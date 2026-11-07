"""CNN-based transit candidate classification using ONNX Runtime.

On macOS with CoreML installed and a sibling `.mlpackage` next to the
ONNX file, inference is routed through Core ML via coremltools. See
`cnn_coreml.try_load_mlpackage`. Measured 2026-10-15 on an M4 (Core ML
compute plan, compute units ALL / CPU_AND_NE / CPU_AND_GPU, fp32 and fp16):
Core ML places every op of this ~5k-parameter model on the CPU, never on
the Neural Engine or GPU, at ~36 us/call vs ~19 us/call for ONNX Runtime
CPU (progress/impl-cnn-none-and-logit.md).
"""

import logging
import os
import numpy as np
import onnxruntime as ort

from cnn_coreml import mlpackage_path_for, try_load_mlpackage
from onnx_providers import select_providers, describe_active_provider
from phase_fold import phase_fold

# The SPOC model outputs the pre-sigmoid logit; the sigmoid is applied in
# classify() with the logit clamped at +/-LOGIT_CLAMP, the same as the FFI path
# (cnn_ffi.py). transit_cnn_logit.onnx is transit_cnn.onnx with its final
# Sigmoid node removed (scripts/export_transit_cnn_logit.py): same weights,
# same graph. The in-graph float32 sigmoid of the old file returned exactly
# 0.0 for logits below about -18 (docs/reviews/2026-10-15-cnn-exact-zero-
# scores.md, fix 3). transit_cnn.onnx (+ .onnx.data) stays in models/ for
# rollback: pass it as model_path, or point MODEL_PATH back at it. classify()
# recognises both by the graph output name.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "transit_cnn_logit.onnx")
LEGACY_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "transit_cnn.onnx")
LOGIT_OUTPUT = "planet_logit"
LOGIT_CLAMP = 60.0
N_BINS = 200


class TransitClassifier:
    def __init__(self, model_path=None, use_gpu=False):
        path = model_path or MODEL_PATH
        if model_path is None and not os.path.exists(path) \
                and os.path.exists(LEGACY_MODEL_PATH):
            # Auto-update can restart the worker with the new cnn.py before
            # transit_cnn_logit.onnx has arrived. Fall back to the legacy
            # model (probability output: classify() recognises it by name and
            # applies no sigmoid) rather than run without a CNN.
            logging.getLogger(__name__).warning(
                "CNN: %s missing, falling back to legacy %s (probability "
                "output, exact-0 underflow below logit ~-18)",
                os.path.basename(MODEL_PATH), os.path.basename(LEGACY_MODEL_PATH))
            path = LEGACY_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"CNN model not found at {path}")
        # Honour the caller's `use_gpu` flag. With onnxruntime-gpu
        # installed and `use_gpu=True`, ORT routes through CUDA;
        # otherwise it falls back to the CPU EP transparently.
        providers = select_providers(gpu_enabled=use_gpu)
        self.session = ort.InferenceSession(path, providers=providers)
        self.backend = f"ONNX-{describe_active_provider(self.session)}"
        onnx_output = self.session.get_outputs()[0].name
        # Only an output explicitly named planet_logit is treated as a logit;
        # anything else (the legacy planet_probability) is used as is.
        self.output_is_logit = onnx_output == LOGIT_OUTPUT
        # CoreML looks for a sibling .mlpackage with the SAME base name, and is
        # only used when its output name equals the ONNX output name, so a
        # probability package can never be read as a logit (or vice versa).
        # transit_cnn_logit.mlpackage is built by
        # scripts/convert_transit_cnn_logit_coreml.py (output planet_logit).
        self._coreml_run = try_load_mlpackage(path)
        if self._coreml_run is not None:
            coreml_outputs = getattr(self._coreml_run, "output_names", [])
            if coreml_outputs[:1] == [onnx_output]:
                self.backend = "CoreML"
            else:
                logging.getLogger(__name__).warning(
                    "CNN: %s outputs %s but %s outputs %r; ignoring the "
                    "mlpackage, using ONNX", mlpackage_path_for(path),
                    coreml_outputs, os.path.basename(path), onnx_output)
                self._coreml_run = None

    def classify(self, time, flux, period, t0, duration):
        phase_flux = self._phase_fold_and_bin(time, flux, period, t0, duration)
        input_data = phase_flux.reshape(1, 1, N_BINS).astype(np.float32)
        inputs = {"phase_folded_flux": input_data}
        if self._coreml_run is not None:
            result = self._coreml_run(inputs)
        else:
            result = self.session.run(None, inputs)
        out = float(result[0][0][0])
        if not self.output_is_logit:
            return out
        logit = max(min(out, LOGIT_CLAMP), -LOGIT_CLAMP)  # guard exp overflow
        return float(1.0 / (1.0 + np.exp(-logit)))

    def _phase_fold_and_bin(self, time, flux, period, t0, duration):
        # Shared with vetting.py (identical convention) — see
        # docs/reviews/2026-07-02-opus48-cleanup-review.md item D5 and
        # tests/worker/test_phase_fold_golden.py for the bit-identical proof.
        phase = phase_fold(time, t0, period, fold_order="mod_div", center="mask")
        bins = np.linspace(-0.5, 0.5, N_BINS + 1)
        binned = np.ones(N_BINS)
        for i in range(N_BINS):
            mask = (phase >= bins[i]) & (phase < bins[i + 1])
            if np.sum(mask) >= 1:
                binned[i] = np.median(flux[mask])
        med = np.median(binned)
        std = np.std(binned)
        if std > 0:
            binned = (binned - med) / std
        return binned
