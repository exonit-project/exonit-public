"""Stage predicates for the recovery funnel (spec 4.2), reading the worker's
candidate dicts.  Production gate logic is IMPORTED, never re-implemented:
``exonit.work_producers.verdict_eligible`` is the single source of truth for the
verdict pre-gate (snr >= 10 AND (cnn >= 0.7 OR exominer_ffi >= 0.5)).

Stage flags are computed independently (``prod`` view); the cumulative funnel of
the spec is their running AND in STAGE_ORDER.  A missing score never passes.
"""
from __future__ import annotations

from exonit.work_producers import verdict_eligible

SNR_GATE = 10.0
CNN_GATE = 0.7
EM_GATE = 0.5          # verdict OR-gate; strong tier uses exominer_ffi_score > v6_min=0.5

# worker/server-side stages, in funnel order
STAGE_ORDER = [
    "S0_emitted", "S0a_rank1", "S1_vetting", "S2_snr10", "S3_gate",
    "S5_verdict", "S6_strong", "S7_finalist",
]
SUB_STAGES = ["S3a_cnn", "S3b_em_ffi", "S4_fpp"]


def worker_stage_flags(cand, cnn, vetting, rank_by_snr):
    """Flags knowable from the worker result alone (S0..S3)."""
    snr = cand.get("snr")
    em = cand.get("exominer_ffi_score")
    return {
        "S0_emitted": True,
        "S0a_rank1": rank_by_snr == 1,
        "S1_vetting": bool(vetting and vetting.get("passed")),
        "S2_snr10": snr is not None and snr >= SNR_GATE,
        "S3_gate": verdict_eligible(snr, cnn, em),
        "S3a_cnn": cnn is not None and cnn >= CNN_GATE,
        "S3b_em_ffi": em is not None and em >= EM_GATE,
    }


def first_failed(flags):
    """First failing stage in STAGE_ORDER (the 'killer'), or None if all pass."""
    for s in STAGE_ORDER:
        if not flags.get(s, False):
            return s
    return None


def cumulative(flags):
    """Running AND over STAGE_ORDER -> {stage: bool}."""
    out, ok = {}, True
    for s in STAGE_ORDER:
        ok = ok and bool(flags.get(s, False))
        out[s] = ok
    return out
