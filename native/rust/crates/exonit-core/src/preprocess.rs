//! Transliterated from `Sources/ExoNitCore/LightCurvePreprocessor.swift`.
//! Reproduces the Python worker's stitch -> clean -> positive sigma clip ->
//! conditional gap fill -> Wotan path.

use crate::detrend::{self, DetrendError};
use crate::detrend_iterative::{self, QuickBlsSearch};
use crate::fits::{LightCurveData, PhotometrySource};
use crate::gaps::{fill_if_allowed_tracked, GapFillError, GapFillSeries, SeededGaussianNoiseSource};
use crate::statistics::{median_or, nan_median};

#[derive(Debug, Clone, PartialEq)]
pub struct ProcessedLightCurve {
    pub time: Vec<f64>,
    pub flat_flux: Vec<f64>,
    pub raw_flux: Vec<f64>,
    pub sector_count: usize,
    pub max_gap_days: f64,
    /// `gap_fill_detrend_only` only (config flag, default off): the full
    /// real+synthetic series the detrend saw. `time`/`flat_flux`/`raw_flux`
    /// above then hold real points only, and every later re-detrend
    /// (multi-planet loop, long-period sweep) must go through this frame.
    /// `None` when the flag is off or the gap fill added no point.
    pub detrend_frame: Option<DetrendFrame>,
}

/// Mirrors python `worker/gap_fill.py::DetrendFrame` (config flag
/// `gap_fill_detrend_only`, docs/reviews/2026-10-15-fill-gaps-sigma-parity.md
/// option A). Gap-fill points exist only to bridge intra-sector gaps for the
/// biweight detrender (otherwise `break_tolerance` splits the curve and
/// `edge_cutoff` NaNs real cadences next to every gap); they never reach BLS.
/// `time`/`raw_flux` are the sorted, finite full series the first detrend
/// produced (identical to the flag-off `ProcessedLightCurve` arrays);
/// `real_index` selects the real points (= the arrays handed to BLS).
/// Transit masks are given on real points and scattered with `false` on the
/// synthetic points (no transit signal; they always stay in the trend fit).
/// This is the one deliberate deviation from flag off, where the phase-based
/// masks also cover synthetic points inside a transit window. With the flag
/// on, the multi-planet 30 % mask-coverage gate counts real points only.
#[derive(Debug, Clone, PartialEq)]
pub struct DetrendFrame {
    pub time: Vec<f64>,
    pub raw_flux: Vec<f64>,
    pub real_index: Vec<usize>,
}

impl DetrendFrame {
    pub fn new(time: Vec<f64>, raw_flux: Vec<f64>, synthetic: &[bool]) -> Result<Self, PreprocessError> {
        if time.len() != raw_flux.len() || time.len() != synthetic.len() {
            return Err(PreprocessError::LengthMismatch);
        }
        let real_index = (0..synthetic.len()).filter(|&i| !synthetic[i]).collect();
        Ok(DetrendFrame { time, raw_flux, real_index })
    }

    pub fn synthetic_count(&self) -> usize {
        self.time.len() - self.real_index.len()
    }

    /// Real-point mask -> full-series mask (`false` on synthetic points).
    pub fn expand_mask(&self, mask_real: &[bool]) -> Result<Vec<bool>, DetrendError> {
        if mask_real.len() != self.real_index.len() {
            return Err(DetrendError::MaskLengthMismatch);
        }
        let mut full = vec![false; self.time.len()];
        for (k, &i) in self.real_index.iter().enumerate() {
            full[i] = mask_real[k];
        }
        Ok(full)
    }

    /// Values of a full-series array at the real points.
    pub fn gather(&self, full: &[f64]) -> Vec<f64> {
        self.real_index.iter().map(|&i| full[i]).collect()
    }

    /// `detrend::flatten_with_params` on the full series, returned at the real
    /// points only (python `DetrendFrame.detrend`).
    pub fn flatten_with_params(
        &self,
        window_length: f64,
        edge_cutoff: f64,
        break_tolerance: f64,
        cval: f64,
        transit_mask_real: Option<&[bool]>,
    ) -> Result<Vec<f64>, DetrendError> {
        let full_mask = match transit_mask_real {
            Some(m) => Some(self.expand_mask(m)?),
            None => None,
        };
        let flat = detrend::flatten_with_params(
            &self.time,
            &self.raw_flux,
            window_length,
            edge_cutoff,
            break_tolerance,
            cval,
            full_mask.as_deref(),
        )?;
        Ok(self.gather(&flat))
    }

    /// [`Self::flatten_with_params`] at wotan defaults (== `detrend::flatten`).
    pub fn flatten(&self, transit_mask_real: Option<&[bool]>) -> Result<Vec<f64>, DetrendError> {
        self.flatten_with_params(
            detrend::DEFAULT_WINDOW_LENGTH,
            detrend::DEFAULT_EDGE_CUTOFF,
            detrend::DEFAULT_BREAK_TOLERANCE,
            detrend::DEFAULT_CVAL,
            transit_mask_real,
        )
    }
}

/// Unified error type for `process()`. Swift's `throws` is untyped and lets
/// `LightCurvePreprocessor.process` propagate either a `DetrendError` or a
/// `GapFillError` interchangeably; Rust needs one concrete type, so this
/// wraps both (language-driven plumbing difference only, no numeric change).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreprocessError {
    LengthMismatch,
    MaskLengthMismatch,
    EmptyInput,
    NonMonotonicTime,
    NonMonotonicCadence,
}

impl std::fmt::Display for PreprocessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            PreprocessError::LengthMismatch => "time and flux lengths differ",
            PreprocessError::MaskLengthMismatch => "transit mask length differs from time",
            PreprocessError::EmptyInput => "empty light curve",
            PreprocessError::NonMonotonicTime => "time is not strictly increasing",
            PreprocessError::NonMonotonicCadence => "cadence is not strictly increasing",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for PreprocessError {}

impl From<DetrendError> for PreprocessError {
    fn from(value: DetrendError) -> Self {
        match value {
            DetrendError::LengthMismatch => PreprocessError::LengthMismatch,
            DetrendError::MaskLengthMismatch => PreprocessError::MaskLengthMismatch,
            DetrendError::EmptyInput => PreprocessError::EmptyInput,
        }
    }
}

impl From<GapFillError> for PreprocessError {
    fn from(value: GapFillError) -> Self {
        match value {
            GapFillError::LengthMismatch => PreprocessError::LengthMismatch,
            GapFillError::NonMonotonicTime => PreprocessError::NonMonotonicTime,
            GapFillError::NonMonotonicCadence => PreprocessError::NonMonotonicCadence,
        }
    }
}

/// Which detrend pass `process` should run. Swift's `iterativeDetrend: Bool`
/// defaults to `true`; Rust has no default parameters, so callers choose
/// explicitly. `Iterative` needs a `QuickBlsSearch` impl (see
/// detrend_iterative.rs doc comment — real BLS is F2 scope).
pub enum DetrendMode<'a> {
    Biweight,
    Iterative(&'a dyn QuickBlsSearch),
}

/// Pre-detrend intermediate: stitched, cleaned (NaN-dropped), positive-
/// sigma-clipped and (conditionally) gap-filled `time`/`raw`, i.e. exactly
/// the arrays `LightCurvePreprocessor.process` hands to the detrend step.
/// Factored out of `process()` so a golden test can inject a real
/// externally-computed quick-BLS result (see detrend_iterative.rs) against
/// the exact same pre-detrend arrays the reference Python pipeline used,
/// without needing real BLS in F1. Also the natural seam for F2's BLS
/// integration later.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedLightCurve {
    pub time: Vec<f64>,
    pub raw: Vec<f64>,
    pub max_gap_days: f64,
    /// Aligned with `time`: `true` = point created by the gap fill (all
    /// `false` when no fill happened). Consumed only by `gap_fill_detrend_only`.
    pub synthetic: Vec<bool>,
}

/// Stitch -> clean -> positive sigma clip -> conditional gap fill. Mirrors
/// the first half of `LightCurvePreprocessor.process` (everything before the
/// detrend call).
///
/// Gap-fill noise is seeded from the data itself (see [`data_fill_seed`]) so
/// the result is deterministic even for callers that know no TIC; the task
/// processor uses [`prepare_seeded`] with the normative (tic, sectors) seed.
pub fn prepare(curves: &[LightCurveData]) -> Result<PreparedLightCurve, PreprocessError> {
    prepare_seeded(curves, data_fill_seed(curves))
}

/// Fallback gap-fill seed for callers without a TIC id: FNV-1a over the
/// number of curves, and each curve's length and first/last time bits.
pub fn data_fill_seed(curves: &[LightCurveData]) -> u32 {
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(&(curves.len() as u64).to_le_bytes());
    for c in curves {
        bytes.extend_from_slice(&(c.time.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&c.time.first().copied().unwrap_or(0.0).to_bits().to_le_bytes());
        bytes.extend_from_slice(&c.time.last().copied().unwrap_or(0.0).to_bits().to_le_bytes());
    }
    crate::gaps::fnv1a_u32(&bytes)
}

/// [`prepare`] with an explicit gap-fill seed (`gaps::gap_fill_seed(tic, sectors)`).
pub fn prepare_seeded(curves: &[LightCurveData], fill_seed: u32) -> Result<PreparedLightCurve, PreprocessError> {
    if curves.is_empty() {
        return Err(PreprocessError::EmptyInput);
    }
    let mut time: Vec<f64> = Vec::new();
    let mut raw: Vec<f64> = Vec::new();
    let all_have_cadence = curves.iter().all(|c| c.cadence.is_some());
    let mut cadence: Option<Vec<i64>> = if all_have_cadence { Some(Vec::new()) } else { None };

    for curve in curves {
        if curve.time.len() != curve.flux.len() {
            return Err(PreprocessError::LengthMismatch);
        }
        if let Some(c) = &curve.cadence {
            if c.len() != curve.time.len() {
                return Err(PreprocessError::LengthMismatch);
            }
        }
        if curves.len() == 1 {
            time.extend_from_slice(&curve.time);
            raw.extend_from_slice(&curve.flux);
        } else {
            let normalization = nan_median(&curve.flux);
            if curve.source == PhotometrySource::Tglc {
                // worker.py's _read_lightcurve() upcasts cal_aper_flux to
                // np.float64 BEFORE constructing the LightCurve object, so
                // lightkurve's normalize()/nanmedian() run entirely in
                // float64 for TGLC/FFI -- never touching the FITS `E`
                // column's native Float32 dtype. Divide in double precision
                // to match.
                for index in 0..curve.time.len() {
                    time.push(curve.time[index]);
                    raw.push(curve.flux[index] / normalization);
                }
            } else {
                // SPOC/QLP go through plain lk.read(path), which DOES
                // preserve the FITS `E` column's native Float32 dtype
                // through normalize() -- keep reproducing that single-
                // rounding division.
                let normalization32 = normalization as f32;
                for index in 0..curve.time.len() {
                    time.push(curve.time[index]);
                    raw.push(((curve.flux[index] as f32) / normalization32) as f64);
                }
            }
        }
        if let Some(values) = &curve.cadence {
            if let Some(c) = cadence.as_mut() {
                c.extend_from_slice(values);
            }
        }
    }

    // LightCurve.remove_nans() removes invalid flux cadences before clipping.
    let mut finite_time: Vec<f64> = Vec::with_capacity(time.len());
    let mut finite_raw: Vec<f64> = Vec::with_capacity(time.len());
    let mut finite_cadence: Option<Vec<i64>> = cadence.as_ref().map(|_| Vec::new());
    for index in 0..time.len() {
        if time[index].is_finite() && raw[index].is_finite() {
            finite_time.push(time[index]);
            finite_raw.push(raw[index]);
            if let (Some(c), Some(fc)) = (&cadence, finite_cadence.as_mut()) {
                fc.push(c[index]);
            }
        }
    }
    let keep = upper_sigma_clip_mask(&finite_raw, 5.0, 5);
    let mut time: Vec<f64> = Vec::new();
    let mut raw: Vec<f64> = Vec::new();
    let mut cadence: Option<Vec<i64>> = finite_cadence.as_ref().map(|_| Vec::new());
    for index in 0..keep.len() {
        if keep[index] {
            time.push(finite_time[index]);
            raw.push(finite_raw[index]);
            if let (Some(fc), Some(c)) = (&finite_cadence, cadence.as_mut()) {
                c.push(fc[index]);
            }
        }
    }

    let mut max_gap = 0.0_f64;
    if time.len() > 1 {
        for index in 1..time.len() {
            max_gap = max_gap.max(time[index] - time[index - 1]);
        }
    }
    // Python calls fill_gaps only when the largest observed gap is strictly
    // below five days; the noise is now seeded (fill-gaps-determinism).
    let mut synthetic = vec![false; time.len()];
    if time.len() > 1 && max_gap < 5.0 {
        // Mirrors Swift `LightCurvePreprocessor`: lightkurve's fill_gaps()
        // re-sorts by cadence number (`ncad[np.argsort(ncad)]`) on the
        // cadenceno path, so a multi-sector SPOC stitch loaded in
        // non-chronological order comes back time-ordered. Reproduce that
        // side effect (stable sort, index-aligned) only when cadence is
        // present; the time-grid branch (TGLC/FFI) never pre-sorts in Python.
        // The max-gap gate above intentionally stays on the pre-sort order.
        if let Some(c) = cadence.as_mut() {
            let mut order: Vec<usize> = (0..time.len()).collect();
            order.sort_by(|&a, &b| time[a].partial_cmp(&time[b]).unwrap_or(std::cmp::Ordering::Equal));
            time = order.iter().map(|&i| time[i]).collect();
            raw = order.iter().map(|&i| raw[i]).collect();
            *c = order.iter().map(|&i| c[i]).collect();
        }
        let mut noise = SeededGaussianNoiseSource::new(fill_seed);
        let series = GapFillSeries {
            time: time.clone(),
            flux: raw.clone(),
            cadence: cadence.clone(),
        };
        let (filled, added) = fill_if_allowed_tracked(series, 5.0, 1.2, None, &mut noise)?;
        time = filled.time;
        raw = filled.flux;
        synthetic = added;
        // ProcessedLightCurve has no cadence field; it only drives gap-grid
        // selection above.
    }

    Ok(PreparedLightCurve {
        time,
        raw,
        max_gap_days: max_gap,
        synthetic,
    })
}

/// Mirrors `LightCurvePreprocessor.process(_:iterativeDetrend:maxConcurrency:)`.
/// `maxConcurrency` is dropped for the same reason as in detrend_iterative.rs
/// (single-threaded F1; the knob only ever reached the out-of-scope BLS
/// search).
pub fn process(curves: &[LightCurveData], mode: DetrendMode) -> Result<ProcessedLightCurve, PreprocessError> {
    process_with_backend(curves, mode, &detrend::DetrendBackend::Cpu)
}

/// `process` restricted to a caller-selected [`detrend::DetrendBackend`]
/// (F4b) — same body as `process`, with the detrend step routed through
/// `detrend::flatten_with_backend`/`detrend_iterative::flatten_with_backend`
/// instead of the CPU-only entry points. `process()` above is a thin
/// `Cpu`-backend wrapper kept for source compatibility with every existing
/// call site (tests, `task_processor.rs` pre-F4b) — behaviourally identical
/// to before this function existed.
pub fn process_with_backend(
    curves: &[LightCurveData],
    mode: DetrendMode,
    detrend_backend: &detrend::DetrendBackend,
) -> Result<ProcessedLightCurve, PreprocessError> {
    process_with_backend_seeded(curves, mode, detrend_backend, data_fill_seed(curves))
}

/// [`process_with_backend`] with an explicit gap-fill seed.
pub fn process_with_backend_seeded(
    curves: &[LightCurveData],
    mode: DetrendMode,
    detrend_backend: &detrend::DetrendBackend,
    fill_seed: u32,
) -> Result<ProcessedLightCurve, PreprocessError> {
    process_with_backend_seeded_opts(curves, mode, detrend_backend, fill_seed, false)
}

/// [`process_with_backend_seeded`] plus the `gap_fill_detrend_only` config
/// flag (task config, default false; mirrors python `_load_star`). Off: the
/// legacy path, bit-identical. On (and the gap fill added points): the
/// detrend runs on the real+synthetic series, the quick BLS and transit mask
/// of the iterative pass ignore synthetic points, the renormalisation median
/// is taken over real points, and the synthetic points are dropped from the
/// returned arrays after the sort/finite filter, the full series being kept
/// in `detrend_frame` for the later re-detrends.
pub fn process_with_backend_seeded_opts(
    curves: &[LightCurveData],
    mode: DetrendMode,
    detrend_backend: &detrend::DetrendBackend,
    fill_seed: u32,
    gap_fill_detrend_only: bool,
) -> Result<ProcessedLightCurve, PreprocessError> {
    let prepared = prepare_seeded(curves, fill_seed)?;
    let time = prepared.time;
    let raw = prepared.raw;
    let max_gap = prepared.max_gap_days;
    let synthetic: Option<Vec<bool>> =
        if gap_fill_detrend_only && prepared.synthetic.iter().any(|&s| s) { Some(prepared.synthetic) } else { None };

    let mut flat: Vec<f64> = match mode {
        DetrendMode::Iterative(search) => {
            detrend_iterative::flatten_with_backend_excluding(
                &time,
                &raw,
                search,
                detrend_iterative::DEFAULT_SNR_THRESHOLD,
                detrend_backend,
                synthetic.as_deref(),
            )?
            .flux
        }
        DetrendMode::Biweight => detrend::flatten_with_backend(&time, &raw, None, detrend_backend)?,
    };
    let flat_median = match &synthetic {
        None => nan_median(&flat),
        Some(syn) => nan_median(&(0..flat.len()).filter(|&i| !syn[i]).map(|i| flat[i]).collect::<Vec<_>>()),
    };
    if flat_median > 1.5 || flat_median < 0.5 {
        for value in flat.iter_mut() {
            *value /= flat_median;
        }
    }

    let mut order: Vec<usize> = (0..time.len()).collect();
    order.sort_by(|&a, &b| time[a].partial_cmp(&time[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut sorted_time: Vec<f64> = Vec::with_capacity(order.len());
    let mut sorted_flat: Vec<f64> = Vec::with_capacity(order.len());
    let mut sorted_raw: Vec<f64> = Vec::with_capacity(order.len());
    let mut sorted_synthetic: Vec<bool> = Vec::new();
    for index in order {
        if time[index].is_finite() && flat[index].is_finite() && raw[index].is_finite() {
            sorted_time.push(time[index]);
            sorted_flat.push(flat[index]);
            sorted_raw.push(raw[index]);
            if let Some(syn) = &synthetic {
                sorted_synthetic.push(syn[index]);
            }
        }
    }
    let mut detrend_frame = None;
    if synthetic.is_some() {
        let frame = DetrendFrame::new(sorted_time, sorted_raw, &sorted_synthetic)?;
        sorted_flat = frame.gather(&sorted_flat);
        sorted_time = frame.gather(&frame.time);
        sorted_raw = frame.gather(&frame.raw_flux);
        detrend_frame = Some(frame);
    }
    Ok(ProcessedLightCurve {
        time: sorted_time,
        flat_flux: sorted_flat,
        raw_flux: sorted_raw,
        sector_count: curves.len(),
        max_gap_days: max_gap,
        detrend_frame,
    })
}

fn upper_sigma_clip_mask(values: &[f64], sigma: f64, max_iterations: u32) -> Vec<bool> {
    let mut keep = vec![true; values.len()];
    for _ in 0..max_iterations {
        let current: Vec<f64> = (0..values.len()).filter(|&i| keep[i]).map(|i| values[i]).collect();
        if current.is_empty() {
            break;
        }
        let center = median_or(&current, f64::NAN);
        let mean = current.iter().sum::<f64>() / current.len() as f64;
        let variance = current.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / current.len() as f64;
        let limit = center + sigma * variance.sqrt();
        let mut changed = false;
        for index in 0..values.len() {
            if keep[index] && values[index] > limit {
                keep[index] = false;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors ExoNitContractTests.testStitchNormalizationSemantics
    /// (main.swift:1703): a single-segment curve must NOT be
    /// stitch-normalized; a two-segment stitch must normalize each segment
    /// to ~1.
    #[test]
    fn single_segment_is_not_stitch_normalized() {
        let time: Vec<f64> = (0..100).map(|i| i as f64 * 0.02).collect();
        let single = LightCurveData {
            time: time.clone(),
            flux: vec![100.0; time.len()],
            raw_row_count: time.len(),
            cadence: None,
            source: PhotometrySource::Unknown,
        };
        let result = process(std::slice::from_ref(&single), DetrendMode::Biweight).unwrap();
        assert!(!result.raw_flux.is_empty());
        assert!(result.raw_flux.iter().all(|&v| v == 100.0));
    }

    #[test]
    fn two_segment_stitch_normalizes_each_segment() {
        let time: Vec<f64> = (0..100).map(|i| i as f64 * 0.02).collect();
        let second_time: Vec<f64> = time.iter().map(|t| t + 10.0).collect();
        let first = LightCurveData {
            time: time.clone(),
            flux: vec![100.0; time.len()],
            raw_row_count: time.len(),
            cadence: None,
            source: PhotometrySource::Unknown,
        };
        let second = LightCurveData {
            time: second_time,
            flux: vec![200.0; time.len()],
            raw_row_count: time.len(),
            cadence: None,
            source: PhotometrySource::Unknown,
        };
        let result = process(&[first, second], DetrendMode::Biweight).unwrap();
        assert!(!result.raw_flux.is_empty());
        assert!(result.raw_flux.iter().all(|&v| (v - 1.0).abs() < 1e-7));
    }

    /// Parity with Swift LightCurvePreprocessor (fill-gaps pre-sort): two
    /// adjacent SPOC sectors (cadence present) stitched out of chronological
    /// order must be time-sorted before gap fill instead of failing with
    /// NonMonotonicTime; a cadence hole is still filled.
    #[test]
    fn out_of_order_adjacent_cadence_segments_are_sorted_before_fill() {
        let dt = 0.02;
        let early_cad: Vec<i64> = (0..100).filter(|&i| i != 50).collect();
        let late_cad: Vec<i64> = (150..250).collect();
        let segment = |cad: &[i64]| LightCurveData {
            time: cad.iter().map(|&c| c as f64 * dt).collect(),
            flux: vec![100.0; cad.len()],
            raw_row_count: cad.len(),
            cadence: Some(cad.to_vec()),
            source: PhotometrySource::Unknown,
        };
        let prepared = prepare_seeded(&[segment(&late_cad), segment(&early_cad)], 7).unwrap();
        assert!(prepared.time.windows(2).all(|w| w[1] >= w[0]));
        assert_eq!(prepared.time.len(), 250);
        assert_eq!(prepared.time.len(), prepared.raw.len());
    }

    #[test]
    fn empty_curves_rejected() {
        assert_eq!(process(&[], DetrendMode::Biweight).unwrap_err(), PreprocessError::EmptyInput);
    }
}
