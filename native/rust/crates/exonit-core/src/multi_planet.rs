//! Transliterated from `Sources/ExoNitCore/MultiPlanetSearcher.swift`.

use std::fmt;

use crate::bls::accelerated::{self, ComputeBackend};
use crate::bls::{BlsError, BlsParams, BlsResult, BlsSearchFn};
use crate::detrend::{self, DetrendError};
use crate::dto::CandidateResult;
use crate::harmonic_epoch_class;
use crate::harmonic_resolver::{self, HarmonicResolutionOptions, LightCurveSource};
use crate::long_period_sweep::{self, LongPeriodDetrender, LongPeriodSweepOptions};
use crate::preprocess::DetrendFrame;
use crate::transit_physics;
use crate::statistics::median_or;

#[derive(Debug, Clone, PartialEq)]
pub struct MultiPlanetOptions {
    pub maximum_planets: i64,
    pub minimum_snr: f64,
    pub period_minimum: f64,
    pub period_maximum: f64,
    pub period_count: usize,
    pub maximum_mask_coverage: f64,
    pub protected_slots_enabled: bool,
    pub protected_slot_minimum_snr: f64,
    /// Mirrors `[ClosedRange<Double>]` as `(lowerBound, upperBound)` pairs.
    pub protected_slot_ranges: Vec<(f64, f64)>,
    pub harmonic_long_period_recovery: bool,
    pub harmonic_long_period_minimum_days: f64,
    pub harmonic_long_snr_factor: f64,
    /// Config `harmonic_epoch_class` (default off = bit-identical).
    pub harmonic_epoch_class: bool,
    pub long_period_sweep: Option<LongPeriodSweepOptions>,
}

impl Default for MultiPlanetOptions {
    fn default() -> Self {
        MultiPlanetOptions {
            maximum_planets: 5,
            minimum_snr: 7.0,
            period_minimum: 0.5,
            period_maximum: 50.0,
            period_count: 50_000,
            maximum_mask_coverage: 0.30,
            protected_slots_enabled: true,
            protected_slot_minimum_snr: 12.0,
            protected_slot_ranges: vec![(3.0, 10.0), (10.0, 50.0)],
            harmonic_long_period_recovery: false,
            harmonic_long_period_minimum_days: 10.0,
            harmonic_long_snr_factor: 0.5,
            harmonic_epoch_class: false,
            long_period_sweep: Some(LongPeriodSweepOptions::default()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiPlanetSearchResult {
    pub candidates: Vec<CandidateResult>,
    pub accumulated_mask: Vec<bool>,
    pub rejected_unphysical_count: usize,
    pub redetrend_failure_count: usize,
}

/// Unifies the two (untyped, in Swift) error sources of
/// `MultiPlanetSearcher.search`: its own length-mismatch guard and the
/// injected BLS `search` step — see `preprocess::PreprocessError` for the
/// same pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum MultiPlanetError {
    LengthMismatch,
    Bls(BlsError),
}

impl fmt::Display for MultiPlanetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultiPlanetError::LengthMismatch => write!(f, "time/flux lengths differ"),
            MultiPlanetError::Bls(e) => write!(f, "BLS search failed: {e}"),
        }
    }
}

impl std::error::Error for MultiPlanetError {}

impl From<BlsError> for MultiPlanetError {
    fn from(value: BlsError) -> Self {
        MultiPlanetError::Bls(value)
    }
}

/// Mirrors `MultiPlanetSearcher.search(time:rawFlux:initialDetrendedFlux:stellarRadiusSolar:stellarMassSolar:source:options:maxConcurrency:blsBackend:)`.
#[allow(clippy::too_many_arguments)]
pub fn search(
    time: &[f64],
    raw_flux: &[f64],
    initial_detrended_flux: &[f64],
    stellar_radius_solar: Option<f64>,
    stellar_mass_solar: Option<f64>,
    source: LightCurveSource,
    options: &MultiPlanetOptions,
    max_concurrency: usize,
    bls_backend: ComputeBackend,
) -> Result<MultiPlanetSearchResult, MultiPlanetError> {
    search_with_frame(
        time,
        raw_flux,
        initial_detrended_flux,
        stellar_radius_solar,
        stellar_mass_solar,
        source,
        options,
        max_concurrency,
        bls_backend,
        None,
    )
}

/// [`search`] with the optional `gap_fill_detrend_only` frame
/// (`ProcessedLightCurve::detrend_frame`). With a frame, `time`/`raw_flux`/
/// `initial_detrended_flux` hold real points only (what every BLS sees) and
/// the re-detrends (loop i>=1, long-period sweep) run on the frame's full
/// real+synthetic series, returned at the real points -- the same series the
/// first detrend saw (python `process_task` + `DetrendFrame.detrend`).
/// `None` == [`search`].
#[allow(clippy::too_many_arguments)]
pub fn search_with_frame(
    time: &[f64],
    raw_flux: &[f64],
    initial_detrended_flux: &[f64],
    stellar_radius_solar: Option<f64>,
    stellar_mass_solar: Option<f64>,
    source: LightCurveSource,
    options: &MultiPlanetOptions,
    max_concurrency: usize,
    bls_backend: ComputeBackend,
    frame: Option<&DetrendFrame>,
) -> Result<MultiPlanetSearchResult, MultiPlanetError> {
    if time.len() != raw_flux.len() || time.len() != initial_detrended_flux.len() {
        return Err(MultiPlanetError::LengthMismatch);
    }
    if let Some(f) = frame {
        if f.real_index.len() != time.len() {
            return Err(MultiPlanetError::LengthMismatch);
        }
    }
    let mut candidates: Vec<CandidateResult> = Vec::new();
    let mut mask = vec![false; time.len()];
    let mut current: Vec<f64> = initial_detrended_flux.to_vec();
    let mut consecutive_detrend_failures = 0i32;
    let mut total_detrend_failures = 0usize;
    let mut rejected = 0usize;

    // `ComputeBackend` is `Clone` but not `Copy` (as of F4a: the `Cuda`
    // variant carries an `Arc<CudaBlsBackend>` handle) — this closure is
    // called once per BLS search across the whole multi-planet loop, so it
    // must clone the captured backend on each call rather than moving it.
    let search_fn = |t: &[f64], f: &[f64], p: &BlsParams| -> Result<BlsResult, BlsError> {
        accelerated::search(t, f, p, bls_backend.clone(), max_concurrency)
    };

    for iteration in 0..options.maximum_planets.max(0) {
        if iteration >= 1 && mask.iter().any(|&m| m) {
            let coverage = mask.iter().filter(|&&m| m).count() as f64 / mask.len() as f64;
            if coverage <= options.maximum_mask_coverage {
                let redetrended = match frame {
                    Some(f) => f.flatten(Some(&mask)),
                    None => detrend::flatten(time, raw_flux, Some(&mask)),
                };
                match redetrended {
                    Ok(mut flattened) => {
                        let good: Vec<usize> =
                            (0..flattened.len()).filter(|&i| flattened[i].is_finite() && !mask[i]).collect();
                        let fill = if good.is_empty() {
                            1.0
                        } else {
                            median_or(&good.iter().map(|&i| flattened[i]).collect::<Vec<_>>(), 1.0)
                        };
                        for index in 0..flattened.len() {
                            if !flattened[index].is_finite() || mask[index] {
                                flattened[index] = fill;
                            }
                        }
                        current = flattened;
                        consecutive_detrend_failures = 0;
                    }
                    Err(_e) => {
                        consecutive_detrend_failures += 1;
                        total_detrend_failures += 1;
                        if consecutive_detrend_failures >= 2 {
                            break;
                        }
                    }
                }
            }
        }
        let parameters = BlsParams {
            period_minimum: options.period_minimum,
            period_maximum: options.period_maximum,
            period_count: options.period_count,
            ..BlsParams::default()
        };
        let seed = search_fn(time, &current, &parameters)?;
        let result = harmonic_resolver::resolve_with_excluded(
            time,
            &current,
            &seed,
            options.period_minimum,
            options.period_maximum,
            &HarmonicResolutionOptions {
                source,
                long_period_recovery: options.harmonic_long_period_recovery,
                long_period_minimum_days: options.harmonic_long_period_minimum_days,
                long_snr_factor: options.harmonic_long_snr_factor,
                epoch_class: options.harmonic_epoch_class,
                ..HarmonicResolutionOptions::default()
            },
            &search_fn,
            Some(&mask),
        );
        if result.snr < options.minimum_snr {
            break;
        }
        let transit_mask = make_mask(time, result.period, result.t0, result.duration);
        let physical = transit_physics::physical_t14_maximum(result.period, stellar_radius_solar, stellar_mass_solar);
        if physical > 0.0 && result.duration > 3.0 * physical {
            rejected += 1;
            apply(&transit_mask, &mut mask, &mut current);
            continue;
        }
        candidates.push(candidate(
            iteration + 1,
            &result,
            time.last().copied().unwrap_or(0.0) - time.first().copied().unwrap_or(0.0),
        ));
        apply(&transit_mask, &mut mask, &mut current);
    }

    if options.protected_slots_enabled {
        let span = time.last().copied().unwrap_or(0.0) - time.first().copied().unwrap_or(0.0);
        for &(lower, upper_bound) in &options.protected_slot_ranges {
            let upper = upper_bound.min(options.period_maximum);
            if !(upper > lower) || !(span >= 2.0 * lower) {
                continue;
            }
            let result = search_fn(
                time,
                &current,
                &BlsParams {
                    period_minimum: lower,
                    period_maximum: upper,
                    period_count: options.period_count,
                    ..BlsParams::default()
                },
            )?;
            let mut duplicate = candidates
                .iter()
                .any(|c| (result.period - c.period_days).abs() / c.period_days.max(1e-9) < 0.02);
            // Epoch-coincident P/k or k*P of an accepted candidate is the same
            // signal; parity decides which side is the alias (flag-gated,
            // default off). Parity needs the accepted candidates' OWN transits:
            // use the pre-loop detrended flux, NOT `current` / `mask` (there
            // they are masked and median-filled -> always 'undecided').
            // Rust has no `duration_refine`, so t0 is always the box start.
            if !duplicate && result.period > 0.0 && options.harmonic_epoch_class {
                duplicate = epoch_class_slot_duplicate(time, initial_detrended_flux, &result, &candidates);
            }
            let physical =
                transit_physics::physical_t14_maximum(result.period, stellar_radius_solar, stellar_mass_solar);
            if result.snr >= options.protected_slot_minimum_snr
                && !duplicate
                && span / result.period >= 2.0
                && !(physical > 0.0 && result.duration > 3.0 * physical)
            {
                candidates.push(candidate(candidates.len() as i64 + 1, &result, span));
            }
        }
    }

    if let Some(long_options) = &options.long_period_sweep {
        if let Ok(Some(long)) = long_period_sweep_step(time, raw_flux, &mask, long_options, frame, &search_fn) {
            let physical =
                transit_physics::physical_t14_maximum(long.result.period, stellar_radius_solar, stellar_mass_solar);
            if !(physical > 0.0 && long.result.duration > 3.0 * physical) {
                let span = time.last().copied().unwrap_or(0.0) - time.first().copied().unwrap_or(0.0);
                if (candidates.len() as i64) < options.maximum_planets {
                    candidates.push(candidate(candidates.len() as i64 + 1, &long.result, span));
                } else if let Some(weakest) = candidates
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.snr.partial_cmp(&b.snr).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                {
                    if long.result.snr > candidates[weakest].snr {
                        let number = candidates[weakest].candidate_number;
                        candidates[weakest] = candidate(number, &long.result, span);
                    }
                }
            }
        }
    }

    Ok(MultiPlanetSearchResult {
        candidates,
        accumulated_mask: mask,
        rejected_unphysical_count: rejected,
        redetrend_failure_count: total_detrend_failures,
    })
}

/// Long-period sweep step of [`search_with_frame`]: with a
/// `gap_fill_detrend_only` frame every window re-detrends the frame's full
/// real+synthetic series (result taken at the real points), so the sweep BLS
/// (`search`) only ever sees `time` = real points; without a frame it is the
/// legacy default detrender on `time`/`raw_flux`.
pub fn long_period_sweep_step(
    time: &[f64],
    raw_flux: &[f64],
    accumulated_mask: &[bool],
    long_options: &LongPeriodSweepOptions,
    frame: Option<&DetrendFrame>,
    search: BlsSearchFn,
) -> Result<Option<long_period_sweep::LongPeriodSweepResult>, long_period_sweep::LongPeriodSweepError> {
    let frame_long_detrender =
        |_t: &[f64], _f: &[f64], window_days: f64, m: Option<&[bool]>| -> Result<Vec<f64>, DetrendError> {
            frame.expect("frame detrender only selected with a frame").flatten_with_params(
                window_days,
                detrend::DEFAULT_EDGE_CUTOFF,
                detrend::DEFAULT_BREAK_TOLERANCE,
                detrend::DEFAULT_CVAL,
                m,
            )
        };
    let long_detrender: LongPeriodDetrender =
        if frame.is_some() { &frame_long_detrender } else { &default_long_detrender };
    long_period_sweep::search(time, raw_flux, accumulated_mask, long_options, long_detrender, search)
}

/// Mirrors `LongPeriodSweep.search`'s default `detrender` argument
/// (`BiweightDetrender.flatten(time:flux:windowLength:transitMask:)` with
/// all other parameters at their Wotan defaults).
fn default_long_detrender(time: &[f64], flux: &[f64], window_days: f64, mask: Option<&[bool]>) -> Result<Vec<f64>, DetrendError> {
    detrend::flatten_with_params(
        time,
        flux,
        window_days,
        detrend::DEFAULT_EDGE_CUTOFF,
        detrend::DEFAULT_BREAK_TOLERANCE,
        detrend::DEFAULT_CVAL,
        mask,
    )
}

fn make_mask(time: &[f64], period: f64, t0: f64, duration: f64) -> Vec<bool> {
    let half = duration / period / 2.0;
    time.iter()
        .map(|&value| {
            let mut phase = (value - t0) % period / period;
            if phase < 0.0 {
                phase += 1.0;
            }
            phase < half || phase > 1.0 - half
        })
        .collect()
}

fn apply(added: &[bool], mask: &mut [bool], flux: &mut [f64]) {
    let finite: Vec<f64> = flux.iter().copied().filter(|v| v.is_finite()).collect();
    let fill = median_or(&finite, 1.0);
    for index in 0..mask.len() {
        if added[index] {
            mask[index] = true;
            flux[index] = fill;
        }
    }
}

/// Flag-gated epoch-class dedup of a protected-slot BLS result against the
/// accepted candidates (`harmonic_epoch_class::slot_duplicate`).
///
/// `t0_is_centre` is the LITERAL `false` and must stay so: every `t0` here
/// (the slot result's and the accepted `CandidateResult.t0`) is the BLS box
/// START, because Rust has no `duration_refine` step that rewrites t0 to the
/// transit centre (the Python worker passes `True` only on that path). Passing
/// `true` would read box starts as centres and shift candidates of different
/// durations against each other. Pinned by
/// `slot_duplicate_reads_t0_as_box_start`.
fn epoch_class_slot_duplicate(
    time: &[f64],
    flux: &[f64],
    result: &BlsResult,
    candidates: &[CandidateResult],
) -> bool {
    let accepted: Vec<harmonic_epoch_class::EpochCandidate> = candidates
        .iter()
        .map(|c| harmonic_epoch_class::EpochCandidate {
            period_days: c.period_days,
            t0: c.t0,
            duration_days: c.duration_days,
        })
        .collect();
    harmonic_epoch_class::slot_duplicate(
        time,
        flux,
        result.period,
        result.t0,
        result.duration,
        &accepted,
        None,
        false,
    )
}

fn candidate(number: i64, result: &BlsResult, span: f64) -> CandidateResult {
    let mut c = CandidateResult::new(
        number,
        rounded(result.period, 4),
        rounded(result.t0, 4),
        rounded(result.duration, 4),
        rounded(result.depth_ppm, 1),
        rounded(result.snr, 1),
    );
    c.bls_power = Some(rounded(result.power, 4));
    c.expected_transits = Some(if result.period > 0.0 { (span / result.period) as i64 } else { 0 });
    c
}

fn rounded(value: f64, places: i32) -> f64 {
    let scale = 10f64.powi(places);
    (value * scale).round_ties_even() / scale
}

#[cfg(test)]
mod slot_t0_convention_tests {
    use super::*;

    /// Flat noisy 60 d light curve: every parity question is decidable but
    /// carries no signal (undecided), so a matched epoch means DROP.
    fn flat_lc() -> (Vec<f64>, Vec<f64>) {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let (mut time, mut flux) = (Vec::new(), Vec::new());
        for i in 0..1200 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
            time.push(i as f64 * 0.05);
            flux.push(1.0 + 2e-4 * u);
        }
        (time, flux)
    }

    fn bls(period: f64, t0: f64, duration: f64) -> BlsResult {
        BlsResult {
            period,
            power: 1.0,
            duration,
            t0,
            depth: 0.004,
            depth_ppm: 4000.0,
            snr: 20.0,
            periods: Vec::new(),
            power_spectrum: Vec::new(),
        }
    }

    /// Accepted P=5 d (t0 2.0, dur 0.1) and a slot result at 2P (t0 3.3, dur 1.0).
    /// Read as box STARTS (Rust: no duration_refine) the epochs differ by
    /// 1.75 d (> tolerance 1.5 d): not the same signal, the slot candidate is
    /// kept. Read as CENTRES (`t0_is_centre = true`) they differ by 1.3 d and
    /// would be dropped. So `false` at the call site is observable.
    #[test]
    fn slot_duplicate_reads_t0_as_box_start() {
        let (time, flux) = flat_lc();
        let accepted = vec![CandidateResult::new(1, 5.0, 2.0, 0.1, 4000.0, 20.0)];
        assert!(
            !epoch_class_slot_duplicate(&time, &flux, &bls(10.0, 3.3, 1.0), &accepted),
            "box-start convention: epochs differ by 1.75 d > 1.5 d tolerance, so the slot candidate is kept"
        );
        // control: the same pair IS a duplicate under the centre convention, so the test can fail
        let as_epoch = [harmonic_epoch_class::EpochCandidate { period_days: 5.0, t0: 2.0, duration_days: 0.1 }];
        assert!(harmonic_epoch_class::slot_duplicate(&time, &flux, 10.0, 3.3, 1.0, &as_epoch, None, true));
    }
}
