//! Transliterated from `Sources/ExoNitCore/HarmonicResolver.swift`.

use crate::bls::{BlsParams, BlsResult, BlsSearchFn};
use crate::harmonic_epoch_class;

/// Mirrors `LightCurveSource` (`spoc`/`ffi`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightCurveSource {
    Spoc,
    Ffi,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarmonicResolutionOptions {
    pub minimum_snr: f64,
    pub depth_ratio_lower: f64,
    pub depth_ratio_upper: f64,
    pub source: LightCurveSource,
    pub long_period_recovery: bool,
    pub long_period_minimum_days: f64,
    pub long_snr_factor: f64,
    /// Config flag `harmonic_epoch_class` (default off = bit-identical to
    /// the pre-flag resolver). When on: the seed is first promoted to k*P by
    /// the epoch-class parity test (a promotion is returned as-is, no further
    /// probes), and a `/k` demotion probe only wins if all k epoch classes of
    /// its own ephemeris carry the signal. See `harmonic_epoch_class.rs`.
    pub epoch_class: bool,
}

impl Default for HarmonicResolutionOptions {
    fn default() -> Self {
        HarmonicResolutionOptions {
            minimum_snr: 5.0,
            depth_ratio_lower: 0.5,
            depth_ratio_upper: 2.0,
            source: LightCurveSource::Spoc,
            long_period_recovery: false,
            long_period_minimum_days: 10.0,
            long_snr_factor: 0.5,
            epoch_class: false,
        }
    }
}

struct Probe {
    period: f64,
    snr_factor: f64,
    transit_guard: bool,
    /// `Some(k)` for the `/k` demotion probes (epoch-class veto only).
    demotion_k: Option<usize>,
}

/// Mirrors `HarmonicResolver.resolve(time:flux:seed:periodMinimum:periodMaximum:options:search:)`.
pub fn resolve(
    time: &[f64],
    flux: &[f64],
    seed: &BlsResult,
    period_minimum: f64,
    period_maximum: f64,
    options: &HarmonicResolutionOptions,
    search: BlsSearchFn,
) -> BlsResult {
    resolve_with_excluded(time, flux, seed, period_minimum, period_maximum, options, search, None)
}

/// `resolve` plus the multi-planet loop's `accumulated_mask` (`excluded`):
/// filler points the epoch-class code must ignore. Unused (and the result
/// bit-identical to `resolve`) when `options.epoch_class` is off. Mirrors
/// Python `resolve_harmonic(..., epoch_class=, excluded=)`.
#[allow(clippy::too_many_arguments)]
pub fn resolve_with_excluded(
    time: &[f64],
    flux: &[f64],
    seed: &BlsResult,
    period_minimum: f64,
    period_maximum: f64,
    options: &HarmonicResolutionOptions,
    search: BlsSearchFn,
    excluded: Option<&[bool]>,
) -> BlsResult {
    if options.epoch_class {
        if let Some(promoted) = harmonic_epoch_class::promote_epoch_class(
            time,
            flux,
            seed,
            period_maximum,
            search,
            options.minimum_snr,
            excluded,
        ) {
            // Parity evidence is final: no amplitude probes from the promoted
            // seed (its re-fit SNR is a narrow-window SDE, not comparable).
            return promoted;
        }
    }
    let mut probes: Vec<Probe> = Vec::new();
    if options.source == LightCurveSource::Ffi {
        for multiplier in [2.0, 3.0, 4.0, 5.0] {
            let period = seed.period * multiplier;
            if period <= period_maximum {
                probes.push(Probe { period, snr_factor: 0.6, transit_guard: true, demotion_k: None });
            }
        }
    } else {
        let doubled = seed.period * 2.0;
        if doubled <= period_maximum {
            probes.push(Probe { period: doubled, snr_factor: 1.0, transit_guard: false, demotion_k: None });
        }
        if options.long_period_recovery {
            for multiplier in [3.0 / 2.0, 5.0 / 3.0, 4.0 / 3.0, 5.0 / 2.0] {
                let period = seed.period * multiplier;
                if period >= options.long_period_minimum_days && period <= period_maximum {
                    probes.push(Probe {
                        period,
                        snr_factor: options.long_snr_factor,
                        transit_guard: true,
                        demotion_k: None,
                    });
                }
            }
        }
    }
    let demotion_factor = if options.source == LightCurveSource::Ffi { 0.6 } else { 1.0 };
    for divisor in [2.0, 3.0, 4.0, 5.0] {
        let period = seed.period / divisor;
        if period >= period_minimum {
            probes.push(Probe {
                period,
                snr_factor: demotion_factor,
                transit_guard: false,
                demotion_k: Some(divisor as usize),
            });
        }
    }

    let span = time.last().copied().unwrap_or(0.0) - time.first().copied().unwrap_or(0.0);
    for probe in &probes {
        // worker.py's resolve_harmonic() calls bls_fn(..., period_min=cand*0.95,
        // period_max=cand*1.05, ...) WITHOUT clamping to the caller's global
        // period_min/period_max — the refine window is allowed to extend
        // outside [period_minimum, period_maximum] (e.g. a subharmonic probe
        // near period_minimum legitimately searches slightly below it). A
        // clamp here narrows the fine-BLS grid vs Python, landing on a
        // different local peak/SNR near marginal-significance systematics
        // (found via TIC 10757037 in the Swift port: identical seed BLS +
        // identical re-detrended flux, but the clamped probe window resolved
        // to a different period/SNR than Python's unclamped window). Do NOT
        // clamp `parameters.period_minimum`/`period_maximum` to
        // `period_minimum`/`period_maximum` here.
        let parameters = BlsParams {
            period_minimum: probe.period * 0.95,
            period_maximum: probe.period * 1.05,
            period_count: 500,
            ..BlsParams::default()
        };
        if !(parameters.period_maximum > parameters.period_minimum) {
            continue;
        }
        let candidate = match search(time, flux, &parameters) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if !(candidate.snr >= options.minimum_snr) {
            continue;
        }
        if !(candidate.snr >= seed.snr * probe.snr_factor) {
            continue;
        }
        if seed.depth > 0.0 {
            let ratio = candidate.depth / seed.depth;
            if !(ratio > options.depth_ratio_lower && ratio < options.depth_ratio_upper) {
                continue;
            }
        }
        if probe.transit_guard && span > 0.0 && candidate.period > 0.0 && span / candidate.period < 2.0 {
            continue;
        }
        // Demotion veto (epoch_class only), mode "vh": v0 (every class must
        // carry) on short baselines, v1 (veto only on the one-class-carries
        // pattern) when the baseline is long relative to the duration.
        if options.epoch_class {
            if let Some(k) = probe.demotion_k {
                if !harmonic_epoch_class::demotion_supported(time, flux, &candidate, k, excluded) {
                    continue;
                }
            }
        }
        return candidate;
    }
    seed.clone()
}
