//! Transliterated from `Sources/ExoNitCore/GapFiller.swift`.
//!
//! Note on gap-fill noise (fill-gaps-determinism, 2026-10-15): lightkurve's
//! `fill_gaps()` draws from the unseeded global numpy RNG, which made repeat
//! runs of the Python worker non-reproducible (IR pilot: 20/100 blank hosts
//! changed candidate lists). Python now seeds a private `RandomState` from
//! FNV-1a 64 of `"{tic}:{sorted sectors}"` truncated to u32 (worker/gap_fill.py)
//! and this port does the same: [`SeededGaussianNoiseSource`] is numpy-legacy
//! `RandomState(seed).normal(mean, sd)` (MT19937 + polar `legacy_gauss`, shared
//! with the FPP port). The unit-normal stream is therefore bit-identical to
//! Python's; the amplitude is NOT: Python uses lightkurve's CDPP as sigma
//! whereas this port (like Swift) uses the population std of the flux.
//! There is no unseeded source any more.

use crate::statistics::median;

#[derive(Debug, Clone, PartialEq)]
pub struct GapFillSeries {
    pub time: Vec<f64>,
    pub flux: Vec<f64>,
    pub cadence: Option<Vec<i64>>,
}

pub trait GaussianNoiseSource {
    fn normal(&mut self, mean: f64, standard_deviation: f64) -> f64;
}

/// numpy-legacy `RandomState(seed).normal(mean, sd)` noise source.
pub struct SeededGaussianNoiseSource {
    rng: crate::fpp::rng::FppRng,
}

impl SeededGaussianNoiseSource {
    pub fn new(seed: u32) -> Self {
        Self { rng: crate::fpp::rng::FppRng::new(seed) }
    }
}

impl GaussianNoiseSource for SeededGaussianNoiseSource {
    fn normal(&mut self, mean: f64, standard_deviation: f64) -> f64 {
        mean + standard_deviation * self.rng.gauss()
    }
}

/// Normative seed key: `"{tic_id}:{s1},{s2},..."` (sectors sorted, de-duplicated).
pub fn gap_fill_seed_key(tic_id: &str, sectors: &[i64]) -> String {
    let mut secs: Vec<i64> = sectors.to_vec();
    secs.sort_unstable();
    secs.dedup();
    let list: Vec<String> = secs.iter().map(|s| s.to_string()).collect();
    format!("{}:{}", tic_id, list.join(","))
}

/// FNV-1a 64 of the seed key, truncated to u32 (== `worker/gap_fill.py`).
pub fn gap_fill_seed(tic_id: &str, sectors: &[i64]) -> u32 {
    fnv1a_u32(gap_fill_seed_key(tic_id, sectors).as_bytes())
}

pub(crate) fn fnv1a_u32(bytes: &[u8]) -> u32 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &byte in bytes {
        h ^= u64::from(byte);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    (h & 0xFFFF_FFFF) as u32
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapFillError {
    LengthMismatch,
    NonMonotonicTime,
    NonMonotonicCadence,
}

impl std::fmt::Display for GapFillError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            GapFillError::LengthMismatch => "time/flux/cadence lengths differ",
            GapFillError::NonMonotonicTime => "time is not strictly increasing",
            GapFillError::NonMonotonicCadence => "cadence is not strictly increasing",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for GapFillError {}

/// Mirrors `GapFiller.fillIfAllowed(_:maximumGlobalGapDays:spacingTolerance:standardDeviation:noise:)`.
/// Rust has no default parameter values, so all Swift defaults
/// (`maximumGlobalGapDays: 5`, `spacingTolerance: 1.2`, `standardDeviation: nil`)
/// must be passed explicitly by the caller.
pub fn fill_if_allowed<R: GaussianNoiseSource>(
    input: GapFillSeries,
    maximum_global_gap_days: f64,
    spacing_tolerance: f64,
    supplied_deviation: Option<f64>,
    noise: &mut R,
) -> Result<GapFillSeries, GapFillError> {
    fill_if_allowed_tracked(input, maximum_global_gap_days, spacing_tolerance, supplied_deviation, noise)
        .map(|(series, _synthetic)| series)
}

/// [`fill_if_allowed`] that also returns the `synthetic` mask aligned with the
/// output (`true` = point created by the gap fill), as Python's
/// `fill_gaps_seeded(..., return_synthetic=True)`. The series is identical.
/// Used by the `gap_fill_detrend_only` config flag (preprocess.rs).
pub fn fill_if_allowed_tracked<R: GaussianNoiseSource>(
    input: GapFillSeries,
    maximum_global_gap_days: f64,
    spacing_tolerance: f64,
    supplied_deviation: Option<f64>,
    noise: &mut R,
) -> Result<(GapFillSeries, Vec<bool>), GapFillError> {
    if input.time.len() != input.flux.len() {
        return Err(GapFillError::LengthMismatch);
    }
    if let Some(cadence) = &input.cadence {
        if cadence.len() != input.time.len() {
            return Err(GapFillError::LengthMismatch);
        }
    }
    if input.time.len() < 2 {
        let n = input.time.len();
        return Ok((input, vec![false; n]));
    }
    let differences: Vec<f64> = input.time.windows(2).map(|w| w[1] - w[0]).collect();
    // Mirrors Swift `GapFiller` (`>= 0`): lightkurve's time-grid branch
    // tolerates an exact tie (duplicate timestamp) by appending it with no
    // filler; only a genuinely descending pair is a caller bug.
    if !differences.iter().all(|&d| d >= 0.0) {
        return Err(GapFillError::NonMonotonicTime);
    }
    let max_diff = differences.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !(max_diff < maximum_global_gap_days) {
        let n = input.time.len();
        return Ok((input, vec![false; n]));
    }

    let finite_flux: Vec<f64> = input.flux.iter().copied().filter(|v| v.is_finite()).collect();
    let mean = if finite_flux.is_empty() {
        f64::NAN
    } else {
        finite_flux.iter().sum::<f64>() / finite_flux.len() as f64
    };
    let deviation = if let Some(d) = supplied_deviation {
        d
    } else if finite_flux.is_empty() {
        f64::NAN
    } else {
        let variance = finite_flux.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>()
            / finite_flux.len() as f64;
        variance.sqrt()
    };

    if let Some(cadence) = input.cadence.clone() {
        fill_cadence_grid(input, &cadence, median(&differences), mean, deviation, noise)
    } else {
        Ok(fill_time_grid(
            input,
            median(&differences),
            spacing_tolerance,
            mean,
            deviation,
            noise,
        ))
    }
}

fn fill_cadence_grid<R: GaussianNoiseSource>(
    input: GapFillSeries,
    cadence: &[i64],
    spacing: f64,
    mean: f64,
    deviation: f64,
    noise: &mut R,
) -> Result<(GapFillSeries, Vec<bool>), GapFillError> {
    let cadence_diff: Vec<i64> = cadence.windows(2).map(|w| w[1] - w[0]).collect();
    if !cadence_diff.iter().all(|&d| d > 0) {
        return Err(GapFillError::NonMonotonicCadence);
    }
    let offsets: Vec<f64> = input
        .time
        .iter()
        .zip(cadence.iter())
        .map(|(&t, &c)| t - spacing * c as f64)
        .collect();
    let mut row_by_cadence: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for (index, &value) in cadence.iter().enumerate() {
        row_by_cadence.insert(value, index);
    }

    let first = cadence[0];
    let last = cadence[cadence.len() - 1];
    let count = (last - first + 1) as usize;
    let mut output = GapFillSeries {
        time: Vec::with_capacity(count),
        flux: Vec::with_capacity(count),
        cadence: Some(Vec::with_capacity(count)),
    };
    let mut synthetic: Vec<bool> = Vec::with_capacity(count);
    let mut lower = 0usize;
    for value in first..=last {
        output.cadence.as_mut().unwrap().push(value);
        if let Some(&row) = row_by_cadence.get(&value) {
            output.time.push(input.time[row]);
            output.flux.push(input.flux[row]);
            synthetic.push(false);
            lower = row;
        } else {
            while lower + 1 < cadence.len() && cadence[lower + 1] < value {
                lower += 1;
            }
            let upper = lower + 1;
            let fraction = (value - cadence[lower]) as f64 / (cadence[upper] - cadence[lower]) as f64;
            let offset = offsets[lower] + fraction * (offsets[upper] - offsets[lower]);
            output.time.push(offset + spacing * value as f64);
            output.flux.push(noise.normal(mean, deviation));
            synthetic.push(true);
        }
    }
    Ok((output, synthetic))
}

fn fill_time_grid<R: GaussianNoiseSource>(
    input: GapFillSeries,
    spacing: f64,
    tolerance: f64,
    mean: f64,
    deviation: f64,
    noise: &mut R,
) -> (GapFillSeries, Vec<bool>) {
    let mut output = GapFillSeries {
        time: vec![input.time[0]],
        flux: vec![input.flux[0]],
        cadence: None,
    };
    let mut synthetic = vec![false];
    for index in 1..input.time.len() {
        while input.time[index] - *output.time.last().unwrap() > tolerance * spacing {
            let next_time = output.time.last().unwrap() + spacing;
            output.time.push(next_time);
            output.flux.push(noise.normal(mean, deviation));
            synthetic.push(true);
        }
        output.time.push(input.time[index]);
        output.flux.push(input.flux[index]);
        synthetic.push(false);
    }
    (output, synthetic)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    struct ReplayNoise {
        values: VecDeque<f64>,
    }

    impl GaussianNoiseSource for ReplayNoise {
        fn normal(&mut self, _mean: f64, _standard_deviation: f64) -> f64 {
            self.values.pop_front().expect("replay noise exhausted")
        }
    }

    /// Mirrors ExoNitContractTests.testGapFillSemantics (main.swift:1724),
    /// cadence-grid branch.
    #[test]
    fn cadence_aware_gap_fill_matches_swift_contract_test() {
        let mut noise = ReplayNoise {
            values: VecDeque::from([999.0]),
        };
        let input = GapFillSeries {
            time: vec![1.0, 2.0, 4.0, 5.0],
            flux: vec![100.0, 101.0, 103.0, 104.0],
            cadence: Some(vec![10, 11, 13, 14]),
        };
        let filled = fill_if_allowed(input, 5.0, 1.2, None, &mut noise).unwrap();
        assert_eq!(filled.time, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(filled.cadence, Some(vec![10, 11, 12, 13, 14]));
        assert_eq!(filled.flux, vec![100.0, 101.0, 999.0, 103.0, 104.0]);
    }

    /// Mirrors ExoNitContractTests.testGapFillSemantics, time-grid branch.
    #[test]
    fn time_grid_gap_fill_matches_swift_contract_test() {
        let mut noise = ReplayNoise {
            values: VecDeque::from([777.0]),
        };
        let input = GapFillSeries {
            time: vec![0.0, 1.0, 2.0, 4.0],
            flux: vec![1.0, 1.0, 1.0, 1.0],
            cadence: None,
        };
        let filled = fill_if_allowed(input, 5.0, 1.2, None, &mut noise).unwrap();
        assert_eq!(filled.time, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(filled.flux, vec![1.0, 1.0, 1.0, 777.0, 1.0]);
    }

    /// Mirrors ExoNitContractTests.testGapFillSemantics, gate branch: a
    /// global gap of exactly `maximumGlobalGapDays` must NOT trigger a fill
    /// (strict `<` comparison), and must consume no randomness.
    #[test]
    fn global_gap_of_exactly_five_days_disables_fill() {
        let mut noise = ReplayNoise {
            values: VecDeque::from([123.0]),
        };
        let input = GapFillSeries {
            time: vec![0.0, 1.0, 6.0],
            flux: vec![1.0, 1.0, 1.0],
            cadence: None,
        };
        let gated = fill_if_allowed(input.clone(), 5.0, 1.2, None, &mut noise).unwrap();
        assert_eq!(gated, input);
        assert_eq!(noise.values.len(), 1, "gate must consume no randomness");
    }

    /// Pinned against worker/gap_fill.py (tests/test_gap_fill_determinism.py).
    #[test]
    fn seed_and_stream_match_python() {
        assert_eq!(gap_fill_seed_key("4206066", &[2, 1, 2]), "4206066:1,2");
        assert_eq!(gap_fill_seed_key("5", &[]), "5:");
        assert_eq!(gap_fill_seed("4206066", &[1, 2]), 717703856);
        assert_eq!(gap_fill_seed("5", &[]), 3031338238);
        let mut n = SeededGaussianNoiseSource::new(717703856);
        assert_eq!(n.normal(0.0, 1.0), 1.8320592014444887);
        assert_eq!(n.normal(0.0, 1.0), -0.5550678836244881);
        assert_eq!(n.normal(0.0, 1.0), 1.023046620261906);
        let mut m = SeededGaussianNoiseSource::new(3031338238);
        assert_eq!(m.normal(0.0, 1.0), -0.870363793639152);
    }

    #[test]
    fn seeded_fill_is_repeatable() {
        let input = GapFillSeries {
            time: vec![0.0, 1.0, 2.0, 4.5, 5.0],
            flux: vec![1.0, 1.1, 0.9, 1.0, 1.0],
            cadence: None,
        };
        let a = fill_if_allowed(input.clone(), 5.0, 1.2, None, &mut SeededGaussianNoiseSource::new(9)).unwrap();
        let b = fill_if_allowed(input, 5.0, 1.2, None, &mut SeededGaussianNoiseSource::new(9)).unwrap();
        assert_eq!(a, b);
    }

    /// Python's time-grid branch appends a duplicate timestamp with no filler
    /// (Swift GapFiller accepts `>= 0`); only a descending pair is an error.
    #[test]
    fn duplicate_timestamp_is_tolerated_descending_rejected() {
        let mut noise = ReplayNoise { values: VecDeque::new() };
        let input = GapFillSeries {
            time: vec![1.0, 2.0, 2.0, 3.0],
            flux: vec![1.0, 2.0, 2.5, 3.0],
            cadence: None,
        };
        let filled = fill_if_allowed(input, 5.0, 1.2, None, &mut noise).unwrap();
        assert_eq!(filled.time, vec![1.0, 2.0, 2.0, 3.0]);
        let mut noise = ReplayNoise { values: VecDeque::new() };
        let input = GapFillSeries { time: vec![1.0, 3.0, 2.0], flux: vec![1.0, 3.0, 2.0], cadence: None };
        assert_eq!(
            fill_if_allowed(input, 5.0, 1.2, None, &mut noise).unwrap_err(),
            GapFillError::NonMonotonicTime
        );
    }
}
