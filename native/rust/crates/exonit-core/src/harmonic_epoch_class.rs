//! Epoch-class (parity) alias check for the harmonic resolver.
//!
//! Transliterated from `worker/harmonic_epoch_class.py` (branch
//! `worktree-alias-2p-fix` of the ExoNit repo; spec: its
//! `docs/reviews/2026-10-14-alias-2p-phase2-report.md` §2 and §7).
//!
//! Numeric contract (bit-exact vs the Python reference, enforced by
//! `tests/harmonic_epoch_class_golden.rs`):
//!
//! * everything is `f64`; `time` is sorted ascending;
//! * every sum is a plain sequential ascending-index loop (`acc += x[i]`,
//!   accumulator starts at `0.0`) — this is exactly what `np.bincount(...,
//!   weights=)` does in the reference. No pairwise/blocked/Kahan sums and no
//!   iterator `.sum()` (std makes no ordering promise there);
//! * medians are sort-based: `s[n/2]` (odd) or `(s[n/2-1] + s[n/2]) * 0.5`;
//! * `rint` is round-half-to-even (`f64::round_ties_even`);
//! * NO `mul_add`, no fast-math. rustc never contracts `a * b + c` into an
//!   FMA on its own, so the expressions below keep the Python evaluation
//!   order verbatim (`((t_ref + es*period) + b) + a*es`, ...);
//! * Python `max(a, b)` / `min(a, b)` return `b` only if strictly greater /
//!   smaller; `py_max` / `py_min` reproduce that.
//!
//! Rollout: behind the worker config flag `harmonic_epoch_class` (default
//! off -> the resolver and the protected slots are bit-identical to before).

use crate::bls::{BlsParams, BlsResult, BlsSearchFn};

// --- thresholds (identical values to the Python module) ----------------------
pub const EPOCH_K_MIN: i64 = 2;
pub const EPOCH_K_MAX: i64 = 5;
pub const EPOCH_WINDOW_HALF_FRAC: f64 = 0.5;
pub const EPOCH_MIN_POINTS: i64 = 3;
pub const EPOCH_MIN_COVER_FRAC: f64 = 0.5;
pub const MIN_EPOCHS_PER_CLASS: i64 = 2;
pub const EPOCH_DIP_SIGMA: f64 = 2.0;
pub const MIN_DIP_EPOCHS: i64 = 2;
pub const PROMOTE_CARRY_SIGMA: f64 = 5.0;
pub const PROMOTE_EMPTY_SIGMA: f64 = 2.0;
pub const PROMOTE_MAX_PASSES: usize = 2;
pub const PROMOTE_REFIT_FRAC: f64 = 0.05;
pub const PROMOTE_REFIT_N_PERIODS: usize = 500;
pub const PROMOTE_MIN_TRANSITS: f64 = 2.0;
pub const VETO_CLASS_SIGMA: f64 = 3.0;
pub const DUP_PERIOD_TOL: f64 = 0.01;
pub const DUP_EPOCH_DUR_FRAC: f64 = 1.5;
pub const DUP_EPOCH_MIN_DAYS: f64 = 0.05;
pub const DRIFT_REL_P: f64 = 1e-4;
pub const DRIFT_MAX_DUR_MULT: f64 = 2.0;
pub const DRIFT_STEP_FRAC: f64 = 0.25;
pub const DRIFT_T0_STEPS: i64 = 2;
pub const DRIFT_MIN_POINTS: usize = 3;
pub const BOX_CENTRE_FRAC: f64 = 0.5;
pub const VH_GRID_REL_P: f64 = 2.0e-4;
const SIGMA_FLOOR: f64 = 1e-12;

/// Python `max(a, b)`: `b` only when strictly greater.
#[inline]
fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python `min(a, b)`: `b` only when strictly smaller.
#[inline]
fn py_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// Normalise a candidate t0 to the transit CENTRE (BLS backends return the
/// box START; `duration_refine` rewrites it to the centre).
pub fn transit_centre(t0: f64, duration: f64, t0_is_centre: bool) -> f64 {
    if t0_is_centre {
        t0
    } else {
        t0 + BOX_CENTRE_FRAC * duration
    }
}

/// Sort-based median (`s[n/2]` or `(s[n/2-1] + s[n/2]) * 0.5`). Empty -> NaN.
fn median(x: &[f64]) -> f64 {
    let mut s: Vec<f64> = x.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n == 0 {
        return f64::NAN;
    }
    let h = n / 2;
    if n % 2 == 1 {
        s[h]
    } else {
        (s[h - 1] + s[h]) * 0.5
    }
}

/// Per-epoch depths at the refined ephemeris (`prepare_epochs` output).
#[derive(Debug, Clone, PartialEq)]
pub struct EpochPrep {
    pub ep0: i64,
    pub n_e: Vec<f64>,
    pub d_e: Vec<f64>,
    pub covered: Vec<bool>,
    pub sigma: f64,
    pub period: f64,
    pub centre: f64,
}

/// Mirrors `prepare_epochs`. `excluded` marks multi-planet filler points.
pub fn prepare_epochs(
    time: &[f64],
    flux: &[f64],
    period: f64,
    t0: f64,
    duration: f64,
    excluded: Option<&[bool]>,
    t0_is_centre: bool,
) -> Option<EpochPrep> {
    let mut t: Vec<f64> = Vec::with_capacity(time.len());
    let mut y: Vec<f64> = Vec::with_capacity(time.len());
    for i in 0..time.len().min(flux.len()) {
        if !(time[i].is_finite() && flux[i].is_finite()) {
            continue;
        }
        if let Some(ex) = excluded {
            if ex[i] {
                continue;
            }
        }
        t.push(time[i]);
        y.push(flux[i]);
    }
    if t.len() < 10 || !(period > 0.0) || !(duration > 0.0) {
        return None;
    }
    let n = t.len();
    let reference = median(&y);
    let abs_dev: Vec<f64> = y.iter().map(|&v| (v - reference).abs()).collect();
    let sigma = py_max(1.4826 * median(&abs_dev), SIGMA_FLOOR);
    let diffs: Vec<f64> = (0..n - 1).map(|i| t[i + 1] - t[i]).collect();
    let dt_med = median(&diffs);
    let half = EPOCH_WINDOW_HALF_FRAC * duration;
    let expected = if dt_med > 0.0 { (2.0 * half) / dt_med } else { 1.0 };
    let need = py_max(EPOCH_MIN_POINTS as f64, EPOCH_MIN_COVER_FRAC * expected);

    let centre0 = transit_centre(t0, duration, t0_is_centre);
    let tmid = 0.5 * (t[0] + t[n - 1]);
    let t_ref = centre0 + ((tmid - centre0) / period).round_ties_even() * period;
    let ep: Vec<f64> = t.iter().map(|&v| ((v - t_ref) / period).round_ties_even()).collect();

    // --- drift refinement ---------------------------------------------------
    let span = t[n - 1] - t[0];
    let w = py_min(DRIFT_REL_P * span, DRIFT_MAX_DUR_MULT * duration);
    let step = DRIFT_STEP_FRAC * duration;
    let j_max: i64 = if w > 0.0 { (w / step).ceil() as i64 } else { 0 };
    let n_edge = py_max(0.5 * span / period, 1.0);
    let b_max = DRIFT_T0_STEPS as f64 * step;
    let limit = half + w + b_max;
    let mut ts: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    let mut es: Vec<f64> = Vec::new();
    for i in 0..n {
        if (t[i] - (t_ref + ep[i] * period)).abs() < limit {
            ts.push(t[i]);
            ys.push(y[i]);
            es.push(ep[i]);
        }
    }
    let score = |a: f64, b: f64| -> Option<f64> {
        let mut sum = 0.0f64;
        let mut cnt: usize = 0;
        for i in 0..ts.len() {
            let c = t_ref + es[i] * period + b + a * es[i];
            if (ts[i] - c).abs() < half {
                sum += ys[i];
                cnt += 1;
            }
        }
        if cnt < DRIFT_MIN_POINTS {
            return None;
        }
        Some(reference - sum / cnt as f64)
    };
    let mut best_a = 0.0f64;
    let mut best_b = 0.0f64;
    let mut best_s = score(0.0, 0.0);
    if j_max > 0 && !ts.is_empty() {
        for j in -j_max..=j_max {
            let a = j as f64 * step / n_edge;
            for i in -DRIFT_T0_STEPS..=DRIFT_T0_STEPS {
                if j == 0 && i == 0 {
                    continue;
                }
                let b = i as f64 * step;
                if let Some(s) = score(a, b) {
                    if best_s.map_or(true, |bs| s > bs) {
                        best_s = Some(s);
                        best_a = a;
                        best_b = b;
                    }
                }
            }
        }
    }
    let p2 = period + best_a;
    let c2 = t_ref + best_b;

    // --- per-epoch depths at the refined ephemeris --------------------------
    let mut ep_in: Vec<i64> = Vec::new();
    let mut y_in: Vec<f64> = Vec::new();
    for i in 0..n {
        let e = ((t[i] - c2) / p2).round_ties_even() as i64;
        if (t[i] - (c2 + e as f64 * p2)).abs() < half {
            ep_in.push(e);
            y_in.push(y[i]);
        }
    }
    if ep_in.is_empty() {
        return None;
    }
    let ep0 = *ep_in.iter().min().unwrap();
    let m = (*ep_in.iter().max().unwrap() - ep0) as usize + 1;
    let mut n_e = vec![0.0f64; m];
    let mut s_e = vec![0.0f64; m];
    for (k, &e) in ep_in.iter().enumerate() {
        let idx = (e - ep0) as usize;
        n_e[idx] += 1.0;
        s_e[idx] += y_in[k] - reference;
    }
    let covered: Vec<bool> = n_e.iter().map(|&c| c >= need).collect();
    let mut d_e = vec![0.0f64; m];
    for i in 0..m {
        if covered[i] {
            d_e[i] = -s_e[i] / n_e[i];
        }
    }
    Some(EpochPrep { ep0, n_e, d_e, covered, sigma, period: p2, centre: c2 })
}

/// Per-class statistics (`class_stats` output). NaN depth/err for classes
/// without covered epochs.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassStats {
    pub depth: Vec<f64>,
    pub err: Vec<f64>,
    pub nep: Vec<i64>,
    pub ndip: Vec<i64>,
}

/// Mirrors `class_stats(prep, k)`.
pub fn class_stats(prep: Option<&EpochPrep>, k: usize) -> ClassStats {
    let prep = match prep {
        None => {
            return ClassStats {
                depth: vec![f64::NAN; k],
                err: vec![f64::NAN; k],
                nep: vec![0; k],
                ndip: vec![0; k],
            }
        }
        Some(p) => p,
    };
    let m = prep.n_e.len();
    let ki = k as i64;
    let cls: Vec<usize> = (0..m).map(|e| (prep.ep0 + e as i64).rem_euclid(ki) as usize).collect();
    let mut nep = vec![0i64; k];
    let mut ndip = vec![0i64; k];
    let mut sum_d = vec![0.0f64; k];
    let mut sum_n = vec![0.0f64; k];
    for e in 0..m {
        if !prep.covered[e] {
            continue;
        }
        let c = cls[e];
        nep[c] += 1;
        sum_d[c] += prep.d_e[e];
        sum_n[c] += prep.n_e[e];
        if prep.d_e[e] >= EPOCH_DIP_SIGMA * prep.sigma / prep.n_e[e].sqrt() {
            ndip[c] += 1;
        }
    }
    let mut depth = vec![f64::NAN; k];
    let mut err_white = vec![f64::NAN; k];
    for c in 0..k {
        if nep[c] > 0 {
            depth[c] = sum_d[c] / nep[c] as f64;
            err_white[c] = prep.sigma / sum_n[c].sqrt();
        }
    }
    let mut sum_dev2 = vec![0.0f64; k];
    for e in 0..m {
        if !prep.covered[e] {
            continue;
        }
        let c = cls[e];
        let dev = prep.d_e[e] - depth[c];
        sum_dev2[c] += dev * dev;
    }
    let mut err = err_white.clone();
    for c in 0..k {
        if nep[c] >= 2 {
            let err_sc = ((sum_dev2[c] / (nep[c] - 1) as f64) / nep[c] as f64).sqrt();
            // np.maximum
            err[c] = if err_sc > err_white[c] { err_sc } else { err_white[c] };
        }
    }
    ClassStats { depth, err, nep, ndip }
}

/// `prepare_epochs` + `class_stats`.
#[allow(clippy::too_many_arguments)]
pub fn epoch_class_stats(
    time: &[f64],
    flux: &[f64],
    period: f64,
    t0: f64,
    duration: f64,
    k: usize,
    t0_is_centre: bool,
    excluded: Option<&[bool]>,
) -> ClassStats {
    let prep = prepare_epochs(time, flux, period, t0, duration, excluded, t0_is_centre);
    class_stats(prep.as_ref(), k)
}

fn carry(s: &ClassStats, sigma_thr: f64) -> (Vec<bool>, Vec<f64>) {
    let z: Vec<f64> = (0..s.depth.len())
        .map(|c| {
            // np.maximum(err, floor)
            let e = if SIGMA_FLOOR > s.err[c] { SIGMA_FLOOR } else { s.err[c] };
            s.depth[c] / e
        })
        .collect();
    let carries = (0..z.len()).map(|c| z[c] >= sigma_thr && s.ndip[c] >= MIN_DIP_EPOCHS).collect();
    (carries, z)
}

fn promotion_pattern(s: &ClassStats) -> bool {
    let (carries, z) = carry(s, PROMOTE_CARRY_SIGMA);
    if carries.iter().filter(|&&c| c).count() != 1 {
        return false;
    }
    (0..z.len()).all(|c| carries[c] || z[c].abs() < PROMOTE_EMPTY_SIGMA)
}

fn all_carry(s: &ClassStats) -> bool {
    carry(s, VETO_CLASS_SIGMA).0.iter().all(|&c| c)
}

fn decidable(s: &ClassStats) -> bool {
    s.nep.iter().all(|&n| n >= MIN_EPOCHS_PER_CLASS)
}

/// Mirrors `epoch_class_promotion_ks` (LARGEST k first).
#[allow(clippy::too_many_arguments)]
pub fn epoch_class_promotion_ks(
    time: &[f64],
    flux: &[f64],
    period: f64,
    t0: f64,
    duration: f64,
    period_max: f64,
    t0_is_centre: bool,
    excluded: Option<&[bool]>,
) -> Vec<i64> {
    let prep = prepare_epochs(time, flux, period, t0, duration, excluded, t0_is_centre);
    let mut out = Vec::new();
    for k in EPOCH_K_MIN..=EPOCH_K_MAX {
        if period * k as f64 > period_max {
            break;
        }
        let s = class_stats(prep.as_ref(), k as usize);
        if !decidable(&s) {
            continue;
        }
        if promotion_pattern(&s) {
            out.push(k);
        }
    }
    out.reverse();
    out
}

/// Mirrors `promote_epoch_class`. Returns `None` when nothing was promoted
/// (Python returns the input object itself in that case).
pub fn promote_epoch_class(
    time: &[f64],
    flux: &[f64],
    result: &BlsResult,
    period_max: f64,
    search: BlsSearchFn,
    snr_min: f64,
    excluded: Option<&[bool]>,
) -> Option<BlsResult> {
    let span = match (time.first(), time.last()) {
        (Some(a), Some(b)) => b - a,
        _ => 0.0,
    };
    let mut cur: Option<BlsResult> = None;
    for _ in 0..PROMOTE_MAX_PASSES {
        let (p, dur, t0) = {
            let c = cur.as_ref().unwrap_or(result);
            (c.period, c.duration, c.t0)
        };
        if !(p > 0.0 && dur > 0.0) {
            break;
        }
        let mut promoted: Option<BlsResult> = None;
        let ks = epoch_class_promotion_ks(time, flux, p, t0, dur, period_max, false, excluded);
        for k in ks {
            let lo = p * k as f64 * (1.0 - PROMOTE_REFIT_FRAC);
            let hi = py_min(p * k as f64 * (1.0 + PROMOTE_REFIT_FRAC), period_max);
            if !(hi > lo) {
                continue;
            }
            let params = BlsParams {
                period_minimum: lo,
                period_maximum: hi,
                period_count: PROMOTE_REFIT_N_PERIODS,
                ..BlsParams::default()
            };
            let sub = match search(time, flux, &params) {
                Ok(s) => s,
                Err(_) => continue,
            };
            // `!(>=)` also rejects a NaN SNR (Python's `<` would let it through).
            if !(sub.snr >= snr_min) {
                continue;
            }
            if span > 0.0 && sub.period > 0.0 && span / sub.period < PROMOTE_MIN_TRANSITS {
                continue;
            }
            // G1 consistency guard: all k classes carry at P_k/k -> the short
            // period is genuine, reject this k (try the next smaller one).
            if short_period_genuine(time, flux, &sub, k as usize, excluded) {
                continue;
            }
            promoted = Some(sub);
            break;
        }
        match promoted {
            None => break,
            Some(s) => cur = Some(s),
        }
    }
    cur
}

/// Mirrors `short_period_genuine` (G1): `sub` = accepted re-fit at ~k*P (raw
/// BLS result). True when every one of the k classes carries (veto rule) at
/// the ephemeris (sub.period / k, sub.t0, sub.duration); undecidable -> false.
pub fn short_period_genuine(
    time: &[f64],
    flux: &[f64],
    sub: &BlsResult,
    k: usize,
    excluded: Option<&[bool]>,
) -> bool {
    let p = sub.period;
    if !(p > 0.0) {
        return false;
    }
    let s = epoch_class_stats(time, flux, p / (k as f64), sub.t0, sub.duration, k, false, excluded);
    if !decidable(&s) {
        return false;
    }
    all_carry(&s)
}

/// Mirrors `demotion_supported` in its only mode "vh" (`sub` = raw BLS result at
/// P/k, t0 = box start). Undecidable -> demotion proceeds. The /k probe grid has a
/// constant relative step of 2e-4; when it is coarser than the BLS peak width
/// (span * 2e-4 > duration, i.e. span > 5000 * duration, strict) the probe
/// ephemeris can be smeared and only positive alias evidence vetoes ("v1": the
/// promotion pattern); otherwise veto unless every class carries ("v0").
/// span = last - first of the raw array (empty -> 0; NaN -> comparison false -> v0).
pub fn demotion_supported(
    time: &[f64],
    flux: &[f64],
    sub: &BlsResult,
    k: usize,
    excluded: Option<&[bool]>,
) -> bool {
    let s = epoch_class_stats(time, flux, sub.period, sub.t0, sub.duration, k, false, excluded);
    if !decidable(&s) {
        return true;
    }
    let span = match (time.first(), time.last()) {
        (Some(a), Some(b)) => *b - *a,
        _ => 0.0,
    };
    if span * VH_GRID_REL_P > sub.duration {
        !promotion_pattern(&s)
    } else {
        all_carry(&s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parity {
    Alias,
    Genuine,
    Undecided,
}

/// Mirrors `harmonic_parity`.
#[allow(clippy::too_many_arguments)]
pub fn harmonic_parity(
    time: &[f64],
    flux: &[f64],
    period: f64,
    t0: f64,
    duration: f64,
    k: usize,
    excluded: Option<&[bool]>,
    t0_is_centre: bool,
) -> Parity {
    let s = epoch_class_stats(time, flux, period, t0, duration, k, t0_is_centre, excluded);
    if !decidable(&s) {
        return Parity::Undecided;
    }
    if promotion_pattern(&s) {
        return Parity::Alias;
    }
    if all_carry(&s) {
        return Parity::Genuine;
    }
    Parity::Undecided
}

/// An accepted candidate as seen by the slot-duplicate test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochCandidate {
    pub period_days: f64,
    pub t0: f64,
    pub duration_days: f64,
}

/// Mirrors `find_harmonic_epoch_match`; returns the index in `accepted`.
pub fn find_harmonic_epoch_match(
    period: f64,
    t0: f64,
    duration: f64,
    accepted: &[EpochCandidate],
    t_first: f64,
    t_last: f64,
    t0_is_centre: bool,
) -> Option<usize> {
    if !(period > 0.0) {
        return None;
    }
    let tc = transit_centre(t0, duration, t0_is_centre);
    let xs = [t_first, 0.5 * (t_first + t_last), t_last];
    for (idx, c) in accepted.iter().enumerate() {
        let pa = c.period_days;
        if !(pa > 0.0) {
            continue;
        }
        let da = c.duration_days;
        let ta = transit_centre(c.t0, da, t0_is_centre);
        let (ps, ts, pl, tl) = if period < pa { (period, tc, pa, ta) } else { (pa, ta, period, tc) };
        let r = pl / ps;
        let k = r.round_ties_even();
        if k < EPOCH_K_MIN as f64 || k > EPOCH_K_MAX as f64 {
            continue;
        }
        if (r - k).abs() / k >= DUP_PERIOD_TOL {
            continue;
        }
        let tol = py_max(DUP_EPOCH_DUR_FRAC * py_max(duration, da), DUP_EPOCH_MIN_DAYS);
        let mut ok = true;
        for &x in &xs {
            let t_long = tl + ((x - tl) / pl).round_ties_even() * pl;
            let t_short = ts + ((t_long - ts) / ps).round_ties_even() * ps;
            if (t_long - t_short).abs() > tol {
                ok = false;
                break;
            }
        }
        if ok {
            return Some(idx);
        }
    }
    None
}

/// Mirrors `slot_duplicate`: true = DROP the protected-slot candidate.
#[allow(clippy::too_many_arguments)]
pub fn slot_duplicate(
    time: &[f64],
    flux: &[f64],
    rm_period: f64,
    rm_t0: f64,
    rm_duration: f64,
    accepted: &[EpochCandidate],
    excluded: Option<&[bool]>,
    t0_is_centre: bool,
) -> bool {
    let fin: Vec<f64> = time.iter().copied().filter(|v| v.is_finite()).collect();
    if fin.len() < 2 || !(rm_period > 0.0) {
        return false;
    }
    let idx = match find_harmonic_epoch_match(
        rm_period,
        rm_t0,
        rm_duration,
        accepted,
        fin[0],
        fin[fin.len() - 1],
        t0_is_centre,
    ) {
        None => return false,
        Some(i) => i,
    };
    let c = &accepted[idx];
    let pa = c.period_days;
    if rm_period < pa {
        let k = (pa / rm_period).round_ties_even() as usize;
        let v = harmonic_parity(time, flux, rm_period, rm_t0, rm_duration, k, excluded, t0_is_centre);
        return v != Parity::Genuine;
    }
    let k = (rm_period / pa).round_ties_even() as usize;
    let v = harmonic_parity(time, flux, pa, c.t0, c.duration_days, k, excluded, t0_is_centre);
    v != Parity::Alias
}
