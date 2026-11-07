//! Ephemeris refinement before FPP (FPP_EPHEM_REFINE, section 8b).
//!
//! Port of `worker/fpp_native.py` lines 4557-4900 and 5751-5784 (commit `03a4bcc`).
//! Pure function of its inputs, deterministic, no RNG, no clock, no global state.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const EPHEM_T14_FACTORS: [f64; 9] = [0.5, 0.65, 0.8, 1.0, 1.2, 1.45, 1.75, 2.1, 2.5];
pub const EPHEM_INGRESS: [f64; 3] = [0.08, 0.2, 0.4];
pub const EPHEM_T0_HALF: f64 = 1.5;
pub const EPHEM_REACH_MAX_P: f64 = 0.25;
pub const EPHEM_DRIFT_HALF: f64 = 1.5;
pub const EPHEM_P_ROUND: f64 = 1e-4;
pub const EPHEM_P_FRAC_MAX: f64 = 0.02;
pub const EPHEM_N_COARSE: usize = 41;
pub const EPHEM_N_ZOOM: usize = 31;
pub const EPHEM_N_ZOOMS: usize = 3;
pub const EPHEM_ZOOM: f64 = 5.0;
pub const EPHEM_EDGE_CELLS: usize = 2;
pub const EPHEM_MIN_DCHI2: f64 = 25.0;
pub const EPHEM_P_FREE_DCHI2: f64 = 9.0;
pub const EPHEM_MAX_FREE_SHIFT: f64 = 0.25;
pub const EPHEM_COVER_POINTS: usize = 3;
pub const EPHEM_MIN_POINTS: usize = 100;
pub const EPHEM_MIN_WINDOW_POINTS: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum T0Convention {
    BoxStart,
    Centre,
}

impl std::str::FromStr for T0Convention {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "box_start" => Ok(T0Convention::BoxStart),
            "centre" => Ok(T0Convention::Centre),
            other => Err(format!(
                "invalid t0_convention: {other:?}, expected 'box_start' or 'centre'"
            )),
        }
    }
}

/// Running integral from -inf to x of the unit trapezoid of half-width hw and ingress length L (0 < L <= hw).
#[inline]
pub fn ephem_cum(x: f64, hw: f64, l: f64) -> f64 {
    let y = if x > 0.0 { -x } else { x };
    let c = if y <= -hw {
        0.0
    } else if y <= l - hw {
        let u = y + hw;
        u * u / (2.0 * l)
    } else {
        0.5 * l + (y + hw - l)
    };
    if x > 0.0 {
        (2.0 * hw - l) - c
    } else {
        c
    }
}

/// Best trapezoid-dip dchi2 at one (P, tc) on the unfolded data; returns (dchi2, shape index, n in window).
#[inline]
pub fn ephem_dchi2(
    t: &[f64],
    r: &[f64],
    period: f64,
    tc: f64,
    t14: &[f64],
    ing: &[f64],
    w: f64,
    inv_s2: f64,
    min_pts: usize,
    smr: &mut [f64],
    smm: &mut [f64],
) -> (f64, i32, usize) {
    let ns = t14.len();
    let ng = ing.len();
    for s in 0..ns * ng {
        smr[s] = 0.0;
        smm[s] = 0.0;
    }
    let hw_exp = if w > 0.0 { 0.5 * w } else { 0.0 };
    let half = 0.5 * t14[ns - 1] + hw_exp;
    let mut npts = 0usize;

    for i in 0..t.len() {
        let x = t[i] - tc + 0.5 * period;
        let dt = x - period * (x / period).floor() - 0.5 * period;
        let a = dt.abs();
        if a >= half {
            continue;
        }
        npts += 1;
        for k in 0..ns {
            let hw = 0.5 * t14[k];
            if a >= hw + hw_exp {
                continue;
            }
            for g in 0..ng {
                let m = if w > 0.0 {
                    let l = ing[g] * t14[k];
                    (ephem_cum(a + hw_exp, hw, l) - ephem_cum(a - hw_exp, hw, l)) / w
                } else {
                    let mut m_val = (hw - a) / (ing[g] * t14[k]);
                    if m_val > 1.0 {
                        m_val = 1.0;
                    }
                    m_val
                };
                smr[k * ng + g] += m * r[i];
                smm[k * ng + g] += m * m;
            }
        }
    }

    let mut best = 0.0;
    let mut bi = -1i32;
    if npts >= min_pts {
        for s in 0..ns * ng {
            if smm[s] > 0.0 && smr[s] > 0.0 {
                let d = smr[s] * smr[s] / smm[s] * inv_s2;
                if d > best {
                    best = d;
                    bi = s as i32;
                }
            }
        }
    }
    (best, bi, npts)
}

/// Grid around (P0, tc0); returns (best i, best j, best dchi2, shape, dchi2 at centre, n in window).
pub fn ephem_grid(
    t: &[f64],
    r: &[f64],
    p0: f64,
    step_p: f64,
    n_p: usize,
    tc0: f64,
    step_t: f64,
    nt: usize,
    t14: &[f64],
    ing: &[f64],
    w: f64,
    inv_s2: f64,
    min_pts: usize,
) -> (usize, usize, f64, i32, f64, usize) {
    let mut smr = [0.0; 27];
    let mut smm = [0.0; 27];
    let ci = (n_p - 1) / 2;
    let cj = (nt - 1) / 2;
    let mut bi = ci;
    let mut bj = cj;
    let mut best = -1.0;
    let mut bshape = -1i32;
    let mut centre = 0.0;
    let mut bn = 0usize;

    for i in 0..n_p {
        let p = p0 + (i as isize - ci as isize) as f64 * step_p;
        for j in 0..nt {
            let tc = tc0 + (j as isize - cj as isize) as f64 * step_t;
            let (d, sh, n) = ephem_dchi2(
                t, r, p, tc, t14, ing, w, inv_s2, min_pts, &mut smr, &mut smm,
            );
            if i == ci && j == cj {
                centre = d;
            }
            if d > best {
                best = d;
                bi = i;
                bj = j;
                bshape = sh;
                bn = n;
            }
        }
    }
    (bi, bj, best, bshape, centre, bn)
}

/// Median of a float64 array: ascending stable sort, middle element or mean of the two middle ones.
pub fn ephem_median(x: &[f64]) -> f64 {
    let mut s = x.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n == 0 {
        return f64::NAN;
    }
    let h = n / 2;
    if n % 2 == 1 {
        s[h]
    } else {
        0.5 * (s[h - 1] + s[h])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomStep {
    pub zoom_index: usize,
    pub zi: usize,
    pub zj: usize,
    pub zshape: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub p: f64,
    pub t: f64,
    pub best: f64,
    pub shape: i32,
    pub centre: f64,
    pub edge_p: bool,
    pub edge_t: bool,
    pub coarse_bi: usize,
    pub coarse_bj: usize,
    pub coarse_shape: i32,
    pub zooms: Vec<ZoomStep>,
}

/// Coarse grid + nested zooms around (P0, tref0); P varies only when p_free.
pub fn ephem_search(
    ts: &[f64],
    rs: &[f64],
    p0: f64,
    dp_half: f64,
    tref0: f64,
    dt_half: f64,
    t14: &[f64],
    ing: &[f64],
    w: f64,
    inv_s2: f64,
    p_free: bool,
) -> SearchResult {
    let n_p = if p_free { EPHEM_N_COARSE } else { 1 };
    let nt = EPHEM_N_COARSE;
    let step_p = if p_free {
        2.0 * dp_half / (n_p - 1) as f64
    } else {
        0.0
    };
    let step_t = 2.0 * dt_half / (nt - 1) as f64;
    let (bi, bj, mut best, mut bshape, centre, _bn) = ephem_grid(
        ts, rs, p0, step_p, n_p, tref0, step_t, nt, t14, ing, w, inv_s2, EPHEM_MIN_WINDOW_POINTS,
    );
    let coarse_bi = bi;
    let coarse_bj = bj;
    let coarse_shape = bshape;
    let ci = (n_p - 1) / 2;
    let cj = (nt - 1) / 2;
    let mut cp = p0 + (bi as isize - ci as isize) as f64 * step_p;
    let mut ct = tref0 + (bj as isize - cj as isize) as f64 * step_t;
    let edge_p = p_free && (bi < EPHEM_EDGE_CELLS || bi > n_p - 1 - EPHEM_EDGE_CELLS);
    let edge_t = bj < EPHEM_EDGE_CELLS || bj > nt - 1 - EPHEM_EDGE_CELLS;
    let mut hp = dp_half;
    let mut ht = dt_half;
    let mut zooms = Vec::with_capacity(EPHEM_N_ZOOMS);

    for z in 0..EPHEM_N_ZOOMS {
        hp /= EPHEM_ZOOM;
        ht /= EPHEM_ZOOM;
        let n_pz = if p_free { EPHEM_N_ZOOM } else { 1 };
        let sp_z = if p_free {
            2.0 * hp / (n_pz - 1) as f64
        } else {
            0.0
        };
        let st_z = 2.0 * ht / (EPHEM_N_ZOOM - 1) as f64;
        let (zi, zj, zb, zs, _zc, _zn) = ephem_grid(
            ts, rs, cp, sp_z, n_pz, ct, st_z, EPHEM_N_ZOOM, t14, ing, w, inv_s2, EPHEM_MIN_WINDOW_POINTS,
        );
        zooms.push(ZoomStep {
            zoom_index: z,
            zi,
            zj,
            zshape: zs,
        });
        if zb > best {
            best = zb;
            bshape = zs;
            cp += (zi as isize - ((n_pz - 1) / 2) as isize) as f64 * sp_z;
            ct += (zj as isize - ((EPHEM_N_ZOOM - 1) / 2) as isize) as f64 * st_z;
        }
    }

    SearchResult {
        p: cp,
        t: ct,
        best,
        shape: bshape,
        centre,
        edge_p,
        edge_t,
        coarse_bi,
        coarse_bj,
        coarse_shape,
        zooms,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefineEphemerisResult {
    pub status: String,
    pub reason: Option<String>,
    pub p: f64,
    pub t0: f64,
    pub p_in: f64,
    pub t0_in: f64,
    pub t_ref_in: f64,
    pub cadence_s: Option<f64>,
    pub n_req: Option<usize>,
    pub n_orbits: Option<f64>,
    pub n_transits: usize,
    pub p_searched: bool,
    pub p_free_accepted: bool,
    pub delta_chi2_p: Option<f64>,
    pub dchi2_p_fixed: Option<f64>,
    pub dchi2_p_free: Option<f64>,
    pub p_fixed: Option<bool>,
    pub shift_h: f64,
    pub dp: f64,
    pub max_offset_h: f64,
    pub dchi2_initial: Option<f64>,
    pub dchi2_refined: Option<f64>,
    pub delta_chi2: Option<f64>,
    pub sigma: Option<f64>,
    pub baseline: Option<f64>,
    pub t14_factor: Option<f64>,
    pub ingress: Option<f64>,
    pub edge_p: bool,
    pub edge_t0: bool,
    pub dp_half: Option<f64>,
    pub dt_half: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchIndicesTrace {
    pub source_search: String,
    pub coarse_bi: Option<usize>,
    pub coarse_bj: Option<usize>,
    pub coarse_shape: Option<i32>,
    pub zooms: Vec<ZoomStep>,
    pub best_shape: Option<i32>,
    pub best_t14_factor_index: Option<usize>,
    pub best_ingress_index: Option<usize>,
}

/// Bounded deterministic local (P, transit-centre) refinement on the unfolded light curve.
pub fn refine_ephemeris(
    time: &[f64],
    flux: &[f64],
    period: f64,
    t0_centre_guess: f64,
    duration: f64,
    cadence_s: Option<f64>,
) -> (RefineEphemerisResult, SearchIndicesTrace) {
    let p_in = period;
    let t0_in = t0_centre_guess;
    let d = duration;

    let mut out = RefineEphemerisResult {
        status: "failed".to_string(),
        reason: None,
        p: p_in,
        t0: t0_in,
        p_in,
        t0_in,
        t_ref_in: t0_in,
        cadence_s: None,
        n_req: None,
        n_orbits: None,
        n_transits: 0,
        p_searched: false,
        p_free_accepted: false,
        delta_chi2_p: None,
        dchi2_p_fixed: None,
        dchi2_p_free: None,
        p_fixed: None,
        shift_h: 0.0,
        dp: 0.0,
        max_offset_h: 0.0,
        dchi2_initial: None,
        dchi2_refined: None,
        delta_chi2: None,
        sigma: None,
        baseline: None,
        t14_factor: None,
        ingress: None,
        edge_p: false,
        edge_t0: false,
        dp_half: None,
        dt_half: None,
    };

    let empty_trace = SearchIndicesTrace {
        source_search: "none".to_string(),
        coarse_bi: None,
        coarse_bj: None,
        coarse_shape: None,
        zooms: Vec::new(),
        best_shape: None,
        best_t14_factor_index: None,
        best_ingress_index: None,
    };

    if !(p_in.is_finite() && p_in > 0.0 && t0_in.is_finite() && d.is_finite() && d > 0.0) {
        out.reason = Some("invalid_ephemeris".to_string());
        return (out, empty_trace);
    }

    let mut t = Vec::with_capacity(time.len());
    let mut f = Vec::with_capacity(flux.len());
    for (&ti, &fi) in time.iter().zip(flux.iter()) {
        if ti.is_finite() && fi.is_finite() {
            t.push(ti);
            f.push(fi);
        }
    }

    if t.len() < EPHEM_MIN_POINTS {
        out.reason = Some("too_few_points".to_string());
        return (out, empty_trace);
    }

    let cad_s = match cadence_s {
        Some(s) => s,
        None => {
            let mut sorted_t = t.clone();
            sorted_t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let diffs: Vec<f64> = sorted_t.windows(2).map(|w| w[1] - w[0]).collect();
            let med_diff = ephem_median(&diffs);
            (med_diff * 86400.0 * 1000.0).round() / 1000.0
        }
    };
    out.cadence_s = Some(cad_s);
    if !(cad_s.is_finite() && cad_s > 0.0) {
        out.reason = Some("cadence_unusable".to_string());
        return (out, empty_trace);
    }

    let w = cad_s / 86400.0;
    let hw = 0.5 * w;
    let tmin = t.iter().copied().fold(f64::INFINITY, f64::min);
    let tmax = t.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    out.n_orbits = Some((tmax - tmin) / p_in);

    let n_ref = ((0.5 * (tmin + tmax) - t0_in) / p_in + 0.5).floor();
    let tref0 = t0_in + n_ref * p_in;
    out.t_ref_in = tref0;

    let mut ep = Vec::with_capacity(t.len());
    let mut dt0 = Vec::with_capacity(t.len());
    for &ti in &t {
        let epi = ((ti - tref0) / p_in + 0.5).floor();
        let x = ti - tref0 + 0.5 * p_in;
        let dt0_i = x - p_in * (x / p_in).floor() - 0.5 * p_in;
        ep.push(epi);
        dt0.push(dt0_i);
    }

    let n_req = EPHEM_COVER_POINTS.min(1.max((d / w).floor() as usize));
    out.n_req = Some(n_req);

    let mut epoch_counts: BTreeMap<i64, usize> = BTreeMap::new();
    for i in 0..t.len() {
        if dt0[i].abs() < 0.5 * d {
            *epoch_counts.entry(ep[i] as i64).or_insert(0) += 1;
        }
    }
    let n_tr = epoch_counts.values().filter(|&&cnt| cnt >= n_req).count();
    out.n_transits = n_tr;
    if n_tr < 1 {
        out.reason = Some("no_transit_covered".to_string());
        return (out, empty_trace);
    }

    let n_max = ep.iter().map(|e| e.abs()).fold(0.0, f64::max);
    let p_search = n_tr >= 2 && n_max >= 1.0;
    out.p_searched = p_search;

    let t14: Vec<f64> = EPHEM_T14_FACTORS.iter().map(|&k| k * d).collect();
    let ing = EPHEM_INGRESS;
    let half_shape = 0.5 * t14[t14.len() - 1] + hw;
    let dt_half = (EPHEM_T0_HALF * d).min((EPHEM_REACH_MAX_P * p_in - half_shape) / 1.25);
    if !(dt_half > 0.0) {
        out.reason = Some("duration_too_long_for_window".to_string());
        return (out, empty_trace);
    }

    let dp_half = if p_search {
        let step_bls = 2e-5f64.max(4e-6 * p_in);
        let dp_drift = (EPHEM_DRIFT_HALF * d / n_max).max(step_bls + EPHEM_P_ROUND);
        dp_drift.min(EPHEM_P_FRAC_MAX * p_in)
    } else {
        0.0
    };
    out.dp_half = Some(dp_half);
    out.dt_half = Some(dt_half);

    let mut sub_idx = Vec::new();
    let mut oot_f = Vec::new();
    for i in 0..t.len() {
        let dt_abs = dt0[i].abs();
        let ep_abs = ep[i].abs();
        let reach = half_shape + 1.5 * (dt_half + ep_abs * dp_half);
        if dt_abs < reach {
            sub_idx.push(i);
        }
        let oot_bound = half_shape + dt_half + ep_abs * dp_half;
        if !(dt_abs < oot_bound) {
            oot_f.push(f[i]);
        }
    }

    let ref_f = if oot_f.len() >= EPHEM_MIN_POINTS {
        oot_f
    } else {
        f.clone()
    };
    let base = ephem_median(&ref_f);
    let dev: Vec<f64> = ref_f.iter().map(|&x| (x - base).abs()).collect();
    let sigma = 1.4826 * ephem_median(&dev);
    out.baseline = Some(base);
    out.sigma = Some(sigma);
    if !(sigma > 0.0 && sigma.is_finite()) {
        out.reason = Some("sigma_unusable".to_string());
        return (out, empty_trace);
    }

    let ts: Vec<f64> = sub_idx.iter().map(|&i| t[i]).collect();
    let rs: Vec<f64> = sub_idx.iter().map(|&i| base - f[i]).collect();
    let inv_s2 = 1.0 / (sigma * sigma);

    let fixed = ephem_search(
        &ts, &rs, p_in, 0.0, tref0, dt_half, &t14, &ing, w, inv_s2, false,
    );
    out.dchi2_p_fixed = Some(fixed.best);

    let mut best_s = fixed.clone();
    let mut free_opt = None;
    if p_search {
        let free = ephem_search(
            &ts, &rs, p_in, dp_half, tref0, dt_half, &t14, &ing, w, inv_s2, true,
        );
        let gain = free.best - fixed.best;
        out.dchi2_p_free = Some(free.best);
        out.delta_chi2_p = Some(gain);
        if gain >= EPHEM_P_FREE_DCHI2 {
            best_s = free.clone();
            out.p_free_accepted = true;
        }
        free_opt = Some(free);
    }

    let p_fixed = !out.p_free_accepted;
    out.p_fixed = Some(p_fixed);
    let centre = fixed.centre;
    let best = best_s.best;
    out.dchi2_initial = Some(centre);
    out.dchi2_refined = Some(best);
    out.delta_chi2 = Some(best - centre);

    let mut t14_factor_idx = None;
    let mut ingress_idx = None;
    if best_s.shape >= 0 {
        let s_idx = best_s.shape as usize;
        let n_ing = EPHEM_INGRESS.len();
        t14_factor_idx = Some(s_idx / n_ing);
        ingress_idx = Some(s_idx % n_ing);
        out.t14_factor = Some(EPHEM_T14_FACTORS[s_idx / n_ing]);
        out.ingress = Some(EPHEM_INGRESS[s_idx % n_ing]);
    }
    out.edge_p = best_s.edge_p;
    out.edge_t0 = best_s.edge_t;

    let source_search = if out.p_free_accepted { "free" } else { "fixed" };
    let active_search = if out.p_free_accepted {
        free_opt.as_ref().unwrap()
    } else {
        &fixed
    };

    let trace = SearchIndicesTrace {
        source_search: source_search.to_string(),
        coarse_bi: Some(active_search.coarse_bi),
        coarse_bj: Some(active_search.coarse_bj),
        coarse_shape: Some(active_search.coarse_shape),
        zooms: active_search.zooms.clone(),
        best_shape: if best_s.shape >= 0 { Some(best_s.shape) } else { None },
        best_t14_factor_index: t14_factor_idx,
        best_ingress_index: ingress_idx,
    };

    if best < EPHEM_MIN_DCHI2 {
        out.reason = Some("no_significant_dip".to_string());
        return (out, trace);
    }
    if best_s.edge_p || best_s.edge_t {
        out.status = "edge".to_string();
        out.reason = Some("optimum_at_search_edge".to_string());
        return (out, trace);
    }

    let cp = best_s.p;
    let ct = best_s.t;
    let dp = cp - p_in;
    let shift = ct - tref0;
    let ep_min = ep.iter().copied().fold(f64::INFINITY, f64::min);
    let ep_max = ep.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let max_off = (shift + ep_min * dp).abs().max((shift + ep_max * dp).abs());
    if max_off > EPHEM_MAX_FREE_SHIFT * d && (best - centre) < EPHEM_P_FREE_DCHI2 {
        out.max_offset_h = max_off * 24.0;
        out.reason = Some("insignificant_large_shift".to_string());
        return (out, trace);
    }

    out.status = "ok".to_string();
    out.p = cp;
    out.t0 = ct;
    out.dp = dp;
    out.shift_h = shift * 24.0;
    out.max_offset_h = max_off * 24.0;

    (out, trace)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefineProvenanceFields {
    pub reason: Option<String>,
    pub p_in: f64,
    pub t0_in: f64,
    pub t_ref_in: f64,
    pub cadence_s: Option<f64>,
    pub n_req: Option<usize>,
    pub n_orbits: Option<f64>,
    pub n_transits: usize,
    pub p_searched: bool,
    pub p_free_accepted: bool,
    pub delta_chi2_p: Option<f64>,
    pub dchi2_p_fixed: Option<f64>,
    pub dchi2_p_free: Option<f64>,
    pub p_fixed: Option<bool>,
    pub dp: f64,
    pub max_offset_h: f64,
    pub dchi2_initial: Option<f64>,
    pub dchi2_refined: Option<f64>,
    pub delta_chi2: Option<f64>,
    pub sigma: Option<f64>,
    pub t14_factor: Option<f64>,
    pub ingress: Option<f64>,
    pub edge_p: bool,
    pub edge_t0: bool,
    pub dp_half: Option<f64>,
    pub dt_half: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EphemProvenance {
    pub ephem_refined: bool,
    pub ephem_status: String,
    pub ephem_t0_convention: String,
    pub ephem_p: f64,
    pub ephem_t0: f64,
    pub ephem_shift_h: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ephem_refine: Option<RefineProvenanceFields>,
}

#[derive(Debug, Clone)]
pub struct FppEphemerisResult {
    pub p_use: f64,
    pub t0_use: f64,
    pub provenance: Option<EphemProvenance>,
    pub refine_failed: bool,
}

/// The (P, t0 centre) the FPP fold uses, plus provenance fields (section 8b).
pub fn fpp_ephemeris(
    time: &[f64],
    flux: &[f64],
    period: f64,
    t0: f64,
    duration: f64,
    convention: T0Convention,
    refine_enabled: bool,
    cadence_s: Option<f64>,
) -> FppEphemerisResult {
    if !refine_enabled {
        return FppEphemerisResult {
            p_use: period,
            t0_use: t0,
            provenance: None,
            refine_failed: false,
        };
    }
    match convention {
        T0Convention::Centre => FppEphemerisResult {
            p_use: period,
            t0_use: t0,
            provenance: Some(EphemProvenance {
                ephem_refined: false,
                ephem_status: "not_requested".to_string(),
                ephem_t0_convention: "centre".to_string(),
                ephem_p: period,
                ephem_t0: t0,
                ephem_shift_h: 0.0,
                ephem_refine: None,
            }),
            refine_failed: false,
        },
        T0Convention::BoxStart => {
            let guess = t0 + 0.5 * duration;
            let (ref_out, _trace) = refine_ephemeris(time, flux, period, guess, duration, cadence_s);
            let ok = ref_out.status == "ok";
            let (p_use, t0_use) = if ok {
                (ref_out.p, ref_out.t0)
            } else {
                (period, guess)
            };
            let shift_h = if ok { ref_out.shift_h } else { 0.0 };
            let prov = EphemProvenance {
                ephem_refined: ok,
                ephem_status: ref_out.status.clone(),
                ephem_t0_convention: "box_start".to_string(),
                ephem_p: p_use,
                ephem_t0: t0_use,
                ephem_shift_h: shift_h,
                ephem_refine: Some(RefineProvenanceFields {
                    reason: ref_out.reason,
                    p_in: ref_out.p_in,
                    t0_in: ref_out.t0_in,
                    t_ref_in: ref_out.t_ref_in,
                    cadence_s: ref_out.cadence_s,
                    n_req: ref_out.n_req,
                    n_orbits: ref_out.n_orbits,
                    n_transits: ref_out.n_transits,
                    p_searched: ref_out.p_searched,
                    p_free_accepted: ref_out.p_free_accepted,
                    delta_chi2_p: ref_out.delta_chi2_p,
                    dchi2_p_fixed: ref_out.dchi2_p_fixed,
                    dchi2_p_free: ref_out.dchi2_p_free,
                    p_fixed: ref_out.p_fixed,
                    dp: ref_out.dp,
                    max_offset_h: ref_out.max_offset_h,
                    dchi2_initial: ref_out.dchi2_initial,
                    dchi2_refined: ref_out.dchi2_refined,
                    delta_chi2: ref_out.delta_chi2,
                    sigma: ref_out.sigma,
                    t14_factor: ref_out.t14_factor,
                    ingress: ref_out.ingress,
                    edge_p: ref_out.edge_p,
                    edge_t0: ref_out.edge_t0,
                    dp_half: ref_out.dp_half,
                    dt_half: ref_out.dt_half,
                }),
            };
            FppEphemerisResult {
                p_use,
                t0_use,
                provenance: Some(prov),
                refine_failed: !ok,
            }
        }
    }
}
