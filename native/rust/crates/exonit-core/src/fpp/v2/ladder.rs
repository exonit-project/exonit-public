//! Fold-bin model ladder (plan 0.3 (j'), N5; every node integrated by the contact-split Gauss-Legendre rule):
//! literal transcription of `_ladder_bin`, `_draw_bin_models` and `production_bin_model` of
//! `worker/fpp_native.py`. Bit-exact.

use super::consts::{
    FPP_LADDER_DELTA_NAT, LADDER_ACCEPT_MAX_LEVEL, LADDER_EXACT_LEVEL, LADDER_K_QUANT,
    LADDER_MAX_K, LADDER_MIN_POINTS,
};
use super::contact::contact_times;
use super::gl::{choose_order, window_mean, DrawParams};
use super::math::cos;
use super::orbit::mid_transit_mean_anomaly;

/// Number of quantile-node levels stored per bin (`LADDER_EXACT_LEVEL - 1`, K = 4 * 2^l for l = 0..5).
pub const NODE_LEVELS: usize = LADDER_EXACT_LEVEL - 1;

/// The production fold in TIME units (days from mid-transit), the arrays `prod_fold_arrays` returns.
#[derive(Clone, Debug, Default)]
pub struct FoldArrays {
    pub n: Vec<i64>,
    pub flux: Vec<f64>,
    pub sigma: Vec<f64>,
    pub tmean: Vec<f64>,
    pub var_t: Vec<f64>,
    pub tmin: Vec<f64>,
    pub tmax: Vec<f64>,
    /// `node_tab[b, l, j]` flattened: `(b * NODE_LEVELS + l) * LADDER_MAX_K + j`.
    pub node_tab: Vec<f64>,
    pub pts_off: Vec<usize>,
    pub pts_t: Vec<f64>,
}

impl FoldArrays {
    #[inline]
    pub fn node(&self, b: usize, l: usize, j: usize) -> f64 {
        self.node_tab[(b * NODE_LEVELS + l) * LADDER_MAX_K + j]
    }
}

/// `_ladder_bin`: model `M_b` of one bin -> `(M_b, window means evaluated, level used)`.
pub fn ladder_bin(
    b: usize,
    w: f64,
    eps_b: f64,
    order_idx: usize,
    fold: &FoldArrays,
    ct: &[f64],
    d: &DrawParams,
) -> (f64, usize, usize) {
    let lo_p = fold.pts_off[b];
    let hi_p = fold.pts_off[b + 1];
    let npts = hi_p - lo_p;
    let mut nwin = 0usize;
    if fold.n[b] as usize <= LADDER_MIN_POINTS {
        let mut acc = 0.0;
        for i in 0..npts {
            acc += window_mean(fold.pts_t[lo_p + i], w, ct, order_idx, d);
        }
        return (acc / npts as f64, nwin + npts, LADDER_EXACT_LEVEL);
    }

    let mut vals = [0.0f64; LADDER_EXACT_LEVEL + 1];
    let w_eff = (w * w + 12.0 * fold.var_t[b]).sqrt();
    vals[0] = window_mean(fold.tmean[b], w_eff, ct, order_idx, d);
    nwin += 1;
    for lvl in 1..LADDER_EXACT_LEVEL {
        let kk = 4usize << (lvl - 1);
        let mut acc = 0.0;
        for j in 0..kk {
            acc += window_mean(fold.node(b, lvl - 1, j), w, ct, order_idx, d);
        }
        vals[lvl] = acc / kk as f64;
        nwin += kk;
        if 2 <= lvl && lvl <= LADDER_ACCEPT_MAX_LEVEL + 2 {
            let c = lvl - 2;
            let d1 = (vals[c] - vals[c + 1]).abs();
            let d2 = (vals[c + 1] - vals[c + 2]).abs();
            if d1 <= eps_b && d2 <= eps_b && (d2 <= d1 || (d1 < 0.05 * eps_b && d2 < 0.05 * eps_b))
            {
                return (vals[c], nwin, c);
            }
        }
    }

    let mut acc = 0.0;
    for i in 0..npts {
        acc += window_mean(fold.pts_t[lo_p + i], w, ct, order_idx, d);
    }
    nwin += npts;
    vals[LADDER_EXACT_LEVEL] = acc / npts as f64;
    let c = LADDER_ACCEPT_MAX_LEVEL;
    let d1 = (vals[c] - vals[c + 1]).abs();
    let d2 = (vals[c + 1] - vals[c + 2]).abs();
    if d1 <= eps_b && d2 <= eps_b && (d2 <= d1 || (d1 < 0.05 * eps_b && d2 < 0.05 * eps_b)) {
        return (vals[c], nwin, c);
    }
    (vals[LADDER_EXACT_LEVEL], nwin, LADDER_EXACT_LEVEL)
}

/// Result of [`draw_bin_models`]: `(N_aff, status, windows evaluated, bins that fell through to the exact level)`
/// plus the affected bin indices and their models.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DrawBinModels {
    pub n_aff: usize,
    pub status: i32,
    pub n_eval: usize,
    pub n_fb: usize,
    pub idx: Vec<usize>,
    pub m: Vec<f64>,
}

/// `_draw_bin_models`: one draw, the ladder model of every fold bin the transit can touch.
#[allow(clippy::too_many_arguments)]
pub fn draw_bin_models(
    w: f64,
    delta: f64,
    fold: &FoldArrays,
    period: f64,
    a_rs: f64,
    inc_rad: f64,
    ecc: f64,
    w_rad: f64,
    k: f64,
    u1: f64,
    u2: f64,
    dil_a: f64,
    dil_b: f64,
) -> DrawBinModels {
    let nb = fold.n.len();
    let ct5 = contact_times(period, a_rs, inc_rad, ecc, w_rad, k);
    let t1 = ct5[0];
    let t4 = ct5[3];
    if t1.is_nan() || t4.is_nan() {
        return DrawBinModels::default();
    }
    let mut ct: Vec<f64> = ct5[..4].iter().copied().filter(|v| !v.is_nan()).collect();
    ct.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let hw = 0.5 * w;
    let aff: Vec<usize> = (0..nb)
        .filter(|&b| fold.tmax[b] + hw > t1 && fold.tmin[b] - hw < t4)
        .collect();
    let n_aff = aff.len();
    if n_aff == 0 {
        return DrawBinModels::default();
    }

    let eps_ref = (2.0 * delta / n_aff as f64).sqrt();
    let mut eps_min = fold.sigma[aff[0]] * eps_ref;
    for &b in aff.iter().skip(1) {
        let e = fold.sigma[b] * eps_ref;
        if e < eps_min {
            eps_min = e;
        }
    }
    let m_t = mid_transit_mean_anomaly(ecc, w_rad);
    let cos_i = cos(inc_rad);
    let d = DrawParams {
        p: period,
        a_rs,
        cos_i,
        ecc,
        w_rad,
        m_t,
        k,
        u1,
        u2,
        dil_a,
        dil_b,
    };
    let (order_idx, status) = choose_order(&ct, w, eps_min, &d);
    let mut out = DrawBinModels {
        n_aff,
        status,
        ..Default::default()
    };
    for &b in &aff {
        let (m_b, nwin, level) =
            ladder_bin(b, w, fold.sigma[b] * eps_ref, order_idx, fold, &ct, &d);
        out.idx.push(b);
        out.m.push(m_b);
        out.n_eval += nwin;
        if level == LADDER_EXACT_LEVEL {
            out.n_fb += 1;
        }
    }
    out
}

/// The fold of uniformly filled bins `[lo[i], hi[i])` exactly as `production_bin_model` builds it (n = 1000,
/// mean time = the middle, var_t = (hi - lo)^2 / 12, level rungs = uniform quantile times, exact rung =
/// `LADDER_K_QUANT` uniform midpoints), with the likelihood sigma `sigma_l` in every bin.
pub fn uniform_bins_fold(lo: &[f64], hi: &[f64], sigma_l: f64) -> FoldArrays {
    let nb = lo.len();
    let mut f = FoldArrays {
        n: vec![1000; nb],
        flux: vec![0.0; nb],
        sigma: vec![sigma_l; nb],
        tmean: Vec::with_capacity(nb),
        var_t: Vec::with_capacity(nb),
        tmin: lo.to_vec(),
        tmax: hi.to_vec(),
        node_tab: vec![0.0; nb * NODE_LEVELS * LADDER_MAX_K],
        pts_off: (0..=nb).map(|i| i * LADDER_K_QUANT).collect(),
        pts_t: Vec::with_capacity(nb * LADDER_K_QUANT),
    };
    for b in 0..nb {
        f.tmean.push(0.5 * (lo[b] + hi[b]));
        f.var_t.push((hi[b] - lo[b]) * (hi[b] - lo[b]) / 12.0);
        let width = hi[b] - lo[b];
        for l in 0..NODE_LEVELS {
            let kk = 4usize << l;
            for j in 0..kk {
                let u = (j as f64 + 0.5) / kk as f64;
                f.node_tab[(b * NODE_LEVELS + l) * LADDER_MAX_K + j] = lo[b] + u * width;
            }
        }
        for j in 0..LADDER_K_QUANT {
            let u = (j as f64 + 0.5) / LADDER_K_QUANT as f64;
            f.pts_t.push(lo[b] + u * width);
        }
    }
    f
}

/// `production_bin_model`: model of the uniformly filled bins at cadence `cadence_s` (1.0 for bins the transit
/// does not touch). `sigma_l = None` is the reference default 2e-5.
#[allow(clippy::too_many_arguments)]
pub fn production_bin_model(
    p: f64,
    k: f64,
    a_r: f64,
    inc_rad: f64,
    ecc: f64,
    w_rad: f64,
    u1: f64,
    u2: f64,
    cadence_s: f64,
    bin_lo: &[f64],
    bin_hi: &[f64],
    sigma_l: Option<f64>,
) -> Vec<f64> {
    let nb = bin_lo.len();
    if nb == 0 {
        return Vec::new();
    }
    let fold = uniform_bins_fold(bin_lo, bin_hi, sigma_l.unwrap_or(2.0e-5));
    let r = draw_bin_models(
        cadence_s / 86400.0,
        FPP_LADDER_DELTA_NAT,
        &fold,
        p,
        a_r,
        inc_rad,
        ecc,
        w_rad,
        k,
        u1,
        u2,
        1.0,
        0.0,
    );
    let mut m = vec![1.0f64; nb];
    for i in 0..r.n_aff {
        m[r.idx[i]] = r.m[i];
    }
    m
}
