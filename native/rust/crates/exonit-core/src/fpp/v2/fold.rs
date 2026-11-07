//! Production fold of plan 0.3 (j'): mean bins + each bin's phase distribution (N5), the type-7 quantile spec,
//! the time-unit fold arrays of the ladder, and the two fold-wide constants of the bin likelihood. Literal
//! transcription of `_fold_quantiles_nb`, `fold_quantiles`, `_fold_phase_nb`, `_fold_bins_nb`,
//! `fold_production`, `prod_fold_arrays`, `build_prod_fold`, `prod_fold_renorm` of `worker/fpp_native.py`.
//! Sequential f64 sums in input order (NOT pairwise: the reference loops are sequential). Bit-exact.

use super::consts::{
    FOLD_K_QUANT, FOLD_N_BINS, FPP_LADDER_DELTA_NAT, LADDER_EXACT_LEVEL, LADDER_K_QUANT,
    LADDER_MAX_K, LN2PI,
};
use super::ladder::{FoldArrays, NODE_LEVELS};

/// CPython / numba float `%` (`fmod` plus the sign correction; result sign follows the divisor).
#[inline]
pub fn py_mod(x: f64, y: f64) -> f64 {
    let mut m = x % y;
    if m != 0.0 {
        if (y < 0.0) != (m < 0.0) {
            m += y;
        }
    } else {
        m = 0.0f64.copysign(y);
    }
    m
}

/// `np.linspace(-0.5, 0.5, n_bins + 1)`: `y_i = i * step + start`, last element exactly `stop`.
pub fn fold_edges(n_bins: usize) -> Vec<f64> {
    let step = 1.0 / n_bins as f64;
    let mut e: Vec<f64> = (0..=n_bins).map(|i| i as f64 * step + -0.5).collect();
    *e.last_mut().unwrap() = 0.5;
    e
}

/// `np.digitize(x, edges)` for increasing edges == `searchsorted(edges, x, side='right')`.
fn digitize(edges: &[f64], x: f64) -> usize {
    let (mut lo, mut hi) = (0usize, edges.len());
    while lo < hi {
        let mid = (lo + hi) / 2;
        if x < edges[mid] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// `_fold_quantiles_nb`: type-7 quantiles of an ascending-sorted slice, `p = (i + 0.5) / K`.
pub fn fold_quantiles_sorted(s: &[f64], kk: usize, out: &mut [f64]) {
    let n = s.len();
    let nm1 = n as i64 - 1;
    for i in 0..kk {
        let p = (i as f64 + 0.5) / kk as f64;
        let h = nm1 as f64 * p;
        let j = h.floor() as i64;
        let g = h - j as f64;
        out[i] = if j + 1 >= n as i64 {
            s[j as usize]
        } else {
            s[j as usize] + g * (s[j as usize + 1] - s[j as usize])
        };
    }
}

/// `fold_quantiles(x, K)`: stable ascending sort, then the spec; NaNs for an empty sample.
pub fn fold_quantiles(x: &[f64], kk: usize) -> Vec<f64> {
    if kk == 0 {
        return Vec::new();
    }
    let mut s = x.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if s.is_empty() {
        return vec![f64::NAN; kk];
    }
    let mut out = vec![0.0; kk];
    fold_quantiles_sorted(&s, kk, &mut out);
    out
}

/// One non-empty bin of `fold_production` (`bin`, `n`, sums, `quantiles` of the bin's phases and, when asked,
/// the bin's phases sorted ascending).
#[derive(Clone, Debug, Default)]
pub struct FoldRow {
    pub bin: usize,
    pub n: usize,
    pub flux_mean: f64,
    pub flux_m2: f64,
    pub phase_mean: f64,
    pub phase_m2: f64,
    pub var_b: f64,
    pub quantiles: Vec<f64>,
    pub phases: Vec<f64>,
}

/// `fold_production(time, flux, t0, period, n_bins, k_quant, keep_phases)`.
pub fn fold_production(
    time: &[f64],
    flux: &[f64],
    t0: f64,
    period: f64,
    n_bins: usize,
    k_quant: usize,
    keep_phases: bool,
) -> Vec<FoldRow> {
    // non-finite time / flux dropped, input order kept
    let (mut tk, mut fk) = (Vec::new(), Vec::new());
    for (&t, &f) in time.iter().zip(flux.iter()) {
        if t.is_finite() && f.is_finite() {
            tk.push(t);
            fk.push(f);
        }
    }
    if tk.is_empty() {
        return Vec::new();
    }
    // _fold_phase_nb: phase = ((t - t0) / period) % 1.0, then phase -= 1.0 when phase > 0.5
    let ph: Vec<f64> = tk
        .iter()
        .map(|&t| {
            let mut p = py_mod((t - t0) / period, 1.0);
            if p > 0.5 {
                p -= 1.0;
            }
            p
        })
        .collect();
    let edges = fold_edges(n_bins);
    let b_idx: Vec<usize> = ph
        .iter()
        .map(|&p| (digitize(&edges, p) as i64 - 1).clamp(0, n_bins as i64 - 1) as usize)
        .collect();
    let mut counts = vec![0usize; n_bins];
    for &b in &b_idx {
        counts[b] += 1;
    }
    // stable argsort by bin: input order kept inside every bin
    let mut order: Vec<usize> = (0..ph.len()).collect();
    order.sort_by_key(|&i| b_idx[i]);
    let ph_s: Vec<f64> = order.iter().map(|&i| ph[i]).collect();
    let fs_s: Vec<f64> = order.iter().map(|&i| fk[i]).collect();
    let mut starts = vec![0usize; n_bins + 1];
    for b in 0..n_bins {
        starts[b + 1] = starts[b] + counts[b];
    }

    let mut rows = Vec::new();
    for b in 0..n_bins {
        let n = counts[b];
        if n == 0 {
            continue;
        }
        let st = starts[b];
        let nf = n as f64;
        let mut s = 0.0;
        for i in 0..n {
            s += fs_s[st + i];
        }
        let fmean = s / nf;
        let mut m2 = 0.0;
        for i in 0..n {
            let d = fs_s[st + i] - fmean;
            m2 += d * d;
        }
        s = 0.0;
        for i in 0..n {
            s += ph_s[st + i];
        }
        let pmean = s / nf;
        let mut pm2 = 0.0;
        for i in 0..n {
            let d = ph_s[st + i] - pmean;
            pm2 += d * d;
        }
        let mut scratch: Vec<f64> = ph_s[st..st + n].to_vec();
        scratch.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut q = vec![0.0; k_quant];
        fold_quantiles_sorted(&scratch, k_quant, &mut q);
        rows.push(FoldRow {
            bin: b,
            n,
            flux_mean: fmean,
            flux_m2: m2,
            phase_mean: pmean,
            phase_m2: pm2,
            var_b: pm2 / nf,
            quantiles: q,
            phases: if keep_phases { scratch } else { Vec::new() },
        });
    }
    rows
}

/// `fold_production` with the default `n_bins = 1000`, `k_quant = 256` and the phases kept.
pub fn fold_production_default(time: &[f64], flux: &[f64], t0: f64, period: f64) -> Vec<FoldRow> {
    fold_production(time, flux, t0, period, FOLD_N_BINS, FOLD_K_QUANT, true)
}

/// `prod_fold_arrays(rows, period, sigma_b)`: the fold in TIME units for the ladder. `sigma_b` holds one value
/// per row, or one value that is repeated.
pub fn prod_fold_arrays(
    rows: &[FoldRow],
    period: f64,
    sigma_b: &[f64],
) -> Result<FoldArrays, String> {
    let nb = rows.len();
    let sig: Vec<f64> = if sigma_b.len() == 1 && nb > 1 {
        vec![sigma_b[0]; nb]
    } else {
        sigma_b.to_vec()
    };
    if sig.len() != nb {
        return Err("sigma_b must hold one value per fold row".into());
    }
    let mut fa = FoldArrays {
        n: rows.iter().map(|r| r.n as i64).collect(),
        flux: rows.iter().map(|r| r.flux_mean).collect(),
        sigma: sig,
        tmean: rows.iter().map(|r| r.phase_mean * period).collect(),
        var_t: rows.iter().map(|r| r.var_b * period * period).collect(),
        tmin: Vec::with_capacity(nb),
        tmax: Vec::with_capacity(nb),
        node_tab: vec![0.0; nb * NODE_LEVELS * LADDER_MAX_K],
        pts_off: vec![0; nb + 1],
        pts_t: Vec::new(),
    };
    for (i, r) in rows.iter().enumerate() {
        let mut ph = r.phases.clone();
        ph.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if ph.is_empty() {
            return Err("fold row without phases (use keep_phases)".into());
        }
        fa.tmin.push(ph[0] * period);
        fa.tmax.push(ph[ph.len() - 1] * period);
        fa.pts_off[i + 1] = fa.pts_off[i] + ph.len();
        for &p in &ph {
            fa.pts_t.push(p * period);
        }
        // N5: a bin with more points than LADDER_K_QUANT is reduced to its 256-point quantile table
        let src: &[f64] = if r.n > LADDER_K_QUANT {
            &r.quantiles
        } else {
            &ph
        };
        for l in 0..NODE_LEVELS {
            let kk = 4usize << l;
            let q = fold_quantiles(src, kk);
            for j in 0..kk {
                fa.node_tab[(i * NODE_LEVELS + l) * LADDER_MAX_K + j] = period * q[j];
            }
        }
    }
    let _ = LADDER_EXACT_LEVEL;
    Ok(fa)
}

/// The two fold-wide constants of the bin likelihood (`build_prod_fold` / `prod_fold_renorm`):
/// `base_chi2 = sum(((f - 1) / sigma)^2)` accumulated sequentially, `lnnorm = -0.5 ln(2 pi) - mean(ln sigma)`.
#[derive(Clone, Debug)]
pub struct ProdFoldConsts {
    pub w: f64,
    pub delta: f64,
    pub base_chi2: f64,
    pub lnnorm: f64,
}

fn chi2_and_lnnorm(f: &[f64], sig: &[f64]) -> (f64, f64) {
    let nb = f.len();
    let mut base_chi2 = 0.0;
    let mut mean_lnsigma = 0.0;
    for i in 0..nb {
        let d = (f[i] - 1.0) / sig[i];
        base_chi2 += d * d;
        mean_lnsigma += sig[i].ln();
    }
    (base_chi2, -0.5 * LN2PI - mean_lnsigma / nb as f64)
}

/// `build_prod_fold` constants: `W = cadence_s / 86400`, `delta = FPP_LADDER_DELTA_NAT`, `base_chi2`, `lnnorm`.
pub fn build_prod_fold_consts(
    fold: &FoldArrays,
    flux_b: &[f64],
    cadence_s: f64,
) -> Result<ProdFoldConsts, String> {
    let nb = fold.n.len();
    if nb == 0 {
        return Err("build_prod_fold: the production fold has no rows".into());
    }
    if flux_b.len() != nb {
        return Err(format!(
            "flux_b must hold one value per fold row ({} != {})",
            flux_b.len(),
            nb
        ));
    }
    if !(cadence_s > 0.0) {
        return Err("cadence_s must be positive".into());
    }
    let (base_chi2, lnnorm) = chi2_and_lnnorm(flux_b, &fold.sigma);
    Ok(ProdFoldConsts {
        w: cadence_s / 86400.0,
        delta: FPP_LADDER_DELTA_NAT,
        base_chi2,
        lnnorm,
    })
}

/// `prod_fold_renorm`: renormalise the bin fluxes / sigmas to a star of flux fraction `fr`
/// (`f' = (f - (1 - fr)) / fr`, `sigma' = sigma / fr`) -> `(f', sigma', base_chi2, lnnorm)`.
pub fn prod_fold_renorm(
    f: &[f64],
    sigma: &[f64],
    fr: f64,
) -> Result<(Vec<f64>, Vec<f64>, f64, f64), String> {
    if !(fr > 0.0) {
        return Err("prod_fold_renorm: fr must be positive".into());
    }
    let f2: Vec<f64> = f.iter().map(|&v| (v - (1.0 - fr)) / fr).collect();
    let s2: Vec<f64> = sigma.iter().map(|&v| v / fr).collect();
    let (base_chi2, lnnorm) = chi2_and_lnnorm(&f2, &s2);
    Ok((f2, s2, base_chi2, lnnorm))
}
