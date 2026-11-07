//! The (q') likelihood sigma of the fixed model (D15, plan 0.3 (q')): first-difference point noise, the
//! scalar `sigma_oot`, the bin- and T14-scale red-noise factors on MEAN bins. Transcription of `_mad`,
//! `_sigma_components_from_fold`, `sigma_components`, `likelihood_sigma` of `worker/fpp_native.py`.
//!
//! Exactness: everything is bit-exact (numpy pairwise sums / medians reproduced by [`super::super::npsum`]), including
//! the low-order phase-curve fit, which is the specified Householder QR of the reference ([`phase_resid_householder`]).

use super::super::npsum::{np_median, np_std, pairwise_sum};
use super::fold::{fold_edges, py_mod};
use super::math::{cos, sin};
use std::f64::consts::PI;

/// Match Python's q-prime phase-fit conditioning threshold.
pub const FIT_ILL_CONDITIONED_RCOND: f64 = 1.0e-8;

/// `_mad(x)`: 1.4826 * median(|x - median(x)|).
pub fn mad(x: &[f64]) -> f64 {
    let m = np_median(x);
    let dev: Vec<f64> = x.iter().map(|&v| (v - m).abs()).collect();
    1.4826 * np_median(&dev)
}

/// The intermediate values of the (q') noise model, including the phase-fit conditioning diagnostic.
#[derive(Clone, Debug)]
pub struct SigmaComponents {
    pub sigma_pt: f64,
    pub beta_bin: f64,
    pub beta_t14: f64,
    pub beta: f64,
    pub sigma_oot: f64,
    pub sigma_b: Vec<f64>,
    /// Min/max absolute diagonal of the unpivoted Householder R factor (1.0 when no fit runs).
    pub fit_rcond: f64,
    /// Python reference predicate `fit_rcond < FIT_ILL_CONDITIONED_RCOND`.
    pub fit_ill_conditioned: bool,
}

/// `_phase_resid_householder(xb, y)`: residual of the low-order phase-curve fit by the SPECIFIED Householder QR
/// (no LAPACK). Literal transcription of the normative spec in `worker/fpp_native.py`: design row
/// `[1, cos t, sin t, cos 2t, sin 2t]` with `t = (2 pi) * xb` and `(4 pi) * xb` for the second harmonic (scalar
/// libm cos / sin); every sum sequential with `i` ascending from 0.0; `f = (2.0 * s) / vn2`; `alpha = -norm` when
/// `R[j][j] >= 0`; no pivoting, no FMA; rank-deficient columns give `coef = 0` (`R[j][j] == 0`) or a skipped
/// reflection (`vn2 == 0`). Bit-exact with the reference.
pub fn phase_resid_householder(xb: &[f64], y: &[f64]) -> Vec<f64> {
    phase_resid_householder_with_rcond(xb, y).0
}

/// Householder phase-fit residuals and the reference conditioning proxy.
pub fn phase_resid_householder_with_rcond(xb: &[f64], y: &[f64]) -> (Vec<f64>, f64) {
    let n = xb.len();
    const P: usize = 5;
    let a: Vec<[f64; P]> = xb
        .iter()
        .map(|&x| {
            let th = 2.0 * PI * x;
            let th2 = 4.0 * PI * x;
            [1.0, cos(th), sin(th), cos(th2), sin(th2)]
        })
        .collect();
    let mut r = a.clone();
    let mut b = y.to_vec();
    let mut v = vec![0.0f64; n];
    for j in 0..P {
        let mut s = 0.0;
        for i in j..n {
            s += r[i][j] * r[i][j];
        }
        let norm = s.sqrt();
        let alpha = if r[j][j] >= 0.0 { -norm } else { norm };
        v[j] = r[j][j] - alpha;
        for i in (j + 1)..n {
            v[i] = r[i][j];
        }
        let mut vn2 = 0.0;
        for i in j..n {
            vn2 += v[i] * v[i];
        }
        if vn2 == 0.0 {
            continue;
        }
        for c in j..P {
            let mut s = 0.0;
            for i in j..n {
                s += v[i] * r[i][c];
            }
            let f = 2.0 * s / vn2;
            for i in j..n {
                r[i][c] -= f * v[i];
            }
        }
        let mut s = 0.0;
        for i in j..n {
            s += v[i] * b[i];
        }
        let f = 2.0 * s / vn2;
        for i in j..n {
            b[i] -= f * v[i];
        }
    }
    let mut coef = [0.0f64; P];
    for j in (0..P).rev() {
        let mut s = b[j];
        for c in (j + 1)..P {
            s -= r[j][c] * coef[c];
        }
        coef[j] = if r[j][j] != 0.0 { s / r[j][j] } else { 0.0 };
    }
    let mut dmin = r[0][0].abs();
    let mut dmax = dmin;
    for j in 1..P {
        let d = r[j][j].abs();
        if d < dmin {
            dmin = d;
        }
        if d > dmax {
            dmax = d;
        }
    }
    let rcond = if dmax > 0.0 { dmin / dmax } else { 0.0 };
    let residuals = (0..n)
        .map(|i| {
            let mut s = a[i][0] * coef[0];
            for c in 1..P {
                s += a[i][c] * coef[c];
            }
            y[i] - s
        })
        .collect();
    (residuals, rcond)
}

/// `_sigma_components_from_fold(time, flux, period, t0, duration, time_pf, flux_pf, counts)`.
#[allow(clippy::too_many_arguments)]
pub fn sigma_components_from_fold(
    time: &[f64],
    flux: &[f64],
    period: f64,
    t0: f64,
    duration: f64,
    time_pf: &[f64],
    flux_pf: &[f64],
    counts: &[f64],
) -> SigmaComponents {
    let guard = 0.75 * duration + 0.02;
    let nt = time.len();
    let mut ok = vec![false; nt];
    for i in 0..nt {
        let ph = py_mod((time[i] - t0) / period + 0.5, 1.0) - 0.5;
        let dt = ph * period;
        let dsec = (ph.abs() - 0.5) * period;
        ok[i] = dt.abs() > guard && dsec.abs() > guard && flux[i].is_finite();
    }
    let mut order: Vec<usize> = (0..nt).collect();
    order.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let tt: Vec<f64> = order.iter().map(|&i| time[i]).collect();
    let ff: Vec<f64> = order.iter().map(|&i| flux[i]).collect();
    let oo: Vec<bool> = order.iter().map(|&i| ok[i]).collect();
    let dtt: Vec<f64> = tt.windows(2).map(|w| w[1] - w[0]).collect();
    let cad = if tt.len() > 2 { np_median(&dtt) } else { 1.0 };
    let mut d: Vec<f64> = Vec::new();
    for i in 0..dtt.len() {
        if oo[i + 1] && oo[i] && dtt[i] < 3.0 * cad {
            d.push(ff[i + 1] - ff[i]);
        }
    }
    let n_ok = ok.iter().filter(|&&v| v).count();
    let sig_pt = if d.len() >= 50 {
        let md = np_median(&d);
        let dev: Vec<f64> = d.iter().map(|&v| (v - md).abs()).collect();
        1.4826 * np_median(&dev) / 2.0f64.sqrt()
    } else if n_ok >= 20 {
        let sel: Vec<f64> = (0..nt).filter(|&i| ok[i]).map(|i| flux[i]).collect();
        mad(&sel)
    } else {
        np_std(flux)
    };

    let npf = time_pf.len();
    let ob: Vec<bool> = (0..npf)
        .map(|i| time_pf[i].abs() > guard && (time_pf[i].abs() - 0.5 * period).abs() > guard)
        .collect();
    let io: Vec<usize> = (0..npf).filter(|&i| ob[i]).collect();
    let mut ia: Vec<usize> = Vec::new();
    for j in 1..io.len() {
        if io[j] - io[j - 1] == 1 && (time_pf[io[j]] - time_pf[io[j - 1]]) < 1.5 * period / 1000.0 {
            ia.push(io[j]);
        }
    }
    let n_ob = ob.iter().filter(|&&v| v).count();
    let mut sig_oot = if ia.len() >= 20 {
        let d_: Vec<f64> = ia.iter().map(|&i| flux_pf[i] - flux_pf[i - 1]).collect();
        let md = np_median(&d_);
        let dev: Vec<f64> = d_.iter().map(|&v| (v - md).abs()).collect();
        1.4826 * np_median(&dev) / 2.0f64.sqrt()
    } else if n_ob >= 20 {
        let sel: Vec<f64> = (0..npf).filter(|&i| ob[i]).map(|i| flux_pf[i]).collect();
        mad(&sel)
    } else {
        np_std(flux_pf)
    };
    // sig_oot = max(sig_oot, 1e-9): Python max returns the 2nd argument only when it is strictly larger
    if 1e-9 > sig_oot {
        sig_oot = 1e-9;
    }
    let n: Vec<f64> = counts
        .iter()
        .map(|&c| if c > 1.0 { c } else { 1.0 })
        .collect();
    let white: Vec<f64> = n.iter().map(|&c| sig_pt / c.sqrt()).collect();
    let okb: Vec<bool> = (0..npf).map(|i| ob[i] && n[i] >= 5.0).collect();
    let n_okb = okb.iter().filter(|&&v| v).count();
    if n_okb < 20 {
        return SigmaComponents {
            sigma_pt: sig_pt,
            beta_bin: f64::NAN,
            beta_t14: f64::NAN,
            beta: f64::NAN,
            sigma_oot: sig_oot,
            sigma_b: vec![sig_oot; npf],
            fit_rcond: 1.0,
            fit_ill_conditioned: false,
        };
    }
    let sel_white: Vec<f64> = (0..npf).filter(|&i| okb[i]).map(|i| white[i]).collect();
    let w_med = np_median(&sel_white);
    let xs: Vec<f64> = (0..npf)
        .filter(|&i| okb[i])
        .map(|i| time_pf[i] / period)
        .collect();
    let ys: Vec<f64> = (0..npf).filter(|&i| okb[i]).map(|i| flux_pf[i]).collect();
    let (r, fit_rcond) = phase_resid_householder_with_rcond(&xs, &ys);
    let mad_r = mad(&r);
    let beta_bin = mad_r / w_med;
    let nb_f = (duration / (period / 1000.0)).round_ties_even();
    let nb = if nb_f > 2.0 { nb_f as usize } else { 2 };
    let k = r.len() / nb;
    let beta_t14 = if k >= 8 {
        let means: Vec<f64> = (0..k)
            .map(|i| pairwise_sum(&r[i * nb..(i + 1) * nb]) / nb as f64)
            .collect();
        mad(&means) / (mad_r / (nb as f64).sqrt())
    } else {
        1.0
    };
    // beta = min(max(1.0, beta_bin, beta_t14), 5.0)
    let mut beta = 1.0f64;
    if beta_bin > beta {
        beta = beta_bin;
    }
    if beta_t14 > beta {
        beta = beta_t14;
    }
    if beta > 5.0 {
        beta = 5.0;
    }
    SigmaComponents {
        sigma_pt: sig_pt,
        beta_bin,
        beta_t14,
        beta,
        sigma_oot: sig_oot,
        sigma_b: white.iter().map(|&w| w * beta).collect(),
        fit_rcond,
        fit_ill_conditioned: fit_rcond < FIT_ILL_CONDITIONED_RCOND,
    }
}

/// `np.nanmean` of a slice (NaN -> 0 in the pairwise sum, divided by the non-NaN count).
fn nanmean(v: &[f64]) -> f64 {
    let cnt = v.iter().filter(|x| !x.is_nan()).count();
    let cleaned: Vec<f64> = v
        .iter()
        .map(|&x| if x.is_nan() { 0.0 } else { x })
        .collect();
    pairwise_sum(&cleaned) / cnt as f64
}

/// `sigma_components(time, flux, period, t0, duration)`: the (q') components of a raw light curve; the mean fold is
/// built as the reference does (`np.nanmean` per bin, bin centre x period as the bin time). `None` for an empty fold.
pub fn sigma_components(
    time: &[f64],
    flux: &[f64],
    period: f64,
    t0: f64,
    duration: f64,
) -> Option<SigmaComponents> {
    const N_BINS: usize = 1000;
    let phase: Vec<f64> = time
        .iter()
        .map(|&t| {
            let mut p = py_mod((t - t0) / period, 1.0);
            if p > 0.5 {
                p -= 1.0;
            }
            p
        })
        .collect();
    let edges = fold_edges(N_BINS);
    let centres: Vec<f64> = (0..N_BINS)
        .map(|i| 0.5 * (edges[i] + edges[i + 1]))
        .collect();
    let bin_idx: Vec<usize> = phase
        .iter()
        .map(|&p| {
            let (mut lo, mut hi) = (0usize, edges.len());
            while lo < hi {
                let mid = (lo + hi) / 2;
                if p < edges[mid] {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            (lo as i64 - 1).clamp(0, N_BINS as i64 - 1) as usize
        })
        .collect();
    let mut order: Vec<usize> = (0..time.len()).collect();
    order.sort_by_key(|&i| bin_idx[i]); // stable: input order kept inside every bin
    let mut per_bin: Vec<Vec<f64>> = vec![Vec::new(); N_BINS];
    for &i in &order {
        per_bin[bin_idx[i]].push(flux[i]);
    }
    let (mut t_out, mut f_out, mut cnt) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..N_BINS {
        if !per_bin[i].is_empty() {
            t_out.push(centres[i] * period);
            f_out.push(nanmean(&per_bin[i]));
            cnt.push(per_bin[i].len() as f64);
        }
    }
    if t_out.is_empty() {
        return None;
    }
    let keep: Vec<usize> = (0..t_out.len())
        .filter(|&i| t_out[i].is_finite() && f_out[i].is_finite())
        .collect();
    let tpf: Vec<f64> = keep.iter().map(|&i| t_out[i]).collect();
    let fpf: Vec<f64> = keep.iter().map(|&i| f_out[i]).collect();
    let cn: Vec<f64> = keep.iter().map(|&i| cnt[i]).collect();
    Some(sigma_components_from_fold(
        time, flux, period, t0, duration, &tpf, &fpf, &cn,
    ))
}

/// `likelihood_sigma`: `(per-bin sigma_b, scalar sigma_oot)` of a raw light curve.
pub fn likelihood_sigma(
    time: &[f64],
    flux: &[f64],
    period: f64,
    t0: f64,
    duration: f64,
) -> Option<(Vec<f64>, f64)> {
    sigma_components(time, flux, period, t0, duration).map(|c| (c.sigma_b, c.sigma_oot))
}
