//! Rust port of the FPP fix plan step-4 S family (STP, SEB, SEBx2P) — a transit / eclipsing binary around
//! the unresolved BOUND companion of the target. Literal transcription of `worker/fpp_native.py` `_lnZ_STP`
//! / `_lnZ_SEB` and their helpers (`stellar_relations_capped`, `ldc_companion_arrays`, `_roche_ok`, the batch
//! lnL kernels, the bound-companion priors, `_lnZ_from_logw`). Bit-exact on glibc x86-64.
//!
//! The two batch kernels take per-draw u1/u2 ARRAYS (the Python groups draws by equal (u1, u2) and calls the
//! scalar-u kernels per group; the per-draw result is independent of the grouping, so the arrays are equivalent).
//!
//! Reference: `worker/fpp_native.py` of the exonit repo. Samplers / splines / flux / LDC / RNG / pairwise sums
//! come from [`super::super`] (`crate::fpp`), the numerical pieces from [`super`] (`crate::fpp::v2`).

use super::super::npsum::{libm_pow, pairwise_sum};
use super::super::priors::{
    sample_ecc, sample_inc, sample_q, sample_q_companion, sample_rp_array, sample_w,
};
use super::super::rng::FppRng;
use super::super::stellar::{
    flux_relation, get_ldc_arrays, ldc_companion_arrays, stellar_relations_capped,
};
use super::consts::{AU, G_CGS, LN2PI, MSUN, PARITY_EXPTIME_D, PARITY_NSUB, REARTH, RSUN};
use super::kernel::transit_flux;
use super::ladder::{draw_bin_models, FoldArrays};
use super::math::{cos, sin};
use super::orbit::{lc_core, mid_transit_mean_anomaly, z_one};
use std::f64::consts::PI;

// --------------------------------------------------------------------------- Roche mask

/// `_roche_ok`: per-draw boolean — the companion clears its Roche limit at periastron. `m_s` / `r_s` are the
/// host (companion) per-draw mass / radius in Solar units, `rho_comp` the per-draw companion density (g/cm^3).
pub fn roche_ok(
    a_cm: &[f64],
    ecc: &[f64],
    m_s: &[f64],
    r_s: &[f64],
    rho_comp: &[f64],
) -> Vec<bool> {
    let c = (4.0 / 3.0) * PI;
    let mut out = Vec::with_capacity(a_cm.len());
    for i in 0..a_cm.len() {
        let r_host_cm = r_s[i] * RSUN;
        let rho_host = (m_s[i] * MSUN) / (c * r_host_cm * r_host_cm * r_host_cm);
        let periastron = a_cm[i] * (1.0 - ecc[i]);
        let limit = 2.44 * r_host_cm * libm_pow(rho_host / rho_comp[i], 1.0 / 3.0);
        out.push(periastron >= limit);
    }
    out
}

// --------------------------------------------------------------------------- secondary minimum flux

/// `_eb_secondary_min_flux`: minimum in-eclipse flux of the EB secondary (swapped geometry, 25 points).
#[allow(clippy::too_many_arguments)]
pub fn eb_secondary_min_flux(
    k_raw: f64,
    p: f64,
    a_rs: f64,
    cos_i: f64,
    ecc: f64,
    w_rad: f64,
    u1: f64,
    u2: f64,
    u1_sec: f64,
    u2_sec: f64,
    sec_consistent: bool,
    k_nudge_upstream: bool,
) -> f64 {
    const N_SAMP: usize = 25;
    const HALF_SPAN: f64 = 0.05;
    let mut k_s = 1.0 / k_raw;
    if k_nudge_upstream && (k_s - 1.0).abs() < 1e-6 {
        k_s *= 0.999;
    }
    let w_sec = w_rad + PI;
    let m_t_s = mid_transit_mean_anomaly(ecc, w_sec);
    let mut sec_flux = f64::INFINITY;
    for i in 0..N_SAMP {
        let t_i = -HALF_SPAN + 2.0 * HALF_SPAN * i as f64 / (N_SAMP as f64 - 1.0);
        let mut z = z_one(t_i, 0.0, p, a_rs, cos_i, ecc, w_sec, m_t_s);
        let (uu1, uu2) = if sec_consistent {
            z *= k_s;
            (u1_sec, u2_sec)
        } else {
            (u1, u2)
        };
        let f = transit_flux(z, k_s, uu1, uu2);
        if f < sec_flux {
            sec_flux = f;
        }
    }
    sec_flux
}

// --------------------------------------------------------------------------- cadence batch kernels

/// `_batch_transit_lnL` (per-draw u1/u2 arrays; `sigma` is per-bin). Returns one lnL per draw, `-inf` never
/// evaluated (not applicable here: the caller already masked, so every draw is evaluated).
#[allow(clippy::too_many_arguments)]
pub fn batch_transit_lnl(
    time: &[f64],
    flux: &[f64],
    sigma: &[f64],
    ks: &[f64],
    p_orbs: &[f64],
    incs_rad: &[f64],
    a_rs_arr: &[f64],
    eccs: &[f64],
    ws_rad: &[f64],
    u1: &[f64],
    u2: &[f64],
    companion_fluxratio: &[f64],
    companion_is_host: bool,
    exptime: f64,
    nsub: usize,
) -> Vec<f64> {
    let n = ks.len();
    let t = time.len();
    let mut mean_lnsigma = 0.0;
    for i in 0..t {
        mean_lnsigma += sigma[i].ln();
    }
    mean_lnsigma /= t as f64;
    let mut lnl = vec![f64::NEG_INFINITY; n];
    for j in 0..n {
        let mut model = lc_core(
            time,
            ks[j],
            0.0,
            p_orbs[j],
            a_rs_arr[j],
            incs_rad[j],
            eccs[j],
            ws_rad[j],
            u1[j],
            u2[j],
            exptime,
            nsub,
        );
        let fr = companion_fluxratio[j];
        if fr > 0.0 {
            let f_dilute = if companion_is_host {
                1.0 / (fr / (1.0 - fr))
            } else {
                fr / (1.0 - fr)
            };
            for m in model.iter_mut() {
                *m = (*m + f_dilute) / (1.0 + f_dilute);
            }
        }
        let mut chi2 = 0.0;
        for i in 0..t {
            let diff = (flux[i] - model[i]) / sigma[i];
            chi2 += diff * diff;
        }
        lnl[j] = -0.5 * LN2PI - mean_lnsigma - 0.5 * chi2;
    }
    lnl
}

/// `_batch_eb_transit_lnL` (per-draw u1/u2 arrays, and per-draw secondary u1_sec/u2_sec).
#[allow(clippy::too_many_arguments)]
pub fn batch_eb_transit_lnl(
    time: &[f64],
    flux: &[f64],
    sigma: &[f64],
    sigma_veto: f64,
    ks: &[f64],
    eb_fluxratios: &[f64],
    p_orbs: &[f64],
    incs_rad: &[f64],
    a_rs_arr: &[f64],
    eccs: &[f64],
    ws_rad: &[f64],
    u1: &[f64],
    u2: &[f64],
    companion_fluxratio: &[f64],
    companion_is_host: bool,
    exptime: f64,
    nsub: usize,
    reject_sec: bool,
    sec_consistent: bool,
    k_nudge_upstream: bool,
    u1_sec: &[f64],
    u2_sec: &[f64],
) -> Vec<f64> {
    let n = ks.len();
    let t = time.len();
    let mut mean_lnsigma = 0.0;
    for i in 0..t {
        mean_lnsigma += sigma[i].ln();
    }
    mean_lnsigma /= t as f64;
    let mut lnl = vec![f64::NEG_INFINITY; n];
    let f_target = 1.0;
    for j in 0..n {
        let k_raw = ks[j];
        let mut k = k_raw;
        if k_nudge_upstream && (k - 1.0).abs() < 1e-6 {
            k *= 0.999;
        }
        let mut flux_m = lc_core(
            time,
            k,
            0.0,
            p_orbs[j],
            a_rs_arr[j],
            incs_rad[j],
            eccs[j],
            ws_rad[j],
            u1[j],
            u2[j],
            exptime,
            nsub,
        );
        let mut sec_flux = eb_secondary_min_flux(
            k_raw,
            p_orbs[j],
            a_rs_arr[j],
            cos(incs_rad[j]),
            eccs[j],
            ws_rad[j],
            u1[j],
            u2[j],
            u1_sec[j],
            u2_sec[j],
            sec_consistent,
            k_nudge_upstream,
        );
        let fr_eb = eb_fluxratios[j];
        let f_eb = if fr_eb < 1.0 {
            fr_eb / (1.0 - fr_eb)
        } else {
            1e6
        };
        let fc = companion_fluxratio[j];
        let f_comp = if fc > 0.0 { fc / (1.0 - fc) } else { 0.0 };
        let f_dilute = if companion_is_host {
            let fc_c = f_comp.max(1e-30);
            let ratio = f_eb / fc_c;
            for m in flux_m.iter_mut() {
                *m = (*m + ratio) / (1.0 + ratio);
            }
            sec_flux = (sec_flux + fc_c / f_eb) / (1.0 + fc_c / f_eb);
            f_target / (fc_c + f_eb)
        } else {
            let ratio = f_eb / f_target;
            for m in flux_m.iter_mut() {
                *m = (*m + ratio) / (1.0 + ratio);
            }
            sec_flux = (sec_flux + f_target / f_eb) / (1.0 + f_target / f_eb);
            f_comp / (f_target + f_eb)
        };
        for m in flux_m.iter_mut() {
            *m = (*m + f_dilute) / (1.0 + f_dilute);
        }
        let secdepth = 1.0 - (sec_flux + f_dilute) / (1.0 + f_dilute);
        if reject_sec && secdepth >= 1.5 * sigma_veto {
            continue;
        }
        let mut chi2 = 0.0;
        for i in 0..t {
            let diff = (flux[i] - flux_m[i]) / sigma[i];
            chi2 += diff * diff;
        }
        lnl[j] = -0.5 * LN2PI - mean_lnsigma - 0.5 * chi2;
    }
    lnl
}

// --------------------------------------------------------------------------- production batch kernels

/// The production-fold inputs of one candidate (renormed): the fold-bin arrays, the renormed bin flux `f`,
/// and the `build_prod_fold` / `prod_fold_renorm` scalars.
pub struct ProdFold<'a> {
    pub fold: &'a FoldArrays,
    pub f: &'a [f64],
    pub base_chi2: f64,
    pub lnnorm: f64,
    pub w: f64,
    pub delta: f64,
}

/// `_batch_transit_lnL_prod`: `(lnL, status, n_aff)` per draw, on the production fold.
#[allow(clippy::too_many_arguments)]
pub fn batch_transit_lnl_prod(
    pf: &ProdFold,
    ks: &[f64],
    p_orbs: &[f64],
    incs_rad: &[f64],
    a_rs_arr: &[f64],
    eccs: &[f64],
    ws_rad: &[f64],
    u1: &[f64],
    u2: &[f64],
    companion_fluxratio: &[f64],
    companion_is_host: bool,
) -> (Vec<f64>, Vec<i64>, Vec<i64>) {
    let n = ks.len();
    let mut lnl = vec![f64::NEG_INFINITY; n];
    let mut status = vec![0i64; n];
    let mut n_aff_out = vec![0i64; n];
    for j in 0..n {
        let fr = companion_fluxratio[j];
        let (dil_a, dil_b) = if fr > 0.0 {
            let f = if companion_is_host {
                1.0 / (fr / (1.0 - fr))
            } else {
                fr / (1.0 - fr)
            };
            (1.0 / (1.0 + f), f / (1.0 + f))
        } else {
            (1.0, 0.0)
        };
        let r = draw_bin_models(
            pf.w,
            pf.delta,
            pf.fold,
            p_orbs[j],
            a_rs_arr[j],
            incs_rad[j],
            eccs[j],
            ws_rad[j],
            ks[j],
            u1[j],
            u2[j],
            dil_a,
            dil_b,
        );
        let mut chi2 = pf.base_chi2;
        for i in 0..r.n_aff {
            let b = r.idx[i];
            let d = (pf.f[b] - r.m[i]) / pf.fold.sigma[b];
            chi2 += d * d;
            let d0 = (pf.f[b] - 1.0) / pf.fold.sigma[b];
            chi2 -= d0 * d0;
        }
        lnl[j] = pf.lnnorm - 0.5 * chi2;
        status[j] = r.status as i64;
        n_aff_out[j] = r.n_aff as i64;
    }
    if super::gate::active() {
        super::gate::kernel_rows(p_orbs, a_rs_arr, incs_rad, eccs, ws_rad, ks, &lnl, &status);
    }
    (lnl, status, n_aff_out)
}

/// `_batch_eb_transit_lnL_prod`.
#[allow(clippy::too_many_arguments)]
pub fn batch_eb_transit_lnl_prod(
    pf: &ProdFold,
    sigma_veto: f64,
    ks: &[f64],
    eb_fluxratios: &[f64],
    p_orbs: &[f64],
    incs_rad: &[f64],
    a_rs_arr: &[f64],
    eccs: &[f64],
    ws_rad: &[f64],
    u1: &[f64],
    u2: &[f64],
    companion_fluxratio: &[f64],
    companion_is_host: bool,
    reject_sec: bool,
    sec_consistent: bool,
    k_nudge_upstream: bool,
    u1_sec: &[f64],
    u2_sec: &[f64],
) -> (Vec<f64>, Vec<i64>, Vec<i64>) {
    let n = ks.len();
    let mut lnl = vec![f64::NEG_INFINITY; n];
    let mut status = vec![0i64; n];
    let mut n_aff_out = vec![0i64; n];
    let f_target = 1.0;
    for j in 0..n {
        let k_raw = ks[j];
        let mut k = k_raw;
        if k_nudge_upstream && (k - 1.0).abs() < 1e-6 {
            k *= 0.999;
        }
        let sec_flux = eb_secondary_min_flux(
            k_raw,
            p_orbs[j],
            a_rs_arr[j],
            cos(incs_rad[j]),
            eccs[j],
            ws_rad[j],
            u1[j],
            u2[j],
            u1_sec[j],
            u2_sec[j],
            sec_consistent,
            k_nudge_upstream,
        );
        let fr_eb = eb_fluxratios[j];
        let f_eb = if fr_eb < 1.0 {
            fr_eb / (1.0 - fr_eb)
        } else {
            1e6
        };
        let fc = companion_fluxratio[j];
        let f_comp = if fc > 0.0 { fc / (1.0 - fc) } else { 0.0 };
        let (x, sec_flux2, f_dilute) = if companion_is_host {
            let fc_c = f_comp.max(1e-30);
            let x = f_eb / fc_c;
            let sf = (sec_flux + fc_c / f_eb) / (1.0 + fc_c / f_eb);
            (x, sf, f_target / (fc_c + f_eb))
        } else {
            let x = f_eb / f_target;
            let sf = (sec_flux + f_target / f_eb) / (1.0 + f_target / f_eb);
            (x, sf, f_comp / (f_target + f_eb))
        };
        let secdepth = 1.0 - (sec_flux2 + f_dilute) / (1.0 + f_dilute);
        if reject_sec && secdepth >= 1.5 * sigma_veto {
            continue;
        }
        let a1 = 1.0 / (1.0 + x);
        let b1 = x / (1.0 + x);
        let a2 = 1.0 / (1.0 + f_dilute);
        let b2 = f_dilute / (1.0 + f_dilute);
        let dil_a = a2 * a1;
        let dil_b = a2 * b1 + b2;
        let r = draw_bin_models(
            pf.w,
            pf.delta,
            pf.fold,
            p_orbs[j],
            a_rs_arr[j],
            incs_rad[j],
            eccs[j],
            ws_rad[j],
            k,
            u1[j],
            u2[j],
            dil_a,
            dil_b,
        );
        let mut chi2 = pf.base_chi2;
        for i in 0..r.n_aff {
            let b = r.idx[i];
            let d = (pf.f[b] - r.m[i]) / pf.fold.sigma[b];
            chi2 += d * d;
            let d0 = (pf.f[b] - 1.0) / pf.fold.sigma[b];
            chi2 -= d0 * d0;
        }
        lnl[j] = pf.lnnorm - 0.5 * chi2;
        status[j] = r.status as i64;
        n_aff_out[j] = r.n_aff as i64;
    }
    if super::gate::active() {
        // the k the kernel hands to draw_bin_models (Python generator: ks * 0.999 near 1 with the upstream nudge)
        let ks_used: Vec<f64> = ks
            .iter()
            .map(|&k| if k_nudge_upstream && (k - 1.0).abs() < 1e-6 { k * 0.999 } else { k })
            .collect();
        super::gate::kernel_rows(p_orbs, a_rs_arr, incs_rad, eccs, ws_rad, &ks_used, &lnl, &status);
    }
    (lnl, status, n_aff_out)
}

// --------------------------------------------------------------------------- bound-companion priors

/// `_separation_at_contrast`: `np.interp(delta_mags, [1.0], [2.2])` = 2.2 for every input (single-point `xp`).
fn separation_at_contrast(delta_mags: &[f64]) -> Vec<f64> {
    delta_mags.iter().map(|_| 2.2).collect()
}

/// Shared tail (the `t4`/`t5` pieces and the piecewise assembly) of `_lnprior_bound_TP` / `_lnprior_bound_EB`,
/// for a given `m_s_eff` (the mass the f1/f2/f3 coefficients are built from), the `log_mP` array, and the
/// `mask_lo` the piecewise mask starts at (3.4 for TP, 1.0 for EB). `t2`/`t3` are only used by the EB form.
#[allow(clippy::too_many_arguments)]
fn bound_rate_pieces(m_s_eff: f64, log_mp: &[f64], mask_lo: f64, with_t2t3: bool) -> Vec<f64> {
    let lm = m_s_eff.log10();
    let f1 = 0.020 + 0.04 * lm + 0.07 * (lm * lm);
    let f2 = 0.039 + 0.07 * lm + 0.01 * (lm * lm);
    let f3 = 0.078 - 0.05 * lm + 0.04 * (lm * lm);
    let alpha = 0.018;
    let dlog_p = 0.7;
    let t4 = alpha * dlog_p * (5.5 - 3.4)
        + f2 * (5.5 - 3.4)
        + (f3 - f2 - alpha * dlog_p) * (0.238095 * (5.5 * 5.5) - 0.952381 * 5.5 + 0.485714);
    let t5 = f3 * (3.33333 - 17.3566 * (-0.3f64 * 8.0f64).exp());
    let t2 = 0.5 * (2.0 - 1.0) * (2.0 * f1 + (f2 - f1 - alpha * dlog_p) * (2.0 - 1.0));
    let t3 = 0.5 * alpha * (3.4 * 3.4 - 5.4 * 3.4 + 6.8) + f2 * (3.4 - 2.0);
    let mut out = vec![0.0f64; log_mp.len()];
    for i in 0..log_mp.len() {
        let lp = log_mp[i];
        let t2_partial = 0.5 * (lp - 1.0) * (2.0 * f1 + (f2 - f1 - alpha * dlog_p) * (lp - 1.0));
        let t3_partial = 0.5 * alpha * (lp * lp - 5.4 * lp + 6.8) + f2 * (lp - 2.0);
        let t4_partial = alpha * dlog_p * (lp - 3.4)
            + f2 * (lp - 3.4)
            + (f3 - f2 - alpha * dlog_p) * (0.238095 * (lp * lp) - 0.952381 * lp + 0.485714);
        let t5_partial = f3 * (3.33333 - 17.3566 * (-0.3 * lp).exp());
        let v = if with_t2t3 {
            if lp >= mask_lo && lp < mask_lo + 1.0 {
                t2_partial
            } else if lp >= mask_lo + 1.0 && lp < 3.4 {
                t2 + t3_partial
            } else if lp >= 3.4 && lp < 5.5 {
                t2 + t3 + t4_partial
            } else if lp >= 5.5 && lp < 8.0 {
                t2 + t3 + t4 + t5_partial
            } else if lp >= 8.0 {
                t2 + t3 + t4 + t5
            } else {
                0.0
            }
        } else {
            if lp >= 3.4 && lp < 5.5 {
                t4_partial
            } else if lp >= 5.5 && lp < 8.0 {
                t4 + t5_partial
            } else if lp >= 8.0 {
                t4 + t5
            } else {
                0.0
            }
        };
        out[i] = v;
    }
    out
}

/// `_lnprior_bound_TP` (Porb_comp > 2500 d).
pub fn lnprior_bound_tp(m_s: f64, plx: f64, delta_mags: &[f64]) -> Vec<f64> {
    let plx = if plx.is_nan() { 0.1 } else { plx };
    let d = 1000.0 / plx;
    let seps: Vec<f64> = separation_at_contrast(delta_mags)
        .iter()
        .map(|&s| d * s)
        .collect();
    let m_s_eff = if m_s >= 1.0 { m_s } else { 1.0 };
    let max_p: Vec<f64> = seps
        .iter()
        .map(|&s| {
            let x3 = libm_pow(s * AU, 3.0);
            libm_pow((4.0 * (PI * PI)) / (G_CGS * m_s_eff * MSUN) * x3, 0.5) / 86400.0
        })
        .collect();
    let log_mp: Vec<f64> = max_p.iter().map(|&x| x.log10()).collect();
    let f_comp = bound_rate_pieces(m_s_eff, &log_mp, 3.4, false);
    if m_s >= 1.0 {
        f_comp.iter().map(|&v| v.ln()).collect()
    } else {
        f_comp
            .iter()
            .map(|&v| {
                let f_act = 0.65 * v + 0.35 * v * m_s;
                let f_act = if f_act < 0.0 { 0.0 } else { f_act };
                f_act.ln()
            })
            .collect()
    }
}

/// `_lnprior_bound_EB` (Porb_comp > 10 d).
pub fn lnprior_bound_eb(m_s: f64, plx: f64, delta_mags: &[f64]) -> Vec<f64> {
    let plx = if plx.is_nan() { 0.1 } else { plx };
    let d = 1000.0 / plx;
    let seps: Vec<f64> = separation_at_contrast(delta_mags)
        .iter()
        .map(|&s| d * s)
        .collect();
    let m_s_eff = if m_s >= 1.0 { m_s } else { 1.0 };
    let max_p: Vec<f64> = seps
        .iter()
        .map(|&s| {
            let x3 = libm_pow(s * AU, 3.0);
            libm_pow((4.0 * (PI * PI)) / (G_CGS * m_s_eff * MSUN) * x3, 0.5) / 86400.0
        })
        .collect();
    let log_mp: Vec<f64> = max_p.iter().map(|&x| x.log10()).collect();
    let f_comp = bound_rate_pieces(m_s_eff, &log_mp, 1.0, true);
    if m_s >= 1.0 {
        f_comp.iter().map(|&v| v.ln()).collect()
    } else {
        f_comp
            .iter()
            .map(|&v| {
                let f_act = 0.65 * v + 0.35 * v * m_s;
                let f_act = if f_act < 0.0 { 0.0 } else { f_act };
                f_act.ln()
            })
            .collect()
    }
}

/// `_lnprior_companion`: per-draw bound-companion log-prior, `<= 0`, `-inf` when the companion is brighter.
pub fn lnprior_companion(m_s: f64, plx: f64, fluxratios_comp: &[f64], scenario: &str) -> Vec<f64> {
    let delta_mags: Vec<f64> = fluxratios_comp
        .iter()
        .map(|&fr| 2.5 * (fr / (1.0 - fr)).log10())
        .collect();
    let abs_dm: Vec<f64> = delta_mags.iter().map(|&x| x.abs()).collect();
    let mut lp = if scenario == "TP" {
        lnprior_bound_tp(m_s, plx, &abs_dm)
    } else {
        lnprior_bound_eb(m_s, plx, &abs_dm)
    };
    for i in 0..lp.len() {
        if lp[i] > 0.0 {
            lp[i] = 0.0;
        }
        if delta_mags[i] > 0.0 {
            lp[i] = f64::NEG_INFINITY;
        }
    }
    lp
}

// --------------------------------------------------------------------------- log-sum-exp

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogWeightReduction {
    pub max: f64,
    pub sum_exp: f64,
    pub ln_sum: f64,
    pub ln_n: f64,
    pub lnz: f64,
}

/// Reduction intermediates for reproducing the reference's array operation
/// order in targeted numerical probes. `None` means no finite draw exists.
///
/// When the step-7 diagnostics sink is active on this thread ([`super::mc::DiagSink`]), it also appends ONE
/// draw-weight record per call, exactly as `_lnZ_from_logw` does into `_DRAW_DIAG` (r2 `:2478-2509`):
/// ESS = (s1 * s1) / s2 with pairwise s1 = sum(e), s2 = sum(e * e); max_share = 1 / s1; the tail scale; and for an
/// input with no finite value {ess 0, n, max_share 0, tail NaN}.  The returned lnZ bits never depend on it.
pub fn logweight_reduction(logw: &[f64]) -> Option<LogWeightReduction> {
    logweight_reduction_impl(logw, super::mc::sink_active())
}

fn logweight_reduction_impl(logw: &[f64], record: bool) -> Option<LogWeightReduction> {
    let n = logw.len();
    let w: Vec<f64> = logw
        .iter()
        .map(|&x| if x.is_nan() { f64::NEG_INFINITY } else { x })
        .collect();
    if !w.iter().any(|&x| x.is_finite()) {
        if record {
            super::mc::sink_push(super::mc::DrawDiag {
                ess: 0.0,
                n,
                max_share: 0.0,
                tail_scale: f64::NAN,
            });
            super::gate::push_reduction();
        }
        return None;
    }
    let m = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = w.iter().map(|&x| (x - m).exp()).collect();
    let s = pairwise_sum(&exps);
    if record {
        let sq: Vec<f64> = exps.iter().map(|&e| e * e).collect();
        let s2 = pairwise_sum(&sq);
        super::mc::sink_push(super::mc::DrawDiag {
            ess: (s * s) / s2,
            n,
            max_share: 1.0 / s,
            tail_scale: super::mc::tail_scale(&w),
        });
        super::gate::push_reduction();
    }
    let ln_sum = s.ln();
    let ln_n = (n as f64).ln();
    Some(LogWeightReduction {
        max: m,
        sum_exp: s,
        ln_sum,
        ln_n,
        lnz: m + ln_sum - ln_n,
    })
}

/// `_lnZ_from_logw`: `log(mean(exp(logw)))` over ALL draws (log-sum-exp), `-inf` when none finite.
/// Records one diagnostics entry when the step-7 sink is active (see [`logweight_reduction`]).
pub fn lnz_from_logw(logw: &[f64]) -> f64 {
    logweight_reduction(logw).map_or(f64::NEG_INFINITY, |r| r.lnz)
}

/// [`lnz_from_logw`] that never records: for probe-only reductions that Python does not perform in the pass.
pub fn lnz_from_logw_unrecorded(logw: &[f64]) -> f64 {
    logweight_reduction_impl(logw, false).map_or(f64::NEG_INFINITY, |r| r.lnz)
}

// --------------------------------------------------------------------------- bound-companion block

struct BoundCompanion {
    mc: Vec<f64>,
    rc: Vec<f64>,
    tc: Vec<f64>,
    frc: Vec<f64>,
    u1c: Vec<f64>,
    u2c: Vec<f64>,
}

/// `_bound_companion_block`: draw the bound companions and derive their properties.
fn bound_companion_block(
    rng: &mut FppRng,
    n: usize,
    m_s: f64,
    r_s: f64,
    teff: f64,
    teff_max: f64,
) -> Result<BoundCompanion, crate::fpp::error::FppError> {
    let qs_comp = sample_q_companion(rng, n, m_s);
    let mc: Vec<f64> = qs_comp.iter().map(|&q| q * m_s).collect();
    let (rc, tc) = stellar_relations_capped(&mc, &vec![r_s; n], &vec![teff; n]);
    let loggc: Vec<f64> = (0..n)
        .map(|i| {
            let r = rc[i] * RSUN;
            ((G_CGS * (mc[i] * MSUN)) / (r * r)).log10()
        })
        .collect();
    let fr_mc = flux_relation(&mc);
    let fr_ms = flux_relation(&[m_s]);
    let frc: Vec<f64> = (0..n).map(|i| fr_mc[i] / (fr_mc[i] + fr_ms[0])).collect();
    let (u1c, u2c) = ldc_companion_arrays(&tc, &loggc, teff_max)?;
    Ok(BoundCompanion {
        mc,
        rc,
        tc,
        frc,
        u1c,
        u2c,
    })
}

/// Per-draw internals of `_lnZ_STP`, exposed for the per-draw goldens: the full-length `lnL` (`-inf` on
/// unmasked draws), the clamped companion log-prior, the transit/collision/Roche mask, the production status
/// (all zeros on the parity path), the companion masses `mc` and the planet radii `rps`.
#[derive(Clone, Debug)]
pub struct StpDetails {
    pub lnz: f64,
    pub lnl: Vec<f64>,
    pub lnprior: Vec<f64>,
    pub mask: Vec<bool>,
    pub status: Vec<i64>,
    pub mc: Vec<f64>,
    pub rps: Vec<f64>,
}

/// `_lnZ_STP` — a planet around the unresolved bound companion. Returns `Err` only on a missing LDC row.
#[allow(clippy::too_many_arguments)]
pub fn lnz_stp(
    time: &[f64],
    flux: &[f64],
    sigma: &[f64],
    p_orb: f64,
    m_s: f64,
    r_s: f64,
    teff: f64,
    n: usize,
    plx: f64,
    rng: &mut FppRng,
    fold: Option<&ProdFold>,
    roche: bool,
) -> Result<f64, crate::fpp::error::FppError> {
    Ok(lnz_stp_with_details(
        time, flux, sigma, p_orb, m_s, r_s, teff, n, plx, rng, fold, roche, true,
    )?
    .lnz)
}

/// `_lnZ_STP` with the per-draw arrays (see `StpDetails`). `exposure_upstream` selects `_exposure_args`:
/// `true` = `FPP_EXPOSURE == "upstream_2min_x20"` (parity), `false` = production `(0.0, 1)`; it only
/// matters on the `fold == None` kernel path.
#[allow(clippy::too_many_arguments)]
pub fn lnz_stp_with_details(
    time: &[f64],
    flux: &[f64],
    sigma: &[f64],
    p_orb: f64,
    m_s: f64,
    r_s: f64,
    teff: f64,
    n: usize,
    plx: f64,
    rng: &mut FppRng,
    fold: Option<&ProdFold>,
    roche: bool,
    exposure_upstream: bool,
) -> Result<StpDetails, crate::fpp::error::FppError> {
    let blk = bound_companion_block(rng, n, m_s, r_s, teff, 10000.0)?;
    let mc = &blk.mc;
    let rc = &blk.rc;
    let frc = &blk.frc;
    let u1c = &blk.u1c;
    let u2c = &blk.u2c;
    let lnprior = lnprior_companion(m_s, plx, frc, "TP");

    let rps = sample_rp_array(rng, n, mc);
    let incs = sample_inc(rng, n);
    let eccs = sample_ecc(rng, n, true, p_orb);
    let argps = sample_w(rng, n);

    let p_arr = vec![p_orb; n];
    let a_cm: Vec<f64> = (0..n)
        .map(|i| {
            let p = p_arr[i] * 86400.0;
            libm_pow(
                ((G_CGS * mc[i]) * MSUN) / (4.0 * (PI * PI)) * (p * p),
                1.0 / 3.0,
            )
        })
        .collect();
    let a_rs: Vec<f64> = (0..n).map(|i| a_cm[i] / (rc[i] * RSUN)).collect();
    let ks: Vec<f64> = (0..n).map(|i| rps[i] * REARTH / (rc[i] * RSUN)).collect();
    let incs_rad: Vec<f64> = incs.iter().map(|&x| x * PI / 180.0).collect();
    let ws_rad: Vec<f64> = argps.iter().map(|&x| (90.0 - x) * PI / 180.0).collect();

    let e_corr: Vec<f64> = (0..n)
        .map(|i| (1.0 + eccs[i] * sin(argps[i] * PI / 180.0)) / (1.0 - eccs[i] * eccs[i]))
        .collect();
    let ptra: Vec<f64> = (0..n)
        .map(|i| (rps[i] * REARTH + rc[i] * RSUN) / a_cm[i] * e_corr[i])
        .collect();
    let mut inc_min = vec![PI / 2.0; n];
    let mut coll = vec![false; n];
    let mut mask = vec![false; n];
    for i in 0..n {
        if ptra[i] <= 1.0 {
            inc_min[i] = ptra[i].acos();
        }
        coll[i] = (rps[i] * REARTH + rc[i] * RSUN) > a_cm[i] * (1.0 - eccs[i]);
        mask[i] = incs_rad[i] >= inc_min[i] && !coll[i];
    }
    if roche {
        let rho_comp = vec![5.51; n];
        let rok = roche_ok(&a_cm, &eccs, mc, rc, &rho_comp);
        for i in 0..n {
            mask[i] &= rok[i];
        }
    }
    let im: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
    let mut lnl = vec![f64::NEG_INFINITY; n];
    let mut status = vec![0i64; n];
    if !im.is_empty() {
        let (exptime, nsub) = if exposure_upstream {
            (PARITY_EXPTIME_D, PARITY_NSUB)
        } else {
            (0.0, 1)
        };
        let sel = |arr: &Vec<f64>| im.iter().map(|&i| arr[i]).collect::<Vec<_>>();
        let (lnl_i, st_i) = match fold {
            None => (
                batch_transit_lnl(
                    time,
                    flux,
                    sigma,
                    &sel(&ks),
                    &sel(&p_arr),
                    &sel(&incs_rad),
                    &sel(&a_rs),
                    &sel(&eccs),
                    &sel(&ws_rad),
                    &sel(u1c),
                    &sel(u2c),
                    &sel(frc),
                    true,
                    exptime,
                    nsub,
                ),
                Vec::new(),
            ),
            Some(pf) => {
                let (l, st, _na) = batch_transit_lnl_prod(
                    pf,
                    &sel(&ks),
                    &sel(&p_arr),
                    &sel(&incs_rad),
                    &sel(&a_rs),
                    &sel(&eccs),
                    &sel(&ws_rad),
                    &sel(u1c),
                    &sel(u2c),
                    &sel(frc),
                    true,
                );
                super::gate::note("main", &im);
                (l, st)
            }
        };
        for (k, &i) in im.iter().enumerate() {
            lnl[i] = lnl_i[k];
            if fold.is_some() {
                status[i] = st_i[k];
            }
        }
    }
    let logw: Vec<f64> = (0..n).map(|i| lnl[i] + lnprior[i]).collect();
    Ok(StpDetails {
        // Python `_lnZ_STP`: `if not np.any(mask): return -np.inf` -- no reduction, so no step-7 diagnostics record
        lnz: if im.is_empty() {
            f64::NEG_INFINITY
        } else {
            super::gate::stage("main", &lnl);
            lnz_from_logw(&logw)
        },
        lnl,
        lnprior,
        mask,
        status,
        mc: mc.clone(),
        rps,
    })
}

/// Per-draw internals of `_lnZ_SEB`, exposed for the per-draw goldens: the full-length `lnL` of the EB (`lnL_eb`)
/// and of the 2P twin (`lnL_tw`), the clamped EB log-prior, the summed |Δmag| separations, the sampled `q` and
/// companion masses, the two masks and the production statuses (all zeros on the parity path).
#[derive(Clone, Debug)]
pub struct SebDetails {
    pub lnz_seb: f64,
    pub lnz_sebx2p: f64,
    pub lnl_eb: Vec<f64>,
    pub lnl_tw: Vec<f64>,
    pub lnprior: Vec<f64>,
    pub mask_eb: Vec<bool>,
    pub mask_tw: Vec<bool>,
    pub status_eb: Vec<i64>,
    pub status_tw: Vec<i64>,
    pub qs: Vec<f64>,
    pub mc: Vec<f64>,
    pub abs_dm: Vec<f64>,
}

/// `_lnZ_SEB` — an EB around the unresolved bound companion. Returns `(lnZ_SEB, lnZ_SEBx2P)`.
#[allow(clippy::too_many_arguments)]
pub fn lnz_seb(
    time: &[f64],
    flux: &[f64],
    sigma: &[f64],
    sigma_veto: f64,
    p_orb: f64,
    m_s: f64,
    r_s: f64,
    teff: f64,
    n: usize,
    plx: f64,
    rng: &mut FppRng,
    fold: Option<&ProdFold>,
    roche: bool,
    sec_consistent: bool,
    k_nudge_upstream: bool,
) -> Result<(f64, f64), crate::fpp::error::FppError> {
    let d = lnz_seb_with_details(
        time,
        flux,
        sigma,
        sigma_veto,
        p_orb,
        m_s,
        r_s,
        teff,
        n,
        plx,
        rng,
        fold,
        roche,
        sec_consistent,
        k_nudge_upstream,
        true,
    )?;
    Ok((d.lnz_seb, d.lnz_sebx2p))
}

/// `_lnZ_SEB` with the per-draw arrays (see `SebDetails`). `exposure_upstream` selects `_exposure_args`
/// on the `fold == None` kernel path (see `lnz_stp_with_details`).
#[allow(clippy::too_many_arguments)]
pub fn lnz_seb_with_details(
    time: &[f64],
    flux: &[f64],
    sigma: &[f64],
    sigma_veto: f64,
    p_orb: f64,
    m_s: f64,
    r_s: f64,
    teff: f64,
    n: usize,
    plx: f64,
    rng: &mut FppRng,
    fold: Option<&ProdFold>,
    roche: bool,
    sec_consistent: bool,
    k_nudge_upstream: bool,
    exposure_upstream: bool,
) -> Result<SebDetails, crate::fpp::error::FppError> {
    let incs = sample_inc(rng, n);
    let qs = sample_q(rng, n, m_s);
    let eccs = sample_ecc(rng, n, false, p_orb);
    let argps = sample_w(rng, n);
    let blk = bound_companion_block(rng, n, m_s, r_s, teff, 13000.0)?;
    let mc = &blk.mc;
    let rc = &blk.rc;
    let tc = &blk.tc;
    let frc = &blk.frc;
    let u1c = &blk.u1c;
    let u2c = &blk.u2c;

    let masses: Vec<f64> = (0..n).map(|i| qs[i] * mc[i]).collect();
    let (reb, teb) = stellar_relations_capped(&masses, rc, tc);
    let fr_masses = flux_relation(&masses);
    let fr_ms = flux_relation(&[m_s]);
    let freb: Vec<f64> = (0..n)
        .map(|i| fr_masses[i] / (fr_masses[i] + fr_ms[0]))
        .collect();
    let dm: Vec<f64> = (0..n)
        .map(|i| 2.5 * ((frc[i] / (1.0 - frc[i])) + (freb[i] / (1.0 - freb[i]))).log10())
        .collect();
    let abs_dm: Vec<f64> = dm.iter().map(|&x| x.abs()).collect();
    let mut lnprior = lnprior_bound_eb(m_s, plx, &abs_dm);
    for i in 0..n {
        if lnprior[i] > 0.0 {
            lnprior[i] = 0.0;
        }
        if dm[i] > 0.0 {
            lnprior[i] = f64::NEG_INFINITY;
        }
    }

    let p_arr = vec![p_orb; n];
    let e_corr: Vec<f64> = (0..n)
        .map(|i| (1.0 + eccs[i] * sin(argps[i] * PI / 180.0)) / (1.0 - eccs[i] * eccs[i]))
        .collect();
    let a_cm: Vec<f64> = (0..n)
        .map(|i| {
            let p = p_arr[i] * 86400.0;
            libm_pow(
                ((G_CGS * (mc[i] + masses[i])) * MSUN) / (4.0 * (PI * PI)) * (p * p),
                1.0 / 3.0,
            )
        })
        .collect();
    let a_tw: Vec<f64> = (0..n)
        .map(|i| {
            let p = 2.0 * p_arr[i] * 86400.0;
            libm_pow(
                ((G_CGS * (mc[i] + masses[i])) * MSUN) / (4.0 * (PI * PI)) * (p * p),
                1.0 / 3.0,
            )
        })
        .collect();
    let ptra: Vec<f64> = (0..n)
        .map(|i| (reb[i] * RSUN + rc[i] * RSUN) / a_cm[i] * e_corr[i])
        .collect();
    let ptra_tw: Vec<f64> = (0..n)
        .map(|i| (reb[i] * RSUN + rc[i] * RSUN) / a_tw[i] * e_corr[i])
        .collect();
    let mut inc_min = vec![PI / 2.0; n];
    let mut inc_min_tw = vec![PI / 2.0; n];
    let mut coll = vec![false; n];
    let mut coll_tw = vec![false; n];
    let incs_rad: Vec<f64> = incs.iter().map(|&x| x * PI / 180.0).collect();
    let ws_rad: Vec<f64> = argps.iter().map(|&x| (90.0 - x) * PI / 180.0).collect();
    let mut mask_eb = vec![false; n];
    let mut mask_tw = vec![false; n];
    for i in 0..n {
        if ptra[i] <= 1.0 {
            inc_min[i] = ptra[i].acos();
        }
        if ptra_tw[i] <= 1.0 {
            inc_min_tw[i] = ptra_tw[i].acos();
        }
        coll[i] = (reb[i] * RSUN + rc[i] * RSUN) > a_cm[i] * (1.0 - eccs[i]);
        coll_tw[i] = (2.0 * rc[i] * RSUN) > a_tw[i] * (1.0 - eccs[i]);
        mask_eb[i] = incs_rad[i] >= inc_min[i] && !coll[i] && qs[i] < 0.95;
        mask_tw[i] = incs_rad[i] >= inc_min_tw[i] && !coll_tw[i] && qs[i] >= 0.95;
    }
    if roche {
        let rho_eb: Vec<f64> = (0..n)
            .map(|i| {
                let r = reb[i] * RSUN;
                (masses[i] * MSUN) / ((4.0 / 3.0) * PI * (r * r * r))
            })
            .collect();
        let rok_eb = roche_ok(&a_cm, &eccs, mc, rc, &rho_eb);
        let rok_tw = roche_ok(&a_tw, &eccs, mc, rc, &rho_eb);
        for i in 0..n {
            mask_eb[i] &= rok_eb[i];
            mask_tw[i] &= rok_tw[i];
        }
    }
    let ks: Vec<f64> = (0..n).map(|i| reb[i] / rc[i]).collect();
    let a_rs: Vec<f64> = (0..n).map(|i| a_cm[i] / (rc[i] * RSUN)).collect();
    let a_tw_rs: Vec<f64> = (0..n).map(|i| a_tw[i] / (rc[i] * RSUN)).collect();

    // secondary-eclipse limb darkening
    let mut u1_sec = u1c.clone();
    let mut u2_sec = u2c.clone();
    if sec_consistent {
        let sel: Vec<usize> = (0..n).filter(|&i| mask_eb[i] || mask_tw[i]).collect();
        if !sel.is_empty() {
            let teb_sel: Vec<f64> = sel.iter().map(|&i| teb[i]).collect();
            let logg_eb: Vec<f64> = sel
                .iter()
                .map(|&i| {
                    let r = reb[i] * RSUN;
                    ((G_CGS * masses[i] * MSUN) / (r * r)).log10()
                })
                .collect();
            let (u1s, u2s) = get_ldc_arrays(&teb_sel, &logg_eb)?;
            for (k, &i) in sel.iter().enumerate() {
                u1_sec[i] = u1s[k];
                u2_sec[i] = u2s[k];
            }
        }
    }

    let (exptime, nsub) = if exposure_upstream {
        (PARITY_EXPTIME_D, PARITY_NSUB)
    } else {
        (0.0, 1)
    };

    let run = |mask: &[bool],
               pv: &[f64],
               ars: &[f64],
               with_sec: bool|
     -> (Vec<f64>, Vec<i64>, Vec<i64>) {
        let mut lnl = vec![f64::NEG_INFINITY; n];
        let im: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
        if im.is_empty() {
            return (lnl, vec![0i64; n], vec![0i64; n]);
        }
        let sel_ks: Vec<f64> = im.iter().map(|&i| ks[i]).collect();
        let sel_pv: Vec<f64> = im.iter().map(|&i| pv[i]).collect();
        let sel_ars: Vec<f64> = im.iter().map(|&i| ars[i]).collect();
        let sel_inc: Vec<f64> = im.iter().map(|&i| incs_rad[i]).collect();
        let sel_ecc: Vec<f64> = im.iter().map(|&i| eccs[i]).collect();
        let sel_w: Vec<f64> = im.iter().map(|&i| ws_rad[i]).collect();
        let sel_u1: Vec<f64> = im.iter().map(|&i| u1c[i]).collect();
        let sel_u2: Vec<f64> = im.iter().map(|&i| u2c[i]).collect();
        let sel_freb: Vec<f64> = im.iter().map(|&i| freb[i]).collect();
        let sel_frc: Vec<f64> = im.iter().map(|&i| frc[i]).collect();
        let sel_u1s: Vec<f64> = im.iter().map(|&i| u1_sec[i]).collect();
        let sel_u2s: Vec<f64> = im.iter().map(|&i| u2_sec[i]).collect();
        let (l, st, na) = match fold {
            None => {
                let l = batch_eb_transit_lnl(
                    time,
                    flux,
                    sigma,
                    sigma_veto,
                    &sel_ks,
                    &sel_freb,
                    &sel_pv,
                    &sel_inc,
                    &sel_ars,
                    &sel_ecc,
                    &sel_w,
                    &sel_u1,
                    &sel_u2,
                    &sel_frc,
                    true,
                    exptime,
                    nsub,
                    with_sec,
                    sec_consistent,
                    k_nudge_upstream,
                    &sel_u1s,
                    &sel_u2s,
                );
                (l, vec![0i64; im.len()], vec![0i64; im.len()])
            }
            Some(pf) => {
                let out = batch_eb_transit_lnl_prod(
                    pf,
                    sigma_veto,
                    &sel_ks,
                    &sel_freb,
                    &sel_pv,
                    &sel_inc,
                    &sel_ars,
                    &sel_ecc,
                    &sel_w,
                    &sel_u1,
                    &sel_u2,
                    &sel_frc,
                    true,
                    with_sec,
                    sec_consistent,
                    k_nudge_upstream,
                    &sel_u1s,
                    &sel_u2s,
                );
                super::gate::note(if with_sec { "eb" } else { "tw" }, &im);
                out
            }
        };
        for (k, &i) in im.iter().enumerate() {
            lnl[i] = l[k];
        }
        let mut status = vec![0i64; n];
        let mut n_aff = vec![0i64; n];
        for (k, &i) in im.iter().enumerate() {
            status[i] = st[k];
            n_aff[i] = na[k];
        }
        (lnl, status, n_aff)
    };

    let (lnl_eb, st_eb, _na_eb) = run(&mask_eb, &p_arr, &a_rs, true);
    let (lnl_tw, st_tw, _na_tw) = run(
        &mask_tw,
        &p_arr.iter().map(|&x| 2.0 * x).collect::<Vec<_>>(),
        &a_tw_rs,
        false,
    );
    let logw_eb: Vec<f64> = (0..n).map(|i| lnl_eb[i] + lnprior[i]).collect();
    let logw_tw: Vec<f64> = (0..n).map(|i| lnl_tw[i] + lnprior[i]).collect();
    Ok(SebDetails {
        lnz_seb: {
            super::gate::stage("eb", &lnl_eb);
            lnz_from_logw(&logw_eb)
        },
        lnz_sebx2p: {
            super::gate::stage("tw", &lnl_tw);
            lnz_from_logw(&logw_tw)
        },
        lnl_eb,
        lnl_tw,
        lnprior,
        mask_eb,
        mask_tw,
        status_eb: st_eb,
        status_tw: st_tw,
        qs,
        mc: mc.clone(),
        abs_dm,
    })
}
