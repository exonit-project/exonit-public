//! Corrected target-host FPP scenarios (TTP, TEB, PTP, PEB).
//!
//! Literal port of `worker/fpp_native.py` `_lnZ_TTP`, `_lnZ_TEB`, `_lnZ_PTP`,
//! and `_lnZ_PEB`. These functions use the v2 likelihood kernels and true
//! log-weight reduction; the legacy `fpp::scenarios` likelihood path has a
//! different normalization and is deliberately not called here.

use super::{
    consts::{G_CGS, MSUN, PARITY_EXPTIME_D, PARITY_NSUB, REARTH, RSUN},
    math::sin,
    s_family::{
        batch_eb_transit_lnl, batch_eb_transit_lnl_prod, batch_transit_lnl, batch_transit_lnl_prod,
        lnprior_companion, lnz_from_logw, roche_ok, ProdFold,
    },
};
use crate::fpp::{
    npsum::{libm_pow, pairwise_sum},
    priors::{sample_ecc, sample_inc, sample_q, sample_q_companion, sample_rp, sample_w},
    rng::FppRng,
    stellar::{flux_relation, get_ldc_arrays, stellar_relations},
};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct TargetFamilyConfig {
    pub roche: bool,
    pub secondary_consistent: bool,
    pub k_nudge_upstream: bool,
    pub upstream_exposure: bool,
}

impl Default for TargetFamilyConfig {
    fn default() -> Self {
        Self {
            roche: false,
            secondary_consistent: false,
            k_nudge_upstream: true,
            upstream_exposure: true,
        }
    }
}

/// NaN likelihoods are unconverged Kepler draws. Callers keep one instance per
/// candidate so a verdict cap cannot accidentally include another candidate's
/// failures.
#[derive(Clone, Debug, Default)]
pub struct TargetDiagnostics {
    pub kepler_unconverged_draws: u64,
    pub exposure_gl_fallback_share: f64,
    pub exposure_unconverged_share: f64,
    pub exposure_unconverged: bool,
    /// Request a small TEB trace for golden/debug comparisons. Disabled by default.
    pub capture_teb_probe: bool,
    /// Last TEB call's first candidates and included draws, when requested.
    pub teb_probe: Option<TebProbe>,
}

#[derive(Clone, Debug)]
pub struct TebProbeDraw {
    pub draw: usize,
    pub q: f64,
    pub inc_deg: f64,
    pub ecc: f64,
    pub arg_deg: f64,
    pub k: f64,
    pub mask: bool,
    pub ln_l: f64,
    pub status: i64,
}

#[derive(Clone, Debug, Default)]
pub struct TebProbeBranch {
    pub first8_candidates: Vec<TebProbeDraw>,
    pub first8_included: Vec<TebProbeDraw>,
    pub all_ln_l: Vec<f64>,
    pub all_logw: Vec<f64>,
    pub all_status: Vec<i64>,
    pub all_q: Vec<f64>,
    pub all_inc_deg: Vec<f64>,
    pub all_ecc: Vec<f64>,
    pub all_arg_deg: Vec<f64>,
    pub all_k: Vec<f64>,
    pub all_mask: Vec<bool>,
}

#[derive(Clone, Debug, Default)]
pub struct TebProbe {
    pub eb: TebProbeBranch,
    pub twin: TebProbeBranch,
    pub lnz_eb: f64,
    pub lnz_twin: f64,
}

impl TargetDiagnostics {
    fn observe(&mut self, values: &[f64]) {
        self.kepler_unconverged_draws += values.iter().filter(|v| v.is_nan()).count() as u64;
    }

    /// Match Python `_note_exposure_stats`: status shares use finite-likelihood
    /// draws, weights `exp(lnL - max(lnL))`, and maxima are retained per
    /// candidate across its production batches.
    pub fn observe_production(&mut self, values: &[f64], status: &[i64]) {
        debug_assert_eq!(values.len(), status.len());
        let max_lnl = values
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        if !max_lnl.is_finite() {
            return;
        }
        let weighted: Vec<(f64, i64)> = values
            .iter()
            .zip(status)
            .filter(|(lnl, _)| lnl.is_finite())
            .map(|(&lnl, &st)| ((lnl - max_lnl).exp(), st))
            .collect();
        let total = pairwise_sum(&weighted.iter().map(|x| x.0).collect::<Vec<_>>());
        if !(total > 0.0) {
            return;
        }
        let fallback = pairwise_sum(
            &weighted
                .iter()
                .filter(|x| x.1 == 1)
                .map(|x| x.0)
                .collect::<Vec<_>>(),
        ) / total;
        let unconverged = pairwise_sum(
            &weighted
                .iter()
                .filter(|x| x.1 == 2)
                .map(|x| x.0)
                .collect::<Vec<_>>(),
        ) / total;
        self.exposure_gl_fallback_share = self.exposure_gl_fallback_share.max(fallback);
        self.exposure_unconverged_share = self.exposure_unconverged_share.max(unconverged);
        self.exposure_unconverged = self.exposure_unconverged_share > 0.01;
    }
}

fn broadcast(values: &[f64], n: usize, name: &str) -> Result<Vec<f64>, String> {
    if values.len() == n {
        Ok(values.to_vec())
    } else if values.len() == 1 {
        Ok(vec![values[0]; n])
    } else {
        Err(format!(
            "{name} must contain one value or N values ({} != {n})",
            values.len()
        ))
    }
}

fn exposure(c: TargetFamilyConfig) -> (f64, usize) {
    if c.upstream_exposure {
        (PARITY_EXPTIME_D, PARITY_NSUB)
    } else {
        (0.0, 1)
    }
}

fn target_axes(period: f64, mass: f64, radius: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let p_sec = period * 86400.0;
    let p2 = p_sec * p_sec;
    let denom = 4.0 * libm_pow(PI, 2.0);
    let mut a_cm = Vec::with_capacity(n);
    let mut a_rs = Vec::with_capacity(n);
    for _ in 0..n {
        let a = libm_pow(G_CGS * mass * MSUN / denom * p2, 1.0 / 3.0);
        a_cm.push(a);
        a_rs.push(a / (radius * RSUN));
    }
    (a_cm, a_rs)
}

fn validate_n(
    n: usize,
    time: &[f64],
    flux: &[f64],
    fold: Option<&ProdFold<'_>>,
) -> Result<(), String> {
    if n == 0 {
        return Err("target-family evaluation requires N > 0".into());
    }
    if fold.is_none() {
        if time.len() != flux.len() {
            return Err("time and flux lengths differ".into());
        }
        if time.is_empty() {
            return Err("target-family evaluation requires a nonempty light curve".into());
        }
    }
    Ok(())
}

/// Target-host transiting planet, also used for nearby NTP with per-draw
/// dilution. `companion_fluxratio` and `lnprior_companion` may be scalar-like
/// (`len == 1`) or contain exactly N values.
#[allow(clippy::too_many_arguments)]
pub fn lnz_ttp(
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    period: f64,
    mass: f64,
    radius: f64,
    n: usize,
    u1: f64,
    u2: f64,
    companion_fluxratio: &[f64],
    companion_is_host: bool,
    lnprior_companion: &[f64],
    rng: &mut FppRng,
    config: TargetFamilyConfig,
    diagnostics: &mut TargetDiagnostics,
    fold: Option<&ProdFold<'_>>,
) -> Result<f64, crate::fpp::error::FppError> {
    validate_n(n, time, flux, fold)?;
    let comp_fr = broadcast(companion_fluxratio, n, "companion_fluxratio")?;
    let prior = broadcast(lnprior_companion, n, "lnprior_companion")?;
    let (a_cm, a_rs) = target_axes(period, mass, radius, n);

    // Upstream consumes four vector draws in this order.
    let rp = sample_rp(rng, n, mass);
    let inc = sample_inc(rng, n);
    let ecc = sample_ecc(rng, n, true, period);
    let arg = sample_w(rng, n);
    let mut k = Vec::with_capacity(n);
    let mut inc_rad = Vec::with_capacity(n);
    let mut w_rad = Vec::with_capacity(n);
    let mut mask = vec![false; n];
    for i in 0..n {
        k.push(rp[i] * REARTH / (radius * RSUN));
        inc_rad.push(inc[i] * PI / 180.0);
        w_rad.push((90.0 - arg[i]) * PI / 180.0);
        let e_corr = (1.0 + ecc[i] * sin(arg[i] * PI / 180.0)) / (1.0 - ecc[i] * ecc[i]);
        let ptra = (rp[i] * REARTH + radius * RSUN) / a_cm[i] * e_corr;
        let inc_min = if ptra <= 1.0 { ptra.acos() } else { PI / 2.0 };
        let coll = (rp[i] * REARTH + radius * RSUN) > a_cm[i] * (1.0 - ecc[i]);
        mask[i] = inc_rad[i] >= inc_min && !coll;
    }
    if config.roche {
        let roche = roche_ok(
            &a_cm,
            &ecc,
            &vec![mass; n],
            &vec![radius; n],
            &vec![5.51; n],
        );
        for i in 0..n {
            mask[i] &= roche[i];
        }
    }

    let mut ln_l = vec![f64::NEG_INFINITY; n];
    let selected: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
    if !selected.is_empty() {
        let ks: Vec<f64> = selected.iter().map(|&i| k[i]).collect();
        let periods = vec![period; selected.len()];
        let incs: Vec<f64> = selected.iter().map(|&i| inc_rad[i]).collect();
        let axes: Vec<f64> = selected.iter().map(|&i| a_rs[i]).collect();
        let eccs: Vec<f64> = selected.iter().map(|&i| ecc[i]).collect();
        let args: Vec<f64> = selected.iter().map(|&i| w_rad[i]).collect();
        let fr: Vec<f64> = selected.iter().map(|&i| comp_fr[i]).collect();
        let u1s = vec![u1; selected.len()];
        let u2s = vec![u2; selected.len()];
        let values = if let Some(pf) = fold {
            let (values, status, _n_aff) = batch_transit_lnl_prod(
                pf,
                &ks,
                &periods,
                &incs,
                &axes,
                &eccs,
                &args,
                &u1s,
                &u2s,
                &fr,
                companion_is_host,
            );
            diagnostics.observe_production(&values, &status);
            super::gate::note("main", &selected);
            values
        } else {
            batch_transit_lnl(
                time,
                flux,
                &vec![sigma; time.len()],
                &ks,
                &periods,
                &incs,
                &axes,
                &eccs,
                &args,
                &u1s,
                &u2s,
                &fr,
                companion_is_host,
                exposure(config).0,
                exposure(config).1,
            )
        };
        diagnostics.observe(&values);
        for (j, &i) in selected.iter().enumerate() {
            ln_l[i] = values[j];
        }
    }
    if selected.is_empty() {
        // Python `_lnZ_TTP`: `if not np.any(mask): return -np.inf` -- no reduction, so no step-7 diagnostics record
        return Ok(f64::NEG_INFINITY);
    }
    super::gate::stage("main", &ln_l);
    for i in 0..n {
        ln_l[i] += prior[i];
    }
    Ok(lnz_from_logw(&ln_l))
}

/// Target-host eclipsing binary and twin branch, also used for nearby NEB.
#[allow(clippy::too_many_arguments)]
pub fn lnz_teb(
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    sigma_veto: f64,
    period: f64,
    mass: f64,
    radius: f64,
    teff: f64,
    n: usize,
    u1: f64,
    u2: f64,
    companion_fluxratio: &[f64],
    companion_is_host: bool,
    lnprior_companion: &[f64],
    rng: &mut FppRng,
    config: TargetFamilyConfig,
    diagnostics: &mut TargetDiagnostics,
    fold: Option<&ProdFold<'_>>,
) -> Result<(f64, f64), crate::fpp::error::FppError> {
    validate_n(n, time, flux, fold)?;
    let comp_fr = broadcast(companion_fluxratio, n, "companion_fluxratio")?;
    let prior = broadcast(lnprior_companion, n, "lnprior_companion")?;

    // Upstream consumes inclination, target-mass q, eccentricity, and omega.
    let inc = sample_inc(rng, n);
    let q = sample_q(rng, n, mass);
    let ecc = sample_ecc(rng, n, false, period);
    let arg = sample_w(rng, n);
    let masses: Vec<f64> = q.iter().map(|&x| x * mass).collect();
    let (mut radii, mut teffs) = stellar_relations(&masses);
    for i in 0..n {
        radii[i] = radii[i].min(radius);
        teffs[i] = teffs[i].min(teff);
    }
    let flux_eb = flux_relation(&masses);
    let flux_target = flux_relation(&[mass])[0];
    let fluxratio_eb: Vec<f64> = flux_eb.iter().map(|&x| x / (x + flux_target)).collect();
    let inc_rad: Vec<f64> = inc.iter().map(|&x| x * PI / 180.0).collect();
    let w_rad: Vec<f64> = arg.iter().map(|&x| (90.0 - x) * PI / 180.0).collect();

    let denom = 4.0 * libm_pow(PI, 2.0);
    let p_sec = period * 86400.0;
    let p2 = p_sec * p_sec;
    let p_twin_sec = 2.0 * period * 86400.0;
    let p_twin2 = p_twin_sec * p_twin_sec;
    let mut a_cm = Vec::with_capacity(n);
    let mut a_twin_cm = Vec::with_capacity(n);
    let mut a_rs = Vec::with_capacity(n);
    let mut a_twin_rs = Vec::with_capacity(n);
    let mut ks = Vec::with_capacity(n);
    let mut mask_eb = vec![false; n];
    let mut mask_twin = vec![false; n];
    for i in 0..n {
        let ac = libm_pow(G_CGS * (mass + masses[i]) * MSUN / denom * p2, 1.0 / 3.0);
        let at = libm_pow(
            G_CGS * (mass + masses[i]) * MSUN / denom * p_twin2,
            1.0 / 3.0,
        );
        a_cm.push(ac);
        a_twin_cm.push(at);
        a_rs.push(ac / (radius * RSUN));
        a_twin_rs.push(at / (radius * RSUN));
        ks.push(radii[i] / radius);
        let e_corr = (1.0 + ecc[i] * sin(arg[i] * PI / 180.0)) / (1.0 - ecc[i] * ecc[i]);
        let ptra = (radii[i] * RSUN + radius * RSUN) / ac * e_corr;
        let ptra_twin = (radii[i] * RSUN + radius * RSUN) / at * e_corr;
        let inc_min = if ptra <= 1.0 { ptra.acos() } else { PI / 2.0 };
        let inc_min_twin = if ptra_twin <= 1.0 {
            ptra_twin.acos()
        } else {
            PI / 2.0
        };
        let collision = (radii[i] * RSUN + radius * RSUN) > ac * (1.0 - ecc[i]);
        let twin_collision = (2.0 * radius * RSUN) > at * (1.0 - ecc[i]);
        mask_eb[i] = inc_rad[i] >= inc_min && !collision && q[i] < 0.95;
        mask_twin[i] = inc_rad[i] >= inc_min_twin && !twin_collision && q[i] >= 0.95;
    }
    if config.roche {
        let rho: Vec<f64> = (0..n)
            .map(|i| {
                let r_cm = radii[i] * RSUN;
                masses[i] * MSUN / ((4.0 / 3.0) * PI * r_cm * r_cm * r_cm)
            })
            .collect();
        // The Roche density belongs to the sampled EB companion, but the
        // Roche-host density is the target star (Python `_lnZ_TEB` passes
        // M_s/R_s with `rho_comp`).
        let target_masses = vec![mass; n];
        let target_radii = vec![radius; n];
        let roche_eb = roche_ok(&a_cm, &ecc, &target_masses, &target_radii, &rho);
        let roche_twin = roche_ok(&a_twin_cm, &ecc, &target_masses, &target_radii, &rho);
        for i in 0..n {
            mask_eb[i] &= roche_eb[i];
            mask_twin[i] &= roche_twin[i];
        }
    }

    let mut u1_sec = vec![u1; n];
    let mut u2_sec = vec![u2; n];
    if config.secondary_consistent {
        let selected: Vec<usize> = (0..n).filter(|&i| mask_eb[i] || mask_twin[i]).collect();
        if !selected.is_empty() {
            let mut teff_sel = Vec::with_capacity(selected.len());
            let mut logg_sel = Vec::with_capacity(selected.len());
            for &i in &selected {
                let rr = radii[i] * RSUN;
                teff_sel.push(teffs[i]);
                logg_sel.push((G_CGS * masses[i] * MSUN / (rr * rr)).log10());
            }
            let (u1_sel, u2_sel) = get_ldc_arrays(&teff_sel, &logg_sel)?;
            for (j, &i) in selected.iter().enumerate() {
                u1_sec[i] = u1_sel[j];
                u2_sec[i] = u2_sel[j];
            }
        }
    }

    let mut ln_l_eb = vec![f64::NEG_INFINITY; n];
    let mut ln_l_twin = vec![f64::NEG_INFINITY; n];
    let mut status_eb = vec![0i64; n];
    let mut status_twin = vec![0i64; n];
    let ex = exposure(config);
    for (branch, keep, branch_period, branch_axis, output, output_status) in [
        (true, &mask_eb, period, &a_rs, &mut ln_l_eb, &mut status_eb),
        (
            false,
            &mask_twin,
            2.0 * period,
            &a_twin_rs,
            &mut ln_l_twin,
            &mut status_twin,
        ),
    ] {
        let selected: Vec<usize> = (0..n).filter(|&i| keep[i]).collect();
        if selected.is_empty() {
            continue;
        }
        let ks_sel: Vec<f64> = selected.iter().map(|&i| ks[i]).collect();
        let fr_eb_sel: Vec<f64> = selected.iter().map(|&i| fluxratio_eb[i]).collect();
        let periods = vec![branch_period; selected.len()];
        let incs: Vec<f64> = selected.iter().map(|&i| inc_rad[i]).collect();
        let axes: Vec<f64> = selected.iter().map(|&i| branch_axis[i]).collect();
        let eccs: Vec<f64> = selected.iter().map(|&i| ecc[i]).collect();
        let args: Vec<f64> = selected.iter().map(|&i| w_rad[i]).collect();
        let comp: Vec<f64> = selected.iter().map(|&i| comp_fr[i]).collect();
        let u1s = vec![u1; selected.len()];
        let u2s = vec![u2; selected.len()];
        let u1sec: Vec<f64> = selected.iter().map(|&i| u1_sec[i]).collect();
        let u2sec: Vec<f64> = selected.iter().map(|&i| u2_sec[i]).collect();
        let result = if let Some(pf) = fold {
            let (values, status, _n_aff) = batch_eb_transit_lnl_prod(
                pf,
                sigma_veto,
                &ks_sel,
                &fr_eb_sel,
                &periods,
                &incs,
                &axes,
                &eccs,
                &args,
                &u1s,
                &u2s,
                &comp,
                companion_is_host,
                branch,
                config.secondary_consistent,
                config.k_nudge_upstream,
                &u1sec,
                &u2sec,
            );
            diagnostics.observe_production(&values, &status);
            super::gate::note(if branch { "eb" } else { "twin" }, &selected);
            for (j, &i) in selected.iter().enumerate() {
                output_status[i] = status[j];
            }
            values
        } else {
            batch_eb_transit_lnl(
                time,
                flux,
                &vec![sigma; time.len()],
                sigma_veto,
                &ks_sel,
                &fr_eb_sel,
                &periods,
                &incs,
                &axes,
                &eccs,
                &args,
                &u1s,
                &u2s,
                &comp,
                companion_is_host,
                ex.0,
                ex.1,
                branch,
                config.secondary_consistent,
                config.k_nudge_upstream,
                &u1sec,
                &u2sec,
            )
        };
        diagnostics.observe(&result);
        for (j, &i) in selected.iter().enumerate() {
            output[i] = result[j];
        }
    }
    if diagnostics.capture_teb_probe {
        let make_branch = |keep: &[bool], lnl: &[f64], status: &[i64], logw: &[f64]| {
            let make_draw = |i: usize| TebProbeDraw {
                draw: i,
                q: q[i],
                inc_deg: inc[i],
                ecc: ecc[i],
                arg_deg: arg[i],
                k: ks[i],
                mask: keep[i],
                ln_l: lnl[i],
                status: status[i],
            };
            TebProbeBranch {
                first8_candidates: (0..n.min(8)).map(make_draw).collect(),
                first8_included: (0..n)
                    .filter(|&i| keep[i] && lnl[i].is_finite())
                    .take(8)
                    .map(make_draw)
                    .collect(),
                all_ln_l: lnl.to_vec(),
                all_logw: logw.to_vec(),
                all_status: status.to_vec(),
                all_q: q.clone(),
                all_inc_deg: inc.clone(),
                all_ecc: ecc.clone(),
                all_arg_deg: arg.clone(),
                all_k: ks.clone(),
                all_mask: keep.to_vec(),
            }
        };
        let prior_eb: Vec<f64> = (0..n).map(|i| ln_l_eb[i] + prior[i]).collect();
        let prior_twin: Vec<f64> = (0..n).map(|i| ln_l_twin[i] + prior[i]).collect();
        diagnostics.teb_probe = Some(TebProbe {
            eb: make_branch(&mask_eb, &ln_l_eb, &status_eb, &prior_eb),
            twin: make_branch(&mask_twin, &ln_l_twin, &status_twin, &prior_twin),
            lnz_eb: super::s_family::lnz_from_logw_unrecorded(&prior_eb),
            lnz_twin: super::s_family::lnz_from_logw_unrecorded(&prior_twin),
        });
    }
    let gate_lnl = super::gate::active().then(|| (ln_l_eb.clone(), ln_l_twin.clone()));
    for i in 0..n {
        ln_l_eb[i] += prior[i];
        ln_l_twin[i] += prior[i];
    }
    if let Some((g, _)) = &gate_lnl {
        super::gate::stage("eb", g);
    }
    let lnz_eb = lnz_from_logw(&ln_l_eb);
    if let Some((_, g)) = &gate_lnl {
        super::gate::stage("twin", g);
    }
    let lnz_twin = lnz_from_logw(&ln_l_twin);
    Ok((lnz_eb, lnz_twin))
}

/// Planet around a bound physical companion. The companion draw precedes the
/// target-host TTP draws, matching Python's RNG stream exactly.
#[allow(clippy::too_many_arguments)]
pub fn lnz_ptp(
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    period: f64,
    mass: f64,
    radius: f64,
    n: usize,
    u1: f64,
    u2: f64,
    parallax_mas: f64,
    rng: &mut FppRng,
    config: TargetFamilyConfig,
    diagnostics: &mut TargetDiagnostics,
    fold: Option<&ProdFold<'_>>,
) -> Result<f64, crate::fpp::error::FppError> {
    validate_n(n, time, flux, fold)?;
    let q_comp = sample_q_companion(rng, n, mass);
    let masses_comp: Vec<f64> = q_comp.iter().map(|&q| q * mass).collect();
    let flux_comp = flux_relation(&masses_comp);
    let flux_target = flux_relation(&[mass])[0];
    let fr: Vec<f64> = flux_comp.iter().map(|&x| x / (x + flux_target)).collect();
    let prior = lnprior_companion(mass, parallax_mas, &fr, "TP");
    lnz_ttp(
        time,
        flux,
        sigma,
        period,
        mass,
        radius,
        n,
        u1,
        u2,
        &fr,
        false,
        &prior,
        rng,
        config,
        diagnostics,
        fold,
    )
}

/// Eclipsing binary around a bound physical companion. As in Python the same
/// per-draw companion prior is applied to both EB branches.
#[allow(clippy::too_many_arguments)]
pub fn lnz_peb(
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    sigma_veto: f64,
    period: f64,
    mass: f64,
    radius: f64,
    teff: f64,
    n: usize,
    u1: f64,
    u2: f64,
    parallax_mas: f64,
    rng: &mut FppRng,
    config: TargetFamilyConfig,
    diagnostics: &mut TargetDiagnostics,
    fold: Option<&ProdFold<'_>>,
) -> Result<(f64, f64), crate::fpp::error::FppError> {
    validate_n(n, time, flux, fold)?;
    let q_comp = sample_q_companion(rng, n, mass);
    let masses_comp: Vec<f64> = q_comp.iter().map(|&q| q * mass).collect();
    let flux_comp = flux_relation(&masses_comp);
    let flux_target = flux_relation(&[mass])[0];
    let fr: Vec<f64> = flux_comp.iter().map(|&x| x / (x + flux_target)).collect();
    let prior = lnprior_companion(mass, parallax_mas, &fr, "EB");
    lnz_teb(
        time,
        flux,
        sigma,
        sigma_veto,
        period,
        mass,
        radius,
        teff,
        n,
        u1,
        u2,
        &fr,
        false,
        &prior,
        rng,
        config,
        diagnostics,
        fold,
    )
}

#[cfg(test)]
mod diagnostics_tests {
    use super::TargetDiagnostics;

    #[test]
    fn production_exposure_shares_ignore_nonfinite_likelihoods_and_keep_maxima() {
        let mut d = TargetDiagnostics::default();
        d.observe_production(
            &[0.0, 0.0, f64::NEG_INFINITY, f64::NAN, -100.0],
            &[1, 2, 2, 2, 0],
        );
        assert_eq!(d.exposure_gl_fallback_share, 0.5);
        assert_eq!(d.exposure_unconverged_share, 0.5);
        assert!(d.exposure_unconverged);

        d.observe_production(&[0.0, 0.0], &[0, 0]);
        assert_eq!(d.exposure_gl_fallback_share, 0.5);
        assert_eq!(d.exposure_unconverged_share, 0.5);
    }
}
