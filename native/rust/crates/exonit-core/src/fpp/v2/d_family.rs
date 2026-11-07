//! D/B nearby-background scenario kernels (DTP, DEB, DEBx2P, BTP, BEB, BEBx2P).
//! The population itself is task supplied and validated by `population_v3`.
use super::ladder::draw_bin_models;
use super::s_family::{
    batch_eb_transit_lnl, batch_eb_transit_lnl_prod, batch_transit_lnl, batch_transit_lnl_prod,
    lnz_from_logw, logweight_reduction, roche_ok, LogWeightReduction, ProdFold,
};
use super::{
    consts::{G_CGS, MSUN, PARITY_EXPTIME_D, PARITY_NSUB, REARTH, RSUN},
    math::sin,
    population_v3::{background_prior, sample_background_indices, BackgroundPopulation},
};
use crate::fpp::{
    error::FppError,
    npsum::{libm_pow, pairwise_sum},
    priors::{
        sample_ecc, sample_inc, sample_q, sample_q_companion, sample_q_from_uniforms, sample_rp,
        sample_rp_array, sample_w,
    },
    rng::FppRng,
    stellar::{flux_relation, get_ldc_arrays, ldc_background_arrays, stellar_relations_capped},
};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct DFamilyConfig {
    pub parity_index: bool,
    pub circular_prior: bool,
    pub target_q: bool,
    pub roche: bool,
    pub secondary_consistent: bool,
    pub k_nudge_upstream: bool,
    pub upstream_exposure: bool,
}
/// Per-evaluation diagnostics; caller creates one per candidate/convention.
#[derive(Clone, Debug, Default)]
pub struct DDiagnostics {
    pub kepler_unconverged_draws: u64,
    pub exposure_gl_fallback_share: f64,
    pub exposure_unconverged_share: f64,
    /// Bounded diagnostic sample retained for parity inspection.
    pub first_dtp_draws: Vec<DtpProbeDraw>,
    /// First eight background-host EB proposals and both likelihood branches.
    pub first_beb_draws: Vec<DBebProbeDraw>,
    /// Last background binary's NumPy-style lnZ reduction, primary/twin.
    /// This is retained for bit-level diagnosis and has no effect on results.
    pub last_binary_reductions: [Option<LogWeightReduction>; 2],
    /// Full last background-binary log-weight vectors, for exact reducer probes.
    pub last_binary_logw: [Vec<f64>; 2],
    /// Opt-in components for exact binary likelihood probes.
    pub last_binary_prior: Vec<f64>,
    pub last_binary_lnl: [Vec<f64>; 2],
    pub last_beb_draw_832: Option<DBebDetailedTrace>,
    /// Full log-weight capture is opt-in and used only by numerical probe tests.
    pub capture_logw: bool,
}
#[derive(Clone, Copy, Debug)]
pub struct DBebProbeDraw {
    pub scenario: usize,
    pub population_index: usize,
    pub mass_ratio: f64,
    pub inclination_deg: f64,
    pub eccentricity: f64,
    pub argument_deg: f64,
    pub included: bool,
    pub prior_ln: f64,
    pub ln_likelihood: f64,
}
#[derive(Clone, Debug, Default)]
pub struct DBebDetailedTrace {
    pub draw: usize,
    pub population_index: usize,
    pub mass_ratio: f64,
    pub inclination_deg: f64,
    pub eccentricity: f64,
    pub argument_deg: f64,
    pub k: f64,
    pub eb_fluxratio: f64,
    pub companion_fluxratio: f64,
    pub period: f64,
    pub inclination_rad: f64,
    pub a_rhost: f64,
    pub a_cm: f64,
    pub a_power_base: f64,
    pub rhost_cm: f64,
    pub w_rad: f64,
    pub u1: f64,
    pub u2: f64,
    pub u1_sec: f64,
    pub u2_sec: f64,
    pub dil_a: f64,
    pub dil_b: f64,
    pub ln_likelihood: f64,
    pub status: i64,
    pub n_aff: i64,
    pub model_indices: Vec<usize>,
    pub model_flux: Vec<f64>,
    pub n_eval: usize,
    pub n_fallback: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct DtpProbeDraw {
    pub population_index: usize,
    pub flux_ratio: f64,
    pub rp: f64,
    pub inclination_deg: f64,
    pub eccentricity: f64,
    pub argument_deg: f64,
    pub inclination_rad: f64,
    pub argument_rad: f64,
    pub included: bool,
    pub prior_ln: f64,
    pub ln_likelihood: f64,
    pub k: f64,
    pub a_rs: f64,
    pub u1: f64,
    pub u2: f64,
}
impl DDiagnostics {
    fn observe(&mut self, lnl: &[f64]) {
        self.kepler_unconverged_draws += lnl.iter().filter(|v| v.is_nan()).count() as u64;
    }

    fn observe_exposure(&mut self, lnl: &[f64], status: &[i64]) {
        let finite: Vec<usize> = (0..lnl.len()).filter(|&i| lnl[i].is_finite()).collect();
        if finite.is_empty() {
            return;
        }
        let max_lnl = finite
            .iter()
            .map(|&i| lnl[i])
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = finite.iter().map(|&i| (lnl[i] - max_lnl).exp()).collect();
        let total = pairwise_sum(&weights);
        if !(total > 0.0) {
            return;
        }
        let fallback = pairwise_sum(
            &finite
                .iter()
                .zip(&weights)
                .filter_map(|(&i, &w)| (status[i] == 1).then_some(w))
                .collect::<Vec<_>>(),
        ) / total;
        let unconverged = pairwise_sum(
            &finite
                .iter()
                .zip(&weights)
                .filter_map(|(&i, &w)| (status[i] == 2).then_some(w))
                .collect::<Vec<_>>(),
        ) / total;
        self.exposure_gl_fallback_share = self.exposure_gl_fallback_share.max(fallback);
        self.exposure_unconverged_share = self.exposure_unconverged_share.max(unconverged);
    }
}
impl Default for DFamilyConfig {
    fn default() -> Self {
        Self {
            parity_index: true,
            circular_prior: false,
            target_q: true,
            roche: false,
            secondary_consistent: false,
            k_nudge_upstream: true,
            upstream_exposure: true,
        }
    }
}
fn exposure(c: DFamilyConfig) -> (f64, usize) {
    if c.upstream_exposure {
        (PARITY_EXPTIME_D, PARITY_NSUB)
    } else {
        (0.0, 1)
    }
}
fn scalar_sigma(time: &[f64], sigma: f64) -> Vec<f64> {
    vec![sigma; time.len()]
}
fn f64_arrays(
    p: &BackgroundPopulation,
    f: impl Fn(&super::population_v3::BackgroundStar) -> f64,
) -> Vec<f64> {
    p.stars.iter().map(f).collect()
}
fn prior_vec(
    p: &BackgroundPopulation,
    fr: &[f64],
    eb: Option<&[f64]>,
    c: DFamilyConfig,
) -> Vec<f64> {
    (0..fr.len())
        .map(|i| background_prior(p.stars.len(), fr[i], eb.map(|x| x[i]), c.circular_prior))
        .collect()
}
fn lz(lnl: Vec<f64>, prior: &[f64]) -> f64 {
    super::gate::stage("main", &lnl);
    lnz_from_logw(
        &lnl.iter()
            .zip(prior)
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>(),
    )
}

/// DTP: a planet transiting the target, diluted by a sampled background star.
#[allow(clippy::too_many_arguments)]
pub fn lnz_dtp(
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    period: f64,
    target_mass: f64,
    target_radius: f64,
    n: usize,
    u1: f64,
    u2: f64,
    population: &BackgroundPopulation,
    rng: &mut FppRng,
    c: DFamilyConfig,
    diagnostics: &mut DDiagnostics,
) -> Result<f64, FppError> {
    lnz_dtp_impl(
        time,
        flux,
        sigma,
        period,
        target_mass,
        target_radius,
        n,
        u1,
        u2,
        population,
        rng,
        c,
        diagnostics,
        None,
    )
}

/// Production-fold DTP entry point. Proposal generation and priors are identical to parity;
/// only the likelihood kernel consumes the production bin ladder.
#[allow(clippy::too_many_arguments)]
pub fn lnz_dtp_with_fold(
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    period: f64,
    target_mass: f64,
    target_radius: f64,
    n: usize,
    u1: f64,
    u2: f64,
    population: &BackgroundPopulation,
    rng: &mut FppRng,
    c: DFamilyConfig,
    diagnostics: &mut DDiagnostics,
    fold: &ProdFold<'_>,
) -> Result<f64, FppError> {
    lnz_dtp_impl(
        time,
        flux,
        sigma,
        period,
        target_mass,
        target_radius,
        n,
        u1,
        u2,
        population,
        rng,
        c,
        diagnostics,
        Some(fold),
    )
}

#[allow(clippy::too_many_arguments)]
fn lnz_dtp_impl(
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    period: f64,
    target_mass: f64,
    target_radius: f64,
    n: usize,
    u1: f64,
    u2: f64,
    population: &BackgroundPopulation,
    rng: &mut FppRng,
    c: DFamilyConfig,
    diagnostics: &mut DDiagnostics,
    fold: Option<&ProdFold<'_>>,
) -> Result<f64, FppError> {
    let idx = sample_background_indices(n, population.stars.len(), c.parity_index, rng)
        ?;
    let rps = sample_rp(rng, n, target_mass);
    let inc = sample_inc(rng, n);
    let ecc = sample_ecc(rng, n, true, period);
    let arg = sample_w(rng, n);
    let p_sec = period * 86400.0;
    let a = libm_pow(
        ((G_CGS * target_mass * MSUN) / (4.0 * libm_pow(PI, 2.0))) * (p_sec * p_sec),
        1.0 / 3.0,
    );
    let mut ks = Vec::with_capacity(n);
    let mut ir = Vec::with_capacity(n);
    let mut wr = Vec::with_capacity(n);
    let mut mask = Vec::with_capacity(n);
    for j in 0..n {
        let e_corr = (1.0 + ecc[j] * sin(arg[j] * PI / 180.0)) / (1.0 - ecc[j] * ecc[j]);
        let ptra = (rps[j] * REARTH + target_radius * RSUN) / a * e_corr;
        let imin = if ptra <= 1.0 { ptra.acos() } else { PI / 2.0 };
        let coll = (rps[j] * REARTH + target_radius * RSUN) > a * (1.0 - ecc[j]);
        ir.push(inc[j] * PI / 180.0);
        wr.push((90.0 - arg[j]) * PI / 180.0);
        ks.push(rps[j] * REARTH / (target_radius * RSUN));
        mask.push(ir[j] >= imin && !coll);
    }
    if c.roche {
        let rok = roche_ok(
            &vec![a; n],
            &ecc,
            &vec![target_mass; n],
            &vec![target_radius; n],
            &vec![5.51; n],
        );
        for j in 0..n {
            mask[j] &= rok[j];
        }
    }
    let ix: Vec<usize> = (0..n).filter(|&j| mask[j]).collect();
    let mut lnl = vec![f64::NEG_INFINITY; n];
    if !ix.is_empty() {
        let sel = |v: &[f64]| ix.iter().map(|&j| v[j]).collect::<Vec<_>>();
        let fr: Vec<f64> = ix
            .iter()
            .map(|&j| population.stars[idx[j]].flux_ratio)
            .collect();
        let (ex, ns) = exposure(c);
        let sig = scalar_sigma(time, sigma);
        let p = vec![period; ix.len()];
        let aa = vec![a / (target_radius * RSUN); ix.len()];
        let val = if let Some(pf) = fold {
            let (lnl, status, _) = batch_transit_lnl_prod(
                pf,
                &sel(&ks),
                &p,
                &sel(&ir),
                &aa,
                &sel(&ecc),
                &sel(&wr),
                &vec![u1; ix.len()],
                &vec![u2; ix.len()],
                &fr,
                false,
            );
            diagnostics.observe_exposure(&lnl, &status);
            super::gate::note("main", &ix);
            lnl
        } else {
            batch_transit_lnl(
                time,
                flux,
                &sig,
                &sel(&ks),
                &p,
                &sel(&ir),
                &aa,
                &sel(&ecc),
                &sel(&wr),
                &vec![u1; ix.len()],
                &vec![u2; ix.len()],
                &fr,
                false,
                ex,
                ns,
            )
        };
        diagnostics.observe(&val);
        for (k, &j) in ix.iter().enumerate() {
            lnl[j] = val[k];
        }
    }
    let fr: Vec<f64> = (0..n)
        .map(|j| population.stars[idx[j]].flux_ratio)
        .collect();
    let prior = prior_vec(population, &fr, None, c);
    diagnostics.first_dtp_draws = (0..n.min(8))
        .map(|j| DtpProbeDraw {
            population_index: idx[j],
            flux_ratio: fr[j],
            rp: rps[j],
            inclination_deg: inc[j],
            eccentricity: ecc[j],
            argument_deg: arg[j],
            inclination_rad: ir[j],
            argument_rad: wr[j],
            included: mask[j],
            prior_ln: prior[j],
            ln_likelihood: lnl[j],
            k: ks[j],
            a_rs: a / (target_radius * RSUN),
            u1,
            u2,
        })
        .collect();
    if ix.is_empty() {
        // Python `_lnZ_DTP` is `_lnZ_TTP`: `if not np.any(mask): return -np.inf` -- no reduction, so no step-7
        // diagnostics record (the lnZ is -inf either way)
        return Ok(f64::NEG_INFINITY);
    }
    Ok(lz(lnl, &prior))
}

/// BTP: a planet transiting a sampled background host.
#[allow(clippy::too_many_arguments)]
pub fn lnz_btp(
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    period: f64,
    n: usize,
    p: &BackgroundPopulation,
    rng: &mut FppRng,
    c: DFamilyConfig,
    diagnostics: &mut DDiagnostics,
) -> Result<f64, FppError> {
    lnz_btp_impl(time, flux, sigma, period, n, p, rng, c, diagnostics, None)
}

/// Production-fold BTP entry point.
#[allow(clippy::too_many_arguments)]
pub fn lnz_btp_with_fold(
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    period: f64,
    n: usize,
    p: &BackgroundPopulation,
    rng: &mut FppRng,
    c: DFamilyConfig,
    diagnostics: &mut DDiagnostics,
    fold: &ProdFold<'_>,
) -> Result<f64, FppError> {
    lnz_btp_impl(
        time,
        flux,
        sigma,
        period,
        n,
        p,
        rng,
        c,
        diagnostics,
        Some(fold),
    )
}

#[allow(clippy::too_many_arguments)]
fn lnz_btp_impl(
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    period: f64,
    n: usize,
    p: &BackgroundPopulation,
    rng: &mut FppRng,
    c: DFamilyConfig,
    diagnostics: &mut DDiagnostics,
    fold: Option<&ProdFold<'_>>,
) -> Result<f64, FppError> {
    let idx = sample_background_indices(n, p.stars.len(), false, rng)?;
    let masses: Vec<f64> = idx.iter().map(|&i| p.stars[i].mass).collect();
    let radii: Vec<f64> = idx.iter().map(|&i| p.stars[i].radius).collect();
    let rps = sample_rp_array(rng, n, &masses);
    let inc = sample_inc(rng, n);
    let ecc = sample_ecc(rng, n, true, period);
    let arg = sample_w(rng, n);
    let teffs = f64_arrays(p, |s| s.teff);
    let logs = f64_arrays(p, |s| s.logg);
    let zs = f64_arrays(p, |s| s.metallicity);
    let (all_u1, all_u2) = ldc_background_arrays(&teffs, &logs, &zs)?;
    let mut ks = vec![0.0; n];
    let mut ir = vec![0.0; n];
    let mut wr = vec![0.0; n];
    let mut axes = vec![0.0; n];
    let mut mask = vec![false; n];
    for j in 0..n {
        let h = &p.stars[idx[j]];
        let p_sec = period * 86400.0;
        let coeff = (G_CGS * h.mass * MSUN) / (4.0 * libm_pow(PI, 2.0));
        let acm = libm_pow(coeff * (p_sec * p_sec), 1.0 / 3.0);
        axes[j] = acm / (radii[j] * RSUN);
        ks[j] = rps[j] * REARTH / (radii[j] * RSUN);
        ir[j] = inc[j] * PI / 180.0;
        wr[j] = (90.0 - arg[j]) * PI / 180.0;
        let ec = (1.0 + ecc[j] * sin(arg[j] * PI / 180.0)) / (1.0 - ecc[j] * ecc[j]);
        let pt = (rps[j] * REARTH + radii[j] * RSUN) / acm * ec;
        let imin = if pt <= 1.0 { pt.acos() } else { PI / 2.0 };
        let col = (rps[j] * REARTH + radii[j] * RSUN) > acm * (1.0 - ecc[j]);
        mask[j] = ir[j] >= imin && !col && h.logg >= 3.5 && h.teff <= 10000.0;
        if c.roche {
            mask[j] &= roche_ok(&[acm], &[ecc[j]], &[h.mass], &[radii[j]], &[5.51])[0];
        }
    }
    let ix: Vec<usize> = (0..n).filter(|&j| mask[j]).collect();
    let mut lnl = vec![f64::NEG_INFINITY; n];
    if !ix.is_empty() {
        let sel = |v: &[f64]| ix.iter().map(|&j| v[j]).collect::<Vec<_>>();
        let fr: Vec<f64> = ix.iter().map(|&j| p.stars[idx[j]].flux_ratio).collect();
        let u1: Vec<f64> = ix.iter().map(|&j| all_u1[idx[j]]).collect();
        let u2: Vec<f64> = ix.iter().map(|&j| all_u2[idx[j]]).collect();
        let (ex, ns) = exposure(c);
        let sig = scalar_sigma(time, sigma);
        let val = if let Some(pf) = fold {
            let (lnl, status, _) = batch_transit_lnl_prod(
                pf,
                &sel(&ks),
                &vec![period; ix.len()],
                &sel(&ir),
                &sel(&axes),
                &sel(&ecc),
                &sel(&wr),
                &u1,
                &u2,
                &fr,
                true,
            );
            diagnostics.observe_exposure(&lnl, &status);
            super::gate::note("main", &ix);
            lnl
        } else {
            batch_transit_lnl(
                time,
                flux,
                &sig,
                &sel(&ks),
                &vec![period; ix.len()],
                &sel(&ir),
                &sel(&axes),
                &sel(&ecc),
                &sel(&wr),
                &u1,
                &u2,
                &fr,
                true,
                ex,
                ns,
            )
        };
        diagnostics.observe(&val);
        for (k, &j) in ix.iter().enumerate() {
            lnl[j] = val[k];
        }
    }
    let fr: Vec<f64> = (0..n).map(|j| p.stars[idx[j]].flux_ratio).collect();
    Ok(lz(lnl, &prior_vec(p, &fr, None, c)))
}

// Remaining binary wrappers share `_binary_background_lnz` exactly; the draw
// order is kept in each public entry point because it is part of the contract.
#[allow(clippy::too_many_arguments)]
fn binary_lnz(
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    sigma_veto: f64,
    period: f64,
    target_mass: f64,
    target_radius: f64,
    target_teff: f64,
    target_u1: f64,
    target_u2: f64,
    n: usize,
    p: &BackgroundPopulation,
    idx: &[usize],
    inc: &[f64],
    q: &[f64],
    ecc: &[f64],
    arg: &[f64],
    background_host: bool,
    c: DFamilyConfig,
    diagnostics: &mut DDiagnostics,
    fold: Option<&ProdFold<'_>>,
) -> Result<(f64, f64), FppError> {
    let mhost: Vec<f64> = (0..n)
        .map(|j| {
            if background_host {
                p.stars[idx[j]].mass
            } else {
                target_mass
            }
        })
        .collect();
    let rhost: Vec<f64> = (0..n)
        .map(|j| {
            if background_host {
                p.stars[idx[j]].radius
            } else {
                target_radius
            }
        })
        .collect();
    let thost: Vec<f64> = (0..n)
        .map(|j| {
            if background_host {
                p.stars[idx[j]].teff
            } else {
                target_teff
            }
        })
        .collect();
    let frc: Vec<f64> = (0..n).map(|j| p.stars[idx[j]].flux_ratio).collect();
    let meb: Vec<f64> = (0..n).map(|j| q[j] * mhost[j]).collect();
    let (reb, teb) = stellar_relations_capped(&meb, &rhost, &thost);
    let fe = flux_relation(&meb);
    let fm = flux_relation(&vec![target_mass]);
    let mut freb: Vec<f64> = (0..n).map(|j| fe[j] / (fe[j] + fm[0])).collect();
    if background_host {
        let fh = flux_relation(&mhost);
        for j in 0..n {
            let fr_bound = fh[j] / (fh[j] + fm[0]);
            freb[j] *= frc[j] / fr_bound;
        }
    }
    let prior = prior_vec(p, &frc, if background_host { Some(&freb) } else { None }, c);
    let mut ir = vec![0.0; n];
    let mut wr = vec![0.0; n];
    let mut ec = vec![0.0; n];
    let mut ac = vec![0.0; n];
    let mut at = vec![0.0; n];
    let mut mask = vec![false; n];
    let mut twin = vec![false; n];
    let mut ks = vec![0.0; n];
    for j in 0..n {
        ir[j] = inc[j] * PI / 180.0;
        wr[j] = (90.0 - arg[j]) * PI / 180.0;
        ec[j] = (1.0 + ecc[j] * sin(arg[j] * PI / 180.0)) / (1.0 - ecc[j] * ecc[j]);
        let psec = period * 86400.0;
        let coeff = (G_CGS * (mhost[j] + meb[j]) * MSUN) / (4.0 * libm_pow(PI, 2.0));
        ac[j] = libm_pow(coeff * (psec * psec), 1.0 / 3.0);
        let pt = (reb[j] * RSUN + rhost[j] * RSUN) / ac[j] * ec[j];
        let imin = if pt <= 1.0 { pt.acos() } else { PI / 2.0 };
        let twosec = 2.0 * psec;
        at[j] = libm_pow(coeff * (twosec * twosec), 1.0 / 3.0);
        let ptw = (reb[j] * RSUN + rhost[j] * RSUN) / at[j] * ec[j];
        let imint = if ptw <= 1.0 { ptw.acos() } else { PI / 2.0 };
        let collision = (reb[j] * RSUN + rhost[j] * RSUN) > ac[j] * (1.0 - ecc[j]);
        let collision_tw = (2.0 * rhost[j] * RSUN) > at[j] * (1.0 - ecc[j]);
        mask[j] = ir[j] >= imin && !collision && q[j] < 0.95;
        twin[j] = ir[j] >= imint && !collision_tw && q[j] >= 0.95;
        if background_host {
            let ok = p.stars[idx[j]].logg >= 3.5 && thost[j] <= 10000.0;
            mask[j] &= ok;
            twin[j] &= ok;
        }
        if c.roche {
            let rho = meb[j] * MSUN
                / ((4.0 / 3.0) * PI * (reb[j] * RSUN) * (reb[j] * RSUN) * (reb[j] * RSUN));
            mask[j] &= roche_ok(&[ac[j]], &[ecc[j]], &[mhost[j]], &[rhost[j]], &[rho])[0];
            twin[j] &= roche_ok(&[at[j]], &[ecc[j]], &[mhost[j]], &[rhost[j]], &[rho])[0];
        }
        ks[j] = reb[j] / rhost[j];
    }
    let (all_u1, all_u2) = if background_host {
        let t = f64_arrays(p, |s| s.teff);
        let g = f64_arrays(p, |s| s.logg);
        let z = f64_arrays(p, |s| s.metallicity);
        ldc_background_arrays(&t, &g, &z)?
    } else {
        (vec![target_u1], vec![target_u2])
    };
    let u1 = (0..n)
        .map(|j| {
            if background_host {
                all_u1[idx[j]]
            } else {
                target_u1
            }
        })
        .collect::<Vec<_>>();
    let u2 = (0..n)
        .map(|j| {
            if background_host {
                all_u2[idx[j]]
            } else {
                target_u2
            }
        })
        .collect::<Vec<_>>();
    let mut u1s = u1.clone();
    let mut u2s = u2.clone();
    if c.secondary_consistent {
        let selected: Vec<usize> = (0..n).filter(|&j| mask[j] || twin[j]).collect();
        if !selected.is_empty() {
            let tt: Vec<f64> = selected.iter().map(|&j| teb[j]).collect();
            let gg: Vec<f64> = selected
                .iter()
                .map(|&j| (G_CGS * meb[j] * MSUN / ((reb[j] * RSUN) * (reb[j] * RSUN))).log10())
                .collect();
            let (a, b) = if background_host {
                let z: Vec<f64> = selected
                    .iter()
                    .map(|&j| p.stars[idx[j]].metallicity)
                    .collect();
                ldc_background_arrays(&tt, &gg, &z)?
            } else {
                get_ldc_arrays(&tt, &gg)?
            };
            for (k, &j) in selected.iter().enumerate() {
                u1s[j] = a[k];
                u2s[j] = b[k];
            }
        }
    }
    let sig = scalar_sigma(time, sigma);
    let (ex, ns) = exposure(c);
    let mut out = [vec![f64::NEG_INFINITY; n], vec![f64::NEG_INFINITY; n]];
    for scenario in 0..2 {
        let keep = if scenario == 0 { &mask } else { &twin };
        let ix: Vec<usize> = (0..n).filter(|&j| keep[j]).collect();
        if ix.is_empty() {
            continue;
        }
        let sel = |v: &[f64]| ix.iter().map(|&j| v[j]).collect::<Vec<_>>();
        let p_orb: Vec<f64> = ix
            .iter()
            .map(|_| period * if scenario == 0 { 1.0 } else { 2.0 })
            .collect();
        let axes: Vec<f64> = ix
            .iter()
            .map(|&j| (if scenario == 0 { ac[j] } else { at[j] }) / (rhost[j] * RSUN))
            .collect();
        let val = if let Some(pf) = fold {
            let (lnl, status, n_aff) = batch_eb_transit_lnl_prod(
                pf,
                sigma_veto,
                &sel(&ks),
                &sel(&freb),
                &p_orb,
                &sel(&ir),
                &axes,
                &sel(&ecc),
                &sel(&wr),
                &sel(&u1),
                &sel(&u2),
                &sel(&frc),
                background_host,
                scenario == 0,
                c.secondary_consistent,
                c.k_nudge_upstream,
                &sel(&u1s),
                &sel(&u2s),
            );
            diagnostics.observe_exposure(&lnl, &status);
            super::gate::note(if scenario == 0 { "s0" } else { "s1" }, &ix);
            if background_host && diagnostics.capture_logw && scenario == 1 {
                if let Some(pos) = ix.iter().position(|&draw| draw == 832) {
                    let draw = 832;
                    let k_raw = ks[draw];
                    let mut k = k_raw;
                    if c.k_nudge_upstream && (k - 1.0).abs() < 1e-6 {
                        k *= 0.999;
                    }
                    let fr_eb = freb[draw];
                    let f_eb = if fr_eb < 1.0 {
                        fr_eb / (1.0 - fr_eb)
                    } else {
                        1e6
                    };
                    let fc = frc[draw];
                    let f_comp = if fc > 0.0 { fc / (1.0 - fc) } else { 0.0 };
                    let x = f_eb / f_comp.max(1e-30);
                    let f_dilute = 1.0 / (f_comp + f_eb);
                    let a1 = 1.0 / (1.0 + x);
                    let b1 = x / (1.0 + x);
                    let a2 = 1.0 / (1.0 + f_dilute);
                    let b2 = f_dilute / (1.0 + f_dilute);
                    let dil_a = a2 * a1;
                    let dil_b = a2 * b1 + b2;
                    let models = draw_bin_models(
                        pf.w, pf.delta, pf.fold, p_orb[pos], axes[pos], ir[draw], ecc[draw],
                        wr[draw], k, u1[draw], u2[draw], dil_a, dil_b,
                    );
                    let psec = period * 86400.0;
                    let twosec = 2.0 * psec;
                    let coeff =
                        (G_CGS * (mhost[draw] + meb[draw]) * MSUN) / (4.0 * libm_pow(PI, 2.0));
                    let a_power_base = coeff * (twosec * twosec);
                    diagnostics.last_beb_draw_832 = Some(DBebDetailedTrace {
                        draw,
                        population_index: idx[draw],
                        mass_ratio: q[draw],
                        inclination_deg: inc[draw],
                        eccentricity: ecc[draw],
                        argument_deg: arg[draw],
                        k,
                        eb_fluxratio: fr_eb,
                        companion_fluxratio: fc,
                        period: p_orb[pos],
                        inclination_rad: ir[draw],
                        a_rhost: axes[pos],
                        a_cm: at[draw],
                        a_power_base,
                        rhost_cm: rhost[draw] * RSUN,
                        w_rad: wr[draw],
                        u1: u1[draw],
                        u2: u2[draw],
                        u1_sec: u1s[draw],
                        u2_sec: u2s[draw],
                        dil_a,
                        dil_b,
                        ln_likelihood: lnl[pos],
                        status: status[pos],
                        n_aff: n_aff[pos],
                        model_indices: models.idx,
                        model_flux: models.m,
                        n_eval: models.n_eval,
                        n_fallback: models.n_fb,
                    });
                }
            }
            lnl
        } else {
            batch_eb_transit_lnl(
                time,
                flux,
                &sig,
                sigma_veto,
                &sel(&ks),
                &sel(&freb),
                &p_orb,
                &sel(&ir),
                &axes,
                &sel(&ecc),
                &sel(&wr),
                &sel(&u1),
                &sel(&u2),
                &sel(&frc),
                background_host,
                ex,
                ns,
                scenario == 0,
                c.secondary_consistent,
                c.k_nudge_upstream,
                &sel(&u1s),
                &sel(&u2s),
            )
        };
        diagnostics.observe(&val);
        for (k, &j) in ix.iter().enumerate() {
            out[scenario][j] = val[k];
        }
        if background_host {
            let mut retained = 0;
            for j in 0..n {
                if keep[j] && retained < 8 {
                    diagnostics.first_beb_draws.push(DBebProbeDraw {
                        scenario,
                        population_index: idx[j],
                        mass_ratio: q[j],
                        inclination_deg: inc[j],
                        eccentricity: ecc[j],
                        argument_deg: arg[j],
                        included: true,
                        prior_ln: prior[j],
                        ln_likelihood: out[scenario][j],
                    });
                    retained += 1;
                }
            }
        }
    }
    let mut reductions = [None, None];
    let mut logweights = [Vec::new(), Vec::new()];
    for scenario in 0..2 {
        let logw: Vec<f64> = out[scenario]
            .iter()
            .zip(&prior)
            .map(|(&likelihood, &prior)| likelihood + prior)
            .collect();
        super::gate::stage(if scenario == 0 { "s0" } else { "s1" }, &out[scenario]);
        reductions[scenario] = logweight_reduction(&logw);
        logweights[scenario] = logw;
    }
    if background_host {
        diagnostics.last_binary_reductions = reductions;
        if diagnostics.capture_logw {
            diagnostics.last_binary_logw = logweights;
            diagnostics.last_binary_prior = prior;
            diagnostics.last_binary_lnl = out;
        }
    }
    Ok((
        reductions[0].map_or(f64::NEG_INFINITY, |r| r.lnz),
        reductions[1].map_or(f64::NEG_INFINITY, |r| r.lnz),
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn lnz_deb(
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    sigma_veto: f64,
    period: f64,
    mass: f64,
    radius: f64,
    teff: f64,
    u1: f64,
    u2: f64,
    n: usize,
    p: &BackgroundPopulation,
    rng: &mut FppRng,
    c: DFamilyConfig,
    diagnostics: &mut DDiagnostics,
) -> Result<(f64, f64), FppError> {
    lnz_deb_impl(
        time,
        flux,
        sigma,
        sigma_veto,
        period,
        mass,
        radius,
        teff,
        u1,
        u2,
        n,
        p,
        rng,
        c,
        diagnostics,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn lnz_deb_with_fold(
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    sigma_veto: f64,
    period: f64,
    mass: f64,
    radius: f64,
    teff: f64,
    u1: f64,
    u2: f64,
    n: usize,
    p: &BackgroundPopulation,
    rng: &mut FppRng,
    c: DFamilyConfig,
    diagnostics: &mut DDiagnostics,
    fold: &ProdFold<'_>,
) -> Result<(f64, f64), FppError> {
    lnz_deb_impl(
        time,
        flux,
        sigma,
        sigma_veto,
        period,
        mass,
        radius,
        teff,
        u1,
        u2,
        n,
        p,
        rng,
        c,
        diagnostics,
        Some(fold),
    )
}

#[allow(clippy::too_many_arguments)]
fn lnz_deb_impl(
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    sigma_veto: f64,
    period: f64,
    mass: f64,
    radius: f64,
    teff: f64,
    u1: f64,
    u2: f64,
    n: usize,
    p: &BackgroundPopulation,
    rng: &mut FppRng,
    c: DFamilyConfig,
    diagnostics: &mut DDiagnostics,
    fold: Option<&ProdFold<'_>>,
) -> Result<(f64, f64), FppError> {
    let inc = sample_inc(rng, n);
    let q = sample_q(rng, n, mass);
    let ecc = sample_ecc(rng, n, false, period);
    let arg = sample_w(rng, n);
    let idx =
        sample_background_indices(n, p.stars.len(), c.parity_index, rng)?;
    binary_lnz(
        time,
        flux,
        sigma,
        sigma_veto,
        period,
        mass,
        radius,
        teff,
        u1,
        u2,
        n,
        p,
        &idx,
        &inc,
        &q,
        &ecc,
        &arg,
        false,
        c,
        diagnostics,
        fold,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn lnz_beb(
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    sigma_veto: f64,
    period: f64,
    mass: f64,
    radius: f64,
    teff: f64,
    u1: f64,
    u2: f64,
    n: usize,
    p: &BackgroundPopulation,
    rng: &mut FppRng,
    c: DFamilyConfig,
    diagnostics: &mut DDiagnostics,
) -> Result<(f64, f64), FppError> {
    lnz_beb_impl(
        time,
        flux,
        sigma,
        sigma_veto,
        period,
        mass,
        radius,
        teff,
        u1,
        u2,
        n,
        p,
        rng,
        c,
        diagnostics,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn lnz_beb_with_fold(
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    sigma_veto: f64,
    period: f64,
    mass: f64,
    radius: f64,
    teff: f64,
    u1: f64,
    u2: f64,
    n: usize,
    p: &BackgroundPopulation,
    rng: &mut FppRng,
    c: DFamilyConfig,
    diagnostics: &mut DDiagnostics,
    fold: &ProdFold<'_>,
) -> Result<(f64, f64), FppError> {
    lnz_beb_impl(
        time,
        flux,
        sigma,
        sigma_veto,
        period,
        mass,
        radius,
        teff,
        u1,
        u2,
        n,
        p,
        rng,
        c,
        diagnostics,
        Some(fold),
    )
}

#[allow(clippy::too_many_arguments)]
fn lnz_beb_impl(
    time: &[f64],
    flux: &[f64],
    sigma: f64,
    sigma_veto: f64,
    period: f64,
    mass: f64,
    radius: f64,
    teff: f64,
    u1: f64,
    u2: f64,
    n: usize,
    p: &BackgroundPopulation,
    rng: &mut FppRng,
    c: DFamilyConfig,
    diagnostics: &mut DDiagnostics,
    fold: Option<&ProdFold<'_>>,
) -> Result<(f64, f64), FppError> {
    let inc = sample_inc(rng, n);
    let mut q = if c.target_q {
        sample_q(rng, n, mass)
    } else {
        vec![0.0; n]
    };
    let q_uniform = if c.target_q {
        None
    } else {
        Some((0..n).map(|_| rng.rand()).collect::<Vec<_>>())
    };
    let _unused = sample_q_companion(rng, n, mass);
    let ecc = sample_ecc(rng, n, false, period);
    let arg = sample_w(rng, n);
    let idx = sample_background_indices(n, p.stars.len(), false, rng)?;
    if let Some(u) = q_uniform {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| p.stars[idx[a]].mass.total_cmp(&p.stars[idx[b]].mass));
        let mut at = 0;
        while at < n {
            let host = p.stars[idx[order[at]]].mass;
            let mut end = at + 1;
            while end < n && p.stars[idx[order[end]]].mass == host {
                end += 1;
            }
            let group: Vec<f64> = order[at..end].iter().map(|&j| u[j]).collect();
            let transformed = sample_q_from_uniforms(group, host);
            for (offset, k) in (at..end).enumerate() {
                q[order[k]] = transformed[offset];
            }
            at = end;
        }
    }
    binary_lnz(
        time,
        flux,
        sigma,
        sigma_veto,
        period,
        mass,
        radius,
        teff,
        u1,
        u2,
        n,
        p,
        &idx,
        &inc,
        &q,
        &ecc,
        &arg,
        true,
        c,
        diagnostics,
        fold,
    )
}
