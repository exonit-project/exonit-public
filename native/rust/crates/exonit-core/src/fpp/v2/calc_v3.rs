//! Corrected source-aware FPP orchestration for the native worker.
//!
//! This is the worker-facing layer above the literal K/D/G kernels: it owns
//! per-convention RNG reset, raw-light-curve preprocessing, geometry and
//! population validation, all scenario dispatch, probability reduction,
//! the step-7 MC-noise stage loop (decision C) and verdict caps. It
//! deliberately does not load catalogues or expected values.
//!
//! Validation order is Python's (`calc_fpp_native`, r2 `:5376-5426`): nearby
//! stars, geometry (`_flux_ratios_for_mode`), population, cross-check type,
//! alarm, target stellar parameters, target limb darkening.

use super::{
    d5::Star as ApertureStar,
    d_family::{self, DDiagnostics, DFamilyConfig},
    dispatch::{self, VerdictCaps},
    fold::{build_prod_fold_consts, fold_production_default, prod_fold_arrays, prod_fold_renorm},
    geometry_v3::{compute_flux_ratios_smooth, GeometryV3, SmoothStar},
    mc::{self, DiagSink, DrawDiag, StagePass, StageRecord},
    population_v3::{validate_population, BackgroundPopulation},
    s_family::{lnz_seb_with_details, lnz_stp_with_details, LogWeightReduction, ProdFold},
    sigma::sigma_components,
    target_family::{self, TargetDiagnostics, TargetFamilyConfig},
};
use crate::fpp::error::{FppError, FppSkipReason, FppUnavailableReason};
use crate::fpp::{
    fold::{phase_fold_bin, renorm_flux},
    inputs::{parse_nearby, python_truthy, StarRec},
    npsum::libm_pow,
    rng::FppRng,
    v2::consts::{G_CGS, MSUN, RSUN},
    stellar,
};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashMap};

/// All corrected inputs are task/API supplied, except for the private seeded
/// stream and the already-loaded light curve.
pub struct CalcV3Inputs<'a> {
    pub tic_id: &'a str,
    pub period: f64,
    pub t0: f64,
    pub duration: f64,
    pub depth_ppm: f64,
    pub time: &'a [f64],
    pub flux: &'a [f64],
    pub nearby_raw: &'a Value,
    pub geometry: Option<&'a Value>,
    pub population: Option<&'a Value>,
    pub population_sha256: Option<&'a str>,
    pub population_version: Option<&'a str>,
    pub population_crosscheck: Option<&'a Value>,
    pub cadence_s: Option<f64>,
    pub lc_source: Option<&'a str>,
    pub tglc_near_edge: Option<bool>,
    pub seed_ephemeris: Option<(f64, f64)>,
}

/// Internal step-7 parameters of `calc_fpp_native` (r2 `:5232-5238`); the worker passes none of them.
/// `adaptive_n = None` follows the switch set (`FPP_ADAPTIVE_N`: on in production, off in parity);
/// `n_cap = None` is `MC_N_CAP`; `futility_stop = None` is on.  `futility_stop = Some(false)` is Python's
/// measurement switch (runs the whole ladder).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct McOptions {
    pub adaptive_n: Option<bool>,
    pub n_cap: Option<usize>,
    pub futility_stop: Option<bool>,
    pub unstable_flag: Option<bool>,
}

#[derive(Clone, Debug)]
pub struct ScenarioResult {
    pub lnz: f64,
    pub probability: f64,
}

#[derive(Clone, Debug)]
pub struct DilutionAlternative {
    pub fpp: f64,
    pub nfpp: f64,
    pub verdict: String,
    pub scenarios: BTreeMap<String, ScenarioResult>,
}

#[derive(Clone, Debug)]
pub struct CalcV3Result {
    pub fpp: f64,
    pub nfpp: f64,
    pub verdict: String,
    pub scenarios: BTreeMap<String, ScenarioResult>,
    pub flags: Vec<String>,
    pub dilution_corrected: bool,
    pub dilution_unverified: bool,
    pub depth_scale: f64,
    pub population: BackgroundPopulation,
    pub population_crosscheck_status: String,
    pub kepler_unconverged_draws: u64,
    pub geometry: super::geometry_v3::GeometryV3Info,
    pub sigma_source: String,
    pub sigma_fit_ill_conditioned: bool,
    pub exposure_unconverged: bool,
    pub exposure_unconverged_share: f64,
    pub exposure_gl_fallback_share: f64,
    pub production_switch_set: bool,
    pub cadence_s: Option<f64>,
    pub high_crowding: bool,
    pub gal_b: f64,
    pub n_nearby: usize,
    pub seed: u32,
    pub dilution_alternatives: BTreeMap<bool, DilutionAlternative>,
    // ---- step 7 (decision C) MC-noise fields of the reported (last) stage, r2 `:5698-5710`
    pub fpp_mc_se: f64,
    pub nfpp_mc_se: f64,
    pub n_draws_final: usize,
    pub n_draws_total: usize,
    /// `calc_fpp_by_source`: draws spent by every dilution run (`:5829`).
    pub n_draws_all_runs: usize,
    pub refinement_stages: Vec<StageRecord>,
    /// Per scenario of the reported stage, in Python's scenario order: [ESS, relative SE of Z, tail scale beta].
    pub mc_scen: Vec<(String, [f64; 3])>,
    pub mc_min_ess: f64,
    pub mc_min_ess_scenario: String,
    pub mc_policy: bool,
    pub borderline: bool,
    pub borderline_reasons: Vec<String>,
    pub d_last_binary_reductions: [Option<LogWeightReduction>; 2],
    pub d_last_binary_logw: [Vec<f64>; 2],
    pub d_last_binary_prior: Vec<f64>,
    pub d_last_binary_lnl: [Vec<f64>; 2],
    pub d_last_beb_draw_832: Option<d_family::DBebDetailedTrace>,
    /// v3 gate observer: this run (and, on the selected result of `calc_fpp_by_source_v3`, every dilution run in
    /// the order Python runs them). Empty unless a `gate::GateRecorder` is active.
    pub gate_runs: Vec<GateRun>,
    // ---- honesty flag (FPP_UNSTABLE_FLAG, Task 44)
    pub unstable_on: bool,
    pub fpp_unstable: bool,
    pub mc_ess_dom: Option<f64>,
    pub mc_dominant_scenario: String,
    pub fpp_interval: [f64; 2],
}

#[derive(Clone)]
struct PreparedStar {
    rec: StarRec,
    smooth: SmoothStar,
}

/// Diagnostics of one Monte-Carlo pass (the reported stage's are the result's).
struct PassDiagnostics {
    target: TargetDiagnostics,
    d: DDiagnostics,
    /// v3 gate observer: per-scenario records of this pass (empty unless a `gate::GateRecorder` is active).
    gate: HashMap<String, Option<super::gate::GateRecord>>,
}

/// One dilution run of the v3 gate dump (Python `calc_fpp_native` call), scenarios in Python's order.
#[derive(Clone, Debug, Default)]
pub struct GateRun {
    pub dilution_corrected: bool,
    pub fpp: f64,
    pub nfpp: f64,
    /// (name, lnZ, probability, record): `None` = the scenario never reached its likelihood reduction
    pub scenarios: Vec<(String, f64, f64, Option<super::gate::GateRecord>)>,
}

fn finite_lc(time: &[f64], flux: &[f64]) -> Result<(Vec<f64>, Vec<f64>), FppError> {
    if time.len() != flux.len() {
        // Python: np.isfinite(time) & np.isfinite(flux) raises on a shape mismatch -> generic failure
        return Err(FppError::failed("time and flux lengths differ"));
    }
    let mut t = Vec::new();
    let mut f = Vec::new();
    for (&ti, &fi) in time.iter().zip(flux) {
        if ti.is_finite() && fi.is_finite() {
            t.push(ti);
            f.push(fi);
        }
    }
    if t.len() < 100 {
        return Err(FppError::skip(
            FppSkipReason::RawLcUnusable,
            "raw light curve unusable: fewer than 100 finite cadences",
        ));
    }
    Ok((t, f))
}

fn target_limb_darkening(star: &StarRec) -> Result<(f64, f64), FppError> {
    let logg = (G_CGS * star.mass * MSUN / (libm_pow(star.rad * RSUN, 2.0))).log10();
    stellar::get_ldc_checked(star.teff, logg, 0.0)
}

fn inferred_cadence_s(time: &[f64]) -> Result<f64, FppError> {
    if time.len() < 2 {
        return Err(FppError::failed("production cadence cannot be inferred from fewer than two samples"));
    }
    let mut sorted = time.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut diffs: Vec<f64> = sorted.windows(2).map(|w| w[1] - w[0]).collect();
    diffs.sort_by(f64::total_cmp);
    let mid = diffs.len() / 2;
    let median = if diffs.len() % 2 == 1 {
        diffs[mid]
    } else {
        (diffs[mid - 1] + diffs[mid]) / 2.0
    };
    let cadence = ((median * 86400.0) * 1000.0).round_ties_even() / 1000.0;
    if !(cadence > 0.0 && cadence.is_finite()) {
        return Err(FppError::failed("production cadence is not positive and finite"));
    }
    Ok(cadence)
}

/// Apply every conservative validation gate after the likelihood values and
/// diagnostics have been produced. Keeping this in the production calc path
/// makes the verdict-caps tests exercise the same finalizer as real tasks.
fn finalize_verdict(
    mut verdict: String,
    flags: &mut Vec<String>,
    caps: VerdictCaps,
    fit_ill_conditioned: bool,
    exposure_unconverged: bool,
    tglc_near_edge: Option<bool>,
    is_tglc: bool,
) -> String {
    dispatch::cap_verdict(&mut verdict, caps, flags);
    if fit_ill_conditioned && verdict == "validated" {
        verdict = "candidate".into();
    }
    if exposure_unconverged && verdict == "validated" {
        verdict = "candidate".into();
    }
    if is_tglc && tglc_near_edge == Some(true) {
        flags.push("tglc_near_edge".into());
        if verdict == "validated" {
            verdict = "candidate".into();
        }
    }
    verdict
}

/// Step-7 verdict rule (r2 `:5642-5646`): with the MC policy on, `validated` needs a resolved stage AND
/// FPP + 3 SE < 0.015 and NFPP + 3 SE < 0.001; a `borderline` result is never `validated`.
fn apply_mc_verdict_rule(
    verdict: &mut String,
    mc_policy: bool,
    borderline: bool,
    fpp: f64,
    fpp_se: f64,
    nfpp: f64,
    nfpp_se: f64,
) {
    if mc_policy
        && verdict == "validated"
        && (borderline || !mc::validated_3sigma(fpp, fpp_se, nfpp, nfpp_se))
    {
        *verdict = "candidate".into();
    }
}

/// The run-level inputs of [`run_level_verdict`].
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct RunLevelGates {
    pub mc_policy: bool,
    pub borderline: bool,
    pub fpp: f64,
    pub fpp_se: f64,
    pub nfpp: f64,
    pub nfpp_se: f64,
    pub crosscheck_ok: bool,
    pub mass_from_relation: bool,
    pub giant_host: bool,
    pub unstable_on: bool,
    pub unstable: bool,
}

/// Python `calc_fpp_native` after the stage loop (r2 `:5640-5693`): the step-7 rule demotes `validated` when the
/// result is `borderline` or fails `mc_validated_3sigma`; the honesty cap (FPP_UNSTABLE_FLAG) demotes `likely_fp` and
/// `validated` to `candidate` when unstable; the giant cap demotes `validated` independently; the run-level flags are
/// `[population_crosscheck_unverified, background_q_target_quirk, mass_from_relation, giant_host, mc_borderline,
/// fpp_unstable]`. The other caps follow in the caller.
#[doc(hidden)]
pub fn run_level_verdict(mut verdict: String, g: &RunLevelGates) -> (String, Vec<String>) {
    apply_mc_verdict_rule(&mut verdict, g.mc_policy, g.borderline, g.fpp, g.fpp_se, g.nfpp, g.nfpp_se);
    if g.unstable_on && g.unstable && (verdict == "likely_fp" || verdict == "validated") {
        verdict = "candidate".into();
    }
    if g.giant_host && verdict == "validated" {
        verdict = "candidate".into();
    }
    let mut flags = Vec::new();
    if !g.crosscheck_ok {
        flags.push("population_crosscheck_unverified".to_string());
    }
    flags.push("background_q_target_quirk".to_string());
    if g.mass_from_relation {
        flags.push("mass_from_relation".to_string());
    }
    if g.giant_host {
        flags.push("giant_host".to_string());
    }
    if g.borderline {
        flags.push("mc_borderline".to_string());
    }
    if g.unstable_on && g.unstable {
        flags.push("fpp_unstable".to_string());
    }
    (verdict, flags)
}

/// Stellar-mass fallback flags of the target (r2 `:5675`, `:5689-5693`), in Python's order, and the giant cap.
/// `giant_host = target.giant_host or (R finite and R > 2.5)`; `mass_from_relation` alone never caps.
/// Python `population_provenance_missing` (7b30ba1): true unless BOTH the sha256 and the version are non-empty strings.
pub fn population_provenance_missing(population_sha256: Option<&str>, population_version: Option<&str>) -> bool {
    !(population_sha256.is_some_and(|s| !s.is_empty()) && population_version.is_some_and(|v| !v.is_empty()))
}

fn mass_fallback_flags(target: &StarRec) -> (bool, bool) {
    let giant_host = target.giant_host || (target.rad.is_finite() && target.rad > 2.5);
    let from_relation = target.mass_source.as_deref() == Some("teff_relation_tic8fit");
    (from_relation, giant_host)
}

/// Python `calc_fpp_native` when M, R or Teff is NaN (r2 `:5409-5422`): the server's hint wins when it is exactly
/// `giant_host` / `stellar_params_missing`; otherwise R finite and R > 2.5 means a giant host.
fn missing_stellar_params_reason(target: &StarRec) -> FppSkipReason {
    match target.mass_unavailable_reason.as_deref() {
        Some("giant_host") => FppSkipReason::GiantHost,
        Some("stellar_params_missing") => FppSkipReason::StellarParamsMissing,
        _ => {
            if target.rad.is_finite() && target.rad > 2.5 {
                FppSkipReason::GiantHost
            } else {
                FppSkipReason::StellarParamsMissing
            }
        }
    }
}

fn geometry_fallback(
    stars: &[ApertureStar],
    depth_ppm: f64,
    geometry_doc: &Value,
    dilution_corrected: bool,
    _mode: &str,
) -> Result<(Vec<SmoothStar>, super::geometry_v3::GeometryV3Info), String> {
    let geometry = if geometry_doc.is_null() {
        None
    } else {
        Some(
            serde_json::from_value::<super::d5::Geometry>(geometry_doc.clone())
                .map_err(|e| format!("geometry_invalid: {e}"))?,
        )
    };
    let (flux_stars, info) =
        super::d5::compute_flux_ratios_d5(stars, depth_ppm, geometry.as_ref(), dilution_corrected)?;
    let output = flux_stars
        .into_iter()
        .map(|s| SmoothStar {
            star: s.star,
            fluxratio: s.fluxratio,
            tdepth_raw: s.tdepth,
            tdepth: s.tdepth,
            viability: if s.viable { 1.0 } else { 0.0 },
            viable: s.viable,
            sep_arcsec: s.sep_arcsec,
        })
        .collect();
    let real = info.geometry_kind.as_deref() == Some("real");
    let (aperture_source, geometry_path) = if real {
        let sectors = geometry_doc
            .get("sectors")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let mut apertures = Vec::new();
        let mut paths = Vec::new();
        for sector in sectors {
            let has_aperture = sector
                .get("aperture")
                .and_then(Value::as_array)
                .is_some_and(|aperture| !aperture.is_empty());
            apertures.push(if has_aperture {
                "spoc_lc_mask"
            } else {
                "default_5x5"
            });
            paths.push(if has_aperture {
                "spoc_mask"
            } else {
                "u_default"
            });
        }
        apertures.sort_unstable();
        apertures.dedup();
        paths.sort_unstable();
        paths.dedup();
        (
            apertures.into_iter().map(str::to_string).collect(),
            paths.into_iter().map(str::to_string).collect(),
        )
    } else {
        (vec!["synthetic".into()], vec!["synthetic".into()])
    };
    let g = super::geometry_v3::GeometryV3Info {
        geometry_kind: info.geometry_kind.unwrap_or_else(|| "synthetic".into()),
        geometry_mode: "upstream".into(),
        geometry_n_sectors: info.geometry_n_sectors,
        geometry_n_task_sectors: info.geometry_n_task_sectors,
        geometry_coverage: info.geometry_coverage,
        geometry_source: info.geometry_source,
        aperture_source,
        geometry_path,
        geometry_path_emulated: false,
        geometry_n_epochs: vec![],
        geometry_sigma_cal: vec![],
        n_nodes: 0,
        n_viable_neighbours: info.n_viable_neighbours,
        n_viability_weighted: 0,
        viability_mass_skipped: 0.0,
        fluxratio_target: info.fluxratio_target,
    };
    Ok((output, g))
}

/// Run the corrected source-aware model with the convention choices dictated
/// by the light-curve product. Unknown/QLP sources are evaluated both ways;
/// each alternative begins from the same seed and the selected result carries
/// its own diagnostics.
pub fn calc_fpp_by_source_v3(
    input: &CalcV3Inputs<'_>,
    n_draws: usize,
    seed: u32,
    production_fold: bool,
) -> Result<CalcV3Result, FppError> {
    calc_fpp_by_source_v3_impl(input, n_draws, seed, production_fold, false, McOptions::default())
}

/// [`calc_fpp_by_source_v3`] with Python's internal step-7 parameters (`adaptive_n`, `n_cap`, `futility_stop`).
pub fn calc_fpp_by_source_v3_with(
    input: &CalcV3Inputs<'_>,
    n_draws: usize,
    seed: u32,
    production_fold: bool,
    options: McOptions,
) -> Result<CalcV3Result, FppError> {
    calc_fpp_by_source_v3_impl(input, n_draws, seed, production_fold, false, options)
}

/// Targeted numerical probe entry point. It captures the most recent full
/// background-binary log-weight vectors for exact reducer comparisons; the
/// normal worker call does not allocate or retain them.
#[doc(hidden)]
pub fn calc_fpp_by_source_v3_with_logw_capture(
    input: &CalcV3Inputs<'_>,
    n_draws: usize,
    seed: u32,
    production_fold: bool,
) -> Result<CalcV3Result, FppError> {
    calc_fpp_by_source_v3_impl(input, n_draws, seed, production_fold, true, McOptions::default())
}

fn calc_fpp_by_source_v3_impl(
    input: &CalcV3Inputs<'_>,
    n_draws: usize,
    seed: u32,
    production_fold: bool,
    capture_logw: bool,
    options: McOptions,
) -> Result<CalcV3Result, FppError> {
    let plan = dispatch::dilution_plan(input.lc_source);
    let mut runs = BTreeMap::new();
    for corrected in plan.conventions {
        let run = run_convention(
            input,
            n_draws,
            seed,
            corrected,
            production_fold,
            capture_logw,
            options,
        )?;
        runs.insert(corrected, run);
    }
    let selected = dispatch::choose_dilution_run(&runs, |r: &CalcV3Result| r.fpp)
        .map_err(FppError::failed)?;
    let alternatives = runs
        .iter()
        .map(|(&corrected, r)| {
            (
                corrected,
                DilutionAlternative {
                    fpp: r.fpp,
                    nfpp: r.nfpp,
                    verdict: r.verdict.clone(),
                    scenarios: r.scenarios.clone(),
                },
            )
        })
        .collect();
    // `n_draws_all_runs = sum(r.get("n_draws_total", 0) for r in runs.values())` (r2 `:5829`)
    let n_draws_all_runs = runs.values().map(|r| r.n_draws_total).sum();
    // v3 gate observer: every run's records, in the order Python runs the conventions
    let mut gate_runs = Vec::new();
    for corrected in dispatch::dilution_plan(input.lc_source).conventions {
        if let Some(r) = runs.get_mut(&corrected) {
            gate_runs.append(&mut r.gate_runs);
        }
    }
    let mut result = runs
        .remove(&selected)
        .ok_or_else(|| FppError::failed("selected dilution run disappeared"))?;
    result.gate_runs = gate_runs;
    result.dilution_alternatives = alternatives;
    result.n_draws_all_runs = n_draws_all_runs;
    Ok(result)
}

/// The per-scenario draw-weight records a scenario function produced (Python `_sc`, r2 `:5465-5472`): none
/// (no reduction) leaves the scenario without a record; a count different from the names is an error, which
/// Python raises as RuntimeError -> `None` (a failure, never a release).
fn take_records(
    sink: &DiagSink,
    n0: usize,
    names: &[String],
    diag: &mut HashMap<String, DrawDiag>,
) -> Result<(), FppError> {
    let got = sink.since(n0);
    if !got.is_empty() && got.len() != names.len() {
        return Err(FppError::failed(format!(
            "{} draw-weight records for {} scenarios",
            got.len(),
            names.len()
        )));
    }
    for (name, d) in names.iter().zip(got) {
        diag.insert(name.clone(), d);
    }
    super::gate::assign(n0, names);
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn run_convention(
    input: &CalcV3Inputs<'_>,
    n_draws: usize,
    seed: u32,
    dilution_corrected: bool,
    production_fold: bool,
    capture_logw: bool,
    options: McOptions,
) -> Result<CalcV3Result, FppError> {
    // r2 `:5283-5285`: refine = switch if adaptive_n is None else adaptive_n; mc_policy = switch or refine
    let mc_switch = production_fold;
    let refine = options.adaptive_n.unwrap_or(mc_switch);
    let mc_policy = mc_switch || refine;

    // Provenance gate first, before any work (Python `fpp-require-pop-provenance` 7b30ba1, decision 2026-10-25 option A):
    // a supplied population whose sha256 or version is missing or empty is refused as `population_invalid` (a
    // release, never a score). A missing population keeps its own `population_missing` refusal below.
    if input.population.is_some() && population_provenance_missing(input.population_sha256, input.population_version) {
        return Err(FppError::unavailable(
            FppUnavailableReason::PopulationInvalid,
            "population provenance is incomplete (population_sha256/population_version missing)",
        ));
    }

    let depth_scale = dispatch::depth_scale(input.lc_source, input.tglc_near_edge);
    let (time_raw, mut flux_raw) = finite_lc(input.time, input.flux)?;
    let cadence_s = if production_fold {
        Some(match input.cadence_s {
            Some(cadence) => cadence,
            None => inferred_cadence_s(&time_raw)?,
        })
    } else {
        None
    };
    let depth_ppm = input.depth_ppm * depth_scale;
    if depth_scale != 1.0 {
        for f in &mut flux_raw {
            *f = 1.0 + (*f - 1.0) * depth_scale;
        }
    }
    let components = sigma_components(&time_raw, &flux_raw, input.period, input.t0, input.duration)
        .ok_or_else(|| {
            FppError::skip(
                FppSkipReason::RawLcUnusable,
                "raw light curve unusable: no q-prime sigma estimate",
            )
        })?;

    let (time_pf, flux_pf, prod_fold_base) = if production_fold {
        let rows = fold_production_default(&time_raw, &flux_raw, input.t0, input.period);
        if rows.is_empty() || rows.len() != components.sigma_b.len() {
            // Python raises RuntimeError -> calc_fpp_native returns None, skip_reason None
            return Err(FppError::failed("production fold and q-prime sigma bins differ"));
        }
        let fold = prod_fold_arrays(&rows, input.period, &components.sigma_b).map_err(FppError::failed)?;
        let flux_b: Vec<f64> = rows.iter().map(|r| r.flux_mean).collect();
        let constants = build_prod_fold_consts(&fold, &flux_b, cadence_s.unwrap()).map_err(FppError::failed)?;
        (Vec::new(), Vec::new(), Some((fold, flux_b, constants)))
    } else {
        let (t0, f0) = phase_fold_bin(&time_raw, &flux_raw, input.period, input.t0, 1000);
        let mut t = Vec::with_capacity(t0.len());
        let mut f = Vec::with_capacity(f0.len());
        for (&ti, &fi) in t0.iter().zip(&f0) {
            if ti.is_finite() && fi.is_finite() {
                t.push(ti);
                f.push(fi);
            }
        }
        if t.len() < 10 {
            // Python: "too few phase-folded points" -> None, skip_reason None
            return Err(FppError::failed("fewer than ten phase-folded bins"));
        }
        (t, f, None)
    };

    let nearby = parse_nearby(input.nearby_raw, input.tic_id)
        .ok_or_else(|| FppError::failed("nearby_unavailable: malformed nearby-star response"))?;
    if nearby.stars.is_empty() {
        return Err(FppError::failed("nearby_unavailable: no TIC stars"));
    }
    let stars: Vec<ApertureStar> = nearby
        .stars
        .iter()
        .map(|s| ApertureStar {
            ra: s.ra,
            dec: s.dec,
            tmag: s.tmag,
            extra: Map::new(),
        })
        .collect();
    // 1. geometry (`_flux_ratios_for_mode`, r2 `:5390`) BEFORE the population (task-30 redo note 3)
    let (smooth_stars, geometry_info) = if production_fold {
        if let Some(geometry_doc) = input.geometry {
            let geometry: GeometryV3 = serde_json::from_value(geometry_doc.clone())
                .map_err(|e| FppError::failed(format!("geometry_invalid: {e}")))?;
            if geometry.kind == "real" && !geometry.sectors.is_empty() {
                compute_flux_ratios_smooth(
                    &stars,
                    depth_ppm,
                    &geometry,
                    dilution_corrected,
                    input.lc_source,
                    0.01,
                )
                // Every smooth-geometry error is a malformed-input failure:
                // the node table that Python can lose at run time
                // (`GeometryUnavailable`) is compiled into this crate.
                .map_err(|e| FppError::failed(format!("geometry_invalid: {e}")))?
            } else {
                geometry_fallback(
                    &stars,
                    depth_ppm,
                    geometry_doc,
                    dilution_corrected,
                    "smooth",
                )
                .map_err(FppError::failed)?
            }
        } else {
            geometry_fallback(
                &stars,
                depth_ppm,
                &Value::Null,
                dilution_corrected,
                "smooth",
            )
            .map_err(FppError::failed)?
        }
    } else {
        geometry_fallback(
            &stars,
            depth_ppm,
            input.geometry.unwrap_or(&Value::Null),
            dilution_corrected,
            "upstream",
        )
        .map_err(FppError::failed)?
    };
    // 2. population, 3. cross-check type, 4. alarm (r2 `:5392-5399`)
    let target_tmag = nearby.stars[0].tmag;
    let population = validate_population(
        input.population,
        target_tmag,
        input.population_sha256,
        input.population_version,
        Some(input.tic_id),
    )?;
    if input
        .population_crosscheck
        .is_some_and(|value| !value.is_null() && !value.is_object())
    {
        return Err(FppError::unavailable(
            FppUnavailableReason::PopulationInvalid,
            "invalid population cross-check",
        ));
    }
    let crosscheck_ok = input
        .population_crosscheck
        .and_then(|value| value.get("gaia_available"))
        .and_then(Value::as_bool)
        == Some(true);
    if input
        .population_crosscheck
        .and_then(|v| v.get("alarm"))
        .is_some_and(python_truthy)
        && population.source != "gaia"
    {
        return Err(FppError::unavailable(
            FppUnavailableReason::PopulationAlarm,
            "R8 alarm requires a server-supplied Gaia population",
        ));
    }
    let prepared: Vec<PreparedStar> = nearby
        .stars
        .into_iter()
        .zip(smooth_stars)
        .map(|(rec, smooth)| PreparedStar { rec, smooth })
        .collect();
    let target = &prepared[0].rec;
    // 5. target stellar parameters (r2 `:5409-5422`): explicit skip_reason, never an invented mass
    if target.mass.is_nan() || target.rad.is_nan() || target.teff.is_nan() {
        return Err(FppError::skip(
            missing_stellar_params_reason(target),
            "missing target stellar parameters",
        ));
    }
    // 6. target limb darkening (LDCNoRow -> skip_reason ldc_no_row)
    let (u1, u2) = target_limb_darkening(target)?;
    let target_flux_frame = dispatch::frame_fluxratio(
        prepared[0].smooth.fluxratio,
        prepared[0].smooth.fluxratio,
        dilution_corrected,
    );
    let (flux_t, sigma_t, fold_t_owned) = if let Some((fold, flux_b, _)) = &prod_fold_base {
        let (f, sig, base_chi2, lnnorm) = prod_fold_renorm(flux_b, &fold.sigma, target_flux_frame).map_err(FppError::failed)?;
        let mut renorm_fold = fold.clone();
        renorm_fold.sigma = sig.clone();
        let mut constants = build_prod_fold_consts(&renorm_fold, &f, cadence_s.unwrap()).map_err(FppError::failed)?;
        constants.base_chi2 = base_chi2;
        constants.lnnorm = lnnorm;
        (
            Vec::new(),
            components.sigma_oot / target_flux_frame,
            Some((renorm_fold, f, sig, constants)),
        )
    } else {
        let (f, sig) = renorm_flux(&flux_pf, components.sigma_oot, target_flux_frame);
        (f, sig, None)
    };
    let fold_t = fold_t_owned.as_ref().map(|(fold, f, _, c)| ProdFold {
        fold,
        f,
        base_chi2: c.base_chi2,
        lnnorm: c.lnnorm,
        w: c.w,
        delta: c.delta,
    });

    let target_config = TargetFamilyConfig {
        roche: production_fold,
        secondary_consistent: production_fold,
        k_nudge_upstream: !production_fold,
        upstream_exposure: !production_fold,
    };
    let d_config = DFamilyConfig {
        parity_index: !production_fold,
        circular_prior: production_fold,
        target_q: true,
        roche: production_fold,
        secondary_consistent: production_fold,
        k_nudge_upstream: !production_fold,
        upstream_exposure: !production_fold,
    };
    let sigma_vec = vec![sigma_t; time_pf.len()];
    let n_nearby = prepared.len().saturating_sub(1);
    let high_crowding = nearby.gal_b.abs() < 10.0 || n_nearby > 15;

    // Neighbour likelihood frames do not depend on the stage: build them once (Python rebuilds identical values
    // inside every pass; the renormalisation is deterministic).
    struct NeighbourFrame {
        tag: String,
        m: f64,
        r: f64,
        teff: f64,
        ln_v: f64,
        flux: Vec<f64>,
        sigma: f64,
        fold: Option<(super::ladder::FoldArrays, Vec<f64>, Vec<f64>, super::fold::ProdFoldConsts)>,
    }
    let mut neighbours = Vec::new();
    let mut viable_index = 0usize;
    for ns in &prepared[1..] {
        if !ns.smooth.viable {
            continue;
        }
        let m = if ns.rec.mass.is_finite() { ns.rec.mass } else { 1.0 };
        let r = if ns.rec.rad.is_finite() { ns.rec.rad } else { 1.0 };
        let teff = if ns.rec.teff.is_finite() { ns.rec.teff } else { 5777.0 };
        let fr = dispatch::frame_fluxratio(
            ns.smooth.fluxratio,
            prepared[0].smooth.fluxratio,
            dilution_corrected,
        );
        let (ns_flux, ns_sigma, ns_fold_owned) = if let Some((base, _, _)) = &prod_fold_base {
            let (f, sig, base_chi2, lnnorm) =
                prod_fold_renorm(&base.flux, &base.sigma, fr).map_err(FppError::failed)?;
            let mut renorm_fold = base.clone();
            renorm_fold.sigma = sig.clone();
            let mut constants = build_prod_fold_consts(&renorm_fold, &f, cadence_s.unwrap())
                .map_err(FppError::failed)?;
            constants.base_chi2 = base_chi2;
            constants.lnnorm = lnnorm;
            (
                Vec::new(),
                components.sigma_oot / fr,
                Some((renorm_fold, f, sig, constants)),
            )
        } else {
            let (f, sig) = renorm_flux(&flux_pf, components.sigma_oot, fr);
            (f, sig, None)
        };
        let ln_v = if ns.smooth.viability > 0.0 {
            ns.smooth.viability.ln()
        } else {
            f64::NEG_INFINITY
        };
        neighbours.push(NeighbourFrame {
            tag: format!("{}:{viable_index}", ns.rec.id),
            m,
            r,
            teff,
            ln_v,
            flux: ns_flux,
            sigma: ns_sigma,
            fold: ns_fold_owned,
        });
        viable_index += 1;
    }

    // One Monte-Carlo pass of every scenario with `n` draws and a FRESH RandomState(seed) (Python `_mc_pass` +
    // `_stage_pass`, r2 `:5455-5625`): the exposure / Kepler diagnostics are per pass, so the reported stage's
    // diagnostics are its own.
    let mc_pass = |n: usize, seed_k: u64| -> Result<StagePass<PassDiagnostics>, FppError> {
        let mut rng = FppRng::new(seed_k as u32);
        let mut d_diag = DDiagnostics {
            capture_logw,
            ..DDiagnostics::default()
        };
        let mut target_diag = TargetDiagnostics::default();
        let sink = DiagSink::start();
        super::gate::begin_pass();
        let mut diag = HashMap::<String, DrawDiag>::new();
        let mut lnz = Vec::<(String, f64)>::new();
        let names = |list: &[&str]| list.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let no_companion = [0.0];

        let n0 = sink.len();
        let tp = target_family::lnz_ttp(
            &time_pf,
            &flux_t,
            sigma_t,
            input.period,
            target.mass,
            target.rad,
            n,
            u1,
            u2,
            &no_companion,
            false,
            &no_companion,
            &mut rng,
            target_config,
            &mut target_diag,
            fold_t.as_ref(),
        )?;
        take_records(&sink, n0, &names(&["TP"]), &mut diag)?;
        lnz.push(("TP".into(), tp));
        let n0 = sink.len();
        let (eb, ebx) = target_family::lnz_teb(
            &time_pf,
            &flux_t,
            sigma_t,
            sigma_t,
            input.period,
            target.mass,
            target.rad,
            target.teff,
            n,
            u1,
            u2,
            &no_companion,
            false,
            &no_companion,
            &mut rng,
            target_config,
            &mut target_diag,
            fold_t.as_ref(),
        )?;
        take_records(&sink, n0, &names(&["EB", "EBx2P"]), &mut diag)?;
        lnz.push(("EB".into(), eb));
        lnz.push(("EBx2P".into(), ebx));
        if !target.plx.is_nan() {
            let n0 = sink.len();
            let ptp = target_family::lnz_ptp(
                &time_pf,
                &flux_t,
                sigma_t,
                input.period,
                target.mass,
                target.rad,
                n,
                u1,
                u2,
                target.plx,
                &mut rng,
                target_config,
                &mut target_diag,
                fold_t.as_ref(),
            )?;
            take_records(&sink, n0, &names(&["PTP"]), &mut diag)?;
            lnz.push(("PTP".into(), ptp));
            let n0 = sink.len();
            let (peb, pebx) = target_family::lnz_peb(
                &time_pf,
                &flux_t,
                sigma_t,
                sigma_t,
                input.period,
                target.mass,
                target.rad,
                target.teff,
                n,
                u1,
                u2,
                target.plx,
                &mut rng,
                target_config,
                &mut target_diag,
                fold_t.as_ref(),
            )?;
            take_records(&sink, n0, &names(&["PEB", "PEBx2P"]), &mut diag)?;
            lnz.push(("PEB".into(), peb));
            lnz.push(("PEBx2P".into(), pebx));
        } else {
            for name in ["PTP", "PEB", "PEBx2P"] {
                lnz.push((name.into(), f64::NEG_INFINITY));
            }
        }
        if !target.plx.is_nan() {
            let n0 = sink.len();
            let stp = lnz_stp_with_details(
                &time_pf,
                &flux_t,
                &sigma_vec,
                input.period,
                target.mass,
                target.rad,
                target.teff,
                n,
                target.plx,
                &mut rng,
                fold_t.as_ref(),
                production_fold,
                !production_fold,
            )?;
            take_records(&sink, n0, &names(&["STP"]), &mut diag)?;
            let s_unconverged = stp.lnl.iter().filter(|x| x.is_nan()).count() as u64;
            target_diag.observe_production(&stp.lnl, &stp.status);
            lnz.push(("STP".into(), stp.lnz));
            let n0 = sink.len();
            let seb = lnz_seb_with_details(
                &time_pf,
                &flux_t,
                &sigma_vec,
                sigma_t,
                input.period,
                target.mass,
                target.rad,
                target.teff,
                n,
                target.plx,
                &mut rng,
                fold_t.as_ref(),
                production_fold,
                production_fold,
                !production_fold,
                !production_fold,
            )?;
            take_records(&sink, n0, &names(&["SEB", "SEBx2P"]), &mut diag)?;
            let s_unconverged = s_unconverged
                + seb.lnl_eb.iter().filter(|x| x.is_nan()).count() as u64
                + seb.lnl_tw.iter().filter(|x| x.is_nan()).count() as u64;
            target_diag.observe_production(&seb.lnl_eb, &seb.status_eb);
            target_diag.observe_production(&seb.lnl_tw, &seb.status_tw);
            lnz.push(("SEB".into(), seb.lnz_seb));
            lnz.push(("SEBx2P".into(), seb.lnz_sebx2p));
            target_diag.kepler_unconverged_draws += s_unconverged;
        } else {
            for name in ["STP", "SEB", "SEBx2P"] {
                lnz.push((name.into(), f64::NEG_INFINITY));
            }
        }
        let n0 = sink.len();
        let dtp = if let Some(pf) = fold_t.as_ref() {
            d_family::lnz_dtp_with_fold(
                &time_pf, &flux_t, sigma_t, input.period, target.mass, target.rad, n, u1, u2,
                &population, &mut rng, d_config, &mut d_diag, pf,
            )?
        } else {
            d_family::lnz_dtp(
                &time_pf, &flux_t, sigma_t, input.period, target.mass, target.rad, n, u1, u2,
                &population, &mut rng, d_config, &mut d_diag,
            )?
        };
        take_records(&sink, n0, &names(&["DTP"]), &mut diag)?;
        lnz.push(("DTP".into(), dtp));
        let n0 = sink.len();
        let (deb, debx) = if let Some(pf) = fold_t.as_ref() {
            d_family::lnz_deb_with_fold(
                &time_pf, &flux_t, sigma_t, sigma_t, input.period, target.mass, target.rad, target.teff,
                u1, u2, n, &population, &mut rng, d_config, &mut d_diag, pf,
            )?
        } else {
            d_family::lnz_deb(
                &time_pf, &flux_t, sigma_t, sigma_t, input.period, target.mass, target.rad, target.teff,
                u1, u2, n, &population, &mut rng, d_config, &mut d_diag,
            )?
        };
        take_records(&sink, n0, &names(&["DEB", "DEBx2P"]), &mut diag)?;
        lnz.push(("DEB".into(), deb));
        lnz.push(("DEBx2P".into(), debx));
        let n0 = sink.len();
        let btp = if let Some(pf) = fold_t.as_ref() {
            d_family::lnz_btp_with_fold(
                &time_pf, &flux_t, sigma_t, input.period, n, &population, &mut rng, d_config, &mut d_diag, pf,
            )?
        } else {
            d_family::lnz_btp(
                &time_pf, &flux_t, sigma_t, input.period, n, &population, &mut rng, d_config, &mut d_diag,
            )?
        };
        take_records(&sink, n0, &names(&["BTP"]), &mut diag)?;
        lnz.push(("BTP".into(), btp));
        let n0 = sink.len();
        let (beb, bebx) = if let Some(pf) = fold_t.as_ref() {
            d_family::lnz_beb_with_fold(
                &time_pf, &flux_t, sigma_t, sigma_t, input.period, target.mass, target.rad, target.teff,
                u1, u2, n, &population, &mut rng, d_config, &mut d_diag, pf,
            )?
        } else {
            d_family::lnz_beb(
                &time_pf, &flux_t, sigma_t, sigma_t, input.period, target.mass, target.rad, target.teff,
                u1, u2, n, &population, &mut rng, d_config, &mut d_diag,
            )?
        };
        take_records(&sink, n0, &names(&["BEB", "BEBx2P"]), &mut diag)?;
        lnz.push(("BEB".into(), beb));
        lnz.push(("BEBx2P".into(), bebx));

        // Nearby star scenarios (NTP, NEB, NEBx2P per viable neighbour, in order)
        let mut nearby_scen = Vec::<(String, f64)>::new();
        for nb in &neighbours {
            // Python computes the neighbour's LDC inside the pass (LDCNoRow -> skip_reason ldc_no_row)
            let ns_logg = (G_CGS * nb.m * MSUN / libm_pow(nb.r * RSUN, 2.0)).log10();
            let (nu1, nu2) = stellar::get_ldc_checked(nb.teff, ns_logg, 0.0)?;
            let ns_fold = nb.fold.as_ref().map(|(fold, f, _, c)| ProdFold {
                fold,
                f,
                base_chi2: c.base_chi2,
                lnnorm: c.lnnorm,
                w: c.w,
                delta: c.delta,
            });
            let ntp_name = format!("NTP:{}", nb.tag);
            let neb_name = format!("NEB:{}", nb.tag);
            let nebx_name = format!("NEBx2P:{}", nb.tag);
            let n0 = sink.len();
            let ntp = target_family::lnz_ttp(
                &time_pf, &nb.flux, nb.sigma, input.period, nb.m, nb.r, n, nu1, nu2, &no_companion, false,
                &no_companion, &mut rng, target_config, &mut target_diag, ns_fold.as_ref(),
            )?;
            take_records(&sink, n0, std::slice::from_ref(&ntp_name), &mut diag)?;
            let n0 = sink.len();
            let (neb, nebx) = target_family::lnz_teb(
                &time_pf, &nb.flux, nb.sigma, nb.sigma, input.period, nb.m, nb.r, nb.teff, n, nu1, nu2,
                &no_companion, false, &no_companion, &mut rng, target_config, &mut target_diag, ns_fold.as_ref(),
            )?;
            take_records(&sink, n0, &[neb_name.clone(), nebx_name.clone()], &mut diag)?;
            nearby_scen.push((ntp_name, ntp + nb.ln_v));
            nearby_scen.push((neb_name, neb + nb.ln_v));
            nearby_scen.push((nebx_name, nebx + nb.ln_v));
        }
        let gate_records = super::gate::take_named();
        drop(sink);
        Ok(StagePass {
            lnz,
            nearby: nearby_scen,
            diag,
            payload: PassDiagnostics {
                target: target_diag,
                d: d_diag,
                gate: gate_records,
            },
        })
    };

    // Stage loop (r2 `:5614-5632`): stage 0 = the candidate's own seed; stage k >= 1 = stage_seed(...).
    let ladder = mc::stage_ladder(n_draws, refine, options.n_cap);
    let (seed_p, seed_t0) = input.seed_ephemeris.unwrap_or((input.period, input.t0));
    let seed_for_stage = |idx: usize| -> u64 {
        if idx == 0 {
            u64::from(seed)
        } else {
            u64::from(mc::stage_seed(
                input.tic_id,
                seed_p,
                seed_t0,
                idx,
                Some(u64::from(seed)),
            ))
        }
    };
    let staged = mc::run_stages(
        mc_pass,
        &ladder,
        seed_for_stage,
        options.futility_stop.unwrap_or(true),
    )?
    .ok_or_else(|| FppError::failed("all_scenarios_non_finite"))?;
    let mc::StagedOutcome {
        pass,
        summary: summ,
        assess,
        log: stage_log,
    } = staged;
    let PassDiagnostics {
        target: target_diag,
        d: d_diag,
        gate: mut gate_records,
    } = pass.payload;

    let fpp = summ.fpp;
    let nfpp = summ.nfpp;
    let borderline = mc_policy && !assess.resolved;
    let verdict = crate::fpp_policy::verdict(fpp, Some(nfpp), high_crowding)
        .as_str()
        .to_string();
    let kepler_unconverged_draws =
        target_diag.kepler_unconverged_draws + d_diag.kepler_unconverged_draws;
    let exposure_gl_fallback_share = target_diag
        .exposure_gl_fallback_share
        .max(d_diag.exposure_gl_fallback_share);
    let exposure_unconverged_share = target_diag
        .exposure_unconverged_share
        .max(d_diag.exposure_unconverged_share);
    let exposure_unconverged = exposure_unconverged_share > 0.01;
    let geometry_real =
        geometry_info.geometry_kind == "real" && geometry_info.geometry_coverage >= 1.0;
    let ffi_reference = geometry_info
        .geometry_source
        .iter()
        .any(|source| source == "ffi_reference");
    let validated_ffi_tglc_path = geometry_info.geometry_mode == "smooth"
        && geometry_info.geometry_path == ["tglc_box"]
        && geometry_info.aperture_source == ["tglc_box_header"]
        && !geometry_info.geometry_path_emulated
        && !geometry_info.geometry_n_epochs.is_empty()
        && geometry_info.geometry_n_epochs.iter().all(|&n| n >= 2);
    let ffi_reference_unvalidated = ffi_reference && !validated_ffi_tglc_path;
    let smooth_path_unvalidated = geometry_info.geometry_mode == "smooth"
        && (geometry_info.geometry_path_emulated
            || geometry_info
                .geometry_path
                .iter()
                .any(|path| !matches!(path.as_str(), "spoc_mask" | "tglc_box")));
    let geometry_unvalidated =
        !geometry_real || ffi_reference_unvalidated || smooth_path_unvalidated;
    let caps = VerdictCaps {
        kepler_unconverged_draws,
        population_crosscheck_verified: crosscheck_ok,
        dilution_unverified: dispatch::dilution_plan(input.lc_source).unverified,
        tglc_near_edge_unknown: dispatch::normalize_lc_source(input.lc_source) == "tglc"
            && input.tglc_near_edge.is_none(),
        geometry_path_emulated: geometry_unvalidated,
        ffi_reference_unvalidated,
    };
    // Run-level verdict rules and flags in Python's order (r2 `:5640-5693`), then `calc_fpp_by_source`'s
    // dilution / TGLC flags and the remaining caps.
    let (mass_from_relation, giant_host) = mass_fallback_flags(target);
    let unstable_on = match options.unstable_flag {
        Some(v) => v,
        None => crate::task_processor::parse_unstable_flag_switch_for_mode(production_fold)?,
    };
    let unstable = mc::is_unstable(&summ);
    let (mut verdict, mut flags) = run_level_verdict(
        verdict,
        &RunLevelGates {
            mc_policy,
            borderline,
            fpp,
            fpp_se: summ.fpp_se,
            nfpp,
            nfpp_se: summ.nfpp_se,
            crosscheck_ok,
            mass_from_relation,
            giant_host,
            unstable_on,
            unstable,
        },
    );
    if caps.dilution_unverified {
        flags.push("dilution_unverified".into());
    }
    if caps.tglc_near_edge_unknown {
        flags.push("tglc_near_edge_unknown".into());
    }
    verdict = finalize_verdict(
        verdict,
        &mut flags,
        caps,
        components.fit_ill_conditioned,
        exposure_unconverged,
        input.tglc_near_edge,
        dispatch::normalize_lc_source(input.lc_source) == "tglc",
    );
    // fpp_scen: lnZ of every scenario of the reported stage with its probability (dict(lnZ, **nearby_scen) order)
    let mut scenarios = BTreeMap::new();
    for (i, name) in summ.names.iter().enumerate() {
        let z = pass
            .lnz
            .iter()
            .chain(&pass.nearby)
            .find(|(n, _)| n == name)
            .map(|(_, z)| *z)
            .expect("summary names come from the pass");
        scenarios.insert(
            name.clone(),
            ScenarioResult {
                lnz: z,
                probability: summ.probs[i],
            },
        );
    }
    let gate_scenarios: Vec<(String, f64, f64, Option<super::gate::GateRecord>)> = if super::gate::active() {
        summ.names
            .iter()
            .map(|name| {
                let s = &scenarios[name];
                (name.clone(), s.lnz, s.probability, gate_records.remove(name).flatten())
            })
            .collect()
    } else {
        Vec::new()
    };
    let mc_scen = summ
        .names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), [summ.ess[i], summ.rel_var[i].sqrt(), summ.tail_scale[i]]))
        .collect();
    Ok(CalcV3Result {
        fpp,
        nfpp,
        verdict,
        scenarios,
        flags,
        dilution_corrected,
        dilution_unverified: caps.dilution_unverified,
        depth_scale,
        population,
        population_crosscheck_status: if crosscheck_ok {
            "passed"
        } else {
            "unverified"
        }
        .into(),
        kepler_unconverged_draws,
        geometry: geometry_info,
        sigma_source: "q_prime".into(),
        sigma_fit_ill_conditioned: components.fit_ill_conditioned,
        exposure_unconverged,
        exposure_unconverged_share,
        exposure_gl_fallback_share,
        production_switch_set: production_fold,
        cadence_s,
        high_crowding,
        gal_b: (nearby.gal_b * 100.0).round_ties_even() / 100.0,
        n_nearby,
        seed,
        dilution_alternatives: BTreeMap::new(),
        fpp_mc_se: summ.fpp_se,
        nfpp_mc_se: summ.nfpp_se,
        n_draws_final: summ.n,
        n_draws_total: stage_log.iter().map(|r| r.n).sum(),
        n_draws_all_runs: stage_log.iter().map(|r| r.n).sum(),
        mc_scen,
        mc_min_ess: summ.min_ess,
        mc_min_ess_scenario: summ.min_ess_scenario.clone(),
        mc_policy,
        borderline,
        borderline_reasons: if borderline {
            assess.reasons.clone()
        } else {
            Vec::new()
        },
        refinement_stages: stage_log,
        d_last_binary_reductions: d_diag.last_binary_reductions,
        d_last_binary_logw: d_diag.last_binary_logw,
        d_last_binary_prior: d_diag.last_binary_prior,
        d_last_binary_lnl: d_diag.last_binary_lnl,
        d_last_beb_draw_832: d_diag.last_beb_draw_832,
        gate_runs: if super::gate::active() {
            vec![GateRun {
                dilution_corrected,
                fpp,
                nfpp,
                scenarios: gate_scenarios,
            }]
        } else {
            Vec::new()
        },
        unstable_on,
        fpp_unstable: unstable_on && unstable,
        mc_ess_dom: if unstable_on {
            let (_, ess) = mc::dominant_ess(&summ);
            if ess.is_finite() {
                Some(ess)
            } else {
                None
            }
        } else {
            None
        },
        mc_dominant_scenario: if unstable_on {
            mc::dominant(&summ).0.to_string()
        } else {
            String::new()
        },
        fpp_interval: if unstable_on {
            [summ.tail_box_fpp.0, summ.tail_box_fpp.1]
        } else {
            [0.0, 0.0]
        },
    })
}


#[cfg(test)]
mod verdict_gate_tests {
    use super::*;

    #[test]
    fn production_verdict_finalizer_demotes_each_registered_cap() {
        let cases = [
            VerdictCaps {
                kepler_unconverged_draws: 1,
                population_crosscheck_verified: true,
                ..Default::default()
            },
            VerdictCaps {
                population_crosscheck_verified: false,
                ..Default::default()
            },
            VerdictCaps {
                population_crosscheck_verified: true,
                dilution_unverified: true,
                ..Default::default()
            },
            VerdictCaps {
                population_crosscheck_verified: true,
                tglc_near_edge_unknown: true,
                ..Default::default()
            },
            VerdictCaps {
                population_crosscheck_verified: true,
                geometry_path_emulated: true,
                ..Default::default()
            },
            VerdictCaps {
                population_crosscheck_verified: true,
                ffi_reference_unvalidated: true,
                ..Default::default()
            },
        ];
        for (idx, caps) in cases.into_iter().enumerate() {
            let mut flags = Vec::new();
            assert_eq!(
                finalize_verdict(
                    "validated".into(),
                    &mut flags,
                    caps,
                    false,
                    false,
                    None,
                    false,
                ),
                "candidate",
                "cap case {idx}"
            );
        }
        assert_eq!(
            finalize_verdict(
                "validated".into(),
                &mut Vec::new(),
                VerdictCaps {
                    population_crosscheck_verified: true,
                    ..Default::default()
                },
                false,
                false,
                None,
                false,
            ),
            "validated"
        );
        assert_eq!(
            finalize_verdict(
                "likely_fp".into(),
                &mut Vec::new(),
                VerdictCaps {
                    kepler_unconverged_draws: 1,
                    ..Default::default()
                },
                false,
                false,
                None,
                false,
            ),
            "likely_fp"
        );
    }
}
