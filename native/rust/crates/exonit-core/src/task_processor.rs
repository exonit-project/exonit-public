//! `PrimaryTaskProcessor` is the F2b integration point DESIGN.md/PLAN.md's
//! F0b task called out by name: the seam a `TaskProcessor` trait implementer
//! plugs into `runtime.rs`'s primary pool. F0b shipped only `StubTaskProcessor`
//! (network legs, no science); F2b wires the real pipeline transliterated
//! from `Sources/ExoNitCore/PrimaryTaskProcessor.swift` in the Swift
//! reference worker: load light curve -> preprocess (iterative detrend,
//! real BLS quick-search) -> `multi_planet::search` (BLS two-pass + harmonic
//! resolution + long-period sweep + protected slots) -> `vetting::vet_all`
//! -> CNN FFI / ExoMiner FFI scoring per candidate (ONNX, optional — see
//! below) -> injection-recovery completeness (optional, config-gated) ->
//! `PrimaryResultRequest`.
//!
//! `StubTaskProcessor` is kept (not deleted): `exonit-fake-integration`
//! still uses it when `EXONIT_MODELS_DIR` is unset (see its module doc), and
//! it remains the simplest network-only double for any future test that
//! wants one without touching ONNX models.

use crate::api::{WorkerApiClient, WorkerApiError};
use crate::bls::accelerated::ComputeBackend;
use crate::bls::quick::RealQuickBlsSearch;
use crate::bls::reference::BlsError;
use crate::detrend::DetrendBackend;
use crate::dto::{PrimaryResultRequest, StarMetadata, WorkerTask};
use crate::exominer_features::{self, FfiStellarMetadata};
use crate::ffi_views::{self, FfiViewError};
use crate::fits::{self, FitsError, LightCurveData, PhotometrySource};
use crate::harmonic_resolver::LightCurveSource;
use crate::injection_recovery::{self, InjectionBackend, InjectionRecoveryOptions};
use crate::long_period_sweep::LongPeriodSweepOptions;
use crate::multi_planet::{self, MultiPlanetError, MultiPlanetOptions};
use crate::onnx_cnn::{CnnFfiClassifier, OnnxFfiError};
use crate::onnx_exominer::ExoMinerFfiClassifier;
use crate::preprocess::{self, DetrendMode, PreprocessError, ProcessedLightCurve};
use crate::runtime::FitsCache;
use crate::vetting::{self, VettingError};
use serde_json::{Map, Value};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Processes one primary-pool task into a `PrimaryResultRequest`. Mirrors
/// the shape of Swift's `PrimaryTaskProcessor.process(_:workerID:)`, minus
/// its `async throws` (this crate is blocking-only, and F0b's stub never
/// fails the task for reasons a retry would fix — see `StubTaskProcessor`
/// doc comment).
pub trait TaskProcessor: Send + Sync {
    fn process(&self, task: &WorkerTask, worker_id: &str) -> PrimaryResultRequest;

    /// Native FPP-backfill processing (F8). Returns `None` when this
    /// processor does not support FPP (the runtime then never polls the
    /// pool). The default keeps every existing implementer (stub, tests)
    /// FPP-less. `raw_task` is the exact server dict, echoed back as
    /// `candidate` in the result, like the Python worker does.
    fn process_fpp(
        &self,
        _task: &crate::dto::BackfillTask,
        _raw_task: &Map<String, Value>,
        _worker_id: &str,
    ) -> Option<FppProcessOutcome> {
        None
    }

    /// Effective `harmonic_epoch_class` of the last primary task this
    /// processor started (python `_effective_harmonic_epoch_class`), `None`
    /// until a task has run. Published in the heartbeat `flag_states`.
    fn last_harmonic_epoch_class(&self) -> Option<bool> {
        None
    }
}

/// Result of one native FPP-backfill task. A calculated result is posted in
/// `request`; an unavailable v3 calculation instead carries a release body
/// and never carries a score sentinel.
pub struct FppProcessOutcome {
    pub request: crate::dto::FppResultRequest,
    pub error: Option<String>,
    /// Python-compatible no-score release request for unavailable v3 FPP.
    pub release: Option<Map<String, Value>>,
}

fn fpp_release_payload(
    task: &crate::dto::BackfillTask,
    raw_task: &Map<String, Value>,
    reason: &str,
) -> Map<String, Value> {
    let mut payload = Map::new();
    payload.insert("candidate_id".into(), Value::from(task.candidate_id));
    payload.insert(
        "claim_id".into(),
        raw_task
            .get("fpp_claim_id")
            .cloned()
            .unwrap_or(Value::Null),
    );
    payload.insert("tic_id".into(), Value::String(task.tic_id.clone()));
    payload.insert("reason".into(), Value::String(reason.to_string()));
    payload
}

fn fpp_release_outcome(
    task: &crate::dto::BackfillTask,
    raw_task: &Map<String, Value>,
    reason: &str,
    message: String,
) -> FppProcessOutcome {
    let mut request = crate::dto::FppResultRequest::new(
        Some(task.candidate_id),
        Some(task.tic_id.clone()),
        None,
    );
    request.candidate = Some(raw_task.clone());
    FppProcessOutcome {
        request,
        error: Some(message),
        release: Some(fpp_release_payload(task, raw_task, reason)),
    }
}

/// Outcome of a v3 calculation that returned an error. The decision is made
/// from the error's TYPE only: `FppError::Unavailable` releases the task with
/// its typed reason and carries no score; every other error is an ordinary
/// failure (`fpp = nfpp = -1`, `verdict = "failed"`), as Python returns
/// `None` for any exception that is not `FPPUnavailable`. The message text is
/// never inspected.
pub(crate) fn fpp_error_outcome(
    task: &crate::dto::BackfillTask,
    raw_task: &Map<String, Value>,
    mut req: crate::dto::FppResultRequest,
    err: crate::fpp::FppError,
) -> FppProcessOutcome {
    if let Some(reason) = err.unavailable_reason() {
        return fpp_release_outcome(task, raw_task, reason.as_str(), err.to_string());
    }
    req.fpp = Some(-1.0);
    req.nfpp = Some(-1.0);
    req.verdict = Some("failed".to_string());
    // WHY it failed, from the error's typed reason (never its text); `null` when Python has none (r2 worker.py
    // `:4058-4072`: `"skip_reason": getattr(calc_fpp, "last_skip_reason", None)`).
    req.skip_reason = Some(err.skip_reason().map(|r| r.as_str().to_string()));
    FppProcessOutcome {
        request: req,
        error: Some(err.to_string()),
        release: None,
    }
}

/// Maps a computed v3 calculation onto the result payload (Python result
/// schema), including the mandatory platform/libm provenance of plan (t) 2.
#[doc(hidden)]
pub fn fpp_v3_result_request(
    req: crate::dto::FppResultRequest,
    task: &crate::dto::BackfillTask,
    r: &crate::fpp::v2::calc_v3::CalcV3Result,
    loaded_lc_source: &str,
) -> crate::dto::FppResultRequest {
    fpp_v3_result_request_with_ephem(
        req,
        task,
        r,
        loaded_lc_source,
        None,
        false,
        parse_ephem_refine_switch_for_mode(r.production_switch_set).unwrap_or(r.production_switch_set),
    )
}

/// Maps a computed v3 calculation and ephemeris refinement onto the result payload.
#[doc(hidden)]
pub fn fpp_v3_result_request_with_ephem(
    mut req: crate::dto::FppResultRequest,
    task: &crate::dto::BackfillTask,
    r: &crate::fpp::v2::calc_v3::CalcV3Result,
    loaded_lc_source: &str,
    ephem_prov: Option<crate::fpp::ephem::EphemProvenance>,
    ephem_failed: bool,
    ephem_refine_on: bool,
) -> crate::dto::FppResultRequest {
    req.fpp = Some(r.fpp);
    req.nfpp = Some(r.nfpp);
    req.verdict = Some(r.verdict.clone());
    req.probability_planet = r.scenarios.get("TP").map(|s| s.probability);
    req.kepler_unconverged_draws = Some(r.kepler_unconverged_draws);
    req.fpp_flags = Some(r.flags.clone());
    req.fpp_model_version = Some(3);
    req.geometry_version = Some(2);
    req.geometry_model_version = Some(2);
    req.dilution_convention = Some(r.dilution_corrected.to_string());
    req.dilution_corrected = Some(r.dilution_corrected);
    req.dilution_unverified = Some(r.dilution_unverified);
    if r.dilution_alternatives.len() > 1 {
        let mut alternatives = Map::new();
        for (corrected, alternative) in &r.dilution_alternatives {
            let mut value = Map::new();
            value.insert("fpp".into(), Value::from(alternative.fpp));
            value.insert("nfpp".into(), Value::from(alternative.nfpp));
            alternatives.insert(
                if *corrected { "true" } else { "false" }.into(),
                Value::Object(value),
            );
        }
        req.dilution_alternatives = Some(alternatives);
    }
    req.depth_scale = Some(r.depth_scale);
    req.lc_source = Some(normalized_fpp_v3_lc_source(Some(&loaded_lc_source)));
    req.fpp_backend = Some("cpu-rust".into());
    req.fpp_platform = Some(crate::fpp::provenance::platform().into());
    req.fpp_libm = Some(crate::fpp::provenance::libm().into());
    req.fpp_build_profile = Some(crate::fpp::provenance::build_profile().into());
    req.fpp_seed = Some(r.seed as u64);
    req.high_crowding = Some(r.high_crowding);
    req.gal_b = Some(r.gal_b);
    req.n_nearby = Some(r.n_nearby);
    req.trilegal_used = Some(false);
    req.population_version = Some(r.population.version.clone());
    req.population_source = Some(r.population.source.clone());
    req.population_sha256 = Some(r.population.sha256.clone());
    req.population_generator = Some(r.population.generator.clone());
    req.population_mapping_version = Some(r.population.mapping_version.clone());
    req.n_comp = Some(r.population.n_comp);
    req.n_comp_fainter = Some(r.population.n_comp_fainter);
    req.population_crosscheck_status = Some(r.population_crosscheck_status.clone());
    req.geometry_kind = Some(r.geometry.geometry_kind.clone());
    req.geometry_mode = Some(r.geometry.geometry_mode.clone());
    req.geometry_coverage = Some(r.geometry.geometry_coverage);
    req.geometry_n_sectors = Some(r.geometry.geometry_n_sectors);
    req.geometry_n_task_sectors = Some(r.geometry.geometry_n_task_sectors);
    req.geometry_source = Some(r.geometry.geometry_source.clone());
    req.aperture_source = Some(r.geometry.aperture_source.clone());
    req.geometry_path = Some(r.geometry.geometry_path.clone());
    req.geometry_path_emulated = Some(r.geometry.geometry_path_emulated);
    if !r.geometry.geometry_n_epochs.is_empty() {
        req.geometry_n_epochs = Some(r.geometry.geometry_n_epochs.clone());
    }
    if !r.geometry.geometry_sigma_cal.is_empty() {
        req.geometry_sigma_cal = Some(r.geometry.geometry_sigma_cal.clone());
    }
    if r.geometry.n_nodes > 0 {
        req.n_nodes = Some(r.geometry.n_nodes);
    }
    let is_tglc = loaded_lc_source == "tglc";
    if is_tglc {
        req.tglc_near_edge = Some(task.tglc_near_edge);
        if task.tglc_near_edge.is_none() {
            req.tglc_near_edge_unknown = Some(true);
        }
    }
    req.n_viable_neighbours = Some(r.geometry.n_viable_neighbours);
    if r.geometry.geometry_mode == "smooth" {
        req.n_viability_weighted = Some(r.geometry.n_viability_weighted);
        req.viability_mass_skipped = Some(r.geometry.viability_mass_skipped);
    }
    req.fluxratio_target = Some(r.geometry.fluxratio_target);
    req.sigma_source = Some(r.sigma_source.clone());
    req.sigma_fit_ill_conditioned = Some(r.sigma_fit_ill_conditioned);
    req.exposure_unconverged = Some(r.exposure_unconverged);
    req.exposure_unconverged_share = Some(r.exposure_unconverged_share);
    req.exposure_gl_fallback_share = Some(r.exposure_gl_fallback_share);
    req.cadence_s = r.cadence_s;
    req.fpp_scen = Some(
        r.scenarios
            .iter()
            .map(|(name, result)| {
                let mut scenario = Map::new();
                scenario.insert("lnZ".into(), Value::from(result.lnz));
                scenario.insert("prob".into(), Value::from(result.probability));
                (name.clone(), Value::Object(scenario))
            })
            .collect(),
    );
    req.fpp_switches = Some(fpp_v3_switches_with_options(r.production_switch_set, ephem_refine_on, r.unstable_on));
    apply_fpp_mc_wire_fields(&mut req, &FppMcWireInput::from_result(r));
    if let Some(prov) = ephem_prov {
        req.ephem_refined = Some(prov.ephem_refined);
        req.ephem_status = Some(prov.ephem_status);
        req.ephem_t0_convention = Some(prov.ephem_t0_convention);
        req.ephem_p = Some(prov.ephem_p);
        req.ephem_t0 = Some(prov.ephem_t0);
        req.ephem_shift_h = Some(prov.ephem_shift_h);
        req.ephem_refine = prov.ephem_refine;
        if ephem_failed {
            let mut flags = req.fpp_flags.take().unwrap_or_default();
            flags.push("ephem_refine_failed".to_string());
            req.fpp_flags = Some(flags);
            req.ephem_refine_failed = Some(true);
            if req.verdict.as_deref() == Some("validated") {
                req.verdict = Some("candidate".to_string());
            }
        }
    }
    req
}

/// The engine-result keys `_fpp_result_model_fields` reads for the step-7 fields (a Python result dict's
/// `fpp_mc_se`, `nfpp_mc_se`, `mc_min_ess`, `n_draws_final`, `n_draws_total`, `mc_policy`, `borderline`,
/// `refinement_stages`, `borderline_reasons`, `mc_min_ess_scenario`, `n_draws_all_runs`, `mc_scen`).
#[derive(Clone, Debug)]
pub struct FppMcWireInput {
    pub fpp_mc_se: f64,
    pub nfpp_mc_se: f64,
    pub mc_min_ess: f64,
    pub n_draws_final: usize,
    pub n_draws_total: usize,
    pub mc_policy: bool,
    pub borderline: bool,
    pub refinement_stages: Vec<crate::fpp::v2::mc::StageRecord>,
    pub borderline_reasons: Vec<String>,
    pub mc_min_ess_scenario: String,
    pub n_draws_all_runs: usize,
    pub mc_scen: Vec<(String, [f64; 3])>,
    pub unstable_on: bool,
    pub mc_ess_dom: Option<f64>,
    pub mc_dominant_scenario: String,
    pub fpp_interval: [f64; 2],
}

impl FppMcWireInput {
    pub fn from_result(r: &crate::fpp::v2::calc_v3::CalcV3Result) -> Self {
        FppMcWireInput {
            fpp_mc_se: r.fpp_mc_se,
            nfpp_mc_se: r.nfpp_mc_se,
            mc_min_ess: r.mc_min_ess,
            n_draws_final: r.n_draws_final,
            n_draws_total: r.n_draws_total,
            mc_policy: r.mc_policy,
            borderline: r.borderline,
            refinement_stages: r.refinement_stages.clone(),
            borderline_reasons: r.borderline_reasons.clone(),
            mc_min_ess_scenario: r.mc_min_ess_scenario.clone(),
            n_draws_all_runs: r.n_draws_all_runs,
            mc_scen: r.mc_scen.clone(),
            unstable_on: r.unstable_on,
            mc_ess_dom: r.mc_ess_dom,
            mc_dominant_scenario: r.mc_dominant_scenario.clone(),
            fpp_interval: r.fpp_interval,
        }
    }
}

/// The 64 KiB cap the server applies to `mc_detail` (`worker._MC_DETAIL_MAX_BYTES`).
pub const MC_DETAIL_MAX_BYTES: usize = 64 * 1024;

fn wire_f64(x: f64) -> Value {
    // `_json_safe`: a non-finite float inside mc_detail is posted as null (serde_json does the same)
    Value::from(x)
}

/// One `refinement_stages` entry as JSON, with Python's `_mc_stage_record` keys.
pub fn fpp_stage_record_json(st: &crate::fpp::v2::mc::StageRecord) -> Value {
    let mut o = Map::new();
    o.insert("stage".into(), Value::from(st.stage as u64));
    o.insert("n".into(), Value::from(st.n as u64));
    o.insert("seed".into(), Value::from(st.seed));
    o.insert("fpp".into(), wire_f64(st.fpp));
    o.insert("nfpp".into(), wire_f64(st.nfpp));
    o.insert("fpp_mc_se".into(), wire_f64(st.fpp_mc_se));
    o.insert("nfpp_mc_se".into(), wire_f64(st.nfpp_mc_se));
    o.insert("min_ess".into(), wire_f64(st.min_ess));
    o.insert("min_ess_scenario".into(), Value::from(st.min_ess_scenario.clone()));
    o.insert(
        "heavy_scenarios".into(),
        Value::Array(st.heavy_scenarios.iter().cloned().map(Value::from).collect()),
    );
    o.insert("n_low_ess".into(), Value::from(st.n_low_ess as u64));
    o.insert("tail_box_fpp".into(), Value::Array(st.tail_box_fpp.iter().map(|&x| wire_f64(x)).collect()));
    o.insert("tail_box_nfpp".into(), Value::Array(st.tail_box_nfpp.iter().map(|&x| wire_f64(x)).collect()));
    o.insert("resolved".into(), Value::Bool(st.resolved));
    o.insert("reasons".into(), Value::Array(st.reasons.iter().cloned().map(Value::from).collect()));
    o.insert(
        "top_scenarios".into(),
        Value::Array(
            st.top_scenarios
                .iter()
                .map(|t| {
                    let mut s = Map::new();
                    s.insert("scenario".into(), Value::from(t.scenario.clone()));
                    s.insert("p".into(), wire_f64(t.p));
                    s.insert("ess".into(), wire_f64(t.ess));
                    s.insert("rel_se".into(), wire_f64(t.rel_se));
                    Value::Object(s)
                })
                .collect(),
        ),
    );
    Value::Object(o)
}

/// `worker._fpp_result_model_fields` for the step-7 fields (r2 worker.py `:606-677`), literally:
/// `fpp_n_draws <- n_draws_final`, `fpp_n_draws_total <- n_draws_total`, `mc_borderline` only when `mc_policy`,
/// `mc_detail = {stages, borderline_reasons, min_ess_scenario, n_draws_all_runs, mc_scen}` with NaN/inf -> null,
/// non-finite top-level `fpp_mc_se` / `nfpp_mc_se` / `mc_min_ess` omitted, and the 64 KiB trimming order measured on
/// the CPython canonical JSON text ([`crate::fpp::pyjson::canonical_dumps`]).
pub fn apply_fpp_mc_wire_fields(req: &mut crate::dto::FppResultRequest, r: &FppMcWireInput) {
    req.fpp_mc_se = Some(r.fpp_mc_se).filter(|x| x.is_finite());
    req.nfpp_mc_se = Some(r.nfpp_mc_se).filter(|x| x.is_finite());
    req.mc_min_ess = Some(r.mc_min_ess).filter(|x| x.is_finite());
    req.fpp_n_draws = Some(r.n_draws_final as u64);
    req.fpp_n_draws_total = Some(r.n_draws_total as u64);
    req.mc_borderline = if r.mc_policy { Some(r.borderline) } else { None };
    let mut detail = Map::new();
    detail.insert(
        "stages".into(),
        Value::Array(r.refinement_stages.iter().map(fpp_stage_record_json).collect()),
    );
    detail.insert(
        "borderline_reasons".into(),
        Value::Array(r.borderline_reasons.iter().cloned().map(Value::from).collect()),
    );
    detail.insert("min_ess_scenario".into(), Value::from(r.mc_min_ess_scenario.clone()));
    detail.insert("n_draws_all_runs".into(), Value::from(r.n_draws_all_runs as u64));
    let mut scen = Map::new();
    for (name, v) in &r.mc_scen {
        scen.insert(name.clone(), Value::Array(v.iter().map(|&x| wire_f64(x)).collect()));
    }
    detail.insert("mc_scen".into(), Value::Object(scen));
    if r.unstable_on {
        detail.insert("ess_dom".into(), wire_f64(r.mc_ess_dom.unwrap_or(f64::NAN)));
        detail.insert("dominant_scenario".into(), Value::from(r.mc_dominant_scenario.clone()));
        detail.insert("fpp_interval".into(), Value::Array(r.fpp_interval.iter().map(|&x| wire_f64(x)).collect()));
    }
    let mut detail = Value::Object(detail);
    trim_mc_detail(&mut detail);
    req.mc_detail = Some(detail);
}

/// The 64 KiB trimming of `_fpp_result_model_fields`: drop `mc_scen` (and mark `truncated`), then the stages'
/// `top_scenarios`; a body still over the cap is posted as is (Python logs a warning).
pub fn trim_mc_detail(detail: &mut Value) {
    let too_big = |d: &Value| crate::fpp::pyjson::canonical_dumps(d).len() > MC_DETAIL_MAX_BYTES;
    if !too_big(detail) {
        return;
    }
    if let Value::Object(o) = detail {
        o.remove("mc_scen");
        o.insert("truncated".into(), Value::Bool(true));
    }
    if !too_big(detail) {
        return;
    }
    if let Some(Value::Array(stages)) = detail.get_mut("stages") {
        for stage in stages.iter_mut() {
            if let Value::Object(s) = stage {
                s.remove("top_scenarios");
            }
        }
    }
    if too_big(detail) {
        eprintln!("[FPP] mc_detail still over {MC_DETAIL_MAX_BYTES} bytes after trimming — the server will reject it");
    }
}

/// Parses the FPP_EPHEM_REFINE environment switch strictly for the given mode (production vs parity).
/// Accepts exactly "on" (true) or "off" (false), with not present defaulting to `production`.
/// Any other value returns a typed FppError::failed, matching Python's ValueError.
pub fn parse_ephem_refine_switch_for_mode(production: bool) -> Result<bool, crate::fpp::FppError> {
    match std::env::var("FPP_EPHEM_REFINE") {
        Ok(v) if v == "on" => Ok(true),
        Ok(v) if v == "off" => Ok(false),
        Ok(other) => Err(crate::fpp::FppError::failed(format!(
            "invalid switch value FPP_EPHEM_REFINE={other:?}: must be 'on' or 'off'"
        ))),
        Err(std::env::VarError::NotPresent) => Ok(production),
        Err(e) => Err(crate::fpp::FppError::failed(format!(
            "failed to read env FPP_EPHEM_REFINE: {e}"
        ))),
    }
}

/// Parses the FPP_EPHEM_REFINE environment switch strictly.
/// When FPP_PARITY is "1", defaults to false ("off"); in production mode, defaults to true ("on").
pub fn parse_ephem_refine_switch() -> Result<bool, crate::fpp::FppError> {
    let parity = std::env::var("FPP_PARITY").is_ok_and(|v| v == "1");
    parse_ephem_refine_switch_for_mode(!parity)
}

/// Parses the FPP_UNSTABLE_FLAG environment switch strictly for the given mode (production vs parity).
/// Accepts exactly "on" (true) or "off" (false), with not present defaulting to `production`.
/// Any other value returns a typed FppError::failed, matching Python's ValueError.
pub fn parse_unstable_flag_switch_for_mode(production: bool) -> Result<bool, crate::fpp::FppError> {
    match std::env::var("FPP_UNSTABLE_FLAG") {
        Ok(v) if v == "on" => Ok(true),
        Ok(v) if v == "off" => Ok(false),
        Ok(other) => Err(crate::fpp::FppError::failed(format!(
            "invalid switch value FPP_UNSTABLE_FLAG={other:?}: must be 'on' or 'off'"
        ))),
        Err(std::env::VarError::NotPresent) => Ok(production),
        Err(e) => Err(crate::fpp::FppError::failed(format!(
            "failed to read env FPP_UNSTABLE_FLAG: {e}"
        ))),
    }
}

/// Parses the FPP_UNSTABLE_FLAG environment switch strictly.
/// When FPP_PARITY is "1", defaults to false ("off"); in production mode, defaults to true ("on").
pub fn parse_unstable_flag_switch() -> Result<bool, crate::fpp::FppError> {
    let parity = std::env::var("FPP_PARITY").is_ok_and(|v| v == "1");
    parse_unstable_flag_switch_for_mode(!parity)
}

/// The v3 switch set posted with every computed result (Python `get_fpp_switches()`, 13 keys).
pub fn fpp_v3_switches(production: bool) -> Map<String, Value> {
    let ephem_refine_on = parse_ephem_refine_switch_for_mode(production).unwrap_or(production);
    let unstable_flag_on = parse_unstable_flag_switch_for_mode(production).unwrap_or(production);
    fpp_v3_switches_with_options(production, ephem_refine_on, unstable_flag_on)
}

/// The v3 switch set posted with every computed result, parameterized by the exact refine switch used.
pub fn fpp_v3_switches_with_refine(production: bool, ephem_refine_on: bool) -> Map<String, Value> {
    let unstable_flag_on = parse_unstable_flag_switch_for_mode(production).unwrap_or(production);
    fpp_v3_switches_with_options(production, ephem_refine_on, unstable_flag_on)
}

/// The v3 switch set parameterized by both refine and unstable switches.
pub fn fpp_v3_switches_with_options(
    production: bool,
    ephem_refine_on: bool,
    unstable_flag_on: bool,
) -> Map<String, Value> {
    let ephem_refine = if ephem_refine_on { "on" } else { "off" };
    let unstable_flag = if unstable_flag_on { "on" } else { "off" };
    let values = if production {
        [
            ("FPP_ADAPTIVE_N", "on"),
            ("FPP_BACKGROUND_INDEX", "correct"),
            ("FPP_BACKGROUND_PRIOR", "circular"),
            ("FPP_BACKGROUND_Q", "target"),
            ("FPP_EPHEM_REFINE", ephem_refine),
            ("FPP_EXPOSURE", "cadence"),
            ("FPP_GEOMETRY", "smooth"),
            ("FPP_K1_NUDGE", "exact"),
            ("FPP_NORM", "upstream"),
            ("FPP_ORBIT_PHYSICS", "roche"),
            ("FPP_Q_BRANCH", "correct"),
            ("FPP_SECONDARY_GEOMETRY", "consistent"),
            ("FPP_UNSTABLE_FLAG", unstable_flag),
        ]
    } else {
        [
            ("FPP_ADAPTIVE_N", "off"),
            ("FPP_BACKGROUND_INDEX", "upstream"),
            ("FPP_BACKGROUND_PRIOR", "upstream"),
            ("FPP_BACKGROUND_Q", "target"),
            ("FPP_EPHEM_REFINE", ephem_refine),
            ("FPP_EXPOSURE", "upstream_2min_x20"),
            ("FPP_GEOMETRY", "upstream"),
            ("FPP_K1_NUDGE", "upstream"),
            ("FPP_NORM", "upstream"),
            ("FPP_ORBIT_PHYSICS", "upstream"),
            ("FPP_Q_BRANCH", "upstream"),
            ("FPP_SECONDARY_GEOMETRY", "upstream"),
            ("FPP_UNSTABLE_FLAG", unstable_flag),
        ]
    };
    values
        .into_iter()
        .map(|(key, value)| (key.into(), Value::String(value.into())))
        .collect()
}

fn normalized_fpp_v3_lc_source(source: Option<&str>) -> String {
    crate::fpp::v2::dispatch::normalize_lc_source(source).to_string()
}

/// F0b's placeholder primary-pool processor. It performs the *real* network
/// legs a production processor needs — `GET /api/metadata/{tic}` then
/// `GET /api/fits/{filename}` for each listed file, through the runtime's
/// `FitsCache` (proving the single-flight + LRU cache path, not just the
/// HTTP client) — but does zero science: no BLS, no vetting, no candidates.
/// This is deliberate, not an oversight: F0b's E2E harness
/// (`exonit-fake-integration`) needs a processor that exercises
/// "register -> task -> FITS -> heartbeat -> result via outbox" against the
/// real `WorkerRuntime` loop (see docs/PLAN.md's F0b gate), and F2's future
/// `BlsTaskProcessor` will need exactly this same metadata/FITS-fetch
/// scaffolding — so it is written once here and marked non-scientific in
/// the result rather than duplicated later.
///
/// A task whose metadata/FITS legs fail outright is reported `"failed"`
/// (a real, worth-surfacing problem); a task with no FITS files listed, or
/// none of the science fields filled in, is still `"completed"` — F0b has
/// no science to run, that is not a task failure.
pub struct StubTaskProcessor {
    api: Arc<WorkerApiClient>,
    fits_cache: Arc<FitsCache>,
}

impl StubTaskProcessor {
    pub fn new(api: Arc<WorkerApiClient>, fits_cache: Arc<FitsCache>) -> Self {
        StubTaskProcessor { api, fits_cache }
    }
}

impl TaskProcessor for StubTaskProcessor {
    fn process(&self, task: &WorkerTask, worker_id: &str) -> PrimaryResultRequest {
        let started = Instant::now();
        let mut result =
            PrimaryResultRequest::new(Some(task.task_id.clone()), Some(task.tic_id.clone()));

        let metadata = match self.api.fetch_metadata(&task.tic_id, Some(worker_id)) {
            Ok(metadata) => metadata,
            Err(error) => {
                result.status = "failed".to_string();
                result.error = Some(format!("metadata fetch failed: {error}"));
                result.processing_time_seconds = started.elapsed().as_secs_f64();
                return result;
            }
        };

        let mut bytes_downloaded: u64 = 0;
        for filename in &metadata.fits_files {
            match self
                .fits_cache
                .get_or_fetch(filename, || self.api.download_fits(filename))
            {
                Ok(bytes) => bytes_downloaded += bytes,
                Err(error) => {
                    result.status = "failed".to_string();
                    result.error = Some(format!("FITS fetch failed for {filename}: {error}"));
                    result.processing_time_seconds = started.elapsed().as_secs_f64();
                    return result;
                }
            }
        }

        result.status = "completed".to_string();
        result.sector_count = metadata.sectors.len() as i64;
        result.ra = metadata.ra;
        result.dec = metadata.dec;
        result.tmag = metadata.tmag;
        result.processing_time_seconds = started.elapsed().as_secs_f64();
        let mut timings = Map::new();
        timings.insert("stub_processor".to_string(), Value::Bool(true));
        timings.insert(
            "fits_bytes_downloaded".to_string(),
            Value::from(bytes_downloaded),
        );
        timings.insert(
            "fits_files_fetched".to_string(),
            Value::from(metadata.fits_files.len() as i64),
        );
        result.timings = Some(timings);
        result
    }
}

// ---------------------------------------------------------------------------
// PrimaryTaskProcessor — F2b real science pipeline.
// ---------------------------------------------------------------------------

/// Unifies every (untyped, in Swift) error source `PrimaryTaskProcessor.process`
/// can throw into one concrete type, mirrored back out as `PrimaryResultRequest`'s
/// `status: "failed"` + `error: <description>` by `TaskProcessor::process`
/// below — exactly Swift's outer `catch { return PrimaryResultRequest(...,
/// status: "failed", error: String(describing: error), ...) }`.
///
/// Faithful transliteration note: like Swift, a single candidate's CNN/
/// ExoMiner view-generation failure (`FfiView`/`Onnx` below) aborts the
/// *whole* task as failed, rather than just skipping that one candidate's
/// score — Swift's `for` loop over candidates uses unguarded `try`, so any
/// per-candidate error propagates out of `process` entirely (see
/// `PrimaryTaskProcessor.swift`'s scoring loop). Exception: a SPARSE view
/// (`FfiViewError::is_sparse`) no longer aborts the task — the CNN score is
/// left absent and ExoMiner-FFI scores 0.0, exactly as the Python worker does
/// (`cnn_ffi.py` / `exominer_ffi_inference.py` on `ViewError`). This worker
/// runs no FPP in-task (`fpp_policy::should_run` has no caller here); FPP
/// only comes from `/api/worker/fpp_task` backfill, whose SQL requires
/// `cnn_score >= 0.7 OR exominer_ffi_score >= 0.5`, so a sparse candidate
/// (cnn NULL, em_ffi 0.0) gets NO FPP on the native workers, whereas the
/// Python worker runs FPP in-task for it. Open item, folded into the FPP
/// plan's server/DDL step (exonit progress/impl-cnn-none-and-logit.md). The
/// remaining view errors need a non-positive period/duration or mismatched
/// arrays, which BLS candidates never have.
#[derive(Debug)]
enum ProcessingError {
    Api(WorkerApiError),
    Fits(FitsError),
    NoLightCurve(String),
    /// Every FITS file listed for the TIC was missing or invalid: python
    /// `_load_star` returns None (`ffi_fallback_on_load_fail` off) and the
    /// task result is `failed` / `load_failed`.
    LoadFailed,
    /// Pre-formatted message (FPP-backfill stage prefixes `metadata: ` / `load: ` / `preprocess: `).
    Staged(String),
    Preprocess(PreprocessError),
    MultiPlanet(MultiPlanetError),
    Vetting(VettingError),
    FfiView(FfiViewError),
    Onnx(OnnxFfiError),
    Injection(BlsError),
}

impl fmt::Display for ProcessingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProcessingError::Api(e) => write!(f, "{e}"),
            ProcessingError::Fits(e) => write!(f, "{e}"),
            ProcessingError::NoLightCurve(tic) => write!(f, "no usable light curve for TIC {tic}"),
            ProcessingError::LoadFailed => write!(f, "load_failed"),
            ProcessingError::Staged(message) => write!(f, "{message}"),
            ProcessingError::Preprocess(e) => write!(f, "{e}"),
            ProcessingError::MultiPlanet(e) => write!(f, "{e}"),
            ProcessingError::Vetting(e) => write!(f, "{e}"),
            ProcessingError::FfiView(e) => write!(f, "{e}"),
            ProcessingError::Onnx(e) => write!(f, "{e}"),
            ProcessingError::Injection(e) => write!(f, "{e}"),
        }
    }
}

impl From<WorkerApiError> for ProcessingError {
    fn from(value: WorkerApiError) -> Self {
        ProcessingError::Api(value)
    }
}
impl From<FitsError> for ProcessingError {
    fn from(value: FitsError) -> Self {
        ProcessingError::Fits(value)
    }
}
impl From<PreprocessError> for ProcessingError {
    fn from(value: PreprocessError) -> Self {
        ProcessingError::Preprocess(value)
    }
}
impl From<MultiPlanetError> for ProcessingError {
    fn from(value: MultiPlanetError) -> Self {
        ProcessingError::MultiPlanet(value)
    }
}
impl From<VettingError> for ProcessingError {
    fn from(value: VettingError) -> Self {
        ProcessingError::Vetting(value)
    }
}
impl From<FfiViewError> for ProcessingError {
    fn from(value: FfiViewError) -> Self {
        ProcessingError::FfiView(value)
    }
}
impl From<OnnxFfiError> for ProcessingError {
    fn from(value: OnnxFfiError) -> Self {
        ProcessingError::Onnx(value)
    }
}
impl From<BlsError> for ProcessingError {
    fn from(value: BlsError) -> Self {
        ProcessingError::Injection(value)
    }
}

/// Mirrors `PrimaryTaskProcessor` (`Sources/ExoNitCore/PrimaryTaskProcessor.swift`).
/// CNN/ExoMiner FFI classifiers are optional: constructed once at startup
/// from `models_dir` (see `new`), `None` when the models directory isn't
/// configured, a model file is missing, or it fails to load — matching the
/// Python worker's real production behaviour (`worker/cnn_ffi.py`'s
/// `FileNotFoundError` -> `classifier_ffi = None` -> "FFI rows will use SPOC
/// classifier (fallback)", `worker/exominer_ffi_inference.py` analogously)
/// rather than Swift's `main.swift` CLI wiring, which always resolves a
/// non-nil model URL and so throws `missingRequiredModel` at startup if the
/// file isn't there (dead code in practice: the Swift release process always
/// ships `Models/`, but not a safe default for this crate to copy — see
/// `docs/PLAN.md`'s F2b session note). `process()` itself already treats
/// `cnn`/`exominer` being absent as "skip that score, don't fail the task",
/// same as Swift's `if let cnn { ... }` — this constructor just extends that
/// same graceful-degradation posture to model *loading*, not only to model
/// *absence-by-design*.
pub struct PrimaryTaskProcessor {
    api: Arc<WorkerApiClient>,
    fits_cache: Arc<FitsCache>,
    cnn: Option<CnnFfiClassifier>,
    exominer: Option<ExoMinerFfiClassifier>,
    max_concurrency: usize,
    compute_backend: ComputeBackend,
    gpu_used: bool,
    /// F4b: injection-recovery's own CUDA batch backend — gated the same
    /// way as `compute_backend` (`config.gpu_enabled` + probe), but a
    /// SEPARATE probe/backend instance (`bls::cuda_batch::CudaBlsBatchBackend`
    /// vs `bls::cuda::CudaBlsBackend`): different kernels, same "probed once
    /// at construction" contract. See `resolve_injection_backend`'s doc
    /// comment for why this mirrors `injection.py`'s `disable_gpu()` gate
    /// rather than `wotan_detrend.py`'s independent env-var gate.
    injection_backend: InjectionBackend,
    injection_gpu_used: bool,
    /// F-opencl: which concrete backend `resolve_injection_backend` actually
    /// resolved to (`"rust-cuda"`/`"rust-opencl"`/`"rust-cpu"`) — read by
    /// `runtime.rs`'s `build_capabilities` for the `injection_backend`
    /// capability field, replacing the previous `injection_gpu_used`-derived
    /// two-way `"rust-cuda"`/`"rust-cpu"` ternary (which had no way to name
    /// OpenCL).
    injection_backend_label: &'static str,
    /// F4b: wotan biweight detrend's CUDA backend — gated by
    /// `EXONIT_WOTAN_BACKEND`, NOT `config.gpu_enabled` (see
    /// `resolve_detrend_backend`'s doc comment; this is a real,
    /// deliberate divergence from the BLS/injection gate, mirroring
    /// Python's own divergent gating for this one backend).
    detrend_backend: DetrendBackend,
    detrend_gpu_used: bool,
    /// Worker-config `gap_fill_detrend_only` (default false). A task-config
    /// key of the same name wins for primary tasks; FPP-backfill tasks carry
    /// no config, so this is what makes backfill FPP and in-task FPP see the
    /// same light curve (python `ExoNitWorker._gap_fill_detrend_only_enabled`).
    gap_fill_detrend_only_default: bool,
    /// Worker-config `harmonic_epoch_class` (default false); a JSON bool in
    /// the task config wins (see `harmonic_epoch_class_for`).
    harmonic_epoch_class_default: bool,
    /// 0 = no task yet, 1 = last task ran OFF, 2 = last task ran ON.
    last_epoch_class: std::sync::atomic::AtomicU8,
    /// python `ffi_fallback_on_load_fail` / `ffi_fetch_timeout` (worker config).
    ffi_fallback_on_load_fail: bool,
    ffi_fetch_timeout: f64,
}

/// Light curve as loaded for one task: metadata, photometry source and the
/// preprocessed arrays (see `ProcessedLightCurve::detrend_frame` for the
/// `gap_fill_detrend_only` semantics).
pub struct LoadedLightCurve {
    pub metadata: StarMetadata,
    pub source: LightCurveSource,
    /// Product detected from the files actually loaded for FPP. An FFI
    /// endpoint response has no local product path and is therefore unknown.
    pub fpp_lc_source: String,
    pub processed: ProcessedLightCurve,
    /// python `lc_load_recovered`: this light curve came from the FFI fallback
    /// (`try_ffi_fallback`), not from the listed FITS files.
    pub recovered: bool,
    /// `n_sectors` of the result. python `_load_star` reports the number of
    /// sectors in the server metadata even when some FITS files were skipped
    /// (missing/invalid); with no file skipped it equals `processed.sector_count`.
    pub sector_count: usize,
}

/// Light curves read from the FITS files that could be obtained (python
/// `_load_star` step 2 skip semantics).
struct LoadedCurves {
    curves: Vec<LightCurveData>,
    used_files: Vec<String>,
    skipped_files: Vec<String>,
}

/// Upper bound accepted for `ffi_fetch_timeout` (one day).
const MAX_FFI_FETCH_TIMEOUT_SECONDS: f64 = 86_400.0;

/// python `fits_cache_helper.ensure_fits_present` attempt count.
const FITS_DOWNLOAD_ATTEMPTS: usize = 3;

fn fpp_lc_source_from_files(files: &[String]) -> String {
    if files.is_empty() {
        return "unknown".into();
    }
    let mut found: Option<&str> = None;
    for file in files {
        let basename = file
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or(file)
            .to_ascii_lowercase();
        let kind = if basename.starts_with("hlsp_tglc") {
            "tglc"
        } else if basename.starts_with("hlsp_qlp") {
            "qlp"
        } else if basename.ends_with("_lc.fits") && !basename.contains("hlsp") {
            "spoc"
        } else {
            return "unknown".into();
        };
        if found.is_some_and(|previous| previous != kind) {
            return "mixed".into();
        }
        found = Some(kind);
    }
    found.unwrap_or("unknown").into()
}

impl PrimaryTaskProcessor {
    pub fn new(
        api: Arc<WorkerApiClient>,
        fits_cache: Arc<FitsCache>,
        models_dir: Option<PathBuf>,
        max_concurrency: usize,
        gpu_enabled: bool,
    ) -> Self {
        let (cnn, exominer) = match &models_dir {
            Some(dir) => (load_cnn(dir), load_exominer(dir)),
            None => (None, None),
        };
        let (compute_backend, gpu_used) = resolve_compute_backend(gpu_enabled);
        let (injection_backend, injection_gpu_used, injection_backend_label) =
            resolve_injection_backend(gpu_enabled);
        let (detrend_backend, detrend_gpu_used) = resolve_detrend_backend();
        PrimaryTaskProcessor {
            api,
            fits_cache,
            cnn,
            exominer,
            max_concurrency: max_concurrency.max(1),
            compute_backend,
            gpu_used,
            injection_backend,
            injection_gpu_used,
            injection_backend_label,
            detrend_backend,
            detrend_gpu_used,
            gap_fill_detrend_only_default: false,
            harmonic_epoch_class_default: false,
            last_epoch_class: std::sync::atomic::AtomicU8::new(0),
            ffi_fallback_on_load_fail: false,
            ffi_fetch_timeout: 30.0,
        }
    }

    /// Worker-config `ffi_fallback_on_load_fail` and `ffi_fetch_timeout` (seconds).
    pub fn with_ffi_fallback(mut self, enabled: bool, fetch_timeout_seconds: f64) -> Self {
        self.ffi_fallback_on_load_fail = enabled;
        // Duration::from_secs_f64 panics on NaN / negative / huge values: only a finite
        // 0 < t <= 1 day is accepted, anything else falls back to python's 30 s.
        self.ffi_fetch_timeout = if fetch_timeout_seconds.is_finite()
            && fetch_timeout_seconds > 0.0
            && fetch_timeout_seconds <= MAX_FFI_FETCH_TIMEOUT_SECONDS
        {
            fetch_timeout_seconds
        } else {
            eprintln!(
                "PrimaryTaskProcessor: ffi_fetch_timeout={fetch_timeout_seconds} is not a finite number in \
                 (0, {MAX_FFI_FETCH_TIMEOUT_SECONDS}] seconds; using 30 s"
            );
            30.0
        };
        self
    }

    /// Effective FFI-fallback fetch timeout in seconds (after validation).
    pub fn ffi_fetch_timeout_seconds(&self) -> f64 {
        self.ffi_fetch_timeout
    }

    /// python `_try_ffi_fallback` (spec 2026-05-08-ffi-fallback-on-lc-load-failure):
    /// last resort when the FITS light curve cannot be loaded. Only with the
    /// flag on. Fetches `/api/ffi/{tic}` (no retries, `ffi_fetch_timeout`),
    /// drops non-finite samples, requires >= 100 finite points, then builds
    /// the light curve EXACTLY like python: NO gap fill and NO detrend (the
    /// FFI cache series is used as is: `flat_flux = raw_flux = flux`), minimal
    /// metadata (tic_id, data_source "ffi", ra/dec/tmag/sectors from a
    /// best-effort 5 s metadata refetch; stellar parameters stay absent),
    /// product tag "unknown", `n_sectors = len(sectors) or 1`,
    /// `lc_load_recovered = true`. `None` = the fallback also failed (python:
    /// the caller then fails the task).
    fn try_ffi_fallback(&self, tic_id: &str, worker_id: &str, reason: &str) -> Option<LoadedLightCurve> {
        if !self.ffi_fallback_on_load_fail {
            return None;
        }
        let timeout = std::time::Duration::try_from_secs_f64(self.ffi_fetch_timeout)
            .unwrap_or(std::time::Duration::from_secs(30));
        let Some((time, flux)) = self.api.fetch_ffi_for_fallback(tic_id, timeout) else {
            eprintln!(
                "PrimaryTaskProcessor: TIC {tic_id}: SPOC load failed ({reason}) -> FFI fallback also failed (server 404)"
            );
            return None;
        };
        let (time, flux): (Vec<f64>, Vec<f64>) =
            time.iter().zip(&flux).filter(|(t, f)| t.is_finite() && f.is_finite()).map(|(t, f)| (*t, *f)).unzip();
        if time.len() < 100 {
            eprintln!("PrimaryTaskProcessor: TIC {tic_id}: FFI fallback rejected - only {} finite points", time.len());
            return None;
        }
        let mut metadata = StarMetadata {
            tic_id: Some(tic_id.to_string()),
            fits_files: Vec::new(),
            sectors: Vec::new(),
            ra: None,
            dec: None,
            tmag: None,
            effective_temperature: None,
            logg: None,
            stellar_radius: None,
            stellar_mass: None,
            data_source: Some("ffi".to_string()),
        };
        if let Ok(refetched) =
            self.api.fetch_metadata_with_timeout(tic_id, Some(worker_id), std::time::Duration::from_secs(5))
        {
            metadata.ra = refetched.ra;
            metadata.dec = refetched.dec;
            metadata.tmag = refetched.tmag;
            metadata.sectors = refetched.sectors;
        }
        let n_sectors = if metadata.sectors.is_empty() { 1 } else { metadata.sectors.len() };
        let max_gap_days = time.windows(2).map(|w| w[1] - w[0]).fold(0.0_f64, f64::max);
        eprintln!(
            "PrimaryTaskProcessor: TIC {tic_id}: SPOC load failed ({reason}) -> FFI fallback OK \
             ({n_sectors} sectors, {} points, source=ffi)",
            time.len()
        );
        let processed = ProcessedLightCurve {
            raw_flux: flux.clone(),
            flat_flux: flux,
            time,
            sector_count: n_sectors,
            max_gap_days,
            detrend_frame: None,
        };
        Some(LoadedLightCurve {
            metadata,
            source: LightCurveSource::Ffi,
            fpp_lc_source: "unknown".to_string(),
            processed,
            recovered: true,
            sector_count: n_sectors,
        })
    }

    /// python `_load_star`: metadata -> FITS (with `ensure_fits_present` skip
    /// semantics) -> preprocess, with the FFI fallback at every point python
    /// has one: metadata unavailable (`metadata_http`/`metadata_exception`),
    /// no FITS listed (`no_fits_files`), no FITS obtainable
    /// (`fits_download_fail`), and ANY failure of python's read+stitch+clean+
    /// detrend `try` block (`lightkurve_exception`: unreadable FITS, empty
    /// light curve after cleaning, any preprocess error).
    /// With the flag OFF nothing changes versus the pre-fallback behaviour.
    ///
    /// `fpp` = the FPP-backfill caller, whose errors keep their historical
    /// stage prefixes (`metadata: `, `load: `, `preprocess: `).
    fn load_star(
        &self,
        tic_id: &str,
        worker_id: &str,
        iterative_detrend: bool,
        gap_fill_detrend_only: bool,
        fpp: bool,
    ) -> Result<LoadedLightCurve, ProcessingError> {
        let staged = |stage: &str, error: ProcessingError| -> ProcessingError {
            if fpp {
                ProcessingError::Staged(format!("{stage}: {error}"))
            } else {
                error
            }
        };
        let metadata = match self.api.fetch_metadata(tic_id, Some(worker_id)) {
            Ok(metadata) => metadata,
            Err(error) => {
                if let Some(recovered) = self.try_ffi_fallback(tic_id, worker_id, "metadata") {
                    return Ok(recovered);
                }
                return Err(staged("metadata", error.into()));
            }
        };
        if self.ffi_fallback_on_load_fail && metadata.fits_files.is_empty() {
            return self
                .try_ffi_fallback(tic_id, worker_id, "no_fits_files")
                .ok_or_else(|| staged("load", ProcessingError::LoadFailed));
        }
        let loaded_curves = match self.load_curves(tic_id, &metadata) {
            Ok(loaded) => loaded,
            Err(error @ ProcessingError::Fits(_)) => {
                if let Some(recovered) = self.try_ffi_fallback(tic_id, worker_id, "lightkurve_exception") {
                    return Ok(recovered);
                }
                return Err(staged("load", error));
            }
            Err(error) => return Err(staged("load", error)),
        };
        if loaded_curves.curves.is_empty() && !metadata.fits_files.is_empty() {
            eprintln!("PrimaryTaskProcessor: No FITS files downloaded for TIC {tic_id}");
            return self
                .try_ffi_fallback(tic_id, worker_id, "fits_download_fail")
                .ok_or_else(|| staged("load", ProcessingError::LoadFailed));
        }
        if fpp && loaded_curves.curves.is_empty() {
            return Err(ProcessingError::Staged(format!("no light curve data for TIC {tic_id}")));
        }
        self.require_curves(tic_id, &metadata, &loaded_curves)?;
        let curves = &loaded_curves.curves;
        let processed =
            match self.preprocess_curves(tic_id, &metadata, curves, iterative_detrend, gap_fill_detrend_only) {
                Ok(processed) => processed,
                Err(error) => {
                    // python wraps read+stitch+clean+detrend in one `try ... except Exception`
                    // (reason `lightkurve_exception`), e.g. an empty light curve after cleaning.
                    if let Some(recovered) = self.try_ffi_fallback(tic_id, worker_id, "lightkurve_exception") {
                        return Ok(recovered);
                    }
                    return Err(staged("preprocess", error.into()));
                }
            };
        let source = if curves.iter().any(|c| c.source == PhotometrySource::Spoc) {
            LightCurveSource::Spoc
        } else {
            LightCurveSource::Ffi
        };
        // python tags the product from the files actually loaded
        let fpp_lc_source = fpp_lc_source_from_files(&loaded_curves.used_files);
        let sector_count = if loaded_curves.skipped_files.is_empty() {
            processed.sector_count
        } else {
            metadata.sectors.len()
        };
        Ok(LoadedLightCurve { metadata, source, fpp_lc_source, processed, recovered: false, sector_count })
    }

    /// Search options for one task: the config-derived options with
    /// `harmonic_epoch_class` resolved by `harmonic_epoch_class_for`; also
    /// returns the effective flag (reported in the result).
    pub fn multi_planet_options_for_task(&self, task_config: &Map<String, Value>) -> (MultiPlanetOptions, bool) {
        let mut options = multi_planet_options_from_config(task_config);
        let epoch_class = self.harmonic_epoch_class_for(Some(task_config));
        options.harmonic_epoch_class = epoch_class;
        (options, epoch_class)
    }

    /// Worker-config `harmonic_epoch_class` (see the field doc comment).
    pub fn with_harmonic_epoch_class(mut self, enabled: bool) -> Self {
        self.harmonic_epoch_class_default = enabled;
        self
    }

    /// `harmonic_epoch_class` precedence (python `process_task`): a JSON bool
    /// in the task config wins, else the worker config, else false. A
    /// non-bool task value ("true", 1, null, ...) is a protocol error: warn
    /// and use the worker value, never truthy-coerce "false".
    pub fn harmonic_epoch_class_for(&self, task_config: Option<&Map<String, Value>>) -> bool {
        match task_config.and_then(|config| config.get("harmonic_epoch_class")) {
            Some(Value::Bool(value)) => *value,
            Some(other) => {
                eprintln!(
                    "PrimaryTaskProcessor: task config harmonic_epoch_class={other} is not a bool; \
                     ignored, using local config ({})",
                    self.harmonic_epoch_class_default
                );
                self.harmonic_epoch_class_default
            }
            None => self.harmonic_epoch_class_default,
        }
    }

    /// Worker-config `gap_fill_detrend_only` (see the field doc comment).
    pub fn with_gap_fill_detrend_only(mut self, enabled: bool) -> Self {
        self.gap_fill_detrend_only_default = enabled;
        self
    }

    /// `gap_fill_detrend_only` precedence: task-config key if it is a JSON
    /// bool, else the worker config, else false. A non-bool task value
    /// ("true", 1, null, ...) is ignored with a warning (python / Swift parity).
    pub fn gap_fill_detrend_only_for(&self, task_config: Option<&Map<String, Value>>) -> bool {
        match task_config.and_then(|config| config.get("gap_fill_detrend_only")) {
            Some(Value::Bool(value)) => *value,
            Some(other) => {
                eprintln!(
                    "PrimaryTaskProcessor: task config gap_fill_detrend_only={other} is not a JSON bool; \
                     ignored (worker value {})",
                    self.gap_fill_detrend_only_default
                );
                self.gap_fill_detrend_only_default
            }
            None => self.gap_fill_detrend_only_default,
        }
    }

    /// Shared tail of both load paths: seeded preprocess with the resolved
    /// `gap_fill_detrend_only` value.
    fn preprocess_curves(
        &self,
        tic_id: &str,
        metadata: &StarMetadata,
        curves: &[LightCurveData],
        iterative_detrend: bool,
        gap_fill_detrend_only: bool,
    ) -> Result<ProcessedLightCurve, PreprocessError> {
        // F7: same `max_concurrency` cap (config.cpu_threads-derived) as the
        // primary multi-planet BLS search — quick-BLS is no longer forced
        // single-threaded (see `RealQuickBlsSearch`'s doc comment).
        let quick_search = RealQuickBlsSearch::new(self.max_concurrency);
        let mode = if iterative_detrend {
            DetrendMode::Iterative(&quick_search)
        } else {
            DetrendMode::Biweight
        };
        // F4b: routed through the resolved wotan detrend backend
        // (`EXONIT_WOTAN_BACKEND` — see `resolve_detrend_backend`'s doc
        // comment) instead of the always-CPU `preprocess::process`.
        let fill_seed = crate::gaps::gap_fill_seed(tic_id, &metadata.sectors);
        // `gap_fill_detrend_only` (default off = bit-identical): gap-fill
        // points feed the detrender only; the result then holds real points
        // only and the re-detrends go through `detrend_frame` (python
        // worker.py `_load_star`/`process_task`, gap_fill.DetrendFrame).
        preprocess::process_with_backend_seeded_opts(
            curves,
            mode,
            &self.detrend_backend,
            fill_seed,
            gap_fill_detrend_only,
        )
    }

    /// Primary-task light curve (`iterative_detrend` and
    /// `gap_fill_detrend_only` resolved from the task config / worker config).
    pub fn load_primary_light_curve(
        &self,
        task: &WorkerTask,
        worker_id: &str,
    ) -> Result<LoadedLightCurve, String> {
        self.load_primary_light_curve_inner(task, worker_id)
            .map_err(|e| e.to_string())
    }

    fn load_primary_light_curve_inner(
        &self,
        task: &WorkerTask,
        worker_id: &str,
    ) -> Result<LoadedLightCurve, ProcessingError> {
        let iterative_detrend = config_bool(&task.config, "iterative_detrend", true);
        let gap_fill_detrend_only = self.gap_fill_detrend_only_for(Some(&task.config));
        self.load_star(&task.tic_id, worker_id, iterative_detrend, gap_fill_detrend_only, false)
    }

    /// FPP-backfill light curve: iterative detrend always (python
    /// `_fetch_and_load_fpp` -> `_load_star`), `gap_fill_detrend_only` from
    /// the worker config (backfill tasks carry no config).
    pub fn load_fpp_light_curve(
        &self,
        task: &crate::dto::BackfillTask,
        worker_id: &str,
    ) -> Result<LoadedLightCurve, String> {
        self.load_star(&task.tic_id, worker_id, true, self.gap_fill_detrend_only_for(None), true)
            .map_err(|e| e.to_string())
    }

    pub fn cnn_available(&self) -> bool {
        self.cnn.is_some()
    }

    pub fn exominer_available(&self) -> bool {
        self.exominer.is_some()
    }

    /// Whether the BLS backend resolved to a real GPU (F4a: CUDA) at
    /// construction time — see `resolve_compute_backend`'s doc comment.
    /// Read by `runtime.rs`'s `build_capabilities` for the worker's
    /// self-reported `bls_backend`/`gpu_available` capabilities, honestly
    /// this time (previously hardcoded `false`, F2b-era comment, now stale).
    pub fn gpu_used(&self) -> bool {
        self.gpu_used
    }

    /// Whether injection-recovery's CUDA batch backend resolved at
    /// construction time (F4b) — see `resolve_injection_backend`'s doc
    /// comment. Read by `runtime.rs`'s `build_capabilities` for the
    /// `injection_backend` capability, replacing the previous hardcoded
    /// `"rust-cpu"`.
    pub fn injection_gpu_used(&self) -> bool {
        self.injection_gpu_used
    }

    /// Which concrete injection-recovery backend actually resolved
    /// (`"rust-cuda"`/`"rust-opencl"`/`"rust-cpu"`) — see the
    /// `injection_backend_label` field's doc comment.
    pub fn injection_backend_label(&self) -> &'static str {
        self.injection_backend_label
    }

    /// Whether the wotan biweight detrend CUDA backend resolved at
    /// construction time (F4b) — see `resolve_detrend_backend`'s doc
    /// comment.
    pub fn detrend_gpu_used(&self) -> bool {
        self.detrend_gpu_used
    }

    /// Mirrors python `fits_cache_helper.ensure_fits_present`: a cached file is
    /// trusted only after validation (invalid -> purged); then up to 3
    /// download+validate attempts, where ANY failure (HTTP 404/5xx, transport
    /// error, invalid file) just burns an attempt. Returns false when all
    /// attempts failed (nothing left on disk) so the caller skips the file.
    fn ensure_fits_present(&self, filename: &str) -> bool {
        let path = self.fits_cache.directory().join(filename);
        if path.exists() {
            match fits::invalid_reason(&path) {
                None => return true,
                Some(reason) => {
                    eprintln!("PrimaryTaskProcessor: Purging invalid cached FITS {filename}: {reason}");
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        for attempt in 1..=FITS_DOWNLOAD_ATTEMPTS {
            let api = &self.api;
            match self.fits_cache.get_or_fetch(filename, || api.download_fits(filename)) {
                Err(error) => {
                    eprintln!(
                        "PrimaryTaskProcessor: Download attempt {attempt}/{FITS_DOWNLOAD_ATTEMPTS} for \
                         {filename} raised {error}"
                    );
                    let _ = std::fs::remove_file(&path);
                    continue;
                }
                Ok(_) => {}
            }
            match fits::invalid_reason(&path) {
                None => return true,
                Some(reason) => {
                    eprintln!(
                        "PrimaryTaskProcessor: Invalid FITS after download (attempt \
                         {attempt}/{FITS_DOWNLOAD_ATTEMPTS}) {filename}: {reason}"
                    );
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        eprintln!("PrimaryTaskProcessor: Giving up on {filename} after {FITS_DOWNLOAD_ATTEMPTS} attempts");
        false
    }

    /// Mirrors `PrimaryTaskProcessor.loadCurves(ticID:metadata:)`: FITS files
    /// listed in metadata take priority (downloaded through the shared
    /// `FitsCache`, single-flighted/LRU'd — see `runtime.rs`) with python's
    /// skip semantics (`ensure_fits_present`); with none listed, falls back to
    /// the FFI endpoint's already-decoded sectors.
    fn load_curves(
        &self,
        tic_id: &str,
        metadata: &StarMetadata,
    ) -> Result<LoadedCurves, ProcessingError> {
        if !metadata.fits_files.is_empty() {
            let mut result = LoadedCurves {
                curves: Vec::with_capacity(metadata.fits_files.len()),
                used_files: Vec::new(),
                skipped_files: Vec::new(),
            };
            for filename in &metadata.fits_files {
                if !self.ensure_fits_present(filename) {
                    result.skipped_files.push(filename.clone());
                    continue;
                }
                let path = self.fits_cache.directory().join(filename);
                result.curves.push(fits::read_light_curve(&path)?);
                result.used_files.push(filename.clone());
            }
            return Ok(result);
        }
        let curves = match self.api.fetch_ffi(tic_id)? {
            None => Vec::new(),
            Some(response) => response
                .sectors
                .into_iter()
                .map(|sector| {
                    let source = if sector.source.to_ascii_lowercase().contains("tglc") {
                        PhotometrySource::Tglc
                    } else {
                        PhotometrySource::Qlp
                    };
                    let point_count = sector.time.len();
                    LightCurveData {
                        time: sector.time,
                        flux: sector.flux,
                        raw_row_count: point_count,
                        cadence: None,
                        source,
                    }
                })
                .collect(),
        };
        Ok(LoadedCurves { curves, used_files: Vec::new(), skipped_files: Vec::new() })
    }

    /// Shared tail of both load paths: all listed files unobtainable is
    /// python's `load_failed`; no FITS listed and no FFI data is `NoLightCurve`.
    fn require_curves(
        &self,
        tic_id: &str,
        metadata: &StarMetadata,
        loaded: &LoadedCurves,
    ) -> Result<(), ProcessingError> {
        if !loaded.curves.is_empty() {
            return Ok(());
        }
        if !metadata.fits_files.is_empty() {
            eprintln!("PrimaryTaskProcessor: No FITS files downloaded for TIC {tic_id}");
            return Err(ProcessingError::LoadFailed);
        }
        Err(ProcessingError::NoLightCurve(tic_id.to_string()))
    }

    /// Mirrors `PrimaryTaskProcessor.process(_:workerID:)`'s `do` block; the
    /// `catch` half lives in `TaskProcessor::process` below.
    fn process_inner(
        &self,
        task: &WorkerTask,
        worker_id: &str,
    ) -> Result<PrimaryResultRequest, ProcessingError> {
        // Alias phase 2: task bool > worker config > off; reported back in the
        // result and (from the moment the task starts, like python) in the
        // heartbeat `flag_states`.
        let (options, epoch_class) = self.multi_planet_options_for_task(&task.config);
        self.last_epoch_class
            .store(if epoch_class { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
        // `gap_fill_detrend_only`: task config > worker config > off (see
        // `gap_fill_detrend_only_for`); `processed` holds real points only
        // when it is on and the gap fill added points.
        let LoadedLightCurve {
            metadata,
            source,
            processed,
            sector_count: loaded_sector_count,
            recovered,
            ..
        } = self.load_primary_light_curve_inner(task, worker_id)?;


        let search = multi_planet::search_with_frame(
            &processed.time,
            &processed.raw_flux,
            &processed.flat_flux,
            metadata.stellar_radius,
            metadata.stellar_mass,
            source,
            &options,
            self.max_concurrency,
            // F4a: CUDA when `config.gpu_enabled` and a GPU actually
            // initialized at construction time (`resolve_compute_backend`);
            // CPU otherwise, with the fallback already logged there. This
            // is a `Clone` (cheap: `Cpu`/`Automatic` are unit variants,
            // `Cuda` clones an `Arc`), not a move, since `self` is reused
            // for every task this processor handles.
            self.compute_backend.clone(),
            processed.detrend_frame.as_ref(),
        )?;
        let mut candidates = search.candidates;

        let vetting = vetting::vet_all(&candidates, &processed.time, &processed.raw_flux)?;

        let mut cnn_scores: Vec<Option<f64>> = vec![None; candidates.len()];
        for index in 0..candidates.len() {
            let (period, t0, duration, depth_ppm, snr) = {
                let c = &candidates[index];
                (c.period_days, c.t0, c.duration_days, c.depth_ppm, c.snr)
        };
            if let Some(cnn) = &self.cnn {
                // Sparse view -> no score (None), task continues (python
                // cnn_ffi.py since 2026-10-15). No FPP follows on this worker:
                // backfill SQL excludes cnn NULL + em_ffi 0.0 (see the
                // ProcessingError doc comment; open item).
                let views = ffi_views::make_cnn_views_if_dense(
                    &processed.time,
                    &processed.flat_flux,
                    period,
                    t0,
                    duration,
                )?;
                if let Some(views) = views {
                    cnn_scores[index] = Some(cnn.classify(&views)?.probability);
                }
            }
            if let Some(exominer) = &self.exominer {
                let stellar = FfiStellarMetadata {
                    tmag: metadata.tmag,
                    teff: metadata.effective_temperature,
                    logg: metadata.logg,
                    radius: metadata.stellar_radius,
                    mass: metadata.stellar_mass,
                };
                let features = match exominer_features::make(
                    &processed.time,
                    &processed.flat_flux,
                    period,
                    t0,
                    duration,
                    depth_ppm,
                    snr,
                    stellar,
                ) {
                    Ok(features) => Some(features),
                    Err(error) if error.is_sparse() => None,
                    Err(error) => return Err(error.into()),
                };
                // Same sparse view as the CNN: python exominer_ffi_inference.py
                // scores it 0.0 (uncalibrated) instead of failing the task.
                // OPEN ITEM (exonit docs/reviews/2026-10-15-cnn-exact-zero-
                // scores.md §7.3): this 0.0 is a sentinel indistinguishable from
                // a real low score; mirrored here for parity until python
                // decides on None (it feeds the backfill OR-gate + verdict).
                let probability = match features {
                    Some(features) => exominer.classify(&features)?.probability,
                    None => 0.0,
                };
                candidates[index].exominer_ffi_score = Some(probability);
                candidates[index].exominer_ffi_model_version =
                    Some("phase3_v6_ensemble_cal".to_string());
            }
        }

        let run_injection = config_bool(
            &task.config,
            "run_injection",
            config_bool(&task.config, "run_injection_recovery", true),
        );
        let completeness = if run_injection {
            let injection = InjectionRecoveryOptions {
                stellar_radius_solar: metadata.stellar_radius.unwrap_or(1.0),
                stellar_mass_solar: metadata.stellar_mass,
                injections_per_cell: config_int(&task.config, "n_injections_per_cell", 10) as usize,
                minimum_snr: options.minimum_snr,
                period_count: config_int(&task.config, "injection_n_periods", 5_000) as usize,
                bin_count: config_int(&task.config, "injection_n_bins", 150) as usize,
                duration_step_count: config_int(&task.config, "injection_n_dur_steps", 10) as usize,
                maximum_seconds: Some(config_double(&task.config, "injection_max_seconds", 300.0)),
                ..InjectionRecoveryOptions::default()
            };
            Some(injection_recovery::run_with_backend(
                &processed.time,
                // Mirrors `PrimaryTaskProcessor.injectionFlux(from:)` ==
                // `processed.flatFlux`.
                &processed.flat_flux,
                &injection,
                self.max_concurrency,
                || false,
                &self.injection_backend,
            )?)
        } else {
            None
        };

        let span = processed.time.last().copied().unwrap_or(0.0)
            - processed.time.first().copied().unwrap_or(0.0);
        let mut timings = Map::new();
        timings.insert("rust_native".to_string(), Value::Bool(true));
        timings.insert(
            "redetrend_failures".to_string(),
            Value::from(search.redetrend_failure_count as i64),
        );
        timings.insert(
            "unphysical_rejections".to_string(),
            Value::from(search.rejected_unphysical_count as i64),
        );

        Ok(PrimaryResultRequest {
            task_id: Some(task.task_id.clone()),
            tic_id: Some(task.tic_id.clone()),
            candidates,
            vetting_results: vetting,
            cnn_scores,
            completeness,
            processing_time_seconds: 0.0, // filled in by `TaskProcessor::process`
            // Honest per CLAUDE.md: reflects whether `resolve_compute_backend`
            // actually resolved to CUDA at construction time (F4a), not
            // just whether `config.gpu_enabled` was requested — mirrors
            // `worker.py`'s `"gpu_used": self.gpu_enabled` field EXCEPT
            // that Python conflates "configured" with "used" here (a known
            // Python imprecision this port deliberately does not copy: see
            // `resolve_compute_backend`'s doc comment).
            gpu_used: self.gpu_used,
            sector_count: loaded_sector_count as i64,
            point_count: processed.time.len() as i64,
            time_span_days: Some(span),
            ra: metadata.ra,
            dec: metadata.dec,
            tmag: metadata.tmag,
            timings: Some(timings),
            light_curve_load_recovered: recovered,
            status: "completed".to_string(),
            error: None,
            harmonic_epoch_class: Some(epoch_class),
        })
    }
}

impl PrimaryTaskProcessor {
    /// Run corrected source-aware FPP. Tasks without its versioned population
    /// inputs are released in `process_fpp`; they never fall back to the
    /// legacy scenario model.
    fn process_fpp_v3_inner(
        &self,
        task: &crate::dto::BackfillTask,
        worker_id: &str,
    ) -> Result<
        (
            crate::fpp::v2::calc_v3::CalcV3Result,
            String,
            Option<crate::fpp::ephem::EphemProvenance>,
            bool,
            bool,
        ),
        crate::fpp::FppError,
    > {
        // Validate the strict switches before any network or compute work so an
        // invalid FPP_EPHEM_REFINE / FPP_UNSTABLE_FLAG fails fast with a typed error.
        let ephem_refine_on = parse_ephem_refine_switch()?;
        let _unstable_flag_on = parse_unstable_flag_switch()?;
        let loaded = self
            .load_fpp_light_curve(task, worker_id)
            .map_err(crate::fpp::FppError::failed)?;
        let nearby_raw = self
            .api
            .fetch_nearby_value(&task.tic_id)
            .map_err(|e| crate::fpp::FppError::failed(format!("nearby: {e}")))?;
        let ephem_res = crate::fpp::ephem::fpp_ephemeris(
            &loaded.processed.time,
            &loaded.processed.flat_flux,
            task.period_days,
            task.t0,
            task.duration_days,
            crate::fpp::ephem::T0Convention::BoxStart,
            ephem_refine_on,
            task.cadence_s,
        );
        let seed = crate::fpp::rng::tic_seed(&task.tic_id, task.period_days, task.t0);
        let seed_ephemeris = if ephem_res.provenance.is_some() {
            Some((task.period_days, task.t0))
        } else {
            None
        };
        let input = crate::fpp::v2::calc_v3::CalcV3Inputs {
            tic_id: &task.tic_id,
            period: ephem_res.p_use,
            t0: ephem_res.t0_use,
            duration: task.duration_days,
            depth_ppm: task.depth_ppm,
            time: &loaded.processed.time,
            flux: &loaded.processed.flat_flux,
            nearby_raw: &nearby_raw,
            geometry: task.geometry.as_ref(),
            population: task.population.as_ref(),
            population_sha256: task.population_sha256.as_deref(),
            population_version: task.population_version.as_deref(),
            population_crosscheck: task.population_crosscheck.as_ref(),
            cadence_s: task.cadence_s,
            lc_source: Some(&loaded.fpp_lc_source),
            tglc_near_edge: task.tglc_near_edge,
            seed_ephemeris,
        };
        // The Python worker initializes one complete switch set from
        // FPP_PARITY; per-switch overrides are only applied by its explicit
        // set_fpp_switches(parity=None) API, which this worker does not call.
        // Match that default worker entrypoint exactly.
        let parity = std::env::var("FPP_PARITY").is_ok_and(|v| v == "1");
        let production_fold = !parity;
        let result = crate::bls::with_pool(self.max_concurrency, || {
            crate::fpp::v2::calc_v3::calc_fpp_by_source_v3(&input, 100_000, seed, production_fold)
        })?;
        Ok((
            result,
            loaded.fpp_lc_source,
            ephem_res.provenance,
            ephem_res.refine_failed,
            ephem_refine_on,
        ))
    }
}

impl TaskProcessor for PrimaryTaskProcessor {
    fn last_harmonic_epoch_class(&self) -> Option<bool> {
        match self.last_epoch_class.load(std::sync::atomic::Ordering::Relaxed) {
            0 => None,
            value => Some(value == 2),
        }
    }

    fn process(&self, task: &WorkerTask, worker_id: &str) -> PrimaryResultRequest {
        let started = Instant::now();
        match self.process_inner(task, worker_id) {
            Ok(mut result) => {
                result.processing_time_seconds = started.elapsed().as_secs_f64();
                result
            }
            Err(error) => {
                let mut result = PrimaryResultRequest::new(
                    Some(task.task_id.clone()),
                    Some(task.tic_id.clone()),
                );
                result.status = "failed".to_string();
                result.error = Some(error.to_string());
                result.processing_time_seconds = started.elapsed().as_secs_f64();
                result
            }
        }
    }

    fn process_fpp(
        &self,
        task: &crate::dto::BackfillTask,
        raw_task: &Map<String, Value>,
        worker_id: &str,
    ) -> Option<FppProcessOutcome> {
        let mut req = crate::dto::FppResultRequest::new(
            Some(task.candidate_id),
            Some(task.tic_id.clone()),
            None,
        );
        req.candidate = Some(raw_task.clone());
        let has_v3_keys = [
            "fpp_model_version",
            "population",
            "population_sha256",
            "population_version",
            "population_crosscheck",
            "geometry",
            "cadence_s",
            "lc_source",
            "tglc_near_edge",
        ]
        .iter()
        .any(|key| raw_task.contains_key(*key));
        let release_reason = if !has_v3_keys
            || raw_task.get("population_missing").and_then(Value::as_bool) == Some(true)
            || !raw_task.get("population").is_some_and(Value::is_object)
        {
            Some(crate::fpp::FppUnavailableReason::PopulationMissing)
        } else if crate::fpp::v2::calc_v3::population_provenance_missing(
            raw_task.get("population_sha256").and_then(Value::as_str),
            raw_task.get("population_version").and_then(Value::as_str),
        ) {
            // Python worker `_population_provenance_missing` (7b30ba1): released before any LC load, after
            // population_missing and before the cross-check checks.
            Some(crate::fpp::FppUnavailableReason::PopulationInvalid)
        } else if !raw_task
            .get("population_crosscheck")
            .and_then(Value::as_object)
            .and_then(|crosscheck| crosscheck.get("gaia_available"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            Some(crate::fpp::FppUnavailableReason::PopulationCrosscheckUnverified)
        } else if raw_task
            .get("population_crosscheck")
            .and_then(Value::as_object)
            .and_then(|crosscheck| crosscheck.get("alarm"))
            .is_some_and(|alarm| match alarm {
                Value::Null => false,
                Value::Bool(value) => *value,
                Value::Number(value) => value.as_f64().is_some_and(|x| x != 0.0),
                Value::String(value) => !value.is_empty(),
                Value::Array(_) | Value::Object(_) => true,
            })
            && raw_task
                .get("population")
                .and_then(|population| population.get("header"))
                .and_then(|header| header.get("source"))
                .and_then(Value::as_str)
                != Some("gaia")
        {
            Some(crate::fpp::FppUnavailableReason::PopulationAlarmUnresolved)
        } else {
            None
        };
        if let Some(reason) = release_reason {
            return Some(fpp_release_outcome(task, raw_task, reason.as_str(), reason.as_str().to_string()));
        }
        match self.process_fpp_v3_inner(task, worker_id) {
            Ok((r, loaded_lc_source, ephem_prov, ephem_failed, ephem_refine_on)) => Some(FppProcessOutcome {
                request: fpp_v3_result_request_with_ephem(
                    req,
                    task,
                    &r,
                    &loaded_lc_source,
                    ephem_prov,
                    ephem_failed,
                    ephem_refine_on,
                ),
                error: None,
                release: None,
            }),
            Err(err) => Some(fpp_error_outcome(task, raw_task, req, err)),
        }
    }
}

/// Loads the CNN FFI classifier from `<models_dir>/cnn_ffi_v1.onnx` (same
/// filename the Python worker's `worker/cnn_ffi.py::MODEL_PATH` uses).
/// Missing file or load failure both degrade to `None` with a log line —
/// see `PrimaryTaskProcessor`'s doc comment for why this is the graceful,
/// production-faithful choice rather than a hard error.
fn load_cnn(models_dir: &Path) -> Option<CnnFfiClassifier> {
    let path = models_dir.join("cnn_ffi_v1.onnx");
    if !path.exists() {
        eprintln!(
            "PrimaryTaskProcessor: CNN FFI model not present at {} — candidates will have no cnn_score (matches the Python worker's FileNotFoundError fallback)",
            path.display()
        );
        return None;
    }
    match CnnFfiClassifier::new(&path) {
        Ok(classifier) => Some(classifier),
        Err(error) => {
            eprintln!("PrimaryTaskProcessor: failed to load CNN FFI model at {}: {error} — continuing without CNN scores", path.display());
            None
        }
    }
}

/// Loads the ExoMiner FFI classifier from `<models_dir>/exominer_ffi_v6_ensemble.onnx`,
/// auto-loading its sibling `<model>.calib.json` if present (see
/// `ExoMinerFfiClassifier::new`). Same missing-file/load-failure degradation
/// as `load_cnn`.
fn load_exominer(models_dir: &Path) -> Option<ExoMinerFfiClassifier> {
    let path = models_dir.join("exominer_ffi_v6_ensemble.onnx");
    if !path.exists() {
        eprintln!(
            "PrimaryTaskProcessor: ExoMiner FFI model not present at {} — candidates will have no exominer_ffi_score",
            path.display()
        );
        return None;
    }
    match ExoMinerFfiClassifier::new(&path) {
        Ok(classifier) => Some(classifier),
        Err(error) => {
            eprintln!("PrimaryTaskProcessor: failed to load ExoMiner FFI model at {}: {error} — continuing without ExoMiner scores", path.display());
            None
        }
    }
}

/// Resolves the BLS compute backend ONCE, at `PrimaryTaskProcessor`
/// construction — mirrors `bls.py`'s import-time `is_cuda_available()`
/// probe timing (see `bls::cuda`'s module doc comment: probed once per
/// process, not once per task/call).
///
/// Honours `config.gpu_enabled` the same way `bls.py`'s backend-selection
/// priority chain does (`worker.py`'s `self.gpu_enabled` gates whether GPU
/// backends are even attempted — see CLAUDE.md's `gpu_enabled` references):
/// `gpu_enabled=false` never attempts CUDA, matching Python exactly.
/// `gpu_enabled=true` attempts `CudaBlsBackend::probe()`; any failure
/// there already logs a specific `eprintln!` (driver/NVRTC missing, no
/// device, compile/load error) and this function degrades to CPU without
/// logging again — one loud line per failure, not two.
///
/// Returns `(backend, gpu_used)` rather than just `backend` because
/// `PrimaryResultRequest.gpu_used` (the per-task result field) must be
/// honest about whether GPU was ACTUALLY used, not just configured —
/// deliberately more precise than `worker.py`'s own per-task
/// `"gpu_used": self.gpu_enabled` (which reports the config flag, not the
/// resolved backend); CLAUDE.md's "never claim capabilities the worker
/// doesn't actually have" argues for the honest field here even though it
/// is not a byte-for-byte Python port.
#[cfg(feature = "cuda")]
fn resolve_compute_backend(gpu_enabled: bool) -> (ComputeBackend, bool) {
    if !gpu_enabled {
        return (ComputeBackend::Cpu, false);
    }
    match crate::bls::cuda::CudaBlsBackend::probe() {
        Some(backend) => (ComputeBackend::Cuda(std::sync::Arc::new(backend)), true),
        None => (ComputeBackend::Cpu, false),
    }
}

/// Non-`cuda`-feature build: no CUDA backend exists in this binary at all
/// (compiled out, not just unavailable at runtime) — always CPU. Logs once
/// if the config asked for GPU anyway, since a silent CPU fallback here
/// would be exactly the "GPU host quietly on CPU" fleet gotcha CLAUDE.md
/// project rule 2 warns about (mirrors `worker.py`'s `_check_gpu_consistency`).
#[cfg(not(feature = "cuda"))]
fn resolve_compute_backend(gpu_enabled: bool) -> (ComputeBackend, bool) {
    if gpu_enabled {
        eprintln!(
            "PrimaryTaskProcessor: config.gpu_enabled=true but this binary was built without \
             the `cuda` Cargo feature — running BLS on CPU. Rebuild with `--features cuda` for \
             GPU BLS on this host."
        );
    }
    (ComputeBackend::Cpu, false)
}

/// Resolves injection-recovery's CUDA batch backend ONCE, at construction —
/// same timing contract as `resolve_compute_backend`, but a SEPARATE probe
/// (`bls::cuda_batch::CudaBlsBatchBackend::probe`, not
/// `bls::cuda::CudaBlsBackend::probe`) since the batch kernels are a
/// different module (see `bls::cuda_batch`'s doc comment for why an
/// independent `CudaContext::new(0)` call is safe here).
///
/// Gating mirrors `injection.py`'s real production gate EXACTLY:
/// `worker.py` calls `injection.disable_gpu()` (forcing CPU, unconditionally,
/// for every injection GPU backend the auto-detect chain might have picked)
/// precisely when its config has `gpu_enabled=false` — see that function's
/// docstring. Injection's GPU choice is therefore gated by the SAME
/// `config.gpu_enabled` flag as BLS, unlike wotan detrend's independent
/// `EXONIT_WOTAN_BACKEND` gate (`resolve_detrend_backend`) — this is a real
/// difference between the two Python modules' selection logic, not an
/// inconsistency introduced by this port.
/// F-opencl (`docs/plans/2026-08-06-opencl-backend-<host>.md`, Task 4):
/// resolves injection-recovery's GPU batch backend ONCE, at construction,
/// in priority order CUDA > OpenCL > CPU — mirrors `injection.py`'s real
/// production auto-detect chain exactly (CUDA > Metal fused > Metal legacy >
/// OpenCL > CPU; this crate has no Metal injection backend, so CUDA and
/// OpenCL are the only two non-CPU cases here — see
/// `docs/plans/2026-08-06-opencl-python-inventory.md`'s backend-priority
/// table). Gating mirrors `injection.py`'s real production gate EXACTLY:
/// `worker.py` calls `injection.disable_gpu()` (forcing CPU, unconditionally,
/// for every injection GPU backend the auto-detect chain might have picked)
/// precisely when its config has `gpu_enabled=false` — same
/// `config.gpu_enabled` flag as `resolve_compute_backend` (BLS primary,
/// CUDA-only — unlike wotan detrend's independent `EXONIT_WOTAN_BACKEND`
/// gate, see `resolve_detrend_backend`'s doc comment for that contrast).
///
/// Returns `(backend, gpu_used, label)`: `gpu_used`/`label` are honest about
/// what was ACTUALLY resolved (probed and confirmed working), not just what
/// `gpu_enabled` requested — same precision argument as
/// `resolve_compute_backend`'s doc comment. `label` is one of
/// `"rust-cuda"`/`"rust-sycl"`/`"rust-opencl"`/`"rust-cpu"`, read by
/// `runtime.rs`'s `build_capabilities` for the `injection_backend`
/// capability field.
///
/// **2026-08-06 Spec 2 integration**: priority chain extended to CUDA >
/// SYCL > OpenCL > CPU (was CUDA > OpenCL > CPU). SYCL ranks above OpenCL
/// because it measurably beat it on the SAME Arc iGPU both target
/// (`spikes/RESULTS_2026-08-06.md`: +31-37% median) — not a guess, a
/// real-measurement result. CUDA still ranks first: it targets different,
/// NVIDIA-only hardware (<host>/<host>) this crate has separately
/// validated, so CUDA and SYCL never actually compete for the SAME
/// physical device in any real deployment — the ordering here is
/// belt-and-suspenders for a hypothetical host with both an NVIDIA and an
/// Intel GPU, not a claim that CUDA beats SYCL on Arc (untested,
/// irrelevant: no such host exists in this fleet).
fn resolve_injection_backend(gpu_enabled: bool) -> (InjectionBackend, bool, &'static str) {
    if !gpu_enabled {
        return (InjectionBackend::Cpu, false, "rust-cpu");
    }
    if let Some(backend) = try_resolve_cuda_injection_backend() {
        return (backend, true, "rust-cuda");
    }
    if let Some(backend) = try_resolve_sycl_injection_backend() {
        return (backend, true, "rust-sycl");
    }
    if let Some(backend) = try_resolve_opencl_injection_backend() {
        return (backend, true, "rust-opencl");
    }
    #[cfg(not(any(feature = "cuda", feature = "sycl", feature = "opencl")))]
    eprintln!(
        "PrimaryTaskProcessor: config.gpu_enabled=true but this binary was built without the \
         `cuda`, `sycl`, or `opencl` Cargo features — running injection-recovery on CPU."
    );
    (InjectionBackend::Cpu, false, "rust-cpu")
}

#[cfg(feature = "cuda")]
fn try_resolve_cuda_injection_backend() -> Option<InjectionBackend> {
    crate::bls::cuda_batch::CudaBlsBatchBackend::probe()
        .map(|backend| InjectionBackend::Cuda(std::sync::Arc::new(backend)))
}

#[cfg(not(feature = "cuda"))]
fn try_resolve_cuda_injection_backend() -> Option<InjectionBackend> {
    None
}

/// 2026-08-06 Spec 2 integration — see `resolve_injection_backend`'s doc
/// comment for the priority-chain rationale.
#[cfg(feature = "sycl")]
fn try_resolve_sycl_injection_backend() -> Option<InjectionBackend> {
    crate::bls::sycl_batch::SyclBlsBatchBackend::probe()
        .map(|backend| InjectionBackend::Sycl(std::sync::Arc::new(backend)))
}

#[cfg(not(feature = "sycl"))]
fn try_resolve_sycl_injection_backend() -> Option<InjectionBackend> {
    None
}

#[cfg(feature = "opencl")]
fn try_resolve_opencl_injection_backend() -> Option<InjectionBackend> {
    crate::bls::opencl_batch::OpenClBlsBatchBackend::probe()
        .map(|backend| InjectionBackend::OpenCl(std::sync::Arc::new(backend)))
}

#[cfg(not(feature = "opencl"))]
fn try_resolve_opencl_injection_backend() -> Option<InjectionBackend> {
    None
}

/// Resolves the wotan biweight detrend CUDA backend ONCE, at construction.
/// Gating mirrors `wotan_detrend.py` EXACTLY — NOT gated by
/// `config.gpu_enabled` at all (see `resolve_injection_backend`'s doc
/// comment for the contrast): `_forced = EXONIT_WOTAN_BACKEND` env var,
/// `"cpu"` always wins (forces CPU regardless of what would otherwise be
/// probed), any other value (unset, `"cuda"`, `"auto"`, ...) attempts the
/// CUDA probe and falls back to CPU on failure — `"cuda"` specifically logs
/// an extra "requested but not available" line (matching Python's own
/// extra warning for that one value), every other non-`"cpu"` value that
/// fails the probe degrades silently-to-CPU-but-loudly-logged exactly like
/// `bls::cuda`'s own probe failure branches (each already `eprintln!`s).
#[cfg(feature = "cuda")]
fn resolve_detrend_backend() -> (DetrendBackend, bool) {
    let forced = std::env::var("EXONIT_WOTAN_BACKEND")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase());
    if forced.as_deref() == Some("cpu") {
        return (DetrendBackend::Cpu, false);
    }
    match crate::detrend_cuda::CudaDetrendBackend::probe() {
        Some(backend) => (DetrendBackend::Cuda(std::sync::Arc::new(backend)), true),
        None => {
            if forced.as_deref() == Some("cuda") {
                eprintln!(
                    "PrimaryTaskProcessor: EXONIT_WOTAN_BACKEND=cuda requested but CUDA not \
                     available; falling back to wotan CPU."
                );
            }
            (DetrendBackend::Cpu, false)
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn resolve_detrend_backend() -> (DetrendBackend, bool) {
    let forced = std::env::var("EXONIT_WOTAN_BACKEND")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase());
    if forced.as_deref() == Some("cuda") {
        eprintln!(
            "PrimaryTaskProcessor: EXONIT_WOTAN_BACKEND=cuda requested but this binary was built \
             without the `cuda` Cargo feature — running wotan detrend on CPU."
        );
    }
    (DetrendBackend::Cpu, false)
}

// ---------------------------------------------------------------------------
// task.config readers — mirrors the private `Dictionary where Key == String,
// Value == JSONValue` extension at the bottom of `PrimaryTaskProcessor.swift`.
// ---------------------------------------------------------------------------

fn config_bool(config: &Map<String, Value>, key: &str, default: bool) -> bool {
    match config.get(key) {
        Some(Value::Bool(value)) => *value,
        _ => default,
    }
}

fn config_double(config: &Map<String, Value>, key: &str, default: f64) -> f64 {
    match config.get(key).and_then(Value::as_f64) {
        Some(value) => value,
        None => default,
    }
}

fn config_int(config: &Map<String, Value>, key: &str, default: i64) -> i64 {
    match config.get(key) {
        Some(Value::Number(n)) => n
            .as_i64()
            .or_else(|| n.as_f64().map(|f| f as i64))
            .unwrap_or(default),
        _ => default,
    }
}

fn config_doubles(config: &Map<String, Value>, key: &str, default: &[f64]) -> Vec<f64> {
    match config.get(key) {
        Some(Value::Array(values)) => {
            let parsed: Vec<f64> = values.iter().filter_map(Value::as_f64).collect();
            if parsed.is_empty() {
                default.to_vec()
            } else {
                parsed
            }
        }
        _ => default.to_vec(),
    }
}

/// Mirrors `Dictionary.doubleRanges(_:)`: an array of 2+-element numeric
/// arrays, each read as `(row[0], row[1])`.
fn config_double_ranges(config: &Map<String, Value>, key: &str) -> Option<Vec<Vec<f64>>> {
    match config.get(key) {
        Some(Value::Array(rows)) => Some(
            rows.iter()
                .map(|row| match row {
                    Value::Array(values) => values.iter().filter_map(Value::as_f64).collect(),
                    _ => Vec::new(),
                })
                .collect(),
        ),
        _ => None,
    }
}

/// Mirrors `PrimaryTaskProcessor.multiPlanetOptions(config:)` field-by-field.
fn multi_planet_options_from_config(config: &Map<String, Value>) -> MultiPlanetOptions {
    let mut options = MultiPlanetOptions::default();
    options.maximum_planets = config_int(config, "max_planets", 5);
    options.minimum_snr = config_double(config, "snr_threshold", 7.0);
    options.period_minimum = config_double(config, "period_min", 0.5);
    let legacy_period_maximum = config_double(config, "period_max_legacy", 50.0);
    options.period_maximum = config_double(config, "period_max", legacy_period_maximum);
    options.period_count = config_int(config, "n_periods", 50_000).max(1) as usize;
    options.protected_slots_enabled = config_bool(config, "medium_period_protected_slot", true);
    options.protected_slot_minimum_snr = config_double(config, "medium_slot_min_snr", 12.0);
    if let Some(ranges) = config_double_ranges(config, "medium_slot_ranges") {
        options.protected_slot_ranges = ranges
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 2 && row[1] > row[0] {
                    Some((row[0], row[1]))
                } else {
                    None
                }
            })
            .collect();
    } else if config.contains_key("medium_slot_min_days")
        || config.contains_key("medium_slot_max_days")
    {
        let minimum = config_double(config, "medium_slot_min_days", 10.0);
        let maximum = config_double(config, "medium_slot_max_days", 50.0);
        options.protected_slot_ranges = if maximum > minimum {
            vec![(minimum, maximum)]
        } else {
            Vec::new()
        };
    }
    options.harmonic_long_period_recovery =
        config_bool(config, "harmonic_long_period_recovery", false);
    options.harmonic_long_period_minimum_days =
        config_double(config, "harmonic_long_period_min_days", 10.0);
    options.harmonic_long_snr_factor = config_double(config, "harmonic_long_snr_factor", 0.5);
    options.harmonic_epoch_class = config_bool(config, "harmonic_epoch_class", false);
    if config_bool(config, "wotan_long_sweep", true) {
        let mut long = LongPeriodSweepOptions {
            baseline_threshold_days: config_double(config, "wotan_long_threshold_days", 100.0),
            windows_days: config_doubles(config, "wotan_long_windows_days", &[2.0, 4.0]),
            period_minimum_days: config_double(config, "wotan_long_period_min_days", 30.0),
            legacy_period_maximum_days: options.period_maximum,
            period_maximum_cap_days: config_double(config, "period_max_cap", 200.0),
            minimum_transits: config_double(config, "period_max_min_transits", 2.5),
            base_period_count: config_int(config, "wotan_long_n_periods", 50_000).max(1) as usize,
            minimum_snr: config_double(config, "wotan_long_snr_threshold", options.minimum_snr),
            ..LongPeriodSweepOptions::default()
        };
        if config.contains_key("wotan_long_n_periods") {
            long.period_count_scale_reference_maximum_days = None;
        }
        options.long_period_sweep = Some(long);
    } else {
        options.long_period_sweep = None;
    }
    options
}

#[cfg(test)]
mod tests {
    use super::*;

    fn release_test_task() -> (crate::dto::BackfillTask, Map<String, Value>) {
        let task: crate::dto::BackfillTask = serde_json::from_value(serde_json::json!({
            "candidate_id": 77,
            "tic_id": "12345",
            "candidate_num": 1,
            "period_days": 3.0,
            "t0": 2459000.0,
            "duration_days": 0.1,
            "depth_ppm": 500.0,
            "snr": 10.0,
        }))
        .unwrap();
        let raw = serde_json::json!({"fpp_claim_id": "claim-77"})
            .as_object()
            .unwrap()
            .clone();
        (task, raw)
    }

    /// R4: the release decision depends on the error TYPE only. Each typed
    /// reason releases even with a message that shares no words with the
    /// production message, and an untyped failure that carries the exact
    /// text the old string matcher keyed on ("background LDC", "population
    /// too small", "FPP geometry unavailable", "nodes missing") must NOT be
    /// released. A message-text classifier fails both halves.
    #[test]
    fn fpp_error_outcome_releases_by_typed_reason_never_by_message_text() {
        use crate::fpp::{FppError, FppUnavailableReason};
        let (task, raw) = release_test_task();
        for reason in [
            FppUnavailableReason::PopulationMissing,
            FppUnavailableReason::PopulationInvalid,
            FppUnavailableReason::PopulationCrosscheckUnverified,
            FppUnavailableReason::PopulationAlarmUnresolved,
            FppUnavailableReason::PopulationAlarm,
            FppUnavailableReason::PopulationLdcInvalid,
        ] {
            let req = crate::dto::FppResultRequest::new(Some(77), Some("12345".into()), None);
            let outcome = fpp_error_outcome(
                &task,
                &raw,
                req,
                FppError::unavailable(reason, "reworded: zzz qqq"),
            );
            let release = outcome.release.as_ref().expect("typed reason must release");
            assert_eq!(release["reason"], reason.as_str());
            assert_eq!(release["claim_id"], "claim-77");
            assert_eq!(release["candidate_id"], 77);
            assert_eq!(outcome.request.fpp, None, "{reason}: no score on release");
            assert_eq!(outcome.request.nfpp, None, "{reason}: no score on release");
            assert_eq!(outcome.request.verdict, None, "{reason}: no verdict on release");
        }
        for text in [
            "background LDC lacks Teff=4000, logg=4.5",
            "background LDC row is not unique: (4000, 4.5, 0)",
            "population too small for upstream D-family indexing",
            "FPP geometry unavailable: nodes missing",
            "population_invalid: invalid background population",
        ] {
            let req = crate::dto::FppResultRequest::new(Some(77), Some("12345".into()), None);
            let outcome = fpp_error_outcome(&task, &raw, req, FppError::failed(text));
            assert!(outcome.release.is_none(), "untyped failure {text:?} must not release");
            assert_eq!(outcome.request.fpp, Some(-1.0));
            assert_eq!(outcome.request.nfpp, Some(-1.0));
            assert_eq!(outcome.request.verdict.as_deref(), Some("failed"));
        }
        // A `String` error converted by `?` is a failure, whatever it says.
        let converted: FppError = String::from("population too small for upstream D-family indexing").into();
        assert_eq!(converted.unavailable_reason(), None);
    }

    /// Task 33 item 3.4: every `-1` post carries Python's `skip_reason` from the TYPED reason (null when Python has
    /// none), with Python's exact body keys; message text never carries it.
    #[test]
    fn failed_posts_carry_the_typed_python_skip_reason() {
        use crate::fpp::{FppError, FppSkipReason};
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../fixtures/fpp/v2/wire_mc_fields_v3.json"
        ))
        .unwrap();
        let (task, raw) = release_test_task();
        let typed = |name: Option<&str>| -> FppError {
            match name {
                None => FppError::failed("raw_lc_unusable: giant_host ldc_no_row (message text is never a carrier)"),
                Some("raw_lc_unusable") => FppError::skip(FppSkipReason::RawLcUnusable, "zzz"),
                Some("giant_host") => FppError::skip(FppSkipReason::GiantHost, "zzz"),
                Some("stellar_params_missing") => FppError::skip(FppSkipReason::StellarParamsMissing, "zzz"),
                Some("ldc_no_row") => FppError::skip(FppSkipReason::LdcNoRow, "zzz"),
                Some(other) => panic!("{other}: not reachable in the Rust port (compiled table)"),
            }
        };
        let mut n = 0;
        for b in fixture["failed_bodies"].as_array().unwrap() {
            let reason = b["skip_reason"].as_str();
            if reason == Some("ldc_table_missing") {
                // Python LDCTableMissing: the LDC table is `include_str!` data here, so it cannot be missing.
                continue;
            }
            let req = crate::dto::FppResultRequest::new(Some(77), Some("12345".into()), None);
            let mut req_with_candidate = req.clone();
            req_with_candidate.candidate = Some(raw.clone());
            let outcome = fpp_error_outcome(&task, &raw, req_with_candidate, typed(reason));
            assert!(outcome.release.is_none(), "{reason:?}: a failure never releases");
            let body = serde_json::to_value(&outcome.request).unwrap();
            let want_keys: std::collections::BTreeSet<&String> = b["body"].as_object().unwrap().keys().collect();
            let got_keys: std::collections::BTreeSet<&String> = body.as_object().unwrap().keys().collect();
            assert_eq!(got_keys, want_keys, "{reason:?}: Python -1 body keys");
            assert_eq!(body["fpp"], -1.0);
            assert_eq!(body["nfpp"], -1.0);
            assert_eq!(body["verdict"], "failed");
            assert_eq!(body["skip_reason"], b["skip_reason"], "{reason:?}");
            n += 1;
        }
        assert_eq!(n, 5);
    }

    /// Task 33 item 4 at the worker boundary: malformed geometry AND an invalid population (the double fault of
    /// `validation_order_v3.json`) through the real calc gives a `-1` failure, never a release.
    #[test]
    fn double_fault_geometry_and_population_fails_at_the_worker_boundary() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../fixtures/fpp/v2/validation_order_v3.json"
        ))
        .unwrap();
        let (task, raw) = release_test_task();
        let bits = |v: &Value| f64::from_bits(u64::from_str_radix(v.as_str().unwrap(), 16).unwrap());
        fn numeric(v: &Value) -> Value {
            match v {
                Value::String(s) if s.len() == 16 && s.bytes().all(|b| b.is_ascii_hexdigit()) => {
                    serde_json::Number::from_f64(f64::from_bits(u64::from_str_radix(s, 16).unwrap()))
                        .map(Value::Number)
                        .unwrap_or(Value::Null)
                }
                Value::Array(a) => Value::Array(a.iter().map(numeric).collect()),
                Value::Object(o) => Value::Object(o.iter().map(|(k, v)| (k.clone(), numeric(v))).collect()),
                _ => v.clone(),
            }
        }
        let inp = &fixture["inputs"];
        let time: Vec<f64> = inp["time"].as_array().unwrap().iter().map(bits).collect();
        let flux: Vec<f64> = inp["flux"].as_array().unwrap().iter().map(bits).collect();
        let nearby = serde_json::json!({"stars": numeric(&inp["stars"]), "gal_b": bits(&fixture["gal_b"])});
        let tic = fixture["tic"].as_u64().unwrap().to_string();
        let mut n = 0;
        for case in fixture["cases"].as_array().unwrap().iter().filter(|c| c["name"] == "both") {
            let geometry = numeric(&case["geometry"]);
            let result = crate::fpp::v2::calc_v3::calc_fpp_by_source_v3(
                &crate::fpp::v2::calc_v3::CalcV3Inputs {
                    tic_id: &tic,
                    period: bits(&inp["period"]),
                    t0: bits(&inp["t0"]),
                    duration: bits(&inp["duration_d"]),
                    depth_ppm: bits(&inp["depth_ppm"]),
                    time: &time,
                    flux: &flux,
                    nearby_raw: &nearby,
                    geometry: Some(&geometry),
                    population: Some(&case["population"]),
                    population_sha256: inp["population_sha256"].as_str(),
                    population_version: inp["population_version"].as_str(),
                    population_crosscheck: Some(&inp["population_crosscheck"]),
                    cadence_s: Some(600.0),
                    lc_source: Some("tglc"),
                    tglc_near_edge: fixture["tglc_near_edge"].as_bool(),
                    seed_ephemeris: None,
                },
                2000,
                fixture["seed"].as_u64().unwrap() as u32,
                case["switch_set"] == "production",
            );
            let err = result.expect_err("double fault");
            let req = crate::dto::FppResultRequest::new(Some(77), Some("12345".into()), None);
            let outcome = fpp_error_outcome(&task, &raw, req, err);
            assert!(outcome.release.is_none(), "{}: Python fails (geometry first), never releases", case["switch_set"]);
            assert_eq!(outcome.request.fpp, Some(-1.0));
            assert_eq!(outcome.request.verdict.as_deref(), Some("failed"));
            assert_eq!(outcome.request.skip_reason, Some(None), "Python skip_reason None");
            n += 1;
        }
        assert_eq!(n, 2, "parity and production");
    }

    #[test]
    fn loaded_fpp_lc_source_comes_from_file_products() {
        assert_eq!(
            fpp_lc_source_from_files(&[
                "/cache/hlsp_tglc_tic-1.fits".to_string(),
                "/cache/hlsp_tglc_tic-1-s2.fits".to_string(),
            ]),
            "tglc"
        );
        assert_eq!(
            fpp_lc_source_from_files(&["/cache/hlsp_qlp_tic-1.fits".to_string()]),
            "qlp"
        );
        assert_eq!(
            fpp_lc_source_from_files(&["/cache/tic-1_lc.fits".to_string()]),
            "spoc"
        );
        assert_eq!(fpp_lc_source_from_files(&[]), "unknown");
        assert_eq!(
            fpp_lc_source_from_files(&[
                "/cache/hlsp_tglc_tic-1.fits".to_string(),
                "/cache/tic-1_lc.fits".to_string(),
            ]),
            "mixed"
        );
    }

    /// Serializes the lib tests that mutate the FPP_* process environment.
    static FPP_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Task path (`process_fpp_v3_inner`): an invalid strict switch must fail
    /// with the typed switch error before any metadata fetch or computation.
    /// The API points at a closed port, so if validation did not run first the
    /// error would be a "metadata: ..." network failure instead.
    #[test]
    fn fpp_v3_task_path_rejects_invalid_switch_values_before_any_work() {
        let _lock = FPP_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let orig_refine = std::env::var("FPP_EPHEM_REFINE").ok();
        let orig_unstable = std::env::var("FPP_UNSTABLE_FLAG").ok();
        let orig_parity = std::env::var("FPP_PARITY").ok();
        std::env::remove_var("FPP_PARITY");

        let api = Arc::new(WorkerApiClient::new("http://127.0.0.1:1", "token"));
        let cache_dir = std::env::temp_dir().join(format!(
            "exonit-rust-switch-task-path-{}-{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let fits_cache = Arc::new(FitsCache::new(cache_dir).expect("create fits cache"));
        let processor = PrimaryTaskProcessor::new(api, fits_cache, None, 1, false);
        let task: crate::dto::BackfillTask = serde_json::from_value(serde_json::json!({
            "candidate_id": 1,
            "tic_id": "1",
            "candidate_num": 1,
            "period_days": 3.0,
            "t0": 1.0,
            "duration_days": 0.1,
            "depth_ppm": 1000.0,
            "snr": 10.0,
        }))
        .expect("backfill task");

        for (name, other) in [
            ("FPP_UNSTABLE_FLAG", "FPP_EPHEM_REFINE"),
            ("FPP_EPHEM_REFINE", "FPP_UNSTABLE_FLAG"),
        ] {
            for bad in ["1", "true", "ON", ""] {
                std::env::remove_var(other);
                std::env::set_var(name, bad);
                match processor.process_fpp_v3_inner(&task, "w") {
                    Err(crate::fpp::FppError::Failed { message, .. }) => assert!(
                        message.contains(&format!("invalid switch value {name}=")),
                        "{name}={bad:?}: wrong error: {message}"
                    ),
                    Err(_) => panic!("{name}={bad:?}: expected FppError::Failed"),
                    Ok(_) => panic!("{name}={bad:?}: invalid switch must not compute"),
                }
            }
        }

        match orig_refine { Some(v) => std::env::set_var("FPP_EPHEM_REFINE", v), None => std::env::remove_var("FPP_EPHEM_REFINE") }
        match orig_unstable { Some(v) => std::env::set_var("FPP_UNSTABLE_FLAG", v), None => std::env::remove_var("FPP_UNSTABLE_FLAG") }
        if let Some(v) = orig_parity { std::env::set_var("FPP_PARITY", v); }
    }

    #[test]
    fn fpp_v3_switch_provenance_matches_both_reference_sets() {
        let _lock = FPP_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let orig_refine = std::env::var("FPP_EPHEM_REFINE").ok();
        let orig_unstable = std::env::var("FPP_UNSTABLE_FLAG").ok();
        let orig_parity = std::env::var("FPP_PARITY").ok();
        std::env::remove_var("FPP_EPHEM_REFINE");
        std::env::remove_var("FPP_UNSTABLE_FLAG");
        std::env::remove_var("FPP_PARITY");

        let fixture: Value = serde_json::from_str(include_str!(
            "../../../fixtures/fpp/v2/calc_fpp_by_source_v3.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let production = case["switch_set"] == "production";
            let mut expected = case["result"]["fpp_switches"].as_object().unwrap().clone();
            if production {
                expected.insert("FPP_EPHEM_REFINE".to_string(), Value::String("on".into()));
                expected.insert("FPP_UNSTABLE_FLAG".to_string(), Value::String("on".into()));
            } else {
                expected.insert("FPP_EPHEM_REFINE".to_string(), Value::String("off".into()));
                expected.insert("FPP_UNSTABLE_FLAG".to_string(), Value::String("off".into()));
            }
            assert_eq!(
                fpp_v3_switches(production),
                expected,
                "TIC {} {}",
                case["tic"],
                case["switch_set"]
            );
        }

        if let Some(v) = orig_refine { std::env::set_var("FPP_EPHEM_REFINE", v); }
        if let Some(v) = orig_unstable { std::env::set_var("FPP_UNSTABLE_FLAG", v); }
        if let Some(v) = orig_parity { std::env::set_var("FPP_PARITY", v); }
    }

    #[test]
    fn fpp_v3_lc_source_uses_python_result_schema_normalization() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../fixtures/fpp/v2/wire_schema_v3.json"
        ))
        .unwrap();
        let mut request = crate::dto::FppResultRequest::new(None, None, None);
        request.lc_source = Some(normalized_fpp_v3_lc_source(Some(" TGLC ")));
        let serialized = serde_json::to_value(request).unwrap();
        assert_eq!(
            serialized["lc_source"],
            fixture["result"]["lc_source"],
            "native task result should use Python's normalized source label"
        );
        assert_eq!(normalized_fpp_v3_lc_source(None), "unknown");
    }

    /// Swift: `testTaskScientificOptions` (`Sources/ExoNitContractTests/main.swift:1065`),
    /// same config keys and expected outcomes.
    #[test]
    fn multi_planet_options_from_config_matches_swift_contract_test() {
        let mut config = Map::new();
        config.insert("period_max".to_string(), Value::from(50));
        config.insert(
            "medium_period_protected_slot".to_string(),
            Value::from(false),
        );
        config.insert("medium_slot_min_snr".to_string(), Value::from(15));
        config.insert(
            "medium_slot_ranges".to_string(),
            Value::from(vec![vec![4, 9], vec![12, 40]]),
        );
        config.insert(
            "harmonic_long_period_recovery".to_string(),
            Value::from(true),
        );
        config.insert("harmonic_long_period_min_days".to_string(), Value::from(14));
        config.insert("harmonic_long_snr_factor".to_string(), Value::from(0.65));
        config.insert("wotan_long_sweep".to_string(), Value::from(false));

        let options = multi_planet_options_from_config(&config);
        assert_eq!(
            options.period_maximum, 50.0,
            "main BLS maximum must not be clipped to span/2"
        );
        assert!(
            !options.protected_slots_enabled,
            "protected-slot disable flag must propagate"
        );
        assert_eq!(
            options.protected_slot_minimum_snr, 15.0,
            "protected-slot SNR must propagate"
        );
        assert_eq!(
            options.protected_slot_ranges,
            vec![(4.0, 9.0), (12.0, 40.0)],
            "protected-slot ranges must propagate"
        );
        assert!(
            options.harmonic_long_period_recovery,
            "harmonic long-period gate must propagate"
        );
        assert_eq!(
            options.harmonic_long_period_minimum_days, 14.0,
            "harmonic long-period minimum must propagate"
        );
        assert_eq!(
            options.harmonic_long_snr_factor, 0.65,
            "harmonic long-period SNR factor must propagate"
        );
        assert!(
            options.long_period_sweep.is_none(),
            "long sweep disable flag must propagate"
        );
    }

    #[test]
    fn multi_planet_options_from_config_defaults_match_swift_struct_defaults() {
        let options = multi_planet_options_from_config(&Map::new());
        let defaults = MultiPlanetOptions::default();
        assert_eq!(
            options, defaults,
            "an empty task config must reproduce MultiPlanetOptions::default()"
        );
    }

    /// `harmonic_epoch_class` (epoch-class alias check): absent = off, and the
    /// config key flips exactly `MultiPlanetOptions::harmonic_epoch_class`.
    #[test]
    fn harmonic_epoch_class_flag_defaults_off_and_propagates() {
        assert!(!multi_planet_options_from_config(&Map::new()).harmonic_epoch_class);
        let mut config = Map::new();
        config.insert("harmonic_epoch_class".to_string(), Value::from(true));
        let options = multi_planet_options_from_config(&config);
        assert!(
            options.harmonic_epoch_class,
            "harmonic_epoch_class must propagate"
        );
        let mut expected = MultiPlanetOptions::default();
        expected.harmonic_epoch_class = true;
        assert_eq!(
            options, expected,
            "no other option may change with the flag"
        );
    }

    /// Mirrors `PrimaryTaskProcessor.injectionFlux(from:)`: injection-recovery
    /// runs against the detrended flat flux, never the raw flux — asserted
    /// directly at the call site in `process_inner` (`&processed.flat_flux`),
    /// this test guards the config-reading half doesn't accidentally disable
    /// that by misreading `run_injection`/`run_injection_recovery` defaults.
    #[test]
    fn run_injection_defaults_to_true_and_legacy_key_is_honoured() {
        assert!(config_bool(
            &Map::new(),
            "run_injection",
            config_bool(&Map::new(), "run_injection_recovery", true)
        ));
        let mut legacy_off = Map::new();
        legacy_off.insert("run_injection_recovery".to_string(), Value::from(false));
        assert!(!config_bool(
            &legacy_off,
            "run_injection",
            config_bool(&legacy_off, "run_injection_recovery", true)
        ));
        let mut explicit_on = Map::new();
        explicit_on.insert("run_injection_recovery".to_string(), Value::from(false));
        explicit_on.insert("run_injection".to_string(), Value::from(true));
        assert!(config_bool(
            &explicit_on,
            "run_injection",
            config_bool(&explicit_on, "run_injection_recovery", true)
        ));
    }

    #[test]
    fn config_readers_fall_back_to_defaults_on_missing_or_wrong_type_keys() {
        let mut config = Map::new();
        config.insert("wrong_type".to_string(), Value::from("not a number"));
        assert_eq!(config_bool(&config, "missing", true), true);
        assert_eq!(config_double(&config, "wrong_type", 3.5), 3.5);
        assert_eq!(config_int(&config, "wrong_type", 7), 7);
        assert_eq!(
            config_doubles(&config, "missing", &[1.0, 2.0]),
            vec![1.0, 2.0]
        );
        assert_eq!(config_double_ranges(&config, "missing"), None);
    }

    /// F2b degradation requirement: a models directory that exists but has
    /// neither model file present must not fail construction — both
    /// classifiers stay absent, matching the Python worker's real
    /// `FileNotFoundError` -> `classifier_ffi = None` fallback (see this
    /// module's `PrimaryTaskProcessor` doc comment).
    #[test]
    fn missing_model_files_degrade_to_absent_classifiers_not_a_crash() {
        let dir = std::env::temp_dir().join(format!(
            "exonit-rust-empty-models-{}-{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("create empty models dir");
        assert!(
            load_cnn(&dir).is_none(),
            "no cnn_ffi_v1.onnx present -> None, not a panic"
        );
        assert!(
            load_exominer(&dir).is_none(),
            "no exominer_ffi_v6_ensemble.onnx present -> None, not a panic"
        );
    }

    /// Same guarantee, but via the public constructor a `WorkerRuntime`
    /// actually calls: no `models_dir` configured at all.
    #[test]
    fn new_without_models_dir_reports_honest_unavailable_capabilities() {
        let api = Arc::new(WorkerApiClient::new("http://127.0.0.1:1", "token"));
        let cache_dir = std::env::temp_dir().join(format!(
            "exonit-rust-no-models-fits-cache-{}-{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let fits_cache = Arc::new(FitsCache::new(cache_dir).expect("create fits cache"));
        let processor = PrimaryTaskProcessor::new(api, fits_cache, None, 1, false);
        assert!(!processor.cnn_available());
        assert!(!processor.exominer_available());
        assert!(
            !processor.gpu_used(),
            "gpu_enabled: false must never resolve to a GPU backend"
        );
    }

    /// F4a: `gpu_enabled: true` without the `cuda` Cargo feature must
    /// degrade to CPU (logged, not silent — see
    /// `resolve_compute_backend`'s `#[cfg(not(feature = "cuda"))]` arm),
    /// never panic or claim GPU. This test runs in both build
    /// configurations.
    ///
    /// Host-agnostic by design (fixed 2026-08-07): the original version of
    /// this test hardcoded `assert!(!processor.gpu_used())` under the
    /// assumption that it only ever ran on <host> (no NVIDIA GPU). That
    /// assumption broke the FIRST time `cargo test --features cuda --all`
    /// ran on a host where the CUDA gate is actually green end-to-end
    /// (<host>'s RTX 4070, driver 610.88, `bls::cuda::CudaBlsBackend::probe()`
    /// succeeding) — the panic message ("no NVIDIA GPU on this host") was
    /// simply false there, and `gpu_used()` was correctly reporting `true`.
    /// The `#[ignore]`d golden tests in `tests/bls_cuda_golden.rs` cover
    /// numerical correctness of the CUDA path; THIS test only needs to
    /// confirm `gpu_used()` tells the truth about whatever
    /// `resolve_compute_backend` actually resolved on THIS host, so it
    /// re-derives the expected value from the same probe instead of
    /// assuming no GPU exists.
    #[test]
    fn new_with_gpu_enabled_never_panics_and_reports_honest_gpu_used() {
        let api = Arc::new(WorkerApiClient::new("http://127.0.0.1:1", "token"));
        let cache_dir = std::env::temp_dir().join(format!(
            "exonit-rust-gpu-enabled-fits-cache-{}-{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let fits_cache = Arc::new(FitsCache::new(cache_dir).expect("create fits cache"));
        let processor = PrimaryTaskProcessor::new(api, fits_cache, None, 1, true);
        #[cfg(feature = "cuda")]
        let expected_gpu_used = crate::bls::cuda::CudaBlsBackend::probe().is_some();
        #[cfg(not(feature = "cuda"))]
        let expected_gpu_used = false;
        assert_eq!(
            processor.gpu_used(),
            expected_gpu_used,
            "gpu_used() must honestly reflect whether bls::cuda::CudaBlsBackend::probe() actually \
             succeeds on this host, never a hardcoded assumption"
        );
    }

    /// F-opencl Task 4 Step 1: `gpu_enabled: false` must resolve the
    /// injection-recovery backend to `Cpu` unconditionally — even on a host
    /// with a real, working OpenCL GPU (<host>'s Arc) and the `opencl`
    /// Cargo feature compiled in. Mirrors `injection.py`'s
    /// `disable_gpu()`/`worker.py`'s `gpu_enabled=false` gate: the config
    /// flag wins before any backend is even probed, same as
    /// `new_without_models_dir_reports_honest_unavailable_capabilities`'s
    /// `gpu_used()` assertion for the primary BLS path above.
    #[test]
    fn injection_backend_never_resolves_gpu_when_disabled_even_with_opencl_feature_compiled() {
        let api = Arc::new(WorkerApiClient::new("http://127.0.0.1:1", "token"));
        let cache_dir = std::env::temp_dir().join(format!(
            "exonit-rust-injection-gpu-disabled-fits-cache-{}-{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let fits_cache = Arc::new(FitsCache::new(cache_dir).expect("create fits cache"));
        let processor = PrimaryTaskProcessor::new(api, fits_cache, None, 1, false);
        assert!(
            !processor.injection_gpu_used(),
            "gpu_enabled: false must never resolve an injection GPU backend"
        );
        assert_eq!(processor.injection_backend_label(), "rust-cpu");
    }
}
