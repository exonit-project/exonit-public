//! Transliterated from `Sources/ExoNitCore/WorkerProtocolDTO.swift`, plus the
//! pure payload/domain types (not the HTTP client itself — that is `api.rs`,
//! deferred to F0b) from `Sources/ExoNitCore/WorkerAPI.swift`.
//!
//! Swift needed a hand-rolled `JSONValue` enum (`Codable`, mirroring
//! `Any`-typed JSON) because `Foundation`'s `Codable` has no native "any
//! JSON value" type. Rust does not have this problem — `serde_json::Value`
//! already *is* that type — so `JSONValue.swift` has no Rust file of its
//! own; every Swift `JSONValue` / `[String: JSONValue]` / `[JSONValue]`
//! below becomes `serde_json::Value` / `serde_json::Map<String, Value>` /
//! `Vec<Value>`.
//!
//! Field names, JSON key renames, optionality and defaults are transliterated
//! field-by-field from the Swift `CodingKeys` + `init(from decoder:)`/
//! memberwise-init pairs. Swift's synthesized `Codable` only skips
//! serializing a field when its Swift type is `Optional` (`encodeIfPresent`);
//! `#[serde(skip_serializing_if = "Option::is_none")]` reproduces that, and
//! `#[serde(default)]` reproduces `decodeIfPresent(...) ?? default` /
//! "missing key is fine for an Optional property".

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

/// Decodes a JSON value that may be a string or an integer as a `String`,
/// trying string first then int64 — mirrors the free function
/// `decodeFlexibleString` in `WorkerProtocolDTO.swift`, used wherever the
/// server may send a TIC ID as either JSON type.
fn deserialize_flexible_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrInt {
        S(String),
        I(i64),
    }
    match StringOrInt::deserialize(deserializer)? {
        StringOrInt::S(s) => Ok(s),
        StringOrInt::I(i) => Ok(i.to_string()),
    }
}

/// Optional counterpart used where Swift does
/// `if c.contains(.key) { field = try? decodeFlexibleString(...) } else { field = nil }`
/// — i.e. a missing key is `None`, and a present-but-unparseable value is
/// swallowed to `None` rather than raising a decode error (the Swift `try?`).
fn deserialize_flexible_string_opt<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value: Option<Value> = Option::deserialize(deserializer)?;
    Ok(value.and_then(|v| match v {
        Value::String(s) => Some(s),
        Value::Number(n) => n.as_i64().map(|i| i.to_string()),
        _ => None,
    }))
}

// ---------------------------------------------------------------------------
// CandidateResult / VettingResult / CompletenessResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateResult {
    #[serde(rename = "candidate_num")]
    pub candidate_number: i64,
    #[serde(rename = "period_days")]
    pub period_days: f64,
    pub t0: f64,
    #[serde(rename = "duration_days")]
    pub duration_days: f64,
    #[serde(rename = "depth_ppm")]
    pub depth_ppm: f64,
    pub snr: f64,
    #[serde(rename = "bls_power", default, skip_serializing_if = "Option::is_none")]
    pub bls_power: Option<f64>,
    #[serde(
        rename = "n_transits_expected",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub expected_transits: Option<i64>,
    #[serde(rename = "triceratops_fpp", default, skip_serializing_if = "Option::is_none")]
    pub triceratops_fpp: Option<f64>,
    #[serde(rename = "triceratops_nfpp", default, skip_serializing_if = "Option::is_none")]
    pub triceratops_nfpp: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nfpp: Option<f64>,
    #[serde(
        rename = "triceratops_prob_planet",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub triceratops_probability_planet: Option<f64>,
    #[serde(
        rename = "triceratops_verdict",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub triceratops_verdict: Option<String>,
    #[serde(rename = "exominer_score", default, skip_serializing_if = "Option::is_none")]
    pub exominer_score: Option<f64>,
    #[serde(
        rename = "exominer_tce_period",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub exominer_tce_period: Option<f64>,
    #[serde(
        rename = "exominer_sector_run",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub exominer_sector_run: Option<String>,
    #[serde(
        rename = "exominer_ffi_score",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub exominer_ffi_score: Option<f64>,
    #[serde(
        rename = "exominer_ffi_model_version",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub exominer_ffi_model_version: Option<String>,
    #[serde(rename = "cross_source_name", default, skip_serializing_if = "Option::is_none")]
    pub cross_source_name: Option<String>,
    #[serde(rename = "cross_source_snr", default, skip_serializing_if = "Option::is_none")]
    pub cross_source_snr: Option<f64>,
    #[serde(
        rename = "cross_source_depth_ppm",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub cross_source_depth_ppm: Option<f64>,
    #[serde(
        rename = "cross_source_depth_ratio",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub cross_source_depth_ratio: Option<f64>,
    #[serde(
        rename = "cross_source_agreement",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub cross_source_agreement: Option<bool>,
}

impl CandidateResult {
    /// Mirrors the Swift convenience `init(candidateNumber:periodDays:...)`
    /// (the six required fields + the two earliest-added optionals).
    pub fn new(
        candidate_number: i64,
        period_days: f64,
        t0: f64,
        duration_days: f64,
        depth_ppm: f64,
        snr: f64,
    ) -> Self {
        CandidateResult {
            candidate_number,
            period_days,
            t0,
            duration_days,
            depth_ppm,
            snr,
            bls_power: None,
            expected_transits: None,
            triceratops_fpp: None,
            triceratops_nfpp: None,
            nfpp: None,
            triceratops_probability_planet: None,
            triceratops_verdict: None,
            exominer_score: None,
            exominer_tce_period: None,
            exominer_sector_run: None,
            exominer_ffi_score: None,
            exominer_ffi_model_version: None,
            cross_source_name: None,
            cross_source_snr: None,
            cross_source_depth_ppm: None,
            cross_source_depth_ratio: None,
            cross_source_agreement: None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VettingResult {
    #[serde(rename = "is_harmonic", default, skip_serializing_if = "Option::is_none")]
    pub is_harmonic: Option<bool>,
    #[serde(rename = "harmonic_of", default, skip_serializing_if = "Option::is_none")]
    pub harmonic_of: Option<i64>,
    #[serde(rename = "harmonic_ratio", default, skip_serializing_if = "Option::is_none")]
    pub harmonic_ratio: Option<f64>,
    #[serde(
        rename = "odd_even_consistent",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub odd_even_consistent: Option<bool>,
    #[serde(rename = "odd_depth", default, skip_serializing_if = "Option::is_none")]
    pub odd_depth: Option<f64>,
    #[serde(rename = "even_depth", default, skip_serializing_if = "Option::is_none")]
    pub even_depth: Option<f64>,
    #[serde(rename = "is_v_shaped", default, skip_serializing_if = "Option::is_none")]
    pub is_v_shaped: Option<bool>,
    #[serde(rename = "v_metric", default, skip_serializing_if = "Option::is_none")]
    pub v_metric: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub passed: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletenessCell {
    #[serde(rename = "radius_earth")]
    pub radius_earth: f64,
    #[serde(rename = "period_days")]
    pub period_days: f64,
    #[serde(rename = "depth_ppm", default, skip_serializing_if = "Option::is_none")]
    pub depth_ppm: Option<f64>,
    #[serde(rename = "detection_prob")]
    pub detection_probability: f64,
    #[serde(rename = "n_recovered", default, skip_serializing_if = "Option::is_none")]
    pub recovered_count: Option<i64>,
    #[serde(rename = "n_injected", default, skip_serializing_if = "Option::is_none")]
    pub injected_count: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletenessResult {
    #[serde(rename = "radius_grid", default, skip_serializing_if = "Option::is_none")]
    pub radius_grid: Option<Vec<f64>>,
    #[serde(rename = "period_grid", default, skip_serializing_if = "Option::is_none")]
    pub period_grid: Option<Vec<f64>>,
    #[serde(
        rename = "completeness_map",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub completeness_map: Option<Vec<Vec<f64>>>,
    pub details: Vec<CompletenessCell>,
    #[serde(rename = "stellar_radius", default, skip_serializing_if = "Option::is_none")]
    pub stellar_radius: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncated: Option<bool>,
    #[serde(
        rename = "periods_evaluated",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub periods_evaluated: Option<i64>,
    #[serde(rename = "radii_evaluated", default, skip_serializing_if = "Option::is_none")]
    pub radii_evaluated: Option<i64>,
}

// ---------------------------------------------------------------------------
// PrimaryResultRequest
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrimaryResultRequest {
    #[serde(rename = "task_id", default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    #[serde(rename = "tic_id", default, skip_serializing_if = "Option::is_none")]
    pub tic_id: Option<String>,
    #[serde(default)]
    pub candidates: Vec<CandidateResult>,
    #[serde(rename = "vetting_results", default)]
    pub vetting_results: Vec<VettingResult>,
    #[serde(rename = "cnn_scores", default)]
    pub cnn_scores: Vec<Option<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completeness: Option<CompletenessResult>,
    #[serde(rename = "processing_time_seconds")]
    pub processing_time_seconds: f64,
    #[serde(rename = "gpu_used")]
    pub gpu_used: bool,
    #[serde(rename = "n_sectors")]
    pub sector_count: i64,
    #[serde(rename = "n_points")]
    pub point_count: i64,
    #[serde(rename = "time_span_days", default, skip_serializing_if = "Option::is_none")]
    pub time_span_days: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ra: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dec: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tmag: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timings: Option<Map<String, Value>>,
    #[serde(rename = "lc_load_recovered")]
    pub light_curve_load_recovered: bool,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Provenance: the effective alias phase-2 flag this result was computed
    /// with (python result key `harmonic_epoch_class`; set on completed
    /// results only, `None` = key absent, as in python's load_failed result).
    #[serde(rename = "harmonic_epoch_class", default, skip_serializing_if = "Option::is_none")]
    pub harmonic_epoch_class: Option<bool>,
}

impl PrimaryResultRequest {
    /// Mirrors the Swift memberwise `init` defaults: `status: "completed"`,
    /// empty arrays, zeroed counters, everything else `nil`/`false`.
    pub fn new(task_id: Option<String>, tic_id: Option<String>) -> Self {
        PrimaryResultRequest {
            task_id,
            tic_id,
            candidates: Vec::new(),
            vetting_results: Vec::new(),
            cnn_scores: Vec::new(),
            completeness: None,
            processing_time_seconds: 0.0,
            gpu_used: false,
            sector_count: 0,
            point_count: 0,
            time_span_days: None,
            ra: None,
            dec: None,
            tmag: None,
            timings: None,
            light_curve_load_recovered: false,
            status: "completed".to_string(),
            error: None,
            harmonic_epoch_class: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Backfill (FPP / ExoMiner) task envelope
// ---------------------------------------------------------------------------

/// Decode-only, like the Swift `Decodable`-only (not full `Codable`) type.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct BackfillTaskEnvelope {
    pub task: Option<BackfillTask>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct BackfillTask {
    #[serde(rename = "candidate_id")]
    pub candidate_id: i64,
    #[serde(rename = "tic_id", deserialize_with = "deserialize_flexible_string")]
    pub tic_id: String,
    #[serde(rename = "candidate_num")]
    pub candidate_number: i64,
    #[serde(rename = "period_days")]
    pub period_days: f64,
    pub t0: f64,
    #[serde(rename = "duration_days")]
    pub duration_days: f64,
    #[serde(rename = "depth_ppm")]
    pub depth_ppm: f64,
    pub snr: f64,
    /// Swift: `decodeIfPresent(Int.self, forKey: .sectorCount) ?? 1`.
    #[serde(rename = "n_sectors", default = "default_sector_count")]
    pub sector_count: i64,
    /// Immutable server-shipped stellar population and its digest.
    #[serde(default)]
    pub population: Option<Value>,
    #[serde(default)]
    pub population_sha256: Option<String>,
    #[serde(default)]
    pub population_version: Option<String>,
    /// Cross-check metadata is separate from the hashed document.
    #[serde(default)]
    pub population_crosscheck: Option<Value>,
    #[serde(default)]
    pub cadence_s: Option<f64>,
    #[serde(default)]
    pub geometry: Option<Value>,
    #[serde(default)]
    pub lc_source: Option<String>,
    #[serde(default)]
    pub tglc_near_edge: Option<bool>,
}

fn default_sector_count() -> i64 {
    1
}

fn deserialize_nullable_string<'de, D>(deserializer: D) -> Result<Option<Option<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer).map(Some)
}

fn deserialize_nullable_bool<'de, D>(deserializer: D) -> Result<Option<Option<bool>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<bool>::deserialize(deserializer).map(Some)
}

// ---------------------------------------------------------------------------
// FPP / ExoMiner result payloads
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FppResultRequest {
    #[serde(rename = "candidate_id", default, skip_serializing_if = "Option::is_none")]
    pub candidate_id: Option<i64>,
    #[serde(rename = "tic_id", default, skip_serializing_if = "Option::is_none")]
    pub tic_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fpp: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nfpp: Option<f64>,
    #[serde(rename = "prob_planet", default, skip_serializing_if = "Option::is_none")]
    pub probability_planet: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verdict: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fpp_scen: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fpp_flags: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fpp_model_version: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry_version: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dilution_convention: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aperture_source: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth_scale: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub population_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub population_source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub population_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub population_generator: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub population_mapping_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_comp: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_comp_fainter: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kepler_unconverged_draws: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dilution_corrected: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dilution_unverified: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dilution_alternatives: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lc_source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fpp_backend: Option<String>,
    /// Platform of the computing binary, `<os>-<arch>` (plan (t) 2 provenance).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fpp_platform: Option<String>,
    /// C math library identity and version, e.g. `glibc-2.43`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fpp_libm: Option<String>,
    /// `debug` or `release` build of the computing binary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fpp_build_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fpp_seed: Option<u64>,
    // ---- step 7 (decision C) Monte-Carlo noise fields, `worker._fpp_result_model_fields` (task 26 / 33)
    /// MC standard error of the reported FPP; omitted when non-finite.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fpp_mc_se: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nfpp_mc_se: Option<f64>,
    /// Draws per scenario of the reported stage (`n_draws_final`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fpp_n_draws: Option<u64>,
    /// Draws per scenario summed over the stages run (`n_draws_total`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fpp_n_draws_total: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mc_min_ess: Option<f64>,
    /// Present only when the MC policy ran (`mc_policy`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mc_borderline: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mc_detail: Option<Value>,
    /// WHY a `-1` / `failed` result has no score (Python `calc_fpp_by_source.last_skip_reason`); `Some(None)`
    /// posts an explicit JSON `null`, as the Python worker does. Never on a computed result.
    #[serde(
        default,
        deserialize_with = "deserialize_nullable_string",
        skip_serializing_if = "Option::is_none"
    )]
    pub skip_reason: Option<Option<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fpp_switches: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub high_crowding: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gal_b: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_nearby: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trilegal_used: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry_model_version: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry_coverage: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry_n_sectors: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry_n_task_sectors: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry_source: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry_path: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry_path_emulated: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry_n_epochs: Option<Vec<usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry_sigma_cal: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_nodes: Option<usize>,
    #[serde(
        default,
        deserialize_with = "deserialize_nullable_bool",
        skip_serializing_if = "Option::is_none"
    )]
    pub tglc_near_edge: Option<Option<bool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_viable_neighbours: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_viability_weighted: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub viability_mass_skipped: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fluxratio_target: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sigma_source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sigma_fit_ill_conditioned: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure_unconverged: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure_unconverged_share: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure_gl_fallback_share: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cadence_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tglc_near_edge_unknown: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub population_crosscheck_status: Option<String>,
    // ---- ephemeris refinement provenance fields (section 8b, FPP_EPHEM_REFINE)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ephem_refined: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ephem_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ephem_t0_convention: Option<String>,
    #[serde(rename = "ephem_P", default, skip_serializing_if = "Option::is_none")]
    pub ephem_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ephem_t0: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ephem_shift_h: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ephem_refine: Option<crate::fpp::ephem::RefineProvenanceFields>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ephem_refine_failed: Option<bool>,
}

impl FppResultRequest {
    pub fn new(candidate_id: Option<i64>, tic_id: Option<String>, fpp: Option<f64>) -> Self {
        FppResultRequest {
            candidate_id,
            tic_id,
            fpp,
            nfpp: None,
            probability_planet: None,
            verdict: None,
            candidate: None,
            fpp_scen: None,
            fpp_flags: None,
            fpp_model_version: None,
            geometry_version: None,
            dilution_convention: None,
            aperture_source: None,
            depth_scale: None,
            population_version: None,
            population_source: None,
            population_sha256: None,
            population_generator: None,
            population_mapping_version: None,
            n_comp: None,
            n_comp_fainter: None,
            kepler_unconverged_draws: None,
            dilution_corrected: None,
            dilution_unverified: None,
            dilution_alternatives: None,
            lc_source: None,
            fpp_backend: None,
            fpp_platform: None,
            fpp_libm: None,
            fpp_build_profile: None,
            fpp_seed: None,
            fpp_mc_se: None,
            nfpp_mc_se: None,
            fpp_n_draws: None,
            fpp_n_draws_total: None,
            mc_min_ess: None,
            mc_borderline: None,
            mc_detail: None,
            skip_reason: None,
            fpp_switches: None,
            high_crowding: None,
            gal_b: None,
            n_nearby: None,
            trilegal_used: None,
            geometry_model_version: None,
            geometry_kind: None,
            geometry_mode: None,
            geometry_coverage: None,
            geometry_n_sectors: None,
            geometry_n_task_sectors: None,
            geometry_source: None,
            geometry_path: None,
            geometry_path_emulated: None,
            geometry_n_epochs: None,
            geometry_sigma_cal: None,
            n_nodes: None,
            tglc_near_edge: None,
            n_viable_neighbours: None,
            n_viability_weighted: None,
            viability_mass_skipped: None,
            fluxratio_target: None,
            sigma_source: None,
            sigma_fit_ill_conditioned: None,
            exposure_unconverged: None,
            exposure_unconverged_share: None,
            exposure_gl_fallback_share: None,
            cadence_s: None,
            tglc_near_edge_unknown: None,
            population_crosscheck_status: None,
            ephem_refined: None,
            ephem_status: None,
            ephem_t0_convention: None,
            ephem_p: None,
            ephem_t0: None,
            ephem_shift_h: None,
            ephem_refine: None,
            ephem_refine_failed: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExoMinerResultRequest {
    #[serde(rename = "candidate_id", default, skip_serializing_if = "Option::is_none")]
    pub candidate_id: Option<i64>,
    #[serde(rename = "tic_id", default, skip_serializing_if = "Option::is_none")]
    pub tic_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    #[serde(rename = "tce_period", default, skip_serializing_if = "Option::is_none")]
    pub tce_period: Option<f64>,
    #[serde(rename = "sector_run", default, skip_serializing_if = "Option::is_none")]
    pub sector_run: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatusResponse {
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
}

// ---------------------------------------------------------------------------
// FFI / nearby-star / metadata payloads
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct FfiSector {
    pub sector: i64,
    pub source: String,
    pub time: Vec<f64>,
    pub flux: Vec<f64>,
    #[serde(default)]
    pub quality: Option<Vec<i64>>,
}

impl FfiSector {
    /// Mirrors the computed property `effectiveQuality`.
    pub fn effective_quality(&self) -> Vec<i64> {
        self.quality
            .clone()
            .unwrap_or_else(|| vec![0; self.time.len()])
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct FfiResponse {
    #[serde(rename = "tic_id", deserialize_with = "deserialize_flexible_string")]
    pub tic_id: String,
    pub sectors: Vec<FfiSector>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct NearbyStar {
    #[serde(rename = "ID", deserialize_with = "deserialize_flexible_string")]
    pub id: String,
    pub ra: f64,
    pub dec: f64,
    #[serde(default)]
    pub mass: Option<f64>,
    #[serde(rename = "rad", default)]
    pub radius: Option<f64>,
    #[serde(rename = "Teff", default)]
    pub effective_temperature: Option<f64>,
    #[serde(rename = "Tmag", default)]
    pub tess_magnitude: Option<f64>,
    #[serde(rename = "plx", default)]
    pub parallax: Option<f64>,
    #[serde(rename = "stellar_source", default)]
    pub stellar_source: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct NearbyResponse {
    #[serde(rename = "tic_id", deserialize_with = "deserialize_flexible_string")]
    pub tic_id: String,
    #[serde(default)]
    pub stars: Vec<NearbyStar>,
    /// Swift: `decodeIfPresent(Double.self, forKey: .galacticLatitude) ?? 0`.
    #[serde(rename = "gal_b", default)]
    pub galactic_latitude: f64,
    #[serde(default)]
    pub source: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpdateManifestFile {
    pub filename: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpdateManifest {
    pub files: Vec<UpdateManifestFile>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct StarMetadata {
    /// Swift: `if c.contains(.ticID) { ticID = try? decodeFlexibleString(...) }
    /// else { ticID = nil }` — missing key or unparseable value both become
    /// `None`, reproduced by `deserialize_flexible_string_opt` + `default`.
    #[serde(rename = "tic_id", default, deserialize_with = "deserialize_flexible_string_opt")]
    pub tic_id: Option<String>,
    #[serde(rename = "fits_files", default)]
    pub fits_files: Vec<String>,
    #[serde(default)]
    pub sectors: Vec<i64>,
    #[serde(default)]
    pub ra: Option<f64>,
    #[serde(default)]
    pub dec: Option<f64>,
    #[serde(default)]
    pub tmag: Option<f64>,
    #[serde(rename = "teff", default)]
    pub effective_temperature: Option<f64>,
    #[serde(default)]
    pub logg: Option<f64>,
    #[serde(rename = "stellar_radius", default)]
    pub stellar_radius: Option<f64>,
    #[serde(rename = "stellar_mass", default)]
    pub stellar_mass: Option<f64>,
    #[serde(rename = "data_source", default)]
    pub data_source: Option<String>,
}

/// Domain enum built programmatically from an HTTP response's status code
/// and body inside `WorkerAPIClient.fetchTFRecord` (F0b `api.rs`) — not a
/// JSON payload shape itself, so (like its Swift counterpart) it carries no
/// `serde` derive.
#[derive(Debug, Clone, PartialEq)]
pub enum TfRecordFetchResult {
    Ready { data: Vec<u8>, sector_run: String, tic_id: String },
    Preprocessing { tic_id: String, sector_run: String },
    NotAvailable,
}

// ---------------------------------------------------------------------------
// Register / task / heartbeat wire types, from WorkerAPI.swift.
// The HTTP client itself (`WorkerAPIClient`) is F0b (`api.rs`); these are
// only the request/response payload shapes it will serialize/deserialize.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerCapabilities {
    pub gpu: String,
    #[serde(rename = "bls_backend")]
    pub bls_backend: String,
    #[serde(rename = "bls_gpu_info")]
    pub bls_gpu_info: String,
    #[serde(rename = "fpp_backend")]
    pub fpp_backend: String,
    #[serde(rename = "cpu_cores")]
    pub cpu_cores: i64,
    #[serde(rename = "fpp_cpu_cores")]
    pub fpp_cpu_cores: i64,
    #[serde(default)]
    pub hardware: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegisterRequest {
    pub name: String,
    pub capabilities: WorkerCapabilities,
}

/// Private in Swift (`WorkerAPI.swift`'s `RegisterResponse`); made `pub` here
/// since `api.rs` (F0b) will live in a sibling module rather than the same
/// file.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RegisterResponseBody {
    #[serde(rename = "worker_id")]
    pub worker_id: String,
}

/// Domain type — not itself `Codable` in Swift (only its wire counterpart
/// `WorkerTaskResponse` is), built from a decoded response.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerTask {
    pub task_id: String,
    pub tic_id: String,
    pub config: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TaskPollResult {
    Task(WorkerTask),
    Wait { seconds: Option<f64> },
}

/// Mirrors the private `WorkerTaskResponse` decode target, including its
/// custom `init(from decoder:)` (flexible `tic_id`, default-empty `config`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct WorkerTaskResponse {
    #[serde(rename = "task_id", default)]
    pub task_id: Option<String>,
    #[serde(rename = "tic_id", default, deserialize_with = "deserialize_flexible_string_opt")]
    pub tic_id: Option<String>,
    #[serde(rename = "wait_seconds", default)]
    pub wait_seconds: Option<f64>,
    #[serde(default, deserialize_with = "deserialize_config_or_empty")]
    pub config: Map<String, Value>,
}

/// `config: null` (or absent) is an empty task config, like Swift's
/// `decodeIfPresent(...) ?? [:]` and python's `task.get("config") or {}`.
fn deserialize_config_or_empty<'de, D>(deserializer: D) -> Result<Map<String, Value>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<Map<String, Value>>::deserialize(deserializer)?.unwrap_or_default())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeartbeatRequest {
    #[serde(rename = "worker_id")]
    pub worker_id: String,
    pub status: String,
    #[serde(rename = "current_tic", default, skip_serializing_if = "Option::is_none")]
    pub current_tic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    #[serde(rename = "phase_progress", default, skip_serializing_if = "Option::is_none")]
    pub phase_progress: Option<f64>,
    #[serde(rename = "star_elapsed_s", default, skip_serializing_if = "Option::is_none")]
    pub star_elapsed_seconds: Option<f64>,
    #[serde(rename = "stars_done")]
    pub stars_done: i64,
    #[serde(rename = "avg_time_per_star")]
    pub average_time_per_star: f64,
    #[serde(rename = "last_error", default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    #[serde(rename = "uptime_s")]
    pub uptime_seconds: f64,
    pub pid: i32,
    #[serde(rename = "git_sha", default, skip_serializing_if = "Option::is_none")]
    pub git_sha: Option<String>,
    #[serde(rename = "flag_states", default)]
    pub flag_states: Map<String, Value>,
    #[serde(rename = "gpu_fallback", default)]
    pub gpu_fallback: Map<String, Value>,
    #[serde(default)]
    pub system: Map<String, Value>,
    #[serde(rename = "gpu", default)]
    pub gpu_telemetry: Map<String, Value>,
    #[serde(rename = "recent_timings", default)]
    pub recent_timings: Vec<Value>,
}

impl HeartbeatRequest {
    /// Mirrors the Swift memberwise `init` required params + defaults.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        worker_id: String,
        status: String,
        current_tic: Option<String>,
        phase: Option<String>,
        stars_done: i64,
        uptime_seconds: f64,
        pid: i32,
    ) -> Self {
        HeartbeatRequest {
            worker_id,
            status,
            current_tic,
            phase,
            phase_progress: None,
            star_elapsed_seconds: None,
            stars_done,
            average_time_per_star: 0.0,
            last_error: None,
            uptime_seconds,
            pid,
            git_sha: None,
            flag_states: Map::new(),
            gpu_fallback: Map::new(),
            system: Map::new(),
            gpu_telemetry: Map::new(),
            recent_timings: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeartbeatCommand {
    #[serde(rename = "continue")]
    Continue,
    #[serde(rename = "re-register")]
    ReRegister,
    #[serde(rename = "pause")]
    Pause,
    #[serde(rename = "resume")]
    Resume,
    #[serde(rename = "shutdown")]
    Shutdown,
    #[serde(rename = "debug_on")]
    DebugOn,
    #[serde(rename = "debug_off")]
    DebugOff,
}

/// Private in Swift (`WorkerAPI.swift`'s `HeartbeatResponse`); made `pub`
/// here for the same reason as `RegisterResponseBody`. Note `command` stays
/// a raw `Option<String>` (not `Option<HeartbeatCommand>`): the server's
/// heartbeat command is looked up leniently ("no command" = continue) and
/// an *unknown* command string must surface as a client-side
/// `WorkerAPIError::malformed_response` (F0b, `api.rs`) rather than a decode
/// failure — see `testUnknownHeartbeatCommand` in the Swift contract suite.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct HeartbeatResponseBody {
    #[serde(default)]
    pub command: Option<String>,
}
