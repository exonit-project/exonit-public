//! Validation and derived D/B inputs for the immutable task-shipped field population.
use crate::fpp::rng::FppRng;
use serde_json::Value;

const COLUMNS: [&str; 5] = ["Tmag", "Mact", "logg", "logTe", "[M/H]"];
const GALAXIA_MAPPINGS: [&str; 2] = ["tmag-stassun2018-jks-cc1", "tmag-stassun2018-jks-v1"];

#[derive(Clone, Debug)]
pub struct BackgroundStar {
    pub tmag: f64,
    pub mass: f64,
    pub logg: f64,
    pub log_te: f64,
    pub metallicity: f64,
    pub teff: f64,
    pub radius: f64,
    pub flux_ratio: f64,
}

#[derive(Clone, Debug)]
pub struct BackgroundPopulation {
    /// This is the post-inclusive-filter row count used by the prior and sampler.
    pub stars: Vec<BackgroundStar>,
    pub source: String,
    pub generator: String,
    pub version: String,
    pub mapping_version: String,
    pub sha256: String,
    /// Original document row count.
    pub n_comp: usize,
    /// Count after `Tmag >= target Tmag`.
    pub n_comp_fainter: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PopulationError {
    /// Typed release reason, set where the condition is detected.
    pub reason: crate::fpp::error::FppUnavailableReason,
    pub message: String,
}
impl std::fmt::Display for PopulationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}
impl std::error::Error for PopulationError {}
fn invalid(message: impl Into<String>) -> PopulationError {
    PopulationError {
        reason: crate::fpp::error::FppUnavailableReason::PopulationInvalid,
        message: message.into(),
    }
}

fn python_float_repr(x: f64) -> String {
    // Rust's Debug formatter uses the same shortest-roundtrip digits as
    // Python repr; normalize only exponent spelling (`e+16`, zero padding).
    let s = format!("{x:?}");
    let Some((mantissa, exponent)) = s.split_once('e') else {
        return s;
    };
    let exponent: i32 = exponent.parse().expect("float exponent");
    format!(
        "{mantissa}e{}{:02}",
        if exponent < 0 { "-" } else { "+" },
        exponent.abs()
    )
}

fn push_python_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c <= '\u{001f}' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if c.is_ascii() => out.push(c),
            c if (c as u32) <= 0xffff => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => {
                let n = c as u32 - 0x10000;
                let hi = 0xd800 + (n >> 10);
                let lo = 0xdc00 + (n & 0x3ff);
                out.push_str(&format!("\\u{hi:04x}\\u{lo:04x}"));
            }
        }
    }
    out.push('"');
}

fn canonical_value(value: &Value, out: &mut String) -> Result<(), PopulationError> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::String(s) => push_python_string(out, s),
        Value::Number(n) if n.is_i64() || n.is_u64() => out.push_str(&n.to_string()),
        Value::Number(n) => {
            let f = n
                .as_f64()
                .ok_or_else(|| invalid("population contains an invalid number"))?;
            if !f.is_finite() {
                return Err(invalid("Out of range float values are not JSON compliant"));
            }
            out.push_str(&python_float_repr(f));
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i != 0 {
                    out.push(',');
                }
                canonical_value(item, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            // serde_json::Map is a BTreeMap here, matching Python sort_keys.
            for (i, (key, val)) in map.iter().enumerate() {
                if i != 0 {
                    out.push(',');
                }
                push_python_string(out, key);
                out.push(':');
                canonical_value(val, out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

fn canonical_json(doc: &Value) -> Result<String, PopulationError> {
    let mut out = String::new();
    canonical_value(doc, &mut out)?;
    Ok(out)
}
fn sha256(s: &str) -> String {
    let mut h = hmac_sha256::Hash::new();
    h.update(s.as_bytes());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
fn get<'a>(v: &'a Value, k: &str) -> Result<&'a Value, PopulationError> {
    v.get(k).ok_or_else(|| invalid(format!("missing {k}")))
}
fn text(v: &Value, name: &str) -> Result<String, PopulationError> {
    v.as_str()
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .ok_or_else(|| invalid(format!("invalid {name}")))
}
fn num(v: &Value, name: &str) -> Result<f64, PopulationError> {
    v.as_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| invalid(format!("invalid {name}")))
}

/// Validate the complete canonical document and prepare target-fainter rows.
pub fn validate_population(
    doc: Option<&Value>,
    target_tmag: f64,
    expected_sha256: Option<&str>,
    expected_version: Option<&str>,
    expected_tic_id: Option<&str>,
) -> Result<BackgroundPopulation, PopulationError> {
    let doc = doc.ok_or_else(|| PopulationError {
        reason: crate::fpp::error::FppUnavailableReason::PopulationMissing,
        message: "background population is missing".into(),
    })?;
    if !target_tmag.is_finite() {
        return Err(invalid("invalid target magnitude"));
    }
    let header = get(doc, "header")?;
    if let Some(tic) = expected_tic_id {
        if get(header, "tic_id")?.to_string().trim_matches('"') != tic {
            return Err(invalid("population belongs to a different TIC"));
        }
    }
    let generator = text(get(header, "source")?, "source")?;
    let is_galaxia = matches!(generator.as_str(), "galaxia-0.7.2" | "galaxia-rs-0.2.2");
    if !is_galaxia && generator != "gaia" {
        return Err(invalid("unsupported population source"));
    }
    let cols = get(header, "columns")?
        .as_array()
        .ok_or_else(|| invalid("population columns must be an array"))?;
    if cols.len() != COLUMNS.len() || cols.iter().zip(COLUMNS).any(|(v, w)| v.as_str() != Some(w)) {
        return Err(invalid("population columns differ from contract"));
    }
    let version = text(get(header, "population_version")?, "population_version")?;
    let mapping = text(get(header, "mapping_version")?, "mapping_version")?;
    if is_galaxia && !GALAXIA_MAPPINGS.contains(&mapping.as_str()) {
        return Err(invalid("unknown Tmag mapping"));
    }
    if num(get(header, "field_area_deg2")?, "field_area_deg2")? != 0.1
        || num(get(header, "tmag_limit")?, "tmag_limit")? != 21.0
    {
        return Err(invalid("unsupported field area or magnitude limit"));
    }
    if expected_version.is_some_and(|e| e != version) {
        return Err(invalid("population version mismatch"));
    }
    let digest = sha256(&canonical_json(doc)?);
    if expected_sha256.is_some_and(|e| e != digest) {
        return Err(invalid("population sha256 mismatch"));
    }
    let rows = get(doc, "rows")?
        .as_array()
        .ok_or_else(|| invalid("population rows must be an array"))?;
    let n_comp = get(header, "n_comp")?
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or_else(|| invalid("empty/invalid population count"))? as usize;
    if rows.len() != n_comp {
        return Err(invalid("invalid population rows"));
    }
    let mut all = Vec::with_capacity(rows.len());
    for row in rows {
        let row = row
            .as_array()
            .ok_or_else(|| invalid("population row must be an array"))?;
        if row.len() != 5 {
            return Err(invalid("population row must contain five columns"));
        }
        let vals = [
            num(&row[0], "Tmag")?,
            num(&row[1], "Mact")?,
            num(&row[2], "logg")?,
            num(&row[3], "logTe")?,
            num(&row[4], "[M/H]")?,
        ];
        if vals[1] <= 0.0 || vals[0] > 21.0 {
            return Err(invalid("invalid magnitude or stellar mass"));
        }
        all.push(vals);
    }
    if all.windows(2).any(|p| {
        for i in 0..5 {
            if p[0][i] < p[1][i] {
                return false;
            }
            if p[0][i] > p[1][i] {
                return true;
            }
        }
        false
    }) {
        return Err(invalid("population is not canonically sorted"));
    }
    let mut stars = Vec::new();
    for r in all.into_iter().filter(|r| r[0] >= target_tmag) {
        let teff = crate::fpp::npsum::libm_pow(10.0, r[3]);
        let radius = (super::consts::G_CGS * r[1] * super::consts::MSUN
            / crate::fpp::npsum::libm_pow(10.0, r[2]))
        .sqrt()
            / super::consts::RSUN;
        let relative = crate::fpp::npsum::libm_pow(10.0, (target_tmag - r[0]) / 2.5);
        let flux_ratio = relative / (1.0 + relative);
        if !teff.is_finite() || teff <= 0.0 || !radius.is_finite() || radius <= 0.0 {
            return Err(invalid("invalid population stellar properties"));
        }
        stars.push(BackgroundStar {
            tmag: r[0],
            mass: r[1],
            logg: r[2],
            log_te: r[3],
            metallicity: r[4],
            teff,
            radius,
            flux_ratio,
        });
    }
    if stars.is_empty() {
        return Err(invalid("no target-fainter population rows"));
    }
    Ok(BackgroundPopulation {
        n_comp,
        n_comp_fainter: stars.len(),
        source: if generator == "gaia" {
            "gaia".into()
        } else {
            "galaxia".into()
        },
        generator,
        version,
        mapping_version: mapping,
        sha256: digest,
        stars,
    })
}

/// `randint(0, high, n)` with the upstream parity exclusion for D draws.
pub fn sample_background_indices(
    n: usize,
    n_comp: usize,
    parity_upstream: bool,
    rng: &mut FppRng,
) -> Result<Vec<usize>, PopulationError> {
    let high = if parity_upstream {
        n_comp.saturating_sub(1)
    } else {
        n_comp
    };
    if high == 0 {
        return Err(invalid(
            "population too small for upstream D-family indexing",
        ));
    }
    Ok((0..n).map(|_| rng.randint(0, high)).collect())
}

/// Sequential scalar form of the documented D/B area, magnitude and prior calculation.
pub fn background_prior(
    n_comp: usize,
    flux_ratio: f64,
    flux_eb: Option<f64>,
    circular: bool,
) -> f64 {
    // Python scalar `** 2` calls libm pow; it is not an `x * x` expression.
    let mut area =
        (n_comp as f64 / 0.1) * super::pow_libm(1.0 / 3600.0, 2.0) * super::pow_libm(2.2, 2.0);
    if circular {
        area *= std::f64::consts::PI;
    }
    let mut prior = area.ln();
    if prior > 0.0 {
        prior = 0.0;
    }
    let mut ratio = flux_ratio / (1.0 - flux_ratio);
    if let Some(f) = flux_eb {
        ratio += f / (1.0 - f);
    }
    let dm = 2.5 * ratio.log10();
    if dm > 0.0 || !dm.is_finite() {
        f64::NEG_INFINITY
    } else {
        prior
    }
}

pub fn crosscheck_verified(crosscheck: Option<&Value>) -> bool {
    crosscheck
        .and_then(|v| v.get("gaia_available"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

#[cfg(test)]
mod canonical_tests {
    use super::canonical_json;
    use serde_json::Value;

    #[test]
    fn canonical_form_uses_python_float_and_ascii_json_spelling() {
        let value: Value = serde_json::from_str(
            r#"{"z":1e-5,"a":1e-4,"large":1e16,"integer_float":1.0,"unicode":"é😀"}"#,
        )
        .unwrap();
        assert_eq!(
            canonical_json(&value).unwrap(),
            r#"{"a":0.0001,"integer_float":1.0,"large":1e+16,"unicode":"\u00e9\ud83d\ude00","z":1e-05}"#
        );
    }
}
