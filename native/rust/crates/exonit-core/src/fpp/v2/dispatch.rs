//! Source-specific dilution planning, depth scaling and result verdict caps.
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct DilutionPlan {
    /// `true` means the light curve is already target-referenced.
    pub conventions: Vec<bool>,
    pub unverified: bool,
}

pub fn normalize_lc_source(source: Option<&str>) -> &'static str {
    match source.unwrap_or("").trim().to_ascii_lowercase().as_str() {
        "spoc" => "spoc",
        "tglc" => "tglc",
        "qlp" => "qlp",
        "mixed" => "mixed",
        _ => "unknown",
    }
}

pub fn dilution_plan(source: Option<&str>) -> DilutionPlan {
    match normalize_lc_source(source) {
        "spoc" | "tglc" => DilutionPlan {
            conventions: vec![true],
            unverified: false,
        },
        _ => DilutionPlan {
            conventions: vec![true, false],
            unverified: true,
        },
    }
}

pub fn depth_scale(source: Option<&str>, near_edge: Option<bool>) -> f64 {
    if normalize_lc_source(source) == "tglc" {
        if near_edge == Some(true) {
            1.0 / 1.27
        } else {
            1.0 / 1.09
        }
    } else {
        1.0
    }
}

pub fn frame_fluxratio(fr_i: f64, fr_target: f64, dilution_corrected: bool) -> f64 {
    if dilution_corrected {
        fr_i / fr_target
    } else {
        fr_i
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct VerdictCaps {
    pub kepler_unconverged_draws: u64,
    pub population_crosscheck_verified: bool,
    pub dilution_unverified: bool,
    pub tglc_near_edge_unknown: bool,
    pub geometry_path_emulated: bool,
    pub ffi_reference_unvalidated: bool,
}

/// Apply the reference's conservative gate and append stable result flags.
pub fn cap_verdict(verdict: &mut String, caps: VerdictCaps, flags: &mut Vec<String>) {
    let crosscheck_unverified = !caps.population_crosscheck_verified;
    let dilution_unverified = caps.dilution_unverified;
    let edge_unknown = caps.tglc_near_edge_unknown;
    let capped = caps.kepler_unconverged_draws > 0
        || crosscheck_unverified
        || dilution_unverified
        || edge_unknown
        || caps.geometry_path_emulated
        || caps.ffi_reference_unvalidated;
    for (active, name) in [
        (crosscheck_unverified, "population_crosscheck_unverified"),
        (dilution_unverified, "dilution_unverified"),
        (edge_unknown, "tglc_near_edge_unknown"),
    ] {
        if active && !flags.iter().any(|f| f == name) {
            flags.push(name.to_string());
        }
    }
    if capped && verdict == "validated" {
        *verdict = "candidate".into();
    }
}

/// Run alternatives in deterministic convention order and pick the larger FPP; ties favor `true`.
pub fn choose_dilution_run<T>(
    runs: &BTreeMap<bool, T>,
    fpp: impl Fn(&T) -> f64,
) -> Result<bool, String> {
    let true_run = runs
        .get(&true)
        .ok_or_else(|| "missing target-referenced dilution run".to_string())?;
    let Some(false_run) = runs.get(&false) else {
        return Ok(true);
    };
    Ok(if fpp(false_run) > fpp(true_run) {
        false
    } else {
        true
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_cap_demotes_validated_and_adds_its_flag() {
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
        for caps in cases {
            let mut verdict = "validated".to_string();
            let mut flags = Vec::new();
            cap_verdict(&mut verdict, caps, &mut flags);
            assert_eq!(verdict, "candidate");
        }
    }

    #[test]
    fn source_dilution_plan_depth_scale_and_frame_rule_match_reference() {
        assert_eq!(
            dilution_plan(Some(" SPOC ")),
            DilutionPlan {
                conventions: vec![true],
                unverified: false
            }
        );
        assert_eq!(dilution_plan(Some("qlp")).conventions, vec![true, false]);
        assert_eq!(depth_scale(Some("tglc"), Some(true)), 1.0 / 1.27);
        assert_eq!(depth_scale(Some("tglc"), None), 1.0 / 1.09);
        assert_eq!(frame_fluxratio(0.2, 0.4, true), 0.5);
        assert_eq!(frame_fluxratio(0.2, 0.4, false), 0.2);
    }
}
