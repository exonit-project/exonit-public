//! Stellar mass→(radius, Teff, flux) relations and TESS limb-darkening
//! lookup — transliteration of `worker/fpp_native.py:437-538`.
//!
//! The FITPACK spline coefficients are embedded bit-for-bit
//! (`stellar_coeffs.rs`, generated) and only the `splev` evaluator (de
//! Boor / `fpbspl`) is ported; `fixtures/fpp/spline_golden.json` pins a
//! dense evaluation mesh per spline, including out-of-range extrapolation
//! (scipy `ext=0`) and the `10**x → inf` overflow of `flux_relation`.

use std::sync::OnceLock;

#[path = "stellar_coeffs.rs"]
mod stellar_coeffs;
use stellar_coeffs as sc;

fn bits(v: &[u64]) -> Vec<f64> {
    v.iter().copied().map(f64::from_bits).collect()
}

pub struct Spline {
    t: Vec<f64>,
    c: Vec<f64>,
    k: usize,
}

impl Spline {
    fn new(t_bits: &[u64], c_bits: &[u64], k: usize) -> Self {
        Spline {
            t: bits(t_bits),
            c: bits(c_bits),
            k,
        }
    }

    /// FITPACK `splev` with `e=0` (extrapolate from the boundary
    /// polynomial): locate `l` with `t[l] <= x < t[l+1]` clamped to the
    /// valid B-spline range, then de Boor (`fpbspl`) in the same
    /// operation order as the Fortran.
    pub fn eval(&self, x: f64) -> f64 {
        let k = self.k;
        let n = self.t.len();
        // 0-based valid interval for l: [k, n-k-2]
        let mut l = k;
        let hi = n - k - 2;
        while l < hi && x >= self.t[l + 1] {
            l += 1;
        }
        // fpbspl: non-zero B-splines h[0..=k] at x for interval l
        let mut h = [0.0f64; 8];
        let mut hh = [0.0f64; 8];
        h[0] = 1.0;
        for j in 1..=k {
            hh[..j].copy_from_slice(&h[..j]);
            h[0] = 0.0;
            for i in 0..j {
                let li = self.t[l + i + 1];
                let lj = self.t[l + i + 1 - j];
                if li == lj {
                    // fpbspl equal-knot guard (repeated boundary knots)
                    h[i + 1] = 0.0;
                    continue;
                }
                let f = hh[i] / (li - lj);
                h[i] += f * (li - x);
                h[i + 1] = f * (x - lj);
            }
        }
        let mut sp = 0.0;
        for j in 0..=k {
            sp += self.c[l - k + j] * h[j];
        }
        sp
    }
}

struct Splines {
    r_torres: Spline,
    t_torres: Spline,
    r_cdwrf: Spline,
    t_cdwrf: Spline,
    flux: Spline,
}

fn splines() -> &'static Splines {
    static S: OnceLock<Splines> = OnceLock::new();
    S.get_or_init(|| Splines {
        r_torres: Spline::new(sc::R_TORRES_T_BITS, sc::R_TORRES_C_BITS, sc::R_TORRES_K),
        t_torres: Spline::new(sc::T_TORRES_T_BITS, sc::T_TORRES_C_BITS, sc::T_TORRES_K),
        r_cdwrf: Spline::new(sc::R_CDWRF_T_BITS, sc::R_CDWRF_C_BITS, sc::R_CDWRF_K),
        t_cdwrf: Spline::new(sc::T_CDWRF_T_BITS, sc::T_CDWRF_C_BITS, sc::T_CDWRF_K),
        flux: Spline::new(sc::FLUX_T_BITS, sc::FLUX_C_BITS, sc::FLUX_K),
    })
}

/// `fpp_native.stellar_relations` — (radii, teffs) from masses in Msun.
pub fn stellar_relations(masses: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let s = splines();
    let mut r = Vec::with_capacity(masses.len());
    let mut t = Vec::with_capacity(masses.len());
    for &m in masses {
        let (ri, ti) = if m > 0.63 {
            (s.r_torres.eval(m), s.t_torres.eval(m))
        } else {
            let mc = m.clamp(0.1, 0.63);
            (s.r_cdwrf.eval(mc), s.t_cdwrf.eval(mc))
        };
        r.push(ri.max(0.1));
        t.push(ti.max(2800.0));
    }
    (r, t)
}

/// `fpp_native.flux_relation` — relative TESS-band flux (`10**spline`,
/// overflow to `inf` preserved).
pub fn flux_relation(masses: &[f64]) -> Vec<f64> {
    let s = splines();
    masses
        .iter()
        .map(|&m| crate::fpp::npsum::libm_pow(10.0, s.flux.eval(m)))
        .collect()
}

/// The embedded flux-spline exponent before conversion to linear relative flux.
/// This is exposed for numerical parity diagnostics; production callers should use [`flux_relation`].
#[doc(hidden)]
pub fn flux_relation_exponents(masses: &[f64]) -> Vec<f64> {
    let s = splines();
    masses.iter().map(|&m| s.flux.eval(m)).collect()
}

/// `fpp_native.stellar_relations_capped` (plan step 4): upstream `stellar_relations(Masses, max_Radii,
/// max_Teffs)`. Spline radius / Teff with the cool-dwarf spline EXTRAPOLATED (no `[0.1, 0.63]` clip, unlike
/// [`stellar_relations`]), then the host caps (`R > max_R -> max_R`, `T > max_T -> max_T`) and only THEN the
/// floors (`R >= 0.1`, `T >= 2800`).
pub fn stellar_relations_capped(
    masses: &[f64],
    max_r: &[f64],
    max_t: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let s = splines();
    let n = masses.len();
    let mut r = vec![0.0f64; n];
    let mut t = vec![0.0f64; n];
    for i in 0..n {
        let (ri, ti) = if masses[i] > 0.63 {
            (s.r_torres.eval(masses[i]), s.t_torres.eval(masses[i]))
        } else {
            (s.r_cdwrf.eval(masses[i]), s.t_cdwrf.eval(masses[i]))
        };
        r[i] = if ri > max_r[i] { max_r[i] } else { ri };
        t[i] = if ti > max_t[i] { max_t[i] } else { ti };
        if r[i] < 0.1 {
            r[i] = 0.1;
        }
        if t[i] < 2800.0 {
            t[i] = 2800.0;
        }
    }
    (r, t)
}

/// `fpp_native.ldc_companion_arrays` (plan step 4): per-draw limb darkening of the S family's host —
/// Teff rounded to 250 K and clipped to `[3500, teff_max]`, logg rounded to 0.5 and clipped to `[3.5, 5.0]`,
/// EXACT row of the table at the nearest Z. Returns `Err` (the Python raises `LDCNoRow`) when a row is missing.
pub fn ldc_companion_arrays(
    teff: &[f64],
    logg: &[f64],
    teff_max: f64,
) -> Result<(Vec<f64>, Vec<f64>), crate::fpp::error::FppError> {
    let tab = ldc_table();
    // nearest Z (single value for every draw), matching np.argmin(np.abs(Zs - Z))
    let zbest = tab.z[argmin_abs_diff(&tab.z, 0.0)];
    let n = teff.len();
    let mut u1 = vec![0.0f64; n];
    let mut u2 = vec![0.0f64; n];
    // cache exact rows by the rounded (Teff, logg) key, as the Python `seen` dict does (logg keyed by bits:
    // it is always an exact multiple of 0.5)
    let mut seen: std::collections::HashMap<(i64, u64), (f64, f64)> =
        std::collections::HashMap::new();
    for i in 0..n {
        let rl = ((logg[i] / 0.5).round_ties_even() * 0.5).clamp(3.5, 5.0);
        let mut rt = (teff[i] / 250.0).round_ties_even() * 250.0;
        if rt < 3500.0 {
            rt = 3500.0;
        }
        if rt > teff_max {
            rt = teff_max;
        }
        let key = (rt as i64, rl.to_bits());
        let pair = match seen.get(&key) {
            Some(p) => *p,
            None => {
                // exact row at (zbest, int(rt), float(rl))
                let mut found = None;
                for j in 0..tab.z.len() {
                    if tab.z[j] == zbest
                        && (tab.teff[j] as i64) == key.0
                        && tab.logg[j].to_bits() == key.1
                    {
                        found = Some((tab.u1[j], tab.u2[j]));
                        break;
                    }
                }
                match found {
                    Some(p) => {
                        seen.insert(key, p);
                        p
                    }
                    None => {
                        // Python: LDCNoRow -> calc_fpp_native skip_reason "ldc_no_row"
                        return Err(crate::fpp::error::FppError::skip(
                            crate::fpp::error::FppSkipReason::LdcNoRow,
                            format!(
                                "no limb-darkening row for Z={zbest} Teff={} logg={} (S family)",
                                key.0,
                                f64::from_bits(key.1)
                            ),
                        ));
                    }
                }
            }
        };
        u1[i] = pair.0;
        u2[i] = pair.1;
    }
    Ok((u1, u2))
}

struct LdcTable {
    z: Vec<f64>,
    teff: Vec<f64>,
    logg: Vec<f64>,
    u1: Vec<f64>,
    u2: Vec<f64>,
}

fn ldc_table() -> &'static LdcTable {
    static T: OnceLock<LdcTable> = OnceLock::new();
    T.get_or_init(|| {
        let csv = include_str!("data/ldc_tess.csv");
        let mut tab = LdcTable {
            z: vec![],
            teff: vec![],
            logg: vec![],
            u1: vec![],
            u2: vec![],
        };
        let mut lines = csv.lines();
        let header: Vec<&str> = lines.next().expect("ldc header").split(',').collect();
        let col = |name: &str| header.iter().position(|h| *h == name).expect("ldc column");
        let (iz, it, ig, ia, ib) = (col("Z"), col("Teff"), col("logg"), col("aLSM"), col("bLSM"));
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split(',').collect();
            tab.z.push(f[iz].parse().unwrap());
            tab.teff.push(f[it].parse().unwrap());
            tab.logg.push(f[ig].parse().unwrap());
            tab.u1.push(f[ia].parse().unwrap());
            tab.u2.push(f[ib].parse().unwrap());
        }
        tab
    })
}

fn argmin_abs_diff(values: &[f64], target: f64) -> usize {
    // np.argmin: first index of the minimum.
    let mut best = 0usize;
    let mut best_v = f64::INFINITY;
    for (i, &v) in values.iter().enumerate() {
        let d = (v - target).abs();
        if d < best_v {
            best_v = d;
            best = i;
        }
    }
    best
}

/// `fpp_native.get_ldc_arrays` — vectorised `get_ldc`: the nearest table Z, Teff and logg chosen INDEPENDENTLY
/// per draw (over the full table), and the joint row must exist (else `Err`, the Python raises `LDCNoRow`).
/// Used by the production secondary-eclipse limb darkening (`FPP_SECONDARY_GEOMETRY = "consistent"`).
pub fn get_ldc_arrays(teff: &[f64], logg: &[f64]) -> Result<(Vec<f64>, Vec<f64>), crate::fpp::error::FppError> {
    if teff.len() != logg.len() {
        // Python: ValueError -> an untyped failure (skip_reason None)
        return Err(crate::fpp::error::FppError::failed(format!(
            "Teff ({}) and logg ({}) must have the same length",
            teff.len(),
            logg.len()
        )));
    }
    let tab = ldc_table();
    let best_z = tab.z[argmin_abs_diff(&tab.z, 0.0)];
    let mut u1 = vec![0.0f64; teff.len()];
    let mut u2 = vec![0.0f64; teff.len()];
    let mut hit: std::collections::HashMap<(u64, u64, u64), (f64, f64)> =
        std::collections::HashMap::new();
    for i in 0..teff.len() {
        let best_t = tab.teff[argmin_abs_diff(&tab.teff, teff[i])];
        let best_g = tab.logg[argmin_abs_diff(&tab.logg, logg[i])];
        let key = (best_z.to_bits(), best_t.to_bits(), best_g.to_bits());
        let pair = match hit.get(&key) {
            Some(p) => *p,
            None => {
                let mut found = None;
                for j in 0..tab.z.len() {
                    if tab.z[j] == best_z && tab.teff[j] == best_t && tab.logg[j] == best_g {
                        found = Some((tab.u1[j], tab.u2[j]));
                        break;
                    }
                }
                match found {
                    Some(p) => {
                        hit.insert(key, p);
                        p
                    }
                    None => {
                        // Python: LDCNoRow -> calc_fpp_native skip_reason "ldc_no_row"
                        return Err(crate::fpp::error::FppError::skip(
                            crate::fpp::error::FppSkipReason::LdcNoRow,
                            format!("no limb-darkening row for Z={best_z} Teff={best_t} logg={best_g}"),
                        ));
                    }
                }
            }
        };
        u1[i] = pair.0;
        u2[i] = pair.1;
    }
    Ok((u1, u2))
}

/// Background-population LDC lookup used by the D/B family. Unlike the
/// companion lookup, this follows the original CSV order: nearest Teff,
/// nearest logg among that Teff, then nearest Z among that pair. `argmin`
/// ties therefore select the first row just as NumPy does.
pub fn ldc_background_arrays(
    teff: &[f64],
    logg: &[f64],
    metallicity: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), crate::fpp::error::FppError> {
    ldc_background_arrays_in(ldc_table(), teff, logg, metallicity)
}

/// Table-parameterised body of [`ldc_background_arrays`]. A missing
/// (Teff, logg) pair or an ambiguous (Teff, logg, Z) row is the Python
/// `BackgroundUnavailable(..., "population_ldc_invalid")`: the error is
/// typed here, where it is detected, so the worker releases the task.
fn ldc_background_arrays_in(
    tab: &LdcTable,
    teff: &[f64],
    logg: &[f64],
    metallicity: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), crate::fpp::error::FppError> {
    use crate::fpp::error::{FppError, FppUnavailableReason};
    let ldc_invalid =
        |message: String| FppError::unavailable(FppUnavailableReason::PopulationLdcInvalid, message);
    if teff.len() != logg.len() || teff.len() != metallicity.len() {
        // Internal invariant of the caller, not a property of the population.
        return Err(FppError::failed("background LDC arrays have different lengths"));
    }
    let mut u1 = Vec::with_capacity(teff.len());
    let mut u2 = Vec::with_capacity(teff.len());
    for i in 0..teff.len() {
        let it = argmin_abs_diff(&tab.teff, teff[i]);
        let t = tab.teff[it];
        let mut ig = None;
        let mut dg = f64::INFINITY;
        for j in 0..tab.teff.len() {
            if tab.teff[j] == t {
                let d = (tab.logg[j] - logg[i]).abs();
                if d < dg {
                    dg = d;
                    ig = Some(j);
                }
            }
        }
        let g = tab.logg[ig.ok_or_else(|| ldc_invalid(format!("background LDC lacks Teff={t}")))?];
        let mut iz = None;
        let mut dz = f64::INFINITY;
        let mut nrow = 0usize;
        for j in 0..tab.teff.len() {
            if tab.teff[j] == t && tab.logg[j] == g {
                nrow += 1;
                let d = (tab.z[j] - metallicity[i]).abs();
                if d < dz {
                    dz = d;
                    iz = Some(j);
                }
            }
        }
        if nrow == 0 {
            return Err(ldc_invalid(format!("background LDC lacks Teff={t}, logg={g}")));
        }
        // Duplicate (Teff, logg, Z) rows make the lookup ambiguous even if
        // the argmin would otherwise pick one by order.
        let row = iz.unwrap();
        let z = tab.z[row];
        if (0..tab.teff.len())
            .filter(|&j| tab.teff[j] == t && tab.logg[j] == g && tab.z[j] == z)
            .count()
            != 1
        {
            return Err(ldc_invalid(format!(
                "background LDC row is not unique: ({t}, {g}, {z})"
            )));
        }
        u1.push(tab.u1[row]);
        u2.push(tab.u2[row]);
    }
    Ok((u1, u2))
}

/// `fpp_native.get_ldc` — nearest-neighbour lookup of the quadratic
/// limb-darkening coefficients (Z, then Teff, then logg; joint mask;
/// solar fallback).
pub fn get_ldc(teff: f64, logg: f64, z: f64) -> (f64, f64) {
    get_ldc_checked(teff, logg, z).unwrap_or((0.40, 0.26))
}

/// `fpp_native.get_ldc` of the corrected (v3) model: the joint row must exist, else `LDCNoRow`
/// (`skip_reason = "ldc_no_row"`), never a default.  [`get_ldc`] keeps the legacy engine's solar fallback.
pub fn get_ldc_checked(teff: f64, logg: f64, z: f64) -> Result<(f64, f64), crate::fpp::error::FppError> {
    let tab = ldc_table();
    let best_z = tab.z[argmin_abs_diff(&tab.z, z)];
    let best_t = tab.teff[argmin_abs_diff(&tab.teff, teff)];
    let best_g = tab.logg[argmin_abs_diff(&tab.logg, logg)];
    for i in 0..tab.z.len() {
        if tab.z[i] == best_z && tab.teff[i] == best_t && tab.logg[i] == best_g {
            return Ok((tab.u1[i], tab.u2[i]));
        }
    }
    Err(crate::fpp::error::FppError::skip(
        crate::fpp::error::FppSkipReason::LdcNoRow,
        format!("no limb-darkening row for Z={best_z} Teff={best_t} logg={best_g}"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fpp::error::FppUnavailableReason;

    fn table(rows: &[(f64, f64, f64)]) -> LdcTable {
        LdcTable {
            teff: rows.iter().map(|r| r.0).collect(),
            logg: rows.iter().map(|r| r.1).collect(),
            z: rows.iter().map(|r| r.2).collect(),
            u1: rows.iter().map(|_| 0.4).collect(),
            u2: rows.iter().map(|_| 0.2).collect(),
        }
    }

    #[test]
    fn ambiguous_background_ldc_row_is_typed_population_ldc_invalid() {
        // Duplicate (Teff, logg, Z) row: Python raises
        // BackgroundUnavailable(..., "population_ldc_invalid").
        let tab = table(&[(5000.0, 4.5, 0.0), (5000.0, 4.5, 0.0), (6000.0, 4.5, 0.0)]);
        let err = ldc_background_arrays_in(&tab, &[5010.0], &[4.4], &[0.1]).unwrap_err();
        assert_eq!(
            err.unavailable_reason(),
            Some(FppUnavailableReason::PopulationLdcInvalid)
        );
        // A unique row resolves normally.
        let (u1, u2) = ldc_background_arrays_in(&tab, &[5990.0], &[4.4], &[0.1]).unwrap();
        assert_eq!((u1, u2), (vec![0.4], vec![0.2]));
    }

    #[test]
    fn background_ldc_length_mismatch_is_a_failure_not_a_release() {
        let tab = table(&[(5000.0, 4.5, 0.0)]);
        let err = ldc_background_arrays_in(&tab, &[5000.0], &[], &[0.0]).unwrap_err();
        assert_eq!(err.unavailable_reason(), None);
    }

    #[test]
    fn shipped_background_ldc_table_has_unique_rows() {
        // With the shipped table the population_ldc_invalid path cannot be
        // reached (as in Python with the unedited CSV); this pins that.
        let tab = ldc_table();
        let mut keys: Vec<(u64, u64, u64)> = (0..tab.teff.len())
            .map(|i| (tab.teff[i].to_bits(), tab.logg[i].to_bits(), tab.z[i].to_bits()))
            .collect();
        let n = keys.len();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), n);
    }
}
