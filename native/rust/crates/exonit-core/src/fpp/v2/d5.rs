//! D5 PSF aperture fluxes and viable-neighbour selection, scalar Python order.
//!
//! Linux GNU calls glibc's `erf` directly, the same symbol as CPython math.erf.
//! macOS uses system libm and Windows the CRT erf; neither is claimed bit-exact
//! with glibc. Decision (t)'s downstream lnL/FPP tolerances and zero gate flips
//! must be certified there before use. No FMA contraction or fast-math allowed.
//! This module consumes server-built geometry; it never opens FITS or contacts
//! a catalogue. Scenario families and worker dispatch are intentionally separate.
use super::{
    pow_libm,
    wcs::{tan_world_to_pixel, TanWcs},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[cfg_attr(unix, link(name = "m"))]
extern "C" {
    #[link_name = "erf"]
    fn c_erf(x: f64) -> f64;
    fn frexp(x: f64, exponent: *mut i32) -> f64;
    fn ldexp(x: f64, exponent: i32) -> f64;
}

#[inline(never)]
pub fn erf_libm(x: f64) -> f64 {
    unsafe { c_erf(x) }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Star {
    pub ra: f64,
    pub dec: f64,
    #[serde(rename = "Tmag")]
    pub tmag: f64,
    /// Preserve catalogue fields when the computed stars are returned.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SectorGeometry {
    pub wcs: TanWcs,
    pub col0: f64,
    pub row0: f64,
    #[serde(default)]
    pub aperture: Option<Vec<[f64; 2]>>,
    #[serde(default = "unknown")]
    pub source: String,
}
fn unknown() -> String {
    "unknown".into()
}
fn synthetic() -> String {
    "synthetic".into()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Geometry {
    #[serde(default = "synthetic")]
    pub kind: String,
    #[serde(default)]
    pub sectors: Vec<SectorGeometry>,
    pub offsets: Option<Vec<[f64; 2]>>,
    pub n_task_sectors: Option<usize>,
}
impl Default for Geometry {
    fn default() -> Self {
        Self {
            kind: synthetic(),
            sectors: vec![],
            offsets: None,
            n_task_sectors: None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct FluxStar {
    #[serde(flatten)]
    pub star: Star,
    pub fluxratio: f64,
    pub tdepth: f64,
    pub viable: bool,
    pub sep_arcsec: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct GeometryInfo {
    pub geometry_kind: Option<String>,
    pub geometry_n_sectors: usize,
    pub geometry_n_task_sectors: usize,
    pub geometry_coverage: f64,
    pub geometry_source: Vec<String>,
    pub n_viable_neighbours: usize,
    pub fluxratio_target: f64,
}

/// Upstream's 5x5 box, with Python/NumPy round-half-to-even.
pub fn default_aperture(target: [f64; 2]) -> Vec<[f64; 2]> {
    let cx = target[0].round_ties_even();
    let cy = target[1].round_ties_even();
    let mut ap = Vec::with_capacity(25);
    for i in -2..=2 {
        for j in -2..=2 {
            ap.push([cx + i as f64, cy + j as f64]);
        }
    }
    ap
}

fn pixel_weight(p: f64, mu: f64) -> f64 {
    let k = 1.0 / (0.75 * 2.0_f64.sqrt());
    0.5 * (erf_libm((p + 0.5 - mu) * k) - erf_libm((p - 0.5 - mu) * k))
}

pub fn sector_flux_ratios(amp: &[f64], mus: &[[f64; 2]], aperture: &[[f64; 2]]) -> Vec<f64> {
    let flux: Vec<f64> = amp
        .iter()
        .zip(mus)
        .map(|(&a, mu)| {
            let mut acc = 0.0;
            for p in aperture {
                acc += pixel_weight(p[0], mu[0]) * pixel_weight(p[1], mu[1]);
            }
            a * acc
        })
        .collect();
    let mut total = 0.0;
    for &f in &flux {
        total += f;
    }
    flux.iter().map(|f| f / total).collect()
}

type ApertureGeometry = (Vec<[f64; 2]>, Vec<[f64; 2]>);
pub fn geometry_apertures(
    stars: &[Star],
    geometry: &Geometry,
) -> Result<Vec<ApertureGeometry>, String> {
    if stars.is_empty() {
        return Ok(vec![]);
    }
    let mut out = vec![];
    if geometry.kind == "real" {
        for sec in &geometry.sectors {
            let mus: Vec<[f64; 2]> = stars
                .iter()
                .map(|s| {
                    let [x, y] = tan_world_to_pixel(&sec.wcs, s.ra, s.dec)?;
                    Ok([sec.col0 + x, sec.row0 + y])
                })
                .collect::<Result<_, String>>()?;
            let ap = match &sec.aperture {
                Some(ap) if !ap.is_empty() => ap.clone(),
                _ => default_aperture(mus[0]),
            };
            out.push((mus, ap));
        }
    } else {
        let t = &stars[0];
        let cosd0 = super::math::cos(t.dec * (std::f64::consts::PI / 180.0));
        let base: Vec<[f64; 2]> = stars
            .iter()
            .map(|s| {
                [
                    (s.ra - t.ra) * cosd0 * 3600.0 / 20.25,
                    (s.dec - t.dec) * 3600.0 / 20.25,
                ]
            })
            .collect();
        let defaults: Vec<[f64; 2]> = [-0.4, 0.0, 0.4]
            .iter()
            .flat_map(|&x| [-0.4, 0.0, 0.4].iter().map(move |&y| [x, y]))
            .collect();
        for &[ox, oy] in geometry.offsets.as_ref().unwrap_or(&defaults) {
            out.push((
                base.iter().map(|&[x, y]| [x - ox, y - oy]).collect(),
                default_aperture([0.0, 0.0]),
            ));
        }
    }
    Ok(out)
}

fn required(fr: &[f64], depth: f64, corrected: bool) -> Vec<f64> {
    let ap_depth = if corrected { depth * fr[0] } else { depth };
    fr.iter()
        .map(|&f| {
            let t = if f != 0.0 { ap_depth / f } else { 0.0 };
            if t > 1.0e6 {
                0.0
            } else {
                t
            }
        })
        .collect()
}

pub fn compute_flux_ratios_d5(
    stars: &[Star],
    depth_ppm: f64,
    geometry: Option<&Geometry>,
    dilution_corrected: bool,
) -> Result<(Vec<FluxStar>, GeometryInfo), String> {
    let mut info = GeometryInfo {
        geometry_kind: None,
        geometry_n_sectors: 0,
        geometry_n_task_sectors: 0,
        geometry_coverage: 0.0,
        geometry_source: vec![],
        n_viable_neighbours: 0,
        fluxratio_target: 0.0,
    };
    if stars.is_empty() {
        return Ok((vec![], info));
    }
    let fallback = Geometry::default();
    let mut geom = geometry.unwrap_or(&fallback);
    if geom.kind == "real" && geom.sectors.is_empty() {
        geom = &fallback;
    }
    let mut tmin = stars[0].tmag;
    for s in &stars[1..] {
        if s.tmag < tmin {
            tmin = s.tmag;
        }
    }
    let amp: Vec<f64> = stars
        .iter()
        .map(|s| pow_libm(10.0, (tmin - s.tmag) / 2.5))
        .collect();
    let per: Vec<Vec<f64>> = geometry_apertures(stars, geom)?
        .iter()
        .map(|(mus, ap)| sector_flux_ratios(&amp, mus, ap))
        .collect();
    let k = per.len();
    if k == 0 {
        return Err("geometry contains no apertures".into());
    }
    let fr: Vec<f64> = (0..stars.len())
        .map(|i| {
            let mut acc = 0.0;
            for sector in &per {
                acc += sector[i];
            }
            acc / k as f64
        })
        .collect();
    let td = required(&fr, depth_ppm, dilution_corrected);
    let mut viable: Vec<bool> = td.iter().map(|&t| t > 0.0).collect();
    if geom.kind == "synthetic" && k > 1 {
        for sector in &per {
            for (v, t) in viable
                .iter_mut()
                .zip(required(sector, depth_ppm, dilution_corrected))
            {
                *v |= t > 0.0;
            }
        }
    }
    let cosd0 = super::math::cos(stars[0].dec * (std::f64::consts::PI / 180.0));
    let out: Vec<FluxStar> = stars
        .iter()
        .enumerate()
        .map(|(i, s)| FluxStar {
            star: s.clone(),
            fluxratio: fr[i],
            tdepth: td[i],
            viable: i == 0 || viable[i],
            sep_arcsec: python_hypot((s.ra - stars[0].ra) * cosd0, s.dec - stars[0].dec) * 3600.0,
        })
        .collect();
    info.geometry_kind = Some(geom.kind.clone());
    if geom.kind == "real" {
        info.geometry_n_sectors = k;
        info.geometry_n_task_sectors = geom.n_task_sectors.unwrap_or(k);
        info.geometry_coverage = (k as f64 / info.geometry_n_task_sectors.max(1) as f64).min(1.0);
        info.geometry_source = geom
            .sectors
            .iter()
            .map(|s| s.source.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
    } else {
        info.geometry_source = vec![synthetic()];
    }
    info.n_viable_neighbours = out[1..].iter().filter(|s| s.viable).count();
    info.fluxratio_target = fr[0];
    Ok((out, info))
}

/// Reference `_compute_flux_ratios` compatibility wrapper.
pub fn compute_flux_ratios(
    stars: &[Star],
    depth_ppm: f64,
    geometry: Option<&Geometry>,
    dilution_corrected: bool,
) -> Result<Vec<FluxStar>, String> {
    Ok(compute_flux_ratios_d5(stars, depth_ppm, geometry, dilution_corrected)?.0)
}

/// Two-coordinate specialization of CPython 3.12 mathmodule.c `vector_norm`.
/// Source: https://github.com/python/cpython/blob/v3.12.3/Modules/mathmodule.c
/// (PSF-2.0). Compensated square/sum and differential sqrt correction avoid
/// the occasional one-ulp difference of glibc hypot. Explicit mul_add computes
/// only the product's residual, as in CPython dl_mul; expressions elsewhere
/// must never be contracted. Coordinates retain their original order.
pub fn python_hypot(x: f64, y: f64) -> f64 {
    let values = [x.abs(), y.abs()];
    let max = values[0].max(values[1]);
    if max.is_infinite() {
        return max;
    }
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if max == 0.0 {
        return max;
    }
    let mut exponent = 0;
    unsafe {
        frexp(max, &mut exponent);
    }
    if exponent < -1023 {
        return f64::MIN_POSITIVE * python_hypot(x / f64::MIN_POSITIVE, y / f64::MIN_POSITIVE);
    }
    let scale = unsafe { ldexp(1.0, -exponent) };
    let mut csum = 1.0;
    let mut frac1 = 0.0;
    let mut frac2 = 0.0;
    for v in values {
        let v = v * scale;
        let hi = v * v;
        let lo = v.mul_add(v, -hi);
        let sum = csum + hi;
        let err = (csum - sum) + hi;
        csum = sum;
        frac1 += lo;
        frac2 += err;
    }
    let mut h = (csum - 1.0 + (frac1 + frac2)).sqrt();
    let hi = -h * h;
    let lo = (-h).mul_add(h, -hi);
    let sum = csum + hi;
    let err = (csum - sum) + hi;
    csum = sum;
    frac1 += lo;
    frac2 += err;
    let residual = csum - 1.0 + (frac1 + frac2);
    h += residual / (2.0 * h);
    h / scale
}
