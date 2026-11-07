//! Product-aperture, epoch-averaged, calibration-marginalized FPP geometry.
//! Arithmetic follows the fixed operation order in `progress/fpp-port-list.md` G.
use super::{
    d5::{erf_libm, python_hypot, Star},
    pow_libm,
    wcs::{tan_world_to_pixel, TanWcs},
};
use serde::Deserialize;
use std::collections::BTreeSet;

const NODE_BITS: [u64; 25] = [
    0x3f68527aba212071,
    0x3f8b404942866157,
    0x3f9676f9c7838aa7,
    0x3f8b404942866157,
    0x3f68527aba212071,
    0x3f8b404942866157,
    0x3fae8862e93c9e5c,
    0x3fb92b856facd857,
    0x3fae8862e93c9e5c,
    0x3f8b404942866157,
    0x3f9676f9c7838aa7,
    0x3fb92b856facd857,
    0x3fc4bfc906c2a8eb,
    0x3fb92b856facd857,
    0x3f9676f9c7838aa7,
    0x3f8b404942866157,
    0x3fae8862e93c9e5c,
    0x3fb92b856facd857,
    0x3fae8862e93c9e5c,
    0x3f8b404942866157,
    0x3f68527aba212071,
    0x3f8b404942866157,
    0x3f9676f9c7838aa7,
    0x3f8b404942866157,
    0x3f68527aba212071,
];
const SIGMA_DEFAULT: [f64; 4] = [0.02, 0.035, 0.05, 0.035];

#[derive(Clone, Debug, Deserialize)]
pub struct GeometryV3 {
    pub kind: String,
    #[serde(default)]
    pub sectors: Vec<SectorV3>,
    #[serde(default)]
    pub n_task_sectors: Option<usize>,
}
#[derive(Clone, Debug, Deserialize)]
pub struct SectorV3 {
    pub wcs: TanWcs,
    pub col0: f64,
    pub row0: f64,
    #[serde(default)]
    pub aperture: Option<Vec<[f64; 2]>>,
    #[serde(default)]
    pub aperture_kind: Option<String>,
    #[serde(default)]
    pub box_centre: Option<[f64; 2]>,
    #[serde(default)]
    pub circle_radius: Option<f64>,
    #[serde(default)]
    pub c_shift: Option<[f64; 2]>,
    #[serde(default)]
    pub epochs: Vec<[f64; 2]>,
    #[serde(default)]
    pub sigma_cal: Option<f64>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub aperture_source: Option<String>,
    #[serde(default)]
    pub tglc_near_edge: Option<bool>,
}

#[derive(Clone, Debug)]
pub struct SmoothStar {
    pub star: Star,
    pub fluxratio: f64,
    pub tdepth_raw: f64,
    pub tdepth: f64,
    pub viability: f64,
    pub viable: bool,
    pub sep_arcsec: f64,
}
#[derive(Clone, Debug)]
pub struct GeometryV3Info {
    pub geometry_kind: String,
    pub geometry_mode: String,
    pub geometry_n_sectors: usize,
    pub geometry_n_task_sectors: usize,
    pub geometry_coverage: f64,
    pub geometry_source: Vec<String>,
    pub aperture_source: Vec<String>,
    pub geometry_path: Vec<String>,
    pub geometry_path_emulated: bool,
    pub geometry_n_epochs: Vec<usize>,
    pub geometry_sigma_cal: Vec<f64>,
    pub n_nodes: usize,
    pub n_viable_neighbours: usize,
    pub n_viability_weighted: usize,
    pub viability_mass_skipped: f64,
    pub fluxratio_target: f64,
}

#[derive(Clone)]
struct ResolvedSector {
    kind: String,
    pixels: Option<Vec<[f64; 3]>>,
    radius: f64,
    emulated: bool,
    source: String,
    path: String,
    shift: [f64; 2],
    epochs: Vec<[f64; 2]>,
    sigma: f64,
}

fn resolve(s: &SectorV3, lc_source: Option<&str>) -> Result<ResolvedSector, String> {
    let ap = s.aperture.as_ref().filter(|a| !a.is_empty());
    let mut kind = s.aperture_kind.clone().unwrap_or_else(|| {
        match super::dispatch::normalize_lc_source(lc_source) {
            "tglc" => "tglc_box".into(),
            "qlp" => "qlp_circle".into(),
            _ if ap.is_some() => "mask".into(),
            _ => "default".into(),
        }
    });
    let emulated = match kind.as_str() {
        "mask" if ap.is_none() => return Err("aperture_kind 'mask' without an aperture".into()),
        "mask" => false,
        "tglc_box" => !(s.box_centre.is_some() || ap.is_some()),
        "qlp_circle" => s.circle_radius.is_none(),
        "default" => false,
        _ => return Err(format!("unknown aperture_kind {kind:?}")),
    };
    let (pixels, src) = match kind.as_str() {
        "mask" => (
            Some(ap.unwrap().iter().map(|p| [p[0], p[1], 1.0]).collect()),
            s.aperture_source
                .clone()
                .unwrap_or_else(|| "spoc_lc_mask".into()),
        ),
        "tglc_box" if !emulated && ap.is_some() => (
            Some(ap.unwrap().iter().map(|p| [p[0], p[1], 1.0]).collect()),
            "tglc_box_header".into(),
        ),
        "tglc_box" if s.box_centre.is_some() => {
            let c = s.box_centre.unwrap();
            (
                Some(box_pixels(c[0].round_ties_even(), c[1].round_ties_even())),
                "tglc_box_header".into(),
            )
        }
        "tglc_box" => (None, "tglc_box_emulated".into()),
        "qlp_circle" => (
            None,
            if emulated {
                "qlp_circle_assumed".into()
            } else {
                "qlp_bestap_circle".into()
            },
        ),
        _ => (None, "default_5x5".into()),
    };
    if kind == "qlp_circle"
        && s.circle_radius.is_some()
        && !(0.5..=8.0).contains(&s.circle_radius.unwrap())
    {
        return Err("circle_radius outside 0.5-8 px".into());
    }
    let radius = s.circle_radius.unwrap_or(3.0);
    let shift = s.c_shift.unwrap_or([0.0, 0.0]);
    if !shift.iter().all(|v| v.is_finite()) || !s.epochs.iter().flatten().all(|v| v.is_finite()) {
        return Err("c_shift / epochs must be finite".into());
    }
    let mut epochs = if s.epochs.is_empty() {
        vec![[0.0, 0.0]]
    } else {
        s.epochs.clone()
    };
    if kind == "qlp_circle" {
        epochs = vec![[0.0, 0.0]];
    }
    let sigma = s.sigma_cal.unwrap_or(
        SIGMA_DEFAULT[match kind.as_str() {
            "mask" => 0,
            "tglc_box" => 1,
            "qlp_circle" => 2,
            _ => 3,
        }],
    );
    if !sigma.is_finite() || sigma < 0.0 {
        return Err("sigma_cal must be finite and >= 0".into());
    }
    let path = match kind.as_str() {
        "mask" => "spoc_mask",
        "tglc_box" => "tglc_box",
        "qlp_circle" => "qlp_circle",
        _ => "u_default",
    }
    .to_string();
    Ok(ResolvedSector {
        kind: std::mem::take(&mut kind),
        pixels,
        radius,
        emulated,
        source: src,
        path,
        shift,
        epochs,
        sigma,
    })
}

fn box_pixels(cx: f64, cy: f64) -> Vec<[f64; 3]> {
    let mut v = Vec::with_capacity(9);
    for i in -1..=1 {
        for j in -1..=1 {
            v.push([cx + i as f64, cy + j as f64, 1.0]);
        }
    }
    v
}
fn default_pixels(cx: f64, cy: f64) -> Vec<[f64; 3]> {
    let mut v = Vec::with_capacity(25);
    for i in -2..=2 {
        for j in -2..=2 {
            v.push([cx + i as f64, cy + j as f64, 1.0]);
        }
    }
    v
}
fn weight(p: f64, mu: f64) -> f64 {
    let k = 1.0 / (0.75 * 2.0_f64.sqrt());
    0.5 * (erf_libm((p + 0.5 - mu) * k) - erf_libm((p - 0.5 - mu) * k))
}
fn aperture_flux(mu: [f64; 2], pixels: &[[f64; 3]]) -> f64 {
    let mut wxs = std::collections::BTreeMap::<u64, f64>::new();
    let mut wys = std::collections::BTreeMap::<u64, f64>::new();
    let mut acc = 0.0;
    for p in pixels {
        let wx = *wxs
            .entry(p[0].to_bits())
            .or_insert_with(|| weight(p[0], mu[0]));
        let wy = *wys
            .entry(p[1].to_bits())
            .or_insert_with(|| weight(p[1], mu[1]));
        acc = acc + ((p[2] * wx) * wy);
    }
    acc
}
fn qlp_pixels(cx: f64, cy: f64, r: f64) -> Vec<[f64; 3]> {
    let sub = 32;
    let r2 = r * r;
    let inv = 1.0 / sub as f64;
    let nn = (sub * sub) as f64;
    let mut out = Vec::new();
    let x0 = (cx - r - 0.5).floor() as i32;
    let x1 = (cx + r + 0.5).ceil() as i32;
    let y0 = (cy - r - 0.5).floor() as i32;
    let y1 = (cy + r + 0.5).ceil() as i32;
    for px in x0..=x1 {
        let xl = px as f64 - 0.5 - cx;
        let xh = px as f64 + 0.5 - cx;
        let near_x = if xl <= 0.0 && 0.0 <= xh {
            0.0
        } else {
            xl.abs().min(xh.abs())
        };
        for py in y0..=y1 {
            let yl = py as f64 - 0.5 - cy;
            let yh = py as f64 + 0.5 - cy;
            let near_y = if yl <= 0.0 && 0.0 <= yh {
                0.0
            } else {
                yl.abs().min(yh.abs())
            };
            if near_x * near_x + near_y * near_y > r2 {
                continue;
            }
            let far_x = xl.abs().max(xh.abs());
            let far_y = yl.abs().max(yh.abs());
            if far_x * far_x + far_y * far_y <= r2 * (1.0 - 1e-9) {
                out.push([px as f64, py as f64, 1.0]);
                continue;
            }
            let mut n = 0;
            for i in 0..sub {
                let x = (px as f64 - 0.5 + (i as f64 + 0.5) * inv) - cx;
                let x2 = pow_libm(x, 2.0);
                for j in 0..sub {
                    let y = (py as f64 - 0.5 + (j as f64 + 0.5) * inv) - cy;
                    let y2 = pow_libm(y, 2.0);
                    if x2 + y2 <= r2 {
                        n += 1;
                    }
                }
            }
            if n > 0 {
                out.push([px as f64, py as f64, n as f64 / nn]);
            }
        }
    }
    out
}

/// Smooth geometry v3. Returns per-star flux/viability weights and geometry provenance.
pub fn compute_flux_ratios_smooth(
    stars: &[Star],
    depth_ppm: f64,
    geometry: &GeometryV3,
    dilution_corrected: bool,
    lc_source: Option<&str>,
    v_min: f64,
) -> Result<(Vec<SmoothStar>, GeometryV3Info), String> {
    if stars.is_empty() {
        return Err("geometry requires a target star".into());
    }
    if geometry.kind != "real" || geometry.sectors.is_empty() {
        return Err("smooth geometry requires real sectors".into());
    }
    let n = stars.len();
    let secs = geometry
        .sectors
        .iter()
        .map(|s| resolve(s, lc_source))
        .collect::<Result<Vec<_>, _>>()?;
    let ns = secs.len();
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
    let mut bases = Vec::with_capacity(ns);
    let mut secs = secs;
    for (s, r) in geometry.sectors.iter().zip(secs.iter_mut()) {
        let mut b = Vec::with_capacity(n);
        for star in stars {
            let [x, y] = tan_world_to_pixel(&s.wcs, star.ra, star.dec)?;
            b.push([(s.col0 + x) + r.shift[0], (s.row0 + y) + r.shift[1]]);
        }
        if r.kind == "tglc_box" && r.pixels.is_none() {
            r.pixels = Some(box_pixels(
                b[0][0].round_ties_even(),
                b[0][1].round_ties_even(),
            ));
        }
        bases.push(b);
    }
    let one_node = secs.iter().all(|s| s.sigma == 0.0);
    let count = if one_node { 1 } else { 25 };
    let mut fr_exp = vec![0.0; n];
    let mut v_exp = vec![0.0; n];
    for ni in 0..count {
        let (a, b, w) = if one_node {
            (0, 0, 1.0)
        } else {
            (
                (ni / 5) as i32 - 2,
                (ni % 5) as i32 - 2,
                f64::from_bits(NODE_BITS[ni]),
            )
        };
        let mut per = Vec::<Vec<f64>>::with_capacity(ns);
        for (si, (base, sr)) in bases.iter().zip(&secs).enumerate() {
            let dx = sr.sigma * a as f64;
            let dy = sr.sigma * b as f64;
            let pixels = if sr.kind == "qlp_circle" {
                qlp_pixels(dx, dy, sr.radius)
            } else if sr.kind == "default" {
                default_pixels(
                    (base[0][0] + dx).round_ties_even(),
                    (base[0][1] + dy).round_ties_even(),
                )
            } else {
                sr.pixels.clone().unwrap_or_default()
            };
            let mut fs = vec![0.0; n];
            let k = sr.epochs.len();
            for i in 0..n {
                let mut acc = 0.0;
                for ep in &sr.epochs {
                    let mu = if sr.kind == "qlp_circle" {
                        [base[i][0] - base[0][0], base[i][1] - base[0][1]]
                    } else {
                        [base[i][0] + dx + ep[0], base[i][1] + dy + ep[1]]
                    };
                    acc += amp[i] * aperture_flux(mu, &pixels);
                }
                fs[i] = acc / k as f64;
            }
            let mut total = 0.0;
            for f in &fs {
                total += *f;
            }
            per.push(fs.iter().map(|f| f / total).collect());
            let _ = si;
        }
        let mut frm = vec![0.0; n];
        for i in 0..n {
            let mut acc = 0.0;
            for sector in &per {
                acc += sector[i];
            }
            frm[i] = acc / ns as f64;
        }
        let apdepth = if dilution_corrected {
            depth_ppm * frm[0]
        } else {
            depth_ppm
        };
        for i in 0..n {
            let td = if frm[i] != 0.0 { apdepth / frm[i] } else { 0.0 };
            let viable = if td > 0.0 && td <= 1.0e6 { 1.0 } else { 0.0 };
            fr_exp[i] += w * frm[i];
            v_exp[i] += w * viable;
        }
    }
    let apdepth = if dilution_corrected {
        depth_ppm * fr_exp[0]
    } else {
        depth_ppm
    };
    let mut weighted = 0;
    let mut skipped = 0.0;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let td = if fr_exp[i] != 0.0 {
            apdepth / fr_exp[i]
        } else {
            0.0
        };
        let v = if i == 0 {
            1.0
        } else {
            let mut v = v_exp[i].clamp(0.0, 1.0);
            if v > 1.0 - 1e-12 {
                v = 1.0;
            }
            v
        };
        let viable = i == 0 || v >= v_min;
        if i > 0 && viable && v < 1.0 {
            weighted += 1;
        }
        if i > 0 && !viable {
            skipped += v;
        }
        let cosd0 = super::math::cos(stars[0].dec * (std::f64::consts::PI / 180.0));
        let sep = python_hypot(
            (stars[i].ra - stars[0].ra) * cosd0,
            stars[i].dec - stars[0].dec,
        ) * 3600.0;
        out.push(SmoothStar {
            star: stars[i].clone(),
            fluxratio: fr_exp[i],
            tdepth_raw: td,
            tdepth: if td > 1e6 { 0.0 } else { td },
            viability: v,
            viable,
            sep_arcsec: sep,
        });
    }
    let geometry_source = geometry
        .sectors
        .iter()
        .map(|s| s.source.clone().unwrap_or_else(|| "unknown".into()))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let aperture_source = secs
        .iter()
        .map(|s| s.source.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let paths = secs
        .iter()
        .map(|s| s.path.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let task_n = geometry.n_task_sectors.unwrap_or(ns);
    let info = GeometryV3Info {
        geometry_kind: "real".into(),
        geometry_mode: "smooth".into(),
        geometry_n_sectors: ns,
        geometry_n_task_sectors: task_n,
        geometry_coverage: (ns as f64 / task_n.max(1) as f64).min(1.0),
        geometry_source,
        aperture_source,
        geometry_path: paths,
        geometry_path_emulated: secs.iter().any(|s| s.emulated),
        geometry_n_epochs: secs.iter().map(|s| s.epochs.len()).collect(),
        geometry_sigma_cal: secs.iter().map(|s| s.sigma).collect(),
        n_nodes: count,
        n_viable_neighbours: out[1..].iter().filter(|s| s.viable).count(),
        n_viability_weighted: weighted,
        viability_mass_skipped: skipped,
        fluxratio_target: fr_exp[0],
    };
    Ok((out, info))
}
