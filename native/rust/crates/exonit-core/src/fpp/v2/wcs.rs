//! Task-shipped TAN WCS. FITS discovery/build_task_geometry remains server-side;
//! workers consume its JSON, including detector origins and aperture pixel order.
use super::math::{cos, sin};
use serde::{Deserialize, Serialize};

fn tan_types() -> [String; 2] {
    ["RA---TAN".into(), "DEC--TAN".into()]
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TanWcs {
    pub crpix: [f64; 2],
    pub crval: [f64; 2],
    pub pc: [f64; 4],
    pub cdelt: [f64; 2],
    #[serde(default = "tan_types")]
    pub ctype: [String; 2],
}

/// Scalar CPython math.radians/degrees constants and operation order.
pub fn tan_world_to_pixel(wcs: &TanWcs, ra_deg: f64, dec_deg: f64) -> Result<[f64; 2], String> {
    if wcs.ctype != tan_types() {
        return Err(format!("unsupported WCS projection {:?}", wcs.ctype));
    }
    let rad = std::f64::consts::PI / 180.0;
    let deg = 180.0 / std::f64::consts::PI;
    let a0 = wcs.crval[0] * rad;
    let d0 = wcs.crval[1] * rad;
    let a = ra_deg * rad;
    let d = dec_deg * rad;
    let da = a - a0;
    let (sd, cd, sd0, cd0) = (sin(d), cos(d), sin(d0), cos(d0));
    let cda = cos(da);
    let cosc = sd0 * sd + cd0 * cd * cda;
    if cosc <= 0.0 {
        return Err("star is on the far side of the tangent point".into());
    }
    let xi = (cd * sin(da) / cosc) * deg;
    let eta = ((cd0 * sd - sd0 * cd * cda) / cosc) * deg;
    let u = xi / wcs.cdelt[0];
    let v = eta / wcs.cdelt[1];
    let [p11, p12, p21, p22] = wcs.pc;
    let det = p11 * p22 - p12 * p21;
    let dx = (p22 * u - p12 * v) / det;
    let dy = (-p21 * u + p11 * v) / det;
    Ok([wcs.crpix[0] - 1.0 + dx, wcs.crpix[1] - 1.0 + dy])
}
