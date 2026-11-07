//! Analytic Mandel & Agol (2002) quadratic-limb-darkening occultation flux, clean-room closed forms with
//! K / E / Pi from [`super::ellip`]: literal transcription of `_transit_flux_core` / `_transit_flux` of
//! `worker/fpp_native.py` (acos of the paper replaced by atan2 of the factored root S). Bit-exact.

use super::ellip::{ellip_ke, ellip_kep};
use super::pow_libm;
use std::f64::consts::PI;

/// `_transit_flux_core(z, p, u1, u2)`.
pub fn transit_flux_core(z: f64, p: f64, u1: f64, u2: f64) -> f64 {
    let c2 = u1 + 2.0 * u2;
    let c4 = -u2;
    let fo = 1.0 - c2 / 3.0 - c4 / 2.0;
    if p <= 0.0 || z >= 1.0 + p {
        return 1.0;
    }
    if p >= 1.0 && z <= p - 1.0 {
        return 0.0;
    }
    let a = (z - p) * (z - p);
    let b = (z + p) * (z + p);
    let q = p * p - z * z;
    let mut theta = if p > z { 1.0 } else { 0.0 };
    let lam_e;
    let eta_d;
    let lam_d;
    if p < 1.0 && z <= 1.0 - p {
        // planet fully inside the stellar disc
        lam_e = p * p;
        eta_d = 0.5 * p * p * (p * p + 2.0 * z * z);
        if z == 0.0 {
            lam_d = -(2.0 / 3.0) * pow_libm(1.0 - p * p, 1.5);
        } else if z == p && p < 0.5 {
            let kc = (1.0 - 4.0 * p * p).sqrt();
            let (k, e) = ellip_ke(kc * kc);
            lam_d = 1.0 / 3.0
                + (2.0 / (9.0 * PI)) * (4.0 * (2.0 * p * p - 1.0) * e + (1.0 - 4.0 * p * p) * k);
            theta = 0.0;
        } else if z == p && p == 0.5 {
            lam_d = 1.0 / 3.0 - 4.0 / (9.0 * PI);
            theta = 0.0;
        } else if z == 1.0 - p {
            lam_d = (2.0 / (3.0 * PI)) * (1.0 - 2.0 * p).acos()
                - (4.0 / (9.0 * PI)) * (3.0 + 2.0 * p - 8.0 * p * p) * (p * (1.0 - p)).sqrt()
                - (2.0 / 3.0) * (if p > 0.5 { 1.0 } else { 0.0 });
        } else {
            let kc = (((1.0 - z - p) * (1.0 + z + p)) / ((1.0 - z + p) * (1.0 + z - p))).sqrt();
            let n = (a - b) / a;
            let (k, e, pi) = ellip_kep(kc * kc, n);
            lam_d = (2.0 / (9.0 * PI * (1.0 - a).sqrt()))
                * ((1.0 - 5.0 * z * z + p * p + q * q) * k
                    + (1.0 - a) * (z * z + 7.0 * p * p - 4.0) * e
                    - 3.0 * (q / a) * pi);
        }
    } else {
        // partial overlap, |1 - p| < z < 1 + p (p > 1 allowed)
        let prod = (1.0 - z + p) * (1.0 + z - p) * (z + p - 1.0) * (z + p + 1.0);
        let s = (if prod > 0.0 { prod } else { 0.0 }).sqrt(); // max(0.0, prod)
        let k0 = s.atan2(p * p + z * z - 1.0);
        let k1 = s.atan2(1.0 - p * p + z * z);
        lam_e = (p * p * k0 + k1 - 0.5 * s) / PI;
        let eta2 = p * p * (p * p + 2.0 * z * z) / 2.0;
        eta_d = (k1 + 2.0 * eta2 * k0 - 0.25 * (1.0 + 5.0 * p * p + z * z) * s) / (2.0 * PI);
        if z == p {
            if p >= 0.5 {
                let kc = (1.0 - 1.0 / (4.0 * p * p)).sqrt();
                let (k, e) = ellip_ke(kc * kc);
                lam_d = 1.0 / 3.0 + 16.0 * p / (9.0 * PI) * (2.0 * p * p - 1.0) * e
                    - (1.0 - 4.0 * p * p) * (3.0 - 8.0 * p * p) / (9.0 * PI * p) * k;
                theta = 0.0;
            } else {
                lam_d = 0.0;
            }
        } else {
            let kc = (((z + p - 1.0) * (z + p + 1.0)) / (4.0 * p * z)).sqrt();
            let n = (a - 1.0) / a;
            let (k, e, pi) = ellip_kep(kc * kc, n);
            lam_d = (1.0 / (9.0 * PI * (p * z).sqrt()))
                * (((1.0 - b) * (2.0 * b + a - 3.0) - 3.0 * q * (b - 2.0)) * k
                    + 4.0 * p * z * (z * z + 7.0 * p * p - 4.0) * e
                    - 3.0 * (q / a) * pi);
        }
    }
    1.0 - ((1.0 - c2) * lam_e + c2 * (lam_d + (2.0 / 3.0) * theta) - c4 * eta_d) / fo
}

/// `_transit_flux(z, p, u1, u2)`: the removable singularity at z == p is bridged by linear interpolation.
pub fn transit_flux(z: f64, p: f64, u1: f64, u2: f64) -> f64 {
    if p > 0.0
        && z > 0.0
        && p != z
        && (z - p).abs() < 1.0e-6
        && z < 1.0 + p
        && !(p >= 1.0 && z <= p - 1.0)
    {
        let h = 1.0e-5;
        let f0 = transit_flux_core(p, p, u1, u2);
        let z1 = p + (if z > p { h } else { -h });
        let f1 = transit_flux_core(z1, p, u1, u2);
        return f0 + (z - p) / (z1 - p) * (f1 - f0);
    }
    transit_flux_core(z, p, u1, u2)
}
