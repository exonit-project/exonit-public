//! Orbit geometry of the fixed model (D1, D2): Kepler solve, true anomaly, mid-transit mean anomaly, the
//! projected separation z(t) with the far side flagged `z = 1e3`, (1/2) d(z^2)/dt, and the parity-mode
//! exposure-supersampled light curve. Literal transcription of `_kepler_eq`, `_true_anomaly`,
//! `_mid_transit_mean_anomaly`, `_z_one`, `_compute_z_arr`, `_half_dz2_dt`, `_lc_core`. Bit-exact.

use super::kernel::transit_flux;
use super::math::{cos, sin};
use super::pow_libm;
use std::f64::consts::PI;

/// Safeguarded scalar Kepler solver from the corrected Python reference.
/// Returns the final eccentric anomaly and whether the root converged.
pub fn kepler_solve(mut m: f64, ecc: f64, maxit: usize) -> (f64, bool) {
    let pi = std::f64::consts::PI;
    let two_pi = 2.0 * pi;
    if m < 0.0 && m >= -1.0e-9 {
        m = 0.0;
    } else if m > two_pi && m <= two_pi + 1.0e-9 {
        m = two_pi;
    }
    if !(m >= 0.0 && m <= two_pi && ecc >= 0.0 && ecc < 1.0) {
        return (f64::NAN, false);
    }
    let (mut lo, mut hi, mut e) = (0.0, two_pi, m);
    for it in 0..maxit {
        // Keep Python's left-to-right evaluation: (E - ecc*sin(E)) - M.
        let f = (e - ecc * sin(e)) - m;
        if f == 0.0 {
            return (e, true);
        }
        if f > 0.0 {
            hi = e;
        } else {
            lo = e;
        }
        let de = f / (1.0 - ecc * cos(e));
        let en = e - de;
        let tiny = de.abs() < 1.0e-10;
        if it < 30 && ((en > lo && en < hi) || tiny) {
            e = en;
            if tiny {
                return (e, true);
            }
        } else {
            let abs_e = e.abs();
            let scale = if abs_e > 1.0 { abs_e } else { 1.0 };
            if hi - lo <= 1.0e-15 * scale {
                return (e, true);
            }
            e = lo + 0.5 * (hi - lo);
        }
    }
    (e, false)
}

/// `_kepler_eq`; unconverged roots map to NaN as in the Python reference.
pub fn kepler_eq(m: f64, ecc: f64) -> f64 {
    let (e, converged) = kepler_solve(m, ecc, 100);
    if converged {
        e
    } else {
        f64::NAN
    }
}

#[cfg(test)]
mod kepler_tests {
    use super::{kepler_eq, kepler_solve, sin};

    #[test]
    fn safeguards_high_eccentricity_and_clamps_roundoff_domain_edges() {
        let tau = 2.0 * std::f64::consts::PI;
        for m in [0.0, -2.8e-14, 0.7, 3.1, tau, tau + 4.4e-14] {
            let (e, ok) = kepler_solve(m, 0.999_999, 100);
            assert!(ok, "M={m}, E={e}");
            assert!(e >= 0.0 && e <= tau);
            let m_clamped = if m < 0.0 {
                0.0
            } else if m > tau {
                tau
            } else {
                m
            };
            assert!((e - 0.999_999 * sin(e) - m_clamped).abs() <= 1.0e-14);
        }
        assert!(kepler_eq(-1.0e-6, 0.4).is_nan());
        assert!(kepler_eq(0.5, 1.0).is_nan());
        let (_, converged) = kepler_solve(0.2, 0.9, 1);
        assert!(!converged);
    }
}

/// `_true_anomaly(M, ecc)`.
pub fn true_anomaly(m: f64, ecc: f64) -> f64 {
    if ecc < 1e-10 {
        return m;
    }
    let e = kepler_eq(m, ecc);
    let cos_e = cos(e);
    ((1.0 - ecc * ecc).sqrt() * sin(e)).atan2(cos_e - ecc)
}

/// `_mid_transit_mean_anomaly(ecc, w_rad)`.
pub fn mid_transit_mean_anomaly(ecc: f64, w_rad: f64) -> f64 {
    let nu_t = PI / 2.0 - w_rad;
    let e_t =
        2.0 * ((1.0 - ecc).sqrt() * sin(nu_t / 2.0)).atan2((1.0 + ecc).sqrt() * cos(nu_t / 2.0));
    e_t - ecc * sin(e_t)
}

/// `_z_one(t, t0, period, a_Rs, cos_i, ecc, w_rad, M_t)`.
#[allow(clippy::too_many_arguments)]
pub fn z_one(
    t: f64,
    t0: f64,
    period: f64,
    a_rs: f64,
    cos_i: f64,
    ecc: f64,
    w_rad: f64,
    m_t: f64,
) -> f64 {
    let two_pi = 2.0 * PI;
    let phase = (t - t0) / period;
    let mut m = two_pi * phase + m_t;
    m -= two_pi * (m / two_pi).floor();
    let nu = true_anomaly(m, ecc);
    let r_factor = (1.0 - ecc * ecc) / (1.0 + ecc * cos(nu));
    let f = nu + w_rad - PI / 2.0;
    let sin_f = sin(f);
    let cos_f = cos(f);
    if cos_f < 0.0 {
        return 1.0e3; // companion behind the star: no transit
    }
    a_rs * r_factor * (sin_f * sin_f + cos_i * cos_i * cos_f * cos_f).sqrt()
}

/// `_compute_z_arr(time, t0, period, a_Rs, inc_rad, ecc, w_rad)`.
pub fn compute_z_arr(
    time: &[f64],
    t0: f64,
    period: f64,
    a_rs: f64,
    inc_rad: f64,
    ecc: f64,
    w_rad: f64,
) -> Vec<f64> {
    let cos_i = cos(inc_rad);
    let m_t = mid_transit_mean_anomaly(ecc, w_rad);
    time.iter()
        .map(|&t| z_one(t, t0, period, a_rs, cos_i, ecc, w_rad, m_t))
        .collect()
}

/// `_half_dz2_dt(t, period, a_Rs, inc_rad, ecc, w_rad, M_t)`: (1/2) d(z^2)/dt in R_s^2 / day.
pub fn half_dz2_dt(
    t: f64,
    period: f64,
    a_rs: f64,
    inc_rad: f64,
    ecc: f64,
    w_rad: f64,
    m_t: f64,
) -> f64 {
    let two_pi = 2.0 * PI;
    let n = two_pi / period;
    let mut m = two_pi * t / period + m_t;
    m -= two_pi * (m / two_pi).floor();
    let nu = true_anomaly(m, ecc);
    let one_minus_e2 = 1.0 - ecc * ecc;
    let q = 1.0 + ecc * cos(nu);
    let r = a_rs * one_minus_e2 / q;
    let nu_dot = n * q * q / pow_libm(one_minus_e2, 1.5);
    let r_dot = n * a_rs * ecc * sin(nu) / one_minus_e2.sqrt();
    let th = w_rad + nu;
    let cos_th = cos(th);
    let sin_th = sin(th);
    let cos_i = cos(inc_rad);
    let x = -r * cos_th;
    let y = -r * sin_th * cos_i;
    let xd = -r_dot * cos_th + r * sin_th * nu_dot;
    let yd = (-r_dot * sin_th - r * cos_th * nu_dot) * cos_i;
    x * xd + y * yd
}

/// `_lc_core(time, k, t0, period, a_Rs, inc_rad, ecc, w_rad, u1, u2, exptime, nsub)`: transit light curve with
/// the PARITY-mode exposure supersampling (end points included); `nsub <= 1` or `exptime <= 0` evaluates the
/// kernel at the cadence times.
#[allow(clippy::too_many_arguments)]
pub fn lc_core(
    time: &[f64],
    k: f64,
    t0: f64,
    period: f64,
    a_rs: f64,
    inc_rad: f64,
    ecc: f64,
    w_rad: f64,
    u1: f64,
    u2: f64,
    exptime: f64,
    nsub: usize,
) -> Vec<f64> {
    let cos_i = cos(inc_rad);
    let m_t = mid_transit_mean_anomaly(ecc, w_rad);
    if nsub <= 1 || exptime <= 0.0 {
        return time
            .iter()
            .map(|&t| {
                transit_flux(
                    z_one(t, t0, period, a_rs, cos_i, ecc, w_rad, m_t),
                    k,
                    u1,
                    u2,
                )
            })
            .collect();
    }
    let nsub_f = nsub as f64;
    time.iter()
        .map(|&t| {
            let mut acc = 0.0;
            for j in 0..nsub {
                let dt = exptime * (j as f64 / (nsub_f - 1.0) - 0.5);
                let z = z_one(t + dt, t0, period, a_rs, cos_i, ecc, w_rad, m_t);
                acc += transit_flux(z, k, u1, u2);
            }
            acc / nsub_f
        })
        .collect()
}
