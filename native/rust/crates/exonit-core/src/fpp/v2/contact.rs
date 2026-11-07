//! Exact contact times of one transit (plan 0.3 (j'')): bisection of z(t) - level and of (1/2) d(z^2)/dt.
//! Literal transcription of `_bisect` and `_contact_times_nb` of `worker/fpp_native.py`. Bit-exact.

use super::math::cos;
use super::orbit::{half_dz2_dt, mid_transit_mean_anomaly, z_one};
use std::f64::consts::PI;

#[allow(clippy::too_many_arguments)]
fn bisect(
    mut flo: f64,
    fhi: f64,
    mut tlo: f64,
    mut thi: f64,
    period: f64,
    a_rs: f64,
    inc_rad: f64,
    ecc: f64,
    w_rad: f64,
    m_t: f64,
    level: f64,
    is_z: bool,
) -> f64 {
    let tol = 1e-13;
    if flo == 0.0 {
        return tlo;
    }
    if fhi == 0.0 {
        return thi;
    }
    if flo * fhi > 0.0 {
        return f64::NAN;
    }
    let mut fhi = fhi;
    for _ in 0..100 {
        if thi - tlo < tol {
            break;
        }
        let tm = 0.5 * (tlo + thi);
        let fm = if is_z {
            z_one(tm, 0.0, period, a_rs, cos(inc_rad), ecc, w_rad, m_t) - level
        } else {
            half_dz2_dt(tm, period, a_rs, inc_rad, ecc, w_rad, m_t)
        };
        if fm == 0.0 {
            return tm;
        }
        if (fm > 0.0) == (flo > 0.0) {
            tlo = tm;
            flo = fm;
        } else {
            thi = tm;
            fhi = fm;
        }
    }
    let _ = fhi; // updated in the reference too, never read afterwards
    0.5 * (tlo + thi)
}

/// `_contact_times_nb(period, a_Rs, inc_rad, ecc, w_rad, k)` -> `[t1, t2, t3, t4, t_min]`, NaN where a contact
/// is absent, days relative to mid-transit.
pub fn contact_times(
    period: f64,
    a_rs: f64,
    inc_rad: f64,
    ecc: f64,
    w_rad: f64,
    k: f64,
) -> [f64; 5] {
    let mut out = [f64::NAN; 5];
    let cos_i = cos(inc_rad);
    let m_t = mid_transit_mean_anomaly(ecc, w_rad);

    // Longest possible transit scale (apastron geometry), never past half an orbit, never below 1e-3 d.
    let mut w = 2.5 * (1.0 + k) * period / (PI * a_rs) / (1.0 - ecc);
    if w > 0.5 * period {
        w = 0.5 * period;
    }
    if w < 1e-3 {
        w = 1e-3;
    }

    let ngrid: i64 = 2001;
    let step = 2.0 * w / (ngrid - 1) as f64;
    let mut best_z = 1.0e3;
    let mut i0: i64 = -1;
    for i in 0..ngrid {
        let z = z_one(
            -w + step * i as f64,
            0.0,
            period,
            a_rs,
            cos_i,
            ecc,
            w_rad,
            m_t,
        );
        if z < 1.0e3 && z < best_z {
            best_z = z;
            i0 = i;
        }
    }
    if i0 < 0 || best_z >= 1.0 + k + 1e-3 {
        return out; // no transit anywhere in the window
    }

    let mut lo_i = i0 - 1;
    if lo_i < 0 {
        lo_i = 0;
    }
    let mut hi_i = i0 + 1;
    if hi_i > ngrid - 1 {
        hi_i = ngrid - 1;
    }
    let tlo = -w + step * lo_i as f64;
    let thi = -w + step * hi_i as f64;
    let t_min = bisect(
        half_dz2_dt(tlo, period, a_rs, inc_rad, ecc, w_rad, m_t),
        half_dz2_dt(thi, period, a_rs, inc_rad, ecc, w_rad, m_t),
        tlo,
        thi,
        period,
        a_rs,
        inc_rad,
        ecc,
        w_rad,
        m_t,
        0.0,
        false,
    );
    if t_min.is_nan() {
        return out;
    }
    let z_min = z_one(t_min, 0.0, period, a_rs, cos_i, ecc, w_rad, m_t);

    let l = 1.0 + k;
    if z_min >= l {
        return out; // the companion never touches the disc
    }

    // Outer contacts; widen the window (x2, at most 4 times) before giving up.
    let mut hw = w;
    let mut t1 = f64::NAN;
    for _ in 0..5 {
        let a = t_min - hw;
        let fa = z_one(a, 0.0, period, a_rs, cos_i, ecc, w_rad, m_t) - l;
        let fb = z_min - l;
        t1 = bisect(
            fa, fb, a, t_min, period, a_rs, inc_rad, ecc, w_rad, m_t, l, true,
        );
        if !t1.is_nan() {
            break;
        }
        hw *= 2.0;
    }
    hw = w;
    let mut t4 = f64::NAN;
    for _ in 0..5 {
        let b = t_min + hw;
        let fa = z_min - l;
        let fb = z_one(b, 0.0, period, a_rs, cos_i, ecc, w_rad, m_t) - l;
        t4 = bisect(
            fa, fb, t_min, b, period, a_rs, inc_rad, ecc, w_rad, m_t, l, true,
        );
        if !t4.is_nan() {
            break;
        }
        hw *= 2.0;
    }

    // Inner contacts: only when the companion is fully inside (or fully in front of) the disc at mid-transit.
    let mut t2 = f64::NAN;
    let mut t3 = f64::NAN;
    let lin = (1.0 - k).abs();
    if z_min < lin {
        if !t1.is_nan() {
            t2 = bisect(
                l - lin,
                z_min - lin,
                t1,
                t_min,
                period,
                a_rs,
                inc_rad,
                ecc,
                w_rad,
                m_t,
                lin,
                true,
            );
        }
        if !t4.is_nan() {
            t3 = bisect(
                z_min - lin,
                l - lin,
                t_min,
                t4,
                period,
                a_rs,
                inc_rad,
                ecc,
                w_rad,
                m_t,
                lin,
                true,
            );
        }
    }

    out[0] = t1;
    out[1] = t2;
    out[2] = t3;
    out[3] = t4;
    out[4] = t_min;
    out
}
