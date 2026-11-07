//! Contact-split Gauss-Legendre exposure integrator and the per-draw order choice (plan 0.3 (j'')).
//! Literal transcription of `GL_X/GL_W` use, `_flux_at`, `_window_mean`, `_choose_order` of
//! `worker/fpp_native.py`. The node / weight tables are DATA (see [`super::gl_tables`]). Bit-exact.

use super::consts::{GL_ORACLE_IDX, GL_SMOOTH_IDX};
use super::gl_tables::{GL_MAX_M, GL_ORDERS, GL_W_BITS, GL_X_BITS};
use super::kernel::transit_flux;
use super::orbit::z_one;

/// Orbit + star parameters shared by every flux evaluation of one draw (`P, a_Rs, cos_i, ecc, w_rad, M_t, k,
/// u1, u2`) and the affine dilution `dil_a * F + dil_b` of `_flux_at`.
#[derive(Clone, Copy, Debug)]
pub struct DrawParams {
    pub p: f64,
    pub a_rs: f64,
    pub cos_i: f64,
    pub ecc: f64,
    pub w_rad: f64,
    pub m_t: f64,
    pub k: f64,
    pub u1: f64,
    pub u2: f64,
    pub dil_a: f64,
    pub dil_b: f64,
}

/// Node `i` of order index `oi` (`GL_X[oi, i]`).
#[inline]
pub fn gl_x(oi: usize, i: usize) -> f64 {
    f64::from_bits(GL_X_BITS[oi * GL_MAX_M + i])
}

/// Weight `i` of order index `oi` (`GL_W[oi, i]`).
#[inline]
pub fn gl_w(oi: usize, i: usize) -> f64 {
    f64::from_bits(GL_W_BITS[oi * GL_MAX_M + i])
}

/// `_flux_at(t, ...)`: model flux at one time, with the affine dilution.
#[inline]
pub fn flux_at(t: f64, d: &DrawParams) -> f64 {
    let z = z_one(t, 0.0, d.p, d.a_rs, d.cos_i, d.ecc, d.w_rad, d.m_t);
    d.dil_a * transit_flux(z, d.k, d.u1, d.u2) + d.dil_b
}

/// `_window_mean(tc, W, ct, order_idx, ...)`: mean model flux over `[tc - W/2, tc + W/2]`, the window cut at
/// the (sorted, finite) contact times `ct` that fall strictly inside it; endpoint substitution when a piece end
/// is a contact.
pub fn window_mean(tc: f64, w: f64, ct: &[f64], order_idx: usize, d: &DrawParams) -> f64 {
    let a = tc - 0.5 * w;
    let b = tc + 0.5 * w;
    let m = GL_ORDERS[order_idx];
    let nct = ct.len();
    let mut acc = 0.0;
    let mut jc = 0usize;
    while jc < nct && ct[jc] <= a {
        jc += 1; // contacts at or below the lower edge
    }
    let mut lo = a;
    let mut lo_c = false;
    loop {
        let hi;
        let hi_c;
        if jc < nct && ct[jc] < b {
            hi = ct[jc]; // interior cut (a contact)
            hi_c = true;
            jc += 1;
        } else {
            hi = b; // upper window edge
            hi_c = false;
        }

        let l = hi - lo;
        if l > 0.0 {
            if !(lo_c || hi_c) {
                let half_l = 0.5 * l;
                let mid = 0.5 * (hi + lo);
                for i in 0..m {
                    let t = half_l * gl_x(order_idx, i) + mid;
                    let wt = half_l * gl_w(order_idx, i);
                    acc += wt * flux_at(t, d);
                }
            } else if lo_c && hi_c {
                for i in 0..m {
                    let u = 0.5 * (gl_x(order_idx, i) + 1.0);
                    let t = lo + l * u * u * (3.0 - 2.0 * u);
                    let wt = 0.5 * gl_w(order_idx, i) * 6.0 * l * u * (1.0 - u);
                    acc += wt * flux_at(t, d);
                }
            } else {
                let s_l = l.sqrt();
                for i in 0..m {
                    let s = 0.5 * s_l * (gl_x(order_idx, i) + 1.0);
                    let wt = 0.5 * s_l * gl_w(order_idx, i) * 2.0 * s;
                    let t = if lo_c { lo + s * s } else { hi - s * s };
                    acc += wt * flux_at(t, d);
                }
            }
        }
        if !hi_c {
            break; // the piece reached the upper window edge
        }
        lo = hi;
        lo_c = true;
    }
    acc / w
}

/// `_choose_order(ct, W, eps, ...)` -> `(order_idx, status)`: status 0 converged by an order <= 32, 1 the
/// smallest-order search failed and the oracle order 128 is used, 2 even 64 vs 128 disagree (unconverged: the
/// caller must flag it, never 'validated').
pub fn choose_order(ct: &[f64], w: f64, eps: f64, d: &DrawParams) -> (usize, i32) {
    let nct = ct.len();
    if nct == 0 {
        return (GL_SMOOTH_IDX, 0);
    }
    let mut chosen = GL_SMOOTH_IDX;
    for &c in ct.iter() {
        let mut v0 = window_mean(c, w, ct, 0, d);
        let mut v1 = window_mean(c, w, ct, 1, d);
        let mut v2 = window_mean(c, w, ct, 2, d);
        let mut found: i32 = -1;
        for i in 0..7usize {
            if (v0 - v1).abs() <= eps && (v0 - v2).abs() <= eps {
                found = i as i32;
                break;
            }
            let v3 = window_mean(c, w, ct, i + 3, d);
            v0 = v1;
            v1 = v2;
            v2 = v3;
        }
        if found < 0 {
            // v0, v1, v2 now hold I_7, I_8, I_9 (m = 24, 32, 64); the oracle I_10 (m = 128) enters the test.
            let v3 = window_mean(c, w, ct, GL_ORACLE_IDX, d);
            if (v0 - v1).abs() <= eps && (v0 - v2).abs() <= eps {
                found = 7;
            } else if (v1 - v2).abs() <= eps && (v1 - v3).abs() <= eps {
                found = 8;
            } else if (v2 - v3).abs() <= eps {
                return (GL_ORACLE_IDX, 1); // plateau holds: oracle fallback
            } else {
                return (GL_ORACLE_IDX, 2); // 64 vs 128 disagree: unconverged
            }
        }
        if found as usize > chosen {
            chosen = found as usize;
        }
    }
    (chosen, 0)
}
