//! Complete elliptic integrals K, E, Pi through Carlson's symmetric forms R_F, R_C, R_J (Carlson 1995,
//! DLMF 19.36): literal transcription of `_carlson_rf/_rc/_rj`, `_ellip_ke`, `_ellip_kep`, `_ellipk`,
//! `_ellipe`, `_ellippi` of `worker/fpp_native.py`. Operation order is the Python order.

use super::pow_libm;

const CARLSON_TOL: f64 = 1.0e-3;

/// `max(abs(dx), abs(dy), abs(dz))` as numba evaluates the 3-argument builtin (left to right).
#[inline]
fn max3(a: f64, b: f64, c: f64) -> f64 {
    let m = if b > a { b } else { a };
    if c > m {
        c
    } else {
        m
    }
}

/// `_carlson_rf`.
pub fn carlson_rf(mut x: f64, mut y: f64, mut z: f64) -> f64 {
    let (mut mu, mut dx, mut dy, mut dz) = (0.0, 0.0, 0.0, 0.0);
    for _ in 0..1100 {
        let sx = x.sqrt();
        let sy = y.sqrt();
        let sz = z.sqrt();
        let lam = sx * sy + sy * sz + sz * sx;
        x = 0.25 * (x + lam);
        y = 0.25 * (y + lam);
        z = 0.25 * (z + lam);
        mu = (x + y + z) / 3.0;
        dx = 1.0 - x / mu;
        dy = 1.0 - y / mu;
        dz = 1.0 - z / mu;
        if max3(dx.abs(), dy.abs(), dz.abs()) < CARLSON_TOL {
            break;
        }
    }
    let e2 = dx * dy - dz * dz;
    let e3 = dx * dy * dz;
    (1.0 - e2 / 10.0 + e3 / 14.0 + e2 * e2 / 24.0 - 3.0 * e2 * e3 / 44.0) / mu.sqrt()
}

/// `_carlson_rc`.
pub fn carlson_rc(mut x: f64, mut y: f64) -> f64 {
    let (mut mu, mut s) = (0.0, 0.0);
    for _ in 0..200 {
        let lam = 2.0 * x.sqrt() * y.sqrt() + y;
        x = 0.25 * (x + lam);
        y = 0.25 * (y + lam);
        mu = (x + 2.0 * y) / 3.0;
        s = (y - mu) / mu;
        if s.abs() < CARLSON_TOL {
            break;
        }
    }
    (1.0 + s * s * (0.3 + s * (1.0 / 7.0 + s * (0.375 + s * 9.0 / 22.0)))) / mu.sqrt()
}

/// `_carlson_rj`.
pub fn carlson_rj(mut x: f64, mut y: f64, mut z: f64, mut p: f64) -> f64 {
    let mut total = 0.0;
    let mut fac = 1.0;
    let (mut mu, mut dx, mut dy, mut dz, mut dp) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for _ in 0..1100 {
        let sx = x.sqrt();
        let sy = y.sqrt();
        let sz = z.sqrt();
        let _sp = p.sqrt(); // evaluated and unused in the reference
        let lam = sx * sy + sy * sz + sz * sx;
        let t = p * (sx + sy + sz) + sx * sy * sz;
        let alpha = t * t; // numba: x ** 2 -> x * x
        let u = p + lam;
        let beta = p * (u * u);
        total += fac * carlson_rc(alpha, beta);
        fac *= 0.25;
        x = 0.25 * (x + lam);
        y = 0.25 * (y + lam);
        z = 0.25 * (z + lam);
        p = 0.25 * (p + lam);
        mu = (x + y + z + 2.0 * p) / 5.0;
        dx = (mu - x) / mu;
        dy = (mu - y) / mu;
        dz = (mu - z) / mu;
        dp = (mu - p) / mu;
        let m1 = if dy.abs() > dx.abs() {
            dy.abs()
        } else {
            dx.abs()
        };
        let m2 = if dp.abs() > dz.abs() {
            dp.abs()
        } else {
            dz.abs()
        };
        let m = if m2 > m1 { m2 } else { m1 };
        if m < CARLSON_TOL {
            break;
        }
    }
    let e2 = dx * dy + dx * dz + dy * dz - 3.0 * dp * dp; // (3.0 * dp) * dp, left to right
    let e3 = dx * dy * dz + 2.0 * e2 * dp + 4.0 * (dp * (dp * dp)); // dp ** 3 -> dp * (dp * dp)
    let e4 = (2.0 * dx * dy * dz + e2 * dp + 3.0 * (dp * (dp * dp))) * dp;
    let e5 = dx * dy * dz * dp * dp;
    let series = 1.0 - 3.0 * e2 / 14.0 + e3 / 6.0 + 9.0 * e2 * e2 / 88.0
        - 3.0 * e4 / 22.0
        - 9.0 * e2 * e3 / 52.0
        + 3.0 * e5 / 26.0;
    3.0 * total + fac * pow_libm(mu, -1.5) * series
}

/// `_ellip_ke`: (K, E) of the modulus whose complementary parameter is `kc2 = 1 - m`.
pub fn ellip_ke(kc2: f64) -> (f64, f64) {
    let kc2 = if kc2 > 1.0e-300 { kc2 } else { 1.0e-300 }; // max(kc2, 1e-300)
    let k = carlson_rf(0.0, kc2, 1.0);
    let e = k - ((1.0 - kc2) / 3.0) * carlson_rj(0.0, kc2, 1.0, 1.0);
    (k, e)
}

/// `_ellip_kep`: (K, E, Pi(n, m)) with m = 1 - kc2; n > 1 is the Cauchy principal value; n != 1.
pub fn ellip_kep(kc2: f64, n: f64) -> (f64, f64, f64) {
    let kc2 = if kc2 > 1.0e-300 { kc2 } else { 1.0e-300 };
    let (k, e) = ellip_ke(kc2);
    let m = 1.0 - kc2;
    let pi = if n < 1.0 {
        if n != 0.0 {
            k + (n / 3.0) * carlson_rj(0.0, kc2, 1.0, 1.0 - n)
        } else {
            k
        }
    } else {
        -(m / (3.0 * n)) * carlson_rj(0.0, kc2, 1.0, 1.0 - m / n)
    };
    (k, e, pi)
}

/// `_ellipk(k2)`.
pub fn ellipk(k2: f64) -> f64 {
    carlson_rf(0.0, 1.0 - k2, 1.0)
}

/// `_ellipe(k2)`.
pub fn ellipe(k2: f64) -> f64 {
    ellip_ke(1.0 - k2).1
}

/// `_ellippi(n, k2)`.
pub fn ellippi(n: f64, k2: f64) -> f64 {
    ellip_kep(1.0 - k2, n).2
}
