import Foundation

/// Complete elliptic integrals K, E, Pi through Carlson's symmetric forms R_F / R_C / R_J
/// (Carlson 1995, DLMF 19.36): literal transcription of `_carlson_rf/_rc/_rj`, `_ellip_ke`,
/// `_ellip_kep`, `_ellipk`, `_ellipe`, `_ellippi` of worker/fpp_native.py. Operation order is
/// the Python order.
public enum FPPV2Ellip {
    static let carlsonTol = 1.0e-3

    /// `max(abs(dx), abs(dy), abs(dz))` as numba evaluates the 3-argument builtin (left to right).
    @inline(__always)
    static func max3(_ a: Double, _ b: Double, _ c: Double) -> Double {
        let m = b > a ? b : a
        return c > m ? c : m
    }

    public static func carlsonRF(_ x0: Double, _ y0: Double, _ z0: Double) -> Double {
        var x = x0, y = y0, z = z0
        var mu = 0.0, dx = 0.0, dy = 0.0, dz = 0.0
        for _ in 0..<1100 {
            let sx = x.squareRoot()
            let sy = y.squareRoot()
            let sz = z.squareRoot()
            let lam = sx * sy + sy * sz + sz * sx
            x = 0.25 * (x + lam)
            y = 0.25 * (y + lam)
            z = 0.25 * (z + lam)
            mu = (x + y + z) / 3.0
            dx = 1.0 - x / mu
            dy = 1.0 - y / mu
            dz = 1.0 - z / mu
            if max3(abs(dx), abs(dy), abs(dz)) < carlsonTol { break }
        }
        let e2 = dx * dy - dz * dz
        let e3 = dx * dy * dz
        return (1.0 - e2 / 10.0 + e3 / 14.0 + e2 * e2 / 24.0 - 3.0 * e2 * e3 / 44.0) / mu.squareRoot()
    }

    public static func carlsonRC(_ x0: Double, _ y0: Double) -> Double {
        var x = x0, y = y0
        var mu = 0.0, s = 0.0
        for _ in 0..<200 {
            let lam = 2.0 * x.squareRoot() * y.squareRoot() + y
            x = 0.25 * (x + lam)
            y = 0.25 * (y + lam)
            mu = (x + 2.0 * y) / 3.0
            s = (y - mu) / mu
            if abs(s) < carlsonTol { break }
        }
        return (1.0 + s * s * (0.3 + s * (1.0 / 7.0 + s * (0.375 + s * 9.0 / 22.0)))) / mu.squareRoot()
    }

    public static func carlsonRJ(_ x0: Double, _ y0: Double, _ z0: Double, _ p0: Double) -> Double {
        var x = x0, y = y0, z = z0, p = p0
        var total = 0.0
        var fac = 1.0
        var mu = 0.0, dx = 0.0, dy = 0.0, dz = 0.0, dp = 0.0
        for _ in 0..<1100 {
            let sx = x.squareRoot()
            let sy = y.squareRoot()
            let sz = z.squareRoot()
            let lam = sx * sy + sy * sz + sz * sx
            let t = p * (sx + sy + sz) + sx * sy * sz
            let alpha = t * t  // numba: x ** 2 -> x * x
            let u = p + lam
            let beta = p * (u * u)
            total += fac * carlsonRC(alpha, beta)
            fac *= 0.25
            x = 0.25 * (x + lam)
            y = 0.25 * (y + lam)
            z = 0.25 * (z + lam)
            p = 0.25 * (p + lam)
            mu = (x + y + z + 2.0 * p) / 5.0
            dx = (mu - x) / mu
            dy = (mu - y) / mu
            dz = (mu - z) / mu
            dp = (mu - p) / mu
            let m1 = abs(dy) > abs(dx) ? abs(dy) : abs(dx)
            let m2 = abs(dp) > abs(dz) ? abs(dp) : abs(dz)
            let m = m2 > m1 ? m2 : m1
            if m < carlsonTol { break }
        }
        let e2 = dx * dy + dx * dz + dy * dz - 3.0 * dp * dp  // (3.0 * dp) * dp, left to right
        let e3 = dx * dy * dz + 2.0 * e2 * dp + 4.0 * (dp * (dp * dp))  // dp ** 3 -> dp * (dp * dp)
        let e4 = (2.0 * dx * dy * dz + e2 * dp + 3.0 * (dp * (dp * dp))) * dp
        let e5 = dx * dy * dz * dp * dp
        let series = 1.0 - 3.0 * e2 / 14.0 + e3 / 6.0 + 9.0 * e2 * e2 / 88.0
            - 3.0 * e4 / 22.0
            - 9.0 * e2 * e3 / 52.0
            + 3.0 * e5 / 26.0
        return 3.0 * total + fac * FPPV2.pow(mu, -1.5) * series
    }

    /// `_ellip_ke`: (K, E) of the modulus whose complementary parameter is `kc2 = 1 - m`.
    public static func ellipKE(_ kc2In: Double) -> (k: Double, e: Double) {
        let kc2 = kc2In > 1.0e-300 ? kc2In : 1.0e-300  // max(kc2, 1e-300)
        let k = carlsonRF(0.0, kc2, 1.0)
        let e = k - ((1.0 - kc2) / 3.0) * carlsonRJ(0.0, kc2, 1.0, 1.0)
        return (k, e)
    }

    /// `_ellip_kep`: (K, E, Pi(n, m)) with m = 1 - kc2; n > 1 is the Cauchy principal value; n != 1.
    public static func ellipKEP(_ kc2In: Double, _ n: Double) -> (k: Double, e: Double, pi: Double) {
        let kc2 = kc2In > 1.0e-300 ? kc2In : 1.0e-300
        let (k, e) = ellipKE(kc2)
        let m = 1.0 - kc2
        let pi: Double
        if n < 1.0 {
            if n != 0.0 {
                pi = k + (n / 3.0) * carlsonRJ(0.0, kc2, 1.0, 1.0 - n)
            } else {
                pi = k
            }
        } else {
            pi = -(m / (3.0 * n)) * carlsonRJ(0.0, kc2, 1.0, 1.0 - m / n)
        }
        return (k, e, pi)
    }

    /// `_ellipk(k2)`.
    public static func ellipK(_ k2: Double) -> Double { carlsonRF(0.0, 1.0 - k2, 1.0) }
    /// `_ellipe(k2)`.
    public static func ellipE(_ k2: Double) -> Double { ellipKE(1.0 - k2).e }
    /// `_ellippi(n, k2)`.
    public static func ellipPi(_ n: Double, _ k2: Double) -> Double { ellipKEP(1.0 - k2, n).pi }
}
