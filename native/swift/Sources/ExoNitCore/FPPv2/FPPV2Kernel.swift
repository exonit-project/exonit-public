import Foundation

/// Analytic Mandel & Agol (2002) quadratic-limb-darkening occultation flux, clean-room closed forms
/// with K / E / Pi from `FPPV2Ellip`: literal transcription of `_transit_flux_core` /
/// `_transit_flux` of worker/fpp_native.py (acos of the paper replaced by atan2 of the factored
/// root S).
public enum FPPV2Kernel {
    /// `_transit_flux_core(z, p, u1, u2)`.
    public static func transitFluxCore(_ z: Double, _ p: Double, _ u1: Double, _ u2: Double) -> Double {
        let pi = Double.pi
        let c2 = u1 + 2.0 * u2
        let c4 = -u2
        let fo = 1.0 - c2 / 3.0 - c4 / 2.0
        if p <= 0.0 || z >= 1.0 + p { return 1.0 }
        if p >= 1.0 && z <= p - 1.0 { return 0.0 }
        let a = (z - p) * (z - p)
        let b = (z + p) * (z + p)
        let q = p * p - z * z
        var theta: Double = p > z ? 1.0 : 0.0
        let lamE: Double
        let etaD: Double
        let lamD: Double
        if p < 1.0 && z <= 1.0 - p {
            // planet fully inside the stellar disc
            lamE = p * p
            etaD = 0.5 * p * p * (p * p + 2.0 * z * z)
            if z == 0.0 {
                lamD = -(2.0 / 3.0) * FPPV2.pow(1.0 - p * p, 1.5)
            } else if z == p && p < 0.5 {
                let kc = (1.0 - 4.0 * p * p).squareRoot()
                let (k, e) = FPPV2Ellip.ellipKE(kc * kc)
                lamD = 1.0 / 3.0
                    + (2.0 / (9.0 * pi)) * (4.0 * (2.0 * p * p - 1.0) * e + (1.0 - 4.0 * p * p) * k)
                theta = 0.0
            } else if z == p && p == 0.5 {
                lamD = 1.0 / 3.0 - 4.0 / (9.0 * pi)
                theta = 0.0
            } else if z == 1.0 - p {
                lamD = (2.0 / (3.0 * pi)) * FPPV2.acos(1.0 - 2.0 * p)
                    - (4.0 / (9.0 * pi)) * (3.0 + 2.0 * p - 8.0 * p * p) * (p * (1.0 - p)).squareRoot()
                    - (2.0 / 3.0) * (p > 0.5 ? 1.0 : 0.0)
            } else {
                let kc = (((1.0 - z - p) * (1.0 + z + p)) / ((1.0 - z + p) * (1.0 + z - p))).squareRoot()
                let n = (a - b) / a
                let (k, e, pip) = FPPV2Ellip.ellipKEP(kc * kc, n)
                lamD = (2.0 / (9.0 * pi * (1.0 - a).squareRoot()))
                    * ((1.0 - 5.0 * z * z + p * p + q * q) * k
                        + (1.0 - a) * (z * z + 7.0 * p * p - 4.0) * e
                        - 3.0 * (q / a) * pip)
            }
        } else {
            // partial overlap, |1 - p| < z < 1 + p (p > 1 allowed)
            let prod = (1.0 - z + p) * (1.0 + z - p) * (z + p - 1.0) * (z + p + 1.0)
            let s = (prod > 0.0 ? prod : 0.0).squareRoot()  // max(0.0, prod)
            let k0 = FPPV2.atan2(s, p * p + z * z - 1.0)
            let k1 = FPPV2.atan2(s, 1.0 - p * p + z * z)
            lamE = (p * p * k0 + k1 - 0.5 * s) / pi
            let eta2 = p * p * (p * p + 2.0 * z * z) / 2.0
            etaD = (k1 + 2.0 * eta2 * k0 - 0.25 * (1.0 + 5.0 * p * p + z * z) * s) / (2.0 * pi)
            if z == p {
                if p >= 0.5 {
                    let kc = (1.0 - 1.0 / (4.0 * p * p)).squareRoot()
                    let (k, e) = FPPV2Ellip.ellipKE(kc * kc)
                    lamD = 1.0 / 3.0 + 16.0 * p / (9.0 * pi) * (2.0 * p * p - 1.0) * e
                        - (1.0 - 4.0 * p * p) * (3.0 - 8.0 * p * p) / (9.0 * pi * p) * k
                    theta = 0.0
                } else {
                    lamD = 0.0
                }
            } else {
                let kc = (((z + p - 1.0) * (z + p + 1.0)) / (4.0 * p * z)).squareRoot()
                let n = (a - 1.0) / a
                let (k, e, pip) = FPPV2Ellip.ellipKEP(kc * kc, n)
                lamD = (1.0 / (9.0 * pi * (p * z).squareRoot()))
                    * (((1.0 - b) * (2.0 * b + a - 3.0) - 3.0 * q * (b - 2.0)) * k
                        + 4.0 * p * z * (z * z + 7.0 * p * p - 4.0) * e
                        - 3.0 * (q / a) * pip)
            }
        }
        return 1.0 - ((1.0 - c2) * lamE + c2 * (lamD + (2.0 / 3.0) * theta) - c4 * etaD) / fo
    }

    /// `_transit_flux(z, p, u1, u2)`: the removable singularity at z == p is bridged by linear
    /// interpolation.
    public static func transitFlux(_ z: Double, _ p: Double, _ u1: Double, _ u2: Double) -> Double {
        if p > 0.0 && z > 0.0 && p != z && abs(z - p) < 1.0e-6 && z < 1.0 + p
            && !(p >= 1.0 && z <= p - 1.0)
        {
            let h = 1.0e-5
            let f0 = transitFluxCore(p, p, u1, u2)
            let z1 = p + (z > p ? h : -h)
            let f1 = transitFluxCore(z1, p, u1, u2)
            return f0 + (z - p) / (z1 - p) * (f1 - f0)
        }
        return transitFluxCore(z, p, u1, u2)
    }
}
