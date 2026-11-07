import Foundation

/// Orbit geometry of the fixed model (D1, D2): Kepler solve, true anomaly, mid-transit mean anomaly,
/// the projected separation z(t) with the far side flagged `z = 1e3`, (1/2) d(z^2)/dt, and the
/// parity-mode exposure-supersampled light curve. Literal transcription of `_kepler_eq`,
/// `_true_anomaly`, `_mid_transit_mean_anomaly`, `_z_one`, `_compute_z_arr`, `_half_dz2_dt`,
/// `_lc_core` of worker/fpp_native.py.
public enum FPPV2Orbit {
    /// Safeguarded solver used by the corrected Python reference. The tuple
    /// carries convergence so callers can count failed draws in the result.
    public static func keplerSolve(_ inputM: Double, _ ecc: Double, maxIterations: Int = 100) -> (eccentricAnomaly: Double, converged: Bool) {
        let pi = FPPV2Consts.pi
        let twoPi = 2.0 * pi
        var m = inputM
        if m < 0.0 && m >= -1.0e-9 {
            m = 0.0
        } else if m > twoPi && m <= twoPi + 1.0e-9 {
            m = twoPi
        }
        guard m >= 0.0 && m <= twoPi && ecc >= 0.0 && ecc < 1.0 else {
            return (.nan, false)
        }
        var lo = 0.0
        var hi = twoPi
        var e = m
        for it in 0..<max(0, maxIterations) {
            let f = (e - ecc * FPPV2.sin(e)) - m
            if f == 0.0 { return (e, true) }
            if f > 0.0 { hi = e } else { lo = e }
            let de = f / (1.0 - ecc * FPPV2.cos(e))
            let en = e - de
            let tiny = abs(de) < 1.0e-10
            if it < 30 && ((en > lo && en < hi) || tiny) {
                e = en
                if tiny { return (e, true) }
            } else {
                let absE = abs(e)
                let scale = absE > 1.0 ? absE : 1.0
                if hi - lo <= 1.0e-15 * scale { return (e, true) }
                e = lo + 0.5 * (hi - lo)
            }
        }
        return (e, false)
    }

    /// `_kepler_eq`; an unconverged solve maps to NaN.
    public static func keplerEq(_ m: Double, _ ecc: Double) -> Double {
        let solved = keplerSolve(m, ecc)
        return solved.converged ? solved.eccentricAnomaly : .nan
    }

    /// `_true_anomaly(M, ecc)`.
    public static func trueAnomaly(_ m: Double, _ ecc: Double) -> Double {
        if ecc < 1e-10 { return m }
        let e = keplerEq(m, ecc)
        let cosE = FPPV2.cos(e)
        return FPPV2.atan2((1.0 - ecc * ecc).squareRoot() * FPPV2.sin(e), cosE - ecc)
    }

    /// `_mid_transit_mean_anomaly(ecc, w_rad)`.
    public static func midTransitMeanAnomaly(_ ecc: Double, _ wRad: Double) -> Double {
        let nuT = Double.pi / 2.0 - wRad
        let eT = 2.0 * FPPV2.atan2((1.0 - ecc).squareRoot() * FPPV2.sin(nuT / 2.0), (1.0 + ecc).squareRoot() * FPPV2.cos(nuT / 2.0))
        return eT - ecc * FPPV2.sin(eT)
    }

    /// `_z_one(t, t0, period, a_Rs, cos_i, ecc, w_rad, M_t)`.
    public static func zOne(
        _ t: Double, _ t0: Double, _ period: Double, _ aRs: Double, _ cosI: Double,
        _ ecc: Double, _ wRad: Double, _ mT: Double
    ) -> Double {
        let twoPi = 2.0 * Double.pi
        let phase = (t - t0) / period
        var m = twoPi * phase + mT
        m -= twoPi * (m / twoPi).rounded(.down)
        let nu = trueAnomaly(m, ecc)
        let rFactor = (1.0 - ecc * ecc) / (1.0 + ecc * FPPV2.cos(nu))
        let f = nu + wRad - Double.pi / 2.0
        let sinF = FPPV2.sin(f)
        let cosF = FPPV2.cos(f)
        if cosF < 0.0 { return 1.0e3 }  // companion behind the star: no transit
        return aRs * rFactor * (sinF * sinF + cosI * cosI * cosF * cosF).squareRoot()
    }

    /// `_compute_z_arr(time, t0, period, a_Rs, inc_rad, ecc, w_rad)`.
    public static func computeZArr(
        _ time: [Double], _ t0: Double, _ period: Double, _ aRs: Double, _ incRad: Double,
        _ ecc: Double, _ wRad: Double
    ) -> [Double] {
        let cosI = FPPV2.cos(incRad)
        let mT = midTransitMeanAnomaly(ecc, wRad)
        return time.map { zOne($0, t0, period, aRs, cosI, ecc, wRad, mT) }
    }

    /// `_half_dz2_dt(t, period, a_Rs, inc_rad, ecc, w_rad, M_t)`: (1/2) d(z^2)/dt in R_s^2 / day.
    public static func halfDz2Dt(
        _ t: Double, _ period: Double, _ aRs: Double, _ incRad: Double, _ ecc: Double,
        _ wRad: Double, _ mT: Double
    ) -> Double {
        let twoPi = 2.0 * Double.pi
        let n = twoPi / period
        var m = twoPi * t / period + mT
        m -= twoPi * (m / twoPi).rounded(.down)
        let nu = trueAnomaly(m, ecc)
        let oneMinusE2 = 1.0 - ecc * ecc
        let q = 1.0 + ecc * FPPV2.cos(nu)
        let r = aRs * oneMinusE2 / q
        let nuDot = n * q * q / FPPV2.pow(oneMinusE2, 1.5)
        let rDot = n * aRs * ecc * FPPV2.sin(nu) / oneMinusE2.squareRoot()
        let th = wRad + nu
        let cosTh = FPPV2.cos(th)
        let sinTh = FPPV2.sin(th)
        let cosI = FPPV2.cos(incRad)
        let x = -r * cosTh
        let y = -r * sinTh * cosI
        let xd = -rDot * cosTh + r * sinTh * nuDot
        let yd = (-rDot * sinTh - r * cosTh * nuDot) * cosI
        return x * xd + y * yd
    }

    /// `_lc_core(...)`: transit light curve with the PARITY-mode exposure supersampling (end points
    /// included); `nsub <= 1` or `exptime <= 0` evaluates the kernel at the cadence times.
    public static func lcCore(
        time: [Double], k: Double, t0: Double, period: Double, aRs: Double, incRad: Double,
        ecc: Double, wRad: Double, u1: Double, u2: Double, exptime: Double, nsub: Int
    ) -> [Double] {
        let cosI = FPPV2.cos(incRad)
        let mT = midTransitMeanAnomaly(ecc, wRad)
        if nsub <= 1 || exptime <= 0.0 {
            return time.map {
                FPPV2Kernel.transitFlux(zOne($0, t0, period, aRs, cosI, ecc, wRad, mT), k, u1, u2)
            }
        }
        let nsubF = Double(nsub)
        return time.map { t in
            var acc = 0.0
            for j in 0..<nsub {
                let dt = exptime * (Double(j) / (nsubF - 1.0) - 0.5)
                let z = zOne(t + dt, t0, period, aRs, cosI, ecc, wRad, mT)
                acc += FPPV2Kernel.transitFlux(z, k, u1, u2)
            }
            return acc / nsubF
        }
    }
}
