import Foundation

/// Exact contact times of one transit (plan 0.3 (j'')): bisection of z(t) - level and of
/// (1/2) d(z^2)/dt. Literal transcription of `_bisect` and `_contact_times_nb` of
/// worker/fpp_native.py.
public enum FPPV2Contact {
    static func bisect(
        _ floIn: Double, _ fhi: Double, _ tloIn: Double, _ thiIn: Double, _ period: Double,
        _ aRs: Double, _ incRad: Double, _ ecc: Double, _ wRad: Double, _ mT: Double,
        _ level: Double, _ isZ: Bool
    ) -> Double {
        let tol = 1e-13
        var flo = floIn, tlo = tloIn, thi = thiIn
        if flo == 0.0 { return tlo }
        if fhi == 0.0 { return thi }
        if flo * fhi > 0.0 { return .nan }
        for _ in 0..<100 {
            if thi - tlo < tol { break }
            let tm = 0.5 * (tlo + thi)
            let fm: Double
            if isZ {
                fm = FPPV2Orbit.zOne(tm, 0.0, period, aRs, FPPV2.cos(incRad), ecc, wRad, mT) - level
            } else {
                fm = FPPV2Orbit.halfDz2Dt(tm, period, aRs, incRad, ecc, wRad, mT)
            }
            if fm == 0.0 { return tm }
            if (fm > 0.0) == (flo > 0.0) {
                tlo = tm
                flo = fm
            } else {
                thi = tm  // the reference also updates fhi here; it is never read afterwards
            }
        }
        return 0.5 * (tlo + thi)
    }

    /// `_contact_times_nb(period, a_Rs, inc_rad, ecc, w_rad, k)` -> `[t1, t2, t3, t4, t_min]`,
    /// NaN where a contact is absent, days relative to mid-transit.
    public static func contactTimes(
        period: Double, aRs: Double, incRad: Double, ecc: Double, wRad: Double, k: Double
    ) -> [Double] {
        contactTimesZMin(period: period, aRs: aRs, incRad: incRad, ecc: ecc, wRad: wRad, k: k).0
    }

    /// The same roots and the separation at t_min. zMin is NaN if the search never reaches t_min;
    /// a near-tangent miss retains zMin even though all five roots are NaN.
    public static func contactTimesZMin(
        period: Double, aRs: Double, incRad: Double, ecc: Double, wRad: Double, k: Double
    ) -> (roots: [Double], zMin: Double) {
        var out = [Double](repeating: .nan, count: 5)
        let cosI = FPPV2.cos(incRad)
        let mT = FPPV2Orbit.midTransitMeanAnomaly(ecc, wRad)

        // Longest possible transit scale (apastron geometry), never past half an orbit, never below 1e-3 d.
        var w = 2.5 * (1.0 + k) * period / (Double.pi * aRs) / (1.0 - ecc)
        if w > 0.5 * period { w = 0.5 * period }
        if w < 1e-3 { w = 1e-3 }

        let ngrid = 2001
        let step = 2.0 * w / Double(ngrid - 1)
        var bestZ = 1.0e3
        var i0 = -1
        for i in 0..<ngrid {
            let z = FPPV2Orbit.zOne(-w + step * Double(i), 0.0, period, aRs, cosI, ecc, wRad, mT)
            if z < 1.0e3 && z < bestZ {
                bestZ = z
                i0 = i
            }
        }
        if i0 < 0 || bestZ >= 1.0 + k + 1e-3 { return (out, .nan) }  // no transit anywhere in the window

        var loI = i0 - 1
        if loI < 0 { loI = 0 }
        var hiI = i0 + 1
        if hiI > ngrid - 1 { hiI = ngrid - 1 }
        let tlo = -w + step * Double(loI)
        let thi = -w + step * Double(hiI)
        let tMin = bisect(
            FPPV2Orbit.halfDz2Dt(tlo, period, aRs, incRad, ecc, wRad, mT),
            FPPV2Orbit.halfDz2Dt(thi, period, aRs, incRad, ecc, wRad, mT),
            tlo, thi, period, aRs, incRad, ecc, wRad, mT, 0.0, false)
        if tMin.isNaN { return (out, .nan) }
        let zMin = FPPV2Orbit.zOne(tMin, 0.0, period, aRs, cosI, ecc, wRad, mT)

        let l = 1.0 + k
        if zMin >= l { return (out, zMin) }  // the companion never touches the disc

        // Outer contacts; widen the window (x2, at most 4 times) before giving up.
        var hw = w
        var t1 = Double.nan
        for _ in 0..<5 {
            let a = tMin - hw
            let fa = FPPV2Orbit.zOne(a, 0.0, period, aRs, cosI, ecc, wRad, mT) - l
            let fb = zMin - l
            t1 = bisect(fa, fb, a, tMin, period, aRs, incRad, ecc, wRad, mT, l, true)
            if !t1.isNaN { break }
            hw *= 2.0
        }
        hw = w
        var t4 = Double.nan
        for _ in 0..<5 {
            let b = tMin + hw
            let fa = zMin - l
            let fb = FPPV2Orbit.zOne(b, 0.0, period, aRs, cosI, ecc, wRad, mT) - l
            t4 = bisect(fa, fb, tMin, b, period, aRs, incRad, ecc, wRad, mT, l, true)
            if !t4.isNaN { break }
            hw *= 2.0
        }

        // Inner contacts: only when the companion is fully inside (or fully in front of) the disc at mid-transit.
        var t2 = Double.nan
        var t3 = Double.nan
        let lin = abs(1.0 - k)
        if zMin < lin {
            if !t1.isNaN {
                t2 = bisect(l - lin, zMin - lin, t1, tMin, period, aRs, incRad, ecc, wRad, mT, lin, true)
            }
            if !t4.isNaN {
                t3 = bisect(zMin - lin, l - lin, tMin, t4, period, aRs, incRad, ecc, wRad, mT, lin, true)
            }
        }
        out[0] = t1
        out[1] = t2
        out[2] = t3
        out[3] = t4
        out[4] = tMin
        return (out, zMin)
    }
}
