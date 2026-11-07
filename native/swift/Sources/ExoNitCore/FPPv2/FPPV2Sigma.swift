import Foundation

/// The (q') noise model up to the low-order phase-curve fit; `FPPV2Sigma.sigmaComponents` completes
/// it with the specified Householder QR (`phaseResidHouseholder`) and the beta factors. When fewer
/// than 20 usable bins exist the reference returns before the fit with `sigma_b = sigma_oot`
/// everywhere (`fitNeeded == false`, `sigmaB` filled).
public struct FPPV2SigmaBase: Sendable {
    public var sigmaPt: Double
    public var sigmaOot: Double
    /// `sigma_pt / sqrt(max(count, 1))` per fold bin (the white-noise expectation of a mean bin).
    public var white: [Double]
    /// Median of `white` over the usable bins (NaN when there are fewer than 20).
    public var wMed: Double
    /// Number of usable bins (out of transit and at least 5 points).
    public var nUsableBins: Int
    /// True when the reference would run the least-squares fit (>= 20 usable bins): beta / sigma_b
    /// are then NOT computed here.
    public var fitNeeded: Bool
    /// Usable bins: `time_pf / period` and the mean flux (inputs of the fit); empty when not `fitNeeded`.
    public var xb: [Double] = []
    public var yb: [Double] = []
    /// The duration and period of the call (for the T14 block size).
    public var duration = 0.0
    public var period = 0.0
    /// `sigma_oot` in every bin when `fitNeeded` is false, otherwise empty.
    public var sigmaB: [Double]
}

/// The (q') likelihood sigma of the fixed model (D15, plan 0.3 (q')): first-difference point noise,
/// the scalar `sigma_oot`. Transcription of `_mad`, `_sigma_components_from_fold`,
/// `sigma_components` of worker/fpp_native.py up to the least-squares fit.
public enum FPPV2Sigma {
    public static let fitIllConditionedRcond = 1.0e-8
    /// `_mad(x)`: 1.4826 * median(|x - median(x)|).
    public static func mad(_ x: [Double]) -> Double {
        let m = FPPNumpy.median(x)
        let dev = x.map { abs($0 - m) }
        return 1.4826 * FPPNumpy.median(dev)
    }

    /// `_sigma_components_from_fold(...)` up to the fit.
    public static func sigmaBaseFromFold(
        time: [Double], flux: [Double], period: Double, t0: Double, duration: Double,
        timePF: [Double], fluxPF: [Double], counts: [Double]
    ) -> FPPV2SigmaBase {
        let guardD = 0.75 * duration + 0.02
        let nt = time.count
        var ok = [Bool](repeating: false, count: nt)
        for i in 0..<nt {
            let ph = FPPV2.pyMod((time[i] - t0) / period + 0.5, 1.0) - 0.5
            let dt = ph * period
            let dsec = (abs(ph) - 0.5) * period
            ok[i] = abs(dt) > guardD && abs(dsec) > guardD && flux[i].isFinite
        }
        let order = (0..<nt).sorted { (time[$0], $0) < (time[$1], $1) }  // stable argsort
        let tt = order.map { time[$0] }
        let ff = order.map { flux[$0] }
        let oo = order.map { ok[$0] }
        var dtt: [Double] = []
        if tt.count >= 2 { for i in 0..<(tt.count - 1) { dtt.append(tt[i + 1] - tt[i]) } }
        let cad = tt.count > 2 ? FPPNumpy.median(dtt) : 1.0
        var d: [Double] = []
        for i in 0..<dtt.count where oo[i + 1] && oo[i] && dtt[i] < 3.0 * cad {
            d.append(ff[i + 1] - ff[i])
        }
        let nOk = ok.filter { $0 }.count
        let sigPt: Double
        if d.count >= 50 {
            let md = FPPNumpy.median(d)
            let dev = d.map { abs($0 - md) }
            sigPt = 1.4826 * FPPNumpy.median(dev) / 2.0.squareRoot()
        } else if nOk >= 20 {
            sigPt = mad((0..<nt).filter { ok[$0] }.map { flux[$0] })
        } else {
            sigPt = FPPNumpy.std(flux)
        }

        let npf = timePF.count
        let ob: [Bool] = (0..<npf).map {
            abs(timePF[$0]) > guardD && abs(abs(timePF[$0]) - 0.5 * period) > guardD
        }
        let io = (0..<npf).filter { ob[$0] }
        var ia: [Int] = []
        if io.count > 1 {
            for j in 1..<io.count
            where io[j] - io[j - 1] == 1 && (timePF[io[j]] - timePF[io[j - 1]]) < 1.5 * period / 1000.0 {
                ia.append(io[j])
            }
        }
        let nOb = ob.filter { $0 }.count
        var sigOot: Double
        if ia.count >= 20 {
            let dd = ia.map { fluxPF[$0] - fluxPF[$0 - 1] }
            let md = FPPNumpy.median(dd)
            let dev = dd.map { abs($0 - md) }
            sigOot = 1.4826 * FPPNumpy.median(dev) / 2.0.squareRoot()
        } else if nOb >= 20 {
            sigOot = mad((0..<npf).filter { ob[$0] }.map { fluxPF[$0] })
        } else {
            sigOot = FPPNumpy.std(fluxPF)
        }
        // sig_oot = max(sig_oot, 1e-9): Python max returns the 2nd argument only when it is strictly larger
        if 1e-9 > sigOot { sigOot = 1e-9 }
        let n = counts.map { $0 > 1.0 ? $0 : 1.0 }
        let white = n.map { sigPt / $0.squareRoot() }
        let okb: [Bool] = (0..<npf).map { ob[$0] && n[$0] >= 5.0 }
        let nOkb = okb.filter { $0 }.count
        if nOkb < 20 {
            return FPPV2SigmaBase(
                sigmaPt: sigPt, sigmaOot: sigOot, white: white, wMed: .nan, nUsableBins: nOkb,
                fitNeeded: false, sigmaB: [Double](repeating: sigOot, count: npf))
        }
        let wMed = FPPNumpy.median((0..<npf).filter { okb[$0] }.map { white[$0] })
        return FPPV2SigmaBase(
            sigmaPt: sigPt, sigmaOot: sigOot, white: white, wMed: wMed, nUsableBins: nOkb,
            fitNeeded: true, xb: (0..<npf).filter { okb[$0] }.map { timePF[$0] / period },
            yb: (0..<npf).filter { okb[$0] }.map { fluxPF[$0] }, duration: duration, period: period,
            sigmaB: [])
    }

    /// `_phase_resid_householder(xb, y)`: residual of the low-order phase-curve fit by the SPECIFIED
    /// Householder QR (no LAPACK). Literal transcription of the normative spec in
    /// worker/fpp_native.py: design row `[1, cos t, sin t, cos 2t, sin 2t]` with `t = (2 pi) * xb`
    /// and `(4 pi) * xb`, scalar libm cos / sin through the non-inlinable shims; every sum
    /// sequential with i ascending from 0.0; `f = (2.0 * s) / vn2`; `alpha = -norm` when
    /// `R[j][j] >= 0`; no pivoting, no FMA; a zero `R[j][j]` gives `coef = 0`, `vn2 == 0` skips the
    /// reflection.
    public static func phaseResidHouseholder(_ xb: [Double], _ y: [Double]) -> [Double] {
        phaseResidHouseholderWithRcond(xb, y).residuals
    }

    /// Returns the residuals and the reference's diagonal ratio from the same QR pass.
    public static func phaseResidHouseholderWithRcond(_ xb: [Double], _ y: [Double]) -> (residuals: [Double], rcond: Double) {
        let n = xb.count
        let p = 5
        let twoPi = 2.0 * Double.pi
        let fourPi = 4.0 * Double.pi
        var a = [Double](repeating: 0.0, count: n * p)  // row-major, a[i * p + c]
        for i in 0..<n {
            let th = twoPi * xb[i]
            let th2 = fourPi * xb[i]
            a[i * p] = 1.0
            a[i * p + 1] = FPPV2.cos(th)
            a[i * p + 2] = FPPV2.sin(th)
            a[i * p + 3] = FPPV2.cos(th2)
            a[i * p + 4] = FPPV2.sin(th2)
        }
        var r = a
        var b = y
        var v = [Double](repeating: 0.0, count: n)
        for j in 0..<p {
            var s = 0.0
            for i in j..<n { s += r[i * p + j] * r[i * p + j] }
            let norm = s.squareRoot()
            let alpha = r[j * p + j] >= 0.0 ? -norm : norm
            v[j] = r[j * p + j] - alpha
            if j + 1 < n { for i in (j + 1)..<n { v[i] = r[i * p + j] } }
            var vn2 = 0.0
            for i in j..<n { vn2 += v[i] * v[i] }
            if vn2 == 0.0 { continue }
            for c in j..<p {
                var s2 = 0.0
                for i in j..<n { s2 += v[i] * r[i * p + c] }
                let f = 2.0 * s2 / vn2
                for i in j..<n { r[i * p + c] -= f * v[i] }
            }
            var s3 = 0.0
            for i in j..<n { s3 += v[i] * b[i] }
            let f = 2.0 * s3 / vn2
            for i in j..<n { b[i] -= f * v[i] }
        }
        var coef = [Double](repeating: 0.0, count: p)
        for j in stride(from: p - 1, through: 0, by: -1) {
            var s = b[j]
            if j + 1 < p { for c in (j + 1)..<p { s -= r[j * p + c] * coef[c] } }
            coef[j] = r[j * p + j] != 0.0 ? s / r[j * p + j] : 0.0
        }
        var dmin = abs(r[0])
        var dmax = dmin
        for j in 1..<p {
            let d = abs(r[j * p + j])
            if d < dmin { dmin = d }
            if d > dmax { dmax = d }
        }
        let rcond = dmax > 0.0 ? dmin / dmax : 0.0
        var out = [Double](repeating: 0.0, count: n)
        for i in 0..<n {
            var s = a[i * p] * coef[0]
            for c in 1..<p { s += a[i * p + c] * coef[c] }
            out[i] = y[i] - s
        }
        return (out, rcond)
    }

    /// All components of the (q') noise model (the reference dict).
    public struct Components: Sendable {
        public var sigmaPt: Double
        public var betaBin: Double
        public var betaT14: Double
        public var beta: Double
        public var sigmaOot: Double
        public var sigmaB: [Double]
        public var fitRcond: Double
        public var fitIllConditioned: Bool
    }

    /// Completes a `FPPV2SigmaBase` with the QR fit and the beta factors
    /// (`_sigma_components_from_fold` after the early return).
    public static func components(from base: FPPV2SigmaBase) -> Components {
        if !base.fitNeeded {
            return Components(
                sigmaPt: base.sigmaPt, betaBin: .nan, betaT14: .nan, beta: .nan,
                sigmaOot: base.sigmaOot, sigmaB: base.sigmaB,
                fitRcond: 1.0, fitIllConditioned: false)
        }
        let fit = phaseResidHouseholderWithRcond(base.xb, base.yb)
        let r = fit.residuals
        let madR = mad(r)
        let betaBin = madR / base.wMed
        let nbF = (base.duration / (base.period / 1000.0)).rounded(.toNearestOrEven)
        let nb = nbF > 2.0 ? Int(nbF) : 2
        let k = r.count / nb
        let betaT14: Double
        if k >= 8 {
            let means = (0..<k).map { i in
                FPPNumpy.pairwiseSum(Array(r[(i * nb)..<((i + 1) * nb)])) / Double(nb)
            }
            betaT14 = mad(means) / (madR / Double(nb).squareRoot())
        } else {
            betaT14 = 1.0
        }
        // beta = min(max(1.0, beta_bin, beta_T14), 5.0)
        var beta = 1.0
        if betaBin > beta { beta = betaBin }
        if betaT14 > beta { beta = betaT14 }
        if beta > 5.0 { beta = 5.0 }
        return Components(
            sigmaPt: base.sigmaPt, betaBin: betaBin, betaT14: betaT14, beta: beta,
            sigmaOot: base.sigmaOot, sigmaB: base.white.map { $0 * beta },
            fitRcond: fit.rcond, fitIllConditioned: fit.rcond < fitIllConditionedRcond)
    }

    /// `sigma_components(time, flux, period, t0, duration)`: the full (q') components; `nil` for an
    /// empty fold.
    public static func sigmaComponents(
        time: [Double], flux: [Double], period: Double, t0: Double, duration: Double
    ) -> Components? {
        sigmaBase(time: time, flux: flux, period: period, t0: t0, duration: duration).map(components(from:))
    }

    /// `np.nanmean` of an array (NaN -> 0 in the pairwise sum, divided by the non-NaN count).
    static func nanmean(_ v: [Double]) -> Double {
        let cnt = v.filter { !$0.isNaN }.count
        let cleaned = v.map { $0.isNaN ? 0.0 : $0 }
        return FPPNumpy.pairwiseSum(cleaned) / Double(cnt)
    }

    /// `sigma_components(time, flux, period, t0, duration)` up to the fit: the mean fold is built as
    /// the reference does (`np.nanmean` per bin, bin centre x period as the bin time). `nil` for an
    /// empty fold.
    public static func sigmaBase(
        time: [Double], flux: [Double], period: Double, t0: Double, duration: Double
    ) -> FPPV2SigmaBase? {
        let nBins = 1000
        let phase: [Double] = time.map { t in
            var p = FPPV2.pyMod((t - t0) / period, 1.0)
            if p > 0.5 { p -= 1.0 }
            return p
        }
        let edges = FPPV2Fold.foldEdges(nBins)
        let centres = (0..<nBins).map { 0.5 * (edges[$0] + edges[$0 + 1]) }
        let binIdx: [Int] = phase.map { min(max(FPPV2Fold.digitize(edges, $0) - 1, 0), nBins - 1) }
        var perBin = [[Double]](repeating: [], count: nBins)
        for i in 0..<time.count { perBin[binIdx[i]].append(flux[i]) }  // input order kept inside every bin
        var tOut: [Double] = [], fOut: [Double] = [], cnt: [Double] = []
        for i in 0..<nBins where !perBin[i].isEmpty {
            tOut.append(centres[i] * period)
            fOut.append(nanmean(perBin[i]))
            cnt.append(Double(perBin[i].count))
        }
        if tOut.isEmpty { return nil }
        let keep = (0..<tOut.count).filter { tOut[$0].isFinite && fOut[$0].isFinite }
        return sigmaBaseFromFold(
            time: time, flux: flux, period: period, t0: t0, duration: duration,
            timePF: keep.map { tOut[$0] }, fluxPF: keep.map { fOut[$0] }, counts: keep.map { cnt[$0] })
    }
}
