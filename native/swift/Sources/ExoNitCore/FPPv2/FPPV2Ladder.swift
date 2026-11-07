import Foundation

/// Result of `FPPV2Ladder.drawBinModels`: `(N_aff, status, windows evaluated, bins that fell
/// through to the exact level)` plus the affected bin indices and their models.
public struct FPPV2DrawBinModels: Sendable {
    public var nAff = 0
    public var status = 0
    public var nEval = 0
    public var nFb = 0
    public var idx: [Int] = []
    public var m: [Double] = []
    /// Gate observer metadata; no likelihood computation reads these fields.
    public var orderIdx = 0
    public var levels: [Int] = []
    public init() {}
}

/// Fold-bin model ladder (plan 0.3 (j'), N5; every node integrated by the contact-split
/// Gauss-Legendre rule): literal transcription of `_ladder_bin`, `_draw_bin_models` and
/// `production_bin_model` of worker/fpp_native.py.
public enum FPPV2Ladder {
    /// `_ladder_bin`: model `M_b` of one bin -> `(M_b, window means evaluated, level used)`.
    public static func ladderBin(
        _ b: Int, _ w: Double, _ epsB: Double, _ orderIdx: Int, _ fold: FPPV2FoldArrays,
        _ ct: [Double], _ d: FPPV2DrawParams
    ) -> (m: Double, nwin: Int, level: Int) {
        let exact = FPPV2Consts.ladderExactLevel
        let accept = FPPV2Consts.ladderAcceptMaxLevel
        let loP = fold.ptsOff[b]
        let hiP = fold.ptsOff[b + 1]
        let npts = hiP - loP
        var nwin = 0
        if fold.n[b] <= FPPV2Consts.ladderMinPoints {
            var acc = 0.0
            for i in 0..<npts {
                acc += FPPV2GL.windowMean(fold.ptsT[loP + i], w, ct, orderIdx, d)
            }
            return (acc / Double(npts), nwin + npts, exact)
        }

        var vals = [Double](repeating: 0.0, count: exact + 1)
        let wEff = (w * w + 12.0 * fold.varT[b]).squareRoot()
        vals[0] = FPPV2GL.windowMean(fold.tmean[b], wEff, ct, orderIdx, d)
        nwin += 1
        for lvl in 1..<exact {
            let kk = 4 << (lvl - 1)
            var acc = 0.0
            for j in 0..<kk {
                acc += FPPV2GL.windowMean(fold.node(b, lvl - 1, j), w, ct, orderIdx, d)
            }
            vals[lvl] = acc / Double(kk)
            nwin += kk
            if 2 <= lvl && lvl <= accept + 2 {
                let c = lvl - 2
                let d1 = abs(vals[c] - vals[c + 1])
                let d2 = abs(vals[c + 1] - vals[c + 2])
                if d1 <= epsB && d2 <= epsB && (d2 <= d1 || (d1 < 0.05 * epsB && d2 < 0.05 * epsB)) {
                    return (vals[c], nwin, c)
                }
            }
        }

        var acc = 0.0
        for i in 0..<npts {
            acc += FPPV2GL.windowMean(fold.ptsT[loP + i], w, ct, orderIdx, d)
        }
        nwin += npts
        vals[exact] = acc / Double(npts)
        let c = accept
        let d1 = abs(vals[c] - vals[c + 1])
        let d2 = abs(vals[c + 1] - vals[c + 2])
        if d1 <= epsB && d2 <= epsB && (d2 <= d1 || (d1 < 0.05 * epsB && d2 < 0.05 * epsB)) {
            return (vals[c], nwin, c)
        }
        return (vals[exact], nwin, exact)
    }

    /// `_draw_bin_models`: one draw, the ladder model of every fold bin the transit can touch.
    public static func drawBinModels(
        w: Double, delta: Double, fold: FPPV2FoldArrays, period: Double, aRs: Double,
        incRad: Double, ecc: Double, wRad: Double, k: Double, u1: Double, u2: Double,
        dilA: Double, dilB: Double
    ) -> FPPV2DrawBinModels {
        let nb = fold.n.count
        let ct5 = FPPV2Contact.contactTimes(
            period: period, aRs: aRs, incRad: incRad, ecc: ecc, wRad: wRad, k: k)
        let t1 = ct5[0]
        let t4 = ct5[3]
        if t1.isNaN || t4.isNaN { return FPPV2DrawBinModels() }
        let ct = ct5[0..<4].filter { !$0.isNaN }.sorted(by: <)

        let hw = 0.5 * w
        let aff = (0..<nb).filter { fold.tmax[$0] + hw > t1 && fold.tmin[$0] - hw < t4 }
        let nAff = aff.count
        if nAff == 0 { return FPPV2DrawBinModels() }

        let epsRef = (2.0 * delta / Double(nAff)).squareRoot()
        var epsMin = fold.sigma[aff[0]] * epsRef
        for b in aff.dropFirst() {
            let e = fold.sigma[b] * epsRef
            if e < epsMin { epsMin = e }
        }
        let mT = FPPV2Orbit.midTransitMeanAnomaly(ecc, wRad)
        let cosI = FPPV2.cos(incRad)
        let d = FPPV2DrawParams(
            p: period, aRs: aRs, cosI: cosI, ecc: ecc, wRad: wRad, mT: mT, k: k, u1: u1, u2: u2,
            dilA: dilA, dilB: dilB)
        let (orderIdx, status) = FPPV2GL.chooseOrder(ct, w, epsMin, d)
        var out = FPPV2DrawBinModels()
        out.nAff = nAff
        out.status = status
        out.orderIdx = orderIdx
        for b in aff {
            let (mB, nwin, level) = ladderBin(b, w, fold.sigma[b] * epsRef, orderIdx, fold, ct, d)
            out.idx.append(b)
            out.m.append(mB)
            out.levels.append(level)
            out.nEval += nwin
            if level == FPPV2Consts.ladderExactLevel { out.nFb += 1 }
        }
        return out
    }

    /// The fold of uniformly filled bins `[lo[i], hi[i])` exactly as `production_bin_model` builds
    /// it (n = 1000, mean time = the middle, var_t = (hi - lo)^2 / 12, level rungs = uniform
    /// quantile times, exact rung = `ladderKQuant` uniform midpoints), with the likelihood sigma
    /// `sigmaL` in every bin.
    public static func uniformBinsFold(lo: [Double], hi: [Double], sigmaL: Double) -> FPPV2FoldArrays {
        let nb = lo.count
        let levels = FPPV2Fold.nodeLevels
        let maxK = FPPV2Consts.ladderMaxK
        let kq = FPPV2Consts.ladderKQuant
        var f = FPPV2FoldArrays()
        f.n = [Int](repeating: 1000, count: nb)
        f.flux = [Double](repeating: 0.0, count: nb)
        f.sigma = [Double](repeating: sigmaL, count: nb)
        f.tmin = lo
        f.tmax = hi
        f.nodeTab = [Double](repeating: 0.0, count: nb * levels * maxK)
        f.ptsOff = (0...nb).map { $0 * kq }
        for b in 0..<nb {
            f.tmean.append(0.5 * (lo[b] + hi[b]))
            f.varT.append((hi[b] - lo[b]) * (hi[b] - lo[b]) / 12.0)
            let width = hi[b] - lo[b]
            for l in 0..<levels {
                let kk = 4 << l
                for j in 0..<kk {
                    let u = (Double(j) + 0.5) / Double(kk)
                    f.nodeTab[(b * levels + l) * maxK + j] = lo[b] + u * width
                }
            }
            for j in 0..<kq {
                let u = (Double(j) + 0.5) / Double(kq)
                f.ptsT.append(lo[b] + u * width)
            }
        }
        return f
    }

    /// `production_bin_model`: model of the uniformly filled bins at cadence `cadenceS` (1.0 for
    /// bins the transit does not touch). `sigmaL = nil` is the reference default 2e-5.
    public static func productionBinModel(
        p: Double, k: Double, aR: Double, incRad: Double, ecc: Double, wRad: Double,
        u1: Double, u2: Double, cadenceS: Double, binLo: [Double], binHi: [Double],
        sigmaL: Double? = nil
    ) -> [Double] {
        let nb = binLo.count
        if nb == 0 { return [] }
        let fold = uniformBinsFold(lo: binLo, hi: binHi, sigmaL: sigmaL ?? 2.0e-5)
        let r = drawBinModels(
            w: cadenceS / 86400.0, delta: FPPV2Consts.ladderDeltaNat, fold: fold, period: p,
            aRs: aR, incRad: incRad, ecc: ecc, wRad: wRad, k: k, u1: u1, u2: u2, dilA: 1.0, dilB: 0.0)
        var m = [Double](repeating: 1.0, count: nb)
        for i in 0..<r.nAff { m[r.idx[i]] = r.m[i] }
        return m
    }
}
