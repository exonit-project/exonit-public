import Foundation

/// Ephemeris refinement before FPP (FPP_EPHEM_REFINE, section 8b).
///
/// Port of `worker/fpp_native.py` lines 4557-4900 and 5751-5784 (commit `03a4bcc`).
/// Pure function of its inputs, deterministic, no RNG, no clock, no global state.
public enum FPPV3Ephem {
    public static let ephemT14Factors: [Double] = [0.5, 0.65, 0.8, 1.0, 1.2, 1.45, 1.75, 2.1, 2.5]
    public static let ephemIngress: [Double] = [0.08, 0.2, 0.4]
    public static let ephemT0Half: Double = 1.5
    public static let ephemReachMaxP: Double = 0.25
    public static let ephemDriftHalf: Double = 1.5
    public static let ephemPRound: Double = 1e-4
    public static let ephemPFracMax: Double = 0.02
    public static let ephemNCoarse: Int = 41
    public static let ephemNZoom: Int = 31
    public static let ephemNZooms: Int = 3
    public static let ephemZoom: Double = 5.0
    public static let ephemEdgeCells: Int = 2
    public static let ephemMinDchi2: Double = 25.0
    public static let ephemPFreeDchi2: Double = 9.0
    public static let ephemMaxFreeShift: Double = 0.25
    public static let ephemCoverPoints: Int = 3
    public static let ephemMinPoints: Int = 100
    public static let ephemMinWindowPoints: Int = 5

    public enum T0Convention: String, Codable {
        case boxStart = "box_start"
        case centre = "centre"
    }

    /// Running integral from -inf to x of the unit trapezoid of half-width hw and ingress length L (0 < L <= hw).
    @inline(__always)
    public static func ephemCum(x: Double, hw: Double, l: Double) -> Double {
        let y = x > 0.0 ? -x : x
        let c: Double
        if y <= -hw {
            c = 0.0
        } else if y <= l - hw {
            let u = y + hw
            c = u * u / (2.0 * l)
        } else {
            c = 0.5 * l + (y + hw - l)
        }
        if x > 0.0 {
            return (2.0 * hw - l) - c
        }
        return c
    }

    /// Best trapezoid-dip dchi2 at one (P, tc) on the unfolded data; returns (dchi2, shape index, n in window).
    @inline(__always)
    public static func ephemDchi2(
        t: [Double],
        r: [Double],
        period: Double,
        tc: Double,
        t14: [Double],
        ing: [Double],
        w: Double,
        invS2: Double,
        minPts: Int,
        smr: inout [Double],
        smm: inout [Double]
    ) -> (best: Double, bi: Int, npts: Int) {
        let ns = t14.count
        let ng = ing.count
        smr.withUnsafeMutableBufferPointer { b in
            b.initialize(repeating: 0.0)
        }
        smm.withUnsafeMutableBufferPointer { b in
            b.initialize(repeating: 0.0)
        }

        let hwExp = w > 0.0 ? 0.5 * w : 0.0
        let half = 0.5 * t14[ns - 1] + hwExp
        var npts = 0

        for i in 0..<t.count {
            let x = t[i] - tc + 0.5 * period
            let dt = x - period * floor(x / period) - 0.5 * period
            let a = abs(dt)
            if a >= half {
                continue
            }
            npts += 1
            for k in 0..<ns {
                let hw = 0.5 * t14[k]
                if a >= hw + hwExp {
                    continue
                }
                for g in 0..<ng {
                    let m: Double
                    if w > 0.0 {
                        let l = ing[g] * t14[k]
                        m = (ephemCum(x: a + hwExp, hw: hw, l: l) - ephemCum(x: a - hwExp, hw: hw, l: l)) / w
                    } else {
                        var mVal = (hw - a) / (ing[g] * t14[k])
                        if mVal > 1.0 {
                            mVal = 1.0
                        }
                        m = mVal
                    }
                    let idx = k * ng + g
                    smr[idx] += m * r[i]
                    smm[idx] += m * m
                }
            }
        }

        var best = 0.0
        var bi = -1
        if npts >= minPts {
            for s in 0..<(ns * ng) {
                if smm[s] > 0.0 && smr[s] > 0.0 {
                    let d = smr[s] * smr[s] / smm[s] * invS2
                    if d > best {
                        best = d
                        bi = s
                    }
                }
            }
        }
        return (best, bi, npts)
    }

    /// Grid around (P0, tc0); returns (best i, best j, best dchi2, shape, dchi2 at centre, n in window).
    public static func ephemGrid(
        t: [Double],
        r: [Double],
        p0: Double,
        stepP: Double,
        nP: Int,
        tc0: Double,
        stepT: Double,
        nt: Int,
        t14: [Double],
        ing: [Double],
        w: Double,
        invS2: Double,
        minPts: Int
    ) -> (bi: Int, bj: Int, best: Double, bshape: Int, centre: Double, bn: Int) {
        var smr = [Double](repeating: 0.0, count: 27)
        var smm = [Double](repeating: 0.0, count: 27)
        let ci = (nP - 1) / 2
        let cj = (nt - 1) / 2
        var bi = ci
        var bj = cj
        var best = -1.0
        var bshape = -1
        var centre = 0.0
        var bn = 0

        for i in 0..<nP {
            let p = p0 + Double(i - ci) * stepP
            for j in 0..<nt {
                let tc = tc0 + Double(j - cj) * stepT
                let (d, sh, n) = ephemDchi2(
                    t: t, r: r, period: p, tc: tc, t14: t14, ing: ing, w: w,
                    invS2: invS2, minPts: minPts, smr: &smr, smm: &smm
                )
                if i == ci && j == cj {
                    centre = d
                }
                if d > best {
                    best = d
                    bi = i
                    bj = j
                    bshape = sh
                    bn = n
                }
            }
        }
        return (bi, bj, best, bshape, centre, bn)
    }

    /// Median of a float64 array: ascending stable sort, middle element or mean of the two middle ones.
    public static func ephemMedian(_ x: [Double]) -> Double {
        if x.isEmpty {
            return Double.nan
        }
        let s = x.sorted()
        let n = s.count
        let h = n / 2
        if n % 2 == 1 {
            return s[h]
        } else {
            return 0.5 * (s[h - 1] + s[h])
        }
    }

    public struct ZoomStep: Codable {
        public let zoomIndex: Int
        public let zi: Int
        public let zj: Int
        public let zshape: Int
    }

    public struct SearchResult {
        public let p: Double
        public let t: Double
        public let best: Double
        public let shape: Int
        public let centre: Double
        public let edgeP: Bool
        public let edgeT: Bool
        public let coarseBi: Int
        public let coarseBj: Int
        public let coarseShape: Int
        public let zooms: [ZoomStep]
    }

    /// Coarse grid + nested zooms around (P0, tref0); P varies only when pFree.
    public static func ephemSearch(
        ts: [Double],
        rs: [Double],
        p0: Double,
        dpHalf: Double,
        tref0: Double,
        dtHalf: Double,
        t14: [Double],
        ing: [Double],
        w: Double,
        invS2: Double,
        pFree: Bool
    ) -> SearchResult {
        let nP = pFree ? ephemNCoarse : 1
        let nt = ephemNCoarse
        let stepP = pFree ? 2.0 * dpHalf / Double(nP - 1) : 0.0
        let stepT = 2.0 * dtHalf / Double(nt - 1)
        let (bi, bj, rawBest, rawBshape, centre, _bn) = ephemGrid(
            t: ts, r: rs, p0: p0, stepP: stepP, nP: nP, tc0: tref0, stepT: stepT, nt: nt,
            t14: t14, ing: ing, w: w, invS2: invS2, minPts: ephemMinWindowPoints
        )
        let coarseBi = bi
        let coarseBj = bj
        let coarseShape = rawBshape
        var best = rawBest
        var bshape = rawBshape
        let ci = (nP - 1) / 2
        let cj = (nt - 1) / 2
        var cp = p0 + Double(bi - ci) * stepP
        var ct = tref0 + Double(bj - cj) * stepT
        let edgeP = pFree && (bi < ephemEdgeCells || bi > nP - 1 - ephemEdgeCells)
        let edgeT = bj < ephemEdgeCells || bj > nt - 1 - ephemEdgeCells
        var hp = dpHalf
        var ht = dtHalf
        var zooms = [ZoomStep]()
        zooms.reserveCapacity(ephemNZooms)

        for z in 0..<ephemNZooms {
            hp /= ephemZoom
            ht /= ephemZoom
            let nPz = pFree ? ephemNZoom : 1
            let spZ = pFree ? 2.0 * hp / Double(nPz - 1) : 0.0
            let stZ = 2.0 * ht / Double(ephemNZoom - 1)
            let (zi, zj, zb, zs, _zc, _zn) = ephemGrid(
                t: ts, r: rs, p0: cp, stepP: spZ, nP: nPz, tc0: ct, stepT: stZ, nt: ephemNZoom,
                t14: t14, ing: ing, w: w, invS2: invS2, minPts: ephemMinWindowPoints
            )
            zooms.append(ZoomStep(zoomIndex: z, zi: zi, zj: zj, zshape: zs))
            if zb > best {
                best = zb
                bshape = zs
                cp += Double(zi - (nPz - 1) / 2) * spZ
                ct += Double(zj - (ephemNZoom - 1) / 2) * stZ
            }
        }

        return SearchResult(
            p: cp,
            t: ct,
            best: best,
            shape: bshape,
            centre: centre,
            edgeP: edgeP,
            edgeT: edgeT,
            coarseBi: coarseBi,
            coarseBj: coarseBj,
            coarseShape: coarseShape,
            zooms: zooms
        )
    }

    public struct RefineEphemerisResult: Codable {
        public var status: String
        public var reason: String?
        public var p: Double
        public var t0: Double
        public var pIn: Double
        public var t0In: Double
        public var tRefIn: Double
        public var cadenceS: Double?
        public var nReq: Int?
        public var nOrbits: Double?
        public var nTransits: Int
        public var pSearched: Bool
        public var pFreeAccepted: Bool
        public var deltaChi2P: Double?
        public var dchi2PFixed: Double?
        public var dchi2PFree: Double?
        public var pFixed: Bool?
        public var shiftH: Double
        public var dp: Double
        public var maxOffsetH: Double
        public var dchi2Initial: Double?
        public var dchi2Refined: Double?
        public var deltaChi2: Double?
        public var sigma: Double?
        public var baseline: Double?
        public var t14Factor: Double?
        public var ingress: Double?
        public var edgeP: Bool
        public var edgeT0: Bool
        public var dpHalf: Double?
        public var dtHalf: Double?

        enum CodingKeys: String, CodingKey {
            case status, reason
            case p = "P"
            case t0
            case pIn = "P_in"
            case t0In = "t0_in"
            case tRefIn = "t_ref_in"
            case cadenceS = "cadence_s"
            case nReq = "n_req"
            case nOrbits = "n_orbits"
            case nTransits = "n_transits"
            case pSearched = "p_searched"
            case pFreeAccepted = "p_free_accepted"
            case deltaChi2P = "delta_chi2_p"
            case dchi2PFixed = "dchi2_p_fixed"
            case dchi2PFree = "dchi2_p_free"
            case pFixed = "p_fixed"
            case shiftH = "shift_h"
            case dp = "dP"
            case maxOffsetH = "max_offset_h"
            case dchi2Initial = "dchi2_initial"
            case dchi2Refined = "dchi2_refined"
            case deltaChi2 = "delta_chi2"
            case sigma, baseline
            case t14Factor = "t14_factor"
            case ingress
            case edgeP = "edge_P"
            case edgeT0 = "edge_t0"
            case dpHalf = "dP_half"
            case dtHalf = "dt_half"
        }
    }

    public struct SearchIndicesTrace {
        public let sourceSearch: String
        public let coarseBi: Int?
        public let coarseBj: Int?
        public let coarseShape: Int?
        public let zooms: [ZoomStep]
        public let bestShape: Int?
        public let bestT14FactorIndex: Int?
        public let bestIngressIndex: Int?
    }

    /// Bounded deterministic local (P, transit-centre) refinement on the unfolded light curve.
    public static func refineEphemeris(
        time: [Double],
        flux: [Double],
        period: Double,
        t0CentreGuess: Double,
        duration: Double,
        cadenceS: Double? = nil
    ) -> (result: RefineEphemerisResult, trace: SearchIndicesTrace) {
        let pIn = period
        let t0In = t0CentreGuess
        let d = duration

        var out = RefineEphemerisResult(
            status: "failed",
            reason: nil,
            p: pIn,
            t0: t0In,
            pIn: pIn,
            t0In: t0In,
            tRefIn: t0In,
            cadenceS: nil,
            nReq: nil,
            nOrbits: nil,
            nTransits: 0,
            pSearched: false,
            pFreeAccepted: false,
            deltaChi2P: nil,
            dchi2PFixed: nil,
            dchi2PFree: nil,
            pFixed: nil,
            shiftH: 0.0,
            dp: 0.0,
            maxOffsetH: 0.0,
            dchi2Initial: nil,
            dchi2Refined: nil,
            deltaChi2: nil,
            sigma: nil,
            baseline: nil,
            t14Factor: nil,
            ingress: nil,
            edgeP: false,
            edgeT0: false,
            dpHalf: nil,
            dtHalf: nil
        )

        let emptyTrace = SearchIndicesTrace(
            sourceSearch: "none",
            coarseBi: nil,
            coarseBj: nil,
            coarseShape: nil,
            zooms: [],
            bestShape: nil,
            bestT14FactorIndex: nil,
            bestIngressIndex: nil
        )

        if !(pIn.isFinite && pIn > 0.0 && t0In.isFinite && d.isFinite && d > 0.0) {
            out.reason = "invalid_ephemeris"
            return (out, emptyTrace)
        }

        var tValid = [Double]()
        var fValid = [Double]()
        tValid.reserveCapacity(time.count)
        fValid.reserveCapacity(flux.count)
        for i in 0..<min(time.count, flux.count) {
            if time[i].isFinite && flux[i].isFinite {
                tValid.append(time[i])
                fValid.append(flux[i])
            }
        }

        if tValid.count < ephemMinPoints {
            out.reason = "too_few_points"
            return (out, emptyTrace)
        }

        let cadS: Double
        if let explicitCad = cadenceS {
            cadS = explicitCad
        } else {
            let sortedT = tValid.sorted()
            var diffs = [Double]()
            diffs.reserveCapacity(sortedT.count - 1)
            for i in 1..<sortedT.count {
                diffs.append(sortedT[i] - sortedT[i - 1])
            }
            let medDiff = ephemMedian(diffs)
            cadS = (medDiff * 86400.0 * 1000.0).rounded() / 1000.0
        }
        out.cadenceS = cadS
        if !(cadS.isFinite && cadS > 0.0) {
            out.reason = "cadence_unusable"
            return (out, emptyTrace)
        }

        let w = cadS / 86400.0
        let hw = 0.5 * w
        var tmin = Double.infinity
        var tmax = -Double.infinity
        for ti in tValid {
            if ti < tmin { tmin = ti }
            if ti > tmax { tmax = ti }
        }
        out.nOrbits = (tmax - tmin) / pIn

        let nRef = floor((0.5 * (tmin + tmax) - t0In) / pIn + 0.5)
        let tref0 = t0In + nRef * pIn
        out.tRefIn = tref0

        var ep = [Double]()
        var dt0 = [Double]()
        ep.reserveCapacity(tValid.count)
        dt0.reserveCapacity(tValid.count)
        for ti in tValid {
            let epi = floor((ti - tref0) / pIn + 0.5)
            let x = ti - tref0 + 0.5 * pIn
            let dt0i = x - pIn * floor(x / pIn) - 0.5 * pIn
            ep.append(epi)
            dt0.append(dt0i)
        }

        let nReq = min(ephemCoverPoints, max(1, Int(floor(d / w))))
        out.nReq = nReq

        var epochCounts = [Int: Int]()
        for i in 0..<tValid.count {
            if abs(dt0[i]) < 0.5 * d {
                let epInt = Int(ep[i])
                epochCounts[epInt, default: 0] += 1
            }
        }
        var nTr = 0
        for count in epochCounts.values {
            if count >= nReq {
                nTr += 1
            }
        }
        out.nTransits = nTr
        if nTr < 1 {
            out.reason = "no_transit_covered"
            return (out, emptyTrace)
        }

        var nMax = 0.0
        for epi in ep {
            let a = abs(epi)
            if a > nMax { nMax = a }
        }
        let pSearch = nTr >= 2 && nMax >= 1.0
        out.pSearched = pSearch

        var t14 = [Double]()
        t14.reserveCapacity(ephemT14Factors.count)
        for factor in ephemT14Factors {
            t14.append(factor * d)
        }
        let ing = ephemIngress
        let halfShape = 0.5 * t14[t14.count - 1] + hw
        let dtHalf = min(ephemT0Half * d, (ephemReachMaxP * pIn - halfShape) / 1.25)
        if !(dtHalf > 0.0) {
            out.reason = "duration_too_long_for_window"
            return (out, emptyTrace)
        }

        let dpHalf: Double
        if pSearch {
            let stepBls = max(2e-5, 4e-6 * pIn)
            let dpDrift = max(ephemDriftHalf * d / nMax, stepBls + ephemPRound)
            dpHalf = min(dpDrift, ephemPFracMax * pIn)
        } else {
            dpHalf = 0.0
        }
        out.dpHalf = dpHalf
        out.dtHalf = dtHalf

        var subIdx = [Int]()
        var ootF = [Double]()
        for i in 0..<tValid.count {
            let dtAbs = abs(dt0[i])
            let epAbs = abs(ep[i])
            let reach = halfShape + 1.5 * (dtHalf + epAbs * dpHalf)
            if dtAbs < reach {
                subIdx.append(i)
            }
            let ootBound = halfShape + dtHalf + epAbs * dpHalf
            if !(dtAbs < ootBound) {
                ootF.append(fValid[i])
            }
        }

        let refF = ootF.count >= ephemMinPoints ? ootF : fValid
        let base = ephemMedian(refF)
        var dev = [Double]()
        dev.reserveCapacity(refF.count)
        for x in refF {
            dev.append(abs(x - base))
        }
        let sigma = 1.4826 * ephemMedian(dev)
        out.baseline = base
        out.sigma = sigma
        if !(sigma > 0.0 && sigma.isFinite) {
            out.reason = "sigma_unusable"
            return (out, emptyTrace)
        }

        var ts = [Double]()
        var rs = [Double]()
        ts.reserveCapacity(subIdx.count)
        rs.reserveCapacity(subIdx.count)
        for idx in subIdx {
            ts.append(tValid[idx])
            rs.append(base - fValid[idx])
        }
        let invS2 = 1.0 / (sigma * sigma)

        let fixed = ephemSearch(
            ts: ts, rs: rs, p0: pIn, dpHalf: 0.0, tref0: tref0, dtHalf: dtHalf,
            t14: t14, ing: ing, w: w, invS2: invS2, pFree: false
        )
        out.dchi2PFixed = fixed.best

        var bestS = fixed
        var freeOpt: SearchResult? = nil
        if pSearch {
            let free = ephemSearch(
                ts: ts, rs: rs, p0: pIn, dpHalf: dpHalf, tref0: tref0, dtHalf: dtHalf,
                t14: t14, ing: ing, w: w, invS2: invS2, pFree: true
            )
            let gain = free.best - fixed.best
            out.dchi2PFree = free.best
            out.deltaChi2P = gain
            if gain >= ephemPFreeDchi2 {
                bestS = free
                out.pFreeAccepted = true
            }
            freeOpt = free
        }

        let pFixed = !out.pFreeAccepted
        out.pFixed = pFixed
        let centre = fixed.centre
        let best = bestS.best
        out.dchi2Initial = centre
        out.dchi2Refined = best
        out.deltaChi2 = best - centre

        var t14FactorIdx: Int? = nil
        var ingressIdx: Int? = nil
        if bestS.shape >= 0 {
            let sIdx = bestS.shape
            let nIng = ephemIngress.count
            t14FactorIdx = sIdx / nIng
            ingressIdx = sIdx % nIng
            out.t14Factor = ephemT14Factors[sIdx / nIng]
            out.ingress = ephemIngress[sIdx % nIng]
        }
        out.edgeP = bestS.edgeP
        out.edgeT0 = bestS.edgeT

        let sourceSearch = out.pFreeAccepted ? "free" : "fixed"
        let activeSearch = out.pFreeAccepted ? freeOpt! : fixed

        let trace = SearchIndicesTrace(
            sourceSearch: sourceSearch,
            coarseBi: activeSearch.coarseBi,
            coarseBj: activeSearch.coarseBj,
            coarseShape: activeSearch.coarseShape,
            zooms: activeSearch.zooms,
            bestShape: bestS.shape >= 0 ? bestS.shape : nil,
            bestT14FactorIndex: t14FactorIdx,
            bestIngressIndex: ingressIdx
        )

        if best < ephemMinDchi2 {
            out.reason = "no_significant_dip"
            return (out, trace)
        }
        if bestS.edgeP || bestS.edgeT {
            out.status = "edge"
            out.reason = "optimum_at_search_edge"
            return (out, trace)
        }

        let cp = bestS.p
        let ct = bestS.t
        let dp = cp - pIn
        let shift = ct - tref0
        var epMin = Double.infinity
        var epMax = -Double.infinity
        for epi in ep {
            if epi < epMin { epMin = epi }
            if epi > epMax { epMax = epi }
        }
        let maxOff = max(abs(shift + epMin * dp), abs(shift + epMax * dp))
        if maxOff > ephemMaxFreeShift * d && (best - centre) < ephemPFreeDchi2 {
            out.maxOffsetH = maxOff * 24.0
            out.reason = "insignificant_large_shift"
            return (out, trace)
        }

        out.status = "ok"
        out.p = cp
        out.t0 = ct
        out.dp = dp
        out.shiftH = shift * 24.0
        out.maxOffsetH = maxOff * 24.0

        return (out, trace)
    }

    public struct RefineProvenanceFields: Codable {
        public let reason: String?
        public let pIn: Double
        public let t0In: Double
        public let tRefIn: Double
        public let cadenceS: Double?
        public let nReq: Int?
        public let nOrbits: Double?
        public let nTransits: Int
        public let pSearched: Bool
        public let pFreeAccepted: Bool
        public let deltaChi2P: Double?
        public let dchi2PFixed: Double?
        public let dchi2PFree: Double?
        public let pFixed: Bool?
        public let dp: Double
        public let maxOffsetH: Double
        public let dchi2Initial: Double?
        public let dchi2Refined: Double?
        public let deltaChi2: Double?
        public let sigma: Double?
        public let t14Factor: Double?
        public let ingress: Double?
        public let edgeP: Bool
        public let edgeT0: Bool
        public let dpHalf: Double?
        public let dtHalf: Double?

        enum CodingKeys: String, CodingKey {
            case reason
            case pIn = "P_in"
            case t0In = "t0_in"
            case tRefIn = "t_ref_in"
            case cadenceS = "cadence_s"
            case nReq = "n_req"
            case nOrbits = "n_orbits"
            case nTransits = "n_transits"
            case pSearched = "p_searched"
            case pFreeAccepted = "p_free_accepted"
            case deltaChi2P = "delta_chi2_p"
            case dchi2PFixed = "dchi2_p_fixed"
            case dchi2PFree = "dchi2_p_free"
            case pFixed = "p_fixed"
            case dp = "dP"
            case maxOffsetH = "max_offset_h"
            case dchi2Initial = "dchi2_initial"
            case dchi2Refined = "dchi2_refined"
            case deltaChi2 = "delta_chi2"
            case sigma
            case t14Factor = "t14_factor"
            case ingress
            case edgeP = "edge_P"
            case edgeT0 = "edge_t0"
            case dpHalf = "dP_half"
            case dtHalf = "dt_half"
        }
    }

    public struct EphemProvenance: Codable {
        public let ephemRefined: Bool
        public let ephemStatus: String
        public let ephemT0Convention: String
        public let ephemP: Double
        public let ephemT0: Double
        public let ephemShiftH: Double
        public let ephemRefine: RefineProvenanceFields?

        enum CodingKeys: String, CodingKey {
            case ephemRefined = "ephem_refined"
            case ephemStatus = "ephem_status"
            case ephemT0Convention = "ephem_t0_convention"
            case ephemP = "ephem_P"
            case ephemT0 = "ephem_t0"
            case ephemShiftH = "ephem_shift_h"
            case ephemRefine = "ephem_refine"
        }
    }

    public struct FppEphemerisResult {
        public let pUse: Double
        public let t0Use: Double
        public let provenance: EphemProvenance?
        public let refineFailed: Bool
    }

    /// The (P, t0 centre) the FPP fold uses, plus provenance fields (section 8b).
    public static func fppEphemeris(
        time: [Double],
        flux: [Double],
        period: Double,
        t0: Double,
        duration: Double,
        convention: T0Convention,
        refineEnabled: Bool,
        cadenceS: Double? = nil
    ) -> FppEphemerisResult {
        if !refineEnabled {
            return FppEphemerisResult(
                pUse: period,
                t0Use: t0,
                provenance: nil,
                refineFailed: false
            )
        }
        switch convention {
        case .centre:
            return FppEphemerisResult(
                pUse: period,
                t0Use: t0,
                provenance: EphemProvenance(
                    ephemRefined: false,
                    ephemStatus: "not_requested",
                    ephemT0Convention: "centre",
                    ephemP: period,
                    ephemT0: t0,
                    ephemShiftH: 0.0,
                    ephemRefine: nil
                ),
                refineFailed: false
            )
        case .boxStart:
            let guess = t0 + 0.5 * duration
            let (refOut, _trace) = refineEphemeris(
                time: time, flux: flux, period: period,
                t0CentreGuess: guess, duration: duration, cadenceS: cadenceS
            )
            let ok = refOut.status == "ok"
            let pUse = ok ? refOut.p : period
            let t0Use = ok ? refOut.t0 : guess
            let shiftH = ok ? refOut.shiftH : 0.0
            let prov = EphemProvenance(
                ephemRefined: ok,
                ephemStatus: refOut.status,
                ephemT0Convention: "box_start",
                ephemP: pUse,
                ephemT0: t0Use,
                ephemShiftH: shiftH,
                ephemRefine: RefineProvenanceFields(
                    reason: refOut.reason,
                    pIn: refOut.pIn,
                    t0In: refOut.t0In,
                    tRefIn: refOut.tRefIn,
                    cadenceS: refOut.cadenceS,
                    nReq: refOut.nReq,
                    nOrbits: refOut.nOrbits,
                    nTransits: refOut.nTransits,
                    pSearched: refOut.pSearched,
                    pFreeAccepted: refOut.pFreeAccepted,
                    deltaChi2P: refOut.deltaChi2P,
                    dchi2PFixed: refOut.dchi2PFixed,
                    dchi2PFree: refOut.dchi2PFree,
                    pFixed: refOut.pFixed,
                    dp: refOut.dp,
                    maxOffsetH: refOut.maxOffsetH,
                    dchi2Initial: refOut.dchi2Initial,
                    dchi2Refined: refOut.dchi2Refined,
                    deltaChi2: refOut.deltaChi2,
                    sigma: refOut.sigma,
                    t14Factor: refOut.t14Factor,
                    ingress: refOut.ingress,
                    edgeP: refOut.edgeP,
                    edgeT0: refOut.edgeT0,
                    dpHalf: refOut.dpHalf,
                    dtHalf: refOut.dtHalf
                )
            )
            return FppEphemerisResult(
                pUse: pUse,
                t0Use: t0Use,
                provenance: prov,
                refineFailed: !ok
            )
        }
    }
}
