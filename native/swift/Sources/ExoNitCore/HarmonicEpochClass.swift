import Foundation

// Epoch-class (parity) alias check for the harmonic resolver.
//
// Bit-exact port of `worker/harmonic_epoch_class.py` (branch worktree-alias-2p-fix of
// the ExoNit repo; spec: docs/reviews/2026-10-14-alias-2p-phase2-report.md section 2).
// Rollout: behind the worker config flag `harmonic_epoch_class` (default false).
// With the flag off none of this code is reached and the resolver / protected-slot
// behaviour is unchanged.
//
// Numeric contract (must stay identical to the Python module and the Rust port):
//  * everything is Double (float64); `time` is sorted ascending;
//  * every sum is SEQUENTIAL in ascending index order (`acc += x[i]`, one accumulator
//    per bin); no pairwise / blocked / compensated summation anywhere;
//  * medians are sort-based (middle element, or `(a + b) * 0.5` of the two middle ones);
//  * `rint` is `.rounded(.toNearestOrEven)`; `sqrt` is IEEE-exact `.squareRoot()`;
//  * NO FMA contraction. Swift emits plain `fmul` / `fadd` (no `contract` flag, and it
//    never emits `llvm.fmuladd`), so LLVM's default FP-op-fusion mode does not fuse
//    them. This file additionally routes every multiplication whose result is added to
//    or subtracted from something through `noFMAMul(_:_:)`, an `@inline(never)` function:
//    a call boundary makes fusion impossible regardless of compiler flags / LTO.
//    `.addingProduct`, `fma`, `Float80` and Accelerate (vDSP) are deliberately not used.
//  * evaluation orders follow numpy left-to-right exactly (see inline comments).

/// A planet ephemeris used by the slot-duplicate rule (an accepted candidate or the slot
/// candidate). `t0` is whatever the BLS produced (box START); see `t0IsCentre`.
public struct EpochEphemeris: Equatable, Sendable {
    public var period: Double
    public var t0: Double
    public var duration: Double
    public init(period: Double, t0: Double, duration: Double) {
        self.period = period; self.t0 = t0; self.duration = duration
    }
}

/// Per-epoch depths at the refined ephemeris (output of `prepareEpochs`).
public struct EpochPrep: Equatable, Sendable {
    public let ep0: Int
    public let nE: [Double]
    public let dE: [Double]
    public let covered: [Bool]
    public let sigma: Double
    public let period: Double
    public let centre: Double
}

/// Per-class statistics (output of `classStats`). `depth` / `err` are NaN for classes
/// without covered epochs.
public struct EpochClassStats: Equatable, Sendable {
    public let depth: [Double]
    public let err: [Double]
    public let nep: [Int]
    public let ndip: [Int]
}

public enum EpochParity: String, Sendable {
    case alias, genuine, undecided
}

/// FMA barrier: a call boundary between a multiply and the following add/sub.
@inline(never)
func noFMAMul(_ a: Double, _ b: Double) -> Double { a * b }

public enum HarmonicEpochClass {
    // --- thresholds (identical to the Python module) --------------------------------
    public static let kMin = 2
    public static let kMax = 5
    public static let windowHalfFraction = 0.5
    public static let minPoints = 3.0
    public static let minCoverFraction = 0.5
    public static let minEpochsPerClass = 2
    public static let dipSigma = 2.0
    public static let minDipEpochs = 2
    public static let promoteCarrySigma = 5.0
    public static let promoteEmptySigma = 2.0
    public static let promoteMaxPasses = 2
    public static let promoteRefitFraction = 0.05
    public static let promoteRefitPeriodCount = 500
    public static let promoteMinTransits = 2.0
    public static let vetoClassSigma = 3.0
    public static let dupPeriodTolerance = 0.01
    public static let dupEpochDurationFraction = 1.5
    public static let dupEpochMinDays = 0.05
    public static let driftRelativePeriod = 1e-4
    public static let driftMaxDurationMultiple = 2.0
    public static let driftStepFraction = 0.25
    public static let driftT0Steps = 2
    public static let driftMinPoints = 3
    public static let boxCentreFraction = 0.5
    public static let vhGridRelativePeriod = 2.0e-4
    public static let sigmaFloor = 1e-12

    // --- primitives -------------------------------------------------------------------

    /// Normalise a candidate t0 to the transit CENTRE (BLS returns the box START).
    public static func transitCentre(_ t0: Double, _ duration: Double, _ t0IsCentre: Bool) -> Double {
        if t0IsCentre { return t0 }
        return t0 + noFMAMul(boxCentreFraction, duration)
    }

    @inline(__always)
    static func rintEven(_ x: Double) -> Double { x.rounded(.toNearestOrEven) }

    /// Sequential ascending-index sum (`acc += x[i]`).
    static func seqSum(_ x: [Double]) -> Double {
        var acc = 0.0
        for v in x { acc += v }
        return acc
    }

    /// Sort-based median: s[n/2] (odd) or (s[n/2-1] + s[n/2]) * 0.5 (even). n >= 1.
    static func median(_ x: [Double]) -> Double {
        let s = x.sorted()
        let n = s.count
        let h = n / 2
        if n % 2 == 1 { return s[h] }
        return (s[h - 1] + s[h]) * 0.5
    }

    // --- prepare_epochs ---------------------------------------------------------------

    public static func prepareEpochs(
        time: [Double], flux: [Double], period: Double, t0: Double, duration: Double,
        excluded: [Bool]? = nil, t0IsCentre: Bool = false
    ) -> EpochPrep? {
        guard time.count == flux.count else { return nil }
        var t: [Double] = []
        var y: [Double] = []
        t.reserveCapacity(time.count); y.reserveCapacity(time.count)
        for i in 0..<time.count {
            guard time[i].isFinite, flux[i].isFinite else { continue }
            if let ex = excluded, i < ex.count, ex[i] { continue }
            t.append(time[i]); y.append(flux[i])
        }
        guard t.count >= 10, period > 0, duration > 0 else { return nil }
        let ref = median(y)
        let rawSigma = 1.4826 * median(y.map { abs($0 - ref) })
        let sigma = sigmaFloor > rawSigma ? sigmaFloor : rawSigma          // max(a, floor)
        var diffs = [Double](repeating: 0, count: t.count - 1)
        for i in 0..<(t.count - 1) { diffs[i] = t[i + 1] - t[i] }
        let dtMed = median(diffs)
        let half = windowHalfFraction * duration
        let expected = dtMed > 0 ? (2.0 * half) / dtMed : 1.0
        let coverNeed = minCoverFraction * expected
        let need = coverNeed > minPoints ? coverNeed : minPoints            // max(3, ...)

        let centre0 = transitCentre(t0, duration, t0IsCentre)
        let tmid = 0.5 * (t[0] + t[t.count - 1])
        let tRef = centre0 + noFMAMul(rintEven((tmid - centre0) / period), period)
        var ep = [Double](repeating: 0, count: t.count)
        for i in 0..<t.count { ep[i] = rintEven((t[i] - tRef) / period) }

        // --- drift refinement ---------------------------------------------------
        let span = t[t.count - 1] - t[0]
        let wA = driftRelativePeriod * span
        let wB = driftMaxDurationMultiple * duration
        let W = wB < wA ? wB : wA                                            // min(a, b)
        let step = driftStepFraction * duration
        var J = 0
        if W > 0 {
            let q = (W / step).rounded(.up)
            guard q.isFinite, q < 1e6 else { return nil }
            J = Int(q)
        }
        let edge = (0.5 * span) / period
        let nEdge = edge > 1.0 ? edge : 1.0                                  // max(a, 1)
        let bMax = Double(driftT0Steps) * step
        let nearLimit = (half + W) + bMax
        var ts: [Double] = [], ys: [Double] = [], es: [Double] = []
        for i in 0..<t.count {
            let centre = tRef + noFMAMul(ep[i], period)
            if abs(t[i] - centre) < nearLimit {
                ts.append(t[i]); ys.append(y[i]); es.append(ep[i])
            }
        }

        func score(_ a: Double, _ b: Double) -> Double? {
            var cnt = 0
            var acc = 0.0
            for i in 0..<ts.count {
                // numpy: t_ref + es*period + b + a*es  ==  ((t_ref + es*period) + b) + a*es
                var c = tRef + noFMAMul(es[i], period)
                c = c + b
                c = c + noFMAMul(a, es[i])
                if abs(ts[i] - c) < half { cnt += 1; acc += ys[i] }
            }
            if cnt < driftMinPoints { return nil }
            return ref - acc / Double(cnt)
        }

        var bestA = 0.0, bestB = 0.0
        var bestS = score(0.0, 0.0)
        if J > 0, !ts.isEmpty {
            for j in -J...J {
                let a = (Double(j) * step) / nEdge
                for i in -driftT0Steps...driftT0Steps {
                    if j == 0 && i == 0 { continue }
                    let b = Double(i) * step
                    if let s = score(a, b), bestS == nil || s > bestS! {
                        bestS = s; bestA = a; bestB = b
                    }
                }
            }
        }
        let p2 = period + bestA
        let c2 = tRef + bestB

        // --- per-epoch depths at the refined ephemeris ---------------------------
        var inIdx: [Int] = [], inEp: [Int] = []
        for i in 0..<t.count {
            let r = rintEven((t[i] - c2) / p2)
            guard r.isFinite, abs(r) < 9e15 else { return nil }
            let e = Int(r)
            let centre = c2 + noFMAMul(Double(e), p2)
            if abs(t[i] - centre) < half { inIdx.append(i); inEp.append(e) }
        }
        guard let ep0 = inEp.min(), let epMax = inEp.max() else { return nil }
        let m = epMax - ep0 + 1
        var nE = [Double](repeating: 0, count: m)
        var sE = [Double](repeating: 0, count: m)
        for q in 0..<inIdx.count {
            let idx = inEp[q] - ep0
            nE[idx] += 1.0
            sE[idx] += (y[inIdx[q]] - ref)
        }
        var covered = [Bool](repeating: false, count: m)
        var dE = [Double](repeating: 0, count: m)
        for e in 0..<m where nE[e] >= need {
            covered[e] = true
            dE[e] = -sE[e] / nE[e]
        }
        return EpochPrep(ep0: ep0, nE: nE, dE: dE, covered: covered,
                         sigma: sigma, period: p2, centre: c2)
    }

    // --- class_stats ------------------------------------------------------------------

    public static func classStats(_ prep: EpochPrep?, k: Int) -> EpochClassStats {
        let nan = [Double](repeating: Double.nan, count: k)
        guard let prep else {
            return EpochClassStats(depth: nan, err: nan,
                                   nep: [Int](repeating: 0, count: k),
                                   ndip: [Int](repeating: 0, count: k))
        }
        let m = prep.nE.count
        var nep = [Int](repeating: 0, count: k)
        var ndip = [Int](repeating: 0, count: k)
        var sumD = [Double](repeating: 0, count: k)
        var sumN = [Double](repeating: 0, count: k)
        var cls = [Int](repeating: 0, count: m)
        for e in 0..<m {
            let c = (((prep.ep0 + e) % k) + k) % k
            cls[e] = c
            guard prep.covered[e] else { continue }
            nep[c] += 1
            sumD[c] += prep.dE[e]
            sumN[c] += prep.nE[e]
            // d_e >= 2 * sigma / sqrt(n_e)  ==  (2*sigma) / sqrt(n_e)
            if prep.dE[e] >= (dipSigma * prep.sigma) / prep.nE[e].squareRoot() { ndip[c] += 1 }
        }
        var depth = nan
        var errWhite = nan
        for c in 0..<k where nep[c] > 0 {
            depth[c] = sumD[c] / Double(nep[c])
            errWhite[c] = prep.sigma / sumN[c].squareRoot()
        }
        var sumDev2 = [Double](repeating: 0, count: k)
        for e in 0..<m where prep.covered[e] {
            let dev = prep.dE[e] - depth[cls[e]]
            sumDev2[cls[e]] += noFMAMul(dev, dev)
        }
        var err = nan
        for c in 0..<k where nep[c] > 0 {
            err[c] = errWhite[c]
            if nep[c] >= 2 {
                // sqrt( (sumdev2 / (nep-1)) / nep )
                let sc = ((sumDev2[c] / Double(nep[c] - 1)) / Double(nep[c])).squareRoot()
                err[c] = errWhite[c] >= sc ? errWhite[c] : sc               // np.maximum
            }
        }
        return EpochClassStats(depth: depth, err: err, nep: nep, ndip: ndip)
    }

    public static func epochClassStats(
        time: [Double], flux: [Double], period: Double, t0: Double, duration: Double,
        k: Int, t0IsCentre: Bool = false, excluded: [Bool]? = nil
    ) -> EpochClassStats {
        classStats(prepareEpochs(time: time, flux: flux, period: period, t0: t0,
                                 duration: duration, excluded: excluded,
                                 t0IsCentre: t0IsCentre), k: k)
    }

    // --- patterns ---------------------------------------------------------------------

    private static func carryFlags(_ s: EpochClassStats, sigma thr: Double) -> (carry: [Bool], z: [Double]) {
        var carry = [Bool](), z = [Double]()
        for c in 0..<s.depth.count {
            let e = s.err[c] >= sigmaFloor ? s.err[c] : sigmaFloor          // np.maximum(err, floor)
            let zc = s.depth[c] / e
            z.append(zc)
            carry.append(zc >= thr && s.ndip[c] >= minDipEpochs)
        }
        return (carry, z)
    }

    static func promotionPattern(_ s: EpochClassStats) -> Bool {
        let (cr, z) = carryFlags(s, sigma: promoteCarrySigma)
        if cr.filter({ $0 }).count != 1 { return false }
        for c in 0..<cr.count where !cr[c] {
            if !(abs(z[c]) < promoteEmptySigma) { return false }
        }
        return true
    }

    static func allCarry(_ s: EpochClassStats) -> Bool {
        carryFlags(s, sigma: vetoClassSigma).carry.allSatisfy { $0 }
    }

    private static func decidable(_ s: EpochClassStats) -> Bool {
        s.nep.allSatisfy { $0 >= minEpochsPerClass }
    }

    // --- promotion --------------------------------------------------------------------

    /// All k in 2...5 (k*period <= periodMax) whose classes show the promotion pattern,
    /// LARGEST k first.
    public static func promotionKs(
        time: [Double], flux: [Double], period: Double, t0: Double, duration: Double,
        periodMax: Double, t0IsCentre: Bool = false, excluded: [Bool]? = nil
    ) -> [Int] {
        let prep = prepareEpochs(time: time, flux: flux, period: period, t0: t0,
                                 duration: duration, excluded: excluded, t0IsCentre: t0IsCentre)
        var out: [Int] = []
        for k in kMin...kMax {
            if period * Double(k) > periodMax { break }
            let s = classStats(prep, k: k)
            if !decidable(s) { continue }
            if promotionPattern(s) { out.append(k) }
        }
        return out.reversed()
    }

    /// Epoch-class promotion of a BLS `result` (up to 2 passes). Returns nil when nothing
    /// was promoted (Python: returns `result` itself). `result.t0` is the raw box start.
    public static func promote(
        time: [Double], flux: [Double], result: BLSResult, periodMax: Double,
        minimumSNR: Double = 5, excluded: [Bool]? = nil, search: BLSSearchFunction
    ) -> BLSResult? {
        var span = 0.0
        if let first = time.first, let last = time.last { span = last - first }
        var cur = result
        var didPromote = false
        for _ in 0..<promoteMaxPasses {
            let P = cur.period
            let dur = cur.duration
            if !(P > 0 && dur > 0) { break }
            var promoted: BLSResult?
            let ks = promotionKs(time: time, flux: flux, period: P, t0: cur.t0, duration: dur,
                                 periodMax: periodMax, excluded: excluded)
            for k in ks {
                let lo = (P * Double(k)) * (1.0 - promoteRefitFraction)
                let hiRaw = (P * Double(k)) * (1.0 + promoteRefitFraction)
                let hi = periodMax < hiRaw ? periodMax : hiRaw              // min(hiRaw, periodMax)
                if !(hi > lo) { continue }
                guard let sub = try? search(time, flux, BLSParameters(
                    periodMinimum: lo, periodMaximum: hi, periodCount: promoteRefitPeriodCount
                )) else { continue }
                if sub.snr < minimumSNR { continue }
                if span > 0, sub.period > 0, span / sub.period < promoteMinTransits { continue }
                // G1 consistency guard: all k classes carry at P_k/k -> the short period is
                // genuine; reject this k (try the next smaller one).
                if shortPeriodGenuine(time: time, flux: flux, sub: sub, k: k,
                                      excluded: excluded) { continue }
                promoted = sub
                break
            }
            guard let next = promoted else { break }
            cur = next
            didPromote = true
        }
        return didPromote ? cur : nil
    }

    // --- G1 promotion guard -----------------------------------------------------------

    /// `sub` = accepted re-fit at ~k*P (raw BLS result). True when every one of the k
    /// classes carries (veto rule) at the ephemeris (sub.period / k, sub.t0, sub.duration):
    /// the short period is genuine and the promotion must be rejected. Undecidable -> false.
    public static func shortPeriodGenuine(
        time: [Double], flux: [Double], sub: BLSResult, k: Int, excluded: [Bool]? = nil
    ) -> Bool {
        let p = sub.period
        if !(p > 0) { return false }
        let s = epochClassStats(time: time, flux: flux, period: p / Double(k), t0: sub.t0,
                                duration: sub.duration, k: k, t0IsCentre: false,
                                excluded: excluded)
        if !decidable(s) { return false }
        return allCarry(s)
    }

    // --- demotion veto ----------------------------------------------------------------

    /// The only mode, "vh". Undecidable -> the demotion proceeds. The /k probe grid has a
    /// constant relative step of 2e-4; when it is coarser than the BLS peak width
    /// (span * 2e-4 > duration, i.e. span > 5000 * duration, strict) the probe ephemeris can
    /// be smeared and only POSITIVE alias evidence vetoes ("v1": the promotion pattern);
    /// otherwise veto unless every class carries ("v0"). span = last - first of the raw array.
    /// False -> veto, true -> the demotion may proceed.
    public static func demotionSupported(
        time: [Double], flux: [Double], sub: BLSResult, k: Int, excluded: [Bool]? = nil
    ) -> Bool {
        let s = epochClassStats(time: time, flux: flux, period: sub.period, t0: sub.t0,
                                duration: sub.duration, k: k, t0IsCentre: false,
                                excluded: excluded)
        if !decidable(s) { return true }
        var span = 0.0
        if let first = time.first, let last = time.last { span = last - first }
        if span * vhGridRelativePeriod > sub.duration {
            return !promotionPattern(s)
        }
        return allCarry(s)
    }

    public static func harmonicParity(
        time: [Double], flux: [Double], period: Double, t0: Double, duration: Double,
        k: Int, excluded: [Bool]? = nil, t0IsCentre: Bool = false
    ) -> EpochParity {
        let s = epochClassStats(time: time, flux: flux, period: period, t0: t0,
                                duration: duration, k: k, t0IsCentre: t0IsCentre,
                                excluded: excluded)
        if !decidable(s) { return .undecided }
        if promotionPattern(s) { return .alias }
        if allCarry(s) { return .genuine }
        return .undecided
    }

    // --- slot duplicate ---------------------------------------------------------------

    /// Index in `accepted` of the first candidate that (period, t0, duration) repeats as
    /// P/k or k*P (k = 2...5, ratio within 1 %) with coincident epochs at the start, middle
    /// and end of [tFirst, tLast]. nil when there is none.
    public static func findHarmonicEpochMatch(
        period: Double, t0: Double, duration: Double, accepted: [EpochEphemeris],
        tFirst: Double, tLast: Double, t0IsCentre: Bool = false
    ) -> Int? {
        guard period > 0 else { return nil }
        let tc = transitCentre(t0, duration, t0IsCentre)
        let xs = [tFirst, 0.5 * (tFirst + tLast), tLast]
        for (index, c) in accepted.enumerated() {
            let pa = c.period
            guard pa > 0 else { continue }
            let da = c.duration
            let ta = transitCentre(c.t0, da, t0IsCentre)
            let ps: Double, ts: Double, pl: Double, tl: Double
            if period < pa { ps = period; ts = tc; pl = pa; tl = ta }
            else { ps = pa; ts = ta; pl = period; tl = tc }
            let r = pl / ps
            let k = rintEven(r)
            if k < Double(kMin) || k > Double(kMax) { continue }
            if abs(r - k) / k >= dupPeriodTolerance { continue }
            let dmax = da > duration ? da : duration                         // max(dur, da)
            let tolA = dupEpochDurationFraction * dmax
            let tol = dupEpochMinDays > tolA ? dupEpochMinDays : tolA        // max(a, 0.05)
            var ok = true
            for x in xs {
                let T = tl + noFMAMul(rintEven((x - tl) / pl), pl)
                let Ts = ts + noFMAMul(rintEven((T - ts) / ps), ps)
                if abs(T - Ts) > tol { ok = false; break }
            }
            if ok { return index }
        }
        return nil
    }

    /// Should a protected-slot candidate be DROPPED as an epoch-coincident P/k or k*P of an
    /// accepted candidate? Direction by parity (see Python `slot_duplicate`). `flux` must be
    /// the PRE-LOOP detrended flux with `excluded` nil (the accepted candidates' own
    /// transits must be present).
    public static func slotDuplicate(
        time: [Double], flux: [Double], slotPeriod: Double, slotT0: Double,
        slotDuration: Double, accepted: [EpochEphemeris],
        excluded: [Bool]? = nil, t0IsCentre: Bool = false
    ) -> Bool {
        var first: Double?, last: Double?, finiteCount = 0
        for v in time where v.isFinite {
            if first == nil { first = v }
            last = v
            finiteCount += 1
        }
        guard finiteCount >= 2, slotPeriod > 0, let tFirst = first, let tLast = last else {
            return false
        }
        guard let index = findHarmonicEpochMatch(
            period: slotPeriod, t0: slotT0, duration: slotDuration, accepted: accepted,
            tFirst: tFirst, tLast: tLast, t0IsCentre: t0IsCentre
        ) else { return false }
        let c = accepted[index]
        if slotPeriod < c.period {
            let k = Int(rintEven(c.period / slotPeriod))
            let v = harmonicParity(time: time, flux: flux, period: slotPeriod, t0: slotT0,
                                   duration: slotDuration, k: k, excluded: excluded,
                                   t0IsCentre: t0IsCentre)
            return v != .genuine
        }
        let k = Int(rintEven(slotPeriod / c.period))
        let v = harmonicParity(time: time, flux: flux, period: c.period, t0: c.t0,
                               duration: c.duration, k: k, excluded: excluded,
                               t0IsCentre: t0IsCentre)
        return v != .alias
    }
}
