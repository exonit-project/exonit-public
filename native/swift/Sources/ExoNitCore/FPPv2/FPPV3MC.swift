import Foundation

// Step 7, decision C: the MC-noise policy of worker/fpp_native.py section 9b (fpp-integration-r2 03a4bcc,
// :4955-5177), ported literally; mirror of the Rust `exonit_core::fpp::v2::mc`.
//
// Operation order is the Python order: numpy array `** 2` is `x * x`, every `np.sum` is the numpy pairwise sum
// (also over the masked subsets), `MC_TAIL_SIGMAS * MC_EV_SD` is formed before it multiplies `beta`, and
// `np.clip` follows the subtraction. Nothing here reads a clock, the environment or the host
// (`v3_mc_decision_code_references_no_clock` checks the source).

public enum FPPV3MC {
    // MARK: constants (verbatim, :4967-4991)
    public static let sigma = 3.0
    public static let gatesFPP: [Double] = [FPPPolicy.validatedFPP, FPPPolicy.candidateFPP]
    public static let gateNFPP = FPPPolicy.validatedNFPP
    public static let stageFactors = [1, 10, 100]
    public static let nCap = 1_000_000
    public static let minRefineN = 100_000
    public static let essMin = 10.0
    public static let heavyPMin = 1e-4
    public static let tailKTop = 10
    public static let tailSigmas = 3.0
    public static let absTol = 1e-9
    public static let futileESSDom = 25.0
    /// `MC_EV_SD = math.pi / math.sqrt(6.0)` (sqrt and `/` are correctly rounded).
    public static let evSD = FPPV2Consts.pi / (6.0).squareRoot()

    /// `f"{g:g}"` of the three registered gates.
    static func gateText(_ g: Double) -> String {
        if g.bitPattern == FPPPolicy.validatedFPP.bitPattern { return "0.015" }
        if g.bitPattern == FPPPolicy.candidateFPP.bitPattern { return "0.5" }
        if g.bitPattern == FPPPolicy.validatedNFPP.bitPattern { return "0.001" }
        preconditionFailure("only the three registered gates are formatted")
    }

    // MARK: draw-weight diagnostics sink

    /// One `_DRAW_DIAG` record.
    public struct DrawDiag: Equatable, Sendable {
        public var ess: Double, n: Int, maxShare: Double, tailScale: Double
        public init(ess: Double, n: Int, maxShare: Double, tailScale: Double) {
            self.ess = ess; self.n = n; self.maxShare = maxShare; self.tailScale = tailScale
        }
    }

    final class SinkBox: @unchecked Sendable { var records: [DrawDiag] = [] }
    static let sinkKey = "exonit.fpp.draw.sink"

    /// Per-thread sink of one Monte-Carlo pass (`_DRAW_DIAG = sink` ... `finally: _DRAW_DIAG = None`): while it is
    /// open, every `FPPV2SFamily.lnZFromLogw` on THIS thread appends one record. Close it with `close()`.
    public final class DiagSink {
        private let box = SinkBox()
        public init() { Thread.current.threadDictionary[FPPV3MC.sinkKey] = box }
        public var count: Int { box.records.count }
        public func since(_ n0: Int) -> [DrawDiag] { Array(box.records[n0...]) }
        public func close() {
            if (Thread.current.threadDictionary[FPPV3MC.sinkKey] as? SinkBox) === box {
                Thread.current.threadDictionary.removeObject(forKey: FPPV3MC.sinkKey)
            }
        }
        deinit { close() }
    }

    static func activeSink() -> SinkBox? { Thread.current.threadDictionary[sinkKey] as? SinkBox }

    /// `_tail_scale(w)`: (max - kk-th largest of the finite log-weights) / H_(kk-1), kk = min(10, #finite); NaN below 3.
    public static func tailScale(_ w: [Double]) -> Double {
        let fin = w.filter(\.isFinite).sorted(by: >)
        let kk = min(tailKTop, fin.count)
        guard kk >= 3 else { return .nan }
        return (fin[0] - fin[kk - 1]) / cpythonFloatSum((1..<kk).map { 1.0 / Double($0) })
    }

    /// CPython >= 3.12 built-in `sum()` of floats (the reference venv is 3.12.13): Neumaier-compensated, the
    /// compensation added at the end when it is non-zero and finite (`builtin_sum_impl`).
    public static func cpythonFloatSum(_ items: [Double]) -> Double {
        var f = 0.0, c = 0.0
        for x in items {
            let t = f + x
            if abs(f) >= abs(x) { c += (f - t) + x } else { c += (x - t) + f }
            f = t
        }
        if c != 0.0 && c.isFinite { f += c }
        return f
    }

    // MARK: seeds and ladder

    public static func stageSeedKey(ticID: String, period: Double, t0: Double, stage: Int, seed0: UInt64? = nil) -> String {
        precondition(stage >= 1, "stage_seed is for refinement stages >= 1; stage 0 uses tic_seed()")
        let s0 = seed0 ?? UInt64(FPPRNG.ticSeed(ticID: ticID, period: period, t0: t0))
        return "\(FPPRNG.ticSeedKey(ticID: ticID, period: period, t0: t0)):stage\(stage):\(s0)"
    }

    /// `stage_seed(tic_id, period, t0, stage, seed0)`: FNV-1a 64 of the key, masked to 32 bits.
    public static func stageSeed(ticID: String, period: Double, t0: Double, stage: Int, seed0: UInt64? = nil) -> UInt32 {
        var h: UInt64 = 0xcbf29ce484222325
        for byte in stageSeedKey(ticID: ticID, period: period, t0: t0, stage: stage, seed0: seed0).utf8 {
            h ^= UInt64(byte); h = h &* 0x00000100000001b3
        }
        return UInt32(truncatingIfNeeded: h)
    }

    /// `_mc_stage_ladder(N, refine, n_cap)`.
    public static func stageLadder(_ n: Int, refine: Bool, nCap: Int? = nil) -> [Int] {
        let cap = nCap ?? self.nCap
        if !refine || n < minRefineN { return [n] }
        var ladder = [n]
        for f in stageFactors.dropFirst() where n * f <= cap { ladder.append(n * f) }
        return ladder
    }

    // MARK: summary

    public struct Summary: Equatable, Sendable {
        public var names: [String], probs: [Double], fpp: Double, nfpp: Double, fppSE: Double, nfppSE: Double
        public var heavy: [String], minESS: Double, minESSScenario: String, n: Int
        public var ess: [Double], relVar: [Double], tailBoxFPP: (Double, Double), tailBoxNFPP: (Double, Double)
        public var nLowESS: Int, tailScale: [Double]
        public static func == (a: Summary, b: Summary) -> Bool {
            guard a.names == b.names, a.heavy == b.heavy, a.minESSScenario == b.minESSScenario else { return false }
            guard a.n == b.n, a.nLowESS == b.nLowESS else { return false }
            let sa: [Double] = [a.fpp, a.nfpp, a.fppSE, a.nfppSE, a.minESS, a.tailBoxFPP.0, a.tailBoxFPP.1,
                                a.tailBoxNFPP.0, a.tailBoxNFPP.1]
            let sb: [Double] = [b.fpp, b.nfpp, b.fppSE, b.nfppSE, b.minESS, b.tailBoxFPP.0, b.tailBoxFPP.1,
                                b.tailBoxNFPP.0, b.tailBoxNFPP.1]
            let va: [Double] = sa + a.probs + a.ess + a.relVar + a.tailScale
            let vb: [Double] = sb + b.probs + b.ess + b.relVar + b.tailScale
            return va.map(\.bitPattern) == vb.map(\.bitPattern)
        }
    }

    static func sumMasked(_ values: [Double], _ mask: [Bool]) -> Double {
        var sub = [Double](); sub.reserveCapacity(values.count)
        for i in values.indices where mask[i] { sub.append(values[i]) }
        return FPPNumpy.pairwiseSum(sub)
    }

    /// `np.maximum(a, b)`: `a` if `a >= b` or `a` is NaN, else `b`.
    static func npMaximum(_ a: Double, _ b: Double) -> Double { (a >= b || a.isNaN) ? a : b }

    /// `_delta_se(probs, in_set, rel_var)`.
    public static func deltaSE(_ probs: [Double], _ inSet: [Bool], _ relVar: [Double]) -> Double {
        let f = sumMasked(probs, inSet)
        let terms = probs.indices.map { i -> Double in
            let d = (inSet[i] ? 1.0 : 0.0) - f
            return ((d * d) * (probs[i] * probs[i])) * relVar[i]
        }
        return FPPNumpy.pairwiseSum(terms).squareRoot()
    }

    /// `_mc_heavy_scenarios(names, probs, ess)`.
    public static func heavyScenarios(_ names: [String], _ probs: [Double], _ ess: [Double]) -> [String] {
        names.indices.filter { !(ess[$0] >= essMin) && probs[$0] >= heavyPMin }.map { names[$0] }
    }

    static func fractions(_ lnz: [Double], _ falseSet: [Bool], _ nearbySet: [Bool]) -> (Double, Double) {
        var m = -Double.infinity
        for x in lnz where x.isFinite { m = Swift.max(m, x) }
        let q0 = lnz.map { FPPV2.exp($0 - m) }
        let total = FPPNumpy.pairwiseSum(q0)
        let q = q0.map { $0 / total }
        return (sumMasked(q, falseSet), sumMasked(q, nearbySet))
    }

    /// `_mc_summarise(lnZ, nearby_scen, diag, N)`; nil when no lnZ is finite or the probability total is not positive.
    public static func summarise(lnZ: [(String, Double)], nearby: [(String, Double)],
                                 diag: [String: DrawDiag], n: Int) -> Summary? {
        var names = [String](), values = [Double]()
        for (name, v) in lnZ + nearby {
            if let i = names.firstIndex(of: name) { values[i] = v } else { names.append(name); values.append(v) }
        }
        let nTarget = lnZ.count
        let allLnZ = values.map { $0.isFinite ? $0 : -Double.infinity }
        guard allLnZ.contains(where: { $0 > -Double.infinity }) else { return nil }
        var maxLnZ = -Double.infinity
        for x in allLnZ where x > -Double.infinity { maxLnZ = Swift.max(maxLnZ, x) }
        var probs = allLnZ.map { FPPV2.exp($0 - maxLnZ) }
        let total = FPPNumpy.pairwiseSum(probs)
        guard total > 0 else { return nil }
        for i in probs.indices { probs[i] /= total }
        func targetProb(_ key: String) -> Double {
            guard let i = names[..<nTarget].firstIndex(of: key) else { return 0.0 }
            return probs[i]
        }
        let fppRaw = 1.0 - targetProb("TP") - targetProb("PTP") - targetProb("DTP")
        let nfppRaw = FPPNumpy.pairwiseSum(Array(probs[nTarget...]))
        let fpp = Swift.min(Swift.max(fppRaw, 0.0), 1.0)
        let nfpp = Swift.min(Swift.max(nfppRaw, 0.0), 1.0)
        let ess = names.indices.map { i -> Double in
            if let d = diag[names[i]] { return d.ess }
            return values[i].isFinite ? .nan : 0.0
        }
        let nDraws = names.map { name -> Double in diag[name].map { Double($0.n) } ?? Double(n) }
        let relVar = names.indices.map { i -> Double in
            let rv = ess[i] > 0.0 ? npMaximum(1.0 / ess[i] - 1.0 / nDraws[i], 0.0) : 0.0
            return probs[i] > 0.0 ? rv : 0.0
        }
        let falseSet = names.map { !["TP", "PTP", "DTP"].contains($0) }
        let nearbySet = names.indices.map { $0 >= nTarget }
        let fppSE = deltaSE(probs, falseSet, relVar)
        let nfppSE = deltaSE(probs, nearbySet, relVar)
        let beta = names.map { diag[$0]?.tailScale ?? .nan }
        let low = names.indices.map { allLnZ[$0].isFinite && !(ess[$0] >= essMin) }
        let known = names.indices.map { low[$0] && beta[$0].isFinite }
        let sel = names.indices.filter { low[$0] && !known[$0] }
        let heavy = heavyScenarios(sel.map { names[$0] }, sel.map { probs[$0] }, sel.map { ess[$0] })
        let k = tailSigmas * evSD
        let half = names.indices.map { known[$0] ? k * beta[$0] : 0.0 }
        let up = names.indices.map { falseSet[$0] ? half[$0] : -half[$0] }
        let upN = names.indices.map { nearbySet[$0] ? half[$0] : -half[$0] }
        func shifted(_ s: [Double], _ plus: Bool) -> [Double] {
            allLnZ.indices.map { plus ? allLnZ[$0] + s[$0] : allLnZ[$0] - s[$0] }
        }
        let fppHi = fractions(shifted(up, true), falseSet, nearbySet).0
        let fppLo = fractions(shifted(up, false), falseSet, nearbySet).0
        let nfppHi = fractions(shifted(upN, true), falseSet, nearbySet).1
        let nfppLo = fractions(shifted(upN, false), falseSet, nearbySet).1
        let essLive = names.indices.map { (probs[$0] > 0.0 && !ess[$0].isNaN) ? ess[$0] : Double.infinity }
        let anyFinite = essLive.contains(where: \.isFinite)
        var iMin = 0
        if anyFinite { for i in essLive.indices.dropFirst() where essLive[i] < essLive[iMin] { iMin = i } }
        return Summary(names: names, probs: probs, fpp: fpp, nfpp: nfpp, fppSE: fppSE, nfppSE: nfppSE,
                       heavy: heavy, minESS: anyFinite ? ess[iMin] : 0.0, minESSScenario: names[iMin], n: n,
                       ess: ess, relVar: relVar, tailBoxFPP: (fppLo, fppHi), tailBoxNFPP: (nfppLo, nfppHi),
                       nLowESS: low.filter { $0 }.count, tailScale: beta)
    }

    // MARK: assessment, futility, verdict rule

    public struct Assess: Equatable, Sendable {
        public var resolved: Bool
        public var reasons: [String]
    }

    /// `_mc_assess(summ, prev)`.
    public static func assess(_ s: Summary, prev: Summary?) -> Assess {
        var reasons = [String]()
        for g in gatesFPP where abs(s.fpp - g) < sigma * s.fppSE { reasons.append("fpp_gate_\(gateText(g))") }
        if abs(s.nfpp - gateNFPP) < sigma * s.nfppSE { reasons.append("nfpp_gate_\(gateText(gateNFPP))") }
        for name in s.heavy { reasons.append("heavy_tail:\(name)") }
        for g in gatesFPP where s.tailBoxFPP.0 < g && g < s.tailBoxFPP.1 { reasons.append("tail_box_fpp_\(gateText(g))") }
        if s.tailBoxNFPP.0 < gateNFPP && gateNFPP < s.tailBoxNFPP.1 { reasons.append("tail_box_nfpp_\(gateText(gateNFPP))") }
        if let prev {
            for (key, now, before, se, prevSE) in [("fpp", s.fpp, prev.fpp, s.fppSE, prev.fppSE),
                                                   ("nfpp", s.nfpp, prev.nfpp, s.nfppSE, prev.nfppSE)] {
                let tol = sigma * FPPV2.hypot(se, prevSE) + absTol
                if abs(now - before) > tol { reasons.append("stage_disagreement_\(key)") }
            }
        }
        return Assess(resolved: reasons.isEmpty, reasons: reasons)
    }

    /// Index of the first scenario of maximal probability (Python argmax).
    public static func dominantIndex(_ s: Summary) -> Int {
        var i = 0
        for j in s.probs.indices.dropFirst() where s.probs[j] > s.probs[i] { i = j }
        return i
    }

    /// ESS of the dominant scenario.
    public static func dominantESS(_ s: Summary) -> Double {
        guard !s.probs.isEmpty else { return .nan }
        return s.ess[dominantIndex(s)]
    }

    /// Name and ESS of the dominant scenario.
    public static func dominant(_ s: Summary) -> (name: String, ess: Double) {
        guard !s.probs.isEmpty else { return ("", .nan) }
        let idx = dominantIndex(s)
        return (s.names[idx], s.ess[idx])
    }

    /// Task 44 / Python `_mc_is_unstable(summ)`: dominant scenario has `not ess >= MC_FUTILE_ESS_DOM`.
    public static func isUnstable(_ s: Summary) -> Bool {
        !(dominantESS(s) >= futileESSDom)
    }

    /// `_mc_futile(summ)`: identical predicate to `isUnstable`.
    public static func futile(_ s: Summary) -> Bool {
        isUnstable(s)
    }

    /// `mc_validated_3sigma(fpp, fpp_se, nfpp, nfpp_se)`.
    public static func validated3Sigma(fpp: Double, fppSE: Double, nfpp: Double, nfppSE: Double) -> Bool {
        fpp + sigma * fppSE < FPPPolicy.validatedFPP && nfpp + sigma * nfppSE < FPPPolicy.validatedNFPP
    }

    // MARK: stage record and stage loop

    public struct TopScenario: Equatable, Sendable {
        public var scenario: String, p: Double, ess: Double, relSE: Double
        public init(scenario: String, p: Double, ess: Double, relSE: Double) {
            self.scenario = scenario; self.p = p; self.ess = ess; self.relSE = relSE
        }
        public static func == (a: TopScenario, b: TopScenario) -> Bool {
            guard a.scenario == b.scenario else { return false }
            return [a.p, a.ess, a.relSE].map(\.bitPattern) == [b.p, b.ess, b.relSE].map(\.bitPattern)
        }
    }

    /// `_mc_stage_record` (one entry of `refinement_stages`).
    public struct StageRecord: Equatable, Sendable {
        public var stage: Int, n: Int, seed: UInt64, fpp: Double, nfpp: Double, fppMCSE: Double, nfppMCSE: Double
        public var minESS: Double, minESSScenario: String, heavyScenarios: [String], nLowESS: Int
        public var tailBoxFPP: [Double], tailBoxNFPP: [Double], resolved: Bool, reasons: [String]
        public var topScenarios: [TopScenario]
        public init(stage: Int, n: Int, seed: UInt64, fpp: Double, nfpp: Double, fppMCSE: Double, nfppMCSE: Double,
                    minESS: Double, minESSScenario: String, heavyScenarios: [String], nLowESS: Int,
                    tailBoxFPP: [Double], tailBoxNFPP: [Double], resolved: Bool, reasons: [String],
                    topScenarios: [TopScenario]) {
            self.stage = stage; self.n = n; self.seed = seed; self.fpp = fpp; self.nfpp = nfpp
            self.fppMCSE = fppMCSE; self.nfppMCSE = nfppMCSE; self.minESS = minESS
            self.minESSScenario = minESSScenario; self.heavyScenarios = heavyScenarios; self.nLowESS = nLowESS
            self.tailBoxFPP = tailBoxFPP; self.tailBoxNFPP = tailBoxNFPP; self.resolved = resolved
            self.reasons = reasons; self.topScenarios = topScenarios
        }
        public static func == (a: StageRecord, b: StageRecord) -> Bool {
            guard a.stage == b.stage, a.n == b.n, a.seed == b.seed else { return false }
            let fa: [Double] = [a.fpp, a.nfpp, a.fppMCSE, a.nfppMCSE, a.minESS] + a.tailBoxFPP + a.tailBoxNFPP
            let fb: [Double] = [b.fpp, b.nfpp, b.fppMCSE, b.nfppMCSE, b.minESS] + b.tailBoxFPP + b.tailBoxNFPP
            guard fa.map(\.bitPattern) == fb.map(\.bitPattern) else { return false }
            guard a.minESSScenario == b.minESSScenario, a.heavyScenarios == b.heavyScenarios else { return false }
            guard a.nLowESS == b.nLowESS, a.resolved == b.resolved, a.reasons == b.reasons else { return false }
            return a.topScenarios == b.topScenarios
        }
    }

    public static func stageRecord(stage: Int, n: Int, seed: UInt64, summary s: Summary, assess a: Assess) -> StageRecord {
        // np.argsort(-probs, kind="stable"): stable ascending sort of -p (no NaN in probs)
        let neg: [Double] = s.probs.map { -$0 }
        let indices: [Int] = Array(neg.indices)
        let order: [Int] = indices.sorted { (l: Int, r: Int) -> Bool in
            if neg[l] < neg[r] { return true }
            if neg[r] < neg[l] { return false }
            return l < r
        }
        var top = [TopScenario]()
        for i in order.prefix(5) {
            top.append(TopScenario(scenario: s.names[i], p: s.probs[i], ess: s.ess[i], relSE: s.relVar[i].squareRoot()))
        }
        return StageRecord(stage: stage, n: n, seed: seed, fpp: s.fpp, nfpp: s.nfpp, fppMCSE: s.fppSE,
                           nfppMCSE: s.nfppSE, minESS: s.minESS, minESSScenario: s.minESSScenario,
                           heavyScenarios: s.heavy, nLowESS: s.nLowESS, tailBoxFPP: [s.tailBoxFPP.0, s.tailBoxFPP.1],
                           tailBoxNFPP: [s.tailBoxNFPP.0, s.tailBoxNFPP.1], resolved: a.resolved, reasons: a.reasons,
                           topScenarios: Array(top))
    }

    /// One Monte-Carlo pass as the stage loop sees it: `(lnZ, nearby_scen, diag)` plus the caller's payload.
    public struct StagePass<P> {
        public var lnZ: [(String, Double)]
        public var nearby: [(String, Double)]
        public var diag: [String: DrawDiag]
        public var payload: P
        public init(lnZ: [(String, Double)], nearby: [(String, Double)], diag: [String: DrawDiag], payload: P) {
            self.lnZ = lnZ; self.nearby = nearby; self.diag = diag; self.payload = payload
        }
    }

    public struct StagedOutcome<P> {
        public var pass: StagePass<P>
        public var summary: Summary
        public var assess: Assess
        public var log: [StageRecord]
    }

    /// `_mc_run_stages(pass_fn, ladder, seed_for_stage, futility)`; nil when a pass has no finite evidence.
    public static func runStages<P>(
        ladder: [Int], seedForStage: (Int) -> UInt64, futility: Bool,
        pass passFn: (Int, UInt64) throws -> StagePass<P>
    ) rethrows -> StagedOutcome<P>? {
        var log = [StageRecord]()
        var prev: Summary?
        var last: (StagePass<P>, Summary, Assess)?
        for (idx, nK) in ladder.enumerated() {
            let seedK = seedForStage(idx)
            let out = try passFn(nK, seedK)
            guard let summ = summarise(lnZ: out.lnZ, nearby: out.nearby, diag: out.diag, n: nK) else { return nil }
            var a = assess(summ, prev: prev)
            let isFutile = futility && idx == 0 && !a.resolved && idx + 1 < ladder.count && futile(summ)
            if isFutile { a.reasons.append("futility_stop") }
            log.append(stageRecord(stage: idx, n: nK, seed: seedK, summary: summ, assess: a))
            let stop = a.resolved || isFutile
            prev = summ
            last = (out, summ, a)
            if stop { break }
        }
        guard let last else { return nil }
        return StagedOutcome(pass: last.0, summary: last.1, assess: last.2, log: log)
    }
}
