import Foundation

/// Observer of the cross-platform v3 FPP gate (plan 2026-10-15-fpp-native-fix-plan.md, decision (t) point 2): the
/// per-scenario per-draw quantities that `scripts/fpp/gen_v3_gate_dumps.py` records from the Python reference, recorded
/// at the same places of this port (the Swift twin of the Rust `fpp::v2::gate`). OFF unless a `Recorder` is open on the
/// thread (only the `FPPGateDump` driver opens one); it never changes a computed bit, it only copies.
///
/// Per scenario reduction of a pass (Python `_lnZ_from_logw` call), in reduction order:
/// * `lnl`: the full per-draw lnL array (all N draws, -inf where not evaluated), BEFORE the prior;
/// * production pass: `state` (int8 per draw: -1 not evaluated by a production fold kernel, else the kernel's
///   exposure-order status) and the contact roots (`FPPV2Contact.contactTimes` with the k the kernel used) of every
///   evaluated draw whose lnL is finite and within `window` nat of the maximum finite lnL of that evaluation;
/// * parity pass: no state and no contacts.
/// Contact rows also carry tangency margins and CSR exposure order, affected bins and ladder levels (F4xs).
/// Device callers supply their own metadata; requireDeviceRoots also requires order/levels, with no CPU fallback.
/// One dilution run of the v3 gate dump (Python `calc_fpp_native` call), scenarios in Python's order.
public struct FPPV3GateRun: Sendable {
    public struct Scenario: Sendable {
        public var name: String
        public var lnZ: Double
        public var probability: Double
        /// nil = the scenario never reached its likelihood reduction
        public var record: FPPV3Gate.Record?
    }
    public var dilutionCorrected: Bool
    public var fpp: Double
    public var nfpp: Double
    public var scenarios: [Scenario]
}

public enum FPPV3Gate {
    /// Per kernel row: exposure order (-1 for no affected bin), ascending affected bin ids and
    /// their ladder levels. CSR offsets have one more entry than order.
    public struct DiscreteRows: Sendable, Equatable {
        public var order: [Int] = []
        public var binsOff: [Int64] = [0]
        public var bins: [Int32] = []
        public var levels: [Int] = []

        public init() {}

        public mutating func push(order: Int, bins: [Int32], levels: [Int]) {
            precondition(bins.count == levels.count, "DiscreteRows: bins and levels must be parallel")
            self.order.append(order)
            self.bins.append(contentsOf: bins)
            self.levels.append(contentsOf: levels)
            binsOff.append(Int64(self.bins.count))
        }

        mutating func pushDraw(_ r: FPPV2DrawBinModels) {
            push(order: r.nAff == 0 ? -1 : r.orderIdx, bins: r.idx.map { Int32($0) }, levels: r.levels)
        }

        mutating func append(_ other: DiscreteRows) {
            let base = Int64(bins.count)
            order.append(contentsOf: other.order)
            bins.append(contentsOf: other.bins)
            levels.append(contentsOf: other.levels)
            binsOff.append(contentsOf: other.binsOff.dropFirst().map { $0 + base })
        }

        /// A device caller's arrays must be checked before slicing or concatenating.
        func defect(rows n: Int) -> String? {
            if order.count != n || binsOff.count != n + 1 {
                return "order \(order.count) / bins_off \(binsOff.count) for \(n) rows"
            }
            if binsOff[0] != 0 || zip(binsOff, binsOff.dropFirst()).contains(where: { $0.1 < $0.0 }) {
                return "bins_off is not a non-decreasing offset list from 0"
            }
            if binsOff[n] != Int64(bins.count) || bins.count != levels.count {
                return "bins \(bins.count) / levels \(levels.count) / bins_off end \(binsOff[n])"
            }
            return nil
        }

        func row(_ j: Int) -> (order: Int, bins: [Int32], levels: [Int]) {
            let range = Int(binsOff[j])..<Int(binsOff[j + 1])
            return (order[j], Array(bins[range]), Array(levels[range]))
        }
    }

    public static func tangencyMargin(zMin: Double, k: Double) -> Double {
        if zMin.isNaN { return .nan }
        return min(abs(zMin - (1.0 + k)), abs(zMin - abs(1.0 - k)))
    }

    public struct Record: Sendable {
        public var lnl: [Double]
        public var state: [Int8]?
        /// (draw indices ascending, roots row-major [n, 5])
        public var contacts: (idx: [Int64], roots: [Double])?
        public var inputDigests: [String] = []
        public var rootsSource: String = "cpu"        // "cpu" | "device"
        /// One entry per contact row, in exactly the same draw order.
        public var tangency: [Double]?
        public var discrete: DiscreteRows?

        public init(
            lnl: [Double],
            state: [Int8]? = nil,
            contacts: (idx: [Int64], roots: [Double])? = nil,
            inputDigests: [String] = [],
            rootsSource: String = "cpu",
            tangency: [Double]? = nil,
            discrete: DiscreteRows? = nil
        ) {
            self.lnl = lnl
            self.state = state
            self.contacts = contacts
            self.inputDigests = inputDigests
            self.rootsSource = rootsSource
            self.tangency = tangency
            self.discrete = discrete
        }
    }

    struct KernelRow {
        var period, aRs, incRad, ecc, wRad, k, lnl: Double
        var status: Int
        var roots: [Double]?
        var tangency: Double?
    }
    struct Note {
        var slot: String
        var selected: [Int]
        var rows: [KernelRow]
        var digests: [String]
        var discrete: DiscreteRows
    }

    final class Box: @unchecked Sendable {
        let production: Bool
        let window: Double
        let requireDeviceRoots: Bool
        var rows: [KernelRow] = []
        var digests: [String] = []
        var discrete = DiscreteRows()
        var notes: [Note] = []
        var staged: Record?? = nil
        var records: [Record?] = []
        var named: [String: Record?] = [:]
        var error: String?
        init(production: Bool, window: Double, requireDeviceRoots: Bool = false) {
            self.production = production
            self.window = window
            self.requireDeviceRoots = requireDeviceRoots
        }
        func fail(_ m: String) { if error == nil { error = m } }
    }

    static let key = "exonit.fpp.v3.gateObserver"
    static func box() -> Box? { Thread.current.threadDictionary[key] as? Box }
    public static var active: Bool { box() != nil }

    /// RAII-style recorder: open for one calculation on this thread, then `close()`.
    public final class Recorder {
        private let b: Box
        /// Device mode requires device roots and discrete rows; missing device tangency remains NaN.
        public init(production: Bool, window: Double, requireDeviceRoots: Bool = false) {
            b = Box(production: production, window: window, requireDeviceRoots: requireDeviceRoots)
            Thread.current.threadDictionary[FPPV3Gate.key] = b
        }
        public var error: String? { b.error }
        public func close() {
            if (Thread.current.threadDictionary[FPPV3Gate.key] as? Box) === b {
                Thread.current.threadDictionary.removeObject(forKey: FPPV3Gate.key)
            }
        }
        deinit { close() }
    }

    /// A production kernel's rows in kernel row order.
    static func kernelRows(periods: [Double], aRs: [Double], incsRad: [Double], eccs: [Double], wsRad: [Double],
                           ksUsed: [Double], lnl: [Double], status: [Int],
                           input: CallInput,
                           deviceRoots: [Double]?, discrete: DiscreteRows? = nil,
                           deviceTangency: [Double]? = nil) {
        guard let b = box() else { return }
        let digest = inputDigest(input)
        b.digests.append(digest)
        let n = lnl.count
        let rootsValid: Bool
        if let dr = deviceRoots {
            if dr.count != n * 5 {
                b.fail("gate: device roots length \(dr.count) != 5 * n \(5 * n)")
                rootsValid = false
            } else {
                rootsValid = true
            }
        } else {
            if b.requireDeviceRoots {
                b.fail("gate: device roots required but None supplied")
            }
            rootsValid = false
        }
        if let discrete {
            if let defect = discrete.defect(rows: n) {
                b.fail("gate: device order/levels malformed: \(defect)")
            } else {
                b.discrete.append(discrete)
            }
        } else if b.requireDeviceRoots {
            b.fail("gate: device order/levels required but None supplied")
        }
        let tangencyValid: Bool
        if let t = deviceTangency {
            tangencyValid = t.count == n
            if !tangencyValid { b.fail("gate: device tangency length \(t.count) != n \(n)") }
        } else {
            tangencyValid = false
        }
        for j in 0..<n {
            let roots: [Double]?
            if rootsValid, let dr = deviceRoots {
                let base = j * 5
                roots = [dr[base], dr[base + 1], dr[base + 2], dr[base + 3], dr[base + 4]]
            } else {
                roots = nil
            }
            b.rows.append(KernelRow(period: periods[j], aRs: aRs[j], incRad: incsRad[j], ecc: eccs[j], wRad: wsRad[j],
                                    k: ksUsed[j], lnl: lnl[j], status: status[j], roots: roots,
                                    tangency: tangencyValid ? deviceTangency?[j] : nil))
        }
    }

    /// Python `_note_exposure_stats`: the kernel rows since the last note belong to the evaluation of `selected`.
    static func note(_ slot: String, _ selected: [Int]) {
        guard let b = box() else { return }
        let rows = b.rows
        b.rows = []
        let digests = b.digests
        b.digests = []
        let discrete = b.discrete
        b.discrete = DiscreteRows()
        if !discrete.order.isEmpty && discrete.order.count != rows.count {
            b.fail("gate note \(slot): order/levels for \(discrete.order.count) of \(rows.count) kernel rows")
        }
        if rows.count != selected.count { b.fail("gate note \(slot): \(rows.count) rows for \(selected.count) draws") }
        b.notes.removeAll { $0.slot == slot }
        b.notes.append(Note(slot: slot, selected: selected, rows: rows, digests: digests, discrete: discrete))
    }

    /// Stage the full lnL array (before the prior) of the reduction that follows.
    static func stage(_ slot: String, _ lnlFull: [Double]) {
        guard let b = box() else { return }
        let n = lnlFull.count
        var state: [Int8]? = nil
        var contacts: (idx: [Int64], roots: [Double])? = nil
        var inputDigests: [String] = []
        var hasDeviceRoots = false
        var tangency: [Double]? = nil
        var discrete: DiscreteRows? = nil
        if let i = b.notes.firstIndex(where: { $0.slot == slot }) {
            let note = b.notes.remove(at: i)
            inputDigests = note.digests
            let nDev = note.rows.filter { $0.roots != nil }.count
            if nDev != 0 && nDev != note.rows.count {
                b.fail("gate stage \(slot): device roots on only \(nDev) of \(note.rows.count) rows")
            }
            hasDeviceRoots = nDev != 0 && nDev == note.rows.count

            guard note.rows.count == note.selected.count else { b.fail("gate stage \(slot): rows/selection"); return }
            var st = [Int8](repeating: -1, count: n)
            for (j, d) in note.selected.enumerated() {
                let r = note.rows[j]
                if d >= n || r.lnl.bitPattern != lnlFull[d].bitPattern {
                    b.fail("gate stage \(slot): batch lnL does not land at draw \(d)"); return
                }
                st[d] = Int8(r.status)
            }
            var mx = -Double.infinity
            for r in note.rows where r.lnl.isFinite && r.lnl > mx { mx = r.lnl }
            let floor = mx - b.window
            // Draw index, kernel row index, roots, margin. Sort all payloads together.
            var keep: [(Int64, Int, [Double], Double)] = []
            // window = inf (gate protocol 2026-10-25, generator 853609f): EVERY evaluated draw, finite lnL or not
            let every = b.window.isInfinite
            if every || floor.isFinite {
                for (j, d) in note.selected.enumerated() {
                    let r = note.rows[j]
                    if every || (r.lnl.isFinite && r.lnl >= floor) {
                        if let roots = r.roots {
                            keep.append((Int64(d), j, roots, r.tangency ?? .nan))
                        } else if b.requireDeviceRoots {
                            b.fail("gate stage: device roots required but row has no device roots")
                            continue
                        } else {
                            let (roots, zMin) = FPPV2Contact.contactTimesZMin(
                                period: r.period, aRs: r.aRs, incRad: r.incRad, ecc: r.ecc, wRad: r.wRad, k: r.k)
                            keep.append((Int64(d), j, roots, r.tangency ?? tangencyMargin(zMin: zMin, k: r.k)))
                        }
                    }
                }
            }
            // stable sort by draw index (Python argsort(kind="stable"))
            let order = keep.indices.sorted { keep[$0].0 != keep[$1].0 ? keep[$0].0 < keep[$1].0 : $0 < $1 }
            state = st
            contacts = (order.map { keep[$0].0 }, order.flatMap { keep[$0].2 })
            tangency = order.map { keep[$0].3 }
            if !note.discrete.order.isEmpty && note.discrete.order.count == note.rows.count {
                var selectedDiscrete = DiscreteRows()
                for i in order {
                    let row = note.discrete.row(keep[i].1)
                    selectedDiscrete.push(order: row.order, bins: row.bins, levels: row.levels)
                }
                discrete = selectedDiscrete
            }
        } else if b.production {
            state = [Int8](repeating: -1, count: n)
            contacts = ([], [])
            tangency = []
        }
        let rootsSource = (b.requireDeviceRoots || hasDeviceRoots) ? "device" : "cpu"
        b.staged = .some(Record(lnl: lnlFull, state: state, contacts: contacts, inputDigests: inputDigests,
                                rootsSource: rootsSource, tangency: tangency, discrete: discrete))
    }

    /// Every RECORDED reduction (in lockstep with the step-7 sink) appends the staged payload, or none.
    static func pushReduction() {
        guard let b = box() else { return }
        if case let .some(rec) = b.staged { b.records.append(rec) } else { b.records.append(nil) }
        b.staged = nil
    }

    static func beginPass() {
        guard let b = box() else { return }
        b.records = []; b.named = [:]; b.rows = []; b.digests = []; b.notes = []; b.staged = nil
        b.discrete = DiscreteRows()
    }

    /// The records since `n0` belong to `names` (the rule of the draw-weight diagnostics).
    static func assign(_ n0: Int, _ names: [String]) {
        guard let b = box() else { return }
        let got: [Record?] = n0 <= b.records.count ? Array(b.records[n0...]) : []
        if !got.isEmpty && got.count != names.count { b.fail("gate: \(got.count) records for \(names.count) scenarios") }
        if n0 <= b.records.count { for i in n0..<b.records.count { b.records[i] = nil } }
        for (i, name) in names.enumerated() { b.named[name] = i < got.count ? got[i] : nil }
    }

    static func takeNamed() -> [String: Record?] {
        guard let b = box() else { return [:] }
        let out = b.named
        b.named = [:]
        return out
    }
}

extension FPPV3Gate {
    /// Canonical per-call input record (addendum v1.1 section 5, parts a-f).
    public struct CallInput: Sendable {
        // a. per-draw arrays, struct order = digest order
        public var k, period, inc, aRs, ecc, wPeri, u1, u2, companionFluxRatio: [Double]
        /// b. EB arrays, in this order: ebFluxRatios, u1Sec, u2Sec. Empty for the non-EB kernel.
        public var ebArrays: [[Double]] = []
        /// c. sigmaVeto for the EB kernel, nil otherwise. (w, delta, baseChi2, lnnorm come from `fold`.)
        public var sigmaVeto: Double? = nil
        /// d. flags in this order: companionIsHost, then EB only: rejectSec, secConsistent, kNudgeUpstream.
        public var flags: [Bool] = []
        /// c (w, delta, baseChi2, lnnorm) and e (fold arrays, f)
        public var fold: FPPV2ProdFold

        public init(
            k: [Double],
            period: [Double],
            inc: [Double],
            aRs: [Double],
            ecc: [Double],
            wPeri: [Double],
            u1: [Double],
            u2: [Double],
            companionFluxRatio: [Double],
            ebArrays: [[Double]] = [],
            sigmaVeto: Double? = nil,
            flags: [Bool] = [],
            fold: FPPV2ProdFold
        ) {
            self.k = k
            self.period = period
            self.inc = inc
            self.aRs = aRs
            self.ecc = ecc
            self.wPeri = wPeri
            self.u1 = u1
            self.u2 = u2
            self.companionFluxRatio = companionFluxRatio
            self.ebArrays = ebArrays
            self.sigmaVeto = sigmaVeto
            self.flags = flags
            self.fold = fold
        }
    }

    static let realGLSHA256: [UInt8] = {
        var b = [UInt8]()
        b.reserveCapacity((FPPV2GLTables.xBits.count + FPPV2GLTables.wBits.count) * 8)
        for v in FPPV2GLTables.xBits {
            withUnsafeBytes(of: v.littleEndian) { b.append(contentsOf: $0) }
        }
        for v in FPPV2GLTables.wBits {
            withUnsafeBytes(of: v.littleEndian) { b.append(contentsOf: $0) }
        }
        let hex = SHA256Swift.hex(b)
        var bytes = [UInt8]()
        bytes.reserveCapacity(32)
        var it = hex.makeIterator()
        while let c1 = it.next(), let c2 = it.next() {
            if let byte = UInt8(String([c1, c2]), radix: 16) {
                bytes.append(byte)
            }
        }
        return bytes
    }()

    private static func appendF64Slice(_ buf: inout [UInt8], _ slice: [Double]) {
        let count = UInt64(slice.count)
        withUnsafeBytes(of: count.littleEndian) { buf.append(contentsOf: $0) }
        for x in slice {
            withUnsafeBytes(of: x.bitPattern.littleEndian) { buf.append(contentsOf: $0) }
        }
    }

    private static func appendI64Slice(_ buf: inout [UInt8], _ slice: [Int]) {
        let count = UInt64(slice.count)
        withUnsafeBytes(of: count.littleEndian) { buf.append(contentsOf: $0) }
        for x in slice {
            let i = Int64(x)
            withUnsafeBytes(of: i.littleEndian) { buf.append(contentsOf: $0) }
        }
    }

    /// Same, with the 32-byte GL table hash given (tests).
    public static func inputDigest(_ c: CallInput, glSHA256: [UInt8]) -> String {
        var buf = [UInt8]()
        // a. per-draw f64 arrays: k, P, inc, aR, ecc, argument of periastron, u1, u2, companion flux ratio
        appendF64Slice(&buf, c.k)
        appendF64Slice(&buf, c.period)
        appendF64Slice(&buf, c.inc)
        appendF64Slice(&buf, c.aRs)
        appendF64Slice(&buf, c.ecc)
        appendF64Slice(&buf, c.wPeri)
        appendF64Slice(&buf, c.u1)
        appendF64Slice(&buf, c.u2)
        appendF64Slice(&buf, c.companionFluxRatio)

        // b. EB only: eb flux ratios, u1 secondary, u2 secondary
        for arr in c.ebArrays {
            appendF64Slice(&buf, arr)
        }

        // c. f64 scalars: exposure width w and delta of the production fold, base_chi2, lnnorm; EB only: sigma_veto
        withUnsafeBytes(of: c.fold.w.bitPattern.littleEndian) { buf.append(contentsOf: $0) }
        withUnsafeBytes(of: c.fold.delta.bitPattern.littleEndian) { buf.append(contentsOf: $0) }
        withUnsafeBytes(of: c.fold.baseChi2.bitPattern.littleEndian) { buf.append(contentsOf: $0) }
        withUnsafeBytes(of: c.fold.lnnorm.bitPattern.littleEndian) { buf.append(contentsOf: $0) }
        if let sv = c.sigmaVeto {
            withUnsafeBytes(of: sv.bitPattern.littleEndian) { buf.append(contentsOf: $0) }
        }

        // d. flags, one byte each: companion_is_host; EB only: reject_sec, sec_consistent, k_nudge_upstream
        for f in c.flags {
            buf.append(f ? 1 : 0)
        }

        // e. fold arrays in the field order of FPPV2FoldArrays (ints as i64), then the production fold flux f
        appendI64Slice(&buf, c.fold.arrays.n)
        appendF64Slice(&buf, c.fold.arrays.flux)
        appendF64Slice(&buf, c.fold.arrays.sigma)
        appendF64Slice(&buf, c.fold.arrays.tmean)
        appendF64Slice(&buf, c.fold.arrays.varT)
        appendF64Slice(&buf, c.fold.arrays.tmin)
        appendF64Slice(&buf, c.fold.arrays.tmax)
        appendF64Slice(&buf, c.fold.arrays.nodeTab)
        appendI64Slice(&buf, c.fold.arrays.ptsOff)
        appendF64Slice(&buf, c.fold.arrays.ptsT)
        appendF64Slice(&buf, c.fold.f)

        // f. the sha256 of the GL_X and GL_W tables, as 32 raw bytes
        buf.append(contentsOf: glSHA256)

        return SHA256Swift.hex(buf)
    }

    /// SHA-256 of the canonical record, 64 lowercase hex characters. Part f uses the real GL tables (cached, lazy).
    public static func inputDigest(_ c: CallInput) -> String {
        inputDigest(c, glSHA256: realGLSHA256)
    }
}
