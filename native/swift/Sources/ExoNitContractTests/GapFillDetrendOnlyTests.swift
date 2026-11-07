import ExoNitCore
import Foundation

// Config flag `gap_fill_detrend_only` (default off), port of python
// worker/gap_fill.py::DetrendFrame + worker.py::_load_star
// (docs/reviews/2026-10-15-fill-gaps-sigma-parity.md, option A). Off: legacy
// path unchanged. On: gap-fill points feed the detrender only; the returned
// arrays (BLS / vetting / CNN / injection input) hold real points only and the
// multi-planet re-detrends run on `detrendFrame`.

private let gapFillDetrendOnlySeed: UInt32 = 12345

/// 12 d of 2-min SPOC-like cadence, two intra-sector gaps (1.0 d and 0.7 d:
/// filled, each longer than wotan's 0.5 d break tolerance), a 3.1 d sinusoid,
/// deterministic noise and (optionally) a 2.3 d box transit.
private func gapFillDetrendOnlyCurve(transit: Bool) -> LightCurveData {
    let dt = 2.0 / 1440.0
    var time: [Double] = [], flux: [Double] = [], cadence: [Int64] = []
    var state: UInt64 = 0x9E37_79B9_7F4A_7C15
    for c in 0..<Int64(12.0 / dt) {
        let rel = Double(c) * dt
        if (rel > 4.0 && rel < 5.0) || (rel > 8.0 && rel < 8.7) { continue }
        state = state &* 6364136223846793005 &+ 1442695040888963407
        let u = Double(state >> 11) / Double(UInt64(1) << 53) - 0.5
        let t = 1000.0 + rel
        var f = 1.0 + 1.4e-3 * u + 2e-3 * sin(2 * Double.pi * rel / 3.1)
        if transit {
            let phase = (t - 1000.37 + 23.0).truncatingRemainder(dividingBy: 2.3) / 2.3
            if phase < 0.02 || phase > 0.98 { f -= 3e-3 }
        }
        time.append(t); flux.append(f); cadence.append(c)
    }
    return LightCurveData(time: time, flux: flux, rawRowCount: time.count, cadence: cadence)
}

private func bitsOf(_ values: [Double]) -> [UInt64] { values.map(\.bitPattern) }

/// FNV-1a 64 over the little-endian bit patterns (golden hash, review R2).
private func gapFillDetrendOnlyHash(_ values: [Double]) -> String {
    var h: UInt64 = 0xcbf29ce484222325
    for v in values {
        var bits = v.bitPattern
        for _ in 0..<8 { h ^= bits & 0xff; h = h &* 0x00000100000001b3; bits >>= 8 }
    }
    return String(format: "%016llx", h)
}

/// Flag OFF pinned to the pre-change build (review R2): hashes produced by a
/// throwaway build of 1ee2aca (before a45c6d9 introduced the flag) running
/// this exact curve through `LightCurvePreprocessor.process(.., maxConcurrency: 2,
/// fillSeed: 12345)` on the .60 Mac. Any flag-OFF drift fails here.
func testGapFillDetrendOnlyFlagOff() async throws {
    let curve = gapFillDetrendOnlyCurve(transit: true)
    let golden: [Bool: (count: Int, time: String, flat: String, raw: String)] = [
        false: (8280, "ea2ed783b0332989", "2d2894580cc21d42", "5ac1419c87bcb4e5"),
        true: (8280, "ea2ed783b0332989", "043898c34cbfc0e9", "5ac1419c87bcb4e5"),
    ]
    for iterative in [false, true] {
        let off = try LightCurvePreprocessor.process(
            [curve], iterativeDetrend: iterative, maxConcurrency: 2, fillSeed: gapFillDetrendOnlySeed,
            gapFillDetrendOnly: false
        )
        let want = golden[iterative]!
        try expect(off.time.count == want.count, "flag off count vs pre-change golden (iterative=\(iterative))")
        try expect(gapFillDetrendOnlyHash(off.time) == want.time, "flag off time vs pre-change golden (iterative=\(iterative))")
        try expect(gapFillDetrendOnlyHash(off.flatFlux) == want.flat, "flag off flat vs pre-change golden (iterative=\(iterative))")
        try expect(gapFillDetrendOnlyHash(off.rawFlux) == want.raw, "flag off raw vs pre-change golden (iterative=\(iterative))")
        try expect(off.detrendFrame == nil, "flag off: no detrend frame")
        let defaulted = try LightCurvePreprocessor.process(
            [curve], iterativeDetrend: iterative, maxConcurrency: 2, fillSeed: gapFillDetrendOnlySeed
        )
        try expect(defaulted == off, "default == explicit off")
    }
}

func testGapFillDetrendOnlyRealPoints() async throws {
    let curve = gapFillDetrendOnlyCurve(transit: false)
    let real = Set(curve.time.map(\.bitPattern))
    let off = try LightCurvePreprocessor.process(
        [curve], iterativeDetrend: false, fillSeed: gapFillDetrendOnlySeed
    )
    let on = try LightCurvePreprocessor.process(
        [curve], iterativeDetrend: false, fillSeed: gapFillDetrendOnlySeed, gapFillDetrendOnly: true
    )
    guard let frame = on.detrendFrame else { throw TestFailure(description: "flag on: frame expected") }
    try expect(frame.syntheticCount > 100, "fixture must contain gap-fill points")
    try expect(on.time.allSatisfy { real.contains($0.bitPattern) }, "synthetic timestamp leaked")
    try expect(on.time.count < off.time.count, "synthetic points dropped")
    try expect(bitsOf(frame.time) == bitsOf(off.time), "frame time == flag-off series")
    try expect(bitsOf(frame.rawFlux) == bitsOf(off.rawFlux), "frame raw == flag-off series")
    var fluxByTime: [UInt64: Double] = [:]
    for (t, f) in zip(curve.time, curve.flux) { fluxByTime[t.bitPattern] = f }
    for (k, i) in frame.realIndex.enumerated() {
        try expect(on.time[k].bitPattern == off.time[i].bitPattern, "real time")
        try expect(on.flatFlux[k].bitPattern == off.flatFlux[i].bitPattern, "real flat == flag off (biweight)")
        try expect(on.rawFlux[k].bitPattern == fluxByTime[on.time[k].bitPattern]!.bitPattern, "real raw untouched")
    }
}

func testGapFillDetrendOnlyIterativeExcludesSynthetic() async throws {
    let curve = gapFillDetrendOnlyCurve(transit: true)
    // Reconstruct the pre-detrend series with the synthetic mask.
    var noise = SeededGaussianNoiseSource(seed: gapFillDetrendOnlySeed)
    let filled = try GapFiller.fillIfAllowed(
        GapFillSeries(time: curve.time, flux: curve.flux, cadence: curve.cadence), noise: &noise
    )
    let realCadence = Set(curve.cadence!)
    let synthetic = filled.cadence!.map { !realCadence.contains($0) }
    try expect(synthetic.contains(true), "fixture must contain gap-fill points")
    let excluded = try IterativeDetrender.flatten(
        time: filled.time, flux: filled.flux, maxConcurrency: 2, excluded: synthetic
    )
    let plain = try IterativeDetrender.flatten(time: filled.time, flux: filled.flux, maxConcurrency: 2)
    try expect(excluded.wasRefined, "fixture transit must trigger the refined pass")
    try expect(excluded.transitMask.contains(true), "mask non-empty")
    for index in synthetic.indices where synthetic[index] {
        try expect(!excluded.transitMask[index], "synthetic point must never be transit-masked")
    }
    if excluded.quickBLS == plain.quickBLS {
        for index in synthetic.indices where !synthetic[index] {
            try expect(excluded.transitMask[index] == plain.transitMask[index], "real-point mask unchanged")
        }
    }
    let on = try LightCurvePreprocessor.process(
        [curve], iterativeDetrend: true, maxConcurrency: 2, fillSeed: gapFillDetrendOnlySeed,
        gapFillDetrendOnly: true
    )
    let real = Set(curve.time.map(\.bitPattern))
    try expect(on.detrendFrame != nil, "iterative flag on: frame")
    try expect(on.time.allSatisfy { real.contains($0.bitPattern) }, "iterative: synthetic timestamp leaked")
}

func testGapFillDetrendOnlyFrameAndMultiPlanet() async throws {
    let time = (0..<400).map { Double($0) * 0.01 }
    let raw = time.map { 1.0 + 0.01 * $0 }
    let synthetic = (0..<400).map { (100..<150).contains($0) }
    let frame = try DetrendFrame(time: time, rawFlux: raw, synthetic: synthetic)
    try expect(frame.syntheticCount == 50, "synthetic count")
    var maskReal = [Bool](repeating: false, count: 350)
    maskReal[0] = true; maskReal[100] = true
    let full = try frame.expandMask(maskReal)
    try expect(full[0] && full[150] && full.filter { $0 }.count == 2, "mask scatter")
    let expected = try BiweightDetrender.flatten(time: time, flux: raw, transitMask: full)
    let got = try frame.flatten(transitMask: maskReal)
    try expect(bitsOf(got) == bitsOf(frame.gather(expected)), "frame flatten")
    try expectThrows("wrong-length mask must throw") { _ = try frame.expandMask([Bool](repeating: false, count: 400)) }

    let curve = gapFillDetrendOnlyCurve(transit: true)
    let on = try LightCurvePreprocessor.process(
        [curve], iterativeDetrend: true, maxConcurrency: 2, fillSeed: gapFillDetrendOnlySeed,
        gapFillDetrendOnly: true
    )
    var options = MultiPlanetOptions()
    options.maximumPlanets = 3
    options.periodMaximum = 10
    options.periodCount = 3000
    options.longPeriodSweep = nil
    let result = try MultiPlanetSearcher.search(
        time: on.time, rawFlux: on.rawFlux, initialDetrendedFlux: on.flatFlux,
        stellarRadiusSolar: nil, stellarMassSolar: nil, source: .spoc, options: options,
        maxConcurrency: 2, detrendFrame: on.detrendFrame
    )
    try expect(!result.candidates.isEmpty, "transit recovered with frame")
    try expect(abs(result.candidates[0].periodDays - 2.3) < 0.05, "period \(result.candidates[0].periodDays)")
    try expect(result.accumulatedMask.count == on.time.count, "mask over real points")
    let bad = try DetrendFrame(time: [0, 1, 2], rawFlux: [1, 1, 1], synthetic: [false, true, false])
    try expectThrows("mismatched frame must throw") {
        _ = try MultiPlanetSearcher.search(
            time: on.time, rawFlux: on.rawFlux, initialDetrendedFlux: on.flatFlux,
            stellarRadiusSolar: nil, stellarMassSolar: nil, source: .spoc, options: options,
            maxConcurrency: 2, detrendFrame: bad
        )
    }
}

// MARK: - worker config / FPP backfill parity

/// FFI (TGLC, no cadence) curve with two 0.5-5 d gaps -> time-grid gap fill.
private func gapFillDetrendOnlyFFIBody() -> Data {
    let dt = 10.0 / 1440.0
    var time: [Double] = [], flux: [Double] = []
    var state: UInt64 = 0x2545_F491_4F6C_DD1D
    for c in 0..<Int64(12.0 / dt) {
        let rel = Double(c) * dt
        if (rel > 4.0 && rel < 5.0) || (rel > 8.0 && rel < 8.7) { continue }
        state = state &* 6364136223846793005 &+ 1442695040888963407
        let u = Double(state >> 11) / Double(UInt64(1) << 53) - 0.5
        let t = 2000.0 + rel
        var f = 1.0 + 1.4e-3 * u + 2e-3 * sin(2 * Double.pi * rel / 3.1)
        if (t - 2000.37 + 23.0).truncatingRemainder(dividingBy: 2.3) / 2.3 < 0.02 { f -= 3e-3 }
        time.append(t); flux.append(f)
    }
    let object: [String: Any] = [
        "tic_id": "777",
        "sectors": [["sector": 14, "source": "tglc", "time": time, "flux": flux]],
    ]
    return try! JSONSerialization.data(withJSONObject: object)
}

private func gapFillDetrendOnlyProcessor(workerFlag: Bool) throws -> PrimaryTaskProcessor {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("exonit-gfdo-\(UUID().uuidString)", isDirectory: true)
    return try PrimaryTaskProcessor(
        api: WorkerAPIClient(
            baseURL: URL(string: "http://127.0.0.1:18400")!,
            token: "test-token", session: mockSession()
        ),
        cacheDirectory: directory, maximumConcurrency: 2, gapFillDetrendOnly: workerFlag
    )
}

func testGapFillDetrendOnlyBackfillParity() async throws {
    let ffi = gapFillDetrendOnlyFFIBody()
    MockURLProtocol.handler = { request in
        let path = request.url!.path
        let ok = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
        if path.hasSuffix("/api/metadata/777") {
            return (ok, Data(#"{"tic_id":"777","fits_files":[],"sectors":[14]}"#.utf8))
        }
        if path.hasSuffix("/api/ffi/777") { return (ok, ffi) }
        let missing = HTTPURLResponse(url: request.url!, statusCode: 404, httpVersion: nil, headerFields: nil)!
        return (missing, Data(#"{"error":"not_found"}"#.utf8))
    }
    for flag in [false, true] {
        let processor = try gapFillDetrendOnlyProcessor(workerFlag: flag)
        let primary = try await processor.loadPrimaryLightCurve(
            WorkerTask(taskID: "t", ticID: "777"), workerID: "wid"
        )
        // exactly the call NativeFPPTaskProcessor / BackfillTaskProcessor make
        let backfill = try await processor.loadProcessedLightCurve(ticID: "777", workerID: "wid")
        try expect(primary.processed == backfill.processed, "in-task and backfill LC differ (flag=\(flag))")
        try expect((primary.processed.detrendFrame != nil) == flag, "frame presence follows flag=\(flag)")
    }
    let on = try gapFillDetrendOnlyProcessor(workerFlag: true)
    try expect(on.gapFillDetrendOnly(taskConfig: nil), "worker value when no task config")
    try expect(on.gapFillDetrendOnly(taskConfig: [:]), "worker value when key absent")
    try expect(!on.gapFillDetrendOnly(taskConfig: ["gap_fill_detrend_only": .bool(false)]), "task config wins")
    let overridden = try await on.loadPrimaryLightCurve(
        WorkerTask(taskID: "t", ticID: "777", config: ["gap_fill_detrend_only": .bool(false)]), workerID: "wid"
    )
    try expect(overridden.processed.detrendFrame == nil, "task config false overrides worker true")
    let off = try gapFillDetrendOnlyProcessor(workerFlag: false)
    try expect(off.gapFillDetrendOnly(taskConfig: ["gap_fill_detrend_only": .bool(true)]), "task config true wins")
    try expect(!off.gapFillDetrendOnly(taskConfig: nil), "default off")
}

func testGapFillDetrendOnlyWorkerConfigAndHeartbeat() async throws {
    let off = try JSONDecoder().decode(
        WorkerConfig.self, from: Data(#"{"server_url":"http://localhost:8400","token":"t"}"#.utf8)
    )
    try expect(!off.gapFillDetrendOnly, "worker config default off")
    let on = try JSONDecoder().decode(
        WorkerConfig.self,
        from: Data(#"{"server_url":"http://localhost:8400","token":"t","gap_fill_detrend_only":true}"#.utf8)
    )
    try expect(on.gapFillDetrendOnly, "worker config key decoded")
    try expect(WorkerRuntime.heartbeatFlagStates(on)["gap_fill_detrend_only"] == .bool(true), "heartbeat on")
    try expect(WorkerRuntime.heartbeatFlagStates(off)["gap_fill_detrend_only"] == .bool(false), "heartbeat off")
}

// MARK: - review 2026-10-15: shared precedence table + long-period sweep

/// Shared cross-language table (python tests/worker/test_gap_fill_detrend_only.py
/// and Rust tests/gap_fill_detrend_only.rs pin the same rows): only real JSON
/// bools count; a non-bool task value falls back to the worker value; a
/// non-bool worker value reads as false.
func testGapFillDetrendOnlyPrecedenceTable() async throws {
    let taskCases: [(String?, Bool?)] = [  // (JSON literal or nil = absent, expected or nil = worker value)
        ("true", true), ("false", false), ("\"true\"", nil), ("\"false\"", nil),
        ("1", nil), ("0", nil), ("null", nil), (nil, nil),
    ]
    let workerCases: [(String?, Bool)] = [
        ("true", true), ("false", false), ("\"true\"", false), ("\"false\"", false),
        ("1", false), ("0", false), ("null", false), (nil, false),
    ]
    for (workerRaw, workerEffective) in workerCases {
        let json = workerRaw.map {
            #"{"server_url":"http://localhost:8400","token":"t","gap_fill_detrend_only":"# + $0 + "}"
        } ?? #"{"server_url":"http://localhost:8400","token":"t"}"#
        let config = try JSONDecoder().decode(WorkerConfig.self, from: Data(json.utf8))
        try expect(config.gapFillDetrendOnly == workerEffective, "worker \(workerRaw ?? "absent")")
        let processor = try gapFillDetrendOnlyProcessor(workerFlag: config.gapFillDetrendOnly)
        try expect(processor.gapFillDetrendOnly(taskConfig: nil) == workerEffective, "backfill: worker value")
        for (taskRaw, expected) in taskCases {
            var task: [String: JSONValue] = [:]
            if let taskRaw {
                task["gap_fill_detrend_only"] = try JSONDecoder().decode(JSONValue.self, from: Data(taskRaw.utf8))
            }
            let want = expected ?? workerEffective
            try expect(
                processor.gapFillDetrendOnly(taskConfig: task) == want,
                "worker \(workerRaw ?? "absent") task \(taskRaw ?? "absent")"
            )
        }
    }
}

private final class SweepSpy: @unchecked Sendable {
    private let lock = NSLock()
    private var calls: [([Double], [Double])] = []
    func record(_ time: [Double], _ flux: [Double]) { lock.lock(); calls.append((time, flux)); lock.unlock() }
    var all: [([Double], [Double])] { lock.lock(); defer { lock.unlock() }; return calls }
}

/// Review item 2: the long-period sweep runs with a frame. Every window's
/// detrend works on the full frame (the BLS flux equals the frame detrend at
/// the real points and differs from a real-points-only detrend), and the
/// sweep BLS sees real timestamps only.
func testGapFillDetrendOnlyLongPeriodSweep() async throws {
    let curve = gapFillDetrendOnlyCurve(transit: true)
    let real = Set(curve.time.map(\.bitPattern))
    let on = try LightCurvePreprocessor.process(
        [curve], iterativeDetrend: false, fillSeed: gapFillDetrendOnlySeed, gapFillDetrendOnly: true
    )
    guard let frame = on.detrendFrame else { throw TestFailure(description: "frame expected") }
    var options = LongPeriodSweepOptions()
    options.baselineThresholdDays = 10
    options.periodMinimumDays = 3.25
    options.legacyPeriodMaximumDays = 10
    options.basePeriodCount = 500
    options.periodCountScaleReferenceMaximumDays = nil
    options.minimumSNR = 0
    let spy = SweepSpy()
    let search: BLSSearchFunction = { time, flux, parameters in
        spy.record(time, flux)
        return try BLSAccelerated.search(time: time, flux: flux, parameters: parameters, backend: .cpu, maxConcurrency: 2)
    }
    let result = try MultiPlanetSearcher.longPeriodSweepStep(
        time: on.time, rawFlux: on.rawFlux, accumulatedMask: [Bool](repeating: false, count: on.time.count),
        options: options, detrendFrame: frame, search: search
    )
    try expect(result != nil, "sweep must run (minimumSNR 0)")
    let calls = spy.all
    try expect(calls.count == options.windowsDays.count, "one BLS per window")
    for ((time, flux), window) in zip(calls, options.windowsDays) {
        try expect(time.allSatisfy { real.contains($0.bitPattern) }, "synthetic timestamp in sweep BLS")
        try expect(bitsOf(time) == bitsOf(on.time), "sweep BLS time == real points")
        let full = try frame.flatten(windowLength: window)
        let finite = full.filter(\.isFinite).sorted()
        let middle = finite.count / 2
        let fill = finite.isEmpty ? 1 : (finite.count.isMultiple(of: 2)
            ? (finite[middle - 1] + finite[middle]) / 2 : finite[middle])
        let expected = full.map { $0.isFinite ? $0 : fill }
        try expect(bitsOf(flux) == bitsOf(expected), "window \(window): sweep detrend must use the full frame")
        let realOnly = try BiweightDetrender.flatten(time: on.time, flux: on.rawFlux, windowLength: window)
        try expect(bitsOf(full) != bitsOf(realOnly), "fixture must distinguish frame vs real-only detrend")
    }
}
