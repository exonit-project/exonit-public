import ExoNitCore
import Foundation

// `ffi_fallback_on_load_fail` (python `ExoNitWorker._try_ffi_fallback` +
// `worker/server_ffi_client.py::fetch_ffi`): flag ON and the FITS light curve
// cannot be loaded -> use the server's FFI cache as is (no gap fill, no detrend),
// minimal metadata, product tag "unknown", lc_load_recovered true. Flag OFF or
// the FFI cache also lacking the TIC -> failed / load_failed. Partial FITS
// success never falls back. Same cases as the Rust `tests/ffi_fallback.rs`.

private func ffbCard(_ text: String) -> String { text.padding(toLength: 80, withPad: " ", startingAt: 0) }

private func ffbHeader(_ cards: [String]) -> Data {
    var text = cards.map(ffbCard).joined() + ffbCard("END")
    text += String(repeating: " ", count: (2880 - text.utf8.count % 2880) % 2880)
    return Data(text.utf8)
}

private func ffbBE<T: FixedWidthInteger>(_ value: T) -> [UInt8] { withUnsafeBytes(of: value.bigEndian) { Array($0) } }

/// Valid SPOC light-curve FITS (40 000 data bytes). `column` = the flux column name (11 chars).
private func ffbSPOCFITS(column: String, quality: Int32 = 0) -> Data {
    let rows = 2000
    var data = ffbHeader([
        "SIMPLE  =                    T", "BITPIX  =                    8",
        "NAXIS   =                    0", "EXTEND  =                    T",
    ])
    data.append(ffbHeader([
        "XTENSION= 'BINTABLE'", "BITPIX  =                    8", "NAXIS   =                    2",
        "NAXIS1  =                   20", "NAXIS2  =           \(String(format: "%10d", rows))",
        "PCOUNT  =                    0", "GCOUNT  =                    1", "TFIELDS =                    3",
        "TTYPE1  = 'TIME    '", "TFORM1  = '1D      '",
        "TTYPE2  = '\(column)'", "TFORM2  = '1D      '",
        "TTYPE3  = 'QUALITY '", "TFORM3  = '1J      '",
    ]))
    var table: [UInt8] = []
    for row in 0..<rows {
        table += ffbBE((2000.0 + Double(row) * 2.0 / 1440.0).bitPattern)
        table += ffbBE((1.0 + 1e-4 * sin(Double(row) * 0.37)).bitPattern)
        table += ffbBE(quality)
    }
    table += [UInt8](repeating: 0, count: (2880 - table.count % 2880) % 2880)
    data.append(contentsOf: table)
    return data
}

/// FFI cache body: 3 sectors delivered out of time order, `nulls` flux samples null (NaN for python).
private func ffbFFIBody(pointsPerSector: Int, nulls: Int) -> Data {
    var sectors: [[String: Any]] = []
    for (k, base) in [(1, 2040.0), (0, 2000.0), (2, 2080.0)] {
        let time = (0..<pointsPerSector).map { base + Double($0) * 0.02 }
        let flux: [Any] = (0..<pointsPerSector).map { i in
            (k == 0 && i < nulls) ? NSNull() : (1.0 + 1e-3 * sin(Double(i + k * 7) * 0.31)) as Any
        }
        sectors.append(["sector": 14 + k, "source": "qlp", "time": time, "flux": flux])
    }
    return try! JSONSerialization.data(withJSONObject: ["tic_id": "888", "sectors": sectors])
}

private final class FFBLog: @unchecked Sendable {
    private let lock = NSLock()
    private var counts: [String: Int] = [:]
    private var firstHeaders: [String: [String: String]] = [:]
    func hit(_ key: String, _ request: URLRequest) {
        lock.lock(); defer { lock.unlock() }
        counts[key, default: 0] += 1
        if firstHeaders[key] == nil { firstHeaders[key] = request.allHTTPHeaderFields ?? [:] }
    }
    func count(_ key: String) -> Int { lock.lock(); defer { lock.unlock() }; return counts[key] ?? 0 }
    func headers(_ key: String) -> [String: String] { lock.lock(); defer { lock.unlock() }; return firstHeaders[key] ?? [:] }
}

private let ffbFiles = ["tess2020-s0014-0000000888-0001-s_lc.fits", "tess2020-s0015-0000000888-0002-s_lc.fits"]

private struct FFBSetup {
    var listed: [String] = ffbFiles
    var metadataStatus = 200
    var fitsStatus: [String: Int] = [ffbFiles[0]: 404, ffbFiles[1]: 404]
    var badFluxColumn = false
    /// every cadence carries a quality flag in the default bitmask -> the cleaned light curve is empty
    var allFlagged = false
    var ffi: Data? = ffbFFIBody(pointsPerSector: 600, nulls: 5)  // nil -> 404
    var ffiStatus = 200
}

private func ffbInstall(_ setup: FFBSetup, _ log: FFBLog) {
    MockURLProtocol.handler = { request in
        let path = request.url!.path
        func reply(_ code: Int, _ body: Data) -> (HTTPURLResponse, Data) {
            (HTTPURLResponse(url: request.url!, statusCode: code, httpVersion: nil, headerFields: nil)!, body)
        }
        if path.hasSuffix("/api/metadata/888") {
            log.hit("metadata", request)
            if setup.metadataStatus != 200 { return reply(setup.metadataStatus, Data(#"{"error":"x"}"#.utf8)) }
            let files = setup.listed.map { "\"\($0)\"" }.joined(separator: ",")
            return reply(200, Data(
                #"{"tic_id":"888","fits_files":[\#(files)],"sectors":[14,15,16],"ra":12.5,"dec":-3.25,"tmag":9.5,"stellar_radius":1.3,"stellar_mass":1.1}"#.utf8
            ))
        }
        if path.hasSuffix("/api/ffi/888") {
            log.hit("ffi", request)
            guard let body = setup.ffi else { return reply(404, Data(#"{"error":"nf"}"#.utf8)) }
            return setup.ffiStatus == 200 ? reply(200, body) : reply(setup.ffiStatus, Data(#"{"error":"x"}"#.utf8))
        }
        if let range = path.range(of: "/api/fits/") {
            let name = String(path[range.upperBound...])
            log.hit(name, request)
            if let code = setup.fitsStatus[name] { return reply(code, Data(#"{"error":"x"}"#.utf8)) }
            return reply(200, ffbSPOCFITS(column: setup.badFluxColumn ? "XXXXXX_FLUX" : "PDCSAP_FLUX", quality: setup.allFlagged ? 1 : 0))
        }
        return reply(404, Data(#"{"error":"not_found"}"#.utf8))
    }
}

private func ffbProcessor(flag: Bool) throws -> PrimaryTaskProcessor {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("exonit-ffifb-\(UUID().uuidString)", isDirectory: true)
    return try PrimaryTaskProcessor(
        api: WorkerAPIClient(baseURL: URL(string: "http://127.0.0.1:18400")!, token: "tok-123", session: mockSession()),
        cacheDirectory: directory, maximumConcurrency: 2, ffiFallbackOnLoadFail: flag
    )
}

func testFFIFallbackUsedWhenAllFilesMissing() async throws {
    let log = FFBLog()
    ffbInstall(FFBSetup(), log)
    let loaded = try await ffbProcessor(flag: true).loadProcessedLightCurve(ticID: "888", workerID: "wid")
    try expect(log.count("ffi") == 1, "no retries on the FFI cache, got \(log.count("ffi"))")
    let headers = log.headers("ffi")
    try expect(headers["Authorization"] == "Bearer tok-123", "Bearer token: \(headers)")
    try expect(headers["X-Worker-ID"] == nil, "python fetch_ffi sends no X-Worker-ID")
    for f in ffbFiles { try expect(log.count(f) == 3, "\(f) attempted \(log.count(f))x before falling back") }
    let p = loaded.processed
    try expect(p.time.count == 1800 - 5, "NaN samples dropped: \(p.time.count)")
    try expect(zip(p.time, p.time.dropFirst()).allSatisfy { $0 <= $1 }, "sorted by time")
    try expect(p.flatFlux == p.rawFlux && p.detrendFrame == nil, "flat == raw == FFI flux as is (no detrend)")
    try expect(p.flatFlux.allSatisfy { $0 > 0.99 && $0 < 1.01 }, "flux not modified")
    try expect(loaded.recovered, "lc_load_recovered")
    try expect(loaded.fppLCSource == "unknown", "product tag \(loaded.fppLCSource)")
    try expect(loaded.metadata.dataSource == "ffi", "data_source ffi")
    try expect(loaded.sectorCount == 3, "n_sectors from the refetched metadata sectors: \(loaded.sectorCount)")
    try expect(loaded.metadata.ra == 12.5 && loaded.metadata.dec == -3.25 && loaded.metadata.tmag == 9.5, "ra/dec/tmag refetched")
    try expect(loaded.metadata.stellarRadius == nil && loaded.metadata.stellarMass == nil, "python fallback meta has no stellar parameters")
    try expect(loaded.metadata.fitsFiles.isEmpty, "no fits files")
    try expect(loaded.source == .ffi, "ffi source")
}

func testFFIFallbackResultMarkedRecoveredEndToEnd() async throws {
    let log = FFBLog()
    ffbInstall(FFBSetup(ffi: ffbFFIBody(pointsPerSector: 600, nulls: 0)), log)
    let config: [String: JSONValue] = [
        "run_injection_recovery": .bool(false), "max_planets": .integer(1), "n_periods": .integer(2000),
        "wotan_long_sweep": .bool(false),
    ]
    let result = try await ffbProcessor(flag: true).process(WorkerTask(taskID: "t", ticID: "888", config: config), workerID: "wid")
    try expect(result.status == "completed", "completed, got \(result.status) \(result.error ?? "")")
    try expect(result.lightCurveLoadRecovered, "lc_load_recovered true")
    try expect(result.sectorCount == 3 && result.pointCount == 1800 && result.ra == 12.5, "n_sectors/n_points/ra")
}

func testFFIFallbackDisabledAndFFIMissingFailLikePython() async throws {
    // flag off: load_failed, the FFI cache is never called
    let log = FFBLog()
    ffbInstall(FFBSetup(), log)
    let off = try ffbProcessor(flag: false)
    do {
        _ = try await off.loadProcessedLightCurve(ticID: "888", workerID: "wid")
        throw TestFailure(description: "flag off must fail")
    } catch PrimaryTaskProcessorError.loadFailed { }
    try expect(log.count("ffi") == 0, "FFI cache untouched with the flag off")
    let failed = try await off.process(WorkerTask(taskID: "t", ticID: "888"), workerID: "wid")
    try expect(failed.status == "failed" && failed.error == "load_failed" && !failed.lightCurveLoadRecovered, "flag off result")
    // flag on, FFI cache 404: load_failed, one attempt
    let log404 = FFBLog()
    ffbInstall(FFBSetup(ffi: nil), log404)
    let on = try ffbProcessor(flag: true)
    do {
        _ = try await on.loadProcessedLightCurve(ticID: "888", workerID: "wid")
        throw TestFailure(description: "FFI 404 must fail")
    } catch PrimaryTaskProcessorError.loadFailed { }
    try expect(log404.count("ffi") == 1, "one attempt, no retries")
    let result = try await on.process(WorkerTask(taskID: "t", ticID: "888"), workerID: "wid")
    try expect(result.status == "failed" && result.error == "load_failed" && result.sectorCount == 0, "FFI 404 result")
    // flag on, FFI HTTP 500 / too few points / malformed / no sectors: all fail
    var setup500 = FFBSetup(); setup500.ffiStatus = 500
    let bodies: [(FFBSetup, String)] = [
        (setup500, "http 500"),
        (FFBSetup(ffi: ffbFFIBody(pointsPerSector: 30, nulls: 0)), "fewer than 100 finite points"),
        (FFBSetup(ffi: Data(#"{"tic_id":"888"}"#.utf8)), "malformed"),
        (FFBSetup(ffi: Data(#"{"tic_id":"888","sectors":[]}"#.utf8)), "no sectors"),
    ]
    for (setup, label) in bodies {
        ffbInstall(setup, FFBLog())
        do {
            _ = try await ffbProcessor(flag: true).loadProcessedLightCurve(ticID: "888", workerID: "wid")
            throw TestFailure(description: "\(label) must fail")
        } catch PrimaryTaskProcessorError.loadFailed { }
    }
}

func testFFIFallbackPartialFilesNeverFallBack() async throws {
    let log = FFBLog()
    var setup = FFBSetup()
    setup.fitsStatus = [ffbFiles[1]: 404]
    ffbInstall(setup, log)
    let loaded = try await ffbProcessor(flag: true).loadProcessedLightCurve(ticID: "888", workerID: "wid")
    try expect(log.count("ffi") == 0, "partial FITS success must not touch the FFI cache")
    try expect(!loaded.recovered && loaded.processed.sectorCount == 1 && loaded.sectorCount == 3, "partial result")
    try expect(loaded.fppLCSource == "spoc", "product tag spoc")
}

func testFFIFallbackOnUnreadableFITSMetadataFailureAndEmptyList() async throws {
    // unreadable FITS (valid structure, no flux column) = python lightkurve_exception
    var unreadable = FFBSetup(); unreadable.fitsStatus = [:]; unreadable.badFluxColumn = true
    let log = FFBLog(); ffbInstall(unreadable, log)
    let recovered = try await ffbProcessor(flag: true).loadProcessedLightCurve(ticID: "888", workerID: "wid")
    try expect(recovered.recovered && log.count("ffi") == 1, "read error falls back")
    let logOff = FFBLog(); ffbInstall(unreadable, logOff)
    do {
        _ = try await ffbProcessor(flag: false).loadProcessedLightCurve(ticID: "888", workerID: "wid")
        throw TestFailure(description: "read error with flag off must fail")
    } catch is FITSError { }
    try expect(logOff.count("ffi") == 0, "no fallback with the flag off")
    // metadata 500: fallback; the best-effort refetch also fails -> sectors [] -> n_sectors 1, no ra
    var noMeta = FFBSetup(); noMeta.metadataStatus = 500
    ffbInstall(noMeta, FFBLog())
    let viaMeta = try await ffbProcessor(flag: true).loadProcessedLightCurve(ticID: "888", workerID: "wid")
    try expect(viaMeta.recovered && viaMeta.sectorCount == 1 && viaMeta.metadata.ra == nil, "metadata failure falls back")
    let logMetaOff = FFBLog(); ffbInstall(noMeta, logMetaOff)
    do {
        _ = try await ffbProcessor(flag: false).loadProcessedLightCurve(ticID: "888", workerID: "wid")
        throw TestFailure(description: "metadata failure with the flag off must fail")
    } catch is WorkerAPIError { }
    try expect(logMetaOff.count("ffi") == 0, "no fallback with the flag off")
    // no FITS listed
    var empty = FFBSetup(); empty.listed = []
    ffbInstall(empty, FFBLog())
    let viaEmpty = try await ffbProcessor(flag: true).loadProcessedLightCurve(ticID: "888", workerID: "wid")
    try expect(viaEmpty.recovered && viaEmpty.sectorCount == 3, "empty FITS list falls back")
}

func testFFIFallbackWorkerConfigStrictBoolTimeoutAndHeartbeat() async throws {
    func decode(_ extra: String) throws -> WorkerConfig {
        try JSONDecoder().decode(
            WorkerConfig.self, from: Data((#"{"server_url":"http://localhost:8400","token":"t""# + extra + "}").utf8)
        )
    }
    let absent = try decode("")
    try expect(!absent.ffiFallbackOnLoadFail && absent.ffiFetchTimeout == 30, "defaults")
    for (literal, want) in [("true", true), ("false", false), ("\"true\"", false), ("1", false), ("null", false)] {
        let config = try decode(#","ffi_fallback_on_load_fail":"# + literal)
        try expect(config.ffiFallbackOnLoadFail == want, "ffi_fallback_on_load_fail \(literal)")
        try expect(WorkerRuntime.heartbeatFlagStates(config)["ffi_fallback_on_load_fail"] == .bool(want), "heartbeat \(literal)")
    }
    let t1 = try decode(#","ffi_fetch_timeout":12.5"#).ffiFetchTimeout
    let t2 = try decode(#","ffi_fetch_timeout":"x""#).ffiFetchTimeout
    let t3 = try decode(#","ffi_fetch_timeout":-1"#).ffiFetchTimeout
    try expect(t1 == 12.5, "timeout 12.5")
    try expect(t2 == 30, "bad timeout -> 30")
    try expect(t3 == 30, "negative timeout -> 30")
}

/// python wraps read + stitch + clean + detrend in ONE `try ... except Exception`: a file that reads
/// fine but leaves an EMPTY light curve after the quality cut also falls back.
func testFFIFallbackOnEmptyLightCurveAfterCleaning() async throws {
    var flagged = FFBSetup(); flagged.fitsStatus = [:]; flagged.allFlagged = true
    let log = FFBLog(); ffbInstall(flagged, log)
    let recovered = try await ffbProcessor(flag: true).loadProcessedLightCurve(ticID: "888", workerID: "wid")
    try expect(recovered.recovered && log.count("ffi") == 1, "empty LC after cleaning falls back")
    let logOff = FFBLog(); ffbInstall(flagged, logOff)
    var failedAsExpected = false
    do {
        _ = try await ffbProcessor(flag: false).loadProcessedLightCurve(ticID: "888", workerID: "wid")
    } catch { failedAsExpected = true }
    try expect(failedAsExpected, "empty LC with the flag off must fail")
    try expect(logOff.count("ffi") == 0, "no fallback with the flag off")
}

/// NaN / negative / zero / huge timeouts are invalid and become python's 30 s; valid ones are kept.
func testFFIFetchTimeoutIsValidated() async throws {
    func processor(_ t: Double) throws -> PrimaryTaskProcessor {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("exonit-ffifb-\(UUID().uuidString)", isDirectory: true)
        return try PrimaryTaskProcessor(
            api: WorkerAPIClient(baseURL: URL(string: "http://127.0.0.1:18400")!, token: "t", session: mockSession()),
            cacheDirectory: directory, ffiFallbackOnLoadFail: true, ffiFetchTimeout: t
        )
    }
    for bad in [Double.nan, .infinity, -.infinity, -1, 0, 1e30, 86_401] {
        let v = try processor(bad).ffiFetchTimeout
        try expect(v == 30, "\(bad) -> \(v)")
    }
    let ok1 = try processor(12.5).ffiFetchTimeout
    let ok2 = try processor(86_400).ffiFetchTimeout
    try expect(ok1 == 12.5 && ok2 == 86_400, "valid timeouts kept")
    // a fallback run with a huge (sanitised) timeout works end to end
    ffbInstall(FFBSetup(), FFBLog())
    let loaded = try await processor(1e30).loadProcessedLightCurve(ticID: "888", workerID: "wid")
    try expect(loaded.recovered, "fallback with a huge (sanitised) timeout")
}
