import ExoNitCore
import Foundation

// Missing SPOC FITS (HTTP 404 on a file listed in the metadata): python
// `_load_star` + `fits_cache_helper.ensure_fits_present` skip the file after 3
// attempts and carry on with the files obtained; only when NONE is obtained
// does the task fail (error "load_failed", n_sectors 0, ffi_fallback_on_load_fail
// off). Before this change Swift failed the whole task on the first 404.

private func fitsCard(_ text: String) -> String {
    text.padding(toLength: 80, withPad: " ", startingAt: 0)
}

private func fitsHeader(_ cards: [String]) -> Data {
    var text = cards.map(fitsCard).joined() + fitsCard("END")
    let pad = (2880 - text.utf8.count % 2880) % 2880
    text += String(repeating: " ", count: pad)
    return Data(text.utf8)
}

private func bigEndianBytes<T: FixedWidthInteger>(_ value: T) -> [UInt8] {
    withUnsafeBytes(of: value.bigEndian) { Array($0) }
}

/// Minimal valid SPOC light-curve FITS (TIME/PDCSAP_FLUX as D, QUALITY as J),
/// 2000 rows -> 40 000 data bytes, comfortably above the 30 000-byte floor.
private func syntheticSPOCFITS(start: Double, rows: Int = 2000) -> Data {
    var data = fitsHeader([
        "SIMPLE  =                    T", "BITPIX  =                    8",
        "NAXIS   =                    0", "EXTEND  =                    T",
    ])
    data.append(fitsHeader([
        "XTENSION= 'BINTABLE'", "BITPIX  =                    8", "NAXIS   =                    2",
        "NAXIS1  =                   20", "NAXIS2  =           \(String(format: "%10d", rows))",
        "PCOUNT  =                    0", "GCOUNT  =                    1", "TFIELDS =                    3",
        "TTYPE1  = 'TIME    '", "TFORM1  = '1D      '",
        "TTYPE2  = 'PDCSAP_FLUX'", "TFORM2  = '1D      '",
        "TTYPE3  = 'QUALITY '", "TFORM3  = '1J      '",
    ]))
    var table: [UInt8] = []
    for row in 0..<rows {
        let t = start + Double(row) * 2.0 / 1440.0
        let f = 1.0 + 1e-4 * sin(Double(row) * 0.37)
        table += bigEndianBytes(t.bitPattern)
        table += bigEndianBytes(f.bitPattern)
        table += bigEndianBytes(Int32(0))
    }
    table += [UInt8](repeating: 0, count: (2880 - table.count % 2880) % 2880)
    data.append(contentsOf: table)
    return data
}

private final class FITSRequestLog: @unchecked Sendable {
    private let lock = NSLock()
    private var counts: [String: Int] = [:]
    func hit(_ name: String) { lock.lock(); counts[name, default: 0] += 1; lock.unlock() }
    func count(_ name: String) -> Int { lock.lock(); defer { lock.unlock() }; return counts[name] ?? 0 }
}

private func missingFITSProcessor() throws -> PrimaryTaskProcessor {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("exonit-missing-fits-\(UUID().uuidString)", isDirectory: true)
    return try PrimaryTaskProcessor(
        api: WorkerAPIClient(
            baseURL: URL(string: "http://127.0.0.1:18400")!,
            token: "test-token", session: mockSession()
        ),
        cacheDirectory: directory, maximumConcurrency: 2
    )
}

private let spocFiles = [
    "tess2020-s0014-0000000888-0001-s_lc.fits",
    "hlsp_qlp_tess_ffi_s0015_llc.fits",  // different product: a tag from the LISTED files would be "mixed"
    "tess2020-s0016-0000000888-0003-s_lc.fits",
]

/// Server mock: `status[name]` overrides (404 / 500 / 200 with a short body);
/// anything else serves the synthetic FITS for that file.
private func installMissingFITSMock(
    listed: [String], sectors: [Int], status: [String: Int], shortBody: Set<String> = [],
    log: FITSRequestLog
) {
    MockURLProtocol.handler = { request in
        let path = request.url!.path
        func reply(_ code: Int, _ body: Data) -> (HTTPURLResponse, Data) {
            (HTTPURLResponse(url: request.url!, statusCode: code, httpVersion: nil, headerFields: nil)!, body)
        }
        if path.hasSuffix("/api/metadata/888") {
            let files = listed.map { "\"\($0)\"" }.joined(separator: ",")
            let secs = sectors.map(String.init).joined(separator: ",")
            return reply(200, Data(#"{"tic_id":"888","fits_files":[\#(files)],"sectors":[\#(secs)]}"#.utf8))
        }
        if let range = path.range(of: "/api/fits/") {
            let name = String(path[range.upperBound...])
            log.hit(name)
            if let code = status[name] { return reply(code, Data(#"{"error":"x"}"#.utf8)) }
            if shortBody.contains(name) { return reply(200, Data("<html>error page</html>".utf8)) }
            let index = spocFiles.firstIndex(of: name) ?? 0
            return reply(200, syntheticSPOCFITS(start: 2000 + Double(index) * 3.0))
        }
        return reply(404, Data(#"{"error":"not_found"}"#.utf8))
    }
}

func testSPOCMissingFITSSkipsFile() async throws {
    let missing = spocFiles[1]
    let log = FITSRequestLog()
    installMissingFITSMock(listed: spocFiles, sectors: [14, 15, 16], status: [missing: 404], log: log)
    let partial = try await missingFITSProcessor().loadProcessedLightCurve(ticID: "888", workerID: "wid")
    // python: 3 attempts on the missing file, then skip it; the others are fetched once
    try expect(log.count(missing) == 3, "missing file attempted \(log.count(missing)) times, want 3")
    try expect(log.count(spocFiles[0]) == 1 && log.count(spocFiles[2]) == 1, "other files fetched once")
    try expect(partial.processed.sectorCount == 2, "two curves stitched")
    // python reports len(metadata sectors), not the number of files obtained
    try expect(partial.sectorCount == 3, "n_sectors from metadata, got \(partial.sectorCount)")
    try expect(partial.fppLCSource == "spoc", "product tag from loaded files: \(partial.fppLCSource)")
    try expect(partial.source == .spoc, "spoc source")

    // The light curve equals the one built from exactly the two obtained files
    // (same gap-fill seed: metadata sectors unchanged).
    let log2 = FITSRequestLog()
    installMissingFITSMock(
        listed: [spocFiles[0], spocFiles[2]], sectors: [14, 15, 16], status: [:], log: log2
    )
    let reference = try await missingFITSProcessor().loadProcessedLightCurve(ticID: "888", workerID: "wid")
    try expect(partial.processed == reference.processed, "partial LC != LC of the obtained files")
    try expect(reference.sectorCount == reference.processed.sectorCount, "no skip: sectorCount unchanged")
}

func testSPOCMissingFITSAllMissingFailsLikePython() async throws {
    let log = FITSRequestLog()
    installMissingFITSMock(
        listed: spocFiles, sectors: [14, 15, 16],
        status: [spocFiles[0]: 404, spocFiles[1]: 404, spocFiles[2]: 500], log: log
    )
    let processor = try missingFITSProcessor()
    do {
        _ = try await processor.loadProcessedLightCurve(ticID: "888", workerID: "wid")
        throw TestFailure(description: "all files missing must fail the load")
    } catch PrimaryTaskProcessorError.loadFailed { }
    for name in spocFiles { try expect(log.count(name) == 3, "\(name) attempted \(log.count(name))x") }
    // task level: python returns status failed / error load_failed / n_sectors 0 / not recovered
    let result = try await processor.process(WorkerTask(taskID: "t", ticID: "888"), workerID: "wid")
    try expect(result.status == "failed", "status \(result.status)")
    try expect(result.error == "load_failed", "error \(String(describing: result.error))")
    try expect(result.sectorCount == 0 && result.pointCount == 0, "n_sectors/n_points 0")
    try expect(!result.lightCurveLoadRecovered && result.candidates.isEmpty, "no recovery, no candidates")
}

func testSPOCMissingFITSInvalidFilesAreSkippedAndCachePurged() async throws {
    // A 200 with a short HTML body is an invalid download (python MIN_FITS_BYTES): retried, then skipped.
    let bad = spocFiles[2]
    let log = FITSRequestLog()
    installMissingFITSMock(listed: spocFiles, sectors: [14, 15, 16], status: [:], shortBody: [bad], log: log)
    let processor = try missingFITSProcessor()
    let loaded = try await processor.loadProcessedLightCurve(ticID: "888", workerID: "wid")
    try expect(log.count(bad) == 3, "invalid download retried 3x, got \(log.count(bad))")
    try expect(loaded.processed.sectorCount == 2 && loaded.sectorCount == 3, "skipped invalid file")

    // An invalid CACHED file is purged and re-downloaded (python step 1), a valid one is trusted.
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("exonit-missing-fits-cache-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try Data("truncated".utf8).write(to: directory.appendingPathComponent(spocFiles[0]))
    try syntheticSPOCFITS(start: 2003).write(to: directory.appendingPathComponent(spocFiles[1]))
    let log2 = FITSRequestLog()
    installMissingFITSMock(listed: Array(spocFiles[0...1]), sectors: [14, 15], status: [:], log: log2)
    let cached = try PrimaryTaskProcessor(
        api: WorkerAPIClient(
            baseURL: URL(string: "http://127.0.0.1:18400")!, token: "test-token", session: mockSession()
        ),
        cacheDirectory: directory, maximumConcurrency: 2
    )
    let result = try await cached.loadProcessedLightCurve(ticID: "888", workerID: "wid")
    try expect(log2.count(spocFiles[0]) == 1, "invalid cached file re-downloaded")
    try expect(log2.count(spocFiles[1]) == 0, "valid cached file not re-downloaded")
    try expect(result.processed.sectorCount == 2 && result.sectorCount == 2, "both loaded")
    try expect(
        FITSReader.invalidReason(at: directory.appendingPathComponent(spocFiles[0])) == nil,
        "re-downloaded cache file is valid"
    )
}

/// Synthetic FITS with an exact row count and no table padding (exact byte sizes).
private func syntheticSPOCFITSUnpadded(rows: Int) -> Data {
    var data = fitsHeader([
        "SIMPLE  =                    T", "BITPIX  =                    8",
        "NAXIS   =                    0", "EXTEND  =                    T",
    ])
    data.append(fitsHeader([
        "XTENSION= 'BINTABLE'", "BITPIX  =                    8", "NAXIS   =                    2",
        "NAXIS1  =                   20", "NAXIS2  =           \(String(format: "%10d", rows))",
        "PCOUNT  =                    0", "GCOUNT  =                    1", "TFIELDS =                    3",
        "TTYPE1  = 'TIME    '", "TFORM1  = '1D      '",
        "TTYPE2  = 'PDCSAP_FLUX'", "TFORM2  = '1D      '",
        "TTYPE3  = 'QUALITY '", "TFORM3  = '1J      '",
    ]))
    var table: [UInt8] = []
    for row in 0..<rows {
        table += bigEndianBytes((2000.0 + Double(row) * 2.0 / 1440.0).bitPattern)
        table += bigEndianBytes((1.0 + 1e-4 * sin(Double(row) * 0.37)).bitPattern)
        table += bigEndianBytes(Int32(0))
    }
    data.append(contentsOf: table)
    return data
}

/// python MIN_FITS_BYTES: 29 999 bytes is too short, exactly 30 000 is accepted; an empty
/// BINTABLE above the floor is "empty data HDU"; a symlinked cache entry is judged by its target.
func testFITSValidityBoundariesEmptyHDUAndSymlinks() throws {
    try withTemporaryDirectory { directory in
        let exact = syntheticSPOCFITSUnpadded(rows: 1212)  // 2 x 2880 header + 1212 x 20 = 30 000
        try expect(exact.count == 30_000, "fixture size \(exact.count)")
        let at = directory.appendingPathComponent("at.fits")
        try exact.write(to: at)
        try expect(FITSReader.invalidReason(at: at) == nil, "exactly 30000 bytes is valid: \(String(describing: FITSReader.invalidReason(at: at)))")
        let below = directory.appendingPathComponent("below.fits")
        try exact.prefix(29_999).write(to: below)
        try expect(
            FITSReader.invalidReason(at: below) == "size too short (29999 bytes < 30000)",
            "29999: \(String(describing: FITSReader.invalidReason(at: below)))"
        )
        let emptyHDU = directory.appendingPathComponent("emptyhdu.fits")
        try (syntheticSPOCFITSUnpadded(rows: 0) + Data(repeating: 0, count: 31_000)).write(to: emptyHDU)
        try expect(FITSReader.invalidReason(at: emptyHDU) == "empty data HDU", "empty HDU: \(String(describing: FITSReader.invalidReason(at: emptyHDU)))")

        let good = directory.appendingPathComponent("good-target.fits")
        try syntheticSPOCFITS(start: 2000).write(to: good)
        let bad = directory.appendingPathComponent("bad-target.fits")
        try Data("truncated".utf8).write(to: bad)
        let goodLink = directory.appendingPathComponent("good-link.fits")
        let badLink = directory.appendingPathComponent("bad-link.fits")
        try FileManager.default.createSymbolicLink(at: goodLink, withDestinationURL: good)
        try FileManager.default.createSymbolicLink(at: badLink, withDestinationURL: bad)
        try expect(FITSReader.invalidReason(at: goodLink) == nil, "symlink to a valid file is valid")
        try expect(FITSReader.invalidReason(at: badLink) != nil, "symlink to an invalid file is invalid")
    }
}

/// A symlinked cache entry that points at a valid file is used as is (not purged, not re-downloaded).
func testSPOCSymlinkedCacheEntryIsKept() async throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("exonit-missing-fits-links-\(UUID().uuidString)", isDirectory: true)
    let store = directory.appendingPathComponent("store", isDirectory: true)
    let cache = directory.appendingPathComponent("cache", isDirectory: true)
    try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let target = store.appendingPathComponent("target.fits")
    try syntheticSPOCFITS(start: 2000).write(to: target)
    try FileManager.default.createSymbolicLink(at: cache.appendingPathComponent(spocFiles[0]), withDestinationURL: target)
    let log = FITSRequestLog()
    installMissingFITSMock(listed: [spocFiles[0]], sectors: [14], status: [:], log: log)
    let processor = try PrimaryTaskProcessor(
        api: WorkerAPIClient(
            baseURL: URL(string: "http://127.0.0.1:18400")!, token: "test-token", session: mockSession()
        ),
        cacheDirectory: cache, maximumConcurrency: 2
    )
    let loaded = try await processor.loadProcessedLightCurve(ticID: "888", workerID: "wid")
    try expect(log.count(spocFiles[0]) == 0, "valid symlinked entry must not be re-downloaded")
    try expect(loaded.processed.sectorCount == 1, "symlinked file was read")
    let attributes = try FileManager.default.attributesOfItem(atPath: cache.appendingPathComponent(spocFiles[0]).path)
    try expect(attributes[.type] as? FileAttributeType == .typeSymbolicLink, "link kept")
}

/// An invalid cached file is purged BEFORE the attempts start, so a server that 404s it still
/// gets all 3 attempts (python: unlink, then the 3-attempt loop).
func testSPOCInvalidCachedFileStillGetsThreeAttempts() async throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("exonit-missing-fits-purge-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try Data("truncated".utf8).write(to: directory.appendingPathComponent(spocFiles[0]))
    let log = FITSRequestLog()
    installMissingFITSMock(listed: Array(spocFiles[0...1]), sectors: [14, 15], status: [spocFiles[0]: 404], log: log)
    let processor = try PrimaryTaskProcessor(
        api: WorkerAPIClient(
            baseURL: URL(string: "http://127.0.0.1:18400")!, token: "test-token", session: mockSession()
        ),
        cacheDirectory: directory, maximumConcurrency: 2
    )
    let loaded = try await processor.loadProcessedLightCurve(ticID: "888", workerID: "wid")
    try expect(log.count(spocFiles[0]) == 3, "3 attempts after the purge, got \(log.count(spocFiles[0]))")
    try expect(!FileManager.default.fileExists(atPath: directory.appendingPathComponent(spocFiles[0]).path), "nothing left behind")
    try expect(loaded.processed.sectorCount == 1 && loaded.sectorCount == 2, "skipped, n_sectors from metadata")
}

func testFITSInvalidReasonMatchesPython() throws {
    try withTemporaryDirectory { directory in
        let empty = directory.appendingPathComponent("empty.fits")
        try Data().write(to: empty)
        try expect(FITSReader.invalidReason(at: empty) == "empty file", "empty")
        let tiny = directory.appendingPathComponent("tiny.fits")
        try Data(repeating: 0, count: 100).write(to: tiny)
        try expect(
            FITSReader.invalidReason(at: tiny) == "size too short (100 bytes < 30000)", "tiny")
        let good = directory.appendingPathComponent("good.fits")
        try syntheticSPOCFITS(start: 2000).write(to: good)
        try expect(FITSReader.invalidReason(at: good) == nil, "good file valid")
        let cut = directory.appendingPathComponent("cut.fits")
        try syntheticSPOCFITS(start: 2000).prefix(35_000).write(to: cut)
        try expect(FITSReader.invalidReason(at: cut) == "truncated data HDU", "cut: \(String(describing: FITSReader.invalidReason(at: cut)))")
        let noExt = directory.appendingPathComponent("noext.fits")
        try (fitsHeader(["SIMPLE  =                    T", "BITPIX  =                    8", "NAXIS   =                    0"])
             + Data(repeating: 0, count: 40_000)).write(to: noExt)
        try expect(FITSReader.invalidReason(at: noExt) != nil, "no usable data HDU")
        try expect(FITSReader.invalidReason(at: directory.appendingPathComponent("nope.fits")) == "file does not exist", "missing")
    }
}
