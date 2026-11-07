import CoreML
import Foundation

public enum PrimaryTaskProcessorError: Error, CustomStringConvertible, Sendable {
    case noLightCurve(String)
    /// Every FITS file listed for the TIC was missing or invalid. Same outcome
    /// as python `_load_star` returning None with `ffi_fallback_on_load_fail`
    /// off: failed task, error "load_failed".
    case loadFailed(String)
    case missingRequiredModel(String)

    public var description: String {
        switch self {
        case let .noLightCurve(tic): return "no usable light curve for TIC \(tic)"
        case .loadFailed: return "load_failed"
        case let .missingRequiredModel(path): return "required Core ML model missing: \(path)"
        }
    }
}

public struct LoadedStar: Sendable {
    public let metadata: StarMetadata
    public let processed: ProcessedLightCurve
    public let source: LightCurveSource
    /// Product label detected from the FITS files used for this light curve.
    /// FFI fallback has no trustworthy product label and is always unknown.
    public let fppLCSource: String
    /// python `lc_load_recovered`: this light curve came from the FFI fallback
    /// (`tryFFIFallback`), not from the listed FITS files.
    public let recovered: Bool
    /// `n_sectors` of the result. python `_load_star` reports the number of
    /// sectors in the server metadata even when some FITS files were skipped
    /// (missing/invalid); with no file skipped this equals `processed.sectorCount`.
    public let sectorCount: Int
}

/// Light curves read from the FITS files that could be obtained, plus which
/// listed files were skipped (python `_load_star` step 2 skip semantics).
struct LoadedCurves {
    var curves: [LightCurveData]
    var usedFilenames: [String]
    var skippedFilenames: [String]
}

extension PrimaryTaskProcessor: FPPLightCurveLoading {
    public func loadFPPLightCurve(ticID: String, workerID: String) async throws -> FPPLoadedLightCurve {
        let loaded = try await loadProcessedLightCurve(ticID: ticID, workerID: workerID)
        return FPPLoadedLightCurve(
            time: loaded.processed.time, flux: loaded.processed.flatFlux, lcSource: loaded.fppLCSource
        )
    }
}

public final class PrimaryTaskProcessor: @unchecked Sendable {
    private let api: WorkerAPIClient
    private let cacheDirectory: URL
    private let cnn: CoreMLFFIClassifier?
    private let exominer: CoreMLExoMinerFFI?
    private let maximumConcurrency: Int
    private let blsBackend: BLSComputeBackend
    private let injectionBackend: BLSComputeBackend
    /// Worker-config `gap_fill_detrend_only` (default false); see
    /// `gapFillDetrendOnly(taskConfig:)` for the precedence.
    public let gapFillDetrendOnlyDefault: Bool
    /// Worker-config `harmonic_epoch_class` (default false); see
    /// `harmonicEpochClass(taskConfig:)`.
    public let harmonicEpochClassDefault: Bool
    /// Worker-config `ffi_fallback_on_load_fail` / `ffi_fetch_timeout` (python parity).
    public let ffiFallbackOnLoadFail: Bool
    public let ffiFetchTimeout: Double
    private let lastEpochClassLock = NSLock()
    private var lastEpochClass: Bool?
    /// Effective `harmonic_epoch_class` of the last primary task this processor
    /// started (python `_effective_harmonic_epoch_class`), nil until a task has
    /// run. Published in the heartbeat `flag_states`.
    private func recordEpochClass(_ value: Bool) {
        lastEpochClassLock.lock(); defer { lastEpochClassLock.unlock() }
        lastEpochClass = value
    }
    public var lastHarmonicEpochClass: Bool? {
        lastEpochClassLock.lock(); defer { lastEpochClassLock.unlock() }
        return lastEpochClass
    }

    public init(
        api: WorkerAPIClient, cacheDirectory: URL,
        cnnModel: URL? = nil, exominerModel: URL? = nil,
        maximumConcurrency: Int = ProcessInfo.processInfo.activeProcessorCount,
        blsBackend: BLSComputeBackend = .cpu,
        injectionBackend: BLSComputeBackend? = nil,
        gpuEnabled: Bool = true,
        gapFillDetrendOnly: Bool = false,
        harmonicEpochClass: Bool = false,
        ffiFallbackOnLoadFail: Bool = false,
        ffiFetchTimeout: Double = 30
    ) throws {
        self.api = api; self.cacheDirectory = cacheDirectory
        self.gapFillDetrendOnlyDefault = gapFillDetrendOnly
        self.harmonicEpochClassDefault = harmonicEpochClass
        self.ffiFallbackOnLoadFail = ffiFallbackOnLoadFail
        // only a finite 0 < t <= 1 day is accepted (NaN / negative / huge are invalid); else python's 30 s
        if ffiFetchTimeout.isFinite && ffiFetchTimeout > 0 && ffiFetchTimeout <= 86_400 {
            self.ffiFetchTimeout = ffiFetchTimeout
        } else {
            Self.warn("ffi_fetch_timeout=\(ffiFetchTimeout) is not a finite number in (0, 86400] seconds; using 30 s")
            self.ffiFetchTimeout = 30
        }
        self.maximumConcurrency = max(1, maximumConcurrency)
        self.blsBackend = blsBackend
        self.injectionBackend = injectionBackend ?? blsBackend
        try FileManager.default.createDirectory(
            at: cacheDirectory, withIntermediateDirectories: true
        )
        if let cnnModel {
            guard FileManager.default.fileExists(atPath: cnnModel.path) else {
                throw PrimaryTaskProcessorError.missingRequiredModel(cnnModel.path)
            }
            cnn = try CoreMLFFIClassifier(
                modelURL: cnnModel,
                computeUnits: gpuEnabled ? .all : .cpuOnly
            )
        } else { cnn = nil }
        if let exominerModel {
            guard FileManager.default.fileExists(atPath: exominerModel.path) else {
                throw PrimaryTaskProcessorError.missingRequiredModel(exominerModel.path)
            }
            exominer = try CoreMLExoMinerFFI(
                modelURL: exominerModel,
                calibration: PiecewiseLinearCalibration(
                    x: [0, 0.38930313157914914, 1], y: [0, 0.5, 1]
                ),
                computeUnits: gpuEnabled ? .all : .cpuOnly
            )
        } else { exominer = nil }
    }

    public static func multiPlanetOptions(
        config: [String: JSONValue]
    ) -> MultiPlanetOptions {
        var options = MultiPlanetOptions()
        options.maximumPlanets = config.int("max_planets", default: 5)
        options.minimumSNR = config.double("snr_threshold", default: 7)
        options.periodMinimum = config.double("period_min", default: 0.5)
        options.periodMaximum = config.double(
            "period_max", default: config.double("period_max_legacy", default: 50)
        )
        options.periodCount = config.int("n_periods", default: 50_000)
        options.protectedSlotsEnabled = config.bool(
            "medium_period_protected_slot", default: true
        )
        options.protectedSlotMinimumSNR = config.double(
            "medium_slot_min_snr", default: 12
        )
        if let configuredRanges = config.doubleRanges("medium_slot_ranges") {
            options.protectedSlotRanges = configuredRanges.compactMap {
                $0.count >= 2 && $0[1] > $0[0] ? $0[0]...$0[1] : nil
            }
        } else if config["medium_slot_min_days"] != nil ||
                    config["medium_slot_max_days"] != nil {
            let minimum = config.double("medium_slot_min_days", default: 10)
            let maximum = config.double("medium_slot_max_days", default: 50)
            options.protectedSlotRanges = maximum > minimum ? [minimum...maximum] : []
        }
        options.harmonicLongPeriodRecovery = config.bool(
            "harmonic_long_period_recovery", default: false
        )
        options.harmonicLongPeriodMinimumDays = config.double(
            "harmonic_long_period_min_days", default: 10
        )
        options.harmonicLongSNRFactor = config.double(
            "harmonic_long_snr_factor", default: 0.5
        )
        options.harmonicEpochClass = config.bool("harmonic_epoch_class", default: false)
        if config.bool("wotan_long_sweep", default: true) {
            var long = LongPeriodSweepOptions()
            long.baselineThresholdDays = config.double(
                "wotan_long_threshold_days", default: 100
            )
            long.windowsDays = config.doubles(
                "wotan_long_windows_days", default: [2, 4]
            )
            long.periodMinimumDays = config.double(
                "wotan_long_period_min_days", default: 30
            )
            long.legacyPeriodMaximumDays = options.periodMaximum
            long.periodMaximumCapDays = config.double("period_max_cap", default: 200)
            long.minimumTransits = config.double("period_max_min_transits", default: 2.5)
            long.basePeriodCount = config.int("wotan_long_n_periods", default: 50_000)
            if config["wotan_long_n_periods"] != nil {
                long.periodCountScaleReferenceMaximumDays = nil
            }
            long.minimumSNR = config.double(
                "wotan_long_snr_threshold", default: options.minimumSNR
            )
            options.longPeriodSweep = long
        } else { options.longPeriodSweep = nil }
        return options
    }

    /// `gap_fill_detrend_only` precedence: task-config key if present, else
    /// the worker config, else false (python `_gap_fill_detrend_only_enabled`).
    /// Only a real JSON bool counts: a non-bool task value ("true", 1, null)
    /// is ignored with a warning (python / Rust parity).
    public func gapFillDetrendOnly(taskConfig: [String: JSONValue]?) -> Bool {
        guard let raw = taskConfig?["gap_fill_detrend_only"] else { return gapFillDetrendOnlyDefault }
        if case let .bool(value) = raw { return value }
        FileHandle.standardError.write(Data(
            ("PrimaryTaskProcessor: task config gap_fill_detrend_only=\(raw) is not a JSON bool; "
             + "ignored (worker value \(gapFillDetrendOnlyDefault))\n").utf8
        ))
        return gapFillDetrendOnlyDefault
    }

    /// `harmonic_epoch_class` (alias phase 2) precedence, python `process_task`:
    /// a JSON bool in the task config wins, else the worker config, else false.
    /// A non-bool task value (string, number, null) is a protocol error: warn
    /// and use the worker value, never truthy-coerce "false".
    public func harmonicEpochClass(taskConfig: [String: JSONValue]?) -> Bool {
        guard let raw = taskConfig?["harmonic_epoch_class"] else { return harmonicEpochClassDefault }
        if case let .bool(value) = raw { return value }
        Self.warn("task config harmonic_epoch_class=\(raw) is not a bool; ignored, using local config")
        return harmonicEpochClassDefault
    }

    /// Search options for one task: the config-derived options with
    /// `harmonicEpochClass` resolved by `harmonicEpochClass(taskConfig:)`; also
    /// returns the effective flag (reported in the result).
    public func multiPlanetOptions(for taskConfig: [String: JSONValue]) -> (MultiPlanetOptions, Bool) {
        var options = Self.multiPlanetOptions(config: taskConfig)
        let epochClass = harmonicEpochClass(taskConfig: taskConfig)
        options.harmonicEpochClass = epochClass
        return (options, epochClass)
    }

    /// Primary-task light curve (`iterative_detrend` and
    /// `gap_fill_detrend_only` resolved from the task / worker config). The
    /// FPP-backfill paths call `loadProcessedLightCurve(ticID:workerID:)`,
    /// which uses the worker value, so for the same flag both see the same
    /// light curve.
    public func loadPrimaryLightCurve(_ task: WorkerTask, workerID: String) async throws -> LoadedStar {
        try await loadProcessedLightCurve(
            ticID: task.ticID, workerID: workerID,
            iterativeDetrend: task.config.bool("iterative_detrend", default: true),
            gapFillDetrendOnly: gapFillDetrendOnly(taskConfig: task.config)
        )
    }

    public static func injectionFlux(from processed: ProcessedLightCurve) -> [Double] {
        processed.flatFlux
    }

    public func process(
        _ task: WorkerTask, workerID: String
    ) async throws -> PrimaryResultRequest {
        let started = ContinuousClock.now
        // Alias phase 2: task bool > worker config > off; reported in the
        // result and, from the moment the task starts (as in python), in the
        // heartbeat flag_states.
        let (options, epochClass) = multiPlanetOptions(for: task.config)
        recordEpochClass(epochClass)
        do {
            try Task.checkCancellation()
            // gap_fill_detrend_only (task > worker config > off): gap-fill
            // points feed the detrender only; `processed` then holds real
            // points only and the re-detrends use `detrendFrame`.
            let loaded = try await loadPrimaryLightCurve(task, workerID: workerID)
            let metadata = loaded.metadata
            let processed = loaded.processed
            let source = loaded.source
            let search = try MultiPlanetSearcher.search(
                time: processed.time, rawFlux: processed.rawFlux,
                initialDetrendedFlux: processed.flatFlux,
                stellarRadiusSolar: metadata.stellarRadius,
                stellarMassSolar: metadata.stellarMass,
                source: source, options: options, maxConcurrency: maximumConcurrency,
                blsBackend: blsBackend, detrendFrame: processed.detrendFrame
            )
            var candidates = search.candidates
            let vetting = try CandidateVetter.vetAll(
                candidates: candidates, time: processed.time, flux: processed.rawFlux
            )
            var cnnScores = [Double?](repeating: nil, count: candidates.count)
            for index in candidates.indices {
                let candidate = candidates[index]
                // Sparse view -> no CNN score (nil), task continues;
                // ExoMiner-FFI scores it 0.0 (uncalibrated). Same as python
                // cnn_ffi.py / exominer_ffi_inference.py on ViewError. No FPP
                // follows on this worker (no in-task FPP; backfill SQL excludes
                // cnn nil + em_ffi 0.0) -- open item, see FFIViewGenerator
                // .makeCNNViewsIfDense.
                if let cnn, let views = try FFIViewGenerator.makeCNNViewsIfDense(
                    time: processed.time, flux: processed.flatFlux,
                    period: candidate.periodDays, t0: candidate.t0,
                    duration: candidate.durationDays
                ) {
                    cnnScores[index] = try cnn.classify(views: views).probability
                }
                if let exominer {
                    let features: ExoMinerFFIFeatures?
                    do {
                        features = try ExoMinerFFIFeatureBuilder.make(
                            time: processed.time, flux: processed.flatFlux,
                            period: candidate.periodDays, t0: candidate.t0,
                            duration: candidate.durationDays, depthPPM: candidate.depthPPM,
                            snr: candidate.snr,
                            stellar: FFIStellarMetadata(
                                tmag: metadata.tmag,
                                teff: metadata.effectiveTemperature,
                                logg: metadata.logg,
                                radius: metadata.stellarRadius,
                                mass: metadata.stellarMass
                            )
                        )
                    } catch let error as FFIViewError where error.isSparse {
                        features = nil
                    }
                    // OPEN ITEM (exonit docs/reviews/2026-10-15-cnn-exact-zero-
                    // scores.md §7.3): the sparse-view 0.0 is a sentinel
                    // indistinguishable from a real low score; mirrored from
                    // python for parity until python decides on nil (it feeds
                    // the backfill OR-gate and the verdict).
                    if let features {
                        candidates[index].exominerFFIScore =
                            try exominer.classify(features: features).probability
                    } else {
                        candidates[index].exominerFFIScore = 0
                    }
                    candidates[index].exominerFFIModelVersion = "phase3_v6_ensemble_cal"
                }
            }
            var completeness: CompletenessResult?
            let runInjection = task.config.bool(
                "run_injection",
                default: task.config.bool("run_injection_recovery", default: true)
            )
            if runInjection {
                var injection = InjectionRecoveryOptions()
                injection.stellarRadiusSolar = metadata.stellarRadius ?? 1
                injection.stellarMassSolar = metadata.stellarMass
                injection.injectionsPerCell = task.config.int("n_injections_per_cell", default: 10)
                injection.minimumSNR = options.minimumSNR
                injection.periodCount = task.config.int("injection_n_periods", default: 5_000)
                injection.binCount = task.config.int("injection_n_bins", default: 150)
                injection.durationStepCount = task.config.int("injection_n_dur_steps", default: 10)
                injection.maximumSeconds = task.config.double("injection_max_seconds", default: 300)
                completeness = try InjectionRecovery.run(
                    time: processed.time, flux: Self.injectionFlux(from: processed),
                    options: injection, maxConcurrency: maximumConcurrency,
                    backend: injectionBackend
                )
            }
            return PrimaryResultRequest(
                taskID: task.taskID, ticID: task.ticID,
                candidates: candidates, vettingResults: vetting,
                cnnScores: cnnScores, completeness: completeness,
                processingTimeSeconds: elapsed(started),
                gpuUsed: BLSMetal.isAvailable && (
                    blsBackend != .cpu || (runInjection && injectionBackend != .cpu)
                ),
                sectorCount: loaded.sectorCount, pointCount: processed.time.count,
                timeSpanDays: (processed.time.last ?? 0) - (processed.time.first ?? 0),
                ra: metadata.ra, dec: metadata.dec, tmag: metadata.tmag,
                timings: [
                    "swift_native": .bool(true),
                    "redetrend_failures": .integer(Int64(search.redetrendFailureCount)),
                    "unphysical_rejections": .integer(Int64(search.rejectedUnphysicalCount)),
                ], lightCurveLoadRecovered: loaded.recovered,
                harmonicEpochClass: epochClass
            )
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            return PrimaryResultRequest(
                taskID: task.taskID, ticID: task.ticID, status: "failed",
                error: String(describing: error), processingTimeSeconds: elapsed(started)
            )
        }
    }

    public func loadProcessedLightCurve(
        ticID: String, workerID: String, iterativeDetrend: Bool = true,
        gapFillDetrendOnly: Bool? = nil
    ) async throws -> LoadedStar {
        // python `_load_star`: FFI fallback (flag ON) at every point python has one:
        // metadata unavailable, no FITS listed, no FITS obtainable, unreadable FITS.
        let metadata: StarMetadata
        do {
            metadata = try await api.fetchMetadata(ticID: ticID, workerID: workerID)
        } catch {
            if error is CancellationError { throw error }
            if let recovered = try await tryFFIFallback(ticID: ticID, workerID: workerID, reason: "metadata") {
                return recovered
            }
            throw error
        }
        if ffiFallbackOnLoadFail && metadata.fitsFiles.isEmpty {
            guard let recovered = try await tryFFIFallback(ticID: ticID, workerID: workerID, reason: "no_fits_files") else {
                throw PrimaryTaskProcessorError.loadFailed(ticID)
            }
            return recovered
        }
        let loadedCurves: LoadedCurves
        do {
            loadedCurves = try await loadCurves(ticID: ticID, metadata: metadata)
        } catch let error as FITSError {
            if let recovered = try await tryFFIFallback(ticID: ticID, workerID: workerID, reason: "lightkurve_exception") {
                return recovered
            }
            throw error
        }
        let curves = loadedCurves.curves
        if curves.isEmpty && !metadata.fitsFiles.isEmpty {
            // Listed files, none obtainable: python logs "No FITS files downloaded",
            // then FFI fallback (flag ON) or load_failed.
            Self.warn("No FITS files downloaded for TIC \(ticID)")
            guard let recovered = try await tryFFIFallback(ticID: ticID, workerID: workerID, reason: "fits_download_fail") else {
                throw PrimaryTaskProcessorError.loadFailed(ticID)
            }
            return recovered
        }
        guard !curves.isEmpty else { throw PrimaryTaskProcessorError.noLightCurve(ticID) }
        let processed: ProcessedLightCurve
        do {
            processed = try LightCurvePreprocessor.process(
                curves, iterativeDetrend: iterativeDetrend,
                maxConcurrency: maximumConcurrency,
                fillSeed: GapFillSeed.seed(ticID: ticID, sectors: metadata.sectors),
                gapFillDetrendOnly: gapFillDetrendOnly ?? gapFillDetrendOnlyDefault
            )
        } catch {
            // python wraps read+stitch+clean+detrend in ONE `try ... except Exception`
            // (reason lightkurve_exception), e.g. an empty light curve after cleaning.
            if error is CancellationError { throw error }
            if let recovered = try await tryFFIFallback(ticID: ticID, workerID: workerID, reason: "lightkurve_exception") {
                return recovered
            }
            throw error
        }
        return LoadedStar(
            metadata: metadata, processed: processed,
            source: curves.contains { $0.source == .spoc } ? .spoc : .ffi,
            // python tags the product from the files actually loaded
            fppLCSource: loadedCurves.usedFilenames.isEmpty
                ? "unknown" : FPPV3LoadedLCSource.detect(fitsPaths: loadedCurves.usedFilenames),
            recovered: false,
            sectorCount: loadedCurves.skippedFilenames.isEmpty
                ? processed.sectorCount : metadata.sectors.count
        )
    }

    /// python `_try_ffi_fallback` (spec 2026-05-08-ffi-fallback-on-lc-load-failure):
    /// last resort, only with the flag ON. Fetches `/api/ffi/{tic}` (no retries,
    /// `ffiFetchTimeout`), drops non-finite samples, requires >= 100 finite points,
    /// then builds the light curve EXACTLY like python: NO gap fill and NO detrend
    /// (`flat = raw = flux`), minimal metadata (tic_id, data_source "ffi", ra/dec/
    /// tmag/sectors from a best-effort 5 s metadata refetch; stellar parameters
    /// stay absent), product tag "unknown", `n_sectors = len(sectors) or 1`,
    /// `lc_load_recovered = true`. nil = the fallback also failed.
    private func tryFFIFallback(ticID: String, workerID: String, reason: String) async throws -> LoadedStar? {
        guard ffiFallbackOnLoadFail else { return nil }
        guard let fetched = await api.fetchFFIForFallback(ticID: ticID, timeout: ffiFetchTimeout) else {
            Self.warn("TIC \(ticID): SPOC load failed (\(reason)) -> FFI fallback also failed (server 404)")
            return nil
        }
        try Task.checkCancellation()
        var time: [Double] = [], flux: [Double] = []
        for index in fetched.time.indices where fetched.time[index].isFinite && fetched.flux[index].isFinite {
            time.append(fetched.time[index]); flux.append(fetched.flux[index])
        }
        guard time.count >= 100 else {
            Self.warn("TIC \(ticID): FFI fallback rejected - only \(time.count) finite points")
            return nil
        }
        var ra: Double?, dec: Double?, tmag: Double?, sectors: [Int] = []
        if let refetched = try? await api.fetchMetadata(ticID: ticID, workerID: workerID, timeout: 5) {
            ra = refetched.ra; dec = refetched.dec; tmag = refetched.tmag; sectors = refetched.sectors
        }
        let sectorCount = sectors.isEmpty ? 1 : sectors.count
        var maxGap = 0.0
        for index in time.indices.dropFirst() { maxGap = max(maxGap, time[index] - time[index - 1]) }
        Self.warn("TIC \(ticID): SPOC load failed (\(reason)) -> FFI fallback OK (\(sectorCount) sectors, \(time.count) points, source=ffi)")
        return LoadedStar(
            metadata: StarMetadata(
                ticID: ticID, sectors: sectors, ra: ra, dec: dec, tmag: tmag, dataSource: "ffi"
            ),
            processed: ProcessedLightCurve(
                time: time, flatFlux: flux, rawFlux: flux, sectorCount: sectorCount, maxGapDays: maxGap
            ),
            source: .ffi, fppLCSource: "unknown", recovered: true, sectorCount: sectorCount
        )
    }

    /// python `fits_cache_helper.ensure_fits_present` attempt count.
    static let fitsDownloadAttempts = 3

    static func warn(_ message: String) {
        FileHandle.standardError.write(Data("PrimaryTaskProcessor: \(message)\n".utf8))
    }

    /// Mirrors python `ensure_fits_present`: a cached file is trusted only
    /// after validation (invalid -> purged); then up to 3 download+validate
    /// attempts, where ANY failure (HTTP 404/5xx, transport error, invalid
    /// file) just burns an attempt. Returns false when all attempts failed
    /// (nothing left on disk) so the caller skips the file.
    private func ensureFITSPresent(filename: String) async throws -> Bool {
        let destination = cacheDirectory.appendingPathComponent(filename)
        if FileManager.default.fileExists(atPath: destination.path) {
            guard let reason = FITSReader.invalidReason(at: destination) else { return true }
            Self.warn("Purging invalid cached FITS \(filename): \(reason)")
            try? FileManager.default.removeItem(at: destination)
        }
        for attempt in 1...Self.fitsDownloadAttempts {
            do {
                let data = try await api.downloadFITS(filename: filename)
                try data.write(to: destination, options: [.atomic])
            } catch {
                if error is CancellationError || Task.isCancelled { throw CancellationError() }
                Self.warn(
                    "Download attempt \(attempt)/\(Self.fitsDownloadAttempts) for \(filename) "
                    + "raised \(error)"
                )
                try? FileManager.default.removeItem(at: destination)
                continue
            }
            guard let reason = FITSReader.invalidReason(at: destination) else { return true }
            Self.warn(
                "Invalid FITS after download (attempt \(attempt)/\(Self.fitsDownloadAttempts)) "
                + "\(filename): \(reason)"
            )
            try? FileManager.default.removeItem(at: destination)
        }
        Self.warn("Giving up on \(filename) after \(Self.fitsDownloadAttempts) attempts")
        return false
    }

    private func loadCurves(ticID: String, metadata: StarMetadata) async throws -> LoadedCurves {
        if !metadata.fitsFiles.isEmpty {
            var result = LoadedCurves(curves: [], usedFilenames: [], skippedFilenames: [])
            for filename in metadata.fitsFiles {
                guard try await ensureFITSPresent(filename: filename) else {
                    result.skippedFilenames.append(filename)
                    continue
                }
                let destination = cacheDirectory.appendingPathComponent(filename)
                result.curves.append(try FITSReader.readLightCurve(at: destination))
                result.usedFilenames.append(filename)
            }
            return result
        }
        guard let ffi = try await api.fetchFFI(ticID: ticID) else {
            return LoadedCurves(curves: [], usedFilenames: [], skippedFilenames: [])
        }
        let curves = ffi.sectors.map { sector in
            LightCurveData(
                time: sector.time, flux: sector.flux, rawRowCount: sector.time.count,
                source: sector.source.lowercased().contains("tglc") ? .tglc : .qlp
            )
        }
        return LoadedCurves(curves: curves, usedFilenames: [], skippedFilenames: [])
    }

    private func elapsed(_ start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: .now)
        return Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
    }
}

private extension Dictionary where Key == String, Value == JSONValue {
    func bool(_ key: String, default fallback: Bool) -> Bool {
        if case let .bool(value)? = self[key] { return value }; return fallback
    }
    func double(_ key: String, default fallback: Double) -> Double {
        switch self[key] {
        case let .number(value): return value
        case let .integer(value): return Double(value)
        default: return fallback
        }
    }
    func int(_ key: String, default fallback: Int) -> Int {
        switch self[key] {
        case let .integer(value): return Int(clamping: value)
        case let .number(value): return Int(value)
        default: return fallback
        }
    }
    func doubleRanges(_ key: String) -> [[Double]]? {
        guard case let .array(rows)? = self[key] else { return nil }
        return rows.compactMap { row in
            guard case let .array(values) = row else { return nil }
            return values.compactMap {
                switch $0 {
                case let .number(value): return value
                case let .integer(value): return Double(value)
                default: return nil
                }
            }
        }
    }

    func doubles(_ key: String, default fallback: [Double]) -> [Double] {
        guard case let .array(values)? = self[key] else { return fallback }
        let result = values.compactMap { value -> Double? in
            switch value {
            case let .number(number): return number
            case let .integer(integer): return Double(integer)
            default: return nil
            }
        }
        return result.isEmpty ? fallback : result
    }
}
