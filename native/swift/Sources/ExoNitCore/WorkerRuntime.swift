import Darwin
import Foundation

public enum WorkerRuntimeError: Error, CustomStringConvertible, Sendable {
    case unsupportedTaskPools([String])
    case missingScienceSidecar
    case canaryNotAuthorized
    case fppMetalStartupFailed(String)

    public var description: String {
        switch self {
        case let .unsupportedTaskPools(pools):
            return "unknown task pools: \(pools.joined(separator: ","))"
        case .missingScienceSidecar:
            return "FPP/ExoMiner pools require the versioned Python science sidecar"
        case .canaryNotAuthorized:
            return "real worker execution requires EXONIT_SWIFT_CANARY=1"
        case let .fppMetalStartupFailed(reason):
            return "fpp_backend=metal startup failed: \(reason); CPU fallback is disabled"
        }
    }
}

public actor WorkerRuntimeState {
    public init() {}
    public private(set) var workerID = ""
    private(set) var paused = false
    private(set) var shutdownRequested = false
    private(set) var debug = false
    public private(set) var currentTIC: String?
    public private(set) var phase: String?
    public private(set) var starsDone = 0
    public private(set) var lastError: String?

    public func registered(_ id: String) { workerID = id }
    public func begin(tic: String, phase: String) { currentTIC = tic; self.phase = phase }
    public func completed() { currentTIC = nil; phase = nil; starsDone += 1; lastError = nil }
    public func released() { currentTIC = nil; phase = nil; lastError = nil }
    public func failed(_ message: String) { currentTIC = nil; phase = nil; lastError = message }
    public func heartbeatFailed(_ message: String) { lastError = message }
    public func requestShutdown() { shutdownRequested = true }
    public func apply(_ command: HeartbeatCommand) {
        switch command {
        case .pause: paused = true
        case .resume: paused = false
        case .shutdown: shutdownRequested = true
        case .debugOn: debug = true
        case .debugOff: debug = false
        case .continue, .reRegister: break
        }
    }
}

public final class WorkerRuntime: @unchecked Sendable {
    public let state = WorkerRuntimeState()
    private let config: WorkerConfig
    private let api: WorkerAPIClient
    private let durableOutbox: DurableOutbox
    private let outbox: WorkerResultOutbox
    private let processor: PrimaryTaskProcessor
    private let backfill: BackfillTaskProcessor?
    private let blsBackend: BLSComputeBackend
    private let injectionBackend: BLSComputeBackend
    private let pools: Set<String>
    private let canaryAuthorized: Bool
    private let nativeFPPEnabled: Bool
    private let fppBackendID: String
    private let fppBackendDevice: String
    private let fppGate: FPPClaimGate
    private let started = ContinuousClock.now

    public init(
        config: WorkerConfig, outboxURL: URL,
        cnnModel: URL? = nil, exominerModel: URL? = nil,
        scienceSidecar: ScienceSidecarConfiguration? = nil,
        canaryAuthorized: Bool = false,
        session: URLSession = .shared,
        fppGPUBackend: (any FPPV3GPUBackend)? = nil
    ) throws {
        let configuredPools = Set(config.taskPools ?? ["primary"])
        let allowed: Set<String> = ["primary", "fpp_backfill", "exominer_backfill"]
        let unsupported = configuredPools.subtracting(allowed).sorted()
        guard unsupported.isEmpty else { throw WorkerRuntimeError.unsupportedTaskPools(unsupported) }
        self.config = config; pools = configuredPools
        self.canaryAuthorized = canaryAuthorized
        self.fppGate = FPPClaimGate(backend: fppGPUBackend)
        self.nativeFPPEnabled = configuredPools.contains("fpp_backfill")
            && ProcessInfo.processInfo.environment["EXONIT_SWIFT_FPP"] == "1"
        let fppBackend = try Self.configureFPPBackend(
            config, nativeFPPEnabled: nativeFPPEnabled,
            fppPoolConfigured: configuredPools.contains("fpp_backfill")
        )
        fppBackendID = fppBackend.id
        fppBackendDevice = fppBackend.device
        api = WorkerAPIClient(baseURL: config.serverURL, token: config.token, session: session)
        durableOutbox = try DurableOutbox(databaseURL: outboxURL)
        outbox = WorkerResultOutbox(outbox: durableOutbox, api: api)
        blsBackend = BLSComputeBackend.configured(gpuEnabled: config.gpuEnabled)
        injectionBackend = BLSComputeBackend.injectionConfigured(gpuEnabled: config.gpuEnabled)
        processor = try PrimaryTaskProcessor(
            api: api, cacheDirectory: URL(fileURLWithPath: config.fitsCacheDirectory),
            cnnModel: cnnModel, exominerModel: exominerModel,
            maximumConcurrency: config.cpuThreads ?? ProcessInfo.processInfo.activeProcessorCount,
            blsBackend: blsBackend,
            injectionBackend: injectionBackend,
            gpuEnabled: config.gpuEnabled,
            gapFillDetrendOnly: config.gapFillDetrendOnly,
            harmonicEpochClass: config.harmonicEpochClass,
            ffiFallbackOnLoadFail: config.ffiFallbackOnLoadFail,
            ffiFetchTimeout: config.ffiFetchTimeout
        )
        if configuredPools.contains("fpp_backfill") || configuredPools.contains("exominer_backfill") {
            let nativeFPP = nativeFPPEnabled
            let fitsCacheDirectory = URL(fileURLWithPath: config.fitsCacheDirectory)
            guard (nativeFPP || scienceSidecar != nil)
                && (!configuredPools.contains("exominer_backfill") || scienceSidecar != nil) else {
                throw WorkerRuntimeError.missingScienceSidecar
            }
            backfill = try BackfillTaskProcessor(
                primary: processor, configuration: scienceSidecar,
                serverURL: config.serverURL, token: config.token,
                workDirectory: fitsCacheDirectory.appendingPathComponent(
                    "swift-sidecar", isDirectory: true
                ),
                nativeFPPEnabled: nativeFPP, api: api,
                nativeFPPCacheDirectory: fitsCacheDirectory
            )
        } else {
            backfill = nil
        }
    }

    private static func configureFPPBackend(
        _ config: WorkerConfig, nativeFPPEnabled: Bool, fppPoolConfigured: Bool
    ) throws -> (id: String, device: String) {
        guard config.fppBackend == .metal else {
            return (
                fppPoolConfigured ? (nativeFPPEnabled ? "swift-native" : "python-sidecar") : "disabled",
                nativeFPPEnabled ? "CPU" : ""
            )
        }
        guard nativeFPPEnabled else {
            throw WorkerRuntimeError.fppMetalStartupFailed(
                "requires task_pools containing fpp_backfill and EXONIT_SWIFT_FPP=1"
            )
        }
        #if canImport(Metal)
        do {
            let backend = try FPPV3MetalBackend()
            let fixture = FPPV3CanaryLoader()
            FPPV3MetalCanary.install(on: backend, fixture: fixture)
            try backend.canary?(backend)
            FPPV3LikelihoodRouter.install(backend)
            return (backend.fppBackendID, backend.backendDevice)
        } catch {
            throw WorkerRuntimeError.fppMetalStartupFailed(String(describing: error))
        }
        #else
        throw WorkerRuntimeError.fppMetalStartupFailed("Metal is unavailable in this build")
        #endif
    }

    public func run() async throws {
        guard canaryAuthorized else { throw WorkerRuntimeError.canaryNotAuthorized }
        // 2026-08-05 fix: DurableOutbox.init already reconciled any leftover
        // 'unknown' rows (pre-fix worker crashes) to 'pending' when it opened
        // this database above — this is no longer fatal, just worth logging.
        if durableOutbox.reconciledUnknownCount > 0 {
            print(
                "[worker] reconciled \(durableOutbox.reconciledUnknownCount) legacy ambiguous-delivery "
                    + "outbox entries to pending at startup (safe retry: server ingestion is idempotent, "
                    + "see exonit/routes/workers.py)"
            )
        }
        if let backfill {
            try await backfill.preflight(
                requireFPP: pools.contains("fpp_backfill"),
                requireExoMiner: pools.contains("exominer_backfill")
            )
        }
        try await register()
        let heartbeat = Task { [self] in await heartbeatLoop() }
        defer { heartbeat.cancel() }
        while !Task.isCancelled, !(await state.shutdownRequested) {
            // Defensive sweep: the .invalidResponse/.malformedResponse paths in
            // WorkerResultOutbox.drain() can still write 'unknown' mid-run.
            // Reconcile it to 'pending' rather than killing the worker.
            let reconciled = try await outbox.reconcileLegacyUnknown()
            if reconciled > 0 {
                print(
                    "[worker] reconciled \(reconciled) ambiguous-delivery outbox entries to pending "
                        + "(safe retry: server ingestion is idempotent)"
                )
            }
            let drain = await outbox.drain()
            if let error = drain.scanError { await state.failed(error) }
            if drain.deliveryUnknown > 0 {
                print(
                    "[worker] WARN: outbox drain reported \(drain.deliveryUnknown) ambiguous "
                        + "delivery/deliveries; reconciled to pending and retried next cycle, not fatal"
                )
                continue
            }
            if try await durableOutbox.count() > 0 {
                try await sleep(seconds: config.pollIntervalSeconds); continue
            }
            if await state.paused {
                try await sleep(seconds: config.pollIntervalSeconds); continue
            }
            let workerID = await state.workerID
            do {
                let didWork = try await processOne(workerID: workerID)
                if !didWork { try await sleep(seconds: config.pollIntervalSeconds) }
            } catch is CancellationError { throw CancellationError() }
            catch {
                await state.failed(String(describing: error))
                try await sleep(seconds: config.pollIntervalSeconds)
            }
        }
    }

    private func processOne(workerID: String) async throws -> Bool {
        if pools.contains("primary"),
           case let .task(task) = try await api.pollTask(workerID: workerID) {
            await state.begin(tic: task.ticID, phase: "primary")
            let result = try await processor.process(task, workerID: workerID)
            try await outbox.enqueuePrimary(result, workerID: workerID)
            try await drainOutboxAfterEnqueue()
            if result.status == "completed" { await state.completed() }
            else { await state.failed(result.error ?? "primary task failed") }
            return true
        }
        if pools.contains("fpp_backfill"), let backfill,
           await fppGate.decide() == .claim,
           let task = try await api.getFPPTask(workerID: workerID) {
            await state.begin(tic: task.ticID, phase: "fpp_backfill")
            let outcome = try await backfill.processFPP(task, workerID: workerID)
            // Note the release reason BEFORE delivering it: the POST in deliver()
            // can throw, and a GPU failure must still latch the gate down.
            if case let .release(request) = outcome { await fppGate.noteRelease(reason: request.reason) }
            switch try await FPPOutcomeDelivery.deliver(
                outcome, workerID: workerID, api: api, results: outbox
            ) {
            case .released:
                await state.released()
            case let .enqueued(result):
                try await drainOutboxAfterEnqueue()
                if result.verdict == "failed" {
                    await state.failed("native FPP processing failed for TIC \(task.ticID)")
                } else {
                    await state.completed()
                }
            }
            return true
        }
        if pools.contains("exominer_backfill"), let backfill,
           let task = try await api.getExoMinerTask(workerID: workerID) {
            await state.begin(tic: task.ticID, phase: "exominer_backfill")
            let result = try await backfill.processExoMiner(task)
            try await outbox.enqueueExoMiner(result, workerID: workerID)
            try await drainOutboxAfterEnqueue()
            await state.completed()
            return true
        }
        return false
    }

    private func drainOutboxAfterEnqueue() async throws {
        let delivered = await outbox.drain()
        if let scanError = delivered.scanError {
            throw WorkerAPIError.malformedResponse("outbox drain: \(scanError)")
        }
        if delivered.deliveryUnknown > 0 {
            // 2026-08-05: no longer fatal — .invalidResponse/.malformedResponse
            // rows get reconciled to 'pending' and retried by the run loop's
            // reconcileLegacyUnknown() sweep, same as any other backoff entry.
            print(
                "[worker] WARN: outbox drain reported \(delivered.deliveryUnknown) ambiguous "
                    + "delivery/deliveries after enqueue; reconciled to pending and retried, not fatal"
            )
        }
    }

    private func register() async throws {
        let cores = config.cpuThreads ?? ProcessInfo.processInfo.activeProcessorCount
        // Primary `.automatic` always executes on CPU (see BLSAccelerated.search), so
        // only the explicit `.metal` opt-in counts as an active Metal primary backend.
        let activeMetal = blsBackend == .metal && BLSMetal.isAvailable
        let activeBLS = BLSComputeBackend.blsCapabilityName(
            backend: blsBackend, metalAvailable: BLSMetal.isAvailable
        )
        let injectionBackendName = BLSComputeBackend.injectionCapabilityName(
            backend: injectionBackend, metalAvailable: BLSMetal.isAvailable
        )
        let workerID = try await api.register(RegisterRequest(
            name: config.workerName,
            capabilities: WorkerCapabilities(
                gpu: BLSMetal.deviceName ?? "none",
                blsBackend: activeBLS,
                blsGPUInfo: activeMetal ? (BLSMetal.deviceName ?? "Metal") : "",
                fppBackend: fppBackendID, cpuCores: cores,
                fppCPUCores: config.fppCPUCores ?? cores,
                hardware: [
                    "runtime": .string("swift-native"),
                    "task_pools": .array(pools.sorted().map(JSONValue.string)),
                    "gpu_configured": .bool(config.gpuEnabled),
                    "gpu_available": .bool(BLSMetal.isAvailable),
                    "bls_backend": .string(activeBLS),
                    "injection_backend": .string(injectionBackendName),
                    "coreml_compute_units": .string(config.gpuEnabled ? "all" : "cpuOnly"),
                    "fpp_backend": .string(fppBackendID),
                    "fpp_backend_device": .string(fppBackendDevice),
                    "exominer_backend": .string(pools.contains("exominer_backfill") ? "python-sidecar" : "disabled")
                ],
            )
        ))
        await state.registered(workerID)
    }

    private func heartbeatLoop() async {
        while !Task.isCancelled, !(await state.shutdownRequested) {
            do {
                try await sleep(seconds: config.heartbeatIntervalSeconds)
                let workerID = await state.workerID
                guard !workerID.isEmpty else { continue }
                let currentTIC = await state.currentTIC
                let command = try await api.heartbeat(HeartbeatRequest(
                    workerID: workerID,
                    status: await state.paused ? "paused" : (currentTIC == nil ? "idle" : "busy"),
                    currentTIC: currentTIC, phase: await state.phase,
                    starsDone: await state.starsDone,
                    uptimeSeconds: uptime(), pid: getpid(),
                    lastError: await state.lastError,
                    flagStates: currentHeartbeatFlagStates()
                ))
                if command == .reRegister { try await register() }
                else { await state.apply(command) }
            } catch is CancellationError { return }
            catch { await state.heartbeatFailed("heartbeat: \(error)") }
        }
    }

    private func sleep(seconds: Double) async throws {
        try await Task.sleep(for: .seconds(seconds))
    }

    /// Worker-config feature flags published in every heartbeat (python
    /// `_resolve_flag_states`); keep it small.
    ///
    /// `harmonic_epoch_class` is dispatched per task by the server (task value >
    /// local config > off): python reports the value the last task ACTUALLY ran
    /// with, never the local config, and omits the key until the first task
    /// has started (`effectiveHarmonicEpochClass` nil).
    public static func heartbeatFlagStates(
        _ config: WorkerConfig, effectiveHarmonicEpochClass: Bool? = nil
    ) -> [String: JSONValue] {
        var states: [String: JSONValue] = [
            "gap_fill_detrend_only": .bool(config.gapFillDetrendOnly),
            "ffi_fallback_on_load_fail": .bool(config.ffiFallbackOnLoadFail),
        ]
        if let effectiveHarmonicEpochClass { states["harmonic_epoch_class"] = .bool(effectiveHarmonicEpochClass) }
        return states
    }

    /// Flag states for the next heartbeat (config flags + the last task's effective alias flag).
    public func currentHeartbeatFlagStates() -> [String: JSONValue] {
        var states = Self.heartbeatFlagStates(
            config, effectiveHarmonicEpochClass: processor.lastHarmonicEpochClass
        )
        states["fpp_backend"] = .string(fppBackendID)
        states["fpp_backend_device"] = .string(fppBackendDevice)
        return states
    }

    /// The primary processor (exposed for contract tests of the heartbeat wiring).
    public var primaryProcessor: PrimaryTaskProcessor { processor }

    private func uptime() -> Double {
        let duration = started.duration(to: .now)
        return Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
    }
}
