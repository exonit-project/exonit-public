import Foundation

/// The light curve a native FPP calculation needs from the primary processor:
/// the processed (flattened) series and the product label of the files that
/// were actually loaded (an FFI fallback is `unknown`).
public struct FPPLoadedLightCurve: Sendable {
    public let time: [Double]
    public let flux: [Double]
    public let lcSource: String

    public init(time: [Double], flux: [Double], lcSource: String) {
        self.time = time; self.flux = flux; self.lcSource = lcSource
    }
}

/// Loads the light curve of an FPP task. `PrimaryTaskProcessor` is the
/// production implementation (FITS/FFI through the server, Darwin only);
/// the protocol keeps `NativeFPPTaskProcessor` platform neutral so the
/// release/no-score behaviour is tested on Linux too.
public protocol FPPLightCurveLoading: Sendable {
    func loadFPPLightCurve(ticID: String, workerID: String) async throws -> FPPLoadedLightCurve
}

/// The GPU capability of a Swift worker that claims FPP tasks. Platform neutral:
/// the Metal implementation lives outside this file. Exactly two requirements:
/// FPP and LoRA share the M4s (decision Q4=B), so there is NO co-tenant,
/// MLX or LoRA probe here — a GPU error under LoRA contention is a failure.
public protocol FPPV3GPUBackend: Sendable {
    /// Device present, allowed device class. Throw FPPV3GPUFailure (any other error is treated as .noDevice).
    func deviceReady() async throws
    /// Canary against the CPU reference. Throw .canaryFailed on mismatch; any other FPPV3GPUFailure if it cannot run.
    func runCanary() async throws
}

/// Decides whether this worker may claim an FPP task, BEFORE `api.getFPPTask`.
/// A GPU that is absent or failing its canary latches the gate down for the rest
/// of the process: no further claims, no CPU compute, one log line. Only a
/// restart re-evaluates. A CPU-declared worker (backend == nil) is unchanged.
public actor FPPClaimGate {
    public enum Decision: Equatable, Sendable { case claim, refuse(String) }   // refuse carries "unavailable" or "gpu_parity_failed"

    private let backend: (any FPPV3GPUBackend)?
    private var checked = false
    private var down: String?

    /// backend == nil means CPU-declared: decide() is always .claim and noteRelease never latches.
    public init(backend: (any FPPV3GPUBackend)?) {
        self.backend = backend
    }

    /// Called before EVERY claim, serially by the run loop. First call: deviceReady() then runCanary() (once per process).
    /// Any throw latches down with FPPV3ReleasePolicy.gpuReason(for:) (non-FPPV3GPUFailure errors -> "unavailable").
    /// Down: .refuse(reason) without touching the backend again.
    public func decide() async -> Decision {
        if let down { return .refuse(down) }
        guard let backend else { return .claim }
        if !checked {
            checked = true
            do {
                try await backend.deviceReady()
                try await backend.runCanary()
            } catch {
                let reason = (error as? FPPV3GPUFailure).map(FPPV3ReleasePolicy.gpuReason) ?? "unavailable"
                latchDown(reason)
                return .refuse(reason)
            }
        }
        return .claim
    }

    /// Called with the reason of every release the worker delivers. A reason in FPPV3ReleasePolicy.gpuReasons latches down
    /// (no-op when backend == nil). population_* reasons are ignored.
    public func noteRelease(reason: String) {
        guard backend != nil else { return }
        guard FPPV3ReleasePolicy.gpuReasons.contains(reason) else { return }
        latchDown(reason)
    }

    public var downReason: String? { down }

    private func latchDown(_ reason: String) {
        guard down == nil else { return }
        down = reason
        print("[worker] fpp gate down reason=\(reason); no further FPP claims this run")
    }
}

/// The durable result queue a computed (or failed) FPP result goes to.
/// `WorkerResultOutbox` is the production implementation.
public protocol FPPResultEnqueuing: Sendable {
    func enqueueFPP(_ result: FPPResultRequest, workerID: String) async throws
}

/// The worker's single path from an FPP outcome to the server, used by
/// `WorkerRuntime`: a release is POSTed to `/api/worker/fpp_release` and is
/// never enqueued as a result; a result (including a failed `-1` result) is
/// enqueued and never released.
public enum FPPOutcomeDelivery {
    public enum Delivered: Equatable, Sendable {
        case released(FPPReleaseRequest)
        case enqueued(FPPResultRequest)
    }

    public static func deliver(
        _ outcome: FPPTaskOutcome, workerID: String,
        api: WorkerAPIClient, results: any FPPResultEnqueuing
    ) async throws -> Delivered {
        switch outcome {
        case let .release(request):
            _ = try await api.releaseFPPTask(request, workerID: workerID)
            return .released(request)
        case let .result(result):
            try await results.enqueueFPP(result, workerID: workerID)
            return .enqueued(result)
        }
    }
}
