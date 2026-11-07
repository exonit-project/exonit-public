import Foundation

/// Native Swift FPP backfill path.  It deliberately reuses the primary
/// processor for FITS/FFI loading (through `FPPLightCurveLoading`, so this
/// file is platform neutral) and keeps the API/sidecar boundary outside the
/// scientific calculation.
public final class NativeFPPTaskProcessor: @unchecked Sendable {
    private let primary: any FPPLightCurveLoading
    private let api: WorkerAPIClient
    private let cacheDirectory: URL

    public init(primary: any FPPLightCurveLoading, api: WorkerAPIClient, cacheDirectory: URL) {
        self.primary = primary
        self.api = api
        self.cacheDirectory = cacheDirectory
    }

    public func processFPP(
        _ task: BackfillTask, workerID: String
    ) async throws -> FPPTaskOutcome {
        do {
            let result = try await calculateFPP(task, workerID: workerID)
            try Task.checkCancellation()
            return .result(result)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            if let reason = FPPV3ReleasePolicy.unavailableReason(for: error) {
                print("[telemetry] swift-native fpp tic=\(task.ticID) status=release reason=\(reason)")
                return .release(FPPReleaseRequest(
                    candidateID: task.candidateID, claimID: task.fppClaimID,
                    ticID: task.ticID, reason: reason))
            }
            if Task.isCancelled { throw CancellationError() }
            print(
                "[telemetry] swift-native fpp tic=\(task.ticID) status=failed "
                    + "error=\(String(describing: error))"
            )
            return .result(Self.failedResult(task, skipReason: FPPV3SkipPolicy.skipReason(for: error)))
        }
    }

    private func calculateFPP(
        _ task: BackfillTask, workerID: String
    ) async throws -> FPPResultRequest {
        try FPPV3LikelihoodRouter.throwIfLatched()
        if let reason = FPPV3ReleasePolicy.preflightReason(
            population: task.population, crosscheck: task.populationCrosscheck,
            populationSHA256: task.populationSHA256, populationVersion: task.populationVersion
        ) {
            throw FPPV3Error(reason, "v3 population preflight is unavailable")
        }
        guard let population = task.population else {
            throw FPPV3Error("population_missing", "background population is missing")
        }
        guard let populationSHA256 = task.populationSHA256, !populationSHA256.isEmpty,
              let populationVersion = task.populationVersion, !populationVersion.isEmpty else {
            throw FPPV3Error("population_invalid", "population provenance is incomplete")
        }
        let loaded = try await primary.loadFPPLightCurve(ticID: task.ticID, workerID: workerID)
        let nearby = try await api.fetchNearby(ticID: task.ticID)
        let nearbyRaw = Self.rawNearby(nearby)
        let ephemRefineOn = try FPPV3Dispatch.parseEphemRefineSwitch()
        let ephemRes = FPPV3Ephem.fppEphemeris(
            time: loaded.time, flux: loaded.flux,
            period: task.periodDays, t0: task.t0, duration: task.durationDays,
            convention: .boxStart, refineEnabled: ephemRefineOn, cadenceS: task.cadenceSeconds
        )
        var input = FPPV3CalcInputs(
            ticID: task.ticID, period: ephemRes.pUse, t0: ephemRes.t0Use,
            durationDays: task.durationDays, depthPPM: task.depthPPM,
            time: loaded.time, flux: loaded.flux, nearby: nearbyRaw,
            geometry: task.geometry, population: population,
            populationSHA256: populationSHA256, populationVersion: populationVersion,
            populationCrosscheck: task.populationCrosscheck, cadenceSeconds: task.cadenceSeconds,
            lcSource: loaded.lcSource, tglcNearEdge: task.tglcNearEdge
        )
        if ephemRes.provenance != nil {
            input.seedEphemeris = (period: task.periodDays, t0: task.t0)
        }
        let selection = FPPV3Dispatch.switchSet(
            parityValue: ProcessInfo.processInfo.environment["FPP_PARITY"]
        )
        if FPPV3LikelihoodRouter.backend != nil && !selection.productionFold {
            throw FPPV3GPUFailure.unsupportedSwitchSet
        }
        let seed = FPPRNG.ticSeed(ticID: task.ticID, period: task.periodDays, t0: task.t0)
        let result: FPPV3CalcResult
        do {
            result = try FPPV3Calculation.calcFPPBySourceV3(
                input, draws: 100_000, seed: seed,
                productionFold: selection.productionFold, switches: selection.switches
            )
        } catch {
            if let failure = FPPV3LikelihoodRouter.latchedFailure { throw failure }   // the GPU failure wins over any downstream symptom
            throw error
        }
        try FPPV3LikelihoodRouter.throwIfLatched()
        return FPPV3ResultMapper.map(
            task: task, result: result, loadedLCSource: loaded.lcSource,
            ephemProv: ephemRes.provenance, refineFailed: ephemRes.refineFailed
        )
    }

    private static func jsonNumber(_ value: Double?) -> JSONValue {
        value.map(JSONValue.number) ?? .null
    }

    private static func rawNearby(_ nearby: NearbyResponse) -> [String: JSONValue] {
        let stars: [JSONValue] = nearby.stars.map { star in
            .object([
                "ID": .string(star.id), "ra": .number(star.ra),
                "dec": .number(star.dec), "mass": jsonNumber(star.mass),
                "rad": jsonNumber(star.radius), "Teff": jsonNumber(star.effectiveTemperature),
                "Tmag": jsonNumber(star.tessMagnitude), "plx": jsonNumber(star.parallax),
            ].merging(massFallbackFields(star)) { current, _ in current })
        }
        return [
            "stars": .array(stars), "gal_b": .number(nearby.galacticLatitude),
        ]
    }

    /// The stellar-mass fallback provenance of `/api/nearby` (task 33 item 3), passed through as served.
    private static func massFallbackFields(_ star: NearbyStar) -> [String: JSONValue] {
        var out = [String: JSONValue]()
        if let v = star.massSource { out["mass_source"] = v }
        if let v = star.giantHost { out["giant_host"] = v }
        if let v = star.massUnavailableReason { out["mass_unavailable_reason"] = v }
        return out
    }

    private static func candidateObject(_ task: BackfillTask) -> [String: JSONValue] {
        [
            "candidate_id": .integer(task.candidateID), "tic_id": .string(task.ticID),
            "candidate_num": .integer(Int64(task.candidateNumber)),
            "period_days": .number(task.periodDays), "t0": .number(task.t0),
            "duration_days": .number(task.durationDays),
            "depth_ppm": .number(task.depthPPM), "snr": .number(task.snr),
            "n_sectors": .integer(Int64(task.sectorCount)),
        ]
    }

    /// The `-1` / `failed` body of Python's worker (r2 worker.py :4058-4072), with `skip_reason` always present:
    /// the typed reason, or JSON null when Python has none.
    static func failedResult(_ task: BackfillTask, skipReason: String?) -> FPPResultRequest {
        var request = FPPResultRequest(
            candidateID: task.candidateID, ticID: task.ticID,
            fpp: -1, nfpp: -1, probabilityPlanet: nil,
            verdict: "failed", candidate: candidateObject(task)
        )
        request.skipReason = skipReason.map(JSONValue.string) ?? .null
        return request
    }
}

public enum NativeFPPError: Error, CustomStringConvertible, Sendable {
    case calculationUnavailable(String)
    case missingAPI

    public var description: String {
        switch self {
        case let .calculationUnavailable(tic): return "native FPP calculation unavailable for TIC \(tic)"
        case .missingAPI: return "native FPP processor requires WorkerAPIClient"
        }
    }
}
