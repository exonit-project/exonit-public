import Foundation

enum FPPV3LoadedLCSource {
    static func detect(fitsPaths: [String]) -> String {
        guard !fitsPaths.isEmpty else { return "unknown" }
        var sources = Set<String>()
        for path in fitsPaths {
            let name = URL(fileURLWithPath: path).lastPathComponent.lowercased()
            if name.hasPrefix("hlsp_tglc") { sources.insert("tglc") }
            else if name.hasPrefix("hlsp_qlp") { sources.insert("qlp") }
            else if name.hasSuffix("_lc.fits"), !name.contains("hlsp") { sources.insert("spoc") }
            else { return "unknown" }
        }
        guard sources.count == 1 else { return "mixed" }
        return sources.first ?? "unknown"
    }
}

/// Converts the scientific v3 result into the worker's wire DTO. Kept in a
/// platform-neutral source file so Linux tests exercise the mapper used by the
/// production worker.
enum FPPV3ResultMapper {
    static func map(
        task: BackfillTask, result: FPPV3CalcResult, loadedLCSource: String?,
        ephemProv: FPPV3Ephem.EphemProvenance? = nil, refineFailed: Bool = false
    ) -> FPPResultRequest {
        func finiteJSON(_ value: Double) -> JSONValue {
            value.isFinite ? .number(value) : .null
        }
        func candidateObject(_ task: BackfillTask) -> [String: JSONValue] {
            [
                "candidate_id": .integer(task.candidateID), "tic_id": .string(task.ticID),
                "candidate_num": .integer(Int64(task.candidateNumber)),
                "period_days": .number(task.periodDays), "t0": .number(task.t0),
                "duration_days": .number(task.durationDays),
                "depth_ppm": .number(task.depthPPM), "snr": .number(task.snr),
                "n_sectors": .integer(Int64(task.sectorCount)),
            ]
        }

        var scenarios = [String: JSONValue]()
        for name in result.scenarioOrder {
            guard let scenario = result.scenarios[name] else { continue }
            scenarios[name] = .object([
                "lnZ": finiteJSON(scenario.lnZ), "prob": finiteJSON(scenario.probability),
            ])
        }
        var switches = [String: JSONValue]()
        for (key, value) in result.switches { switches[key] = .string(value) }
        var alternatives = [String: JSONValue]()
        if result.dilutionAlternatives.count > 1 {
            for (corrected, alternative) in result.dilutionAlternatives {
                alternatives[corrected ? "true" : "false"] = .object([
                    "fpp": finiteJSON(alternative.fpp), "nfpp": finiteJSON(alternative.nfpp),
                ])
            }
        }
        let info = result.geometry
        let source = FPPV3Dispatch.normalizeLCSource(loadedLCSource)
        let isTGLC = source == "tglc"
        let edge: Bool? = {
            guard case let .bool(value)? = task.tglcNearEdge else { return nil }
            return value
        }()
        let provenance = FPPV3LikelihoodRouter.backend as? FPPV3BackendProvenance
        var request = FPPResultRequest(
            candidateID: task.candidateID, ticID: task.ticID, fpp: result.fpp, nfpp: result.nfpp,
            probabilityPlanet: nil, verdict: result.verdict, candidate: candidateObject(task),
            fppModelVersion: 3, geometryVersion: 2,
            dilutionConvention: result.dilutionCorrected ? "true" : "false",
            apertureSource: info.apertureSource, depthScale: result.depthScale,
            populationVersion: result.population.version, populationSource: result.population.source,
            populationSHA256: result.population.sha256, populationGenerator: result.population.generator,
            populationMappingVersion: result.population.mappingVersion,
            populationCrosscheck: task.populationCrosscheck,
            nComp: result.population.nComp, nCompFainter: result.population.nCompFainter,
            fppFlags: result.flags, keplerUnconvergedDraws: result.keplerUnconvergedDraws,
            fppScen: scenarios, dilutionAlternatives: alternatives.isEmpty ? nil : .object(alternatives),
            dilutionCorrected: result.dilutionCorrected,
            dilutionUnverified: result.dilutionUnverified,
            exposureGLFallbackShare: result.exposureGLFallbackShare,
            exposureUnconverged: result.exposureUnconverged,
            exposureUnconvergedShare: result.exposureUnconvergedShare,
            fluxratioTarget: info.fluxratioTarget, fppBackend: provenance?.fppBackendID ?? "cpu-swift", fppSeed: result.seed,
            fppSwitches: switches, galacticLatitude: result.galacticLatitude,
            geometryCoverage: info.geometryCoverage, geometryKind: info.geometryKind,
            geometryMode: info.geometryMode, geometryModelVersion: 2,
            geometryNSectors: info.geometryNSectors, geometryNTaskSectors: info.geometryNTaskSectors,
            geometryPath: info.geometryPath, geometryPathEmulated: info.geometryPathEmulated,
            geometrySource: info.geometrySource, highCrowding: result.highCrowding,
            lcSource: source, nNearby: result.nearbyCount,
            nViableNeighbours: info.nViableNeighbours,
            populationCrosscheckStatus: result.populationCrosscheckStatus,
            sigmaFitIllConditioned: result.sigmaFitIllConditioned, sigmaSource: result.sigmaSource,
            tglcNearEdge: isTGLC ? (task.tglcNearEdge ?? .null) : nil, trilegalUsed: false,
            nNodes: info.nNodes, nViabilityWeighted: info.nViabilityWeighted,
            viabilityMassSkipped: info.viabilityMassSkipped, geometryNEpochs: info.geometryNEpochs,
            geometrySigmaCal: info.geometrySigmaCal, tglcNearEdgeUnknown: isTGLC && edge == nil,
            cadenceSeconds: result.cadenceSeconds,
            fppPlatform: FPPV3Provenance.platform, fppLibm: FPPV3Provenance.libm,
            fppBuildProfile: FPPV3Provenance.buildProfile
        )
        request.fppBackendDevice = provenance?.backendDevice
        request.gpuKernelVersion = provenance?.gpuKernelVersion
        // step-7 MC fields, exactly as worker._fpp_result_model_fields maps them (task 33 item 2.8)
        FPPV3MCWire.apply(&request, FPPV3MCWire.Input(result: result))
        if let prov = ephemProv {
            request.ephemRefined = prov.ephemRefined
            request.ephemStatus = prov.ephemStatus
            request.ephemT0Convention = prov.ephemT0Convention
            request.ephemP = prov.ephemP
            request.ephemT0 = prov.ephemT0
            request.ephemShiftH = prov.ephemShiftH
            if let refine = prov.ephemRefine {
                var refDict = [String: JSONValue]()
                refDict["reason"] = refine.reason.map(JSONValue.string) ?? .null
                refDict["P_in"] = .number(refine.pIn)
                refDict["t0_in"] = .number(refine.t0In)
                refDict["t_ref_in"] = .number(refine.tRefIn)
                refDict["cadence_s"] = refine.cadenceS.map(JSONValue.number) ?? .null
                refDict["n_req"] = refine.nReq.map { .integer(Int64($0)) } ?? .null
                refDict["n_orbits"] = refine.nOrbits.map(JSONValue.number) ?? .null
                refDict["n_transits"] = .integer(Int64(refine.nTransits))
                refDict["p_searched"] = .bool(refine.pSearched)
                refDict["p_free_accepted"] = .bool(refine.pFreeAccepted)
                refDict["delta_chi2_p"] = refine.deltaChi2P.map(JSONValue.number) ?? .null
                refDict["dchi2_p_fixed"] = refine.dchi2PFixed.map(JSONValue.number) ?? .null
                refDict["dchi2_p_free"] = refine.dchi2PFree.map(JSONValue.number) ?? .null
                refDict["p_fixed"] = refine.pFixed.map(JSONValue.bool) ?? .null
                refDict["dP"] = .number(refine.dp)
                refDict["max_offset_h"] = .number(refine.maxOffsetH)
                refDict["dchi2_initial"] = refine.dchi2Initial.map(JSONValue.number) ?? .null
                refDict["dchi2_refined"] = refine.dchi2Refined.map(JSONValue.number) ?? .null
                refDict["delta_chi2"] = refine.deltaChi2.map(JSONValue.number) ?? .null
                refDict["sigma"] = refine.sigma.map(JSONValue.number) ?? .null
                refDict["t14_factor"] = refine.t14Factor.map(JSONValue.number) ?? .null
                refDict["ingress"] = refine.ingress.map(JSONValue.number) ?? .null
                refDict["edge_P"] = .bool(refine.edgeP)
                refDict["edge_t0"] = .bool(refine.edgeT0)
                refDict["dP_half"] = refine.dpHalf.map(JSONValue.number) ?? .null
                refDict["dt_half"] = refine.dtHalf.map(JSONValue.number) ?? .null
                request.ephemRefine = .object(refDict)
            }
            if refineFailed {
                var flags = request.fppFlags ?? []
                flags.append("ephem_refine_failed")
                request.fppFlags = flags
                request.ephemRefineFailed = true
                if request.verdict == "validated" {
                    request.verdict = "candidate"
                }
            }
        }
        return request
    }
}
