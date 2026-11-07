import Foundation

public enum JSONValue: Codable, Equatable, Sendable {
    case null
    case bool(Bool)
    case integer(Int64)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null }
        else if let value = try? container.decode(Bool.self) { self = .bool(value) }
        else if let value = try? container.decode(Int64.self) { self = .integer(value) }
        else if let value = try? container.decode(Double.self) { self = .number(value) }
        else if let value = try? container.decode(String.self) { self = .string(value) }
        else if let value = try? container.decode([JSONValue].self) { self = .array(value) }
        else if let value = try? container.decode([String: JSONValue].self) { self = .object(value) }
        else { throw DecodingError.dataCorruptedError(in: container, debugDescription: "unsupported JSON value") }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case let .bool(value): try container.encode(value)
        case let .integer(value): try container.encode(value)
        case let .number(value): try container.encode(value)
        case let .string(value): try container.encode(value)
        case let .array(value): try container.encode(value)
        case let .object(value): try container.encode(value)
        }
    }
}

private func decodeFlexibleString<K: CodingKey>(
    _ container: KeyedDecodingContainer<K>, forKey key: K
) throws -> String {
    if let value = try? container.decode(String.self, forKey: key) { return value }
    if let value = try? container.decode(Int64.self, forKey: key) { return String(value) }
    throw DecodingError.dataCorruptedError(forKey: key, in: container, debugDescription: "expected string or integer")
}

public struct CandidateResult: Codable, Equatable, Sendable {
    public var candidateNumber: Int
    public var periodDays: Double
    public var t0: Double
    public var durationDays: Double
    public var depthPPM: Double
    public var snr: Double
    public var blsPower: Double?
    public var expectedTransits: Int?
    public var triceratopsFPP: Double? = nil
    public var triceratopsNFPP: Double? = nil
    public var nfpp: Double? = nil
    public var triceratopsProbabilityPlanet: Double? = nil
    public var triceratopsVerdict: String? = nil
    public var exominerScore: Double? = nil
    public var exominerTCEPeriod: Double? = nil
    public var exominerSectorRun: String? = nil
    public var exominerFFIScore: Double? = nil
    public var exominerFFIModelVersion: String? = nil
    public var crossSourceName: String? = nil
    public var crossSourceSNR: Double? = nil
    public var crossSourceDepthPPM: Double? = nil
    public var crossSourceDepthRatio: Double? = nil
    public var crossSourceAgreement: Bool? = nil

    enum CodingKeys: String, CodingKey {
        case candidateNumber = "candidate_num", periodDays = "period_days", t0
        case durationDays = "duration_days", depthPPM = "depth_ppm", snr
        case blsPower = "bls_power", expectedTransits = "n_transits_expected"
        case triceratopsFPP = "triceratops_fpp", triceratopsNFPP = "triceratops_nfpp", nfpp
        case triceratopsProbabilityPlanet = "triceratops_prob_planet", triceratopsVerdict = "triceratops_verdict"
        case exominerScore = "exominer_score", exominerTCEPeriod = "exominer_tce_period", exominerSectorRun = "exominer_sector_run"
        case exominerFFIScore = "exominer_ffi_score", exominerFFIModelVersion = "exominer_ffi_model_version"
        case crossSourceName = "cross_source_name", crossSourceSNR = "cross_source_snr"
        case crossSourceDepthPPM = "cross_source_depth_ppm", crossSourceDepthRatio = "cross_source_depth_ratio"
        case crossSourceAgreement = "cross_source_agreement"
    }

    public init(candidateNumber: Int, periodDays: Double, t0: Double, durationDays: Double, depthPPM: Double, snr: Double, blsPower: Double? = nil, expectedTransits: Int? = nil) {
        self.candidateNumber = candidateNumber; self.periodDays = periodDays; self.t0 = t0
        self.durationDays = durationDays; self.depthPPM = depthPPM; self.snr = snr
        self.blsPower = blsPower; self.expectedTransits = expectedTransits
    }
}

public struct VettingResult: Codable, Equatable, Sendable {
    public var isHarmonic: Bool?
    public var harmonicOf: Int?
    public var harmonicRatio: Double?
    public var oddEvenConsistent: Bool?
    public var oddDepth: Double?
    public var evenDepth: Double?
    public var isVShaped: Bool?
    public var vMetric: Double?
    public var passed: Bool?

    enum CodingKeys: String, CodingKey {
        case isHarmonic = "is_harmonic", harmonicOf = "harmonic_of", harmonicRatio = "harmonic_ratio"
        case oddEvenConsistent = "odd_even_consistent", oddDepth = "odd_depth", evenDepth = "even_depth"
        case isVShaped = "is_v_shaped", vMetric = "v_metric", passed
    }
}

public struct CompletenessCell: Codable, Equatable, Sendable {
    public var radiusEarth: Double
    public var periodDays: Double
    public var depthPPM: Double?
    public var detectionProbability: Double
    public var recoveredCount: Int?
    public var injectedCount: Int?

    enum CodingKeys: String, CodingKey {
        case radiusEarth = "radius_earth", periodDays = "period_days", depthPPM = "depth_ppm"
        case detectionProbability = "detection_prob", recoveredCount = "n_recovered", injectedCount = "n_injected"
    }
}

public struct CompletenessResult: Codable, Equatable, Sendable {
    public var radiusGrid: [Double]?
    public var periodGrid: [Double]?
    public var completenessMap: [[Double]]?
    public var details: [CompletenessCell]
    public var stellarRadius: Double?
    public var truncated: Bool?
    public var periodsEvaluated: Int?
    public var radiiEvaluated: Int?

    enum CodingKeys: String, CodingKey {
        case radiusGrid = "radius_grid", periodGrid = "period_grid", completenessMap = "completeness_map"
        case details, stellarRadius = "stellar_radius", truncated
        case periodsEvaluated = "periods_evaluated", radiiEvaluated = "radii_evaluated"
    }
}

public struct PrimaryResultRequest: Codable, Equatable, Sendable {
    public var taskID: String?
    public var ticID: String?
    public var candidates: [CandidateResult]
    public var vettingResults: [VettingResult]
    public var cnnScores: [Double?]
    public var completeness: CompletenessResult?
    public var processingTimeSeconds: Double
    public var gpuUsed: Bool
    public var sectorCount: Int
    public var pointCount: Int
    public var timeSpanDays: Double?
    public var ra: Double?
    public var dec: Double?
    public var tmag: Double?
    public var timings: [String: JSONValue]?
    public var lightCurveLoadRecovered: Bool
    public var status: String
    public var error: String?
    /// Provenance: the effective alias phase-2 flag this result was computed
    /// with (python result key `harmonic_epoch_class`, set on completed
    /// results only; nil = key absent, as in python's load_failed result).
    public var harmonicEpochClass: Bool?

    enum CodingKeys: String, CodingKey {
        case harmonicEpochClass = "harmonic_epoch_class"
        case taskID = "task_id", ticID = "tic_id", candidates
        case vettingResults = "vetting_results", cnnScores = "cnn_scores", completeness
        case processingTimeSeconds = "processing_time_seconds", gpuUsed = "gpu_used"
        case sectorCount = "n_sectors", pointCount = "n_points", timeSpanDays = "time_span_days"
        case ra, dec, tmag, timings, lightCurveLoadRecovered = "lc_load_recovered", status, error
    }

    public init(taskID: String?, ticID: String?, status: String = "completed", error: String? = nil, candidates: [CandidateResult] = [], vettingResults: [VettingResult] = [], cnnScores: [Double?] = [], completeness: CompletenessResult? = nil, processingTimeSeconds: Double = 0, gpuUsed: Bool = false, sectorCount: Int = 0, pointCount: Int = 0, timeSpanDays: Double? = nil, ra: Double? = nil, dec: Double? = nil, tmag: Double? = nil, timings: [String: JSONValue]? = nil, lightCurveLoadRecovered: Bool = false, harmonicEpochClass: Bool? = nil) {
        self.taskID = taskID; self.ticID = ticID; self.status = status; self.error = error
        self.harmonicEpochClass = harmonicEpochClass
        self.candidates = candidates; self.vettingResults = vettingResults; self.cnnScores = cnnScores
        self.completeness = completeness; self.processingTimeSeconds = processingTimeSeconds
        self.gpuUsed = gpuUsed; self.sectorCount = sectorCount; self.pointCount = pointCount
        self.timeSpanDays = timeSpanDays; self.ra = ra; self.dec = dec; self.tmag = tmag
        self.timings = timings; self.lightCurveLoadRecovered = lightCurveLoadRecovered
    }
}

public struct BackfillTaskEnvelope: Decodable, Equatable, Sendable { public let task: BackfillTask? }

public struct BackfillTask: Decodable, Equatable, Sendable {
    public let candidateID: Int64
    public let ticID: String
    public let candidateNumber: Int
    public let periodDays: Double
    public let t0: Double
    public let durationDays: Double
    public let depthPPM: Double
    public let snr: Double
    public let sectorCount: Int
    public let cadenceSeconds: Double?
    public let lcSource: String?
    /// Preserves the distinction between a missing field and an explicit JSON null.
    public let tglcNearEdge: JSONValue?
    public let geometry: JSONValue?
    public let population: JSONValue?
    public let populationSHA256: String?
    public let populationVersion: String?
    public let populationCrosscheck: JSONValue?
    /// Original FPP claim token, including an explicit null for unclaimed legacy tasks.
    public let fppClaimID: JSONValue

    enum CodingKeys: String, CodingKey {
        case candidateID = "candidate_id", ticID = "tic_id", candidateNumber = "candidate_num"
        case periodDays = "period_days", t0, durationDays = "duration_days", depthPPM = "depth_ppm", snr
        case sectorCount = "n_sectors"
        case cadenceSeconds = "cadence_s", lcSource = "lc_source", tglcNearEdge = "tglc_near_edge"
        case geometry, population, populationSHA256 = "population_sha256"
        case populationVersion = "population_version", populationCrosscheck = "population_crosscheck"
        case fppClaimID = "fpp_claim_id"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        candidateID = try c.decode(Int64.self, forKey: .candidateID)
        ticID = try decodeFlexibleString(c, forKey: .ticID)
        candidateNumber = try c.decode(Int.self, forKey: .candidateNumber)
        periodDays = try c.decode(Double.self, forKey: .periodDays); t0 = try c.decode(Double.self, forKey: .t0)
        durationDays = try c.decode(Double.self, forKey: .durationDays); depthPPM = try c.decode(Double.self, forKey: .depthPPM)
        snr = try c.decode(Double.self, forKey: .snr); sectorCount = try c.decodeIfPresent(Int.self, forKey: .sectorCount) ?? 1
        cadenceSeconds = try c.decodeIfPresent(Double.self, forKey: .cadenceSeconds)
        lcSource = try c.decodeIfPresent(String.self, forKey: .lcSource)
        tglcNearEdge = c.contains(.tglcNearEdge) ? try c.decode(JSONValue.self, forKey: .tglcNearEdge) : nil
        geometry = try c.decodeIfPresent(JSONValue.self, forKey: .geometry)
        population = try c.decodeIfPresent(JSONValue.self, forKey: .population)
        populationSHA256 = try c.decodeIfPresent(String.self, forKey: .populationSHA256)
        populationVersion = try c.decodeIfPresent(String.self, forKey: .populationVersion)
        populationCrosscheck = c.contains(.populationCrosscheck)
            ? try c.decode(JSONValue.self, forKey: .populationCrosscheck) : nil
        fppClaimID = c.contains(.fppClaimID) ? try c.decode(JSONValue.self, forKey: .fppClaimID) : .null
    }
}

/// One durable outbox row (stored by `DurableOutbox`, sent by `WorkerAPIClient`).
/// Kept here, free of SQLite, so the API client also builds on Linux.
public struct OutboxEntry: Equatable, Sendable {
    public let id: String
    public let endpoint: String
    public let workerID: String
    public let payload: Data
    public let createdAt: Date
    public let attempts: Int
    public let nextAttemptAt: Date
    public let lastError: String?
}

public struct FPPReleaseRequest: Codable, Equatable, Sendable {
    public let candidateID: Int64
    public let claimID: JSONValue
    public let ticID: String
    public let reason: String

    enum CodingKeys: String, CodingKey {
        case candidateID = "candidate_id", claimID = "claim_id", ticID = "tic_id", reason
    }

    public init(candidateID: Int64, claimID: JSONValue, ticID: String, reason: String) {
        self.candidateID = candidateID; self.claimID = claimID
        self.ticID = ticID; self.reason = reason
    }
}

public enum FPPTaskOutcome: Equatable, Sendable {
    case result(FPPResultRequest)
    case release(FPPReleaseRequest)
}

public struct FPPResultRequest: Codable, Equatable, Sendable {
    public var candidateID: Int64?; public var ticID: String?; public var fpp: Double?; public var nfpp: Double?
    public var probabilityPlanet: Double?; public var verdict: String?; public var candidate: [String: JSONValue]?
    public var fppModelVersion: Int?
    public var geometryVersion: Int?
    public var dilutionConvention: String?
    public var apertureSource: [String]?
    public var depthScale: Double?
    public var populationVersion: String?
    public var populationSource: String?
    public var populationSHA256: String?
    public var populationGenerator: String?
    public var populationMappingVersion: String?
    public var populationCrosscheck: JSONValue?
    public var nComp: Int?
    public var nCompFainter: Int?
    public var fppFlags: [String]?
    public var keplerUnconvergedDraws: Int?
    public var fppScen: [String: JSONValue]?
    public var dilutionAlternatives: JSONValue?
    public var dilutionCorrected: Bool?
    public var dilutionUnverified: Bool?
    public var exposureGLFallbackShare: Double?
    public var exposureUnconverged: Bool?
    public var exposureUnconvergedShare: Double?
    public var fluxratioTarget: Double?
    public var fppBackend: String?
    public var fppBackendDevice: String?
    public var gpuKernelVersion: String?
    /// Platform / libm / build provenance of the computing binary (plan (t) 2).
    public var fppPlatform: String?
    public var fppLibm: String?
    public var fppBuildProfile: String?
    public var fppSeed: UInt32?
    public var fppSwitches: [String: JSONValue]?
    public var galacticLatitude: Double?
    public var geometryCoverage: Double?
    public var geometryKind: String?
    public var geometryMode: String?
    public var geometryModelVersion: Int?
    public var geometryNSectors: Int?
    public var geometryNTaskSectors: Int?
    public var geometryPath: [String]?
    public var geometryPathEmulated: Bool?
    public var geometrySource: [String]?
    public var highCrowding: Bool?
    public var lcSource: String?
    public var nNearby: Int?
    public var nViableNeighbours: Int?
    public var nNodes: Int?
    public var nViabilityWeighted: Int?
    public var viabilityMassSkipped: Double?
    public var geometryNEpochs: [Int]?
    public var geometrySigmaCal: [Double]?
    public var populationCrosscheckStatus: String?
    public var sigmaFitIllConditioned: Bool?
    public var sigmaSource: String?
    /// JSON null is meaningful for TGLC tasks with unknown edge status.
    public var tglcNearEdge: JSONValue?
    public var tglcNearEdgeUnknown: Bool?
    public var cadenceSeconds: Double?
    public var trilegalUsed: Bool?
    // ephemeris refinement provenance fields (section 8b, FPP_EPHEM_REFINE)
    public var ephemRefined: Bool?
    public var ephemStatus: String?
    public var ephemT0Convention: String?
    public var ephemP: Double?
    public var ephemT0: Double?
    public var ephemShiftH: Double?
    public var ephemRefine: JSONValue?
    public var ephemRefineFailed: Bool?
    // step 7 (decision C) Monte-Carlo noise fields, `worker._fpp_result_model_fields` (task 26 / 33)
    /// MC standard errors of the reported FPP / NFPP; omitted when non-finite.
    public var fppMCSE: Double?
    public var nfppMCSE: Double?
    /// Draws per scenario of the reported stage (`n_draws_final`) and summed over the stages (`n_draws_total`).
    public var fppNDraws: Int?
    public var fppNDrawsTotal: Int?
    public var mcMinESS: Double?
    /// Present only when the MC policy ran.
    public var mcBorderline: Bool?
    public var mcDetail: JSONValue?
    /// WHY a `-1` / `failed` result has no score (Python `calc_fpp_by_source.last_skip_reason`): `.null` posts an
    /// explicit JSON null, as the Python worker does. Never on a computed result.
    public var skipReason: JSONValue?

    enum CodingKeys: String, CodingKey {
        case fppMCSE = "fpp_mc_se", nfppMCSE = "nfpp_mc_se", fppNDraws = "fpp_n_draws"
        case fppNDrawsTotal = "fpp_n_draws_total", mcMinESS = "mc_min_ess", mcBorderline = "mc_borderline"
        case mcDetail = "mc_detail", skipReason = "skip_reason"
        case candidateID = "candidate_id", ticID = "tic_id", fpp, nfpp, probabilityPlanet = "prob_planet", verdict, candidate
        case fppModelVersion = "fpp_model_version", geometryVersion = "geometry_version"
        case dilutionConvention = "dilution_convention", apertureSource = "aperture_source", depthScale = "depth_scale"
        case populationVersion = "population_version", populationSource = "population_source"
        case populationSHA256 = "population_sha256", populationGenerator = "population_generator"
        case populationMappingVersion = "population_mapping_version", populationCrosscheck = "population_crosscheck"
        case nComp = "n_comp", nCompFainter = "n_comp_fainter", fppFlags = "fpp_flags"
        case keplerUnconvergedDraws = "kepler_unconverged_draws", fppScen = "fpp_scen"
        case dilutionAlternatives = "dilution_alternatives"
        case dilutionCorrected = "dilution_corrected", dilutionUnverified = "dilution_unverified"
        case exposureGLFallbackShare = "exposure_gl_fallback_share", exposureUnconverged = "exposure_unconverged"
        case exposureUnconvergedShare = "exposure_unconverged_share", fluxratioTarget = "fluxratio_target"
        case fppBackend = "fpp_backend", fppBackendDevice = "fpp_backend_device", gpuKernelVersion = "gpu_kernel_version"
        case fppSeed = "fpp_seed", fppSwitches = "fpp_switches", galacticLatitude = "gal_b"
        case fppPlatform = "fpp_platform", fppLibm = "fpp_libm", fppBuildProfile = "fpp_build_profile"
        case geometryCoverage = "geometry_coverage", geometryKind = "geometry_kind", geometryMode = "geometry_mode"
        case geometryModelVersion = "geometry_model_version", geometryNSectors = "geometry_n_sectors"
        case geometryNTaskSectors = "geometry_n_task_sectors", geometryPath = "geometry_path"
        case geometryPathEmulated = "geometry_path_emulated", geometrySource = "geometry_source"
        case highCrowding = "high_crowding", lcSource = "lc_source", nNearby = "n_nearby"
        case nViableNeighbours = "n_viable_neighbours", populationCrosscheckStatus = "population_crosscheck_status"
        case nNodes = "n_nodes", nViabilityWeighted = "n_viability_weighted"
        case viabilityMassSkipped = "viability_mass_skipped", geometryNEpochs = "geometry_n_epochs"
        case geometrySigmaCal = "geometry_sigma_cal", tglcNearEdgeUnknown = "tglc_near_edge_unknown"
        case cadenceSeconds = "cadence_s"
        case sigmaFitIllConditioned = "sigma_fit_ill_conditioned", sigmaSource = "sigma_source"
        case tglcNearEdge = "tglc_near_edge", trilegalUsed = "trilegal_used"
        case ephemRefined = "ephem_refined", ephemStatus = "ephem_status", ephemT0Convention = "ephem_t0_convention"
        case ephemP = "ephem_P", ephemT0 = "ephem_t0", ephemShiftH = "ephem_shift_h"
        case ephemRefine = "ephem_refine", ephemRefineFailed = "ephem_refine_failed"
    }
    public init(candidateID: Int64?, ticID: String?, fpp: Double?, nfpp: Double? = nil,
                probabilityPlanet: Double? = nil, verdict: String? = nil, candidate: [String: JSONValue]? = nil,
                fppModelVersion: Int? = nil, geometryVersion: Int? = nil, dilutionConvention: String? = nil,
                apertureSource: [String]? = nil, depthScale: Double? = nil, populationVersion: String? = nil,
                populationSource: String? = nil, populationSHA256: String? = nil, populationGenerator: String? = nil,
                populationMappingVersion: String? = nil, populationCrosscheck: JSONValue? = nil,
                nComp: Int? = nil, nCompFainter: Int? = nil, fppFlags: [String]? = nil,
                keplerUnconvergedDraws: Int? = nil, fppScen: [String: JSONValue]? = nil,
                dilutionAlternatives: JSONValue? = nil,
                dilutionCorrected: Bool? = nil, dilutionUnverified: Bool? = nil,
                exposureGLFallbackShare: Double? = nil, exposureUnconverged: Bool? = nil,
                exposureUnconvergedShare: Double? = nil, fluxratioTarget: Double? = nil,
                fppBackend: String? = nil, fppSeed: UInt32? = nil, fppSwitches: [String: JSONValue]? = nil,
                galacticLatitude: Double? = nil, geometryCoverage: Double? = nil, geometryKind: String? = nil,
                geometryMode: String? = nil, geometryModelVersion: Int? = nil, geometryNSectors: Int? = nil,
                geometryNTaskSectors: Int? = nil, geometryPath: [String]? = nil, geometryPathEmulated: Bool? = nil,
                geometrySource: [String]? = nil, highCrowding: Bool? = nil, lcSource: String? = nil,
                nNearby: Int? = nil, nViableNeighbours: Int? = nil, populationCrosscheckStatus: String? = nil,
                sigmaFitIllConditioned: Bool? = nil, sigmaSource: String? = nil, tglcNearEdge: JSONValue? = nil,
                trilegalUsed: Bool? = nil, nNodes: Int? = nil, nViabilityWeighted: Int? = nil,
                viabilityMassSkipped: Double? = nil, geometryNEpochs: [Int]? = nil,
                geometrySigmaCal: [Double]? = nil, tglcNearEdgeUnknown: Bool? = nil,
                cadenceSeconds: Double? = nil, fppPlatform: String? = nil, fppLibm: String? = nil,
                fppBuildProfile: String? = nil, fppBackendDevice: String? = nil,
                gpuKernelVersion: String? = nil) {
        self.candidateID = candidateID; self.ticID = ticID; self.fpp = fpp; self.nfpp = nfpp
        self.probabilityPlanet = probabilityPlanet; self.verdict = verdict; self.candidate = candidate
        self.fppModelVersion = fppModelVersion; self.geometryVersion = geometryVersion
        self.dilutionConvention = dilutionConvention; self.apertureSource = apertureSource
        self.depthScale = depthScale; self.populationVersion = populationVersion
        self.populationSource = populationSource; self.populationSHA256 = populationSHA256
        self.populationGenerator = populationGenerator; self.populationMappingVersion = populationMappingVersion
        self.populationCrosscheck = populationCrosscheck; self.nComp = nComp; self.nCompFainter = nCompFainter
        self.fppFlags = fppFlags; self.keplerUnconvergedDraws = keplerUnconvergedDraws; self.fppScen = fppScen
        self.dilutionAlternatives = dilutionAlternatives
        self.dilutionCorrected = dilutionCorrected; self.dilutionUnverified = dilutionUnverified
        self.exposureGLFallbackShare = exposureGLFallbackShare; self.exposureUnconverged = exposureUnconverged
        self.exposureUnconvergedShare = exposureUnconvergedShare; self.fluxratioTarget = fluxratioTarget
        self.fppBackend = fppBackend; self.fppSeed = fppSeed; self.fppSwitches = fppSwitches
        self.galacticLatitude = galacticLatitude; self.geometryCoverage = geometryCoverage
        self.geometryKind = geometryKind; self.geometryMode = geometryMode; self.geometryModelVersion = geometryModelVersion
        self.geometryNSectors = geometryNSectors; self.geometryNTaskSectors = geometryNTaskSectors
        self.geometryPath = geometryPath; self.geometryPathEmulated = geometryPathEmulated
        self.geometrySource = geometrySource; self.highCrowding = highCrowding; self.lcSource = lcSource
        self.nNearby = nNearby; self.nViableNeighbours = nViableNeighbours
        self.populationCrosscheckStatus = populationCrosscheckStatus; self.sigmaFitIllConditioned = sigmaFitIllConditioned
        self.sigmaSource = sigmaSource; self.tglcNearEdge = tglcNearEdge; self.trilegalUsed = trilegalUsed
        self.nNodes = nNodes; self.nViabilityWeighted = nViabilityWeighted
        self.viabilityMassSkipped = viabilityMassSkipped; self.geometryNEpochs = geometryNEpochs
        self.geometrySigmaCal = geometrySigmaCal; self.tglcNearEdgeUnknown = tglcNearEdgeUnknown
        self.cadenceSeconds = cadenceSeconds
        self.fppPlatform = fppPlatform; self.fppLibm = fppLibm; self.fppBuildProfile = fppBuildProfile
        self.fppBackendDevice = fppBackendDevice; self.gpuKernelVersion = gpuKernelVersion
    }
}

/// V3 fields added to the task protocol; kept as a small Codable value for
/// Python-generated payload round-trip verification and worker task parsing.
public struct FPPV3TaskPayload: Codable, Equatable, Sendable {
    public var cadenceSeconds: Double
    public var geometry: JSONValue
    public var population: JSONValue
    public var populationCrosscheck: JSONValue?
    public var populationSHA256: String
    public var populationVersion: String
    enum CodingKeys: String, CodingKey {
        case cadenceSeconds = "cadence_s", geometry, population
        case populationCrosscheck = "population_crosscheck", populationSHA256 = "population_sha256"
        case populationVersion = "population_version"
    }
    public init(cadenceSeconds: Double, geometry: JSONValue, population: JSONValue,
                populationCrosscheck: JSONValue?, populationSHA256: String, populationVersion: String) {
        self.cadenceSeconds = cadenceSeconds; self.geometry = geometry; self.population = population
        self.populationCrosscheck = populationCrosscheck; self.populationSHA256 = populationSHA256
        self.populationVersion = populationVersion
    }
}

public struct ExoMinerResultRequest: Codable, Equatable, Sendable {
    public var candidateID: Int64?; public var ticID: String?; public var score: Double?
    public var tcePeriod: Double?; public var sectorRun: String?
    enum CodingKeys: String, CodingKey { case candidateID = "candidate_id", ticID = "tic_id", score, tcePeriod = "tce_period", sectorRun = "sector_run" }
    public init(candidateID: Int64?, ticID: String?, score: Double?, tcePeriod: Double? = nil, sectorRun: String? = nil) {
        self.candidateID = candidateID; self.ticID = ticID; self.score = score; self.tcePeriod = tcePeriod; self.sectorRun = sectorRun
    }
}

public struct StatusResponse: Codable, Equatable, Sendable { public let status: String; public let outcome: String? }

public struct FFISector: Decodable, Equatable, Sendable {
    public let sector: Int; public let source: String; public let time: [Double]; public let flux: [Double]; public let quality: [Int]?
    public var effectiveQuality: [Int] { quality ?? Array(repeating: 0, count: time.count) }
}
public struct FFIResponse: Decodable, Equatable, Sendable {
    public let ticID: String; public let sectors: [FFISector]
    enum CodingKeys: String, CodingKey { case ticID = "tic_id", sectors }
    public init(from decoder: Decoder) throws { let c = try decoder.container(keyedBy: CodingKeys.self); ticID = try decodeFlexibleString(c, forKey: .ticID); sectors = try c.decode([FFISector].self, forKey: .sectors) }
}

public struct NearbyStar: Decodable, Equatable, Sendable {
    public let id: String; public let ra: Double; public let dec: Double; public let mass: Double?; public let radius: Double?
    public let effectiveTemperature: Double?; public let tessMagnitude: Double?; public let parallax: Double?; public let stellarSource: String?
    /// Stellar-mass fallback provenance served by `/api/nearby` (task 33 item 3), kept as raw JSON so the FPP
    /// calculation applies Python's semantics (`mass_source` / hint string compares, `bool(giant_host)`).
    public let massSource: JSONValue?; public let giantHost: JSONValue?; public let massUnavailableReason: JSONValue?
    enum CodingKeys: String, CodingKey { case id = "ID", ra, dec, mass, radius = "rad", effectiveTemperature = "Teff", tessMagnitude = "Tmag", parallax = "plx", stellarSource = "stellar_source", massSource = "mass_source", giantHost = "giant_host", massUnavailableReason = "mass_unavailable_reason" }
    public init(from decoder: Decoder) throws { let c = try decoder.container(keyedBy: CodingKeys.self); id = try decodeFlexibleString(c, forKey: .id); ra = try c.decode(Double.self, forKey: .ra); dec = try c.decode(Double.self, forKey: .dec); mass = try c.decodeIfPresent(Double.self, forKey: .mass); radius = try c.decodeIfPresent(Double.self, forKey: .radius); effectiveTemperature = try c.decodeIfPresent(Double.self, forKey: .effectiveTemperature); tessMagnitude = try c.decodeIfPresent(Double.self, forKey: .tessMagnitude); parallax = try c.decodeIfPresent(Double.self, forKey: .parallax); stellarSource = try c.decodeIfPresent(String.self, forKey: .stellarSource); massSource = try c.decodeIfPresent(JSONValue.self, forKey: .massSource); giantHost = try c.decodeIfPresent(JSONValue.self, forKey: .giantHost); massUnavailableReason = try c.decodeIfPresent(JSONValue.self, forKey: .massUnavailableReason) }
}
public struct NearbyResponse: Decodable, Equatable, Sendable {
    public let ticID: String; public let stars: [NearbyStar]; public let galacticLatitude: Double; public let source: String?
    enum CodingKeys: String, CodingKey { case ticID = "tic_id", stars, galacticLatitude = "gal_b", source }
    public init(from decoder: Decoder) throws { let c = try decoder.container(keyedBy: CodingKeys.self); ticID = try decodeFlexibleString(c, forKey: .ticID); stars = try c.decodeIfPresent([NearbyStar].self, forKey: .stars) ?? []; galacticLatitude = try c.decodeIfPresent(Double.self, forKey: .galacticLatitude) ?? 0; source = try c.decodeIfPresent(String.self, forKey: .source) }
}

public struct UpdateManifestFile: Codable, Equatable, Sendable { public let filename: String; public let sha256: String }
public struct UpdateManifest: Codable, Equatable, Sendable { public let files: [UpdateManifestFile] }

public struct StarMetadata: Decodable, Equatable, Sendable {
    public let ticID: String?; public let fitsFiles: [String]; public let sectors: [Int]
    public let ra: Double?; public let dec: Double?; public let tmag: Double?
    public let effectiveTemperature: Double?; public let logg: Double?
    public let stellarRadius: Double?; public let stellarMass: Double?; public let dataSource: String?
    enum CodingKeys: String, CodingKey {
        case ticID = "tic_id", fitsFiles = "fits_files", sectors, ra, dec, tmag
        case effectiveTemperature = "teff", logg
        case stellarRadius = "stellar_radius", stellarMass = "stellar_mass"
        case dataSource = "data_source"
    }
    public init(
        ticID: String?, fitsFiles: [String] = [], sectors: [Int] = [], ra: Double? = nil, dec: Double? = nil,
        tmag: Double? = nil, effectiveTemperature: Double? = nil, logg: Double? = nil,
        stellarRadius: Double? = nil, stellarMass: Double? = nil, dataSource: String? = nil
    ) {
        self.ticID = ticID; self.fitsFiles = fitsFiles; self.sectors = sectors; self.ra = ra; self.dec = dec
        self.tmag = tmag; self.effectiveTemperature = effectiveTemperature; self.logg = logg
        self.stellarRadius = stellarRadius; self.stellarMass = stellarMass; self.dataSource = dataSource
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if c.contains(.ticID) { ticID = try? decodeFlexibleString(c, forKey: .ticID) } else { ticID = nil }
        fitsFiles = try c.decodeIfPresent([String].self, forKey: .fitsFiles) ?? []
        sectors = try c.decodeIfPresent([Int].self, forKey: .sectors) ?? []
        ra = try c.decodeIfPresent(Double.self, forKey: .ra)
        dec = try c.decodeIfPresent(Double.self, forKey: .dec)
        tmag = try c.decodeIfPresent(Double.self, forKey: .tmag)
        effectiveTemperature = try c.decodeIfPresent(Double.self, forKey: .effectiveTemperature)
        logg = try c.decodeIfPresent(Double.self, forKey: .logg)
        stellarRadius = try c.decodeIfPresent(Double.self, forKey: .stellarRadius)
        stellarMass = try c.decodeIfPresent(Double.self, forKey: .stellarMass)
        dataSource = try c.decodeIfPresent(String.self, forKey: .dataSource)
    }
}

public enum TFRecordFetchResult: Equatable, Sendable {
    case ready(Data, sectorRun: String, ticID: String)
    case preprocessing(ticID: String, sectorRun: String)
    case notAvailable
}
