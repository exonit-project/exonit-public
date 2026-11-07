import Foundation

/// Inputs to the source-aware native FPP calculation. Catalogues and light curves are supplied by the caller;
/// this layer only consumes the task payload and the already loaded samples.
public struct FPPV3CalcInputs: Sendable {
    public var ticID: String
    public var period: Double
    public var t0: Double
    public var durationDays: Double
    public var depthPPM: Double
    public var time: [Double]
    public var flux: [Double]
    public var nearby: [String: JSONValue]
    public var geometry: JSONValue?
    public var population: JSONValue?
    public var populationSHA256: String?
    public var populationVersion: String?
    public var populationCrosscheck: JSONValue?
    public var cadenceSeconds: Double?
    public var lcSource: String?
    /// `nil` means absent; `.null` preserves an explicit JSON null from the Python task contract.
    public var tglcNearEdge: JSONValue?
    public var seedEphemeris: (period: Double, t0: Double)?

    public init(ticID: String, period: Double, t0: Double, durationDays: Double, depthPPM: Double,
                time: [Double], flux: [Double], nearby: [String: JSONValue], geometry: JSONValue? = nil,
                population: JSONValue? = nil, populationSHA256: String? = nil, populationVersion: String? = nil,
                populationCrosscheck: JSONValue? = nil, cadenceSeconds: Double? = nil, lcSource: String? = nil,
                tglcNearEdge: JSONValue? = nil, seedEphemeris: (period: Double, t0: Double)? = nil) {
        self.ticID = ticID; self.period = period; self.t0 = t0; self.durationDays = durationDays
        self.depthPPM = depthPPM; self.time = time; self.flux = flux; self.nearby = nearby
        self.geometry = geometry; self.population = population; self.populationSHA256 = populationSHA256
        self.populationVersion = populationVersion; self.populationCrosscheck = populationCrosscheck
        self.cadenceSeconds = cadenceSeconds; self.lcSource = lcSource; self.tglcNearEdge = tglcNearEdge
        self.seedEphemeris = seedEphemeris
    }
}

public struct FPPV3ScenarioResult: Sendable {
    public var lnZ: Double
    public var probability: Double
    public init(lnZ: Double, probability: Double) { self.lnZ = lnZ; self.probability = probability }
}

public struct FPPV3DilutionAlternative: Sendable {
    public var fpp: Double
    public var nfpp: Double
    public var verdict: String
    public var scenarios: [String: FPPV3ScenarioResult]
    public var scenarioOrder: [String]
}

public struct FPPV3CalcResult: Sendable {
    public var fpp: Double
    public var nfpp: Double
    public var verdict: String
    public var scenarios: [String: FPPV3ScenarioResult]
    /// Dict insertion order is part of Python's scenario-weight reduction; retain it separately from the map.
    public var scenarioOrder: [String]
    public var flags: [String]
    public var dilutionCorrected: Bool
    public var dilutionUnverified: Bool
    public var depthScale: Double
    public var lcSource: String
    public var population: FPPV3BackgroundPopulation
    public var populationCrosscheckStatus: String
    public var keplerUnconvergedDraws: Int
    public var geometry: FPPV3GeometryInfo
    public var sigmaSource: String
    public var sigmaFitRcond: Double
    public var sigmaFitIllConditioned: Bool
    public var exposureUnconverged: Bool
    public var exposureUnconvergedShare: Double
    public var exposureGLFallbackShare: Double
    public var productionSwitchSet: Bool
    public var cadenceSeconds: Double?
    public var highCrowding: Bool
    public var galacticLatitude: Double
    public var nearbyCount: Int
    public var seed: UInt32
    public var dilutionAlternatives: [Bool: FPPV3DilutionAlternative]
    public var switches: [String: String]
    // step 7 (decision C) MC-noise fields of the reported (last) stage, r2 :5698-5710
    public var fppMCSE: Double = .nan
    public var nfppMCSE: Double = .nan
    public var nDrawsFinal: Int = 0
    public var nDrawsTotal: Int = 0
    /// `calc_fpp_by_source`: draws spent by every dilution run (:5829).
    public var nDrawsAllRuns: Int = 0
    public var refinementStages: [FPPV3MC.StageRecord] = []
    /// Per scenario of the reported stage, in Python's order: [ESS, relative SE of Z, tail scale beta].
    public var mcScen: [(String, [Double])] = []
    public var mcMinESS: Double = .nan
    public var mcMinESSScenario: String = ""
    public var mcPolicy: Bool = false
    public var borderline: Bool = false
    public var borderlineReasons: [String] = []
    // Task 44: unstable honesty flag
    public var unstableOn: Bool = false
    public var fppUnstable: Bool = false
    public var mcESSDom: Double = .nan
    public var mcDominantScenario: String = ""
    public var fppInterval: [Double] = []
    /// v3 gate observer: this run (on the selected result of `calcFPPBySourceV3`: every dilution run in the order
    /// Python runs them). Empty unless an `FPPV3Gate.Recorder` is open.
    public var gateRuns: [FPPV3GateRun] = []
}

/// Internal step-7 parameters of `calc_fpp_native` (r2 :5232-5238); the worker passes none of them.
public struct FPPV3MCOptions: Equatable, Sendable {
    public var adaptiveN: Bool?
    public var nCap: Int?
    public var futilityStop: Bool?
    public var unstableFlag: Bool?
    public init(adaptiveN: Bool? = nil, nCap: Int? = nil, futilityStop: Bool? = nil, unstableFlag: Bool? = nil) {
        self.adaptiveN = adaptiveN; self.nCap = nCap; self.futilityStop = futilityStop; self.unstableFlag = unstableFlag
    }
}

/// Run-level inputs of `FPPV3Calculation.runLevelVerdict`.
public struct FPPV3RunLevelGates: Sendable {
    public var mcPolicy: Bool, borderline: Bool, fpp: Double, fppSE: Double, nfpp: Double, nfppSE: Double
    public var crosscheckOk: Bool, backgroundQTarget: Bool, massFromRelation: Bool, giantHost: Bool
    public var unstableOn: Bool, unstable: Bool
    public init(mcPolicy: Bool, borderline: Bool, fpp: Double, fppSE: Double, nfpp: Double, nfppSE: Double,
                crosscheckOk: Bool, backgroundQTarget: Bool = true, massFromRelation: Bool, giantHost: Bool,
                unstableOn: Bool = false, unstable: Bool = false) {
        self.mcPolicy = mcPolicy; self.borderline = borderline; self.fpp = fpp; self.fppSE = fppSE
        self.nfpp = nfpp; self.nfppSE = nfppSE; self.crosscheckOk = crosscheckOk
        self.backgroundQTarget = backgroundQTarget; self.massFromRelation = massFromRelation; self.giantHost = giantHost
        self.unstableOn = unstableOn; self.unstable = unstable
    }
}

private struct FPPV3PreparedStar {
    var record: FPPStarRecord
    var smooth: FPPV3SmoothStar
}

private struct FPPV3RunResult {
    var result: FPPV3CalcResult
}

public enum FPPV3Calculation {
    private static func finiteLC(_ time: [Double], _ flux: [Double]) throws -> (time: [Double], flux: [Double]) {
        // Python: a shape mismatch raises -> generic failure (skip_reason None)
        guard time.count == flux.count else { throw FPPV3Error("failed", "time and flux lengths differ") }
        var t = [Double](), f = [Double](); t.reserveCapacity(time.count); f.reserveCapacity(flux.count)
        for (ti, fi) in zip(time, flux) where ti.isFinite && fi.isFinite { t.append(ti); f.append(fi) }
        guard t.count >= 100 else { throw FPPV3Error("raw_lc_unusable", "raw light curve unusable: fewer than 100 finite cadences") }
        return (t, f)
    }

    private static func truthy(_ value: JSONValue?) -> Bool {
        guard let value else { return false }
        switch value {
        case .null: return false
        case let .bool(x): return x
        case let .integer(x): return x != 0
        case let .number(x): return x != 0.0
        case let .string(x): return !x.isEmpty
        case let .array(x): return !x.isEmpty
        case let .object(x): return !x.isEmpty
        }
    }

    private static func nearEdgeValue(_ value: JSONValue?) -> Bool? {
        guard case let .bool(x)? = value else { return nil }
        return x
    }

    private static func targetLimbDarkening(_ star: FPPStarRecord) throws -> (Double, Double) {
        let logg = FPPV2.log10(FPPV2Consts.gCGS * star.mass * FPPV2Consts.msun /
                               FPPV2.pow(star.radius * FPPV2Consts.rsun, 2.0))
        let ld = try FPPV2SFamily.getLDC(teff: star.teff, logg: logg)
        return (ld.u1, ld.u2)
    }

    private static func inferredCadence(_ time: [Double]) throws -> Double {
        guard time.count >= 2 else { throw FPPV3Error("failed", "production cadence cannot be inferred from fewer than two samples") }
        let sorted = time.sorted()
        let diffs = zip(sorted, sorted.dropFirst()).map { $1 - $0 }.sorted()
        let median = diffs.count % 2 == 1 ? diffs[diffs.count / 2] :
            (diffs[diffs.count / 2 - 1] + diffs[diffs.count / 2]) / 2.0
        let cadence = (((median * 86400.0) * 1000.0).rounded(.toNearestOrEven)) / 1000.0
        guard cadence > 0.0 && cadence.isFinite else { throw FPPV3Error("failed", "production cadence is not positive and finite") }
        return cadence
    }

    private static func upstreamGeometry(
        stars: [FPPV2Star], depthPPM: Double, document: FPPV3Geometry?, dilutionCorrected: Bool
    ) throws -> (stars: [FPPV3SmoothStar], info: FPPV3GeometryInfo) {
        let old: FPPV2Geometry?
        if let document {
            old = FPPV2Geometry(kind: document.kind, sectors: document.sectors.map {
                FPPV2SectorGeometry(wcs: $0.wcs, col0: $0.col0, row0: $0.row0, aperture: $0.aperture,
                                    source: $0.source ?? "unknown")
            }, nTaskSectors: document.nTaskSectors)
        } else { old = nil }
        let computed = try FPPV2.computeFluxRatiosD5(stars: stars, depthPPM: depthPPM,
                                                     geometry: old, dilutionCorrected: dilutionCorrected)
        let smooth = computed.stars.map { row in
            FPPV3SmoothStar(star: row.star, fluxratio: row.fluxratio, tdepthRaw: row.tdepth, tdepth: row.tdepth,
                            viability: row.viable ? 1.0 : 0.0, viable: row.viable, sepArcsec: row.sepArcsec)
        }
        let real = computed.info.geometryKind == "real"
        let sectors = document?.sectors ?? []
        let apertures = real ? Array(Set(sectors.map { ($0.aperture?.isEmpty == false) ? "spoc_lc_mask" : "default_5x5" })).sorted() : ["synthetic"]
        let paths = real ? Array(Set(sectors.map { ($0.aperture?.isEmpty == false) ? "spoc_mask" : "u_default" })).sorted() : ["synthetic"]
        var info = FPPV3GeometryInfo()
        info.geometryKind = computed.info.geometryKind ?? "synthetic"
        info.geometryMode = "upstream"
        info.geometryNSectors = computed.info.geometryNSectors
        info.geometryNTaskSectors = computed.info.geometryNTaskSectors
        info.geometryCoverage = computed.info.geometryCoverage
        info.geometrySource = computed.info.geometrySource
        info.apertureSource = apertures
        info.geometryPath = paths
        info.nViableNeighbours = computed.info.nViableNeighbours
        info.fluxratioTarget = computed.info.fluxratioTarget
        return (smooth, info)
    }

    /// Run every scenario and both dilution conventions when the source is unknown/QLP.
    public static func calcFPPBySourceV3(
        _ input: FPPV3CalcInputs, draws: Int, seed: UInt32,
        productionFold: Bool, switches: FPPV3Switches, options: FPPV3MCOptions = FPPV3MCOptions()
    ) throws -> FPPV3CalcResult {
        let plan = FPPV3Dispatch.dilutionPlan(input.lcSource)
        var runs = [Bool: FPPV3CalcResult]()
        for corrected in plan.conventions {
            runs[corrected] = try runConvention(input, draws: draws, seed: seed, corrected: corrected,
                                                productionFold: productionFold, switches: switches,
                                                dilutionUnverified: plan.unverified, options: options)
        }
        guard let trueRun = runs[true] else { throw FPPV3Error("dilution_run_missing", "target-referenced dilution run missing") }
        let selectedCorrected = FPPV3Dispatch.chooseDilutionRun(trueRun: trueRun, falseRun: runs[false]) { $0.fpp }.corrected
        guard var selected = runs[selectedCorrected] else { throw FPPV3Error("dilution_run_missing", "selected dilution run disappeared") }
        if runs.count > 1 {
            selected.dilutionAlternatives = Dictionary(uniqueKeysWithValues: runs.map { entry in
                let (corrected, run) = entry
                return (corrected, FPPV3DilutionAlternative(fpp: run.fpp, nfpp: run.nfpp, verdict: run.verdict,
                                                             scenarios: run.scenarios, scenarioOrder: run.scenarioOrder))
            })
        } else {
            selected.dilutionAlternatives = [:]
        }
        // `n_draws_all_runs = sum(r.get("n_draws_total", 0) for r in runs.values())` (r2 :5829)
        selected.nDrawsAllRuns = runs.values.reduce(0) { $0 + $1.nDrawsTotal }
        if FPPV3Gate.active { selected.gateRuns = plan.conventions.flatMap { runs[$0]?.gateRuns ?? [] } }
        return selected
    }

    /// Python `calc_fpp_native` after the stage loop (r2 :5640-5693): the step-7 rule demotes `validated` when the
    /// result is borderline or fails `mc_validated_3sigma`; the giant cap demotes it independently; the run-level
    /// flags are [population_crosscheck_unverified, background_q_target_quirk, mass_from_relation, giant_host,
    /// mc_borderline]. The remaining caps follow in the caller.
    public static func runLevelVerdict(_ verdict: String, _ g: FPPV3RunLevelGates) -> (verdict: String, flags: [String]) {
        var v = verdict
        if g.mcPolicy && v == "validated"
            && (g.borderline || !FPPV3MC.validated3Sigma(fpp: g.fpp, fppSE: g.fppSE, nfpp: g.nfpp, nfppSE: g.nfppSE)) {
            v = "candidate"
        }
        if g.unstableOn && g.unstable && (v == "likely_fp" || v == "validated") {
            v = "candidate"
        }
        if g.giantHost && v == "validated" { v = "candidate" }
        var flags = [String]()
        if !g.crosscheckOk { flags.append("population_crosscheck_unverified") }
        if g.backgroundQTarget { flags.append("background_q_target_quirk") }
        if g.massFromRelation { flags.append("mass_from_relation") }
        if g.giantHost { flags.append("giant_host") }
        if g.borderline { flags.append("mc_borderline") }
        if g.unstableOn && g.unstable { flags.append("fpp_unstable") }
        return (v, flags)
    }

    /// Python (r2 :5409-5422): the server's hint wins when it is exactly giant_host / stellar_params_missing;
    /// otherwise R finite and R > 2.5 means a giant host.
    static func missingStellarParamsReason(_ target: FPPStarRecord) -> String {
        switch target.massUnavailableReason {
        case "giant_host"?: return "giant_host"
        case "stellar_params_missing"?: return "stellar_params_missing"
        default: return target.radius.isFinite && target.radius > 2.5 ? "giant_host" : "stellar_params_missing"
        }
    }

    /// The per-scenario draw-weight records of one scenario function (Python `_sc`, r2 :5465-5472).
    private static func takeRecords(_ sink: FPPV3MC.DiagSink, _ n0: Int, _ names: [String],
                                    _ diag: inout [String: FPPV3MC.DrawDiag]) throws {
        let got = sink.since(n0)
        if !got.isEmpty && got.count != names.count {
            throw FPPV3Error("failed", "\(got.count) draw-weight records for \(names.count) scenarios")
        }
        for (name, d) in zip(names, got) { diag[name] = d }
        FPPV3Gate.assign(n0, names)
    }

    private struct PassDiagnostics {
        var target = FPPV3Diagnostics()
        var d = FPPV3Diagnostics()
        var s = FPPV2ExposureStats()
        /// v3 gate observer: per-scenario records of this pass (empty unless a recorder is open)
        var gate: [String: FPPV3Gate.Record?] = [:]
    }

    private struct NeighbourFrame {
        var tag: String, mass: Double, radius: Double, teff: Double, logV: Double
        var flux: [Double], sigma: Double, fold: FPPV2ProdFold?
    }

    private static func runConvention(
        _ input: FPPV3CalcInputs, draws: Int, seed: UInt32, corrected: Bool,
        productionFold: Bool, switches: FPPV3Switches, dilutionUnverified: Bool, options: FPPV3MCOptions
    ) throws -> FPPV3CalcResult {
        // r2 :5283-5285: refine = switch if adaptive_n is None else adaptive_n; mc_policy = switch or refine
        let mcSwitch = switches.adaptiveN == "on"
        let refine = options.adaptiveN ?? mcSwitch
        let mcPolicy = mcSwitch || refine
        // Provenance gate first, before any work (Python 7b30ba1, decision 2026-10-25 option A): a supplied
        // population whose sha256 or version is missing or empty is refused as `population_invalid` (a release).
        // A missing population keeps its own `population_missing` refusal below.
        if input.population != nil,
           FPPV3ReleasePolicy.provenanceMissing(sha256: input.populationSHA256, version: input.populationVersion) {
            throw FPPV3Error("population_invalid",
                             "population provenance is incomplete (population_sha256/population_version missing)")
        }
        let source = FPPV3Dispatch.normalizeLCSource(input.lcSource)
        let edge = nearEdgeValue(input.tglcNearEdge)
        let depthScale = FPPV3Dispatch.depthScale(input.lcSource, nearEdge: edge)
        let raw = try finiteLC(input.time, input.flux)
        let cadence: Double?
        if productionFold {
            if let provided = input.cadenceSeconds {
                cadence = provided
            } else {
                cadence = try inferredCadence(raw.time)
            }
        } else {
            cadence = nil
        }
        let depthPPM = input.depthPPM * depthScale
        var fluxRaw = raw.flux
        if depthScale != 1.0 { for i in fluxRaw.indices { fluxRaw[i] = 1.0 + (fluxRaw[i] - 1.0) * depthScale } }
        guard let sigma = FPPV2Sigma.sigmaComponents(time: raw.time, flux: fluxRaw, period: input.period,
                                                     t0: input.t0, duration: input.durationDays) else {
            throw FPPV3Error("raw_lc_unusable", "raw light curve unusable: no q-prime sigma estimate")
        }

        let timePF: [Double], fluxPF: [Double], baseFold: FPPV2ProdFold?
        if productionFold {
            let rows = FPPV2Fold.foldProduction(time: raw.time, flux: fluxRaw, t0: input.t0,
                                                period: input.period, keepPhases: true)
            guard !rows.isEmpty && rows.count == sigma.sigmaB.count else {
                // Python raises RuntimeError -> None, skip_reason None
                throw FPPV3Error("failed", "production fold and q-prime sigma bins differ")
            }
            let fluxB = rows.map(\.fluxMean)
            baseFold = try FPPV2ProdFold.build(rows: rows, period: input.period, sigmaB: sigma.sigmaB,
                                               fluxB: fluxB, cadenceS: cadence!)
            timePF = []; fluxPF = []
        } else {
            let folded = FPPFold.bin(time: raw.time, flux: fluxRaw, period: input.period, t0: input.t0, bins: 1000)
            var t = [Double](), f = [Double]()
            for (ti, fi) in zip(folded.time, folded.flux) where ti.isFinite && fi.isFinite { t.append(ti); f.append(fi) }
            // Python: "too few phase-folded points" -> None, skip_reason None
            guard t.count >= 10 else { throw FPPV3Error("failed", "fewer than ten phase-folded bins") }
            timePF = t; fluxPF = f; baseFold = nil
        }

        guard let nearby = FPPInputs.parseNearby(input.nearby, ticID: input.ticID), !nearby.stars.isEmpty else {
            throw FPPV3Error("nearby_unavailable", "malformed or empty nearby-star response")
        }
        // 1. geometry (`_flux_ratios_for_mode`, r2 :5390) BEFORE the population (task-30 redo note 3)
        let geometry = try FPPV3Geometry.decode(input.geometry)
        let d5Stars = nearby.stars.map { FPPV2Star(id: $0.id, ra: $0.ra, dec: $0.dec, tmag: $0.tmag) }
        let smooth: [FPPV3SmoothStar], geometryInfo: FPPV3GeometryInfo
        if productionFold, let geometry, geometry.kind == "real", !geometry.sectors.isEmpty {
            let computed = try FPPV3GeometryEngine.computeFluxRatiosSmooth(stars: d5Stars, depthPPM: depthPPM,
                                                                            geometry: geometry, dilutionCorrected: corrected,
                                                                            lcSource: input.lcSource, vMin: 0.01)
            smooth = computed.stars; geometryInfo = computed.info
        } else {
            let computed = try upstreamGeometry(stars: d5Stars, depthPPM: depthPPM,
                                                document: geometry, dilutionCorrected: corrected)
            smooth = computed.stars; geometryInfo = computed.info
        }
        // 2. population, 3. cross-check type, 4. alarm (r2 :5392-5399)
        let population: FPPV3BackgroundPopulation
        do {
            population = try FPPV3Population.validate(input.population, targetTmag: nearby.stars[0].tmag,
                                                      expectedSHA256: input.populationSHA256,
                                                      expectedVersion: input.populationVersion,
                                                      expectedTICID: input.ticID)
        } catch let error as FPPV3Error {
            throw error  // population_missing / population_invalid, typed at the source
        } catch {
            throw FPPV3Error("population_invalid", "background population failed validation: \(error)")
        }
        if let check = input.populationCrosscheck, case .null = check {
            // A JSON null is equivalent to an absent cross-check.
        } else if let check = input.populationCrosscheck, case .object = check {
            // Valid cross-check object; fields are interpreted below.
        } else if input.populationCrosscheck != nil {
            // Python: BackgroundUnavailable('invalid population cross-check') -> population_invalid.
            throw FPPV3Error("population_invalid", "invalid population cross-check")
        }
        let crossObject: [String: JSONValue] = {
            guard case let .object(value)? = input.populationCrosscheck else { return [:] }; return value
        }()
        let crossOk: Bool = { if case .bool(true)? = crossObject["gaia_available"] { return true }; return false }()
        if truthy(crossObject["alarm"]) && population.source != "gaia" {
            throw FPPV3Error("population_alarm", "R8 alarm requires a server-supplied Gaia population")
        }
        let prepared = zip(nearby.stars, smooth).map { FPPV3PreparedStar(record: $0.0, smooth: $0.1) }
        let target = prepared[0].record
        // 5. target stellar parameters (r2 :5409-5422): explicit skip_reason, never an invented mass
        guard !target.mass.isNaN && !target.radius.isNaN && !target.teff.isNaN else {
            throw FPPV3Error(missingStellarParamsReason(target), "missing target stellar parameters")
        }
        // 6. target limb darkening (FPPV2LDCNoRow -> skip_reason ldc_no_row)
        let targetLD = try targetLimbDarkening(target)
        let targetFrame = FPPV3Dispatch.frameFluxratio(prepared[0].smooth.fluxratio,
                                                       targetRatio: prepared[0].smooth.fluxratio, corrected: corrected)
        let fluxT: [Double], sigmaT: Double, foldT: FPPV2ProdFold?
        if let baseFold {
            let renorm = try baseFold.renorm(fr: targetFrame)
            fluxT = []; sigmaT = sigma.sigmaOot / targetFrame; foldT = renorm
        } else {
            let renorm = FPPFold.renorm(flux: fluxPF, fluxError: sigma.sigmaOot, starFluxRatio: targetFrame)
            fluxT = renorm.flux; sigmaT = renorm.error; foldT = nil
        }
        let dConfig = FPPV3DFamilyConfig.from(switches)
        let zero = [0.0], sigmaArray = [Double](repeating: sigmaT, count: timePF.count)
        let nearbyCount = prepared.count - 1
        let highCrowding = abs(nearby.galacticLatitude) < 10.0 || nearbyCount > 15

        // neighbour frames do not depend on the stage (Python rebuilds the same values inside every pass)
        var neighbours = [NeighbourFrame]()
        var viableIndex = 0
        for neighbor in prepared.dropFirst() where neighbor.smooth.viable {
            let m = neighbor.record.mass.isNaN ? 1.0 : neighbor.record.mass
            let r = neighbor.record.radius.isNaN ? 1.0 : neighbor.record.radius
            let teff = neighbor.record.teff.isNaN ? 5777.0 : neighbor.record.teff
            let fr = FPPV3Dispatch.frameFluxratio(neighbor.smooth.fluxratio,
                targetRatio: prepared[0].smooth.fluxratio, corrected: corrected)
            let nsFlux: [Double], nsSigma: Double, nsFold: FPPV2ProdFold?
            if let baseFold {
                nsFold = try baseFold.renorm(fr: fr); nsFlux = []; nsSigma = sigma.sigmaOot / fr
            } else {
                let rn = FPPFold.renorm(flux: fluxPF, fluxError: sigma.sigmaOot, starFluxRatio: fr)
                nsFlux = rn.flux; nsSigma = rn.error; nsFold = nil
            }
            let logV = neighbor.smooth.viability > 0.0 ? FPPV2.log(neighbor.smooth.viability) : -.infinity
            neighbours.append(NeighbourFrame(tag: "\(neighbor.record.id):\(viableIndex)", mass: m, radius: r,
                                             teff: teff, logV: logV, flux: nsFlux, sigma: nsSigma, fold: nsFold))
            viableIndex += 1
        }

        // One Monte-Carlo pass with `n` draws and a FRESH RandomState(seed) (Python `_mc_pass` + `_stage_pass`,
        // r2 :5455-5625): the exposure / Kepler diagnostics are per pass.
        func mcPass(_ n: Int, _ seedK: UInt64) throws -> FPPV3MC.StagePass<PassDiagnostics> {
            var rng = FPPRNG(seed: UInt32(truncatingIfNeeded: seedK))
            var pd = PassDiagnostics()
            let sink = FPPV3MC.DiagSink()
            defer { sink.close() }
            FPPV3Gate.beginPass()
            var diag = [String: FPPV3MC.DrawDiag]()
            var lnz = [(String, Double)]()

            var n0 = sink.count
            let tp = try FPPV3TargetFamily.lnZTTP(time: timePF, flux: fluxT, sigma: sigmaT,
                period: input.period, mass: target.mass, radius: target.radius, n: n,
                u1: targetLD.0, u2: targetLD.1, companionFluxRatio: zero, lnpriorCompanion: zero,
                rng: &rng, switches: switches.core, diagnostics: &pd.target, fold: foldT)
            try takeRecords(sink, n0, ["TP"], &diag); lnz.append(("TP", tp))
            n0 = sink.count
            let eb = try FPPV3TargetFamily.lnZTEB(time: timePF, flux: fluxT, sigma: sigmaT, sigmaVeto: sigmaT,
                period: input.period, mass: target.mass, radius: target.radius, teff: target.teff, n: n,
                u1: targetLD.0, u2: targetLD.1, companionFluxRatio: zero, lnpriorCompanion: zero,
                rng: &rng, switches: switches.core, diagnostics: &pd.target, fold: foldT)
            try takeRecords(sink, n0, ["EB", "EBx2P"], &diag); lnz.append(("EB", eb.eb)); lnz.append(("EBx2P", eb.twin))
            if !target.parallax.isNaN {
                n0 = sink.count
                let ptp = try FPPV3TargetFamily.lnZPTP(time: timePF, flux: fluxT, sigma: sigmaT,
                    period: input.period, mass: target.mass, radius: target.radius, n: n,
                    u1: targetLD.0, u2: targetLD.1, parallaxMas: target.parallax,
                    rng: &rng, switches: switches.core, diagnostics: &pd.target, fold: foldT)
                try takeRecords(sink, n0, ["PTP"], &diag); lnz.append(("PTP", ptp))
                n0 = sink.count
                let peb = try FPPV3TargetFamily.lnZPEB(time: timePF, flux: fluxT, sigma: sigmaT, sigmaVeto: sigmaT,
                    period: input.period, mass: target.mass, radius: target.radius, teff: target.teff, n: n,
                    u1: targetLD.0, u2: targetLD.1, parallaxMas: target.parallax,
                    rng: &rng, switches: switches.core, diagnostics: &pd.target, fold: foldT)
                try takeRecords(sink, n0, ["PEB", "PEBx2P"], &diag)
                lnz.append(("PEB", peb.eb)); lnz.append(("PEBx2P", peb.twin))
                n0 = sink.count
                let stp = try FPPV2SFamily.lnZSTPWithDetails(time: timePF, flux: fluxT, sigma: sigmaArray,
                    period: input.period, massSolar: target.mass, radiusSolar: target.radius, teff: target.teff,
                    count: n, parallax: target.parallax, rng: &rng, switches: switches.core,
                    fold: foldT, sigmaVeto: sigmaT, exposureStats: &pd.s)
                try takeRecords(sink, n0, ["STP"], &diag)
                pd.target.keplerUnconvergedDraws += stp.lnL.filter(\.isNaN).count
                lnz.append(("STP", stp.lnZ))
                n0 = sink.count
                let seb = try FPPV2SFamily.lnZSEBWithDetails(time: timePF, flux: fluxT, sigma: sigmaArray,
                    period: input.period, massSolar: target.mass, radiusSolar: target.radius, teff: target.teff,
                    count: n, parallax: target.parallax, rng: &rng, switches: switches.core,
                    fold: foldT, sigmaVeto: sigmaT, exposureStats: &pd.s)
                try takeRecords(sink, n0, ["SEB", "SEBx2P"], &diag)
                pd.target.keplerUnconvergedDraws += seb.lnLSEB.filter(\.isNaN).count + seb.lnLSEBx2P.filter(\.isNaN).count
                lnz.append(("SEB", seb.lnZSEB)); lnz.append(("SEBx2P", seb.lnZSEBx2P))
            } else {
                for name in ["PTP", "PEB", "PEBx2P", "STP", "SEB", "SEBx2P"] { lnz.append((name, -.infinity)) }
            }

            n0 = sink.count
            let dtp: Double
            if let foldT {
                dtp = try FPPV3DFamily.lnZDTP(time: timePF, flux: fluxT, sigma: sigmaT, period: input.period,
                    targetMass: target.mass, targetRadius: target.radius, n: n, u1: targetLD.0, u2: targetLD.1,
                    population: population, rng: &rng, config: dConfig, diagnostics: &pd.d, fold: foldT)
            } else {
                dtp = try FPPV3DFamily.lnZDTP(time: timePF, flux: fluxT, sigma: sigmaT, period: input.period,
                    targetMass: target.mass, targetRadius: target.radius, n: n, u1: targetLD.0, u2: targetLD.1,
                    population: population, rng: &rng, config: dConfig, diagnostics: &pd.d)
            }
            try takeRecords(sink, n0, ["DTP"], &diag); lnz.append(("DTP", dtp))
            n0 = sink.count
            let deb: (eb: Double, twin: Double)
            if let foldT {
                deb = try FPPV3DFamily.lnZDEB(time: timePF, flux: fluxT, sigma: sigmaT, sigmaVeto: sigmaT,
                    period: input.period, mass: target.mass, radius: target.radius, teff: target.teff,
                    u1: targetLD.0, u2: targetLD.1, n: n, population: population, rng: &rng,
                    config: dConfig, diagnostics: &pd.d, fold: foldT)
            } else {
                deb = try FPPV3DFamily.lnZDEB(time: timePF, flux: fluxT, sigma: sigmaT, sigmaVeto: sigmaT,
                    period: input.period, mass: target.mass, radius: target.radius, teff: target.teff,
                    u1: targetLD.0, u2: targetLD.1, n: n, population: population, rng: &rng,
                    config: dConfig, diagnostics: &pd.d)
            }
            try takeRecords(sink, n0, ["DEB", "DEBx2P"], &diag); lnz.append(("DEB", deb.eb)); lnz.append(("DEBx2P", deb.twin))
            n0 = sink.count
            let btp: Double
            if let foldT {
                btp = try FPPV3DFamily.lnZBTP(time: timePF, flux: fluxT, sigma: sigmaT,
                    period: input.period, n: n, population: population, rng: &rng,
                    config: dConfig, diagnostics: &pd.d, fold: foldT)
            } else {
                btp = try FPPV3DFamily.lnZBTP(time: timePF, flux: fluxT, sigma: sigmaT,
                    period: input.period, n: n, population: population, rng: &rng,
                    config: dConfig, diagnostics: &pd.d)
            }
            try takeRecords(sink, n0, ["BTP"], &diag); lnz.append(("BTP", btp))
            n0 = sink.count
            let beb: (eb: Double, twin: Double)
            if let foldT {
                beb = try FPPV3DFamily.lnZBEB(time: timePF, flux: fluxT, sigma: sigmaT, sigmaVeto: sigmaT,
                    period: input.period, targetMass: target.mass, targetRadius: target.radius,
                    targetTeff: target.teff, targetU1: targetLD.0, targetU2: targetLD.1,
                    n: n, population: population, rng: &rng, config: dConfig, diagnostics: &pd.d, fold: foldT)
            } else {
                beb = try FPPV3DFamily.lnZBEB(time: timePF, flux: fluxT, sigma: sigmaT, sigmaVeto: sigmaT,
                    period: input.period, targetMass: target.mass, targetRadius: target.radius,
                    targetTeff: target.teff, targetU1: targetLD.0, targetU2: targetLD.1,
                    n: n, population: population, rng: &rng, config: dConfig, diagnostics: &pd.d)
            }
            try takeRecords(sink, n0, ["BEB", "BEBx2P"], &diag); lnz.append(("BEB", beb.eb)); lnz.append(("BEBx2P", beb.twin))

            var nearbyScen = [(String, Double)]()
            for nb in neighbours {
                // Python computes the neighbour's LDC inside the pass (LDCNoRow -> skip_reason ldc_no_row)
                let ld = try targetLimbDarkening(FPPStarRecord(id: "", ra: 0, dec: 0, mass: nb.mass, radius: nb.radius,
                                                              teff: nb.teff, tmag: 0, parallax: .nan))
                let ntpName = "NTP:\(nb.tag)", nebName = "NEB:\(nb.tag)", nebxName = "NEBx2P:\(nb.tag)"
                n0 = sink.count
                let ntp = try FPPV3TargetFamily.lnZTTP(time: timePF, flux: nb.flux, sigma: nb.sigma,
                    period: input.period, mass: nb.mass, radius: nb.radius, n: n, u1: ld.0, u2: ld.1,
                    companionFluxRatio: zero, lnpriorCompanion: zero, rng: &rng, switches: switches.core,
                    diagnostics: &pd.target, fold: nb.fold)
                try takeRecords(sink, n0, [ntpName], &diag)
                n0 = sink.count
                let neb = try FPPV3TargetFamily.lnZTEB(time: timePF, flux: nb.flux, sigma: nb.sigma,
                    sigmaVeto: nb.sigma, period: input.period, mass: nb.mass, radius: nb.radius, teff: nb.teff, n: n,
                    u1: ld.0, u2: ld.1, companionFluxRatio: zero, lnpriorCompanion: zero,
                    rng: &rng, switches: switches.core, diagnostics: &pd.target, fold: nb.fold)
                try takeRecords(sink, n0, [nebName, nebxName], &diag)
                nearbyScen.append((ntpName, ntp + nb.logV))
                nearbyScen.append((nebName, neb.eb + nb.logV))
                nearbyScen.append((nebxName, neb.twin + nb.logV))
            }
            pd.gate = FPPV3Gate.takeNamed()
            return FPPV3MC.StagePass(lnZ: lnz, nearby: nearbyScen, diag: diag, payload: pd)
        }

        // Stage loop (r2 :5614-5632): stage 0 = the candidate's own seed; stage k >= 1 = stage_seed(...).
        let ladder = FPPV3MC.stageLadder(draws, refine: refine, nCap: options.nCap)
        let seedP = input.seedEphemeris?.period ?? input.period
        let seedT0 = input.seedEphemeris?.t0 ?? input.t0
        let staged = try FPPV3MC.runStages(
            ladder: ladder,
            seedForStage: { idx in
                idx == 0 ? UInt64(seed) : UInt64(FPPV3MC.stageSeed(ticID: input.ticID, period: seedP,
                                                                   t0: seedT0, stage: idx, seed0: UInt64(seed)))
            },
            futility: options.futilityStop ?? true, pass: mcPass)
        guard let staged else { throw FPPV3Error("all_scenarios_non_finite", "all scenario likelihoods are non-finite") }
        let summ = staged.summary, pd = staged.pass.payload
        let fpp = summ.fpp, nfpp = summ.nfpp
        let borderline = mcPolicy && !staged.assess.resolved
        let baseVerdict = FPPPolicy.verdict(fpp: fpp, nfpp: nfpp, highCrowding: highCrowding).rawValue
        let giantHost = target.giantHost || (target.radius.isFinite && target.radius > 2.5)
        let fromRelation = target.massSource == "teff_relation_tic8fit"
        let isProd = switches == .production
        let unstableOn = try options.unstableFlag ?? FPPV3Dispatch.parseUnstableFlagSwitch(production: isProd)
        let unstable = FPPV3MC.isUnstable(summ)
        let (domName, domESS) = FPPV3MC.dominant(summ)
        var (verdict, flags) = runLevelVerdict(baseVerdict, FPPV3RunLevelGates(
            mcPolicy: mcPolicy, borderline: borderline, fpp: fpp, fppSE: summ.fppSE, nfpp: nfpp, nfppSE: summ.nfppSE,
            crosscheckOk: crossOk, backgroundQTarget: switches.backgroundQ == "target",
            massFromRelation: fromRelation, giantHost: giantHost,
            unstableOn: unstableOn, unstable: unstable))
        if dilutionUnverified { flags.append("dilution_unverified") }
        if source == "tglc" && edge == nil { flags.append("tglc_near_edge_unknown") }
        let kCount = pd.target.keplerUnconvergedDraws + pd.d.keplerUnconvergedDraws
        var diagnostics = FPPV3Diagnostics()
        diagnostics.keplerUnconvergedDraws = kCount
        let fallbackShare = max(pd.target.exposureGLFallbackShare, max(pd.d.exposureGLFallbackShare, pd.s.fallbackShare))
        let uncShare = max(pd.target.exposureUnconvergedShare, max(pd.d.exposureUnconvergedShare, pd.s.unconvergedShare))
        diagnostics.exposureGLFallbackShare = fallbackShare; diagnostics.exposureUnconvergedShare = uncShare
        let exposureUnconverged = uncShare > 0.01
        let geometryReal = geometryInfo.geometryKind == "real" && geometryInfo.geometryCoverage >= 1.0
        let ffiReference = geometryInfo.geometrySource.contains("ffi_reference")
        let validTGLCFFI = geometryInfo.geometryMode == "smooth" && geometryInfo.geometryPath == ["tglc_box"]
            && geometryInfo.apertureSource == ["tglc_box_header"] && !geometryInfo.geometryPathEmulated
            && !geometryInfo.geometryNEpochs.isEmpty && geometryInfo.geometryNEpochs.allSatisfy { $0 >= 2 }
        let ffiUnvalidated = ffiReference && !validTGLCFFI
        let smoothUnvalidated = geometryInfo.geometryMode == "smooth" &&
            (geometryInfo.geometryPathEmulated || geometryInfo.geometryPath.contains { !["spoc_mask", "tglc_box"].contains($0) })
        let geometryUnvalidated = !geometryReal || ffiUnvalidated || smoothUnvalidated
        let nearUnknown = source == "tglc" && edge == nil
        verdict = FPPV3Dispatch.finalizeVerdict(verdict, flags: &flags, diagnostics: diagnostics,
            populationCrosscheckVerified: crossOk, dilutionUnverified: dilutionUnverified,
            tglcNearEdgeUnknown: nearUnknown, geometryPathEmulated: geometryUnvalidated,
            ffiReferenceUnvalidated: ffiUnvalidated, sigmaFitIllConditioned: sigma.fitIllConditioned,
            exposureUnconverged: exposureUnconverged, isTGLC: source == "tglc", tglcNearEdge: edge)
        var lnzByName = [String: Double]()
        for (name, value) in staged.pass.lnZ + staged.pass.nearby { lnzByName[name] = value }
        var scenarioResults = [String: FPPV3ScenarioResult]()
        for (i, name) in summ.names.enumerated() {
            scenarioResults[name] = FPPV3ScenarioResult(lnZ: lnzByName[name]!, probability: summ.probs[i])
        }
        var result = FPPV3CalcResult(fpp: fpp, nfpp: nfpp, verdict: verdict, scenarios: scenarioResults,
            scenarioOrder: summ.names, flags: flags, dilutionCorrected: corrected, dilutionUnverified: dilutionUnverified,
            depthScale: depthScale, lcSource: source, population: population,
            populationCrosscheckStatus: crossOk ? "passed" : "unverified",
            keplerUnconvergedDraws: kCount, geometry: geometryInfo, sigmaSource: "q_prime",
            sigmaFitRcond: sigma.fitRcond, sigmaFitIllConditioned: sigma.fitIllConditioned,
            exposureUnconverged: exposureUnconverged, exposureUnconvergedShare: uncShare,
            exposureGLFallbackShare: fallbackShare, productionSwitchSet: productionFold,
            cadenceSeconds: cadence, highCrowding: highCrowding,
            galacticLatitude: ((nearby.galacticLatitude * 100.0).rounded(.toNearestOrEven)) / 100.0,
            nearbyCount: nearbyCount, seed: seed, dilutionAlternatives: [:],
            switches: FPPV3Dispatch.switchDictionary(switches, ephemRefine: input.seedEphemeris != nil || ((try? FPPV3Dispatch.parseEphemRefineSwitch(production: isProd)) ?? isProd), unstableFlag: unstableOn))
        if FPPV3Gate.active {
            result.gateRuns = [FPPV3GateRun(dilutionCorrected: corrected, fpp: fpp, nfpp: nfpp,
                scenarios: summ.names.map { name in
                    FPPV3GateRun.Scenario(name: name, lnZ: scenarioResults[name]!.lnZ,
                                          probability: scenarioResults[name]!.probability,
                                          record: pd.gate[name] ?? nil)
                })]
        }
        result.fppMCSE = summ.fppSE
        result.nfppMCSE = summ.nfppSE
        result.nDrawsFinal = summ.n
        result.nDrawsTotal = staged.log.reduce(0) { $0 + $1.n }
        result.nDrawsAllRuns = result.nDrawsTotal
        result.refinementStages = staged.log
        result.mcScen = summ.names.indices.map { (summ.names[$0], [summ.ess[$0], summ.relVar[$0].squareRoot(), summ.tailScale[$0]]) }
        result.mcMinESS = summ.minESS
        result.mcMinESSScenario = summ.minESSScenario
        result.mcPolicy = mcPolicy
        result.borderline = borderline
        result.unstableOn = unstableOn
        result.fppUnstable = unstableOn && unstable
        result.mcESSDom = unstableOn ? domESS : .nan
        result.mcDominantScenario = unstableOn ? domName : ""
        result.fppInterval = unstableOn ? [summ.tailBoxFPP.0, summ.tailBoxFPP.1] : []
        result.borderlineReasons = borderline ? staged.assess.reasons : []
        return result
    }
}
