import Foundation

public struct FPPV3DilutionPlan: Equatable, Sendable {
    public var conventions: [Bool]
    public var unverified: Bool
}

public enum FPPV3Dispatch {
    public static func switchSet(parityValue: String?) -> (switches: FPPV3Switches, productionFold: Bool) {
        let parity = parityValue == "1"
        return (parity ? .parity : .production, !parity)
    }

    public static func normalizeLCSource(_ value: String?) -> String {
        let s = (value ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ["spoc", "tglc", "qlp", "mixed"].contains(s) ? s : "unknown"
    }
    public static func dilutionPlan(_ source: String?) -> FPPV3DilutionPlan {
        switch normalizeLCSource(source) {
        case "spoc", "tglc": return FPPV3DilutionPlan(conventions: [true], unverified: false)
        default: return FPPV3DilutionPlan(conventions: [true, false], unverified: true)
        }
    }
    public static func depthScale(_ source: String?, nearEdge: Bool?) -> Double {
        guard normalizeLCSource(source) == "tglc" else { return 1.0 }
        return 1.0 / (nearEdge == true ? 1.27 : 1.09)
    }
    public static func frameFluxratio(_ ratio: Double, targetRatio: Double, corrected: Bool) -> Double {
        corrected ? ratio / targetRatio : ratio
    }
    public static func chooseDilutionRun<T>(trueRun: T, falseRun: T?, fpp: (T) -> Double) -> (run: T, corrected: Bool) {
        guard let falseRun, fpp(falseRun) > fpp(trueRun) else { return (trueRun, true) }
        return (falseRun, false)
    }

    public static func capVerdict(_ verdict: inout String, diagnostics: FPPV3Diagnostics,
                                  populationCrosscheckVerified: Bool, dilutionUnverified: Bool,
                                  tglcNearEdgeUnknown: Bool, geometryPathEmulated: Bool,
                                  ffiReferenceUnvalidated: Bool, flags: inout [String]) {
        let additions: [(Bool, String)] = [
            (!populationCrosscheckVerified, "population_crosscheck_unverified"),
            (dilutionUnverified, "dilution_unverified"),
            (tglcNearEdgeUnknown, "tglc_near_edge_unknown"),
        ]
        for (active, name) in additions where active && !flags.contains(name) { flags.append(name) }
        let capped = diagnostics.keplerUnconvergedDraws > 0 || !populationCrosscheckVerified
            || dilutionUnverified || tglcNearEdgeUnknown || geometryPathEmulated || ffiReferenceUnvalidated
        if capped && verdict == "validated" { verdict = "candidate" }
    }

    /// Shared final verdict gate used by calculation and one-factor cap tests.
    public static func finalizeVerdict(
        _ verdict: String, flags: inout [String], diagnostics: FPPV3Diagnostics,
        populationCrosscheckVerified: Bool, dilutionUnverified: Bool,
        tglcNearEdgeUnknown: Bool, geometryPathEmulated: Bool, ffiReferenceUnvalidated: Bool,
        sigmaFitIllConditioned: Bool, exposureUnconverged: Bool,
        isTGLC: Bool, tglcNearEdge: Bool?
    ) -> String {
        var value = verdict
        capVerdict(&value, diagnostics: diagnostics,
            populationCrosscheckVerified: populationCrosscheckVerified,
            dilutionUnverified: dilutionUnverified, tglcNearEdgeUnknown: tglcNearEdgeUnknown,
            geometryPathEmulated: geometryPathEmulated,
            ffiReferenceUnvalidated: ffiReferenceUnvalidated, flags: &flags)
        if sigmaFitIllConditioned && value == "validated" { value = "candidate" }
        if exposureUnconverged && value == "validated" { value = "candidate" }
        if isTGLC && tglcNearEdge == true {
            if !flags.contains("tglc_near_edge") { flags.append("tglc_near_edge") }
            if value == "validated" { value = "candidate" }
        }
        return value
    }

    public static func isProductionMode(_ parityValue: String? = nil) -> Bool {
        let p = parityValue ?? ProcessInfo.processInfo.environment["FPP_PARITY"]
        return p != "1"
    }

    public static func parseEphemRefineSwitch(production: Bool? = nil) throws -> Bool {
        let prod = production ?? isProductionMode()
        guard let env = ProcessInfo.processInfo.environment["FPP_EPHEM_REFINE"] else {
            return prod
        }
        switch env {
        case "on": return true
        case "off": return false
        default:
            throw FPPV3Error("invalid_switch", "invalid switch value FPP_EPHEM_REFINE=\(env): must be 'on' or 'off'")
        }
    }

    public static func parseUnstableFlagSwitch(production: Bool? = nil) throws -> Bool {
        let prod = production ?? isProductionMode()
        guard let env = ProcessInfo.processInfo.environment["FPP_UNSTABLE_FLAG"] else {
            return prod
        }
        switch env {
        case "on": return true
        case "off": return false
        default:
            throw FPPV3Error("invalid_switch", "invalid switch value FPP_UNSTABLE_FLAG=\(env): must be 'on' or 'off'")
        }
    }

    public static func switchDictionary(
        _ switches: FPPV3Switches,
        ephemRefine: Bool? = nil,
        unstableFlag: Bool? = nil
    ) -> [String: String] {
        let c = switches.core
        let isProd = switches == .production
        let refineOn: Bool
        if let ephemRefine {
            refineOn = ephemRefine
        } else {
            refineOn = (try? parseEphemRefineSwitch(production: isProd)) ?? isProd
        }
        let unstableOn: Bool
        if let unstableFlag {
            unstableOn = unstableFlag
        } else {
            unstableOn = (try? parseUnstableFlagSwitch(production: isProd)) ?? isProd
        }
        return [
            "FPP_SECONDARY_GEOMETRY": c.secondaryGeometryConsistent ? "consistent" : "upstream",
            "FPP_ORBIT_PHYSICS": c.orbitPhysicsRoche ? "roche" : "upstream",
            "FPP_K1_NUDGE": c.k1NudgeUpstream ? "upstream" : "exact",
            "FPP_Q_BRANCH": c.qBranchUpstream ? "upstream" : "correct",
            "FPP_EXPOSURE": c.exposureUpstream2MinX20 ? "upstream_2min_x20" : "cadence",
            "FPP_NORM": c.normData ? "data" : "upstream",
            "FPP_BACKGROUND_INDEX": switches.backgroundIndex,
            "FPP_BACKGROUND_PRIOR": switches.backgroundPrior,
            "FPP_BACKGROUND_Q": switches.backgroundQ,
            "FPP_GEOMETRY": switches.geometry,
            "FPP_ADAPTIVE_N": switches.adaptiveN,
            "FPP_EPHEM_REFINE": refineOn ? "on" : "off",
            "FPP_UNSTABLE_FLAG": unstableOn ? "on" : "off",
        ]
    }
}
