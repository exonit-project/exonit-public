import Foundation

/// Python worker preflight and exception mapping for tasks that cannot produce
/// a v3 FPP score. These reasons trigger fpp_release, never an FPP result.
enum FPPV3ReleasePolicy {
    /// Python worker `_population_provenance_missing` (7b30ba1, decision 2026-10-25 option A): true unless BOTH
    /// the sha256 and the version are non-empty strings.
    static func provenanceMissing(sha256: String?, version: String?) -> Bool {
        !((sha256.map { !$0.isEmpty } ?? false) && (version.map { !$0.isEmpty } ?? false))
    }

    /// The backfill preflight in the Python worker's order: population_missing, then the provenance
    /// (`population_invalid`), then the cross-check, then the alarm. No light curve is loaded before it.
    static func preflightReason(population: JSONValue?, crosscheck: JSONValue?,
                                populationSHA256: String?, populationVersion: String?) -> String? {
        guard case let .object(pop)? = population else { return "population_missing" }
        if provenanceMissing(sha256: populationSHA256, version: populationVersion) { return "population_invalid" }
        guard case let .object(check)? = crosscheck,
              case .bool(true)? = check["gaia_available"] else {
            return "population_crosscheck_unverified"
        }
        if truthy(check["alarm"]) {
            let header: [String: JSONValue] = {
                guard case let .object(value)? = pop["header"] else { return [:] }
                return value
            }()
            guard case .string("gaia")? = header["source"] else {
                return "population_alarm_unresolved"
            }
        }
        return nil
    }

    static func unavailableReason(for error: FPPV3Error) -> String? {
        switch error.reason {
        // The Python worker's FPPUnavailable skip reasons. `geometry_nodes_unavailable`
        // and `gl_tables_unavailable` cannot occur here: both tables are compiled in.
        case "population_missing", "population_invalid",
             "population_crosscheck_unverified", "population_alarm_unresolved",
             "population_alarm", "population_ldc_invalid":
            return error.reason
        default:
            return nil
        }
    }

    /// The release decision depends on the error TYPE and its typed reason,
    /// set where the condition is detected; the message text is never read.
    static func unavailableReason(for error: Error) -> String? {
        if let failure = error as? FPPV3GPUFailure { return gpuReason(for: failure) }
        if let error = error as? FPPV3Error { return unavailableReason(for: error) }
        return nil
    }

    /// The only two GPU reasons the server accepts (addendum v1.1 section 6).
    static let gpuReasons: Set<String> = ["unavailable", "gpu_parity_failed"]

    /// CLOSED switch: every case listed, no `default`, so a new GPU failure case
    /// cannot compile until it has a server reason.
    static func gpuReason(for failure: FPPV3GPUFailure) -> String {
        switch failure {
        case .noDevice, .missingFunction, .unsupportedSwitchSet,
             .compile, .commandFailed, .vramCapExceeded:
            return "unavailable"
        case .canaryFailed, .nonFinite:
            return "gpu_parity_failed"
        }
    }

    private static func truthy(_ value: JSONValue?) -> Bool {
        guard let value else { return false }
        switch value {
        case .null: return false
        case let .bool(value): return value
        case let .integer(value): return value != 0
        case let .number(value): return value != 0
        case let .string(value): return !value.isEmpty
        case let .array(value): return !value.isEmpty
        case let .object(value): return !value.isEmpty
        }
    }
}

/// A GPU failure the Swift FPP worker can detect. The set is closed: the release
/// mapping in `FPPV3ReleasePolicy.gpuReason(for:)` lists every case, so adding a
/// case without a server reason is a compile error. The later Metal card extends it.
public enum FPPV3GPUFailure: Error, Equatable, Sendable, CaseIterable, CustomStringConvertible {
    case noDevice             // no device, wrong device class or OS, driver missing
    case missingFunction      // Metal function not found in the library
    case unsupportedSwitchSet // non-production switch set
    case compile              // shader/pipeline compile failed
    case commandFailed        // command buffer error or device removed mid-task
    case vramCapExceeded      // out of memory or buffer allocation failed
    case canaryFailed         // canary ran but did not match the CPU reference
    case nonFinite            // non-finite device output (C5): refusal of the whole task

    /// For logs only; never used to decide a release.
    public var description: String {
        switch self {
        case .noDevice: return "FPPV3GPUFailure.noDevice"
        case .missingFunction: return "FPPV3GPUFailure.missingFunction"
        case .unsupportedSwitchSet: return "FPPV3GPUFailure.unsupportedSwitchSet"
        case .compile: return "FPPV3GPUFailure.compile"
        case .commandFailed: return "FPPV3GPUFailure.commandFailed"
        case .vramCapExceeded: return "FPPV3GPUFailure.vramCapExceeded"
        case .canaryFailed: return "FPPV3GPUFailure.canaryFailed"
        case .nonFinite: return "FPPV3GPUFailure.nonFinite"
        }
    }
}

/// The `skip_reason` of a `-1` post (task 33 item 3.4): Python's `calc_fpp_native.last_skip_reason`, from the error's
/// TYPE and its typed reason code (set where the condition is detected), never from message text. `nil` (posted as
/// JSON null) for every other failure. `ldc_table_missing` / `gl_tables_unavailable` cannot occur here: both tables
/// are compiled in.
enum FPPV3SkipPolicy {
    static let skipReasons: Set<String> = ["raw_lc_unusable", "giant_host", "stellar_params_missing", "ldc_no_row"]

    static func skipReason(for error: Error) -> String? {
        if error is FPPV2LDCNoRow { return "ldc_no_row" }
        if let error = error as? FPPV3Error, skipReasons.contains(error.reason) { return error.reason }
        return nil
    }
}
