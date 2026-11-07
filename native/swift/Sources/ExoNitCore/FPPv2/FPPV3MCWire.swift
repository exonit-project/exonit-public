import Foundation

/// `worker._fpp_result_model_fields` for the step-7 MC fields (fpp-integration-r2 worker.py :606-677), literally;
/// mirror of the Rust `task_processor::apply_fpp_mc_wire_fields`:
/// `fpp_n_draws <- n_draws_final`, `fpp_n_draws_total <- n_draws_total`, `mc_borderline` only when `mc_policy`,
/// `mc_detail = {stages, borderline_reasons, min_ess_scenario, n_draws_all_runs, mc_scen}` with NaN/inf -> null,
/// non-finite top-level `fpp_mc_se` / `nfpp_mc_se` / `mc_min_ess` omitted, and the 64 KiB trimming measured on the
/// CPython canonical JSON text (`FPPV3PyJSON.canonical`).
public enum FPPV3MCWire {
    public static let mcDetailMaxBytes = 64 * 1024

    /// The engine-result keys the Python helper reads.
    public struct Input: Sendable {
        public var fppMCSE: Double, nfppMCSE: Double, mcMinESS: Double
        public var nDrawsFinal: Int, nDrawsTotal: Int, mcPolicy: Bool, borderline: Bool
        public var refinementStages: [FPPV3MC.StageRecord], borderlineReasons: [String]
        public var mcMinESSScenario: String, nDrawsAllRuns: Int, mcScen: [(String, [Double])]
        public var unstableOn: Bool, mcESSDom: Double, mcDominantScenario: String, fppInterval: [Double]
        public init(fppMCSE: Double, nfppMCSE: Double, mcMinESS: Double, nDrawsFinal: Int, nDrawsTotal: Int,
                    mcPolicy: Bool, borderline: Bool, refinementStages: [FPPV3MC.StageRecord],
                    borderlineReasons: [String], mcMinESSScenario: String, nDrawsAllRuns: Int,
                    mcScen: [(String, [Double])],
                    unstableOn: Bool = false, mcESSDom: Double = .nan,
                    mcDominantScenario: String = "", fppInterval: [Double] = []) {
            self.fppMCSE = fppMCSE; self.nfppMCSE = nfppMCSE; self.mcMinESS = mcMinESS
            self.nDrawsFinal = nDrawsFinal; self.nDrawsTotal = nDrawsTotal; self.mcPolicy = mcPolicy
            self.borderline = borderline; self.refinementStages = refinementStages
            self.borderlineReasons = borderlineReasons; self.mcMinESSScenario = mcMinESSScenario
            self.nDrawsAllRuns = nDrawsAllRuns; self.mcScen = mcScen
            self.unstableOn = unstableOn; self.mcESSDom = mcESSDom
            self.mcDominantScenario = mcDominantScenario; self.fppInterval = fppInterval
        }
        public init(result r: FPPV3CalcResult) {
            self.init(fppMCSE: r.fppMCSE, nfppMCSE: r.nfppMCSE, mcMinESS: r.mcMinESS, nDrawsFinal: r.nDrawsFinal,
                      nDrawsTotal: r.nDrawsTotal, mcPolicy: r.mcPolicy, borderline: r.borderline,
                      refinementStages: r.refinementStages, borderlineReasons: r.borderlineReasons,
                      mcMinESSScenario: r.mcMinESSScenario, nDrawsAllRuns: r.nDrawsAllRuns, mcScen: r.mcScen,
                      unstableOn: r.unstableOn, mcESSDom: r.mcESSDom,
                      mcDominantScenario: r.mcDominantScenario, fppInterval: r.fppInterval)
        }
    }

    /// `_json_safe`: a non-finite float inside mc_detail is posted as null.
    static func safe(_ x: Double) -> JSONValue { x.isFinite ? .number(x) : .null }

    public static func stageJSON(_ st: FPPV3MC.StageRecord) -> JSONValue {
        .object([
            "stage": .integer(Int64(st.stage)), "n": .integer(Int64(st.n)), "seed": .integer(Int64(st.seed)),
            "fpp": safe(st.fpp), "nfpp": safe(st.nfpp), "fpp_mc_se": safe(st.fppMCSE), "nfpp_mc_se": safe(st.nfppMCSE),
            "min_ess": safe(st.minESS), "min_ess_scenario": .string(st.minESSScenario),
            "heavy_scenarios": .array(st.heavyScenarios.map(JSONValue.string)), "n_low_ess": .integer(Int64(st.nLowESS)),
            "tail_box_fpp": .array(st.tailBoxFPP.map(safe)), "tail_box_nfpp": .array(st.tailBoxNFPP.map(safe)),
            "resolved": .bool(st.resolved), "reasons": .array(st.reasons.map(JSONValue.string)),
            "top_scenarios": .array(st.topScenarios.map {
                .object(["scenario": .string($0.scenario), "p": safe($0.p), "ess": safe($0.ess), "rel_se": safe($0.relSE)])
            }),
        ])
    }

    public static func apply(_ req: inout FPPResultRequest, _ r: Input) {
        req.fppMCSE = r.fppMCSE.isFinite ? r.fppMCSE : nil
        req.nfppMCSE = r.nfppMCSE.isFinite ? r.nfppMCSE : nil
        req.mcMinESS = r.mcMinESS.isFinite ? r.mcMinESS : nil
        req.fppNDraws = r.nDrawsFinal
        req.fppNDrawsTotal = r.nDrawsTotal
        req.mcBorderline = r.mcPolicy ? r.borderline : nil
        var scen = [String: JSONValue]()
        for (name, v) in r.mcScen { scen[name] = .array(v.map(safe)) }
        var detail: [String: JSONValue] = [
            "stages": .array(r.refinementStages.map(stageJSON)),
            "borderline_reasons": .array(r.borderlineReasons.map(JSONValue.string)),
            "min_ess_scenario": .string(r.mcMinESSScenario),
            "n_draws_all_runs": .integer(Int64(r.nDrawsAllRuns)),
            "mc_scen": .object(scen),
        ]
        if r.unstableOn {
            detail["ess_dom"] = safe(r.mcESSDom)
            detail["dominant_scenario"] = .string(r.mcDominantScenario)
            detail["fpp_interval"] = .array(r.fppInterval.map(safe))
        }
        trim(&detail)
        req.mcDetail = .object(detail)
    }

    /// The 64 KiB trimming: drop `mc_scen` (and mark `truncated`), then the stages' `top_scenarios`.
    public static func trim(_ detail: inout [String: JSONValue]) {
        func tooBig(_ d: [String: JSONValue]) -> Bool { FPPV3PyJSON.canonical(.object(d)).utf8.count > mcDetailMaxBytes }
        guard tooBig(detail) else { return }
        detail.removeValue(forKey: "mc_scen")
        detail["truncated"] = .bool(true)
        guard tooBig(detail) else { return }
        if case let .array(stages)? = detail["stages"] {
            detail["stages"] = .array(stages.map { stage in
                guard case var .object(s) = stage else { return stage }
                s.removeValue(forKey: "top_scenarios")
                return .object(s)
            })
        }
        if tooBig(detail) {
            print("[FPP] mc_detail still over \(mcDetailMaxBytes) bytes after trimming — the server will reject it")
        }
    }
}

/// Byte-exact port of CPython `json.dumps(value, sort_keys=True, separators=(",", ":"))` (ensure_ascii=True) for
/// `JSONValue` trees; floats use `float.__repr__`.
public enum FPPV3PyJSON {
    /// CPython `repr(float)` (shortest round-trip digits; exponent form when the decimal point position is <= -4 or
    /// > 16; two-digit minimum exponent; `.0` on integral values).
    public static func floatRepr(_ x: Double) -> String {
        if x.isNaN { return "NaN" }
        if x.isInfinite { return x > 0 ? "Infinity" : "-Infinity" }
        if x == 0 { return x.sign == .minus ? "-0.0" : "0.0" }
        // Swift's description prints the shortest round-trip digits; parse them back to (digits, decpt).
        var text = "\(abs(x))"
        var exp10 = 0
        if let eIndex = text.firstIndex(where: { $0 == "e" || $0 == "E" }) {
            exp10 = Int(text[text.index(after: eIndex)...])!
            text = String(text[..<eIndex])
        }
        let parts = text.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
        let intPart = parts[0], fracPart = parts.count > 1 ? parts[1] : ""
        var digits = intPart + fracPart
        var decpt = intPart.count + exp10
        while digits.hasPrefix("0") && digits.count > 1 { digits.removeFirst(); decpt -= 1 }
        while digits.hasSuffix("0") && digits.count > 1 { digits.removeLast() }
        var out = x < 0 ? "-" : ""
        if decpt <= -4 || decpt > 16 {
            out += String(digits.prefix(1))
            if digits.count > 1 { out += "." + digits.dropFirst() }
            let e = decpt - 1
            out += "e" + (e < 0 ? "-" : "+") + String(format: "%02d", abs(e))
        } else if decpt <= 0 {
            out += "0." + String(repeating: "0", count: -decpt) + digits
        } else if decpt >= digits.count {
            out += digits + String(repeating: "0", count: decpt - digits.count) + ".0"
        } else {
            let i = digits.index(digits.startIndex, offsetBy: decpt)
            out += digits[..<i] + "." + digits[i...]
        }
        return out
    }

    static func appendString(_ s: String, _ out: inout String) {
        out += "\""
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if scalar.value < 0x20 || scalar.value > 0x7e {
                    for unit in String(scalar).utf16 { out += String(format: "\\u%04x", unit) }
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        out += "\""
    }

    static func append(_ v: JSONValue, _ out: inout String) {
        switch v {
        case .null: out += "null"
        case let .bool(b): out += b ? "true" : "false"
        case let .integer(i): out += String(i)
        case let .number(x): out += floatRepr(x)
        case let .string(s): appendString(s, &out)
        case let .array(a):
            out += "["
            for (i, x) in a.enumerated() { if i > 0 { out += "," }; append(x, &out) }
            out += "]"
        case let .object(o):
            out += "{"
            // Python sorts str keys by code point; compare the unicode scalars, not Swift's String ordering
            let keys = o.keys.sorted { $0.unicodeScalars.lexicographicallyPrecedes($1.unicodeScalars) }
            for (i, k) in keys.enumerated() {
                if i > 0 { out += "," }
                appendString(k, &out); out += ":"; append(o[k]!, &out)
            }
            out += "}"
        }
    }

    public static func canonical(_ v: JSONValue) -> String {
        var out = ""
        append(v, &out)
        return out
    }
}
