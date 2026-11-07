import Foundation

/// v3 FPP gate DUMP driver (Swift port), the twin of the Rust `fpp_gate_dump` binary: reads one staged gate-input case
/// (`scripts/fpp/v3_gate_format.py`, schema `fpp-v3-gate-input/1`), runs `calcFPPBySourceV3` as the Python reference
/// generator (`scripts/fpp/gen_v3_gate_dumps.py`) runs `calc_fpp_by_source` (production switch set, FPP_EPHEM_REFINE
/// off in this port, fixed N: `adaptive_n = False`, the staged seed and star list), and writes the dump directory
/// `<out>/<tic>/<switch_set>/N<N>/` in the generator's format.
///
/// `--profile production-full` (gate-release-r1, docs/reviews/gate-release-r1/spec.md sections 4, 5, 10 WI-4) runs the production
/// worker path of `NativeFPPTaskProcessor.calculateFPP` on the staged case: the case's `t0_convention` goes through
/// `FPPV3Ephem.fppEphemeris` (refinement per `parseEphemRefineSwitch(production: true)`, env `FPP_EPHEM_REFINE=on` or unset),
/// the fold uses the refined (P, t0), `seedEphemeris` is the input (P, t0) whenever the engine returned provenance,
/// `calcFPPBySourceV3` runs the real 1e5 -> 1e6 ladder (`FPPV3MCOptions()`), and the dump carries a `production` block: the
/// ephemeris provenance, the stage log, the wire payload built by `FPPV3ResultMapper.map`, the switch dict and the producer.
/// `FPP_PARITY` and `FPP_UNSTABLE_FLAG` must be unset. Without `--profile` the kit behaviour (refinement off, fixed N) is unchanged.
///
/// BLIND RULE: a case whose `set` is not `dev` is refused unless `--gate-run` is given (the pre-registered gate run).
///   fpp_gate_dump --case STAGE/cases/<tic>.json --out DUMPS --N 100000 [--switch-set production|parity]
///                 [--profile kit|production-full] [--contacts-window 41] [--gate-run] [--require-device-roots]
/// Exit: 0 = dump written (any status), 2 = usage / unreadable input, 3 = observer inconsistency.
public enum FPPV3GateDump {
    enum ObserverError: Error, CustomStringConvertible {
        case invalidCode(String)
        var description: String {
            switch self { case let .invalidCode(message): return message }
        }
    }

    static func hx(_ x: Double) -> String {
        let s = String(x.bitPattern, radix: 16)
        return String(repeating: "0", count: 16 - s.count) + s
    }

    static func unhx(_ v: Any?) -> Double {
        Double(bitPattern: UInt64((v as! String), radix: 16)!)
    }

    static func isHex16(_ s: String) -> Bool {
        s.utf8.count == 16 && s.utf8.allSatisfy { ($0 >= 48 && $0 <= 57) || ($0 >= 97 && $0 <= 102) }
    }

    static func decodeStar(_ v: JSONValue) -> JSONValue {
        guard case let .object(o) = v else { return v }
        var out = [String: JSONValue]()
        for (k, x) in o {
            if k != "ID", case let .string(s) = x, isHex16(s) {
                let d = Double(bitPattern: UInt64(s, radix: 16)!)
                out[k] = d.isFinite ? .number(d) : .null
            } else {
                out[k] = x
            }
        }
        return .object(out)
    }

    /// JSON text of a dump value (keys sorted, compact; the format has no float numbers besides `window_nat`).
    indirect enum J {
        case null, bool(Bool), int(Int64), num(Double), str(String), arr([J]), obj([(String, J)])
        func text() -> String {
            switch self {
            case .null: return "null"
            case let .bool(b): return b ? "true" : "false"
            case let .int(i): return String(i)
            case let .num(d): return d == d.rounded() && abs(d) < 1e15 ? String(format: "%.1f", d) : "\(d)"
            case let .str(s):
                var r = "\""
                for u in s.unicodeScalars {
                    switch u {
                    case "\"": r += "\\\""
                    case "\\": r += "\\\\"
                    case "\n": r += "\\n"
                    default:
                        if u.value < 0x20 || u.value > 0x7e { r += String(format: "\\u%04x", u.value) } else { r.unicodeScalars.append(u) }
                    }
                }
                return r + "\""
            case let .arr(a): return "[" + a.map { $0.text() }.joined(separator: ",") + "]"
            case let .obj(o):
                return "{" + o.sorted { $0.0 < $1.0 }.map { J.str($0.0).text() + ":" + $0.1.text() }.joined(separator: ",") + "}"
            }
        }
    }

    static func fromJSONValue(_ v: JSONValue) -> J {
        switch v {
        case .null: return .null
        case let .bool(b): return .bool(b)
        case let .integer(i): return .int(i)
        case let .number(d): return .num(d)
        case let .string(s): return .str(s)
        case let .array(a): return .arr(a.map(fromJSONValue))
        case let .object(o): return .obj(o.map { ($0.key, fromJSONValue($0.value)) })
        }
    }

    final class Writer {
        let dir: URL
        init(_ dir: URL) { self.dir = dir }
        func array(_ rel: String, _ bytes: [UInt8], _ n: Int) throws -> J {
            let url = dir.appendingPathComponent("bin").appendingPathComponent(rel)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(bytes).write(to: url)
            return .obj([("file", .str("bin/" + rel)), ("sha256", .str(SHA256Swift.hex(bytes))), ("n", .int(Int64(n)))])
        }
        func f64s(_ rel: String, _ a: [Double]) throws -> J {
            var b = [UInt8](); b.reserveCapacity(a.count * 8)
            for x in a { withUnsafeBytes(of: x.bitPattern.littleEndian) { b.append(contentsOf: $0) } }
            return try array(rel, b, a.count)
        }
        func i8s(_ rel: String, _ a: [Int8]) throws -> J { try array(rel, a.map { UInt8(bitPattern: $0) }, a.count) }
        func i64s(_ rel: String, _ a: [Int64]) throws -> J {
            var b = [UInt8](); b.reserveCapacity(a.count * 8)
            for x in a { withUnsafeBytes(of: x.littleEndian) { b.append(contentsOf: $0) } }
            return try array(rel, b, a.count)
        }
        func i32s(_ rel: String, _ a: [Int32]) throws -> J {
            var b = [UInt8](); b.reserveCapacity(a.count * 4)
            for x in a { withUnsafeBytes(of: x.littleEndian) { b.append(contentsOf: $0) } }
            return try array(rel, b, a.count)
        }
        func i8sChecked(_ rel: String, _ a: [Int]) throws -> J {
            let bytes: [Int8] = try a.map { x in
                guard let byte = Int8(exactly: x) else {
                    throw ObserverError.invalidCode("gate dump \(rel): value \(x) does not fit the i8 file")
                }
                return byte
            }
            return try i8s(rel, bytes)
        }
    }

    /// Python's result dict holds a non-finite lnZ as None (JSON null): -inf (never reached / zero viability) and NaN.
    static func windowJ(_ w: Double) -> J { w.isInfinite ? .str("inf") : .num(w) }

    static func lnZJ(_ z: Double) -> J { z.isFinite ? .str(hx(z)) : .null }

    static func scenarioEntry(_ w: Writer, _ base: String, _ s: FPPV3GateRun.Scenario, _ window: Double) throws -> J {
        var e: [(String, J)] = [("name", .str(s.name)), ("lnZ", lnZJ(s.lnZ)), ("prob", .str(hx(s.probability)))]
        var lnl: J = .null, state: J = .null, contacts: J = .null
        if let r = s.record {
            lnl = try w.f64s("\(base).lnl.f64", r.lnl)
            if let st = r.state { state = try w.i8s("\(base).state.i8", st) }
            if let c = r.contacts {
                if c.idx.isEmpty {
                    contacts = .obj([("idx_file", .null), ("roots_file", .null), ("n", .int(0)), ("window_nat", windowJ(window))])
                } else {
                    let ie = try w.i64s("\(base).cidx.i64", c.idx)
                    let re = try w.f64s("\(base).croots.f64", c.roots)
                    var fields: [(String, J)] = [("idx_file", ie), ("roots_file", re), ("n", .int(Int64(c.idx.count))),
                                                  ("window_nat", windowJ(window))]
                    // New files only; selection and row order are those of cidx/croots.
                    if let t = r.tangency {
                        fields.append(("tangency_file", try w.f64s("\(base).ctang.f64", t)))
                    }
                    if let d = r.discrete {
                        fields.append(("order_file", try w.i8sChecked("\(base).corder.i8", d.order)))
                        fields.append(("bins_off_file", try w.i64s("\(base).cbinsoff.i64", d.binsOff)))
                        fields.append(("bins_file", try w.i32s("\(base).cbins.i32", d.bins)))
                        fields.append(("levels_file", try w.i8sChecked("\(base).clevels.i8", d.levels)))
                    }
                    contacts = .obj(fields)
                }
            }
            e += [("input_digests", .arr(r.inputDigests.map { .str($0) })), ("roots_source", .str(r.rootsSource))]
        }
        e += [("lnl", lnl), ("state", state), ("contacts", contacts)]
        return .obj(e)
    }

    /// The compiler of THIS build (a compile-time stamp, not a run-time claim): the gate pins Swift 6.3.3.
    static let compilerStamp: String = {
        #if compiler(>=6.3.4)
        return "swift>=6.3.4"
        #elseif compiler(>=6.3.3)
        return "swift-6.3.3"
        #else
        return "swift<6.3.3"
        #endif
    }()

    // ---- production-full: the `production` block (spec section 5, item 2); floats as u64 bits like the rest of the dump ----

    static func hxOpt(_ x: Double?) -> J { x.map { .str(hx($0)) } ?? .null }

    /// `JSONValue` (the encoded wire DTO) -> dump encoding: every JSON float becomes its u64 bits; integers stay integers.
    static func bitsFromJSONValue(_ v: JSONValue) -> J {
        switch v {
        case .null: return .null
        case let .bool(b): return .bool(b)
        case let .integer(i): return .int(i)
        case let .number(d): return .str(hx(d))
        case let .string(s): return .str(s)
        case let .array(a): return .arr(a.map(bitsFromJSONValue))
        case let .object(o): return .obj(o.map { ($0.key, bitsFromJSONValue($0.value)) })
        }
    }

    /// One `refinement_stages` entry with Python's `_mc_stage_record` keys (floats as bits).
    static func stageBits(_ st: FPPV3MC.StageRecord) -> J {
        .obj([("stage", .int(Int64(st.stage))), ("n", .int(Int64(st.n))), ("seed", .int(Int64(truncatingIfNeeded: st.seed))),
              ("fpp", .str(hx(st.fpp))), ("nfpp", .str(hx(st.nfpp))), ("fpp_mc_se", .str(hx(st.fppMCSE))),
              ("nfpp_mc_se", .str(hx(st.nfppMCSE))), ("min_ess", .str(hx(st.minESS))),
              ("min_ess_scenario", .str(st.minESSScenario)), ("heavy_scenarios", .arr(st.heavyScenarios.map { .str($0) })),
              ("n_low_ess", .int(Int64(st.nLowESS))), ("tail_box_fpp", .arr(st.tailBoxFPP.map { .str(hx($0)) })),
              ("tail_box_nfpp", .arr(st.tailBoxNFPP.map { .str(hx($0)) })), ("resolved", .bool(st.resolved)),
              ("reasons", .arr(st.reasons.map { .str($0) })),
              ("top_scenarios", .arr(st.topScenarios.map {
                  .obj([("scenario", .str($0.scenario)), ("p", .str(hx($0.p))), ("ess", .str(hx($0.ess))),
                        ("rel_se", .str(hx($0.relSE)))])
              }))])
    }

    /// The ephemeris provenance with Python's key names, `ephem_refine_failed` always explicit; nil provenance = the
    /// engine added nothing (refinement off).
    static func ephemBits(_ prov: FPPV3Ephem.EphemProvenance?, _ failed: Bool) -> J {
        guard let p = prov else { return .null }
        var refine: J = .null
        if let f = p.ephemRefine {
            refine = .obj([
                ("reason", f.reason.map { .str($0) } ?? .null), ("P_in", .str(hx(f.pIn))), ("t0_in", .str(hx(f.t0In))),
                ("t_ref_in", .str(hx(f.tRefIn))), ("cadence_s", hxOpt(f.cadenceS)),
                ("n_req", f.nReq.map { .int(Int64($0)) } ?? .null), ("n_orbits", hxOpt(f.nOrbits)),
                ("n_transits", .int(Int64(f.nTransits))), ("p_searched", .bool(f.pSearched)),
                ("p_free_accepted", .bool(f.pFreeAccepted)), ("delta_chi2_p", hxOpt(f.deltaChi2P)),
                ("dchi2_p_fixed", hxOpt(f.dchi2PFixed)), ("dchi2_p_free", hxOpt(f.dchi2PFree)),
                ("p_fixed", f.pFixed.map { .bool($0) } ?? .null), ("dP", .str(hx(f.dp))),
                ("max_offset_h", .str(hx(f.maxOffsetH))), ("dchi2_initial", hxOpt(f.dchi2Initial)),
                ("dchi2_refined", hxOpt(f.dchi2Refined)), ("delta_chi2", hxOpt(f.deltaChi2)), ("sigma", hxOpt(f.sigma)),
                ("t14_factor", hxOpt(f.t14Factor)), ("ingress", hxOpt(f.ingress)), ("edge_P", .bool(f.edgeP)),
                ("edge_t0", .bool(f.edgeT0)), ("dP_half", hxOpt(f.dpHalf)), ("dt_half", hxOpt(f.dtHalf)),
            ])
        }
        return .obj([("ephem_refined", .bool(p.ephemRefined)), ("ephem_status", .str(p.ephemStatus)),
                     ("ephem_t0_convention", .str(p.ephemT0Convention)), ("ephem_P", .str(hx(p.ephemP))),
                     ("ephem_t0", .str(hx(p.ephemT0))), ("ephem_shift_h", .str(hx(p.ephemShiftH))),
                     ("ephem_refine_failed", .bool(failed)), ("ephem_refine", refine)])
    }

    /// Host-provenance keys of the wire payload: never part of the comparison (spec section 5).
    static let wireHostKeys = ["fpp_backend", "fpp_backend_device", "fpp_platform", "fpp_libm", "fpp_build_profile"]

    /// What the production worker would POST for this result, built by the production mapping function
    /// (`FPPV3ResultMapper.map`, the function `NativeFPPTaskProcessor.calculateFPP` returns), minus the host keys.
    static func wireBits(period: Double, t0: Double, duration: Double, depth: Double, tic: String, lcSource: String,
                         nearEdge: JSONValue?, result: FPPV3CalcResult, prov: FPPV3Ephem.EphemProvenance?,
                         failed: Bool) throws -> J {
        var taskJ: [String: JSONValue] = [
            "candidate_id": .integer(0), "tic_id": .string(tic), "candidate_num": .integer(1), "period_days": .number(period),
            "t0": .number(t0), "duration_days": .number(duration), "depth_ppm": .number(depth), "snr": .number(0.0),
            "lc_source": .string(lcSource)]
        if let nearEdge { taskJ["tglc_near_edge"] = nearEdge }
        let task = try JSONDecoder().decode(BackfillTask.self, from: JSONEncoder().encode(JSONValue.object(taskJ)))
        let req = FPPV3ResultMapper.map(task: task, result: result, loadedLCSource: lcSource, ephemProv: prov, refineFailed: failed)
        var v = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(req))
        if case var .object(o) = v {
            for k in wireHostKeys { o[k] = nil }
            v = .object(o)
        }
        return bitsFromJSONValue(v)
    }

    public static func main(_ args: [String]) -> Int32 {
        func arg(_ k: String) -> String? { args.firstIndex(of: k).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }
        let metal = args.contains("--metal")
        let bench = args.contains("--bench")
        let requireDeviceRoots = args.contains("--require-device-roots")
        if metal && !bench && !requireDeviceRoots {
            FileHandle.standardError.write("--metal dump requires --require-device-roots\n".data(using: .utf8)!)
            return 2
        }
        #if !canImport(Metal)
        if metal {
            FileHandle.standardError.write("--metal requires a build with Metal support\n".data(using: .utf8)!)
            return 2
        }
        #endif
        guard let casePath = arg("--case"), let outRoot = arg("--out") else {
            FileHandle.standardError.write("usage: fpp_gate_dump --case CASE.json --out DUMPS --N 100000 [--switch-set production] [--gate-run] [--require-device-roots] [--metal] [--bench]\n".data(using: .utf8)!)
            return 2
        }
        let nDraws = Int(arg("--N") ?? "100000")!
        let switchSet = arg("--switch-set") ?? "production"
        let window = Double(arg("--contacts-window") ?? "41.0")!
        let gateRun = args.contains("--gate-run")
        guard switchSet == "production" || switchSet == "parity" else { return 2 }
        let production = switchSet == "production"
        let profile = arg("--profile") ?? "kit"
        guard profile == "kit" || profile == "production-full" else {
            FileHandle.standardError.write("unknown profile \(profile)\n".data(using: .utf8)!)
            return 2
        }
        let full = profile == "production-full"
        if full {
            // the production worker exports no FPP_PARITY / FPP_UNSTABLE_FLAG, and FPP_EPHEM_REFINE is exactly "on" (the
            // deploy sets it in every Rust / Swift entry of ops/workers.json) or unset (= the production default, also "on")
            let env = ProcessInfo.processInfo.environment
            let stray = ["FPP_PARITY", "FPP_UNSTABLE_FLAG"].filter { env[$0] != nil }
            let refineOK = env["FPP_EPHEM_REFINE"].map { $0 == "on" } ?? true
            if !production || !stray.isEmpty || !refineOK {
                FileHandle.standardError.write(("production-full: needs --switch-set production, FPP_PARITY / FPP_UNSTABLE_FLAG unset "
                    + "and FPP_EPHEM_REFINE unset or 'on' (stray: \(stray), refineOK: \(refineOK))\n").data(using: .utf8)!)
                return 2
            }
        }
        guard let raw = FileManager.default.contents(atPath: casePath),
              let c = (try? JSONSerialization.jsonObject(with: raw)) as? [String: Any],
              let cv = try? JSONDecoder().decode(JSONValue.self, from: raw), case let .object(co) = cv,
              case let .object(inpV)? = co["inputs"] else {
            FileHandle.standardError.write("\(casePath): unreadable case\n".data(using: .utf8)!)
            return 2
        }
        guard (c["schema"] as? String) == "fpp-v3-gate-input/1" else { return 2 }
        if (c["set"] as? String) != "dev" && !gateRun {
            FileHandle.standardError.write("held-out case refused (blind rule; --gate-run is for the gate run only)\n".data(using: .utf8)!)
            return 2
        }
        if let open = c["open"] as? [Any], !open.isEmpty { return 2 }
        let tic = (c["tic"] as? NSNumber).map { $0.stringValue } ?? String(describing: c["tic"]!)
        let inp = c["inputs"] as! [String: Any]
        let stars: [JSONValue] = { if case let .array(a)? = inpV["stars"] { return a.map(decodeStar) } else { return [] } }()
        func doc(_ k: String) -> JSONValue? { if let v = inpV[k], v != .null { return v } else { return nil } }
        let nearEdge: JSONValue? = (c["tglc_near_edge"] as? Bool).map { .bool($0) }
        let seed = UInt32((c["seed"] as! NSNumber).uint64Value)
        let started = Date()
        let (pIn, t0In, durIn, depthIn) = (unhx(inp["period"]), unhx(inp["t0"]), unhx(inp["duration_d"]), unhx(inp["depth_ppm"]))
        let timeArr = (inp["time"] as! [Any]).map { unhx($0) }
        let fluxArr = (inp["flux"] as! [Any]).map { unhx($0) }
        let cadenceS: Double? = (c["cadence_s"] is NSNull || c["cadence_s"] == nil) ? nil : unhx(c["cadence_s"])
        // production-full: NativeFPPTaskProcessor.calculateFPP step for step (ephemeris, seed from the INPUT epoch, seedEphemeris)
        var ephemRes: FPPV3Ephem.FppEphemerisResult?
        var refineOn = false
        if full {
            guard let conv = FPPV3Ephem.T0Convention(rawValue: (c["t0_convention"] as? String) ?? "") else {
                FileHandle.standardError.write("\(casePath): bad t0_convention\n".data(using: .utf8)!)
                return 2
            }
            do { refineOn = try FPPV3Dispatch.parseEphemRefineSwitch(production: true) } catch {
                FileHandle.standardError.write("\(error)\n".data(using: .utf8)!)
                return 2
            }
            let want = FPPRNG.ticSeed(ticID: tic, period: pIn, t0: t0In)
            guard want == seed else {
                FileHandle.standardError.write("TIC \(tic): staged seed \(seed) != tic_seed \(want)\n".data(using: .utf8)!)
                return 2
            }
            ephemRes = FPPV3Ephem.fppEphemeris(time: timeArr, flux: fluxArr, period: pIn, t0: t0In, duration: durIn,
                                               convention: conv, refineEnabled: refineOn, cadenceS: cadenceS)
        }
        var input = FPPV3CalcInputs(
            ticID: tic, period: ephemRes?.pUse ?? pIn, t0: ephemRes?.t0Use ?? t0In, durationDays: durIn,
            depthPPM: depthIn, time: timeArr, flux: fluxArr,
            nearby: ["stars": .array(stars), "gal_b": .number(unhx(c["gal_b"]))],
            geometry: doc("geometry"), population: doc("population"),
            populationSHA256: inp["population_sha256"] as? String, populationVersion: inp["population_version"] as? String,
            populationCrosscheck: doc("population_crosscheck"),
            cadenceSeconds: cadenceS,
            lcSource: c["lc_source"] as? String, tglcNearEdge: nearEdge)
        if ephemRes?.provenance != nil { input.seedEphemeris = (period: pIn, t0: t0In) }
        // N2 go/no-go: `--bench` times the whole calculation with no recorder and writes no dump; with `--metal` the
        // likelihood runs on the GPU route. Prints one line and exits. Not a gate path.
        if bench {
            let t = Date()
            var probeS: Double = 0.0
            #if canImport(Metal)
            if metal {
                let tp = Date()
                do {
                    let backend = try FPPV3MetalBackend()
                    let fixture = FPPV3CanaryLoader()
                    FPPV3MetalCanary.install(on: backend, fixture: fixture)
                    try backend.canary?(backend)
                    FPPV3LikelihoodRouter.install(backend)
                } catch {
                    FileHandle.standardError.write("Metal backend probe failed: \(error)\n".data(using: .utf8)!)
                    return 2
                }
                probeS = Date().timeIntervalSince(tp)
            }
            #endif
            let outcome: Result<FPPV3CalcResult, Error>
            do {
                outcome = .success(try FPPV3Calculation.calcFPPBySourceV3(
                    input, draws: nDraws, seed: seed, productionFold: production,
                    switches: production ? .production : .parity, options: full ? FPPV3MCOptions() : FPPV3MCOptions(adaptiveN: false)))
            } catch {
                outcome = .failure(error)
            }
            let calc = Date().timeIntervalSince(t) - probeS
            let whole = Date().timeIntervalSince(started) - probeS
            switch outcome {
            case let .success(r):
                print(String(format: "BENCH tic=%@ calc_s=%.2f whole_s=%.2f fpp=%e nfpp=%e verdict=%@ stages=%ld",
                             tic, calc, whole, r.fpp, r.nfpp, r.verdict, r.refinementStages.count))
            case let .failure(e):
                print(String(format: "BENCH tic=%@ calc_s=%.2f whole_s=%.2f error=%@",
                             tic, calc, whole, "\(e)"))
            }
            return 0
        }
        let dir = URL(fileURLWithPath: outRoot).appendingPathComponent(tic).appendingPathComponent(switchSet)
            .appendingPathComponent("N\(nDraws)")
        try? FileManager.default.removeItem(at: dir.appendingPathComponent("bin"))
        try! FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var backendID = "cpu-swift"
        var backendDevice: J = .null
        var gpuKernelVersion: J = .null
        #if canImport(Metal)
        if metal {
            do {
                let backend = try FPPV3MetalBackend()
                let fixture = FPPV3CanaryLoader()
                FPPV3MetalCanary.install(on: backend, fixture: fixture)
                try backend.canary?(backend)
                FPPV3LikelihoodRouter.install(backend)
                backendID = backend.fppBackendID
                backendDevice = .str(backend.backendDevice)
                gpuKernelVersion = .str(backend.gpuKernelVersion)
            } catch {
                FileHandle.standardError.write("Metal backend failed: \(error)\n".data(using: .utf8)!)
                return 2
            }
        }
        #endif
        let recorder = FPPV3Gate.Recorder(production: production, window: window, requireDeviceRoots: requireDeviceRoots)
        var outcome: Result<FPPV3CalcResult, Error>
        do {
            outcome = .success(try FPPV3Calculation.calcFPPBySourceV3(
                input, draws: nDraws, seed: seed, productionFold: production,
                switches: production ? .production : .parity, options: full ? FPPV3MCOptions() : FPPV3MCOptions(adaptiveN: false)))
        } catch {
            outcome = .failure(error)
        }
        let observerError = recorder.error
        recorder.close()
        let runtime = Date().timeIntervalSince(started)
        if let observerError {
            FileHandle.standardError.write("TIC \(tic): gate observer inconsistency: \(observerError)\n".data(using: .utf8)!)
            return 3
        }
        let w = Writer(dir)
        var status = "ok", skip: J = .null, result: J = .null, gates: J = .null, runs: [J] = []
        var errorText: J = .null
        var productionBlock: [(String, J)] = []
        do {
            switch outcome {
            case let .success(r):
                if full, let e = ephemRes {
                    // the library keeps the stage log of the reported run only (the SPOC / TGLC path has exactly one run)
                    let stages: [J] = r.gateRuns.map { run in
                        run.dilutionCorrected == r.dilutionCorrected ? J.arr(r.refinementStages.map(stageBits)) : J.null
                    }
                    productionBlock = [
                        ("schema", .str("fpp-v3-gate-production/1")), ("ephem", ephemBits(e.provenance, e.refineFailed)),
                        ("stages", .arr(stages)),
                        ("wire", try wireBits(period: pIn, t0: t0In, duration: durIn, depth: depthIn, tic: tic,
                                              lcSource: (c["lc_source"] as? String) ?? "unknown", nearEdge: nearEdge,
                                              result: r, prov: e.provenance, failed: e.refineFailed)),
                    ]
                }
                for (ri, run) in r.gateRuns.enumerated() {
                    var scen = [J]()
                    for (si, s) in run.scenarios.enumerated() {
                        scen.append(try scenarioEntry(w, String(format: "run%d/s%03d", ri, si), s, window))
                    }
                    runs.append(.obj([("dilution_corrected", .bool(run.dilutionCorrected)), ("fpp", .str(hx(run.fpp))),
                                      ("nfpp", .str(hx(run.nfpp))), ("scenarios", .arr(scen))]))
                }
                var fs = [(String, J)]()
                for run in r.gateRuns where run.dilutionCorrected == r.dilutionCorrected {
                    for s in run.scenarios { fs.append((s.name, .obj([("lnZ", lnZJ(s.lnZ)), ("prob", .str(hx(s.probability)))]))) }
                }
                result = .obj([("fpp", .str(hx(r.fpp))), ("nfpp", .str(hx(r.nfpp))), ("verdict", .str(r.verdict)),
                               ("fpp_flags", .arr(r.flags.map { .str($0) })), ("fpp_scen", .obj(fs)),
                               ("dilution_corrected", .bool(r.dilutionCorrected)), ("fpp_seed", .int(Int64(r.seed))),
                               ("fpp_mc_se", .str(hx(r.fppMCSE))), ("nfpp_mc_se", .str(hx(r.nfppMCSE))),
                               ("mc_policy", .bool(r.mcPolicy)), ("borderline", .bool(r.borderline)),
                               ("borderline_reasons", .arr(r.borderlineReasons.map { .str($0) })),
                               ("n_draws_final", .int(Int64(r.nDrawsFinal))), ("n_draws_total", .int(Int64(r.nDrawsTotal)))])
                gates = .obj([("fpp_lt_0.015", .bool(r.fpp < 0.015)), ("fpp_lt_0.5", .bool(r.fpp < 0.5)),
                              ("nfpp_lt_0.001", .bool(r.nfpp < 0.001))])
            case let .failure(e):
                errorText = .str(String(describing: e))
                if let u = FPPV3ReleasePolicy.unavailableReason(for: e) {
                    status = "unavailable"; skip = .str(u)
                } else {
                    status = "skipped"; skip = FPPV3SkipPolicy.skipReason(for: e).map { .str($0) } ?? .null
                }
            }
        } catch {
            FileHandle.standardError.write("TIC \(tic): write failed: \(error)\n".data(using: .utf8)!)
            return error is ObserverError ? 3 : 2
        }
        let fppEnv = ProcessInfo.processInfo.environment.filter { $0.key.hasPrefix("FPP_") && $0.key != "FPP_GATE_GIT_SHA" }
        let producer = J.obj([("impl", .str("swift")), ("git_sha", .str(ProcessInfo.processInfo.environment["FPP_GATE_GIT_SHA"] ?? "unknown")),
                              ("compiler", .str(compilerStamp)),
                              ("platform", .str(FPPV3Provenance.platform)), ("libm", .str(FPPV3Provenance.libm)),
                              ("build_profile", .str(FPPV3Provenance.buildProfile)), ("fpp_backend", .str(backendID)),
                              ("fpp_backend_device", backendDevice), ("gpu_kernel_version", gpuKernelVersion),
                              ("fpp_env", .obj(fppEnv.map { ($0.key, .str($0.value)) })),
                              ("error", errorText)])
        let sw = FPPV3Dispatch.switchDictionary(production ? .production : .parity)
        let caseJ = J.obj([("tic", fromJSONValue(co["tic"] ?? .null)), ("toi", fromJSONValue(co["toi"] ?? .null)),
                           ("set", fromJSONValue(co["set"] ?? .null)), ("input_sha256", .str(SHA256Swift.hex([UInt8](raw))))])
        let switchesJ = J.obj(sw.map { ($0.key, .str($0.value)) })
        var dumpFields: [(String, J)] = [
            ("schema", .str("fpp-v3-gate-dump/1")), ("producer", producer), ("case", caseJ),
            ("switch_set", .str(switchSet)),
            ("production_profile", full ? .str("production-full") : (production ? .str("port-ephem-off") : .null)),
            ("switches", switchesJ), ("N", .int(Int64(nDraws))),
            ("seed", .int(Int64(seed))), ("status", .str(status)), ("skip_reason", skip),
            ("result", result), ("gates", gates), ("runs", .arr(runs))]
        if full {
            dumpFields.append(("production", productionBlock.isEmpty ? .null
                : .obj(productionBlock + [("switches", switchesJ), ("producer", producer)])))
        }
        let dump = J.obj(dumpFields)
        let tmp = dir.appendingPathComponent("dump.json.tmp")
        try! (dump.text() + "\n").data(using: .utf8)!.write(to: tmp)
        _ = try? FileManager.default.removeItem(at: dir.appendingPathComponent("dump.json"))
        try! FileManager.default.moveItem(at: tmp, to: dir.appendingPathComponent("dump.json"))
        let meta = J.obj([("runtime_s", .num(runtime)), ("impl", .str("swift")), ("build_profile", .str(FPPV3Provenance.buildProfile))])
        try! (meta.text() + "\n").data(using: .utf8)!.write(to: dir.appendingPathComponent("run_meta.json"))
        print("TIC \(tic) \(switchSet) N=\(nDraws): \(status) \(String(format: "%.1f", runtime))s")
        return 0
    }
}
