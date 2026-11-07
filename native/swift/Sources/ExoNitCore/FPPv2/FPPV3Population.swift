import Foundation

public struct FPPV3BackgroundStar: Sendable, Equatable {
    public var tmag: Double
    public var mass: Double
    public var logg: Double
    public var logTe: Double
    public var metallicity: Double
    public var teff: Double
    public var radius: Double
    public var fluxRatio: Double

    public init(tmag: Double, mass: Double, logg: Double, logTe: Double, metallicity: Double,
                teff: Double, radius: Double, fluxRatio: Double) {
        self.tmag = tmag; self.mass = mass; self.logg = logg; self.logTe = logTe
        self.metallicity = metallicity; self.teff = teff; self.radius = radius
        self.fluxRatio = fluxRatio
    }
}

public struct FPPV3BackgroundPopulation: Sendable, Equatable {
    /// Inclusive target-fainter rows. This count drives the background prior.
    public var stars: [FPPV3BackgroundStar]
    public var source: String
    public var generator: String
    public var version: String
    public var mappingVersion: String
    public var sha256: String
    public var nComp: Int
    public var nCompFainter: Int

    public init(stars: [FPPV3BackgroundStar], source: String, generator: String, version: String,
                mappingVersion: String, sha256: String, nComp: Int, nCompFainter: Int) {
        self.stars = stars; self.source = source; self.generator = generator; self.version = version
        self.mappingVersion = mappingVersion; self.sha256 = sha256; self.nComp = nComp
        self.nCompFainter = nCompFainter
    }
}

public struct FPPV3Error: Error, Equatable, CustomStringConvertible, Sendable {
    public var reason: String
    public var description: String
    public init(_ reason: String, _ description: String) {
        self.reason = reason; self.description = description
    }
}

public struct FPPV3Diagnostics: Sendable {
    public var keplerUnconvergedDraws = 0
    public var exposureGLFallbackShare = 0.0
    public var exposureUnconvergedShare = 0.0
    public init() {}
}

public struct FPPV3Switches: Equatable, Sendable {
    public var core: FPPV2Switches
    public var backgroundIndex: String
    public var backgroundPrior: String
    public var backgroundQ: String
    public var geometry: String
    /// `FPP_ADAPTIVE_N` (step 7, decision C): `on` in production, `off` in parity. `FPP_EPHEM_REFINE` is not a
    /// field: the port does not implement the ephemeris refinement, so it is `off` in both sets.
    public var adaptiveN: String

    public init(core: FPPV2Switches, backgroundIndex: String, backgroundPrior: String,
                backgroundQ: String, geometry: String, adaptiveN: String = "off") {
        self.core = core; self.backgroundIndex = backgroundIndex; self.backgroundPrior = backgroundPrior
        self.backgroundQ = backgroundQ; self.geometry = geometry; self.adaptiveN = adaptiveN
    }
    public static let parity = FPPV3Switches(core: .parity, backgroundIndex: "upstream",
                                              backgroundPrior: "upstream", backgroundQ: "target", geometry: "upstream",
                                              adaptiveN: "off")
    public static let production = FPPV3Switches(core: .production, backgroundIndex: "correct",
                                                  backgroundPrior: "circular", backgroundQ: "target", geometry: "smooth",
                                                  adaptiveN: "on")
}

public struct FPPV3DFamilyConfig: Sendable {
    public var parityIndex: Bool
    public var circularPrior: Bool
    public var targetQ: Bool
    public var roche: Bool
    public var secondaryConsistent: Bool
    public var kNudgeUpstream: Bool
    public var exposureUpstream: Bool
    public var upstreamExposure: Bool { exposureUpstream }
    public init(parityIndex: Bool, circularPrior: Bool, targetQ: Bool, roche: Bool,
                secondaryConsistent: Bool, kNudgeUpstream: Bool, exposureUpstream: Bool) {
        self.parityIndex = parityIndex; self.circularPrior = circularPrior; self.targetQ = targetQ
        self.roche = roche; self.secondaryConsistent = secondaryConsistent
        self.kNudgeUpstream = kNudgeUpstream; self.exposureUpstream = exposureUpstream
    }
    public static func from(_ s: FPPV3Switches) -> FPPV3DFamilyConfig {
        FPPV3DFamilyConfig(parityIndex: s.backgroundIndex == "upstream",
                           circularPrior: s.backgroundPrior == "circular", targetQ: s.backgroundQ == "target",
                           roche: s.core.orbitPhysicsRoche, secondaryConsistent: s.core.secondaryGeometryConsistent,
                           kNudgeUpstream: s.core.k1NudgeUpstream, exposureUpstream: s.core.exposureUpstream2MinX20)
    }
}

public enum FPPV3Population {
    private static let expectedColumns = ["Tmag", "Mact", "logg", "logTe", "[M/H]"]

    public static func validate(_ document: JSONValue?, targetTmag: Double,
                                expectedSHA256: String? = nil, expectedVersion: String? = nil,
                                expectedTICID: String? = nil) throws -> FPPV3BackgroundPopulation {
        guard let document else { throw FPPV3Error("population_missing", "background population is missing") }
        guard targetTmag.isFinite else { throw invalid("invalid target magnitude") }
        guard case let .object(root) = document,
              case let .object(header)? = root["header"], case let .array(rawRows)? = root["rows"] else {
            throw invalid("population document must contain header and rows")
        }
        let source = try string(header["source"], "source")
        let isGalaxia = source == "galaxia-0.7.2" || source == "galaxia-rs-0.2.2"
        guard isGalaxia || source == "gaia" else { throw invalid("unsupported population source") }
        if let expectedTICID {
            guard try stringOrInteger(header["tic_id"], "tic_id") == expectedTICID else {
                throw invalid("population belongs to a different TIC")
            }
        }
        guard case let .array(cols)? = header["columns"], cols.count == expectedColumns.count,
              zip(cols, expectedColumns).allSatisfy({ value, expected in value.stringValue == expected }) else {
            throw invalid("population columns differ from contract")
        }
        let version = try string(header["population_version"], "population_version")
        let mapping = try string(header["mapping_version"], "mapping_version")
        if isGalaxia && mapping != "tmag-stassun2018-jks-cc1" && mapping != "tmag-stassun2018-jks-v1" {
            throw invalid("unknown Tmag mapping")
        }
        guard try number(header["field_area_deg2"], "field_area_deg2") == 0.1,
              try number(header["tmag_limit"], "tmag_limit") == 21.0 else {
            throw invalid("unsupported field area or magnitude limit")
        }
        if let expectedVersion, expectedVersion != version { throw invalid("population version mismatch") }

        let canonical = canonicalPopulationJSON(document)
        let digest = SHA256Swift.hex(Array(canonical.utf8))
        if let expectedSHA256, expectedSHA256 != digest { throw invalid("population sha256 mismatch") }
        guard let nComp = header["n_comp"]?.integerValue, nComp > 0, nComp == rawRows.count else {
            throw invalid("invalid population row count")
        }
        var rows = [[Double]]()
        rows.reserveCapacity(rawRows.count)
        for raw in rawRows {
            guard case let .array(items) = raw, items.count == 5 else { throw invalid("population row must contain five columns") }
            let values = try ["Tmag", "Mact", "logg", "logTe", "[M/H]"].enumerated().map { i, name in
                try number(items[i], name)
            }
            guard values[1] > 0.0, values[0] <= 21.0 else { throw invalid("invalid magnitude or stellar mass") }
            rows.append(values)
        }
        if rows.count > 1 {
            for i in 1..<rows.count {
                if lexicographicallyPrecedes(rows[i], rows[i - 1]) { throw invalid("population rows are not canonical order") }
            }
        }
        let filtered = rows.filter { $0[0] >= targetTmag }
        let stars = filtered.map { v -> FPPV3BackgroundStar in
            let teff = FPPV2.pow(10.0, v[3])
            let radius = ((FPPV2Consts.gCGS * v[1] * FPPV2Consts.msun) / FPPV2.pow(10.0, v[2])).squareRoot() / FPPV2Consts.rsun
            let rel = FPPV2.pow(10.0, (targetTmag - v[0]) / 2.5)
            let fr = rel / (1.0 + rel)
            return FPPV3BackgroundStar(tmag: v[0], mass: v[1], logg: v[2], logTe: v[3], metallicity: v[4],
                                       teff: teff, radius: radius, fluxRatio: fr)
        }
        guard !stars.isEmpty else { throw invalid("no target-fainter population rows") }
        return FPPV3BackgroundPopulation(stars: stars, source: source == "gaia" ? "gaia" : "galaxia",
                                         generator: source, version: version,
                                         mappingVersion: mapping, sha256: digest, nComp: nComp,
                                         nCompFainter: stars.count)
    }

    private static func invalid(_ message: String) -> FPPV3Error { FPPV3Error("population_invalid", message) }
    private static func number(_ value: JSONValue?, _ name: String) throws -> Double {
        let x: Double
        switch value { case let .number(v): x = v; case let .integer(v): x = Double(v); default: throw invalid("invalid \(name)") }
        guard x.isFinite else { throw invalid("invalid \(name)") }
        return x
    }
    private static func string(_ value: JSONValue?, _ name: String) throws -> String {
        guard case let .string(s)? = value, !s.isEmpty else { throw invalid("invalid \(name)") }
        return s
    }
    private static func stringOrInteger(_ value: JSONValue?, _ name: String) throws -> String {
        switch value { case let .string(s): return s; case let .integer(i): return String(i); default: throw invalid("invalid \(name)") }
    }
    private static func lexicographicallyPrecedes(_ a: [Double], _ b: [Double]) -> Bool {
        for i in 0..<min(a.count, b.count) { if a[i] != b[i] { return a[i] < b[i] } }
        return a.count < b.count
    }
    private static func canonicalJSON(_ value: JSONValue) -> String {
        canonicalJSON(value, floatNumeric: false)
    }
    private static func canonicalPopulationJSON(_ value: JSONValue) -> String {
        guard case let .object(root) = value else { return canonicalJSON(value) }
        return "{" + root.keys.sorted().map { key in
            let encoded: String
            if key == "rows" {
                encoded = canonicalJSON(root[key]!, floatNumeric: true)
            } else if key == "header", case let .object(header) = root[key]! {
                encoded = "{" + header.keys.sorted().map { field in
                    quote(field) + ":" + canonicalJSON(header[field]!, floatNumeric: field == "field_area_deg2" || field == "tmag_limit")
                }.joined(separator: ",") + "}"
            } else {
                encoded = canonicalJSON(root[key]!, floatNumeric: false)
            }
            return quote(key) + ":" + encoded
        }.joined(separator: ",") + "}"
    }
    private static func canonicalJSON(_ value: JSONValue, floatNumeric: Bool) -> String {
        switch value {
        case .null: return "null"
        case let .bool(v): return v ? "true" : "false"
        case let .integer(v): return floatNumeric ? pythonFloat(Double(v)) : String(v)
        case let .number(v): return pythonFloat(v)
        case let .string(s): return quote(s)
        case let .array(a): return "[" + a.map { canonicalJSON($0, floatNumeric: floatNumeric) }.joined(separator: ",") + "]"
        case let .object(o): return "{" + o.keys.sorted().map { quote($0) + ":" + canonicalJSON(o[$0]!, floatNumeric: floatNumeric) }.joined(separator: ",") + "}"
        }
    }
    private static func pythonFloat(_ value: Double) -> String {
        if value == 0.0 && value.sign == .minus { return "-0.0" }
        var s = String(value).lowercased()
        if !s.contains(".") && !s.contains("e") { s += ".0" }
        if let i = s.firstIndex(of: "e") {
            let m = String(s[..<i]); let e = Int(s[s.index(after: i)...]) ?? 0
            s = "\(m)e\(e < 0 ? "-" : "+")\(String(format: "%02d", abs(e)))"
        }
        return s
    }
    private static func quote(_ value: String) -> String {
        let data = try! JSONSerialization.data(withJSONObject: [value], options: [.fragmentsAllowed, .withoutEscapingSlashes])
        let encoded = String(data: data, encoding: .utf8)!
        return String(encoded.dropFirst().dropLast())
    }
}

private extension JSONValue {
    var stringValue: String? { if case let .string(v) = self { return v }; return nil }
    var integerValue: Int? { if case let .integer(v) = self { return Int(exactly: v) }; return nil }
}

enum SHA256Swift {  // internal: also used by the v3 gate dump driver
    private static let k: [UInt32] = [
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2]
    static func hex(_ input: [UInt8]) -> String {
        var bytes = input; let bitCount = UInt64(bytes.count) &* 8
        bytes.append(0x80); while bytes.count % 64 != 56 { bytes.append(0) }
        for shift in stride(from: 56, through: 0, by: -8) { bytes.append(UInt8(truncatingIfNeeded: bitCount >> UInt64(shift))) }
        var h: [UInt32] = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]
        for start in stride(from: 0, to: bytes.count, by: 64) {
            var w = [UInt32](repeating: 0, count: 64)
            for i in 0..<16 { let j = start + i * 4; w[i] = UInt32(bytes[j]) << 24 | UInt32(bytes[j+1]) << 16 | UInt32(bytes[j+2]) << 8 | UInt32(bytes[j+3]) }
            for i in 16..<64 { let a = w[i-15], b = w[i-2]; let s0 = rotr(a,7) ^ rotr(a,18) ^ (a >> 3); let s1 = rotr(b,17) ^ rotr(b,19) ^ (b >> 10); w[i] = w[i-16] &+ s0 &+ w[i-7] &+ s1 }
            var a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7]
            for i in 0..<64 { let s1=rotr(e,6)^rotr(e,11)^rotr(e,25); let ch=(e & f) ^ (~e & g); let t1=hh &+ s1 &+ ch &+ k[i] &+ w[i]; let s0=rotr(a,2)^rotr(a,13)^rotr(a,22); let maj=(a & b)^(a & c)^(b & c); let t2=s0 &+ maj; hh=g;g=f;f=e;e=d &+ t1;d=c;c=b;b=a;a=t1 &+ t2 }
            h[0] &+= a;h[1] &+= b;h[2] &+= c;h[3] &+= d;h[4] &+= e;h[5] &+= f;h[6] &+= g;h[7] &+= hh
        }
        return h.map { String(format:"%08x",$0) }.joined()
    }
    private static func rotr(_ x: UInt32, _ n: UInt32) -> UInt32 { (x >> n) | (x << (32-n)) }
}
