import Foundation

public struct FPPV3SectorGeometry: Sendable {
    public var wcs: FPPV2TanWCS
    public var col0: Double
    public var row0: Double
    public var aperture: [(Double, Double)]?
    public var apertureKind: String?
    public var boxCentre: (Double, Double)?
    public var circleRadius: Double?
    public var cShift: (Double, Double)?
    public var epochs: [(Double, Double)]
    public var sigmaCal: Double?
    public var source: String?
    public var apertureSource: String?
    public var tglcNearEdge: Bool?

    public init(wcs: FPPV2TanWCS, col0: Double, row0: Double, aperture: [(Double, Double)]? = nil,
                apertureKind: String? = nil, boxCentre: (Double, Double)? = nil, circleRadius: Double? = nil,
                cShift: (Double, Double)? = nil, epochs: [(Double, Double)] = [], sigmaCal: Double? = nil,
                source: String? = nil, apertureSource: String? = nil, tglcNearEdge: Bool? = nil) {
        self.wcs = wcs; self.col0 = col0; self.row0 = row0; self.aperture = aperture
        self.apertureKind = apertureKind; self.boxCentre = boxCentre; self.circleRadius = circleRadius
        self.cShift = cShift; self.epochs = epochs; self.sigmaCal = sigmaCal; self.source = source
        self.apertureSource = apertureSource; self.tglcNearEdge = tglcNearEdge
    }
}

public struct FPPV3Geometry: Sendable {
    public var kind: String
    public var sectors: [FPPV3SectorGeometry]
    public var nTaskSectors: Int?
    public init(kind: String = "synthetic", sectors: [FPPV3SectorGeometry] = [], nTaskSectors: Int? = nil) {
        self.kind = kind; self.sectors = sectors; self.nTaskSectors = nTaskSectors
    }

    public static func decode(_ value: JSONValue?) throws -> FPPV3Geometry? {
        guard let value else { return nil }
        guard case let .object(root) = value, case let .string(kind)? = root["kind"] else {
            throw FPPV3Error("geometry_invalid", "geometry must be an object with kind")
        }
        let taskCount = root["n_task_sectors"]?.intNumber
        let rawSectors: [JSONValue]
        if case let .array(a)? = root["sectors"] { rawSectors = a } else { rawSectors = [] }
        let sectors = try rawSectors.map { raw -> FPPV3SectorGeometry in
            guard case let .object(s) = raw, case let .object(w)? = s["wcs"] else {
                throw FPPV3Error("geometry_invalid", "sector is missing wcs")
            }
            func pair(_ key: String, _ object: [String: JSONValue]) throws -> (Double, Double)? {
                guard let v = object[key] else { return nil }
                guard case let .array(a) = v, a.count == 2,
                      let x = a[0].doubleNumber, let y = a[1].doubleNumber else {
                    throw FPPV3Error("geometry_invalid", "invalid \(key)")
                }
                return (x, y)
            }
            guard let crpix = try pair("crpix", w), let crval = try pair("crval", w),
                  let cdelt = try pair("cdelt", w), case let .array(pcRaw)? = w["pc"], pcRaw.count == 4,
                  let pc0 = pcRaw[0].doubleNumber, let pc1 = pcRaw[1].doubleNumber,
                  let pc2 = pcRaw[2].doubleNumber, let pc3 = pcRaw[3].doubleNumber else {
                throw FPPV3Error("geometry_invalid", "invalid WCS fields")
            }
            var ctype = ["RA---TAN", "DEC--TAN"]
            if case let .array(c)? = w["ctype"] { ctype = c.compactMap(\.stringValue) }
            let wcs = FPPV2TanWCS(crpix: crpix, crval: crval, pc: (pc0,pc1,pc2,pc3), cdelt: cdelt, ctype: ctype)
            var aperture: [(Double, Double)]?
            if case let .array(a)? = s["aperture"] {
                aperture = try a.map { p in
                    guard case let .array(xy) = p, xy.count == 2,
                          let x = xy[0].doubleNumber, let y = xy[1].doubleNumber else {
                        throw FPPV3Error("geometry_invalid", "invalid aperture pixel")
                    }
                    return (x,y)
                }
            }
            var epochs = [(Double, Double)]()
            if case let .array(e)? = s["epochs"] {
                epochs = try e.map { row in
                    guard case let .array(xy) = row, xy.count == 2,
                          let x = xy[0].doubleNumber, let y = xy[1].doubleNumber else {
                        throw FPPV3Error("geometry_invalid", "invalid epoch offset")
                    }
                    return (x,y)
                }
            }
            let centre = try pair("box_centre", s), shift = try pair("c_shift", s)
            return FPPV3SectorGeometry(wcs: wcs, col0: s["col0"]?.doubleNumber ?? 0.0,
                                       row0: s["row0"]?.doubleNumber ?? 0.0, aperture: aperture,
                                       apertureKind: s["aperture_kind"]?.stringValue,
                                       boxCentre: centre, circleRadius: s["circle_radius"]?.doubleNumber,
                                       cShift: shift, epochs: epochs, sigmaCal: s["sigma_cal"]?.doubleNumber,
                                       source: s["source"]?.stringValue,
                                       apertureSource: s["aperture_source"]?.stringValue,
                                       tglcNearEdge: s["tglc_near_edge"]?.boolValue)
        }
        return FPPV3Geometry(kind: kind, sectors: sectors, nTaskSectors: taskCount)
    }
}

public struct FPPV3SmoothStar: Sendable {
    public var star: FPPV2Star
    public var fluxratio: Double
    public var tdepthRaw: Double
    public var tdepth: Double
    public var viability: Double
    public var viable: Bool
    public var sepArcsec: Double
}

public struct FPPV3GeometryInfo: Sendable {
    public var geometryKind = "real"
    public var geometryMode = "smooth"
    public var geometryNSectors = 0
    public var geometryNTaskSectors = 0
    public var geometryCoverage = 0.0
    public var geometrySource = [String]()
    public var apertureSource = [String]()
    public var geometryPath = [String]()
    public var geometryPathEmulated = false
    public var geometryNEpochs = [Int]()
    public var geometrySigmaCal = [Double]()
    public var nNodes = 0
    public var nViableNeighbours = 0
    public var nViabilityWeighted = 0
    public var viabilityMassSkipped = 0.0
    public var fluxratioTarget = 0.0
}

public enum FPPV3GeometryEngine {
    private struct Pixel { var x: Double; var y: Double; var weight: Double }
    private struct Sector {
        var kind: String; var pixels: [Pixel]?; var radius: Double; var emulated: Bool
        var source: String; var path: String; var shift: (Double, Double); var epochs: [(Double, Double)]; var sigma: Double
    }
    private static let nodeBits: [UInt64] = [
        0x3f68527aba212071,0x3f8b404942866157,0x3f9676f9c7838aa7,0x3f8b404942866157,0x3f68527aba212071,
        0x3f8b404942866157,0x3fae8862e93c9e5c,0x3fb92b856facd857,0x3fae8862e93c9e5c,0x3f8b404942866157,
        0x3f9676f9c7838aa7,0x3fb92b856facd857,0x3fc4bfc906c2a8eb,0x3fb92b856facd857,0x3f9676f9c7838aa7,
        0x3f8b404942866157,0x3fae8862e93c9e5c,0x3fb92b856facd857,0x3fae8862e93c9e5c,0x3f8b404942866157,
        0x3f68527aba212071,0x3f8b404942866157,0x3f9676f9c7838aa7,0x3f8b404942866157,0x3f68527aba212071]

    public static func computeFluxRatiosSmooth(stars: [FPPV2Star], depthPPM: Double,
                                                geometry: FPPV3Geometry, dilutionCorrected: Bool,
                                                lcSource: String? = nil, vMin: Double = 0.01) throws
        -> (stars: [FPPV3SmoothStar], info: FPPV3GeometryInfo) {
        guard !stars.isEmpty else { throw FPPV3Error("geometry_invalid", "geometry requires a target star") }
        guard geometry.kind == "real", !geometry.sectors.isEmpty else {
            throw FPPV3Error("geometry_invalid", "smooth geometry requires real sectors")
        }
        var sectors = try geometry.sectors.map { try resolve($0, lcSource: lcSource) }
        let n = stars.count, ns = sectors.count
        let tmin = stars.map(\.tmag).min()!
        let amp = stars.map { FPPV2.pow(10.0, (tmin - $0.tmag) / 2.5) }
        var bases = [[(Double, Double)]](); bases.reserveCapacity(ns)
        for i in sectors.indices {
            let src = geometry.sectors[i], sec = sectors[i]
            var base = [(Double, Double)](); base.reserveCapacity(n)
            for star in stars {
                let xy = try FPPV2.tanWorldToPixel(src.wcs, raDeg: star.ra, decDeg: star.dec)
                base.append(((src.col0 + xy.x) + sec.shift.0, (src.row0 + xy.y) + sec.shift.1))
            }
            if sec.kind == "tglc_box" && sectors[i].pixels == nil {
                sectors[i].pixels = boxPixels(base[0].0.rounded(.toNearestOrEven), base[0].1.rounded(.toNearestOrEven))
            }
            bases.append(base)
        }
        let oneNode = sectors.allSatisfy { $0.sigma == 0.0 }
        let count = oneNode ? 1 : 25
        var frExp = [Double](repeating: 0.0, count: n), vExp = frExp
        for ni in 0..<count {
            let a = oneNode ? 0 : ni / 5 - 2, b = oneNode ? 0 : ni % 5 - 2
            let weight = oneNode ? 1.0 : Double(bitPattern: nodeBits[ni])
            var per = [[Double]](); per.reserveCapacity(ns)
            for si in 0..<ns {
                let sec = sectors[si], base = bases[si]
                let dx = sec.sigma * Double(a), dy = sec.sigma * Double(b)
                let pixels: [Pixel]
                if sec.kind == "qlp_circle" { pixels = qlpPixels(dx, dy, sec.radius) }
                else if sec.kind == "default" {
                    pixels = defaultPixels((base[0].0 + dx).rounded(.toNearestOrEven), (base[0].1 + dy).rounded(.toNearestOrEven))
                } else { pixels = sec.pixels ?? [] }
                var flux = [Double](repeating: 0.0, count: n)
                for i in 0..<n {
                    var acc = 0.0
                    for epoch in sec.epochs {
                        let mu = sec.kind == "qlp_circle"
                            ? (base[i].0 - base[0].0, base[i].1 - base[0].1)
                            : (base[i].0 + dx + epoch.0, base[i].1 + dy + epoch.1)
                        acc = acc + amp[i] * apertureFlux(mu, pixels)
                    }
                    flux[i] = acc / Double(sec.epochs.count)
                }
                var total = 0.0; for f in flux { total += f }
                per.append(flux.map { $0 / total })
            }
            var fr = [Double](repeating: 0.0, count: n)
            for i in 0..<n { var acc = 0.0; for sector in per { acc += sector[i] }; fr[i] = acc / Double(ns) }
            let apDepth = dilutionCorrected ? depthPPM * fr[0] : depthPPM
            for i in 0..<n {
                let td = fr[i] != 0.0 ? apDepth / fr[i] : 0.0
                frExp[i] += weight * fr[i]
                vExp[i] += weight * (td > 0.0 && td <= 1.0e6 ? 1.0 : 0.0)
            }
        }
        let apDepth = dilutionCorrected ? depthPPM * frExp[0] : depthPPM
        var weighted = 0, skipped = 0.0, out = [FPPV3SmoothStar](); out.reserveCapacity(n)
        for i in 0..<n {
            let td = frExp[i] != 0.0 ? apDepth / frExp[i] : 0.0
            var viability = i == 0 ? 1.0 : min(1.0, max(0.0, vExp[i]))
            if viability > 1.0 - 1.0e-12 { viability = 1.0 }
            let viable = i == 0 || viability >= vMin
            if i > 0 && viable && viability < 1.0 { weighted += 1 }
            if i > 0 && !viable { skipped += viability }
            let cosd0 = FPPV2.cos(stars[0].dec * (FPPV2Consts.pi / 180.0))
            let sep = FPPV2.pythonHypot((stars[i].ra - stars[0].ra) * cosd0, stars[i].dec - stars[0].dec) * 3600.0
            out.append(FPPV3SmoothStar(star: stars[i], fluxratio: frExp[i], tdepthRaw: td,
                                      tdepth: td > 1.0e6 ? 0.0 : td, viability: viability,
                                      viable: viable, sepArcsec: sep))
        }
        var info = FPPV3GeometryInfo()
        info.geometryNSectors = ns; info.geometryNTaskSectors = geometry.nTaskSectors ?? ns
        info.geometryCoverage = min(1.0, Double(ns) / Double(max(1, info.geometryNTaskSectors)))
        info.geometrySource = Array(Set(geometry.sectors.map { $0.source ?? "unknown" })).sorted()
        info.apertureSource = Array(Set(sectors.map(\.source))).sorted()
        info.geometryPath = Array(Set(sectors.map(\.path))).sorted()
        info.geometryPathEmulated = sectors.contains { $0.emulated }
        info.geometryNEpochs = sectors.map { $0.epochs.count }; info.geometrySigmaCal = sectors.map(\.sigma)
        info.nNodes = count; info.nViableNeighbours = out.dropFirst().filter(\.viable).count
        info.nViabilityWeighted = weighted; info.viabilityMassSkipped = skipped; info.fluxratioTarget = frExp[0]
        return (out, info)
    }

    private static func resolve(_ s: FPPV3SectorGeometry, lcSource: String?) throws -> Sector {
        let ap = s.aperture?.isEmpty == false ? s.aperture : nil
        let source = (lcSource ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let kind = s.apertureKind ?? (source == "tglc" ? "tglc_box" : source == "qlp" ? "qlp_circle" : (ap != nil ? "mask" : "default"))
        let emulated: Bool
        switch kind {
        case "mask": guard ap != nil else { throw FPPV3Error("geometry_invalid", "mask aperture missing") }; emulated = false
        case "tglc_box": emulated = !(s.boxCentre != nil || ap != nil)
        case "qlp_circle": emulated = s.circleRadius == nil
        case "default": emulated = false
        default: throw FPPV3Error("geometry_invalid", "unknown aperture_kind \(kind)")
        }
        let pixels: [Pixel]?
        let apertureSource: String
        switch kind {
        case "mask": pixels = ap?.map { Pixel(x: $0.0, y: $0.1, weight: 1.0) }; apertureSource = s.apertureSource ?? "spoc_lc_mask"
        case "tglc_box" where !emulated && ap != nil:
            pixels = ap?.map { Pixel(x: $0.0, y: $0.1, weight: 1.0) }; apertureSource = "tglc_box_header"
        case "tglc_box" where s.boxCentre != nil:
            let c = s.boxCentre!; pixels = boxPixels(c.0.rounded(.toNearestOrEven),c.1.rounded(.toNearestOrEven)); apertureSource = "tglc_box_header"
        case "tglc_box": pixels = nil; apertureSource = "tglc_box_emulated"
        case "qlp_circle": pixels = nil; apertureSource = emulated ? "qlp_circle_assumed" : "qlp_bestap_circle"
        default: pixels = nil; apertureSource = "default_5x5"
        }
        if kind == "qlp_circle", let r = s.circleRadius, !(0.5...8.0).contains(r) {
            throw FPPV3Error("geometry_invalid", "circle_radius outside 0.5-8 px")
        }
        let radius = s.circleRadius ?? 3.0, shift = s.cShift ?? (0.0,0.0)
        guard shift.0.isFinite && shift.1.isFinite && s.epochs.allSatisfy({ $0.0.isFinite && $0.1.isFinite }) else {
            throw FPPV3Error("geometry_invalid", "c_shift / epochs must be finite")
        }
        var epochs = s.epochs.isEmpty ? [(0.0,0.0)] : s.epochs
        if kind == "qlp_circle" { epochs = [(0.0,0.0)] }
        let sigmaDefault: Double = kind == "mask" ? 0.02 : kind == "tglc_box" ? 0.035 : kind == "qlp_circle" ? 0.05 : 0.035
        let sigma = s.sigmaCal ?? sigmaDefault
        guard sigma.isFinite && sigma >= 0.0 else { throw FPPV3Error("geometry_invalid", "sigma_cal must be finite and >= 0") }
        let path = kind == "mask" ? "spoc_mask" : kind == "tglc_box" ? "tglc_box" : kind == "qlp_circle" ? "qlp_circle" : "u_default"
        return Sector(kind: kind, pixels: pixels, radius: radius, emulated: emulated, source: apertureSource,
                      path: path, shift: shift, epochs: epochs, sigma: sigma)
    }
    private static func boxPixels(_ cx: Double, _ cy: Double) -> [Pixel] {
        var out = [Pixel](); for i in -1...1 { for j in -1...1 { out.append(Pixel(x: cx+Double(i), y: cy+Double(j), weight: 1.0)) } }; return out
    }
    private static func defaultPixels(_ cx: Double, _ cy: Double) -> [Pixel] {
        var out = [Pixel](); for i in -2...2 { for j in -2...2 { out.append(Pixel(x: cx+Double(i), y: cy+Double(j), weight: 1.0)) } }; return out
    }
    private static func apertureFlux(_ mu: (Double, Double), _ pixels: [Pixel]) -> Double {
        var cacheX = [UInt64: Double](), cacheY = [UInt64: Double](), acc = 0.0
        for p in pixels {
            let wx: Double
            if let cached = cacheX[p.x.bitPattern] { wx = cached }
            else { wx = FPPV2.pixelWeight(p.x, mu.0); cacheX[p.x.bitPattern] = wx }
            let wy: Double
            if let cached = cacheY[p.y.bitPattern] { wy = cached }
            else { wy = FPPV2.pixelWeight(p.y, mu.1); cacheY[p.y.bitPattern] = wy }
            acc = acc + ((p.weight * wx) * wy)
        }
        return acc
    }
    private static func qlpPixels(_ cx: Double, _ cy: Double, _ radius: Double) -> [Pixel] {
        let sub = 32, r2 = radius * radius, inv = 1.0 / Double(sub), nn = Double(sub * sub)
        let x0 = Int(floor(cx-radius-0.5)), x1 = Int(ceil(cx+radius+0.5))
        let y0 = Int(floor(cy-radius-0.5)), y1 = Int(ceil(cy+radius+0.5))
        var out = [Pixel]()
        for px in x0...x1 {
            let xl = Double(px)-0.5-cx, xh = Double(px)+0.5-cx
            let nearX = xl <= 0.0 && 0.0 <= xh ? 0.0 : min(abs(xl),abs(xh))
            for py in y0...y1 {
                let yl = Double(py)-0.5-cy, yh = Double(py)+0.5-cy
                let nearY = yl <= 0.0 && 0.0 <= yh ? 0.0 : min(abs(yl),abs(yh))
                if nearX*nearX + nearY*nearY > r2 { continue }
                let farX=max(abs(xl),abs(xh)), farY=max(abs(yl),abs(yh))
                if farX*farX + farY*farY <= r2*(1.0-1e-9) { out.append(Pixel(x:Double(px),y:Double(py),weight:1.0)); continue }
                var n=0
                for i in 0..<sub {
                    let x=(Double(px)-0.5+(Double(i)+0.5)*inv)-cx, x2=FPPV2.pow(x,2.0)
                    for j in 0..<sub {
                        let y=(Double(py)-0.5+(Double(j)+0.5)*inv)-cy, y2=FPPV2.pow(y,2.0)
                        if x2+y2 <= r2 { n += 1 }
                    }
                }
                if n>0 { out.append(Pixel(x:Double(px),y:Double(py),weight:Double(n)/nn)) }
            }
        }
        return out
    }
}

private extension JSONValue {
    var doubleNumber: Double? { switch self { case let .number(v): return v; case let .integer(v): return Double(v); default: return nil } }
    var intNumber: Int? { switch self { case let .integer(v): return Int(exactly:v); case let .number(v): return Int(exactly:v); default: return nil } }
    var stringValue: String? { if case let .string(v) = self { return v }; return nil }
    var boolValue: Bool? { if case let .bool(v) = self { return v }; return nil }
}
