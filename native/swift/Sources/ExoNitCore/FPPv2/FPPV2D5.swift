import Foundation

// D5 PSF aperture fluxes and viable-neighbour selection, scalar Python order.
//
// Linux GNU calls glibc's `erf` via `FPPV2.erf`, the same symbol as CPython `math.erf`.
// macOS uses system libm; downstream lnL/FPP tolerances and zero gate flips apply per
// plan decision (t). No FMA contraction or fast-math is used anywhere, except the
// explicit `.addingProduct` in `pythonHypot` matching CPython 3.12 `dl_mul`.
// This module consumes server-built geometry; it never opens FITS or contacts a catalogue.
// Scenario families and worker dispatch are intentionally separate.

public enum FPPV2D5Error: Error, CustomStringConvertible {
    case emptyApertures

    public var description: String {
        switch self {
        case .emptyApertures:
            return "geometry contains no apertures"
        }
    }
}

public struct FPPV2Star: Sendable, Equatable {
    public var id: String
    public var ra: Double
    public var dec: Double
    public var tmag: Double

    public init(id: String = "", ra: Double, dec: Double, tmag: Double) {
        self.id = id
        self.ra = ra
        self.dec = dec
        self.tmag = tmag
    }
}

public struct FPPV2SectorGeometry: Sendable, Equatable {
    public var wcs: FPPV2TanWCS
    public var col0: Double
    public var row0: Double
    public var aperture: [(Double, Double)]?
    public var source: String

    public init(
        wcs: FPPV2TanWCS,
        col0: Double,
        row0: Double,
        aperture: [(Double, Double)]? = nil,
        source: String = "unknown"
    ) {
        self.wcs = wcs
        self.col0 = col0
        self.row0 = row0
        self.aperture = aperture
        self.source = source
    }

    public static func == (lhs: FPPV2SectorGeometry, rhs: FPPV2SectorGeometry) -> Bool {
        guard lhs.wcs == rhs.wcs && lhs.col0 == rhs.col0 && lhs.row0 == rhs.row0 && lhs.source == rhs.source else {
            return false
        }
        switch (lhs.aperture, rhs.aperture) {
        case (nil, nil):
            return true
        case (let a?, let b?):
            guard a.count == b.count else { return false }
            return zip(a, b).allSatisfy { $0.0 == $1.0 }
        default:
            return false
        }
    }
}

public struct FPPV2Geometry: Sendable, Equatable {
    public var kind: String
    public var sectors: [FPPV2SectorGeometry]
    public var offsets: [(Double, Double)]?
    public var nTaskSectors: Int?

    public init(
        kind: String = "synthetic",
        sectors: [FPPV2SectorGeometry] = [],
        offsets: [(Double, Double)]? = nil,
        nTaskSectors: Int? = nil
    ) {
        self.kind = kind
        self.sectors = sectors
        self.offsets = offsets
        self.nTaskSectors = nTaskSectors
    }

    public static func == (lhs: FPPV2Geometry, rhs: FPPV2Geometry) -> Bool {
        guard lhs.kind == rhs.kind && lhs.sectors == rhs.sectors && lhs.nTaskSectors == rhs.nTaskSectors else {
            return false
        }
        switch (lhs.offsets, rhs.offsets) {
        case (nil, nil):
            return true
        case (let a?, let b?):
            guard a.count == b.count else { return false }
            return zip(a, b).allSatisfy { $0.0 == $1.0 }
        default:
            return false
        }
    }
}

public struct FPPV2FluxStar: Sendable, Equatable {
    public var star: FPPV2Star
    public var fluxratio: Double
    public var tdepth: Double
    public var viable: Bool
    public var sepArcsec: Double

    public init(
        star: FPPV2Star,
        fluxratio: Double,
        tdepth: Double,
        viable: Bool,
        sepArcsec: Double
    ) {
        self.star = star
        self.fluxratio = fluxratio
        self.tdepth = tdepth
        self.viable = viable
        self.sepArcsec = sepArcsec
    }
}

public struct FPPV2GeometryInfo: Sendable, Equatable {
    public var geometryKind: String?
    public var geometryNSectors: Int
    public var geometryNTaskSectors: Int
    public var geometryCoverage: Double
    public var geometrySource: [String]
    public var nViableNeighbours: Int
    public var fluxratioTarget: Double

    public init(
        geometryKind: String? = nil,
        geometryNSectors: Int = 0,
        geometryNTaskSectors: Int = 0,
        geometryCoverage: Double = 0.0,
        geometrySource: [String] = [],
        nViableNeighbours: Int = 0,
        fluxratioTarget: Double = 0.0
    ) {
        self.geometryKind = geometryKind
        self.geometryNSectors = geometryNSectors
        self.geometryNTaskSectors = geometryNTaskSectors
        self.geometryCoverage = geometryCoverage
        self.geometrySource = geometrySource
        self.nViableNeighbours = nViableNeighbours
        self.fluxratioTarget = fluxratioTarget
    }
}

extension FPPV2 {
    public static let psfSigmaPix: Double = 0.75
    public static let synthPixArcsec: Double = 20.25
    public static let synthOffsets: [(Double, Double)] = {
        let grid: [Double] = [-0.4, 0.0, 0.4]
        var out: [(Double, Double)] = []
        out.reserveCapacity(9)
        for ox in grid {
            for oy in grid {
                out.append((ox, oy))
            }
        }
        return out
    }()

    /// Upstream's 5x5 box centred on `round(target pixel)` (Python/NumPy round-half-to-even).
    public static func defaultAperture(_ target: (Double, Double)) -> [(Double, Double)] {
        let cx = target.0.rounded(.toNearestOrEven)
        let cy = target.1.rounded(.toNearestOrEven)
        var ap: [(Double, Double)] = []
        ap.reserveCapacity(25)
        for i in -2...2 {
            for j in -2...2 {
                ap.append((cx + Double(i), cy + Double(j)))
            }
        }
        return ap
    }

    static func pixelWeight(_ p: Double, _ mu: Double) -> Double {
        let k = 1.0 / (0.75 * (2.0).squareRoot())
        return 0.5 * (FPPV2.erf((p + 0.5 - mu) * k) - FPPV2.erf((p - 0.5 - mu) * k))
    }

    /// Flux ratios of all stars in ONE aperture: `amp[i] * PSF flux`, normalised by the sequential total.
    public static func sectorFluxRatios(
        amp: [Double],
        mus: [(Double, Double)],
        aperture: [(Double, Double)]
    ) -> [Double] {
        var flux: [Double] = []
        flux.reserveCapacity(amp.count)
        for i in 0..<min(amp.count, mus.count) {
            let mu = mus[i]
            var acc = 0.0
            for p in aperture {
                acc += pixelWeight(p.0, mu.0) * pixelWeight(p.1, mu.1)
            }
            flux.append(amp[i] * acc)
        }
        var total = 0.0
        for f in flux {
            total += f
        }
        return flux.map { $0 / total }
    }

    /// List of `(mus, aperture)` pairs, one per sector (real) or per sub-pixel offset (synthetic).
    public static func geometryApertures(
        stars: [FPPV2Star],
        geometry: FPPV2Geometry
    ) throws -> [(mus: [(Double, Double)], aperture: [(Double, Double)])] {
        if stars.isEmpty {
            return []
        }
        var out: [(mus: [(Double, Double)], aperture: [(Double, Double)])] = []
        if geometry.kind == "real" {
            for sec in geometry.sectors {
                var mus: [(Double, Double)] = []
                mus.reserveCapacity(stars.count)
                for s in stars {
                    let (x, y) = try tanWorldToPixel(sec.wcs, raDeg: s.ra, decDeg: s.dec)
                    mus.append((sec.col0 + x, sec.row0 + y))
                }
                let ap: [(Double, Double)]
                if let custom = sec.aperture, !custom.isEmpty {
                    ap = custom
                } else {
                    ap = defaultAperture(mus[0])
                }
                out.append((mus: mus, aperture: ap))
            }
        } else {
            let t = stars[0]
            let cosd0 = FPPV2.cos(t.dec * (FPPV2Consts.pi / 180.0))
            var base: [(Double, Double)] = []
            base.reserveCapacity(stars.count)
            for s in stars {
                let dx = (s.ra - t.ra) * cosd0 * 3600.0 / synthPixArcsec
                let dy = (s.dec - t.dec) * 3600.0 / synthPixArcsec
                base.append((dx, dy))
            }
            let ap = defaultAperture((0.0, 0.0))
            let offsets = geometry.offsets ?? synthOffsets
            for (ox, oy) in offsets {
                let mus = base.map { (dx, dy) in (dx - ox, dy - oy) }
                out.append((mus: mus, aperture: ap))
            }
        }
        return out
    }

    fileprivate static func requiredDepths(
        _ frs: [Double],
        depth: Double,
        dilutionCorrected: Bool
    ) -> [Double] {
        let apDepth = dilutionCorrected ? depth * frs[0] : depth
        return frs.map { f in
            let t = f != 0.0 ? (apDepth / f) : 0.0
            return t > 1.0e6 ? 0.0 : t
        }
    }

    /// Plan D5: flux ratios, required depths and the viable-neighbour set. Returns `(stars, info)`.
    public static func computeFluxRatiosD5(
        stars: [FPPV2Star],
        depthPPM: Double,
        geometry: FPPV2Geometry? = nil,
        dilutionCorrected: Bool = false
    ) throws -> (stars: [FPPV2FluxStar], info: FPPV2GeometryInfo) {
        var info = FPPV2GeometryInfo()
        if stars.isEmpty {
            return ([], info)
        }
        var geom = geometry ?? FPPV2Geometry()
        if geom.kind == "real" && geom.sectors.isEmpty {
            geom = FPPV2Geometry()
        }
        var tmin = stars[0].tmag
        for s in stars.dropFirst() {
            if s.tmag < tmin {
                tmin = s.tmag
            }
        }
        let amp = stars.map { s in FPPV2.pow(10.0, (tmin - s.tmag) / 2.5) }
        let apertures = try geometryApertures(stars: stars, geometry: geom)
        let per = apertures.map { pair in
            sectorFluxRatios(amp: amp, mus: pair.mus, aperture: pair.aperture)
        }
        let k = per.count
        if k == 0 {
            throw FPPV2D5Error.emptyApertures
        }
        let n = stars.count
        var fr: [Double] = []
        fr.reserveCapacity(n)
        for i in 0..<n {
            var acc = 0.0
            for sector in per {
                acc += sector[i]
            }
            fr.append(acc / Double(k))
        }
        let td = requiredDepths(fr, depth: depthPPM, dilutionCorrected: dilutionCorrected)
        var viable = td.map { $0 > 0.0 }
        if geom.kind == "synthetic" && k > 1 {
            for sector in per {
                let tdk = requiredDepths(sector, depth: depthPPM, dilutionCorrected: dilutionCorrected)
                for i in 0..<viable.count {
                    viable[i] = viable[i] || (tdk[i] > 0.0)
                }
            }
        }
        let cosd0 = FPPV2.cos(stars[0].dec * (FPPV2Consts.pi / 180.0))
        var out: [FPPV2FluxStar] = []
        out.reserveCapacity(n)
        for (i, s) in stars.enumerated() {
            let sep = pythonHypot((s.ra - stars[0].ra) * cosd0, s.dec - stars[0].dec) * 3600.0
            out.append(
                FPPV2FluxStar(
                    star: s,
                    fluxratio: fr[i],
                    tdepth: td[i],
                    viable: i == 0 ? true : viable[i],
                    sepArcsec: sep
                )
            )
        }
        info.geometryKind = geom.kind
        if geom.kind == "real" {
            info.geometryNSectors = k
            let nTask = geom.nTaskSectors ?? k
            info.geometryNTaskSectors = nTask
            info.geometryCoverage = min(1.0, Double(k) / Double(max(1, nTask)))
            info.geometrySource = Array(Set(geom.sectors.map { $0.source })).sorted()
        } else {
            info.geometrySource = ["synthetic"]
        }
        info.nViableNeighbours = out.dropFirst().filter { $0.viable }.count
        info.fluxratioTarget = fr[0]
        return (out, info)
    }

    /// Reference `_compute_flux_ratios` compatibility wrapper.
    public static func computeFluxRatios(
        stars: [FPPV2Star],
        depthPPM: Double,
        geometry: FPPV2Geometry? = nil,
        dilutionCorrected: Bool = false
    ) throws -> [FPPV2FluxStar] {
        try computeFluxRatiosD5(
            stars: stars,
            depthPPM: depthPPM,
            geometry: geometry,
            dilutionCorrected: dilutionCorrected
        ).stars
    }

    /// Two-coordinate specialization of CPython 3.12 `mathmodule.c` `vector_norm`.
    /// Source: https://github.com/python/cpython/blob/v3.12.3/Modules/mathmodule.c (PSF-2.0).
    /// Compensated square/sum and differential sqrt correction avoid the occasional 1-ULP
    /// difference of libm `hypot`. Explicit `.addingProduct` computes only the product's
    /// residual, as in CPython `dl_mul`; expressions elsewhere are never contracted.
    public static func pythonHypot(_ x: Double, _ y: Double) -> Double {
        let values = [abs(x), abs(y)]
        let maxVal = max(values[0], values[1])
        if maxVal.isInfinite {
            return maxVal
        }
        if x.isNaN || y.isNaN {
            return .nan
        }
        if maxVal == 0.0 {
            return maxVal
        }
        let (_, exponent) = Foundation.frexp(maxVal)
        if exponent < -1023 {
            let minPos = Double.leastNormalMagnitude
            return minPos * pythonHypot(x / minPos, y / minPos)
        }
        let scale = Foundation.scalbn(1.0, -exponent)
        var csum = 1.0
        var frac1 = 0.0
        var frac2 = 0.0
        for v0 in values {
            let v = v0 * scale
            let hi = v * v
            let lo = (-hi).addingProduct(v, v)
            let sum = csum + hi
            let err = (csum - sum) + hi
            csum = sum
            frac1 += lo
            frac2 += err
        }
        var h = (csum - 1.0 + (frac1 + frac2)).squareRoot()
        let hi = -h * h
        let lo = (-hi).addingProduct(-h, h)
        let sum = csum + hi
        let err = (csum - sum) + hi
        csum = sum
        frac1 += lo
        frac2 += err
        let residual = csum - 1.0 + (frac1 + frac2)
        h += residual / (2.0 * h)
        return h / scale
    }
}
