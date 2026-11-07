import Foundation

// Task-shipped TAN WCS. FITS discovery/build_task_geometry remains server-side;
// workers consume its JSON, including detector origins and aperture pixel order.

public enum FPPV2WCSError: Error, CustomStringConvertible {
    case unsupportedProjection([String])
    case farSideOfTangentPoint

    public var description: String {
        switch self {
        case .unsupportedProjection(let ctype):
            return "unsupported WCS projection \(ctype)"
        case .farSideOfTangentPoint:
            return "star is on the far side of the tangent point"
        }
    }
}

public struct FPPV2TanWCS: Sendable, Equatable {
    public var crpix: (Double, Double)
    public var crval: (Double, Double)
    public var pc: (Double, Double, Double, Double)
    public var cdelt: (Double, Double)
    public var ctype: [String]

    public init(
        crpix: (Double, Double),
        crval: (Double, Double),
        pc: (Double, Double, Double, Double),
        cdelt: (Double, Double),
        ctype: [String] = ["RA---TAN", "DEC--TAN"]
    ) {
        self.crpix = crpix
        self.crval = crval
        self.pc = pc
        self.cdelt = cdelt
        self.ctype = ctype
    }

    public static func == (lhs: FPPV2TanWCS, rhs: FPPV2TanWCS) -> Bool {
        lhs.crpix == rhs.crpix && lhs.crval == rhs.crval && lhs.pc == rhs.pc
            && lhs.cdelt == rhs.cdelt && lhs.ctype == rhs.ctype
    }
}

extension FPPV2 {
    /// 0-based pixel (x, y) of the image grid of a TAN WCS (astropy `all_world2pix(ra, dec, 0)` equivalent).
    /// Scalar CPython `math.radians` / `math.degrees` constants and operation order, using `@inline(never)`
    /// libm shims (`FPPV2.sin`, `FPPV2.cos`) to prevent `___sincos_stret` fusion.
    public static func tanWorldToPixel(
        _ wcs: FPPV2TanWCS,
        raDeg: Double,
        decDeg: Double
    ) throws -> (x: Double, y: Double) {
        if wcs.ctype.count != 2 || wcs.ctype[0] != "RA---TAN" || wcs.ctype[1] != "DEC--TAN" {
            throw FPPV2WCSError.unsupportedProjection(wcs.ctype)
        }
        let rad = FPPV2Consts.pi / 180.0
        let deg = 180.0 / FPPV2Consts.pi
        let a0 = wcs.crval.0 * rad
        let d0 = wcs.crval.1 * rad
        let a = raDeg * rad
        let d = decDeg * rad
        let da = a - a0
        let sd = FPPV2.sin(d)
        let cd = FPPV2.cos(d)
        let sd0 = FPPV2.sin(d0)
        let cd0 = FPPV2.cos(d0)
        let cda = FPPV2.cos(da)
        let cosc = sd0 * sd + cd0 * cd * cda
        if cosc <= 0.0 {
            throw FPPV2WCSError.farSideOfTangentPoint
        }
        let xi = (cd * FPPV2.sin(da) / cosc) * deg
        let eta = ((cd0 * sd - sd0 * cd * cda) / cosc) * deg
        let u = xi / wcs.cdelt.0
        let v = eta / wcs.cdelt.1
        let (p11, p12, p21, p22) = wcs.pc
        let det = p11 * p22 - p12 * p21
        let dx = (p22 * u - p12 * v) / det
        let dy = (-p21 * u + p11 * v) / det
        return (wcs.crpix.0 - 1.0 + dx, wcs.crpix.1 - 1.0 + dy)
    }
}
