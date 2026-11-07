import Foundation

// Swift port of FPP Step 4 (D6 S family: STP, SEB, SEBx2P) and its building blocks.
// Reference: worker/fpp_native.py on branch worktree-dossier-regen.
//
// Arithmetic order is the Python/NumPy/numba order, literally:
// - All transcendentals (sin, cos, acos, atan2, log, log10, exp, pow, erf) go through
//   the `@inline(never)` shims on `FPPV2` so the compiler never fuses `sin`/`cos` into
//   `___sincos_stret` or folds/rewrites libm calls between debug and release.
// - NumPy array `** 2` is lowered to `x * x`, while NumPy array `** 3` and Python scalar
//   `** 2` / `** 3` call `FPPV2.pow`.
// - Per-draw limb darkening takes per-draw `u1`/`u2` arrays directly with the exact same
//   per-draw operation order as Python's `_call_by_ldc` over the scalar-u batch kernels.
// - `_lnZ_from_logw` uses `FPPNumpy.pairwiseSum` over the full length-N `exp(w - m)` array.

/// Raised when no row in `ldc_tess.csv` matches the requested `(Z, Teff, logg)` combination.
public struct FPPV2LDCNoRow: Error, CustomStringConvertible, Equatable, Sendable {
    public let skipReason: String = "ldc_no_row"
    public let description: String

    public init(_ description: String) {
        self.description = description
    }
}

/// Raised on invalid array shapes or arguments in the S-family port.
public struct FPPV2SFamilyError: Error, CustomStringConvertible, Equatable, Sendable {
    public let description: String

    public init(_ description: String) {
        self.description = description
    }
}

/// Switch set controlling parity vs production behaviour (`_PARITY_SET` / `_PRODUCTION_SET`).
public struct FPPV2Switches: Equatable, Sendable {
    /// `FPP_SECONDARY_GEOMETRY == "consistent"` (`false` = `"upstream"`).
    public var secondaryGeometryConsistent: Bool
    /// `FPP_ORBIT_PHYSICS == "roche"` (`false` = `"upstream"`).
    public var orbitPhysicsRoche: Bool
    /// `FPP_K1_NUDGE == "upstream"` (`false` = `"exact"`).
    public var k1NudgeUpstream: Bool
    /// `FPP_Q_BRANCH == "upstream"` (`false` = `"correct"`).
    public var qBranchUpstream: Bool
    /// `FPP_EXPOSURE == "upstream_2min_x20"` (`false` = `"cadence"`).
    public var exposureUpstream2MinX20: Bool
    /// `FPP_NORM == "data"` (`false` = `"upstream"`).
    public var normData: Bool

    public init(
        secondaryGeometryConsistent: Bool,
        orbitPhysicsRoche: Bool,
        k1NudgeUpstream: Bool,
        qBranchUpstream: Bool,
        exposureUpstream2MinX20: Bool,
        normData: Bool = false
    ) {
        self.secondaryGeometryConsistent = secondaryGeometryConsistent
        self.orbitPhysicsRoche = orbitPhysicsRoche
        self.k1NudgeUpstream = k1NudgeUpstream
        self.qBranchUpstream = qBranchUpstream
        self.exposureUpstream2MinX20 = exposureUpstream2MinX20
        self.normData = normData
    }

    public static let parity = FPPV2Switches(
        secondaryGeometryConsistent: false,
        orbitPhysicsRoche: false,
        k1NudgeUpstream: true,
        qBranchUpstream: true,
        exposureUpstream2MinX20: true,
        normData: false
    )

    public static let production = FPPV2Switches(
        secondaryGeometryConsistent: true,
        orbitPhysicsRoche: true,
        k1NudgeUpstream: false,
        qBranchUpstream: false,
        exposureUpstream2MinX20: false,
        normData: false
    )

    public var exposureArgs: (exptime: Double, nsub: Int) {
        if exposureUpstream2MinX20 {
            return (FPPV2Consts.parityExptimeD, FPPV2Consts.parityNsub)
        }
        return (0.0, 1)
    }
}

/// Complete production fold (`build_prod_fold` / `prod_fold_renorm` dictionary in Python).
public struct FPPV2ProdFold: Sendable {
    public var arrays: FPPV2FoldArrays
    public var f: [Double]
    public var w: Double
    public var delta: Double
    public var nb: Int
    public var baseChi2: Double
    public var lnnorm: Double

    public init(
        arrays: FPPV2FoldArrays,
        f: [Double],
        w: Double,
        delta: Double,
        nb: Int,
        baseChi2: Double,
        lnnorm: Double
    ) {
        self.arrays = arrays
        self.f = f
        self.w = w
        self.delta = delta
        self.nb = nb
        self.baseChi2 = baseChi2
        self.lnnorm = lnnorm
    }

    /// `build_prod_fold(rows, period, sigma_b, flux_b, cadence_s)`.
    public static func build(
        rows: [FPPV2FoldRow],
        period: Double,
        sigmaB: [Double],
        fluxB: [Double],
        cadenceS: Double
    ) throws -> FPPV2ProdFold {
        let arr = try FPPV2Fold.prodFoldArrays(rows: rows, period: period, sigmaB: sigmaB)
        let c = try FPPV2Fold.buildProdFoldConsts(fold: arr, fluxB: fluxB, cadenceS: cadenceS)
        return FPPV2ProdFold(
            arrays: arr,
            f: fluxB,
            w: c.w,
            delta: c.delta,
            nb: rows.count,
            baseChi2: c.baseChi2,
            lnnorm: c.lnnorm
        )
    }

    /// `prod_fold_renorm(fold, fr)`.
    public func renorm(fr: Double) throws -> FPPV2ProdFold {
        let rn = try FPPV2Fold.prodFoldRenorm(f: self.f, sigma: self.arrays.sigma, fr: fr)
        var arr2 = self.arrays
        arr2.sigma = rn.sigma
        return FPPV2ProdFold(
            arrays: arr2,
            f: rn.f,
            w: self.w,
            delta: self.delta,
            nb: self.nb,
            baseChi2: rn.baseChi2,
            lnnorm: rn.lnnorm
        )
    }
}

/// Output of `_bound_companion_block`.
public struct FPPV2BoundCompanionBlock: Sendable {
    public var qsComp: [Double]
    public var mc: [Double]
    public var rc: [Double]
    public var tc: [Double]
    public var loggc: [Double]
    public var frc: [Double]
    public var u1c: [Double]
    public var u2c: [Double]
}

/// Exposure diagnostics tracker (`_EXPOSURE_STATS` / `_note_exposure_stats`).
public struct FPPV2ExposureStats: Equatable, Sendable {
    public var fallbackShare: Double = 0.0
    public var unconvergedShare: Double = 0.0

    public init() {}

    public mutating func note(lnL: [Double], status: [Int]) {
        var lw: [Double] = []
        var st: [Int] = []
        for i in 0..<min(lnL.count, status.count) where lnL[i].isFinite {
            lw.append(lnL[i])
            st.append(status[i])
        }
        guard let m = lw.max() else { return }
        let w = lw.map { FPPV2.exp($0 - m) }
        let tot = FPPNumpy.pairwiseSum(w)
        guard tot > 0.0 else { return }
        var w1: [Double] = []
        var w2: [Double] = []
        for i in 0..<w.count {
            if st[i] == 1 { w1.append(w[i]) }
            if st[i] == 2 { w2.append(w[i]) }
        }
        let fb = FPPNumpy.pairwiseSum(w1) / tot
        let unc = FPPNumpy.pairwiseSum(w2) / tot
        if fb > fallbackShare { fallbackShare = fb }
        if unc > unconvergedShare { unconvergedShare = unc }
    }
}

/// Detailed result of `_lnZ_STP` (including per-draw arrays for golden testing).
public struct FPPV2STPResult: Sendable {
    public var lnZ: Double
    public var lnL: [Double]
    public var lnprior: [Double]
    public var mask: [Bool]
    public var status: [Int]
}

/// Detailed result of `_lnZ_SEB` (including per-draw arrays for golden testing).
public struct FPPV2SEBResult: Sendable {
    public var lnZSEB: Double
    public var lnZSEBx2P: Double
    public var lnLSEB: [Double]
    public var lnLSEBx2P: [Double]
    public var lnprior: [Double]
    public var maskEB: [Bool]
    public var maskTwin: [Bool]
    public var statusEB: [Int]
    public var statusTwin: [Int]
}

public enum FPPV2SFamily {
    // MARK: - FITPACK splines

    private struct Spline: Sendable {
        let knots: [Double]
        let coefficients: [Double]
        let degree: Int

        init(knotBits: [UInt64], coefficientBits: [UInt64], degree: Int) {
            self.knots = knotBits.map(Double.init(bitPattern:))
            self.coefficients = coefficientBits.map(Double.init(bitPattern:))
            self.degree = degree
        }

        /// FITPACK `splev(..., ext=0)` (`fpbspl` recurrence, extrapolated outside end knots).
        func evaluate(_ x: Double) -> Double {
            let k = degree
            let n = knots.count
            var interval = k
            let upper = n - k - 2
            while interval < upper && x >= knots[interval + 1] { interval += 1 }
            var h = [Double](repeating: 0.0, count: 8)
            var hh = [Double](repeating: 0.0, count: 8)
            h[0] = 1.0
            if k > 0 {
                for j in 1...k {
                    for i in 0..<j { hh[i] = h[i] }
                    h[0] = 0.0
                    for i in 0..<j {
                        let li = knots[interval + i + 1]
                        let lj = knots[interval + i + 1 - j]
                        if li == lj {
                            h[i + 1] = 0.0
                            continue
                        }
                        let f = hh[i] / (li - lj)
                        h[i] += f * (li - x)
                        h[i + 1] = f * (x - lj)
                    }
                }
            }
            var result = 0.0
            for j in 0...k { result += coefficients[interval - k + j] * h[j] }
            return result
        }
    }

    private static let rTorres = Spline(
        knotBits: FPPStellarCoefficients.R_TORRES_T_BITS,
        coefficientBits: FPPStellarCoefficients.R_TORRES_C_BITS,
        degree: FPPStellarCoefficients.R_TORRES_K
    )
    private static let tTorres = Spline(
        knotBits: FPPStellarCoefficients.T_TORRES_T_BITS,
        coefficientBits: FPPStellarCoefficients.T_TORRES_C_BITS,
        degree: FPPStellarCoefficients.T_TORRES_K
    )
    private static let rCool = Spline(
        knotBits: FPPStellarCoefficients.R_CDWRF_T_BITS,
        coefficientBits: FPPStellarCoefficients.R_CDWRF_C_BITS,
        degree: FPPStellarCoefficients.R_CDWRF_K
    )
    private static let tCool = Spline(
        knotBits: FPPStellarCoefficients.T_CDWRF_T_BITS,
        coefficientBits: FPPStellarCoefficients.T_CDWRF_C_BITS,
        degree: FPPStellarCoefficients.T_CDWRF_K
    )
    private static let fluxSpline = Spline(
        knotBits: FPPStellarCoefficients.FLUX_T_BITS,
        coefficientBits: FPPStellarCoefficients.FLUX_C_BITS,
        degree: FPPStellarCoefficients.FLUX_K
    )

    // MARK: - Stellar relations

    /// `stellar_relations(masses)`: cool dwarf input clipped to `[0.1, 0.63]`, floors `R >= 0.1`, `T >= 2800`.
    public static func stellarRelations(masses: [Double]) -> (radii: [Double], temperatures: [Double]) {
        var r = [Double](repeating: 0.0, count: masses.count)
        var t = [Double](repeating: 0.0, count: masses.count)
        for i in 0..<masses.count {
            let m = masses[i]
            var ri: Double
            var ti: Double
            if m > 0.63 {
                ri = rTorres.evaluate(m)
                ti = tTorres.evaluate(m)
            } else {
                let mc = min(max(m, 0.1), 0.63)
                ri = rCool.evaluate(mc)
                ti = tCool.evaluate(mc)
            }
            if ri < 0.1 { ri = 0.1 }
            if ti < 2800.0 { ti = 2800.0 }
            r[i] = ri
            t[i] = ti
        }
        return (r, t)
    }

    /// `stellar_relations_capped(masses, max_R, max_T)`:
    /// Torres spline above 0.63 Msun, cool-dwarf spline at/below 0.63 Msun (EXTRAPOLATED outside
    /// its nodes, NOT input-clipped), then caps (`R > max_R -> max_R`, `T > max_T -> max_T`), and
    /// ONLY THEN floors (`R >= 0.1`, `T >= 2800.0`).
    public static func stellarRelationsCapped(
        masses: [Double],
        maxR: [Double],
        maxT: [Double]
    ) -> (radii: [Double], temperatures: [Double]) {
        let n = masses.count
        var r = [Double](repeating: 0.0, count: n)
        var t = [Double](repeating: 0.0, count: n)
        for i in 0..<n {
            let m = masses[i]
            let mr = maxR.count == 1 ? maxR[0] : maxR[i]
            let mt = maxT.count == 1 ? maxT[0] : maxT[i]
            var ri: Double
            var ti: Double
            if m > 0.63 {
                ri = rTorres.evaluate(m)
                ti = tTorres.evaluate(m)
            } else {
                ri = rCool.evaluate(m)
                ti = tCool.evaluate(m)
            }
            if ri > mr { ri = mr }
            if ti > mt { ti = mt }
            if ri < 0.1 { ri = 0.1 }
            if ti < 2800.0 { ti = 2800.0 }
            r[i] = ri
            t[i] = ti
        }
        return (r, t)
    }

    public static func stellarRelationsCapped(
        masses: [Double],
        maxR: Double,
        maxT: Double
    ) -> (radii: [Double], temperatures: [Double]) {
        stellarRelationsCapped(masses: masses, maxR: [maxR], maxT: [maxT])
    }

    /// `flux_relation(masses) = 10.0 ** _spl_flux(masses)`.
    public static func fluxRelation(masses: [Double]) -> [Double] {
        masses.map { FPPV2.pow(10.0, fluxSpline.evaluate($0)) }
    }

    public static func fluxRelationScalar(_ mass: Double) -> Double {
        FPPV2.pow(10.0, fluxSpline.evaluate(mass))
    }

    // MARK: - Limb darkening table & lookups

    private struct LDCKey: Hashable, Sendable {
        let zBits: UInt64
        let teff: Int
        let loggBits: UInt64

        init(z: Double, teff: Int, logg: Double) {
            self.zBits = (z == 0.0 ? 0.0 : z).bitPattern
            self.teff = teff
            self.loggBits = (logg == 0.0 ? 0.0 : logg).bitPattern
        }
    }

    private struct LDCTable: Sendable {
        let zs: [Double]
        let ts: [Int]
        let gs: [Double]
        let rows: [LDCKey: (u1: Double, u2: Double)]
    }

    private static let ldcTable: LDCTable = {
        var zSet = Set<Double>()
        var tSet = Set<Int>()
        var gSet = Set<Double>()
        var map: [LDCKey: (u1: Double, u2: Double)] = [:]
        map.reserveCapacity(FPPLimbDarkeningData.rows.count)
        for row in FPPLimbDarkeningData.rows {
            let z = row.z == 0.0 ? 0.0 : row.z
            let t = Int(row.teff)
            let g = row.logg == 0.0 ? 0.0 : row.logg
            zSet.insert(z)
            tSet.insert(t)
            gSet.insert(g)
            map[LDCKey(z: z, teff: t, logg: g)] = (row.u1, row.u2)
        }
        return LDCTable(
            zs: zSet.sorted(),
            ts: tSet.sorted(),
            gs: gSet.sorted(),
            rows: map
        )
    }()

    private static func nearestZ(_ z: Double) -> Double {
        let zs = ldcTable.zs
        var best = zs[0]
        var bestDist = abs(best - z)
        for cand in zs.dropFirst() {
            let d = abs(cand - z)
            if d < bestDist {
                best = cand
                bestDist = d
            }
        }
        return best
    }

    private static func nearestT(_ teff: Double) -> Int {
        let ts = ldcTable.ts
        var best = ts[0]
        var bestDist = abs(teff - Double(best))
        for cand in ts.dropFirst() {
            let d = abs(teff - Double(cand))
            if d < bestDist {
                best = cand
                bestDist = d
            }
        }
        return best
    }

    private static func nearestG(_ logg: Double) -> Double {
        let gs = ldcTable.gs
        var best = gs[0]
        var bestDist = abs(logg - best)
        for cand in gs.dropFirst() {
            let d = abs(logg - cand)
            if d < bestDist {
                best = cand
                bestDist = d
            }
        }
        return best
    }

    /// `get_ldc(Teff, logg, Z=0.0)`.
    public static func getLDC(teff: Double, logg: Double, z: Double = 0.0) throws -> (u1: Double, u2: Double) {
        let bz = nearestZ(z)
        let bt = nearestT(teff)
        let bg = nearestG(logg)
        guard let row = ldcTable.rows[LDCKey(z: bz, teff: bt, logg: bg)] else {
            throw FPPV2LDCNoRow("no limb-darkening row for Z=\(bz) Teff=\(bt) logg=\(bg)")
        }
        return row
    }

    /// `get_ldc_arrays(Teff, logg, Z=0.0)`.
    public static func getLDCArrays(
        teff: [Double],
        logg: [Double],
        z: Double = 0.0
    ) throws -> (u1: [Double], u2: [Double]) {
        if teff.count != logg.count {
            throw FPPV2SFamilyError("Teff (\(teff.count)) and logg (\(logg.count)) must have the same length")
        }
        let bz = nearestZ(z)
        let n = teff.count
        var u1 = [Double](repeating: 0.0, count: n)
        var u2 = [Double](repeating: 0.0, count: n)
        for i in 0..<n {
            let bt = nearestT(teff[i])
            let bg = nearestG(logg[i])
            guard let row = ldcTable.rows[LDCKey(z: bz, teff: bt, logg: bg)] else {
                throw FPPV2LDCNoRow("no limb-darkening row for Z=\(bz) Teff=\(bt) logg=\(bg)")
            }
            u1[i] = row.u1
            u2[i] = row.u2
        }
        return (u1, u2)
    }

    /// D6 background lookup: nearest Teff in the original CSV order, then
    /// nearest logg among that Teff's rows, then nearest metallicity.
    public static func ldcBackgroundArrays(
        teff: [Double], logg: [Double], metallicity: [Double]
    ) throws -> (u1: [Double], u2: [Double]) {
        guard teff.count == logg.count && teff.count == metallicity.count else {
            throw FPPV2SFamilyError("background LDC arrays must have equal lengths")
        }
        var u1 = [Double](repeating: 0.0, count: teff.count)
        var u2 = [Double](repeating: 0.0, count: teff.count)
        for i in teff.indices {
            var candidates = FPPLimbDarkeningData.rows
            guard let first = candidates.first else { throw FPPV2LDCNoRow("background LDC table is empty") }
            var chosenTeff = first.teff
            var best = abs(teff[i] - first.teff)
            for row in candidates.dropFirst() {
                let distance = abs(teff[i] - row.teff)
                if distance < best { best = distance; chosenTeff = row.teff }
            }
            candidates = candidates.filter { $0.teff == chosenTeff }
            guard let firstG = candidates.first else { throw FPPV2LDCNoRow("no background LDC Teff row") }
            var chosenLogG = firstG.logg
            best = abs(logg[i] - firstG.logg)
            for row in candidates.dropFirst() {
                let distance = abs(logg[i] - row.logg)
                if distance < best { best = distance; chosenLogG = row.logg }
            }
            candidates = candidates.filter { $0.logg == chosenLogG }
            guard let firstZ = candidates.first else { throw FPPV2LDCNoRow("no background LDC gravity row") }
            var chosenZ = firstZ.z
            best = abs(metallicity[i] - firstZ.z)
            for row in candidates.dropFirst() {
                let distance = abs(metallicity[i] - row.z)
                if distance < best { best = distance; chosenZ = row.z }
            }
            let matches = candidates.filter { $0.z == chosenZ }
            guard matches.count == 1, let row = matches.first else {
                throw FPPV2LDCNoRow("background LDC lookup is not unique")
            }
            u1[i] = row.u1; u2[i] = row.u2
        }
        return (u1, u2)
    }

    /// `ldc_companion_arrays(Teff, logg, teff_max, Z=0.0)`:
    /// `logg` rounded to 0.5 (ties-to-even) and clipped to `[3.5, 5.0]`,
    /// `Teff` rounded to 250 K (ties-to-even) and clipped to `[3500.0, teffMax]`,
    /// exact table lookup at nearest `Z`; throws `FPPV2LDCNoRow` on a missing row.
    public static func ldcCompanionArrays(
        teff: [Double],
        logg: [Double],
        teffMax: Double,
        z: Double = 0.0
    ) throws -> (u1: [Double], u2: [Double]) {
        if teff.count != logg.count {
            throw FPPV2SFamilyError("Teff (\(teff.count)) and logg (\(logg.count)) must have the same length")
        }
        let bz = nearestZ(z)
        let n = teff.count
        var u1 = [Double](repeating: 0.0, count: n)
        var u2 = [Double](repeating: 0.0, count: n)
        for i in 0..<n {
            var rl = (logg[i] / 0.5).rounded(.toNearestOrEven) * 0.5
            if rl < 3.5 { rl = 3.5 }
            if rl > 5.0 { rl = 5.0 }
            var rt = (teff[i] / 250.0).rounded(.toNearestOrEven) * 250.0
            if rt < 3500.0 { rt = 3500.0 }
            if rt > teffMax { rt = teffMax }
            let key = LDCKey(z: bz, teff: Int(rt), logg: rl)
            guard let row = ldcTable.rows[key] else {
                throw FPPV2LDCNoRow("no limb-darkening row for Z=\(bz) Teff=\(Int(rt)) logg=\(rl) (S family)")
            }
            u1[i] = row.u1
            u2[i] = row.u2
        }
        return (u1, u2)
    }

    // MARK: - Samplers

    private static func uniformVector(_ rng: inout FPPRNG, count: Int) -> [Double] {
        var out = [Double](repeating: 0.0, count: count)
        for i in 0..<count { out[i] = rng.rand() }
        return out
    }

    private static func transformRpOne(_ u: Double, heavy: Bool) -> Double {
        let rBreak1 = 3.0, rBreak2 = 6.0, rMin = 0.5, rMax = 20.0
        let p1 = 0.0
        let p2 = heavy ? -4.0 : -7.0
        let p3 = -0.5
        let a1 = FPPV2.pow(rBreak1, p1) / FPPV2.pow(rBreak1, p2)
        let a2 = FPPV2.pow(rBreak2, p2) / FPPV2.pow(rBreak2, p3)
        let i1 = (FPPV2.pow(rBreak1, p1 + 1.0) - FPPV2.pow(rMin, p1 + 1.0)) / (p1 + 1.0)
        let i2 = a1 * (FPPV2.pow(rBreak2, p2 + 1.0) - FPPV2.pow(rBreak1, p2 + 1.0)) / (p2 + 1.0)
        let i3 = a2 * a1 * (FPPV2.pow(rMax, p3 + 1.0) - FPPV2.pow(rBreak2, p3 + 1.0)) / (p3 + 1.0)
        let norm = 1.0 / (i1 + i2 + i3)
        if u <= norm * i1 {
            return FPPV2.pow(u / norm * (p1 + 1.0) + FPPV2.pow(rMin, p1 + 1.0), 1.0 / (p1 + 1.0))
        } else if u > norm * i1 && u <= norm * (i1 + i2) {
            return FPPV2.pow((u / norm - i1) * (p2 + 1.0) / a1 + FPPV2.pow(rBreak1, p2 + 1.0), 1.0 / (p2 + 1.0))
        } else {
            // Python: `x[m3] = (...)**(1 / (p3 + 1))` on a numpy ARRAY with the scalar exponent 2.0, which numpy's
            // fast scalar power evaluates as np.square (x * x), not libm pow; glibc pow(x, 2.0) differs from x * x
            // in the last bit for some x (task 33: draw 1027 of the rad = 3 mass-fallback case). Rust's `powf(2.0)`
            // is lowered to x * x by LLVM, which is why only Swift differed.
            let base = (u / norm - i1 - i2) * (p3 + 1.0) / (a1 * a2) + FPPV2.pow(rBreak2, p3 + 1.0)
            let exponent = 1.0 / (p3 + 1.0)
            return exponent == 2.0 ? base * base : FPPV2.pow(base, exponent)
        }
    }

    /// `_sample_rp(N, M_s, rng)` for scalar `M_s`.
    public static func sampleRp(count n: Int, hostMass: Double, rng: inout FPPRNG) -> [Double] {
        var x = uniformVector(&rng, count: n)
        let heavy = hostMass > 0.45
        for i in 0..<n {
            x[i] = transformRpOne(x[i], heavy: heavy)
        }
        return x
    }

    /// `_sample_rp(N, M_s_array, rng)` for array `M_s` (S family).
    public static func sampleRp(count n: Int, hostMasses: [Double], rng: inout FPPRNG) throws -> [Double] {
        let x = uniformVector(&rng, count: n)
        if hostMasses.count != n {
            throw FPPV2SFamilyError("host mass array must have shape (\(n),), got (\(hostMasses.count),)")
        }
        var out = [Double](repeating: 0.0, count: n)
        for i in 0..<n {
            out[i] = transformRpOne(x[i], heavy: hostMasses[i] > 0.45)
        }
        return out
    }

    /// `_sample_inc(N, rng)`.
    public static func sampleInc(count n: Int, rng: inout FPPRNG) -> [Double] {
        let x = uniformVector(&rng, count: n)
        let norm = 1.0
        return x.map { FPPV2.acos(1.0 - $0 / norm) * 180.0 / FPPV2Consts.pi }
    }

    /// `_sample_ecc(N, planet, P_orb, rng)`.
    public static func sampleEcc(count n: Int, planet: Bool, period: Double, rng: inout FPPRNG) -> [Double] {
        if planet {
            return (0..<n).map { _ in rng.beta(0.867, 3.030) }
        }
        let nuPlus1 = period <= 10.0 ? 0.2 : 0.6
        return (0..<n).map { _ in rng.power(nuPlus1) }
    }

    /// `_sample_w(N, rng)`.
    public static func sampleW(count n: Int, rng: inout FPPRNG) -> [Double] {
        uniformVector(&rng, count: n).map { $0 * 360.0 }
    }

    private static func transformMassRatios(
        _ x: inout [Double],
        qMin: Double,
        p1: Double,
        p2: Double,
        fTwin: Double,
        threeSeg: Bool,
        upstream: Bool
    ) {
        // NumPy's float64 power ufunc has a dedicated exponent-2 path that
        // evaluates x * x. Keep that operation order for the vectorized
        // `_sample_q` result; libm pow(x, 2) can differ by one ULP.
        @inline(__always) func numpyPower(_ value: Double, _ exponent: Double) -> Double {
            exponent == 2.0 ? value * value : FPPV2.pow(value, exponent)
        }
        if threeSeg {
            let a1 = FPPV2.pow(0.3, p1) / FPPV2.pow(0.3, p2)
            let a2 = 1.0 + (fTwin / (1.0 - fTwin))
                * ((FPPV2.pow(1.0, p2 + 1.0) - FPPV2.pow(0.3, p2 + 1.0)) / (p2 + 1.0))
                / ((FPPV2.pow(1.0, p2 + 1.0) - FPPV2.pow(0.95, p2 + 1.0)) / (p2 + 1.0))
            let i1 = (FPPV2.pow(0.3, p1 + 1.0) - FPPV2.pow(qMin, p1 + 1.0)) / (p1 + 1.0)
            let i2 = a1 * (FPPV2.pow(0.95, p2 + 1.0) - FPPV2.pow(0.3, p2 + 1.0)) / (p2 + 1.0)
            let i3 = a2 * a1 * (FPPV2.pow(1.0, p2 + 1.0) - FPPV2.pow(0.95, p2 + 1.0)) / (p2 + 1.0)
            let norm = 1.0 / (i1 + i2 + i3)
            for i in 0..<x.count {
                let u = x[i]
                let m1 = u <= norm * i1
                let m2 = (u > norm * i1) && (u <= norm * (i1 + i2))
                let m3 = u > norm * (i1 + i2)
                if m1 {
                    x[i] = numpyPower(u / norm * (p1 + 1.0) + FPPV2.pow(qMin, p1 + 1.0), 1.0 / (p1 + 1.0))
                } else if m2 {
                    x[i] = numpyPower((u / norm - i1) * (p2 + 1.0) / a1 + FPPV2.pow(0.3, p2 + 1.0), 1.0 / (p2 + 1.0))
                } else if m3 {
                    x[i] = numpyPower((u / norm - i1 - i2) * (p2 + 1.0) / (a1 * a2) + FPPV2.pow(0.95, p2 + 1.0), 1.0 / (p2 + 1.0))
                }
            }
        } else {
            let a2 = 1.0 + (fTwin / (1.0 - fTwin))
                * ((FPPV2.pow(1.0, p2 + 1.0) - FPPV2.pow(qMin, p2 + 1.0)) / (p2 + 1.0))
                / ((FPPV2.pow(1.0, p2 + 1.0) - FPPV2.pow(0.95, p2 + 1.0)) / (p2 + 1.0))
            let i2 = (FPPV2.pow(0.95, p2 + 1.0) - FPPV2.pow(qMin, p2 + 1.0)) / (p2 + 1.0)
            let i3 = a2 * (FPPV2.pow(1.0, p2 + 1.0) - FPPV2.pow(0.95, p2 + 1.0)) / (p2 + 1.0)
            let norm = 1.0 / (i2 + i3)
            for i in 0..<x.count {
                let u = x[i]
                if u <= norm * i2 {
                    x[i] = numpyPower(u / norm * (p2 + 1.0) + FPPV2.pow(qMin, p2 + 1.0), 1.0 / (p2 + 1.0))
                } else {
                    x[i] = numpyPower((u / norm - i2) * (p2 + 1.0) / a2 + FPPV2.pow(0.95, p2 + 1.0), 1.0 / (p2 + 1.0))
                }
            }
        }
        if !upstream {
            for i in 0..<x.count {
                x[i] = min(1.0, max(qMin, x[i]))
            }
        }
    }

    /// `_sample_q(N, M_s, rng)`.
    public static func sampleQ(
        count n: Int,
        hostMass: Double,
        rng: inout FPPRNG,
        qBranchUpstream: Bool = false
    ) -> [Double] {
        var x = uniformVector(&rng, count: n)
        let qMin: Double
        if hostMass >= 1.0 {
            qMin = 0.1
        } else if hostMass > 0.1 {
            qMin = 0.1 / hostMass
        } else {
            return [Double](repeating: 1.0, count: n)
        }
        let threeSeg = qBranchUpstream ? (hostMass >= 0.3) : (qMin < 0.3)
        transformMassRatios(
            &x, qMin: qMin, p1: 0.3, p2: -0.5, fTwin: 0.30,
            threeSeg: threeSeg, upstream: qBranchUpstream
        )
        return x
    }

    /// Transform an already-drawn uniform vector through `_sample_q`; used
    /// when background hosts require one host-mass-specific transform per group.
    public static func sampleQFromUniforms(
        _ uniforms: [Double], hostMass: Double, qBranchUpstream: Bool = false
    ) -> [Double] {
        var x = uniforms
        let qMin: Double
        if hostMass >= 1.0 { qMin = 0.1 }
        else if hostMass > 0.1 { qMin = 0.1 / hostMass }
        else { return [Double](repeating: 1.0, count: x.count) }
        let threeSeg = qBranchUpstream ? (hostMass >= 0.3) : (qMin < 0.3)
        transformMassRatios(&x, qMin: qMin, p1: 0.3, p2: -0.5, fTwin: 0.30,
                            threeSeg: threeSeg, upstream: qBranchUpstream)
        return x
    }

    /// `_sample_q_companion(N, M_s, rng)`.
    public static func sampleQCompanion(
        count n: Int,
        hostMass: Double,
        rng: inout FPPRNG,
        qBranchUpstream: Bool = false
    ) -> [Double] {
        var x = uniformVector(&rng, count: n)
        let qMin = hostMass > 0.1 ? max(0.1 / hostMass, 0.1) : 1.0
        if hostMass <= 0.1 {
            return [Double](repeating: 1.0, count: n)
        }
        let threeSeg = qBranchUpstream ? (hostMass >= 0.3) : (qMin < 0.3)
        transformMassRatios(
            &x, qMin: qMin, p1: 0.3, p2: -0.95, fTwin: 0.05,
            threeSeg: threeSeg, upstream: qBranchUpstream
        )
        return x
    }

    // MARK: - Bound companion priors (N1 constants)

    /// `_separation_at_contrast(delta_mags)`: single-point `np.interp` returns 2.2 arcsec.
    public static func separationAtContrast(_ deltaMags: [Double]) -> [Double] {
        [Double](repeating: 2.2, count: deltaMags.count)
    }

    private static func boundCompanionCoeffs(_ mS: Double) -> (f1: Double, f2: Double, f3: Double, alpha: Double, dlogP: Double) {
        let logM = FPPV2.log10(mS)
        let logM2 = FPPV2.pow(logM, 2.0)
        let f1 = 0.020 + 0.04 * logM + 0.07 * logM2
        let f2 = 0.039 + 0.07 * logM + 0.01 * logM2
        let f3 = 0.078 - 0.05 * logM + 0.04 * logM2
        return (f1, f2, f3, 0.018, 0.7)
    }

    private static func log10MaxPorbs(seps: [Double], mS: Double) -> [Double] {
        let num = 4.0 * FPPV2.pow(FPPV2Consts.pi, 2.0)
        let den = (FPPV2Consts.gCGS * mS) * FPPV2Consts.msun
        let coeff = num / den
        return seps.map { sep in
            let sepAu = sep * FPPV2Consts.au
            let cubed = FPPV2.pow(sepAu, 3.0)
            let maxP = (coeff * cubed).squareRoot() / 86400.0
            return FPPV2.log10(maxP)
        }
    }

    private static func fCompTP(mS: Double, seps: [Double]) -> [Double] {
        let (_, f2, f3, alpha, dlogP) = boundCompanionCoeffs(mS)
        let logMP = log10MaxPorbs(seps: seps, mS: mS)
        let t4 = alpha * dlogP * (5.5 - 3.4) + f2 * (5.5 - 3.4)
            + (f3 - f2 - alpha * dlogP)
            * (0.238095 * FPPV2.pow(5.5, 2.0) - 0.952381 * 5.5 + 0.485714)
        let t5 = f3 * (3.33333 - 17.3566 * FPPV2.exp(-0.3 * 8.0))
        return logMP.map { lm in
            if lm >= 3.4 && lm < 5.5 {
                return alpha * dlogP * (lm - 3.4) + f2 * (lm - 3.4)
                    + (f3 - f2 - alpha * dlogP)
                    * (0.238095 * (lm * lm) - 0.952381 * lm + 0.485714)
            } else if lm >= 5.5 && lm < 8.0 {
                return t4 + f3 * (3.33333 - 17.3566 * FPPV2.exp(-0.3 * lm))
            } else if lm >= 8.0 {
                return t4 + t5
            } else {
                return 0.0
            }
        }
    }

    private static func fCompEB(mS: Double, seps: [Double]) -> [Double] {
        let (f1, f2, f3, alpha, dlogP) = boundCompanionCoeffs(mS)
        let logMP = log10MaxPorbs(seps: seps, mS: mS)
        let t2 = 0.5 * (2.0 - 1.0) * (2.0 * f1 + (f2 - f1 - alpha * dlogP) * (2.0 - 1.0))
        let t3 = 0.5 * alpha * (FPPV2.pow(3.4, 2.0) - 5.4 * 3.4 + 6.8) + f2 * (3.4 - 2.0)
        let t4 = alpha * dlogP * (5.5 - 3.4) + f2 * (5.5 - 3.4)
            + (f3 - f2 - alpha * dlogP)
            * (0.238095 * FPPV2.pow(5.5, 2.0) - 0.952381 * 5.5 + 0.485714)
        let t5 = f3 * (3.33333 - 17.3566 * FPPV2.exp(-0.3 * 8.0))
        return logMP.map { lm in
            if lm >= 1.0 && lm < 2.0 {
                return 0.5 * (lm - 1.0) * (2.0 * f1 + (f2 - f1 - alpha * dlogP) * (lm - 1.0))
            } else if lm >= 2.0 && lm < 3.4 {
                return t2 + 0.5 * alpha * (lm * lm - 5.4 * lm + 6.8) + f2 * (lm - 2.0)
            } else if lm >= 3.4 && lm < 5.5 {
                return t2 + t3 + (alpha * dlogP * (lm - 3.4) + f2 * (lm - 3.4)
                    + (f3 - f2 - alpha * dlogP)
                    * (0.238095 * (lm * lm) - 0.952381 * lm + 0.485714))
            } else if lm >= 5.5 && lm < 8.0 {
                return t2 + t3 + t4 + f3 * (3.33333 - 17.3566 * FPPV2.exp(-0.3 * lm))
            } else if lm >= 8.0 {
                return t2 + t3 + t4 + t5
            } else {
                return 0.0
            }
        }
    }

    /// `_lnprior_bound_TP(M_s, plx, delta_mags)`.
    public static func lnpriorBoundTP(
        massSolar mS: Double,
        parallax plx: Double,
        deltaMagsAbs: [Double]
    ) -> [Double] {
        let effPlx = plx.isNaN ? 0.1 : plx
        let d = 1000.0 / effPlx
        let seps = separationAtContrast(deltaMagsAbs).map { d * $0 }
        if mS >= 1.0 {
            return fCompTP(mS: mS, seps: seps).map { FPPV2.log($0) }
        }
        let mAct = mS
        return fCompTP(mS: 1.0, seps: seps).map { fc in
            var fAct = 0.65 * fc + 0.35 * fc * mAct
            if fAct < 0.0 { fAct = 0.0 }
            return FPPV2.log(fAct)
        }
    }

    /// `_lnprior_bound_EB(M_s, plx, delta_mags)`.
    public static func lnpriorBoundEB(
        massSolar mS: Double,
        parallax plx: Double,
        deltaMagsAbs: [Double]
    ) -> [Double] {
        let effPlx = plx.isNaN ? 0.1 : plx
        let d = 1000.0 / effPlx
        let seps = separationAtContrast(deltaMagsAbs).map { d * $0 }
        if mS >= 1.0 {
            return fCompEB(mS: mS, seps: seps).map { FPPV2.log($0) }
        }
        let mAct = mS
        return fCompEB(mS: 1.0, seps: seps).map { fc in
            var fAct = 0.65 * fc + 0.35 * fc * mAct
            if fAct < 0.0 { fAct = 0.0 }
            return FPPV2.log(fAct)
        }
    }

    /// `_lnprior_companion(M_s, plx, fluxratios_comp, scenario)`.
    public static func lnpriorCompanion(
        massSolar mS: Double,
        parallax plx: Double,
        fluxRatiosCompanion: [Double],
        scenario: String
    ) -> [Double] {
        let dm = fluxRatiosCompanion.map { 2.5 * FPPV2.log10($0 / (1.0 - $0)) }
        let dmAbs = dm.map { abs($0) }
        var lp = scenario == "TP"
            ? lnpriorBoundTP(massSolar: mS, parallax: plx, deltaMagsAbs: dmAbs)
            : lnpriorBoundEB(massSolar: mS, parallax: plx, deltaMagsAbs: dmAbs)
        for i in 0..<lp.count {
            if lp[i] > 0.0 { lp[i] = 0.0 }
            if dm[i] > 0.0 { lp[i] = -.infinity }
        }
        return lp
    }

    // MARK: - Companion block, Roche mask, logsumexp

    /// `_bound_companion_block(N, M_s, R_s, Teff, rng, teff_max)`.
    public static func boundCompanionBlock(
        count n: Int,
        massSolar mS: Double,
        radiusSolar rS: Double,
        teff: Double,
        rng: inout FPPRNG,
        teffMax: Double,
        qBranchUpstream: Bool = false
    ) throws -> FPPV2BoundCompanionBlock {
        let qsComp = sampleQCompanion(count: n, hostMass: mS, rng: &rng, qBranchUpstream: qBranchUpstream)
        let mc = qsComp.map { $0 * mS }
        let (rc, tc) = stellarRelationsCapped(masses: mc, maxR: rS, maxT: teff)
        let loggc: [Double] = (0..<n).map { i in
            let mcMsun = mc[i] * FPPV2Consts.msun
            let rcRsun = rc[i] * FPPV2Consts.rsun
            return FPPV2.log10((FPPV2Consts.gCGS * mcMsun) / (rcRsun * rcRsun))
        }
        let fMc = fluxRelation(masses: mc)
        let fMs = fluxRelationScalar(mS)
        let frc = fMc.map { $0 / ($0 + fMs) }
        let (u1c, u2c) = try ldcCompanionArrays(teff: tc, logg: loggc, teffMax: teffMax)
        return FPPV2BoundCompanionBlock(
            qsComp: qsComp, mc: mc, rc: rc, tc: tc,
            loggc: loggc, frc: frc, u1c: u1c, u2c: u2c
        )
    }

    /// `_roche_ok(a_cm, ecc, M_s, R_s, rho_comp)`.
    public static func rocheOk(
        aCm: [Double],
        ecc: [Double],
        hostMass: [Double],
        hostRadius: [Double],
        rhoComp: [Double]
    ) -> [Bool] {
        let n = aCm.count
        var out = [Bool](repeating: false, count: n)
        let fourThirdsPi = (4.0 / 3.0) * FPPV2Consts.pi
        for i in 0..<n {
            let ms = hostMass.count == 1 ? hostMass[0] : hostMass[i]
            let rs = hostRadius.count == 1 ? hostRadius[0] : hostRadius[i]
            let rc = rhoComp.count == 1 ? rhoComp[0] : rhoComp[i]
            let rHostCm = rs * FPPV2Consts.rsun
            let rhoHost = (ms * FPPV2Consts.msun) / (fourThirdsPi * FPPV2.pow(rHostCm, 3.0))
            let periastron = aCm[i] * (1.0 - ecc[i])
            let limit = (2.44 * rHostCm) * FPPV2.pow(rhoHost / rc, 1.0 / 3.0)
            out[i] = periastron >= limit
        }
        return out
    }

    /// `_lnZ_from_logw(logw)`: log-mean-exp using `FPPNumpy.pairwiseSum` over the full N-element array.
    /// With a step-7 sink open on this thread (`FPPV3MC.DiagSink`) it appends ONE draw-weight record, as Python
    /// appends to `_DRAW_DIAG` (r2 :2478-2509): ESS = (s1 * s1) / s2 with pairwise s1 = sum(e), s2 = sum(e * e);
    /// max_share = 1 / s1; the tail scale; {ess 0, n, max_share 0, tail NaN} when nothing is finite.
    /// The returned lnZ bits never depend on it.
    public static func lnZFromLogw(_ logw: [Double]) -> Double {
        lnZFromLogw(logw, sink: FPPV3MC.activeSink())
    }

    /// `lnZFromLogw` that never records (probe-only reductions Python does not perform in the pass).
    public static func lnZFromLogwUnrecorded(_ logw: [Double]) -> Double {
        lnZFromLogw(logw, sink: nil)
    }

    static func lnZFromLogw(_ logw: [Double], sink: FPPV3MC.SinkBox?) -> Double {
        let n = logw.count
        if n == 0 {
            sink?.records.append(FPPV3MC.DrawDiag(ess: 0.0, n: 0, maxShare: 0.0, tailScale: .nan))
            if sink != nil { FPPV3Gate.pushReduction() }
            return -.infinity
        }
        var w = [Double](repeating: -.infinity, count: n)
        var anyFinite = false
        var m = -Double.infinity
        for i in 0..<n {
            let v = logw[i].isNaN ? -Double.infinity : logw[i]
            w[i] = v
            if v.isFinite {
                anyFinite = true
                if v > m { m = v }
            } else if v > m {
                m = v
            }
        }
        if !anyFinite {
            sink?.records.append(FPPV3MC.DrawDiag(ess: 0.0, n: n, maxShare: 0.0, tailScale: .nan))
            if sink != nil { FPPV3Gate.pushReduction() }
            return -.infinity
        }
        let expW = w.map { FPPV2.exp($0 - m) }
        let sumExp = FPPNumpy.pairwiseSum(expW)
        if let sink {
            let s2 = FPPNumpy.pairwiseSum(expW.map { $0 * $0 })
            sink.records.append(FPPV3MC.DrawDiag(ess: (sumExp * sumExp) / s2, n: n, maxShare: 1.0 / sumExp,
                                                 tailScale: FPPV3MC.tailScale(w)))
            FPPV3Gate.pushReduction()
        }
        return m + FPPV2.log(sumExp) - FPPV2.log(Double(n))
    }

    // MARK: - Batch likelihood kernels (per-draw u1 / u2)

    /// `_batch_transit_lnL` with per-draw `(u1, u2)` arrays.
    public static func batchTransitLnL(
        time: [Double],
        flux: [Double],
        sigma: [Double],
        ks: [Double],
        pOrbs: [Double],
        incsRad: [Double],
        aRsArr: [Double],
        eccs: [Double],
        wsRad: [Double],
        u1: [Double],
        u2: [Double],
        companionFluxRatio: [Double],
        companionIsHost: Bool,
        exptime: Double,
        nsub: Int
    ) -> [Double] {
        let n = ks.count
        let tCount = time.count
        var lnL = [Double](repeating: -.infinity, count: n)
        if n == 0 { return lnL }

        var meanLnSigma = 0.0
        for t in 0..<tCount {
            meanLnSigma += FPPV2.log(sigma[t])
        }
        meanLnSigma /= Double(tCount)

        for j in 0..<n {
            let u1j = u1.count == 1 ? u1[0] : u1[j]
            let u2j = u2.count == 1 ? u2[0] : u2[j]
            var model = FPPV2Orbit.lcCore(
                time: time, k: ks[j], t0: 0.0, period: pOrbs[j], aRs: aRsArr[j],
                incRad: incsRad[j], ecc: eccs[j], wRad: wsRad[j],
                u1: u1j, u2: u2j, exptime: exptime, nsub: nsub
            )
            let fr = companionFluxRatio[j]
            if fr > 0.0 {
                let fDilute: Double
                if companionIsHost {
                    fDilute = 1.0 / (fr / (1.0 - fr))
                } else {
                    fDilute = fr / (1.0 - fr)
                }
                for t in 0..<tCount {
                    model[t] = (model[t] + fDilute) / (1.0 + fDilute)
                }
            }
            var chi2 = 0.0
            for t in 0..<tCount {
                let diff = (flux[t] - model[t]) / sigma[t]
                chi2 += diff * diff
            }
            lnL[j] = -0.5 * FPPV2Consts.ln2pi - meanLnSigma - 0.5 * chi2
        }
        return lnL
    }

    /// `_batch_transit_lnL_prod` with per-draw `(u1, u2)` arrays.
    public static func batchTransitLnLProd(
        fold: FPPV2ProdFold,
        ks: [Double],
        pOrbs: [Double],
        incsRad: [Double],
        aRsArr: [Double],
        eccs: [Double],
        wsRad: [Double],
        u1: [Double],
        u2: [Double],
        companionFluxRatio: [Double],
        companionIsHost: Bool
    ) -> (lnL: [Double], status: [Int], nAff: [Int]) {
        if let gpu = FPPV3LikelihoodRouter.backend {
            return FPPV3LikelihoodRouter.routeTransit(gpu, fold: fold, ks: ks, pOrbs: pOrbs, incsRad: incsRad, aRsArr: aRsArr,
                eccs: eccs, wsRad: wsRad, u1: u1, u2: u2, companionFluxRatio: companionFluxRatio, companionIsHost: companionIsHost)
        }
        let n = ks.count
        var lnL = [Double](repeating: -.infinity, count: n)
        var status = [Int](repeating: 0, count: n)
        var nAffOut = [Int](repeating: 0, count: n)
        let f = fold.f
        let sig = fold.arrays.sigma
        var gateDiscrete: FPPV3Gate.DiscreteRows? = FPPV3Gate.active ? .init() : nil

        for j in 0..<n {
            let fr = companionFluxRatio[j]
            var dilA = 1.0
            var dilB = 0.0
            if fr > 0.0 {
                let fRatio: Double
                if companionIsHost {
                    fRatio = 1.0 / (fr / (1.0 - fr))
                } else {
                    fRatio = fr / (1.0 - fr)
                }
                dilA = 1.0 / (1.0 + fRatio)
                dilB = fRatio / (1.0 + fRatio)
            }
            let u1j = u1.count == 1 ? u1[0] : u1[j]
            let u2j = u2.count == 1 ? u2[0] : u2[j]
            let r = FPPV2Ladder.drawBinModels(
                w: fold.w, delta: fold.delta, fold: fold.arrays,
                period: pOrbs[j], aRs: aRsArr[j], incRad: incsRad[j],
                ecc: eccs[j], wRad: wsRad[j], k: ks[j],
                u1: u1j, u2: u2j, dilA: dilA, dilB: dilB
            )
            var chi2 = fold.baseChi2
            for i in 0..<r.nAff {
                let b = r.idx[i]
                let d = (f[b] - r.m[i]) / sig[b]
                chi2 += d * d
                let d0 = (f[b] - 1.0) / sig[b]
                chi2 -= d0 * d0
            }
            lnL[j] = fold.lnnorm - 0.5 * chi2
            status[j] = r.status
            nAffOut[j] = r.nAff
            gateDiscrete?.pushDraw(r)
        }
        if FPPV3Gate.active {
            let input = FPPV3Gate.CallInput(
                k: ks, period: pOrbs, inc: incsRad, aRs: aRsArr, ecc: eccs, wPeri: wsRad,
                u1: u1, u2: u2, companionFluxRatio: companionFluxRatio,
                ebArrays: [], sigmaVeto: nil, flags: [companionIsHost], fold: fold
            )
            FPPV3Gate.kernelRows(periods: pOrbs, aRs: aRsArr, incsRad: incsRad, eccs: eccs, wsRad: wsRad,
                                 ksUsed: ks, lnl: lnL, status: status,
                                 input: input, deviceRoots: nil, discrete: gateDiscrete)
        }
        return (lnL, status, nAffOut)
    }

    /// `_eb_secondary_min_flux`.
    public static func ebSecondaryMinFlux(
        kRaw: Double,
        p: Double,
        aRs: Double,
        cosI: Double,
        ecc: Double,
        wRad: Double,
        u1: Double,
        u2: Double,
        u1Sec: Double,
        u2Sec: Double,
        secConsistent: Bool,
        kNudgeUpstream: Bool
    ) -> Double {
        let nSamp = 25
        let halfSpan = 0.05
        var kS = 1.0 / kRaw
        if kNudgeUpstream && abs(kS - 1.0) < 1e-6 {
            kS = kS * 0.999
        }
        let wSec = wRad + FPPV2Consts.pi
        let mTS = FPPV2Orbit.midTransitMeanAnomaly(ecc, wSec)
        var secFlux = Double.infinity
        for i in 0..<nSamp {
            let tI = -halfSpan + 2.0 * halfSpan * Double(i) / Double(nSamp - 1)
            var z = FPPV2Orbit.zOne(tI, 0.0, p, aRs, cosI, ecc, wSec, mTS)
            let uu1: Double
            let uu2: Double
            if secConsistent {
                z = z * kS
                uu1 = u1Sec
                uu2 = u2Sec
            } else {
                uu1 = u1
                uu2 = u2
            }
            let f = FPPV2Kernel.transitFlux(z, kS, uu1, uu2)
            if f < secFlux {
                secFlux = f
            }
        }
        return secFlux
    }

    /// `_batch_eb_transit_lnL` with per-draw `(u1, u2)` arrays.
    public static func batchEBTransitLnL(
        time: [Double],
        flux: [Double],
        sigma: [Double],
        sigmaVeto: Double,
        ks: [Double],
        ebFluxRatios: [Double],
        pOrbs: [Double],
        incsRad: [Double],
        aRsArr: [Double],
        eccs: [Double],
        wsRad: [Double],
        u1: [Double],
        u2: [Double],
        companionFluxRatio: [Double],
        companionIsHost: Bool,
        exptime: Double,
        nsub: Int,
        rejectSec: Bool,
        secConsistent: Bool,
        kNudgeUpstream: Bool,
        u1Sec: [Double],
        u2Sec: [Double]
    ) -> [Double] {
        let n = ks.count
        let tCount = time.count
        var lnL = [Double](repeating: -.infinity, count: n)
        if n == 0 { return lnL }

        var meanLnSigma = 0.0
        for t in 0..<tCount {
            meanLnSigma += FPPV2.log(sigma[t])
        }
        meanLnSigma /= Double(tCount)

        let fTarget = 1.0
        for j in 0..<n {
            let kRaw = ks[j]
            var k = kRaw
            if kNudgeUpstream && abs(k - 1.0) < 1e-6 {
                k = k * 0.999
            }
            let u1j = u1.count == 1 ? u1[0] : u1[j]
            let u2j = u2.count == 1 ? u2[0] : u2[j]
            var fluxM = FPPV2Orbit.lcCore(
                time: time, k: k, t0: 0.0, period: pOrbs[j], aRs: aRsArr[j],
                incRad: incsRad[j], ecc: eccs[j], wRad: wsRad[j],
                u1: u1j, u2: u2j, exptime: exptime, nsub: nsub
            )
            var secFlux = ebSecondaryMinFlux(
                kRaw: kRaw, p: pOrbs[j], aRs: aRsArr[j], cosI: FPPV2.cos(incsRad[j]),
                ecc: eccs[j], wRad: wsRad[j], u1: u1j, u2: u2j,
                u1Sec: u1Sec[j], u2Sec: u2Sec[j],
                secConsistent: secConsistent, kNudgeUpstream: kNudgeUpstream
            )
            let frEB = ebFluxRatios[j]
            let fEB = frEB < 1.0 ? (frEB / (1.0 - frEB)) : 1e6
            let fc = companionFluxRatio[j]
            let fComp = fc > 0.0 ? (fc / (1.0 - fc)) : 0.0

            let fDilute: Double
            if companionIsHost {
                let fcClamped = max(fComp, 1e-30)
                let ratio = fEB / fcClamped
                for t in 0..<tCount {
                    fluxM[t] = (fluxM[t] + ratio) / (1.0 + ratio)
                }
                secFlux = (secFlux + fcClamped / fEB) / (1.0 + fcClamped / fEB)
                fDilute = fTarget / (fcClamped + fEB)
            } else {
                let ratio = fEB / fTarget
                for t in 0..<tCount {
                    fluxM[t] = (fluxM[t] + ratio) / (1.0 + ratio)
                }
                secFlux = (secFlux + fTarget / fEB) / (1.0 + fTarget / fEB)
                fDilute = fComp / (fTarget + fEB)
            }

            for t in 0..<tCount {
                fluxM[t] = (fluxM[t] + fDilute) / (1.0 + fDilute)
            }
            let secdepth = 1.0 - (secFlux + fDilute) / (1.0 + fDilute)
            if rejectSec && secdepth >= 1.5 * sigmaVeto {
                continue
            }

            var chi2 = 0.0
            for t in 0..<tCount {
                let diff = (flux[t] - fluxM[t]) / sigma[t]
                chi2 += diff * diff
            }
            lnL[j] = -0.5 * FPPV2Consts.ln2pi - meanLnSigma - 0.5 * chi2
        }
        return lnL
    }

    /// Host f64 preparation shared by the CPU production kernel and Metal.
    /// Keep the operation order of `_batch_eb_transit_lnL_prod`, including the
    /// secondary veto before computing the primary's two-stage affine.
    struct EBProdPreparation {
        let k: Double
        let dilA: Double
        let dilB: Double
        let secdepth: Double
        let rejected: Bool
    }

    static func prepareEBTransitProdDraw(
        sigmaVeto: Double, kRaw: Double, ebFluxRatio: Double,
        period: Double, incRad: Double, aRs: Double, ecc: Double, wRad: Double,
        u1: Double, u2: Double, companionFluxRatio: Double, companionIsHost: Bool,
        rejectSec: Bool, secConsistent: Bool, kNudgeUpstream: Bool,
        u1Sec: Double, u2Sec: Double
    ) -> EBProdPreparation {
        let fTarget = 1.0
        var k = kRaw
        if kNudgeUpstream && abs(k - 1.0) < 1e-6 {
            k = k * 0.999
        }
        var secFlux = ebSecondaryMinFlux(
            kRaw: kRaw, p: period, aRs: aRs, cosI: FPPV2.cos(incRad),
            ecc: ecc, wRad: wRad, u1: u1, u2: u2,
            u1Sec: u1Sec, u2Sec: u2Sec,
            secConsistent: secConsistent, kNudgeUpstream: kNudgeUpstream
        )
        let frEB = ebFluxRatio
        let fEB = frEB < 1.0 ? (frEB / (1.0 - frEB)) : 1e6
        let fc = companionFluxRatio
        let fComp = fc > 0.0 ? (fc / (1.0 - fc)) : 0.0

        let x: Double
        let fDilute: Double
        if companionIsHost {
            let fcClamped = max(fComp, 1e-30)
            x = fEB / fcClamped
            secFlux = (secFlux + fcClamped / fEB) / (1.0 + fcClamped / fEB)
            fDilute = fTarget / (fcClamped + fEB)
        } else {
            x = fEB / fTarget
            secFlux = (secFlux + fTarget / fEB) / (1.0 + fTarget / fEB)
            fDilute = fComp / (fTarget + fEB)
        }

        let secdepth = 1.0 - (secFlux + fDilute) / (1.0 + fDilute)
        if rejectSec && secdepth >= 1.5 * sigmaVeto {
            return EBProdPreparation(k: k, dilA: 1.0, dilB: 0.0, secdepth: secdepth, rejected: true)
        }

        let a1 = 1.0 / (1.0 + x)
        let b1 = x / (1.0 + x)
        let a2 = 1.0 / (1.0 + fDilute)
        let b2 = fDilute / (1.0 + fDilute)
        let dilA = a2 * a1
        let dilB = a2 * b1 + b2
        return EBProdPreparation(k: k, dilA: dilA, dilB: dilB, secdepth: secdepth, rejected: false)
    }

    /// `_batch_eb_transit_lnL_prod` with per-draw `(u1, u2)` arrays.
    public static func batchEBTransitLnLProd(
        fold: FPPV2ProdFold,
        sigmaVeto: Double,
        ks: [Double],
        ebFluxRatios: [Double],
        pOrbs: [Double],
        incsRad: [Double],
        aRsArr: [Double],
        eccs: [Double],
        wsRad: [Double],
        u1: [Double],
        u2: [Double],
        companionFluxRatio: [Double],
        companionIsHost: Bool,
        rejectSec: Bool,
        secConsistent: Bool,
        kNudgeUpstream: Bool,
        u1Sec: [Double],
        u2Sec: [Double]
    ) -> (lnL: [Double], status: [Int], nAff: [Int]) {
        if let gpu = FPPV3LikelihoodRouter.backend {
            return FPPV3LikelihoodRouter.routeEB(gpu, fold: fold, sigmaVeto: sigmaVeto, ks: ks, ebFluxRatios: ebFluxRatios,
                pOrbs: pOrbs, incsRad: incsRad, aRsArr: aRsArr, eccs: eccs, wsRad: wsRad, u1: u1, u2: u2,
                companionFluxRatio: companionFluxRatio, companionIsHost: companionIsHost, rejectSec: rejectSec,
                secConsistent: secConsistent, kNudgeUpstream: kNudgeUpstream, u1Sec: u1Sec, u2Sec: u2Sec)
        }
        let n = ks.count
        var lnL = [Double](repeating: -.infinity, count: n)
        var status = [Int](repeating: 0, count: n)
        var nAffOut = [Int](repeating: 0, count: n)
        let f = fold.f
        let sig = fold.arrays.sigma
        var gateDiscrete: FPPV3Gate.DiscreteRows? = FPPV3Gate.active ? .init() : nil

        for j in 0..<n {
            let u1j = u1.count == 1 ? u1[0] : u1[j]
            let u2j = u2.count == 1 ? u2[0] : u2[j]
            let prep = prepareEBTransitProdDraw(
                sigmaVeto: sigmaVeto, kRaw: ks[j], ebFluxRatio: ebFluxRatios[j],
                period: pOrbs[j], incRad: incsRad[j], aRs: aRsArr[j], ecc: eccs[j], wRad: wsRad[j],
                u1: u1j, u2: u2j, companionFluxRatio: companionFluxRatio[j], companionIsHost: companionIsHost,
                rejectSec: rejectSec, secConsistent: secConsistent, kNudgeUpstream: kNudgeUpstream,
                u1Sec: u1Sec[j], u2Sec: u2Sec[j]
            )
            if prep.rejected {
                gateDiscrete?.push(order: -1, bins: [], levels: [])
                continue
            }

            let r = FPPV2Ladder.drawBinModels(
                w: fold.w, delta: fold.delta, fold: fold.arrays,
                period: pOrbs[j], aRs: aRsArr[j], incRad: incsRad[j],
                ecc: eccs[j], wRad: wsRad[j], k: prep.k,
                u1: u1j, u2: u2j, dilA: prep.dilA, dilB: prep.dilB
            )
            var chi2 = fold.baseChi2
            for i in 0..<r.nAff {
                let b = r.idx[i]
                let d = (f[b] - r.m[i]) / sig[b]
                chi2 += d * d
                let d0 = (f[b] - 1.0) / sig[b]
                chi2 -= d0 * d0
            }
            lnL[j] = fold.lnnorm - 0.5 * chi2
            status[j] = r.status
            nAffOut[j] = r.nAff
            gateDiscrete?.pushDraw(r)
        }
        if FPPV3Gate.active {
            // the k the kernel hands to drawBinModels (Python generator: ks * 0.999 near 1 with the upstream nudge)
            let ksUsed = ks.map { kNudgeUpstream && abs($0 - 1.0) < 1e-6 ? $0 * 0.999 : $0 }
            let input = FPPV3Gate.CallInput(
                k: ks, period: pOrbs, inc: incsRad, aRs: aRsArr, ecc: eccs, wPeri: wsRad,
                u1: u1, u2: u2, companionFluxRatio: companionFluxRatio,
                ebArrays: [ebFluxRatios, u1Sec, u2Sec], sigmaVeto: sigmaVeto,
                flags: [companionIsHost, rejectSec, secConsistent, kNudgeUpstream], fold: fold
            )
            FPPV3Gate.kernelRows(periods: pOrbs, aRs: aRsArr, incsRad: incsRad, eccs: eccs, wsRad: wsRad,
                                 ksUsed: ksUsed, lnl: lnL, status: status,
                                 input: input, deviceRoots: nil, discrete: gateDiscrete)
        }
        return (lnL, status, nAffOut)
    }

    // MARK: - S-family marginal likelihoods (_lnZ_STP, _lnZ_SEB)

    private static func sigmaArray(_ sigma: [Double], count n: Int) throws -> [Double] {
        if sigma.count == 1 {
            return [Double](repeating: sigma[0], count: n)
        }
        if sigma.count != n {
            throw FPPV2SFamilyError("sigma array has \(sigma.count) entries, expected \(n)")
        }
        return sigma
    }

    /// `_lnZ_STP` returning full per-draw details in addition to `lnZ`.
    public static func lnZSTPWithDetails(
        time: [Double],
        flux: [Double],
        sigma: [Double],
        period pOrb: Double,
        massSolar mS: Double,
        radiusSolar rS: Double,
        teff: Double,
        count n: Int,
        parallax plx: Double,
        rng: inout FPPRNG,
        switches: FPPV2Switches = .production,
        fold: FPPV2ProdFold? = nil,
        sigmaVeto: Double? = nil,
        exposureStats: inout FPPV2ExposureStats
    ) throws -> FPPV2STPResult {
        let sigArr: [Double] = fold != nil ? [] : try sigmaArray(sigma, count: time.count)
        let blk = try boundCompanionBlock(
            count: n, massSolar: mS, radiusSolar: rS, teff: teff,
            rng: &rng, teffMax: 10000.0, qBranchUpstream: switches.qBranchUpstream
        )
        let mc = blk.mc
        let rc = blk.rc
        let frc = blk.frc
        let u1c = blk.u1c
        let u2c = blk.u2c
        let lnprior = lnpriorCompanion(massSolar: mS, parallax: plx, fluxRatiosCompanion: frc, scenario: "TP")

        let rps = try sampleRp(count: n, hostMasses: mc, rng: &rng)
        let incs = sampleInc(count: n, rng: &rng)
        let eccs = sampleEcc(count: n, planet: true, period: pOrb, rng: &rng)
        let argps = sampleW(count: n, rng: &rng)

        let pArr = [Double](repeating: pOrb, count: n)
        let fourPi2 = 4.0 * FPPV2.pow(FPPV2Consts.pi, 2.0)
        var aCm = [Double](repeating: 0.0, count: n)
        var aRs = [Double](repeating: 0.0, count: n)
        var ks = [Double](repeating: 0.0, count: n)
        var incsRad = [Double](repeating: 0.0, count: n)
        var wsRad = [Double](repeating: 0.0, count: n)
        var mask = [Bool](repeating: false, count: n)

        for i in 0..<n {
            let pSec = pArr[i] * 86400.0
            let t1 = ((FPPV2Consts.gCGS * mc[i]) * FPPV2Consts.msun) / fourPi2
            let ac = FPPV2.pow(t1 * (pSec * pSec), 1.0 / 3.0)
            aCm[i] = ac
            let rcCm = rc[i] * FPPV2Consts.rsun
            aRs[i] = ac / rcCm
            let rpCm = rps[i] * FPPV2Consts.rearth
            ks[i] = rpCm / rcCm
            let ir = (incs[i] * FPPV2Consts.pi) / 180.0
            incsRad[i] = ir
            wsRad[i] = ((90.0 - argps[i]) * FPPV2Consts.pi) / 180.0

            let eCorr = (1.0 + eccs[i] * FPPV2.sin((argps[i] * FPPV2Consts.pi) / 180.0)) / (1.0 - eccs[i] * eccs[i])
            let ptra = ((rpCm + rcCm) / ac) * eCorr
            let incMin = ptra <= 1.0 ? FPPV2.acos(ptra) : (FPPV2Consts.pi / 2.0)
            let coll = (rpCm + rcCm) > ac * (1.0 - eccs[i])
            mask[i] = (ir >= incMin) && (!coll)
        }

        if switches.orbitPhysicsRoche {
            let rOk = rocheOk(aCm: aCm, ecc: eccs, hostMass: mc, hostRadius: rc, rhoComp: [5.51])
            for i in 0..<n { mask[i] = mask[i] && rOk[i] }
        }

        var im: [Int] = []
        for i in 0..<n where mask[i] { im.append(i) }
        var lnL = [Double](repeating: -.infinity, count: n)
        var statusFull = [Int](repeating: 0, count: n)
        if im.isEmpty {
            return FPPV2STPResult(lnZ: -.infinity, lnL: lnL, lnprior: lnprior, mask: mask, status: statusFull)
        }

        let ksM = im.map { ks[$0] }
        let pM = im.map { pArr[$0] }
        let incM = im.map { incsRad[$0] }
        let aRsM = im.map { aRs[$0] }
        let eccM = im.map { eccs[$0] }
        let wM = im.map { wsRad[$0] }
        let u1M = im.map { u1c[$0] }
        let u2M = im.map { u2c[$0] }
        let frcM = im.map { frc[$0] }

        if let prodFold = fold {
            let (lM, stM, _) = batchTransitLnLProd(
                fold: prodFold, ks: ksM, pOrbs: pM, incsRad: incM, aRsArr: aRsM,
                eccs: eccM, wsRad: wM, u1: u1M, u2: u2M,
                companionFluxRatio: frcM, companionIsHost: true
            )
            for (k, idx) in im.enumerated() {
                lnL[idx] = lM[k]
                statusFull[idx] = stM[k]
            }
            exposureStats.note(lnL: lM, status: stM)
            FPPV3Gate.note("main", im)
        } else {
            let (exptime, nsub) = switches.exposureArgs
            let lM = batchTransitLnL(
                time: time, flux: flux, sigma: sigArr,
                ks: ksM, pOrbs: pM, incsRad: incM, aRsArr: aRsM,
                eccs: eccM, wsRad: wM, u1: u1M, u2: u2M,
                companionFluxRatio: frcM, companionIsHost: true,
                exptime: exptime, nsub: nsub
            )
            for (k, idx) in im.enumerated() {
                lnL[idx] = lM[k]
            }
        }

        let logw = (0..<n).map { lnL[$0] + lnprior[$0] }
        FPPV3Gate.stage("main", lnL)
        let lnZ = lnZFromLogw(logw)
        return FPPV2STPResult(lnZ: lnZ, lnL: lnL, lnprior: lnprior, mask: mask, status: statusFull)
    }

    /// `_lnZ_STP` convenience wrapper returning scalar `lnZ`.
    public static func lnZSTP(
        time: [Double],
        flux: [Double],
        sigma: [Double],
        period pOrb: Double,
        massSolar mS: Double,
        radiusSolar rS: Double,
        teff: Double,
        count n: Int,
        parallax plx: Double,
        rng: inout FPPRNG,
        switches: FPPV2Switches = .production,
        fold: FPPV2ProdFold? = nil,
        sigmaVeto: Double? = nil
    ) throws -> Double {
        var stats = FPPV2ExposureStats()
        return try lnZSTPWithDetails(
            time: time, flux: flux, sigma: sigma, period: pOrb,
            massSolar: mS, radiusSolar: rS, teff: teff, count: n,
            parallax: plx, rng: &rng, switches: switches, fold: fold,
            sigmaVeto: sigmaVeto, exposureStats: &stats
        ).lnZ
    }

    /// `_lnZ_SEB` returning full per-draw details in addition to `(lnZ_SEB, lnZ_SEBx2P)`.
    public static func lnZSEBWithDetails(
        time: [Double],
        flux: [Double],
        sigma: [Double],
        period pOrb: Double,
        massSolar mS: Double,
        radiusSolar rS: Double,
        teff: Double,
        count n: Int,
        parallax plx: Double,
        rng: inout FPPRNG,
        switches: FPPV2Switches = .production,
        fold: FPPV2ProdFold? = nil,
        sigmaVeto: Double? = nil,
        exposureStats: inout FPPV2ExposureStats
    ) throws -> FPPV2SEBResult {
        let sigArr: [Double] = fold != nil ? [] : try sigmaArray(sigma, count: time.count)
        let veto: Double
        if let sv = sigmaVeto {
            veto = sv
        } else if fold != nil {
            throw FPPV2SFamilyError("sigma_veto is required when a production fold is given")
        } else if sigma.count == 1 {
            veto = sigma[0]
        } else {
            throw FPPV2SFamilyError("sigma_veto is required when sigma is a per-bin array")
        }

        let incs = sampleInc(count: n, rng: &rng)
        let qs = sampleQ(count: n, hostMass: mS, rng: &rng, qBranchUpstream: switches.qBranchUpstream)
        let eccs = sampleEcc(count: n, planet: false, period: pOrb, rng: &rng)
        let argps = sampleW(count: n, rng: &rng)
        let blk = try boundCompanionBlock(
            count: n, massSolar: mS, radiusSolar: rS, teff: teff,
            rng: &rng, teffMax: 13000.0, qBranchUpstream: switches.qBranchUpstream
        )
        let mc = blk.mc
        let rc = blk.rc
        let tc = blk.tc
        let frc = blk.frc
        let u1c = blk.u1c
        let u2c = blk.u2c

        let masses = (0..<n).map { qs[$0] * mc[$0] }
        let (reb, teb) = stellarRelationsCapped(masses: masses, maxR: rc, maxT: tc)
        let fMasses = fluxRelation(masses: masses)
        let fMs = fluxRelationScalar(mS)
        let freb = fMasses.map { $0 / ($0 + fMs) }
        let dm = (0..<n).map { i in
            2.5 * FPPV2.log10((frc[i] / (1.0 - frc[i])) + (freb[i] / (1.0 - freb[i])))
        }
        var lnprior = lnpriorBoundEB(massSolar: mS, parallax: plx, deltaMagsAbs: dm.map { abs($0) })
        for i in 0..<n {
            if lnprior[i] > 0.0 { lnprior[i] = 0.0 }
            if dm[i] > 0.0 { lnprior[i] = -.infinity }
        }

        let pArr = [Double](repeating: pOrb, count: n)
        let pTwArr = pArr.map { 2.0 * $0 }
        let fourPi2 = 4.0 * FPPV2.pow(FPPV2Consts.pi, 2.0)
        var aCm = [Double](repeating: 0.0, count: n)
        var aTw = [Double](repeating: 0.0, count: n)
        var incsRad = [Double](repeating: 0.0, count: n)
        var wsRad = [Double](repeating: 0.0, count: n)
        var maskEB = [Bool](repeating: false, count: n)
        var maskTw = [Bool](repeating: false, count: n)
        var ks = [Double](repeating: 0.0, count: n)
        var aRs = [Double](repeating: 0.0, count: n)
        var aTwRs = [Double](repeating: 0.0, count: n)

        for i in 0..<n {
            let eCorr = (1.0 + eccs[i] * FPPV2.sin((argps[i] * FPPV2Consts.pi) / 180.0)) / (1.0 - eccs[i] * eccs[i])
            let mTot = ((FPPV2Consts.gCGS * (mc[i] + masses[i])) * FPPV2Consts.msun) / fourPi2
            let pSec = pArr[i] * 86400.0
            let pTwSec = (2.0 * pArr[i]) * 86400.0
            let ac = FPPV2.pow(mTot * (pSec * pSec), 1.0 / 3.0)
            let atw = FPPV2.pow(mTot * (pTwSec * pTwSec), 1.0 / 3.0)
            aCm[i] = ac
            aTw[i] = atw

            let rebCm = reb[i] * FPPV2Consts.rsun
            let rcCm = rc[i] * FPPV2Consts.rsun
            let ptra = ((rebCm + rcCm) / ac) * eCorr
            let ptraTw = ((rebCm + rcCm) / atw) * eCorr
            let incMin = ptra <= 1.0 ? FPPV2.acos(ptra) : (FPPV2Consts.pi / 2.0)
            let incMinTw = ptraTw <= 1.0 ? FPPV2.acos(ptraTw) : (FPPV2Consts.pi / 2.0)
            let coll = (rebCm + rcCm) > ac * (1.0 - eccs[i])
            let collTw = ((2.0 * rc[i]) * FPPV2Consts.rsun) > atw * (1.0 - eccs[i])
            let ir = (incs[i] * FPPV2Consts.pi) / 180.0
            incsRad[i] = ir
            wsRad[i] = ((90.0 - argps[i]) * FPPV2Consts.pi) / 180.0
            maskEB[i] = (ir >= incMin) && (!coll) && (qs[i] < 0.95)
            maskTw[i] = (ir >= incMinTw) && (!collTw) && (qs[i] >= 0.95)
            ks[i] = reb[i] / rc[i]
            aRs[i] = ac / rcCm
            aTwRs[i] = atw / rcCm
        }

        if switches.orbitPhysicsRoche {
            let fourThirdsPi = (4.0 / 3.0) * FPPV2Consts.pi
            let rhoEB: [Double] = (0..<n).map { i in
                (masses[i] * FPPV2Consts.msun) / (fourThirdsPi * FPPV2.pow(reb[i] * FPPV2Consts.rsun, 3.0))
            }
            let rOkEB = rocheOk(aCm: aCm, ecc: eccs, hostMass: mc, hostRadius: rc, rhoComp: rhoEB)
            let rOkTw = rocheOk(aCm: aTw, ecc: eccs, hostMass: mc, hostRadius: rc, rhoComp: rhoEB)
            for i in 0..<n {
                maskEB[i] = maskEB[i] && rOkEB[i]
                maskTw[i] = maskTw[i] && rOkTw[i]
            }
        }

        let secConsistent = switches.secondaryGeometryConsistent
        let kNudgeUpstream = switches.k1NudgeUpstream
        var u1Sec = u1c
        var u2Sec = u2c
        if secConsistent {
            var sel: [Int] = []
            for i in 0..<n where maskEB[i] || maskTw[i] { sel.append(i) }
            if !sel.isEmpty {
                let tebSel = sel.map { teb[$0] }
                let loggEB = sel.map { i -> Double in
                    let rCm = reb[i] * FPPV2Consts.rsun
                    return FPPV2.log10(((FPPV2Consts.gCGS * masses[i]) * FPPV2Consts.msun) / (rCm * rCm))
                }
                let (u1S, u2S) = try getLDCArrays(teff: tebSel, logg: loggEB)
                for (k, idx) in sel.enumerated() {
                    u1Sec[idx] = u1S[k]
                    u2Sec[idx] = u2S[k]
                }
            }
        }

        func runBranch(mask: [Bool], pV: [Double], aRsV: [Double], withSec: Bool) -> ([Double], [Int]) {
            var lnL = [Double](repeating: -.infinity, count: n)
            var statusFull = [Int](repeating: 0, count: n)
            var im: [Int] = []
            for i in 0..<n where mask[i] { im.append(i) }
            if im.isEmpty { return (lnL, statusFull) }

            let ksM = im.map { ks[$0] }
            let frebM = im.map { freb[$0] }
            let pM = im.map { pV[$0] }
            let incM = im.map { incsRad[$0] }
            let aRsM = im.map { aRsV[$0] }
            let eccM = im.map { eccs[$0] }
            let wM = im.map { wsRad[$0] }
            let u1M = im.map { u1c[$0] }
            let u2M = im.map { u2c[$0] }
            let frcM = im.map { frc[$0] }
            let u1SecM = im.map { u1Sec[$0] }
            let u2SecM = im.map { u2Sec[$0] }

            if let prodFold = fold {
                let (lM, stM, _) = batchEBTransitLnLProd(
                    fold: prodFold, sigmaVeto: veto, ks: ksM, ebFluxRatios: frebM,
                    pOrbs: pM, incsRad: incM, aRsArr: aRsM, eccs: eccM, wsRad: wM,
                    u1: u1M, u2: u2M, companionFluxRatio: frcM, companionIsHost: true,
                    rejectSec: withSec, secConsistent: secConsistent,
                    kNudgeUpstream: kNudgeUpstream, u1Sec: u1SecM, u2Sec: u2SecM
                )
                for (k, idx) in im.enumerated() {
                    lnL[idx] = lM[k]
                    statusFull[idx] = stM[k]
                }
                exposureStats.note(lnL: lM, status: stM)
                FPPV3Gate.note(withSec ? "eb" : "tw", im)
            } else {
                let (exptime, nsub) = switches.exposureArgs
                let lM = batchEBTransitLnL(
                    time: time, flux: flux, sigma: sigArr, sigmaVeto: veto,
                    ks: ksM, ebFluxRatios: frebM, pOrbs: pM, incsRad: incM,
                    aRsArr: aRsM, eccs: eccM, wsRad: wM, u1: u1M, u2: u2M,
                    companionFluxRatio: frcM, companionIsHost: true,
                    exptime: exptime, nsub: nsub, rejectSec: withSec,
                    secConsistent: secConsistent, kNudgeUpstream: kNudgeUpstream,
                    u1Sec: u1SecM, u2Sec: u2SecM
                )
                for (k, idx) in im.enumerated() {
                    lnL[idx] = lM[k]
                }
            }
            return (lnL, statusFull)
        }

        let (lnLEB, stEB) = runBranch(mask: maskEB, pV: pArr, aRsV: aRs, withSec: true)
        let (lnLTw, stTw) = runBranch(mask: maskTw, pV: pTwArr, aRsV: aTwRs, withSec: false)
        FPPV3Gate.stage("eb", lnLEB)
        let lnZEB = lnZFromLogw((0..<n).map { lnLEB[$0] + lnprior[$0] })
        FPPV3Gate.stage("tw", lnLTw)
        let lnZTw = lnZFromLogw((0..<n).map { lnLTw[$0] + lnprior[$0] })
        return FPPV2SEBResult(
            lnZSEB: lnZEB, lnZSEBx2P: lnZTw,
            lnLSEB: lnLEB, lnLSEBx2P: lnLTw, lnprior: lnprior,
            maskEB: maskEB, maskTwin: maskTw, statusEB: stEB, statusTwin: stTw
        )
    }

    /// `_lnZ_SEB` convenience wrapper returning `(lnZ_SEB, lnZ_SEBx2P)`.
    public static func lnZSEB(
        time: [Double],
        flux: [Double],
        sigma: [Double],
        period pOrb: Double,
        massSolar mS: Double,
        radiusSolar rS: Double,
        teff: Double,
        count n: Int,
        parallax plx: Double,
        rng: inout FPPRNG,
        switches: FPPV2Switches = .production,
        fold: FPPV2ProdFold? = nil,
        sigmaVeto: Double? = nil
    ) throws -> (lnZSEB: Double, lnZSEBx2P: Double) {
        var stats = FPPV2ExposureStats()
        let res = try lnZSEBWithDetails(
            time: time, flux: flux, sigma: sigma, period: pOrb,
            massSolar: mS, radiusSolar: rS, teff: teff, count: n,
            parallax: plx, rng: &rng, switches: switches, fold: fold,
            sigmaVeto: sigmaVeto, exposureStats: &stats
        )
        return (res.lnZSEB, res.lnZSEBx2P)
    }
}
