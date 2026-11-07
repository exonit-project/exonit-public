import Foundation

/// Orbit + star parameters shared by every flux evaluation of one draw and the affine dilution
/// `dilA * F + dilB` of `_flux_at`.
public struct FPPV2DrawParams: Sendable {
    public var p: Double
    public var aRs: Double
    public var cosI: Double
    public var ecc: Double
    public var wRad: Double
    public var mT: Double
    public var k: Double
    public var u1: Double
    public var u2: Double
    public var dilA: Double
    public var dilB: Double

    public init(
        p: Double, aRs: Double, cosI: Double, ecc: Double, wRad: Double, mT: Double, k: Double,
        u1: Double, u2: Double, dilA: Double, dilB: Double
    ) {
        self.p = p; self.aRs = aRs; self.cosI = cosI; self.ecc = ecc; self.wRad = wRad
        self.mT = mT; self.k = k; self.u1 = u1; self.u2 = u2; self.dilA = dilA; self.dilB = dilB
    }
}

/// Contact-split Gauss-Legendre exposure integrator and the per-draw order choice (plan 0.3
/// (j'')). Literal transcription of `GL_X/GL_W` use, `_flux_at`, `_window_mean`, `_choose_order`
/// of worker/fpp_native.py. The node / weight tables are DATA (`FPPV2GLTables`).
public enum FPPV2GL {
    static let xs: [Double] = FPPV2GLTables.xBits.map { Double(bitPattern: $0) }
    static let ws: [Double] = FPPV2GLTables.wBits.map { Double(bitPattern: $0) }

    /// Node `i` of order index `oi` (`GL_X[oi, i]`).
    @inline(__always)
    public static func glX(_ oi: Int, _ i: Int) -> Double { xs[oi * FPPV2GLTables.maxM + i] }
    /// Weight `i` of order index `oi` (`GL_W[oi, i]`).
    @inline(__always)
    public static func glW(_ oi: Int, _ i: Int) -> Double { ws[oi * FPPV2GLTables.maxM + i] }

    /// `_flux_at(t, ...)`: model flux at one time, with the affine dilution.
    @inline(__always)
    public static func fluxAt(_ t: Double, _ d: FPPV2DrawParams) -> Double {
        let z = FPPV2Orbit.zOne(t, 0.0, d.p, d.aRs, d.cosI, d.ecc, d.wRad, d.mT)
        return d.dilA * FPPV2Kernel.transitFlux(z, d.k, d.u1, d.u2) + d.dilB
    }

    /// `_window_mean(tc, W, ct, order_idx, ...)`: mean model flux over `[tc - W/2, tc + W/2]`, the
    /// window cut at the (sorted, finite) contact times `ct` that fall strictly inside it;
    /// endpoint substitution when a piece end is a contact.
    public static func windowMean(
        _ tc: Double, _ w: Double, _ ct: [Double], _ orderIdx: Int, _ d: FPPV2DrawParams
    ) -> Double {
        let a = tc - 0.5 * w
        let b = tc + 0.5 * w
        let m = FPPV2GLTables.orders[orderIdx]
        let nct = ct.count
        var acc = 0.0
        var jc = 0
        while jc < nct && ct[jc] <= a { jc += 1 }  // contacts at or below the lower edge
        var lo = a
        var loC = false
        while true {
            let hi: Double
            let hiC: Bool
            if jc < nct && ct[jc] < b {
                hi = ct[jc]  // interior cut (a contact)
                hiC = true
                jc += 1
            } else {
                hi = b  // upper window edge
                hiC = false
            }

            let l = hi - lo
            if l > 0.0 {
                if !(loC || hiC) {
                    let halfL = 0.5 * l
                    let mid = 0.5 * (hi + lo)
                    for i in 0..<m {
                        let t = halfL * glX(orderIdx, i) + mid
                        let wt = halfL * glW(orderIdx, i)
                        acc += wt * fluxAt(t, d)
                    }
                } else if loC && hiC {
                    for i in 0..<m {
                        let u = 0.5 * (glX(orderIdx, i) + 1.0)
                        let t = lo + l * u * u * (3.0 - 2.0 * u)
                        let wt = 0.5 * glW(orderIdx, i) * 6.0 * l * u * (1.0 - u)
                        acc += wt * fluxAt(t, d)
                    }
                } else {
                    let sL = l.squareRoot()
                    for i in 0..<m {
                        let s = 0.5 * sL * (glX(orderIdx, i) + 1.0)
                        let wt = 0.5 * sL * glW(orderIdx, i) * 2.0 * s
                        let t = loC ? lo + s * s : hi - s * s
                        acc += wt * fluxAt(t, d)
                    }
                }
            }
            if !hiC { break }  // the piece reached the upper window edge
            lo = hi
            loC = true
        }
        return acc / w
    }

    /// `_choose_order(ct, W, eps, ...)` -> `(orderIdx, status)`: status 0 converged by an order
    /// <= 32, 1 the smallest-order search failed and the oracle order 128 is used, 2 even 64 vs
    /// 128 disagree (unconverged: the caller must flag it, never 'validated').
    public static func chooseOrder(
        _ ct: [Double], _ w: Double, _ eps: Double, _ d: FPPV2DrawParams
    ) -> (orderIdx: Int, status: Int) {
        if ct.isEmpty { return (FPPV2Consts.glSmoothIdx, 0) }
        var chosen = FPPV2Consts.glSmoothIdx
        for c in ct {
            var v0 = windowMean(c, w, ct, 0, d)
            var v1 = windowMean(c, w, ct, 1, d)
            var v2 = windowMean(c, w, ct, 2, d)
            var found = -1
            for i in 0..<7 {
                if abs(v0 - v1) <= eps && abs(v0 - v2) <= eps {
                    found = i
                    break
                }
                let v3 = windowMean(c, w, ct, i + 3, d)
                v0 = v1
                v1 = v2
                v2 = v3
            }
            if found < 0 {
                // v0, v1, v2 now hold I_7, I_8, I_9 (m = 24, 32, 64); the oracle I_10 (m = 128) enters the test.
                let v3 = windowMean(c, w, ct, FPPV2Consts.glOracleIdx, d)
                if abs(v0 - v1) <= eps && abs(v0 - v2) <= eps {
                    found = 7
                } else if abs(v1 - v2) <= eps && abs(v1 - v3) <= eps {
                    found = 8
                } else if abs(v2 - v3) <= eps {
                    return (FPPV2Consts.glOracleIdx, 1)  // plateau holds: oracle fallback
                } else {
                    return (FPPV2Consts.glOracleIdx, 2)  // 64 vs 128 disagree: unconverged
                }
            }
            if found > chosen { chosen = found }
        }
        return (chosen, 0)
    }
}
