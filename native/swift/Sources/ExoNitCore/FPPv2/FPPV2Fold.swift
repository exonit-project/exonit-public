import Foundation

/// The production fold in TIME units (days from mid-transit), the arrays `prod_fold_arrays` returns.
public struct FPPV2FoldArrays: Sendable {
    public var n: [Int] = []
    public var flux: [Double] = []
    public var sigma: [Double] = []
    public var tmean: [Double] = []
    public var varT: [Double] = []
    public var tmin: [Double] = []
    public var tmax: [Double] = []
    /// `node_tab[b, l, j]` flattened: `(b * nodeLevels + l) * ladderMaxK + j`.
    public var nodeTab: [Double] = []
    public var ptsOff: [Int] = []
    public var ptsT: [Double] = []

    public init() {}

    @inline(__always)
    public func node(_ b: Int, _ l: Int, _ j: Int) -> Double {
        nodeTab[(b * FPPV2Fold.nodeLevels + l) * FPPV2Consts.ladderMaxK + j]
    }
}

/// One non-empty bin of `fold_production` (`bin`, `n`, sums, `quantiles` of the bin's phases and,
/// when asked, the bin's phases sorted ascending).
public struct FPPV2FoldRow: Sendable {
    public var bin = 0
    public var n = 0
    public var fluxMean = 0.0
    public var fluxM2 = 0.0
    public var phaseMean = 0.0
    public var phaseM2 = 0.0
    public var varB = 0.0
    public var quantiles: [Double] = []
    public var phases: [Double] = []
    public init() {}
}

/// The two fold-wide constants of the bin likelihood (`build_prod_fold` / `prod_fold_renorm`).
public struct FPPV2ProdFoldConsts: Sendable {
    public var w: Double
    public var delta: Double
    public var baseChi2: Double
    public var lnnorm: Double
}

public struct FPPV2FoldError: Error, CustomStringConvertible {
    public let description: String
}

/// Production fold of plan 0.3 (j'): mean bins + each bin's phase distribution (N5), the type-7
/// quantile spec, the time-unit fold arrays of the ladder, and the two fold-wide constants of the
/// bin likelihood. Literal transcription of `_fold_quantiles_nb`, `fold_quantiles`,
/// `_fold_phase_nb`, `_fold_bins_nb`, `fold_production`, `prod_fold_arrays`, `build_prod_fold`,
/// `prod_fold_renorm` of worker/fpp_native.py. Sequential f64 sums in input order (NOT pairwise:
/// the reference loops are sequential).
public enum FPPV2Fold {
    /// Number of quantile-node levels stored per bin (`ladderExactLevel - 1`, K = 4 * 2^l for l = 0..5).
    public static let nodeLevels = FPPV2Consts.ladderExactLevel - 1

    /// `np.linspace(-0.5, 0.5, n_bins + 1)`: `y_i = i * step + start`, last element exactly `stop`.
    public static func foldEdges(_ nBins: Int) -> [Double] {
        let step = 1.0 / Double(nBins)
        var e = (0...nBins).map { Double($0) * step + -0.5 }
        e[nBins] = 0.5
        return e
    }

    /// `np.digitize(x, edges)` for increasing edges == `searchsorted(edges, x, side='right')`.
    static func digitize(_ edges: [Double], _ x: Double) -> Int {
        var lo = 0, hi = edges.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if x < edges[mid] { hi = mid } else { lo = mid + 1 }
        }
        return lo
    }

    static func sortedAscending(_ a: [Double]) -> [Double] { a.sorted(by: <) }

    /// `_fold_quantiles_nb`: type-7 quantiles of an ascending-sorted array, `p = (i + 0.5) / K`.
    public static func foldQuantilesSorted(_ s: [Double], _ kk: Int) -> [Double] {
        let n = s.count
        let nm1 = n - 1
        var out = [Double](repeating: 0.0, count: kk)
        for i in 0..<kk {
            let p = (Double(i) + 0.5) / Double(kk)
            let h = Double(nm1) * p
            let j = Int(h.rounded(.down))
            let g = h - Double(j)
            out[i] = j + 1 >= n ? s[j] : s[j] + g * (s[j + 1] - s[j])
        }
        return out
    }

    /// `fold_quantiles(x, K)`: ascending sort, then the spec; NaNs for an empty sample.
    public static func foldQuantiles(_ x: [Double], _ kk: Int) -> [Double] {
        if kk == 0 { return [] }
        let s = sortedAscending(x)
        if s.isEmpty { return [Double](repeating: .nan, count: kk) }
        return foldQuantilesSorted(s, kk)
    }

    /// `fold_production(time, flux, t0, period, n_bins, k_quant, keep_phases)`.
    public static func foldProduction(
        time: [Double], flux: [Double], t0: Double, period: Double,
        nBins: Int = FPPV2Consts.foldNBins, kQuant: Int = FPPV2Consts.foldKQuant,
        keepPhases: Bool = true
    ) -> [FPPV2FoldRow] {
        // non-finite time / flux dropped, input order kept
        var tk: [Double] = [], fk: [Double] = []
        for (t, f) in zip(time, flux) where t.isFinite && f.isFinite {
            tk.append(t)
            fk.append(f)
        }
        if tk.isEmpty { return [] }
        // _fold_phase_nb: phase = ((t - t0) / period) % 1.0, then phase -= 1.0 when phase > 0.5
        let ph: [Double] = tk.map { t in
            var p = FPPV2.pyMod((t - t0) / period, 1.0)
            if p > 0.5 { p -= 1.0 }
            return p
        }
        let edges = foldEdges(nBins)
        let bIdx: [Int] = ph.map { min(max(digitize(edges, $0) - 1, 0), nBins - 1) }
        var counts = [Int](repeating: 0, count: nBins)
        for b in bIdx { counts[b] += 1 }
        // stable argsort by bin: input order kept inside every bin
        let order = (0..<ph.count).sorted { (bIdx[$0], $0) < (bIdx[$1], $1) }
        let phS = order.map { ph[$0] }
        let fsS = order.map { fk[$0] }
        var starts = [Int](repeating: 0, count: nBins + 1)
        for b in 0..<nBins { starts[b + 1] = starts[b] + counts[b] }

        var rows: [FPPV2FoldRow] = []
        for b in 0..<nBins {
            let n = counts[b]
            if n == 0 { continue }
            let st = starts[b]
            let nf = Double(n)
            var s = 0.0
            for i in 0..<n { s += fsS[st + i] }
            let fmean = s / nf
            var m2 = 0.0
            for i in 0..<n {
                let d = fsS[st + i] - fmean
                m2 += d * d
            }
            s = 0.0
            for i in 0..<n { s += phS[st + i] }
            let pmean = s / nf
            var pm2 = 0.0
            for i in 0..<n {
                let d = phS[st + i] - pmean
                pm2 += d * d
            }
            let scratch = sortedAscending(Array(phS[st..<(st + n)]))
            var row = FPPV2FoldRow()
            row.bin = b
            row.n = n
            row.fluxMean = fmean
            row.fluxM2 = m2
            row.phaseMean = pmean
            row.phaseM2 = pm2
            row.varB = pm2 / nf
            row.quantiles = foldQuantilesSorted(scratch, kQuant)
            row.phases = keepPhases ? scratch : []
            rows.append(row)
        }
        return rows
    }

    /// `prod_fold_arrays(rows, period, sigma_b)`: the fold in TIME units for the ladder. `sigmaB`
    /// holds one value per row, or one value that is repeated.
    public static func prodFoldArrays(
        rows: [FPPV2FoldRow], period: Double, sigmaB: [Double]
    ) throws -> FPPV2FoldArrays {
        let nb = rows.count
        let sig: [Double] = (sigmaB.count == 1 && nb > 1) ? [Double](repeating: sigmaB[0], count: nb) : sigmaB
        if sig.count != nb { throw FPPV2FoldError(description: "sigma_b must hold one value per fold row") }
        var fa = FPPV2FoldArrays()
        fa.n = rows.map { $0.n }
        fa.flux = rows.map { $0.fluxMean }
        fa.sigma = sig
        fa.tmean = rows.map { $0.phaseMean * period }
        fa.varT = rows.map { $0.varB * period * period }
        fa.nodeTab = [Double](repeating: 0.0, count: nb * nodeLevels * FPPV2Consts.ladderMaxK)
        fa.ptsOff = [Int](repeating: 0, count: nb + 1)
        for (i, r) in rows.enumerated() {
            let ph = sortedAscending(r.phases)
            if ph.isEmpty { throw FPPV2FoldError(description: "fold row without phases (use keepPhases)") }
            fa.tmin.append(ph[0] * period)
            fa.tmax.append(ph[ph.count - 1] * period)
            fa.ptsOff[i + 1] = fa.ptsOff[i] + ph.count
            for p in ph { fa.ptsT.append(p * period) }
            // N5: a bin with more points than ladderKQuant is reduced to its 256-point quantile table
            let src = r.n > FPPV2Consts.ladderKQuant ? r.quantiles : ph
            for l in 0..<nodeLevels {
                let kk = 4 << l
                let q = foldQuantiles(src, kk)
                for j in 0..<kk {
                    fa.nodeTab[(i * nodeLevels + l) * FPPV2Consts.ladderMaxK + j] = period * q[j]
                }
            }
        }
        return fa
    }

    /// `base_chi2 = sum(((f - 1) / sigma)^2)` accumulated sequentially,
    /// `lnnorm = -0.5 ln(2 pi) - mean(ln sigma)`.
    static func chi2AndLnnorm(_ f: [Double], _ sig: [Double]) -> (Double, Double) {
        let nb = f.count
        var baseChi2 = 0.0
        var meanLnSigma = 0.0
        for i in 0..<nb {
            let d = (f[i] - 1.0) / sig[i]
            baseChi2 += d * d
            meanLnSigma += FPPV2.log(sig[i])
        }
        return (baseChi2, -0.5 * FPPV2Consts.ln2pi - meanLnSigma / Double(nb))
    }

    /// `build_prod_fold` constants: `W = cadence_s / 86400`, `delta = ladderDeltaNat`, `base_chi2`, `lnnorm`.
    public static func buildProdFoldConsts(
        fold: FPPV2FoldArrays, fluxB: [Double], cadenceS: Double
    ) throws -> FPPV2ProdFoldConsts {
        let nb = fold.n.count
        if nb == 0 { throw FPPV2FoldError(description: "build_prod_fold: the production fold has no rows") }
        if fluxB.count != nb {
            throw FPPV2FoldError(description: "flux_b must hold one value per fold row (\(fluxB.count) != \(nb))")
        }
        if !(cadenceS > 0.0) { throw FPPV2FoldError(description: "cadence_s must be positive") }
        let (baseChi2, lnnorm) = chi2AndLnnorm(fluxB, fold.sigma)
        return FPPV2ProdFoldConsts(
            w: cadenceS / 86400.0, delta: FPPV2Consts.ladderDeltaNat, baseChi2: baseChi2, lnnorm: lnnorm)
    }

    /// `prod_fold_renorm`: renormalise the bin fluxes / sigmas to a star of flux fraction `fr`
    /// (`f' = (f - (1 - fr)) / fr`, `sigma' = sigma / fr`) -> `(f', sigma', base_chi2, lnnorm)`.
    public static func prodFoldRenorm(
        f: [Double], sigma: [Double], fr: Double
    ) throws -> (f: [Double], sigma: [Double], baseChi2: Double, lnnorm: Double) {
        if !(fr > 0.0) { throw FPPV2FoldError(description: "prod_fold_renorm: fr must be positive") }
        let f2 = f.map { ($0 - (1.0 - fr)) / fr }
        let s2 = sigma.map { $0 / fr }
        let (baseChi2, lnnorm) = chi2AndLnnorm(f2, s2)
        return (f2, s2, baseChi2, lnnorm)
    }
}
