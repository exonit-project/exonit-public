import Foundation

/// Corrected v3 target-host and bound-companion scenario kernels.
/// These are literal ports of `_lnZ_TTP`, `_lnZ_TEB`, `_lnZ_PTP`, and `_lnZ_PEB`.
public enum FPPV3TargetFamily {
    private static func note(_ values: [Double], _ diagnostics: inout FPPV3Diagnostics) {
        diagnostics.keplerUnconvergedDraws += values.reduce(into: 0) { n, x in
            if x.isNaN { n += 1 }
        }
    }

    private static func noteExposure(_ values: [Double], _ status: [Int], _ diagnostics: inout FPPV3Diagnostics) {
        var finite: [Int] = []
        for i in 0..<min(values.count, status.count) where values[i].isFinite { finite.append(i) }
        guard let maximum = finite.map({ values[$0] }).max() else { return }
        let weights = finite.map { FPPV2.exp(values[$0] - maximum) }
        let total = FPPNumpy.pairwiseSum(weights)
        guard total > 0.0 else { return }
        var fallback: [Double] = [], unconverged: [Double] = []
        for (j, i) in finite.enumerated() {
            if status[i] == 1 { fallback.append(weights[j]) }
            if status[i] == 2 { unconverged.append(weights[j]) }
        }
        diagnostics.exposureGLFallbackShare = max(
            diagnostics.exposureGLFallbackShare, FPPNumpy.pairwiseSum(fallback) / total)
        diagnostics.exposureUnconvergedShare = max(
            diagnostics.exposureUnconvergedShare, FPPNumpy.pairwiseSum(unconverged) / total)
    }

    private static func validate(n: Int, time: [Double], flux: [Double], fold: FPPV2ProdFold?) throws {
        guard n > 0 else { throw FPPV2SFamilyError("target-family evaluation requires N > 0") }
        if fold == nil {
            guard !time.isEmpty else { throw FPPV2SFamilyError("target-family evaluation requires a nonempty light curve") }
            guard time.count == flux.count else { throw FPPV2SFamilyError("time and flux lengths differ") }
        }
    }

    private static func broadcast(_ x: [Double], n: Int, name: String) throws -> [Double] {
        if x.count == n { return x }
        if x.count == 1 { return [Double](repeating: x[0], count: n) }
        throw FPPV2SFamilyError("\(name) must contain one value or N values (\(x.count) != \(n))")
    }

    private static func targetAxes(period: Double, mass: Double, radius: Double, count n: Int) -> (cm: [Double], rs: [Double]) {
        let pSec = period * 86400.0
        let p2 = pSec * pSec
        let denom = 4.0 * FPPV2.pow(FPPV2Consts.pi, 2.0)
        var cm: [Double] = [], rs: [Double] = []
        cm.reserveCapacity(n); rs.reserveCapacity(n)
        for _ in 0..<n {
            let a = FPPV2.pow(FPPV2Consts.gCGS * mass * FPPV2Consts.msun / denom * p2, 1.0 / 3.0)
            cm.append(a)
            rs.append(a / (radius * FPPV2Consts.rsun))
        }
        return (cm, rs)
    }

    /// Target-host planet, including nearby NTP through per-draw dilution and companion priors.
    public static func lnZTTP(
        time: [Double], flux: [Double], sigma: Double, period: Double,
        mass: Double, radius: Double, n: Int, u1: Double, u2: Double,
        companionFluxRatio: [Double], companionIsHost: Bool = false,
        lnpriorCompanion: [Double], rng: inout FPPRNG,
        switches: FPPV2Switches = .production,
        diagnostics: inout FPPV3Diagnostics,
        fold: FPPV2ProdFold? = nil
    ) throws -> Double {
        try validate(n: n, time: time, flux: flux, fold: fold)
        let comp = try broadcast(companionFluxRatio, n: n, name: "companion_flux_ratio")
        let prior = try broadcast(lnpriorCompanion, n: n, name: "lnprior_companion")
        let axes = targetAxes(period: period, mass: mass, radius: radius, count: n)
        let rp = FPPV2SFamily.sampleRp(count: n, hostMass: mass, rng: &rng)
        let inc = FPPV2SFamily.sampleInc(count: n, rng: &rng)
        let ecc = FPPV2SFamily.sampleEcc(count: n, planet: true, period: period, rng: &rng)
        let arg = FPPV2SFamily.sampleW(count: n, rng: &rng)
        var k = [Double](repeating: 0.0, count: n), ir = k, wr = k
        var mask = [Bool](repeating: false, count: n)
        for i in 0..<n {
            k[i] = rp[i] * FPPV2Consts.rearth / (radius * FPPV2Consts.rsun)
            ir[i] = inc[i] * FPPV2Consts.pi / 180.0
            wr[i] = (90.0 - arg[i]) * FPPV2Consts.pi / 180.0
            let ec = (1.0 + ecc[i] * FPPV2.sin(arg[i] * FPPV2Consts.pi / 180.0)) / (1.0 - ecc[i] * ecc[i])
            let ptra = (rp[i] * FPPV2Consts.rearth + radius * FPPV2Consts.rsun) / axes.cm[i] * ec
            let minInc = ptra <= 1.0 ? FPPV2.acos(ptra) : FPPV2Consts.pi / 2.0
            let collision = (rp[i] * FPPV2Consts.rearth + radius * FPPV2Consts.rsun) > axes.cm[i] * (1.0 - ecc[i])
            mask[i] = ir[i] >= minInc && !collision
        }
        if switches.orbitPhysicsRoche {
            let roche = FPPV2SFamily.rocheOk(
                aCm: axes.cm, ecc: ecc,
                hostMass: [Double](repeating: mass, count: n),
                hostRadius: [Double](repeating: radius, count: n),
                rhoComp: [Double](repeating: 5.51, count: n))
            for i in 0..<n { mask[i] = mask[i] && roche[i] }
        }
        var lnl = [Double](repeating: -.infinity, count: n)
        let selected = (0..<n).filter { mask[$0] }
        if !selected.isEmpty {
            let ks = selected.map { k[$0] }, periods = [Double](repeating: period, count: selected.count)
            let incs = selected.map { ir[$0] }, ar = selected.map { axes.rs[$0] }
            let es = selected.map { ecc[$0] }, ws = selected.map { wr[$0] }, fr = selected.map { comp[$0] }
            let us1 = [Double](repeating: u1, count: selected.count), us2 = [Double](repeating: u2, count: selected.count)
            let values: [Double]
            if let fold {
                let result = FPPV2SFamily.batchTransitLnLProd(
                    fold: fold, ks: ks, pOrbs: periods, incsRad: incs, aRsArr: ar,
                    eccs: es, wsRad: ws, u1: us1, u2: us2,
                    companionFluxRatio: fr, companionIsHost: companionIsHost)
                values = result.lnL
                noteExposure(values, result.status, &diagnostics)
                FPPV3Gate.note("main", selected)
            } else {
                values = FPPV2SFamily.batchTransitLnL(
                    time: time, flux: flux, sigma: [Double](repeating: sigma, count: time.count),
                    ks: ks, pOrbs: periods, incsRad: incs, aRsArr: ar, eccs: es, wsRad: ws,
                    u1: us1, u2: us2, companionFluxRatio: fr, companionIsHost: companionIsHost,
                    exptime: switches.exposureArgs.exptime, nsub: switches.exposureArgs.nsub)
            }
            note(values, &diagnostics)
            for (j, i) in selected.enumerated() { lnl[i] = values[j] }
        }
        // Python `_lnZ_TTP`: `if not np.any(mask): return -np.inf` -- no reduction, so no step-7 diagnostics record
        if selected.isEmpty { return -.infinity }
        FPPV3Gate.stage("main", lnl)
        for i in 0..<n { lnl[i] += prior[i] }
        return FPPV2SFamily.lnZFromLogw(lnl)
    }

    /// Target-host EB and twin branch, also used for nearby NEB.
    public static func lnZTEB(
        time: [Double], flux: [Double], sigma: Double, sigmaVeto: Double,
        period: Double, mass: Double, radius: Double, teff: Double, n: Int,
        u1: Double, u2: Double, companionFluxRatio: [Double], companionIsHost: Bool = false,
        lnpriorCompanion: [Double], rng: inout FPPRNG,
        switches: FPPV2Switches = .production,
        diagnostics: inout FPPV3Diagnostics,
        fold: FPPV2ProdFold? = nil
    ) throws -> (eb: Double, twin: Double) {
        try validate(n: n, time: time, flux: flux, fold: fold)
        let comp = try broadcast(companionFluxRatio, n: n, name: "companion_flux_ratio")
        let prior = try broadcast(lnpriorCompanion, n: n, name: "lnprior_companion")
        let inc = FPPV2SFamily.sampleInc(count: n, rng: &rng)
        let q = FPPV2SFamily.sampleQ(count: n, hostMass: mass, rng: &rng,
                                    qBranchUpstream: switches.qBranchUpstream)
        let ecc = FPPV2SFamily.sampleEcc(count: n, planet: false, period: period, rng: &rng)
        let arg = FPPV2SFamily.sampleW(count: n, rng: &rng)
        let masses = q.map { $0 * mass }
        let stellar = FPPV2SFamily.stellarRelations(masses: masses)
        var radii = stellar.radii, teffs = stellar.temperatures
        for i in 0..<n { radii[i] = min(radii[i], radius); teffs[i] = min(teffs[i], teff) }
        let fluxEB = FPPV2SFamily.fluxRelation(masses: masses)
        let fluxTarget = FPPV2SFamily.fluxRelation(masses: [mass])[0]
        let frEB = fluxEB.map { $0 / ($0 + fluxTarget) }
        let ir = inc.map { $0 * FPPV2Consts.pi / 180.0 }
        let wr = arg.map { (90.0 - $0) * FPPV2Consts.pi / 180.0 }
        let denom = 4.0 * FPPV2.pow(FPPV2Consts.pi, 2.0)
        let pSec = period * 86400.0
        let p2 = pSec * pSec
        let twinSec = 2.0 * period * 86400.0
        let twin2 = twinSec * twinSec
        var a: [Double] = [], at: [Double] = [], ars: [Double] = [], atrs: [Double] = []
        var ks: [Double] = [], ebMask = [Bool](repeating: false, count: n), twinMask = ebMask
        for i in 0..<n {
            let ac = FPPV2.pow(FPPV2Consts.gCGS * (mass + masses[i]) * FPPV2Consts.msun / denom * p2, 1.0 / 3.0)
            // Preserve Python grouping: coefficient * (twin_seconds ** 2).
            let atc = FPPV2.pow(FPPV2Consts.gCGS * (mass + masses[i]) * FPPV2Consts.msun / denom * twin2, 1.0 / 3.0)
            a.append(ac); at.append(atc)
            ars.append(ac / (radius * FPPV2Consts.rsun)); atrs.append(atc / (radius * FPPV2Consts.rsun))
            ks.append(radii[i] / radius)
            let ec = (1.0 + ecc[i] * FPPV2.sin(arg[i] * FPPV2Consts.pi / 180.0)) / (1.0 - ecc[i] * ecc[i])
            let ptra = (radii[i] * FPPV2Consts.rsun + radius * FPPV2Consts.rsun) / ac * ec
            let ptraTwin = (radii[i] * FPPV2Consts.rsun + radius * FPPV2Consts.rsun) / atc * ec
            let minI = ptra <= 1.0 ? FPPV2.acos(ptra) : FPPV2Consts.pi / 2.0
            let minITwin = ptraTwin <= 1.0 ? FPPV2.acos(ptraTwin) : FPPV2Consts.pi / 2.0
            let coll = (radii[i] * FPPV2Consts.rsun + radius * FPPV2Consts.rsun) > ac * (1.0 - ecc[i])
            let collTwin = (2.0 * radius * FPPV2Consts.rsun) > atc * (1.0 - ecc[i])
            ebMask[i] = ir[i] >= minI && !coll && q[i] < 0.95
            twinMask[i] = ir[i] >= minITwin && !collTwin && q[i] >= 0.95
        }
        if switches.orbitPhysicsRoche {
            var rho = [Double](repeating: 0.0, count: n)
            for i in 0..<n {
                let rcm = radii[i] * FPPV2Consts.rsun
                rho[i] = masses[i] * FPPV2Consts.msun / ((4.0 / 3.0) * FPPV2Consts.pi * rcm * rcm * rcm)
            }
            let rm = [Double](repeating: mass, count: n), rr = [Double](repeating: radius, count: n)
            let re = FPPV2SFamily.rocheOk(aCm: a, ecc: ecc, hostMass: rm, hostRadius: rr, rhoComp: rho)
            let rt = FPPV2SFamily.rocheOk(aCm: at, ecc: ecc, hostMass: rm, hostRadius: rr, rhoComp: rho)
            for i in 0..<n { ebMask[i] = ebMask[i] && re[i]; twinMask[i] = twinMask[i] && rt[i] }
        }
        var u1Sec = [Double](repeating: u1, count: n), u2Sec = [Double](repeating: u2, count: n)
        if switches.secondaryGeometryConsistent {
            let chosen = (0..<n).filter { ebMask[$0] || twinMask[$0] }
            if !chosen.isEmpty {
                let ts = chosen.map { teffs[$0] }
                let gs = chosen.map { i -> Double in
                    let rr = radii[i] * FPPV2Consts.rsun
                    return FPPV2.log10(FPPV2Consts.gCGS * masses[i] * FPPV2Consts.msun / (rr * rr))
                }
                let ldc = try FPPV2SFamily.getLDCArrays(teff: ts, logg: gs)
                for (j, i) in chosen.enumerated() { u1Sec[i] = ldc.u1[j]; u2Sec[i] = ldc.u2[j] }
            }
        }

        func branch(_ twin: Bool) -> [Double] {
            let keep = twin ? twinMask : ebMask
            let axis = twin ? atrs : ars
            let branchPeriod = twin ? 2.0 * period : period
            let selected = (0..<n).filter { keep[$0] }
            var out = [Double](repeating: -.infinity, count: n)
            guard !selected.isEmpty else { return out }
            let kSel = selected.map { ks[$0] }, frSel = selected.map { frEB[$0] }
            let ps = [Double](repeating: branchPeriod, count: selected.count)
            let irSel = selected.map { ir[$0] }, arSel = selected.map { axis[$0] }
            let eSel = selected.map { ecc[$0] }, wSel = selected.map { wr[$0] }
            let cSel = selected.map { comp[$0] }
            let u1s = [Double](repeating: u1, count: selected.count), u2s = [Double](repeating: u2, count: selected.count)
            let us1 = selected.map { u1Sec[$0] }, us2 = selected.map { u2Sec[$0] }
            let values: [Double]
            if let fold {
                let result = FPPV2SFamily.batchEBTransitLnLProd(
                    fold: fold, sigmaVeto: sigmaVeto, ks: kSel, ebFluxRatios: frSel,
                    pOrbs: ps, incsRad: irSel, aRsArr: arSel, eccs: eSel, wsRad: wSel,
                    u1: u1s, u2: u2s, companionFluxRatio: cSel, companionIsHost: companionIsHost,
                    rejectSec: !twin, secConsistent: switches.secondaryGeometryConsistent,
                    kNudgeUpstream: switches.k1NudgeUpstream, u1Sec: us1, u2Sec: us2)
                values = result.lnL
                noteExposure(values, result.status, &diagnostics)
                FPPV3Gate.note(twin ? "twin" : "eb", selected)
            } else {
                values = FPPV2SFamily.batchEBTransitLnL(
                    time: time, flux: flux, sigma: [Double](repeating: sigma, count: time.count),
                    sigmaVeto: sigmaVeto, ks: kSel, ebFluxRatios: frSel, pOrbs: ps,
                    incsRad: irSel, aRsArr: arSel, eccs: eSel, wsRad: wSel,
                    u1: u1s, u2: u2s, companionFluxRatio: cSel, companionIsHost: companionIsHost,
                    exptime: switches.exposureArgs.exptime, nsub: switches.exposureArgs.nsub,
                    rejectSec: !twin, secConsistent: switches.secondaryGeometryConsistent,
                    kNudgeUpstream: switches.k1NudgeUpstream, u1Sec: us1, u2Sec: us2)
            }
            note(values, &diagnostics)
            for (j, i) in selected.enumerated() { out[i] = values[j] }
            return out
        }
        var lnlEB = branch(false), lnlTwin = branch(true)
        let gateLnl = FPPV3Gate.active ? (lnlEB, lnlTwin) : nil
        for i in 0..<n { lnlEB[i] += prior[i]; lnlTwin[i] += prior[i] }
        if let g = gateLnl { FPPV3Gate.stage("eb", g.0) }
        let lnZEB = FPPV2SFamily.lnZFromLogw(lnlEB)
        if let g = gateLnl { FPPV3Gate.stage("twin", g.1) }
        let lnZTwin = FPPV2SFamily.lnZFromLogw(lnlTwin)
        return (lnZEB, lnZTwin)
    }

    /// Planet orbiting a bound companion; companion proposal/prior precedes target-host draws.
    public static func lnZPTP(
        time: [Double], flux: [Double], sigma: Double, period: Double,
        mass: Double, radius: Double, n: Int, u1: Double, u2: Double,
        parallaxMas: Double, rng: inout FPPRNG, switches: FPPV2Switches = .production,
        diagnostics: inout FPPV3Diagnostics, fold: FPPV2ProdFold? = nil
    ) throws -> Double {
        let q = FPPV2SFamily.sampleQCompanion(count: n, hostMass: mass, rng: &rng,
                                             qBranchUpstream: switches.qBranchUpstream)
        let mc = q.map { $0 * mass }, fComp = FPPV2SFamily.fluxRelation(masses: mc)
        let fTarget = FPPV2SFamily.fluxRelation(masses: [mass])[0]
        let fr = fComp.map { $0 / ($0 + fTarget) }
        let prior = FPPV2SFamily.lnpriorCompanion(
            massSolar: mass, parallax: parallaxMas, fluxRatiosCompanion: fr, scenario: "TP")
        return try lnZTTP(time: time, flux: flux, sigma: sigma, period: period, mass: mass, radius: radius,
                          n: n, u1: u1, u2: u2, companionFluxRatio: fr, companionIsHost: false,
                          lnpriorCompanion: prior, rng: &rng, switches: switches,
                          diagnostics: &diagnostics, fold: fold)
    }

    /// Eclipsing binary around a bound companion; same companion prior on EB and twin branches.
    public static func lnZPEB(
        time: [Double], flux: [Double], sigma: Double, sigmaVeto: Double, period: Double,
        mass: Double, radius: Double, teff: Double, n: Int, u1: Double, u2: Double,
        parallaxMas: Double, rng: inout FPPRNG, switches: FPPV2Switches = .production,
        diagnostics: inout FPPV3Diagnostics, fold: FPPV2ProdFold? = nil
    ) throws -> (eb: Double, twin: Double) {
        let q = FPPV2SFamily.sampleQCompanion(count: n, hostMass: mass, rng: &rng,
                                             qBranchUpstream: switches.qBranchUpstream)
        let mc = q.map { $0 * mass }, fComp = FPPV2SFamily.fluxRelation(masses: mc)
        let fTarget = FPPV2SFamily.fluxRelation(masses: [mass])[0]
        let fr = fComp.map { $0 / ($0 + fTarget) }
        let prior = FPPV2SFamily.lnpriorCompanion(
            massSolar: mass, parallax: parallaxMas, fluxRatiosCompanion: fr, scenario: "EB")
        return try lnZTEB(time: time, flux: flux, sigma: sigma, sigmaVeto: sigmaVeto,
                          period: period, mass: mass, radius: radius, teff: teff, n: n, u1: u1, u2: u2,
                          companionFluxRatio: fr, companionIsHost: false, lnpriorCompanion: prior,
                          rng: &rng, switches: switches, diagnostics: &diagnostics, fold: fold)
    }
}
