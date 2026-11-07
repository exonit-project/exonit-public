import Foundation

/// Corrected D/B nearby-background kernels (DTP, DEB, BTP, BEB and twin branches).
public enum FPPV3DFamily {
    private static func backgroundLDC(
        teff: [Double], logg: [Double], metallicity: [Double]
    ) throws -> (u1: [Double], u2: [Double]) {
        do {
            return try FPPV2SFamily.ldcBackgroundArrays(teff: teff, logg: logg, metallicity: metallicity)
        } catch let error as FPPV2LDCNoRow {
            throw FPPV3Error("population_ldc_invalid", error.description)
        }
    }

    private static func note(_ values: [Double], _ diagnostics: inout FPPV3Diagnostics) {
        diagnostics.keplerUnconvergedDraws += values.reduce(into: 0) { n, x in if x.isNaN { n += 1 } }
    }

    private static func noteExposure(_ values: [Double], _ status: [Int], _ diagnostics: inout FPPV3Diagnostics) {
        let valid = (0..<min(values.count, status.count)).filter { values[$0].isFinite }
        guard let m = valid.map({ values[$0] }).max() else { return }
        let weights = valid.map { FPPV2.exp(values[$0] - m) }
        let total = FPPNumpy.pairwiseSum(weights)
        guard total > 0 else { return }
        var fb: [Double] = [], unc: [Double] = []
        for (j, i) in valid.enumerated() {
            if status[i] == 1 { fb.append(weights[j]) }
            if status[i] == 2 { unc.append(weights[j]) }
        }
        diagnostics.exposureGLFallbackShare = max(diagnostics.exposureGLFallbackShare, FPPNumpy.pairwiseSum(fb) / total)
        diagnostics.exposureUnconvergedShare = max(diagnostics.exposureUnconvergedShare, FPPNumpy.pairwiseSum(unc) / total)
    }

    private static func validate(n: Int, time: [Double], flux: [Double], fold: FPPV2ProdFold?) throws {
        guard n > 0 else { throw FPPV2SFamilyError("D-family evaluation requires N > 0") }
        if fold == nil {
            guard !time.isEmpty else { throw FPPV2SFamilyError("D-family evaluation requires a nonempty light curve") }
            guard time.count == flux.count else { throw FPPV2SFamilyError("time and flux lengths differ") }
        }
    }

    private static func indices(_ n: Int, _ population: FPPV3BackgroundPopulation,
                                parity: Bool, _ rng: inout FPPRNG) throws -> [Int] {
        let high = parity ? population.stars.count - 1 : population.stars.count
        guard high > 0 else {
            // Python BackgroundUnavailable(...), skip_reason population_invalid.
            throw FPPV3Error("population_invalid", "population too small for upstream D-family indexing")
        }
        return (0..<n).map { _ in rng.randint(low: 0, high: high) }
    }

    private static func prior(_ population: FPPV3BackgroundPopulation, _ fr: Double,
                              eb: Double? = nil, circular: Bool) -> Double {
        var area = (Double(population.stars.count) / 0.1)
            * FPPV2.pow(1.0 / 3600.0, 2.0) * FPPV2.pow(2.2, 2.0)
        if circular { area *= FPPV2Consts.pi }
        var lp = FPPV2.log(area)
        if lp > 0.0 { lp = 0.0 }
        var ratio = fr / (1.0 - fr)
        if let eb { ratio += eb / (1.0 - eb) }
        let dm = 2.5 * FPPV2.log10(ratio)
        return dm > 0.0 || !dm.isFinite ? -.infinity : lp
    }

    private static func exposure(_ config: FPPV3DFamilyConfig) -> (Double, Int) {
        config.upstreamExposure ? (FPPV2Consts.parityExptimeD, FPPV2Consts.parityNsub) : (0.0, 1)
    }

    private static func selected(_ a: [Double], _ ix: [Int]) -> [Double] { ix.map { a[$0] } }

    /// DTP: target planet diluted by one sampled background row.
    public static func lnZDTP(
        time: [Double], flux: [Double], sigma: Double, period: Double,
        targetMass: Double, targetRadius: Double, n: Int, u1: Double, u2: Double,
        population: FPPV3BackgroundPopulation, rng: inout FPPRNG,
        config: FPPV3DFamilyConfig, diagnostics: inout FPPV3Diagnostics,
        fold: FPPV2ProdFold? = nil
    ) throws -> Double {
        try validate(n: n, time: time, flux: flux, fold: fold)
        let idx = try indices(n, population, parity: config.parityIndex, &rng)
        let rp = FPPV2SFamily.sampleRp(count: n, hostMass: targetMass, rng: &rng)
        let inc = FPPV2SFamily.sampleInc(count: n, rng: &rng)
        let ecc = FPPV2SFamily.sampleEcc(count: n, planet: true, period: period, rng: &rng)
        let arg = FPPV2SFamily.sampleW(count: n, rng: &rng)
        let pSec = period * 86400.0
        let a = FPPV2.pow(((FPPV2Consts.gCGS * targetMass * FPPV2Consts.msun)
                           / (4.0 * FPPV2.pow(FPPV2Consts.pi, 2.0))) * (pSec * pSec), 1.0 / 3.0)
        var k = [Double](repeating: 0, count: n), ir = k, wr = k, mask = [Bool](repeating: false, count: n)
        for j in 0..<n {
            let ec = (1.0 + ecc[j] * FPPV2.sin(arg[j] * FPPV2Consts.pi / 180.0)) / (1.0 - ecc[j] * ecc[j])
            let ptra = (rp[j] * FPPV2Consts.rearth + targetRadius * FPPV2Consts.rsun) / a * ec
            let imin = ptra <= 1.0 ? FPPV2.acos(ptra) : FPPV2Consts.pi / 2.0
            let collision = (rp[j] * FPPV2Consts.rearth + targetRadius * FPPV2Consts.rsun) > a * (1.0 - ecc[j])
            ir[j] = inc[j] * FPPV2Consts.pi / 180.0
            wr[j] = (90.0 - arg[j]) * FPPV2Consts.pi / 180.0
            k[j] = rp[j] * FPPV2Consts.rearth / (targetRadius * FPPV2Consts.rsun)
            mask[j] = ir[j] >= imin && !collision
        }
        if config.roche {
            let ok = FPPV2SFamily.rocheOk(aCm: [Double](repeating: a, count: n), ecc: ecc,
                                          hostMass: [Double](repeating: targetMass, count: n),
                                          hostRadius: [Double](repeating: targetRadius, count: n),
                                          rhoComp: [Double](repeating: 5.51, count: n))
            for j in 0..<n { mask[j] = mask[j] && ok[j] }
        }
        let ix = (0..<n).filter { mask[$0] }
        var lnl = [Double](repeating: -.infinity, count: n)
        if !ix.isEmpty {
            let rows = ix.map { population.stars[idx[$0]] }
            let fr = rows.map { $0.fluxRatio }
            let ks = selected(k, ix), irs = selected(ir, ix), es = selected(ecc, ix), ws = selected(wr, ix)
            let axes = [Double](repeating: a / (targetRadius * FPPV2Consts.rsun), count: ix.count)
            let ps = [Double](repeating: period, count: ix.count), us1 = [Double](repeating: u1, count: ix.count), us2 = [Double](repeating: u2, count: ix.count)
            let val: [Double]
            if let fold {
                let result = FPPV2SFamily.batchTransitLnLProd(fold: fold, ks: ks, pOrbs: ps, incsRad: irs,
                    aRsArr: axes, eccs: es, wsRad: ws, u1: us1, u2: us2,
                    companionFluxRatio: fr, companionIsHost: false)
                val = result.lnL; noteExposure(val, result.status, &diagnostics)
                FPPV3Gate.note("main", ix)
            } else {
                val = FPPV2SFamily.batchTransitLnL(time: time, flux: flux,
                    sigma: [Double](repeating: sigma, count: time.count), ks: ks, pOrbs: ps,
                    incsRad: irs, aRsArr: axes, eccs: es, wsRad: ws, u1: us1, u2: us2,
                    companionFluxRatio: fr, companionIsHost: false,
                    exptime: exposure(config).0, nsub: exposure(config).1)
            }
            note(val, &diagnostics)
            for (j, i) in ix.enumerated() { lnl[i] = val[j] }
        }
        let logw = (0..<n).map { lnl[$0] + prior(population, population.stars[idx[$0]].fluxRatio, circular: config.circularPrior) }
        // Python `_lnZ_DTP` is `_lnZ_TTP`: `if not np.any(mask): return -np.inf` -- no reduction, no step-7 record
        if ix.isEmpty { return -.infinity }
        FPPV3Gate.stage("main", lnl)
        return FPPV2SFamily.lnZFromLogw(logw)
    }

    /// BTP: planet orbiting a sampled background star.
    public static func lnZBTP(
        time: [Double], flux: [Double], sigma: Double, period: Double, n: Int,
        population: FPPV3BackgroundPopulation, rng: inout FPPRNG,
        config: FPPV3DFamilyConfig, diagnostics: inout FPPV3Diagnostics,
        fold: FPPV2ProdFold? = nil
    ) throws -> Double {
        try validate(n: n, time: time, flux: flux, fold: fold)
        let idx = try indices(n, population, parity: false, &rng)
        let masses = idx.map { population.stars[$0].mass }, radii = idx.map { population.stars[$0].radius }
        let rp = try FPPV2SFamily.sampleRp(count: n, hostMasses: masses, rng: &rng)
        let inc = FPPV2SFamily.sampleInc(count: n, rng: &rng)
        let ecc = FPPV2SFamily.sampleEcc(count: n, planet: true, period: period, rng: &rng)
        let arg = FPPV2SFamily.sampleW(count: n, rng: &rng)
        let teff = population.stars.map { $0.teff }, logg = population.stars.map { $0.logg }, z = population.stars.map { $0.metallicity }
        let ldc = try backgroundLDC(teff: teff, logg: logg, metallicity: z)
        var ks = [Double](repeating: 0, count: n), ir = ks, wr = ks, axes = ks
        var mask = [Bool](repeating: false, count: n), acm = ks
        for j in 0..<n {
            let h = population.stars[idx[j]], psec = period * 86400.0
            let coeff = (FPPV2Consts.gCGS * h.mass * FPPV2Consts.msun) / (4.0 * FPPV2.pow(FPPV2Consts.pi, 2.0))
            acm[j] = FPPV2.pow(coeff * (psec * psec), 1.0 / 3.0)
            axes[j] = acm[j] / (radii[j] * FPPV2Consts.rsun)
            ks[j] = rp[j] * FPPV2Consts.rearth / (radii[j] * FPPV2Consts.rsun)
            ir[j] = inc[j] * FPPV2Consts.pi / 180.0; wr[j] = (90.0 - arg[j]) * FPPV2Consts.pi / 180.0
            let ec = (1.0 + ecc[j] * FPPV2.sin(arg[j] * FPPV2Consts.pi / 180.0)) / (1.0 - ecc[j] * ecc[j])
            let pt = (rp[j] * FPPV2Consts.rearth + radii[j] * FPPV2Consts.rsun) / acm[j] * ec
            let imin = pt <= 1.0 ? FPPV2.acos(pt) : FPPV2Consts.pi / 2.0
            let collision = (rp[j] * FPPV2Consts.rearth + radii[j] * FPPV2Consts.rsun) > acm[j] * (1.0 - ecc[j])
            mask[j] = ir[j] >= imin && !collision && h.logg >= 3.5 && h.teff <= 10000.0
            if config.roche {
                mask[j] = mask[j] && FPPV2SFamily.rocheOk(aCm: [acm[j]], ecc: [ecc[j]], hostMass: [h.mass], hostRadius: [radii[j]], rhoComp: [5.51])[0]
            }
        }
        let ix = (0..<n).filter { mask[$0] }
        var lnl = [Double](repeating: -.infinity, count: n)
        if !ix.isEmpty {
            let rows = ix.map { population.stars[idx[$0]] }
            let fr = rows.map { $0.fluxRatio }, u1 = ix.map { ldc.u1[idx[$0]] }, u2 = ix.map { ldc.u2[idx[$0]] }
            let val: [Double]
            if let fold {
                let result = FPPV2SFamily.batchTransitLnLProd(fold: fold, ks: selected(ks, ix),
                    pOrbs: [Double](repeating: period, count: ix.count), incsRad: selected(ir, ix),
                    aRsArr: selected(axes, ix), eccs: selected(ecc, ix), wsRad: selected(wr, ix),
                    u1: u1, u2: u2, companionFluxRatio: fr, companionIsHost: true)
                val = result.lnL; noteExposure(val, result.status, &diagnostics)
                FPPV3Gate.note("main", ix)
            } else {
                val = FPPV2SFamily.batchTransitLnL(time: time, flux: flux,
                    sigma: [Double](repeating: sigma, count: time.count), ks: selected(ks, ix),
                    pOrbs: [Double](repeating: period, count: ix.count), incsRad: selected(ir, ix),
                    aRsArr: selected(axes, ix), eccs: selected(ecc, ix), wsRad: selected(wr, ix),
                    u1: u1, u2: u2, companionFluxRatio: fr, companionIsHost: true,
                    exptime: exposure(config).0, nsub: exposure(config).1)
            }
            note(val, &diagnostics)
            for (j, i) in ix.enumerated() { lnl[i] = val[j] }
        }
        FPPV3Gate.stage("main", lnl)
        return FPPV2SFamily.lnZFromLogw((0..<n).map {
            lnl[$0] + prior(population, population.stars[idx[$0]].fluxRatio, circular: config.circularPrior)
        })
    }

    /// DEB: target-host EB and twin, with a sampled background dilution.
    public static func lnZDEB(
        time: [Double], flux: [Double], sigma: Double, sigmaVeto: Double, period: Double,
        mass: Double, radius: Double, teff: Double, u1: Double, u2: Double, n: Int,
        population: FPPV3BackgroundPopulation, rng: inout FPPRNG,
        config: FPPV3DFamilyConfig, diagnostics: inout FPPV3Diagnostics,
        fold: FPPV2ProdFold? = nil
    ) throws -> (eb: Double, twin: Double) {
        try validate(n: n, time: time, flux: flux, fold: fold)
        let inc = FPPV2SFamily.sampleInc(count: n, rng: &rng)
        let q = FPPV2SFamily.sampleQ(count: n, hostMass: mass, rng: &rng)
        let ecc = FPPV2SFamily.sampleEcc(count: n, planet: false, period: period, rng: &rng)
        let arg = FPPV2SFamily.sampleW(count: n, rng: &rng)
        let idx = try indices(n, population, parity: config.parityIndex, &rng)
        return try binary(time: time, flux: flux, sigma: sigma, sigmaVeto: sigmaVeto, period: period,
            targetMass: mass, targetRadius: radius, targetTeff: teff, targetU1: u1, targetU2: u2,
            n: n, population: population, idx: idx, inc: inc, q: q, ecc: ecc, arg: arg,
            backgroundHost: false, config: config, diagnostics: &diagnostics, fold: fold)
    }

    /// BEB: sampled background-host EB and twin. RNG order preserves the unused companion-q draw.
    public static func lnZBEB(
        time: [Double], flux: [Double], sigma: Double, sigmaVeto: Double, period: Double,
        targetMass: Double, targetRadius: Double, targetTeff: Double, targetU1: Double, targetU2: Double,
        n: Int, population: FPPV3BackgroundPopulation, rng: inout FPPRNG,
        config: FPPV3DFamilyConfig, diagnostics: inout FPPV3Diagnostics,
        fold: FPPV2ProdFold? = nil
    ) throws -> (eb: Double, twin: Double) {
        try validate(n: n, time: time, flux: flux, fold: fold)
        let inc = FPPV2SFamily.sampleInc(count: n, rng: &rng)
        var q = [Double](repeating: 0, count: n)
        var qUniform: [Double]? = nil
        if config.targetQ {
            q = FPPV2SFamily.sampleQ(count: n, hostMass: targetMass, rng: &rng)
        } else {
            qUniform = (0..<n).map { _ in rng.rand() }
        }
        _ = FPPV2SFamily.sampleQCompanion(count: n, hostMass: targetMass, rng: &rng)
        let ecc = FPPV2SFamily.sampleEcc(count: n, planet: false, period: period, rng: &rng)
        let arg = FPPV2SFamily.sampleW(count: n, rng: &rng)
        let idx = try indices(n, population, parity: false, &rng)
        if let uniforms = qUniform {
            let order = (0..<n).sorted {
                let a = population.stars[idx[$0]].mass, b = population.stars[idx[$1]].mass
                return a == b ? $0 < $1 : a < b
            }
            var at = 0
            while at < n {
                let host = population.stars[idx[order[at]]].mass
                var end = at + 1
                while end < n && population.stars[idx[order[end]]].mass == host { end += 1 }
                let group = (at..<end).map { uniforms[order[$0]] }
                let transformed = FPPV2SFamily.sampleQFromUniforms(group, hostMass: host)
                for offset in 0..<(end - at) { q[order[at + offset]] = transformed[offset] }
                at = end
            }
        }
        return try binary(time: time, flux: flux, sigma: sigma, sigmaVeto: sigmaVeto, period: period,
            targetMass: targetMass, targetRadius: targetRadius, targetTeff: targetTeff,
            targetU1: targetU1, targetU2: targetU2, n: n, population: population,
            idx: idx, inc: inc, q: q, ecc: ecc, arg: arg, backgroundHost: true,
            config: config, diagnostics: &diagnostics, fold: fold)
    }

    private static func binary(
        time: [Double], flux: [Double], sigma: Double, sigmaVeto: Double, period: Double,
        targetMass: Double, targetRadius: Double, targetTeff: Double, targetU1: Double, targetU2: Double,
        n: Int, population: FPPV3BackgroundPopulation, idx: [Int], inc: [Double], q: [Double],
        ecc: [Double], arg: [Double], backgroundHost: Bool, config: FPPV3DFamilyConfig,
        diagnostics: inout FPPV3Diagnostics, fold: FPPV2ProdFold?
    ) throws -> (eb: Double, twin: Double) {
        let mh = (0..<n).map { backgroundHost ? population.stars[idx[$0]].mass : targetMass }
        let rh = (0..<n).map { backgroundHost ? population.stars[idx[$0]].radius : targetRadius }
        let th = (0..<n).map { backgroundHost ? population.stars[idx[$0]].teff : targetTeff }
        let frc = (0..<n).map { population.stars[idx[$0]].fluxRatio }
        let meb = (0..<n).map { q[$0] * mh[$0] }
        let stellar = FPPV2SFamily.stellarRelationsCapped(masses: meb, maxR: rh, maxT: th)
        let reb = stellar.radii, teb = stellar.temperatures
        let fe = FPPV2SFamily.fluxRelation(masses: meb), fm = FPPV2SFamily.fluxRelation(masses: [targetMass])[0]
        var freb = (0..<n).map { fe[$0] / (fe[$0] + fm) }
        if backgroundHost {
            let fh = FPPV2SFamily.fluxRelation(masses: mh)
            for j in 0..<n {
                let frBound = fh[j] / (fh[j] + fm)
                freb[j] *= frc[j] / frBound
            }
        }
        let priors = (0..<n).map { prior(population, frc[$0], eb: backgroundHost ? freb[$0] : nil, circular: config.circularPrior) }
        var ir = inc.map { $0 * FPPV2Consts.pi / 180.0 }
        let wr = arg.map { (90.0 - $0) * FPPV2Consts.pi / 180.0 }
        var ec = [Double](repeating: 0, count: n), ac = ec, at = ec, ks = ec
        var mask = [Bool](repeating: false, count: n), twin = mask
        for j in 0..<n {
            ec[j] = (1.0 + ecc[j] * FPPV2.sin(arg[j] * FPPV2Consts.pi / 180.0)) / (1.0 - ecc[j] * ecc[j])
            let psec = period * 86400.0
            let coeff = (FPPV2Consts.gCGS * (mh[j] + meb[j]) * FPPV2Consts.msun) / (4.0 * FPPV2.pow(FPPV2Consts.pi, 2.0))
            ac[j] = FPPV2.pow(coeff * (psec * psec), 1.0 / 3.0)
            let pt = (reb[j] * FPPV2Consts.rsun + rh[j] * FPPV2Consts.rsun) / ac[j] * ec[j]
            let imin = pt <= 1.0 ? FPPV2.acos(pt) : FPPV2Consts.pi / 2.0
            let twinSec = 2.0 * psec
            // Python expression groups the period square before applying the coefficient.
            at[j] = FPPV2.pow(coeff * (twinSec * twinSec), 1.0 / 3.0)
            let ptw = (reb[j] * FPPV2Consts.rsun + rh[j] * FPPV2Consts.rsun) / at[j] * ec[j]
            let imint = ptw <= 1.0 ? FPPV2.acos(ptw) : FPPV2Consts.pi / 2.0
            let collision = (reb[j] * FPPV2Consts.rsun + rh[j] * FPPV2Consts.rsun) > ac[j] * (1.0 - ecc[j])
            let collisionTwin = (2.0 * rh[j] * FPPV2Consts.rsun) > at[j] * (1.0 - ecc[j])
            mask[j] = ir[j] >= imin && !collision && q[j] < 0.95
            twin[j] = ir[j] >= imint && !collisionTwin && q[j] >= 0.95
            if backgroundHost {
                let row = population.stars[idx[j]], ok = row.logg >= 3.5 && th[j] <= 10000.0
                mask[j] = mask[j] && ok; twin[j] = twin[j] && ok
            }
            if config.roche {
                let rc = reb[j] * FPPV2Consts.rsun
                let rho = meb[j] * FPPV2Consts.msun / ((4.0 / 3.0) * FPPV2Consts.pi * rc * rc * rc)
                mask[j] = mask[j] && FPPV2SFamily.rocheOk(aCm: [ac[j]], ecc: [ecc[j]], hostMass: [mh[j]], hostRadius: [rh[j]], rhoComp: [rho])[0]
                twin[j] = twin[j] && FPPV2SFamily.rocheOk(aCm: [at[j]], ecc: [ecc[j]], hostMass: [mh[j]], hostRadius: [rh[j]], rhoComp: [rho])[0]
            }
            ks[j] = reb[j] / rh[j]
        }
        var u1 = [Double](repeating: targetU1, count: n), u2 = [Double](repeating: targetU2, count: n)
        var u1sec = u1, u2sec = u2
        if backgroundHost {
            let all = try backgroundLDC(teff: population.stars.map { $0.teff },
                logg: population.stars.map { $0.logg }, metallicity: population.stars.map { $0.metallicity })
            for j in 0..<n { u1[j] = all.u1[idx[j]]; u2[j] = all.u2[idx[j]] }
            u1sec = u1; u2sec = u2
        }
        if config.secondaryConsistent {
            let selectedRows = (0..<n).filter { mask[$0] || twin[$0] }
            if !selectedRows.isEmpty {
                let ts = selectedRows.map { teb[$0] }
                let gs = selectedRows.map { j -> Double in
                    let rc = reb[j] * FPPV2Consts.rsun
                    return FPPV2.log10(FPPV2Consts.gCGS * meb[j] * FPPV2Consts.msun / (rc * rc))
                }
                let ldc: (u1: [Double], u2: [Double])
                if backgroundHost {
                    ldc = try backgroundLDC(teff: ts, logg: gs,
                        metallicity: selectedRows.map { population.stars[idx[$0]].metallicity })
                } else { ldc = try FPPV2SFamily.getLDCArrays(teff: ts, logg: gs) }
                for (j, row) in selectedRows.enumerated() { u1sec[row] = ldc.u1[j]; u2sec[row] = ldc.u2[j] }
            }
        }
        var output = [[Double](repeating: -.infinity, count: n), [Double](repeating: -.infinity, count: n)]
        for scenario in 0..<2 {
            let keep = scenario == 0 ? mask : twin
            let ix = (0..<n).filter { keep[$0] }
            guard !ix.isEmpty else { continue }
            let branchPeriod = ix.map { _ in period * (scenario == 0 ? 1.0 : 2.0) }
            let axes = ix.map { (scenario == 0 ? ac[$0] : at[$0]) / (rh[$0] * FPPV2Consts.rsun) }
            let values: [Double]
            if let fold {
                let r = FPPV2SFamily.batchEBTransitLnLProd(fold: fold, sigmaVeto: sigmaVeto,
                    ks: selected(ks, ix), ebFluxRatios: selected(freb, ix), pOrbs: branchPeriod,
                    incsRad: selected(ir, ix), aRsArr: axes, eccs: selected(ecc, ix), wsRad: selected(wr, ix),
                    u1: selected(u1, ix), u2: selected(u2, ix), companionFluxRatio: selected(frc, ix),
                    companionIsHost: backgroundHost, rejectSec: scenario == 0,
                    secConsistent: config.secondaryConsistent, kNudgeUpstream: config.kNudgeUpstream,
                    u1Sec: selected(u1sec, ix), u2Sec: selected(u2sec, ix))
                values = r.lnL; noteExposure(values, r.status, &diagnostics)
                FPPV3Gate.note(scenario == 0 ? "s0" : "s1", ix)
            } else {
                values = FPPV2SFamily.batchEBTransitLnL(time: time, flux: flux,
                    sigma: [Double](repeating: sigma, count: time.count), sigmaVeto: sigmaVeto,
                    ks: selected(ks, ix), ebFluxRatios: selected(freb, ix), pOrbs: branchPeriod,
                    incsRad: selected(ir, ix), aRsArr: axes, eccs: selected(ecc, ix), wsRad: selected(wr, ix),
                    u1: selected(u1, ix), u2: selected(u2, ix), companionFluxRatio: selected(frc, ix),
                    companionIsHost: backgroundHost, exptime: exposure(config).0, nsub: exposure(config).1,
                    rejectSec: scenario == 0, secConsistent: config.secondaryConsistent,
                    kNudgeUpstream: config.kNudgeUpstream, u1Sec: selected(u1sec, ix), u2Sec: selected(u2sec, ix))
            }
            note(values, &diagnostics)
            for (j, i) in ix.enumerated() { output[scenario][i] = values[j] }
        }
        FPPV3Gate.stage("s0", output[0])
        let outEB = FPPV2SFamily.lnZFromLogw((0..<n).map { output[0][$0] + priors[$0] })
        FPPV3Gate.stage("s1", output[1])
        let outTwin = FPPV2SFamily.lnZFromLogw((0..<n).map { output[1][$0] + priors[$0] })
        return (outEB, outTwin)
    }
}
