// NOT COMPILED BY THE AUTHOR (2026-11-05). <host> has no Swift toolchain or Metal device.

import Foundation

/// Process-wide router for transit likelihood evaluations.
/// Routes batch transit likelihood calls to an installed `FPPV3LikelihoodBackend` (e.g. Metal on Apple Silicon).
/// If no backend is installed, caller entry points compute on CPU.
public enum FPPV3LikelihoodRouter {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var _backend: (any FPPV3LikelihoodBackend)? = nil
    nonisolated(unsafe) private static var _latchedFailure: FPPV3GPUFailure? = nil
    nonisolated(unsafe) internal static var log: (String) -> Void = { print($0) }

    /// Process-wide, set once at worker start by the code that builds the Metal backend; nil = CPU-declared instance.
    /// Installing resets the latch. NOT thread-bound: a lock protects both.
    public static func install(_ backend: (any FPPV3LikelihoodBackend)?) {
        lock.lock()
        defer { lock.unlock() }
        _backend = backend
        _latchedFailure = nil
    }

    public static var backend: (any FPPV3LikelihoodBackend)? {
        lock.lock()
        defer { lock.unlock() }
        return _backend
    }

    /// The first failure since install, or nil. Once set it never clears (only install(...) does): "the instance stays down for the run".
    public static var latchedFailure: FPPV3GPUFailure? {
        lock.lock()
        defer { lock.unlock() }
        return _latchedFailure
    }

    /// Throws the latched failure if any (used by NativeFPPTaskProcessor).
    public static func throwIfLatched() throws {
        lock.lock()
        defer { lock.unlock() }
        if let failure = _latchedFailure {
            throw failure
        }
    }

    /// Latches the given error. If a failure has already been latched, this is a no-op (first failure wins).
    /// Logs one line on first latch: "[worker] fpp gpu failure case=<case>; instance stays down for the run".
    internal static func latch(_ error: Error) {
        lock.lock()
        defer { lock.unlock() }
        if _latchedFailure != nil {
            return
        }
        let failure = (error as? FPPV3GPUFailure) ?? .commandFailed
        _latchedFailure = failure
        log("[worker] fpp gpu failure case=\(failure); instance stays down for the run")
    }

    /// Called by the two heads. Returns (lnL, status, nAff). Behaviour:
    ///  - if a failure is latched: return placeholders at once, never call the backend;
    ///  - else call the backend; on a thrown FPPV3GPUFailure latch it; on any other error latch .commandFailed (fail closed);
    ///    print ONE line "[worker] fpp gpu failure case=<case>; instance stays down for the run" when latching;
    ///  - placeholders = lnL [-inf]*n, status [0]*n, nAff [0]*n (the caller's result is discarded: the task throws at the end);
    ///  - if FPPV3Gate.active: pass the CPU-shaped CallInput and the backend's device roots,
    ///    discrete rows and tangency margins to FPPV3Gate.kernelRows.
    public static func routeTransit(
        _ b: any FPPV3LikelihoodBackend,
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
        let n = ks.count
        if latchedFailure != nil {
            return (
                lnL: [Double](repeating: -.infinity, count: n),
                status: [Int](repeating: 0, count: n),
                nAff: [Int](repeating: 0, count: n)
            )
        }
        let result: FPPV3LnLProdResult
        do {
            result = try b.batchTransitLnLProd(
                fold: fold,
                ks: ks,
                pOrbs: pOrbs,
                incsRad: incsRad,
                aRsArr: aRsArr,
                eccs: eccs,
                wsRad: wsRad,
                u1: u1,
                u2: u2,
                companionFluxRatio: companionFluxRatio,
                companionIsHost: companionIsHost
            )
        } catch {
            latch(error)
            return (
                lnL: [Double](repeating: -.infinity, count: n),
                status: [Int](repeating: 0, count: n),
                nAff: [Int](repeating: 0, count: n)
            )
        }
        if FPPV3Gate.active {
            let input = FPPV3Gate.CallInput(
                k: ks,
                period: pOrbs,
                inc: incsRad,
                aRs: aRsArr,
                ecc: eccs,
                wPeri: wsRad,
                u1: u1,
                u2: u2,
                companionFluxRatio: companionFluxRatio,
                ebArrays: [],
                sigmaVeto: nil,
                flags: [companionIsHost],
                fold: fold
            )
            FPPV3Gate.kernelRows(
                periods: pOrbs,
                aRs: aRsArr,
                incsRad: incsRad,
                eccs: eccs,
                wsRad: wsRad,
                ksUsed: ks,
                lnl: result.lnL,
                status: result.status,
                input: input,
                deviceRoots: result.deviceRoots,
                discrete: result.discrete,
                deviceTangency: result.deviceTangency
            )
        }
        return (lnL: result.lnL, status: result.status, nAff: result.nAff)
    }

    public static func routeEB(
        _ b: any FPPV3LikelihoodBackend,
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
        let n = ks.count
        if latchedFailure != nil {
            return (
                lnL: [Double](repeating: -.infinity, count: n),
                status: [Int](repeating: 0, count: n),
                nAff: [Int](repeating: 0, count: n)
            )
        }
        let result: FPPV3LnLProdResult
        do {
            result = try b.batchEBTransitLnLProd(
                fold: fold,
                sigmaVeto: sigmaVeto,
                ks: ks,
                ebFluxRatios: ebFluxRatios,
                pOrbs: pOrbs,
                incsRad: incsRad,
                aRsArr: aRsArr,
                eccs: eccs,
                wsRad: wsRad,
                u1: u1,
                u2: u2,
                companionFluxRatio: companionFluxRatio,
                companionIsHost: companionIsHost,
                rejectSec: rejectSec,
                secConsistent: secConsistent,
                kNudgeUpstream: kNudgeUpstream,
                u1Sec: u1Sec,
                u2Sec: u2Sec
            )
        } catch {
            latch(error)
            return (
                lnL: [Double](repeating: -.infinity, count: n),
                status: [Int](repeating: 0, count: n),
                nAff: [Int](repeating: 0, count: n)
            )
        }
        if FPPV3Gate.active {
            let ksUsed = ks.map { kNudgeUpstream && abs($0 - 1.0) < 1e-6 ? $0 * 0.999 : $0 }
            let input = FPPV3Gate.CallInput(
                k: ks,
                period: pOrbs,
                inc: incsRad,
                aRs: aRsArr,
                ecc: eccs,
                wPeri: wsRad,
                u1: u1,
                u2: u2,
                companionFluxRatio: companionFluxRatio,
                ebArrays: [ebFluxRatios, u1Sec, u2Sec],
                sigmaVeto: sigmaVeto,
                flags: [companionIsHost, rejectSec, secConsistent, kNudgeUpstream],
                fold: fold
            )
            FPPV3Gate.kernelRows(
                periods: pOrbs,
                aRs: aRsArr,
                incsRad: incsRad,
                eccs: eccs,
                wsRad: wsRad,
                ksUsed: ksUsed,
                lnl: result.lnL,
                status: result.status,
                input: input,
                deviceRoots: result.deviceRoots,
                discrete: result.discrete,
                deviceTangency: result.deviceTangency
            )
        }
        return (lnL: result.lnL, status: result.status, nAff: result.nAff)
    }
}
