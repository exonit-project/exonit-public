// NOT COMPILED BY THE AUTHOR (2026-11-05). <host> has no Swift toolchain or Metal device.

import Foundation

/// Result of a transit likelihood evaluation on the likelihood backend.
public struct FPPV3LnLProdResult: Sendable {
    public var lnL: [Double]          // n
    public var status: [Int]          // n, exposure-order status 0/1/2 (K2Draw), 0 for empty draws
    public var nAff: [Int]            // n
    public var deviceRoots: [Double]? // n*5, the K1 roots as recorded by the device; nil only for non-device backends
    public var discrete: FPPV3Gate.DiscreteRows? // n orders and CSR affected bins / ladder levels
    public var deviceTangency: [Double]? // n, computed from deviceZMin and the kernel's k
    public var deviceZMin: [Double]? // n, NaN if not reached (or rejected before K2)

    public init(
        lnL: [Double],
        status: [Int],
        nAff: [Int],
        deviceRoots: [Double]? = nil,
        discrete: FPPV3Gate.DiscreteRows? = nil,
        deviceTangency: [Double]? = nil,
        deviceZMin: [Double]? = nil
    ) {
        self.lnL = lnL
        self.status = status
        self.nAff = nAff
        self.deviceRoots = deviceRoots
        self.discrete = discrete
        self.deviceTangency = deviceTangency
        self.deviceZMin = deviceZMin
    }
}

/// Abstract backend interface for likelihood evaluations (Metal on Apple Silicon, CPU fallback, etc.).
public protocol FPPV3LikelihoodBackend: Sendable {
    func batchTransitLnLProd(
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
    ) throws -> FPPV3LnLProdResult

    func batchEBTransitLnLProd(
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
    ) throws -> FPPV3LnLProdResult
}
