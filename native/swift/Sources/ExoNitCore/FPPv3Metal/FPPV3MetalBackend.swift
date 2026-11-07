// NOT COMPILED BY THE AUTHOR (2026-11-05). <host> has no Swift toolchain or Metal device.

#if canImport(Metal)
import Foundation
import Metal
import Darwin

// MARK: - LAYOUT v1 Types & Structures

private struct K1ParamsLayout {
    var n: UInt32 = 0
    var nb: UInt32 = 0
    var companionIsHost: UInt32 = 0
    var u1Stride: UInt32 = 0
    var u2Stride: UInt32 = 0
    var segOff: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    var pad0: UInt32 = 0 // bit 0: EB dilA in segment 6
    var pad1: UInt32 = 0 // EB active count; any trailing K1 rows are recorder roots only
    var foldW: UInt64 = 0
    var foldDelta: UInt64 = 0
}

private struct K1OutRow {
    var root: (UInt64, UInt64, UInt64, UInt64, UInt64)
    var k1State: UInt32
    var firstAff: UInt32
    var nAff: UInt32
    var epsRef: Float
    var epsMin: Float
    var dilA: Float
}

private struct K2DrawRow {
    var orderStatus: UInt32
    var nEval: UInt32
}

private struct BinOutRow {
    var D: Float
    var levelNwin: UInt32
}

private final class PooledSlot {
    let drawIn: MTLBuffer
    let k1Params: MTLBuffer
    let k1Out: MTLBuffer
    let zMinOut: MTLBuffer
    let drawC: MTLBuffer
    let binOff: MTLBuffer
    let k2Draw: MTLBuffer
    var binOut: MTLBuffer?
    let fixedBytes: Int
    var binOutAllocatedBytes: Int = 0

    init(
        drawIn: MTLBuffer,
        k1Params: MTLBuffer,
        k1Out: MTLBuffer,
        zMinOut: MTLBuffer,
        drawC: MTLBuffer,
        binOff: MTLBuffer,
        k2Draw: MTLBuffer,
        fixedBytes: Int
    ) {
        self.drawIn = drawIn
        self.k1Params = k1Params
        self.k1Out = k1Out
        self.zMinOut = zMinOut
        self.drawC = drawC
        self.binOff = binOff
        self.k2Draw = k2Draw
        self.fixedBytes = fixedBytes
    }
}

private final class CachedFold {
    let foldBuf: MTLBuffer
    let foldParams: [UInt32]
    let words: [UInt64]
    let totalBytes: Int

    init(foldBuf: MTLBuffer, foldParams: [UInt32], words: [UInt64], totalBytes: Int) {
        self.foldBuf = foldBuf
        self.foldParams = foldParams
        self.words = words
        self.totalBytes = totalBytes
    }
}

private struct PendingChunk {
    let slotIndex: Int
    let drawOffset: Int
    let drawCount: Int
    let cb2: MTLCommandBuffer
    let totalBins: Int
    let chunkIndex: Int
}

// MARK: - FPPV3MetalBackend

public final class FPPV3MetalBackend: FPPV3LikelihoodBackend, FPPV3GPUBackend, @unchecked Sendable {
    public struct Config: Sendable {
        public var chunk = 65_536
        public var memoryCapBytes = 512 << 20
        public var allowedDeviceNamePrefixes = ["Apple M4"]      // device scope (design 4.1); anything else -> .noDevice
        public var binOutCapBytes = 128 << 20

        public init(
            chunk: Int = 65_536,
            memoryCapBytes: Int = 512 << 20,
            allowedDeviceNamePrefixes: [String] = ["Apple M4"],
            binOutCapBytes: Int = 128 << 20
        ) {
            self.chunk = chunk
            self.memoryCapBytes = memoryCapBytes
            self.allowedDeviceNamePrefixes = allowedDeviceNamePrefixes
            self.binOutCapBytes = binOutCapBytes
        }
    }

    struct TestHooks: Sendable {
        var failInit: FPPV3GPUFailure? = nil
        var failCommandAtChunk: Int? = nil
        var nanBinOutAtChunk: Int? = nil
        var breakContiguity: Bool = false
    }

    public struct Stats: Sendable {
        public var gpuSecondsK1: Double = 0
        public var gpuSecondsPrefix: Double = 0
        public var gpuSecondsOrder: Double = 0
        public var gpuSecondsLadder: Double = 0
        public var hostAssemblySeconds: Double = 0
        public var chunks: Int = 0
        public var nEvaluated: Int = 0
        public var bytesLive: Int = 0
        public var recheckedBins: Int = 0

        public var recheckCount: Int { recheckedBins }

        public init(
            gpuSecondsK1: Double = 0,
            gpuSecondsPrefix: Double = 0,
            gpuSecondsOrder: Double = 0,
            gpuSecondsLadder: Double = 0,
            hostAssemblySeconds: Double = 0,
            chunks: Int = 0,
            nEvaluated: Int = 0,
            bytesLive: Int = 0,
            recheckedBins: Int = 0
        ) {
            self.gpuSecondsK1 = gpuSecondsK1
            self.gpuSecondsPrefix = gpuSecondsPrefix
            self.gpuSecondsOrder = gpuSecondsOrder
            self.gpuSecondsLadder = gpuSecondsLadder
            self.hostAssemblySeconds = hostAssemblySeconds
            self.chunks = chunks
            self.nEvaluated = nEvaluated
            self.bytesLive = bytesLive
            self.recheckedBins = recheckedBins
        }
    }

    public let config: Config
    let hooks: TestHooks

    public let deviceName: String
    public let osBuild: String                           // sysctl kern.osversion ("unknown" if unreadable)
    public var backendDevice: String {
        "\(deviceName) | macOS \(osBuild)"
    }
    public var gpuKernelVersion: String {
        FPPV3MetalSource.gpuKernelVersion
    }

    public var canary: (@Sendable (FPPV3MetalBackend) throws -> Void)?
    public private(set) var lastStats: Stats = Stats()

    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary

    let pK1: MTLComputePipelineState
    let pPrefix: MTLComputePipelineState
    let pOrder: MTLComputePipelineState
    let pLadder: MTLComputePipelineState

    let glx: MTLBuffer
    let glw: MTLBuffer
    let orders: MTLBuffer

    // Memory tracking
    private var currentLiveBytes: Int = 0
    private let liveLock = NSLock()

    // Pooled slots (two slots, allocated on first call)
    private var pooledSlots: [PooledSlot]? = nil
    private let poolLock = NSLock()

    // Fold cache (LRU up to 4)
    private var foldCache = [UInt64: CachedFold]()
    private var foldLRU = [UInt64]()
    private let foldLock = NSLock()

    // Debug tracking of last call's bound inputs
    private var lastBoundDrawArraysSegments: [[Double]] = []
    private var lastBoundFoldBufferWords: [UInt64] = []
    var lastFoldBuffer: MTLBuffer? = nil

    private static func getOSVersion() -> String {
        var size: size_t = 0
        sysctlbyname("kern.osversion", nil, &size, nil, 0)
        guard size > 0 else { return "unknown" }
        var buffer = [CChar](repeating: 0, count: size)
        let ret = sysctlbyname("kern.osversion", &buffer, &size, nil, 0)
        guard ret == 0 else { return "unknown" }
        return String(cString: buffer)
    }

    public convenience init(config: Config = Config()) throws {
        try self.init(config: config, hooks: TestHooks())
    }

    init(config: Config, hooks: TestHooks) throws {
        self.config = config
        self.hooks = hooks

        if let fail = hooks.failInit {
            throw fail
        }

        guard let dev = MTLCreateSystemDefaultDevice() else {
            print("[metal] No system default Metal device found")
            throw FPPV3GPUFailure.noDevice
        }

        let name = dev.name
        self.deviceName = name
        self.osBuild = Self.getOSVersion()

        let matches = config.allowedDeviceNamePrefixes.contains { name.hasPrefix($0) }
        guard matches else {
            print("[metal] Device name '\(name)' does not match allowed prefixes \(config.allowedDeviceNamePrefixes)")
            throw FPPV3GPUFailure.noDevice
        }

        guard let q = dev.makeCommandQueue() else {
            print("[metal] Failed to make command queue")
            throw FPPV3GPUFailure.noDevice
        }
        self.device = dev
        self.queue = q

        let opts = MTLCompileOptions()
        if #available(macOS 15.0, *) {
            opts.mathMode = .safe
            opts.mathFloatingPointFunctions = .precise
        } else {
            opts.fastMathEnabled = false
        }

        let lib: MTLLibrary
        do {
            lib = try dev.makeLibrary(source: FPPV3MetalSource.text, options: opts)
        } catch {
            print("[metal] Library compilation failed: \(error)")
            throw FPPV3GPUFailure.compile
        }
        self.library = lib

        guard let fnK1 = lib.makeFunction(name: "k1_contacts_gate"),
              let fnPrefix = lib.makeFunction(name: "k_prefix_naff"),
              let fnOrder = lib.makeFunction(name: "k2_order"),
              let fnLadder = lib.makeFunction(name: "k2_ladder") else {
            print("[metal] Missing required kernel function")
            throw FPPV3GPUFailure.missingFunction
        }

        do {
            let pk1 = try dev.makeComputePipelineState(function: fnK1)
            let pprefix = try dev.makeComputePipelineState(function: fnPrefix)
            let porder = try dev.makeComputePipelineState(function: fnOrder)
            let pladder = try dev.makeComputePipelineState(function: fnLadder)

            guard pk1.threadExecutionWidth == 32 &&
                  pprefix.threadExecutionWidth == 32 &&
                  porder.threadExecutionWidth == 32 &&
                  pladder.threadExecutionWidth == 32 else {
                print("[metal] Pipeline threadExecutionWidth != 32")
                throw FPPV3GPUFailure.compile
            }
            self.pK1 = pk1
            self.pPrefix = pprefix
            self.pOrder = porder
            self.pLadder = pladder
        } catch let failure as FPPV3GPUFailure {
            throw failure
        } catch {
            print("[metal] Pipeline creation failed: \(error)")
            throw FPPV3GPUFailure.compile
        }

        // B7: GL tables (glx, glw, orders)
        var x = [SIMD2<Float>]()
        var w = [Float]()
        for oi in 0..<FPPV2GLTables.orders.count {
            for i in 0..<FPPV2GLTables.maxM {
                let v = FPPV2GL.glX(oi, i)
                let hi = Float(v)
                x.append(SIMD2<Float>(hi, Float(v - Double(hi))))
                w.append(Float(FPPV2GL.glW(oi, i)))
            }
        }
        var ordersArr = FPPV2GLTables.orders.map { Int32($0) }
        let glxLen = x.count * MemoryLayout<SIMD2<Float>>.stride
        let glwLen = w.count * MemoryLayout<Float>.stride
        let ordLen = ordersArr.count * MemoryLayout<Int32>.stride
        let totalTablesBytes = glxLen + glwLen + ordLen

        guard totalTablesBytes <= config.memoryCapBytes else {
            throw FPPV3GPUFailure.vramCapExceeded
        }

        guard let bgx = dev.makeBuffer(bytes: x, length: glxLen, options: .storageModeShared),
              let bgw = dev.makeBuffer(bytes: w, length: glwLen, options: .storageModeShared),
              let bord = dev.makeBuffer(bytes: &ordersArr, length: ordLen, options: .storageModeShared) else {
            throw FPPV3GPUFailure.vramCapExceeded
        }
        self.glx = bgx
        self.glw = bgw
        self.orders = bord
        self.currentLiveBytes = totalTablesBytes
    }

    public func deviceReady() async throws {
        let matches = config.allowedDeviceNamePrefixes.contains { deviceName.hasPrefix($0) }
        guard matches else {
            throw FPPV3GPUFailure.noDevice
        }
    }

    public func runCanary() async throws {
        guard let canary = canary else {
            throw FPPV3GPUFailure.canaryFailed
        }
        do {
            try canary(self)
        } catch let failure as FPPV3GPUFailure {
            throw failure
        } catch {
            throw FPPV3GPUFailure.canaryFailed
        }
    }

    public func batchEBTransitLnLProd(
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
    ) throws -> FPPV3LnLProdResult {
        let n = ks.count
        guard ebFluxRatios.count == n, pOrbs.count == n, incsRad.count == n,
              aRsArr.count == n, eccs.count == n, wsRad.count == n,
              companionFluxRatio.count == n, u1Sec.count == n, u2Sec.count == n,
              (u1.count == n || u1.count == 1), (u2.count == n || u2.count == 1) else {
            throw FPPV3GPUFailure.unsupportedSwitchSet
        }

        var ksUsed = [Double]()
        var dilAs = [Double]()
        var active = [Bool]()
        ksUsed.reserveCapacity(n)
        dilAs.reserveCapacity(n)
        active.reserveCapacity(n)
        for j in 0..<n {
            let u1j = u1.count == 1 ? u1[0] : u1[j]
            let u2j = u2.count == 1 ? u2[0] : u2[j]
            let prep = FPPV2SFamily.prepareEBTransitProdDraw(
                sigmaVeto: sigmaVeto, kRaw: ks[j], ebFluxRatio: ebFluxRatios[j],
                period: pOrbs[j], incRad: incsRad[j], aRs: aRsArr[j], ecc: eccs[j], wRad: wsRad[j],
                u1: u1j, u2: u2j, companionFluxRatio: companionFluxRatio[j], companionIsHost: companionIsHost,
                rejectSec: rejectSec, secConsistent: secConsistent, kNudgeUpstream: kNudgeUpstream,
                u1Sec: u1Sec[j], u2Sec: u2Sec[j]
            )
            if !prep.rejected {
                guard [prep.k, pOrbs[j], incsRad[j], aRsArr[j], eccs[j], wsRad[j],
                       u1j, u2j, prep.dilA, prep.dilB].allSatisfy({ $0.isFinite }) else {
                    throw FPPV3GPUFailure.nonFinite
                }
                // K2 consumes the deficit dilA * (1 - flux), so its affine
                // must sum to one before the per-draw f64 dilA is uploaded.
                guard abs(prep.dilA + prep.dilB - 1.0) <= 1e-12 else {
                    throw FPPV3GPUFailure.unsupportedSwitchSet
                }
            }
            ksUsed.append(prep.k)
            dilAs.append(prep.dilA)
            active.append(!prep.rejected)
        }

        // Compact active draws first. Rejected draws are appended only for
        // recorder roots; they never enter a K2 dispatch. Scatter back to the
        // caller's order, keeping CPU veto placeholders on every rejected row.
        let activeIndices = (0..<n).filter { active[$0] }
        let indices = activeIndices + (FPPV3Gate.active ? (0..<n).filter { !active[$0] } : [])
        let prepared = try batchPreparedTransitLnLProd(
            fold: fold, ks: indices.map { ksUsed[$0] }, pOrbs: indices.map { pOrbs[$0] },
            incsRad: indices.map { incsRad[$0] }, aRsArr: indices.map { aRsArr[$0] },
            eccs: indices.map { eccs[$0] }, wsRad: indices.map { wsRad[$0] },
            u1: u1.count == 1 ? u1 : indices.map { u1[$0] },
            u2: u2.count == 1 ? u2 : indices.map { u2[$0] },
            companionFluxRatio: indices.map { dilAs[$0] }, companionIsHost: companionIsHost,
            ebActiveCount: activeIndices.count
        )
        var lnL = [Double](repeating: -.infinity, count: n)
        var status = [Int](repeating: 0, count: n)
        var nAff = [Int](repeating: 0, count: n)
        var roots = [Double](repeating: Double(bitPattern: 0x7ff8000000000000), count: n * 5)
        var zMin = [Double](repeating: Double(bitPattern: 0x7ff8000000000000), count: n)
        var tangency = zMin
        var discrete = FPPV3Gate.DiscreteRows()
        var preparedRowForDraw = [Int](repeating: -1, count: n)
        guard let preparedRoots = prepared.deviceRoots, preparedRoots.count == indices.count * 5 else {
            throw FPPV3GPUFailure.nonFinite
        }
        guard let preparedDiscrete = prepared.discrete,
              preparedDiscrete.defect(rows: indices.count) == nil,
              let preparedTangency = prepared.deviceTangency, preparedTangency.count == indices.count,
              let preparedZMin = prepared.deviceZMin, preparedZMin.count == indices.count else {
            throw FPPV3GPUFailure.nonFinite
        }
        for (i, j) in indices.enumerated() {
            lnL[j] = prepared.lnL[i]
            status[j] = prepared.status[i]
            nAff[j] = prepared.nAff[i]
            for r in 0..<5 { roots[j * 5 + r] = preparedRoots[i * 5 + r] }
            if active[j] {
                preparedRowForDraw[j] = i
                zMin[j] = preparedZMin[i]
                tangency[j] = preparedTangency[i]
            }
        }
        for j in 0..<n {
            let i = preparedRowForDraw[j]
            if i >= 0 {
                let row = preparedDiscrete.row(i)
                discrete.push(order: row.order, bins: row.bins, levels: row.levels)
            } else {
                discrete.push(order: -1, bins: [], levels: [])
            }
        }
        return FPPV3LnLProdResult(
            lnL: lnL, status: status, nAff: nAff, deviceRoots: roots,
            discrete: discrete, deviceTangency: tangency, deviceZMin: zMin
        )
    }

    public func debugBoundDrawArrays() -> [[Double]] {
        return lastBoundDrawArraysSegments
    }

    func debugBoundFoldWords() -> [UInt64] {
        return lastBoundFoldBufferWords
    }

    // MARK: - Fold Upload & Cache

    private static func fnv1a64Fold(fold: FPPV2ProdFold) -> UInt64 {
        var hash: UInt64 = 0xcbf29ce484222325
        let prime: UInt64 = 0x100000001b3

        func appendBytes<T>(_ value: inout T) {
            withUnsafeBytes(of: &value) { buf in
                for b in buf {
                    hash = (hash ^ UInt64(b)) &* prime
                }
            }
        }
        func appendSlice<T>(_ slice: [T]) {
            slice.withUnsafeBytes { buf in
                for b in buf {
                    hash = (hash ^ UInt64(b)) &* prime
                }
            }
        }

        var nb = Int64(fold.nb)
        appendBytes(&nb)
        var wBits = fold.w.bitPattern
        appendBytes(&wBits)
        var deltaBits = fold.delta.bitPattern
        appendBytes(&deltaBits)
        var baseChi2Bits = fold.baseChi2.bitPattern
        appendBytes(&baseChi2Bits)
        var lnnormBits = fold.lnnorm.bitPattern
        appendBytes(&lnnormBits)

        appendSlice(fold.arrays.n)
        appendSlice(fold.arrays.sigma)
        appendSlice(fold.arrays.tmean)
        appendSlice(fold.arrays.varT)
        appendSlice(fold.arrays.tmin)
        appendSlice(fold.arrays.tmax)
        appendSlice(fold.arrays.nodeTab)
        appendSlice(fold.arrays.ptsOff)
        appendSlice(fold.arrays.ptsT)
        appendSlice(fold.f)

        return hash
    }

    private static func validateFold(_ fold: FPPV2ProdFold) throws {
        let nb = fold.nb
        guard fold.arrays.tmin.count == nb && fold.arrays.tmax.count == nb else {
            throw FPPV3GPUFailure.unsupportedSwitchSet
        }
        for b in 0..<nb {
            let tmin = fold.arrays.tmin[b]
            let tmax = fold.arrays.tmax[b]
            guard tmin <= tmax else {
                throw FPPV3GPUFailure.unsupportedSwitchSet
            }
            if b > 0 {
                guard fold.arrays.tmin[b] >= fold.arrays.tmin[b - 1] &&
                      fold.arrays.tmax[b] >= fold.arrays.tmax[b - 1] else {
                    throw FPPV3GPUFailure.unsupportedSwitchSet
                }
            }
        }
    }

    private func getOrCreateFoldBuffers(
        _ fold: FPPV2ProdFold
    ) throws -> CachedFold {
        try Self.validateFold(fold)

        let key = Self.fnv1a64Fold(fold: fold)
        foldLock.lock()
        defer { foldLock.unlock() }

        if let cached = foldCache[key] {
            if let idx = foldLRU.firstIndex(of: key) {
                foldLRU.remove(at: idx)
            }
            foldLRU.append(key)
            self.lastFoldBuffer = cached.foldBuf
            self.lastBoundFoldBufferWords = cached.words
            return cached
        }

        let nb = fold.nb
        let offN = 0
        let offSigma = offN + nb
        let offTmean = offSigma + nb
        let offVarT = offTmean + nb
        let offTmin = offVarT + nb
        let offTmax = offTmin + nb
        let offNodeTab = offTmax + nb
        let offPtsOff = offNodeTab + nb * 6 * 128
        let offPtsT = offPtsOff + (nb + 1)
        let ptsTotal = fold.arrays.ptsOff.count > nb ? fold.arrays.ptsOff[nb] : 0
        let totalWords = offPtsT + ptsTotal
        let totalBytes = totalWords * 8

        while foldLRU.count >= 4 {
            let evictKey = foldLRU.removeFirst()
            if let evicted = foldCache.removeValue(forKey: evictKey) {
                liveLock.lock()
                currentLiveBytes -= evicted.totalBytes
                liveLock.unlock()
            }
        }

        liveLock.lock()
        if currentLiveBytes + totalBytes > config.memoryCapBytes {
            liveLock.unlock()
            throw FPPV3GPUFailure.vramCapExceeded
        }
        liveLock.unlock()

        var words = [UInt64](repeating: 0, count: totalWords)
        for i in 0..<nb {
            words[offN + i] = UInt64(bitPattern: Int64(fold.arrays.n[i]))
            words[offSigma + i] = fold.arrays.sigma[i].bitPattern
            words[offTmean + i] = fold.arrays.tmean[i].bitPattern
            words[offVarT + i] = fold.arrays.varT[i].bitPattern
            words[offTmin + i] = fold.arrays.tmin[i].bitPattern
            words[offTmax + i] = fold.arrays.tmax[i].bitPattern
        }
        let nodeCount = min(fold.arrays.nodeTab.count, nb * 6 * 128)
        for i in 0..<nodeCount {
            words[offNodeTab + i] = fold.arrays.nodeTab[i].bitPattern
        }
        let ptsOffCount = min(fold.arrays.ptsOff.count, nb + 1)
        for i in 0..<ptsOffCount {
            words[offPtsOff + i] = UInt64(bitPattern: Int64(fold.arrays.ptsOff[i]))
        }
        let ptsTCount = min(fold.arrays.ptsT.count, ptsTotal)
        for i in 0..<ptsTCount {
            words[offPtsT + i] = fold.arrays.ptsT[i].bitPattern
        }

        guard let buf = device.makeBuffer(bytes: words, length: totalBytes, options: .storageModeShared) else {
            throw FPPV3GPUFailure.vramCapExceeded
        }

        var fv = [UInt32](repeating: 0, count: 16)
        fv[0] = UInt32(nb)
        fv[1] = UInt32(offN)
        fv[2] = UInt32(offSigma)
        fv[3] = UInt32(offTmean)
        fv[4] = UInt32(offVarT)
        fv[5] = UInt32(offTmin)
        fv[6] = UInt32(offTmax)
        fv[7] = UInt32(offNodeTab)
        fv[8] = UInt32(offPtsOff)
        fv[9] = UInt32(offPtsT)
        fv[10] = 6
        fv[11] = 128
        fv[12] = UInt32(ptsTotal)

        liveLock.lock()
        currentLiveBytes += totalBytes
        liveLock.unlock()

        let cached = CachedFold(foldBuf: buf, foldParams: fv, words: words, totalBytes: totalBytes)
        foldCache[key] = cached
        foldLRU.append(key)
        self.lastFoldBuffer = buf
        self.lastBoundFoldBufferWords = words
        return cached
    }

    // MARK: - Pooled Slots

    private func ensurePooledSlots() throws -> [PooledSlot] {
        poolLock.lock()
        defer { poolLock.unlock() }

        if let slots = pooledSlots {
            return slots
        }

        let maxN = config.chunk
        let drawInWords = 9 * maxN
        let drawInBytes = drawInWords * 8
        let k1ParamsBytes = 80
        let k1OutBytes = maxN * 64
        let zMinOutBytes = maxN * 8
        let drawCBytes = maxN * 64
        let binOffBytes = (maxN + 1) * 4
        let k2DrawBytes = maxN * 8

        let bytesPerSlot = drawInBytes + k1ParamsBytes + k1OutBytes + zMinOutBytes + drawCBytes + binOffBytes + k2DrawBytes
        let totalPooledBytes = bytesPerSlot * 2

        liveLock.lock()
        if currentLiveBytes + totalPooledBytes > config.memoryCapBytes {
            liveLock.unlock()
            throw FPPV3GPUFailure.vramCapExceeded
        }
        liveLock.unlock()

        func makeSlot() throws -> PooledSlot {
            guard let bIn = device.makeBuffer(length: drawInBytes, options: .storageModeShared),
                  let bParam = device.makeBuffer(length: k1ParamsBytes, options: .storageModeShared),
                  let bOut = device.makeBuffer(length: k1OutBytes, options: .storageModeShared),
                  let bZMin = device.makeBuffer(length: zMinOutBytes, options: .storageModeShared),
                  let bC = device.makeBuffer(length: drawCBytes, options: .storageModeShared),
                  let bOff = device.makeBuffer(length: binOffBytes, options: .storageModeShared),
                  let bDraw = device.makeBuffer(length: k2DrawBytes, options: .storageModeShared) else {
                throw FPPV3GPUFailure.vramCapExceeded
            }
            return PooledSlot(
                drawIn: bIn,
                k1Params: bParam,
                k1Out: bOut,
                zMinOut: bZMin,
                drawC: bC,
                binOff: bOff,
                k2Draw: bDraw,
                fixedBytes: bytesPerSlot
            )
        }

        let s0 = try makeSlot()
        let s1 = try makeSlot()

        liveLock.lock()
        currentLiveBytes += totalPooledBytes
        liveLock.unlock()

        let slots = [s0, s1]
        self.pooledSlots = slots
        return slots
    }

    // MARK: - Batch Transit Likelihood

    public func batchTransitLnLProd(
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
    ) throws -> FPPV3LnLProdResult {
        return try batchPreparedTransitLnLProd(
            fold: fold, ks: ks, pOrbs: pOrbs, incsRad: incsRad, aRsArr: aRsArr,
            eccs: eccs, wsRad: wsRad, u1: u1, u2: u2,
            companionFluxRatio: companionFluxRatio, companionIsHost: companionIsHost
        )
    }

    // For EB, segment 6 carries the shared host preparation's dilA instead
    // of the TP companion flux ratio. The rest of K1/K2 and host assembly
    // are shared with batchTransitLnLProd.
    private func batchPreparedTransitLnLProd(
        fold: FPPV2ProdFold,
        ks: [Double], pOrbs: [Double], incsRad: [Double], aRsArr: [Double],
        eccs: [Double], wsRad: [Double], u1: [Double], u2: [Double],
        companionFluxRatio: [Double], companionIsHost: Bool,
        ebActiveCount: Int? = nil
    ) throws -> FPPV3LnLProdResult {
        let totalN = ks.count
        guard pOrbs.count == totalN,
              incsRad.count == totalN,
              aRsArr.count == totalN,
              eccs.count == totalN,
              wsRad.count == totalN,
              companionFluxRatio.count == totalN else {
            throw FPPV3GPUFailure.unsupportedSwitchSet
        }

        let u1Stride: Int
        if u1.count == totalN {
            u1Stride = 1
        } else if u1.count == 1 {
            u1Stride = 0
        } else {
            throw FPPV3GPUFailure.unsupportedSwitchSet
        }

        let u2Stride: Int
        if u2.count == totalN {
            u2Stride = 1
        } else if u2.count == 1 {
            u2Stride = 0
        } else {
            throw FPPV3GPUFailure.unsupportedSwitchSet
        }

        if totalN == 0 {
            self.lastStats = Stats(
                gpuSecondsK1: 0,
                gpuSecondsPrefix: 0,
                gpuSecondsOrder: 0,
                gpuSecondsLadder: 0,
                hostAssemblySeconds: 0,
                chunks: 0,
                nEvaluated: 0,
                bytesLive: currentLiveBytes
            )
            return FPPV3LnLProdResult(
                lnL: [], status: [], nAff: [], deviceRoots: [],
                discrete: FPPV3Gate.DiscreteRows(), deviceTangency: [], deviceZMin: []
            )
        }

        let cachedFold = try getOrCreateFoldBuffers(fold)
        let slots = try ensurePooledSlots()

        var lnLOut = [Double](repeating: 0, count: totalN)
        var statusOut = [Int](repeating: 0, count: totalN)
        var nAffOut = [Int](repeating: 0, count: totalN)
        var deviceRootsOut = [Double](repeating: 0, count: totalN * 5)
        var discreteOut = FPPV3Gate.DiscreteRows()
        var deviceZMinOut = [Double](repeating: Double(bitPattern: 0x7ff8000000000000), count: totalN)
        var deviceTangencyOut = deviceZMinOut

        var gpuSecondsK1Total: Double = 0
        var gpuSecondsLadderTotal: Double = 0
        var hostAssemblySecondsTotal: Double = 0
        var chunkCount: Int = 0
        var recheckedBinsTotal: Int = 0

        var currentDrawIdx = 0
        var pending: PendingChunk? = nil

        func assemblePending(_ p: PendingChunk) throws {
            let t0 = Date()
            p.cb2.waitUntilCompleted()
            if p.cb2.status != .completed || p.cb2.error != nil {
                print("[metal] CB2 failed: status=\(p.cb2.status.rawValue), error=\(String(describing: p.cb2.error))")
                throw FPPV3GPUFailure.commandFailed
            }
            gpuSecondsLadderTotal += max(0, p.cb2.gpuEndTime - p.cb2.gpuStartTime)

            let slot = slots[p.slotIndex]

            if hooks.nanBinOutAtChunk == p.chunkIndex {
                if let bOut = slot.binOut {
                    let ptr = bOut.contents().bindMemory(to: Float.self, capacity: max(1, p.totalBins * 2))
                    ptr[0] = Float.nan
                }
            }

            if hooks.breakContiguity {
                let k1Words = slot.k1Out.contents().bindMemory(to: UInt32.self, capacity: p.drawCount * 16)
                k1Words[10] = 2 // k1State offset 40 bytes = word 10
            }

            let k1Ptr = slot.k1Out.contents().bindMemory(to: K1OutRow.self, capacity: p.drawCount)
            let zMinPtr = slot.zMinOut.contents().bindMemory(to: UInt64.self, capacity: p.drawCount)
            let binOffPtr = slot.binOff.contents().bindMemory(to: UInt32.self, capacity: p.drawCount + 1)
            let k2DrawPtr = slot.k2Draw.contents().bindMemory(to: K2DrawRow.self, capacity: p.drawCount)
            let binOutPtr = slot.binOut?.contents().bindMemory(to: BinOutRow.self, capacity: max(1, p.totalBins))

            let f = fold.f
            let sig = fold.arrays.sigma

            // Re-decide near-threshold ladder bins flagged by the device in f64 (card M2.5)
            if let bOut = binOutPtr {
                for j in 0..<p.drawCount {
                    let drawIdx = p.drawOffset + j
                    if let ebActiveCount, drawIdx >= ebActiveCount { continue }
                    let row = k1Ptr[j]
                    guard row.k1State == 1 else { continue }
                    let nAff = Int(row.nAff)
                    guard nAff > 0 else { continue }
                    let binBase = Int(binOffPtr[j])

                    var drawParams: FPPV2DrawParams? = nil
                    var ct: [Double]? = nil
                    var epsRef: Double? = nil

                    for i in 0..<nAff {
                        let binIdx = binBase + i
                        if (bOut[binIdx].levelNwin & 0xFF) == 255 {
                            recheckedBinsTotal += 1
                            if drawParams == nil {
                                let r0 = Double(bitPattern: row.root.0)
                                let r1 = Double(bitPattern: row.root.1)
                                let r2 = Double(bitPattern: row.root.2)
                                let r3 = Double(bitPattern: row.root.3)
                                ct = [r0, r1, r2, r3].filter { !$0.isNaN }.sorted()
                                let dilA = Double(row.dilA)
                                let dilB = 1.0 - dilA
                                let pOrb = pOrbs[drawIdx]
                                let aRs = aRsArr[drawIdx]
                                let inc = incsRad[drawIdx]
                                let cosI = FPPV2.cos(inc)
                                let ecc = eccs[drawIdx]
                                let wRad = wsRad[drawIdx]
                                let mT = FPPV2Orbit.midTransitMeanAnomaly(ecc, wRad)
                                let k = ks[drawIdx]
                                let u1Val = (u1Stride == 1 ? u1[drawIdx] : u1[0])
                                let u2Val = (u2Stride == 1 ? u2[drawIdx] : u2[0])
                                drawParams = FPPV2DrawParams(
                                    p: pOrb, aRs: aRs, cosI: cosI, ecc: ecc, wRad: wRad, mT: mT,
                                    k: k, u1: u1Val, u2: u2Val, dilA: dilA, dilB: dilB
                                )
                                epsRef = (2.0 * fold.delta / Double(nAff)).squareRoot()
                            }
                            let b = Int(row.firstAff) + i
                            let orderIdx = Int(k2DrawPtr[j].orderStatus & 0xFF)
                            let (m, _, level) = FPPV2Ladder.ladderBin(
                                b, fold.w, sig[b] * epsRef!, orderIdx, fold.arrays, ct!, drawParams!
                            )
                            bOut[binIdx].D = Float(1.0 - m)
                            bOut[binIdx].levelNwin = UInt32(level) | (bOut[binIdx].levelNwin & 0xFFFFFF00)
                        }
                    }
                }
            }

            for j in 0..<p.drawCount {
                let drawIdx = p.drawOffset + j
                let row = k1Ptr[j]

                let r0 = Double(bitPattern: row.root.0)
                let r1 = Double(bitPattern: row.root.1)
                let r2 = Double(bitPattern: row.root.2)
                let r3 = Double(bitPattern: row.root.3)
                let r4 = Double(bitPattern: row.root.4)
                let baseRootIdx = drawIdx * 5
                deviceRootsOut[baseRootIdx] = r0
                deviceRootsOut[baseRootIdx + 1] = r1
                deviceRootsOut[baseRootIdx + 2] = r2
                deviceRootsOut[baseRootIdx + 3] = r3
                deviceRootsOut[baseRootIdx + 4] = r4

                if let ebActiveCount, drawIdx >= ebActiveCount {
                    lnLOut[drawIdx] = -.infinity
                    // K1 emitted an empty row: the prefix excludes it and
                    // K2 performs no order selection or bin integration.
                    statusOut[drawIdx] = 0
                    nAffOut[drawIdx] = 0
                    discreteOut.push(order: -1, bins: [], levels: [])
                    continue
                }

                let zMin = Double(bitPattern: zMinPtr[j])
                deviceZMinOut[drawIdx] = zMin
                deviceTangencyOut[drawIdx] = FPPV3Gate.tangencyMargin(zMin: zMin, k: ks[drawIdx])
                if row.k1State == 2 {
                    throw FPPV3GPUFailure.nonFinite
                } else if row.k1State == 0 {
                    let emptyLnL = fold.lnnorm - 0.5 * fold.baseChi2
                    guard emptyLnL.isFinite else {
                        throw FPPV3GPUFailure.nonFinite
                    }
                    lnLOut[drawIdx] = emptyLnL
                    statusOut[drawIdx] = 0
                    nAffOut[drawIdx] = 0
                    discreteOut.push(order: -1, bins: [], levels: [])
                } else if row.k1State == 1 {
                    guard row.epsMin.isFinite else {
                        throw FPPV3GPUFailure.nonFinite
                    }
                    let nAff = Int(row.nAff)
                    let firstAff = Int(row.firstAff)
                    let binBase = Int(binOffPtr[j])
                    guard let bOut = binOutPtr else {
                        throw FPPV3GPUFailure.nonFinite
                    }

                    var chi2 = fold.baseChi2
                    var bins = [Int32]()
                    var levels = [Int]()
                    bins.reserveCapacity(nAff)
                    levels.reserveCapacity(nAff)
                    for i in 0..<nAff {
                        let binIdx = binBase + i
                        let D = bOut[binIdx].D
                        guard D.isFinite else {
                            throw FPPV3GPUFailure.nonFinite
                        }
                        let b = firstAff + i
                        bins.append(Int32(b))
                        levels.append(Int(bOut[binIdx].levelNwin & 0xFF))
                        let d = ((f[b] - 1.0) + Double(D)) / sig[b]
                        chi2 += d * d
                        let d0 = (f[b] - 1.0) / sig[b]
                        chi2 -= d0 * d0
                    }

                    let valLnL = fold.lnnorm - 0.5 * chi2
                    guard valLnL.isFinite else {
                        throw FPPV3GPUFailure.nonFinite
                    }
                    lnLOut[drawIdx] = valLnL
                    statusOut[drawIdx] = Int((k2DrawPtr[j].orderStatus >> 8) & 0xFF)
                    nAffOut[drawIdx] = nAff
                    discreteOut.push(order: Int(k2DrawPtr[j].orderStatus & 0xFF), bins: bins, levels: levels)
                } else {
                    throw FPPV3GPUFailure.nonFinite
                }
            }

            hostAssemblySecondsTotal += Date().timeIntervalSince(t0)
        }

        while currentDrawIdx < totalN {
            var currN = min(config.chunk, totalN - currentDrawIdx)
            var currentSlotIdx = 0
            if let p = pending {
                currentSlotIdx = 1 - p.slotIndex
            }

            let slot = slots[currentSlotIdx]

            while true {
                // Check failCommand hook
                if hooks.failCommandAtChunk == chunkCount {
                    throw FPPV3GPUFailure.commandFailed
                }

                // 1. Copy raw f64 arrays with ONE memcpy per array
                let b0Ptr = slot.drawIn.contents().bindMemory(to: UInt64.self, capacity: 9 * currN)
                var segOff = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                segOff[0] = 0
                segOff[1] = segOff[0] + currN
                segOff[2] = segOff[1] + currN
                segOff[3] = segOff[2] + currN
                segOff[4] = segOff[3] + currN
                segOff[5] = segOff[4] + currN
                segOff[6] = segOff[5] + currN
                segOff[7] = segOff[6] + currN
                let u1Count = (u1Stride == 1 ? currN : 1)
                segOff[8] = segOff[7] + u1Count
                let u2Count = (u2Stride == 1 ? currN : 1)
                let activeN = ebActiveCount.map { max(0, min(currN, $0 - currentDrawIdx)) } ?? currN

                ks[currentDrawIdx ..< currentDrawIdx + currN].withUnsafeBytes { src in
                    memcpy(b0Ptr.advanced(by: segOff[0]), src.baseAddress!, currN * 8)
                }
                pOrbs[currentDrawIdx ..< currentDrawIdx + currN].withUnsafeBytes { src in
                    memcpy(b0Ptr.advanced(by: segOff[1]), src.baseAddress!, currN * 8)
                }
                incsRad[currentDrawIdx ..< currentDrawIdx + currN].withUnsafeBytes { src in
                    memcpy(b0Ptr.advanced(by: segOff[2]), src.baseAddress!, currN * 8)
                }
                aRsArr[currentDrawIdx ..< currentDrawIdx + currN].withUnsafeBytes { src in
                    memcpy(b0Ptr.advanced(by: segOff[3]), src.baseAddress!, currN * 8)
                }
                eccs[currentDrawIdx ..< currentDrawIdx + currN].withUnsafeBytes { src in
                    memcpy(b0Ptr.advanced(by: segOff[4]), src.baseAddress!, currN * 8)
                }
                wsRad[currentDrawIdx ..< currentDrawIdx + currN].withUnsafeBytes { src in
                    memcpy(b0Ptr.advanced(by: segOff[5]), src.baseAddress!, currN * 8)
                }
                companionFluxRatio[currentDrawIdx ..< currentDrawIdx + currN].withUnsafeBytes { src in
                    memcpy(b0Ptr.advanced(by: segOff[6]), src.baseAddress!, currN * 8)
                }

                if u1Stride == 1 {
                    u1[currentDrawIdx ..< currentDrawIdx + currN].withUnsafeBytes { src in
                        memcpy(b0Ptr.advanced(by: segOff[7]), src.baseAddress!, currN * 8)
                    }
                } else {
                    u1.withUnsafeBytes { src in
                        memcpy(b0Ptr.advanced(by: segOff[7]), src.baseAddress!, 8)
                    }
                }

                if u2Stride == 1 {
                    u2[currentDrawIdx ..< currentDrawIdx + currN].withUnsafeBytes { src in
                        memcpy(b0Ptr.advanced(by: segOff[8]), src.baseAddress!, currN * 8)
                    }
                } else {
                    u2.withUnsafeBytes { src in
                        memcpy(b0Ptr.advanced(by: segOff[8]), src.baseAddress!, 8)
                    }
                }

                // Record bound arrays for last chunk test aid
                if currentDrawIdx + currN >= totalN {
                    var segs = [[Double]]()
                    for s in 0..<7 {
                        let count = currN
                        let start = segOff[s]
                        var arr = [Double](repeating: 0, count: count)
                        for k in 0..<count {
                            arr[k] = Double(bitPattern: b0Ptr[start + k])
                        }
                        segs.append(arr)
                    }
                    var u1Arr = [Double](repeating: 0, count: u1Count)
                    for k in 0..<u1Count {
                        u1Arr[k] = Double(bitPattern: b0Ptr[segOff[7] + k])
                    }
                    segs.append(u1Arr)

                    var u2Arr = [Double](repeating: 0, count: u2Count)
                    for k in 0..<u2Count {
                        u2Arr[k] = Double(bitPattern: b0Ptr[segOff[8] + k])
                    }
                    segs.append(u2Arr)
                    self.lastBoundDrawArraysSegments = segs
                }

                // 2. Write K1Params
                var k1Params = K1ParamsLayout(
                    n: UInt32(currN),
                    nb: UInt32(fold.nb),
                    companionIsHost: companionIsHost ? 1 : 0,
                    u1Stride: UInt32(u1Stride),
                    u2Stride: UInt32(u2Stride),
                    segOff: (UInt32(segOff[0]), UInt32(segOff[1]), UInt32(segOff[2]), UInt32(segOff[3]),
                             UInt32(segOff[4]), UInt32(segOff[5]), UInt32(segOff[6]), UInt32(segOff[7]),
                             UInt32(segOff[8])),
                    pad0: ebActiveCount == nil ? 0 : 1,
                    pad1: ebActiveCount == nil ? 0 : UInt32(activeN),
                    foldW: fold.w.bitPattern,
                    foldDelta: fold.delta.bitPattern
                )
                memcpy(slot.k1Params.contents(), &k1Params, 80)

                // 3. Command Buffer 1: k1_contacts + k_prefix_naff
                guard let cb1 = queue.makeCommandBuffer(),
                      let enc1 = cb1.makeComputeCommandEncoder() else {
                    throw FPPV3GPUFailure.commandFailed
                }

                enc1.setComputePipelineState(pK1)
                enc1.setBuffer(slot.drawIn, offset: 0, index: 0)
                enc1.setBuffer(slot.k1Params, offset: 0, index: 1)
                enc1.setBuffer(cachedFold.foldBuf, offset: 0, index: 2)
                var fp = cachedFold.foldParams
                enc1.setBytes(&fp, length: 64, index: 3)
                enc1.setBuffer(slot.k1Out, offset: 0, index: 4)
                enc1.setBuffer(slot.drawC, offset: 0, index: 5)
                enc1.setBuffer(slot.zMinOut, offset: 0, index: 6)
                let tgK1 = min(pK1.maxTotalThreadsPerThreadgroup, 64)
                enc1.dispatchThreads(
                    MTLSize(width: currN, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: tgK1, height: 1, depth: 1)
                )
                enc1.endEncoding()

                guard let encPrefix = cb1.makeComputeCommandEncoder() else {
                    throw FPPV3GPUFailure.commandFailed
                }
                encPrefix.setComputePipelineState(pPrefix)
                encPrefix.setBuffer(slot.k1Out, offset: 0, index: 0)
                encPrefix.setBuffer(slot.k1Params, offset: 0, index: 1)
                encPrefix.setBuffer(slot.binOff, offset: 0, index: 2)
                let tgPrefix = min(1024, pPrefix.maxTotalThreadsPerThreadgroup)
                encPrefix.dispatchThreadgroups(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: tgPrefix, height: 1, depth: 1)
                )
                encPrefix.endEncoding()

                cb1.commit()
                cb1.waitUntilCompleted()

                if cb1.status != .completed || cb1.error != nil {
                    print("[metal] CB1 failed: status=\(cb1.status.rawValue), error=\(String(describing: cb1.error))")
                    throw FPPV3GPUFailure.commandFailed
                }
                gpuSecondsK1Total += max(0, cb1.gpuEndTime - cb1.gpuStartTime)

                let binOffPtr = slot.binOff.contents().bindMemory(to: UInt32.self, capacity: currN + 1)
                let total = Int(binOffPtr[currN])

                if total * 8 > config.binOutCapBytes {
                    if currN <= 256 {
                        throw FPPV3GPUFailure.vramCapExceeded
                    }
                    currN = currN / 2
                    continue
                }

                // Allocate/grow binOut if needed
                let neededBinOutBytes = max(total * 8, 256)
                if slot.binOut == nil || slot.binOutAllocatedBytes < neededBinOutBytes {
                    let diff = neededBinOutBytes - slot.binOutAllocatedBytes
                    liveLock.lock()
                    if currentLiveBytes + diff > config.memoryCapBytes {
                        liveLock.unlock()
                        throw FPPV3GPUFailure.vramCapExceeded
                    }
                    liveLock.unlock()

                    guard let newBinOut = device.makeBuffer(length: neededBinOutBytes, options: .storageModeShared) else {
                        throw FPPV3GPUFailure.vramCapExceeded
                    }
                    liveLock.lock()
                    currentLiveBytes += diff
                    liveLock.unlock()
                    slot.binOut = newBinOut
                    slot.binOutAllocatedBytes = neededBinOutBytes
                }

                // CB1 is complete. K2 sees only the active prefix of this
                // chunk; trailing recorder-only rows have nAff = 0. The
                // same slot is not reused until its CB2 has completed.
                k1Params.n = UInt32(activeN)
                memcpy(slot.k1Params.contents(), &k1Params, 80)

                // 4. Command Buffer 2: k2_order + k2_ladder
                guard let cb2 = queue.makeCommandBuffer() else {
                    throw FPPV3GPUFailure.commandFailed
                }

                if activeN > 0 {
                    guard let encOrder = cb2.makeComputeCommandEncoder() else {
                        throw FPPV3GPUFailure.commandFailed
                    }
                    encOrder.setComputePipelineState(pOrder)
                    encOrder.setBuffer(slot.k1Out, offset: 0, index: 0)
                    encOrder.setBuffer(slot.k1Params, offset: 0, index: 1)
                    encOrder.setBuffer(slot.drawC, offset: 0, index: 2)
                    encOrder.setBuffer(glx, offset: 0, index: 3)
                    encOrder.setBuffer(glw, offset: 0, index: 4)
                    encOrder.setBuffer(orders, offset: 0, index: 5)
                    encOrder.setBuffer(slot.k2Draw, offset: 0, index: 6)
                    let tgOrder = min(pOrder.maxTotalThreadsPerThreadgroup, 64)
                    encOrder.dispatchThreads(
                        MTLSize(width: activeN * 32, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: tgOrder, height: 1, depth: 1)
                    )
                    encOrder.endEncoding()
                }

                if total > 0 {
                    guard let encLadder = cb2.makeComputeCommandEncoder() else {
                        throw FPPV3GPUFailure.commandFailed
                    }
                    encLadder.setComputePipelineState(pLadder)
                    encLadder.setBuffer(slot.k1Out, offset: 0, index: 0)
                    encLadder.setBuffer(slot.k1Params, offset: 0, index: 1)
                    encLadder.setBuffer(slot.drawC, offset: 0, index: 2)
                    encLadder.setBuffer(cachedFold.foldBuf, offset: 0, index: 3)
                    var fpLadder = cachedFold.foldParams
                    encLadder.setBytes(&fpLadder, length: 64, index: 4)
                    encLadder.setBuffer(glx, offset: 0, index: 5)
                    encLadder.setBuffer(glw, offset: 0, index: 6)
                    encLadder.setBuffer(orders, offset: 0, index: 7)
                    encLadder.setBuffer(slot.k2Draw, offset: 0, index: 8)
                    encLadder.setBuffer(slot.binOff, offset: 0, index: 9)
                    encLadder.setBuffer(slot.binOut!, offset: 0, index: 10)
                    let tgLadder = min(pLadder.maxTotalThreadsPerThreadgroup, 64)
                    encLadder.dispatchThreads(
                        MTLSize(width: total * 32, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: tgLadder, height: 1, depth: 1)
                    )
                    encLadder.endEncoding()
                }

                cb2.commit() // Commit WITHOUT waiting

                // Overlap: wait for previous chunk and assemble it
                if let prev = pending {
                    try assemblePending(prev)
                    pending = nil
                }

                pending = PendingChunk(
                    slotIndex: currentSlotIdx,
                    drawOffset: currentDrawIdx,
                    drawCount: currN,
                    cb2: cb2,
                    totalBins: total,
                    chunkIndex: chunkCount
                )

                chunkCount += 1
                currentDrawIdx += currN
                break
            }
        }

        // Wait for and assemble the last chunk
        if let last = pending {
            try assemblePending(last)
            pending = nil
        }

        self.lastStats = Stats(
            gpuSecondsK1: gpuSecondsK1Total,
            gpuSecondsPrefix: 0,
            gpuSecondsOrder: 0,
            gpuSecondsLadder: gpuSecondsLadderTotal,
            hostAssemblySeconds: hostAssemblySecondsTotal,
            chunks: chunkCount,
            nEvaluated: ebActiveCount ?? totalN,
            bytesLive: currentLiveBytes,
            recheckedBins: recheckedBinsTotal
        )

        return FPPV3LnLProdResult(
            lnL: lnLOut,
            status: statusOut,
            nAff: nAffOut,
            deviceRoots: deviceRootsOut,
            discrete: discreteOut,
            deviceTangency: deviceTangencyOut,
            deviceZMin: deviceZMinOut
        )
    }
}
#endif
