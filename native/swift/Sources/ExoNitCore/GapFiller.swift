import Foundation

public struct GapFillSeries: Equatable, Sendable {
    public var time: [Double]
    public var flux: [Double]
    public var cadence: [Int64]?

    public init(time: [Double], flux: [Double], cadence: [Int64]?) {
        self.time = time
        self.flux = flux
        self.cadence = cadence
    }
}

public protocol GaussianNoiseSource {
    mutating func normal(mean: Double, standardDeviation: Double) -> Double
}

/// numpy-legacy `RandomState(seed).normal(mean, sd)` noise source
/// (fill-gaps-determinism, 2026-10-15). lightkurve's fill_gaps() draws from
/// the unseeded global numpy RNG, which made repeat runs non-reproducible
/// (IR pilot: 20/100 blank hosts changed candidate lists). Python now seeds a
/// private RandomState from `GapFillSeed.seed(ticID:sectors:)`
/// (worker/gap_fill.py); this source reproduces its unit-normal stream bit for
/// bit. The amplitude differs: Python uses lightkurve's CDPP as sigma, this
/// port the population std of the flux. There is no unseeded source any more.
public struct SeededGaussianNoiseSource: GaussianNoiseSource {
    private var rng: FPPRNG
    public init(seed: UInt32) { rng = FPPRNG(seed: seed) }

    public mutating func normal(mean: Double, standardDeviation: Double) -> Double {
        mean + standardDeviation * rng.gauss()
    }
}

public enum GapFillSeed {
    /// `"{tic}:{s1},{s2},..."` with sectors sorted and de-duplicated.
    public static func key(ticID: String, sectors: [Int]) -> String {
        "\(ticID):" + Set(sectors).sorted().map(String.init).joined(separator: ",")
    }

    /// FNV-1a 64 of the key, truncated to UInt32 (== worker/gap_fill.py).
    public static func seed(ticID: String, sectors: [Int]) -> UInt32 {
        fnv1a32(Array(key(ticID: ticID, sectors: sectors).utf8))
    }

    static func fnv1a32(_ bytes: [UInt8]) -> UInt32 {
        var h: UInt64 = 0xcbf29ce484222325
        for byte in bytes { h ^= UInt64(byte); h = h &* 0x00000100000001b3 }
        return UInt32(truncatingIfNeeded: h)
    }

    /// Fallback for callers that know no TIC: FNV-1a over curve count and
    /// each curve's length and first/last time bits (still deterministic).
    public static func dataSeed(_ curves: [LightCurveData]) -> UInt32 {
        var bytes: [UInt8] = []
        func put(_ v: UInt64) { for i in 0..<8 { bytes.append(UInt8(truncatingIfNeeded: v >> (8 * UInt64(i)))) } }
        put(UInt64(curves.count))
        for c in curves {
            put(UInt64(c.time.count))
            put((c.time.first ?? 0).bitPattern)
            put((c.time.last ?? 0).bitPattern)
        }
        return fnv1a32(bytes)
    }
}

public enum GapFillError: Error, Equatable, Sendable {
    case lengthMismatch
    case nonMonotonicTime
    case nonMonotonicCadence
}

public enum GapFiller {
    public static func fillIfAllowed<R: GaussianNoiseSource>(
        _ input: GapFillSeries,
        maximumGlobalGapDays: Double = 5,
        spacingTolerance: Double = 1.2,
        standardDeviation suppliedDeviation: Double? = nil,
        noise: inout R
    ) throws -> GapFillSeries {
        guard input.time.count == input.flux.count,
              input.cadence == nil || input.cadence?.count == input.time.count else {
            throw GapFillError.lengthMismatch
        }
        guard input.time.count >= 2 else { return input }
        let differences = zip(input.time.dropFirst(), input.time).map(-)
        // Callers are expected to sort by time before reaching here (see
        // LightCurvePreprocessor.process), matching the side effect of
        // lightkurve's lc.fill_gaps(): when cadenceno is present it rebuilds
        // the time grid via `ncad[np.argsort(ncad)]`, which re-orders the
        // data regardless of how LightCurveCollection.stitch() concatenated
        // the sectors. What Python never does is *reject* the input for
        // being unsorted or containing an exact tie — its non-cadence path
        // (`while (t - prevtime) > 1.2 * dt`) silently tolerates a zero
        // diff (duplicate timestamp) by just appending it, no filler
        // inserted, no exception. Only a genuinely descending pair (which
        // should not exist post-sort, and would indicate a real caller bug)
        // is treated as an error here.
        guard differences.allSatisfy({ $0 >= 0 }) else { throw GapFillError.nonMonotonicTime }
        guard (differences.max() ?? .infinity) < maximumGlobalGapDays else { return input }

        let finiteFlux = input.flux.filter(\.isFinite)
        let mean = finiteFlux.isEmpty ? .nan : finiteFlux.reduce(0, +) / Double(finiteFlux.count)
        let deviation: Double
        if let suppliedDeviation {
            deviation = suppliedDeviation
        } else if finiteFlux.isEmpty {
            deviation = .nan
        } else {
            let variance = finiteFlux.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(finiteFlux.count)
            deviation = sqrt(variance)
        }

        if let cadence = input.cadence {
            return try fillCadenceGrid(input, cadence: cadence, spacing: median(differences), mean: mean, deviation: deviation, noise: &noise)
        }
        return fillTimeGrid(input, spacing: median(differences), tolerance: spacingTolerance, mean: mean, deviation: deviation, noise: &noise)
    }

    private static func fillCadenceGrid<R: GaussianNoiseSource>(
        _ input: GapFillSeries, cadence: [Int64], spacing: Double,
        mean: Double, deviation: Double, noise: inout R
    ) throws -> GapFillSeries {
        let cadenceDiff = zip(cadence.dropFirst(), cadence).map(-)
        guard cadenceDiff.allSatisfy({ $0 > 0 }) else { throw GapFillError.nonMonotonicCadence }
        let offsets = zip(input.time, cadence).map { $0 - spacing * Double($1) }
        var rowByCadence: [Int64: Int] = [:]
        for (index, value) in cadence.enumerated() { rowByCadence[value] = index }
        var output = GapFillSeries(time: [], flux: [], cadence: [])
        let count = Int(cadence.last! - cadence.first! + 1)
        output.time.reserveCapacity(count); output.flux.reserveCapacity(count); output.cadence?.reserveCapacity(count)
        var lower = 0
        for value in cadence.first!...cadence.last! {
            output.cadence?.append(value)
            if let row = rowByCadence[value] {
                output.time.append(input.time[row]); output.flux.append(input.flux[row]); lower = row
            } else {
                while lower + 1 < cadence.count, cadence[lower + 1] < value { lower += 1 }
                let upper = lower + 1
                let fraction = Double(value - cadence[lower]) / Double(cadence[upper] - cadence[lower])
                let offset = offsets[lower] + fraction * (offsets[upper] - offsets[lower])
                output.time.append(offset + spacing * Double(value))
                output.flux.append(noise.normal(mean: mean, standardDeviation: deviation))
            }
        }
        return output
    }

    private static func fillTimeGrid<R: GaussianNoiseSource>(
        _ input: GapFillSeries, spacing: Double, tolerance: Double,
        mean: Double, deviation: Double, noise: inout R
    ) -> GapFillSeries {
        var output = GapFillSeries(time: [input.time[0]], flux: [input.flux[0]], cadence: nil)
        for index in 1..<input.time.count {
            while input.time[index] - output.time.last! > tolerance * spacing {
                output.time.append(output.time.last! + spacing)
                output.flux.append(noise.normal(mean: mean, standardDeviation: deviation))
            }
            output.time.append(input.time[index]); output.flux.append(input.flux[index])
        }
        return output
    }

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
    }
}
