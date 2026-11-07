import Foundation

public struct IterativeDetrendResult: Equatable, Sendable {
    public let flux: [Double]
    public let firstPassFlux: [Double]
    public let quickBLS: BLSResult
    public let transitMask: [Bool]
    public let wasRefined: Bool
}

public enum IterativeDetrender {
    public static let quickParameters = BLSParameters(
        periodMinimum: 0.5, periodMaximum: 50, periodCount: 2_000,
        binCount: 100, durationStepCount: 5,
        durationMinimumFraction: 0.01, durationMaximumFraction: 0.15
    )

    public static func flatten(
        time: [Double], flux: [Double],
        snrThreshold: Double = 7,
        maxConcurrency: Int = ProcessInfo.processInfo.activeProcessorCount,
        excluded: [Bool]? = nil
    ) throws -> IterativeDetrendResult {
        // `excluded` (gap_fill_detrend_only): gap-fill points stay in both
        // wotan passes but are kept out of the quick BLS and never masked.
        guard time.count == flux.count else { throw DetrendError.lengthMismatch }
        if let excluded, excluded.count != time.count { throw DetrendError.maskLengthMismatch }
        let first = try BiweightDetrender.flatten(time: time, flux: flux)
        var finiteTime: [Double] = []
        var finiteFlux: [Double] = []
        for index in time.indices
        where time[index].isFinite && first[index].isFinite && !(excluded?[index] ?? false) {
            finiteTime.append(time[index]); finiteFlux.append(first[index])
        }
        let quick = try BLSCPUOptimized.search(
            time: finiteTime, flux: finiteFlux, parameters: quickParameters,
            maxConcurrency: maxConcurrency
        )
        guard quick.snr >= snrThreshold else {
            return IterativeDetrendResult(
                flux: first, firstPassFlux: first, quickBLS: quick,
                transitMask: Array(repeating: false, count: time.count), wasRefined: false
            )
        }
        let durationFraction = quick.duration / quick.period
        let mask = time.indices.map { index -> Bool in
            var phase = (time[index] - quick.t0).truncatingRemainder(dividingBy: quick.period) / quick.period
            if phase < 0 { phase += 1 }
            return (phase < durationFraction || phase > 1 - durationFraction) && !(excluded?[index] ?? false)
        }
        let refined = try BiweightDetrender.flatten(time: time, flux: flux, transitMask: mask)
        return IterativeDetrendResult(
            flux: refined, firstPassFlux: first, quickBLS: quick,
            transitMask: mask, wasRefined: true
        )
    }
}
