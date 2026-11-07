import ExoNitCore
import Foundation

// Sparse CNN-FFI view -> no CNN score (nil), not a failed task. Port of python
// worker/cnn_ffi.py (ViewError -> None since 2026-10-15, exonit
// docs/reviews/2026-10-15-cnn-exact-zero-scores.md fix 2). The native primary
// task runs no FPP in-task, and the server backfill excludes such rows, so no
// FPP follows here (open item); this test only covers the view/score half.

func testCNNSparseViewIsNil() async throws {
    // 4 points: fewer than the 5 the view window needs.
    let sparseTime = [0.0, 0.01, 0.02, 0.03]
    let sparseFlux = [1.0, 1.0, 1.0, 1.0]
    try expectThrows("sparse view must still throw from makeCNNViews") {
        _ = try FFIViewGenerator.makeCNNViews(
            time: sparseTime, flux: sparseFlux, period: 2, t0: 1, duration: 0.1
        )
    }
    let none = try FFIViewGenerator.makeCNNViewsIfDense(
        time: sparseTime, flux: sparseFlux, period: 2, t0: 1, duration: 0.1
    )
    try expect(none == nil, "sparse view -> nil, not an error")

    let count = 3_000
    let time = (0..<count).map { 27.0 * Double($0) / Double(count - 1) }
    let flux = time.enumerated().map { index, t -> Double in
        let phase = (t - 1).truncatingRemainder(dividingBy: 2)
        let dip = (phase >= 0 && phase < 0.05) ? 0.02 : 0
        return 1 - dip + 1e-4 * (Double(index * 7_919 % 1_000) / 1_000 - 0.5)
    }
    let direct = try FFIViewGenerator.makeCNNViews(
        time: time, flux: flux, period: 2, t0: 1, duration: 0.1
    )
    let dense = try FFIViewGenerator.makeCNNViewsIfDense(
        time: time, flux: flux, period: 2, t0: 1, duration: 0.1
    )
    try expect(dense == direct, "dense view must be returned unchanged")

    // Caller bugs are not "sparse" and still throw, as in Python.
    try expectThrows("invalid ephemeris must still throw") {
        _ = try FFIViewGenerator.makeCNNViewsIfDense(
            time: time, flux: flux, period: 0, t0: 1, duration: 0.1
        )
    }
    try expect(FFIViewError.allBinsEmpty.isSparse, "allBinsEmpty is sparse")
    try expect(!FFIViewError.mismatchedSeries.isSparse, "mismatchedSeries is not sparse")

    // Gate-policy parity with python should_run_triceratops (nil -> true).
    // Note: the native primary task never calls FPPPolicy.shouldRun, so this
    // does NOT mean FPP runs for a sparse candidate on this worker.
    try expect(FPPPolicy.shouldRun(vettingPassed: true, snr: 12, cnnScore: nil),
               "FPPPolicy.shouldRun(nil) must match python (true)")
}
