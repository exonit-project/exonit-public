import Foundation

// Swift port, part 1, of the APPROVED pieces of the fixed FPP model (fix plan
// docs/superpowers/plans/2026-10-15-fpp-native-fix-plan.md, step 9; mirror of the Rust
// crates/exonit-core/src/fpp/v2). Reference: worker/fpp_native.py (CPU / numba f64).
//
// Arithmetic order is the Python order, literally. numba lowers `x ** 2` / `x ** 3`
// (constant integer exponents) to `x * x` / `x * (x * x)`; `x ** 1.5` / `x ** -1.5` call
// libm `pow` (here `FPPV2.pow`). No FMA: Swift does not contract `a * b + c` by default
// and no `.addingProduct` is used anywhere in this port. The ACTUAL release-build hazard is the
// optimiser fusing sin(x)/cos(x) into `___sincos_stret`: every sin/cos/atan2/acos/log/pow call
// in this port goes through the `@inline(never)` shims below (checked by
// dev/check-fppv2-reproducible.sh).
//
// Platform contract (plan decision (t)): bit-exact against the reference holds on glibc
// x86-64. macOS libm differs in the last bit for sin / cos / atan2 / acos / pow / log, so
// on macOS the golden test reports the bit-identical count and the max ULP difference
// instead of failing on a last-bit difference (a PROPOSED tolerance tier, not yet ratified).

/// Namespace + shared helpers of the v2 port.
public enum FPPV2 {
    /// Actual libm `pow` call, never folded or rewritten to `x * x` by the optimiser.
    @inline(never)
    public static func pow(_ x: Double, _ y: Double) -> Double {
        Foundation.pow(x, y)
    }

    // Transcendentals go through non-inlinable shims so the optimiser can never pair a `sin(x)`
    // with a `cos(x)` of the same argument into one Apple `___sincos_stret` call (whose last bit
    // can differ from separate calls) or otherwise rewrite them: debug and release builds must
    // evaluate exactly the same libm calls.
    @inline(never) public static func sin(_ x: Double) -> Double { Foundation.sin(x) }
    @inline(never) public static func cos(_ x: Double) -> Double { Foundation.cos(x) }
    @inline(never) public static func atan2(_ y: Double, _ x: Double) -> Double { Foundation.atan2(y, x) }
    @inline(never) public static func acos(_ x: Double) -> Double { Foundation.acos(x) }
    @inline(never) public static func log(_ x: Double) -> Double { Foundation.log(x) }
    @inline(never) public static func log10(_ x: Double) -> Double { Foundation.log10(x) }
    @inline(never) public static func exp(_ x: Double) -> Double { Foundation.exp(x) }
    @inline(never) public static func erf(_ x: Double) -> Double { Foundation.erf(x) }
    /// libm `hypot` (numpy `np.hypot` on scalars), step-7 stage-agreement test.
    @inline(never) public static func hypot(_ x: Double, _ y: Double) -> Double { Foundation.hypot(x, y) }

    /// CPython / numba float `%` (fmod plus the sign correction; result sign follows the divisor).
    public static func pyMod(_ x: Double, _ y: Double) -> Double {
        var m = fmod(x, y)
        if m != 0.0 {
            if (y < 0.0) != (m < 0.0) { m += y }
        } else {
            m = copysign(0.0, y)
        }
        return m
    }
}

/// N1 constants block of the fixed FPP model: the astropy 7.2.2 cgs values used by upstream
/// TRICERATOPS, as EXACT f64 bit patterns (the Python reference writes them with `float.fromhex`).
public enum FPPV2Consts {
    /// `G_cgs`, cm^3 g^-1 s^-2 (0x1.1ea8b45edf98bp-24).
    public static let gCGS = Double(bitPattern: 0x3e71ea8b45edf98b)
    /// `Msun`, g (0x1.88250b3b66e99p+110).
    public static let msun = Double(bitPattern: 0x46d88250b3b66e99)
    /// `Rsun`, cm (0x1.032b1f4800000p+36).
    public static let rsun = Double(bitPattern: 0x423032b1f4800000)
    /// `Rearth`, cm (0x1.3021aa8000000p+29).
    public static let rearth = Double(bitPattern: 0x41c3021aa8000000)
    /// `au`, cm (0x1.b363196a46000p+43).
    public static let au = Double(bitPattern: 0x42ab363196a46000)
    public static let pi = Double.pi
    /// `ln2pi = np.log(2 * pi)` as the reference computed it, pinned as bits.
    public static let ln2pi = Double(bitPattern: 0x3ffd67f1c864beb4)

    public static let fppModelVersion = 2

    /// Gauss-Legendre exposure integrator: smooth / oracle order indices.
    public static let glSmoothIdx = 2
    public static let glOracleIdx = 10

    /// Quantile-node ladder constants.
    public static let ladderDeltaNat = 0.02
    public static let ladderKQuant = 256
    public static let ladderMaxK = 128
    public static let ladderExactLevel = 7
    public static let ladderMinPoints = 8
    public static let ladderAcceptMaxLevel = 5

    /// Fold constants.
    public static let foldNBins = 1000
    public static let foldKQuant = 256

    /// Parity-mode exposure supersampling.
    public static let parityExptimeD = 0.00139
    public static let parityNsub = 20
}
