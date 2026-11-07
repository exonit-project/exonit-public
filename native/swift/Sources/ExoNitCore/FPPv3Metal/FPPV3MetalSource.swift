// NOT COMPILED BY THE AUTHOR (2026-11-05). <host> has no Swift toolchain or Metal device.

import Foundation

/// Assembles the complete MSL source text for the V3 Metal likelihood pipeline and defines kernel version tags.
public enum FPPV3MetalSource {
    public static let kernelTag = "fpp-v3-metal-p6.0"

    /// DF64 + DF64Trig + K1Scan + K1 + K2Flux + K2Window + K2Ladder, in this order, joined with "\n".
    public static let text: String = [
        FPPV3MetalDF64.text,
        FPPV3MetalDF64Trig.text,
        FPPV3MetalK1Scan.text,
        FPPV3MetalK1.text,
        FPPV3MetalK2Flux.text,
        FPPV3MetalK2Window.text,
        FPPV3MetalK2Ladder.text,
    ].joined(separator: "\n")

    /// SHA-256 hex string of the complete MSL source text.
    public static var textSHA256: String {
        SHA256Swift.hex(Array(text.utf8))
    }

    /// Pre-registered compilation flags for Metal library compilation.
    public static let compileFlags = "mathMode=safe;mathFloatingPointFunctions=precise;fp-contract=off;language=default"

    /// SHA-256 hex string of the compilation flags.
    public static var compileFlagsSHA256: String {
        SHA256Swift.hex(Array(compileFlags.utf8))
    }

    /// Canonical GPU kernel version identifier: "\(kernelTag):\(textSHA256):\(compileFlagsSHA256)".
    public static var gpuKernelVersion: String {
        "\(kernelTag):\(textSHA256):\(compileFlagsSHA256)"
    }
}
