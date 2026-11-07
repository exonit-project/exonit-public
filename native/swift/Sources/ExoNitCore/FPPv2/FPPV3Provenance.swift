import Foundation
#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif

/// Platform and libm provenance recorded with every computed FPP result
/// (plan decision (t) point 2, "[Amended 2026-10-25]": mandatory; glibc
/// x86-64 Linux is the canonical platform). Task 33 contract (shared with the
/// Rust port `exonit_core::fpp::provenance` and, for Python, the coordinator):
///
/// * `fpp_platform` = `<os>-<arch>`, os in {`linux`, `darwin`, `windows`},
///   arch in {`x86_64`, `arm64`}; any other component is the literal `unknown`;
/// * `fpp_libm` = `glibc-<gnu_get_libc_version()>` on glibc Linux, `musl` on musl,
///   `apple-libm-macos-<kern.osproductversion>-<kern.osversion>` on macOS,
///   `msvc-ucrt` on Windows; anything that cannot be determined (a missing
///   symbol, ONE failed sysctl, a malformed version) makes the WHOLE value `unknown`;
/// * `fpp_build_profile` = `debug` or `release` (assert configuration).
public enum FPPV3Provenance {
    public static let platform: String = {
        #if os(Linux)
        let os = "linux"
        #elseif os(macOS)
        let os = "darwin"
        #elseif os(Windows)
        let os = "windows"
        #else
        let os = "unknown"
        #endif
        #if arch(x86_64)
        let arch = "x86_64"
        #elseif arch(arm64)
        let arch = "arm64"
        #else
        let arch = "unknown"
        #endif
        return "\(os)-\(arch)"
    }()

    /// `glibc-<version>` when the version is dot-separated digits (at least two components), else `unknown`.
    public static func glibcLibmString(_ version: String?) -> String {
        guard let version else { return "unknown" }
        let parts = version.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count >= 2, parts.allSatisfy({ !$0.isEmpty && $0.allSatisfy(\.isASCII) && $0.allSatisfy(\.isNumber) })
        else { return "unknown" }
        return "glibc-\(version)"
    }

    /// `apple-libm-macos-<product>-<build>`; one missing, empty, non-ASCII or dash-containing component (or a value
    /// over 256 characters) gives `unknown` for the whole value.
    public static func macOSLibmString(product: String?, build: String?) -> String {
        func ok(_ s: String) -> Bool {
            !s.isEmpty && s.allSatisfy(\.isASCII) && !s.contains("-") && !s.unicodeScalars.contains { $0.value < 0x20 }
        }
        guard let product, let build, ok(product), ok(build) else { return "unknown" }
        let value = "apple-libm-macos-\(product)-\(build)"
        return value.count <= 256 ? value : "unknown"
    }

    public static let libm: String = {
        #if os(Linux)
        // glibc exports gnu_get_libc_version(); resolve it at run time so the
        // version is that of the library the process actually loaded.
        typealias VersionFunction = @convention(c) () -> UnsafePointer<CChar>
        guard let symbol = dlsym(UnsafeMutableRawPointer(bitPattern: 0), "gnu_get_libc_version") else {
            #if canImport(Musl)
            return "musl"
            #else
            return "unknown"
            #endif
        }
        let version = unsafeBitCast(symbol, to: VersionFunction.self)
        return glibcLibmString(String(validatingUTF8: version()))
        #elseif os(macOS)
        func sysctlString(_ name: String) -> String? {
            var size = 0
            guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
            var buffer = [CChar](repeating: 0, count: size)
            guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
            return String(validatingUTF8: buffer)
        }
        return macOSLibmString(product: sysctlString("kern.osproductversion"), build: sysctlString("kern.osversion"))
        #elseif os(Windows)
        return "msvc-ucrt"
        #else
        return "unknown"
        #endif
    }()

    public static let buildProfile: String = _isDebugAssertConfiguration() ? "debug" : "release"

    // MARK: exact matchers of the task-33 contract regexes (no regex engine needed)

    /// `^(linux|darwin|windows|unknown)-(x86_64|arm64|unknown)$`
    public static func matchesPlatformContract(_ s: String) -> Bool {
        let p = s.split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false).map(String.init)
        guard p.count == 2 else { return false }
        return ["linux", "darwin", "windows", "unknown"].contains(p[0]) && ["x86_64", "arm64", "unknown"].contains(p[1])
    }

    /// `^(glibc-[0-9]+(\.[0-9]+)+|musl|apple-libm-macos-[^-]+-[^-]+|msvc-ucrt|mingw-msvcrt|unknown)$`
    public static func matchesLibmContract(_ s: String) -> Bool {
        if ["musl", "msvc-ucrt", "mingw-msvcrt", "unknown"].contains(s) { return true }
        if s.hasPrefix("glibc-") {
            let parts = s.dropFirst(6).split(separator: ".", omittingEmptySubsequences: false)
            return parts.count >= 2 && parts.allSatisfy { !$0.isEmpty && $0.allSatisfy { ("0"..."9").contains($0) } }
        }
        if s.hasPrefix("apple-libm-macos-") {
            let parts = s.dropFirst(17).split(separator: "-", omittingEmptySubsequences: false)
            return parts.count == 2 && parts.allSatisfy { !$0.isEmpty }
        }
        return false
    }

    /// `^(debug|release)$`
    public static func matchesProfileContract(_ s: String) -> Bool { s == "debug" || s == "release" }
}
