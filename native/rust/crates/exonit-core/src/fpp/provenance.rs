//! Platform and libm provenance recorded with every computed FPP result
//! (plan decision (t) point 2, "[Amended 2026-10-25]": the platform and libm
//! provenance of every result is mandatory; glibc x86-64 Linux is the
//! canonical platform).
//!
//! The strings are shared with the Swift port (`FPPV3Provenance.swift`) and
//! proposed for the Python reference, so the server can group rows by them:
//!
//! * `fpp_platform` = `<os>-<arch>`, os in {`linux`, `darwin`, `windows`},
//!   arch in {`x86_64`, `arm64`} (Rust's `aarch64` is spelled `arm64`).
//! * `fpp_libm` = the C math library that served `sin/cos/atan2/acos/pow/
//!   exp/log/erf`: `glibc-<gnu_get_libc_version()>` on glibc Linux,
//!   `musl` on musl Linux, `apple-libm-macos-<kern.osproductversion>-<kern.osversion>`
//!   on macOS (libsystem_m ships with, and is versioned by, the OS build),
//!   `msvc-ucrt` / `mingw-msvcrt` on Windows (the CRT version is not read).
//! * `fpp_build_profile` = `debug` or `release` (`cfg!(debug_assertions)`).

use std::sync::OnceLock;

/// `<os>-<arch>` of the running binary, e.g. `linux-x86_64`, `darwin-arm64`. Task 33 contract: os in
/// {`linux`, `darwin`, `windows`}, arch in {`x86_64`, `arm64`}; any other component is the literal `unknown`.
pub fn platform() -> &'static str {
    static P: OnceLock<String> = OnceLock::new();
    P.get_or_init(|| platform_string(std::env::consts::OS, std::env::consts::ARCH))
}

/// The contract mapping of Rust's `std::env::consts::{OS, ARCH}` (pure, for tests).
pub fn platform_string(os: &str, arch: &str) -> String {
    let os = match os {
        "linux" => "linux",
        "macos" => "darwin",
        "windows" => "windows",
        _ => "unknown",
    };
    let arch = match arch {
        "x86_64" => "x86_64",
        "aarch64" => "arm64",
        _ => "unknown",
    };
    format!("{os}-{arch}")
}

/// Identity and version of the C math library, e.g. `glibc-2.43`.
pub fn libm() -> &'static str {
    static L: OnceLock<String> = OnceLock::new();
    L.get_or_init(detect_libm)
}

/// `debug` or `release`.
pub const fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn detect_libm() -> String {
    extern "C" {
        fn gnu_get_libc_version() -> *const std::os::raw::c_char;
    }
    // SAFETY: glibc returns a pointer to a static NUL-terminated string.
    let version = unsafe { std::ffi::CStr::from_ptr(gnu_get_libc_version()) };
    glibc_libm_string(version.to_str().ok())
}

/// Task 33 contract for glibc: `glibc-<gnu_get_libc_version()>`; a version that is not dot-separated digits
/// (at least two components) cannot be determined and gives `unknown`.
pub fn glibc_libm_string(version: Option<&str>) -> String {
    match version {
        Some(v)
            if v.split('.').count() >= 2
                && v.split('.').all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit())) =>
        {
            format!("glibc-{v}")
        }
        _ => "unknown".to_string(),
    }
}

#[cfg(all(target_os = "linux", target_env = "musl"))]
fn detect_libm() -> String {
    "musl".to_string()
}

#[cfg(all(target_os = "linux", not(any(target_env = "gnu", target_env = "musl"))))]
fn detect_libm() -> String {
    "unknown".to_string()
}

#[cfg(target_os = "macos")]
fn detect_libm() -> String {
    fn sysctl_string(name: &str) -> Option<String> {
        let key = std::ffi::CString::new(name).expect("sysctl name");
        let mut len: libc::size_t = 0;
        // SAFETY: size query with a null buffer, then a read into a buffer of that size.
        unsafe {
            if libc::sysctlbyname(key.as_ptr(), std::ptr::null_mut(), &mut len, std::ptr::null_mut(), 0) != 0
                || len == 0
            {
                return None;
            }
            let mut buf = vec![0u8; len];
            if libc::sysctlbyname(key.as_ptr(), buf.as_mut_ptr().cast(), &mut len, std::ptr::null_mut(), 0) != 0 {
                return None;
            }
            buf.truncate(len);
            while buf.last() == Some(&0) {
                buf.pop();
            }
            String::from_utf8(buf).ok()
        }
    }
    macos_libm_string(
        sysctl_string("kern.osproductversion").as_deref(),
        sysctl_string("kern.osversion").as_deref(),
    )
}

/// Task 33 contract for macOS: `apple-libm-macos-<kern.osproductversion>-<kern.osversion>`; ONE failed (or empty,
/// non-ASCII, dash-containing, over-long) sysctl makes the WHOLE value `unknown`.
pub fn macos_libm_string(product: Option<&str>, build: Option<&str>) -> String {
    let ok = |s: &str| !s.is_empty() && s.is_ascii() && !s.contains('-') && !s.chars().any(|c| c.is_ascii_control());
    match (product, build) {
        (Some(p), Some(b)) if ok(p) && ok(b) => {
            let v = format!("apple-libm-macos-{p}-{b}");
            if v.len() <= 256 {
                v
            } else {
                "unknown".to_string()
            }
        }
        _ => "unknown".to_string(),
    }
}

#[cfg(all(target_os = "windows", target_env = "msvc"))]
fn detect_libm() -> String {
    "msvc-ucrt".to_string()
}

#[cfg(all(target_os = "windows", not(target_env = "msvc")))]
fn detect_libm() -> String {
    "mingw-msvcrt".to_string()
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn detect_libm() -> String {
    "unknown".to_string()
}

/// Exact matchers of the task-33 contract regexes (no regex dependency in this crate):
/// `^(linux|darwin|windows|unknown)-(x86_64|arm64|unknown)$`.
pub fn matches_platform_contract(s: &str) -> bool {
    match s.split_once('-') {
        Some((os, arch)) => {
            matches!(os, "linux" | "darwin" | "windows" | "unknown")
                && matches!(arch, "x86_64" | "arm64" | "unknown")
        }
        None => false,
    }
}

/// `^(glibc-[0-9]+(\.[0-9]+)+|musl|apple-libm-macos-[^-]+-[^-]+|msvc-ucrt|mingw-msvcrt|unknown)$`.
pub fn matches_libm_contract(s: &str) -> bool {
    if matches!(s, "musl" | "msvc-ucrt" | "mingw-msvcrt" | "unknown") {
        return true;
    }
    if let Some(v) = s.strip_prefix("glibc-") {
        let parts: Vec<&str> = v.split('.').collect();
        return parts.len() >= 2 && parts.iter().all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()));
    }
    if let Some(rest) = s.strip_prefix("apple-libm-macos-") {
        let parts: Vec<&str> = rest.split('-').collect();
        return parts.len() == 2 && parts.iter().all(|p| !p.is_empty());
    }
    false
}

/// `^(debug|release)$`.
pub fn matches_profile_contract(s: &str) -> bool {
    matches!(s, "debug" | "release")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_strings_name_the_running_platform_and_libm() {
        let p = platform();
        let l = libm();
        eprintln!("FPP-PROVENANCE fpp_platform={p} fpp_libm={l} fpp_build_profile={}", build_profile());
        assert!(matches_platform_contract(p), "{p}");
        assert!(matches_libm_contract(l), "{l}");
        assert!(matches_profile_contract(build_profile()));
        assert!(p.len() <= 256 && l.len() <= 256 && p.is_ascii() && l.is_ascii());
        #[cfg(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"))]
        {
            assert_eq!(p, "linux-x86_64");
            assert!(l.starts_with("glibc-") && l != "unknown", "{l}");
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            assert_eq!(p, "darwin-arm64");
            assert!(l.starts_with("apple-libm-macos-"), "{l}");
        }
        assert_eq!(build_profile(), if cfg!(debug_assertions) { "debug" } else { "release" });
    }

    #[test]
    fn provenance_contract_maps_unknown_components_and_failed_lookups() {
        assert_eq!(platform_string("linux", "x86_64"), "linux-x86_64");
        assert_eq!(platform_string("macos", "aarch64"), "darwin-arm64");
        assert_eq!(platform_string("windows", "x86_64"), "windows-x86_64");
        assert_eq!(platform_string("linux", "riscv64"), "linux-unknown");
        assert_eq!(platform_string("freebsd", "x86_64"), "unknown-x86_64");
        assert_eq!(platform_string("freebsd", "powerpc"), "unknown-unknown");
        for s in ["linux-unknown", "unknown-arm64", "darwin-arm64"] {
            assert!(matches_platform_contract(s), "{s}");
        }
        for s in ["macos-arm64", "linux-aarch64", "linux", "linux-x86_64-extra"] {
            assert!(!matches_platform_contract(s), "{s}");
        }
        assert_eq!(macos_libm_string(Some("26.5.2"), Some("25F71")), "apple-libm-macos-26.5.2-25F71");
        assert_eq!(macos_libm_string(None, Some("25F71")), "unknown");
        assert_eq!(macos_libm_string(Some("26.5.2"), None), "unknown");
        assert_eq!(macos_libm_string(Some(""), Some("25F71")), "unknown");
        assert_eq!(glibc_libm_string(Some("2.43")), "glibc-2.43");
        assert_eq!(glibc_libm_string(Some("2.43.1")), "glibc-2.43.1");
        assert_eq!(glibc_libm_string(Some("2")), "unknown");
        assert_eq!(glibc_libm_string(Some("2.43-ubuntu")), "unknown");
        assert_eq!(glibc_libm_string(None), "unknown");
        for s in ["glibc-2.43", "musl", "apple-libm-macos-26.5.2-25F71", "msvc-ucrt", "mingw-msvcrt", "unknown"] {
            assert!(matches_libm_contract(s), "{s}");
        }
        for s in ["unknown-libc", "glibc-2", "glibc 2.43", "apple-libm-macos-26.5.2", "apple-libm-macos-unknown-x-y"] {
            assert!(!matches_libm_contract(s), "{s}");
        }
        assert!(matches_profile_contract("debug") && matches_profile_contract("release") && !matches_profile_contract("dev"));
    }
}
