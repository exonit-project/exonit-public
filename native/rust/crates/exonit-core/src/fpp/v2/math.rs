//! Scalar libm wrappers used where the Python reference makes distinct calls.
//!
//! Keep sin and cos in separate, non-inlined C-library calls. LLVM can combine
//! adjacent `f64::sin` / `f64::cos` operations into Apple's `sincos` entry point
//! in optimized builds, changing their last bits relative to Python `math` and
//! making debug and release calculations disagree.

#[cfg_attr(unix, link(name = "m"))]
extern "C" {
    #[link_name = "sin"]
    fn c_sin(x: f64) -> f64;
    #[link_name = "cos"]
    fn c_cos(x: f64) -> f64;
}

#[inline(never)]
pub fn sin(x: f64) -> f64 {
    unsafe { c_sin(x) }
}

#[inline(never)]
pub fn cos(x: f64) -> f64 {
    unsafe { c_cos(x) }
}
