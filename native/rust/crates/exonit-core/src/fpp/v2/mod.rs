//! Rust port of the approved step-2 numerical pieces and step-3 D5 geometry (fix plan
//! `docs/superpowers/plans/2026-10-15-fpp-native-fix-plan.md`, steps 2 and 10; branch `fpp-v2-port`).
//!
//! Reference: `worker/fpp_native.py` (CPU / numba f64) of the exonit repo. Bit-exact with it on glibc x86-64,
//! including the q-prime phase-curve fit (the specified Householder QR). Scenario
//! families, batch likelihoods, priors/samplers and worker dispatch are outside
//! this numerical module and the D5 port.
//!
//! * [`consts`]  the N1 constants block (astropy cgs values as exact bit patterns)
//! * [`ellip`]   Carlson R_F / R_C / R_J and K, E, Pi
//! * [`kernel`]  the analytic Mandel-Agol quadratic-limb-darkening flux
//! * [`orbit`]   Kepler / z(t) geometry, the parity-mode exposure light curve
//! * [`contact`] exact contact times (bisection of z(t) and of (1/2) d z^2/dt)
//! * [`gl`]      the contact-split Gauss-Legendre exposure integrator and the order choice
//! * [`ladder`]  the quantile-node bin-model ladder and the per-draw bin models
//! * [`fold`]    quantile spec, the production (mean-bin) fold, fold arrays and constants
//! * [`d5`]     aperture flux ratios and viable neighbours
//! * [`wcs`]    task-shipped TAN sky-to-pixel geometry
//! * [`sigma`]   the (q') likelihood sigma components
//!
//! Arithmetic order is the Python order, literally. numba lowers `x ** 2` / `x ** 3` (constant integer
//! exponents) to multiplications (`x*x`, `x*(x*x)`) and `x ** 1.5` / `x ** -1.5` to libm `pow`
//! (here [`pow_libm`]); see `gotcha_python_scalar_pow2_vs_numpy_array_pow2`.

// Literal transcriptions of numba loops: index loops and negated comparisons are kept as in the reference
// (a NaN-safe `!(x > 0.0)` is the Python `not x > 0` semantics), so these style lints are silenced here.
#![allow(
    clippy::needless_range_loop,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::manual_range_contains
)]

pub mod calc_v3;
pub mod consts;
pub mod contact;
pub mod d5;
pub mod d_family;
pub mod dispatch;
pub mod ellip;
pub mod fold;
pub mod gate;
pub mod geometry_v3;
pub mod gl;
pub mod gl_tables;
pub mod kernel;
pub mod ladder;
pub mod math;
pub mod mc;
pub mod orbit;
pub mod population_v3;
pub mod s_family;
pub mod sigma;
pub mod target_family;
pub mod wcs;

pub use super::npsum::libm_pow as pow_libm;
