//! N1 constants block of the fixed FPP model (plan fix N1): the astropy 7.2.2 cgs values used by upstream
//! TRICERATOPS, as EXACT f64 bit patterns (the Python reference writes them with `float.fromhex`). The old
//! decimal literals of `fpp::scenarios` (`G_CGS = 6.674e-8`, `MSUN = 1.989e33`, ...) are NOT these values
//! and are replaced when the scenario families are ported.

/// `G_cgs`, cm^3 g^-1 s^-2 (0x1.1ea8b45edf98bp-24).
pub const G_CGS: f64 = f64::from_bits(0x3e71ea8b45edf98b);
/// `Msun`, g (0x1.88250b3b66e99p+110).
pub const MSUN: f64 = f64::from_bits(0x46d88250b3b66e99);
/// `Rsun`, cm (0x1.032b1f4800000p+36).
pub const RSUN: f64 = f64::from_bits(0x423032b1f4800000);
/// `Rearth`, cm (0x1.3021aa8000000p+29).
pub const REARTH: f64 = f64::from_bits(0x41c3021aa8000000);
/// `au`, cm (0x1.b363196a46000p+43).
pub const AU: f64 = f64::from_bits(0x42ab363196a46000);
/// `pi` (numpy's `np.pi`, identical to `std::f64::consts::PI`).
pub const PI: f64 = std::f64::consts::PI;
/// `ln2pi = np.log(2 * pi)` as the reference computed it: pinned as bits (numpy's scalar `log` is not
/// guaranteed to be glibc's), asserted within 1 ulp of `(2*pi).ln()` by the golden test.
pub const LN2PI: f64 = f64::from_bits(0x3ffd67f1c864beb4);

/// `FPP_MODEL_VERSION` of the reference (integer; the shipped model was 1).
pub const FPP_MODEL_VERSION: u32 = 2;

/// Gauss-Legendre exposure integrator: orders, oracle and smooth indices (`GL_ORDERS`, ...).
pub const GL_SMOOTH_IDX: usize = 2;
pub const GL_ORACLE_IDX: usize = 10;

/// Quantile-node ladder constants (`FPP_LADDER_DELTA_NAT`, `LADDER_*`).
pub const FPP_LADDER_DELTA_NAT: f64 = 0.02;
pub const LADDER_K_QUANT: usize = 256;
pub const LADDER_MAX_K: usize = 128;
pub const LADDER_EXACT_LEVEL: usize = 7;
pub const LADDER_MIN_POINTS: usize = 8;
pub const LADDER_ACCEPT_MAX_LEVEL: usize = 5;

/// Fold constants (`FOLD_N_BINS`, `FOLD_K_QUANT`).
pub const FOLD_N_BINS: usize = 1000;
pub const FOLD_K_QUANT: usize = 256;

/// Parity-mode exposure supersampling (`PARITY_EXPTIME_D`, `PARITY_NSUB`).
pub const PARITY_EXPTIME_D: f64 = 0.00139;
pub const PARITY_NSUB: usize = 20;
