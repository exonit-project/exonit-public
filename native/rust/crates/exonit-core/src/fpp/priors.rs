//! Prior samplers — transliteration of `worker/fpp_native.py:545-689`.
//!
//! RNG-consumption contract: numpy fills the whole `rand(N)` vector FIRST
//! and transforms it in place afterwards; every sampler here does the
//! same (including the degenerate `return ones(N)` branches, which in the
//! Python still consume N uniform draws before returning).

use super::rng::FppRng;

const PI: f64 = std::f64::consts::PI;

fn rand_vec(rng: &mut FppRng, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.rand()).collect()
}

/// `_sample_rp` — broken power-law planet radii (Earth radii).
pub fn sample_rp(rng: &mut FppRng, n: usize, m_s: f64) -> Vec<f64> {
    let mut x = rand_vec(rng, n);
    let (r_break1, r_break2, r_min, r_max) = (3.0f64, 6.0f64, 0.5f64, 20.0f64);
    let (p1, p2, p3) = if m_s > 0.45 {
        (0.0, -4.0, -0.5)
    } else {
        (0.0, -7.0, -0.5)
    };

    let a1 = r_break1.powf(p1) / r_break1.powf(p2);
    let a2 = r_break2.powf(p2) / r_break2.powf(p3);
    let i1 = (r_break1.powf(p1 + 1.0) - r_min.powf(p1 + 1.0)) / (p1 + 1.0);
    let i2 = a1 * (r_break2.powf(p2 + 1.0) - r_break1.powf(p2 + 1.0)) / (p2 + 1.0);
    let i3 = a2 * a1 * (r_max.powf(p3 + 1.0) - r_break2.powf(p3 + 1.0)) / (p3 + 1.0);
    let norm = 1.0 / (i1 + i2 + i3);

    // Task 33: numpy applies its fast scalar power to the masked ARRAYS (m3's exponent is exactly 2.0 -> x * x);
    // `powf(2.0)` only matched because LLVM lowers it to x * x when optimising, which a Debug build need not do.
    for v in x.iter_mut() {
        let u = *v;
        *v = if u <= norm * i1 {
            npow(u / norm * (p1 + 1.0) + r_min.powf(p1 + 1.0), 1.0 / (p1 + 1.0))
        } else if u <= norm * (i1 + i2) {
            npow((u / norm - i1) * (p2 + 1.0) / a1 + r_break1.powf(p2 + 1.0), 1.0 / (p2 + 1.0))
        } else {
            npow(
                (u / norm - i1 - i2) * (p3 + 1.0) / (a1 * a2) + r_break2.powf(p3 + 1.0),
                1.0 / (p3 + 1.0),
            )
        };
    }
    x
}

/// `_sample_rp` with an ARRAY of host masses (plan step 4, the S family): each draw uses the exponents of ITS
/// host (`M > 0.45: p2 = -4, else -7`). The per-group constants are computed with the exact scalar arithmetic
/// (same `powf` calls), then applied to the selected draws with numpy's umath fast paths (`npow`).
pub fn sample_rp_array(rng: &mut FppRng, n: usize, m_s: &[f64]) -> Vec<f64> {
    let x = rand_vec(rng, n);
    let (r_break1, r_break2, r_min, r_max) = (3.0f64, 6.0f64, 0.5f64, 20.0f64);
    let (p1, p3) = (0.0f64, -0.5f64);
    let mut out = vec![0.0f64; n];
    for (heavy, p2) in [(true, -4.0f64), (false, -7.0f64)] {
        let sel: Vec<usize> = (0..n).filter(|&i| (m_s[i] > 0.45) == heavy).collect();
        if sel.is_empty() {
            continue;
        }
        let a1 = r_break1.powf(p1) / r_break1.powf(p2);
        let a2 = r_break2.powf(p2) / r_break2.powf(p3);
        let i1 = (r_break1.powf(p1 + 1.0) - r_min.powf(p1 + 1.0)) / (p1 + 1.0);
        let i2 = a1 * (r_break2.powf(p2 + 1.0) - r_break1.powf(p2 + 1.0)) / (p2 + 1.0);
        let i3 = a2 * a1 * (r_max.powf(p3 + 1.0) - r_break2.powf(p3 + 1.0)) / (p3 + 1.0);
        let norm = 1.0 / (i1 + i2 + i3);
        for &i in &sel {
            let u = x[i];
            // numpy applies the umath fast paths to the selected draws (m3's exponent is
            // exactly 2.0 -> `x * x`, 1 ULP off `pow(x, 2.0)` on some inputs).
            out[i] = if u <= norm * i1 {
                npow(
                    u / norm * (p1 + 1.0) + r_min.powf(p1 + 1.0),
                    1.0 / (p1 + 1.0),
                )
            } else if u <= norm * (i1 + i2) {
                npow(
                    (u / norm - i1) * (p2 + 1.0) / a1 + r_break1.powf(p2 + 1.0),
                    1.0 / (p2 + 1.0),
                )
            } else {
                npow(
                    (u / norm - i1 - i2) * (p3 + 1.0) / (a1 * a2) + r_break2.powf(p3 + 1.0),
                    1.0 / (p3 + 1.0),
                )
            };
        }
    }
    out
}

/// `_sample_inc` — sin(i) prior, degrees.
pub fn sample_inc(rng: &mut FppRng, n: usize) -> Vec<f64> {
    let x = rand_vec(rng, n);
    x.into_iter()
        .map(|u| (1.0 - u / 1.0).acos() * 180.0 / PI)
        .collect()
}

/// `_sample_ecc` — Beta(0.867, 3.030) for planets; scipy powerlaw for
/// binaries (`nu_plus_1` = 0.2 if P ≤ 10 d else 0.6).
pub fn sample_ecc(rng: &mut FppRng, n: usize, planet: bool, p_orb: f64) -> Vec<f64> {
    if planet {
        (0..n).map(|_| rng.beta(0.867, 3.030)).collect()
    } else {
        let nu_plus_1 = if p_orb <= 10.0 { 0.2 } else { 0.6 };
        (0..n).map(|_| rng.power(nu_plus_1)).collect()
    }
}

/// `_sample_w` — uniform argument of periastron, degrees.
pub fn sample_w(rng: &mut FppRng, n: usize) -> Vec<f64> {
    rand_vec(rng, n).into_iter().map(|u| u * 360.0).collect()
}

/// numpy's umath fast path for float64 arrays: an exponent of exactly 2.0 is computed as
/// `x * x`, which can differ by 1 ULP from `pow(x, 2.0)`.
fn npow(y: f64, e: f64) -> f64 {
    if e == 2.0 {
        y * y
    } else {
        y.powf(e)
    }
}

/// Shared body of the two q samplers once (q_min, p1, p2, F_twin) are set.
fn q_transform(x: &mut [f64], q_min: f64, p1: f64, p2: f64, f_twin: f64) {
    if q_min < 0.3 {
        let a1 = 0.3f64.powf(p1) / 0.3f64.powf(p2);
        let a2 = 1.0
            + (f_twin / (1.0 - f_twin))
                * ((1.0f64.powf(p2 + 1.0) - 0.3f64.powf(p2 + 1.0)) / (p2 + 1.0))
                / ((1.0f64.powf(p2 + 1.0) - 0.95f64.powf(p2 + 1.0)) / (p2 + 1.0));
        let i1 = (0.3f64.powf(p1 + 1.0) - q_min.powf(p1 + 1.0)) / (p1 + 1.0);
        let i2 = a1 * (0.95f64.powf(p2 + 1.0) - 0.3f64.powf(p2 + 1.0)) / (p2 + 1.0);
        let i3 = a2 * a1 * (1.0f64.powf(p2 + 1.0) - 0.95f64.powf(p2 + 1.0)) / (p2 + 1.0);
        let norm = 1.0 / (i1 + i2 + i3);
        for v in x.iter_mut() {
            let u = *v;
            *v = if u <= norm * i1 {
                npow(
                    u / norm * (p1 + 1.0) + q_min.powf(p1 + 1.0),
                    1.0 / (p1 + 1.0),
                )
            } else if u <= norm * (i1 + i2) {
                npow(
                    (u / norm - i1) * (p2 + 1.0) / a1 + 0.3f64.powf(p2 + 1.0),
                    1.0 / (p2 + 1.0),
                )
            } else {
                npow(
                    (u / norm - i1 - i2) * (p2 + 1.0) / (a1 * a2) + 0.95f64.powf(p2 + 1.0),
                    1.0 / (p2 + 1.0),
                )
            };
        }
    } else {
        let a2 = 1.0
            + (f_twin / (1.0 - f_twin))
                * ((1.0f64.powf(p2 + 1.0) - q_min.powf(p2 + 1.0)) / (p2 + 1.0))
                / ((1.0f64.powf(p2 + 1.0) - 0.95f64.powf(p2 + 1.0)) / (p2 + 1.0));
        let i2 = (0.95f64.powf(p2 + 1.0) - q_min.powf(p2 + 1.0)) / (p2 + 1.0);
        let i3 = a2 * (1.0f64.powf(p2 + 1.0) - 0.95f64.powf(p2 + 1.0)) / (p2 + 1.0);
        let norm = 1.0 / (i2 + i3);
        for v in x.iter_mut() {
            let u = *v;
            *v = if u <= norm * i2 {
                npow(
                    u / norm * (p2 + 1.0) + q_min.powf(p2 + 1.0),
                    1.0 / (p2 + 1.0),
                )
            } else {
                npow(
                    (u / norm - i2) * (p2 + 1.0) / a2 + 0.95f64.powf(p2 + 1.0),
                    1.0 / (p2 + 1.0),
                )
            };
        }
    }
    for v in x.iter_mut() {
        *v = v.clamp(q_min, 1.0);
    }
}

/// `_sample_q` — mass ratios, short-period binaries (Moe & Di Stefano 2017).
pub fn sample_q(rng: &mut FppRng, n: usize, m_s: f64) -> Vec<f64> {
    let mut x = rand_vec(rng, n);
    let q_min = if m_s >= 1.0 {
        0.1
    } else if m_s >= 0.3 {
        0.1 / m_s
    } else if m_s > 0.1 {
        0.1 / m_s
    } else {
        return vec![1.0; n];
    };
    q_transform(&mut x, q_min, 0.3, -0.5, 0.30);
    x
}

/// Transform an already-consumed vector of uniforms with the scalar host-mass
/// parameters used by `_sample_q` (BEB's `FPP_BACKGROUND_Q=host` path).
pub fn sample_q_from_uniforms(mut x: Vec<f64>, m_s: f64) -> Vec<f64> {
    let q_min = if m_s >= 1.0 {
        0.1
    } else if m_s > 0.1 {
        0.1 / m_s
    } else {
        return vec![1.0; x.len()];
    };
    q_transform(&mut x, q_min, 0.3, -0.5, 0.30);
    x
}

/// `_sample_q_companion` — mass ratios, long-period companions.
pub fn sample_q_companion(rng: &mut FppRng, n: usize, m_s: f64) -> Vec<f64> {
    let mut x = rand_vec(rng, n);
    if m_s <= 0.1 {
        return vec![1.0; n];
    }
    let q_min = if m_s > 0.1 { (0.1 / m_s).max(0.1) } else { 1.0 };
    q_transform(&mut x, q_min, 0.3, -0.95, 0.05);
    x
}
