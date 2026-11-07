import Foundation

/// Metal Shading Language source for the V3 K2 transit/eclipse deficit kernel (contract D5).
///
/// The Carlson R_F / R_C / R_J and K / E / Pi code (`// ---- BEGIN p4.0 lines 40-142`) and the
/// Mandel & Agol branch structure are the frozen p4.0 f32 flux kernel, copied unchanged. Only the
/// two early returns and the last line differ: these functions return the DEFICIT `D = 1 - flux`
/// directly and never form the flux `m ~ 1` followed by a subtraction. The `transit_deficit` bridge
/// is the p4.0 quadratic z ~ p bridge applied to the deficit values (the bridge is linear in the
/// values, so it commutes with `D = 1 - flux`).
public enum FPPV3MetalK2Flux {
    public static let text = #"""
#ifndef FPPV3_MSL_PRELUDE
#define FPPV3_MSL_PRELUDE
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#endif

constant float PI_F = 3.14159265358979323846f;
constant float CARLSON_TOL = 1.0e-3f;
// F32: the f64 reference clamps kc2 at 1e-300; the smallest safe f32 value is used instead.
constant float KC2_MIN = 1.0e-30f;
// F32: relative Newton tolerance on dE (the f64 reference uses 1e-10 absolute, unreachable in f32).
#ifndef FPPV3_NEWTON_RTOL
#define FPPV3_NEWTON_RTOL
constant float NEWTON_RTOL = 2.0e-7f;
#endif
// F32: for |p - 1| < P1_NEAR the differences (1 +- p), (p - 1), (p^2 - 1) are formed first and
// exactly (z is not absorbed when p ~ 1); elsewhere the f64 forms have fewer roundings.
constant float P1_NEAR = 0.25f;
// p4: for n < N_SWITCH the characteristic transformation replaces Pi = K + (n/3) R_J(0, kc2, 1, 1 - n),
// which subtracts two ~K terms for large |n| (a priori constant, see docs/fpp-v2-metal-gate-p4.md).
constant float N_SWITCH = -1.0f;
// p4 guard (found on the dev / probe set): for kc2 < KC2_TR the transformation loses accuracy / overflows in R_J(.., 1-N tiny); the old form is used there.
constant float KC2_TR = 1.0e-6f;
// F32: half-width and node offset of the bridge around z == p (f64: 1e-6 / 1e-5, linear).
constant float BRIDGE_H = 1.0e-2f;

inline float max3f(float a, float b, float c) {
    float m = b > a ? b : a;
    return c > m ? c : m;
}

// ---- BEGIN p4.0 lines 40-142 (unchanged)
// ---------------------------------------------------------------- Carlson (DLMF 19.36)
float carlson_rf(float x, float y, float z) {
    float mu = 0.0f, dx = 0.0f, dy = 0.0f, dz = 0.0f;
    for (int i = 0; i < 200; i++) {
        float sx = sqrt(x), sy = sqrt(y), sz = sqrt(z);
        float lam = sx * sy + sy * sz + sz * sx;
        x = 0.25f * (x + lam);
        y = 0.25f * (y + lam);
        z = 0.25f * (z + lam);
        mu = (x + y + z) / 3.0f;
        dx = 1.0f - x / mu;
        dy = 1.0f - y / mu;
        dz = 1.0f - z / mu;
        if (max3f(fabs(dx), fabs(dy), fabs(dz)) < CARLSON_TOL) break;
    }
    float e2 = dx * dy - dz * dz;
    float e3 = dx * dy * dz;
    return (1.0f - e2 / 10.0f + e3 / 14.0f + e2 * e2 / 24.0f - 3.0f * e2 * e3 / 44.0f) / sqrt(mu);
}

float carlson_rc(float x, float y) {
    float mu = 0.0f, s = 0.0f;
    for (int i = 0; i < 200; i++) {
        float lam = 2.0f * sqrt(x) * sqrt(y) + y;
        x = 0.25f * (x + lam);
        y = 0.25f * (y + lam);
        mu = (x + 2.0f * y) / 3.0f;
        s = (y - mu) / mu;
        if (fabs(s) < CARLSON_TOL) break;
    }
    return (1.0f + s * s * (0.3f + s * (1.0f / 7.0f + s * (0.375f + s * 9.0f / 22.0f)))) / sqrt(mu);
}

float carlson_rj(float x, float y, float z, float p) {
    float total = 0.0f, fac = 1.0f;
    float mu = 0.0f, dx = 0.0f, dy = 0.0f, dz = 0.0f, dp = 0.0f;
    for (int i = 0; i < 200; i++) {
        float sx = sqrt(x), sy = sqrt(y), sz = sqrt(z);
        float lam = sx * sy + sy * sz + sz * sx;
        float t = p * (sx + sy + sz) + sx * sy * sz;
        float alpha = t * t;
        float u = p + lam;
        float beta = p * (u * u);
        total += fac * carlson_rc(alpha, beta);
        fac *= 0.25f;
        x = 0.25f * (x + lam);
        y = 0.25f * (y + lam);
        z = 0.25f * (z + lam);
        p = 0.25f * (p + lam);
        mu = (x + y + z + 2.0f * p) / 5.0f;
        dx = (mu - x) / mu;
        dy = (mu - y) / mu;
        dz = (mu - z) / mu;
        dp = (mu - p) / mu;
        float m1 = fabs(dy) > fabs(dx) ? fabs(dy) : fabs(dx);
        float m2 = fabs(dp) > fabs(dz) ? fabs(dp) : fabs(dz);
        float m = m2 > m1 ? m2 : m1;
        if (m < CARLSON_TOL) break;
    }
    float e2 = dx * dy + dx * dz + dy * dz - 3.0f * dp * dp;
    float e3 = dx * dy * dz + 2.0f * e2 * dp + 4.0f * (dp * (dp * dp));
    float e4 = (2.0f * dx * dy * dz + e2 * dp + 3.0f * (dp * (dp * dp))) * dp;
    float e5 = dx * dy * dz * dp * dp;
    float series = 1.0f - 3.0f * e2 / 14.0f + e3 / 6.0f + 9.0f * e2 * e2 / 88.0f
        - 3.0f * e4 / 22.0f
        - 9.0f * e2 * e3 / 52.0f
        + 3.0f * e5 / 26.0f;
    return 3.0f * total + fac * (1.0f / (mu * sqrt(mu))) * series;  // mu ** -1.5
}

float2 ellip_ke(float kc2) {
    kc2 = kc2 > KC2_MIN ? kc2 : KC2_MIN;
    float k = carlson_rf(0.0f, kc2, 1.0f);
    float e = k - ((1.0f - kc2) / 3.0f) * carlson_rj(0.0f, kc2, 1.0f, 1.0f);
    return float2(k, e);
}

float3 ellip_kep(float kc2, float n) {
    kc2 = kc2 > KC2_MIN ? kc2 : KC2_MIN;
    float2 ke = ellip_ke(kc2);
    float m = 1.0f - kc2;
    float pi;
    if (n < N_SWITCH && kc2 > KC2_TR) {
        // A&S 17.7.15 (complete case) / Byrd & Friedman 117 / DLMF 19.7.9 family, N = (m - n)/(1 - n):
        //   Pi(n|m) = [-n (1-m) / ((1-n)(m-n))] Pi(N|m) + [m/(m-n)] K(m), Pi(N|m) = K + (N/3) R_J(0, kc2, 1, 1-N),
        // 1 - N = kc2/(1 - n). Same mathematical quantity as the branch below; all terms positive.
        float omn = 1.0f - n;
        float p1 = kc2 / omn;
        float piN = ke.x + ((1.0f - p1) / 3.0f) * carlson_rj(0.0f, kc2, 1.0f, p1);
        float c1 = -n * kc2 / (omn * (m - n));
        float c2 = m / (m - n);
        pi = c1 * piN + c2 * ke.x;
    } else if (n < 1.0f) {
        if (n != 0.0f) {
            pi = ke.x + (n / 3.0f) * carlson_rj(0.0f, kc2, 1.0f, 1.0f - n);
        } else {
            pi = ke.x;
        }
    } else {
        pi = -(m / (3.0f * n)) * carlson_rj(0.0f, kc2, 1.0f, 1.0f - m / n);
    }
    return float3(ke.x, ke.y, pi);
}
// ---- END p4.0 lines 40-142

// ---------------------------------------------------------------- Mandel & Agol
float transit_deficit_core(float z, float p, float u1, float u2) {
    float c2 = u1 + 2.0f * u2;
    float c4 = -u2;
    float fo = 1.0f - c2 / 3.0f - c4 / 2.0f;
    if (p <= 0.0f || z >= 1.0f + p) return 0.0f;
    if (p >= 1.0f && z <= p - 1.0f) return 1.0f;
    float a = (z - p) * (z - p);
    float b = (z + p) * (z + p);
    float q = p * p - z * z;
    float theta = p > z ? 1.0f : 0.0f;
    float lamE, etaD, lamD;
    if (p < 1.0f && z <= 1.0f - p) {
        lamE = p * p;
        etaD = 0.5f * p * p * (p * p + 2.0f * z * z);
        if (z == 0.0f) {
            float om = 1.0f - p * p;
            lamD = -(2.0f / 3.0f) * (om * sqrt(om));  // (1 - p^2) ** 1.5
        } else if (z == p && p < 0.5f) {
            float kc = sqrt(1.0f - 4.0f * p * p);
            float2 ke = ellip_ke(kc * kc);
            lamD = 1.0f / 3.0f
                + (2.0f / (9.0f * PI_F)) * (4.0f * (2.0f * p * p - 1.0f) * ke.y + (1.0f - 4.0f * p * p) * ke.x);
            theta = 0.0f;
        } else if (z == p && p == 0.5f) {
            lamD = 1.0f / 3.0f - 4.0f / (9.0f * PI_F);
            theta = 0.0f;
        } else if (z == 1.0f - p) {
            lamD = (2.0f / (3.0f * PI_F)) * acos(clamp(1.0f - 2.0f * p, -1.0f, 1.0f))
                - (4.0f / (9.0f * PI_F)) * (3.0f + 2.0f * p - 8.0f * p * p) * sqrt(p * (1.0f - p))
                - (2.0f / 3.0f) * (p > 0.5f ? 1.0f : 0.0f);
        } else {
            // F32: factors grouped so that (1 - p) and (1 + p) are formed first (exact near p ~ 1).
            float kc = sqrt(((1.0f - z - p) * (1.0f + z + p)) / ((1.0f - z + p) * (1.0f + z - p)));
            if (fabs(p - 1.0f) < P1_NEAR) {
                kc = sqrt((((1.0f - p) - z) * (1.0f + z + p)) / (((1.0f + p) - z) * ((1.0f - p) + z)));
            }
            float n = -4.0f * p * z / a;  // p4: = (a - b)/a, a - b = -4 p z formed exactly
            float3 kep = ellip_kep(kc * kc, n);
            lamD = (2.0f / (9.0f * PI_F * sqrt(1.0f - a)))
                * ((1.0f - 5.0f * z * z + p * p + q * q) * kep.x
                    + (1.0f - a) * (z * z + 7.0f * p * p - 4.0f) * kep.y
                    - 3.0f * (q / a) * kep.z);
        }
    } else {
        // F32: (1 - z + p) = (1 + p) - z, (1 + z - p) = (1 - p) + z, (z + p - 1) = z + (p - 1),
        // p^2 - 1 = (p - 1)(p + 1): the f64 left-to-right forms absorb z when p ~ 1 (z < 6e-8).
        bool near1 = fabs(p - 1.0f) < P1_NEAR;
        float prod = (1.0f - z + p) * (1.0f + z - p) * (z + p - 1.0f) * (z + p + 1.0f);
        float a0 = p * p + z * z - 1.0f;
        float a1 = 1.0f - p * p + z * z;
        if (near1) {
            prod = ((1.0f + p) - z) * ((1.0f - p) + z) * (z + (p - 1.0f)) * (z + p + 1.0f);
            a0 = (p - 1.0f) * (p + 1.0f) + z * z;
            a1 = (1.0f - p) * (1.0f + p) + z * z;
        }
        float s = sqrt(prod > 0.0f ? prod : 0.0f);
        float k0 = atan2(s, a0);
        float k1 = atan2(s, a1);
        lamE = (p * p * k0 + k1 - 0.5f * s) / PI_F;
        float eta2 = p * p * (p * p + 2.0f * z * z) / 2.0f;
        etaD = (k1 + 2.0f * eta2 * k0 - 0.25f * (1.0f + 5.0f * p * p + z * z) * s) / (2.0f * PI_F);
        if (z == p) {
            if (p >= 0.5f) {
                float kc = sqrt(1.0f - 1.0f / (4.0f * p * p));
                float2 ke = ellip_ke(kc * kc);
                lamD = 1.0f / 3.0f + 16.0f * p / (9.0f * PI_F) * (2.0f * p * p - 1.0f) * ke.y
                    - (1.0f - 4.0f * p * p) * (3.0f - 8.0f * p * p) / (9.0f * PI_F * p) * ke.x;
                theta = 0.0f;
            } else {
                lamD = 0.0f;
            }
        } else {
            float kc = sqrt(((z + p - 1.0f) * (z + p + 1.0f)) / (4.0f * p * z));
            if (near1) kc = sqrt(((z + (p - 1.0f)) * (z + p + 1.0f)) / (4.0f * p * z));
            float n = (a - 1.0f) / a;
            float3 kep = ellip_kep(kc * kc, n);
            lamD = (1.0f / (9.0f * PI_F * sqrt(p * z)))
                * (((1.0f - b) * (2.0f * b + a - 3.0f) - 3.0f * q * (b - 2.0f)) * kep.x
                    + 4.0f * p * z * (z * z + 7.0f * p * p - 4.0f) * kep.y
                    - 3.0f * (q / a) * kep.z);
        }
    }
    return ((1.0f - c2) * lamE + c2 * (lamD + (2.0f / 3.0f) * theta) - c4 * etaD) / fo;
}

float transit_deficit(float z, float p, float u1, float u2) {
    // F32: the generic branch loses ~5e-8/|z - p| of absolute accuracy near z == p (the (q/a) Pi term),
    // so a quadratic (3-point) bridge through deficit_core(p-h), deficit_core(p), deficit_core(p+h) replaces the f64 linear one.
    float h = min(BRIDGE_H, 0.5f * p);
    if (p > 0.0f && z > 0.0f && p != z && fabs(z - p) < h && z < 1.0f + p
        && !(p >= 1.0f && z <= p - 1.0f)) {
        float fm = transit_deficit_core(p - h, p, u1, u2);
        float f0 = transit_deficit_core(p, p, u1, u2);
        float fp = transit_deficit_core(p + h, p, u1, u2);
        float x = (z - p) / h;
        return f0 + 0.5f * x * (fp - fm) + 0.5f * x * x * (fp + fm - 2.0f * f0);
    }
    return transit_deficit_core(z, p, u1, u2);
}


// rows (z, p, u1, u2) -> D
kernel void k_deficit(device const float* in [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    const uint base = 4u * gid;
    out[gid] = transit_deficit(in[base], in[base + 1u], in[base + 2u], in[base + 3u]);
}

// rows (z, p, u1, u2) -> D without the z ~ p bridge
kernel void k_deficit_core(device const float* in [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
    const uint base = 4u * gid;
    out[gid] = transit_deficit_core(in[base], in[base + 1u], in[base + 2u], in[base + 3u]);
}
"""#
}
