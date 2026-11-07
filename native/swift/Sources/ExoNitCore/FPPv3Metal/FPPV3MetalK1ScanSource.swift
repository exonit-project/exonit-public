import Foundation

/// MSL source text holder for K1 orbit reduction, f32 screen, and df64 contact grid scan.
public enum FPPV3MetalK1Scan {
    public static let text: String = #"""
#ifndef FPPV3_MSL_PRELUDE
#define FPPV3_MSL_PRELUDE
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#endif

// LAYOUT v1 structs
struct K1Params {
    uint n;                // draws in this chunk (<= 65536)
    uint nb;               // fold bins
    uint companionIsHost;  // 0 or 1
    uint u1Stride, u2Stride;      // 1 = per-draw array, 0 = length-1 broadcast
    uint segOff[9];
    uint pad0, pad1;       // 0
    ulong foldW, foldDelta;       // f64 bits of fold.w and fold.delta
};

struct FoldParams { uint v[16]; };

struct K1Out {
    ulong root[5];         // t1 t2 t3 t4 tMin as f64 bit patterns, days from mid-transit; absent = 0x7ff8000000000000 (NaN)
    uint  k1State;         // 0 empty, 1 ok, 2 affected bins NOT contiguous
    uint  firstAff;        // first affected bin index
    uint  nAff;            // number of affected bins
    float epsRef;          // sqrt(2 delta / nAff), df64 -> f32
    float epsMin;          // min over affected bins of sigma[b] * epsRef, df64 -> f32
    float dilA;            // 1 / (1 + fRatio) in df64 -> f32
};

struct K1Orbit {
    df64 P, aRs, cosI, ecc, w, mT, n;            // period, a/R*, cos(inc), e, arg. of periastron, M_T, n = 2 pi / P
    float sET, cET, q, A, B;                     // p4.0 A' constants, f32
    float aRs32, cosI32, ecc32, n32, nLo;        // f32 roundings; nLo = low word of n
    bool  ok;                                    // false if any input is non-finite, P <= 0, aRs <= 0, ecc < 0 or ecc >= 1
};

#ifndef FPPV3_NEWTON_RTOL
#define FPPV3_NEWTON_RTOL
constant float NEWTON_RTOL = 2.0e-7f;
#endif
constant float K1_ULP_FN = 8.0f;

inline K1Orbit k1_orbit(ulong Pbits, ulong aRbits, ulong incBits, ulong eccBits, ulong wBits) {
    df64 P = df64_from_bits(Pbits);
    df64 aRs = df64_from_bits(aRbits);
    df64 inc = df64_from_bits(incBits);
    df64 ecc = df64_from_bits(eccBits);
    df64 w = df64_from_bits(wBits);

    bool ok = df64_isfinite(P) && df64_isfinite(aRs) && df64_isfinite(inc) &&
              df64_isfinite(ecc) && df64_isfinite(w);
    if (ok) {
        if (df64_le(P, DF64_ZERO) || df64_le(aRs, DF64_ZERO) ||
            df64_lt(ecc, DF64_ZERO) || df64_ge(ecc, DF64_ONE)) {
            ok = false;
        }
    }

    if (!ok) {
        K1Orbit bad;
        bad.P = P; bad.aRs = aRs; bad.cosI = DF64_ZERO; bad.ecc = ecc; bad.w = w;
        bad.mT = DF64_ZERO; bad.n = DF64_ZERO;
        bad.sET = 0.0f; bad.cET = 0.0f; bad.q = 0.0f; bad.A = 0.0f; bad.B = 0.0f;
        bad.aRs32 = 0.0f; bad.cosI32 = 0.0f; bad.ecc32 = 0.0f; bad.n32 = 0.0f; bad.nLo = 0.0f;
        bad.ok = false;
        return bad;
    }

    df64 cosI = df64_cos(inc);
    df64 n = df64_div(DF64_TWO_PI, P);

    df64 nuT = df64_sub(DF64_HALF_PI, w);
    df64 halfNuT = df64_mul_f(nuT, 0.5f);
    df64_sc sc_nuT = df64_sincos(halfNuT);
    df64 one_m_e = df64_sub(DF64_ONE, ecc);
    df64 one_p_e = df64_add(DF64_ONE, ecc);
    df64 sq_m = df64_sqrt(one_m_e);
    df64 sq_p = df64_sqrt(one_p_e);
    df64 y_eT = df64_mul(sq_m, sc_nuT.s);
    df64 x_eT = df64_mul(sq_p, sc_nuT.c);
    df64 atan2_val = df64_atan2(y_eT, x_eT); // E_T / 2
    df64 eT = df64_mul_f(atan2_val, 2.0f);
    df64 sin_eT = df64_sin(eT);
    df64 mT = df64_sub(eT, df64_mul(ecc, sin_eT));

    df64_sc sc_eT = df64_sincos(eT);
    df64 q_df = df64_div(sq_p, sq_m);
    df64_sc sc_halfET = df64_sincos(atan2_val);
    df64 q2 = df64_mul(q_df, q_df);
    df64 sin2_h = df64_mul(sc_halfET.s, sc_halfET.s);
    df64 cos2_h = df64_mul(sc_halfET.c, sc_halfET.c);
    df64 A_df = df64_add(cos2_h, df64_mul(q2, sin2_h));
    df64 q2_minus_1 = df64_sub(q2, DF64_ONE);
    df64 sincos_h = df64_mul(sc_halfET.s, sc_halfET.c);
    df64 B_df = df64_mul(q2_minus_1, sincos_h);

    K1Orbit o;
    o.P = P;
    o.aRs = aRs;
    o.cosI = cosI;
    o.ecc = ecc;
    o.w = w;
    o.mT = mT;
    o.n = n;
    o.sET = df64_to_float(sc_eT.s);
    o.cET = df64_to_float(sc_eT.c);
    o.q = df64_to_float(q_df);
    o.A = df64_to_float(A_df);
    o.B = df64_to_float(B_df);
    o.aRs32 = df64_to_float(aRs);
    o.cosI32 = df64_to_float(cosI);
    o.ecc32 = df64_to_float(ecc);
    o.n32 = df64_to_float(n);
    o.nLo = n.lo;
    o.ok = true;
    return o;
}

inline float2 k1_z32(float dM, thread const K1Orbit& o) {
    if (!o.ok) return float2(NAN, NAN);
    float twoPi = 2.0f * 3.14159265358979323846f;
    float aRs = o.aRs32, cosI = o.cosI32, ecc = o.ecc32;
    float sET = o.sET, cET = o.cET, q = o.q, A = o.A, B = o.B;
    dM = dM - twoPi * rint(dM / twoPi);  // wrap to (-pi, pi]
    float dE = dM;
    bool conv = false;
    for (int i = 0; i < 12; i++) {
        float sh = sin(0.5f * dE), sd = sin(dE), cd = cos(dE);
        float g = dE - ecc * (cET * sd - 2.0f * sET * sh * sh) - dM;
        float gp = 1.0f - ecc * (cET * cd - sET * sd);
        float de = g / gp;
        dE -= de;
        if (fabs(de) <= NEWTON_RTOL * fabs(dE) + 1.0e-30f) {
            conv = true;
            break;
        }
    }
    if (!conv) {
        return float2(NAN, NAN);
    }
    float sh = sin(0.5f * dE), ch = cos(0.5f * dE);
    float N = q * sh;
    float D = A * ch + B * sh;
    float n2 = N * N + D * D;
    float sinF = 2.0f * N * D / n2;
    float cosF = (D * D - N * N) / n2;
    float r = 1.0f - ecc * (cET * cos(dE) - sET * sin(dE));
    float z = aRs * r * sqrt(sinF * sinF + cosI * cosI * cosF * cosF);
    return float2(z, cosF);
}

inline df64 k1_zone_df64(df64 t, thread const K1Orbit& o) {
    if (!o.ok) return df64{NAN, NAN};
    df64 phase = df64_div(t, o.P);
    df64 m = df64_add(df64_mul(DF64_TWO_PI, phase), o.mT);
    df64 m_div = df64_div(m, DF64_TWO_PI);
    df64 k_fl = df64_floor(m_div);
    m = df64_sub(m, df64_mul(DF64_TWO_PI, k_fl));

    df64 nu;
    if (o.ecc.hi < 1e-10f) {
        nu = m;
    } else {
        df64 e = m;
        df64 lo = DF64_ZERO;
        df64 hi = DF64_TWO_PI;
        for (int it = 0; it < 30; it++) {
            df64_sc sc_e = df64_sincos(e);
            df64 f = df64_sub(df64_sub(e, df64_mul(o.ecc, sc_e.s)), m);
            if (f.hi == 0.0f && f.lo == 0.0f) break;
            if (f.hi > 0.0f || (f.hi == 0.0f && f.lo > 0.0f)) {
                hi = e;
            } else {
                lo = e;
            }
            df64 gp = df64_sub(DF64_ONE, df64_mul(o.ecc, sc_e.c));
            df64 de = df64_div(f, gp);
            df64 en = df64_sub(e, de);
            bool tiny = abs(de.hi) < 1.0e-10f;
            if (it < 30 && ((df64_gt(en, lo) && df64_lt(en, hi)) || tiny)) {
                e = en;
                if (tiny) break;
            } else {
                float scale = abs(e.hi) > 1.0f ? abs(e.hi) : 1.0f;
                df64 diff = df64_sub(hi, lo);
                if (diff.hi <= 1.0e-15f * scale) break;
                e = df64_add(lo, df64_mul_f(diff, 0.5f));
            }
        }
        df64_sc sc_e = df64_sincos(e);
        df64 one_m_e2 = df64_sub(DF64_ONE, df64_mul(o.ecc, o.ecc));
        df64 sq = df64_sqrt(one_m_e2);
        df64 y = df64_mul(sq, sc_e.s);
        df64 x = df64_sub(sc_e.c, o.ecc);
        nu = df64_atan2(y, x);
    }

    df64 one_m_e2 = df64_sub(DF64_ONE, df64_mul(o.ecc, o.ecc));
    df64 cos_nu = df64_cos(nu);
    df64 denom = df64_add(DF64_ONE, df64_mul(o.ecc, cos_nu));
    df64 rFactor = df64_div(one_m_e2, denom);

    df64 f = df64_sub(df64_add(nu, o.w), DF64_HALF_PI);
    df64_sc sc_f = df64_sincos(f);
    if (sc_f.c.hi < 0.0f || (sc_f.c.hi == 0.0f && sc_f.c.lo < 0.0f)) {
        return df64_from_float(1000.0f);
    }

    df64 sinF2 = df64_mul(sc_f.s, sc_f.s);
    df64 cosF2 = df64_mul(sc_f.c, sc_f.c);
    df64 cosI2 = df64_mul(o.cosI, o.cosI);
    df64 val = df64_add(sinF2, df64_mul(cosI2, cosF2));
    df64 sqrt_val = df64_sqrt(val);
    return df64_mul(df64_mul(o.aRs, rFactor), sqrt_val);
}

struct K1Grid { df64 w; df64 step; };

inline K1Grid k1_grid(ulong kBits, thread const K1Orbit& o) {
    if (!o.ok) {
        return K1Grid{df64_from_float(1.0e-3f), df64_div(df64_from_float(2.0e-3f), df64_from_float(2000.0f))};
    }
    df64 k = df64_from_bits(kBits);
    df64 one_p_k = df64_add(DF64_ONE, k);
    df64 num = df64_mul(df64_from_float(2.5f), df64_mul(one_p_k, o.P));
    df64 one_m_e = df64_sub(DF64_ONE, o.ecc);
    df64 den = df64_mul(df64_mul(DF64_PI, o.aRs), one_m_e);
    df64 w = df64_div(num, den);

    df64 half_P = df64_mul_f(o.P, 0.5f);
    if (df64_gt(w, half_P)) {
        w = half_P;
    }
    df64 min_w = df64_from_float(1.0e-3f);
    if (df64_lt(w, min_w)) {
        w = min_w;
    }
    df64 step = df64_div(df64_mul_f(w, 2.0f), df64_from_float(2000.0f));
    return K1Grid{w, step};
}

inline float k1_band_cos(float z32, float cosF32, float tAbs32, thread const K1Orbit& o) {
    float ecc = o.ecc32;
    float one_m_e = max(1.0e-4f, 1.0f - ecc);
    float q = sqrt((1.0f + ecc) / one_m_e);
    float eps0 = 0x1.0p-24f;
    float b_cos = q * (2.0e-7f * 3.14159265358979323846f + 16.0f * eps0 / one_m_e) + 32.0f * eps0;
    return max(b_cos, 1.0e-5f);
}

inline float k1_band(float z32, float cosF32, float tAbs32, thread const K1Orbit& o) {
    float ecc = o.ecc32;
    float one_m_e = max(1.0e-4f, 1.0f - ecc);
    float K_rel = 28.0f + 30.0f / one_m_e;
    float z_floor = 0.18f * o.aRs32 + 1.0f * o.aRs32 * o.n32 * tAbs32;
    float eps0 = 0x1.0p-24f;
    return K_rel * eps0 * (z32 + z_floor);
}

struct K1Scan { df64 bestZ; int i0; uint nRecheck; uint flags; };

inline K1Scan k1_scan(thread const K1Orbit& o, thread const K1Grid& g) {
    if (!o.ok) {
        return K1Scan{df64_from_float(1000.0f), -1, 0, 0};
    }
    uint flags = 0;
    float U = 1.0e30f;
    bool has_front = false;

    // Pass 1: compute upper bound U from certain front points and classify
    for (int i = 0; i < 2001; i++) {
        df64 fi = df64_from_float((float)i);
        df64 ti = df64_add(df64_neg(g.w), df64_mul(g.step, fi));
        df64 n_ti = df64_mul(o.n, ti);
        df64 scaled = df64_div(n_ti, DF64_TWO_PI);
        df64 k_rint = df64_rint(scaled);
        df64 dM_df = df64_sub(n_ti, df64_mul(DF64_TWO_PI, k_rint));
        float dM32 = df64_to_float(dM_df);
        float tAbs32 = abs(df64_to_float(ti));

        float2 z_cos = k1_z32(dM32, o);
        float z32 = z_cos.x;
        float cosF32 = z_cos.y;
        float b_cos = k1_band_cos(z32, cosF32, tAbs32, o);
        float b_z = k1_band(z32, cosF32, tAbs32, o);

        if (!isfinite(z32) || !isfinite(cosF32)) {
            flags |= 1;
        } else if (cosF32 >= b_cos) {
            has_front = true;
            float upper = z32 + b_z;
            if (upper < U) U = upper;
        } else if (cosF32 < -b_cos) {
            // certain back
        } else {
            flags |= 1;
        }
    }

    df64 bestZ = df64_from_float(1000.0f);
    int i0 = -1;
    uint nRecheck = 0;

    // Pass 2: evaluate points in candidate set S
    for (int i = 0; i < 2001; i++) {
        df64 fi = df64_from_float((float)i);
        df64 ti = df64_add(df64_neg(g.w), df64_mul(g.step, fi));
        df64 n_ti = df64_mul(o.n, ti);
        df64 scaled = df64_div(n_ti, DF64_TWO_PI);
        df64 k_rint = df64_rint(scaled);
        df64 dM_df = df64_sub(n_ti, df64_mul(DF64_TWO_PI, k_rint));
        float dM32 = df64_to_float(dM_df);
        float tAbs32 = abs(df64_to_float(ti));

        float2 z_cos = k1_z32(dM32, o);
        float z32 = z_cos.x;
        float cosF32 = z_cos.y;
        float b_cos = k1_band_cos(z32, cosF32, tAbs32, o);
        float b_z = k1_band(z32, cosF32, tAbs32, o);

        bool in_S = false;
        if (!isfinite(z32) || !isfinite(cosF32)) {
            in_S = true;
        } else if (cosF32 >= b_cos) {
            if (!has_front || (z32 - b_z <= U)) {
                in_S = true;
            }
        } else if (cosF32 < -b_cos) {
            in_S = false;
        } else {
            in_S = true;
        }

        if (in_S) {
            nRecheck++;
            df64 z = k1_zone_df64(ti, o);
            if (df64_lt(z, df64_from_float(1000.0f)) && df64_lt(z, bestZ)) {
                bestZ = z;
                i0 = i;
            }
        }
    }

    return K1Scan{bestZ, i0, nRecheck, flags};
}

struct K1ScanProbeOut { ulong bestZBits; int i0; uint nRecheck; };

kernel void k_k1_scan_probe(device const ulong* drawIn [[buffer(0)]], constant K1Params& P [[buffer(1)]],
                            device K1ScanProbeOut* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= P.n) return;
    ulong kBits   = drawIn[P.segOff[0] + gid];
    ulong Pbits   = drawIn[P.segOff[1] + gid];
    ulong incBits = drawIn[P.segOff[2] + gid];
    ulong aRbits  = drawIn[P.segOff[3] + gid];
    ulong eccBits = drawIn[P.segOff[4] + gid];
    ulong wBits   = drawIn[P.segOff[5] + gid];

    K1Orbit o = k1_orbit(Pbits, aRbits, incBits, eccBits, wBits);
    K1Grid g = k1_grid(kBits, o);
    K1Scan s = k1_scan(o, g);
    out[gid] = K1ScanProbeOut{ df64_to_bits(s.bestZ), s.i0, s.nRecheck };
}

kernel void k_k1_z32_probe(device const ulong* drawIn [[buffer(0)]], constant K1Params& P [[buffer(1)]],
                           constant uint& gridIndex [[buffer(2)]], device float* out [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= P.n) return;
    ulong Pbits   = drawIn[P.segOff[1] + gid];
    ulong incBits = drawIn[P.segOff[2] + gid];
    ulong aRbits  = drawIn[P.segOff[3] + gid];
    ulong eccBits = drawIn[P.segOff[4] + gid];
    ulong wBits   = drawIn[P.segOff[5] + gid];
    ulong kBits   = drawIn[P.segOff[0] + gid];

    K1Orbit o = k1_orbit(Pbits, aRbits, incBits, eccBits, wBits);
    K1Grid g = k1_grid(kBits, o);

    df64 fi = df64_from_float((float)gridIndex);
    df64 ti = df64_add(df64_neg(g.w), df64_mul(g.step, fi));
    df64 n_ti = df64_mul(o.n, ti);
    df64 scaled = df64_div(n_ti, DF64_TWO_PI);
    df64 k_rint = df64_rint(scaled);
    df64 dM_df = df64_sub(n_ti, df64_mul(DF64_TWO_PI, k_rint));
    float dM32 = df64_to_float(dM_df);

    float2 z_cos = k1_z32(dM32, o);
    out[2 * gid] = z_cos.x;
    out[2 * gid + 1] = z_cos.y;
}
"""#
}
