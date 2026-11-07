import Foundation

/// MSL source text holder for double-float (df64) trigonometry and transcendentals.
public enum FPPV3MetalDF64Trig {
    public static let text: String = #"""
#ifndef FPPV3_MSL_PRELUDE
#define FPPV3_MSL_PRELUDE
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#endif

struct df64_sc {
    df64 s;
    df64 c;
};

// pi/2 three-float decomposition for argument reduction:
constant float DF64_TRIG_P1 = 0x1.921fb6p+0f;
constant float DF64_TRIG_P2 = -0x1.777a5cp-25f;
constant float DF64_TRIG_P3 = -0x1.ee59dap-50f;

// 2 / pi:
constant df64 DF64_TRIG_TWO_O_PI = df64{0x1.45f306p-1f, 0x1.b93910p-26f};

// Taylor coefficients for sin: (-1)^k / (2k+1)!
constant df64 DF64_SIN_C3 = df64{-0x1.555556p-3f, 0x1.555556p-28f};
constant df64 DF64_SIN_C5 = df64{0x1.111112p-7f, -0x1.dddddep-32f};
constant df64 DF64_SIN_C7 = df64{-0x1.a01a02p-13f, 0x1.7f97fap-39f};
constant df64 DF64_SIN_C9 = df64{0x1.71de3ap-19f, 0x1.55b1ccp-45f};
constant df64 DF64_SIN_C11 = df64{-0x1.ae6456p-26f, -0x1.fd5138p-52f};
constant df64 DF64_SIN_C13 = df64{0x1.612462p-33f, -0x1.8af25ep-58f};
constant df64 DF64_SIN_C15 = df64{-0x1.ae7f3ep-41f, -0x1.ccee08p-67f};
constant df64 DF64_SIN_C17 = df64{0x1.952c78p-49f, -0x1.f9ea56p-74f};

// Taylor coefficients for cos: (-1)^k / (2k)!
constant df64 DF64_COS_C2 = df64{-0x1.000000p-1f, 0.0f};
constant df64 DF64_COS_C4 = df64{0x1.555556p-5f, -0x1.555556p-30f};
constant df64 DF64_COS_C6 = df64{-0x1.6c16c2p-10f, 0x1.27d27ep-35f};
constant df64 DF64_COS_C8 = df64{0x1.a01a02p-16f, -0x1.7f97fap-42f};
constant df64 DF64_COS_C10 = df64{-0x1.27e4fcp-22f, 0x1.10ec14p-47f};
constant df64 DF64_COS_C12 = df64{0x1.1eed8ep-29f, 0x1.ff1b12p-54f};
constant df64 DF64_COS_C14 = df64{-0x1.93974ap-37f, -0x1.180f94p-62f};
constant df64 DF64_COS_C16 = df64{0x1.ae7f3ep-45f, 0x1.ccee08p-71f};

inline df64_sc df64_sincos(df64 x) {
    if (df64_isnan(x) || abs(x.hi) > 64.0f) {
        return df64_sc{df64{NAN, NAN}, df64{NAN, NAN}};
    }

    df64 x_scaled = df64_mul(x, DF64_TRIG_TWO_O_PI);
    df64 k_df = df64_rint(x_scaled);
    float kf = df64_to_float(k_df);
    int k = (int)kf;

    df64 r = df64_sub(x, df_two_prod(kf, DF64_TRIG_P1));
    r = df64_sub(r, df_two_prod(kf, DF64_TRIG_P2));
    r = df64_add_f(r, -kf * DF64_TRIG_P3);

    df64 r2 = df64_mul(r, r);

    df64 s_poly = DF64_SIN_C17;
    s_poly = df64_add(DF64_SIN_C15, df64_mul(r2, s_poly));
    s_poly = df64_add(DF64_SIN_C13, df64_mul(r2, s_poly));
    s_poly = df64_add(DF64_SIN_C11, df64_mul(r2, s_poly));
    s_poly = df64_add(DF64_SIN_C9, df64_mul(r2, s_poly));
    s_poly = df64_add(DF64_SIN_C7, df64_mul(r2, s_poly));
    s_poly = df64_add(DF64_SIN_C5, df64_mul(r2, s_poly));
    s_poly = df64_add(DF64_SIN_C3, df64_mul(r2, s_poly));
    df64 s_val = df64_add(r, df64_mul(r, df64_mul(r2, s_poly)));

    df64 c_poly = DF64_COS_C16;
    c_poly = df64_add(DF64_COS_C14, df64_mul(r2, c_poly));
    c_poly = df64_add(DF64_COS_C12, df64_mul(r2, c_poly));
    c_poly = df64_add(DF64_COS_C10, df64_mul(r2, c_poly));
    c_poly = df64_add(DF64_COS_C8, df64_mul(r2, c_poly));
    c_poly = df64_add(DF64_COS_C6, df64_mul(r2, c_poly));
    c_poly = df64_add(DF64_COS_C4, df64_mul(r2, c_poly));
    c_poly = df64_add(DF64_COS_C2, df64_mul(r2, c_poly));
    df64 c_val = df64_add(DF64_ONE, df64_mul(r2, c_poly));

    int m = k % 4;
    if (m < 0) m += 4;
    if (m == 0) {
        return df64_sc{s_val, c_val};
    } else if (m == 1) {
        return df64_sc{c_val, df64_neg(s_val)};
    } else if (m == 2) {
        return df64_sc{df64_neg(s_val), df64_neg(c_val)};
    } else {
        return df64_sc{df64_neg(c_val), s_val};
    }
}

inline df64 df64_sin(df64 x) {
    return df64_sincos(x).s;
}

inline df64 df64_cos(df64 x) {
    return df64_sincos(x).c;
}

inline df64 df64_atan2(df64 y, df64 x) {
    if (df64_isnan(y) || df64_isnan(x)) {
        return df64{NAN, NAN};
    }

    int sy = signbit(y.hi) ? 1 : 0;
    int sx = signbit(x.hi) ? 1 : 0;

    if (y.hi == 0.0f && y.lo == 0.0f) {
        if (x.hi == 0.0f && x.lo == 0.0f) {
            if (sx == 0) {
                return df64{sy ? -0.0f : 0.0f, 0.0f};
            } else {
                return sy ? df64_neg(DF64_PI) : DF64_PI;
            }
        }
        if (sx == 0) {
            return df64{sy ? -0.0f : 0.0f, 0.0f};
        } else {
            return sy ? df64_neg(DF64_PI) : DF64_PI;
        }
    }

    if (x.hi == 0.0f && x.lo == 0.0f) {
        return sy ? df64_neg(DF64_HALF_PI) : DF64_HALF_PI;
    }

    float t0 = atan2(y.hi, x.hi);
    df64 theta = df64_from_float(t0);

    for (int step = 0; step < 2; ++step) {
        df64_sc sc = df64_sincos(theta);
        df64 num = df64_sub(df64_mul(y, sc.c), df64_mul(x, sc.s));
        df64 den = df64_add(df64_mul(x, sc.c), df64_mul(y, sc.s));
        df64 delta = df64_div(num, den);
        theta = df64_add(theta, delta);
    }

    return theta;
}

inline df64 df64_acos(df64 x) {
    if (df64_isnan(x)) {
        return df64{NAN, NAN};
    }
    if (df64_gt(x, DF64_ONE) || df64_lt(x, df64_neg(DF64_ONE))) {
        return df64{NAN, NAN};
    }
    if (df64_cmp(x, DF64_ONE) == 0) {
        return DF64_ZERO;
    }
    if (df64_cmp(x, df64_neg(DF64_ONE)) == 0) {
        return DF64_PI;
    }

    df64 one_minus_x = df64_sub(DF64_ONE, x);
    df64 one_plus_x = df64_add(DF64_ONE, x);
    df64 prod = df64_mul(one_minus_x, one_plus_x);
    if (prod.hi < 0.0f || (prod.hi == 0.0f && prod.lo < 0.0f)) {
        prod = DF64_ZERO;
    }
    df64 y = df64_sqrt(prod);
    return df64_atan2(y, x);
}

inline df64 df64_pow15(df64 x) {
    if (df64_isnan(x) || x.hi < 0.0f || (x.hi == 0.0f && x.lo < 0.0f)) {
        return df64{NAN, NAN};
    }
    if (x.hi == 0.0f && x.lo == 0.0f) {
        return DF64_ZERO;
    }
    df64 s = df64_sqrt(x);
    return df64_mul(x, s);
}

kernel void k_df64_probe2(
    device const ulong* in [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant uint& op [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    ulong raw_a = in[2 * gid];
    ulong raw_b = in[2 * gid + 1];
    df64 a = df64_from_bits(raw_a);
    df64 b = df64_from_bits(raw_b);
    ulong res = 0;
    switch (op) {
        case 0: res = df64_to_bits(df64_sin(a)); break;
        case 1: res = df64_to_bits(df64_cos(a)); break;
        case 2: res = df64_to_bits(df64_atan2(a, b)); break;
        case 3: res = df64_to_bits(df64_acos(a)); break;
        case 4: res = df64_to_bits(df64_pow15(a)); break;
        default: break;
    }
    out[gid] = res;
}
"""#
}
