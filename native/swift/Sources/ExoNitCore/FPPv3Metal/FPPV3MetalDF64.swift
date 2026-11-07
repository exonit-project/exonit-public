import Foundation

/// MSL source text holder for double-float (df64) arithmetic.
public enum FPPV3MetalDF64 {
    public static let text: String = #"""
#ifndef FPPV3_MSL_PRELUDE
#define FPPV3_MSL_PRELUDE
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#endif

// Metal flushes f32 subnormals: lo is lost below |x| ~ 2^-78 (3e-24), where the split keeps only hi. Far below any ExoNit input.
struct df64 {
    float hi;
    float lo;
};

inline df64 df_two_sum(float a, float b) {
    float s = a + b;
    float v = s - a;
    float e = (a - (s - v)) + (b - v);
    return df64{s, e};
}

inline df64 df_quick_two_sum(float a, float b) {
    float s = a + b;
    float e = b - (s - a);
    return df64{s, e};
}

inline df64 df_two_prod(float a, float b) {
    float p = a * b;
    float e = fma(a, b, -p);
    return df64{p, e};
}

inline df64 df64_make(float hi, float lo) {
    return df_quick_two_sum(hi, lo);
}

inline df64 df64_from_float(float x) {
    return df64{x, 0.0f};
}

inline float df64_to_float(df64 a) {
    return a.hi;
}

inline df64 df64_neg(df64 a) {
    return df64{-a.hi, -a.lo};
}

inline bool df64_isnan(df64 a) {
    return isnan(a.hi) || isnan(a.lo);
}

inline bool df64_isfinite(df64 a) {
    return isfinite(a.hi) && isfinite(a.lo);
}

inline df64 df64_abs(df64 a) {
    if (df64_isnan(a)) return a;
    if (a.hi < 0.0f || (a.hi == 0.0f && (signbit(a.hi) || signbit(a.lo)))) {
        return df64_neg(a);
    }
    return a;
}

inline df64 df64_add(df64 a, df64 b) {
    df64 s = df_two_sum(a.hi, b.hi);
    df64 t = df_two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = df_quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return df_quick_two_sum(s.hi, s.lo);
}

inline df64 df64_sub(df64 a, df64 b) {
    return df64_add(a, df64_neg(b));
}

inline df64 df64_mul_f(df64 a, float b) {
    df64 p = df_two_prod(a.hi, b);
    p.lo += a.lo * b;
    return df_quick_two_sum(p.hi, p.lo);
}

inline df64 df64_add_f(df64 a, float b) {
    return df64_add(a, df64_from_float(b));
}

inline df64 df64_mul(df64 a, df64 b) {
    df64 p = df_two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return df_quick_two_sum(p.hi, p.lo);
}

inline df64 df64_div(df64 a, df64 b) {
    float q1 = a.hi / b.hi;
    df64 r = df64_sub(a, df64_mul_f(b, q1));
    float q2 = r.hi / b.hi;
    r = df64_sub(r, df64_mul_f(b, q2));
    float q3 = r.hi / b.hi;
    df64 q = df_quick_two_sum(q1, q2);
    q.lo += q3;
    return df_quick_two_sum(q.hi, q.lo);
}

inline df64 df64_sqrt(df64 a) {
    if (df64_isnan(a) || a.hi < 0.0f || (a.hi == 0.0f && a.lo < 0.0f)) {
        return df64{NAN, NAN};
    }
    if (a.hi == 0.0f && a.lo == 0.0f) {
        return df64{0.0f, 0.0f};
    }
    float x = 1.0f / sqrt(a.hi);
    float ax = a.hi * x;
    df64 diff = df64_sub(a, df_two_prod(ax, ax));
    return df_two_sum(ax, diff.hi * x * 0.5f);
}

inline int df64_cmp(df64 a, df64 b) {
    if (df64_isnan(a) || df64_isnan(b)) return 2;
    if (a.hi < b.hi) return -1;
    if (a.hi > b.hi) return 1;
    if (a.lo < b.lo) return -1;
    if (a.lo > b.lo) return 1;
    return 0;
}

inline bool df64_lt(df64 a, df64 b) { return df64_cmp(a, b) == -1; }
inline bool df64_le(df64 a, df64 b) { int c = df64_cmp(a, b); return c == -1 || c == 0; }
inline bool df64_gt(df64 a, df64 b) { return df64_cmp(a, b) == 1; }
inline bool df64_ge(df64 a, df64 b) { int c = df64_cmp(a, b); return c == 1 || c == 0; }

inline df64 df64_floor(df64 a) {
    if (!df64_isfinite(a)) return a;
    float f_hi = floor(a.hi);
    if (f_hi == a.hi) {
        float f_lo = floor(a.lo);
        return df_quick_two_sum(f_hi, f_lo);
    } else {
        return df64{f_hi, 0.0f};
    }
}

inline df64 df64_rint(df64 a) {
    if (!df64_isfinite(a)) return a;
    float f_hi = floor(a.hi);
    if (f_hi == a.hi) {
        float i_lo = floor(a.lo);
        float frac = a.lo - i_lo;
        float add = 0.0f;
        if (frac > 0.5f) {
            add = 1.0f;
        } else if (frac == 0.5f) {
            // parity of hi + i_lo, without forming the sum: above 2^24 it does not fit an f32
            float s = fmod(abs(a.hi), 2.0f) + fmod(abs(i_lo), 2.0f);
            if (s == 1.0f) add = 1.0f;
        }
        float r_lo = i_lo + add;
        return df_quick_two_sum(a.hi, r_lo);
    } else {
        float frac_hi = a.hi - f_hi;
        if (frac_hi < 0.5f) return df64{f_hi, 0.0f};
        if (frac_hi > 0.5f) return df64{f_hi + 1.0f, 0.0f};
        if (a.lo < 0.0f) return df64{f_hi, 0.0f};
        if (a.lo > 0.0f) return df64{f_hi + 1.0f, 0.0f};
        float s = fmod(abs(f_hi), 2.0f);
        return df64{s == 1.0f ? f_hi + 1.0f : f_hi, 0.0f};
    }
}

inline df64 df64_from_bits(ulong bits) {
    ulong sign = bits >> 63;
    ulong e = (bits >> 52) & 0x7ff;
    ulong m = bits & ((1ULL << 52) - 1ULL);
    if (e == 0x7ff) {
        if (m != 0) {
            return df64{NAN, 0.0f};
        }
        return df64{sign ? -INFINITY : INFINITY, 0.0f};
    }
    if (e == 0) {
        return df64{sign ? -0.0f : 0.0f, 0.0f};
    }
    int unbiased_e = (int)e - 1023;
    if (unbiased_e >= 128) {
        return df64{sign ? -INFINITY : INFINITY, 0.0f};
    }
    if (unbiased_e < -126) {
        return df64{sign ? -0.0f : 0.0f, 0.0f};
    }
    ulong M = (1ULL << 52) | m;
    ulong q = M >> 29;
    ulong rem = M & ((1ULL << 29) - 1ULL);
    if (rem > (1ULL << 28) || (rem == (1ULL << 28) && (q & 1ULL))) {
        q += 1ULL;
    }
    float hi = ldexp((float)q, (int)e - 1046);
    long R = (long)M - (long)(q << 29);
    float lo = ldexp((float)R, (int)e - 1075);
    if (sign) {
        hi = -hi;
        lo = -lo;
    }
    return df64{hi, lo};
}

inline ulong df64_to_bits_exact(df64 a) {
    if (df64_isnan(a)) {
        return 0x7ff8000000000000ULL;
    }
    if (isinf(a.hi)) {
        ulong sign = signbit(a.hi) ? 1ULL : 0ULL;
        return (sign << 63) | 0x7ff0000000000000ULL;
    }
    if (a.hi == 0.0f) {
        ulong sign = signbit(a.hi) ? 1ULL : 0ULL;
        return sign << 63;
    }
    uint u_hi = as_type<uint>(a.hi);
    ulong sign_hi = (ulong)(u_hi >> 31);
    int exp_hi = (int)((u_hi >> 23) & 0xff);
    uint mant_hi = u_hi & 0x7fffffu;
    ulong M_hi = (1ULL << 23) | (ulong)mant_hi;
    int e_hi = exp_hi - 127;

    if (a.lo == 0.0f) {
        ulong e_f64 = (ulong)(e_hi + 1023);
        return (sign_hi << 63) | (e_f64 << 52) | ((ulong)mant_hi << 29);
    }

    uint u_lo = as_type<uint>(a.lo);
    ulong sign_lo = (ulong)(u_lo >> 31);
    int exp_lo = (int)((u_lo >> 23) & 0xff);
    uint mant_lo = u_lo & 0x7fffffu;
    ulong M_lo;
    int e_lo;
    if (exp_lo > 0) {
        M_lo = (1ULL << 23) | (ulong)mant_lo;
        e_lo = exp_lo - 127;
    } else {
        M_lo = (ulong)mant_lo;
        e_lo = -126;
    }

    int shift = e_hi - e_lo;
    ulong scaled_hi = M_hi << 37;
    ulong scaled_lo = 0;
    ulong sticky = 0;
    if (shift <= 60) {
        if (37 - shift >= 0) {
            scaled_lo = M_lo << (37 - shift);
        } else {
            int right = shift - 37;
            scaled_lo = M_lo >> right;
            sticky = (M_lo & ((1ULL << right) - 1ULL)) ? 1ULL : 0ULL;
        }
    } else {
        sticky = 1ULL;
    }

    long val_hi = sign_hi ? -(long)scaled_hi : (long)scaled_hi;
    long val_lo = sign_lo ? -(long)scaled_lo : (long)scaled_lo;
    long total = val_hi + val_lo;
    if (sticky) {
        total += sign_lo ? -1L : 1L;
    }

    ulong res_sign = total < 0 ? 1ULL : 0ULL;
    ulong abs_total = (ulong)(total < 0 ? -total : total);
    if (abs_total == 0) {
        return res_sign << 63;
    }

    int msb = 63 - clz(abs_total);
    int res_exp = e_hi + (msb - 60) + 1023;

    int round_pos = msb - 53;
    ulong M_53;
    if (round_pos >= 0) {
        M_53 = abs_total >> (round_pos + 1);
        ulong round_bit = (abs_total >> round_pos) & 1ULL;
        ulong sticky_bits = abs_total & ((1ULL << round_pos) - 1ULL);
        if (round_bit && (sticky_bits || (M_53 & 1ULL))) {
            M_53 += 1ULL;
            if (M_53 >= (1ULL << 53)) {
                M_53 >>= 1;
                res_exp += 1;
            }
        }
    } else {
        M_53 = abs_total << (-round_pos - 1);
    }
    ulong mant_52 = M_53 & ((1ULL << 52) - 1ULL);
    return (res_sign << 63) | ((ulong)res_exp << 52) | mant_52;
}

inline ulong df64_to_bits(df64 a) {
    if (!df64_isfinite(a) || (a.hi == 0.0f && a.lo == 0.0f)) {
        return df64_to_bits_exact(a);
    }
    return df64_to_bits_exact(df_quick_two_sum(a.hi, a.lo));
}

constant df64 DF64_PI = df64{0x1.921fb6p+1f, -0x1.777a5cp-24f};
constant df64 DF64_TWO_PI = df64{0x1.921fb6p+2f, -0x1.777a5cp-23f};
constant df64 DF64_HALF_PI = df64{0x1.921fb6p+0f, -0x1.777a5cp-25f};
constant df64 DF64_ONE = df64{1.0f, 0.0f};
constant df64 DF64_ZERO = df64{0.0f, 0.0f};

kernel void k_df64_probe(
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
        case 0: res = df64_to_bits(df64_add(a, b)); break;
        case 1: res = df64_to_bits(df64_sub(a, b)); break;
        case 2: res = df64_to_bits(df64_mul(a, b)); break;
        case 3: res = df64_to_bits(df64_div(a, b)); break;
        case 4: res = df64_to_bits(df64_sqrt(a)); break;
        case 5: res = df64_to_bits(df64_floor(a)); break;
        case 6: res = df64_to_bits(df64_rint(a)); break;
        case 7: res = df64_to_bits(a); break;
        case 8: {
            int c = df64_cmp(a, b);
            res = (ulong)(long)c;
            break;
        }
        case 9: res = df64_to_bits_exact(a); break;
        default: break;
    }
    out[gid] = res;
}
"""#
}
