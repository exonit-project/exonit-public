import Foundation

/// Metal Shading Language source for the V3 K2 order choice, the fold-bin ladder and the `nAff`
/// prefix sum (card M2.3).
///
/// This text is concatenated LAST: after `FPPV3MetalDF64`, `FPPV3MetalDF64Trig`,
/// `FPPV3MetalK1Scan`, `FPPV3MetalK1`, `FPPV3MetalK2Flux` and `FPPV3MetalK2Window`. It uses, from
/// those texts, `df64` / `df64_from_bits` / `df64_to_float`, `K1Out`, `K1Params`, `FoldParams`,
/// `k2_window_mean` (the SIMD-group window mean of the deficit), and `k2_neumaier_add` (the
/// compensated f32 accumulator). The guarded prelude is repeated so the text is self-contained
/// apart from those dependencies.
///
/// All three kernels are SIMD-group kernels: one group of 32 lanes per draw (`k2_order`) or per
/// affected bin (`k2_ladder`). `k2_window_mean` is called by every lane with the same arguments and
/// returns the same value on all lanes, so the control flow of `chooseOrder` and `ladderBin` is
/// uniform across the group and no cross-lane communication is needed here.
public enum FPPV3MetalK2Ladder {
    public static let text = #"""
#ifndef FPPV3_MSL_PRELUDE
#define FPPV3_MSL_PRELUDE
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#endif

// Card M2.3: K2 `chooseOrder`, the fold-bin ladder and the `nAff` prefix sum.
// Concatenated last (see FPPV3MetalK2LadderSource.swift).
#pragma clang fp contract(off)

// FPPV2Consts.glSmoothIdx / glOracleIdx: the CPU order indices of the initial choice and of the
// oracle (m = 128).
constant uint K2_GL_SMOOTH_IDX = 2u;
constant uint K2_GL_ORACLE_IDX = 10u;
// FPPV2Consts.ladderExactLevel / ladderAcceptMaxLevel / ladderMinPoints.
constant uint K2_LADDER_EXACT = 7u;
constant uint K2_LADDER_ACCEPT = 5u;
constant uint K2_LADDER_MIN_POINTS = 8u;

// Deficit units; covers the p99.9 kernel tier of window means.
#define FPP_LADDER_BAND 5.0e-7f

// Interval decision: each rung difference is known to +-FPP_LADDER_BAND. Decide in f32 only when the CPU test
// has the same outcome over the whole interval; otherwise the bin is redone on the host in f64.
inline bool k2_ladder_accept_f32(float d1, float d2, float epsB, thread bool& recheck) {
    if (!isfinite(d1) || !isfinite(d2)) {
        recheck = true;
        return false;
    }
    const float B = FPP_LADDER_BAND;
    float hi1 = d1 + B;
    float hi2 = d2 + B;
    float lo1 = max(d1 - B, 0.0f);
    float lo2 = max(d2 - B, 0.0f);
    float small = 0.05f * epsB;
    if (lo1 > epsB || lo2 > epsB) {
        return false;
    }
    bool under = hi1 <= epsB && hi2 <= epsB;
    bool small_yes = hi1 < small && hi2 < small;
    bool small_no = lo1 >= small || lo2 >= small;
    bool ord_yes = hi2 <= lo1;
    bool ord_no = lo2 > hi1;
    if (under && (ord_yes || small_yes)) {
        return true;
    }
    if (ord_no && small_no) {
        return false;
    }
    recheck = true;
    return false;
}

// B8: one row per draw. orderStatus = orderIdx | (status << 8).
struct K2Draw {
    uint orderStatus;   // bits 0-7 orderIdx (0..10), bits 8-15 status (0, 1, 2)
    uint nEval;         // window means evaluated by k2_order (diagnostic)
};

// B9: one row per affected bin, at binOff[j] + i for the i-th affected bin of draw j.
struct BinOut {
    float D;            // model DEFICIT of the bin (D5): 1 - M_b, never formed as 1 - flux
    uint levelNwin;     // bits 0-7 ladder level (0..7), bits 8-23 windows evaluated
};

// Result of one `ladderBin` call (the CPU tuple `(m, nwin, level)`, in deficit form).
struct K2LadderOut {
    float D;
    uint level;
    uint nwin;
};

// The contacts of a K1 row: the finite roots among root[0..3] (t1 t2 t3 t4), sorted ascending.
// `tMin` (root[4]) is not a contact. Insertion sort keeps the order deterministic; at most four
// elements compare, so no data-dependent divergence beyond the (uniform) values themselves.
inline uint k2_contacts(thread df64* ct, device const K1Out* row) {
    uint nct = 0u;
    for (uint i = 0u; i < 4u; i++) {
        df64 t = df64_from_bits(row->root[i]);
        if (df64_isnan(t)) {
            continue;
        }
        uint p = nct;
        while (p > 0u && df64_lt(t, ct[p - 1u])) {
            ct[p] = ct[p - 1u];
            p -= 1u;
        }
        ct[p] = t;
        nct += 1u;
    }
    return nct;
}

// Group -> draw: the largest j with binOff[j] <= g (upper bound minus one, over binOff[0..n]).
// Because g < binOff[n], that j is the draw that owns group g (zero-`nAff` draws can never own a
// dispatched group, so the "ties resolved to the last j with nAff > 0" rule holds by construction).
inline uint k2_find_draw(device const uint* binOff, uint n, uint g) {
    uint lo = 0u;
    uint hi = n;
    while (lo < hi) {
        uint mid = (lo + hi + 1u) >> 1;
        if (binOff[mid] <= g) {
            lo = mid;
        } else {
            hi = mid - 1u;
        }
    }
    return lo;
}

// The CPU `FPPV2GL.chooseOrder` on deficit window means. Every lane calls it with the same
// arguments (the window means are identical on all lanes), so the early returns are uniform.
// nEval counts the window means evaluated (the diagnostic of K2Draw.nEval).
inline void k2_choose_order(thread const df64* ct, uint nct, df64 w, float eps,
                            device const float* dc, device const float2* glx,
                            device const float* glw, device const int* orders, uint lane,
                            thread uint& outIdx, thread uint& outStatus, thread uint& outNEval) {
    if (nct == 0u) {
        outIdx = K2_GL_SMOOTH_IDX;
        outStatus = 0u;
        outNEval = 0u;
        return;
    }
    uint chosen = K2_GL_SMOOTH_IDX;
    uint nEval = 0u;
    for (uint ci = 0u; ci < nct; ci++) {
        df64 c = ct[ci];
        float v0 = k2_window_mean(c, w, ct, nct, 0u, dc, glx, glw, orders, lane);
        nEval += 1u;
        float v1 = k2_window_mean(c, w, ct, nct, 1u, dc, glx, glw, orders, lane);
        nEval += 1u;
        float v2 = k2_window_mean(c, w, ct, nct, 2u, dc, glx, glw, orders, lane);
        nEval += 1u;
        int found = -1;
        for (int i = 0; i < 7; i++) {
            if (fabs(v0 - v1) <= eps && fabs(v0 - v2) <= eps) {
                found = i;
                break;
            }
            float v3 = k2_window_mean(c, w, ct, nct, (uint)(i + 3), dc, glx, glw, orders, lane);
            nEval += 1u;
            v0 = v1;
            v1 = v2;
            v2 = v3;
        }
        if (found < 0) {
            // v0, v1, v2 now hold I_7, I_8, I_9 (m = 24, 32, 64); the oracle I_10 (m = 128) enters.
            float v3 = k2_window_mean(c, w, ct, nct, K2_GL_ORACLE_IDX, dc, glx, glw, orders, lane);
            nEval += 1u;
            if (fabs(v0 - v1) <= eps && fabs(v0 - v2) <= eps) {
                found = 7;
            } else if (fabs(v1 - v2) <= eps && fabs(v1 - v3) <= eps) {
                found = 8;
            } else if (fabs(v2 - v3) <= eps) {
                outIdx = K2_GL_ORACLE_IDX;   // plateau holds: oracle fallback
                outStatus = 1u;
                outNEval = nEval;
                return;
            } else {
                outIdx = K2_GL_ORACLE_IDX;   // 64 vs 128 disagree: unconverged
                outStatus = 2u;
                outNEval = nEval;
                return;
            }
        }
        if ((uint)found > chosen) {
            chosen = (uint)found;
        }
    }
    outIdx = chosen;
    outStatus = 0u;
    outNEval = nEval;
}

// CUDA k2_signed_rung_pair: reference = vals[0], lower rung negative, upper positive.
// Keep one compensated pair per lane across all windows of both rungs, then reduce once.
// nwin counts every recomputed window as well as the windows used for accepted values.
inline float k2_signed_rung_pair(uint b, uint lowerRung, uint upperRung, float reference,
                                  uint npts, uint loP, df64 w, df64 wEff, uint orderIdx,
                                  thread const df64* ct, uint nct,
                                  device const ulong* foldBuf, thread const uint* Fv,
                                  device const float* dc, device const float2* glx,
                                  device const float* glw, device const int* orders, uint lane,
                                  thread uint& nwin) {
    float s = 0.0f;
    float c = 0.0f;
    for (uint side = 0u; side < 2u; side++) {
        uint rung = side == 0u ? lowerRung : upperRung;
        float sign = side == 0u ? -1.0f : 1.0f;
        uint count = rung == 0u ? 1u :
            (rung < K2_LADDER_EXACT ? (4u << (rung - 1u)) : npts);
        df64 width = rung == 0u ? wEff : w;
        float signedScale = (sign / (float)count) / df64_to_float(width);
        for (uint item = 0u; item < count; item++) {
            df64 centre;
            if (rung == 0u) {
                centre = df64_from_bits(foldBuf[Fv[3] + b]);
            } else if (rung < K2_LADDER_EXACT) {
                centre = df64_from_bits(
                    foldBuf[Fv[7] + (b * Fv[10] + rung - 1u) * Fv[11] + item]);
            } else {
                centre = df64_from_bits(foldBuf[Fv[9] + loP + item]);
            }
            k2_window_terms(centre, width, ct, nct, orderIdx, dc, glx, glw, orders, lane,
                            true, signedScale, reference, s, c);
        }
        nwin += count;
    }
    return k2_reduce_pair(s, c, lane);
}

// The CPU `FPPV2Ladder.ladderBin` on deficit window means, in f32. `vals[]` and every level mean
// are compensated f32 sums (Neumaier pair, fixed order); comparisons use centred signed node terms.
// Every accumulation is over the windows of one bin only, so the order is fixed and identical on
// all lanes. Non-finite means are not clipped: they propagate to D.
inline K2LadderOut k2_ladder_bin(uint b, df64 w, float epsB, uint orderIdx,
                                 thread const df64* ct, uint nct,
                                 device const ulong* foldBuf, thread const uint* Fv,
                                 device const float* dc, device const float2* glx,
                                 device const float* glw, device const int* orders, uint lane) {
    K2LadderOut out;
    uint loP = (uint)(long)foldBuf[Fv[8] + b];
    uint hiP = (uint)(long)foldBuf[Fv[8] + b + 1u];
    uint npts = hiP - loP;
    int nb = (int)(long)foldBuf[Fv[1] + b];  // v[1] = offN (v[0] is nb, the bin count)
    uint nwin = 0u;

    if (nb <= (int)K2_LADDER_MIN_POINTS) {
        float s = 0.0f;
        float c = 0.0f;
        for (uint i = 0u; i < npts; i++) {
            df64 t = df64_from_bits(foldBuf[Fv[9] + loP + i]);
            k2_neumaier_add(
                s, c, k2_window_mean(t, w, ct, nct, orderIdx, dc, glx, glw, orders, lane));
        }
        nwin += npts;
        out.D = (s + c) / (float)npts;
        out.level = K2_LADDER_EXACT;
        out.nwin = nwin;
        return out;
    }

    // wEff = sqrt(w*w + 12 varT[b]) in df64; every window centre is a df64 from its bits.
    df64 varT = df64_from_bits(foldBuf[Fv[4] + b]);
    df64 tmean = df64_from_bits(foldBuf[Fv[3] + b]);
    df64 wEff = df64_sqrt(df64_add(df64_mul(w, w), df64_mul_f(varT, 12.0f)));

    float vals[8];
    vals[0] = k2_window_mean(tmean, wEff, ct, nct, orderIdx, dc, glx, glw, orders, lane);
    nwin += 1u;
    for (uint lvl = 1u; lvl < K2_LADDER_EXACT; lvl++) {
        uint kk = 4u << (lvl - 1u);
        uint lrung = lvl - 1u;
        float s = 0.0f;
        float c = 0.0f;
        for (uint jn = 0u; jn < kk; jn++) {
            df64 centre = df64_from_bits(
                foldBuf[Fv[7] + (b * Fv[10] + lrung) * Fv[11] + jn]);
            k2_neumaier_add(
                s, c, k2_window_mean(centre, w, ct, nct, orderIdx, dc, glx, glw, orders, lane));
        }
        vals[lvl] = (s + c) / (float)kk;
        nwin += kk;
        if (lvl >= 2u && lvl <= K2_LADDER_ACCEPT + 2u) {
            uint cc = lvl - 2u;
            float d1 = fabs(k2_signed_rung_pair(
                b, cc, cc + 1u, vals[0], npts, loP, w, wEff, orderIdx,
                ct, nct, foldBuf, Fv, dc, glx, glw, orders, lane, nwin));
            float d2 = fabs(k2_signed_rung_pair(
                b, cc + 1u, cc + 2u, vals[0], npts, loP, w, wEff, orderIdx,
                ct, nct, foldBuf, Fv, dc, glx, glw, orders, lane, nwin));
            bool recheck = false;
            if (k2_ladder_accept_f32(d1, d2, epsB, recheck)) {
                out.D = vals[cc];
                out.level = cc;
                out.nwin = nwin;
                return out;
            }
            if (recheck) {
                out.D = 0.0f;
                out.level = 255u;
                out.nwin = nwin;
                return out;
            }
        }
    }

    float s = 0.0f;
    float c = 0.0f;
    for (uint i = 0u; i < npts; i++) {
        df64 t = df64_from_bits(foldBuf[Fv[9] + loP + i]);
        k2_neumaier_add(s, c, k2_window_mean(t, w, ct, nct, orderIdx, dc, glx, glw, orders, lane));
    }
    nwin += npts;
    vals[K2_LADDER_EXACT] = (s + c) / (float)npts;
    uint cc = K2_LADDER_ACCEPT;
    float d1 = fabs(k2_signed_rung_pair(
        b, cc, cc + 1u, vals[0], npts, loP, w, wEff, orderIdx,
        ct, nct, foldBuf, Fv, dc, glx, glw, orders, lane, nwin));
    float d2 = fabs(k2_signed_rung_pair(
        b, cc + 1u, cc + 2u, vals[0], npts, loP, w, wEff, orderIdx,
        ct, nct, foldBuf, Fv, dc, glx, glw, orders, lane, nwin));
    bool recheck = false;
    if (k2_ladder_accept_f32(d1, d2, epsB, recheck)) {
        out.D = vals[cc];
        out.level = cc;
    } else if (recheck) {
        out.D = 0.0f;
        out.level = 255u;
    } else {
        out.D = vals[K2_LADDER_EXACT];
        out.level = K2_LADDER_EXACT;
    }
    out.nwin = nwin;
    return out;
}

// ------------------------------------------------------------------ k_prefix_naff (B6)
// ONE threadgroup (the host dispatches 1024 threads). The prefix is a plain serial scan by thread
// 0 over the chunk: binOff[0] = 0, binOff[j+1] = binOff[j] + k1Out[j].nAff. Integer adds only, so
// the result is exact and its order does not matter for a serial scan. No shared memory and no
// barrier are needed, since only thread 0 touches the buffer.
kernel void k_prefix_naff(device const K1Out* k1Out [[buffer(0)]], constant K1Params& P [[buffer(1)]],
                          device uint* binOff [[buffer(2)]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tg [[threads_per_threadgroup]]) {
    if (tid == 0u) {
        uint acc = 0u;
        binOff[0] = 0u;
        for (uint j = 0u; j < P.n; j++) {
            acc += k1Out[j].nAff;
            binOff[j + 1u] = acc;
        }
    }
}

// ------------------------------------------------------------------ k2_order (B8)
// Dispatch P.n * 32 threads; group j = gid / 32 runs `chooseOrder` for draw j.
kernel void k2_order(device const K1Out* k1Out [[buffer(0)]], constant K1Params& P [[buffer(1)]],
                     device const float* drawC [[buffer(2)]], device const float2* glx [[buffer(3)]],
                     device const float* glw [[buffer(4)]], device const int* orders [[buffer(5)]],
                     device K2Draw* k2Draw [[buffer(6)]],
                     uint gid [[thread_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]]) {
    uint j = gid / 32u;
    if (j >= P.n) {
        return;
    }
    device const K1Out* row = &k1Out[j];
    if (row->k1State != 1u) {
        if (lane == 0u) {
            k2Draw[j].orderStatus = K2_GL_SMOOTH_IDX;   // orderIdx 2, status 0
            k2Draw[j].nEval = 0u;
        }
        return;
    }

    df64 ct[4];
    uint nct = k2_contacts(ct, row);
    df64 w = df64_from_bits(P.foldW);
    device const float* dc = drawC + j * 16u;

    uint idx = K2_GL_SMOOTH_IDX;
    uint status = 0u;
    uint nEval = 0u;
    k2_choose_order(ct, nct, w, row->epsMin, dc, glx, glw, orders, lane, idx, status, nEval);
    if (lane == 0u) {
        k2Draw[j].orderStatus = idx | (status << 8);
        k2Draw[j].nEval = nEval;
    }
}

// ------------------------------------------------------------------ k2_ladder (B9)
// Dispatch binOff[P.n] * 32 threads; group g = gid / 32 is the (g - binOff[j])-th affected bin of
// draw j = k2_find_draw(g).
kernel void k2_ladder(device const K1Out* k1Out [[buffer(0)]], constant K1Params& P [[buffer(1)]],
                      device const float* drawC [[buffer(2)]], device const ulong* foldBuf [[buffer(3)]],
                      constant FoldParams& F [[buffer(4)]], device const float2* glx [[buffer(5)]],
                      device const float* glw [[buffer(6)]], device const int* orders [[buffer(7)]],
                      device const K2Draw* k2Draw [[buffer(8)]], device const uint* binOff [[buffer(9)]],
                      device BinOut* binOut [[buffer(10)]],
                      uint gid [[thread_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint g = gid / 32u;
    uint total = binOff[P.n];
    if (g >= total) {
        return;
    }
    uint j = k2_find_draw(binOff, P.n, g);
    device const K1Out* row = &k1Out[j];
    uint b = row->firstAff + (g - binOff[j]);

    df64 ct[4];
    uint nct = k2_contacts(ct, row);
    df64 w = df64_from_bits(P.foldW);
    // epsB = sigma[b] * epsRef, both read at the precision the device holds (f32 product).
    float epsB = df64_to_float(df64_from_bits(foldBuf[F.v[2] + b])) * row->epsRef;
    uint orderIdx = k2Draw[j].orderStatus & 0xffu;

    uint Fv[16];
    for (uint i = 0u; i < 16u; i++) {
        Fv[i] = F.v[i];
    }

    device const float* dc = drawC + j * 16u;
    K2LadderOut r = k2_ladder_bin(
        b, w, epsB, orderIdx, ct, nct, foldBuf, Fv, dc, glx, glw, orders, lane);
    if (lane == 0u) {
        binOut[g].D = r.D;
        binOut[g].levelNwin = r.level | (r.nwin << 8);
    }
}
"""#
}
