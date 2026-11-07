import Foundation

/// Metal Shading Language source for the V3 K2 exposure-window mean of the model deficit (card M2.2).
///
/// This text is concatenated AFTER `FPPV3MetalDF64.text`, `FPPV3MetalDF64Trig.text` and
/// `FPPV3MetalK2Flux.text`. It uses, from those texts, the df64 arithmetic and the constants
/// `DF64_ONE` / `DF64_TWO_PI`, the p4.0 constants `PI_F` / `NEWTON_RTOL`, and `transit_deficit`
/// (the D5 deficit kernel of card M2.1). The guarded prelude is repeated so the text is
/// self-contained apart from those dependencies.
///
/// The window is cut at the contacts only, the same topology as `FPPV2GL.windowMean`: contacts at or
/// below the lower edge are skipped, a contact equal to the upper edge is not a cut, and pieces with
/// `l <= 0` are dropped. A node with `cos f < 0` has deficit exactly 0: the CPU `FPPV2Orbit.zOne`
/// returns `1e3` there, `transitFlux` returns 1 for `z >= 1 + p`, so the CPU flux is exactly 1 and
/// the CPU deficit exactly 0. No switch cut at `cos f = 0` is made (the device tests `cosF` itself).
public enum FPPV3MetalK2Window {
    public static let text = #"""
#ifndef FPPV3_MSL_PRELUDE
#define FPPV3_MSL_PRELUDE
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#endif

// Card M2.2: K2 piece placement in df64 and the SIMD-group (32 lanes) window mean of the deficit.
// Concatenated after DF64, DF64Trig and K2Flux (see FPPV3MetalK2WindowSource.swift).

constant uint K2_MAX_CT = 4;
constant uint K2_MAX_PIECES = 5;

// One piece of the exposure window [lo, hi], length l, and the CPU branch kind:
// 0 plain (neither end a contact), 1 both ends contacts, 2 lower end a contact only,
// 3 upper end a contact only.
struct K2Piece {
    df64 lo;
    df64 hi;
    df64 l;
    int kind;
};

// One quadrature node: time in df64, f32 weight (normalised for a mean, raw for a signed pair).
struct K2Node {
    df64 t;
    float wt;
};

// Pieces of the window [a, b] cut at the sorted finite contacts ct[0..nct), the loop of
// FPPV2GL.windowMean: contacts <= a are skipped, an interior contact is one with a < ct < b,
// pieces with l <= 0 are dropped. Writes at most K2_MAX_PIECES and returns the piece count.
inline uint k2_place_pieces(df64 a, df64 b, thread const df64* ct, uint nct, thread K2Piece* out) {
    uint nc = nct < K2_MAX_CT ? nct : K2_MAX_CT;
    uint jc = 0u;
    while (jc < nc && df64_le(ct[jc], a)) {
        jc += 1u;
    }
    df64 lo = a;
    bool loC = false;
    uint np = 0u;
    while (true) {
        df64 hi;
        bool hiC;
        if (jc < nc && df64_lt(ct[jc], b)) {
            hi = ct[jc];
            hiC = true;
            jc += 1u;
        } else {
            hi = b;
            hiC = false;
        }
        df64 l = df64_sub(hi, lo);
        if (df64_gt(l, DF64_ZERO)) {
            if (np < K2_MAX_PIECES) {
                out[np].lo = lo;
                out[np].hi = hi;
                out[np].l = l;
                out[np].kind = loC ? (hiC ? 1 : 2) : (hiC ? 3 : 0);
                np += 1u;
            }
        }
        if (!hiC) {
            break;
        }
        lo = hi;
        loC = true;
    }
    return np;
}

// Node time and f32 weight of node i (0 <= i < m) of order oi in piece pc: the three formulas of
// FPPV2GL.windowMean, evaluated in df64. x is the device's (hi, lo) split of the f64 node, gw the
// f32 weight widened to df64, invW = 1 / w in df64 for a mean, or 1 for raw signed terms.
// wt is rounded to f32 after this scaling.
inline K2Node k2_node(thread const K2Piece& pc, uint i, uint oi, df64 invW,
                      device const float2* glx, device const float* glw) {
    float2 x2 = glx[oi * 128u + i];
    df64 x = df64_make(x2.x, x2.y);
    df64 gw = df64_from_float(glw[oi * 128u + i]);
    K2Node nd;
    df64 wt;
    if (pc.kind == 0) {
        // t = halfL * x + mid, wt = halfL * gw
        df64 hl = df64_mul_f(pc.l, 0.5f);
        df64 mid = df64_mul_f(df64_add(pc.hi, pc.lo), 0.5f);
        nd.t = df64_add(df64_mul(hl, x), mid);
        wt = df64_mul(hl, gw);
    } else if (pc.kind == 1) {
        // u = 0.5 * (x + 1), t = lo + l * u * u * (3 - 2u), wt = 0.5 * gw * 6 * l * u * (1 - u)
        df64 u = df64_mul_f(df64_add(x, DF64_ONE), 0.5f);
        df64 lu = df64_mul(pc.l, u);
        df64 bend = df64_sub(df64_from_float(3.0f), df64_mul_f(u, 2.0f));
        nd.t = df64_add(pc.lo, df64_mul(df64_mul(lu, u), bend));
        wt = df64_mul_f(gw, 0.5f);
        wt = df64_mul_f(wt, 6.0f);
        wt = df64_mul(wt, pc.l);
        wt = df64_mul(wt, u);
        wt = df64_mul(wt, df64_sub(DF64_ONE, u));
    } else {
        // sL = sqrt(l), s = 0.5 * sL * (x + 1), wt = 0.5 * sL * gw * 2 * s,
        // t = (kind 2) lo + s * s else hi - s * s
        df64 sL = df64_sqrt(pc.l);
        df64 hl = df64_mul_f(sL, 0.5f);
        df64 s = df64_mul(hl, df64_add(x, DF64_ONE));
        df64 ss = df64_mul(s, s);
        nd.t = (pc.kind == 2) ? df64_add(pc.lo, ss) : df64_sub(pc.hi, ss);
        wt = df64_mul(hl, gw);
        wt = df64_mul_f(wt, 2.0f);
        wt = df64_mul(wt, s);
    }
    nd.wt = df64_to_float(df64_mul(wt, invW));
    return nd;
}

// dM for the A' orbit: dM = n * t in df64 (n = df64(drawC[0], drawC[14])), wrapped to (-pi, pi] by
// df64_rint, rounded ONCE to f32.
inline float k2_node_dM(df64 t, device const float* dc) {
    df64 n = df64_make(dc[0], dc[14]);
    df64 dM = df64_mul(n, t);
    df64 k = df64_rint(df64_div(dM, DF64_TWO_PI));
    dM = df64_sub(dM, df64_mul(k, DF64_TWO_PI));
    return df64_to_float(dM);
}

// ---------------------------------------------------------------- orbit (design A'), p4.0 z_one
// The host reduces every draw in f64 (nu_T = pi/2 - w, E_T, M_T, n = 2 pi / P, ...) and the device
// works with OFFSETS from mid-transit only: dM = n t, dE = E - E_T, f = nu - nu_T. No f32 quantity
// of order 1 is subtracted from another of order 1, so f keeps relative accuracy near transit.
// draw layout (16 floats): n, aRs, cosI, ecc, sinET, cosET, q, A, B, k, u1, u2, dilA, dilB, n_lo, 0.
// q = sqrt((1+e)/(1-e)); A = cos^2(E_T/2) + q^2 sin^2(E_T/2); B = (q^2-1) sin(E_T/2) cos(E_T/2).
// p4.0 returned z alone (its host cut the window at cos f = 0); this copy also returns cosF, and the
// caller (k2_deficit_at) tests cosF < 0, so the window keeps the CPU topology (contacts only).
inline float2 k2_z_one_cosf(float dM, device const float* d) {
    float twoPi = 2.0f * PI_F;
    float aRs = d[1], cosI = d[2], ecc = d[3];
    float sET = d[4], cET = d[5], q = d[6], A = d[7], B = d[8];
    dM = dM - twoPi * rint(dM / twoPi);  // wrap to (-pi, pi]
    float dE = dM;
    for (int i = 0; i < 12; i++) {
        float sh = sin(0.5f * dE), sd = sin(dE), cd = cos(dE);
        float g = dE - ecc * (cET * sd - 2.0f * sET * sh * sh) - dM;
        float gp = 1.0f - ecc * (cET * cd - sET * sd);
        float de = g / gp;
        dE -= de;
        if (fabs(de) <= NEWTON_RTOL * fabs(dE) + 1.0e-30f) break;
    }
    float sh = sin(0.5f * dE), ch = cos(0.5f * dE);
    float N = q * sh;
    float D = A * ch + B * sh;
    float n2 = N * N + D * D;
    float sinF = 2.0f * N * D / n2;
    float cosF = (D * D - N * N) / n2;
    float r = 1.0f - ecc * (cET * cos(dE) - sET * sin(dE));
    return float2(aRs * r * sqrt(sinF * sinF + cosI * cosI * cosF * cosF), cosF);
}

// Deficit of one node: the p4.0 z_one with cosF; cosF < 0 -> 0.0f (the CPU zOne returns 1e3 there
// and transitFlux 1, so the CPU deficit is exactly 0); else dilA * transit_deficit(z, k, u1, u2),
// never formed as 1 - flux.
inline float k2_deficit_at(float dM, device const float* dc) {
    float2 zc = k2_z_one_cosf(dM, dc);
    if (zc.y < 0.0f) {
        return 0.0f;
    }
    return dc[12] * transit_deficit(zc.x, dc[9], dc[10], dc[11]);
}

// ---------------------------------------------------------------- fixed-order reduction
// Neumaier add: (s, c) <- (s, c) + term, every step rounded to f32.
inline void k2_neumaier_add(thread float& s, thread float& c, float term) {
    float t = s + term;
    if (fabs(s) >= fabs(term)) {
        c += (s - t) + term;
    } else {
        c += (term - t) + s;
    }
    s = t;
}

// Knuth two-sum in f32: s + e == a + b exactly.
inline void k2_two_sum(float a, float b, thread float& s, thread float& e) {
    s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

// Merge A = (sa, ca) (lower lane) and B = (sb, cb) (higher lane) with two-sum, in a fixed order:
// (t, e) = two_sum(sa, sb); ctot = (e + ca) + cb; (s, c) = two_sum(t, ctot).
inline void k2_merge_pair(thread float& s, thread float& c, float sa, float ca, float sb, float cb) {
    float t, e;
    k2_two_sum(sa, sb, t, e);
    float ctot = (e + ca) + cb;
    k2_two_sum(t, ctot, s, c);
}

// Append this window's node terms to each lane's Neumaier pair. Signed comparisons
// centre every deficit BEFORE a window mean can round away the rung difference. A node
// behind the star still contributes its weight times (0 - reference).
inline void k2_window_terms(df64 tc, df64 w, thread const df64* ct, uint nct, uint orderIdx,
                            device const float* dc, device const float2* glx, device const float* glw,
                            device const int* orders, uint lane, bool signedTerms,
                            float signedScale, float reference, thread float& s, thread float& c) {
    df64 halfW = df64_mul_f(w, 0.5f);
    df64 a = df64_sub(tc, halfW);
    df64 b = df64_add(tc, halfW);
    K2Piece pieces[K2_MAX_PIECES];
    uint np = k2_place_pieces(a, b, ct, nct, pieces);
    uint m = (uint)orders[orderIdx];
    // Signed comparisons scale raw weights before centring; ordinary means retain their
    // original df64 normalisation and rounding.
    df64 invW = signedTerms ? DF64_ONE : df64_div(DF64_ONE, w);

    uint total = np * m;
    for (uint flat = lane; flat < total; flat += 32u) {
        uint pi = flat / m;
        uint i = flat - pi * m;
        K2Node nd = k2_node(pieces[pi], i, orderIdx, invW, glx, glw);
        float dM = k2_node_dM(nd.t, dc);
        float deficit = k2_deficit_at(dM, dc);
        float term;
        if (signedTerms) {
            float scaledWeight = signedScale * nd.wt;
            term = scaledWeight * (deficit - reference);
        } else {
            term = nd.wt * deficit;
        }
        k2_neumaier_add(s, c, term);
    }
}

// Fixed xor butterfly, lower lane first, unchanged from the window-mean reduction.
// For a signed pair this is called only after BOTH rungs have entered the lane pairs.
inline float k2_reduce_pair(float s, float c, uint lane) {
    for (uint off = 16u; off >= 1u; off >>= 1u) {
        float ps = simd_shuffle_xor(s, (ushort)off);
        float pcc = simd_shuffle_xor(c, (ushort)off);
        float sa, ca, sb, cb;
        if ((lane & off) == 0u) {
            sa = s;
            ca = c;
            sb = ps;
            cb = pcc;
        } else {
            sa = ps;
            ca = pcc;
            sb = s;
            cb = c;
        }
        k2_merge_pair(s, c, sa, ca, sb, cb);
    }
    return s + c;
}

// Mean deficit: all 32 lanes call with identical arguments and get the same result.
// Flattened (piece, node) indices go to lane L, L + 32, ... in their original order.
inline float k2_window_mean(df64 tc, df64 w, thread const df64* ct, uint nct, uint orderIdx,
                            device const float* dc, device const float2* glx, device const float* glw,
                            device const int* orders, uint lane) {
    float s = 0.0f;
    float c = 0.0f;
    k2_window_terms(tc, w, ct, nct, orderIdx, dc, glx, glw, orders, lane,
                    false, 0.0f, 0.0f, s, c);
    return k2_reduce_pair(s, c, lane);
}

// Probe: one SIMD-group of 32 threads per window. No shared memory is declared anywhere.
struct K2WinIn {
    ulong tcBits, wBits;
    ulong ct[4];
    uint nct, orderIdx, drawRow, pad;
};

kernel void k2_window_probe(device const K2WinIn* wins [[buffer(0)]], device const float* drawC [[buffer(1)]],
                            device const float2* glx [[buffer(2)]], device const float* glw [[buffer(3)]],
                            device const int* orders [[buffer(4)]], device float* out [[buffer(5)]],
                            uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint widx = gid / 32u;
    device const K2WinIn& win = wins[widx];
    df64 tc = df64_from_bits(win.tcBits);
    df64 w = df64_from_bits(win.wBits);
    df64 ct[K2_MAX_CT];
    for (uint i = 0u; i < K2_MAX_CT; i++) {
        ct[i] = df64_from_bits(win.ct[i]);
    }
    device const float* dc = drawC + (uint)win.drawRow * 16u;
    float res = k2_window_mean(tc, w, ct, win.nct, win.orderIdx, dc, glx, glw, orders, lane);
    if (lane == 0u) {
        out[widx] = res;
    }
}

#ifdef K2_DEBUG_PIECES
// Debug variant: the piece table of every window. dbg[w * 6] = piece count,
// dbg[w * 6 + 1 + p] = kind of piece p (0 for p >= count).
kernel void k2_window_pieces_probe(device const K2WinIn* wins [[buffer(0)]], device const float* drawC [[buffer(1)]],
                                   device const float2* glx [[buffer(2)]], device const float* glw [[buffer(3)]],
                                   device const int* orders [[buffer(4)]], device uint* dbg [[buffer(6)]],
                                   uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint widx = gid / 32u;
    device const K2WinIn& win = wins[widx];
    df64 tc = df64_from_bits(win.tcBits);
    df64 w = df64_from_bits(win.wBits);
    df64 ct[K2_MAX_CT];
    for (uint i = 0u; i < K2_MAX_CT; i++) {
        ct[i] = df64_from_bits(win.ct[i]);
    }
    df64 halfW = df64_mul_f(w, 0.5f);
    K2Piece pieces[K2_MAX_PIECES];
    uint np = k2_place_pieces(df64_sub(tc, halfW), df64_add(tc, halfW), ct, win.nct, pieces);
    if (lane == 0u) {
        dbg[widx * 6u] = np;
        for (uint p = 0u; p < K2_MAX_PIECES; p++) {
            dbg[widx * 6u + 1u + p] = (p < np) ? (uint)pieces[p].kind : 0u;
        }
    }
}

// Debug variant for the node-dM rounding: for every (window, flattened node) writes the node time
// as the device's df64 split (hi, lo bits) and dM as f32 bits. 3 uints per node, 640 nodes per
// window. The host rebuilds dM in f64 from (n, t) and measures |dM32 - dM64|.
kernel void k2_window_dm_probe(device const K2WinIn* wins [[buffer(0)]], device const float* drawC [[buffer(1)]],
                               device const float2* glx [[buffer(2)]], device const float* glw [[buffer(3)]],
                               device const int* orders [[buffer(4)]], device uint* dbg [[buffer(6)]],
                               uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint widx = gid / 32u;
    device const K2WinIn& win = wins[widx];
    df64 tc = df64_from_bits(win.tcBits);
    df64 w = df64_from_bits(win.wBits);
    df64 ct[K2_MAX_CT];
    for (uint i = 0u; i < K2_MAX_CT; i++) {
        ct[i] = df64_from_bits(win.ct[i]);
    }
    df64 halfW = df64_mul_f(w, 0.5f);
    K2Piece pieces[K2_MAX_PIECES];
    uint np = k2_place_pieces(df64_sub(tc, halfW), df64_add(tc, halfW), ct, win.nct, pieces);
    uint m = (uint)orders[win.orderIdx];
    df64 invW = df64_div(DF64_ONE, w);
    device const float* dc = drawC + (uint)win.drawRow * 16u;
    uint total = np * m;
    for (uint flat = lane; flat < total; flat += 32u) {
        uint pi = flat / m;
        uint i = flat - pi * m;
        K2Node nd = k2_node(pieces[pi], i, win.orderIdx, invW, glx, glw);
        float dM = k2_node_dM(nd.t, dc);
        uint base = (widx * 640u + flat) * 3u;
        dbg[base] = as_type<uint>(nd.t.hi);
        dbg[base + 1u] = as_type<uint>(nd.t.lo);
        dbg[base + 2u] = as_type<uint>(dM);
    }
}
#endif
"""#
}
