import Foundation

/// MSL source text holder for K1 contacts: the df64 bisections, the five contact roots, the affected
/// fold bins with `epsRef` / `epsMin`, and the K1 output rows (`K1Out` + the p4.0 `drawC` row).
///
/// Appended after `FPPV3MetalDF64`, `FPPV3MetalDF64Trig` and `FPPV3MetalK1Scan`: it reuses
/// `df64`, `K1Orbit`, `K1Grid`, `K1Scan`, `K1Params`, `FoldParams`, `K1Out`, `k1_orbit`, `k1_grid`,
/// `k1_scan` and `k1_zone_df64` from those texts.
public enum FPPV3MetalK1 {
    public static let text: String = #"""
    // Card M1.4: K1 part 2 -- bisections, the five contact roots, affected bins, K1 output rows.
    #pragma clang fp contract(off)

    // Addendum 3 / LAYOUT v1: an absent root is stored as this exact f64 NaN pattern.
    constant ulong K1_ROOT_ABSENT = 0x7ff8000000000000ULL;

    struct K1Contacts { df64 t[5]; df64 zMin; }; // zMin survives a near-tangent miss

    // True anomaly from the mean anomaly with the safeguarded solver of `k1_zone_df64`.
    inline df64 k1_true_anomaly(df64 m, thread const K1Orbit& o) {
        if (o.ecc.hi < 1e-10f) return m;
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
            if ((df64_gt(en, lo) && df64_lt(en, hi)) || tiny) {
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
        return df64_atan2(y, x);
    }

    // Literal FPPV2Orbit.halfDz2Dt in df64: (1/2) d(z^2)/dt. `inc` enters only through o.cosI.
    inline df64 k1_half_dz2dt(df64 t, thread const K1Orbit& o) {
        if (!o.ok) return df64{NAN, NAN};
        df64 two_pi = DF64_TWO_PI;
        df64 m = df64_add(df64_div(df64_mul(two_pi, t), o.P), o.mT);
        m = df64_sub(m, df64_mul(two_pi, df64_floor(df64_div(m, two_pi))));
        df64 nu = k1_true_anomaly(m, o);
        df64 one_m_e2 = df64_sub(DF64_ONE, df64_mul(o.ecc, o.ecc));
        df64 cos_nu = df64_cos(nu);
        df64 q = df64_add(DF64_ONE, df64_mul(o.ecc, cos_nu));
        df64 r = df64_div(df64_mul(o.aRs, one_m_e2), q);
        df64 nuDot = df64_div(df64_mul(df64_mul(o.n, q), q), df64_pow15(one_m_e2));
        df64 rDot = df64_div(
            df64_mul(df64_mul(df64_mul(o.n, o.aRs), o.ecc), df64_sin(nu)), df64_sqrt(one_m_e2));
        df64 th = df64_add(o.w, nu);
        df64_sc sc_th = df64_sincos(th);
        df64 x = df64_neg(df64_mul(r, sc_th.c));
        df64 y = df64_neg(df64_mul(df64_mul(r, sc_th.s), o.cosI));
        df64 xd = df64_add(
            df64_neg(df64_mul(rDot, sc_th.c)), df64_mul(df64_mul(r, sc_th.s), nuDot));
        df64 yd = df64_mul(
            df64_add(df64_neg(df64_mul(rDot, sc_th.s)),
                     df64_neg(df64_mul(df64_mul(r, sc_th.c), nuDot))),
            o.cosI);
        return df64_add(df64_mul(x, xd), df64_mul(y, yd));
    }

    // Literal FPPV2Contact.bisect: tol 1e-13 d, at most 100 iterations.
    inline df64 k1_bisect(df64 floIn, df64 fhi, df64 tloIn, df64 thi, thread const K1Orbit& o,
                          df64 level, bool isZ) {
        df64 tol = df64_from_float(1.0e-13f);
        df64 flo = floIn;
        df64 tlo = tloIn;
        if (df64_cmp(flo, DF64_ZERO) == 0) return tlo;
        if (df64_cmp(fhi, DF64_ZERO) == 0) return thi;
        if (df64_gt(df64_mul(flo, fhi), DF64_ZERO)) return df64{NAN, NAN};
        for (int it = 0; it < 100; it++) {
            if (df64_lt(df64_sub(thi, tlo), tol)) break;
            df64 tm = df64_mul_f(df64_add(tlo, thi), 0.5f);
            df64 fm;
            if (isZ) {
                fm = df64_sub(k1_zone_df64(tm, o), level);
            } else {
                fm = k1_half_dz2dt(tm, o);
            }
            if (df64_cmp(fm, DF64_ZERO) == 0) return tm;
            if (df64_gt(fm, DF64_ZERO) == df64_gt(flo, DF64_ZERO)) {
                tlo = tm;
                flo = fm;   // `fhi` is never read again, as on the CPU
            } else {
                thi = tm;
            }
        }
        return df64_mul_f(df64_add(tlo, thi), 0.5f);
    }

    // Literal FPPV2Contact.contactTimes lines 44-120 (grid scan from M1.3), in df64.
    inline K1Contacts k1_contact_times(thread const K1Orbit& o, ulong kBits) {
        K1Contacts out;
        for (int i = 0; i < 5; i++) out.t[i] = df64{NAN, NAN};
        out.zMin = df64{NAN, NAN};
        if (!o.ok) return out;

        df64 k = df64_from_bits(kBits);
        K1Grid g = k1_grid(kBits, o);
        K1Scan s = k1_scan(o, g);
        df64 limit = df64_add(df64_add(DF64_ONE, k), df64_from_float(1.0e-3f));
        if (s.i0 < 0 || df64_ge(s.bestZ, limit)) return out;   // no transit in the window

        int loI = s.i0 - 1;
        if (loI < 0) loI = 0;
        int hiI = s.i0 + 1;
        if (hiI > 2000) hiI = 2000;
        df64 neg_w = df64_neg(g.w);
        df64 tlo = df64_add(neg_w, df64_mul_f(g.step, (float)loI));
        df64 thi = df64_add(neg_w, df64_mul_f(g.step, (float)hiI));
        df64 tMin = k1_bisect(
            k1_half_dz2dt(tlo, o), k1_half_dz2dt(thi, o), tlo, thi, o, DF64_ZERO, false);
        if (df64_isnan(tMin)) return out;
        df64 zMin = k1_zone_df64(tMin, o);
        out.zMin = zMin;

        df64 l = df64_add(DF64_ONE, k);
        if (df64_ge(zMin, l)) return out;   // the companion never touches the disc

        // Outer contacts; widen the window (x2, at most 4 times) before giving up.
        df64 hw = g.w;
        df64 t1 = df64{NAN, NAN};
        for (int it = 0; it < 5; it++) {
            df64 a = df64_sub(tMin, hw);
            df64 fa = df64_sub(k1_zone_df64(a, o), l);
            df64 fb = df64_sub(zMin, l);
            t1 = k1_bisect(fa, fb, a, tMin, o, l, true);
            if (!df64_isnan(t1)) break;
            hw = df64_mul_f(hw, 2.0f);
        }
        hw = g.w;
        df64 t4 = df64{NAN, NAN};
        for (int it = 0; it < 5; it++) {
            df64 b = df64_add(tMin, hw);
            df64 fa = df64_sub(zMin, l);
            df64 fb = df64_sub(k1_zone_df64(b, o), l);
            t4 = k1_bisect(fa, fb, tMin, b, o, l, true);
            if (!df64_isnan(t4)) break;
            hw = df64_mul_f(hw, 2.0f);
        }

        // Inner contacts: only when the companion is fully inside (or in front of) the disc at tMin.
        df64 t2 = df64{NAN, NAN};
        df64 t3 = df64{NAN, NAN};
        df64 lin = df64_abs(df64_sub(DF64_ONE, k));
        if (df64_lt(zMin, lin)) {
            if (!df64_isnan(t1)) {
                t2 = k1_bisect(
                    df64_sub(l, lin), df64_sub(zMin, lin), t1, tMin, o, lin, true);
            }
            if (!df64_isnan(t4)) {
                t3 = k1_bisect(
                    df64_sub(zMin, lin), df64_sub(l, lin), tMin, t4, o, lin, true);
            }
        }
        out.t[0] = t1;
        out.t[1] = t2;
        out.t[2] = t3;
        out.t[3] = t4;
        out.t[4] = tMin;
        return out;
    }

    // One thread per draw: roots, dilution, affected bins, epsRef/epsMin, K1Out row and drawC row.
    inline void k1_contacts_impl(device const ulong* drawIn, constant K1Params& P,
                                  device const ulong* foldBuf, constant FoldParams& F,
                                  device K1Out* k1Out, device float* drawC,
                                  uint gid, thread ulong& zMinBits) {
        zMinBits = K1_ROOT_ABSENT;
        if (gid >= P.n) return;

        ulong kBits   = drawIn[P.segOff[0] + gid];
        ulong Pbits   = drawIn[P.segOff[1] + gid];
        ulong incBits = drawIn[P.segOff[2] + gid];
        ulong aRbits  = drawIn[P.segOff[3] + gid];
        ulong eccBits = drawIn[P.segOff[4] + gid];
        ulong wBits   = drawIn[P.segOff[5] + gid];
        ulong compBits = drawIn[P.segOff[6] + gid];
        uint u1i = P.segOff[7] + (P.u1Stride == 1u ? gid : 0u);
        uint u2i = P.segOff[8] + (P.u2Stride == 1u ? gid : 0u);
        ulong u1Bits = drawIn[u1i];
        ulong u2Bits = drawIn[u2i];

        device K1Out* row = &k1Out[gid];
        device float* dc = &drawC[16u * gid];

        // M2c uses the two reserved K1Params words without changing its ABI:
        // pad0 bit 0 = host f64 EB dilA in segment 6; pad1 = active count.
        // Any trailing rejected draws are sent here only for gate roots.
        bool ebMode = (P.pad0 & 1u) != 0u;
        if (ebMode && gid >= P.pad1) {
            for (int i = 0; i < 5; i++) row->root[i] = K1_ROOT_ABSENT;
            row->k1State = 0u;
            row->firstAff = 0u;
            row->nAff = 0u;
            row->epsRef = 0.0f;
            row->epsMin = 0.0f;
            row->dilA = 1.0f;
            for (int i = 0; i < 16; i++) dc[i] = 0.0f;
            K1Orbit maskedOrbit = k1_orbit(Pbits, aRbits, incBits, eccBits, wBits);
            K1Contacts maskedContacts = k1_contact_times(maskedOrbit, kBits);
            zMinBits = df64_to_bits(maskedContacts.zMin);
            for (int i = 0; i < 5; i++) {
                row->root[i] = df64_isnan(maskedContacts.t[i]) ? K1_ROOT_ABSENT : df64_to_bits(maskedContacts.t[i]);
            }
            return; // nAff stays zero: no K2 work for a rejected draw
        }

        K1Orbit o = k1_orbit(Pbits, aRbits, incBits, eccBits, wBits);
        if (!o.ok) {
            for (int i = 0; i < 5; i++) row->root[i] = K1_ROOT_ABSENT;
            row->k1State = 0;
            row->firstAff = 0;
            row->nAff = 0;
            row->epsRef = 0.0f;
            row->epsMin = 0.0f;
            row->dilA = 1.0f;
            for (int i = 0; i < 16; i++) dc[i] = 0.0f;
            dc[9] = df64_to_float(df64_from_bits(kBits));
            dc[10] = df64_to_float(df64_from_bits(u1Bits));
            dc[11] = df64_to_float(df64_from_bits(u2Bits));
            dc[12] = 1.0f;
            return;
        }

        // TP-form dilution, or the literal CPU EB preparation's f64 dilA.
        df64 comp = df64_from_bits(compBits);
        df64 dilA_df = DF64_ONE;
        if (ebMode) {
            dilA_df = comp;
        } else if (df64_gt(comp, DF64_ZERO)) {
            df64 fRatio;
            if (P.companionIsHost == 1u) {
                fRatio = df64_div(DF64_ONE, df64_div(comp, df64_sub(DF64_ONE, comp)));
            } else {
                fRatio = df64_div(comp, df64_sub(DF64_ONE, comp));
            }
            dilA_df = df64_div(DF64_ONE, df64_add(DF64_ONE, fRatio));
        }
        float dilA = df64_to_float(dilA_df);
        float dilB = 1.0f - dilA;

        K1Contacts ct = k1_contact_times(o, kBits);
        zMinBits = df64_to_bits(ct.zMin);
        for (int i = 0; i < 5; i++) {
            row->root[i] = df64_isnan(ct.t[i]) ? K1_ROOT_ABSENT : df64_to_bits(ct.t[i]);
        }

        df64 foldW = df64_from_bits(P.foldW);
        df64 delta = df64_from_bits(P.foldDelta);
        df64 hw = df64_mul_f(foldW, 0.5f);
        df64 t1 = ct.t[0];
        df64 t4 = ct.t[3];

        uint first = 0;
        uint nAff = 0;
        uint last = 0;
        if (!df64_isnan(t1) && !df64_isnan(t4)) {
            for (uint b = 0; b < P.nb; b++) {
                df64 tmax = df64_from_bits(foldBuf[F.v[6] + b]);
                df64 tmin = df64_from_bits(foldBuf[F.v[5] + b]);
                if (df64_gt(df64_add(tmax, hw), t1) && df64_lt(df64_sub(tmin, hw), t4)) {
                    if (nAff == 0u) first = b;
                    nAff++;
                    last = b;
                }
            }
        }

        uint state = 0;
        float epsRef = 0.0f;
        float epsMin = 0.0f;
        if (nAff > 0u) {
            if (last == first + nAff - 1u) {
                state = 1;
                df64 epsRef_df = df64_sqrt(
                    df64_div(df64_mul_f(delta, 2.0f), df64_from_float((float)nAff)));
                df64 eMin = df64_mul(df64_from_bits(foldBuf[F.v[2] + first]), epsRef_df);
                for (uint b = first + 1u; b < first + nAff; b++) {
                    df64 e = df64_mul(df64_from_bits(foldBuf[F.v[2] + b]), epsRef_df);
                    if (df64_lt(e, eMin)) eMin = e;
                }
                epsRef = df64_to_float(epsRef_df);
                epsMin = df64_to_float(eMin);
            } else {
                state = 2;   // affected bins not contiguous: the host refuses
            }
        }

        row->k1State = state;
        row->firstAff = (state == 1u) ? first : 0u;
        row->nAff = (state == 1u) ? nAff : 0u;
        row->epsRef = epsRef;
        row->epsMin = epsMin;
        row->dilA = dilA;

        dc[0] = o.n32;
        dc[1] = o.aRs32;
        dc[2] = o.cosI32;
        dc[3] = o.ecc32;
        dc[4] = o.sET;
        dc[5] = o.cET;
        dc[6] = o.q;
        dc[7] = o.A;
        dc[8] = o.B;
        dc[9] = df64_to_float(df64_from_bits(kBits));
        dc[10] = df64_to_float(df64_from_bits(u1Bits));
        dc[11] = df64_to_float(df64_from_bits(u2Bits));
        dc[12] = dilA;
        dc[13] = dilB;
        dc[14] = o.nLo;
        dc[15] = 0.0f;
    }

    // Preserve the six-buffer entry point used by the K1/K2 probes. Both entry
    // points execute the same root calculation and retain the 64-byte K1Out ABI.
    kernel void k1_contacts(device const ulong* drawIn [[buffer(0)]], constant K1Params& P [[buffer(1)]],
                            device const ulong* foldBuf [[buffer(2)]], constant FoldParams& F [[buffer(3)]],
                            device K1Out* k1Out [[buffer(4)]], device float* drawC [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
        ulong zMinBits;
        k1_contacts_impl(drawIn, P, foldBuf, F, k1Out, drawC, gid, zMinBits);
    }

    kernel void k1_contacts_gate(device const ulong* drawIn [[buffer(0)]], constant K1Params& P [[buffer(1)]],
                                 device const ulong* foldBuf [[buffer(2)]], constant FoldParams& F [[buffer(3)]],
                                 device K1Out* k1Out [[buffer(4)]], device float* drawC [[buffer(5)]],
                                 device ulong* zMinOut [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
        if (gid >= P.n) return;
        ulong zMinBits;
        k1_contacts_impl(drawIn, P, foldBuf, F, k1Out, drawC, gid, zMinBits);
        zMinOut[gid] = zMinBits;
    }
    """#
}
