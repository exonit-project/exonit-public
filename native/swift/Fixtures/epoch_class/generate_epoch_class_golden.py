#!/usr/bin/env python3
"""Golden fixtures for the Swift port of the ExoNit epoch-class harmonic resolver.

Runs the PYTHON implementation (worker/harmonic_epoch_class.py + worker.worker.
resolve_harmonic) from an ExoNit checkout on synthetic light curves and stores inputs +
outputs as JSON. Every float64 is stored as its IEEE-754 bit pattern (unsigned 64-bit
integer) so the Swift test can assert BIT-EXACT equality with no text round-trip.

    EXONIT_REPO=/path/to/exonit-checkout python3 -I generate_epoch_class_golden.py [out.json]

EXONIT_REPO must contain worker/harmonic_epoch_class.py (branch worktree-alias-2p-fix or
later). Needs numpy only (+ whatever worker.worker imports; use the server venv).
Deterministic: fixed numpy Generator seeds. The INPUTS are stored in the JSON, so the
Swift test does not depend on the numpy version used to generate them.
"""
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get(
    "EXONIT_REPO", "~/exonit/.claude/worktrees/alias-2p-fix"))
sys.path.insert(0, str(REPO / "worker"))
sys.path.insert(0, str(REPO))
import harmonic_epoch_class as hec  # noqa: E402
from worker.worker import resolve_harmonic  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path(__file__).resolve().parent / "epoch_class_golden.json"


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def barr(a):
    return [bits(v) for v in np.asarray(a, dtype=np.float64)]


# --------------------------------------------------------------------------- data
def lc(planets, span=120.0, cadence=1.0 / 24.0, noise=3e-4, seed=7,
       gaps=(), red=0.0, bumps=()):
    """planets [(P, t0 centre, dur, depth)]; gaps [(lo, hi)]; red = AR(1) phi;
    bumps [(lo, hi, dflux)]."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, span, cadence)
    f = np.ones_like(t)
    for P, t0, dur, dep in planets:
        ph = np.abs(((t - t0 + 0.5 * P) % P) - 0.5 * P)
        f[ph < 0.5 * dur] -= dep
    w = rng.normal(0.0, noise, t.size)
    if red > 0:
        n = np.empty_like(w)
        n[0] = w[0]
        for i in range(1, w.size):
            n[i] = red * n[i - 1] + np.sqrt(1 - red * red) * w[i]
        w = n
    f += w
    for lo, hi, d in bumps:
        f[(t >= lo) & (t < hi)] += d
    keep = np.ones(t.size, bool)
    for lo, hi in gaps:
        keep &= ~((t >= lo) & (t <= hi))
    return t[keep], f[keep]


DUR = 0.2
P_TRUE = 12.47
P_HALF = P_TRUE / 2.0


def eph(P, t0c, dur=DUR, centre=False, use_excluded=False):
    """Ephemeris with t0 given as the transit centre; stored as box START unless
    centre=True (the raw-BLS convention the worker uses)."""
    t0 = t0c if centre else t0c - 0.5 * dur
    return dict(period=P, t0=t0, duration=dur, centre=centre,
                use_excluded=use_excluded)


# --------------------------------------------------------------------------- ops
def enc_prep(p):
    if p is None:
        return None
    return dict(ep0=int(p["ep0"]), n_e=barr(p["n_e"]), d_e=barr(p["d_e"]),
                covered=[bool(v) for v in p["covered"]], sigma=bits(p["sigma"]),
                period=bits(p["period"]), centre=bits(p["centre"]))


def run_eph(t, f, e, excl):
    ex = excl if e["use_excluded"] else None
    P, t0, d, c = e["period"], e["t0"], e["duration"], e["centre"]
    prep = hec.prepare_epochs(t, f, P, t0, d, ex, c)
    out = dict(period=bits(P), t0=bits(t0), duration=bits(d), centre=c,
               use_excluded=e["use_excluded"], prep=enc_prep(prep),
               transit_centre=bits(hec.transit_centre(t0, d, c)))
    stats, parity, dem, grd, dem0, dem1 = {}, {}, {}, {}, {}, {}
    for k in range(2, 6):
        dep, err, nep, ndip = hec.class_stats(prep, k)
        stats[str(k)] = dict(depth=barr(dep), err=barr(err),
                             nep=[int(v) for v in nep],
                             ndip=[int(v) for v in ndip])
        parity[str(k)] = hec.harmonic_parity(t, f, P, t0, d, k, ex, c)
        if not c:
            sub = dict(period=P, t0=t0, duration=d)
            dem[str(k)] = bool(hec.demotion_supported(t, f, sub, k,
                                                      excluded=ex))
            grd[str(k)] = bool(hec.short_period_genuine(t, f, sub, k,
                                                        excluded=ex))
            dem0[str(k)] = bool(hec.demotion_supported(t, f, sub, k, excluded=ex,
                                                       mode="v0"))
            dem1[str(k)] = bool(hec.demotion_supported(t, f, sub, k, excluded=ex,
                                                       mode="v1"))
    out["stats"] = stats
    out["parity"] = parity
    out["demotion"] = dem
    out["guard_genuine"] = grd
    out["demotion_v0"] = dem0
    out["demotion_v1"] = dem1
    out["ks"] = [int(k) for k in hec.epoch_class_promotion_ks(
        t, f, P, t0, d, 50.0, c, ex)]
    # period_max cuts the larger k off (exercises the `break`)
    out["ks_short_max"] = [int(k) for k in hec.epoch_class_promotion_ks(
        t, f, P, t0, d, P * 2.5, c, ex)]
    out["period_max_short"] = bits(P * 2.5)
    return out


def run_slot(t, f, rm, accepted, centre):
    """rm / accepted use period, t0 (box start unless centre), duration."""
    acc = [dict(period_days=a["period"], t0=a["t0"],
                duration_days=a["duration"]) for a in accepted]
    fin = t[np.isfinite(t)]
    m = hec.find_harmonic_epoch_match(rm["period"], rm["t0"], rm["duration"],
                                      acc, float(fin[0]), float(fin[-1]),
                                      centre)
    idx = -1 if m is None else acc.index(m)
    drop = hec.slot_duplicate(t, f, rm["period"], rm["t0"], rm["duration"],
                              acc, None, centre)
    return dict(rm=dict(period=bits(rm["period"]), t0=bits(rm["t0"]),
                        duration=bits(rm["duration"])),
                accepted=[dict(period=bits(a["period"]), t0=bits(a["t0"]),
                               duration=bits(a["duration"])) for a in accepted],
                centre=centre, match=idx, drop=bool(drop))


def mock_bls(table, log):
    """Deterministic canned BLS: first table row whose period is within 5 % of
    the window centre; otherwise a null (snr 0) result."""
    def fn(time, flux, *, period_min, period_max, n_periods):
        log.append([bits(period_min), bits(period_max), int(n_periods)])
        c = 0.5 * (period_min + period_max)
        for r in table:
            if abs(c - r["period"]) / r["period"] < 0.05:
                return dict(period=r["period"], snr=r["snr"], depth=r["depth"],
                            depth_ppm=r["depth"] * 1e6, duration=r["duration"],
                            t0=r["t0"], power=1.0)
        return dict(period=c, snr=0.0, depth=0.0, depth_ppm=0.0,
                    duration=0.1, t0=0.0, power=0.0)
    return fn


def row(P, snr, depth, t0c, dur=DUR):
    return dict(period=P, snr=snr, depth=depth, duration=dur, t0=t0c - 0.5 * dur)


def enc_row(r):
    return dict(period=bits(r["period"]), snr=bits(r["snr"]),
                depth=bits(r["depth"]), duration=bits(r["duration"]),
                t0=bits(r["t0"]))


def seed_dict(P, snr, depth, t0c, dur=DUR):
    return dict(period=P, snr=snr, depth=depth, depth_ppm=depth * 1e6,
                duration=dur, t0=t0c - 0.5 * dur, power=1.0)


def enc_res(r):
    return dict(period=bits(r["period"]), snr=bits(r["snr"]),
                depth=bits(r["depth"]), duration=bits(r["duration"]),
                t0=bits(r["t0"]))


def run_promote(t, f, seed, period_max, table, excl=None, snr_min=5.0):
    log = []
    seed_obj = dict(seed)
    out = hec.promote_epoch_class(t, f, seed_obj, period_max,
                                  mock_bls(table, log), snr_min=snr_min,
                                  excluded=excl)
    return dict(seed=enc_res(seed), period_max=bits(period_max),
                snr_min=bits(snr_min), table=[enc_row(r) for r in table],
                use_excluded=excl is not None,
                promoted=out is not seed_obj,
                result=enc_res(out), calls=log)


def run_resolve(t, f, seed, pmin, pmax, source, table, excl=None,
                epoch_class=True, long_recovery=False):
    log = []
    out = resolve_harmonic(t, f, dict(seed), pmin, pmax, mock_bls(table, log),
                           data_source=source, long_period_recovery=long_recovery,
                           epoch_class=epoch_class, excluded=excl)
    return dict(seed=enc_res(seed), period_min=bits(pmin), period_max=bits(pmax),
                source=source, epoch_class=epoch_class,
                use_excluded=excl is not None,
                table=[enc_row(r) for r in table], result=enc_res(out),
                calls=log)


# --------------------------------------------------------------------------- cases
def case(name, t, f, ephs, excl=None, slots=(), promotes=(), resolves=()):
    d = dict(name=name, time=barr(t), flux=barr(f),
             excluded=None if excl is None else [int(v) for v in excl])
    d["ephemerides"] = [run_eph(t, f, e, excl) for e in ephs]
    d["slots"] = [run_slot(t, f, rm, acc, c) for rm, acc, c in slots]
    d["promotes"] = [run_promote(t, f, s, pm, tab, excl if ue else None)
                     for s, pm, tab, ue in promotes]
    d["resolves"] = [run_resolve(t, f, s, 0.5, 50.0, src, tab,
                                 excl if ue else None)
                     for s, src, tab, ue in resolves]
    return d


cases = []

# 1. promotion: true 12.47 d planet, BLS seed at P/2 and P/3
t, f = lc([(P_TRUE, 3.0, DUR, 0.002)])
cases.append(case(
    "promotion", t, f,
    [eph(P_HALF, 3.0), eph(P_TRUE, 3.0), eph(P_TRUE / 3, 3.0),
     eph(P_HALF, 3.0, centre=True)],
    slots=[(eph(P_TRUE, 3.0), [eph(P_HALF, 3.0)], False),     # long, accepted alias
           (eph(P_HALF, 3.0), [eph(P_TRUE, 3.0)], False),     # short alias of accepted
           (eph(9.1, 3.0), [eph(P_TRUE, 3.0)], False)],       # unrelated -> keep
    promotes=[(seed_dict(P_HALF, 200.0, 0.002, 3.0), 50.0,
               [row(P_TRUE, 50.0, 0.002, 3.0)], False),
              (seed_dict(P_HALF, 200.0, 0.002, 3.0), 50.0,
               [row(P_TRUE, 3.0, 0.002, 3.0)], False),       # snr below min -> none
              (seed_dict(P_HALF, 200.0, 0.002, 3.0), 13.0,
               [row(P_TRUE, 50.0, 0.002, 3.0)], False)],      # period_max clamps window
    resolves=[(seed_dict(P_HALF, 200.0, 0.002, 3.0), "spoc",
               [row(P_TRUE, 50.0, 0.002, 3.0)], False),
              (seed_dict(P_HALF, 200.0, 0.002, 3.0), "ffi",
               [row(P_TRUE, 50.0, 0.002, 3.0)], False),
              # veto: seed is the true period, noise-stronger /2 probe must be vetoed
              (seed_dict(P_TRUE, 100.0, 0.002, 3.0), "spoc",
               [row(P_HALF, 300.0, 0.002, 3.0)], False)]))

# 2. genuine short period: planet at 6.235, all epochs carry
t, f = lc([(P_HALF, 3.0, DUR, 0.002)], seed=11)
cases.append(case(
    "genuine_short", t, f,
    [eph(P_HALF, 3.0), eph(P_TRUE, 3.0)],
    slots=[(eph(P_TRUE, 3.0), [eph(P_HALF, 3.0)], False),     # long of genuine -> drop
           (eph(P_HALF, 3.0), [eph(P_TRUE, 3.0)], False)],    # short genuine -> keep
    promotes=[(seed_dict(P_HALF, 200.0, 0.002, 3.0), 50.0,
               [row(P_TRUE, 50.0, 0.002, 3.0)], False)],
    resolves=[(seed_dict(P_TRUE, 100.0, 0.002, 3.0), "spoc",
               [row(P_HALF, 300.0, 0.002, 3.0)], False),     # genuine /2 allowed
              (seed_dict(P_TRUE, 100.0, 0.002, 3.0), "ffi",
               [row(P_HALF, 300.0, 0.002, 3.0)], False)]))

# 3. near-resonant pair 6.2 / 12.5 d with coincident reference t0
t, f = lc([(6.2, 3.0, DUR, 0.002), (12.5, 3.1, DUR, 0.002)], span=200.0, seed=13)
cases.append(case(
    "near_resonant", t, f,
    [eph(6.2, 3.0), eph(12.5, 3.1), eph(12.4, 3.0)],
    slots=[(eph(12.5, 3.1), [eph(6.2, 3.0)], False),
           (eph(6.2, 3.0), [eph(12.5, 3.1)], False),
           (eph(12.4, 3.0), [eph(6.2, 3.0)], False)],         # exact 2x -> match
    resolves=[(seed_dict(6.2, 100.0, 0.002, 3.0), "spoc",
               [row(12.5, 90.0, 0.002, 3.1)], False)]))

# 4. gap: odd epochs of the 12.47 d planet are not in the data
gaps = [(3.0 + P_TRUE * n - 2.0, 3.0 + P_TRUE * n + 2.0) for n in range(1, 10, 2)]
t, f = lc([(P_TRUE, 3.0, DUR, 0.002)], gaps=gaps, seed=17)
cases.append(case(
    "gap", t, f,
    [eph(P_HALF, 3.0), eph(P_TRUE, 3.0)],
    slots=[(eph(P_TRUE, 3.0), [eph(P_HALF, 3.0)], False)],
    promotes=[(seed_dict(P_HALF, 200.0, 0.002, 3.0), 50.0,
               [row(P_TRUE, 50.0, 0.002, 3.0)], False)],
    resolves=[(seed_dict(P_HALF, 200.0, 0.002, 3.0), "spoc",
               [row(P_TRUE, 50.0, 0.002, 3.0)], False)]))

# 5. red noise + a flare epoch (+2 %) and a deep systematics epoch (-2 %)
t, f = lc([(P_TRUE, 3.0, DUR, 0.0015)], span=200.0, noise=4e-4, red=0.95, seed=19,
          bumps=[(3.0 + P_TRUE * 2 - 0.1, 3.0 + P_TRUE * 2 + 0.1, 0.02),
                 (3.0 + P_TRUE * 5 - 0.1, 3.0 + P_TRUE * 5 + 0.1, -0.02)])
cases.append(case(
    "red_noise_flare", t, f,
    [eph(P_HALF, 3.0), eph(P_TRUE, 3.0), eph(P_TRUE / 4, 3.0)],
    slots=[(eph(P_TRUE, 3.0), [eph(P_HALF, 3.0)], False)],
    promotes=[(seed_dict(P_HALF, 200.0, 0.0015, 3.0), 50.0,
               [row(P_TRUE, 50.0, 0.0015, 3.0)], False)]))

# 6. long span (1000 d, 0.1 d cadence): period-grid drift refinement active (J >= 1)
t, f = lc([(P_TRUE, 3.0, 0.6, 0.003)], span=1000.0, cadence=0.1, noise=5e-4,
          seed=23)
cases.append(case(
    "long_drift", t, f,
    [eph(P_HALF * (1 + 1.2e-4), 3.0, dur=0.6),
     eph(P_TRUE * (1 + 1.2e-4), 3.0, dur=0.6),
     eph(P_TRUE * (1 - 2.5e-4), 3.0, dur=0.6),
     eph(P_HALF * (1 - 0.8e-4), 3.0, dur=0.6)],
    slots=[(eph(P_TRUE * (1 + 1.2e-4), 3.0, dur=0.6),
            [eph(P_HALF * (1 + 1.2e-4), 3.0, dur=0.6)], False)],
    promotes=[(seed_dict(P_HALF * (1 + 1.2e-4), 200.0, 0.003, 3.0, dur=0.6), 50.0,
               [row(P_TRUE * (1 + 1.2e-4), 50.0, 0.003, 3.0, dur=0.6)], False)]))

# 7. multi-planet-loop state: odd-epoch transits median-filled and flagged excluded,
#    plus non-finite flux / time points
t, f = lc([(P_TRUE, 3.0, DUR, 0.002), (P_TRUE * 0.37, 1.1, DUR, 0.002)], seed=29)
ph = np.abs(((t - 3.0 + 0.5 * P_TRUE) % P_TRUE) - 0.5 * P_TRUE)
ep = np.rint((t - 3.0) / P_TRUE)
excl = (ph < 0.6) & (np.mod(ep, 2) == 1)
f[excl] = np.median(f[~excl])
f[[5, 400, 1201]] = np.nan
t = t.copy()
t[[17, 2000]] = np.nan
cases.append(case(
    "excluded_nonfinite", t, f,
    [eph(P_HALF, 3.0, use_excluded=True), eph(P_HALF, 3.0, use_excluded=False),
     eph(P_TRUE, 3.0, use_excluded=True)],
    excl=excl,
    slots=[(eph(P_TRUE, 3.0), [eph(P_HALF, 3.0)], False)],
    promotes=[(seed_dict(P_HALF, 200.0, 0.002, 3.0), 50.0,
               [row(P_TRUE, 50.0, 0.002, 3.0)], True),
              (seed_dict(P_HALF, 200.0, 0.002, 3.0), 50.0,
               [row(P_TRUE, 50.0, 0.002, 3.0)], False)],
    resolves=[(seed_dict(P_HALF, 200.0, 0.002, 3.0), "spoc",
               [row(P_TRUE, 50.0, 0.002, 3.0)], True)]))

# 8. revision 3 (WASP-164 b regression): two sectors 700 d apart, genuine 1.77711 d
#    planet; probes/refits 0.06 % off the true period (smeared in the far sector)
P_W, DUR_W, DEP_W = 1.77711, 0.06, 0.008
cad = 10.0 / 720.0
tw = np.concatenate([np.arange(0.0, 27.0, cad), np.arange(700.0, 727.0, cad)])
fw = np.ones_like(tw)
phw = np.abs(((tw - 0.4 + 0.5 * P_W) % P_W) - 0.5 * P_W)
fw[phw < 0.5 * DUR_W] -= DEP_W
fw += np.random.default_rng(11).normal(0.0, 1.5e-3, tw.size)
cases.append(case(
    "two_sector_smeared", tw, fw,
    [eph(P_W * (1 - 6e-4), 0.4, dur=DUR_W), eph(P_W, 0.4, dur=DUR_W),
     eph(P_W * (1 + 6e-4), 0.4, dur=DUR_W),
     eph(2 * P_W * (1 - 6e-4), 0.4, dur=DUR_W)],
    promotes=[(seed_dict(P_W * (1 - 6e-4), 80.0, DEP_W, 0.4, dur=DUR_W), 50.0,
               [row(2 * P_W * (1 - 6e-4), 40.0, DEP_W, 0.4, dur=DUR_W)], False)],
    resolves=[(seed_dict(P_W * (1 - 6e-4), 80.0, DEP_W, 0.4, dur=DUR_W), "spoc",
               [row(2 * P_W * (1 - 6e-4), 40.0, DEP_W, 0.4, dur=DUR_W)], False),
              # the A/B regression: seed 2P, the (smeared) true /2 probe must be taken
              (seed_dict(2 * P_W * (1 - 6e-4), 60.0, DEP_W, 0.4, dur=DUR_W), "spoc",
               [row(P_W * (1 - 6e-4), 300.0, DEP_W, 0.4, dur=DUR_W)], False)]))

# 9. G1 guard fires inside promote: planet A 12.47 d (the half-period seed sees one-sided
#    classes) + planet B 6.235 d offset by 2 d; the canned re-fit sits on B's ephemeris
t, f = lc([(P_TRUE, 3.0, DUR, 0.002), (P_HALF, 5.0, DUR, 0.002)], seed=71)
cases.append(case(
    "guard_fires", t, f,
    [eph(P_HALF, 3.0), eph(P_TRUE, 3.0), eph(P_TRUE, 5.0), eph(P_TRUE / 3, 5.0)],
    promotes=[(seed_dict(P_HALF, 200.0, 0.002, 3.0), 50.0,
               [row(P_TRUE, 20.0, 0.002, 5.0)], False),       # guard rejects -> none
              (seed_dict(P_HALF, 200.0, 0.002, 3.0), 50.0,
               [row(P_TRUE, 20.0, 0.002, 3.0)], False)],      # A-aligned -> promoted
    resolves=[(seed_dict(P_HALF, 200.0, 0.002, 3.0), "spoc",
               [row(P_TRUE, 20.0, 0.002, 5.0)], False)]))

# 10. k = 3 guard firing: planet A 12 d + planet B 4 d offset by 1 d; the 4 d seed on A's
#     ephemeris is one-sided in 3 classes (ks = [3]); the canned 12 d re-fit sits on B's
#     ephemeris where all 3 classes carry at P_k/3 -> guard rejects
PU = 4.0
t, f = lc([(3 * PU, 3.0, DUR, 0.002), (PU, 4.0, DUR, 0.002)], seed=73)
cases.append(case(
    "guard_k3", t, f,
    [eph(PU, 3.0), eph(3 * PU, 3.0), eph(3 * PU, 4.0)],
    promotes=[(seed_dict(PU, 200.0, 0.002, 3.0), 50.0,
               [row(3 * PU, 20.0, 0.002, 4.0)], False),        # guard rejects -> none
              (seed_dict(PU, 200.0, 0.002, 3.0), 50.0,
               [row(3 * PU, 20.0, 0.002, 3.0)], False)],       # A-aligned -> promoted
    resolves=[(seed_dict(PU, 200.0, 0.002, 3.0), "spoc",
               [row(3 * PU, 20.0, 0.002, 4.0)], False)]))

# 11. vh switch discriminating cases (revision 5): v0 != v1 on each side of the threshold;
#     boundary at span == 5000 x duration (v0) and just above (v1)
t = np.arange(0, 27.0, 0.5 / 24)
rng = np.random.default_rng(0)
f = 1 + rng.normal(0, 1e-3, t.size)
ph = np.abs(((t - 1.3 + 0.5 * 4.0) % 4.0) - 0.5 * 4.0)
f[ph < 0.12 / 2] -= 1.0e-3
cases.append(case("vh_short", t, f, [eph(2.0, 1.3, dur=0.12)]))
cad = 0.5 / 24
tl = np.concatenate([np.arange(0, 27.0, cad), np.arange(700, 727.0, cad)])
span_l = float(tl[-1]) - float(tl[0])
dur_eq = span_l * 2e-4
rng = np.random.default_rng(0)
fl = 1 + rng.normal(0, 1e-3, tl.size)
phl = np.abs(((tl - 0.4 + 0.5 * 3.0) % 3.0) - 0.5 * 3.0)
fl[phl < 0.12 / 2] -= 1.5e-3
cases.append(case("vh_long", tl, fl, [eph(3.0 * (1 - 6e-4), 0.4, dur=0.12)]))
rng = np.random.default_rng(0)
fb = 1 + rng.normal(0, 1e-3, tl.size)
phb = np.abs(((tl - 0.4 + 0.5 * 3.0) % 3.0) - 0.5 * 3.0)
fb[phb < dur_eq / 2] -= 1.5e-3
cases.append(case("vh_boundary", tl, fb,
                  [eph(3.0 * (1 - 6e-4), 0.4, dur=dur_eq),
                   eph(3.0 * (1 - 6e-4), 0.4, dur=float(np.nextafter(dur_eq, 0.0))),
                   eph(3.0 * (1 - 6e-4), 0.4, dur=float(np.nextafter(dur_eq, 1.0)))]))

# --------------------------------------------------------------------------- flag off
# Canned scenarios whose expected values were captured from the PRE-CHANGE resolver
# (branch alias-2p-phase1); the Python suite asserts the same constants.
GT = np.arange(0, 100, 0.1)
GF = np.ones_like(GT)
GOLD = {
    "spoc_x2": (4.0, [(8.0, 12.0, 0.0021)], "spoc", (8.0, 12.0, 0.0021)),
    "ffi_x3": (4.0, [(12.0, 7.0, 0.0019)], "ffi", (12.0, 7.0, 0.0019)),
    "spoc_demote": (8.0, [(4.0, 15.0, 0.0018)], "spoc", (4.0, 15.0, 0.0018)),
    "ffi_demote3": (9.0, [(3.0, 9.0, 0.0016)], "ffi", (3.0, 9.0, 0.0016)),
    "none": (5.0, [], "spoc", (5.0, 10.0, 0.002)),
    "depth_reject": (4.0, [(8.0, 12.0, 0.0050)], "spoc", (4.0, 10.0, 0.002)),
}
flag_off = []
for name in sorted(GOLD):
    P, rows, src, (gp, gs, gd) = GOLD[name]
    table = [dict(period=p, snr=s, depth=d, duration=0.1, t0=0.5)
             for p, s, d in rows]
    seed = dict(period=P, snr=10.0, depth=0.002, depth_ppm=2000.0,
                duration=0.1, t0=0.5, power=1.0)
    rec = run_resolve(GT, GF, seed, 0.5, 50.0, src, table, epoch_class=False)
    got = (rec["result"]["period"], rec["result"]["snr"], rec["result"]["depth"])
    assert got == (bits(gp), bits(gs), bits(gd)), (name, got)
    rec["name"] = name
    flag_off.append(rec)

head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
doc = dict(
    description="Python goldens for the Swift epoch-class port; floats are IEEE-754 "
                "bit patterns (UInt64).",
    source_repo_head=head, numpy=np.__version__,
    flag_off_n=int(GT.size), flag_off=flag_off, cases=cases)
OUT.write_text(json.dumps(doc, separators=(",", ":")))
print("wrote", OUT, OUT.stat().st_size, "bytes;",
      sum(len(c["ephemerides"]) for c in cases), "ephemerides")
