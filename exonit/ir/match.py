"""Recovery / alias classification of worker candidates against injection truth
(spec section 4.1).  Pure functions, no I/O.

Epoch test: a candidate's reference epoch T0_c is compared with the lattice of
the *true* ephemeris (spacing P_i/q for the ratio p/q), evaluated at T0_c itself.
That makes the test independent of how the candidate's period error would
accumulate between T0_c and the injected T0 (BLS reports a mid-span epoch).
"""
from __future__ import annotations

import math

import numpy as np

PERIOD_TOL = 0.01
EPOCH_FLOOR_DAYS = 0.02            # 29 min: BLS epoch resolution floor
ALIAS_KS = (2, 3, 4, 5)
# (name, p, q): candidate period = (p/q) * P_injected
CLASSES = (
    [("exact", 1, 1)]
    + [(f"sub{k}", 1, k) for k in ALIAS_KS]       # P/k
    + [(f"super{k}", k, 1) for k in ALIAS_KS]     # k*P
    + [("ratio3_2", 3, 2), ("ratio2_3", 2, 3)]
)
HEADLINE = {"exact": "exact", "ratio3_2": "ratio", "ratio2_3": "ratio"}


def family_of(cls: str) -> str:
    if cls in HEADLINE:
        return HEADLINE[cls]
    if cls.startswith("sub"):
        return "sub"
    if cls.startswith("super"):
        return "super"
    return cls


def lattice_offset(t0_c, t0_ref, spacing):
    """Signed distance (days) from t0_c to the nearest point of t0_ref + n*spacing."""
    return (t0_c - t0_ref + 0.5 * spacing) % spacing - 0.5 * spacing


def epoch_tol(t14):
    return max(0.5 * t14, EPOCH_FLOOR_DAYS)


def classify(p_c, t0_c, truth_p, truth_t0, truth_t14, tol=PERIOD_TOL):
    """Return (class_name, epoch_offset_days).  class_name is one of CLASSES,
    'period_only' (period matches a class but the epoch does not) or 'wrong'."""
    best_period_only = False
    for name, p, q in CLASSES:
        target = truth_p * p / q
        if abs(p_c - target) / target > tol:
            continue
        off = lattice_offset(t0_c, truth_t0, truth_p / q)
        if abs(off) < epoch_tol(truth_t14):
            return name, float(off)
        best_period_only = True
    return ("period_only" if best_period_only else "wrong"), None


def explained_by_baseline(cand, baseline, tol=PERIOD_TOL):
    """True if the candidate repeats a candidate of the un-injected blank run:
    period within ``tol`` and same epoch lattice (within half a duration)."""
    for b in baseline:
        pb = b["period_days"]
        if not pb or abs(cand["period_days"] - pb) / pb > tol:
            continue
        dur = max(cand.get("duration_days") or 0.0, b.get("duration_days") or 0.0, 0.04)
        if abs(lattice_offset(cand["t0"], b["t0"], pb)) < 0.5 * dur:
            return True
    return False


def production_period_max(span_days, legacy=50.0, cap=200.0, divisor=2.5):
    """Mirror of worker.process_task: main loop 50 d, extended to span/2.5 (cap 200)
    when span > legacy*divisor."""
    if span_days > legacy * divisor:
        return min(span_days / divisor, cap)
    return legacy


def count_observed_transits(valid_times, t0, period, t14, cadence):
    """Number of transits with >= 50 % of the expected in-transit cadences valid
    (>= 1).  ``valid_times`` = times of cadences surviving the quality mask."""
    vt = np.sort(np.asarray(valid_times, dtype=np.float64))
    if vt.size == 0:
        return 0
    n0 = math.ceil((vt[0] - t14 - t0) / period)
    n1 = math.floor((vt[-1] + t14 - t0) / period)
    need = max(1.0, 0.5 * t14 / max(cadence, 1e-6))
    cnt = 0
    for n in range(n0, n1 + 1):
        tc = t0 + n * period
        lo = np.searchsorted(vt, tc - 0.5 * t14, side="left")
        hi = np.searchsorted(vt, tc + 0.5 * t14, side="right")
        if hi - lo >= need:
            cnt += 1
    return cnt


def unrecoverable(truth_p, n_transits, span_days):
    """(flag, reason) for the spec's by_design_unrecoverable."""
    if n_transits < 2:
        return True, "n_transits<2"
    if truth_p > production_period_max(span_days):
        return True, "P>period_max"
    return False, ""


def best_match(cands, truth, baseline=(), tol=PERIOD_TOL):
    """Pick the best candidate for one injection.

    Returns dict(cls, family, cand_index, rank_by_snr, epoch_off, depth_ratio,
    depth_mismatch, baseline_removed) or cls == 'none'.  Preference: exact, then
    the highest-SNR candidate of the best class; baseline-explained candidates
    are removed first (counted in baseline_removed).
    """
    order = sorted(range(len(cands)), key=lambda i: -(cands[i].get("snr") or 0.0))
    rank = {i: r + 1 for r, i in enumerate(order)}
    best = None
    removed = 0
    prio = {name: n for n, (name, _, _) in enumerate(CLASSES)}
    for i in order:
        c = cands[i]
        if c.get("period_days") is None or c.get("t0") is None:
            continue
        cls, off = classify(c["period_days"], c["t0"], truth["period"], truth["t0"],
                            truth["t14"], tol)
        if cls in ("wrong", "period_only"):
            continue
        if explained_by_baseline(c, baseline, tol):
            removed += 1
            continue
        key = prio[cls]
        if best is None or key < best[0]:
            best = (key, i, cls, off)
    if best is None:
        return dict(cls="none", family="none", cand_index=None, rank_by_snr=None,
                    epoch_off=None, depth_ratio=None, depth_mismatch=None,
                    baseline_removed=removed)
    _, i, cls, off = best
    dr = None
    mism = None
    if cands[i].get("depth_ppm") and truth.get("depth_mean_ppm"):
        dr = cands[i]["depth_ppm"] / truth["depth_mean_ppm"]
        mism = not (0.5 <= dr <= 2.0)
    return dict(cls=cls, family=family_of(cls), cand_index=i, rank_by_snr=rank[i],
                epoch_off=off, depth_ratio=dr, depth_mismatch=mism, baseline_removed=removed)
