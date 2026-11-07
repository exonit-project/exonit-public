"""Harmonic-duplicate detection between candidates of the SAME TIC and sector.

Why (docs/reviews/2026-10-14-alias-2p-diagnosis.md): the medium protected slot
([3,10] d) of the worker only de-duplicates within 2 % of an accepted period,
so P/k (or k*P) of an already accepted candidate is re-reported as a "new"
candidate. The strong list / dossier keeps ONE row per TIC, chosen by v6 score,
and could publish that sub-harmonic duplicate instead of the fundamental
(TIC 31943299 s1: 12.47 d SNR 397 accepted, 6.24 d published).

A candidate D is a *harmonic duplicate* of K when, for K != D of the same
(tic_id, sector):
  * K.snr > D.snr (strictly; the stronger signal is the one kept),
  * the period ratio long/short is within ``rtol`` (default 1 %) of an
    integer k in 2..``kmax`` (5), and
  * the transit epochs coincide: the t0 difference modulo the SHORTER period,
    folded to [-P/2, P/2], is <= tol, with
        tol = min(max(1.5 * max(dur_D, dur_K), 0.05 d), 0.25 * P_short).
    1.5 x the longer duration absorbs BLS t0 jitter (~ +-0.5 dur) plus the
    epoch drift of a <= 1 % period error over a few cycles; the 0.05 d floor
    covers very short durations; the 0.25 P_short cap keeps the test
    meaningful for short periods (a tolerance >= P/2 would match anything).
    A real second planet at a near-resonant period has an unrelated phase, so
    it fails this test and survives.
Any missing / non-finite / non-positive period, t0, duration or SNR means
"cannot prove it" -> never marked.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

DEFAULT_RTOL = 0.01
DEFAULT_KMAX = 5
MIN_EPOCH_TOL_D = 0.05
EPOCH_TOL_DUR_FACTOR = 1.5
EPOCH_TOL_MAX_PERIOD_FRAC = 0.25


def _num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def harmonic_relation(dup, keeper, rtol: float = DEFAULT_RTOL,
                      kmax: int = DEFAULT_KMAX) -> Optional[int]:
    """Return k (2..kmax) if ``dup`` is a harmonic duplicate of ``keeper``.

    Both are mappings with period_days, t0, duration_days, snr. Returns None
    when not provable (see module docstring).
    """
    pd_, pk = _num(dup.get("period_days")), _num(keeper.get("period_days"))
    td, tk = _num(dup.get("t0")), _num(keeper.get("t0"))
    dd, dk = _num(dup.get("duration_days")), _num(keeper.get("duration_days"))
    sd, sk = _num(dup.get("snr")), _num(keeper.get("snr"))
    if None in (pd_, pk, td, tk, dd, dk, sd, sk):
        return None
    if pd_ <= 0 or pk <= 0 or dd <= 0 or dk <= 0:
        return None
    if not sk > sd:
        return None
    p_short, p_long = (pd_, pk) if pd_ <= pk else (pk, pd_)
    ratio = p_long / p_short
    k = round(ratio)
    if k < 2 or k > kmax or abs(ratio - k) / k > rtol:
        return None
    tol = max(EPOCH_TOL_DUR_FACTOR * max(dd, dk), MIN_EPOCH_TOL_D)
    tol = min(tol, EPOCH_TOL_MAX_PERIOD_FRAC * p_short)
    delta = (td - tk + 0.5 * p_short) % p_short - 0.5 * p_short
    return k if abs(delta) <= tol else None


def find_harmonic_duplicates(cands: Iterable[dict], protected_ids=(),
                             rtol: float = DEFAULT_RTOL,
                             kmax: int = DEFAULT_KMAX) -> dict:
    """Map duplicate candidate id -> (keeper candidate dict, k).

    ``cands`` are rows with id, tic_id, sector, period_days, t0, duration_days,
    snr; only rows sharing (tic_id, sector) are compared. ``protected_ids``
    are never marked (anti-TOI protection). The keeper of each duplicate is
    its strongest (highest-SNR) related candidate.
    """
    protected = set(protected_ids)
    groups: dict = {}
    for c in cands:
        groups.setdefault((c.get("tic_id"), c.get("sector")), []).append(c)
    out: dict = {}
    for grp in groups.values():
        if len(grp) < 2:
            continue
        for d in grp:
            if d["id"] in protected:
                continue
            best = None
            for kp in grp:
                if kp["id"] == d["id"]:
                    continue
                k = harmonic_relation(d, kp, rtol=rtol, kmax=kmax)
                if k is not None and (best is None or float(kp["snr"]) > float(best[0]["snr"])):
                    best = (kp, k)
            if best is not None:
                out[d["id"]] = best
    return out
