"""Epoch-class (parity) alias check for the harmonic resolver.

Background (docs/reviews/2026-10-14-alias-2p-diagnosis.md): BLS often reports
a sub-harmonic P/k of the true period, and the amplitude-based gates of
``resolve_harmonic`` (snr_factor, depth-ratio window) cannot tell it apart
from the real thing. The discriminating evidence is epoch parity: fold the
in-data epochs into k classes (epoch index mod k). At the TRUE period P_true =
k * P_seed exactly ONE class carries the transit and the other k-1 classes
(which ARE covered by data) are flat. At a genuine P all classes carry it.

Everything here is pure, deterministic, numpy-only (no I/O, no randomness).
It is meant to be ported bit-exactly to Swift and Rust, so the arithmetic
order is fixed and documented (spec: docs/reviews/2026-10-14-alias-2p-phase2-
report.md §2 and §7):

* all inputs are float64; ``time`` is sorted ascending (as in the worker);
* every sum is SEQUENTIAL in ascending index order (``_seqsum`` /
  ``np.bincount(..., weights=)`` accumulate ``acc[bin] += w[i]`` for i = 0..n-1);
  numpy's pairwise ``sum``/``mean`` are NOT used for anything that feeds a
  threshold;
* medians are sort-based (``_median``): middle element, or ``(a+b)*0.5`` of the
  two middle elements for even n;
* ``np.rint`` is round-half-to-even; no FMA contraction in the ports.

Rollout: behind the worker config flag ``harmonic_epoch_class`` (default False
-> behaviour bit-identical to before).
"""
from __future__ import annotations

import numpy as np

# --- thresholds (all named; the ports must use the same values) -------------
EPOCH_K_MIN = 2                    # smallest class count tested
EPOCH_K_MAX = 5                    # largest class count tested
EPOCH_WINDOW_HALF_FRAC = 0.5       # in-window half-width = frac * duration
EPOCH_MIN_POINTS = 3               # min in-window points for a covered epoch
EPOCH_MIN_COVER_FRAC = 0.5         # ... and >= this fraction of expected pts
MIN_EPOCHS_PER_CLASS = 2           # covered epochs every class must have
EPOCH_DIP_SIGMA = 2.0              # an epoch "dips" if its depth >= 2 sigma_e
MIN_DIP_EPOCHS = 2                 # a carrying class needs >= 2 dipping epochs
PROMOTE_CARRY_SIGMA = 5.0          # carrying class depth >= 5 sigma
PROMOTE_EMPTY_SIGMA = 2.0          # every other class |depth| < 2 sigma
PROMOTE_MAX_PASSES = 2             # at most 2 successive promotions
PROMOTE_REFIT_FRAC = 0.05          # BLS re-fit window +-5 % around k*P
PROMOTE_REFIT_N_PERIODS = 500      # same as resolve_harmonic probes
PROMOTE_MIN_TRANSITS = 2.0         # promoted period must fit >= 2 in baseline
VETO_CLASS_SIGMA = 3.0             # v0 demotion rule / G1 guard: class >= 3 sigma
DEMOTION_CARRY_SIGMA_V1 = 5.0      # v1 veto: carrying class z (= promotion)
DEMOTION_CARRY_SIGMA_V1P = 3.0     # v1p veto: carrying class z
DEMOTION_VETO_MODE = "vh"          # the ONE mode used by the worker (no config key)
DEMOTION_VETO_MODES = ("vh", "v0", "v1", "v1p", "vs")   # all but vh: A/B harness only
VH_GRID_REL_P = 2.0e-4             # vh: dP/P of the +-5 %/500-period probe grid
DUP_PERIOD_TOL = 0.01              # P/k or k*P match tolerance (1 %)
DUP_EPOCH_DUR_FRAC = 1.5           # epoch coincidence: 1.5 * duration ...
DUP_EPOCH_MIN_DAYS = 0.05          # ... but at least 0.05 d
DRIFT_REL_P = 1e-4                 # period uncertainty searched: dP/P (probe
                                   # grid half-step is 1e-4 * P at +-5 %/500)
DRIFT_MAX_DUR_MULT = 2.0           # accumulated shift searched <= 2 * duration
DRIFT_STEP_FRAC = 0.25             # shift grid step = 0.25 * duration
DRIFT_T0_STEPS = 2                 # t0 offsets searched: +-2 steps (= +-dur/2)
DRIFT_MIN_POINTS = 3               # min in-window points in a trial
BOX_CENTRE_FRAC = 0.5              # centre = box start + 0.5 * duration
_SIGMA_FLOOR = 1e-12               # guards division on noiseless data


def transit_centre(t0, duration, t0_is_centre):
    """Normalise a candidate t0 to the transit CENTRE.

    Every BLS backend (numba/numpy ``bls_fast``, CUDA, Metal, OpenCL) returns
    ``t0 = best_bin_index / n_bins * period + t0_ref`` = the START of the best
    box window [j, j+width). ``apply_duration_refine`` (config
    ``duration_refine``) rewrites t0 to the box CENTRE. Callers say which one
    they hold; everything in this module works on the centre.
    """
    t0 = float(t0)
    if t0_is_centre:
        return t0
    return t0 + BOX_CENTRE_FRAC * float(duration)


def _seqsum(x):
    """Sequential ascending-index float64 sum (``acc += x[i]``)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.bincount(np.zeros(x.size, dtype=np.intp), weights=x,
                             minlength=1)[0])


def _median(x):
    """Sort-based median: s[n//2] (odd) or (s[n/2-1] + s[n/2]) * 0.5 (even)."""
    s = np.sort(np.asarray(x, dtype=np.float64))
    n = s.size
    h = n // 2
    if n % 2:
        return float(s[h])
    return float((s[h - 1] + s[h]) * 0.5)


def prepare_epochs(time, flux, period, t0, duration, excluded=None,
                   t0_is_centre=False):
    """Fix the ephemeris and the per-epoch depths used by every class test.

    Steps (all float64, see module docstring for the arithmetic rules):
    1. Keep finite points that are not ``excluded`` (``excluded`` marks points
       the multi-planet loop filled with the median after masking an earlier
       transit; they are not data). < 10 points or period/duration <= 0 -> None.
    2. ``ref = median(y)``, ``sigma = max(1.4826*median(|y-ref|), 1e-12)``,
       ``cad = median(diff(t))``, ``half = 0.5*duration``,
       ``need = max(3, 0.5 * (2*half/cad))``.
    3. centre0 = ``transit_centre(t0, duration, t0_is_centre)``; reference
       epoch at mid-baseline: ``t_ref = centre0 + rint((tmid-centre0)/P)*P``.
    4. Drift refinement (period-grid error accumulated over the baseline):
       ``span = t[-1]-t[0]``, ``W = min(1e-4*span, 2*duration)``,
       ``step = 0.25*duration``, ``J = ceil(W/step)``,
       ``n_edge = max(0.5*span/P, 1)``. Epochs are ``ep = rint((t-t_ref)/P)``.
       Trials (j = -J..J outer, i = -2..2 inner): ``a = j*step/n_edge``,
       ``b = i*step``; window ``|t - (t_ref + ep*P + b + a*ep)| < half``; score
       = ``ref - seqsum(y_in)/count`` (needs count >= 3). The (0,0) trial is
       evaluated first and only a STRICTLY larger score replaces the best.
       Refined: ``P' = P + a_best``, ``centre' = t_ref + b_best``.
    5. Epochs at the refined ephemeris: ``ep' = rint((t-centre')/P')``,
       in-window ``|t - (centre' + ep'*P')| < half``. Per epoch: ``n_e``
       points, ``d_e = -seqsum(y-ref)/n_e``. Covered iff ``n_e >= need``.

    Returns a dict with epoch-level arrays or None.
    """
    t = np.asarray(time, dtype=np.float64)
    y = np.asarray(flux, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(y)
    if excluded is not None:
        ok &= ~np.asarray(excluded, dtype=bool)
    t = t[ok]
    y = y[ok]
    period = float(period)
    duration = float(duration)
    if t.size < 10 or not (period > 0) or not (duration > 0):
        return None
    ref = _median(y)
    sigma = max(1.4826 * _median(np.abs(y - ref)), _SIGMA_FLOOR)
    dt_med = _median(np.diff(t))
    half = EPOCH_WINDOW_HALF_FRAC * duration
    expected = (2.0 * half) / dt_med if dt_med > 0 else 1.0
    need = max(float(EPOCH_MIN_POINTS), EPOCH_MIN_COVER_FRAC * expected)

    centre0 = transit_centre(t0, duration, t0_is_centre)
    tmid = 0.5 * (t[0] + t[-1])
    t_ref = centre0 + np.rint((tmid - centre0) / period) * period
    ep = np.rint((t - t_ref) / period)

    # --- drift refinement --------------------------------------------------
    span = float(t[-1] - t[0])
    W = min(DRIFT_REL_P * span, DRIFT_MAX_DUR_MULT * duration)
    step = DRIFT_STEP_FRAC * duration
    J = int(np.ceil(W / step)) if W > 0 else 0
    n_edge = max(0.5 * span / period, 1.0)
    b_max = DRIFT_T0_STEPS * step
    near = np.abs(t - (t_ref + ep * period)) < half + W + b_max
    ts, ys, es = t[near], y[near], ep[near]

    def _score(a, b):
        c = t_ref + es * period + b + a * es
        m = np.abs(ts - c) < half
        cnt = int(m.sum())
        if cnt < DRIFT_MIN_POINTS:
            return None
        return ref - _seqsum(ys[m]) / cnt

    best_a, best_b = 0.0, 0.0
    best_s = _score(0.0, 0.0)
    if J > 0 and ts.size:
        for j in range(-J, J + 1):
            a = j * step / n_edge
            for i in range(-DRIFT_T0_STEPS, DRIFT_T0_STEPS + 1):
                if j == 0 and i == 0:
                    continue
                s = _score(a, i * step)
                if s is not None and (best_s is None or s > best_s):
                    best_s, best_a, best_b = s, a, i * step
    P2 = period + best_a
    c2 = t_ref + best_b

    # --- per-epoch depths at the refined ephemeris ----------------------------
    ep2 = np.rint((t - c2) / P2).astype(np.int64)
    inw = np.abs(t - (c2 + ep2 * P2)) < half
    if not inw.any():
        return None
    ep_in = ep2[inw]
    ep0 = int(ep_in.min())
    idx = ep_in - ep0
    m = int(idx.max()) + 1
    n_e = np.bincount(idx, minlength=m).astype(np.float64)
    s_e = np.bincount(idx, weights=y[inw] - ref, minlength=m)
    covered = n_e >= need
    d_e = np.zeros(m)
    d_e[covered] = -s_e[covered] / n_e[covered]
    return {"ep0": ep0, "n_e": n_e, "d_e": d_e, "covered": covered,
            "sigma": sigma, "period": P2, "centre": c2}


def class_stats(prep, k):
    """Per-class statistics from ``prepare_epochs`` output.

    Class of epoch e = ``(ep0 + e) mod k`` (non-negative modulo). For each
    class c over its covered epochs (ascending epoch order):
    ``nep`` = count; ``depth`` = seqsum(d_e)/nep; ``ndip`` = #epochs with
    ``d_e >= 2 * sigma/sqrt(n_e)``; ``err = max(err_white, err_scatter)``
    with ``err_white = sigma/sqrt(sum n_e)`` and, for nep >= 2,
    ``err_scatter = sqrt( seqsum((d_e-depth)^2) / (nep-1) / nep )`` (scatter
    of the per-epoch means: absorbs red noise and single outlier epochs).
    Classes without covered epochs have nan depth/err.
    Returns (depth[k], err[k], nep[k] int64, ndip[k] int64).
    """
    k = int(k)
    nan_k = np.full(k, np.nan)
    zero_k = np.zeros(k, dtype=np.int64)
    if prep is None:
        return nan_k.copy(), nan_k.copy(), zero_k, zero_k.copy()
    n_e, d_e, cov = prep["n_e"], prep["d_e"], prep["covered"]
    sigma = prep["sigma"]
    m = n_e.size
    cls = np.mod(prep["ep0"] + np.arange(m, dtype=np.int64), k)
    dip = np.zeros(m, dtype=bool)
    dip[cov] = d_e[cov] >= EPOCH_DIP_SIGMA * sigma / np.sqrt(n_e[cov])
    cc = cls[cov]
    de = d_e[cov]
    ne = n_e[cov]
    nep = np.bincount(cc, minlength=k).astype(np.int64)
    ndip = np.bincount(cls[dip], minlength=k).astype(np.int64)
    sum_d = np.bincount(cc, weights=de, minlength=k)
    sum_n = np.bincount(cc, weights=ne, minlength=k)
    depth = nan_k.copy()
    err = nan_k.copy()
    has = nep > 0
    depth[has] = sum_d[has] / nep[has]
    err_white = nan_k.copy()
    err_white[has] = sigma / np.sqrt(sum_n[has])
    dev2 = (de - depth[cc]) ** 2
    sum_dev2 = np.bincount(cc, weights=dev2, minlength=k)
    err[has] = err_white[has]
    two = nep >= 2
    err_sc = np.zeros(k)
    err_sc[two] = np.sqrt(sum_dev2[two] / (nep[two] - 1) / nep[two])
    err[two] = np.maximum(err_white[two], err_sc[two])
    return depth, err, nep, ndip


def epoch_class_stats(time, flux, period, t0, duration, k, t0_is_centre=False,
                      excluded=None):
    """prepare_epochs + class_stats in one call: (depth, err, nep, ndip)."""
    return class_stats(
        prepare_epochs(time, flux, period, t0, duration, excluded,
                       t0_is_centre), k)


def _carry(depth, err, nep, ndip, sigma_thr):
    z = depth / np.maximum(err, _SIGMA_FLOOR)
    return (z >= sigma_thr) & (ndip >= MIN_DIP_EPOCHS), z


def _pattern(depth, err, nep, ndip, carry_sigma):
    """True iff exactly one class carries (z >= carry_sigma and >= 2 dipping
    epochs) and every other class has |z| < 2."""
    carry, z = _carry(depth, err, nep, ndip, carry_sigma)
    if int(carry.sum()) != 1:
        return False
    return bool(np.all(np.abs(z[~carry]) < PROMOTE_EMPTY_SIGMA))


def _promotion_pattern(depth, err, nep, ndip):
    """The promotion pattern (carrying class z >= 5)."""
    return _pattern(depth, err, nep, ndip, PROMOTE_CARRY_SIGMA)


def _all_carry(depth, err, nep, ndip):
    carry, _ = _carry(depth, err, nep, ndip, VETO_CLASS_SIGMA)
    return bool(np.all(carry))


def epoch_class_promotion_ks(time, flux, period, t0, duration, period_max,
                             t0_is_centre=False, excluded=None):
    """All k in 2..5 (k*period <= period_max) whose epoch classes show the
    promotion pattern, LARGEST k first (empty list when none). k is skipped
    (undecidable) unless every class has >= MIN_EPOCHS_PER_CLASS covered
    epochs."""
    prep = prepare_epochs(time, flux, period, t0, duration, excluded,
                          t0_is_centre)
    out = []
    for k in range(EPOCH_K_MIN, EPOCH_K_MAX + 1):
        if period * k > period_max:
            break
        depth, err, nep, ndip = class_stats(prep, k)
        if not np.all(nep >= MIN_EPOCHS_PER_CLASS):
            continue
        if _promotion_pattern(depth, err, nep, ndip):
            out.append(k)
    return out[::-1]


def short_period_genuine(time, flux, sub, k, excluded=None):
    """G1 consistency guard for a promotion to ``k*P``.

    ``sub`` is the accepted BLS re-fit at ~k*P (raw result, t0 = box start).
    Re-run the class test on the ephemeris (sub.period / k, sub.t0,
    sub.duration) with k classes. If every class carries (veto rule, z >= 3
    and >= 2 dipping epochs) the short period is genuine and the promotion
    evidence (a smeared seed ephemeris that looked one-sided) is wrong ->
    True (reject the promotion). Undecidable (some class < MIN_EPOCHS_PER_CLASS
    covered epochs) -> False (the promotion stands).
    """
    p = float(sub.get("period", 0.0))
    if not (p > 0):
        return False
    depth, err, nep, ndip = epoch_class_stats(
        time, flux, p / int(k), sub.get("t0", 0.0), sub.get("duration", 0.0),
        int(k), t0_is_centre=False, excluded=excluded)
    if not np.all(nep >= MIN_EPOCHS_PER_CLASS):
        return False
    return _all_carry(depth, err, nep, ndip)


def promote_epoch_class(time, flux, result, period_max, bls_fn,
                        snr_min=5.0, excluded=None, trace=None):
    """Epoch-class promotion of a BLS ``result`` (up to PROMOTE_MAX_PASSES).

    Each pass: find the passing ks, try them from largest to smallest and for
    each re-fit BLS in ``[kP*0.95, min(kP*1.05, period_max)]`` (500 periods).
    The first re-fit with snr >= snr_min and >= PROMOTE_MIN_TRANSITS transits
    in the baseline is accepted (snr_factor and the depth-ratio window are
    deliberately bypassed). The next pass starts from the re-fitted result.
    G1 guard: an otherwise accepted re-fit is REJECTED (next smaller k is
    tried) when ``short_period_genuine`` holds for it. ``trace`` (optional
    list) receives one dict per decision: {"pass", "k", "action", "reason",
    "period"}. ``result`` t0 is the raw BLS box start. Returns ``result``
    itself (same object) when nothing was promoted.
    """
    span = 0.0
    try:
        span = float(time[-1] - time[0])
    except Exception:
        span = 0.0
    cur = result
    for ps in range(PROMOTE_MAX_PASSES):
        P = cur.get("period", 0.0)
        dur = cur.get("duration", 0.0)
        if not (P > 0 and dur > 0):
            break
        promoted = None
        ks = epoch_class_promotion_ks(time, flux, P, cur["t0"], dur,
                                      period_max, excluded=excluded)
        for k in ks:
            lo = P * k * (1.0 - PROMOTE_REFIT_FRAC)
            hi = min(P * k * (1.0 + PROMOTE_REFIT_FRAC), period_max)
            if not (hi > lo):
                continue
            try:
                sub = bls_fn(time, flux, period_min=lo, period_max=hi,
                             n_periods=PROMOTE_REFIT_N_PERIODS)
            except Exception:
                continue
            if sub.get("snr", 0.0) < snr_min:
                continue
            if span > 0 and sub.get("period", 0.0) > 0 \
                    and span / sub["period"] < PROMOTE_MIN_TRANSITS:
                continue
            if short_period_genuine(time, flux, sub, k, excluded):
                if trace is not None:
                    trace.append({"pass": ps, "k": k, "action": "reject",
                                  "reason": "short_period_genuine_all_classes_carry",
                                  "period": float(sub["period"])})
                continue
            if trace is not None:
                trace.append({"pass": ps, "k": k, "action": "promote",
                              "reason": "one_class_carries",
                              "period": float(sub["period"])})
            promoted = sub
            break
        if promoted is None:
            break
        cur = promoted
    return cur


def _segment_local_proceed(time, flux, sub, k, excluded=None):
    """Segment-local V0 (experimental mode "vs", pre-registered in the report
    section 10): split the finite, non-excluded points at gaps longer than the
    probe period, run the class test in each decidable segment (every class
    >= MIN_EPOCHS_PER_CLASS covered epochs; segment >= 10 points) and apply the
    old V0 rule (every class carries) in the segment with the largest summed
    class depth. Returns True (proceed) / False (veto), or None when no segment
    is decidable (caller falls back to V1)."""
    t = np.asarray(time, dtype=np.float64)
    y = np.asarray(flux, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(y)
    if excluded is not None:
        ok &= ~np.asarray(excluded, dtype=bool)
    idx = np.nonzero(ok)[0]
    if idx.size < 10:
        return None
    tt = t[idx]
    cuts = np.nonzero(np.diff(tt) > float(sub.get("period", 0.0)))[0] + 1
    bounds = [0] + [int(c) for c in cuts] + [int(idx.size)]
    best, best_score = None, None
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 10:
            continue
        sl = idx[a:b]
        depth, err, nep, ndip = epoch_class_stats(
            t[sl], y[sl], sub.get("period", 0.0), sub.get("t0", 0.0),
            sub.get("duration", 0.0), k, t0_is_centre=False)
        if not np.all(nep >= MIN_EPOCHS_PER_CLASS):
            continue
        score = _seqsum(depth)
        if best is None or score > best_score:
            best, best_score = (depth, err, nep, ndip), score
    if best is None:
        return None
    return _all_carry(*best)


def demotion_supported(time, flux, sub, k, excluded=None,
                      mode=DEMOTION_VETO_MODE):
    """Veto test for a ``/k`` demotion probe ``sub`` (raw BLS result at P/k).

    Returns False = veto the demotion, True = it may proceed. Undecidable
    (some class < MIN_EPOCHS_PER_CLASS covered epochs) always proceeds
    (existing behaviour). ``mode`` is NOT a worker config option (one mode
    fleet-wide, so Python cannot diverge from the native workers); the other
    modes exist for the A/B harness and raise ValueError when unknown:

    * ``"vh"`` (default, the only mode the worker uses): the hybrid below.
    * ``"v0"``  (revision 0-2): veto unless EVERY class carries (z >= 3,
      >= 2 dipping epochs). Strongest protection against a spurious /k, but
      also vetoes genuine short periods whose probe ephemeris is smeared
      (WASP-164 b).
    * ``"v1"``  (revision 3): veto only on positive alias evidence, the
      promotion pattern at the probe's own ephemeris: exactly one class
      carries at z >= DEMOTION_CARRY_SIGMA_V1 (5) and every other class is
      flat (|z| < 2).
    * ``"v1p"``: same pattern with the carrying class at z >=
      DEMOTION_CARRY_SIGMA_V1P (3).

    vh derivation. The /k probe is a BLS run over +-5 % of P/k with 500
    periods, i.e. a grid of constant relative step dP_grid/P = 2e-4
    (VH_GRID_REL_P, independent of P). Over a baseline ``span`` the folded
    transit has a period uncertainty of ~ P*duration/span (the width of the
    BLS peak). If the grid is finer than the peak, dP_grid/P <= duration/span,
    the best grid period is within about one duration of the truth at the
    baseline ends and the drift search (which absorbs 2 durations) recovers
    the ephemeris -> the class test is reliable -> use v0. If the grid is
    coarser than the peak, span * 2e-4 > duration, the best grid point can be
    several durations off at the ends (cycle-count ambiguity across long
    gaps) and the class depths are diluted -> only positive alias evidence
    (v1) may veto. Equivalent forms: span > 5000 * duration; strict ``>`` so
    equality takes v0. Both span and duration are float64; the test is the
    single product ``span * 2e-4 > duration`` (span = time[-1] - time[0] of
    the raw array; NaN or an empty array -> comparison False -> v0).
    """
    if mode not in DEMOTION_VETO_MODES:
        raise ValueError(f"unknown demotion veto mode {mode!r}")
    depth, err, nep, ndip = epoch_class_stats(
        time, flux, sub.get("period", 0.0), sub.get("t0", 0.0),
        sub.get("duration", 0.0), k, t0_is_centre=False, excluded=excluded)
    if not np.all(nep >= MIN_EPOCHS_PER_CLASS):
        return True
    if mode in ("vh", "vs"):
        span = 0.0
        try:
            span = float(time[-1]) - float(time[0])
        except Exception:
            span = 0.0
        if span * VH_GRID_REL_P > float(sub.get("duration", 0.0)):
            if mode == "vs":
                seg = _segment_local_proceed(time, flux, sub, k, excluded)
                if seg is not None:
                    return seg
            mode = "v1"
        else:
            mode = "v0"
    if mode == "v0":
        return _all_carry(depth, err, nep, ndip)
    sigma = DEMOTION_CARRY_SIGMA_V1P if mode == "v1p" \
        else DEMOTION_CARRY_SIGMA_V1
    return not _pattern(depth, err, nep, ndip, sigma)


def harmonic_parity(time, flux, period, t0, duration, k, excluded=None,
                    t0_is_centre=False):
    """Parity verdict of ephemeris (period, t0, duration) for k classes:
    'alias' (promotion pattern: exactly one class carries, the others are
    flat), 'genuine' (all k classes carry) or 'undecided'."""
    depth, err, nep, ndip = epoch_class_stats(
        time, flux, period, t0, duration, k, t0_is_centre, excluded)
    if not np.all(nep >= MIN_EPOCHS_PER_CLASS):
        return "undecided"
    if _promotion_pattern(depth, err, nep, ndip):
        return "alias"
    if _all_carry(depth, err, nep, ndip):
        return "genuine"
    return "undecided"


def find_harmonic_epoch_match(period, t0, duration, accepted, t_first, t_last,
                              t0_is_centre=False):
    """Return the first accepted candidate dict that (period, t0, duration)
    repeats as P/k or k*P (k=2..5, ratio within 1 %) AND whose epochs coincide
    at the start, the middle and the end of the baseline [t_first, t_last].

    Coincidence at x: take the long ephemeris' transit nearest x, T =
    tl + rint((x-tl)/Pl)*Pl; its distance to the short ephemeris' nearest
    transit must be <= max(1.5*max(dur, dur_acc), 0.05 d). Evaluating at three
    epochs (not just the reference one) rejects genuine near-resonant pairs
    whose ratio 2.00-2.02 drifts apart over the baseline. t0 are normalised to
    centres (``t0_is_centre`` applies to candidate and accepted dicts).
    Returns None when there is no match.
    """
    if not (period > 0):
        return None
    tc = transit_centre(t0, duration, t0_is_centre)
    xs = (float(t_first), 0.5 * (float(t_first) + float(t_last)),
          float(t_last))
    for c in accepted:
        pa = c.get("period_days", 0.0)
        if not (pa > 0):
            continue
        da = float(c.get("duration_days", 0.0))
        ta = transit_centre(c.get("t0", 0.0), da, t0_is_centre)
        if period < pa:
            ps, ts, pl, tl = period, tc, pa, ta
        else:
            ps, ts, pl, tl = pa, ta, period, tc
        r = pl / ps
        k = float(np.rint(r))
        if k < EPOCH_K_MIN or k > EPOCH_K_MAX:
            continue
        if abs(r - k) / k >= DUP_PERIOD_TOL:
            continue
        tol = max(DUP_EPOCH_DUR_FRAC * max(float(duration), da),
                  DUP_EPOCH_MIN_DAYS)
        ok = True
        for x in xs:
            T = tl + np.rint((x - tl) / pl) * pl
            Ts = ts + np.rint((T - ts) / ps) * ps
            if abs(float(T - Ts)) > tol:
                ok = False
                break
        if ok:
            return c
    return None


def slot_duplicate(time, flux, rm_period, rm_t0, rm_duration, accepted,
                   excluded=None, t0_is_centre=False):
    """Decide whether a protected-slot candidate should be DROPPED because it
    is an epoch-coincident P/k or k*P of an accepted candidate, with the
    direction decided by parity:

    * slot candidate is the SHORT one (P/k of accepted): parity of its own
      ephemeris with k classes: 'alias' -> drop (accepted long one is the
      truth); 'genuine' -> keep (the accepted long period is the multiple);
      'undecided' -> drop (previous behaviour).
    * slot candidate is the LONG one (k*P of accepted): parity of the
      ACCEPTED short ephemeris: 'alias' -> keep (the long one is the truth);
      'genuine' -> drop; 'undecided' -> drop.
    No harmonic/epoch match -> keep (False).
    """
    t = np.asarray(time, dtype=np.float64)
    fin = t[np.isfinite(t)]
    if fin.size < 2 or not (rm_period > 0):
        return False
    c = find_harmonic_epoch_match(rm_period, rm_t0, rm_duration, accepted,
                                  float(fin[0]), float(fin[-1]), t0_is_centre)
    if c is None:
        return False
    pa = float(c["period_days"])
    if rm_period < pa:
        k = int(np.rint(pa / rm_period))
        v = harmonic_parity(time, flux, rm_period, rm_t0, rm_duration, k,
                            excluded, t0_is_centre)
        return v != "genuine"
    k = int(np.rint(rm_period / pa))
    v = harmonic_parity(time, flux, pa, c["t0"], c.get("duration_days", 0.0),
                        k, excluded, t0_is_centre)
    return v != "alias"
