"""Systematic-period veto v2 (EXONIT_VETO_V2) — density-excess families.

Why (docs/reviews/2026-10-15-ir-pilot-loss-analysis.md, finding 1-2): the v1
families are 2 %-wide period bins holding >= 10 distinct TICs.  On a campaign
population of tens of millions of rows every 2 % bin below 2 d holds thousands
of TICs, so the 73 "families" cover 100 % of 0.5-2 d and veto every non-KP/CP
short-period candidate (a blanket cut, not a systematic detector); the
epoch-clustered path (T0 mod P concentration >= 0.4, phase window +-10 % of P)
is no more selective than random at >= 5 d.

v2 flags a period only when ALL of these hold (constants below, calibrated in
docs/reviews/2026-10-15-systematic-veto-fix-report.md):

 1. DENSITY EXCESS: the number of distinct TICs in a fine log-period bin
    (``V2_BIN_LN`` = 0.25 %) is >= ``V2_DENSITY_K`` x the local background (the
    higher of the left / right medians of the neighbouring bins,
    ``V2_BG_GAP``..``V2_BG_HALF`` bins away) and a Poisson z >= ``V2_MIN_Z`` and
    >= ``V2_MIN_TICS`` TICs.  Contiguous flagged bins (+ ``V2_PERIOD_PAD_BINS``
    of peak wing) form one family; its period interval [p_lo, p_hi] IS the
    match tolerance (data-driven, ~+-0.3-0.8 % instead of a fixed +-2 %).
    Families are detected per sector (plus a pooled epoch-agnostic set), see
    ``find_systematic_families_v2``.
 2. EPOCH (P >= ``V2_AGNOSTIC_MAX_PERIOD``): the family's T0 values are
    clustered in absolute time.  Measured on the campaign DB the long-period
    systematics are single instrumental events (T0 = 1348.83 BTJD for s1,
    1696.39 for the s14 mid-sector orbit gap, 1723.6 for s15, 1791.3 for s18,
    ...: 75-98 % of a family's rows sit in <= 3 bins of 0.25 d), whereas
    T0 mod P is NOT stable inside a family (P differs by up to +-0.5 % between
    rows, 100+ cycles -> the phase drifts by days).  A family keeps the
    ``t0_modes`` (0.25-d T0 bins holding >= ``V2_T0_MODE_FRAC`` of its rows) and
    is accepted only when those modes hold >= ``V2_MIN_T0_MASS`` of the rows; a
    candidate shares the epoch when its T0 lies in a mode (+- ``V2_T0_PAD``).
    Below ``V2_AGNOSTIC_MAX_PERIOD`` families are epoch-agnostic (the
    daily/sub-daily artefacts), as in v1.
 3. SHAPE SIMILARITY: the family carries a duration band and a depth band, the
    central ``V2_DUR_BAND_MASS`` / ``V2_DEPTH_BAND_MASS`` of the EXCESS (family
    histogram minus the scaled neighbour histogram) plus a margin; a candidate
    is vetoed only inside both bands.
 4. SOFT TAG: an epoch-agnostic match with SNR >= ``V2_SNR_SOFT`` is NOT
    vetoed (dossier annotation only); an epoch-clustered match (same epoch)
    is vetoed whatever the SNR.

Everything here is additive: v1 code paths are untouched; v2 families carry
``"v2": True`` and ``exonit.systematic_veto.is_vetoed`` only takes the v2 branch
for such entries.
"""
from __future__ import annotations

import math
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Named constants (calibration: docs/reviews/2026-10-15-systematic-veto-fix-report.md)
# ---------------------------------------------------------------------------
V2_BIN_LN: float = 0.0025          # log-period bin width (0.25 %), ~ BLS peak width
V2_BG_GAP: int = 2                 # epoch-agnostic: neighbour bins skipped on each side of a bin
V2_BG_HALF: int = 12               # epoch-agnostic: neighbour window half-width (bins, +-3 %)
V2_BG_GAP_EPOCH: int = 4           # epoch-clustered: broader humps (13.7 d orbit gap is ~+-4 % wide)
V2_BG_HALF_EPOCH: int = 40         # -> background window out to +-10 %
V2_DENSITY_K: float = 3.0          # epoch-agnostic family: distinct-TIC density >= K x local background
V2_DENSITY_K_EPOCH: float = 2.0    # epoch-clustered family (P >= agnostic max): lower, the epoch test carries the proof
V2_MIN_Z: float = 5.0              # Poisson significance of the excess
V2_MIN_TICS: int = 30              # absolute floor on distinct TICs in a flagged bin
V2_T0_BIN: float = 0.25            # epoch-clustered: absolute-T0 bin (days)
V2_T0_MODE_FRAC: float = 0.05      # a T0 bin is a mode if it holds >= this fraction of the family rows
V2_MIN_T0_MASS: float = 0.5        # epoch-clustered family needs >= this fraction of rows in T0 modes
V2_T0_PAD: float = 0.125           # extra T0 tolerance each side of a mode (days)
V2_MAX_T0_MODES: int = 24
V2_PERIOD_PAD_BINS: int = 1        # match interval = flagged bins +- this many bins (BLS peak wings)
V2_RUN_GAP: int = 1                # flagged bins separated by <= this many bins form one family
V2_SNR_SOFT: float = 50.0          # epoch-agnostic match with SNR >= this: soft tag only
V2_DUR_BAND_MASS: float = 0.98     # central mass of the family's excess kept in the duration band
V2_DEPTH_BAND_MASS: float = 0.998  # ... and in the depth band (depth is broad: wide band)
V2_DUR_MARGIN_DEX: float = 0.05    # margin added to each side of the duration band
V2_DEPTH_MARGIN_DEX: float = 0.1   # ... and of the depth band
V2_AGNOSTIC_MAX_PERIOD: float = 2.0
# --- T0-event test (round 2): a candidate whose (period class, T0) cell holds far more
# TICs than the uniform-T0 expectation of its sector/period class is the same instrumental
# event as thousands of other rows, whatever its period family.
V2_T0EVENT_PB: float = 0.05        # period class width (ln P, 5 %)
V2_T0EVENT_MIN_TICS: int = 30      # distinct TICs in a (sector, period class, T0 bin) cell
V2_T0EVENT_SIGMA: float = 2.0      # K_T0 = exp(median + SIGMA * sigma_left) of the log cell/expectation ratio
V2_T0EVENT_MIN_PERIOD: float = 1.0     # classes with P_mid < 1 d (support < 4 bins) are skipped anyway
V2_T0EVENT_SECTOR_SPAN_DAYS: float = 28.5   # physical cap of one TESS sector's T0 span (27.4 d + margin)
V2_T0EVENT_ALIVE_FRAC: float = 0.1     # a T0 bin is 'alive' (not in a gap) if the sector-wide count >= this x median
V2_T0EVENT_EDGE_FRAC: float = 0.01    # also test the adjacent period class within 1 % of the shared edge
V2_T0EVENT_MIN_CELLS: int = 50     # need at least this many populated cells to estimate the background

_LN_P0 = math.log(0.2)             # origin of the log-period grid
_DUR_EDGES = np.arange(-3.0, 0.5001, 0.05)     # log10(days)
_DEPTH_EDGES = np.arange(0.0, 9.5001, 0.1)     # log10(ppm)


def veto_v2_mode() -> str:
    """EXONIT_VETO_V2: ``"off"`` (default, unset/0/anything else), ``"on"`` (=1) or
    ``"shadow"``.  Shadow: v1 drives everything (DB, dossier, strong list byte-identical to
    off); v2 is only computed for the side report ``<run>_veto_v2_shadow.{json,md}``.
    Default OFF until a shadow run + scorecard exist (code review 2026-10-15)."""
    v = os.environ.get("EXONIT_VETO_V2", "0").strip().lower()
    return "on" if v == "1" else ("shadow" if v == "shadow" else "off")


def veto_v2_enabled() -> bool:
    """True only when v2 DRIVES the veto (``EXONIT_VETO_V2=1``); shadow mode is False."""
    return veto_v2_mode() == "on"


def veto_v2_shadow() -> bool:
    return veto_v2_mode() == "shadow"


def _neighbour_median(cnt: np.ndarray, gap: int, half: int) -> np.ndarray:
    """Local background of every bin: the HIGHER of the left and right
    neighbour medians (bins ``gap+1``..``half`` away on each side).

    Taking the max of the two sides means a step at the edge of the data (or of
    a density plateau) is not mistaken for a spike: a real spike must exceed
    the background on BOTH sides.
    """
    nb = len(cnt)
    pad = np.full(nb + 2 * half, np.nan)
    pad[half:half + nb] = cnt
    sides = []
    for sign in (-1, 1):
        offs = [sign * o for o in range(gap + 1, half + 1)]
        stack = np.stack([pad[half + o: half + o + nb] for o in offs], axis=1)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sides.append(np.nanmedian(stack, axis=1))
    left, right = sides
    # array edge (one side all-NaN): use the other side
    bg = np.fmax(left, right)
    # a side beyond the first/last populated bin counts as 0 (data edge)
    populated = np.nonzero(cnt > 0)[0]
    if len(populated):
        lo, hi = populated[0], populated[-1]
        idx = np.arange(nb)
        l0 = np.where(idx - gap - 1 < lo, 0.0, left)
        r0 = np.where(idx + gap + 1 > hi, 0.0, right)
        bg = np.fmax(l0, r0)
    return np.where(np.isfinite(bg), bg, 0.0)


def _hist_band(vals: np.ndarray, edges: np.ndarray, excess_w: float,
               bg_vals: np.ndarray, mass: float,
               margin: float) -> Optional[Tuple[float, float]]:
    """Band [lo, hi] (log10 units) of the central ``mass`` of the excess.

    excess = hist(family rows) - excess_w_scaled hist(background rows), where
    the background histogram is scaled to the expected background mass inside
    the family (``bg_frac`` = 1 - 1/ratio carried in ``excess_w``).
    """
    ok = np.isfinite(vals)
    h_f, _ = np.histogram(vals[ok], bins=edges)
    okb = np.isfinite(bg_vals)
    h_b, _ = np.histogram(bg_vals[okb], bins=edges)
    n_f, n_b = int(h_f.sum()), int(h_b.sum())
    if n_f == 0:
        return None
    if n_b > 0:
        exp_bg = h_b / n_b * (n_f * excess_w)
    else:
        exp_bg = np.zeros_like(h_b, dtype=float)
    exc = np.clip(h_f - exp_bg, 0.0, None)
    tot = exc.sum()
    if tot <= 0:
        return None
    c = np.cumsum(exc) / tot
    tail = (1.0 - mass) / 2.0
    i_lo = int(np.searchsorted(c, tail, side="left"))
    i_hi = int(np.searchsorted(c, 1.0 - tail, side="left"))
    i_hi = min(i_hi, len(edges) - 2)
    return float(edges[i_lo]) - margin, float(edges[i_hi + 1]) + margin


def _families_one_scope(
    pop,
    density_k: float = V2_DENSITY_K,
    density_k_epoch: float = V2_DENSITY_K_EPOCH,
    min_z: float = V2_MIN_Z,
    min_tics: int = V2_MIN_TICS,
    min_t0_mass: float = V2_MIN_T0_MASS,
    agnostic_max_period: float = V2_AGNOSTIC_MAX_PERIOD,
) -> List[Dict[str, Any]]:
    """v2 families of ONE scope (one sector, or the pooled population)."""
    periods, tics, t0s = pop.period, pop.tic, pop.t0
    have_shape = len(getattr(pop, "dur", ())) == len(periods) and len(periods) > 0
    valid = np.isfinite(periods) & (periods > 0)
    vidx = np.nonzero(valid)[0]
    if len(vidx) == 0:
        return []
    lnp = np.log(periods[vidx])
    bins = np.floor((lnp - _LN_P0) / V2_BIN_LN).astype(np.int64)
    keep = bins >= 0
    vidx, bins = vidx[keep], bins[keep]
    if len(vidx) == 0:
        return []
    nb = int(bins.max()) + 1
    tic_v = tics[vidx]
    if tic_v.size and int(tic_v.max()) >= (1 << 32):
        raise ValueError("find_systematic_families_v2: tic_id >= 2^32 unsupported")
    ukey = np.unique((bins << 32) | tic_v.astype(np.int64))
    cnt = np.bincount((ukey >> 32).astype(np.int64), minlength=nb).astype(float)
    del ukey
    centers = np.exp(_LN_P0 + (np.arange(nb) + 0.5) * V2_BIN_LN)
    is_ag = centers < agnostic_max_period
    bg = np.where(is_ag, _neighbour_median(cnt, V2_BG_GAP, V2_BG_HALF),
                  _neighbour_median(cnt, V2_BG_GAP_EPOCH, V2_BG_HALF_EPOCH))
    ratio = cnt / np.maximum(bg, 1.0)
    z = (cnt - bg) / np.sqrt(bg + 1.0)
    k_bin = np.where(is_ag, density_k, density_k_epoch)
    flag = (cnt >= k_bin * np.maximum(bg, 1.0)) & (z >= min_z) & (cnt >= min_tics)
    if not flag.any():
        return []
    # contiguous runs of flagged bins
    fb = np.nonzero(flag)[0]
    splits = np.nonzero(np.diff(fb) > V2_RUN_GAP + 1)[0] + 1
    runs = [(int(r[0]), int(r[-1])) for r in np.split(fb, splits)]

    order = np.argsort(bins, kind="stable")
    sbins = bins[order]
    svidx = vidx[order]

    def rows_in(b0: int, b1: int) -> np.ndarray:
        a = np.searchsorted(sbins, b0, side="left")
        b = np.searchsorted(sbins, b1, side="right")
        return svidx[a:b]

    out: List[Dict[str, Any]] = []
    for b0, b1 in runs:
        pk = b0 + int(np.argmax(cnt[b0:b1 + 1]))
        p_peak = float(centers[pk])
        p_lo = float(math.exp(_LN_P0 + (b0 - V2_PERIOD_PAD_BINS) * V2_BIN_LN))
        p_hi = float(math.exp(_LN_P0 + (b1 + 1 + V2_PERIOD_PAD_BINS) * V2_BIN_LN))
        agnostic = p_peak < agnostic_max_period
        rows = rows_in(b0, b1)
        entry: Dict[str, Any] = {
            "period": round(p_peak, 4),
            "n_stars": int(len(np.unique(tics[rows]))),
            "epoch_agnostic": bool(agnostic),
            "v2": True,
            "ratio": round(float(ratio[pk]), 2),
            "p_lo": round(p_lo, 5),
            "p_hi": round(p_hi, 5),
        }
        if not agnostic:
            t = t0s[rows]
            t = t[np.isfinite(t)]
            if len(t) < min_tics:
                continue
            tb = np.floor(t / V2_T0_BIN).astype(np.int64)
            ub, uc = np.unique(tb, return_counts=True)
            sel = uc >= V2_T0_MODE_FRAC * len(t)
            if not sel.any():
                continue
            ub, uc = ub[sel], uc[sel]
            t0_mass = float(uc.sum()) / len(t)
            if t0_mass < min_t0_mass:
                continue
            modes: List[List[float]] = []
            for k in np.argsort(ub):                # merge adjacent bins
                lo_t = float(ub[k]) * V2_T0_BIN - V2_T0_PAD
                hi_t = float(ub[k] + 1) * V2_T0_BIN + V2_T0_PAD
                if modes and lo_t <= modes[-1][1]:
                    modes[-1][1] = max(modes[-1][1], hi_t)
                else:
                    modes.append([lo_t, hi_t])
            if len(modes) > V2_MAX_T0_MODES:
                continue                            # too diffuse to be a handful of events
            entry["t0_modes"] = [[round(a, 3), round(b, 3)] for a, b in modes]
            entry["t0_mass"] = round(t0_mass, 3)
        if have_shape:
            # background rows: neighbouring bins that are not themselves flagged
            nbr = []
            gap, half = ((V2_BG_GAP, V2_BG_HALF) if agnostic
                         else (V2_BG_GAP_EPOCH, V2_BG_HALF_EPOCH))
            for lo_b, hi_b in ((b0 - half, b0 - gap - 1), (b1 + gap + 1, b1 + half)):
                lo_b = max(lo_b, 0)
                if hi_b >= lo_b:
                    nbr.append(rows_in(lo_b, hi_b))
            bg_rows = np.concatenate(nbr) if nbr else np.empty(0, dtype=np.int64)
            run_ratio = float(max(ratio[b0:b1 + 1].mean(), 1.0))
            w = 1.0 / run_ratio                      # background fraction inside the family
            with np.errstate(all="ignore"):
                f_dur = np.log10(np.clip(pop.dur[rows], 1e-6, None))
                b_dur = np.log10(np.clip(pop.dur[bg_rows], 1e-6, None))
                f_dep = np.log10(np.clip(np.abs(pop.depth[rows]), 1.0, None))
                b_dep = np.log10(np.clip(np.abs(pop.depth[bg_rows]), 1.0, None))
            f_dur[~np.isfinite(pop.dur[rows])] = np.nan
            f_dep[~np.isfinite(pop.depth[rows])] = np.nan
            b_dur[~np.isfinite(pop.dur[bg_rows])] = np.nan
            b_dep[~np.isfinite(pop.depth[bg_rows])] = np.nan
            bd = _hist_band(f_dur, _DUR_EDGES, w, b_dur, V2_DUR_BAND_MASS, V2_DUR_MARGIN_DEX)
            bp = _hist_band(f_dep, _DEPTH_EDGES, w, b_dep, V2_DEPTH_BAND_MASS, V2_DEPTH_MARGIN_DEX)
            if bd is not None:
                entry["dur_lo"], entry["dur_hi"] = round(10 ** bd[0], 5), round(10 ** bd[1], 5)
            if bp is not None:
                entry["depth_lo"], entry["depth_hi"] = round(10 ** bp[0], 1), round(10 ** bp[1], 1)
        out.append(entry)
    out.sort(key=lambda e: -e["n_stars"])
    return out


def find_systematic_families_v2(
    pop,
    density_k: float = V2_DENSITY_K,
    density_k_epoch: float = V2_DENSITY_K_EPOCH,
    min_z: float = V2_MIN_Z,
    min_tics: int = V2_MIN_TICS,
    min_t0_mass: float = V2_MIN_T0_MASS,
    agnostic_max_period: float = V2_AGNOSTIC_MAX_PERIOD,
    t0_events: bool = True,
) -> List[Dict[str, Any]]:
    """Return v2 systematic families from a ``ColumnarPopulation``.

    ``t0_events`` (default True) appends ONE extra entry (``"kind": "t0_events"``)
    holding the flagged (period class, T0) cells, see ``_t0_event_cells``.

    Families are detected PER SECTOR when ``pop.sector`` is filled (each
    family carries ``"sector"``): instrumental systematics are sector-specific
    (their T0 is a given event of a given sector), and on a pooled campaign
    the biggest sectors (s14 = 30 % of the rows) drown the density excess of
    the small ones.  The epoch-agnostic (P < 2 d) families are ALSO detected on
    the pooled population (no ``"sector"`` key) because daily artefacts are not
    sector-specific.  Without ``pop.sector`` the population is treated as one
    scope (no ``"sector"`` key; matches every candidate).

    ``pop`` needs ``tic``, ``period``, ``t0`` and (for the shape bands)
    non-empty ``dur`` / ``depth`` arrays (``ColumnarPopulation.from_rows(...,
    with_shape=True)``); without them the shape bands are omitted and the
    similarity test is skipped.
    """
    from exonit.systematic_veto import ColumnarPopulation
    kw = dict(density_k=density_k, density_k_epoch=density_k_epoch, min_z=min_z,
              min_tics=min_tics, min_t0_mass=min_t0_mass,
              agnostic_max_period=agnostic_max_period)
    n = len(pop.period)
    sec = getattr(pop, "sector", None)
    if sec is None or len(sec) != n or n == 0:
        return _families_one_scope(pop, **kw)
    have_shape = len(pop.dur) == n
    out: List[Dict[str, Any]] = []
    for sv in np.unique(sec):
        m = sec == sv
        sub = ColumnarPopulation(
            tic=pop.tic[m], period=pop.period[m], t0=pop.t0[m],
            dur=pop.dur[m] if have_shape else np.empty(0),
            depth=pop.depth[m] if have_shape else np.empty(0))
        for f in _families_one_scope(sub, **kw):
            f["sector"] = int(sv)
            out.append(f)
    # Daily/sub-daily artefacts (0.5 d, 1.0 d, 2/3 d ...) are not sector-specific and
    # the small sectors lack the statistics to see them: add the POOLED epoch-agnostic
    # families (no "sector" key = match every candidate).
    short = np.isfinite(pop.period) & (pop.period < agnostic_max_period * 1.05)
    if short.any():
        sub = ColumnarPopulation(
            tic=pop.tic[short], period=pop.period[short], t0=pop.t0[short],
            dur=pop.dur[short] if have_shape else np.empty(0),
            depth=pop.depth[short] if have_shape else np.empty(0))
        out.extend(f for f in _families_one_scope(sub, **kw) if f["epoch_agnostic"])
    out.sort(key=lambda e: (e.get("sector", 10**6), -e["n_stars"]))
    if t0_events:
        ev = _t0_event_cells(pop)
        if ev is not None:
            out.append(ev)
    return out


def _t0_event_cells(pop) -> Optional[Dict[str, Any]]:
    """Entry with the (period class, T0) cells that are T0 EVENTS, or None.

    NULL (measured, round 2c).  The T0 the pipeline reports is the FIRST transit in the data.
    Its support in a sector is the first ``min(P, L_s)`` days, ``L_s`` = the sector's T0 span:
    s23 (L_s = 21 d) at P = 16.0 / 18.6 / 19.6 d has an observed 1-99 % T0 span of
    17.6 / 18.8 / 19.7 d, whereas the "at least 2 transits" support ``L - P`` would be
    4.9 / 2.3 / 1.3 d; s7, s17, s26 agree (span ~ P up to ~20 d, then ~ L_s).  So the expected
    distinct-TIC count of a 0.25-d T0 bin is ``N_window / n_support`` where n_support = number
    of ALIVE 0.25-d bins in the first ``min(P_mid, L_s)`` days of the sector's T0 window:
    * ``L_s`` / the sector window: the densest ``V2_T0EVENT_SECTOR_SPAN_DAYS`` T0 window of all
      the sector's rows; rows outside it (stray epochs of stitched multi-sector light curves)
      are excluded from the null and never create cells;
    * gaps (mid-sector orbit gap, data gaps): bins whose sector-wide count is < ``V2_T0EVENT_ALIVE_FRAC``
      x the median populated bin are not counted in n_support (no events are expected there).
    Period classes are ``V2_T0EVENT_PB`` (5 % in ln P) wide; classes with fewer than 4 support
    bins are skipped.  A cell is an event if it holds >= ``V2_T0EVENT_MIN_TICS`` TICs, a Poisson
    z >= ``V2_MIN_Z`` and cell/expectation >= K_T0, where K_T0 = exp(median + SIGMA * sigma_left) of the
    log ratio over all cells (sigma_left = median - 15.9th percentile): estimated from the
    population's own background, derivation unchanged since round 2 (SIGMA = 2).  Matching is by
    absolute T0 (sector-independent) and period class; NaN T0 never matches.
    """
    sec = getattr(pop, "sector", None)
    n = len(pop.period)
    if n == 0:
        return None
    if sec is None or len(sec) != n:
        sec = np.zeros(n, dtype=np.int16)
    PB, TB = V2_T0EVENT_PB, V2_T0_BIN
    W = int(math.ceil(V2_T0EVENT_SECTOR_SPAN_DAYS / TB))
    ok = np.isfinite(pop.period) & (pop.period >= V2_T0EVENT_MIN_PERIOD) & np.isfinite(pop.t0)
    cells_all = []                     # (pb, tb, count, expectation)
    for sv in np.unique(sec[ok]):
        m = ok & (sec == sv)
        pb = np.floor(np.log(pop.period[m]) / PB).astype(np.int64)
        tb = np.floor(pop.t0[m] / TB).astype(np.int64)
        tic = pop.tic[m].astype(np.int64)
        if tic.size and int(tic.max()) >= (1 << 32):
            raise ValueError("t0 events: tic_id >= 2^32 unsupported")
        # --- sector T0 window (densest W bins over all periods) and alive mask
        tb0 = int(tb.min())
        occ = np.bincount(tb - tb0).astype(float)
        cso = np.concatenate(([0.0], np.cumsum(occ)))
        ends = np.minimum(np.arange(len(occ)) + W, len(occ))
        ws = int(np.argmax(cso[ends] - cso[np.arange(len(occ))])) + tb0
        win = occ[ws - tb0: ws - tb0 + W]
        pos = win[win > 0]
        alive = win >= V2_T0EVENT_ALIVE_FRAC * (float(np.median(pos)) if len(pos) else 0.0)
        alive &= win > 0
        inw = (tb >= ws) & (tb < ws + W)
        pb, tb, tic = pb[inw], tb[inw], tic[inw]
        if len(pb) == 0:
            continue
        pb0, tbm = int(pb.min()), ws
        key = ((pb - pb0) << 52) | ((tb - tbm) << 32) | tic
        u = np.unique(key)
        ck, cc = np.unique(u >> 32, return_counts=True)
        cpb = (ck >> 20) + pb0
        ctb = (ck & 0xFFFFF) + tbm
        order = np.argsort(cpb, kind="stable")
        cpb, ctb, cc = cpb[order], ctb[order], cc[order]
        bounds = np.flatnonzero(np.diff(cpb)) + 1
        for a, b in zip(np.concatenate(([0], bounds)), np.concatenate((bounds, [len(cpb)]))):
            p = int(cpb[a])
            cnt, tbs = cc[a:b].astype(float), ctb[a:b]
            pmid = math.exp((p + 0.5) * PB)
            sup_bins = min(int(math.ceil(pmid / TB)), W)
            if sup_bins < 4:
                continue                       # period class too short: <1 d windows carry no spike information
            # class window = densest min(P_hi, L_s)-wide T0 window inside the sector window
            width = min(int(math.ceil(math.exp((p + 1) * PB) / TB)) + 2, W)
            cs = np.concatenate(([0.0], np.cumsum(cnt)))
            e2 = np.searchsorted(tbs, tbs + width, side="left")
            best = int(np.argmax(cs[e2] - cs[np.arange(len(tbs))]))
            t_start = int(tbs[best])
            inwin = (tbs >= t_start) & (tbs < t_start + width)
            lo_i = t_start - ws
            sup = alive[max(lo_i, 0): max(lo_i, 0) + sup_bins]
            n_sup = int(sup.sum())
            if n_sup < 4:
                continue
            expect = cnt[inwin].sum() / n_sup
            for t, c in zip(tbs[inwin], cnt[inwin]):
                if c >= V2_T0EVENT_MIN_TICS:
                    cells_all.append((p, int(t), float(c), float(max(expect, 1e-9))))
    if len(cells_all) < V2_T0EVENT_MIN_CELLS:
        return None                    # too few cells to calibrate the background (small population)
    arr = np.array(cells_all)
    ratio = arr[:, 2] / arr[:, 3]
    lr = np.log(np.clip(ratio, 1e-3, None))
    med = float(np.median(lr))
    sigma_left = med - float(np.percentile(lr, 15.87))
    k_t0 = math.exp(med + V2_T0EVENT_SIGMA * sigma_left)
    z = (arr[:, 2] - arr[:, 3]) / np.sqrt(arr[:, 3] + 1.0)
    hot = (ratio >= k_t0) & (z >= V2_MIN_Z)
    if not hot.any():
        return None
    by_pb: Dict[int, List[int]] = {}
    for p, t in zip(arr[hot, 0].astype(int), arr[hot, 1].astype(int)):
        by_pb.setdefault(int(p), []).append(int(t))
    cells: Dict[str, List[List[float]]] = {}
    for p, ts in by_pb.items():
        iv: List[List[float]] = []
        for t in sorted(set(ts)):
            lo, hi = t * TB - V2_T0_PAD, (t + 1) * TB + V2_T0_PAD
            if iv and lo <= iv[-1][1]:
                iv[-1][1] = max(iv[-1][1], hi)
            else:
                iv.append([lo, hi])
        cells[str(p)] = [[round(a, 3), round(b, 3)] for a, b in iv]
    return {"period": 0.0, "n_stars": int(hot.sum()), "epoch_agnostic": False, "v2": True,
            "kind": "t0_events", "t0_k": round(k_t0, 3), "t0_cells": cells,
            "t0_n_cells_calibration": int(len(arr)), "t0_n_classes": len(cells)}


def _t0_event_match(s: Dict[str, Any], period: float, t0: Optional[float]) -> bool:
    """T0 in a flagged cell of the candidate's period class, or of the adjacent class when the
    period lies within ``V2_T0EVENT_EDGE_FRAC`` of the shared edge (no holes at class edges)."""
    if t0 is None or period is None or not math.isfinite(t0) or not math.isfinite(period) or period <= 0:
        return False
    x = math.log(period) / V2_T0EVENT_PB
    p = int(math.floor(x))
    classes = [p]
    frac = V2_T0EVENT_EDGE_FRAC / V2_T0EVENT_PB           # edge distance in class units
    if x - p < frac:
        classes.append(p - 1)
    elif (p + 1) - x < frac:
        classes.append(p + 1)
    for k in classes:
        for a, b in s["t0_cells"].get(str(k), ()):
            if a <= t0 <= b:
                return True
    return False


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def family_match_v2(s: Dict[str, Any], period: float, t0: Optional[float],
                    duration: Optional[float] = None,
                    depth: Optional[float] = None,
                    sector: Optional[int] = None) -> bool:
    """True if (period, t0, duration, depth) matches v2 family ``s``.

    Missing duration/depth/t0 skip the corresponding test (conservative: the
    candidate is treated as similar), matching the v1 convention for t0.  A
    family with a ``"sector"`` only matches candidates of that sector; a
    candidate without sector matches families of every sector.
    """
    if sector is not None and s.get("sector") is not None and s["sector"] != sector:
        return False
    if not (s["p_lo"] <= period <= s["p_hi"]):
        return False
    if not s.get("epoch_agnostic"):
        if t0 is not None and not any(a <= t0 <= b for a, b in s.get("t0_modes", ())):
            return False
    # NaN/inf = missing data: skip the test (conservative), like None
    if duration is not None and math.isfinite(duration) and "dur_lo" in s:
        if not (s["dur_lo"] <= duration <= s["dur_hi"]):
            return False
    if depth is not None and math.isfinite(depth) and "depth_lo" in s:
        if not (s["depth_lo"] <= abs(depth) <= s["depth_hi"]):
            return False
    return True


def family_decision_v2(s: Dict[str, Any], period: float, t0: Optional[float],
                       duration: Optional[float] = None,
                       depth: Optional[float] = None,
                       snr: Optional[float] = None,
                       sector: Optional[int] = None) -> Optional[str]:
    """"veto" | "soft" | None for one v2 family."""
    if s.get("kind") == "t0_events":
        return "veto" if _t0_event_match(s, period, t0) else None
    if not family_match_v2(s, period, t0, duration, depth, sector):
        return None
    if (s.get("epoch_agnostic") and snr is not None
            and math.isfinite(snr) and snr >= V2_SNR_SOFT):
        return "soft"
    return "veto"


def soft_family(period: float, t0: Optional[float], systematics: List[Dict[str, Any]],
                duration: Optional[float] = None, depth: Optional[float] = None,
                snr: Optional[float] = None,
                sector: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """First v2 family that SOFT-tags this candidate (no family hard-vetoes it).

    Used for the dossier annotation only; returns None when a family hard-vetoes
    (the candidate is then demoted, not soft-tagged) or none matches.
    """
    soft = None
    for s in systematics:
        if not s.get("v2"):
            continue
        d = family_decision_v2(s, period, t0, duration, depth, snr, sector)
        if d == "veto":
            return None
        if d == "soft" and soft is None:
            soft = s
    return soft
