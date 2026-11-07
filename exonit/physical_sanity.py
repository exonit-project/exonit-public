"""Physical-sanity annotations for strong / finalist transit candidates.

Motivation (Planet Hunters TESS forum review): a candidate whose transit depth
and host radius imply a >20 R_earth "planet" (e.g. TIC 63280211: R*=8.4 R_sun,
depth 6002 ppm -> Rp ~ 71 R_earth) is a blended / evolved-star / eclipsing-binary
signal, not a planet. This module derives, per candidate:

  * Rp [R_earth, R_jup] = sqrt(depth) * R* * 109.08 (R_sun -> R_earth)
  * the mid-transit epoch T0 in BTJD and BJD_TDB (= BTJD + 2457000)
  * the duration in hours
  * flags (named constants below)

Rp bias depends on the light-curve product (``lc_source``, from the metadata
``fits_files`` the worker actually read):

  * SPOC PDCSAP / QLP: Rp is (approximately) a LOWER BOUND -- the observed depth
    is diluted by any unresolved flux (QLP is only partially decontaminated) and
    BLS box depths are biased low for ingress/egress-dominated transits.
  * TGLC (99.99 % of the discovery campaign): NOT a lower bound. TGLC is
    decontaminated but normalises ``cal_aper_flux`` by a Tmag-PREDICTED flux, so
    depths are too deep by c = 1.09 (Rp by sqrt(c), +4.4 %);
    docs/reviews/2026-10-15-tglc-depth-excess.md. With ``EXONIT_TGLC_DEPTH_CORR``
    (default 1) TGLC rows get ``depth_ppm_corr = depth/c`` and ``rp_rearth_corr =
    Rp/sqrt(c)`` with the correction uncertainty, flag ``tglc_depth_corrected``
    (``tglc_near_edge`` for NEAREDGE stars, provisional c = 1.27), and the
    ``rp_implausible`` cut uses the corrected Rp. Rows whose product is unknown
    (or mixed), and rows of FFI-fallback tasks (``tasks.lc_load_recovered``: the
    worker's ``_try_ffi_fallback``, ON fleet-wide via worker/config.json, read the
    server's FFI cache instead of ``fits_files``), stay uncorrected with
    ``lc_source_unknown``. The correction is
    applied at CONSUMPTION only: the stored depths and every model input stay
    uncorrected (the ML models were trained on uncorrected TGLC amplitudes).

A flagged ``rp_implausible`` row is robustly implausible only up to the per-star
scatter of the correction (~6 % in depth); an unflagged one is not thereby
validated.

T0 conventions (the epoch stored in ``candidates.t0``):
  * raw BLS (worker bls_fast/bls_cuda/bls_metal*): ``t0 = phase_bin_start * P +
    t0_ref`` -- the START of the best box, so mid = t0 + duration/2. This is what
    every committed discovery run used (``duration_refine`` is OFF fleet-wide,
    see project status, open items: activation waits for Swift/Rust parity + scorecard A/B).
  * ``worker/duration_refine.refine_duration``: returns the box CENTRE
    (``t0 + best_offset`` where the offset is the centre of the best box), so
    mid = t0. Select with ``t0_is_centre=True`` / ``EXONIT_PHYS_T0_IS_CENTRE=1``
    once duration_refine is enabled on the fleet.

The T0 convention is a GLOBAL switch, not per candidate (the DB stores no marker of
whether duration_refine was applied). ``EXONIT_PHYS_T0_IS_CENTRE`` must therefore be
flipped to 1 ONLY together with enabling ``duration_refine`` fleet-wide (not enabled
as of 2026-10-15; project status, open items). Dossiers regenerated/produced before the flip
keep the box-start convention (the section text names the convention used); any
candidate produced by a host with duration_refine enabled while the switch is still 0
would have its T0 mid off by duration/2.

``EXONIT_PHYS_SANITY`` (default 1): 0 disables the whole feature (callers keep
the exact old behaviour). ``EXONIT_TGLC_DEPTH_CORR`` (default 1): 0 restores the
uncorrected Rp, the old table layout and the old flags.

Pure functions + small READ-ONLY cache loaders; no online calls in here.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
from typing import Iterable, Optional

R_SUN_TO_R_EARTH = 109.08          # 1 R_sun in R_earth
R_JUP_TO_R_EARTH = 11.209          # 1 R_jup (equatorial) in R_earth
BTJD_TO_BJD = 2457000.0            # BJD_TDB = BTJD + 2457000

# --- flag thresholds (named constants; change here only) --------------------
RP_IMPLAUSIBLE_REARTH = 22.0       # planets are rarely >20 R_earth, inflated HJs <~18
GIANT_HOST_RSTAR_RSUN = 3.0        # R* above this: evolved host
GIANT_HOST_LOGG = 3.5              # logg below this: evolved host (when known)
# Hard cut: TIC R* of subgiants (R* 3-4 Rsun, which do host real planets) carries
# ~10-20 % error, so R*>3 only ANNOTATES; the finalist exclusion needs a clearly
# evolved/giant host.
GIANT_HOST_HARD_RSTAR_RSUN = 5.0
RUWE_HIGH = 1.4                    # Gaia RUWE above this: likely non-single star

FLAG_RP_IMPLAUSIBLE = "rp_implausible"
FLAG_GIANT_HOST = "giant_host"              # annotation only
FLAG_GIANT_HOST_HARD = "giant_host_hard"    # R* > 5 Rsun, excludes finalists
FLAG_RUWE_HIGH = "ruwe_high"
FLAG_MISSING_RSTAR = "missing_rstar"

# --- TGLC depth correction (the ONLY place these numbers live) ---------------
# Provenance: docs/reviews/2026-10-15-tglc-depth-excess.md ("Final result", sections
# 6-8), commit 738dedc. c = TGLC depth / true depth; constant across sector family
# (primary 1.090, extended 1.091), Tmag, camera and crowding.
TGLC_DEPTH_C = 1.09
TGLC_DEPTH_C_STAT = 0.02           # 1 sigma, absolute on c (sem of the published check)
TGLC_DEPTH_C_SYST = 0.03           # 1 sigma, absolute on c (zero point vs published)
TGLC_DEPTH_SCATTER_FRAC = 0.06     # intrinsic star-to-star scatter, fractional on depth
# NEAREDGE = True (LOC_BG == 0, no PSF branch): provisional, n = 6 of 80 files.
# Interpretation of +-0.12: uncertainty of the near-edge CONSTANT (the report does
# not say whether it is a sem or a spread); the 6 % per-star scatter is added on top
# in quadrature. If it is a spread this double counts (11.2 % instead of ~9.4 % on
# depth): conservative, acceptable for a provisional, flagged constant.
TGLC_NEAR_EDGE_C = 1.27
TGLC_NEAR_EDGE_C_ERR = 0.12
# Unchecked NEAREDGE (no header read): c = 1.09 is assumed, and the mixture spread
# of the edge prior is added to sigma_c in quadrature: p = 33/811 near-edge files
# (header snapshot 2026-10-15), sqrt(p (1-p)) x (1.27 - 1.09) = 0.036. The mean
# bias p x 0.18 = +0.007 on c is NOT applied (disclosed in the dossier).
TGLC_UNCHECKED_EDGE_SIGMA = 0.036
TGLC_DEPTH_CORR_REF = "docs/reviews/2026-10-15-tglc-depth-excess.md @ 738dedc"

LC_TGLC, LC_SPOC, LC_QLP, LC_MIXED = "tglc", "spoc", "qlp", "mixed"
# The worker's FFI fallback (worker._try_ffi_fallback; ON in the committed
# worker/config.json, ffi_fallback_on_load_fail = true) replaces the metadata
# fits_files by the server's Phase-2.5 FFI cache when every download fails or the
# load raises; recorded as tasks.lc_load_recovered. Such rows are labelled this
# (an unknown product: left uncorrected, flag lc_source_unknown).
LC_FFI_FALLBACK = "ffi_fallback"
# ``recovered`` sentinel: the tasks lookup failed -> every row is treated as recovered.
RECOVERED_ALL = "all"
FLAG_TGLC_DEPTH_CORRECTED = "tglc_depth_corrected"   # annotation
FLAG_TGLC_NEAR_EDGE = "tglc_near_edge"               # annotation (provisional c used)
FLAG_LC_SOURCE_UNKNOWN = "lc_source_unknown"         # annotation (left uncorrected)

# Finalists additionally EXCLUDE these (Rp>22 R_earth / a R*>5 Rsun host cannot be
# confirmed as a planet with TESS photometry alone: the depth-radius budget points
# at a stellar companion or a blend, and giant hosts are dominated by
# oscillation / granulation noise). giant_host (R*>3 / logg<3.5), ruwe_high and
# missing_rstar only annotate.
HARD_FLAGS = (FLAG_RP_IMPLAUSIBLE, FLAG_GIANT_HOST_HARD)

SECTION_HEADING = "## Physical sanity"
TABLE_HEADER = ("| TIC | P(d) | T0 mid (BTJD) | T0 mid (BJD) | dur(h) | R*(R☉) | "
                "Rp(R⊕) | RUWE | logg | flags |")
# EXONIT_TGLC_DEPTH_CORR=1 layout: raw depth/Rp next to the corrected values.
TABLE_HEADER_CORR = ("| TIC | P(d) | T0 mid (BTJD) | T0 mid (BJD) | dur(h) | R*(R☉) | "
                     "LC | depth(ppm) | depth corr(ppm) | Rp(R⊕) | Rp corr(R⊕) | "
                     "RUWE | logg | flags |")


def enabled() -> bool:
    """EXONIT_PHYS_SANITY (default ON); 0/false/off/no disables."""
    return os.environ.get("EXONIT_PHYS_SANITY", "1").strip().lower() not in (
        "0", "false", "off", "no", "")


def tglc_corr_enabled() -> bool:
    """EXONIT_TGLC_DEPTH_CORR (default ON); 0/false/off/no disables."""
    return os.environ.get("EXONIT_TGLC_DEPTH_CORR", "1").strip().lower() not in (
        "0", "false", "off", "no", "")


def tglc_header_cache_path() -> str:
    """Server-side TGLC header (NEAREDGE) cache: EXONIT_TGLC_HEADER_CACHE, default
    ~/.cache/exonit-tglc/tglc_headers.db (outside the repo and the prod data dir)."""
    return os.environ.get("EXONIT_TGLC_HEADER_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "exonit-tglc", "tglc_headers.db")


def lc_source_from_fits_files(fits_files) -> Optional[str]:
    """Light-curve product of a ``metadata.fits_files`` list (the files the worker
    downloads and stitches for the TIC; same name patterns as the worker's
    ``_detect_data_source``): 'tglc' / 'spoc' / 'qlp' when every file is that
    product, 'mixed' when they differ, None when empty/unrecognised.

    Exception: when the worker's FFI fallback fired (ON fleet-wide via
    worker/config.json; ``tasks.lc_load_recovered``) the light curve did NOT come
    from these files. Callers must override such rows with
    :data:`LC_FFI_FALLBACK` (see :func:`is_recovered`)."""
    if isinstance(fits_files, str):
        try:
            fits_files = json.loads(fits_files)
        except ValueError:
            return None
    kinds = set()
    for f in fits_files or ():
        b = os.path.basename(str(f)).lower()
        if b.startswith("hlsp_tglc"):
            kinds.add(LC_TGLC)
        elif b.startswith("hlsp_qlp"):
            kinds.add(LC_QLP)
        elif b.endswith("_lc.fits") and "hlsp" not in b:
            kinds.add(LC_SPOC)
        else:
            return None
    if not kinds:
        return None
    return kinds.pop() if len(kinds) == 1 else LC_MIXED


def tglc_depth_scale(near_edge=None) -> tuple:
    """(c, rel_err_depth, edge_state) for a TGLC light curve.

    ``near_edge``: NEAREDGE of the TGLC file(s) the worker stitched -- one
    bool/None or a list (one per file); None = not checked. The stitched depth
    is an equal-weight mix of the files, so c is the mean of the per-file c
    (1.09, or the provisional 1.27 for a NEAREDGE file); an unchecked file
    counts as not near-edge. ``rel_err_depth`` = 1 sigma fractional
    uncertainty of the corrected depth: sigma_c/c (stat and syst in quadrature;
    1.27 +- 0.12 for near-edge files, averaged linearly because the files share
    the constant) in quadrature with the 6 % per-star scatter. The Rp
    uncertainty is half of it. ``edge_state``: 'no' / 'yes' (every file) /
    'partial' / 'unchecked' (at least one file unknown, none near-edge).
    'partial': the true weights of the files in the stitched BLS depth are
    in-transit point counts (cadence differs up to 9x between eras), so the
    equal-weight mean is only a centre: half the [1.09, 1.27] range is added to
    sigma_c in quadrature (the 1 sigma band spans both constants).
    'unchecked': :data:`TGLC_UNCHECKED_EDGE_SIGMA` is added in quadrature.
    """
    vals = list(near_edge) if isinstance(near_edge, (list, tuple)) else [near_edge]
    if not vals:
        vals = [None]
    sig_ok = math.hypot(TGLC_DEPTH_C_STAT, TGLC_DEPTH_C_SYST)
    cs = [TGLC_NEAR_EDGE_C if v is True else TGLC_DEPTH_C for v in vals]
    ss = [TGLC_NEAR_EDGE_C_ERR if v is True else sig_ok for v in vals]
    c = sum(cs) / len(cs)
    sig_c = sum(ss) / len(ss)
    n_edge = sum(1 for v in vals if v is True)
    if n_edge == len(vals):
        state = "yes"
    elif n_edge:
        state = "partial"
        sig_c = math.hypot(sig_c, (TGLC_NEAR_EDGE_C - TGLC_DEPTH_C) / 2)
    elif any(v is None for v in vals):
        state = "unchecked"
        sig_c = math.hypot(sig_c, TGLC_UNCHECKED_EDGE_SIGMA)
    else:
        state = "no"
    rel = math.hypot(sig_c / c, TGLC_DEPTH_SCATTER_FRAC)
    return c, rel, state


def is_recovered(recovered, tic, sector=None) -> bool:
    """True when the candidate (TIC, sector) came from an FFI-fallback task.
    ``recovered``: set of (tic str, sector int|None) -- a None sector matches every
    sector of the TIC -- or :data:`RECOVERED_ALL` (lookup failed), or None/empty."""
    if not recovered:
        return False
    if recovered == RECOVERED_ALL:
        return True
    t = str(tic)
    if (t, None) in recovered:
        return True
    try:
        return sector is not None and (t, int(sector)) in recovered
    except (TypeError, ValueError):
        return False


def t0_is_centre_env() -> bool:
    return os.environ.get("EXONIT_PHYS_T0_IS_CENTRE", "0").strip() == "1"


def _num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def rp_rearth(depth_ppm, rstar_rsun) -> Optional[float]:
    d, r = _num(depth_ppm), _num(rstar_rsun)
    if d is None or r is None or d <= 0 or r <= 0:
        return None
    return math.sqrt(d * 1e-6) * r * R_SUN_TO_R_EARTH


def t0_mid_btjd(t0, duration_days, t0_is_centre: bool = False) -> Optional[float]:
    t, d = _num(t0), _num(duration_days)
    if t is None:
        return None
    if t0_is_centre:
        return t
    if d is None:
        return None
    return t + 0.5 * d


def assess(period_days, t0, duration_days, depth_ppm, rstar=None, teff=None,
           logg=None, ruwe=None, t0_is_centre: bool = False, *,
           lc_source: Optional[str] = None, near_edge=None,
           tglc_corr: Optional[bool] = None) -> dict:
    """Physical-sanity record for one candidate (R* in R_sun, RUWE unitless).

    ``lc_source``: 'tglc'/'spoc'/'qlp'/'mixed'/None (see
    :func:`lc_source_from_fits_files`); ``near_edge``: NEAREDGE of the TGLC
    file(s), see :func:`tglc_depth_scale`; ``tglc_corr``: None = read
    ``EXONIT_TGLC_DEPTH_CORR``. ``rp_rearth`` is always the RAW value;
    ``rp_rearth_used`` is the one the ``rp_implausible`` cut used (corrected
    for TGLC rows when the correction is on). ``*_corr_err`` = 1 sigma from the
    correction only (the R* error is not included)."""
    if tglc_corr is None:
        tglc_corr = tglc_corr_enabled()
    rs, lg, ru = _num(rstar), _num(logg), _num(ruwe)
    if rs is not None and rs <= 0:
        rs = None
    rp = rp_rearth(depth_ppm, rs)
    mid = t0_mid_btjd(t0, duration_days, t0_is_centre)
    dur = _num(duration_days)
    dep = _num(depth_ppm)
    corr = {"lc_source": lc_source, "depth_ppm": dep, "depth_ppm_corr": None,
            "depth_ppm_corr_err": None, "rp_rearth_corr": None,
            "rp_rearth_corr_err": None, "tglc_depth_c": None, "tglc_near_edge": None}
    rp_used = rp
    flags, src_flags = [], []
    if rs is None:
        flags.append(FLAG_MISSING_RSTAR)
    if tglc_corr and lc_source == LC_TGLC:
        c, rel, state = tglc_depth_scale(near_edge)
        corr.update(tglc_depth_c=c, tglc_near_edge=state)
        if dep is not None and dep > 0:
            corr["depth_ppm_corr"] = dep / c
            corr["depth_ppm_corr_err"] = dep / c * rel
        if rp is not None:
            rp_used = rp / math.sqrt(c)
            corr["rp_rearth_corr"] = rp_used
            corr["rp_rearth_corr_err"] = rp_used * 0.5 * rel
        src_flags.append(FLAG_TGLC_DEPTH_CORRECTED)
        if state in ("yes", "partial"):
            src_flags.append(FLAG_TGLC_NEAR_EDGE)
    elif tglc_corr and lc_source not in (LC_SPOC, LC_QLP):
        src_flags.append(FLAG_LC_SOURCE_UNKNOWN)      # None or mixed: left uncorrected
    if rp_used is not None and rp_used > RP_IMPLAUSIBLE_REARTH:
        flags.append(FLAG_RP_IMPLAUSIBLE)
    if (rs is not None and rs > GIANT_HOST_RSTAR_RSUN) or (
            lg is not None and lg < GIANT_HOST_LOGG):
        flags.append(FLAG_GIANT_HOST)
    if rs is not None and rs > GIANT_HOST_HARD_RSTAR_RSUN:
        flags.append(FLAG_GIANT_HOST_HARD)
    if ru is not None and ru > RUWE_HIGH:
        flags.append(FLAG_RUWE_HIGH)
    flags += src_flags                                # provenance annotations last
    rec = {
        "period_days": _num(period_days),
        "t0_mid_btjd": mid,
        "t0_mid_bjd": None if mid is None else mid + BTJD_TO_BJD,
        "duration_h": None if dur is None else dur * 24.0,
        "rstar": rs, "teff": _num(teff), "logg": lg, "ruwe": ru,
        "rp_rearth": rp,
        "rp_rjup": None if rp is None else rp / R_JUP_TO_R_EARTH,
        "flags": flags,
    }
    if tglc_corr:
        rec.update(corr, rp_rearth_used=rp_used)
    return rec


def hard_excluded(rec: Optional[dict]) -> bool:
    return bool(rec) and any(f in HARD_FLAGS for f in rec["flags"])


# --------------------------------------------------------------------------- #
# dossier formatting
# --------------------------------------------------------------------------- #
def _f(v, nd) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def _pm(v, e, nd) -> str:
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if e is None else f"{v:.{nd}f} ± {e:.{nd}f}"


def _lc_cell(rec: dict) -> str:
    src = rec.get("lc_source")
    if src is None:
        return "?"
    if src == LC_TGLC:
        return f"tglc (edge: {rec.get('tglc_near_edge') or 'unchecked'})"
    return src


def format_row(tic, rec: Optional[dict], printed_period: Optional[float] = None,
               tglc_corr: bool = False) -> str:
    if rec is None:
        p = "-" if printed_period is None else f"{printed_period:.2f}"
        n_dash = 11 if tglc_corr else 7
        return f"| {tic} | {p} | " + "- | " * n_dash + "unmatched |"
    head = (f"| {tic} | {_f(rec['period_days'], 2)} | {_f(rec['t0_mid_btjd'], 4)} | "
            f"{_f(rec['t0_mid_bjd'], 4)} | {_f(rec['duration_h'], 2)} | "
            f"{_f(rec['rstar'], 2)} | ")
    tail = (f"{_f(rec['ruwe'], 2)} | {_f(rec['logg'], 2)} | "
            f"{', '.join(rec['flags']) or 'ok'} |")
    if not tglc_corr:
        return head + f"{_f(rec['rp_rearth'], 1)} | " + tail
    return (head + f"{_lc_cell(rec)} | {_f(rec.get('depth_ppm'), 0)} | "
            f"{_pm(rec.get('depth_ppm_corr'), rec.get('depth_ppm_corr_err'), 0)} | "
            f"{_f(rec['rp_rearth'], 1)} | "
            f"{_pm(rec.get('rp_rearth_corr'), rec.get('rp_rearth_corr_err'), 1)} | " + tail)


def _rp_sentence(tglc_corr: bool) -> str:
    if not tglc_corr:
        return ("Rp = sqrt(depth) x R* x 109.08 is a "
                "LOWER BOUND (blend dilution only deepens the true transit). ")
    return ("Rp = sqrt(depth) x R* x 109.08 (raw). For SPOC/QLP light curves it is "
            "approximately a lower bound (blend dilution only deepens the true transit). "
            "TGLC light curves (LC = tglc) are decontaminated but their depths are too "
            f"deep by c = {TGLC_DEPTH_C:g} (flux normalised by a Tmag-predicted flux; "
            f"{TGLC_DEPTH_CORR_REF}): depth corr = depth/c, Rp corr = Rp/sqrt(c), "
            f"± 1 sigma of the correction only (c ± {TGLC_DEPTH_C_STAT:g} stat ± "
            f"{TGLC_DEPTH_C_SYST:g} syst, {TGLC_DEPTH_SCATTER_FRAC:.0%} per-star scatter "
            f"in depth; the R* error is not included); NEAREDGE stars use the "
            f"provisional c = {TGLC_NEAR_EDGE_C:g} ± {TGLC_NEAR_EDGE_C_ERR:g} "
            f"(edge: unchecked = header not read, c = {TGLC_DEPTH_C:g} assumed, error "
            f"widened by ± {TGLC_UNCHECKED_EDGE_SIGMA:g} on c; partial = some stitched "
            f"files near-edge, c = mean, error spans both constants). "
            f"`{FLAG_RP_IMPLAUSIBLE}` uses Rp corr for TGLC rows. ")


def dossier_lines(entries: list, *, t0_is_centre: bool = False,
                  tglc_corr: Optional[bool] = None) -> list:
    """Section lines (no leading/trailing blank). ``entries`` = [(tic, rec|None,
    printed_period|None)] in strong-table order. ``tglc_corr`` None = read
    ``EXONIT_TGLC_DEPTH_CORR``; off = the pre-correction layout byte for byte."""
    if tglc_corr is None:
        tglc_corr = tglc_corr_enabled()
    conv = ("centre of the fitted box" if t0_is_centre
            else "start of the BLS box + duration/2")
    src_flags = (f" `{FLAG_TGLC_DEPTH_CORRECTED}` TGLC depth/Rp corrected, "
                 f"`{FLAG_TGLC_NEAR_EDGE}` TGLC NEAREDGE star (provisional c), "
                 f"`{FLAG_LC_SOURCE_UNKNOWN}` light-curve product unknown, mixed or "
                 f"worker FFI fallback (LC = {LC_FFI_FALLBACK}) "
                 f"(left uncorrected).") if tglc_corr else ""
    out = [SECTION_HEADING, "",
           f"Per-candidate physical plausibility of the strong rows above. T0 mid = "
           f"{conv} (BJD_TDB = BTJD + 2457000). " + _rp_sentence(tglc_corr) + "Flags: "
           f"`{FLAG_RP_IMPLAUSIBLE}` Rp > {RP_IMPLAUSIBLE_REARTH:g} R⊕, "
           f"`{FLAG_GIANT_HOST}` R* > {GIANT_HOST_RSTAR_RSUN:g} R☉ or logg < "
           f"{GIANT_HOST_LOGG:g} (annotation only), `{FLAG_GIANT_HOST_HARD}` R* > "
           f"{GIANT_HOST_HARD_RSTAR_RSUN:g} R☉, `{FLAG_RUWE_HIGH}` Gaia RUWE > {RUWE_HIGH:g}, "
           f"`{FLAG_MISSING_RSTAR}` no stellar radius available. Finalists exclude "
           f"`{FLAG_RP_IMPLAUSIBLE}` and `{FLAG_GIANT_HOST_HARD}`; the others only annotate."
           + src_flags, ""]
    if tglc_corr:
        out += [TABLE_HEADER_CORR, "|" + "---|" * 14]
    else:
        out += [TABLE_HEADER, "|---|---|---|---|---|---|---|---|---|---|"]
    out += [format_row(t, r, p, tglc_corr) for t, r, p in entries]
    return out


# --------------------------------------------------------------------------- #
# READ-ONLY cache loaders (no online calls)
# --------------------------------------------------------------------------- #
def _ro_sqlite(path: str):
    """Open a local SQLite cache read-only without creating -wal/-shm files."""
    if not path or not os.path.exists(path):
        return None
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def _json_cache(path: str, table: str, tics: Iterable) -> dict:
    con = _ro_sqlite(path)
    out: dict = {}
    if con is None:
        return out
    try:
        for t in set(str(x) for x in tics):
            row = con.execute(f"SELECT data_json FROM {table} WHERE tic_id=?",
                              (t,)).fetchone()
            if row:
                try:
                    out[t] = json.loads(row[0])
                except ValueError:
                    pass
    except sqlite3.Error:
        pass
    finally:
        con.close()
    return out


def load_ruwe(tics: Iterable, cache_dir: str) -> dict:
    """{tic: RUWE} from the server's Gaia DR3 cache (data/gaia_cache.db);
    TICs not in the cache (or without RUWE) are absent."""
    raw = _json_cache(os.path.join(cache_dir, "gaia_cache.db"), "gaia_cache", tics)
    return {t: v for t, d in raw.items() if (v := _num(d.get("ruwe"))) is not None}


def load_ticv8_stellar(tics: Iterable, cache_dir: str) -> dict:
    """{tic: {rad, teff, logg}} from the server's TICv8 cache (data/ticv8_cache.db)."""
    raw = _json_cache(os.path.join(cache_dir, "ticv8_cache.db"), "ticv8_cache", tics)
    return {t: {"rad": _num(d.get("rad")), "teff": _num(d.get("Teff", d.get("teff"))),
                "logg": _num(d.get("logg"))} for t, d in raw.items()}


def merge_stellar(*sources: dict) -> dict:
    """First source with a non-null field wins, field by field.
    Each source: {tic: {rad, teff, logg}}."""
    out: dict = {}
    for src in sources:
        for t, d in (src or {}).items():
            cur = out.setdefault(t, {"rad": None, "teff": None, "logg": None})
            for k in cur:
                if cur[k] is None and _num(d.get(k)) is not None:
                    cur[k] = _num(d.get(k))
    return out


TGLC_HEADER_TABLE = "tglc_headers"


def load_tglc_near_edge(filenames: Iterable, cache_path: Optional[str]) -> dict:
    """{fits basename: NEAREDGE (bool)} from the TGLC header cache written by
    ``scripts/tglc_header_cache.py``: the SQLite cache (opened read-only, NOT
    immutable: the tool may be writing it) or its committed CSV snapshot
    (``*.csv``, columns filename,near_edge,...). Files absent from the cache (or a
    missing cache) are absent -> 'unchecked'; a read error is logged to stderr."""
    out: dict = {}
    if not cache_path or not os.path.exists(cache_path):
        return out
    names = {os.path.basename(str(f)) for f in filenames}
    if str(cache_path).endswith(".csv"):
        import csv
        try:
            with open(cache_path, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r["filename"] in names and r.get("near_edge") not in (None, ""):
                        out[r["filename"]] = r["near_edge"].strip() in ("1", "True", "true")
        except (OSError, KeyError, csv.Error) as exc:
            print(f"WARNING phys-sanity: TGLC header snapshot {cache_path} unreadable "
                  f"({exc!r}); NEAREDGE unchecked", file=sys.stderr)
            return {}
        return out
    try:
        con = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        print(f"WARNING phys-sanity: TGLC header cache {cache_path} unreadable "
              f"({exc!r}); NEAREDGE unchecked", file=sys.stderr)
        return out
    try:
        names = sorted({os.path.basename(str(f)) for f in filenames})
        names = sorted(names)
        for i in range(0, len(names), 500):
            part = names[i:i + 500]
            for fn, ne in con.execute(
                    f"SELECT filename, near_edge FROM {TGLC_HEADER_TABLE} "
                    f"WHERE filename IN ({','.join('?' * len(part))})", part):
                if ne is not None:
                    out[fn] = bool(ne)
    except sqlite3.Error as exc:
        print(f"WARNING phys-sanity: TGLC header cache {cache_path} read failed "
              f"({exc!r}); NEAREDGE unchecked", file=sys.stderr)
        out = {}
    finally:
        con.close()
    return out


def tic_source_info(fits_files_by_tic: dict, header_cache: Optional[str]) -> tuple:
    """({tic: lc_source}, {tic: [NEAREDGE per TGLC file, None = unchecked]}) from
    {tic: metadata.fits_files}. The near-edge list is only built for TGLC TICs."""
    src, ff = {}, {}
    for t, files in fits_files_by_tic.items():
        if isinstance(files, str):
            try:
                files = json.loads(files)
            except ValueError:
                files = []
        files = [os.path.basename(str(f)) for f in (files or [])]
        src[str(t)] = lc_source_from_fits_files(files)
        if src[str(t)] == LC_TGLC:
            ff[str(t)] = files
    ne = load_tglc_near_edge([f for fl in ff.values() for f in fl], header_cache)
    return src, {t: [ne.get(f) for f in fl] for t, fl in ff.items()}


def assess_many(cands: list, stellar: dict, ruwe: dict,
                t0_is_centre: bool = False, lc_source: Optional[dict] = None,
                near_edge: Optional[dict] = None, recovered=None) -> dict:
    """{candidate id: record} for candidate dicts (tic_id,id,period_days,t0,
    duration_days,depth_ppm[,sector]); ``lc_source`` / ``near_edge`` keyed by TIC;
    ``recovered`` (see :func:`is_recovered`) overrides the source of FFI-fallback
    candidates with :data:`LC_FFI_FALLBACK`."""
    out = {}
    lc_source, near_edge = lc_source or {}, near_edge or {}
    for c in cands:
        t = str(c["tic_id"])
        st = stellar.get(t, {})
        src = lc_source.get(t)
        if is_recovered(recovered, t, c.get("sector")):
            src = LC_FFI_FALLBACK
        out[c["id"]] = assess(c.get("period_days"), c.get("t0"), c.get("duration_days"),
                              c.get("depth_ppm"), st.get("rad"), st.get("teff"),
                              st.get("logg"), ruwe.get(t), t0_is_centre,
                              lc_source=src, near_edge=near_edge.get(t))
    return out
