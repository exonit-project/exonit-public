"""Host-star parameter derivation shared by every path that serves nearby-star
data to the FPP worker (GET /api/nearby cache path AND live-MAST path) and by
the FPP gate (server.target_mass_unavailable_reason).

Why this exists (2026-10-23, 100-star offline FPP run): the live-MAST path of
/api/nearby filled a missing TIC mass from logg + radius, the cache path did
not (the nearby_stars table has no logg column), so every target cached with a
NaN mass reached the worker with M=NaN and got no FPP ("missing stellar
params"). One module now owns the decision; both paths and the gate call it.

Order for a star (nothing is ever invented):
  1. TIC mass present (finite, > 0)                         -> mass_source 'tic'
  2. giant host: R > 2.5 Rsun or (logg finite and < 3.5)    -> NO mass,
     mass_unavailable_reason 'giant_host' ("FPP not applicable: giant host").
     TIC v8 deliberately gives giants no mass; the derivations below are not
     applied to them (a sub-giant like TOI 803.01, R 4.2, is a giant_host).
  3. M = g R^2 / G = R^2 * 10**(logg - 4.438) when logg and R are known and
     the result is in [0.08, 5] Msun with 3.5 <= logg <= 5.5 -> 'derived_logg_radius'.
     The FPP host models cover main-sequence/sub-giant hosts; below 0.08 Msun
     (sub-stellar) and above 5 Msun (no TESS planet-host population) are out.
     These bounds are OUR choice, not a TRICERATOPS constant.
     A logg equal to 4.438 (to 5e-4) is rejected as the TIC/solar placeholder
     (M would equal R^2). A placeholder at another value that also gives an
     in-range mass cannot be detected (the TIC logg error is not stored).
  4. dwarf (R <= 2.5, logg >= 3.5 if known) with Teff/R inside the validity of
     the pinned empirical relation data/catalogs/tic8_teff_mass_relation.json
     (fitted on TIC v8's own dwarf masses, scripts/fpp/fit_tic8_teff_mass.py,
     SHA-256 pinned below; no extrapolation)       -> 'teff_relation_tic8fit'.
  5. Berger+2023 overlay (when the TIC is in the catalogue) overrides
     mass/rad/Teff and wins over everything above  -> 'berger'.
  6. otherwise no mass: reason 'giant_host' or 'stellar_params_missing'.
Giants WITH a mass (TIC or Berger) keep being computed; every star carries ``giant_host`` (bool, evaluated on the
final radius) and the worker adds the flag 'giant_host' to fpp_flags and never validates them.
Neighbours go through the same function: a NaN-mass dwarf neighbour now gets a derived/relation mass (this changes
the NTP/NEB inputs; before, fpp_native substituted 1.0 Msun); a giant neighbour gets no mass and keeps that 1.0 default.
The enrichment runs on READ only; the nearby_stars cache always keeps the original TIC values.

mass_unavailable_reason ends up in candidates.fpp_skip_reason. Legacy rows
keep 'no_stellar_mass' (written by the pre-2026-10-23 gate = giant_host OR
stellar_params_missing, indistinguishable after the fact); history is not renamed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from typing import Dict, Optional

import numpy as np

LOGG_SUN = 4.438
PLACEHOLDER_LOGG = 4.438
PLACEHOLDER_TOL = 5e-4
LOGG_MIN, LOGG_MAX = 3.5, 5.5
MASS_MIN, MASS_MAX = 0.08, 5.0
GIANT_R_MIN = 2.5          # Rsun: R above this -> giant host
GIANT_LOGG_MAX = 3.5       # logg below this -> giant host
RELATION_SHA256 = "0284aa57434d15db14e6dbd970619791982036ca03e865bc11de280071ece140"

_REL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "catalogs", "tic8_teff_mass_relation.json")
_rel_cache: dict = {}


def _finite(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def is_giant_host(rad, logg) -> bool:
    return (_finite(rad) and float(rad) > GIANT_R_MIN) or (_finite(logg) and float(logg) < GIANT_LOGG_MAX)


def classify_no_mass(rad, logg) -> str:
    """Skip reason for a star that ends up without a usable mass."""
    return "giant_host" if is_giant_host(rad, logg) else "stellar_params_missing"


def derive_mass_from_logg_rad(logg, rad) -> Optional[float]:
    """Stellar mass (Msun) from logg and radius (Rsun), or None when the inputs
    are missing, a placeholder, or the result is outside the accepted range."""
    if not (_finite(logg) and _finite(rad)):
        return None
    logg, rad = float(logg), float(rad)
    if rad <= 0 or not (LOGG_MIN <= logg <= LOGG_MAX):
        return None
    if abs(logg - PLACEHOLDER_LOGG) < PLACEHOLDER_TOL:
        return None
    mass = rad ** 2 * 10 ** (logg - LOGG_SUN)
    if not (MASS_MIN <= mass <= MASS_MAX):
        return None
    return mass


_REL_ERR_LOG_EVERY_S = 60.0
_rel_err = {"last_log": 0.0, "error": None}


def _relation():
    """The pinned relation (SHA-256 verified), or None if missing/altered. A success is cached for the process; a
    FAILURE is not (the file may be deployed later) and is logged at ERROR on every use, at most once a minute.
    Those stars then get stellar_params_missing, never a silently different mass."""
    if "rel" in _rel_cache:
        return _rel_cache["rel"]
    try:
        with open(_REL_PATH, "rb") as fh:
            raw = fh.read()
        if hashlib.sha256(raw).hexdigest() != RELATION_SHA256:
            raise ValueError("SHA-256 mismatch")
        rel = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        _rel_err["error"] = f"{type(exc).__name__}: {exc}"
        now = time.monotonic()
        if now - _rel_err["last_log"] >= _REL_ERR_LOG_EVERY_S:
            _rel_err["last_log"] = now
            logging.getLogger("exonit").error(
                "Teff-mass relation unavailable (%s: %s): no relation masses are produced", _REL_PATH, exc)
        return None
    _rel_cache["rel"] = rel
    _rel_err["error"] = None
    return rel


def relation_status() -> dict:
    """For the health endpoint: whether the pinned relation loads (re-checks, no caching of failures)."""
    ok = _relation() is not None
    return {"ok": ok, "sha256": RELATION_SHA256, "error": None if ok else _rel_err["error"]}


def mass_from_teff_relation(teff, rad, logg=None) -> Optional[float]:
    """TIC-v8-like dwarf mass from the pinned empirical relation; None outside its validity (dwarf cut, Teff in
    [teff_min_K, teff_max_K], R inside the cool branch's knot range)."""
    rel = _relation()
    if rel is None or not (_finite(teff) and _finite(rad)) or float(rad) <= 0 or is_giant_host(rad, logg):
        return None
    teff, rad = float(teff), float(rad)
    if not (rel["teff_min_K"] <= teff <= rel["teff_max_K"]):
        return None
    if teff >= rel["t_switch_K"]:
        b, x = rel["teff_branch"], math.log10(teff)
    else:
        b, x = rel["radius_branch"], math.log10(rad)
    if not (b["x"][0] <= x <= b["x"][-1]):
        return None
    return float(np.interp(x, b["x"], b["m"]))


def enrich_nearby_stars(stars, berger=None,
                        logg_by_id: Optional[Dict[str, float]] = None) -> Dict[str, int]:
    """In-place, identical for the cache and the live path (see module docstring for the order).
    ``logg_by_id`` maps star ID (str) -> TIC logg; stars without an entry have logg unknown.
    Sets ``mass_source`` (and ``mass_unavailable_reason`` when there is no mass). Returns counts per source."""
    logg_by_id = logg_by_id or {}
    counts = {"tic": 0, "derived_logg_radius": 0, "teff_relation_tic8fit": 0, "berger": 0, "none": 0}
    for s in stars:
        m = s.get("mass")
        logg = logg_by_id.get(str(s.get("ID")))
        rad = s.get("rad")
        s.pop("mass_unavailable_reason", None)
        if _finite(m) and float(m) > 0:
            src = "tic"
        elif is_giant_host(rad, logg):
            src = "none"
        else:
            d = derive_mass_from_logg_rad(logg, rad)
            src = "derived_logg_radius"
            if d is None:
                d = mass_from_teff_relation(s.get("Teff"), rad, logg)
                src = "teff_relation_tic8fit"
            if d is not None:
                s["mass"] = d
            else:
                src = "none"
        if berger is not None:
            try:
                bg = berger.lookup_by_tic(int(s["ID"]))
            except (ValueError, TypeError, KeyError):
                bg = None
            if bg is not None:
                s["mass"], s["rad"], s["Teff"] = bg["mass"], bg["rad"], bg["teff"]
                s["stellar_source"] = "Berger+2023"
                src = "berger"
        if src == "none":
            s["mass_unavailable_reason"] = classify_no_mass(rad, logg)
        # a giant that DOES have a mass (TIC or Berger) is still computed, but flagged: the worker caps validated
        s["giant_host"] = bool(is_giant_host(s.get("rad"), logg))
        s["mass_source"] = src
        counts[src] += 1
    return counts
