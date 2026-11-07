"""Server-side contract of the fixed FPP model (FPP fix plan step 6).

Everything the server needs to know about a "new model" FPP result, kept out of
``db.py`` / the routes so that both ingest paths (``/api/worker/fpp_result`` and the
primary result post that carries an in-task FPP) and the verdict gate share one
definition:

* ``MODEL_COLUMNS``       the additive nullable ``candidates`` columns (DDL in
                          ``scripts/migration/2026-10-17_fpp_model_columns.sql``);
* ``extract_model_fields`` validate + normalise the new fields of a result body;
* ``gate_candidate_for_verdict`` verdict gating (plan step 6 (v), until step 12 passes);
* ``task_extras``         the new fields of the FPP task payload (``lc_source``,
                          ``cadence_s``, the population, the geometry stub).

Nothing here touches the worker (``worker/fpp_native.py`` is owned by another change):
the worker-side field names are defined HERE and documented in
``progress/impl-fpp-step6.md``.
"""

from __future__ import annotations

import collections
import concurrent.futures
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: ``fpp_model_version`` written by the fixed model.  Legacy rows are NULL (== model 1).
FPP_MODEL_VERSION_NEW = 2

#: First model version whose results carry the geometry-v3 provenance (``GEOMETRY_COLUMNS``) and
#: the population provenance: such a result is refused (503) unless every one of those columns exists.
FPP_MODEL_VERSION_GEOMETRY = 3

#: column -> (PostgreSQL type, SQLite type).  All nullable, no default: ``ADD COLUMN`` is
#: metadata-only on the 50M-row partitioned ``candidates`` table.
MODEL_COLUMNS = {
    "fpp_model_version": ("SMALLINT", "INTEGER"),
    "triceratops_fpp_v1": ("DOUBLE PRECISION", "REAL"),
    "triceratops_nfpp_v1": ("DOUBLE PRECISION", "REAL"),
    "fpp_scen": ("JSONB", "TEXT"),
    "fpp_backend": ("TEXT", "TEXT"),
    "fpp_backend_device": ("TEXT", "TEXT"),
    "fpp_mc_se": ("DOUBLE PRECISION", "REAL"),
    "fpp_n_draws": ("BIGINT", "INTEGER"),
    "population_version": ("TEXT", "TEXT"),
    "population_source": ("TEXT", "TEXT"),
    "population_sha256": ("TEXT", "TEXT"),
    "fpp_flags": ("TEXT", "TEXT"),
    # model >= 3 (FFI geometry v3 + per-product dilution convention), DDL
    # scripts/migration/2026-10-18_fpp_geometry_columns.sql
    "geometry_version": ("SMALLINT", "INTEGER"),
    "dilution_convention": ("TEXT", "TEXT"),
    "aperture_source": ("TEXT", "TEXT"),
    "depth_scale": ("DOUBLE PRECISION", "REAL"),
    # step-7 Monte-Carlo result fields
    "nfpp_mc_se": ("DOUBLE PRECISION", "REAL"),
    "fpp_n_draws_total": ("BIGINT", "INTEGER"),
    "mc_min_ess": ("DOUBLE PRECISION", "REAL"),
    "mc_borderline": ("BOOLEAN", "INTEGER"),
    "mc_detail": ("JSONB", "TEXT"),
    # ephemeris refinement provenance (FPP_EPHEM_REFINE), DDL scripts/migration/2026-10-30_fpp_ephem_columns.sql
    # (NOT APPLIED; the server stores them only where the columns exist)
    "ephem_status": ("TEXT", "TEXT"),
    "ephem_reason": ("TEXT", "TEXT"),
    "ephem_period_days": ("DOUBLE PRECISION", "REAL"),
    "ephem_t0": ("DOUBLE PRECISION", "REAL"),
    "ephem_shift_h": ("DOUBLE PRECISION", "REAL"),
    "ephem_delta_chi2": ("DOUBLE PRECISION", "REAL"),
    # D17 + rehearsal P2, DDL scripts/migration/2026-10-31_fpp_switches_verdict_v1.sql (NOT APPLIED):
    # the FPP switch set that produced the result, and the legacy verdict saved when the row is armed
    # (``triceratops_verdict_v1`` is filled by the server / arming script, never accepted from workers)
    "fpp_switches": ("JSONB", "TEXT"),
    "triceratops_verdict_v1": ("TEXT", "TEXT"),
    # alias phase-2 provenance, DDL scripts/migration/2026-11-02_harmonic_epoch_class.sql (NOT APPLIED): whether the
    # worker ran with the ``harmonic_epoch_class`` flag for THIS scan. Not an FPP field: it is NOT in RESULT_FIELDS
    # (the primary-result handler reads it from the result body and calls ``set_candidate_harmonic_epoch_class``).
    "harmonic_epoch_class": ("BOOLEAN", "INTEGER"),
}

#: Ephemeris provenance: the engines' result keys (Python ``fpp_ephemeris`` / Rust / Swift) are ``ephem_status``,
#: ``ephem_P``, ``ephem_t0``, ``ephem_shift_h`` and the nested record ``ephem_refine`` (only its ``reason`` and
#: ``delta_chi2`` are stored); the columns are ``EPHEM_COLUMNS``.
EPHEM_STATUSES = ("ok", "edge", "failed", "not_requested")
EPHEM_COLUMNS = ("ephem_status", "ephem_reason", "ephem_period_days", "ephem_t0", "ephem_shift_h",
                 "ephem_delta_chi2")

#: Columns a model >= ``FPP_MODEL_VERSION_GEOMETRY`` result must be able to store.  Without
#: them the number could not be attributed to the geometry / dilution convention that made it.
GEOMETRY_COLUMNS = ("geometry_version", "dilution_convention", "aperture_source", "depth_scale",
                    "population_version", "population_source", "population_sha256")

#: Fields a worker may send (top level of ``/api/worker/fpp_result`` or of one entry of
#: ``candidates`` in the primary result).  ``triceratops_fpp_v1`` / ``_nfpp_v1`` are NOT
#: accepted from workers: the server fills them from the row's old values.
RESULT_FIELDS = ("fpp_model_version", "fpp_scen", "fpp_backend", "fpp_backend_device",
                 "fpp_mc_se", "fpp_n_draws", "population_version",
                 "population_source", "population_sha256", "fpp_flags",
                 "geometry_version", "dilution_convention", "aperture_source", "depth_scale",
                 "nfpp_mc_se", "fpp_n_draws_total", "mc_min_ess", "mc_borderline", "mc_detail",
                 "ephem_status", "ephem_P", "ephem_t0", "ephem_shift_h", "ephem_refine",
                 "fpp_switches",
                 # accepted (the ports send them) but derivable, so not stored: refined = status ok,
                 # convention = box_start for stored rows, failed = status in (edge, failed)
                 "ephem_refined", "ephem_t0_convention", "ephem_refine_failed")

#: What every model >= ``FPP_MODEL_VERSION_GEOMETRY`` result must carry (400 otherwise).
GEOMETRY_RESULT_FIELDS = ("geometry_version", "dilution_convention", "aperture_source", "depth_scale")

DILUTION_CONVENTIONS = ("true", "false")

#: fpp_backend values of decision (s) 5 of the plan.
FPP_BACKENDS = ("cpu-python", "cpu-swift", "cpu-rust", "cuda", "metal", "opencl",
                "unavailable", "gpu_parity_failed")

MAX_SCEN_BYTES = 64 * 1024
#: fpp_switches: flat {switch name: mode}; today 13 entries of < 30 characters.
MAX_SWITCHES = 64
MAX_SWITCHES_BYTES = 4 * 1024
MAX_TEXT = 256


class FppFieldError(ValueError):
    """A new-model field of a result body is malformed."""


class FppModelSchemaMissing(RuntimeError):
    """A new-model FPP result arrived but the model columns do not exist (DDL not applied)."""


def _num(name, v, lo=None, integer=False):
    try:
        x = int(v) if integer else float(v)
    except (TypeError, ValueError):
        raise FppFieldError(f"{name} must be a number, got {v!r}") from None
    if not integer and x != x:                      # NaN would become NULL-ish garbage in PG
        raise FppFieldError(f"{name} is NaN")
    if lo is not None and x < lo:
        raise FppFieldError(f"{name} must be >= {lo}, got {x}")
    return x


def _finite(name, v, lo=0.0):
    x = _num(name, v, lo)
    if x in (float("inf"), float("-inf")):
        raise FppFieldError(f"{name} must be finite, got {x}")
    return x


def extract_model_fields(body: Optional[dict]) -> Dict[str, Any]:
    """The validated new-model fields present in *body* (absent keys stay absent; an
    explicit ``None`` is dropped).  Raises ``FppFieldError`` on a malformed value: the
    route turns that into HTTP 400 instead of writing a half-valid row.

    A result that carries any new field but no ``fpp_model_version`` is rejected: the
    version is what separates the new numbers from the legacy ones everywhere downstream.
    """
    if not body:
        return {}
    raw = {k: body[k] for k in RESULT_FIELDS if body.get(k) is not None}
    if not raw:
        return {}
    if "fpp_model_version" not in raw:
        raise FppFieldError("new FPP fields sent without fpp_model_version")
    out: Dict[str, Any] = {"fpp_model_version": _num("fpp_model_version", raw["fpp_model_version"], 1, True)}
    if "fpp_scen" in raw:
        scen = raw["fpp_scen"]
        if isinstance(scen, str):
            try:
                scen = json.loads(scen)
            except ValueError:
                raise FppFieldError("fpp_scen is not valid JSON") from None
        if not isinstance(scen, (dict, list)):
            raise FppFieldError("fpp_scen must be a JSON object or array")
        try:
            text = json.dumps(scen, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except ValueError:
            raise FppFieldError("fpp_scen contains NaN/Infinity") from None
        if len(text.encode("utf-8")) > MAX_SCEN_BYTES:
            raise FppFieldError(f"fpp_scen larger than {MAX_SCEN_BYTES} bytes")
        out["fpp_scen"] = text
    if "fpp_backend" in raw:
        if raw["fpp_backend"] not in FPP_BACKENDS:
            raise FppFieldError(f"fpp_backend {raw['fpp_backend']!r} not in {FPP_BACKENDS}")
        out["fpp_backend"] = raw["fpp_backend"]
    if "fpp_backend_device" in raw:
        out["fpp_backend_device"] = str(raw["fpp_backend_device"])[:MAX_TEXT]
    if "population_version" in raw:
        out["population_version"] = str(raw["population_version"])[:2 * MAX_TEXT]
    if "population_source" in raw:
        out["population_source"] = str(raw["population_source"])[:2 * MAX_TEXT]
    if "population_sha256" in raw:
        sha256 = str(raw["population_sha256"])
        if not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
            raise FppFieldError("population_sha256 must be 64 hexadecimal characters")
        out["population_sha256"] = sha256.lower()
    if "fpp_flags" in raw:
        flags = raw["fpp_flags"]
        if isinstance(flags, (list, tuple)):
            flags = ",".join(sorted(str(f) for f in flags))
        out["fpp_flags"] = str(flags)[:MAX_TEXT]
    if "geometry_version" in raw:
        out["geometry_version"] = _num("geometry_version", raw["geometry_version"], 1, True)
    if "dilution_convention" in raw:
        if raw["dilution_convention"] not in DILUTION_CONVENTIONS:
            raise FppFieldError(f"dilution_convention {raw['dilution_convention']!r} not in {DILUTION_CONVENTIONS}")
        out["dilution_convention"] = raw["dilution_convention"]
    if "aperture_source" in raw:
        src = raw["aperture_source"]
        if isinstance(src, (list, tuple)):
            src = ",".join(sorted(str(x) for x in src))
        out["aperture_source"] = str(src)[:MAX_TEXT]
    if "depth_scale" in raw:
        scale = _num("depth_scale", raw["depth_scale"], 0.0)
        if not (0.0 < scale < float("inf")):
            raise FppFieldError(f"depth_scale must be a positive finite number, got {scale}")
        out["depth_scale"] = scale
    if out["fpp_model_version"] >= FPP_MODEL_VERSION_GEOMETRY:
        missing = [k for k in GEOMETRY_RESULT_FIELDS if k not in out]
        if missing:
            raise FppFieldError(f"model {out['fpp_model_version']} result without {', '.join(missing)}")
    if "fpp_mc_se" in raw:
        out["fpp_mc_se"] = _finite("fpp_mc_se", raw["fpp_mc_se"], 0.0)
    if "fpp_n_draws" in raw:
        out["fpp_n_draws"] = _num("fpp_n_draws", raw["fpp_n_draws"], 0, True)
    if "nfpp_mc_se" in raw:
        out["nfpp_mc_se"] = _finite("nfpp_mc_se", raw["nfpp_mc_se"], 0.0)
    if "mc_min_ess" in raw:
        out["mc_min_ess"] = _finite("mc_min_ess", raw["mc_min_ess"], 0.0)
    if "fpp_n_draws_total" in raw:
        v = raw["fpp_n_draws_total"]
        # int(nan) / int(inf) raise ValueError, and a fractional draw count is not a count:
        # check before converting.
        if isinstance(v, bool) or (isinstance(v, float) and (v != v or abs(v) == float("inf") or v != int(v))):
            raise FppFieldError(f"fpp_n_draws_total must be an integer, got {v!r}")
        out["fpp_n_draws_total"] = _num("fpp_n_draws_total", raw["fpp_n_draws_total"], 0, True)
        if "fpp_n_draws" in out and out["fpp_n_draws_total"] < out["fpp_n_draws"]:
            raise FppFieldError("fpp_n_draws_total must be >= fpp_n_draws")
    if "mc_borderline" in raw:
        if not isinstance(raw["mc_borderline"], bool):
            raise FppFieldError("mc_borderline must be a boolean")
        out["mc_borderline"] = raw["mc_borderline"]
    if "mc_detail" in raw:
        detail = raw["mc_detail"]
        if isinstance(detail, str):
            try:
                detail = json.loads(detail)
            except ValueError:
                raise FppFieldError("mc_detail is not valid JSON") from None
        if not isinstance(detail, dict):
            raise FppFieldError("mc_detail must be a JSON object")
        try:
            text = json.dumps(detail, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except ValueError:
            raise FppFieldError("mc_detail contains NaN/Infinity") from None
        if len(text.encode("utf-8")) > MAX_SCEN_BYTES:
            raise FppFieldError(f"mc_detail larger than {MAX_SCEN_BYTES} bytes")
        out["mc_detail"] = text
    if "fpp_switches" in raw:
        out["fpp_switches"] = _switches_text(raw["fpp_switches"])
    out.update(_ephem_fields(raw))
    return out


def _switches_text(sw) -> str:
    """Validated ``fpp_switches`` (flat dict str -> str, bounded) as canonical JSON text."""
    if isinstance(sw, str):
        try:
            sw = json.loads(sw)
        except ValueError:
            raise FppFieldError("fpp_switches is not valid JSON") from None
    if not isinstance(sw, dict) or not sw:
        raise FppFieldError("fpp_switches must be a non-empty JSON object")
    if len(sw) > MAX_SWITCHES:
        raise FppFieldError(f"fpp_switches has more than {MAX_SWITCHES} entries")
    for k, v in sw.items():
        if not (isinstance(k, str) and isinstance(v, str)):
            raise FppFieldError("fpp_switches must map strings to strings")
        if not k or len(k) > MAX_TEXT or len(v) > MAX_TEXT:
            raise FppFieldError("fpp_switches key/value empty or too long")
    text = json.dumps(sw, sort_keys=True, separators=(",", ":"))
    if len(text.encode("utf-8")) > MAX_SWITCHES_BYTES:
        raise FppFieldError(f"fpp_switches larger than {MAX_SWITCHES_BYTES} bytes")
    return text


_EPHEM_WARNED: set = set()


def _ephem_fields(raw: dict) -> Dict[str, Any]:
    """Validated ephemeris-refinement provenance columns from the engines' keys (``ephem_status``,
    ``ephem_P``, ``ephem_t0``, ``ephem_shift_h`` and the nested ``ephem_refine``).

    LENIENT on purpose: this is provenance, not a result.  A malformed or non-finite value is dropped (and
    logged once per field), never raised: a 400 here would throw away a finished FPP computation that can take
    hours.  Whether the columns exist is decided later by the db layer (absent columns = field ignored)."""
    out: Dict[str, Any] = {}

    def drop(field, why):
        if field not in _EPHEM_WARNED:
            _EPHEM_WARNED.add(field)
            logger.warning("FPP result carried malformed ephemeris field %s (%s): dropped (warned once per field)",
                           field, why)

    def number(field, v, positive=False):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return drop(field, f"not a number: {v!r}")
        x = float(v)
        if x != x or x in (float("inf"), float("-inf")) or (positive and x <= 0.0):
            return drop(field, f"not finite{' and > 0' if positive else ''}: {v!r}")
        return x

    if "ephem_status" in raw:
        if raw["ephem_status"] in EPHEM_STATUSES:
            out["ephem_status"] = raw["ephem_status"]
        else:
            drop("ephem_status", f"not in {EPHEM_STATUSES}: {raw['ephem_status']!r}")
    for wire, col, positive in (("ephem_P", "ephem_period_days", True), ("ephem_t0", "ephem_t0", False),
                                ("ephem_shift_h", "ephem_shift_h", False)):
        if wire in raw:
            x = number(wire, raw[wire], positive)
            if x is not None:
                out[col] = x
    rec = raw.get("ephem_refine")
    if rec is not None:
        if not isinstance(rec, dict):
            drop("ephem_refine", "not an object")
        else:
            if isinstance(rec.get("reason"), str):
                out["ephem_reason"] = rec["reason"][:MAX_TEXT]
            if rec.get("delta_chi2") is not None:
                x = number("ephem_refine.delta_chi2", rec["delta_chi2"])
                if x is not None:
                    out["ephem_delta_chi2"] = x
    return out


def verdict_gate_active() -> bool:
    """Step 6 (v): until step 12 passes, new-model FPP numbers are NOT fed to the verdict
    classifiers.  ``EXONIT_FPP_V2_VERDICT_GATE=0`` lifts the gate (step 12 only)."""
    return os.environ.get("EXONIT_FPP_V2_VERDICT_GATE", "1") != "0"


def is_new_model(candidate: dict) -> bool:
    v = candidate.get("fpp_model_version")
    try:
        return v is not None and int(v) >= FPP_MODEL_VERSION_NEW
    except (TypeError, ValueError):
        return False


def gate_candidate_for_verdict(candidate: dict) -> dict:
    """The candidate dict the verdict classifiers should see.

    A new-model row (``fpp_model_version >= 2``) keeps feeding the classifiers what they
    were trained on: the legacy FPP/NFPP (``triceratops_fpp_v1`` / ``_nfpp_v1``); when
    there is no legacy value (a first-time computation by the new model) the FPP inputs
    are absent (None), exactly as for a never-computed FPP, and the FPP-derived
    ``triceratops_prob_planet`` / ``triceratops_verdict`` are dropped.  Any other row is
    returned unchanged (the same object).  The input is never mutated.
    """
    if not verdict_gate_active() or not is_new_model(candidate):
        return candidate
    gated = dict(candidate)
    gated["triceratops_fpp"] = candidate.get("triceratops_fpp_v1")
    gated["triceratops_nfpp"] = candidate.get("triceratops_nfpp_v1")
    if "nfpp" in gated:
        gated["nfpp"] = candidate.get("triceratops_nfpp_v1")
    gated["triceratops_prob_planet"] = None
    gated["triceratops_verdict"] = None
    gated["fpp_verdict_gated"] = True
    return gated


def cnn_null_fpp_enabled() -> bool:
    """The cnn-NULL FPP eligibility arm (``EXONIT_FPP_CNN_NULL=1``).  OFF by default: it
    widens the FPP queue to every row with ``cnn_score IS NULL`` and needs the ``_n``
    partial indexes (migration file) first; the size of the widening is a cost decision."""
    return os.environ.get("EXONIT_FPP_CNN_NULL", "0") == "1"


def task_extras(*, tic_id, exptime, fits_files, population: Optional[dict],
                geometry: Optional[dict] = None) -> Dict[str, Any]:
    """New fields of the FPP task payload (plan step 6 (iii)).

    * ``lc_source``   'spoc' / 'tglc' / 'qlp' / 'mixed' or None when it cannot be told
                      (the worker then runs both dilution conventions, D8);
    * ``cadence_s``   the light curve's exposure time in seconds (exposure integration, 0.3 (j));
    * ``population``  the cached Galaxia document (dict) or None, with
                      ``population_missing`` so the worker never silently falls back to
                      an online query (workers have no external connectivity);
    * ``geometry``    per-sector WCS/pixel geometry (decision (f)); ``{"kind": "synthetic"}``
                      until the step-3 provider exists: such a task can never be 'validated'.
    * ``tglc_near_edge``  NEAREDGE of the TGLC sectors of ``geometry`` (True if any sector, False if all
                      known, None if it cannot be told): selects the TGLC depth constant (1.27 / 1.09).
                      None for a TGLC light curve makes the engine flag the result and not validate it.
    """
    from exonit.physical_sanity import lc_source_from_fits_files
    try:
        cadence = float(exptime) if exptime is not None else None
    except (TypeError, ValueError):
        cadence = None
    return {
        "lc_source": lc_source_from_fits_files(fits_files) if fits_files else None,
        "cadence_s": cadence,
        "population": population,
        "population_missing": population is None,
        "geometry": geometry if geometry is not None else {"kind": "synthetic"},
        "tglc_near_edge": _tglc_near_edge(geometry),
    }


_SECTOR_IN_LC_NAME = (
    re.compile(r"^hlsp_tglc_tess_ffi_gaiaid-\d+-s(\d{4})-cam\d-ccd\d_"),
    re.compile(r"^hlsp_qlp_tess_ffi_s(\d{4})-"),
    re.compile(r"^tess\d+-s(\d{4})-\d+-\d+-s_lc\.fits$"),
)


def sector_of_lc_file(name) -> Optional[int]:
    """Sector number parsed from a TGLC / QLP / SPOC light-curve file name, or None."""
    base = os.path.basename(str(name or "")).lower()
    for rx in _SECTOR_IN_LC_NAME:
        m = rx.match(base)
        if m:
            return int(m.group(1))
    return None


def lc_files_by_sector(fits_files, paths: Dict[str, str]) -> Dict[int, str]:
    """``{sector: path}`` of the light-curve files of a metadata entry.

    ``paths`` is ``{file name: full path}`` from the ``fits_paths`` table.  A file the table does not know stays in
    the map under its bare NAME: ``fpp_geometry.product_of`` still tells its product from the name, the header read
    fails (not a file), and the sector then gets no header-derived value (fail closed), but is never mistaken for
    another product.  A sector with several files keeps the first in name order (deterministic).  A name without a
    parsable sector is skipped."""
    out: Dict[int, str] = {}
    for name in sorted(str(f) for f in (fits_files or ())):
        sector = sector_of_lc_file(name)
        if sector is not None and sector not in out:
            out[sector] = paths.get(name) or name
    return out


GEOMETRY_BUDGET_S = 2.0                       # hard per-task time budget of the geometry build (serving path)
_GEOM_POOL = None
_GEOM_POOL_LOCK = threading.Lock()
_OUTCOMES = collections.Counter()             # geometry outcomes + blob-read outcomes, per process
_OUTCOMES_LOCK = threading.Lock()
_OUTCOME_LOG_EVERY = 100                      # one INFO line with the counters per this many geometry builds


def _geom_pool():
    global _GEOM_POOL
    with _GEOM_POOL_LOCK:
        if _GEOM_POOL is None:
            _GEOM_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="fppgeom")
        return _GEOM_POOL


# Circuit breaker for the remote header reads: after BREAKER_THRESHOLD consecutive failed builds (header-read error or
# budget exceeded, with a remote reader) the geometry build is skipped for BREAKER_COOLDOWN_S and the flagged stub is
# shipped at once (``breaker_open``).  After the cool-down one failure re-opens it; one header read that works closes it.
BREAKER_THRESHOLD = 5
BREAKER_COOLDOWN_S = 60.0
_BREAKER = {"fails": 0, "open_until": 0.0}
_BREAKER_LOCK = threading.Lock()


def _breaker_is_open() -> bool:
    with _BREAKER_LOCK:
        if _BREAKER["open_until"] <= 0.0:
            return False
        if time.monotonic() < _BREAKER["open_until"]:
            return True
        _BREAKER["open_until"] = 0.0                    # half-open: the next failure re-opens at once
        _BREAKER["fails"] = max(0, BREAKER_THRESHOLD - 1)
        return False


def _breaker_note(failed: bool, ok: bool) -> None:
    with _BREAKER_LOCK:
        if ok:
            _BREAKER["fails"] = 0
        elif failed:
            _BREAKER["fails"] += 1
            if _BREAKER["fails"] >= BREAKER_THRESHOLD and _BREAKER["open_until"] <= 0.0:
                _BREAKER["open_until"] = time.monotonic() + BREAKER_COOLDOWN_S
                logger.warning("fpp geometry: %d consecutive failed remote header reads, skipping them for %.0f s",
                               _BREAKER["fails"], BREAKER_COOLDOWN_S)


def geometry_outcome_counts() -> Dict[str, int]:
    """Counters of the geometry builds of this process: ``built`` and the stub reasons (``no_radec``, ``no_sectors``,
    ``no_real_sector``, ``budget_exceeded``, ``error``) and the header-read outcomes (``blob_ok`` / ``blob_missing`` /
    ``blob_error`` / ``blob_bad``)."""
    with _OUTCOMES_LOCK:
        return dict(_OUTCOMES)


def _record_outcome(reason, blob_stats=None):
    with _OUTCOMES_LOCK:
        _OUTCOMES[reason] += 1
        _OUTCOMES["builds"] += 1
        for k, v in (blob_stats or {}).items():
            _OUTCOMES[k] += v
        n, snap = _OUTCOMES["builds"], dict(_OUTCOMES)
    if n % _OUTCOME_LOG_EVERY == 0:
        logger.info("fpp geometry outcomes (stub reasons and header reads): %s", snap)


def task_geometry(*, tic_id, entry, star, paths, roots, ffi_cache_dir, meta=None, store=None,
                  budget_s: float = GEOMETRY_BUDGET_S) -> Optional[dict]:
    """Real per-sector geometry blob of an FPP task, or None when none can be built (the caller then ships the
    synthetic stub exactly as before: such a task can never be 'validated').  See ``task_geometry_ex`` for the reason."""
    return task_geometry_ex(tic_id=tic_id, entry=entry, star=star, paths=paths, roots=roots, ffi_cache_dir=ffi_cache_dir,
                            meta=meta, store=store, budget_s=budget_s)[0]


def task_geometry_ex(*, tic_id, entry, star, paths, roots, ffi_cache_dir, meta=None, store=None,
                     budget_s: float = GEOMETRY_BUDGET_S):
    """``(blob or None, reason)``; ``reason`` is ``"built"`` or why the stub is shipped.

    ``entry``: metadata-cache entry (``ra``/``dec``/``sectors``/``fits_files``); ``star``: ``{"ra", "dec"}`` from the
    ``stars`` table, used when the entry has none; ``paths``: ``{file name: full path}`` (``fits_paths``).
    A light-curve file that is not a local file is read through ``meta.get_fits_blob`` + ``store.get_range`` (the
    ``/api/fits`` route's own path; header bytes only, ``exonit.lc_header_reader``) when ``meta`` and ``store`` are given.
    A sector without a K-epoch FFI WCS cache entry, or with an unreadable file / failed MinIO read, is listed in
    ``missing_sectors`` (the engine flags it and ``tglc_near_edge`` is None); nothing is ever invented for it.
    The whole build has a hard time budget (``budget_s``): past it the stub is shipped (``budget_exceeded``) and the
    serving path is never held longer.  Never raises."""
    from exonit.lc_header_reader import GeometryBudgetExceeded, RemoteHeaderReader
    deadline = time.monotonic() + float(budget_s)
    reader = RemoteHeaderReader(meta, store, deadline=deadline) if (meta is not None and store is not None) else None
    if reader is not None and _breaker_is_open():
        _record_outcome("breaker_open", None)
        return None, "breaker_open"

    def _build():
        from exonit import fpp_geometry as fg
        ra = entry.get("ra") if entry.get("ra") is not None else (star or {}).get("ra")
        dec = entry.get("dec") if entry.get("dec") is not None else (star or {}).get("dec")
        if ra is None or dec is None:
            return None, "no_radec"
        lc_files = lc_files_by_sector(entry.get("fits_files"), paths)
        sectors = sorted({int(s) for s in (entry.get("sectors") or ())} | set(lc_files))
        if not sectors:
            return None, "no_sectors"
        camera_ccd = {}
        for sector, path in lc_files.items():
            if fg.product_of(path) == "spoc":      # TGLC / QLP: build_task_geometry reads their headers itself
                prim, _ = fg._read_lc_header(path, reader)
                if prim is not None and "CAMERA" in prim and "CCD" in prim:
                    camera_ccd[sector] = (int(prim["CAMERA"]), int(prim["CCD"]))
        blob = fg.build_task_geometry(int(tic_id), float(ra), float(dec), sectors, roots=tuple(roots),
                                      ffi_cache_dir=ffi_cache_dir, camera_ccd=camera_ccd or None,
                                      lc_files=lc_files, header_reader=reader)
        if blob.get("kind") != "real":
            return None, "no_real_sector"
        json.dumps(blob)                            # must survive the task payload; fail closed otherwise
        return blob, "built"

    reason, blob = "error", None
    try:
        fut = _geom_pool().submit(_build)
        try:
            blob, reason = fut.result(timeout=max(0.0, deadline - time.monotonic()))
        except concurrent.futures.TimeoutError:
            fut.cancel()                            # a running build cannot be killed; its next read hits the deadline
            reason = "budget_exceeded"
            logger.warning("fpp geometry TIC %s: %.1f s budget exceeded, task ships the synthetic stub", tic_id, budget_s)
    except GeometryBudgetExceeded:
        reason = "budget_exceeded"
        logger.warning("fpp geometry TIC %s: %.1f s budget exceeded, task ships the synthetic stub", tic_id, budget_s)
    except Exception:
        logger.warning("fpp geometry TIC %s: not built, task ships the synthetic stub", tic_id, exc_info=True)
    stats = reader.stats_snapshot() if reader is not None else None
    _record_outcome(reason, stats)
    if reader is not None:
        _breaker_note(failed=reason == "budget_exceeded" or stats.get("blob_error", 0) > 0,
                      ok=stats.get("blob_ok", 0) > 0)
    return blob, reason


def _tglc_near_edge(geometry):
    from exonit.fpp_geometry import tglc_near_edge
    return tglc_near_edge(geometry)
