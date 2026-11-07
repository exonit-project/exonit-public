"""R8 Gaia reference counts WITHOUT millions of TAP jobs (fleet portability report, section 6.7).

The per-field cone query of ``fpp_background.gaia_tmag_tap`` is one async ESA job per field: 6.9M jobs
is not a credible load.  Instead Gaia DR3 is aggregated ONCE per sky region (a few thousand jobs) into a
lattice of 0.1 deg cells with per-Tmag-bin counts, cached in a local SQLite file; the Gaia reference of any
0.1 deg^2 field is then the area-weighted sum of the cells under its disc.

Lattice: dec zones of ``ZONE`` = 1 deg; zone z has ``n_z = floor(360 cos(dec_c) / H)`` ra cells of equal ra
width ``360 / n_z`` (true width ~ H) and ``ZONE / H`` dec bands: cell = (z, k, i).  One query covers one zone
and a run of whole ra columns, so no cell is ever split between two queries (counts are complete).

Tmag bins: the R8 bins of ``fpp_background.XCHECK_BINS`` (12-14, 14-16, 16-18, 18-21.5) plus T < 12 and the
total, from Gaia DR3 G < 21 with the Stassun+2019 G-T relation exactly as ``gaia_to_tmag`` (sources without
BP/RP: G - 0.43, a second job per query).  Approximation to keep in mind: counts are uniform inside a cell
(0.01 deg^2), the disc weights are numeric; the error against exact cone counts is measured on the dev fields
(``scripts/galaxia_gaia_counts.py calibrate``) and stored with the results.  Never blocks the population run:
R8 is evaluated later from the stored Galaxia bin counts and this file.
"""

from __future__ import annotations

import math
import os
import sqlite3
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from exonit import fpp_background as fb
except ImportError:
    import fpp_background as fb                    # type: ignore

H = 0.1                      # cell size (deg)
ZONE = 1.0                   # dec zone height (deg)
BANDS = int(round(ZONE / H))
DISC_RADIUS = math.sqrt(fb.FIELD_AREA_DEG2 / math.pi)        # 0.1784 deg
G_LIMIT = 21.0
# half-magnitude bin edges of the stored histogram, as integers 2*T: R8 bins are unions of them
BIN_EDGES_2T = (24, 28, 32, 36, 43)                           # T = 12, 14, 16, 18, 21.5
COLS = ("n_lt12", "n_12_14", "n_14_16", "n_16_18", "n_18_215", "n_total")

DDL = """
CREATE TABLE IF NOT EXISTS gaia_cells (
  z INTEGER NOT NULL, k INTEGER NOT NULL, i INTEGER NOT NULL,
  n_lt12 INTEGER NOT NULL, n_12_14 INTEGER NOT NULL, n_14_16 INTEGER NOT NULL, n_16_18 INTEGER NOT NULL,
  n_18_215 INTEGER NOT NULL, n_total INTEGER NOT NULL, PRIMARY KEY (z, k, i)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS gaia_queries (
  z INTEGER NOT NULL, i0 INTEGER NOT NULL, i1 INTEGER NOT NULL, status TEXT NOT NULL, n_rows INTEGER,
  seconds REAL, error TEXT, completed_at TEXT, PRIMARY KEY (z, i0, i1));
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


# --------------------------------------------------------------------------
# Lattice
# --------------------------------------------------------------------------

def zone_of(dec: float) -> int:
    return min(int(math.floor((dec + 90.0) / ZONE)), int(180 / ZONE) - 1)


def zone_geometry(z: int) -> Tuple[float, int, float]:
    """(zone lower dec edge, number of ra cells, ra width in degrees)."""
    zlo = -90.0 + z * ZONE
    zc = zlo + ZONE / 2.0
    n = max(1, int(math.floor(360.0 * math.cos(math.radians(zc)) / H)))
    return zlo, n, 360.0 / n


def cell_of(ra: float, dec: float) -> Tuple[int, int, int]:
    z = zone_of(dec)
    zlo, n, w = zone_geometry(z)
    k = min(int(math.floor((dec - zlo) / H)), BANDS - 1)
    return z, k, min(int(math.floor((ra % 360.0) / w)), n - 1)


def _ang_dist_deg(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = (np.radians(x) for x in (ra1, dec1, ra2, dec2))
    s = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def field_cells(ra0: float, dec0: float, r: float = DISC_RADIUS, m: int = 8) -> List[Tuple[Tuple[int, int, int], float]]:
    """Cells under the disc of radius ``r`` around (ra0, dec0) with the fraction of each cell inside the disc
    (``m`` x ``m`` sample points per cell)."""
    out: Dict[Tuple[int, int, int], float] = {}
    dlo, dhi = max(-90.0, dec0 - r), min(90.0, dec0 + r)
    for z in range(zone_of(dlo), zone_of(dhi) + 1):
        zlo, n, w = zone_geometry(z)
        cosmax = max(1e-6, min(math.cos(math.radians(max(abs(dlo), abs(dhi)))), 1.0))
        half = min(180.0, r / cosmax) + w
        i_lo = int(math.floor(((ra0 - half) % 360.0) / w))
        n_cols = min(n, int(math.ceil(2 * half / w)) + 1)
        for dk in range(n_cols):
            i = (i_lo + dk) % n
            for k in range(BANDS):
                d_lo = zlo + k * H
                if d_lo > dhi or d_lo + H < dlo:
                    continue
                s = (np.arange(m) + 0.5) / m
                dd = d_lo + H * s
                rr = i * w + w * s
                RR, DD = np.meshgrid(rr, dd)
                # unwrap ra around the field for the distance (only matters across ra = 0/360)
                inside = _ang_dist_deg(RR, DD, ra0 % 360.0, dec0) <= r
                f = float(inside.mean())
                if f > 0:
                    out[(z, k, i)] = out.get((z, k, i), 0.0) + f
    return sorted(out.items())


# --------------------------------------------------------------------------
# Query planning
# --------------------------------------------------------------------------

def target_columns(b_deg: float) -> int:
    """Ra columns (~0.1 deg of sky each, 1 deg tall) per ADQL job: 1 deg^2 in the plane, 8 deg^2 at high latitude
    (the density, hence the server time, varies by ~40x)."""
    ab = abs(b_deg)
    return 10 if ab < 5 else (20 if ab < 15 else 80)


def plan_queries(fields: Iterable[Tuple[float, float]], gap_columns: int = 3) -> List[Tuple[int, int, int]]:
    """Queries ``(z, i0, i1)`` (i1 exclusive) covering every cell under the discs of the given (ra, dec) fields.
    Columns of a zone are grouped into runs (gaps <= ``gap_columns`` are bridged) of at most
    ``target_columns(b)`` columns."""
    cols: Dict[int, set] = {}
    for ra0, dec0 in fields:
        for (z, k, i), _ in field_cells(ra0, dec0):
            cols.setdefault(z, set()).add(i)
    plan: List[Tuple[int, int, int]] = []
    for z, s in sorted(cols.items()):
        zlo, n, w = zone_geometry(z)
        ordered = sorted(s)
        runs, start, prev = [], ordered[0], ordered[0]
        for i in ordered[1:]:
            if i - prev - 1 > gap_columns:
                runs.append((start, prev + 1))
                start = i
            prev = i
        runs.append((start, prev + 1))
        for a, b in runs:
            i = a
            while i < b:
                ra_c = (i + 0.5) * w
                _, bb = fb.galactic_lb(ra_c, zlo + ZONE / 2.0)
                step = target_columns(bb)
                j = min(b, i + step)
                plan.append((z, i, j))
                i = j
    return plan


def _runs_to_plan(cols: Dict[int, "Sequence[int]"], gap_columns: int) -> List[Tuple[int, int, int]]:
    plan: List[Tuple[int, int, int]] = []
    for z, ordered in sorted(cols.items()):
        zlo, n, w = zone_geometry(z)
        runs, start, prev = [], ordered[0], ordered[0]
        for i in ordered[1:]:
            if i - prev - 1 > gap_columns:
                runs.append((start, prev + 1))
                start = i
            prev = i
        runs.append((start, prev + 1))
        for a, b in runs:
            i = a
            while i < b:
                _, bb = fb.galactic_lb((i + 0.5) * w, zlo + ZONE / 2.0)
                j = min(b, i + target_columns(bb))
                plan.append((z, i, j))
                i = j
    return plan


def plan_queries_fast(ra: "np.ndarray", dec: "np.ndarray", gap_columns: int = 3) -> List[Tuple[int, int, int]]:
    """Vectorised ``plan_queries`` for millions of fields: every field needs the columns i-2..i+2 of its own
    zone (a disc is 0.36 deg wide, a column ~0.1) and of the neighbouring zone when it is within the disc radius
    of a zone edge.  Over-covers ``plan_queries`` slightly (never under-covers; unit-tested)."""
    ra = np.asarray(ra, dtype=float) % 360.0
    dec = np.asarray(dec, dtype=float)
    cols: Dict[int, set] = {}
    z0 = np.minimum(np.floor((dec + 90.0) / ZONE).astype(int), int(180 / ZONE) - 1)
    frac = (dec + 90.0) - z0 * ZONE
    for dz in (-1, 0, 1):
        sel = np.ones(len(dec), dtype=bool)
        if dz == -1:
            sel = frac < DISC_RADIUS + 0.01
        elif dz == 1:
            sel = frac > ZONE - DISC_RADIUS - 0.01
        zz = z0 + dz
        sel &= (zz >= 0) & (zz < int(180 / ZONE))
        for z in np.unique(zz[sel]):
            zlo, n, w = zone_geometry(int(z))
            m = sel & (zz == z)
            i0 = np.floor(ra[m] / w).astype(int)
            # a field near the pole spans more columns: cos(dec) in the half-width
            span = int(math.ceil(DISC_RADIUS / max(1e-3, math.cos(math.radians(min(89.9, abs(zlo) + ZONE)))) / w)) + 1
            span = min(span, n // 2)
            allc = np.unique(np.concatenate([(i0 + d) % n for d in range(-span, span + 1)]))
            cols.setdefault(int(z), set()).update(int(x) for x in allc)
    return _runs_to_plan({z: sorted(v) for z, v in cols.items()}, gap_columns)


def _tmag_adql(with_colour: bool) -> str:
    if with_colour:
        c = "LEAST(GREATEST(phot_bp_mean_mag - phot_rp_mean_mag, -0.2), 3.5)"
        return (f"(phot_g_mean_mag - 0.00522555*POWER({c},3) + 0.0891337*POWER({c},2) - 0.633923*{c} + 0.0324473)")
    return "(phot_g_mean_mag - 0.43)"


def query_adql(z: int, i0: int, i1: int, with_colour: bool) -> str:
    zlo, n, w = zone_geometry(z)
    colour = ("phot_bp_mean_mag IS NOT NULL AND phot_rp_mean_mag IS NOT NULL" if with_colour
              else "(phot_bp_mean_mag IS NULL OR phot_rp_mean_mag IS NULL)")
    return (f"SELECT FLOOR((dec - {zlo})/{H}) AS k, FLOOR(ra/{w!r}) AS i, FLOOR(2*{_tmag_adql(with_colour)}) AS tb, COUNT(*) AS n "
            f"FROM gaiadr3.gaia_source WHERE dec >= {zlo} AND dec < {zlo + ZONE} AND ra >= {i0 * w!r} AND ra < {i1 * w!r} "
            f"AND phot_g_mean_mag < {G_LIMIT:g} AND {colour} GROUP BY k, i, tb")


def rows_to_cells(z: int, rows: Iterable[Tuple[int, int, int, int]]) -> Dict[Tuple[int, int, int], List[int]]:
    """Fold ``(k, i, tb, n)`` rows (tb = floor(2T)) into per-cell ``COLS`` counts."""
    cells: Dict[Tuple[int, int, int], List[int]] = {}
    for k, i, tb, n in rows:
        c = cells.setdefault((z, int(k), int(i)), [0] * len(COLS))
        tb = int(tb)
        idx = 0
        while idx < 5 and tb >= BIN_EDGES_2T[idx]:
            idx += 1
        # idx: 0 -> T < 12 ... 4 -> [18, 21.5); 5 -> T >= 21.5 (counted in the total only)
        if idx < 5:
            c[idx] += int(n)
        c[5] += int(n)
    return cells


# --------------------------------------------------------------------------
# Store + run
# --------------------------------------------------------------------------

def open_db(path: str) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=60)
    db.executescript(DDL)
    return db


def _covered(db: sqlite3.Connection, z: int, i0: int, i1: int) -> bool:
    """True when the columns [i0, i1) of zone z are already inside completed queries (a re-plan after a change of
    the field list or of the planner does not repeat finished work)."""
    pos = i0
    for a, b in db.execute("SELECT i0, i1 FROM gaia_queries WHERE z=? AND status='ok' AND i1>? AND i0<? ORDER BY i0", (z, i0, i1)):
        if a > pos:
            return False
        pos = max(pos, b)
        if pos >= i1:
            return True
    return pos >= i1


TAP_TIMEOUT_S = 900.0            # one ADQL job; typical 10-25 s, so a quarter of an hour means a hung ESA job


def call_with_timeout(fn: Callable, timeout_s: float, *args):
    """Run ``fn(*args)`` in a daemon thread and give up after ``timeout_s`` (TimeoutError).  astroquery's
    ``launch_job_async`` polls the ESA job without any limit; a hung job would otherwise stall the pre-step for good.
    The abandoned thread only polls and is a daemon (it never keeps the process alive); ``run_plan`` records the
    timeout as a failed query and carries on."""
    box: dict = {}

    def target():
        try:
            box["v"] = fn(*args)
        except BaseException as exc:                     # noqa: BLE001 - re-raised in the caller's thread
            box["e"] = exc
    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise TimeoutError(f"TAP call exceeded {timeout_s:.0f} s")
    if "e" in box:
        raise box["e"]
    return box["v"]


def run_plan(db: sqlite3.Connection, plan: Sequence[Tuple[int, int, int]], tap: Callable[[str], Iterable[tuple]],
             *, max_seconds: Optional[float] = None, log: Callable[[str], None] = lambda s: None) -> dict:
    """Execute the planned queries sequentially (resumable: done queries are skipped).  ``tap(adql)`` returns the
    result rows ``(k, i, tb, n)``; any failure is recorded and the loop continues -- R8 simply stays
    unavailable for fields whose cells are not covered."""
    res = {"done": 0, "skipped": 0, "failed": 0, "seconds": 0.0}
    t_start = time.time()
    for z, i0, i1 in plan:
        if _covered(db, z, i0, i1):
            res["skipped"] += 1
            continue
        if max_seconds is not None and time.time() - t_start > max_seconds:
            break
        t0 = time.time()
        try:
            rows = list(tap(query_adql(z, i0, i1, True))) + list(tap(query_adql(z, i0, i1, False)))
            cells = rows_to_cells(z, rows)
            with db:
                db.execute("DELETE FROM gaia_cells WHERE z=? AND i>=? AND i<?", (z, i0, i1))
                db.executemany("INSERT INTO gaia_cells VALUES (?,?,?,?,?,?,?,?,?)",
                               [(zz, k, i) + tuple(v) for (zz, k, i), v in sorted(cells.items())])
                db.execute("INSERT OR REPLACE INTO gaia_queries VALUES (?,?,?,?,?,?,?,?)",
                           (z, i0, i1, "ok", len(rows), round(time.time() - t0, 2), None,
                            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
            res["done"] += 1
        except Exception as exc:                         # noqa: BLE001 - recorded, never raised
            with db:
                db.execute("INSERT OR REPLACE INTO gaia_queries VALUES (?,?,?,?,?,?,?,?)",
                           (z, i0, i1, "failed", None, round(time.time() - t0, 2), f"{type(exc).__name__}: {exc}"[:400],
                            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
            res["failed"] += 1
            log(f"query z={z} {i0}-{i1} failed: {exc}")
        res["seconds"] = round(time.time() - t_start, 1)
    return res


def field_gaia_counts(db: sqlite3.Connection, ra: float, dec: float) -> Optional[dict]:
    """Area-weighted Gaia reference of one field: ``{"bins": [4 R8 bins], "lt12", "total", "source"}`` (counts
    are expected values, rounded to integers) or None when a cell under the disc is not covered by a completed query."""
    cells = field_cells(ra, dec)
    acc = np.zeros(len(COLS))
    for (z, k, i), f in cells:
        if not db.execute("SELECT 1 FROM gaia_queries WHERE z=? AND i0<=? AND i1>? AND status='ok'", (z, i, i)).fetchone():
            return None
        r = db.execute("SELECT n_lt12, n_12_14, n_14_16, n_16_18, n_18_215, n_total FROM gaia_cells WHERE z=? AND k=? AND i=?",
                       (z, k, i)).fetchone()
        if r:
            acc += f * np.array(r, dtype=float)
    acc = np.rint(acc).astype(int)
    return {"lt12": int(acc[0]), "bins": [int(x) for x in acc[1:5]], "total": int(acc[5]), "source": "cells-v1"}


def evaluate_r8(db: sqlite3.Connection, ra: float, dec: float, n_galaxia_bins: Sequence[int], n_galaxia: int) -> dict:
    """R8 for one stored field against the cell counts (same rule as ``fb.crosscheck_from_counts``)."""
    g = field_gaia_counts(db, ra, dec)
    if g is None:
        cx = fb.crosscheck_from_counts(n_galaxia_bins, n_galaxia, None, None)
        cx["gaia_error"] = "cells not covered"
        return cx
    cx = fb.crosscheck_from_counts(n_galaxia_bins, n_galaxia, g["bins"], g["total"])
    cx["gaia_source"] = g["source"]
    return cx
