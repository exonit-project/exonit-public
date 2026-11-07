"""Fleet-wide Galaxia background-population prefetch (FPP plan step 5, bulk generation).

Why a fleet: one population per TIC (design A) is ~13 s of Galaxia per field; 6.9M fields is
~100 core-days per host class, so the compute is spread over every machine the owner owns
(docs/reviews/2026-10-17-galaxia-prefetch-sizing.md, ...-fleet-portability.md).  Decisions of
2026-10-17: arm64 Macs take part under their own platform tag (1A), data lives on <host>
``/data/usb8tb/exonit-galaxia`` as one SQLite file per chunk, merged later (2A), no machine of a collaborator (3A).

What is stored (a different thing from the step-5 blob): the RAW Galaxia rows (``fb.RAW_COLUMNS``)
with a ``raw_sha256``, the per-Tmag-bin Galaxia counts under every known mapping (R8 stays
re-evaluable), and the field identity.  The step-5 document (``fb.serve_document``) is built from the
raw rows at serve / materialise time under a versioned mapping, so reverting cc1 -> v1 (plan A5)
never regenerates a field.  The served document is byte-identical to the previous one-step code.

Pieces (stdlib + numpy only, so a host needs no exonit install):

* work list    ``build_worklist``: coordinates CSV -> ~20k-TIC chunks sorted along a Morton curve on the
               sky, cut by estimated cost (plane fields cost more), chunk 0 = the validation fields.
* coordinator  ``Coordinator`` (SQLite state, atomic claim, heartbeat, stale release) behind a tiny
               HTTP API (``serve``): hosts need only outgoing HTTP, no ssh keys, no shared disk.
* runner       ``run_loop``: claim a chunk, run N Galaxia processes, 2 retries per TIC, write a
               per-chunk SQLite shard, publish it (sha256-verified upload) only when complete.
* merge        ``merge_shards``: ``ON CONFLICT DO NOTHING`` with a raw-hash conflict check.
* serving      ``RawStore``: raw store -> served document / legacy ``fpp_background`` rows.
* status       ``format_status``: chunks per host, failures, ETA.
"""

from __future__ import annotations

import csv
import hashlib
import http.client
import json
import math
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:                                              # repo layout
    from exonit import fpp_background as fb
except ImportError:                               # flat bundle (hosts without the exonit package)
    import fpp_background as fb                   # type: ignore
try:
    from exonit import galaxia_rs as rs
except ImportError:
    import galaxia_rs as rs                       # type: ignore

CHUNK_TICS = 20000
PRIORITY_CHUNK_TICS = 5000          # tiers 1-2: small chunks so the slowest host cannot hold the priority work for days
TIER_NAMES = {0: "validation", 1: "strong/finalist", 2: "TOI host", 3: "S1 (0<=FPP<0.5)", 4: "rest of scope"}
STALE_S = 600.0                  # a claim without heartbeat for this long is released
HEARTBEAT_S = 60.0
MAX_CLAIM_ATTEMPTS = 3           # a chunk released this many times needs an operator
MAX_CPU_FRACTION = 0.75          # CLAUDE.md: never use more than 75 % of a machine's CPUs
MIN_AVAIL_GB = 6.0               # runner: do not start a new Galaxia process below this much available RAM
LOCK_STALE_S = 180.0             # runner.lock whose heartbeat is older than this is dead
LOCK_TOUCH_S = 30.0

RAW_SCHEMA = """
CREATE TABLE IF NOT EXISTS fpp_background_raw (
  tic_id TEXT NOT NULL, raw_version TEXT NOT NULL,
  ra REAL NOT NULL, dec REAL NOT NULL, gal_l REAL NOT NULL, gal_b REAL NOT NULL, seed INTEGER NOT NULL,
  n_galaxia INTEGER NOT NULL, n_raw INTEGER NOT NULL,
  raw_meta TEXT NOT NULL, raw_sha256 TEXT NOT NULL, raw_blob BLOB NOT NULL,
  counts_json TEXT NOT NULL,
  host TEXT NOT NULL, chunk_id INTEGER NOT NULL, elapsed_s REAL, created_at TEXT NOT NULL,
  PRIMARY KEY (tic_id, raw_version));
CREATE TABLE IF NOT EXISTS fpp_background_fail (
  tic_id TEXT NOT NULL, chunk_id INTEGER NOT NULL, host TEXT NOT NULL, attempts INTEGER NOT NULL,
  error TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY (tic_id, chunk_id, host));
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""
_RAW_COLS = ("tic_id", "raw_version", "ra", "dec", "gal_l", "gal_b", "seed", "n_galaxia", "n_raw",
             "raw_meta", "raw_sha256", "raw_blob", "counts_json", "host", "chunk_id", "elapsed_s", "created_at")


def _key(tic) -> str:
    return str(int(tic))


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_sha256(path: str) -> str:
    return fb.file_sha256(path)


def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ==========================================================================
# Work list
# ==========================================================================

def _gal_b_vec(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Galactic latitude (deg) of arrays of ICRS positions; ordering / cost estimate only."""
    r, d = np.radians(ra), np.radians(dec)
    v = np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])
    z = fb._ICRS_TO_GAL[2] @ v
    return np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))


def field_cost(b_deg: np.ndarray) -> np.ndarray:
    """Relative Galaxia cost of a field (1.0 = high latitude).  Measured: 11.4 s at |b| > 18, 16.8 s at
    b = -3, 15.0 s at b = +0.3 (portability report, section 4)."""
    return 1.0 + 0.45 * np.exp(-np.abs(b_deg) / 8.0)


def _morton(ra: np.ndarray, dec: np.ndarray, bits: int = 16) -> np.ndarray:
    """Z-order key on a uniform (ra, dec) grid: consecutive keys are spatial neighbours."""
    x = np.minimum(((ra % 360.0) / 360.0 * (1 << bits)).astype(np.uint64), (1 << bits) - 1)
    y = np.minimum(((dec + 90.0) / 180.0 * (1 << bits)).astype(np.uint64), (1 << bits) - 1)
    key = np.zeros(len(x), dtype=np.uint64)
    for i in range(bits):
        key |= ((x >> np.uint64(i)) & np.uint64(1)) << np.uint64(2 * i)
        key |= ((y >> np.uint64(i)) & np.uint64(1)) << np.uint64(2 * i + 1)
    return key


def read_coords(path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """``tic,ra,dec`` CSV -> (tics, ra, dec); rows with unparsable or impossible coordinates are
    returned with NaN so that the caller can list them."""
    tics, ra, dec = [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                tics.append(_key(r["tic"]))
            except (ValueError, KeyError):
                continue
            try:
                ra.append(float(r["ra"]))
                dec.append(float(r["dec"]))
            except (ValueError, TypeError, KeyError):
                ra.append(float("nan"))
                dec.append(float("nan"))
    return tics, np.array(ra, dtype=float), np.array(dec, dtype=float)


QUEUE_DDL = """
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY, kind TEXT NOT NULL, n_tics INTEGER NOT NULL, cost REAL NOT NULL,
  csv_sha256 TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
  host TEXT, platform TEXT, claimed_at REAL, heartbeat_at REAL,
  progress_done INTEGER NOT NULL DEFAULT 0, progress_failed INTEGER NOT NULL DEFAULT 0,
  attempts INTEGER NOT NULL DEFAULT 0, finished_at REAL, shard_sha256 TEXT, shard_bytes INTEGER,
  n_ok INTEGER, n_fail INTEGER, elapsed_s REAL, peak_rss_mb REAL, tier INTEGER);
CREATE TABLE IF NOT EXISTS hosts (
  host TEXT PRIMARY KEY, platform TEXT, procs INTEGER, last_seen REAL, peak_rss_mb REAL);
CREATE TABLE IF NOT EXISTS events (ts REAL NOT NULL, chunk_id INTEGER, host TEXT, what TEXT NOT NULL);
-- the engine / raw-version family of the whole queue (a population is computed by ONE family, never mixed)
CREATE TABLE IF NOT EXISTS queue_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _write_chunk_csv(path: str, tics: Sequence[str], ra: np.ndarray, dec: np.ndarray) -> str:
    with open(path, "w", newline="") as f:
        f.write("tic,ra,dec\n")
        for t, a, d in zip(tics, ra, dec):
            f.write(f"{t},{a:.6f},{d:.6f}\n")
    return file_sha256(path)


def read_tiers(path: str) -> Dict[str, int]:
    """``tic,tier`` CSV (tier 1..3; header required) -> {tic: tier}.  A TIC listed twice keeps its lowest tier."""
    out: Dict[str, int] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            tr = int(r["tier"])
            if tr not in (1, 2, 3):
                raise ValueError(f"tier {tr} for TIC {r['tic']}: only 1..3 are listed (4 = everything else, 0 = validation)")
            k = _key(r["tic"])
            out[k] = min(tr, out.get(k, 9))
    return out


def build_worklist(out_dir: str, *, validation_csv: str, coords_csv: Optional[str] = None,
                   chunk_size: int = CHUNK_TICS, replicas: int = 1, source_note: str = "",
                   tiers_csv: Optional[str] = None, priority_chunk_size: int = PRIORITY_CHUNK_TICS,
                   engine: str = "cpp", family: Optional[str] = None) -> dict:
    """Create ``out_dir/{chunks/NNNN.csv, queue.db, manifest.json}``.

    Chunk 0 = the validation fields (explicit coordinates; ``replicas`` > 1 duplicates it as chunks
    0..replicas-1, used only by dry runs to compare hosts on the same fields).  The rest: ``coords_csv``
    (``tic,ra,dec`` from the production ``stars`` table), minus the validation TICs, minus invalid
    coordinates (listed in ``invalid_coords.csv``), Morton-sorted, cut so that every chunk has about
    ``chunk_size`` high-latitude-equivalents of work.

    Priority tiers (``tiers_csv`` = ``tic,tier`` with tier 1..3; a TIC not listed is tier 4; the validation
    fields are tier 0): chunks never mix tiers, tier 0 first, then 1 (strong / finalist candidates), 2 (TOI hosts),
    3 (S1: 0 <= FPP < 0.5), 4 (everything else); the claim order is (tier, id).  Inside a tier the TICs keep the
    Morton (sky) order, so every chunk is spatially compact, and the chunks are processed heaviest first.
    Tiers 1-2 are cut into ``priority_chunk_size`` chunks (a 20k chunk takes ~37 h on the slowest host).

    ``engine`` ("cpp" | "rs") and ``family`` (the raw_version family every shard of this queue must carry: ``galaxia-0.7.2``
    for the C++ engine, ``galaxia-rs-<cargo version>`` for galaxia-rs; required for "rs") are recorded in the queue:
    the coordinator refuses a runner or a shard of another family."""
    if engine not in ("cpp", "rs"):
        raise ValueError(f"engine must be 'cpp' or 'rs', got {engine!r}")
    if engine == "rs" and not family:
        raise ValueError("engine rs needs the raw-version family (galaxia-rs-<cargo version>): pass the binary or --family")
    family = family or fb.SOURCE
    if (engine == "rs") != family.startswith("galaxia-rs-"):
        raise ValueError(f"family {family!r} does not belong to engine {engine!r}")
    os.makedirs(os.path.join(out_dir, "chunks"), exist_ok=True)
    qpath = os.path.join(out_dir, "queue.db")
    if os.path.exists(qpath):
        raise FileExistsError(f"{qpath} exists: refusing to overwrite a work list")
    vt, vra, vdec = read_coords(validation_csv)
    if not vt or np.isnan(vra).any() or np.isnan(vdec).any():
        raise ValueError("validation list needs explicit finite coordinates for every TIC")
    vset = set(vt)
    chunks = []                                            # (id, kind, tics, ra, dec, cost)
    vcost = float(field_cost(_gal_b_vec(vra, vdec)).sum())
    for k in range(max(1, replicas)):
        chunks.append((k, "validation" if replicas == 1 else "dry", vt, vra, vdec, vcost, 0))
    n_invalid = 0
    n_main = 0
    if coords_csv:
        t, ra, dec = read_coords(coords_csv)
        t = np.array(t)
        ok = np.isfinite(ra) & np.isfinite(dec) & (ra >= 0) & (ra < 360) & (np.abs(dec) <= 90)
        bad = t[~ok]
        _, first = np.unique(t, return_index=True)         # first occurrence of every TIC
        keep = np.zeros(len(t), dtype=bool)
        keep[first] = True
        keep &= ok & ~np.isin(t, list(vset))
        n_invalid = int((~ok).sum())
        with open(os.path.join(out_dir, "invalid_coords.csv"), "w") as f:
            f.write("tic\n" + "".join(f"{x}\n" for x in bad))
        t, ra, dec = t[keep], ra[keep], dec[keep]
        tier_of = read_tiers(tiers_csv) if tiers_csv else {}
        tier = np.array([tier_of.get(x, 4) for x in t], dtype=int)
        n_main = int(len(t))
        nid = max(1, replicas)
        for tr in (1, 2, 3, 4):
            sel = np.nonzero(tier == tr)[0]
            if not len(sel):
                continue
            tt, rr, dd = t[sel], ra[sel], dec[sel]
            order = np.argsort(_morton(rr, dd), kind="stable")
            tt, rr, dd = tt[order], rr[order], dd[order]
            cost = field_cost(_gal_b_vec(rr, dd))
            cum_before = np.concatenate([[0.0], np.cumsum(cost)[:-1]])
            cid = np.floor(cum_before / float(priority_chunk_size if tr in (1, 2) else chunk_size)).astype(int)
            groups = []
            for c in range(int(cid.max()) + 1):
                idx = np.nonzero(cid == c)[0]
                groups.append((idx, float(cost[idx].sum() / len(idx))))
            groups.sort(key=lambda g: -g[1])               # heaviest chunks first: the light tail ends the tier
            for idx, _ in groups:
                chunks.append((nid, "main", list(tt[idx]), rr[idx], dd[idx], float(cost[idx].sum()), tr))
                nid += 1
    q = sqlite3.connect(qpath)
    q.executescript(QUEUE_DDL)
    q.executemany("INSERT INTO queue_meta VALUES (?,?)", [("engine", engine), ("family", family)])
    for cid_, kind, tics, ra_, dec_, cost_, tier_ in chunks:
        sha = _write_chunk_csv(os.path.join(out_dir, "chunks", f"{cid_:04d}.csv"), tics, ra_, dec_)
        q.execute("INSERT INTO chunks(id, kind, n_tics, cost, csv_sha256, tier) VALUES (?,?,?,?,?,?)",
                  (cid_, kind, len(tics), cost_, sha, tier_))
    q.commit()
    q.close()
    manifest = {"created": _now_iso(), "chunk_size": chunk_size, "n_chunks": len(chunks),
                "n_validation": len(vt), "n_main_tics": n_main, "n_invalid_coords": n_invalid,
                "coords_csv": coords_csv and {"path": coords_csv, "sha256": file_sha256(coords_csv)},
                "validation_csv": {"path": validation_csv, "sha256": file_sha256(validation_csv)},
                "replicas": replicas, "note": source_note, "engine": engine, "family": family,
                "tiers_csv": tiers_csv and {"path": tiers_csv, "sha256": file_sha256(tiers_csv)},
                "tiers": {str(tr): {"name": TIER_NAMES[tr], "chunks": sum(1 for c in chunks if c[6] == tr),
                                    "tics": sum(len(c[2]) for c in chunks if c[6] == tr)} for tr in range(5)}}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    return manifest


# ==========================================================================
# Coordinator
# ==========================================================================

class Coordinator:
    """Queue state in ``<queue_dir>/queue.db``; shards in ``<shards_dir>/NNNN/<host>.db``.

    Every mutation runs in a ``BEGIN IMMEDIATE`` transaction, so claims are atomic across the
    threads of the HTTP server (and across processes)."""

    def __init__(self, queue_dir: str, shards_dir: str, stale_s: float = STALE_S):
        self.queue_dir, self.shards_dir, self.stale_s = queue_dir, shards_dir, stale_s
        self.path = os.path.join(queue_dir, "queue.db")
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)
        if shards_dir:
            os.makedirs(shards_dir, exist_ok=True)
        c = self._conn()
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("CREATE TABLE IF NOT EXISTS queue_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")   # queues built before engines
        if "tier" not in {r[1] for r in c.execute("PRAGMA table_info(chunks)")}:     # queue built before priority tiers
            c.execute("ALTER TABLE chunks ADD COLUMN tier INTEGER")
            c.execute("UPDATE chunks SET tier = CASE WHEN kind IN ('validation','dry') THEN 0 ELSE 4 END")
        c.close()

    def queue_family(self) -> Optional[str]:
        """The raw-version family this queue was built for; None for a queue that predates engines (implicitly C++)."""
        c = self._conn()
        try:
            r = c.execute("SELECT value FROM queue_meta WHERE key='family'").fetchone()
            return r[0] if r else None
        finally:
            c.close()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        return c

    def _tx(self):
        c = self._conn()
        c.execute("BEGIN IMMEDIATE")
        return c

    # -- stale handling
    def release_stale(self, c: Optional[sqlite3.Connection] = None, now: Optional[float] = None) -> List[int]:
        own = c is None
        c = c or self._tx()
        now = now if now is not None else time.time()
        try:
            rows = c.execute("SELECT id, host, attempts FROM chunks WHERE state='running' AND heartbeat_at < ?",
                             (now - self.stale_s,)).fetchall()
            released = []
            for r in rows:
                new = "failed" if r["attempts"] >= MAX_CLAIM_ATTEMPTS else "pending"
                c.execute("UPDATE chunks SET state=?, host=NULL, platform=NULL, claimed_at=NULL, heartbeat_at=NULL,"
                          " progress_done=0, progress_failed=0 WHERE id=?", (new, r["id"]))
                c.execute("INSERT INTO events VALUES (?,?,?,?)",
                          (now, r["id"], r["host"], f"stale claim released -> {new}"))
                released.append(r["id"])
            if own:
                c.execute("COMMIT")
            return released
        except Exception:
            if own:
                c.execute("ROLLBACK")
            raise
        finally:
            if own:
                c.close()

    # -- claim / heartbeat / release
    def claim(self, host: str, platform: str, chunk_ids: Optional[Sequence[int]] = None,
              procs: int = 0, family: Optional[str] = None) -> dict:
        qf = self.queue_family()
        if family is not None and qf is not None and family != qf:
            c = self._tx()
            try:
                c.execute("INSERT INTO events VALUES (?,?,?,?)", (time.time(), None, host,
                                                                   f"claim refused: runner family {family} != queue {qf}"))
                c.execute("COMMIT")
            finally:
                c.close()
            return {"chunk_id": None, "remaining": None,
                    "refused": f"this queue is for raw-version family {qf}; the runner is {family}"}
        now = time.time()
        c = self._tx()
        try:
            self.release_stale(c, now)
            c.execute("INSERT INTO hosts(host, platform, procs, last_seen) VALUES (?,?,?,?) "
                      "ON CONFLICT(host) DO UPDATE SET platform=excluded.platform, procs=excluded.procs, "
                      "last_seen=excluded.last_seen", (host, platform, procs, now))
            allowed = ""
            args: list = []
            if chunk_ids:
                allowed = f" AND id IN ({','.join('?' * len(chunk_ids))})"
                args = [int(i) for i in chunk_ids]
            row = c.execute(f"SELECT * FROM chunks WHERE state='running' AND host=?{allowed} ORDER BY id LIMIT 1",
                            [host] + args).fetchone()
            resumed = row is not None
            if row is None:
                row = c.execute(f"SELECT * FROM chunks WHERE state='pending'{allowed} ORDER BY COALESCE(tier, 4), id LIMIT 1",
                                args).fetchone()
            if row is None:
                left = c.execute("SELECT COUNT(*) FROM chunks WHERE state IN ('pending','running')").fetchone()[0]
                c.execute("COMMIT")
                return {"chunk_id": None, "remaining": left}
            c.execute("UPDATE chunks SET state='running', host=?, platform=?, claimed_at=COALESCE(?, claimed_at),"
                      " heartbeat_at=?, attempts=attempts+? WHERE id=?",
                      (host, platform, None if resumed else now, now, 0 if resumed else 1, row["id"]))
            c.execute("INSERT INTO events VALUES (?,?,?,?)",
                      (now, row["id"], host, "claim resumed" if resumed else "claimed"))
            c.execute("COMMIT")
            return {"chunk_id": row["id"], "n_tics": row["n_tics"], "csv_sha256": row["csv_sha256"],
                    "kind": row["kind"], "resumed": resumed}
        except Exception:
            c.execute("ROLLBACK")
            raise
        finally:
            c.close()

    def heartbeat(self, host: str, chunk_id: int, done: int, failed: int, procs: int = 0,
                  rss_mb: Optional[float] = None) -> dict:
        now = time.time()
        c = self._tx()
        try:
            r = c.execute("SELECT state, host FROM chunks WHERE id=?", (chunk_id,)).fetchone()
            ok = r is not None and r["state"] == "running" and r["host"] == host
            if ok:
                c.execute("UPDATE chunks SET heartbeat_at=?, progress_done=?, progress_failed=? WHERE id=?",
                          (now, done, failed, chunk_id))
            c.execute("UPDATE hosts SET last_seen=?, procs=COALESCE(NULLIF(?,0), procs), "
                      "peak_rss_mb=MAX(COALESCE(peak_rss_mb,0), COALESCE(?,0)) WHERE host=?",
                      (now, procs, rss_mb, host))
            c.execute("COMMIT")
            return {"ok": ok, "state": None if r is None else r["state"]}
        except Exception:
            c.execute("ROLLBACK")
            raise
        finally:
            c.close()

    def release(self, host: str, chunk_id: int, reason: str = "") -> bool:
        c = self._tx()
        try:
            r = c.execute("UPDATE chunks SET state='pending', host=NULL, platform=NULL, claimed_at=NULL, "
                          "heartbeat_at=NULL, progress_done=0, progress_failed=0 WHERE id=? AND host=? AND state='running'",
                          (chunk_id, host)).rowcount
            c.execute("INSERT INTO events VALUES (?,?,?,?)", (time.time(), chunk_id, host, f"released by host: {reason}"))
            c.execute("COMMIT")
            return bool(r)
        except Exception:
            c.execute("ROLLBACK")
            raise
        finally:
            c.close()

    def chunk_csv_path(self, chunk_id: int) -> str:
        return os.path.join(self.queue_dir, "chunks", f"{int(chunk_id):04d}.csv")

    def shard_path(self, chunk_id: int, host: str) -> str:
        return os.path.join(self.shards_dir, f"{int(chunk_id):04d}", f"{host}.db")

    def complete(self, host: str, chunk_id: int, shard_sha: str, nbytes: int, n_ok: int, n_fail: int,
                 elapsed_s: float, rss_mb: Optional[float]) -> dict:
        """Mark a chunk done after its shard file was stored and its sha256 verified.  A valid complete
        shard from a host whose claim had meanwhile been released is accepted (the work is real) unless
        the chunk is already done."""
        now = time.time()
        c = self._tx()
        try:
            r = c.execute("SELECT state FROM chunks WHERE id=?", (chunk_id,)).fetchone()
            if r is None:
                c.execute("ROLLBACK")
                return {"ok": False, "error": "unknown chunk"}
            if r["state"] == "done":
                prev = c.execute("SELECT host, shard_sha256 FROM chunks WHERE id=?", (chunk_id,)).fetchone()
                c.execute("ROLLBACK")
                if prev["host"] == host and prev["shard_sha256"] == shard_sha:       # a retried PUT whose answer was lost
                    return {"ok": True, "duplicate": True}
                return {"ok": False, "error": "chunk already done by another host", "done_by": prev["host"]}
            plat = c.execute("SELECT platform FROM hosts WHERE host=?", (host,)).fetchone()
            c.execute("UPDATE chunks SET state='done', host=?, platform=?, finished_at=?, shard_sha256=?, shard_bytes=?,"
                      " n_ok=?, n_fail=?, elapsed_s=?, peak_rss_mb=?, progress_done=?, progress_failed=? WHERE id=?",
                      (host, plat["platform"] if plat else None, now, shard_sha, nbytes, n_ok, n_fail, elapsed_s,
                       rss_mb, n_ok + n_fail, n_fail, chunk_id))
            c.execute("INSERT INTO events VALUES (?,?,?,?)", (now, chunk_id, host, f"done ok={n_ok} fail={n_fail}"))
            c.execute("COMMIT")
            return {"ok": True}
        except Exception:
            c.execute("ROLLBACK")
            raise
        finally:
            c.close()

    def chunk_done_info(self, chunk_id: int) -> Optional[dict]:
        """``{"host", "shard_sha256", "shard_bytes"}`` if the chunk is done, else None."""
        c = self._conn()
        try:
            r = c.execute("SELECT host, shard_sha256, shard_bytes FROM chunks WHERE id=? AND state='done'",
                          (chunk_id,)).fetchone()
            return dict(r) if r else None
        finally:
            c.close()

    # -- status
    def status(self, now: Optional[float] = None) -> dict:
        now = now if now is not None else time.time()
        c = self._conn()
        try:
            self.release_stale(None, now)
            ch = [dict(r) for r in c.execute("SELECT * FROM chunks ORDER BY id")]
            hosts = {r["host"]: dict(r) for r in c.execute("SELECT * FROM hosts")}
        finally:
            c.close()
        st = compute_status(ch, hosts, now, self.stale_s)
        st["family"] = self.queue_family()
        return st


def compute_status(chunks: List[dict], hosts: Dict[str, dict], now: float, stale_s: float) -> dict:
    """Pure aggregation (unit-tested): per-host rates from completed chunks plus the running chunk's
    progress; ETA = remaining cost / summed rate of the hosts that are alive."""
    total_cost = sum(c["cost"] for c in chunks) or 1.0
    per: Dict[str, dict] = {}
    remaining = 0.0
    counts = {"pending": 0, "running": 0, "done": 0, "failed": 0}
    tics_total = sum(c["n_tics"] for c in chunks)
    tics_done = 0
    for c in chunks:
        counts[c["state"]] = counts.get(c["state"], 0) + 1
        unit = c["cost"] / max(1, c["n_tics"])
        if c["state"] == "done":
            tics_done += c["n_tics"]
        elif c["state"] == "running":
            tics_done += c["progress_done"]
            remaining += max(0.0, c["cost"] - c["progress_done"] * unit)
        else:
            remaining += c["cost"]
        h = c["host"]
        if h is None:
            continue
        p = per.setdefault(h, {"platform": c["platform"] or (hosts.get(h) or {}).get("platform"),
                               "done_chunks": [], "running": None, "failed_tics": 0, "cost_done": 0.0,
                               "elapsed_h": 0.0, "tics_done": 0})
        if c["state"] == "done":
            p["done_chunks"].append(c["id"])
            p["failed_tics"] += c["n_fail"] or 0
            p["cost_done"] += c["cost"]
            p["tics_done"] += c["n_tics"]
            p["elapsed_h"] += (c["elapsed_s"] or (c["finished_at"] - c["claimed_at"])) / 3600.0
        elif c["state"] == "running":
            el = max(1.0, now - (c["claimed_at"] or now))
            p["running"] = {"chunk": c["id"], "done": c["progress_done"], "failed": c["progress_failed"],
                            "n_tics": c["n_tics"], "age_min": round(el / 60.0, 1),
                            "heartbeat_age_s": round(now - (c["heartbeat_at"] or now))}
            p["cost_done"] += c["progress_done"] * unit
            p["tics_done"] += c["progress_done"]
            p["elapsed_h"] += el / 3600.0
    alive_rate = 0.0
    for h, p in per.items():
        hs = hosts.get(h) or {}
        p["procs"] = hs.get("procs")
        p["peak_rss_mb"] = hs.get("peak_rss_mb")
        p["rate_cost_per_h"] = p["cost_done"] / p["elapsed_h"] if p["elapsed_h"] > 0 else 0.0
        p["alive"] = bool(p["running"] and p["running"]["heartbeat_age_s"] < stale_s)
        if p["alive"]:
            alive_rate += p["rate_cost_per_h"]
    tiers: Dict[int, dict] = {}
    for c in chunks:
        tr = c.get("tier")
        tr = (0 if c["kind"] in ("validation", "dry") else 4) if tr is None else tr
        t = tiers.setdefault(tr, {"name": TIER_NAMES.get(tr, str(tr)), "chunks": 0, "chunks_done": 0, "tics": 0, "tics_done": 0,
                                  "remaining_cost": 0.0})
        unit = c["cost"] / max(1, c["n_tics"])
        t["chunks"] += 1
        t["tics"] += c["n_tics"]
        if c["state"] == "done":
            t["chunks_done"] += 1
            t["tics_done"] += c["n_tics"]
        elif c["state"] == "running":
            t["tics_done"] += c["progress_done"]
            t["remaining_cost"] += max(0.0, c["cost"] - c["progress_done"] * unit)
        else:
            t["remaining_cost"] += c["cost"]
    cum = 0.0
    for tr in sorted(tiers):                      # tiers are worked in order: "tiers <= t done" needs the cost of all of them
        cum += tiers[tr].pop("remaining_cost")
        tiers[tr]["eta_h_through"] = round(cum / alive_rate, 1) if alive_rate > 0 else (0.0 if cum <= 0 else None)
    return {"chunks": counts, "tics_total": tics_total, "tics_done": tics_done, "tiers": tiers,
            "remaining_cost_frac": round(remaining / total_cost, 4),
            "rate_cost_per_h_alive": round(alive_rate, 1),
            "eta_h": round(remaining / alive_rate, 1) if alive_rate > 0 else None,
            "hosts": per,
            "failed_chunks": [c["id"] for c in chunks if c["state"] == "failed"],
            "chunks_with_failed_tics": {c["id"]: c["n_fail"] for c in chunks if c["state"] == "done" and c["n_fail"]}}


def format_status(st: dict) -> str:
    cc = st["chunks"]
    lines = [f"chunks: done {cc['done']}  running {cc['running']}  pending {cc['pending']}  failed {cc['failed']}   "
             f"TICs {st['tics_done']:,}/{st['tics_total']:,}   remaining work {st['remaining_cost_frac']*100:.1f} %"]
    if st.get("family"):
        lines.insert(0, f"queue family: {st['family']}")
    eta = st["eta_h"]
    lines.append("ETA: " + ("unknown (no live host)" if eta is None else f"{eta:.1f} h = {eta/24:.1f} days")
                 + f"   (live rate {st['rate_cost_per_h_alive']:.0f} field-cost units/h)")
    lines.append(f"{'host':<20}{'plat':<8}{'procs':>6}{'rss MB':>8}{'rate/h':>9}  {'done chunks':<14}{'failed TICs':>11}  running")
    for h, p in sorted(st["hosts"].items()):
        r = p["running"]
        run = "-" if not r else (f"#{r['chunk']} {r['done']}/{r['n_tics']} (+{r['failed']} failed) age {r['age_min']} min"
                                 + ("" if p["alive"] else "  STALE"))
        rss = "" if p["peak_rss_mb"] is None else f"{p['peak_rss_mb']:.0f}"
        lines.append(f"{h:<20}{(p['platform'] or ''):<8}{p['procs'] or 0:>6}{rss:>8}{p['rate_cost_per_h']:>9.0f}  "
                     f"{len(p['done_chunks']):<14}{p['failed_tics']:>11}  {run}")
    if len(st.get("tiers", {})) > 1 or any(k != 4 for k in st.get("tiers", {})):
        lines.append("tier  name                  chunks done/total   TICs done/total        ETA until this tier is done")
        for tr, t in sorted(st["tiers"].items()):
            eta = t["eta_h_through"]
            lines.append(f"{tr:<6}{t['name']:<22}{t['chunks_done']:>5}/{t['chunks']:<10}{t['tics_done']:>10,}/{t['tics']:<11,}"
                         + ("unknown (no live host)" if eta is None else f"{eta:.1f} h = {eta / 24:.1f} days"))
    if st["failed_chunks"]:
        lines.append(f"FAILED chunks (claim limit reached, need an operator): {st['failed_chunks']}")
    if st["chunks_with_failed_tics"]:
        lines.append(f"chunks with failed TICs: {st['chunks_with_failed_tics']}")
    return "\n".join(lines)


# ==========================================================================
# HTTP API
# ==========================================================================

def shard_is_intact(path: str, sha: str, nbytes: Optional[int] = None) -> bool:
    """The stored shard exists, has the recorded size and the recorded sha256."""
    try:
        if nbytes is not None and os.path.getsize(path) != nbytes:
            return False
        return file_sha256(path) == sha
    except OSError:
        return False


def shard_families(path: str) -> set:
    """Raw-version families inside a shard: its ``meta.raw_version`` and every row's ``raw_version``.  Raises
    ``sqlite3.Error`` when the file is not a readable shard."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        fams = {fb.raw_family(r[0]) for r in db.execute("SELECT DISTINCT raw_version FROM fpp_background_raw")}
        r = db.execute("SELECT value FROM meta WHERE key='raw_version'").fetchone()
        if r:
            fams.add(fb.raw_family(json.loads(r[0])))
        return fams
    finally:
        db.close()


def check_shard_family(path: str, queue_family: Optional[str]) -> Optional[str]:
    """Why a shard may NOT be accepted by a queue of ``queue_family`` (None = acceptable).  A queue that records its
    family needs a shard whose every raw_version is of exactly that family (an unreadable shard, or one that names no
    version at all, cannot prove it).  A queue that predates engines is implicitly C++: it only refuses a galaxia-rs
    shard and, as before, does not look inside anything else."""
    try:
        fams = shard_families(path)
    except (sqlite3.Error, ValueError) as exc:
        return None if queue_family is None else f"the shard is not a readable shard ({exc}): its raw-version family cannot be established"
    if queue_family is None:
        rs_fams = sorted(f for f in fams if f.startswith("galaxia-rs-"))
        return f"this queue is the C++ engine's; the shard is {rs_fams}" if rs_fams else None
    if not fams:
        return "the shard carries no raw_version (no rows, no meta): its family cannot be established"
    if fams != {queue_family}:
        return f"raw-version family {sorted(fams)} != the queue's {queue_family}"
    return None


def verify_done_shards(coord: "Coordinator", merged_dbs: Sequence[str] = (), *, full: bool = True,
                       reopen: bool = True) -> dict:
    """Every chunk marked done must have its shard file with the recorded size (``full``: and sha256).  A
    shard that was removed by ``merge --delete-merged`` is fine when its file sha256 is registered in one of
    ``merged_dbs`` (``merged_shards``).  Anything else reopens the chunk (back to ``pending``; a damaged file
    is renamed ``*.corrupt``, never deleted) so another claim regenerates it."""
    merged = set()
    for m in merged_dbs:
        try:
            db = sqlite3.connect(f"file:{m}?mode=ro", uri=True)
            merged |= {r[0] for r in db.execute("SELECT file_sha256 FROM merged_shards")}
            db.close()
        except sqlite3.Error:
            pass
    c = coord._conn()
    try:
        rows = c.execute("SELECT id, host, shard_sha256, shard_bytes FROM chunks WHERE state='done'").fetchall()
    finally:
        c.close()
    rep = {"checked": len(rows), "ok": 0, "merged_away": 0, "reopened": [], "problems": {}}
    for r in rows:
        p = coord.shard_path(r["id"], r["host"])
        if os.path.exists(p):
            ok = (os.path.getsize(p) == r["shard_bytes"]) and (not full or file_sha256(p) == r["shard_sha256"])
            if ok:
                rep["ok"] += 1
                continue
            reason = "shard file differs from the recorded size/sha256"
        elif r["shard_sha256"] in merged:
            rep["merged_away"] += 1
            continue
        else:
            reason = "shard file missing (and not registered in a merged store)"
        rep["problems"][r["id"]] = reason
        if reopen:
            if os.path.exists(p):
                os.replace(p, p + f".corrupt-{int(time.time())}")
            tx = coord._tx()
            try:
                tx.execute("UPDATE chunks SET state='pending', host=NULL, platform=NULL, claimed_at=NULL, heartbeat_at=NULL,"
                           " finished_at=NULL, shard_sha256=NULL, shard_bytes=NULL, n_ok=NULL, n_fail=NULL, elapsed_s=NULL,"
                           " peak_rss_mb=NULL, progress_done=0, progress_failed=0, attempts=0 WHERE id=? AND state='done'",
                           (r["id"],))
                tx.execute("INSERT INTO events VALUES (?,?,?,?)", (time.time(), r["id"], r["host"], f"reopened: {reason}"))
                tx.execute("COMMIT")
            except Exception:
                tx.execute("ROLLBACK")
                raise
            finally:
                tx.close()
            rep["reopened"].append(r["id"])
    return rep


def make_handler(coord: Coordinator, token: str):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):             # quiet; the events table is the log
            pass

        def _auth(self) -> bool:
            if self.headers.get("Authorization", "") != f"Bearer {token}":
                self._json({"error": "unauthorized"}, 401)
                return False
            return True

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _body_json(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if not self._auth():
                return
            u = urllib.parse.urlparse(self.path)
            if u.path == "/status":
                return self._json(coord.status())
            if u.path.startswith("/chunk/"):
                try:
                    with open(coord.chunk_csv_path(int(u.path.split("/")[2])), "rb") as f:
                        b = f.read()
                except (OSError, ValueError):
                    return self._json({"error": "no such chunk"}, 404)
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            if not self._auth():
                return
            try:
                d = self._body_json()
                p = urllib.parse.urlparse(self.path).path
                if p == "/claim":
                    return self._json(coord.claim(d["host"], d["platform"], d.get("chunk_ids"), d.get("procs", 0),
                                                  d.get("family")))
                if p == "/heartbeat":
                    return self._json(coord.heartbeat(d["host"], int(d["chunk_id"]), int(d["done"]), int(d["failed"]),
                                                      int(d.get("procs", 0)), d.get("rss_mb")))
                if p == "/release":
                    return self._json({"ok": coord.release(d["host"], int(d["chunk_id"]), d.get("reason", ""))})
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                return self._json({"error": f"bad request: {exc}"}, 400)
            self._json({"error": "not found"}, 404)

        def do_PUT(self):
            if not self._auth():
                return
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            try:
                chunk_id = int(u.path.split("/")[2])
                host, want = q["host"][0], q["sha256"][0]
                n_ok, n_fail = int(q["n_ok"][0]), int(q["n_fail"][0])
                elapsed = float(q.get("elapsed_s", ["0"])[0])
                rss = float(q["rss_mb"][0]) if "rss_mb" in q else None
                length = int(self.headers["Content-Length"])
            except (KeyError, ValueError, IndexError):
                return self._json({"error": "bad request"}, 400)
            dest = coord.shard_path(chunk_id, host)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            tmp = dest + ".partial"
            h = hashlib.sha256()
            got = 0
            with open(tmp, "wb") as f:
                while got < length:
                    b = self.rfile.read(min(1 << 20, length - got))
                    if not b:
                        break
                    f.write(b)
                    h.update(b)
                    got += len(b)
            if got != length or h.hexdigest() != want:
                os.remove(tmp)
                return self._json({"error": f"shard upload corrupt (got {got}/{length} bytes, sha {h.hexdigest()[:12]})"}, 400)
            bad = check_shard_family(tmp, coord.queue_family())
            if bad:                                              # a population is one family: never store a mixed shard
                os.remove(tmp)
                return self._json({"ok": False, "error": f"shard refused: {bad}", "family_mismatch": True}, 409)
            # F2 ordering: the shard is stored (atomic rename) BEFORE the chunk is marked done, so a crash
            # between the two steps leaves a stored shard + a not-done chunk (the client's retry completes it),
            # never a done chunk without its shard.
            prev = coord.chunk_done_info(chunk_id)
            if prev is not None and not (prev["host"] == host and prev["shard_sha256"] == want):
                os.remove(tmp)
                return self._json({"ok": False, "error": "chunk already done by another host", "done_by": prev["host"]}, 409)
            if prev is not None and shard_is_intact(dest, want, prev["shard_bytes"]):
                os.remove(tmp)                                   # retried PUT of a stored shard: nothing to do
                return self._json({"ok": True, "path": dest, "duplicate": True})
            os.replace(tmp, dest)
            res = coord.complete(host, chunk_id, want, length, n_ok, n_fail, elapsed, rss)
            if not res["ok"]:                                    # lost a race with another host between the two steps
                if res.get("done_by") != host:
                    try:
                        os.remove(dest)
                    except OSError:
                        pass
                return self._json(res, 409)
            self._json({"ok": True, "path": dest})

    return H


def serve(queue_dir: str, shards_dir: str, token_file: str, host: str = "0.0.0.0", port: int = 0,
          stale_s: float = STALE_S, merged_dir: Optional[str] = None) -> None:
    token = open(token_file).read().strip()
    coord = Coordinator(queue_dir, shards_dir, stale_s)
    merged = sorted(os.path.join(merged_dir, n) for n in os.listdir(merged_dir)) if merged_dir and os.path.isdir(merged_dir) else []
    chk = verify_done_shards(coord, [m for m in merged if m.endswith(".db")], full=False, reopen=False)   # report only
    _log(f"startup check of done chunks: {chk['checked']} checked, {chk['ok']} ok, {chk['merged_away']} merged away, "
         + (f"PROBLEMS {chk['problems']} (run 'verify-done' to reopen them)" if chk["problems"] else "no problems"))
    srv = ThreadingHTTPServer((host, port), make_handler(coord, token))

    def watchdog():
        while True:
            time.sleep(60)
            try:
                coord.release_stale()
            except Exception:                           # noqa: BLE001 - the watchdog must not die
                traceback.print_exc()
    threading.Thread(target=watchdog, daemon=True).start()
    _log(f"coordinator listening on {srv.server_address} queue={queue_dir} shards={shards_dir} stale={stale_s}s")
    srv.serve_forever()


class Client:
    def __init__(self, url: str, token: str, retries: int = 6):
        self.url, self.token, self.retries = url.rstrip("/"), token, retries

    def _req(self, method: str, path: str, obj=None, raw: bool = False):
        delay = 2.0
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(
                    self.url + path, method=method,
                    data=None if obj is None else json.dumps(obj).encode(),
                    headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    b = r.read()
                return b if raw else json.loads(b)
            except urllib.error.HTTPError as exc:
                if 400 <= exc.code < 500:
                    raise
                err = exc
            except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as exc:
                err = exc
            if attempt == self.retries:
                raise err
            time.sleep(delay)
            delay = min(delay * 2, 120.0)

    def claim(self, host, platform, chunk_ids=None, procs=0, family=None):
        body = {"host": host, "platform": platform, "chunk_ids": chunk_ids, "procs": procs}
        if family is not None:
            body["family"] = family
        return self._req("POST", "/claim", body)

    def heartbeat(self, host, chunk_id, done, failed, procs=0, rss_mb=None):
        return self._req("POST", "/heartbeat", {"host": host, "chunk_id": chunk_id, "done": done, "failed": failed,
                                                "procs": procs, "rss_mb": rss_mb})

    def release(self, host, chunk_id, reason=""):
        return self._req("POST", "/release", {"host": host, "chunk_id": chunk_id, "reason": reason})

    def chunk(self, chunk_id: int) -> bytes:
        return self._req("GET", f"/chunk/{chunk_id}", raw=True)

    def status(self) -> dict:
        return self._req("GET", "/status")

    def upload_shard(self, chunk_id: int, host: str, path: str, sha: str, n_ok: int, n_fail: int,
                     elapsed_s: float, rss_mb: Optional[float]) -> dict:
        u = urllib.parse.urlparse(self.url)
        q = {"host": host, "sha256": sha, "n_ok": n_ok, "n_fail": n_fail, "elapsed_s": f"{elapsed_s:.1f}"}
        if rss_mb is not None:
            q["rss_mb"] = f"{rss_mb:.1f}"
        size = os.path.getsize(path)
        delay = 5.0
        for attempt in range(8):
            try:
                conn = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
                with open(path, "rb") as f:
                    conn.request("PUT", f"/shard/{chunk_id}?{urllib.parse.urlencode(q)}", body=f,
                                 headers={"Authorization": f"Bearer {self.token}", "Content-Length": str(size)})
                    resp = conn.getresponse()
                    body = json.loads(resp.read() or b"{}")
                conn.close()
                if resp.status == 200:
                    return body
                if resp.status == 409:
                    return {**body, "ok": False, "error": body.get("error", "conflict"), "final": True}
                err = body.get("error", resp.status)
            except (OSError, http.client.HTTPException) as exc:
                err = exc
            _log(f"shard upload attempt {attempt + 1} failed: {err}; retry in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 300.0)
        raise RuntimeError(f"shard upload of chunk {chunk_id} failed after retries")


# ==========================================================================
# Shard (per-chunk SQLite) and the runner
# ==========================================================================

def open_shard(path: str) -> sqlite3.Connection:
    c = sqlite3.connect(path, timeout=60)
    c.execute("PRAGMA journal_mode=DELETE")            # one self-contained file at the end
    c.execute("PRAGMA synchronous=NORMAL")
    c.executescript(RAW_SCHEMA)
    return c


def insert_raw(db: sqlite3.Connection, tic: str, raw_version: str, raw: dict, pk: dict, counts: dict,
               host: str, chunk_id: int, elapsed_s: float) -> None:
    db.execute(f"INSERT OR IGNORE INTO fpp_background_raw({','.join(_RAW_COLS)}) VALUES ({','.join('?' * len(_RAW_COLS))})",
               (tic, raw_version, raw["ra"], raw["dec"], raw["gal_l"], raw["gal_b"], raw["seed"],
                raw["n_galaxia"], int(len(raw["arrays"]["rad"])), pk["meta"], pk["sha256"],
                sqlite3.Binary(pk["blob"]), json.dumps(counts, sort_keys=True, separators=(",", ":")),
                host, chunk_id, round(elapsed_s, 2), _now_iso()))


def fleet_commit() -> Optional[str]:
    """Commit of the code bundle: ``<bundle>/COMMIT`` written by push_code.sh, else None (a dev checkout)."""
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "COMMIT")
    try:
        return open(p).read().strip() or None
    except OSError:
        return None


def child_peak_rss_mb() -> Optional[float]:
    """Largest resident set of any finished child process (= one Galaxia process); None on Windows."""
    try:
        import resource
    except ImportError:
        return None
    v = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return v / (1024.0 * 1024.0) if sys.platform == "darwin" else v / 1024.0


class RunnerConfig:
    def __init__(self, **kw):
        self.url: str = kw["url"]
        self.token: str = kw["token"]
        self.host: str = kw["host"]
        self.engine: str = kw.get("engine", "cpp")                  # "cpp" (Galaxia 0.7.2, one process per TIC) | "rs" (galaxia-rs batch)
        self.galaxia_root: Optional[str] = kw.get("galaxia_root")
        self.rs_binary: Optional[str] = kw.get("rs_binary")
        self.rs_data_dir: Optional[str] = kw.get("rs_data_dir")
        self.rs_backend: Optional[str] = kw.get("rs_backend")
        self.allow_fp32: bool = bool(kw.get("allow_fp32", False))   # test runs only: an fp32 backend is refused otherwise
        self.rs_threads: int = max(1, int(kw.get("rs_threads") or 1))   # galaxia-rs --threads: N threads share ONE resident load
        st = kw.get("rs_slice_tics")
        self.rs_slice_tics: int = int(st) if st else (max(rs.SLICE_TICS, 25 * self.rs_threads) if self.rs_threads > 1 else rs.SLICE_TICS)
        self.work_dir: str = kw["work_dir"]
        self.procs: int = int(kw["procs"])
        self.cores: Optional[List[int]] = kw.get("cores")            # Linux pinning only
        self.nice: int = int(kw.get("nice", 10))
        self.retries: int = int(kw.get("retries", 2))
        self.timeout_s: float = float(kw.get("timeout_s", 900.0))
        self.tmag_ceiling: Optional[float] = kw.get("tmag_ceiling")
        if self.engine == "rs" and self.tmag_ceiling is None:
            self.tmag_ceiling = rs.DEFAULT_CEILING                  # the rs population is +C22.5
        self.chunk_ids: Optional[List[int]] = kw.get("chunk_ids")
        self.max_chunks: int = int(kw.get("max_chunks", 0))
        self.exit_when_empty: bool = bool(kw.get("exit_when_empty", True))
        self.golden_path: Optional[str] = kw.get("golden_path")
        self.skip_conformance: bool = bool(kw.get("skip_conformance", False))
        self.force_procs: bool = bool(kw.get("force_procs", False))
        self.limit_tics: int = int(kw.get("limit_tics", 0))          # test aid: process only the first N TICs
        self.min_avail_gb: float = float(kw.get("min_avail_gb", MIN_AVAIL_GB))   # 0 disables the memory gate
        self.mem_poll_s: float = float(kw.get("mem_poll_s", 5.0))
        self.mem_settle_s: float = float(kw.get("mem_settle_s", 1.0))


def make_install(cfg: "RunnerConfig"):
    """The engine install a runner config describes: ``fb.GalaxiaInstall`` (cpp) or ``rs.RsInstall`` (rs)."""
    if cfg.engine == "rs":
        missing = [n for n in ("rs_binary", "rs_data_dir", "rs_backend") if not getattr(cfg, n)]
        if missing:
            raise ValueError(f"engine rs needs {', '.join('--' + m.replace('_', '-') for m in missing)}")
        return rs.RsInstall(cfg.rs_binary, cfg.rs_data_dir, cfg.rs_backend, cfg.allow_fp32)
    if cfg.engine != "cpp":
        raise ValueError(f"unknown engine {cfg.engine!r}")
    if not cfg.galaxia_root:
        raise ValueError("engine cpp needs --galaxia-root")
    return fb.GalaxiaInstall(cfg.galaxia_root)


def gate_floor_gb(min_avail_gb: float, install) -> float:
    """Available RAM required before a new process starts.  One C++ Galaxia process peaks at ~1 GB, so the floor is the
    headroom to keep; a galaxia-rs batch process holds ~``rs.RESIDENT_GB`` of resident data on top of it, so for that
    engine the floor is the headroom PLUS the process itself (0 still disables the gate)."""
    if min_avail_gb > 0 and isinstance(install, rs.RsInstall):
        return float(min_avail_gb) + rs.RESIDENT_GB
    return float(min_avail_gb)


def platform_label(install) -> str:
    """What a runner reports to the coordinator / writes into its shard as platform."""
    return install.platform if isinstance(install, rs.RsInstall) else fb.platform_tag()


def available_memory_mb() -> Optional[float]:
    """RAM that can be given to a new process without swapping, in MB; None when this OS cannot tell.
    Linux: ``MemAvailable``; macOS: free + inactive + speculative + purgeable pages of ``vm_stat``; Windows: ullAvailPhys."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1024.0
            return None
        if sys.platform == "darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10, check=False).stdout
            page = int(re.search(r"page size of (\d+) bytes", out).group(1))
            pages = 0
            for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
                m = re.search(rf"{key}:\s+(\d+)", out)
                pages += int(m.group(1)) if m else 0
            return pages * page / (1024.0 * 1024.0)
        if sys.platform == "win32":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return ms.ullAvailPhys / (1024.0 * 1024.0)
    except Exception:                                  # noqa: BLE001 - an unreadable probe must not stop the run
        return None
    return None


class MemoryGate:
    """F4: no new Galaxia process starts while the available RAM is below the floor.  Starts are serialised
    and spaced by ``settle_s`` so that processes started a moment ago have grown before the next check
    (one Galaxia process peaks at 0.6-1.2 GB)."""

    def __init__(self, floor_gb: float, poll_s: float = 5.0, settle_s: float = 1.0, log=None):
        self.floor_mb = float(floor_gb) * 1024.0
        self.poll_s, self.settle_s = poll_s, settle_s
        self.lock = threading.Lock()
        self.log = log or _log
        self.n_waits = 0
        self.unknown_logged = False

    def wait(self, stop: threading.Event, abort: threading.Event, should_stop=None) -> bool:
        """Block until a process may start; False if the runner is stopping meanwhile.  ``should_stop`` (the STOP
        file check) is polled while waiting: a starved runner must still obey ``stop``."""
        if self.floor_mb <= 0:
            return True
        with self.lock:
            t_wait = time.time()
            last_log = 0.0
            while True:
                if should_stop is not None and should_stop():
                    stop.set()
                if stop.is_set() or abort.is_set():
                    return False
                avail = available_memory_mb()
                if avail is None:
                    if not self.unknown_logged:
                        self.log("memory gate: available RAM cannot be read on this OS; gate inactive")
                        self.unknown_logged = True
                    return True
                if avail >= self.floor_mb:
                    break
                if time.time() - last_log > 60.0:
                    self.log(f"memory gate: {avail / 1024.0:.1f} GB available < floor {self.floor_mb / 1024.0:.1f} GB; "
                             f"not starting new Galaxia processes (waiting {time.time() - t_wait:.0f} s)")
                    last_log = time.time()
                    self.n_waits += 1
                stop.wait(self.poll_s)
            if self.settle_s > 0:
                stop.wait(self.settle_s)
            return not (stop.is_set() or abort.is_set())


def _pid_alive(pid: int) -> Optional[bool]:
    """True / False on POSIX, None where it cannot be told cheaply (Windows: the heartbeat decides)."""
    if sys.platform == "win32":
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunnerLockHeld(RuntimeError):
    pass


class RunnerLock:
    """F5: one runner per work dir (= per host name).  ``<work_dir>/runner.lock`` holds the pid and a heartbeat
    timestamp refreshed every ``LOCK_TOUCH_S``; a lock is live while its heartbeat is younger than
    ``LOCK_STALE_S`` (and, on POSIX, its pid exists).  A second runner refuses to start."""

    def __init__(self, work_dir: str, host: str):
        self.path = os.path.join(work_dir, "runner.lock")
        self.host, self.work_dir = host, work_dir
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

    @staticmethod
    def holder(work_dir: str, now: Optional[float] = None) -> Optional[dict]:
        """The live holder's lock content, or None (no lock / stale lock)."""
        path = os.path.join(work_dir, "runner.lock")
        t_now = now if now is not None else time.time()
        try:
            with open(path) as f:
                info = json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):                   # unreadable / half-written: judge by the file's mtime
            try:
                fresh = t_now - os.path.getmtime(path) < LOCK_STALE_S
            except OSError:
                return None
            return {"pid": None, "host": "?", "ts": os.path.getmtime(path)} if fresh else None
        age = t_now - float(info.get("ts", 0))
        if age >= LOCK_STALE_S:
            return None
        alive = _pid_alive(int(info["pid"])) if info.get("pid") else None
        if alive is False:
            return None
        return info

    def _write(self, exclusive: bool = False) -> None:
        data = json.dumps({"pid": os.getpid(), "host": self.host, "ts": time.time()})
        if exclusive:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(data)
        else:
            tmp = self.path + f".{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                f.write(data)
            os.replace(tmp, self.path)

    def acquire(self) -> None:
        os.makedirs(self.work_dir, exist_ok=True)
        for _ in range(3):
            try:
                self._write(exclusive=True)
                break
            except FileExistsError:
                h = self.holder(self.work_dir)
                if h is not None:
                    raise RunnerLockHeld(f"another runner holds {self.path}: pid {h.get('pid')} host {h.get('host')}, heartbeat "
                                         f"{time.time() - float(h.get('ts', 0)):.0f} s ago (stale after {LOCK_STALE_S:.0f} s)")
                try:                                    # stale: move it aside atomically (the loser of a race finds nothing)
                    os.replace(self.path, self.path + f".stale.{os.getpid()}")
                    os.remove(self.path + f".stale.{os.getpid()}")
                except OSError:
                    pass
        else:
            raise RunnerLockHeld(f"could not take {self.path}")

        def touch():
            while not self._stop.wait(LOCK_TOUCH_S):
                try:
                    self._write()
                except OSError:
                    pass                                 # Windows: a reader may hold the file for a moment
        self._thr = threading.Thread(target=touch, daemon=True)
        self._thr.start()

    def release(self) -> None:
        self._stop.set()
        try:
            with open(self.path) as f:
                if json.load(f).get("pid") == os.getpid():
                    os.remove(self.path)
        except (OSError, ValueError):
            pass


def check_procs(procs: int, force: bool = False) -> None:
    """25 % of every machine stays free (CLAUDE.md)."""
    cpus = os.cpu_count() or 1
    cap = max(1, int(cpus * MAX_CPU_FRACTION))
    if procs > cap and not force:
        raise SystemExit(f"--procs {procs} exceeds 75 % of {cpus} CPUs (= {cap}); refusing")


def stop_requested(cfg: RunnerConfig) -> bool:
    return os.path.exists(os.path.join(cfg.work_dir, "STOP"))


def run_chunk(cfg: RunnerConfig, install: "fb.GalaxiaInstall", chunk_id: int,
              items: Sequence[Tuple[str, float, float]], shard_path: str,
              stop: threading.Event, abort: threading.Event, counters: dict) -> dict:
    """Generate every TIC of a chunk into ``shard_path`` (resumable: TICs already stored are skipped).
    Returns ``{"status": "complete"|"stopped"|"aborted", "n_ok", "n_fail"}``."""
    db = open_shard(shard_path)
    raw_version = install.raw_version(cfg.tmag_ceiling)
    have = {r[0] for r in db.execute("SELECT tic_id FROM fpp_background_raw WHERE raw_version=?", (raw_version,))}
    prior_fail = {r[0] for r in db.execute("SELECT tic_id FROM fpp_background_fail WHERE chunk_id=?", (chunk_id,))}
    if isinstance(install, rs.RsInstall):
        other = [r[0] for r in db.execute("SELECT DISTINCT raw_version FROM fpp_background_raw WHERE raw_version<>?", (raw_version,))]
        if other:                                       # e.g. the binary was rebuilt between two runs of this chunk
            db.close()
            raise rs.RsFatal(f"the partial shard {shard_path} holds rows of another raw version {other}; this runner is "
                             f"{raw_version}: remove the shard (a chunk is never mixed)")
    todo = [(t, ra, dec) for (t, ra, dec) in items if t not in have]
    counters["done"] = len(have)
    counters["failed"] = 0
    free: "list[Optional[int]]" = list(cfg.cores) if cfg.cores and sys.platform.startswith("linux") else []
    lock = threading.Lock()

    gate = MemoryGate(gate_floor_gb(cfg.min_avail_gb, install), cfg.mem_poll_s, cfg.mem_settle_s)
    counters["mem_gate_waits"] = 0
    counters["peak_field_rss_mb"] = 0.0
    counters["peak_field_tic"] = None

    def job(item):
        tic, ra, dec = item
        with lock:
            cpu = free.pop() if free else None
        try:
            last = "stopped"
            for attempt in range(1, cfg.retries + 2):
                if stop.is_set() or abort.is_set():
                    return item, None, "stopped", attempt - 1
                if not gate.wait(stop, abort, lambda: stop_requested(cfg)):
                    return item, None, "stopped", attempt - 1
                try:
                    t0 = time.time()
                    st: dict = {}
                    raw = fb.generate_raw(install, tic, ra, dec, cpu=cpu, nice=cfg.nice, timeout_s=cfg.timeout_s,
                                          workdir=os.path.join(cfg.work_dir, "scratch"), tmag_ceiling=cfg.tmag_ceiling,
                                          stats=st)
                    pk = fb.pack_raw(raw["arrays"])
                    counts = fb.mapping_bin_counts(raw["arrays"])
                    return item, (raw, pk, counts, time.time() - t0, st.get("galaxia_rss_mb")), None, attempt
                except Exception as exc:                # noqa: BLE001 - retried, then recorded
                    last = f"{type(exc).__name__}: {exc}"[:600]
            return item, None, last, cfg.retries + 1
        finally:
            if cpu is not None:
                with lock:
                    free.append(cpu)

    n_new = 0

    def store(item, res, err, attempts):
        """Record one finished TIC (a raw field, or its failure) in the shard; ``err == "stopped"`` records nothing."""
        nonlocal n_new
        tic = item[0]
        if res is not None:
            raw, pk, counts, dt, rss = res
            insert_raw(db, tic, raw_version, raw, pk, counts, cfg.host, chunk_id, dt)
            _log(f"chunk {chunk_id} tic {tic}: n_raw {len(raw['arrays']['rad'])} galaxia {dt:.1f} s"
                 + ("" if rss is None else f" peak RSS {rss:.0f} MB"))
            if rss is not None and rss > counters["peak_field_rss_mb"]:
                counters["peak_field_rss_mb"], counters["peak_field_tic"] = rss, tic
            db.execute("DELETE FROM fpp_background_fail WHERE tic_id=? AND chunk_id=?", (tic, chunk_id))
            counters["done"] += 1
            n_new += 1
        elif err != "stopped":
            db.execute("INSERT OR REPLACE INTO fpp_background_fail VALUES (?,?,?,?,?,?)",
                       (tic, chunk_id, cfg.host, attempts, err, _now_iso()))
            counters["failed"] += 1
            n_new += 1
        if n_new % 200 == 0:
            db.commit()

    if isinstance(install, rs.RsInstall):
        try:
            _drain_rs(cfg, install, todo, store, db, gate, stop, abort, counters, free, lock)
            db.commit()
        finally:
            db.commit()
    else:
        pending = set()
        it = iter(todo)
        exhausted = False
        ex = ThreadPoolExecutor(max_workers=max(1, cfg.procs))
        try:
            while True:
                while not exhausted and len(pending) < cfg.procs * 2 and not stop.is_set() and not abort.is_set():
                    if cfg.limit_tics and n_new + len(pending) >= cfg.limit_tics:
                        exhausted = True
                        break
                    nxt = next(it, None)
                    if nxt is None:
                        exhausted = True
                        break
                    pending.add(ex.submit(job, nxt))
                if not pending:
                    break
                done_set, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done_set:
                    item, res, err, attempts = fut.result()
                    store(item, res, err, attempts)
                if stop_requested(cfg):
                    stop.set()
            db.commit()
        finally:
            ex.shutdown(wait=True)
            db.commit()
    counters["mem_gate_waits"] = gate.n_waits
    n_ok = db.execute("SELECT COUNT(*) FROM fpp_background_raw").fetchone()[0]
    n_fail = db.execute("SELECT COUNT(*) FROM fpp_background_fail WHERE chunk_id=?", (chunk_id,)).fetchone()[0]
    full = (n_ok + n_fail) >= len(items) if not cfg.limit_tics else True
    status = "aborted" if abort.is_set() else ("stopped" if stop.is_set() or not full else "complete")
    if status == "complete":
        for k, v in (("chunk_id", chunk_id), ("host", cfg.host), ("platform", platform_label(install)),
                     ("engine", getattr(install, "engine", "cpp")), ("family", fb.raw_family(raw_version)),
                     ("raw_version", raw_version), ("finished", _now_iso()), ("n_ok", n_ok), ("n_fail", n_fail),
                     ("procs", cfg.procs), ("python", sys.version.split()[0]), ("numpy", np.__version__),
                     ("binary_sha256", install.provenance().get("binary_sha256", "")),
                     ("tmag_ceiling", cfg.tmag_ceiling), ("peak_child_rss_mb", child_peak_rss_mb()),
                     ("peak_field_rss_mb", counters.get("peak_field_rss_mb")), ("peak_field_tic", counters.get("peak_field_tic")),
                     ("min_avail_gb", cfg.min_avail_gb), ("mem_gate_waits", counters.get("mem_gate_waits")),
                     ("repo_commit", fleet_commit())):
            db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, json.dumps(v)))
        db.commit()
    db.close()
    return {"status": status, "n_ok": n_ok, "n_fail": n_fail}


def _drain_rs(cfg: RunnerConfig, install: "rs.RsInstall", todo: Sequence[Tuple[str, float, float]], store, db,
              gate: "MemoryGate", stop: threading.Event, abort: threading.Event, counters: dict,
              free: "list[Optional[int]]", lock: threading.Lock) -> None:
    """galaxia-rs engine: ``cfg.procs`` batch processes work through the not-yet-stored TICs in small slices (``cfg.rs_slice_tics``,
    but never so large that a process has nothing to share: a short todo is split evenly over the processes).  Each process
    is pinned to one of ``cfg.cores`` on Linux (none on macOS / Windows); starts go through the memory gate and are serialised
    until the resident data is loaded.  Fields reach this (the calling) thread through a bounded queue as the binary finishes
    them, so the shard, the counters and the heartbeat advance field by field and the DB is only touched by one thread."""
    import queue as _queue
    todo = list(todo)
    if cfg.limit_tics:
        todo = todo[:cfg.limit_tics]
    if not todo:
        return
    procs = max(1, cfg.procs)
    size = max(1, min(cfg.rs_slice_tics, -(-len(todo) // procs)))
    slices = [todo[i:i + size] for i in range(0, len(todo), size)]
    expect = install.opencl_device()["name"] if install.backend == "opencl" else None
    q: "_queue.Queue" = _queue.Queue(maxsize=max(8, procs * 4))
    start_lock = threading.Lock()
    kill = threading.Event()                        # abort of the batch processes: claim lost, or a fatal error elsewhere
    scratch = os.path.join(cfg.work_dir, "scratch")
    os.makedirs(scratch, exist_ok=True)

    def emit(item, raw, err, attempts, dt, rss):
        if raw is not None:
            pk = fb.pack_raw(raw["arrays"])
            counts = fb.mapping_bin_counts(raw["arrays"])
            q.put(("item", item, (raw, pk, counts, dt, rss), None, attempts))
        else:
            q.put(("item", item, None, err, attempts))

    def job(sl):
        if cfg.rs_threads > 1 and cfg.cores and sys.platform.startswith("linux"):
            cpu = ",".join(str(c) for c in cfg.cores)       # threads share the whole core list (taskset -c a,b)
        else:
            with lock:
                cpu = free.pop() if free else None
        info = {"rss_mb": None}
        try:
            info = rs.generate_slice(install, sl, scratch, retries=cfg.retries, nice=cfg.nice, stall_s=cfg.timeout_s,
                                     ceiling=cfg.tmag_ceiling, cpu=cpu, threads=cfg.rs_threads, emit=emit, stop=stop, abort=kill,
                                     start_lock=start_lock, gate_wait=lambda: gate.wait(stop, abort, lambda: stop_requested(cfg)),
                                     expect_device=expect, log=_log)
        except BaseException as exc:                # noqa: BLE001 - reported to the draining thread
            q.put(("fatal", exc))
        finally:
            if cpu is not None and cfg.rs_threads <= 1:
                with lock:
                    free.append(cpu)
            q.put(("done", info))

    ex = ThreadPoolExecutor(max_workers=procs)
    futs = [ex.submit(job, sl) for sl in slices]
    left, fatal = len(futs), None
    try:
        while left:
            if abort.is_set():
                kill.set()
            try:
                msg = q.get(timeout=1.0)
            except _queue.Empty:
                if stop_requested(cfg):
                    stop.set()
                continue
            if msg[0] == "item":
                store(msg[1], msg[2], msg[3], msg[4])
            elif msg[0] == "fatal":
                fatal = fatal or msg[1]
                kill.set()
            else:
                left -= 1
                rss = (msg[1] or {}).get("rss_mb")
                if rss is not None and rss > counters["peak_field_rss_mb"]:
                    counters["peak_field_rss_mb"], counters["peak_field_tic"] = rss, "batch process"
                db.commit()
            if stop_requested(cfg):
                stop.set()
    finally:
        kill.set()
        while not all(f.done() for f in futs):      # producers blocked on a full queue must be able to finish
            try:
                q.get(timeout=0.2)
            except _queue.Empty:
                pass
        ex.shutdown(wait=True)
    if fatal is not None:
        raise fatal


def read_chunk_items(csv_bytes: bytes) -> List[Tuple[str, float, float]]:
    rows = csv.DictReader(csv_bytes.decode().splitlines())
    return [(_key(r["tic"]), float(r["ra"]), float(r["dec"])) for r in rows]


# -- conformance gate -------------------------------------------------------

def load_golden(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def raw_fingerprint(install: "fb.GalaxiaInstall", tic, ra, dec, *, cpu=None, nice=10, workdir=None) -> dict:
    raw = fb.generate_raw(install, tic, ra, dec, cpu=cpu, nice=nice, workdir=workdir)
    pk = fb.pack_raw(raw["arrays"])
    return {"ra": raw["ra"], "dec": raw["dec"], "n_raw": int(len(raw["arrays"]["rad"])), "raw_sha256": pk["sha256"]}


def rs_fingerprints(install: "rs.RsInstall", fields: Dict[str, Tuple[float, float]], procs: int, work_dir: str,
                    cores: Optional[Sequence[int]] = None, *, min_avail_gb: float = MIN_AVAIL_GB,
                    ceiling: Optional[float] = rs.DEFAULT_CEILING, nice: int = 10, stall_s: float = 900.0,
                    retries: int = 2) -> Dict[str, dict]:
    """``{tic: {ra, dec, n_raw, raw_sha256}}`` of ``fields`` generated by galaxia-rs on THIS host (``procs`` batch processes,
    the fields split evenly between them; ``ceiling`` is part of what is hashed, like in a stored row).  Raises when a field
    cannot be generated: a fingerprint is never made from a partial result."""
    scratch = os.path.join(work_dir, "scratch")
    os.makedirs(scratch, exist_ok=True)
    tics = sorted(fields)
    n = max(1, min(procs, len(tics)))
    parts = [[(t, fields[t][0], fields[t][1]) for t in tics[i::n]] for i in range(n)]
    free = list(cores) if cores and sys.platform.startswith("linux") else []
    lock = threading.Lock()
    start_lock = threading.Lock()
    gate = MemoryGate(gate_floor_gb(min_avail_gb, install), 5.0, 1.0)
    never = threading.Event()
    expect = install.opencl_device()["name"] if install.backend == "opencl" else None
    out: Dict[str, dict] = {}
    errors: Dict[str, str] = {}
    guard = threading.Lock()

    def emit(item, raw, err, attempts, dt, rss):
        with guard:
            if raw is None:
                errors[item[0]] = err or "failed"
                return
        pk = fb.pack_raw(raw["arrays"])
        with guard:
            out[item[0]] = {"ra": raw["ra"], "dec": raw["dec"], "n_raw": int(len(raw["arrays"]["rad"])), "raw_sha256": pk["sha256"]}

    def one(part):
        with lock:
            cpu = free.pop() if free else None
        try:
            rs.generate_slice(install, part, scratch, retries=retries, nice=nice, stall_s=stall_s, ceiling=ceiling, cpu=cpu,
                              emit=emit, start_lock=start_lock, gate_wait=lambda: gate.wait(never, never),
                              expect_device=expect, log=_log)
        finally:
            if cpu is not None:
                with lock:
                    free.append(cpu)
    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(one, parts))
    if errors:
        raise RuntimeError(f"galaxia-rs could not generate {len(errors)} field(s) for the fingerprint: {errors}")
    return out


def conformance_check(install: "fb.GalaxiaInstall", golden: dict, procs: int, work_dir: str,
                      cores: Optional[Sequence[int]] = None, min_avail_gb: float = MIN_AVAIL_GB) -> dict:
    """Regenerate the golden fields on THIS host and compare their ``raw_sha256``: a host whose build,
    libm or data tree does not reproduce its platform class never takes a chunk.  The golden file is keyed by the
    host class: the CPU class for the C++ engine; for galaxia-rs the full version class (family + source + data +
    platform + backend + precision), so every build / backend / OS proves itself against its own fingerprints."""
    if isinstance(install, rs.RsInstall):
        plat = install.class_key()
        gold = golden.get(plat)
        if not gold:
            raise RuntimeError(f"no golden fingerprints for host class {plat}; create them with 'golden' on a proven host")
        fps = rs_fingerprints(install, {t: (g["ra"], g["dec"]) for t, g in gold.items()}, procs, work_dir, cores,
                              min_avail_gb=min_avail_gb)
        bad = {t: {"expected": gold[t]["raw_sha256"][:16], "got": fps[t]["raw_sha256"][:16]} for t in sorted(gold)
               if fps[t]["raw_sha256"] != gold[t]["raw_sha256"] or fps[t]["n_raw"] != gold[t]["n_raw"]}
        if bad:
            raise RuntimeError(f"CONFORMANCE FAILED on {plat}: {bad}")
        return {"platform": plat, "fields": len(gold)}
    plat = fb.platform_tag()
    gold = golden.get(plat)
    if not gold:
        raise RuntimeError(f"no golden fingerprints for platform {plat}; create them with 'golden' on a proven host")
    scratch = os.path.join(work_dir, "scratch")
    os.makedirs(scratch, exist_ok=True)
    free = list(cores) if cores and sys.platform.startswith("linux") else []
    lock = threading.Lock()

    def one(tic):
        with lock:
            cpu = free.pop() if free else None
        try:
            g = gold[tic]
            return tic, raw_fingerprint(install, tic, g["ra"], g["dec"], cpu=cpu, workdir=scratch)
        finally:
            if cpu is not None:
                with lock:
                    free.append(cpu)
    bad = {}
    with ThreadPoolExecutor(max_workers=max(1, min(procs, len(gold)))) as ex:
        for tic, fp in ex.map(one, sorted(gold)):
            if fp["raw_sha256"] != gold[tic]["raw_sha256"] or fp["n_raw"] != gold[tic]["n_raw"]:
                bad[tic] = {"expected": gold[tic]["raw_sha256"][:16], "got": fp["raw_sha256"][:16]}
    if bad:
        raise RuntimeError(f"CONFORMANCE FAILED on {plat}: {bad}")
    return {"platform": plat, "fields": len(gold)}


# -- the loop -----------------------------------------------------------------

def run_loop(cfg: RunnerConfig) -> int:
    check_procs(cfg.procs, cfg.force_procs)
    lock = RunnerLock(cfg.work_dir, cfg.host)
    try:
        lock.acquire()
    except RunnerLockHeld as exc:
        _log(f"refusing to start a second runner: {exc}")
        return 4
    try:
        return _run_loop(cfg)
    finally:
        lock.release()


def _run_loop(cfg: RunnerConfig) -> int:
    try:
        install = make_install(cfg)
    except (ValueError, rs.RsError) as exc:
        _log(f"cannot set up the {cfg.engine} engine: {exc}")
        return 2
    if not install.available():
        _log("no usable galaxia-rs binary / data dir" if cfg.engine == "rs"
             else f"no usable Galaxia install / provenance-fleet.json at {cfg.galaxia_root}")
        return 2
    try:
        family = fb.raw_family(install.raw_version(cfg.tmag_ceiling))      # also refuses an fp32 / unprovable backend
    except rs.RsError as exc:
        _log(f"REFUSING to run: {exc}")
        return 2
    plat = platform_label(install)
    _log(f"engine {cfg.engine}: raw version {install.raw_version(cfg.tmag_ceiling)}")
    os.makedirs(os.path.join(cfg.work_dir, "scratch"), exist_ok=True)
    marker = os.path.join(cfg.work_dir, "conformance.ok")
    stamp = install.raw_version() + "|" + install.provenance().get("binary_sha256", "")
    if not cfg.skip_conformance and not (os.path.exists(marker) and open(marker).read() == stamp):
        if not cfg.golden_path:
            _log("conformance gate needs --golden (or --skip-conformance for a non-production test)")
            return 2
        _log("conformance gate: regenerating the golden fields ...")
        r = conformance_check(install, load_golden(cfg.golden_path), cfg.procs, cfg.work_dir, cfg.cores,
                              min_avail_gb=cfg.min_avail_gb)
        open(marker, "w").write(stamp)
        _log(f"conformance OK: {r}")
    client = Client(cfg.url, cfg.token)
    chunks_done = 0
    while not stop_requested(cfg):
        got = client.claim(cfg.host, plat, cfg.chunk_ids, cfg.procs, family)
        if got.get("refused"):
            _log(f"coordinator refused the claim: {got['refused']}")
            return 5
        cid = got["chunk_id"]
        if cid is None:
            if cfg.exit_when_empty:
                _log(f"no chunk left to claim ({got.get('remaining')} not done); exiting")
                break
            time.sleep(120)
            continue
        raw_csv = client.chunk(cid)
        if hashlib.sha256(raw_csv).hexdigest() != got["csv_sha256"]:
            _log(f"chunk {cid}: csv sha256 mismatch; releasing")
            client.release(cfg.host, cid, "csv sha mismatch")
            return 3
        items = read_chunk_items(raw_csv)
        shard = os.path.join(cfg.work_dir, f"chunk_{cid:04d}_{cfg.host}.db")
        if not got.get("resumed") and os.path.exists(shard):
            os.remove(shard)                                   # a previous claim of ours was released: start clean
        _log(f"chunk {cid}: {len(items)} TICs ({'resumed' if got.get('resumed') else 'new'}), {cfg.procs} procs")
        stop, abort = threading.Event(), threading.Event()
        hb_done = threading.Event()                           # F6: the heartbeat outlives the compute, through the upload
        uploading = threading.Event()
        counters = {"done": 0, "failed": 0}
        t0 = time.time()

        def beat():
            while not hb_done.wait(HEARTBEAT_S) and not abort.is_set():
                try:
                    r = client.heartbeat(cfg.host, cid, counters["done"], counters["failed"], cfg.procs,
                                         child_peak_rss_mb())
                    if not r.get("ok"):
                        if uploading.is_set():                # the coordinator still accepts a valid shard of a released claim
                            _log(f"chunk {cid}: claim lost during the upload ({r.get('state')}); upload continues")
                        else:
                            _log(f"chunk {cid}: claim lost ({r.get('state')}); aborting")
                            abort.set()
                except Exception as exc:                       # noqa: BLE001 - keep computing through a coordinator outage
                    _log(f"heartbeat failed: {exc}")
                _log(f"chunk {cid}: {counters['done']}/{len(items)} done, {counters['failed']} failed"
                     + (" (uploading)" if uploading.is_set() else ""))
        hb = threading.Thread(target=beat, daemon=True)
        hb.start()
        try:
            try:
                res = run_chunk(cfg, install, cid, items, shard, stop, abort, counters)
            finally:
                stop.set()
            if res["status"] == "aborted":
                _log(f"chunk {cid}: aborted, discarding {shard}")
                if os.path.exists(shard):
                    os.remove(shard)
                continue
            if res["status"] == "stopped":
                _log(f"chunk {cid}: stop requested, releasing (partial kept at {shard})")
                client.release(cfg.host, cid, "stop requested")
                break
            _log(f"chunk {cid}: peak RSS of one field {counters.get('peak_field_rss_mb') or 0:.0f} MB "
                 f"(tic {counters.get('peak_field_tic')}), memory-gate waits {counters.get('mem_gate_waits')}")
            sha = file_sha256(shard)
            uploading.set()
            up = client.upload_shard(cid, cfg.host, shard, sha, res["n_ok"], res["n_fail"], time.time() - t0,
                                     child_peak_rss_mb())
        finally:
            hb_done.set()
            hb.join(timeout=5)
        if up.get("ok"):
            _log(f"chunk {cid}: published ({res['n_ok']} ok, {res['n_fail']} failed)")
            os.remove(shard)
        else:
            _log(f"chunk {cid}: not accepted ({up.get('error')}); shard kept at {shard}")
            if up.get("family_mismatch"):                  # re-claiming the same chunk would only be refused again
                return 6
        chunks_done += 1
        if cfg.max_chunks and chunks_done >= cfg.max_chunks:
            break
    return 0


# ==========================================================================
# Merge
# ==========================================================================

MERGED_EXTRA = """
CREATE TABLE IF NOT EXISTS merged_shards (
  file_sha256 TEXT PRIMARY KEY, name TEXT NOT NULL, chunk_id INTEGER, host TEXT, platform TEXT,
  n_rows INTEGER NOT NULL, n_inserted INTEGER NOT NULL, n_fail INTEGER NOT NULL, merged_at TEXT NOT NULL);
-- F3: problems found while merging are part of the store, not of one run's printout.  An open row
-- (resolved_at IS NULL) makes every later merge report it and exit non-zero.
CREATE TABLE IF NOT EXISTS merge_conflicts (
  tic_id TEXT NOT NULL, raw_version TEXT NOT NULL, stored_sha256 TEXT NOT NULL, new_sha256 TEXT NOT NULL,
  shard TEXT NOT NULL, shard_sha256 TEXT NOT NULL, chunk_id INTEGER, recorded_at TEXT NOT NULL, resolved_at TEXT,
  PRIMARY KEY (tic_id, raw_version, new_sha256, shard_sha256));
CREATE TABLE IF NOT EXISTS merge_corrupt (
  tic_id TEXT NOT NULL, raw_version TEXT NOT NULL, shard TEXT NOT NULL, shard_sha256 TEXT NOT NULL, chunk_id INTEGER,
  error TEXT NOT NULL, recorded_at TEXT NOT NULL, requeued_at TEXT, resolved_at TEXT,
  PRIMARY KEY (tic_id, raw_version, shard_sha256));
"""

_PLAT_RE = re.compile(r"\+(?:plat|be)\.[^+]+")                # galaxia-rs also carries its backend class (+be.cpu-fp64)


def _norm_version(raw_version: str) -> str:
    """raw_version without the platform tag: one population per TIC (decision 1A), so a good row of ANY platform
    class resolves a corrupt record of the same TIC."""
    return _PLAT_RE.sub("", raw_version)


def open_merge_problems(out: sqlite3.Connection) -> dict:
    conf = [dict(zip(("tic", "raw_version", "stored", "new", "shard", "chunk_id", "recorded_at"), r)) for r in out.execute(
        "SELECT tic_id, raw_version, stored_sha256, new_sha256, shard, chunk_id, recorded_at FROM merge_conflicts "
        "WHERE resolved_at IS NULL ORDER BY tic_id, shard")]
    cor = [dict(zip(("tic", "raw_version", "shard", "chunk_id", "error", "recorded_at", "requeued_at"), r)) for r in out.execute(
        "SELECT tic_id, raw_version, shard, chunk_id, error, recorded_at, requeued_at FROM merge_corrupt "
        "WHERE resolved_at IS NULL ORDER BY tic_id, shard")]
    return {"conflicts": conf, "corrupt": cor}


def merge_shards(out_path: str, shard_paths: Iterable[str], *, verify_blobs: bool = True,
                 delete_merged: bool = False) -> dict:
    """Merge shards into one raw store.  Rows are immutable: ``INSERT ... ON CONFLICT DO NOTHING``,
    but a conflicting key with a DIFFERENT ``raw_sha256`` is a reproducibility failure that is
    recorded, reported and never overwritten.  ``verify_blobs`` decodes every new row's blob and
    checks it against its ``raw_sha256`` (corruption in transit / on disk).  ``delete_merged`` removes a
    shard file once ALL of its rows are in the merged store (inserted or identical) and committed: the raw
    store is ~2 TB, so keeping shards and the merged file side by side does not fit the 4.5 TB disk.

    Conflicts and corrupt rows are recorded persistently in the merged store (``merge_conflicts`` /
    ``merge_corrupt``).  A shard that had any is NOT registered in ``merged_shards``, so a rerun processes it
    again; ``report["conflicts"]`` / ``["corrupt"]`` list every OPEN record of the store (not only this run's),
    so the rerun reports them again and the CLI exits non-zero until they are resolved.  A corrupt TIC is
    resolved when a good row for it arrives (see ``requeue_corrupt``)."""
    out = sqlite3.connect(out_path, timeout=120)
    out.executescript(RAW_SCHEMA + MERGED_EXTRA)
    rep = {"shards": 0, "skipped_already_merged": 0, "rows_seen": 0, "inserted": 0, "unchanged": 0,
           "new_conflicts": 0, "new_corrupt": 0, "fail_rows": 0, "refused": []}
    # a merged store holds ONE raw-version family (C++ and galaxia-rs populations are never mixed): recorded in its
    # meta table, or taken from the rows of a store that predates the record
    r_fam = out.execute("SELECT value FROM meta WHERE key='raw_family'").fetchone()
    if r_fam:
        store_family: Optional[str] = json.loads(r_fam[0])
    else:
        fams = {fb.raw_family(r[0]) for r in out.execute("SELECT DISTINCT raw_version FROM fpp_background_raw")}
        store_family = next(iter(fams)) if len(fams) == 1 else None
    open_corrupt = {(t, _norm_version(v)) for t, v in out.execute(
        "SELECT tic_id, raw_version FROM merge_corrupt WHERE resolved_at IS NULL")}

    def resolve(tic, rv):
        if (tic, _norm_version(rv)) in open_corrupt:
            out.execute("UPDATE merge_corrupt SET resolved_at=? WHERE tic_id=? AND resolved_at IS NULL", (_now_iso(), tic))
            open_corrupt.discard((tic, _norm_version(rv)))

    for sp in shard_paths:
        sha = file_sha256(sp)
        if out.execute("SELECT 1 FROM merged_shards WHERE file_sha256=?", (sha,)).fetchone():
            rep["skipped_already_merged"] += 1
            continue
        name = os.path.basename(sp)
        try:
            sfams = shard_families(sp)
        except sqlite3.Error as exc:
            rep["refused"].append({"shard": name, "reason": f"not a readable shard: {exc}"})
            continue
        if len(sfams) > 1 or (store_family is not None and sfams and sfams != {store_family}):
            rep["refused"].append({"shard": name, "families": sorted(sfams), "store_family": store_family,
                                   "reason": "a merged store holds one raw-version family"})
            continue
        if store_family is None and len(sfams) == 1:
            store_family = next(iter(sfams))
            out.execute("INSERT OR REPLACE INTO meta VALUES ('raw_family', ?)", (json.dumps(store_family),))
        sh = sqlite3.connect(f"file:{sp}?mode=ro", uri=True)
        meta = {k: json.loads(v) for k, v in sh.execute("SELECT key, value FROM meta")}
        n_rows = n_ins = n_bad = 0
        cur = sh.execute(f"SELECT {','.join(_RAW_COLS)} FROM fpp_background_raw ORDER BY tic_id")
        while True:
            batch = cur.fetchmany(500)
            if not batch:
                break
            for row in batch:
                d = dict(zip(_RAW_COLS, row))
                n_rows += 1
                old = out.execute("SELECT raw_sha256 FROM fpp_background_raw WHERE tic_id=? AND raw_version=?",
                                  (d["tic_id"], d["raw_version"])).fetchone()
                if old is not None and old[0] == d["raw_sha256"]:
                    rep["unchanged"] += 1
                    resolve(d["tic_id"], d["raw_version"])
                    continue
                if verify_blobs:
                    try:
                        fb.unpack_raw(d["raw_meta"], d["raw_blob"], d["raw_sha256"])
                    except Exception as exc:            # noqa: BLE001
                        n_bad += 1
                        cur_ins = out.execute(
                            "INSERT OR IGNORE INTO merge_corrupt(tic_id, raw_version, shard, shard_sha256, chunk_id, error, recorded_at)"
                            " VALUES (?,?,?,?,?,?,?)",
                            (d["tic_id"], d["raw_version"], name, sha, d["chunk_id"], str(exc)[:300], _now_iso()))
                        rep["new_corrupt"] += cur_ins.rowcount
                        continue
                if old is not None:
                    n_bad += 1
                    cur_ins = out.execute(
                        "INSERT OR IGNORE INTO merge_conflicts(tic_id, raw_version, stored_sha256, new_sha256, shard, shard_sha256,"
                        " chunk_id, recorded_at) VALUES (?,?,?,?,?,?,?,?)",
                        (d["tic_id"], d["raw_version"], old[0], d["raw_sha256"], name, sha, d["chunk_id"], _now_iso()))
                    rep["new_conflicts"] += cur_ins.rowcount
                    continue
                out.execute(f"INSERT INTO fpp_background_raw({','.join(_RAW_COLS)}) VALUES ({','.join('?' * len(_RAW_COLS))}) "
                            "ON CONFLICT(tic_id, raw_version) DO NOTHING", row)
                n_ins += 1
                resolve(d["tic_id"], d["raw_version"])
        fails = sh.execute("SELECT tic_id, chunk_id, host, attempts, error, created_at FROM fpp_background_fail").fetchall()
        for f in fails:
            out.execute("INSERT INTO fpp_background_fail VALUES (?,?,?,?,?,?) ON CONFLICT DO NOTHING", f)
        rep["fail_rows"] += len(fails)
        if n_bad == 0:                                   # only a clean shard counts as merged (F3)
            out.execute("INSERT INTO merged_shards VALUES (?,?,?,?,?,?,?,?,?)",
                        (sha, name, meta.get("chunk_id"), meta.get("host"), meta.get("platform"),
                         n_rows, n_ins, len(fails), _now_iso()))
        out.commit()
        sh.close()
        if delete_merged and n_bad == 0:
            os.remove(sp)
            rep["deleted"] = rep.get("deleted", 0) + 1
        rep["shards"] += 1
        rep["rows_seen"] += n_rows
        rep["inserted"] += n_ins
    rep.update(open_merge_problems(out))
    out.close()
    return rep


def _make_retry_chunks(c: sqlite3.Connection, coord: "Coordinator", tics: Sequence[str],
                       coords: Dict[str, Tuple[float, float]], source_chunks: Sequence[int] = ()) -> List[int]:
    """Create 'retry' chunks for ``tics`` inside the open queue transaction ``c``; they take the best (lowest)
    tier of the chunks the TICs came from, so a retried strong candidate is not queued behind the rest of scope."""
    tier = 4
    if source_chunks:
        q = ",".join("?" * len(source_chunks))
        tier = c.execute(f"SELECT MIN(COALESCE(tier, 4)) FROM chunks WHERE id IN ({q})", list(source_chunks)).fetchone()[0]
        tier = 4 if tier is None else tier
    new_id = (c.execute("SELECT COALESCE(MAX(id),0) FROM chunks").fetchone()[0]) + 1
    made = []
    for i in range(0, len(tics), CHUNK_TICS):
        part = list(tics[i:i + CHUNK_TICS])
        ra = np.array([coords[t][0] for t in part])
        dec = np.array([coords[t][1] for t in part])
        cost = float(field_cost(_gal_b_vec(ra, dec)).sum())
        sha = _write_chunk_csv(coord.chunk_csv_path(new_id), part, ra, dec)
        c.execute("INSERT INTO chunks(id, kind, n_tics, cost, csv_sha256, tier) VALUES (?,?,?,?,?,?)",
                  (new_id, "retry", len(part), cost, sha, tier))
        made.append(new_id)
        new_id += 1
    return made


def requeue_corrupt(coord: "Coordinator", merged_db: str, max_rounds: int = 3) -> dict:
    """F3: TICs whose row was corrupt in a shard (open ``merge_corrupt`` records not yet requeued) go into new
    'retry' chunks of the queue.  A TIC that was corrupt in ``max_rounds`` different shards is listed in
    ``permanent_corrupt.csv`` instead.  Idempotent: a record is requeued once (``requeued_at``)."""
    m = sqlite3.connect(merged_db, timeout=120)
    m.executescript(RAW_SCHEMA + MERGED_EXTRA)
    c = coord._tx()
    try:
        recs = m.execute("SELECT tic_id, raw_version, chunk_id, shard_sha256 FROM merge_corrupt "
                         "WHERE resolved_at IS NULL AND requeued_at IS NULL").fetchall()
        tics = sorted({r[0] for r in recs})
        stored = {t for t in tics if m.execute("SELECT 1 FROM fpp_background_raw WHERE tic_id=? LIMIT 1", (t,)).fetchone()}
        rounds = {t: m.execute("SELECT COUNT(DISTINCT shard_sha256) FROM merge_corrupt WHERE tic_id=?", (t,)).fetchone()[0]
                  for t in tics}
        coords: Dict[str, Tuple[float, float]] = {}
        for cid in sorted({r[2] for r in recs if r[2] is not None}):
            try:
                for it in read_chunk_items(open(coord.chunk_csv_path(cid), "rb").read()):
                    coords[it[0]] = (it[1], it[2])
            except OSError:
                pass
        retry = [t for t in tics if t not in stored and rounds[t] < max_rounds and t in coords]
        perm = [t for t in tics if t not in stored and (rounds[t] >= max_rounds or t not in coords)]
        made = _make_retry_chunks(c, coord, retry, coords, sorted({r[2] for r in recs if r[2] is not None})) if retry else []
        c.execute("COMMIT")
    except Exception:
        c.execute("ROLLBACK")
        raise
    finally:
        c.close()
    now = _now_iso()
    for t in retry:
        m.execute("UPDATE merge_corrupt SET requeued_at=? WHERE tic_id=? AND resolved_at IS NULL", (now, t))
    for t in stored:                                      # a good row exists meanwhile: nothing to retry
        m.execute("UPDATE merge_corrupt SET resolved_at=? WHERE tic_id=? AND resolved_at IS NULL", (now, t))
    m.commit()
    m.close()
    if perm:
        with open(os.path.join(coord.queue_dir, "permanent_corrupt.csv"), "w") as f:
            f.write("tic\n" + "".join(f"{t}\n" for t in perm))
    return {"requeued_tics": len(retry), "new_chunks": made, "permanent": len(perm)}


# ==========================================================================
# Serving from the raw store
# ==========================================================================

class RawStore:
    """Read side of a (merged or per-chunk) raw store."""

    def __init__(self, path: str):
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row

    def versions(self, tic) -> List[str]:
        return [r[0] for r in self.db.execute("SELECT raw_version FROM fpp_background_raw WHERE tic_id=? ORDER BY raw_version",
                                              (_key(tic),))]

    def get_row(self, tic, raw_version: Optional[str] = None) -> Optional[sqlite3.Row]:
        if raw_version is None:
            vs = self.versions(tic)
            if len(vs) != 1:
                return None if not vs else self._ambiguous(tic, vs)
            raw_version = vs[0]
        return self.db.execute("SELECT * FROM fpp_background_raw WHERE tic_id=? AND raw_version=?",
                               (_key(tic), raw_version)).fetchone()

    @staticmethod
    def _ambiguous(tic, vs):
        raise ValueError(f"TIC {tic} has several raw versions {vs}; pass raw_version explicitly")

    def raw(self, tic, raw_version: Optional[str] = None) -> Optional[Tuple[sqlite3.Row, Dict[str, np.ndarray]]]:
        row = self.get_row(tic, raw_version)
        if row is None:
            return None
        return row, fb.unpack_raw(row["raw_meta"], row["raw_blob"], row["raw_sha256"])

    def document(self, tic, mapping: str = fb.MAPPING_VERSION, raw_version: Optional[str] = None) -> Optional[dict]:
        """The served step-5 document for ``tic`` under ``mapping`` (same dict as ``fb.serve_document``)."""
        got = self.raw(tic, raw_version)
        if got is None:
            return None
        row, arrays = got
        return fb.serve_document(arrays, tic_id=row["tic_id"], ra=row["ra"], dec=row["dec"], gal_l=row["gal_l"],
                                 gal_b=row["gal_b"], seed=row["seed"], raw_version=row["raw_version"], mapping=mapping)

    def counts(self, tic, mapping: str, raw_version: Optional[str] = None) -> Optional[dict]:
        row = self.get_row(tic, raw_version)
        return None if row is None else json.loads(row["counts_json"])[mapping]

    def tics(self) -> Iterable[str]:
        for r in self.db.execute("SELECT DISTINCT tic_id FROM fpp_background_raw ORDER BY tic_id"):
            yield r[0]

    def materialize(self, store: "fb.FppBackgroundStore", mapping: str = fb.MAPPING_VERSION,
                    tics: Optional[Iterable] = None, raw_version: Optional[str] = None) -> dict:
        """Write served documents into a legacy ``fpp_background`` table (what the route and the step-6 PG
        import consume).  Immutable rows: an existing row with another hash raises ``ReproducibilityError``."""
        res = {"inserted": 0, "unchanged": 0}
        for t in (tics if tics is not None else self.tics()):
            doc = self.document(t, mapping, raw_version)
            if doc is None:
                continue
            c = self.counts(t, mapping, raw_version)
            cx = fb.crosscheck_from_counts(c["bins"], c["n"], None, None)      # Gaia side: see galaxia_gaia_counts
            r = store.put(t, doc["header"], doc["payload"], doc["sha256"], cx)
            res[r] += 1
        return res


# ==========================================================================
# Operator helpers
# ==========================================================================

def glob_shards(shards_dir: str, chunk_id: int) -> List[str]:
    d = os.path.join(shards_dir, f"{int(chunk_id):04d}")
    return [os.path.join(d, n) for n in os.listdir(d) if n.endswith(".db")] if os.path.isdir(d) else []


def make_golden(install: "fb.GalaxiaInstall", fields: Dict[str, Tuple[float, float]], procs: int, work_dir: str,
                cores: Optional[Sequence[int]] = None, min_avail_gb: float = MIN_AVAIL_GB) -> dict:
    """Fingerprints (``raw_sha256``, ``n_raw``) of the conformance fields on THIS host/platform."""
    if isinstance(install, rs.RsInstall):
        return rs_fingerprints(install, fields, procs, work_dir, cores, min_avail_gb=min_avail_gb)
    scratch = os.path.join(work_dir, "scratch")
    os.makedirs(scratch, exist_ok=True)
    free = list(cores) if cores and sys.platform.startswith("linux") else []
    lock = threading.Lock()

    def one(tic):
        with lock:
            cpu = free.pop() if free else None
        try:
            ra, dec = fields[tic]
            return tic, raw_fingerprint(install, tic, ra, dec, cpu=cpu, workdir=scratch)
        finally:
            if cpu is not None:
                with lock:
                    free.append(cpu)
    with ThreadPoolExecutor(max_workers=max(1, procs)) as ex:
        return {tic: fp for tic, fp in ex.map(one, sorted(fields))}


def verify_raw_sample(raw_db: str, install: "fb.GalaxiaInstall", n: int, cores: Sequence[int], seed: int = 1,
                      host_platform: Optional[str] = None, min_avail_gb: float = MIN_AVAIL_GB) -> dict:
    """Acceptance check of a stored raw store: regenerate ``n`` random stored fields of THIS platform class
    (ideally on a different host than the one that stored them) and compare ``raw_sha256``.  Fields stored
    under another platform tag are skipped, never compared."""
    import random
    if isinstance(install, rs.RsInstall):
        return _verify_raw_sample_rs(raw_db, install, n, cores, seed, min_avail_gb)
    plat = host_platform or fb.platform_tag()
    db = sqlite3.connect(f"file:{raw_db}?mode=ro", uri=True)
    keys = [r for r in db.execute("SELECT tic_id, raw_version, ra, dec, raw_sha256, host FROM fpp_background_raw "
                                  "WHERE raw_version LIKE ?", (f"%+plat.{plat}+%",))]
    random.Random(seed).shuffle(keys)
    keys = keys[:n]
    free = list(cores) if sys.platform.startswith("linux") else []
    lock = threading.Lock()
    res = {"platform": plat, "checked": 0, "ok": 0, "mismatches": [], "failed": {}}

    def one(k):
        tic, rv, ra, dec, sha, host = k
        with lock:
            cpu = free.pop() if free else None
        try:
            fp = raw_fingerprint(install, tic, ra, dec, cpu=cpu)
            return k, fp, None
        except Exception as exc:                          # noqa: BLE001
            return k, None, f"{type(exc).__name__}: {exc}"[:300]
        finally:
            if cpu is not None:
                with lock:
                    free.append(cpu)
    with ThreadPoolExecutor(max_workers=max(1, len(cores))) as ex:
        for k, fp, err in ex.map(one, keys):
            if err:
                res["failed"][k[0]] = err
                continue
            res["checked"] += 1
            if fp["raw_sha256"] == k[4]:
                res["ok"] += 1
            else:
                res["mismatches"].append({"tic": k[0], "stored": k[4][:16], "regenerated": fp["raw_sha256"][:16], "stored_by": k[5]})
    return res


def _verify_raw_sample_rs(raw_db: str, install: "rs.RsInstall", n: int, cores: Sequence[int], seed: int,
                          min_avail_gb: float = MIN_AVAIL_GB) -> dict:
    """verify-sample for galaxia-rs: only rows of THIS host class (the version up to and including ``be.``) are regenerated,
    each with the ceiling written in its own raw_version."""
    import random
    import tempfile
    cls = install.class_key()
    db = sqlite3.connect(f"file:{raw_db}?mode=ro", uri=True)
    keys = [r for r in db.execute("SELECT tic_id, raw_version, ra, dec, raw_sha256, host FROM fpp_background_raw "
                                  "WHERE substr(raw_version, 1, ?) = ? ORDER BY tic_id", (len(cls) + 1, cls + "+"))]
    random.Random(seed).shuffle(keys)
    keys = keys[:n]
    res = {"platform": cls, "checked": 0, "ok": 0, "mismatches": [], "failed": {}}
    by_ceiling: Dict[Optional[float], list] = {}
    for k in keys:
        m = re.search(r"\+C([0-9.]+)$", k[1])
        by_ceiling.setdefault(float(m.group(1)) if m else None, []).append(k)
    work = tempfile.mkdtemp(prefix="rs_verify_")
    try:
        for ceiling, ks in by_ceiling.items():
            fields = {k[0]: (k[2], k[3]) for k in ks}
            try:
                fps = rs_fingerprints(install, fields, len(cores), work, cores, ceiling=ceiling, min_avail_gb=min_avail_gb)
            except RuntimeError as exc:
                for k in ks:
                    res["failed"][k[0]] = str(exc)[:300]
                continue
            for k in ks:
                res["checked"] += 1
                if fps[k[0]]["raw_sha256"] == k[4]:
                    res["ok"] += 1
                else:
                    res["mismatches"].append({"tic": k[0], "stored": k[4][:16], "regenerated": fps[k[0]]["raw_sha256"][:16],
                                              "stored_by": k[5]})
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return res


def requeue_failed(coord: Coordinator, max_rounds: int = 3) -> dict:
    """Put the TICs that failed in finished chunks into new 'retry' chunks.  A TIC that already failed in
    ``max_rounds`` different chunks is listed as permanently failed (``permanent_failures.csv`` in the
    queue dir) instead of being queued again."""
    c = coord._tx()
    try:
        done = c.execute("SELECT id FROM chunks WHERE state='done' AND n_fail > 0").fetchall()
        coords: Dict[str, Tuple[float, float]] = {}
        fails: Dict[str, int] = {}
        ok_tics = set()
        for r in done:
            cid = r["id"]
            for dbp in sorted(glob_shards(coord.shards_dir, cid)):
                sh = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True)
                for (t,) in sh.execute("SELECT tic_id FROM fpp_background_fail WHERE chunk_id=?", (cid,)):
                    fails[t] = fails.get(t, 0) + 1
                for (t,) in sh.execute("SELECT tic_id FROM fpp_background_raw"):
                    ok_tics.add(t)
                sh.close()
            for it in read_chunk_items(open(coord.chunk_csv_path(cid), "rb").read()):
                coords[it[0]] = (it[1], it[2])
        retry = sorted(t for t in fails if t not in ok_tics and fails[t] < max_rounds and t in coords)
        perm = sorted(t for t in fails if t not in ok_tics and fails[t] >= max_rounds)
        made = _make_retry_chunks(c, coord, retry, coords, [r["id"] for r in done]) if retry else []
        c.execute("COMMIT")
    except Exception:
        c.execute("ROLLBACK")
        raise
    finally:
        c.close()
    with open(os.path.join(coord.queue_dir, "permanent_failures.csv"), "w") as f:
        f.write("tic\n" + "".join(f"{t}\n" for t in perm))
    return {"requeued_tics": len(retry), "new_chunks": made, "permanent_failures": len(perm)}
