"""Serve FPP background populations from the materialized galaxia-rs part files (read-only).

Design: progress/impl-fpp-galaxia-store-ingest.md, option (b).  The materializer
(``scripts/galaxia_materialize_full.py``, branch galaxia-materialize-prerun) writes the served documents
as N immutable SQLite PART files ``part_NNNN.db`` in the ``fpp_background`` table format
(``fb.SQLITE_DDL`` + ``created_at``), plus ``manifest.json`` (per part: ``file_sha256``, ``n_tics``,
``run_id``; global: ``n_parts``, ``n_tics``, ``population_versions`` counts).  Parts are assigned
round-robin over fleet chunks, so a TIC's part cannot be derived from its id: a TIC -> part INDEX is needed.

Two steps:

* :func:`build_index` (offline, explicit CLI ``scripts/fpp_galaxia_store_index.py``) verifies the store
  against its manifest (complete part set of ONE run, every part's full-file sha256, row counts,
  population_version tallies, every version inside the served FAMILY) and writes a small local SQLite
  index (TIC, population_version, part, blob_sha256 + the light columns the readiness gate reads).  It
  REFUSES (writes no index) when any TIC has more than one row in the family, in one part or across
  parts (review 2026-10-25 fpp-integration-r2 note 4): the materializer guarantees one row per TIC, so a
  second row means a mixed/corrupt store, and choosing one would be an arbitrary pick.
* :class:`GalaxiaPartStore` (server, read-only) opens the index, checks that the manifest still has the
  sha256 the index was built from and that every part still has the size/mtime it had when its sha256
  was verified, and serves one document per lookup.  Each served document is re-verified: the part
  row's ``blob_sha256`` equals the index (i.e. the value inside the manifest-verified part), the
  uncompressed payload hashes to it, and the payload header names the same TIC and population_version,
  which must be one of the manifest's versions and inside the family.  Any failure raises
  :class:`GalaxiaStoreError`; it is never answered from another population (no fallback to the
  ``fpp_background`` table).

Switch: ``EXONIT_FPP_GALAXIA_STORE_INDEX`` (path of the index).  Unset = current behaviour (the
``fpp_background`` table).  Set: the store is the ONLY population source and
``EXONIT_FPP_POPULATION_FAMILY`` must be set and equal the family the index was built for.
``EXONIT_FPP_GALAXIA_STORE_DIR`` optionally overrides the parts directory recorded in the index (e.g. the
same files seen through another mount point; size/mtime/manifest sha must still match).
``EXONIT_FPP_GALAXIA_REVALIDATE_S`` (default 30; <=0 = on every call) is how often a cached store is
re-checked (manifest sha256 + size/mtime of every part, stat-only): a mount that disappears after the
server started closes the store (:class:`GalaxiaStoreConfigError`, 503 / population_missing, readiness
gate closed) instead of leaving it "ready" until a TIC happens to need the lost part.
``GET /api/health/fpp_population`` (:func:`health`) runs the full live check.
``EXONIT_FPP_GALAXIA_PROBE_TICS`` (optional, comma-separated) names TICs that must resolve at open
(:meth:`GalaxiaPartStore.consistency_check`); the open also compares manifest and index ``n_parts`` /
``n_tics``.  Any mismatch refuses the whole store (no fallback), same as every other open failure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import zlib
from typing import Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

STORE_ENV = "EXONIT_FPP_GALAXIA_STORE_INDEX"
PARTS_DIR_ENV = "EXONIT_FPP_GALAXIA_STORE_DIR"
#: Comma-separated TIC ids that MUST resolve (verified end to end) for the store to be served.  Unset =
#: one TIC sampled from the index itself.  Deploy value for the v0.2.2 store: 143526444 (a multi-version TIC).
PROBE_TICS_ENV = "EXONIT_FPP_GALAXIA_PROBE_TICS"
#: How often (seconds) a cached store re-validates its manifest sha256 and part size/mtime; <=0 = every call.
REVALIDATE_ENV = "EXONIT_FPP_GALAXIA_REVALIDATE_S"
DEFAULT_REVALIDATE_S = 30.0
INDEX_SCHEMA_VERSION = "1"
STORE_NAME = "galaxia-parts"
PART_TABLE_COLS = ("tic_id", "population_version", "n_comp", "source", "blob_sha256", "blob",
                   "crosscheck_json")
#: A failed open is retried after this many seconds (e.g. the NFS mount came back), not on every call.
REOPEN_AFTER_S = 60.0

INDEX_DDL = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE parts (part INTEGER PRIMARY KEY, file_name TEXT NOT NULL, size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL, file_sha256 TEXT NOT NULL, n_tics INTEGER NOT NULL);
CREATE TABLE tic_rows (tic_id TEXT NOT NULL, population_version TEXT NOT NULL, part INTEGER NOT NULL,
                       blob_sha256 TEXT NOT NULL, n_comp INTEGER NOT NULL, source TEXT NOT NULL,
                       crosscheck_json TEXT, PRIMARY KEY (tic_id, population_version));
"""


class GalaxiaStoreError(RuntimeError):
    """The materialized store (or one document of it) failed a verification: refuse, never fall back."""


class GalaxiaStoreSwitchOnError(GalaxiaStoreError):
    """A writer of the ``fpp_background`` table was started while the store switch is on: the table is
    ignored by every consumer in that mode, so the write would be invisible (store-ingest review note 4)."""


class GalaxiaStoreConfigError(GalaxiaStoreError):
    """The store switch is on but the store cannot be used (family unset/mismatch, index or parts
    unreadable or changed).  Every consumer refuses (population_missing / 503); logged once per message."""


def _key(tic_id) -> str:
    return str(int(tic_id))


def _family_of(version: str) -> str:
    from exonit.fpp_background import family_of
    return family_of(version)


def _ro(path: str, immutable: bool = True) -> sqlite3.Connection:
    """Strictly read-only SQLite connection.  ``immutable=1`` (no locking, no change detection) is safe
    because the files are verified immutable artefacts and every read is guarded by a size/mtime check."""
    uri = f"file:{os.path.abspath(path)}?mode=ro" + ("&immutable=1" if immutable else "")
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _part_file(idx: int) -> str:
    return f"part_{idx:04d}.db"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------------------------------------
# Offline: verify the store against its manifest and build the TIC index
# --------------------------------------------------------------------------------------------------

def _load_manifest(parts_dir: str) -> tuple[dict, str]:
    path = os.path.join(parts_dir, "manifest.json")
    if not os.path.exists(path):
        raise GalaxiaStoreError(f"no manifest.json in {parts_dir}: the materializer has not finished "
                                "(--merge-manifest writes it last)")
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw), _sha256_bytes(raw)


def _check_manifest(man: dict, family: str) -> List[dict]:
    """Structural checks of manifest.json; returns its parts sorted by index."""
    n_parts = man.get("n_parts")
    parts = man.get("parts")
    if not isinstance(n_parts, int) or n_parts < 1 or not isinstance(parts, list):
        raise GalaxiaStoreError("manifest has no valid n_parts / parts")
    by_idx = {p.get("part"): p for p in parts}
    if sorted(by_idx) != list(range(n_parts)) or len(parts) != n_parts:
        raise GalaxiaStoreError(f"manifest part set is not 0..{n_parts - 1}: {sorted(by_idx)}")
    run_id = man.get("run_id")
    for p in parts:
        if p.get("status") != "complete":
            raise GalaxiaStoreError(f"manifest part {p.get('part')} status {p.get('status')!r} != 'complete'")
        if not p.get("file_sha256"):
            raise GalaxiaStoreError(f"manifest part {p.get('part')} has no file_sha256")
        if run_id is None or p.get("run_id") != run_id:
            raise GalaxiaStoreError(f"manifest part {p.get('part')} run_id {p.get('run_id')} != {run_id}")
    if sum(int(p.get("n_tics", -1)) for p in parts) != man.get("n_tics"):
        raise GalaxiaStoreError("manifest n_tics != sum of the part n_tics")
    versions = man.get("population_versions") or {}
    if not versions:
        raise GalaxiaStoreError("manifest lists no population_versions")
    foreign = sorted(v for v in versions if _family_of(v) != family)
    if foreign:
        raise GalaxiaStoreError(f"manifest holds population_versions outside family {family}: {foreign[:5]}")
    if sum(versions.values()) != man["n_tics"]:
        raise GalaxiaStoreError("manifest population_versions counts != n_tics")
    return [by_idx[i] for i in range(n_parts)]


def _verify_document(tic: str, version: str, sha: str, blob: bytes) -> None:
    payload = zlib.decompress(bytes(blob))
    if _sha256_bytes(payload) != sha:
        raise GalaxiaStoreError(f"TIC {tic} {version}: payload sha256 != blob_sha256 {sha}")
    header = json.loads(payload)["header"]
    if _key(header.get("tic_id")) != tic or header.get("population_version") != version:
        raise GalaxiaStoreError(f"TIC {tic} {version}: payload header names TIC {header.get('tic_id')} "
                                f"{header.get('population_version')}")


def build_index(parts_dir: str, index_path: str, family: str, *, deep: bool = False,
                file_sha256: Optional[Callable[[str], str]] = None,
                progress: Optional[Callable[[str], None]] = None) -> dict:
    """Verify the materialized store in ``parts_dir`` against its manifest and write the TIC index at
    ``index_path`` (atomically: ``.tmp`` then rename; an existing index is never overwritten).

    Refuses (raises :class:`GalaxiaStoreError`, no index written) on: missing/incomplete manifest, a part
    whose full-file sha256 differs from the manifest, a row count or version tally that disagrees with the
    manifest, any population_version outside ``family``, and ANY TIC with more than one row in the family
    (one part or across parts).  ``deep=True`` also decompresses and re-hashes every document (reads the
    whole store a second time; the default relies on the manifest file sha256 plus the per-serve check)."""
    from exonit.fpp_background import family_of
    if not family or family_of(family) != family:
        raise GalaxiaStoreError(f"{family!r} is a full population_version, not a family")
    if os.path.exists(index_path):
        raise GalaxiaStoreError(f"{index_path} exists; refusing to overwrite an index (remove it first)")
    file_sha256 = file_sha256 or _file_sha256
    say = progress or (lambda msg: logger.info("%s", msg))
    parts_dir = os.path.abspath(parts_dir)
    man, man_sha = _load_manifest(parts_dir)
    parts = _check_manifest(man, family)
    tmp = index_path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    idx = sqlite3.connect(tmp)
    try:
        idx.executescript(INDEX_DDL)
        tally: Dict[str, int] = {}
        duplicates: List[tuple] = []
        for p in parts:
            i = p["part"]
            path = os.path.join(parts_dir, _part_file(i))
            if not os.path.exists(path):
                raise GalaxiaStoreError(f"part {i}: {path} missing")
            st_before = os.stat(path)
            say(f"part {i}: sha256 of {path} ({st_before.st_size / 1e9:.1f} GB)")
            actual = file_sha256(path)
            if actual != p["file_sha256"]:
                raise GalaxiaStoreError(f"part {i}: file sha256 {actual} != manifest {p['file_sha256']}")
            conn = _ro(path)
            try:
                cols = {r["name"] for r in conn.execute("PRAGMA table_info(fpp_background)")}
                missing = sorted(set(PART_TABLE_COLS) - cols)
                if missing:
                    raise GalaxiaStoreError(f"part {i}: fpp_background lacks columns {missing}")
                sel = ("tic_id, population_version, blob_sha256, n_comp, source, crosscheck_json"
                       + (", blob" if deep else ""))
                n = 0
                for r in conn.execute(f"SELECT {sel} FROM fpp_background"):
                    tic, ver = _key(r["tic_id"]), r["population_version"]
                    if str(r["tic_id"]) != tic:
                        raise GalaxiaStoreError(f"part {i}: non-canonical tic_id {r['tic_id']!r}")
                    if family_of(ver) != family:
                        raise GalaxiaStoreError(f"part {i}: TIC {tic} version {ver} outside family {family}")
                    if deep:
                        _verify_document(tic, ver, r["blob_sha256"], r["blob"])
                    try:
                        idx.execute("INSERT INTO tic_rows VALUES (?,?,?,?,?,?,?)",
                                    (tic, ver, i, r["blob_sha256"], r["n_comp"], r["source"],
                                     r["crosscheck_json"]))
                    except sqlite3.IntegrityError:
                        duplicates.append((tic, ver, i))
                    tally[ver] = tally.get(ver, 0) + 1
                    n += 1
            finally:
                conn.close()
            st_after = os.stat(path)
            if (st_after.st_size, st_after.st_mtime_ns) != (st_before.st_size, st_before.st_mtime_ns):
                raise GalaxiaStoreError(f"part {i} changed while it was being verified")
            if n != p["n_tics"]:
                raise GalaxiaStoreError(f"part {i}: {n} rows != manifest n_tics {p['n_tics']}")
            idx.execute("INSERT INTO parts VALUES (?,?,?,?,?,?)",
                        (i, _part_file(i), st_after.st_size, st_after.st_mtime_ns, actual, n))
        multi = [r[0] for r in idx.execute(
            "SELECT tic_id FROM tic_rows GROUP BY tic_id HAVING COUNT(*) > 1 ORDER BY tic_id")]
        ambiguous = sorted(set(multi) | {d[0] for d in duplicates}, key=int)
        if ambiguous:
            raise GalaxiaStoreError(
                f"{len(ambiguous)} TIC(s) have more than one row in family {family} "
                f"(first: {ambiguous[:20]}); refusing to index the store (never pick one)")
        if tally != man["population_versions"]:
            raise GalaxiaStoreError(f"population_version tally {tally} != manifest {man['population_versions']}")
        meta = {"schema_version": INDEX_SCHEMA_VERSION, "family": family, "parts_dir": parts_dir,
                "manifest_sha256": man_sha, "run_id": man["run_id"], "n_parts": str(man["n_parts"]),
                "n_tics": str(man["n_tics"]), "population_versions": json.dumps(sorted(tally)),
                "deep_verified": "1" if deep else "0",
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        idx.executemany("INSERT INTO meta VALUES (?,?)", sorted(meta.items()))
        idx.commit()
    except BaseException:
        idx.close()
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    idx.close()
    os.replace(tmp, index_path)
    say(f"index {index_path}: {man['n_tics']} TICs, {man['n_parts']} parts, family {family}")
    return {"index": index_path, "n_tics": man["n_tics"], "n_parts": man["n_parts"],
            "run_id": man["run_id"], "manifest_sha256": man_sha, "deep_verified": deep,
            "population_versions": tally}


def _file_sha256(path: str) -> str:
    from exonit.fpp_background import file_sha256
    return file_sha256(path)


# --------------------------------------------------------------------------------------------------
# Runtime: read-only store used by the server
# --------------------------------------------------------------------------------------------------

class GalaxiaPartStore:
    """Read-only, verified view of a materialized store through its index.  Thread-safe: one SQLite
    connection per (thread, file), opened read-only."""

    def __init__(self, index_path: str, family: str, parts_dir: Optional[str] = None):
        self.index_path = os.path.abspath(index_path)
        if not os.path.exists(self.index_path):
            raise GalaxiaStoreConfigError(f"galaxia store index {self.index_path} not found")
        conn = _ro(self.index_path)
        try:
            meta = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM meta")}
            self._parts = {r["part"]: dict(r) for r in conn.execute("SELECT * FROM parts")}
        except sqlite3.Error as exc:
            raise GalaxiaStoreConfigError(f"galaxia store index {self.index_path} unreadable: {exc}") from exc
        finally:
            conn.close()
        if meta.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise GalaxiaStoreConfigError(f"index schema {meta.get('schema_version')} != {INDEX_SCHEMA_VERSION}")
        if meta.get("family") != family:
            raise GalaxiaStoreConfigError(
                f"index built for family {meta.get('family')!r}, served family is {family!r}")
        self.family = family
        self.meta = meta
        self.parts_dir = os.path.abspath(parts_dir or meta["parts_dir"])
        self.versions = frozenset(json.loads(meta["population_versions"]))
        if len(self._parts) != int(meta["n_parts"]):
            raise GalaxiaStoreConfigError("index part table incomplete")
        try:
            self._manifest, man_sha = _load_manifest(self.parts_dir)
        except (OSError, ValueError, GalaxiaStoreError) as exc:
            raise GalaxiaStoreConfigError(f"manifest unreadable in {self.parts_dir}: {exc}") from exc
        if man_sha != meta["manifest_sha256"]:
            raise GalaxiaStoreConfigError(
                f"manifest sha256 {man_sha} != {meta['manifest_sha256']} the index was built from; rebuild the index")
        for i in sorted(self._parts):
            self._check_part(i, exc_type=GalaxiaStoreConfigError)
        self._local = threading.local()
        self.validated_at = time.monotonic()

    def revalidate(self) -> None:
        """Stat-only re-check of what ``__init__`` proved: the manifest still has the sha256 the index was
        built from and every part still has its verified size/mtime.  Raises
        :class:`GalaxiaStoreConfigError` (e.g. the NFS mount is gone)."""
        try:
            _man, man_sha = _load_manifest(self.parts_dir)
        except (OSError, ValueError, GalaxiaStoreError) as exc:
            raise GalaxiaStoreConfigError(f"manifest unreadable in {self.parts_dir}: {exc}") from exc
        if man_sha != self.meta["manifest_sha256"]:
            raise GalaxiaStoreConfigError(
                f"manifest sha256 {man_sha} != {self.meta['manifest_sha256']} the index was built from")
        for i in sorted(self._parts):
            self._check_part(i, exc_type=GalaxiaStoreConfigError)
        self.validated_at = time.monotonic()

    # -- consistency ------------------------------------------------------------------------------
    def consistency_check(self, probe_tics: Optional[Iterable] = None) -> dict:
        """Read-only end-to-end consistency of store dir, index and family; raises
        :class:`GalaxiaStoreConfigError` (the server then refuses every population lookup).

        Beyond what ``__init__`` already proves (family, schema, manifest sha256, size/mtime of every part):
        the manifest's ``n_parts`` / ``n_tics`` equal the index meta's and the index ``parts`` table, every
        manifest part file exists in the parts dir, and each probe TIC resolves to a document that passes the
        full per-document verification (``get_by_family``).  *probe_tics* None = ``PROBE_TICS_ENV``, else one
        TIC sampled from the index (an empty index is itself an error)."""
        man = self._manifest
        n_parts_meta, n_tics_meta = int(self.meta["n_parts"]), int(self.meta["n_tics"])
        if man.get("n_parts") != n_parts_meta or len(self._parts) != n_parts_meta:
            raise GalaxiaStoreConfigError(
                f"n_parts disagree: manifest {man.get('n_parts')}, index meta {n_parts_meta}, "
                f"index parts table {len(self._parts)}")
        if man.get("n_tics") != n_tics_meta:
            raise GalaxiaStoreConfigError(f"n_tics disagree: manifest {man.get('n_tics')}, index meta {n_tics_meta}")
        for i in sorted(self._parts):
            if not os.path.isfile(self._part_path(i)):
                raise GalaxiaStoreConfigError(f"part {i} file {self._part_path(i)} missing from {self.parts_dir}")
        if probe_tics is None:
            raw = os.environ.get(PROBE_TICS_ENV, "").strip()
            probe_tics = [t for t in (x.strip() for x in raw.split(",")) if t]
        probe_tics = list(probe_tics)
        if not probe_tics:
            try:
                row = self._index().execute("SELECT tic_id FROM tic_rows LIMIT 1").fetchone()
            except sqlite3.Error as exc:
                raise GalaxiaStoreConfigError(f"index tic_rows unreadable: {exc}") from exc
            if row is None:
                raise GalaxiaStoreConfigError("index has no TIC rows")
            probe_tics = [row["tic_id"]]
        for tic in probe_tics:
            try:
                doc = self.get_by_family(tic, self.family)
            except GalaxiaStoreError as exc:
                raise GalaxiaStoreConfigError(f"probe TIC {tic} failed verification: {exc}") from exc
            except (ValueError, sqlite3.Error, zlib.error, OSError) as exc:
                raise GalaxiaStoreConfigError(f"probe TIC {tic} unreadable: {exc}") from exc
            if doc is None:
                raise GalaxiaStoreConfigError(f"probe TIC {tic} does not resolve in family {self.family}")
        return {"n_parts": n_parts_meta, "n_tics": n_tics_meta, "parts_dir": self.parts_dir,
                "family": self.family, "probe_tics": [str(t) for t in probe_tics]}

    # -- connections ------------------------------------------------------------------------------
    def _conn(self, name: str, path: str) -> sqlite3.Connection:
        conns = getattr(self._local, "conns", None)
        if conns is None:
            conns = self._local.conns = {}
        c = conns.get(name)
        if c is None:
            c = conns[name] = _ro(path)
        return c

    def _index(self) -> sqlite3.Connection:
        return self._conn("index", self.index_path)

    def _part_path(self, i: int) -> str:
        return os.path.join(self.parts_dir, self._parts[i]["file_name"])

    def _check_part(self, i: int, exc_type=GalaxiaStoreError) -> str:
        """The part still has the size/mtime it had when its full-file sha256 was verified."""
        info = self._parts[i]
        path = self._part_path(i)
        try:
            st = os.stat(path)
        except OSError as exc:
            raise exc_type(f"part {i} ({path}) unavailable: {exc}") from exc
        if (st.st_size, st.st_mtime_ns) != (info["size"], info["mtime_ns"]):
            raise exc_type(f"part {i} ({path}) changed since its sha256 was verified "
                           f"(size/mtime {st.st_size}/{st.st_mtime_ns} != {info['size']}/{info['mtime_ns']}); "
                           "rebuild the index")
        return path

    # -- lookups ----------------------------------------------------------------------------------
    def _rows(self, tic) -> List[sqlite3.Row]:
        return self._index().execute(
            "SELECT * FROM tic_rows WHERE tic_id = ?", (_key(tic),)).fetchall()

    def versions_in_family(self, tic_id, family: str) -> List[str]:
        return sorted(r["population_version"] for r in self._rows(tic_id)
                      if _family_of(r["population_version"]) == family)

    def get(self, tic_id, population_version: str) -> Optional[dict]:
        """The verified document of (TIC, exact population_version), ``FppBackgroundStore.get`` shape
        (+ ``store``/``part``), or None when the index has no such row.  A failed check raises."""
        tic = _key(tic_id)
        row = self._index().execute(
            "SELECT * FROM tic_rows WHERE tic_id = ? AND population_version = ?",
            (tic, population_version)).fetchone()
        if row is None:
            return None
        return self._serve(tic, row)

    def get_by_family(self, tic_id, family: str) -> Optional[dict]:
        """The single row of the TIC within ``family``; none -> None; several -> ERROR + None (D2)."""
        from exonit.fpp_background import _log_ambiguous_family
        tic = _key(tic_id)
        rows = [r for r in self._rows(tic) if _family_of(r["population_version"]) == family]
        if not rows:
            return None
        if len(rows) > 1:
            _log_ambiguous_family(tic, family, [r["population_version"] for r in rows])
            return None
        return self._serve(tic, rows[0])

    def _serve(self, tic: str, row: sqlite3.Row) -> dict:
        version, sha, i = row["population_version"], row["blob_sha256"], row["part"]
        if version not in self.versions or _family_of(version) != self.family:
            raise GalaxiaStoreError(f"TIC {tic}: {version} is not a manifest version of family {self.family}")
        path = self._check_part(i)
        prow = self._conn(f"part{i}", path).execute(
            "SELECT tic_id, population_version, n_comp, blob_sha256, blob, crosscheck_json "
            "FROM fpp_background WHERE tic_id = ? AND population_version = ?", (tic, version)).fetchone()
        if prow is None:
            raise GalaxiaStoreError(f"TIC {tic} {version}: indexed in part {i} but absent from it")
        if prow["blob_sha256"] != sha:
            raise GalaxiaStoreError(f"TIC {tic} {version}: part {i} blob_sha256 {prow['blob_sha256']} != "
                                    f"verified {sha}")
        payload = zlib.decompress(bytes(prow["blob"]))
        if _sha256_bytes(payload) != sha:
            raise GalaxiaStoreError(f"TIC {tic} {version}: payload fails its sha256 check (part {i})")
        header = json.loads(payload)["header"]
        if _key(header.get("tic_id")) != tic or header.get("population_version") != version:
            raise GalaxiaStoreError(f"TIC {tic} {version}: payload header names TIC {header.get('tic_id')} "
                                    f"{header.get('population_version')}")
        cx = prow["crosscheck_json"]
        return {"payload": payload, "sha256": sha, "population_version": version,
                "n_comp": prow["n_comp"], "crosscheck": json.loads(cx) if cx else None,
                "store": STORE_NAME, "part": i}

    def fetch_rows(self, tic_ids: Iterable, *, family: Optional[str] = None,
                   version: Optional[str] = None, chunk_size: int = 5000) -> Dict[str, dict]:
        """Light rows (``tic_id, population_version, source, crosscheck_json``) per TIC from the index,
        same contract as ``fpp_background.fetch_population_rows`` (ambiguous TICs absent + logged)."""
        from exonit.fpp_background import _log_ambiguous_family
        ids = sorted({_key(t) for t in tic_ids})
        out: Dict[str, dict] = {}
        conn = self._index()
        for k in range(0, len(ids), chunk_size):
            batch = ids[k:k + chunk_size]
            ph = ",".join("?" * len(batch))
            rows = conn.execute(
                f"SELECT tic_id, population_version, source, crosscheck_json FROM tic_rows "
                f"WHERE tic_id IN ({ph})", tuple(batch)).fetchall()
            by_tic: Dict[str, list] = {}
            for r in rows:
                ver = r["population_version"]
                if (family and _family_of(ver) == family) or (not family and ver == version):
                    by_tic.setdefault(r["tic_id"], []).append(dict(r))
            for tic, hits in by_tic.items():
                if len(hits) == 1:
                    out[tic] = hits[0]
                else:
                    _log_ambiguous_family(tic, family or version, [h["population_version"] for h in hits])
        return out

    def close(self) -> None:
        for c in (getattr(self._local, "conns", None) or {}).values():
            c.close()
        self._local = threading.local()


# --------------------------------------------------------------------------------------------------
# Process-wide switch
# --------------------------------------------------------------------------------------------------

_CACHE_LOCK = threading.Lock()
_CACHE: dict = {}                 # (index, family, parts_dir) -> GalaxiaPartStore | (error, t_failed)
_ERRORS_LOGGED: set = set()


def store_index_path() -> Optional[str]:
    """``EXONIT_FPP_GALAXIA_STORE_INDEX`` or None (switch off: current behaviour)."""
    return os.environ.get(STORE_ENV, "").strip() or None


def log_store_error(err: Exception) -> None:
    """ERROR once per process and message (consumers call it per TIC)."""
    key = str(err)
    with _CACHE_LOCK:
        if key in _ERRORS_LOGGED:
            return
        _ERRORS_LOGGED.add(key)
    logger.error("FPP galaxia store unusable: %s. Every FPP population lookup is refused "
                 "(population_missing / 503) until fixed or %s is unset; no other population is served.",
                 err, STORE_ENV)


def _revalidate_interval() -> float:
    raw = os.environ.get(REVALIDATE_ENV, "").strip()
    try:
        return float(raw) if raw else DEFAULT_REVALIDATE_S
    except ValueError:
        return DEFAULT_REVALIDATE_S


def _evict(key, store: GalaxiaPartStore, err: Exception) -> None:
    """Drop a store that failed re-validation; the failure is cached like a failed open (retried after
    :data:`REOPEN_AFTER_S`, then reopened from scratch)."""
    with _CACHE_LOCK:
        if _CACHE.get(key) is store:
            _CACHE[key] = (err, time.monotonic())
    store.close()


def _still_valid(key, store: GalaxiaPartStore) -> GalaxiaPartStore:
    """The cached store, re-validated when its last check is older than the interval.  Fails closed."""
    interval = _revalidate_interval()
    with _CACHE_LOCK:
        due = interval <= 0 or time.monotonic() - store.validated_at >= interval
        if due:
            store.validated_at = time.monotonic()      # one thread re-validates, the others keep serving
    if not due:
        return store
    try:
        store.revalidate()
    except GalaxiaStoreConfigError as err:
        _evict(key, store, err)
        raise
    return store


def active_store() -> Optional[GalaxiaPartStore]:
    """The configured store, or None when the switch is off.  Switch on but unusable ->
    :class:`GalaxiaStoreConfigError` (never None: None would mean "use the fpp_background table")."""
    index = store_index_path()
    if index is None:
        return None
    from exonit.fpp_background import PopulationFamilyConfigError, population_family
    try:
        family = population_family()
    except PopulationFamilyConfigError as err:
        raise GalaxiaStoreConfigError(f"{STORE_ENV} set but family misconfigured: {err}") from err
    if not family:
        raise GalaxiaStoreConfigError(f"{STORE_ENV} is set but EXONIT_FPP_POPULATION_FAMILY is not: "
                                      "the store is served by TIC within an explicit family only")
    parts_dir = os.environ.get(PARTS_DIR_ENV, "").strip() or None
    key = (index, family, parts_dir)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
    if isinstance(hit, GalaxiaPartStore):
        return _still_valid(key, hit)
    if isinstance(hit, tuple) and time.monotonic() - hit[1] < REOPEN_AFTER_S:
        raise hit[0]
    try:
        store = GalaxiaPartStore(index, family, parts_dir)
        try:
            store.consistency_check()
        except GalaxiaStoreError:
            store.close()
            raise
    except GalaxiaStoreConfigError as err:
        with _CACHE_LOCK:
            _CACHE[key] = (err, time.monotonic())
        raise
    except Exception as exc:  # noqa: BLE001 - any open failure is a refusal, never a fallback
        err = GalaxiaStoreConfigError(f"galaxia store {index} could not be opened: {exc}")
        with _CACHE_LOCK:
            _CACHE[key] = (err, time.monotonic())
        raise err from exc
    with _CACHE_LOCK:
        _CACHE[key] = store
    logger.info("FPP populations served from galaxia store %s (%s parts, %s TICs, family %s, run %s)",
                index, store.meta["n_parts"], store.meta["n_tics"], family, store.meta["run_id"])
    return store


_PUBLIC_ERROR = "galaxia population store unavailable (detail in the server log)"


def health() -> dict:
    """Live health of the population source for ``GET /api/health/fpp_population`` (no paths, no secrets: on failure only
    ``error_type`` and a generic message, the detail is logged server-side).

    Switch off -> ``ok`` with ``enabled: False``.  Switch on -> the store is opened (or taken from the cache),
    FORCE re-validated (manifest sha256 + size/mtime of every part) and one probe TIC is read end to end
    (``consistency_check``: part row, payload sha256, header TIC/version, family); ``latency_ms`` covers the
    whole check.  Any failure -> ``ok: False`` with ``error_type`` / ``error``; the cached store is evicted
    so the next call reopens it."""
    index = store_index_path()
    if index is None:
        return {"ok": True, "enabled": False, "source": "fpp_background-table"}
    out: dict = {"ok": False, "enabled": True, "source": STORE_NAME}
    t0 = time.perf_counter()
    try:
        store = active_store()
        key = (index, store.family, os.environ.get(PARTS_DIR_ENV, "").strip() or None)
        try:
            store.revalidate()
            rep = store.consistency_check()
        except GalaxiaStoreError as err:
            _evict(key, store, err)
            raise
    except GalaxiaStoreError as err:
        log_store_error(err)             # detail (may name mount/part paths) goes to the server log only
        out.update(error_type=type(err).__name__, error=_PUBLIC_ERROR)
        return out
    except Exception as exc:  # noqa: BLE001 - any failure of the live probe is "unhealthy"
        err = GalaxiaStoreConfigError(f"health probe failed: {exc}")
        log_store_error(err)
        out.update(error_type=type(err).__name__, error=_PUBLIC_ERROR)
        return out
    out.update(ok=True, family=rep["family"], run_id=store.meta["run_id"], n_parts=rep["n_parts"],
               n_tics=rep["n_tics"], probe_tics=rep["probe_tics"],
               latency_ms=round((time.perf_counter() - t0) * 1000, 1))
    return out


def refuse_if_switch_on(what: str) -> None:
    """Writers of the ``fpp_background`` table call this first: with the store switch on the table is
    never read, so the write would be invisible (store-ingest review note 4)."""
    if store_index_path() is not None:
        raise GalaxiaStoreSwitchOnError(
            f"{what} refused: {STORE_ENV} is set, the galaxia store is the only FPP population source and "
            "the fpp_background table is ignored; unset it (or run on a host without it) to write the table")


def check_store_config() -> Optional[GalaxiaPartStore]:
    """Startup check (``server.create_app``): open the store once and log; never raises."""
    try:
        return active_store()
    except GalaxiaStoreError as err:
        log_store_error(err)
        return None


def reset_cache() -> None:
    """Drop cached stores and log-once state (tests; a rebuilt index in a running process)."""
    with _CACHE_LOCK:
        stores = [v for v in _CACHE.values() if isinstance(v, GalaxiaPartStore)]
        _CACHE.clear()
        _ERRORS_LOGGED.clear()
    for s in stores:
        s.close()
