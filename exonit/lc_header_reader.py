"""Header-only reads of packed light-curve FITS members (MinIO containers), for the FPP task geometry.

The server cannot open most TGLC / QLP files (they are packed in MinIO containers or live on another host), but the
geometry needs only the primary header and the first extension header (TGLC STAR_X/Y, NEAREDGE; QLP BESTAP; CAMERA/CCD).
The bytes are fetched exactly the way ``/api/fits`` does it: ``MetadataCache.get_fits_blob`` (the ``fits_blobs`` row ->
container key, offset, length) plus ``store.get_range``.  Only the first ``first_bytes`` of the member are read; more only
while the END card of the second header has not been seen.  Members are plain FITS (``/api/fits`` serves the raw bytes with
``application/fits``; the name must end in ``.fits``); a gzip magic is treated as unreadable.

Fail closed: any lookup/MinIO error gives ``(None, None)`` for that file (the sector is then simply ``missing``);
only the time budget raises (``GeometryBudgetExceeded``), so a slow store can never hold the serving path.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter

logger = logging.getLogger("exonit.lc_header_reader")

BLOCK = 2880
FIRST_BYTES = 5 * BLOCK          # 14 400 B: the headers of a TGLC/QLP member end well inside this
MAX_BYTES = 256 * 1024           # never read more than this for two headers


class GeometryBudgetExceeded(Exception):
    """The per-task time budget of the geometry build ran out."""


def parse_headers(buf: bytes, n_hdu: int = 2):
    """-> (headers or None, bytes needed).  ``None`` headers: ``buf`` is too short; ``needed`` is a lower bound of the length
    that would contain them (a multiple of BLOCK).  Raises ValueError for bytes that are not FITS."""
    from astropy.io import fits
    if buf[:2] == b"\x1f\x8b":
        raise ValueError("gzip member")
    if buf[:6] != b"SIMPLE":
        raise ValueError("not a FITS primary header")
    hdrs, pos = [], 0
    for _ in range(n_hdu):
        end, p = None, pos
        while end is None and p + BLOCK <= len(buf):
            blk = buf[p:p + BLOCK]
            p += BLOCK
            if any(blk[i:i + 8] == b"END     " for i in range(0, BLOCK, 80)):
                end = p
        if end is None:
            return None, max(len(buf), p) + BLOCK
        h = fits.Header.fromstring(buf[pos:end].decode("ascii", "replace"))
        hdrs.append(h)
        naxis = int(h.get("NAXIS", 0) or 0)
        size = 0
        if naxis:
            size = 1
            for k in range(1, naxis + 1):
                size *= int(h.get(f"NAXIS{k}", 0) or 0)
            size = abs(int(h.get("BITPIX", 8))) // 8 * int(h.get("GCOUNT", 1) or 1) * (
                int(h.get("PCOUNT", 0) or 0) + size)
        pos = end + -(-size // BLOCK) * BLOCK
    return hdrs, pos


class RemoteHeaderReader:
    """Callable ``reader(path_or_name) -> (primary header, first extension header)`` or ``(None, None)``.

    ``meta``: a MetadataCache (``get_fits_blob``); ``store``: an object store (``get_range``); ``deadline``: a
    ``time.monotonic()`` instant after which every call raises ``GeometryBudgetExceeded``.
    ``stats`` counts the outcome per file (``blob_ok`` / ``blob_missing`` / ``blob_error`` / ``blob_bad``)."""

    def __init__(self, meta, store, deadline=None, first_bytes=None, max_bytes=None):
        self.meta, self.store, self.deadline = meta, store, deadline
        self.first_bytes = int(first_bytes if first_bytes is not None else FIRST_BYTES)
        self.max_bytes = int(max_bytes if max_bytes is not None else MAX_BYTES)
        self.stats = Counter()
        self._stats_lock = threading.Lock()   # an abandoned build thread may still count while the caller reads
        self._memo = {}

    def _count(self, key):
        with self._stats_lock:
            self.stats[key] += 1

    def stats_snapshot(self):
        """Copy of the outcome counters, safe against a build thread that is still running."""
        with self._stats_lock:
            return dict(self.stats)

    def _check(self):
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise GeometryBudgetExceeded()

    def __call__(self, path_or_name):
        name = os.path.basename(str(path_or_name))
        if name in self._memo:
            return self._memo[name]
        res = self._read(name)
        self._memo[name] = res
        return res

    def _read(self, name):
        self._check()
        try:
            blob = self.meta.get_fits_blob(name) if self.meta is not None else None
        except Exception as exc:
            logger.warning("fits_blobs lookup failed for %s: %s", name, exc)
            self._count("blob_error")
            return None, None
        if blob is None or self.store is None:
            self._count("blob_missing")
            return None, None
        container_key, offset, length = blob[0], int(blob[1]), int(blob[2])
        n = min(self.first_bytes, length)
        try:
            while True:
                self._check()
                buf = self.store.get_range(container_key, offset, n)
                hdrs, needed = parse_headers(buf)
                if hdrs is not None:
                    self._count("blob_ok")
                    return hdrs[0], hdrs[1]
                if n >= length or n >= self.max_bytes:
                    self._count("blob_bad")
                    return None, None
                n = min(length, self.max_bytes, max(needed, 2 * n))
        except GeometryBudgetExceeded:
            raise
        except ValueError as exc:
            logger.warning("unreadable header bytes for %s: %s", name, exc)
            self._count("blob_bad")
        except Exception as exc:          # MinIO error / timeout: that file is missing, the task still ships
            logger.warning("range read failed for %s: %s", name, exc)
            self._count("blob_error")
        return None, None
