"""Server-side cache of reference TESS FFI WCS headers, one per (sector, camera, CCD) (fix plan 1.5 (f), step-3 review ruling).

Why: TGLC and QLP light curves carry no WCS (docs/fpp-ffi-wcs-reference-preregistration.md). D5 needs the sky-to-pixel map of every sector of an
FFI-only candidate. This module reads ONLY the FITS headers (primary + image extension, ~20 kB, HTTP Range) of three calibrated FFIs per
(sector, camera, CCD) from MAST -- the first, the middle and the last of the sector -- and keeps the WCS cards with checksums. The middle one is the
reference; the first/last give the measured intra-sector drift. Workers never contact MAST: the server derives a LOCAL TAN WCS at the target from the
SIP WCS (`local_tan`) and ships it in the FPP task like a SPOC / TESScut WCS.

Network access is injected (`HttpGet`) so everything is testable offline. Nothing is ever written except the JSON cache files in `cache_dir`.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request

import numpy as np

MAST_FFI_BASE = "https://archive.stsci.edu/missions/tess/ffi"
HEADER_FIRST_BYTES = 12 * 2880          # primary (1 block) + image extension (6 blocks) = 7 blocks for SPOC FFIs
HEADER_MAX_BYTES = 60 * 2880
CACHE_VERSION = 2                       # 2: entries may carry K midpoint-bin epochs (geometry v3); v1 entries stay valid
SUPPORTED_CACHE_VERSIONS = (1, 2)
EPOCH_RULE = "midpoint-bins"
# DQUALITY bits that make an FFI's pointing / WCS untrustworthy: lightkurve's default TESS bitmask (1 attitude tweak, 2 safe mode,
# 4 coarse point, 8 Earth point, 32 reaction-wheel desaturation, 128 manual exclude).  Bit 2048 (scattered light) and the like do NOT
# touch the WCS, and a strict `DQUALITY == 0` leaves multi-day holes in the scattered-light part of an orbit (found on the first
# smoke test, s45 1-3, bin 3 of 6: every one of the 16 nearest frames carried 2048).  Decided BEFORE any pilot result was seen.
DQ_WCS_MASK = 175
MAX_CANDIDATES = 16
MAX_EPOCH_SPAN = 2048              # how deep (in frames, nearest first) an epoch may search for a usable frame
DRIFT_THRESHOLD_PX = 0.1                # pre-registered
# science area of a TESS CCD in FFI image coordinates (RAWX / RAWY of the pixel centre)
SCI_COLS = (45.0, 2092.0)
SCI_ROWS = (1.0, 2048.0)
_WCS_KEY = re.compile(r"^(WCSAXES|CTYPE\d|CRPIX\d|CRVAL\d|CDELT\d|CD\d_\d|PC\d_\d|LONPOLE|LATPOLE|RADESYS|EQUINOX|"
                      r"[AB]P?_(ORDER|\d_\d|DMAX)|CRPIX\dP|CRVAL\dP|CDELT\dP|CTYPE\dP)$")
_META_KEYS = ("CAMERA", "CCD", "TSTART", "TSTOP", "DATE-OBS", "NAXIS1", "NAXIS2", "EXTVER", "DQUALITY")


class HttpGet:
    """Default network layer: urllib with Range support, retries and a descriptive User-Agent."""

    def __init__(self, timeout=60.0, retries=3, pause=1.0):
        self.timeout, self.retries, self.pause = timeout, retries, pause

    def __call__(self, url, byte_range=None):
        """-> (status, body bytes, response headers dict with lower-case keys)."""
        last = None
        for k in range(self.retries):
            req = urllib.request.Request(url, headers={"User-Agent": "exonit-ffi-wcs-cache/1 (server-side header reads)"})
            if byte_range is not None:
                req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return r.status, r.read(), {k_.lower(): v for k_, v in r.headers.items()}
            except urllib.error.HTTPError as e:
                if e.code in (404, 416):
                    return e.code, b"", {}
                last = e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = e
            time.sleep(self.pause * (k + 1))
        raise OSError(f"GET {url} failed: {last}")


# --------------------------------------------------------------------------- MAST directory layout

def _hrefs(html: bytes):
    return re.findall(r'href="([^"]+)"', html.decode("latin-1"))


def sector_days(sector, get):
    """Sorted [(year, doy)] of the FFI directories of a sector (a sector can span two years)."""
    out = []
    st, body, _ = get(f"{MAST_FFI_BASE}/s{int(sector):04d}/")
    if st != 200:
        return out
    for y in sorted(h.strip("/") for h in _hrefs(body) if re.fullmatch(r"\d{4}/", h)):
        st, body, _ = get(f"{MAST_FFI_BASE}/s{int(sector):04d}/{y}/")
        if st != 200:
            continue
        for d in sorted(h.strip("/") for h in _hrefs(body) if re.fullmatch(r"\d{3}/", h)):
            out.append((int(y), int(d)))
    return out


def ccd_files(sector, year, doy, camera, ccd, get):
    """Sorted calibrated-FFI (`*_ffic.fits`) file names of one day directory of one (camera, CCD)."""
    st, body, _ = get(f"{MAST_FFI_BASE}/s{int(sector):04d}/{int(year)}/{int(doy):03d}/{int(camera)}-{int(ccd)}/")
    if st != 200:
        return []
    return sorted(h for h in _hrefs(body) if h.endswith("_ffic.fits"))


def pick_epochs(sector, camera, ccd, get, days=None):
    """{'first','mid','last'} -> list of up to MAX_CANDIDATES (year, doy, filename) in order of preference (the first FFIs of a sector can lack a WCS). First file of the first day directory that holds calibrated FFIs, last file of the last
    such directory, the middle file of the day directory nearest to the middle of the list (some day directories hold only CBV / collateral files)."""
    days = days if days is not None else sector_days(sector, get)
    if not days:
        raise LookupError(f"no FFI directories for sector {sector}")
    cache = {}

    def files(k):
        if k not in cache:
            y, d = days[k]
            cache[k] = ccd_files(sector, y, d, camera, ccd, get)
        return cache[k]

    def first_nonempty(order):
        for k in order:
            if files(k):
                return k
        raise LookupError(f"no ffic files for sector {sector} {camera}-{ccd}")

    n = len(days)
    kf = first_nonempty(range(n))
    kl = first_nonempty(range(n - 1, -1, -1))
    mid0 = n // 2
    km = first_nonempty([mid0 + (k // 2 + 1) * (1 if k % 2 else -1) if k else mid0 for k in range(2 * n)
                         if 0 <= (mid0 + ((k // 2 + 1) * (1 if k % 2 else -1) if k else 0)) < n])
    sel = {}
    for tag, k in (("first", kf), ("mid", km), ("last", kl)):
        fl = files(k)
        if tag == "mid":
            m = len(fl) // 2
            order = [m] + [m + (j // 2 + 1) * (1 if j % 2 else -1) for j in range(len(fl))
                           if 0 <= m + (j // 2 + 1) * (1 if j % 2 else -1) < len(fl)]
            sel[tag] = [(days[k][0], days[k][1], fl[i]) for i in order[:MAX_CANDIDATES]]
            continue
        # first / last: the first FFIs of a sector (or a whole first day) can lack a WCS, so walk inwards day by day
        step = 1 if tag == "first" else -1
        cands = []
        kk = k
        while 0 <= kk < n and len(cands) < MAX_CANDIDATES:
            fk = files(kk)
            if fk:
                pick = fk[:2] if tag == "first" else fk[::-1][:2]
                cands += [(days[kk][0], days[kk][1], nm) for nm in pick]
            kk += step
        sel[tag] = cands[:MAX_CANDIDATES]
    return sel


# --------------------------------------------------------------------------- K epochs (geometry v3, design section 4)

_FFI_TIME = re.compile(r"^tess(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})-")


def ffi_name_time(name):
    """Time of an FFI from its file name `tessYYYYDDDHHMMSS-...` in days since 2000-01-01 (only used to place the epochs;
    the header TSTART / TSTOP are stored with the epoch)."""
    import datetime
    m = _FFI_TIME.match(name)
    if not m:
        raise ValueError(f"not an FFI file name: {name}")
    y, doy, hh, mm, ss = (int(g) for g in m.groups())
    return (datetime.date(y, 1, 1) - datetime.date(2000, 1, 1)).days + (doy - 1) + (hh * 3600 + mm * 60 + ss) / 86400.0


def _day_start(year, doy):
    import datetime
    return float((datetime.date(int(year), 1, 1) - datetime.date(2000, 1, 1)).days + int(doy) - 1)


def pick_epochs_k(sector, camera, ccd, get, K=6, days=None):
    """K midpoint-rule epochs: the sector's calibrated-FFI time span [first file, last file] is cut into K equal-time bins and,
    for bin k, the candidates are the (up to MAX_EPOCH_CANDIDATES) FFI files of that CCD nearest to the bin centre (k + 1/2) / K.
    -> (t_lo, t_hi, [[(year, doy, filename), ...] for k in range(K)]).  Bin centres avoid the orbit gap at ~0.5 of the span."""
    days = days if days is not None else sector_days(sector, get)
    if not days:
        raise LookupError(f"no FFI directories for sector {sector}")
    cache = {}

    def files(k):
        if k not in cache:
            y, d = days[k]
            cache[k] = ccd_files(sector, y, d, camera, ccd, get)
        return cache[k]

    n = len(days)
    kf = next((k for k in range(n) if files(k)), None)
    kl = next((k for k in range(n - 1, -1, -1) if files(k)), None)
    if kf is None:
        raise LookupError(f"no ffic files for sector {sector} {camera}-{ccd}")
    t_lo = ffi_name_time(files(kf)[0])
    t_hi = ffi_name_time(files(kl)[-1])
    def candidates(tc):
        """Lazy candidate stream for a bin centre: day directories by distance (listed on demand), the files of a day by
        distance, thinned by a growing stride (every file for the first 8, every 4th up to 64, every 16th up to 512, then every
        64th, at most MAX_EPOCH_SPAN files deep): a frame flagged for pointing near an orbit boundary hides a whole window of the
        stream, and the nearest USABLE frame is what the epoch needs, not the nearest frame."""
        order = sorted(range(n), key=lambda j: abs(_day_start(*days[j]) + 0.5 - tc))
        i, nxt = 0, 0
        for j in order:
            fl = files(j)
            for c in sorted(((days[j][0], days[j][1], nm) for nm in fl), key=lambda c: (abs(ffi_name_time(c[2]) - tc), c[2])):
                if i >= MAX_EPOCH_SPAN:
                    return
                if i == nxt:
                    yield c
                    nxt = i + (1 if i < 8 else 4 if i < 64 else 16 if i < 512 else 64)
                i += 1

    return t_lo, t_hi, [candidates(t_lo + (k + 0.5) / K * (t_hi - t_lo)) for k in range(int(K))]


def build_epochs(sector, camera, ccd, get, K=6, days=None):
    """K epoch headers of one (sector, camera, CCD), midpoint rule.  -> (epochs list, epoch_spec).  Each epoch is
    fetch_header(...) plus k, frac (nominal), frac_actual, t_name, tstart, tstop, dquality, skipped_without_wcs.
    A candidate is skipped when it has no TAN-SIP WCS, a pointing-related DQUALITY bit (DQ_WCS_MASK), or was already used by another
    epoch (K distinct files)."""
    t_lo, t_hi, cands_k = pick_epochs_k(sector, camera, ccd, get, K, days)
    used, epochs = set(), []
    span = max(t_hi - t_lo, 1e-9)
    for k, cands in enumerate(cands_k):
        n_tried = 0
        for n_skipped, (y, d, name) in enumerate(cands):
            n_tried = n_skipped + 1
            # K distinct files in time order: a bin whose nearest usable frame is the one (or an earlier one) of the previous bin
            # keeps looking (two bins inside one flagged window)
            if name in used or (epochs and ffi_name_time(name) <= epochs[-1]["t_name"]):
                continue
            h = fetch_header(sector, y, d, camera, ccd, name, get)
            if "RA---TAN-SIP" not in str(h["cards"].get("CTYPE1", "")) or int(h["meta"].get("DQUALITY", 0) or 0) & DQ_WCS_MASK:
                continue
            if int(h["meta"].get("CAMERA", camera)) != int(camera) or int(h["meta"].get("CCD", ccd)) != int(ccd):
                raise ValueError(f"{name}: header camera/CCD differ from the directory")
            used.add(name)
            t = ffi_name_time(name)
            h.update({"k": k, "frac": (k + 0.5) / K, "frac_actual": (t - t_lo) / span, "t_name": t,
                      "tstart": h["meta"].get("TSTART"), "tstop": h["meta"].get("TSTOP"),
                      "dquality": int(h["meta"].get("DQUALITY", 0) or 0), "skipped_without_wcs": n_skipped})
            epochs.append(h)
            break
        else:
            raise ValueError(f"s{sector} {camera}-{ccd} epoch {k}/{K}: none of {n_tried} candidate FFIs is usable")
    ts = [e["t_name"] for e in epochs]
    if any(b <= a for a, b in zip(ts, ts[1:])):
        raise ValueError(f"s{sector} {camera}-{ccd}: epoch times are not monotone")
    return epochs, {"K": int(K), "rule": EPOCH_RULE, "dq_mask": DQ_WCS_MASK}


def epoch_positions(entry, ra, dec):
    """CCD (col, row) of (ra, dec) in each epoch of an entry -> list of (col, row)."""
    out = []
    for e in entry["epochs"]:
        c, r = world_to_ccd(e["cards"], ra, dec)
        out.append((float(c[0]), float(r[0])))
    return out


def epoch_summary(entry, ra, dec):
    """dict(mean (col, row), offsets [(dx, dy)] from the mean in epoch order, drift_rms_px, mid_index, positions) of an entry
    with epochs.  `mid_index` = the epoch nearest to the middle of the span (its WCS gives the local Jacobian)."""
    pos = epoch_positions(entry, ra, dec)
    K = len(pos)
    mc = sum(p[0] for p in pos) / K
    mr = sum(p[1] for p in pos) / K
    offs = [(p[0] - mc, p[1] - mr) for p in pos]
    rms = math.sqrt(sum(o[0] ** 2 + o[1] ** 2 for o in offs) / K)
    mid = min(range(K), key=lambda i: abs(entry["epochs"][i]["frac_actual"] - 0.5))
    return {"mean": (mc, mr), "offsets": offs, "drift_rms_px": rms, "mid_index": mid, "positions": pos}


# --------------------------------------------------------------------------- header parsing

def _split_headers(raw: bytes):
    """-> (primary_bytes, extension_bytes) or None if the second END card is not in `raw`."""
    ends = []
    for b in range(0, len(raw) - 2879, 2880):
        blk = raw[b:b + 2880]
        if any(blk[i:i + 8] == b"END     " for i in range(0, 2880, 80)):
            ends.append(b + 2880)
            if len(ends) == 2:
                return raw[:ends[0]], raw[ends[0]:ends[1]]
    return None


def fetch_header(sector, year, doy, camera, ccd, name, get):
    """Range-read the primary + image-extension header of one FFI. -> dict(cards, meta, sha256, bytes, etag, size)."""
    from astropy.io import fits
    url = f"{MAST_FFI_BASE}/s{int(sector):04d}/{int(year)}/{int(doy):03d}/{int(camera)}-{int(ccd)}/{name}"
    n = HEADER_FIRST_BYTES
    while True:
        st, body, hdrs = get(url, (0, n - 1))
        if st not in (200, 206):
            raise OSError(f"{url}: HTTP {st}")
        parts = _split_headers(body)
        if parts is not None:
            break
        if n >= HEADER_MAX_BYTES:
            raise ValueError(f"{url}: header longer than {HEADER_MAX_BYTES} bytes")
        n = min(2 * n, HEADER_MAX_BYTES)
    prim = fits.Header.fromstring(parts[0].decode("ascii"))
    ext = fits.Header.fromstring(parts[1].decode("ascii"))
    cards = {k: ext[k] for k in ext if _WCS_KEY.match(k)}
    meta = {k: ext[k] for k in _META_KEYS if k in ext}
    meta.update({k: prim[k] for k in ("SECTOR", "CAMERA", "CCD") if k in prim})
    total = hdrs.get("content-range", "").rsplit("/", 1)[-1]
    return {"filename": name, "path": f"s{int(sector):04d}/{int(year)}/{int(doy):03d}/{int(camera)}-{int(ccd)}",
            "etag": hdrs.get("etag"), "file_size": int(total) if total.isdigit() else None,
            "sha256": hashlib.sha256(parts[0] + parts[1]).hexdigest(), "header_bytes": len(parts[0]) + len(parts[1]),
            "meta": meta, "cards": cards}


# --------------------------------------------------------------------------- WCS maths (astropy, server only)

def astropy_wcs(cards):
    from astropy.io import fits
    from astropy.wcs import WCS
    h = fits.Header()
    for k, v in cards.items():
        h[k] = v
    return WCS(h)


def _offset(cards):
    """FFI pixel-centre coordinate (RAWX / RAWY) = CRVAL?P + (i0 + 1 - CRPIX?P) for the 0-based image index i0."""
    return (cards.get("CRVAL1P", 1) + 1 - cards.get("CRPIX1P", 1), cards.get("CRVAL2P", 1) + 1 - cards.get("CRPIX2P", 1))


def world_to_ccd(cards, ra, dec, wcs=None):
    """(col, row) CCD coordinates of the pixel-centre convention used by upstream (`col0 + all_world2pix(.., 0)`)."""
    w = wcs or astropy_wcs(cards)
    x, y = w.all_world2pix(np.atleast_1d(ra), np.atleast_1d(dec), 0, quiet=True)
    ox, oy = _offset(cards)
    return np.asarray(x) + ox, np.asarray(y) + oy


def ccd_to_world(cards, col, row, wcs=None):
    w = wcs or astropy_wcs(cards)
    ox, oy = _offset(cards)
    ra, dec = w.all_pix2world(np.atleast_1d(col) - ox, np.atleast_1d(row) - oy, 0)
    return np.asarray(ra), np.asarray(dec)


def drift_positions():
    """The 20 pre-registered positions: the 4 corners and a 4 x 4 interior grid of the science area."""
    c0, c1 = SCI_COLS
    r0, r1 = SCI_ROWS
    pts = [(c0, r0), (c0, r1), (c1, r0), (c1, r1)]
    for fc in (0.2, 0.4, 0.6, 0.8):
        for fr in (0.2, 0.4, 0.6, 0.8):
            pts.append((c0 + fc * (c1 - c0), r0 + fr * (r1 - r0)))
    return pts


def wcs_shift_px(cards_ref, cards_other, positions=None):
    """Pixel shift (|dx|, |dy| combined) of the sky position of each CCD position when mapped by `cards_other` instead of `cards_ref`."""
    pts = positions or drift_positions()
    col = np.array([p[0] for p in pts])
    row = np.array([p[1] for p in pts])
    wr, wo = astropy_wcs(cards_ref), astropy_wcs(cards_other)
    ra, dec = ccd_to_world(cards_ref, col, row, wr)
    c2, r2 = world_to_ccd(cards_other, ra, dec, wo)
    return np.hypot(c2 - col, r2 - row)


def drift_of_entry(first, mid, last):
    """max over the 20 positions of max(|first - mid|, |last - mid|) (pre-registered statistic) and the per-pair maxima."""
    a = wcs_shift_px(mid["cards"], first["cards"])
    b = wcs_shift_px(mid["cards"], last["cards"])
    return {"drift_px": float(max(a.max(), b.max())), "first_mid_px": float(a.max()), "last_mid_px": float(b.max())}


def _tan_inverse(ra0, dec0, xi_deg, eta_deg):
    a0, d0 = math.radians(ra0), math.radians(dec0)
    xi, eta = math.radians(xi_deg), math.radians(eta_deg)
    rho = math.hypot(xi, eta)
    if rho == 0.0:
        return ra0, dec0
    c = math.atan(rho)
    dec = math.asin(math.cos(c) * math.sin(d0) + eta * math.sin(c) * math.cos(d0) / rho)
    ra = a0 + math.atan2(xi * math.sin(c), rho * math.cos(d0) * math.cos(c) - eta * math.sin(d0) * math.sin(c))
    return math.degrees(ra) % 360.0, math.degrees(dec)


def local_tan(cards, ra, dec, half=11, wcs=None):
    """LOCAL TAN WCS at (ra, dec) from the SIP WCS: the Jacobian of the exact map at the target, expressed as a pseudo-cutout
    (origin col0, row0 in CCD coordinates) in the same form as a TESScut / SPOC `APERTURE` WCS.
    -> (wcs dict {ctype, crpix, crval, cdelt, pc}, col0, row0, target (col, row))."""
    w = wcs or astropy_wcs(cards)
    cx, cy = (float(v[0]) for v in world_to_ccd(cards, ra, dec, w))
    h = 1.0e-3

    def pix(xi, eta):
        r, d = _tan_inverse(ra, dec, xi, eta)
        x, y = w.all_world2pix(r, d, 0, quiet=True)
        return float(x), float(y)

    pxp, pxm = pix(h, 0.0), pix(-h, 0.0)
    pyp, pym = pix(0.0, h), pix(0.0, -h)
    # d(pixel) / d(xi, eta) in pixels per degree
    J = np.array([[(pxp[0] - pxm[0]) / (2 * h), (pyp[0] - pym[0]) / (2 * h)],
                  [(pxp[1] - pxm[1]) / (2 * h), (pyp[1] - pym[1]) / (2 * h)]])
    PC = np.linalg.inv(J)                            # (xi, eta) = PC @ (dx, dy)
    col0 = int(math.floor(cx)) - int(half)
    row0 = int(math.floor(cy)) - int(half)
    wcs_d = {"ctype": ["RA---TAN", "DEC--TAN"], "crpix": [cx - col0 + 1.0, cy - row0 + 1.0], "crval": [float(ra), float(dec)],
             "cdelt": [1.0, 1.0], "pc": [float(PC[0, 0]), float(PC[0, 1]), float(PC[1, 0]), float(PC[1, 1])]}
    return wcs_d, col0, row0, (cx, cy)


def local_tan_residual_px(cards, ra, dec, radius_arcsec=90.0, n=24, wcs=None):
    """Max pixel difference between the SIP WCS and the local TAN over a circle (+ centre) of the given radius."""
    from astropy.wcs import WCS
    w = wcs or astropy_wcs(cards)
    wd, col0, row0, _ = local_tan(cards, ra, dec, wcs=w)
    lw = WCS(naxis=2)
    lw.wcs.ctype, lw.wcs.crpix, lw.wcs.crval = wd["ctype"], wd["crpix"], wd["crval"]
    lw.wcs.cdelt = wd["cdelt"]
    lw.wcs.pc = [[wd["pc"][0], wd["pc"][1]], [wd["pc"][2], wd["pc"][3]]]
    worst = 0.0
    rad = radius_arcsec / 3600.0
    pts = [(0.0, 0.0)] + [(rad * math.cos(2 * math.pi * k / n), rad * math.sin(2 * math.pi * k / n)) for k in range(n)]
    for xi, eta in pts:
        r, d = _tan_inverse(ra, dec, xi, eta)
        a = w.all_world2pix(r, d, 0, quiet=True)
        ox, oy = _offset(cards)
        ccol, crow = float(a[0]) + ox, float(a[1]) + oy
        b = lw.all_world2pix(r, d, 0)
        worst = max(worst, math.hypot(ccol - (col0 + float(b[0])), crow - (row0 + float(b[1]))))
    return worst


# --------------------------------------------------------------------------- cache files

def build_entry(sector, camera, ccd, get, days=None):
    """First / middle / last headers of one (sector, camera, CCD) plus the pre-registered drift statistic."""
    sel = pick_epochs(sector, camera, ccd, get, days)
    ep = {}
    for tag, cands in sel.items():
        for n_skipped, (y, d, name) in enumerate(cands):
            h = fetch_header(sector, y, d, camera, ccd, name, get)
            if "RA---TAN-SIP" in str(h["cards"].get("CTYPE1", "")):
                h["skipped_without_wcs"] = n_skipped
                ep[tag] = h
                break
        else:
            raise ValueError(f"s{sector} {camera}-{ccd} {tag}: none of {len(cands)} candidate FFIs carries a TAN-SIP WCS")
    for tag, e in ep.items():
        if int(e["meta"].get("CAMERA", camera)) != int(camera) or int(e["meta"].get("CCD", ccd)) != int(ccd):
            raise ValueError(f"{e['filename']}: header camera/CCD differ from the directory")
        if "RA---TAN-SIP" not in str(e["cards"].get("CTYPE1", "")):
            raise ValueError(f"{e['filename']}: unexpected projection {e['cards'].get('CTYPE1')}")
    dr = drift_of_entry(ep["first"], ep["mid"], ep["last"])
    return {**ep, **dr, "usable": dr["drift_px"] <= DRIFT_THRESHOLD_PX}


def cache_path(cache_dir, sector):
    return os.path.join(cache_dir, f"ffi_wcs_s{int(sector):04d}.json")


def save_sector(cache_dir, sector, entries):
    """entries: {"cam-ccd": build_entry(...) [+ epochs]}. Canonical JSON. A file whose entries carry no epochs is written as
    version 1 (byte-identical to the files of the first build); one with epochs as version 2."""
    os.makedirs(cache_dir, exist_ok=True)
    version = CACHE_VERSION if any(e.get("epochs") for e in entries.values()) else 1
    body = {"version": version, "sector": int(sector), "base": MAST_FFI_BASE, "entries": entries}
    text = json.dumps(body, sort_keys=True, indent=1) + "\n"
    with open(cache_path(cache_dir, sector), "w", newline="\n") as fh:
        fh.write(text)
    return hashlib.sha256(text.encode()).hexdigest()


_BODY_LRU = collections.OrderedDict()      # (path, mtime_ns, size) -> parsed cache file (about 0.5 MB of JSON per sector)
_BODY_LRU_MAX = 64                          # >= the most sectors of one TIC (28 measured, 2026-10-18) with margin; <= ~32 MB per process
_BODY_LRU_LOCK = threading.Lock()


def load_entry(cache_dir, sector, camera, ccd):
    """-> entry dict or None (no file / no entry). Verifies the stored per-header sha256 only structurally; the file sha is kept by the caller.

    The parsed file is memoised per (path, mtime, size) in a small LRU; the returned entry is shared between callers and is read-only."""
    p = cache_path(cache_dir, sector)
    try:
        st = os.stat(p)
    except OSError:
        return None
    key = (p, st.st_mtime_ns, st.st_size)
    with _BODY_LRU_LOCK:
        body = _BODY_LRU.get(key)
        if body is not None:
            _BODY_LRU.move_to_end(key)
    if body is None:
        with open(p) as fh:
            body = json.load(fh)
        if body.get("version") not in SUPPORTED_CACHE_VERSIONS:
            raise ValueError(f"{p}: cache version {body.get('version')} not in {SUPPORTED_CACHE_VERSIONS}")
        with _BODY_LRU_LOCK:
            _BODY_LRU[key] = body
            while len(_BODY_LRU) > _BODY_LRU_MAX:
                _BODY_LRU.popitem(last=False)
    return body["entries"].get(f"{int(camera)}-{int(ccd)}")
