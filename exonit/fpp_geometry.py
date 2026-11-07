"""Server-side builder of the per-sector pixel geometry shipped inside an FPP task.

Workers have no external connectivity, so the geometry (2-D TAN WCS, camera/CCD,
target reference pixel, SPOC optimal-aperture mask) is read here from the
read-only FITS cache and carried in the task; see
docs/superpowers/plans/2026-10-15-fpp-native-fix-plan.md, decision (f), step 3.
Every FITS file is opened with mode="readonly" and memmap=False inside a `with`
block; nothing is ever written.
"""
from __future__ import annotations

import glob
import os
import re
from math import cos, hypot, radians

import numpy as np
from astropy.io import fits

GEOMETRY_VERSION = 2               # 1: WCS + SPOC mask; 2: + aperture_kind, c_shift, epochs, sigma_cal (geometry v3; worker fpp_native.GEOMETRY_VERSION)
POS_CORR_K = 6                      # equal-count bins of the good cadences: the SPOC arm's epochs
# sign of POS_CORR relative to the star's position drift: cosine +1.00 (median, 41 pairs, docs/fpp-ffi-geometry-v3-data/diag/poscorr.py)
POS_CORR_SIGN = 1.0
# calibration uncertainty per axis [px] by product (design section 3); must equal worker/fpp_native.SIGMA_CAL_DEFAULT
SIGMA_CAL = {"mask": 0.02, "tglc_box": 0.035, "qlp_circle": 0.05, "default": 0.035}
SCI_COLS = (45, 2092)               # science area of a CCD (RAWX); rows (RAWY) 1..2048
SCI_ROWS = (1, 2048)
LOCAL_TAN_MAX_RESID_PX = 0.01       # pre-registered: local TAN vs the SIP WCS within 90 arcsec
DEFAULT_ROOTS = ("/data/tess_cache/mastDownload/TESS", "/data/tess_cache/tesscut")

# SIP distortion keywords (a TESS aperture header never carries them; if one
# does, the plain PC/CDelt WCS below would silently be wrong).
_SIP_PREFIXES = ("A_", "B_", "AP_", "BP_")

# "tess-s0001-2-2_349.556791_-56.903899_22x22_astrocut.fits" -> centre RA/Dec.
_TESSCUT_CENTRE_RE = re.compile(r"_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_")

_TESSCUT_MAX_SEP_DEG = 4.0 / 60.0  # 4 arcmin


def _wcs_from_header(h):
    """Return the 2-D TAN WCS of an aperture header as a JSON-safe dict."""
    try:
        ctype = [h["CTYPE1"], h["CTYPE2"]]
        crpix = [float(h["CRPIX1"]), float(h["CRPIX2"])]
        crval = [float(h["CRVAL1"]), float(h["CRVAL2"])]
        pc = [float(h["PC1_1"]), float(h["PC1_2"]), float(h["PC2_1"]), float(h["PC2_2"])]
    except KeyError as exc:
        raise ValueError(f"missing WCS keyword {exc}") from None
    if ctype[0] != "RA---TAN" or ctype[1] != "DEC--TAN":
        raise ValueError(f"unsupported WCS type {ctype}")
    for key in h:
        if key.startswith(_SIP_PREFIXES):
            raise ValueError(f"unsupported SIP distortion keyword {key}")
    return {
        "ctype": ctype,
        "crpix": crpix,
        "crval": crval,
        "cdelt": [float(h.get("CDELT1", 1.0)), float(h.get("CDELT2", 1.0))],
        "pc": pc,
    }


def sector_geometry_from_spoc_lc(path):
    """Geometry of one sector from a SPOC 2-minute light curve's APERTURE HDU."""
    with fits.open(path, mode="readonly", memmap=False) as hdul:
        primary = hdul[0].header
        ap = hdul["APERTURE"]
        wcs = _wcs_from_header(ap.header)
        col0 = int(ap.header["CRVAL1P"])
        row0 = int(ap.header["CRVAL2P"])
        camera, ccd, sector = int(primary["CAMERA"]), int(primary["CCD"]), int(primary["SECTOR"])
        data = ap.data.astype(int)
        pos_corr = _pos_corr_epochs(hdul)
    rows, cols = np.nonzero(np.bitwise_and(data, 2))  # optimal aperture = bit 1
    if rows.size == 0:
        aperture, aperture_default = None, True
    else:
        aperture = [[int(c) + col0, int(r) + row0] for r, c in zip(rows, cols)]
        aperture_default = False
    kind = "default" if aperture_default else "mask"
    out = {"sector": sector, "camera": camera, "ccd": ccd,
           "source": "spoc_lc_aperture", "file": os.path.basename(path),
           "wcs": wcs, "col0": col0, "row0": row0, "shape": list(data.shape),
           "aperture": aperture, "aperture_default": aperture_default,
           "aperture_kind": kind, "sigma_cal": SIGMA_CAL[kind], "c_shift": [0.0, 0.0]}
    if not aperture_default:
        out["aperture_source"] = "spoc_lc_mask"
    if pos_corr is not None:
        out["c_shift"] = pos_corr["c_shift"]
        out["epochs"] = pos_corr["epochs"]
        out["pos_corr_n_good"] = pos_corr["n_good"]
    return out


def _pos_corr_epochs(hdul, K=POS_CORR_K):
    """Time-averaged position of the SPOC arm from the light curve itself: c_shift = mean(POS_CORR1/2) over the QUALITY == 0
    cadences, epochs = the means over K equal-count bins of those cadences minus their mean (pre-registration v3, truth arm A).
    None if the table has no usable POS_CORR (< 100 good cadences)."""
    try:
        d = hdul[1].data
        t = np.asarray(d["TIME"], dtype=float)
        q = np.asarray(d["QUALITY"])
        p1 = np.asarray(d["POS_CORR1"], dtype=float)
        p2 = np.asarray(d["POS_CORR2"], dtype=float)
    except (KeyError, TypeError, IndexError, AttributeError):
        return None
    good = (q == 0) & np.isfinite(p1) & np.isfinite(p2) & np.isfinite(t)
    n = int(good.sum())
    if n < 100:
        return None
    idx = np.where(good)[0]
    means = []
    for j in range(K):
        b = idx[j * n // K:(j + 1) * n // K]
        means.append((float(np.mean(p1[b])), float(np.mean(p2[b]))))
    mx = sum(m[0] for m in means) / K
    my = sum(m[1] for m in means) / K
    return {"c_shift": [POS_CORR_SIGN * float(np.mean(p1[good])), POS_CORR_SIGN * float(np.mean(p2[good]))],
            "epochs": [[POS_CORR_SIGN * (m[0] - mx), POS_CORR_SIGN * (m[1] - my)] for m in means], "n_good": n}


def sector_geometry_from_tesscut(path):
    """Geometry of one sector from a TESScut/astrocut cutout (no SPOC aperture)."""
    with fits.open(path, mode="readonly", memmap=False) as hdul:
        primary = hdul[0].header
        wcs = _wcs_from_header(hdul[2].header)
        col0 = int(hdul[1].header["1CRV4P"])
        row0 = int(hdul[1].header["2CRV4P"])
        camera, ccd, sector = int(primary["CAMERA"]), int(primary["CCD"]), int(primary["SECTOR"])
        shape = list(hdul[2].data.shape)
    return {"sector": sector, "camera": camera, "ccd": ccd,
            "source": "tesscut_aperture", "file": os.path.basename(path),
            "wcs": wcs, "col0": col0, "row0": row0, "shape": shape,
            "aperture": None, "aperture_default": True,
            "aperture_kind": "default", "sigma_cal": SIGMA_CAL["default"], "c_shift": [0.0, 0.0]}


def find_spoc_lc(tic, sector, roots=DEFAULT_ROOTS):
    """First cached SPOC 2-min LC file for (tic, sector), or None."""
    for root in roots:
        matches = sorted(glob.glob(f"{root}/tess*-s{int(sector):04d}-{int(tic):016d}-*-s/*_lc.fits"))
        if matches:
            return matches[0]
    return None


def find_tesscut(sector, ra, dec, roots=DEFAULT_ROOTS):
    """Cached TESScut cutout whose centre is nearest (ra, dec), within 4 arcmin.

    The centre is read from the file name only — no FITS file is opened here.
    """
    best, best_sep = None, None
    for root in roots:
        if not os.path.isdir(root):
            continue
        for path in sorted(glob.glob(f"{root}/tess-s{int(sector):04d}-*_astrocut.fits")):
            m = _TESSCUT_CENTRE_RE.search(os.path.basename(path))
            if m is None:
                continue
            dra = (ra - float(m.group(1))) * cos(radians(dec))  # small angle
            sep = hypot(dra, dec - float(m.group(2)))
            if sep <= _TESSCUT_MAX_SEP_DEG and (best_sep is None or sep < best_sep):
                best, best_sep = path, sep
    return best


def sector_geometry_from_ffi_reference(entry, sector, ra, dec, drift_gate="corners"):
    """Geometry of one sector of an FFI-only candidate from the cached reference FFI WCS (exonit/ffi_wcs_cache.py).

    Returns a dict like the other sources (local TAN WCS at the target, pseudo-cutout origin col0 / row0, no aperture) with the provenance
    `ref_file`, `ref_sha256` and the measured `drift_px`; None if the entry is not usable (drift above the pre-registered threshold) or
    if the local TAN does not reproduce the SIP WCS within LOCAL_TAN_MAX_RESID_PX.
    """
    from exonit import ffi_wcs_cache as fw
    # drift_gate "corners" = the PRE-REGISTERED rule (entry usable iff the 20-position drift statistic <= 0.1 px); "off" records the drift and
    # uses the middle-epoch reference regardless (only after the coordinator has ruled on the conflict documented in
    # docs/reviews/2026-10-17-fpp-ffi-wcs-reference.md).
    if drift_gate not in ("corners", "off"):
        raise ValueError(f"unknown drift_gate {drift_gate!r}")
    if entry is None or (drift_gate == "corners" and not entry.get("usable")):
        return None
    cards = entry["mid"]["cards"]
    if fw.local_tan_residual_px(cards, ra, dec) > LOCAL_TAN_MAX_RESID_PX:
        return None
    wcs, col0, row0, (cx, cy) = fw.local_tan(cards, ra, dec)
    return {"sector": int(sector), "camera": int(entry["mid"]["meta"]["CAMERA"]), "ccd": int(entry["mid"]["meta"]["CCD"]),
            "source": "ffi_reference", "file": entry["mid"]["filename"], "wcs": wcs, "col0": col0, "row0": row0,
            "shape": [2 * 11 + 1, 2 * 11 + 1], "aperture": None, "aperture_default": True,
            "ref_file": entry["mid"]["filename"], "ref_sha256": entry["mid"]["sha256"],
            "drift_px": float(entry["drift_px"])}


def product_of(name):
    """Light-curve product of a file name: 'tglc' / 'qlp' / 'spoc' / None (same patterns as exonit.physical_sanity)."""
    if not name:
        return None
    b = os.path.basename(str(name)).lower()
    if b.startswith("hlsp_tglc"):
        return "tglc"
    if b.startswith("hlsp_qlp"):
        return "qlp"
    if b.endswith("_lc.fits") and "hlsp" not in b:
        return "spoc"
    return None


def _read_lc_header(path, header_reader=None):
    """(primary header, first table extension header) of a light-curve file, or (None, None) if it cannot be read.

    A path that is not a local file goes to ``header_reader`` (``exonit.lc_header_reader.RemoteHeaderReader``: the packed
    MinIO member, header bytes only) when one is given."""
    if not path:
        return None, None
    if not os.path.isfile(path):
        return header_reader(path) if header_reader is not None else (None, None)
    try:
        with fits.open(path, mode="readonly", memmap=False) as hdul:
            return hdul[0].header.copy(), (hdul[1].header.copy() if len(hdul) > 1 else None)
    except (OSError, ValueError, KeyError):
        return None, None


def _clip_to_science_area(pixels):
    return [[c, r] for c, r in pixels if SCI_COLS[0] <= c <= SCI_COLS[1] and SCI_ROWS[0] <= r <= SCI_ROWS[1]]


def sector_geometry_from_ffi_epochs(entry, sector, ra, dec, product=None, lc_path=None, drift_gate="corners",
                                   lc_headers=None):
    """Geometry of one sector from the FFI WCS cache (cache v2: K epochs; a v1 entry gives one mid-sector epoch) for a given product.

    The relative geometry is the local TAN of the mid-most epoch; the common-mode target position is the epoch mean (`c_shift`
    = mean minus the shipped local-TAN position) and the epoch offsets the drift.  The APERTURE comes from the product:
      tglc: 3x3 box around k + round(STAR) with k = round(position - STAR) from the TGLC header `STAR_X/Y` (clipped to the
            science area; without a header the worker emulates it and flags it);
      qlp:  circle of radius BESTAP (header) centred on the target;
      None: the default 5x5 (path U); the pre-registered v1 drift gate still applies when the entry has no epochs.
    Returns None if the entry is missing / not usable or the local TAN fails its residual check.
    """
    from exonit import ffi_wcs_cache as fw
    if drift_gate not in ("corners", "off"):
        raise ValueError(f"unknown drift_gate {drift_gate!r}")
    if entry is None:
        return None
    has_epochs = bool(entry.get("epochs"))
    if product is None and not has_epochs and drift_gate == "corners" and not entry.get("usable"):
        return None
    if has_epochs:
        s = fw.epoch_summary(entry, ra, dec)
        ref = entry["epochs"][s["mid_index"]]
    else:
        s, ref = None, entry["mid"]
    cards = ref["cards"]
    if fw.local_tan_residual_px(cards, ra, dec) > LOCAL_TAN_MAX_RESID_PX:
        return None
    wcs, col0, row0, (cx, cy) = fw.local_tan(cards, ra, dec)
    kind = {"tglc": "tglc_box", "qlp": "qlp_circle"}.get(product, "default")
    out = {"sector": int(sector), "camera": int(ref["meta"]["CAMERA"]), "ccd": int(ref["meta"]["CCD"]),
           "source": "ffi_reference", "file": ref["filename"], "wcs": wcs, "col0": col0, "row0": row0,
           "shape": [2 * 11 + 1, 2 * 11 + 1], "aperture": None, "aperture_default": True,
           "ref_file": ref["filename"], "ref_sha256": ref["sha256"],
           "aperture_kind": kind, "sigma_cal": SIGMA_CAL[kind], "c_shift": [0.0, 0.0]}
    if entry.get("drift_px") is not None:
        out["drift_px"] = float(entry["drift_px"])
    mean_c, mean_r = (cx, cy)
    if has_epochs:
        mean_c, mean_r = s["mean"]
        out["c_shift"] = [float(mean_c - cx), float(mean_r - cy)]
        out["epochs"] = [[float(o[0]), float(o[1])] for o in s["offsets"]]
        out["drift_rms_px"] = float(s["drift_rms_px"])
        out["epoch_sha256"] = [e["sha256"] for e in entry["epochs"]]
    prim, ext = lc_headers if lc_headers is not None else _read_lc_header(lc_path)
    if kind == "tglc_box" and prim is not None and "STAR_X" in prim and "STAR_Y" in prim:
        sx, sy = float(prim["STAR_X"]), float(prim["STAR_Y"])
        kx, ky = round(mean_c - sx), round(mean_r - sy)                 # integer offset cut frame -> CCD
        bx, by = kx + round(sx), ky + round(sy)
        out["box_centre"] = [int(bx), int(by)]
        out["aperture"] = _clip_to_science_area([[bx + i, by + j] for i in (-1, 0, 1) for j in (-1, 0, 1)])
        out["aperture_default"] = False
        out["tglc_star_xy"] = [sx, sy]
        out["tglc_hdr_vs_cache_px"] = [float(mean_c - kx - sx), float(mean_r - ky - sy)]
        ne = (ext.get("NEAREDGE") if ext is not None else None)
        out["tglc_near_edge"] = bool(ne) if ne is not None else None
    elif kind == "qlp_circle" and (prim is not None or ext is not None):
        for hd in (prim, ext):
            if hd is not None and "BESTAP" in hd:
                try:
                    out["circle_radius"] = float(str(hd["BESTAP"]).split(":")[0])
                except ValueError:
                    pass
                break
    return out


def _spoc_geometry_for_product(geom, product):
    """A SPOC LC / TESScut geometry used for a TGLC / QLP light curve: the WCS (and POS_CORR drift) stand, the aperture does not."""
    g = dict(geom)
    kind = {"tglc": "tglc_box", "qlp": "qlp_circle"}[product]
    g.update({"aperture": None, "aperture_default": True, "aperture_kind": kind, "sigma_cal": SIGMA_CAL[kind]})
    g.pop("aperture_source", None)
    return g


def tglc_near_edge(geometry):
    """NEAREDGE of the TGLC light curves of a task geometry blob: the value the FPP depth scale needs.

    True if ANY TGLC sector says NEAREDGE (TGLC normalises by a Tmag-predicted flux and a near-edge star
    needs c = 1.27 instead of 1.09), False only if EVERY TGLC sector of the task is known and says no, and
    None when it cannot be told: no TGLC sector in the blob, a TGLC sector without the header value, or a
    task sector the blob does not cover.  None is never a silent "no": the FPP engine flags it and keeps the
    result unvalidated (geometry review N4)."""
    if not isinstance(geometry, dict):
        return None
    sectors = [s for s in (geometry.get("sectors") or []) if isinstance(s, dict) and s.get("aperture_kind") == "tglc_box"]
    if not sectors:
        return None
    flags = [s.get("tglc_near_edge") for s in sectors]
    if any(f is True for f in flags):
        return True
    if all(f is False for f in flags) and not geometry.get("missing_sectors"):
        return False
    return None


def build_task_geometry(tic, ra, dec, sectors, roots=DEFAULT_ROOTS, ffi_cache_dir=None, camera_ccd=None,
                        ffi_drift_gate="corners", lc_files=None, header_reader=None):
    """Build the geometry blob for an FPP task.

    Per sector, by the product of the light curve the worker uses (`lc_files`: {sector: file name or path of that light curve};
    absent = unknown product, the v1 behaviour): a SPOC light curve gets its own WCS, optimal-aperture mask and POS_CORR epochs
    (then TESScut, then the FFI cache); a TGLC / QLP light curve gets the FFI epoch cache with the product aperture from the file
    header and, if the cache has no entry, the SPOC / TESScut WCS of the same star with an emulated product aperture.

    `camera_ccd`: {sector: (camera, CCD)} read from the candidate's light-curve files (TGLC / QLP headers carry them; taken from
    the file header when `lc_files` points at a readable file); `ffi_cache_dir`: directory of the server-side FFI WCS cache;
    `header_reader`: reads the headers of a light-curve file that is not a local file (packed MinIO member).
    """
    if ffi_drift_gate not in ("corners", "off"):
        raise ValueError(f"unknown ffi_drift_gate {ffi_drift_gate!r}")
    built, missing = [], []
    for sector in sectors:
        geom = None
        lc_path = (lc_files or {}).get(int(sector))
        product = product_of(lc_path)
        if product in ("tglc", "qlp"):
            lc_headers = _read_lc_header(lc_path, header_reader)      # read once: camera/ccd, STAR_X/Y, NEAREDGE, BESTAP
            cc = (camera_ccd or {}).get(int(sector))
            if cc is None:
                prim = lc_headers[0]
                if prim is not None and "CAMERA" in prim and "CCD" in prim:
                    cc = (int(prim["CAMERA"]), int(prim["CCD"]))
            if ffi_cache_dir and cc is not None:
                from exonit import ffi_wcs_cache as fw
                try:
                    geom = sector_geometry_from_ffi_epochs(fw.load_entry(ffi_cache_dir, sector, cc[0], cc[1]), sector, ra, dec,
                                                           product=product, lc_path=lc_path, drift_gate=ffi_drift_gate,
                                                           lc_headers=lc_headers)
                except (ValueError, OSError, KeyError):
                    geom = None
            if geom is None:
                path = find_spoc_lc(tic, sector, roots)
                if path is not None:
                    try:
                        geom = _spoc_geometry_for_product(sector_geometry_from_spoc_lc(path), product)
                    except (ValueError, OSError, KeyError):
                        geom = None
            if geom is None:
                path = find_tesscut(sector, ra, dec, roots)
                if path is not None:
                    try:
                        geom = _spoc_geometry_for_product(sector_geometry_from_tesscut(path), product)
                    except (ValueError, OSError, KeyError):
                        geom = None
        else:
            path = find_spoc_lc(tic, sector, roots)
            if path is not None:
                try:
                    geom = sector_geometry_from_spoc_lc(path)
                except (ValueError, OSError, KeyError):
                    geom = None
            if geom is None:
                path = find_tesscut(sector, ra, dec, roots)
                if path is not None:
                    try:
                        geom = sector_geometry_from_tesscut(path)
                    except (ValueError, OSError, KeyError):
                        geom = None
            if geom is None and ffi_cache_dir and camera_ccd and int(sector) in camera_ccd:
                from exonit import ffi_wcs_cache as fw
                cam, ccd = camera_ccd[int(sector)]
                try:
                    geom = sector_geometry_from_ffi_epochs(fw.load_entry(ffi_cache_dir, sector, cam, ccd), sector, ra, dec,
                                                           product=None, drift_gate=ffi_drift_gate)
                except (ValueError, OSError, KeyError):
                    geom = None
        if geom is None:
            missing.append(int(sector))
        else:
            built.append(geom)
    return {"version": GEOMETRY_VERSION,
            "kind": "real" if built else "synthetic",
            "sources": sorted({g["source"] for g in built}),
            "sectors": built,
            "n_task_sectors": len(sectors),
            "missing_sectors": missing}
