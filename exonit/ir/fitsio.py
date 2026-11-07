"""Read a TESS FITS, apply a column-wise operation, write a new file.

The output keeps the original file name pattern (``_detect_data_source`` looks
at it) and gets ONE opaque header card ``EXNT_ID`` (hash, never the truth).
"""
from __future__ import annotations

import re
from pathlib import Path

from astropy.io import fits

from exonit.ir.transit_inject import detect_family


def process_file(src, dst, fn, opaque_id):
    """``fn(hdu_data, family) -> dict`` mutates the table in place.  Returns
    (family, fn result)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with fits.open(src, memmap=False) as hdul:
        family = detect_family(hdul[1].columns.names)
        res = fn(hdul[1].data, family)
        hdul[0].header["EXNT_ID"] = (opaque_id, "opaque harness id")
        hdul.writeto(dst, overwrite=False)
    return family, res


_NAME_PATTERNS = [
    (re.compile(r"-(\d{16})-"), 16),       # SPOC  ...-s0001-<tic16>-0120-s_lc.fits
    (re.compile(r"-(\d{16})_"), 16),       # QLP   ..._s0017-<tic16>_tess_v01_llc.fits
    (re.compile(r"gaiaid-(\d+)-"), None),  # TGLC  ..._gaiaid-<id>-s0001-cam2-ccd4_...
]


def synth_name(name, synth_id):
    """Rename a TESS file for a harness variant so that (a) the data-source
    pattern the worker keys on is untouched and (b) different variants of one
    host never share a cache file name.  Replaces the host id field by the
    zero-padded synthetic id."""
    sid = str(int(synth_id))
    for pat, width in _NAME_PATTERNS:
        m = pat.search(name)
        if m:
            w = width or len(m.group(1))
            if len(sid) > w:
                raise ValueError(f"synthetic id {sid} too long for {name}")
            return name[:m.start(1)] + sid.zfill(w) + name[m.end(1):]
    raise ValueError(f"unrecognised TESS file name: {name}")
