"""False-alarm transforms of FITS flux columns (spec section 5).

Pilot scope: ``inv`` only (invert about the median).  The other variants of the
spec (scr_shuf, scr_phase, sector_swap, inv_inj, masked-host filler) are Stage 3
work and are intentionally NOT implemented here (one change per iteration).
"""
from __future__ import annotations

import numpy as np

from exonit.ir.transit_inject import (FAMILIES, _col, valid_cadence_mask)


def invert_about_median(hdu_data, family):
    """f -> 2*median(f) - f on every flux column, the median being the median of
    that column over the cadences the production quality mask keeps.

    Error columns are left unchanged (an inverted dip has the same noise).
    All cadences (also flagged ones) are inverted so the quality mask removes
    exactly the same cadences as in production.  Returns the columns touched.
    """
    valid = valid_cadence_mask(hdu_data, family)
    names = hdu_data.columns.names
    done = []
    for cname in FAMILIES[family]["flux"]:
        real = _col(names, cname)
        if real is None:
            continue
        arr = hdu_data[real]
        x = np.asarray(arr, dtype=np.float64)
        sel = valid & np.isfinite(x)
        if not sel.any():
            continue
        med = float(np.median(x[sel]))
        hdu_data[real] = (2.0 * med - x).astype(arr.dtype)
        done.append(real)
    return done
