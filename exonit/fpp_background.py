"""Galaxia background-star populations for the FPP D/B scenario families.

FPP fix plan step 5 (docs/superpowers/plans/2026-10-15-fpp-native-fix-plan.md
sections 0.3, 4.2-4.5, R7, R8).  Design A of 4.5: a SERVER-SIDE batch job runs
the pinned, unmodified Galaxia 0.7.2 (AGPL-3.0, never patched, never shipped:
only its numerical OUTPUT leaves this module) for a 0.1 deg^2 field around a
TIC, maps the stars to Tmag, and caches the result in the ``fpp_background``
table.  Workers fetch it through ``GET /api/fpp_background/{tic}`` (route in
``exonit/routes/fpp_background.py``); it is NEVER generated on demand inside a
worker request.

Pieces, all pure functions except the two subprocess runners:

* ``galactic_lb``          ICRS -> Galactic (fixed Hipparcos rotation matrix).
* ``stassun_tmag``         Tmag from extincted J, Ks (Stassun+2018, TIC v8).
* ``map_population``       Galaxia arrays -> Tmag/Mact/logg/logTe/[M/H] rows.
* ``build_document``       canonical JSON document + sha256 (the blob contract).
* ``crosscheck_counts``    R8 per-Tmag-bin cross-check against Gaia counts.
* ``GalaxiaInstall``       the pinned install under ~/tools/galaxia.
* ``generate_raw``         one field: par file -> galaxia -> reader -> RAW arrays (fleet store).
* ``serve_document``       RAW arrays + a mapping version -> the canonical served document.
* ``generate_population``  ``generate_raw`` + ``serve_document`` in one call (legacy / small batches).
* ``FppBackgroundStore``   cache behind the Backend interface (PG in
                           production, SQLite for tests / file cache).
* ``prefetch_backgrounds`` the batch helper (one galaxia process per core).

Blob contract (section 4.5): a JSON document ``{"header": {...}, "rows":
[[Tmag, Mact, logg, logTe, [M/H]], ...]}`` serialised with sorted keys and
compact separators; rows are ROUNDED first (``ROW_DECIMALS``) and sorted, and
``blob_sha256`` is the sha256 of exactly those serialised bytes, so what a
worker parses is bit-exactly what is hashed.  The header carries no timestamp
and the stored blob is the zlib stream of the document (the hash is over the
UNcompressed bytes, because zlib output is not guaranteed stable across zlib
versions).  Same field + same pinned install -> byte-identical document.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import platform as _platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

GALAXIA_VERSION = "0.7.2"
# Tmag mapping versions.  "v1" = the plain Stassun+2018 J-Ks relation; "cc1" = v1 plus the
# colour-binned correction below.  The mapping version is part of ``population_version``, so a
# cached population always says which mapping produced it (and v1 stays reproducible: it is the
# parity switch, FPP_BACKGROUND_TMAG_CORRECTION=0, for comparisons with upstream-style mappings).
MAPPING_PLAIN = "tmag-stassun2018-jks-v1"
MAPPING_CC1 = "tmag-stassun2018-jks-cc1"
MAPPING_VERSION = MAPPING_CC1                   # production default (plan 4.2/R8 amendment, 2026-10-17)
# Colour-binned correction cc1: T_corrected = T_stassun + delta for lo < J-Ks <= hi.
# Source: median of T_TRILEGAL(native TESS band) - T_stassun(J, Ks) over the 94 207 corrected-TRILEGAL
# stars of the 30 bench fields (scripts/bench/fpp_vs_triceratops/results/tmag_mapping_residual_by_color.csv):
# the only bin with both a large residual and a large population (50 % of the stars; the join of the two
# branches of the relation) is 0.7 < J-Ks <= 0.85, +0.148 mag.  The other bins are either ~-0.013 (bins
# below 0.7, i.e. already within the 0.006-0.013 mag scatter) or hold < 1 % of the stars (J-Ks < 0,
# > 0.85) where the median is not a stable estimate: they are deliberately NOT corrected.
# CAVEAT: fitted on the same 30 fields it is validated on (in-sample); the out-of-sample check is the
# step-8 held-out fields (plan amendment A5).
TMAG_COLOR_CORRECTION_CC1 = ((0.7, 0.85, 0.148),)
SOURCE = f"galaxia-{GALAXIA_VERSION}"
# sha256 of the pinned upstream tarball (SourceForge galaxia-0.7.2.tar.gz); identifies the SOURCE of
# the generator independently of the per-host binary (compiler, OS and install path differ per host
# without changing the numbers on x86-64; see docs/reviews/2026-10-17-galaxia-fleet-portability.md).
GALAXIA_TARBALL_SHA256 = "550cfd154ff329ef0ef2e68affde48d58da478e73e133465262e3dd23a34d10d"
FIELD_AREA_DEG2 = 0.1
TMAG_LIMIT = 21.0
# Galaxia magnitude limit (intrinsic J) so that extincted T <= TMAG_LIMIT is complete
# (plan 4.2: "J < 22 intrinsic", i.e. TMAG_LIMIT + 1).
GALAXIA_MAG_LIMIT = TMAG_LIMIT + 1.0

RV = 3.1
# A_X / A_V (plan 4.2): J, H, Ks.
A_OVER_AV = {"J": 0.282, "H": 0.175, "Ks": 0.112}

# Decimals of the rounded row columns (Tmag, Mact, logg, logTe, [M/H]).
ROW_DECIMALS = (4, 4, 4, 5, 4)

DEFAULT_ROOT = os.path.expanduser("~/tools/galaxia/0.7.2")
READER_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "fpp_background_read_ebf.py")

# R8 (plan section 8): bins and alarm thresholds of the Galaxia / Gaia cross-check.
XCHECK_BINS = ((12.0, 14.0), (14.0, 16.0), (16.0, 18.0), (18.0, 21.5))
XCHECK_BIN_RANGE = (0.7, 1.4)       # ratio N_galaxia / N_gaia of a bin at T < 18 outside it ...
XCHECK_SIGMA = 3.0                  # ... AND |N_galaxia - N_gaia| >= XCHECK_SIGMA * sqrt(N_galaxia + N_gaia)
XCHECK_TOTAL_RANGE = (0.4, 3.0)     # total N_galaxia / N_gaia outside it -> alarm
XCHECK_MIN_GAIA_PER_BIN = 20        # a bin with fewer Gaia stars has no usable ratio

_PAR_TEMPLATE = """outputFile                          {name}
outputDir                           {outdir}/
photoSys                            DCMC
magcolorNames                       J,J-Ks
appMagLimits[0]                     -100
appMagLimits[1]                     {mlim}
absMagLimits[0]                     -1000
absMagLimits[1]                     1000
colorLimits[0]                      -1000
colorLimits[1]                      1000
geometryOption                      1
longitude                           {l:.6f}
latitude                            {b:.6f}
surveyArea                          {area}
fSample                             1.0
popID                               -1
warpFlareOn                         1
seed                                {seed}
r_max                               1000
starType                            0
photoError                          0
"""


# --------------------------------------------------------------------------
# Pure functions
# --------------------------------------------------------------------------

# ICRS -> Galactic rotation (Hipparcos/ESA 1997, vol. 1 sect. 1.5.3), row-major.
_ICRS_TO_GAL = np.array([
    [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
    [+0.4941094278755837, -0.4448296299600112, +0.7469822444972189],
    [-0.8676661490190047, -0.1980763734312015, +0.4559837761750669],
])


def galactic_lb(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    """Galactic (l, b) in degrees of an ICRS position (deterministic, no astropy)."""
    ra, dec = math.radians(ra_deg), math.radians(dec_deg)
    v = np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])
    x, y, z = _ICRS_TO_GAL @ v
    b = math.degrees(math.asin(max(-1.0, min(1.0, float(z)))))
    l = math.degrees(math.atan2(float(y), float(x))) % 360.0
    return l, b


def seed_for_tic(tic_id) -> int:
    """Deterministic Galaxia seed of a field (the TIC is the cell, design A)."""
    return int(tic_id) % 100000 + 1


def stassun_tmag(J: np.ndarray, Ks: np.ndarray) -> np.ndarray:
    """TIC-v8 Tmag from 2MASS-like J, Ks (Stassun+2018), as triceratops'
    ``funcs.trilegal_results`` applies it to TRILEGAL 1.5 output."""
    J = np.asarray(J, dtype=float)
    c = J - np.asarray(Ks, dtype=float)
    T = np.full_like(J, np.nan)
    m = (-0.1 <= c) & (c <= 0.70)
    T[m] = J[m] + 1.22163 * c[m] ** 3 - 1.74299 * c[m] ** 2 + 1.89115 * c[m] + 0.0563
    m = (0.7 < c) & (c <= 1.0)
    T[m] = J[m] - 269.372 * c[m] ** 3 + 668.453 * c[m] ** 2 - 545.64 * c[m] + 147.811
    m = c < -0.1
    T[m] = J[m] + 0.5
    m = c > 1.0
    T[m] = J[m] + 1.75
    return T


def tmag_correction_enabled() -> bool:
    """Parity switch: FPP_BACKGROUND_TMAG_CORRECTION=0 selects the plain mapping (v1)."""
    return os.environ.get("FPP_BACKGROUND_TMAG_CORRECTION", "1") != "0"


def mapping_version(tmag_correction: Optional[bool] = None) -> str:
    if tmag_correction is None:
        tmag_correction = tmag_correction_enabled()
    return MAPPING_CC1 if tmag_correction else MAPPING_PLAIN


def apply_color_correction(T: np.ndarray, color: np.ndarray) -> np.ndarray:
    """cc1: add the per-colour-bin offset to the Stassun Tmag (bins are (lo, hi])."""
    T = np.array(T, dtype=float)
    for lo, hi, delta in TMAG_COLOR_CORRECTION_CC1:
        T[(color > lo) & (color <= hi)] += delta
    return T


def map_population(raw: Dict[str, np.ndarray],
                   tmag_correction: Optional[bool] = None) -> Dict[str, np.ndarray]:
    """Galaxia arrays (keys rad, mact, grav, teff, feh, dcmc_j, dcmc_h, dcmc_ks,
    ebv; DCMC magnitudes ABSOLUTE and intrinsic, rad in kpc, teff = log10) ->
    stars with extincted Tmag <= TMAG_LIMIT (plan 4.2)."""
    rad = np.asarray(raw["rad"], dtype=float)
    ebv = np.asarray(raw["ebv"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dm = 5.0 * np.log10(rad * 1000.0) - 5.0
    J = raw["dcmc_j"] + dm + A_OVER_AV["J"] * RV * ebv
    Ks = raw["dcmc_ks"] + dm + A_OVER_AV["Ks"] * RV * ebv
    T = stassun_tmag(J, Ks)
    if tmag_correction is None:
        tmag_correction = tmag_correction_enabled()
    if tmag_correction:
        T = apply_color_correction(T, J - Ks)
    keep = np.isfinite(T) & (T <= TMAG_LIMIT)
    return {
        "Tmag": T[keep],
        "Mact": np.asarray(raw["mact"], dtype=float)[keep],
        "logg": np.asarray(raw["grav"], dtype=float)[keep],
        "logTe": np.asarray(raw["teff"], dtype=float)[keep],
        "feh": np.asarray(raw["feh"], dtype=float)[keep],
    }


def canonical_rows(pop: Dict[str, np.ndarray]) -> List[list]:
    """Rounded (ROW_DECIMALS), lexicographically sorted
    ``[Tmag, Mact, logg, logTe, [M/H]]`` rows."""
    cols = [np.round(np.asarray(pop[k], dtype=float), d)
            for k, d in zip(("Tmag", "Mact", "logg", "logTe", "feh"), ROW_DECIMALS)]
    if len(cols[0]) == 0:
        return []
    # np.round can return -0.0; normalise so that the serialisation is unique.
    cols = [c + 0.0 for c in cols]
    order = np.lexsort(tuple(reversed(cols)))     # primary key = Tmag
    return [[float(c[i]) for c in cols] for i in order]


def build_document(rows: List[list], header: dict) -> tuple[bytes, str]:
    """Serialise ``{"header": header, "rows": rows}`` canonically.
    Returns ``(payload_bytes, sha256_hex)``; the hash is over the payload."""
    payload = json.dumps({"header": header, "rows": rows}, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return payload, hashlib.sha256(payload).hexdigest()


def crosscheck_counts(galaxia_tmag: Sequence[float],
                      gaia_tmag: Optional[Sequence[float]]) -> dict:
    """R8 cross-check of the Galaxia population against Gaia counts, PER Tmag bin.

    Direction everywhere: ratio = N_galaxia / N_gaia (the plan's R8 label "Gaia/Galaxia" was an
    error; its own example, the |b| = 2.9 deg field with 18 845 / 8 048 = 2.3, is Galaxia/Gaia).

    Bin alarm (bins with T < 18 only, the faint bin is expected at ~1.45 and is not scaled):
    N_gaia >= ``XCHECK_MIN_GAIA_PER_BIN`` AND ratio outside ``XCHECK_BIN_RANGE`` AND
    |N_galaxia - N_gaia| >= ``XCHECK_SIGMA`` * sigma, with sigma = sqrt(N_galaxia + N_gaia) (both counts
    are Poisson samples, so the variance of their difference is the sum).  Total alarm: the total ratio
    outside ``XCHECK_TOTAL_RANGE``.  The thresholds were set on the 30 bench fields (development
    calibration, not an independent test).  The raw ratios and per-bin ``z`` are always stored, so the
    rule can be revisited.  The consequence of an alarm (Gaia-based population, recorded as
    ``population_source``) belongs to the FPP side.
    """
    g = np.asarray(galaxia_tmag, dtype=float)
    n_gal_bins = [int(((g >= lo) & (g < hi)).sum()) for lo, hi in XCHECK_BINS]
    if gaia_tmag is None:
        return crosscheck_from_counts(n_gal_bins, len(g), None, None)
    a = np.asarray(gaia_tmag, dtype=float)
    n_gaia_bins = [int(((a >= lo) & (a < hi)).sum()) for lo, hi in XCHECK_BINS]
    return crosscheck_from_counts(n_gal_bins, len(g), n_gaia_bins, len(a))


def crosscheck_from_counts(n_galaxia_bins: Sequence[int], n_galaxia_total: int,
                           n_gaia_bins: Optional[Sequence[int]],
                           n_gaia_total: Optional[int]) -> dict:
    """The R8 rule of ``crosscheck_counts`` evaluated on per-bin COUNTS (the fleet stores Galaxia bin
    counts per field and aggregated Gaia counts per sky cell, so R8 is re-evaluable offline).
    ``n_gaia_bins is None`` -> no Gaia reference (``gaia_available: false``)."""
    out = {"n_galaxia": int(n_galaxia_total), "bins": [list(b) for b in XCHECK_BINS],
           "ratio_bins": [], "z_bins": [], "n_galaxia_bins": [int(x) for x in n_galaxia_bins],
           "n_gaia_bins": [], "alarm": False,
           "alarm_reasons": [], "gaia_available": n_gaia_bins is not None,
           "rule": {"bin_range": list(XCHECK_BIN_RANGE), "sigma": XCHECK_SIGMA,
                    "sigma_def": "sqrt(N_galaxia + N_gaia)", "min_gaia_per_bin": XCHECK_MIN_GAIA_PER_BIN,
                    "total_range": list(XCHECK_TOTAL_RANGE), "direction": "galaxia/gaia"}}
    if n_gaia_bins is None:
        out["ratio_bins"] = [None] * len(XCHECK_BINS)
        out["z_bins"] = [None] * len(XCHECK_BINS)
        return out
    out["n_gaia"] = int(n_gaia_total)
    for (lo, hi), ng, na in zip(XCHECK_BINS, out["n_galaxia_bins"], n_gaia_bins):
        na = int(na)
        out["n_gaia_bins"].append(na)
        if na < XCHECK_MIN_GAIA_PER_BIN:
            out["ratio_bins"].append(None)
            out["z_bins"].append(None)
            continue
        ratio = ng / na
        z = (ng - na) / math.sqrt(ng + na)
        out["ratio_bins"].append(round(ratio, 4))
        out["z_bins"].append(round(z, 3))
        if (hi <= 18.0 and not (XCHECK_BIN_RANGE[0] <= ratio <= XCHECK_BIN_RANGE[1])
                and abs(z) >= XCHECK_SIGMA):
            out["alarm"] = True
            out["alarm_reasons"].append(
                f"bin {lo}-{hi}: Galaxia/Gaia {ratio:.2f} outside {XCHECK_BIN_RANGE} at {abs(z):.1f} sigma")
    if n_gaia_total > 0:
        tot = n_galaxia_total / n_gaia_total
        out["galaxia_over_gaia_total"] = round(tot, 4)
        if not (XCHECK_TOTAL_RANGE[0] <= tot <= XCHECK_TOTAL_RANGE[1]):
            out["alarm"] = True
            out["alarm_reasons"].append(f"total Galaxia/Gaia {tot:.2f} outside {XCHECK_TOTAL_RANGE}")
    return out


def gaia_to_tmag(G, BP, RP):
    """Tmag from Gaia DR3 G, BP, RP: Stassun+2019 (TIC v8) G-T colour relation, the same published
    relation upstream's background query uses (colour clipped to [-0.2, 3.5]; G - 0.43 without colour)."""
    G = np.asarray(G, dtype=float)
    c = np.clip(np.asarray(BP, dtype=float) - np.asarray(RP, dtype=float), -0.2, 3.5)
    T = G - 0.00522555 * c ** 3 + 0.0891337 * c ** 2 - 0.633923 * c + 0.0324473
    return np.where(np.isfinite(c), T, G - 0.43)


def gaia_tmag_tap(ra: float, dec: float) -> np.ndarray:
    """Tmag list of the Gaia DR3 sources (G < 21) in the 0.1 deg^2 circle around (ra, dec): the R8
    reference.  Server-side only (public ESA TAP service); raises on failure so that a caller can
    record ``gaia_unavailable`` instead of silently skipping the check."""
    from astroquery.gaia import Gaia
    radius = math.sqrt(FIELD_AREA_DEG2 / math.pi)
    adql = ("SELECT phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag FROM gaiadr3.gaia_source "
            f"WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius})) "
            f"AND phot_g_mean_mag < {TMAG_LIMIT}")
    Gaia.ROW_LIMIT = -1
    tbl = Gaia.launch_job_async(adql, verbose=False).get_results()
    return gaia_to_tmag(np.asarray(tbl["phot_g_mean_mag"], dtype=float),
                        np.asarray(tbl["phot_bp_mean_mag"], dtype=float),
                        np.asarray(tbl["phot_rp_mean_mag"], dtype=float))


# --------------------------------------------------------------------------
# The pinned install and one-field generation
# --------------------------------------------------------------------------

def platform_tag() -> str:
    """CPU class that enters every stored raw field and ``population_version``.

    Galaxia is bit-identical across x86-64 hosts (Linux g++ and Windows MinGW, proven on the
    conformance fields) but NOT on arm64 (Apple libm): same statistics within Poisson noise, other
    random realisation (docs/reviews/2026-10-17-galaxia-fleet-portability.md, decision 1A).  An
    unknown CPU class is refused rather than guessed."""
    m = _platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return "x86_64"
    if m in ("arm64", "aarch64"):
        return "arm64"
    raise RuntimeError(f"unsupported CPU class {m!r}: no conformance data for Galaxia on it")


def _tree_ignored(name: str) -> bool:
    """Finder / AppleDouble litter is not data (a .DS_Store ships in the upstream tarball, and macOS
    creates ``._*`` files on non-HFS copies), so it must not change the tree hash."""
    return name == ".DS_Store" or name.startswith("._")


def tree_sha256(root: str) -> str:
    """sha256 over the sorted (POSIX relative path, file sha256) list of a directory tree, identical
    on Linux, macOS and Windows for identical content (path separator normalised, OS litter ignored)."""
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in filenames:
            if _tree_ignored(fn):
                continue
            p = os.path.join(dirpath, fn)
            entries.append((os.path.relpath(p, root).replace(os.sep, "/"), file_sha256(p)))
    entries.sort()
    h = hashlib.sha256()
    for rel, digest in entries:
        h.update(f"{rel}\0{digest}\n".encode())
    return h.hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def served_version(raw_version: str, mapping: str) -> str:
    """``population_version`` of the document served for ``raw_version`` under ``mapping``:
    ``galaxia-0.7.2+src.S+data.D+plat.P+<mapping>+A0.1+T21``."""
    parts = raw_version.split("+")
    # the "A<area>" part sits after the platform tag: index 4 for the C++ engine (family+src+data+plat+A...),
    # index 5 for galaxia-rs (family+src+data+plat+be+A...)
    idx = next((i for i in (4, 5) if len(parts) > i + 1 and re.fullmatch(r"A\d[\d.]*", parts[i])), None)
    if idx is None:
        raise ValueError(f"not a raw_version: {raw_version!r}")
    return "+".join(parts[:idx] + [mapping, parts[idx], f"T{TMAG_LIMIT:g}"])


def raw_family(raw_version: str) -> str:
    """Engine + version family of a raw_version: its first ``+`` field (``galaxia-0.7.2`` for the C++ engine,
    ``galaxia-rs-<cargo version>`` for galaxia-rs).  Rows of different families are never one population."""
    return raw_version.split("+", 1)[0]


@dataclass(frozen=True)
class GalaxiaInstall:
    """The pinned Galaxia install (docs/runbooks/galaxia-install.md): ``<root>/prefix/bin/galaxia``,
    ``<root>/GalaxiaData`` (the data directory is compiled into the binary, so it must be where
    ``configure --datadir`` said), optional ``<root>/venv`` (numpy + ebfpy) and ``<root>/ebf/ebf.py``.

    ``provenance-fleet.json`` (written once by ``write_provenance`` after the build) holds the source
    tarball sha256 and the portable data-tree sha256; with the runtime platform tag they form every
    ``population_version``.  The per-host binary hash is informational only (it differs per host
    because of compiler, OS and install path even where the output is bit-identical).
    """
    root: str = DEFAULT_ROOT

    @property
    def binary(self) -> str:
        exe = "galaxia.exe" if sys.platform == "win32" else "galaxia"
        return os.path.join(os.path.abspath(self.root), "prefix", "bin", exe)

    @property
    def python(self) -> str:
        sub = ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")
        venv = os.path.join(os.path.abspath(self.root), "venv", *sub)
        return venv if os.path.exists(venv) else sys.executable

    @property
    def ebf_dir(self) -> Optional[str]:
        d = os.path.join(os.path.abspath(self.root), "ebf")
        return d if os.path.isdir(d) else None

    @property
    def provenance_path(self) -> str:
        return os.path.join(self.root, "provenance-fleet.json")

    def available(self) -> bool:
        return all(os.path.exists(p) for p in (self.binary, self.provenance_path))

    def write_provenance(self) -> dict:
        prov = {
            "schema": 2,
            "galaxia_version": GALAXIA_VERSION,
            "src_tarball_sha256": GALAXIA_TARBALL_SHA256,
            "data_tree_sha256": tree_sha256(os.path.join(self.root, "GalaxiaData")),
            "binary_sha256": file_sha256(self.binary),            # informational, not in any version
            "platform": platform_tag(),
        }
        with open(self.provenance_path, "w") as f:
            json.dump(prov, f, indent=2, sort_keys=True)
        return prov

    def provenance(self) -> dict:
        with open(self.provenance_path) as f:
            prov = json.load(f)
        if prov.get("schema") != 2:
            raise ValueError(f"{self.provenance_path}: not a schema-2 provenance")
        if prov.get("platform") != platform_tag():
            raise ValueError(f"{self.provenance_path} was written on {prov.get('platform')}, "
                             f"this host is {platform_tag()}")
        return prov

    def raw_version(self, tmag_ceiling: Optional[float] = None) -> str:
        """Identity of the RAW Galaxia output (no mapping): the raw store is keyed by it.
        ``tmag_ceiling`` (optional, see ``generate_raw``) is part of it because it changes the stored rows."""
        p = self.provenance()
        v = (f"{SOURCE}+src.{p['src_tarball_sha256'][:12]}+data.{p['data_tree_sha256'][:12]}"
             f"+plat.{platform_tag()}+A{FIELD_AREA_DEG2}+M{GALAXIA_MAG_LIMIT:g}")
        return v if tmag_ceiling is None else f"{v}+C{tmag_ceiling:g}"

    def population_version(self, tmag_correction: Optional[bool] = None,
                           mapping: Optional[str] = None) -> str:
        return served_version(self.raw_version(), mapping or mapping_version(tmag_correction))


def current_population_version(install: Optional[GalaxiaInstall] = None) -> Optional[str]:
    """Population version of the pinned install, or None when this host has none."""
    install = install or GalaxiaInstall(os.environ.get("EXONIT_GALAXIA_ROOT", DEFAULT_ROOT))
    try:
        return install.population_version() if install.available() else None
    except (OSError, KeyError, ValueError):
        return None


#: Env var naming the accepted population FAMILY (decision D2, 2026-10-24).  Explicit, never inferred.
FAMILY_ENV = "EXONIT_FPP_POPULATION_FAMILY"
_PLATFORM_TOKENS = ("src.", "plat.")


class PopulationFamilyConfigError(ValueError):
    """``EXONIT_FPP_POPULATION_FAMILY`` is set but is not a family (it still carries ``src.``/``plat.``
    tokens).  A ``ValueError`` subclass; consumers refuse (no population is ever picked)."""


def family_of(population_version: str) -> str:
    """``population_version`` without its ``+src.*`` and ``+plat.*`` tokens.

    The Galaxia-rs version string embeds the source checkout and the platform that computed the
    field; the three platform classes are verified byte-identical, so the remaining tokens (code
    version, data tree, backend, Tmag mapping, area, limit) define what is scientifically the same
    population.  The per-row full string stays as provenance."""
    return "+".join(t for t in population_version.split("+") if not t.startswith(_PLATFORM_TOKENS))


def population_family() -> Optional[str]:
    """The configured accepted family (``EXONIT_FPP_POPULATION_FAMILY``), or None (legacy exact-version
    path).  A value that still carries ``src.``/``plat.`` tokens is a misconfiguration (it would pin one
    platform while looking like a family) and raises ``ValueError`` rather than being silently reinterpreted."""
    fam = os.environ.get(FAMILY_ENV, "").strip()
    if not fam:
        return None
    if family_of(fam) != fam:
        raise PopulationFamilyConfigError(f"{FAMILY_ENV}={fam!r} must not contain +src.* / +plat.* tokens")
    return fam


#: Values of ``EXONIT_FPP_POPULATION_FAMILY`` whose misconfiguration was already logged (once per value).
_FAMILY_CONFIG_ERRORS_LOGGED: set = set()
#: Guards the log-once sets below and above (the server answers from several threads).
_LOG_ONCE_LOCK = threading.Lock()


def log_population_family_config_error(err: Exception) -> None:
    """Log a misconfigured ``EXONIT_FPP_POPULATION_FAMILY`` at ERROR once per process and value
    (startup check and per-TIC consumers share it, so the log is not flooded per TIC)."""
    key = str(err)
    with _LOG_ONCE_LOCK:
        if key in _FAMILY_CONFIG_ERRORS_LOGGED:
            return
        _FAMILY_CONFIG_ERRORS_LOGGED.add(key)
    logger.error("FPP population family misconfigured: %s. Every FPP task will be refused "
                 "(population_missing) and the population route / readiness gate return 500 until "
                 "%s is fixed or unset (unset = legacy exact-version path).", err, FAMILY_ENV)


def check_population_family_config() -> Optional[str]:
    """Startup / config-load check of ``EXONIT_FPP_POPULATION_FAMILY``.  Returns the family, or None when
    unset (legacy path) or misconfigured.  A misconfiguration is logged ONCE at ERROR and never raised:
    the server must still start, and every consumer keeps refusing (it never picks a population)."""
    try:
        fam = population_family()
    except PopulationFamilyConfigError as err:
        log_population_family_config_error(err)
        return None
    if fam:
        logger.info("FPP population served by TIC within family %s (%s)", fam, FAMILY_ENV)
    return fam


#: (tic, family, versions) already logged as ambiguous: the ERROR is emitted once per distinct state,
#: not on every gate / poll call.  Bounded so a pathological store cannot grow it without limit.
_AMBIGUOUS_LOGGED: set = set()
_AMBIGUOUS_LOGGED_MAX = 100_000


def _log_ambiguous_family(tic, family: str, versions: Sequence[str]) -> None:
    """ERROR for a TIC with several rows in the served family, once per (tic, family, set of versions);
    a later repeat is DEBUG.  A change in the row set (e.g. one more platform) logs again."""
    key = (_key(tic), family, tuple(sorted(versions)))
    with _LOG_ONCE_LOCK:
        first = key not in _AMBIGUOUS_LOGGED
        if first:
            if len(_AMBIGUOUS_LOGGED) >= _AMBIGUOUS_LOGGED_MAX:
                _AMBIGUOUS_LOGGED.clear()
            _AMBIGUOUS_LOGGED.add(key)
    log = logger.error if first else logger.debug
    log("fpp_background TIC %s: %d rows in family %s (%s): refusing to pick one",
        key[0], len(versions), family, list(key[2]))


# --------------------------------------------------------------------------
# RAW Galaxia rows (fleet store) and the mapping applied at build / serve time
# --------------------------------------------------------------------------

# Every array the reader extracts (J, H, Ks, distance, E(B-V) and the stellar parameters): the cc1 ->
# v1 revert (plan A5) or a new mapping must never force regenerating a field.
RAW_COLUMNS = ("rad", "mact", "grav", "teff", "feh", "dcmc_j", "dcmc_h", "dcmc_ks", "ebv")
RAW_LAYOUT = "colmajor-byteshuffle-zlib6-v1"
_MAPPING_FLAGS = {MAPPING_PLAIN: False, MAPPING_CC1: True}      # mapping_version -> tmag_correction


def mapping_flag(mapping: str) -> bool:
    try:
        return _MAPPING_FLAGS[mapping]
    except KeyError:
        raise ValueError(f"unknown mapping version {mapping!r}; known: {sorted(_MAPPING_FLAGS)}") from None


def _storage_dtype(a: np.ndarray) -> str:
    """'<f4' when the float64 values survive the float32 round trip exactly (Galaxia writes float32),
    else '<f8': storage is always lossless."""
    a = np.asarray(a, dtype=np.float64)
    return "<f4" if np.array_equal(a.astype("<f4").astype(np.float64), a, equal_nan=True) else "<f8"


def raw_canonical_bytes(arrays: Dict[str, np.ndarray], dtypes: Sequence[str]) -> bytes:
    """Column-major concatenation of the RAW columns in ``RAW_COLUMNS`` order, little-endian: the
    bytes ``raw_sha256`` is defined over (independent of the compression layout)."""
    return b"".join(np.ascontiguousarray(np.asarray(arrays[c], dtype=np.float64).astype(dt)).tobytes()
                    for c, dt in zip(RAW_COLUMNS, dtypes))


def pack_raw(arrays: Dict[str, np.ndarray]) -> dict:
    """``{"meta": json str, "sha256", "blob"}``: the storage form of the raw arrays.  The blob is the
    zlib stream of the byte-plane-shuffled canonical bytes (compresses ~15 % better than plain)."""
    n = int(len(arrays["rad"]))
    dtypes = [_storage_dtype(arrays[c]) for c in RAW_COLUMNS]
    canon = raw_canonical_bytes(arrays, dtypes)
    parts, off = [], 0
    for dt in dtypes:
        size = n * np.dtype(dt).itemsize
        col = np.frombuffer(canon, dtype=np.uint8, count=size, offset=off).reshape(n, np.dtype(dt).itemsize)
        parts.append(np.ascontiguousarray(col.T).tobytes())
        off += size
    meta = json.dumps({"n": n, "columns": list(RAW_COLUMNS), "dtypes": dtypes, "layout": RAW_LAYOUT},
                      sort_keys=True, separators=(",", ":"))
    return {"meta": meta, "sha256": hashlib.sha256(canon).hexdigest(),
            "blob": zlib.compress(b"".join(parts), 6)}


def unpack_raw(meta: str, blob: bytes, sha256: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Inverse of ``pack_raw`` (float64 arrays, bit-equal to what the reader produced); raises when
    ``sha256`` is given and the decoded bytes do not match it."""
    m = json.loads(meta)
    if m["layout"] != RAW_LAYOUT or m["columns"] != list(RAW_COLUMNS):
        raise ValueError(f"unsupported raw layout {m.get('layout')!r}")
    n = m["n"]
    shuffled = zlib.decompress(bytes(blob))
    canon_parts, off = [], 0
    for dt in m["dtypes"]:
        isz = np.dtype(dt).itemsize
        size = n * isz
        col = np.frombuffer(shuffled, dtype=np.uint8, count=size, offset=off).reshape(isz, n)
        canon_parts.append(np.ascontiguousarray(col.T).tobytes())
        off += size
    canon = b"".join(canon_parts)
    if sha256 is not None and hashlib.sha256(canon).hexdigest() != sha256:
        raise ValueError("raw blob fails its sha256 check")
    arrays, off = {}, 0
    for c, dt in zip(RAW_COLUMNS, m["dtypes"]):
        size = n * np.dtype(dt).itemsize
        arrays[c] = np.frombuffer(canon, dtype=dt, count=n, offset=off).astype(np.float64)
        off += size
    return arrays


def mapping_bin_counts(arrays: Dict[str, np.ndarray]) -> dict:
    """Per-Tmag-bin Galaxia counts (the R8 bins) under every known mapping, stored with the raw
    field so that R8 can be re-evaluated against Gaia counts later without touching the raw blob."""
    out = {}
    for name, flag in _MAPPING_FLAGS.items():
        t = map_population(arrays, flag)["Tmag"]
        out[name] = {"n": int(len(t)),
                     "bins": [int(((t >= lo) & (t < hi)).sum()) for lo, hi in XCHECK_BINS]}
    return out


def _run_galaxia(install: GalaxiaInstall, par: str, scratch: str, name: str, *,
                 cpu: Optional[int], nice: int, timeout_s: float, stats: Optional[dict] = None) -> None:
    """Run one galaxia process on ``par``.  Galaxia exits with status 1 on success; the caller judges
    success by the output file.  Priority: ``nice`` on POSIX, BELOW_NORMAL on Windows; ``taskset`` only
    on Linux (macOS and Windows have no per-process affinity: the fleet limits the PROCESS COUNT).

    ``stats`` (optional dict): on POSIX receives ``galaxia_rss_mb``, the peak resident set of THIS process
    (``wait4`` rusage; RUSAGE_CHILDREN is process-wide and cannot tell fields apart).  Not available on Windows."""
    kw = {}
    if sys.platform == "win32":
        kw["creationflags"] = 0x00004000                     # BELOW_NORMAL_PRIORITY_CLASS
        cmd = [install.binary, "-r", par]
    else:
        cmd = ["nice", "-n", str(nice)]
        if cpu is not None and sys.platform.startswith("linux") and shutil.which("taskset"):
            cmd += ["taskset", "-c", str(cpu)]
        cmd += [install.binary, "-r", par]
    with open(os.path.join(scratch, name + ".log"), "wb") as log:
        if stats is None or sys.platform == "win32":
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=scratch,
                           timeout=timeout_s, check=False, **kw)
            return
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=scratch, **kw)
        timed_out = threading.Event()

        def _kill():
            timed_out.set()
            try:
                p.kill()
            except OSError:
                pass
        timer = threading.Timer(timeout_s, _kill)
        timer.start()
        try:
            _, status, ru = os.wait4(p.pid, 0)
        finally:
            timer.cancel()
        p.returncode = os.waitstatus_to_exitcode(status)     # we reaped it ourselves
        stats["galaxia_rss_mb"] = ru.ru_maxrss / (1024.0 * 1024.0) if sys.platform == "darwin" else ru.ru_maxrss / 1024.0
        if timed_out.is_set():
            raise subprocess.TimeoutExpired(cmd, timeout_s)


def apply_tmag_ceiling(arrays: Dict[str, np.ndarray], tmag_ceiling: float) -> Dict[str, np.ndarray]:
    """Keep the stars whose plain-mapping (v1) extincted Tmag is finite and <= ``tmag_ceiling``.  The ONE
    definition of the raw-store ceiling: ``generate_raw`` (C++ engine) and the galaxia-rs read-back both use it."""
    rad = arrays["rad"]
    with np.errstate(divide="ignore", invalid="ignore"):
        dm = 5.0 * np.log10(rad * 1000.0) - 5.0
    T = stassun_tmag(arrays["dcmc_j"] + dm + A_OVER_AV["J"] * RV * arrays["ebv"],
                     arrays["dcmc_ks"] + dm + A_OVER_AV["Ks"] * RV * arrays["ebv"])
    keep = np.isfinite(T) & (T <= tmag_ceiling)
    return {k: v[keep] for k, v in arrays.items()}


def generate_raw(install: GalaxiaInstall, tic_id, ra: float, dec: float, *,
                 cpu: Optional[int] = None, nice: int = 10,
                 timeout_s: float = 900.0, workdir: Optional[str] = None,
                 tmag_ceiling: Optional[float] = None, stats: Optional[dict] = None) -> dict:
    """Run Galaxia for the 0.1 deg^2 field around (ra, dec) and return the RAW arrays:
    ``{"tic_id", "ra", "dec", "gal_l", "gal_b", "seed", "arrays", "n_galaxia"}`` (no mapping applied).

    ``tmag_ceiling`` (default None = keep every star): drop stars whose plain-mapping (v1) extincted
    Tmag exceeds it.  The mapping keeps T <= 21 and cc1 only shifts by +0.148, so any ceiling >= 22.5
    leaves both mappings unchanged; it is part of ``raw_version`` because it changes the stored rows.

    One galaxia process (pinned to ``cpu`` on Linux only) at ``nice``; scratch directory removed
    afterwards.  Galaxia exits with status 1 on success, so success is judged by the output file.
    ``stats`` (optional dict) receives ``galaxia_rss_mb`` (POSIX): the peak RSS of this field's process.
    """
    ra, dec = round(float(ra), 6), round(float(dec), 6)   # the header stores these: regeneration from it is exact
    l, b = galactic_lb(ra, dec)
    seed = seed_for_tic(tic_id)
    base = os.path.abspath(workdir or os.path.join(install.root, "run"))
    os.makedirs(base, exist_ok=True)
    scratch = os.path.abspath(tempfile.mkdtemp(prefix=f"gx_{tic_id}_", dir=base))
    try:
        name = f"gx_{tic_id}"
        par = os.path.join(scratch, name + ".par")
        with open(par, "w", newline="\n") as f:               # LF on Windows too (Galaxia's reader is not CRLF-safe)
            f.write(_PAR_TEMPLATE.format(name=name, outdir=scratch.replace("\\", "/"), mlim=GALAXIA_MAG_LIMIT,
                                         l=l, b=b, area=FIELD_AREA_DEG2, seed=seed))
        _run_galaxia(install, par, scratch, name, cpu=cpu, nice=nice, timeout_s=timeout_s, stats=stats)
        ebf = os.path.join(scratch, name + ".ebf")
        if not os.path.exists(ebf):
            try:
                with open(os.path.join(scratch, name + ".log"), "rb") as lf:
                    tail = lf.read()[-400:].decode("utf-8", "replace").strip().replace("\n", " | ")
            except OSError:
                tail = "no log"
            raise RuntimeError(f"galaxia produced no output for TIC {tic_id}; log tail: {tail}")
        npz = os.path.join(scratch, name + ".npz")
        # -I: the reader imports ebfpy only; keep cwd/user site out of sys.path.  A host without a dedicated
        # venv runs it with the current interpreter, whose numpy may live in the user site (Windows
        # store python): -E there (ignore PYTHON* variables) instead of -I.
        cmd = [install.python, "-E" if install.python == sys.executable else "-I", READER_SCRIPT, ebf, npz]
        if install.ebf_dir:
            cmd.append(install.ebf_dir)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"ebf reader failed for TIC {tic_id}: {r.stderr.strip()[:500]}")
        with np.load(npz) as z:
            arrays = {k: z[k] for k in RAW_COLUMNS}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    n_galaxia = int(len(arrays["rad"]))
    if tmag_ceiling is not None:
        arrays = apply_tmag_ceiling(arrays, tmag_ceiling)
    return {"tic_id": str(tic_id), "ra": ra, "dec": dec, "gal_l": l, "gal_b": b, "seed": seed,
            "arrays": arrays, "n_galaxia": n_galaxia}


def serve_document(arrays: Dict[str, np.ndarray], *, tic_id, ra: float, dec: float,
                   gal_l: float, gal_b: float, seed: int, raw_version: str,
                   mapping: str = MAPPING_VERSION) -> dict:
    """RAW arrays + a mapping version -> the served document of the step-5 data contract.
    Returns ``{"rows", "header", "payload", "sha256", "tmag", "seed", "gal_l", "gal_b"}``; the document
    is exactly what the previous one-step ``generate_population`` produced for the same field."""
    pop = map_population(arrays, mapping_flag(mapping))
    rows = canonical_rows(pop)
    header = {
        "tic_id": str(tic_id), "n_comp": len(rows), "field_area_deg2": FIELD_AREA_DEG2,
        "tmag_limit": TMAG_LIMIT, "source": raw_family(raw_version),
        "population_version": served_version(raw_version, mapping),
        "mapping_version": mapping, "seed": seed, "generator": raw_family(raw_version),
        "gal_l": round(gal_l, 6), "gal_b": round(gal_b, 6),
        "ra": round(float(ra), 6), "dec": round(float(dec), 6),
        "columns": ["Tmag", "Mact", "logg", "logTe", "[M/H]"],
        "decimals": list(ROW_DECIMALS),
    }
    payload, sha = build_document(rows, header)
    return {"rows": rows, "header": header, "payload": payload, "sha256": sha,
            "tmag": pop["Tmag"], "seed": seed, "gal_l": gal_l, "gal_b": gal_b}


def generate_population(install: GalaxiaInstall, tic_id, ra: float, dec: float, *,
                        cpu: Optional[int] = None, nice: int = 10,
                        timeout_s: float = 900.0, workdir: Optional[str] = None,
                        tmag_correction: Optional[bool] = None) -> dict:
    """``generate_raw`` + ``serve_document`` for one field (small batches / tests; the fleet stores
    the raw field and serves later)."""
    raw = generate_raw(install, tic_id, ra, dec, cpu=cpu, nice=nice, timeout_s=timeout_s, workdir=workdir)
    if tmag_correction is None:
        tmag_correction = tmag_correction_enabled()
    return serve_document(raw["arrays"], tic_id=tic_id, ra=raw["ra"], dec=raw["dec"], gal_l=raw["gal_l"],
                          gal_b=raw["gal_b"], seed=raw["seed"], raw_version=install.raw_version(),
                          mapping=mapping_version(tmag_correction))


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

_COLS = ("tic_id", "population_version", "ra", "dec", "gal_l", "gal_b", "n_comp",
         "field_area_deg2", "tmag_limit", "source", "generator", "seed",
         "mapping_version", "blob_sha256", "blob", "crosscheck_json")

# SQLite twin of scripts/migration/2026-10-17_fpp_background.sql (tests / file cache only).
SQLITE_DDL = """CREATE TABLE IF NOT EXISTS fpp_background (
  tic_id TEXT NOT NULL, population_version TEXT NOT NULL,
  ra REAL NOT NULL, dec REAL NOT NULL, gal_l REAL NOT NULL, gal_b REAL NOT NULL,
  n_comp INTEGER NOT NULL, field_area_deg2 REAL NOT NULL, tmag_limit REAL NOT NULL,
  source TEXT NOT NULL, generator TEXT NOT NULL, seed INTEGER NOT NULL,
  mapping_version TEXT NOT NULL,
  blob_sha256 TEXT NOT NULL, blob BLOB NOT NULL, crosscheck_json TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (tic_id, population_version))"""


class ReproducibilityError(RuntimeError):
    """A regenerated field does not reproduce the stored blob hash."""


def _key(tic_id) -> str:
    """Canonical TIC key ("007" == 7)."""
    return str(int(tic_id))


class FppBackgroundStore:
    """``fpp_background`` cache over an ``exonit.db_backend.Backend``.

    On PostgreSQL the table is created by the DB owner (migration file, step 6);
    this class never issues DDL there.  ``ensure_sqlite_schema()`` exists for
    tests and for a file cache.
    """

    def __init__(self, backend):
        self.backend = backend

    def ensure_sqlite_schema(self) -> None:
        from exonit.db_backend import PgDialect
        if isinstance(self.backend.dialect, PgDialect):
            raise RuntimeError("fpp_background DDL on PostgreSQL is applied by the DB owner (step 6)")
        self.backend.execute(SQLITE_DDL)

    def has(self, tic_id, population_version: str) -> bool:
        return self.backend.query_one(
            "SELECT 1 AS x FROM fpp_background WHERE tic_id = ? AND population_version = ?",
            (_key(tic_id), population_version)) is not None

    def versions_in_family(self, tic_id, family: str) -> List[str]:
        """Stored ``population_version`` strings of the TIC whose family is ``family`` (PK prefix
        scan on ``tic_id``; no blob is read)."""
        rows = self.backend.query(
            "SELECT population_version FROM fpp_background WHERE tic_id = ?", (_key(tic_id),))
        return sorted(r["population_version"] for r in rows
                      if family_of(r["population_version"]) == family)

    def get_by_family(self, tic_id, family: str) -> Optional[dict]:
        """Like :meth:`get`, for the single row of the TIC within ``family``.  No row -> None.  More
        than one (different ``src``/``plat``) -> logged at ERROR and None: never picked arbitrarily."""
        versions = self.versions_in_family(tic_id, family)
        if not versions:
            return None
        if len(versions) > 1:
            _log_ambiguous_family(tic_id, family, versions)
            return None
        return self.get(tic_id, versions[0])

    def put(self, tic_id, header: dict, payload: bytes, sha256: str,
            crosscheck: Optional[dict] = None) -> str:
        """Insert a (field, version) row.  Rows are immutable: returns ``"inserted"``, or
        ``"unchanged"`` when the stored row has the same hash.  A stored row with a DIFFERENT
        hash is a reproducibility failure: it is logged at ERROR and raised as
        ``ReproducibilityError``, never dropped silently."""
        key = _key(tic_id)
        old = self.backend.query_one(
            "SELECT blob_sha256 FROM fpp_background WHERE tic_id = ? AND population_version = ?",
            (key, header["population_version"]))
        if old is not None:
            if old["blob_sha256"] == sha256:
                return "unchanged"
            msg = (f"REPRODUCIBILITY FAILURE TIC {key} {header['population_version']}: stored "
                   f"{old['blob_sha256']} != regenerated {sha256}")
            logger.error(msg)
            raise ReproducibilityError(msg)
        row = (key, header["population_version"], header["ra"], header["dec"],
               header["gal_l"], header["gal_b"], header["n_comp"], header["field_area_deg2"],
               header["tmag_limit"], header["source"], header["generator"], header["seed"],
               header["mapping_version"], sha256, zlib.compress(payload, 9),
               None if crosscheck is None else json.dumps(crosscheck, sort_keys=True))
        self.backend.execute(self.backend.dialect.insert_ignore("fpp_background", _COLS), row)
        return "inserted"

    def get(self, tic_id, population_version: Optional[str] = None) -> Optional[dict]:
        """``{"payload": bytes, "sha256", "population_version", "n_comp", "crosscheck"}``
        or None.  ``population_version=None`` -> the most recently created version."""
        if population_version is None:
            row = self.backend.query_one(
                "SELECT * FROM fpp_background WHERE tic_id = ? ORDER BY created_at DESC LIMIT 1",
                (_key(tic_id),))
        else:
            row = self.backend.query_one(
                "SELECT * FROM fpp_background WHERE tic_id = ? AND population_version = ?",
                (_key(tic_id), population_version))
        if row is None:
            return None
        payload = zlib.decompress(bytes(row["blob"]))
        sha = hashlib.sha256(payload).hexdigest()
        if sha != row["blob_sha256"]:
            raise RuntimeError(f"fpp_background blob of TIC {tic_id} fails its sha256 check")
        cx = row["crosscheck_json"]
        return {"payload": payload, "sha256": sha, "population_version": row["population_version"],
                "n_comp": row["n_comp"], "crosscheck": json.loads(cx) if cx else None}


def serve_population(store: "FppBackgroundStore", tic_id) -> Optional[dict]:
    """The population served for ``tic_id`` (``FppBackgroundStore.get`` shape, carrying the row's own
    full ``population_version``) or None.  Family configured -> by-TIC family lookup; otherwise the
    legacy exact ``current_population_version()`` (pinned local install).  Single entry point of the
    route and the task builder (the readiness gate / drain counter use ``fetch_population_rows``).

    ``EXONIT_FPP_GALAXIA_STORE_INDEX`` set: the materialized galaxia-rs store
    (``exonit.fpp_galaxia_store``) is the ONLY source and ``store`` is not consulted; an unusable
    store raises ``GalaxiaStoreConfigError`` (refused), a TIC absent from it is None (never served
    from the ``fpp_background`` table)."""
    from exonit import fpp_galaxia_store
    galaxia = fpp_galaxia_store.active_store()
    if galaxia is not None:
        return galaxia.get_by_family(tic_id, galaxia.family)
    family = population_family()
    if family:
        return store.get_by_family(tic_id, family)
    version = current_population_version()
    return store.get(tic_id, version) if version else None


def fetch_population_rows(query, tic_ids: Iterable, *, version: Optional[str] = None,
                          family: Optional[str] = None, chunk_size: int = 5000) -> Dict[str, dict]:
    """Light rows (``tic_id, population_version, source, crosscheck_json``; no blob) of the served
    population per TIC, ``{tic_key: row}``.  ``query(sql, params)`` is ``db._query``.  Exact mode
    (``version``) or family mode (``family``); a TIC with no row, or with SEVERAL rows in the family,
    is absent (the latter is logged at ERROR, never resolved arbitrarily).  With the galaxia store
    switch on, the rows come from its index only (``query`` is not used)."""
    from exonit import fpp_galaxia_store
    galaxia = fpp_galaxia_store.active_store()
    if galaxia is not None:
        return galaxia.fetch_rows(tic_ids, family=family, version=version, chunk_size=chunk_size)
    ids = sorted({_key(t) for t in tic_ids})
    out: Dict[str, dict] = {}
    cols = "tic_id, population_version, source, crosscheck_json"
    for i in range(0, len(ids), chunk_size):
        batch = ids[i:i + chunk_size]
        ph = ",".join("?" * len(batch))
        if family:
            rows = query(f"SELECT {cols} FROM fpp_background WHERE tic_id IN ({ph})", tuple(batch))
            by_tic: Dict[str, list] = {}
            for r in rows:
                if family_of(r["population_version"]) == family:
                    by_tic.setdefault(str(r["tic_id"]), []).append(r)
            for tic, hits in by_tic.items():
                if len(hits) == 1:
                    out[tic] = hits[0]
                else:
                    _log_ambiguous_family(tic, family, [h["population_version"] for h in hits])
        else:
            for r in query(f"SELECT {cols} FROM fpp_background WHERE population_version = ? "
                           f"AND tic_id IN ({ph})", (version, *batch)):
                out[str(r["tic_id"])] = r
    return out


# --------------------------------------------------------------------------
# Batch prefetch
# --------------------------------------------------------------------------

def prefetch_backgrounds(store: FppBackgroundStore, install: GalaxiaInstall,
                         coords: Dict[str, tuple], *, cores: Sequence[int],
                         gaia_tmag_fn: Optional[Callable[[str, float, float], Sequence[float]]] = None,
                         tmag_correction: Optional[bool] = None,
                         nice: int = 10, progress: Optional[Callable[[str, str], None]] = None) -> dict:
    """Generate and cache the populations of ``coords`` = ``{tic: (ra, dec)}``.

    Skips fields already stored for the pinned version.  With a population family configured
    (``EXONIT_FPP_POPULATION_FAMILY``) a field is skipped as soon as the TIC has ANY row in the family
    of the pinned version, so a build with a different ``src``/``plat`` never inserts a second row in
    that family (which would make the TIC ambiguous, hence refused).  One galaxia process per
    entry of ``cores`` (each pinned with ``taskset``); the store is written from
    the calling thread only.  ``gaia_tmag_fn(tic, ra, dec)`` supplies the Gaia Tmag list
    for the R8 cross-check (e.g. ``gaia_tmag_tap``); its result is stored in ``crosscheck_json``.
    If it raises, the population is still stored, with ``gaia_available: false`` and the error
    recorded, and the tic is listed in ``gaia_failed`` (the check did NOT run).  Failures of
    Galaxia are collected, not raised, so one bad field does not stop a batch; the caller must
    look at ``failed`` / ``gaia_failed`` / ``alarms`` BEFORE the FPP phase starts.
    Returns ``{"version", "generated", "cached", "failed": {tic: reason}, "alarms": [tic],
    "gaia_failed": {tic: reason}}``.
    """
    from exonit import fpp_galaxia_store
    fpp_galaxia_store.refuse_if_switch_on("prefetch_backgrounds")
    version = install.population_version(tmag_correction)
    served_family = population_family()        # misconfigured -> raises: prefetch nothing
    family = family_of(version) if served_family else None
    if served_family and family != served_family:
        logger.warning("fpp_background prefetch: install version %s is outside the served family %s "
                       "(%s); its rows will not be served", version, served_family, FAMILY_ENV)
    todo = []
    cached = 0
    for tic, (ra, dec) in coords.items():
        if store.versions_in_family(tic, family) if family else store.has(tic, version):
            cached += 1
        else:
            todo.append((_key(tic), float(ra), float(dec)))
    free: "queue.Queue[int]" = queue.Queue()
    for c in cores:
        free.put(int(c))

    def job(item):
        tic, ra, dec = item
        cpu = free.get()
        try:
            return tic, generate_population(install, tic, ra, dec, cpu=cpu, nice=nice,
                                            tmag_correction=tmag_correction), None
        except Exception as exc:  # noqa: BLE001 - reported per field
            return tic, None, f"{type(exc).__name__}: {exc}"
        finally:
            free.put(cpu)

    res = {"version": version, "generated": 0, "cached": cached, "failed": {}, "alarms": [],
           "gaia_failed": {}}
    with ThreadPoolExecutor(max_workers=max(1, len(cores))) as ex:
        for tic, gen, err in ex.map(job, todo):
            if err is not None:
                logger.warning("fpp_background TIC %s failed: %s", tic, err)
                res["failed"][tic] = err
                if progress:
                    progress(tic, "failed")
                continue
            gaia = None
            gaia_err = None
            if gaia_tmag_fn is not None:
                try:
                    gaia = gaia_tmag_fn(tic, gen["header"]["ra"], gen["header"]["dec"])
                except Exception as exc:  # noqa: BLE001 - recorded, not silent
                    gaia_err = f"{type(exc).__name__}: {exc}"
                    res["gaia_failed"][tic] = gaia_err
                    logger.warning("fpp_background TIC %s: Gaia cross-check unavailable: %s", tic, gaia_err)
            cx = crosscheck_counts(gen["tmag"], gaia)
            if gaia_err:
                cx["gaia_error"] = gaia_err
            if cx["alarm"]:
                res["alarms"].append(tic)
                logger.warning("fpp_background TIC %s R8 ALARM: %s", tic, "; ".join(cx["alarm_reasons"]))
            store.put(tic, gen["header"], gen["payload"], gen["sha256"], cx)
            res["generated"] += 1
            if progress:
                progress(tic, "ok")
    return res


def verify_backgrounds(store: FppBackgroundStore, install: GalaxiaInstall, tics: Iterable,
                       *, cores: Sequence[int], tmag_correction: Optional[bool] = None,
                       nice: int = 10) -> dict:
    """Regenerate stored fields and compare ``blob_sha256`` (the step-5 acceptance: same cell ->
    byte-identical blob).  Returns ``{"checked", "ok", "mismatches": [{tic, stored, regenerated}],
    "missing": [tic], "failed": {tic: reason}}``; every mismatch is logged at ERROR."""
    version = install.population_version(tmag_correction)
    res = {"version": version, "checked": 0, "ok": 0, "mismatches": [], "missing": [], "failed": {}}
    free: "queue.Queue[int]" = queue.Queue()
    for c in cores:
        free.put(int(c))
    items = []
    for tic in tics:
        hit = store.get(tic, version)
        if hit is None:
            res["missing"].append(_key(tic))
            continue
        hdr = json.loads(hit["payload"])["header"]
        items.append((_key(tic), hdr["ra"], hdr["dec"], hit["sha256"]))

    def job(item):
        tic, ra, dec, stored = item
        cpu = free.get()
        try:
            g = generate_population(install, tic, ra, dec, cpu=cpu, nice=nice,
                                    tmag_correction=tmag_correction)
            return tic, stored, g["sha256"], None
        except Exception as exc:  # noqa: BLE001
            return tic, stored, None, f"{type(exc).__name__}: {exc}"
        finally:
            free.put(cpu)

    with ThreadPoolExecutor(max_workers=max(1, len(cores))) as ex:
        for tic, stored, new, err in ex.map(job, items):
            if err is not None:
                res["failed"][tic] = err
                continue
            res["checked"] += 1
            if new == stored:
                res["ok"] += 1
            else:
                logger.error("REPRODUCIBILITY FAILURE TIC %s: stored %s != regenerated %s", tic, stored, new)
                res["mismatches"].append({"tic": tic, "stored": stored, "regenerated": new})
    return res
