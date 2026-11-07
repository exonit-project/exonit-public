"""Physical transit model + FITS writer for the full-pipeline injection harness.

Design (spec 2026-10-15, section 1):
  * quadratic limb-darkened transit, circular orbit, a/R* from the stellar
    density, impact parameter drawn as U(0, 1+k);
  * the model is MULTIPLIED into every flux column of the original FITS
    (raw, pre-detrend, quality-flagged cadences included) so the whole
    production worker sees the injection as it would see a real planet;
  * counter-based Philox RNG indexed by (TIC, draw index, variant) so adding
    hosts/draws never changes earlier draws.

Model note (finding, 2026-10-15): ``worker/fpp_native._transit_flux_ring``
(100 midpoint rings) was measured against a 4000-ring reference and is off by
up to 1e-4 absolute (~9 % of the depth at k = 0.02, ~1 % at k = 0.1).  That is
far outside the spec's 2e-6 acceptance, so this module carries its own
Mandel-Agol-class integrator: the exact-arc ring integral with a Gauss-Legendre
quadrature in an angle substitution that removes the square-root end-point
singularities (error < 1e-9 absolute, see tests/ir/test_ir_transit_inject.py).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

G_CGS = 6.67430e-8
RSUN_CM = 6.957e10
MSUN_G = 1.98847e33
REARTH_RSUN = 6378.1366e5 / RSUN_CM
DAY_S = 86400.0

P_MIN, P_MAX = 0.5, 200.0
RP_MIN, RP_MAX = 0.5, 16.0
K_MAX = 0.5

_LDC_PATH = Path(__file__).resolve().parents[2] / "worker" / "ldc_tess.csv"
_ldc = None


# ---------------------------------------------------------------- limb darkening
def _load_ldc():
    global _ldc
    if _ldc is None:
        rows = list(csv.DictReader(open(_LDC_PATH)))
        _ldc = {
            "Z": np.array([float(r["Z"]) for r in rows]),
            "Teff": np.array([float(r["Teff"]) for r in rows]),
            "logg": np.array([float(r["logg"]) for r in rows]),
            "u1": np.array([float(r["aLSM"]) for r in rows]),
            "u2": np.array([float(r["bLSM"]) for r in rows]),
        }
    return _ldc


def get_ldc(teff, logg, z=0.0):
    """Nearest-grid Claret (2017) TESS quadratic LD; same lookup as
    ``worker/fpp_native.get_ldc``.  Returns (u1, u2, fallback_flag)."""
    if teff is None or logg is None or not (math.isfinite(teff) and math.isfinite(logg)):
        return 0.40, 0.26, True
    d = _load_ldc()
    bz = d["Z"][np.argmin(np.abs(d["Z"] - z))]
    bt = d["Teff"][np.argmin(np.abs(d["Teff"] - teff))]
    bg = d["logg"][np.argmin(np.abs(d["logg"] - logg))]
    m = (d["Z"] == bz) & (d["Teff"] == bt) & (d["logg"] == bg)
    if np.any(m):
        return float(d["u1"][m][0]), float(d["u2"][m][0]), False
    return 0.40, 0.26, True


# ---------------------------------------------------------------- geometry
def stellar_density_cgs(mass_msun, radius_rsun):
    return mass_msun * MSUN_G / (4.0 / 3.0 * math.pi * (radius_rsun * RSUN_CM) ** 3)


def mass_from_logg(logg, radius_rsun):
    """M = g R^2 / G, in solar masses."""
    return (10.0 ** logg) * (radius_rsun * RSUN_CM) ** 2 / G_CGS / MSUN_G


def a_over_rstar(period_d, rho_cgs):
    """Kepler III for a circular orbit, planet mass neglected."""
    return (G_CGS * rho_cgs * (period_d * DAY_S) ** 2 / (3.0 * math.pi)) ** (1.0 / 3.0)


def duration_t14(period_d, a_rs, k, b):
    """Total duration T14 (days) of a circular orbit; 0 when not transiting."""
    arg_num = (1.0 + k) ** 2 - b * b
    if arg_num <= 0:
        return 0.0
    sin_i = math.sqrt(max(1.0 - (b / a_rs) ** 2, 1e-12))
    s = math.sqrt(arg_num) / (a_rs * sin_i)
    return period_d / math.pi * math.asin(min(s, 1.0))


def n_supersample(exp_days, k, t14):
    """Sub-steps so that each is <= 1/3 of the ingress time (spec 1.1)."""
    ingress = max(k * t14 / (1.0 + k), 1e-6)
    return int(np.clip(math.ceil(exp_days / (ingress / 3.0)), 1, 15))


# ---------------------------------------------------------------- flux model
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)
_PHI = (_GL_X + 1.0) * (math.pi / 2.0)          # nodes on [0, pi]
_PHI_W = _GL_W * (math.pi / 2.0)


def _full_cover(s_a, u1, u2):
    """Integral of I(mu) 2 mu dmu over the fully blocked inner disc, s = 1-mu."""
    return (2.0 * s_a - (1.0 + u1) * s_a ** 2 + (2.0 / 3.0) * (u1 - u2) * s_a ** 3
            + 0.5 * u2 * s_a ** 4)


def quad_ld_flux(z, k, u1, u2):
    """Normalised flux of a dark disc (radius ratio k) at projected separation
    z (stellar radii), quadratic limb darkening.  Vectorised over z."""
    z = np.atleast_1d(np.asarray(z, dtype=np.float64))
    out = np.ones_like(z)
    omega = 1.0 - u1 / 3.0 - u2 / 6.0
    active = np.nonzero(z < 1.0 + k)[0]
    if active.size == 0:
        return out
    za = np.maximum(z[active], 1e-9)
    r1 = np.abs(za - k)
    r2 = np.minimum(za + k, 1.0)
    blocked = np.zeros_like(za)
    # fully covered inner disc (planet larger than the offset)
    cover = np.minimum(np.maximum(k - za, 0.0), 1.0)
    s_a = 1.0 - np.sqrt(1.0 - cover ** 2)
    blocked += _full_cover(s_a, u1, u2)
    # partially covered annulus [r1, r2]
    ok = r2 > r1
    if np.any(ok):
        zz, rr1, rr2 = za[ok], r1[ok], r2[ok]
        c = 0.5 * (rr1 + rr2)[:, None]
        h = 0.5 * (rr2 - rr1)[:, None]
        r = c - h * np.cos(_PHI)[None, :]
        dr_dphi = h * np.sin(_PHI)[None, :]
        zc = zz[:, None]
        arg = (r * r + zc * zc - k * k) / (2.0 * r * zc)
        frac = np.arccos(np.clip(arg, -1.0, 1.0)) / math.pi
        mu = np.sqrt(np.clip(1.0 - r * r, 0.0, 1.0))
        s = 1.0 - mu
        inten = 1.0 - u1 * s - u2 * s * s
        integrand = inten * frac * 2.0 * r * dr_dphi
        blocked[ok] += (integrand * _PHI_W[None, :]).sum(axis=1)
    out[active] = 1.0 - blocked / omega
    return out


def separation_circular(t, t0, period, a_rs, b):
    """Projected separation z(t) for e = 0: z = a sqrt(sin^2 th + cos^2 i cos^2 th)."""
    th = 2.0 * math.pi * (t - t0) / period
    cos_i = b / a_rs
    return a_rs * np.sqrt(np.sin(th) ** 2 + (cos_i * np.cos(th)) ** 2)


def model_flux(time, period, t0, k, b, a_rs, u1, u2, exp_days=0.0, n_ss=1):
    """Exposure-integrated transit model on ``time`` (non-finite times -> 1.0).

    Only the sub-samples that can overlap the stellar disc are integrated.
    The planet is on the near side only (cos th > 0 half of the orbit): the
    secondary-eclipse half has z large because the planet is behind the star
    and is explicitly excluded.
    """
    time = np.asarray(time, dtype=np.float64)
    out = np.ones(time.shape, dtype=np.float64)
    fin = np.isfinite(time)
    if not fin.any():
        return out
    offs = ((np.arange(n_ss) + 0.5) / n_ss - 0.5) * exp_days if n_ss > 1 else np.array([0.0])
    acc = np.zeros(time.shape, dtype=np.float64)
    for o in offs:
        tt = np.where(fin, time + o, 0.0)
        ph = ((tt - t0) / period + 0.5) % 1.0 - 0.5            # -0.5..0.5, 0 = transit
        z = separation_circular(tt, t0, period, a_rs, b)
        near = np.abs(ph) < 0.25                               # near-side half only
        sel = fin & near & (z < 1.0 + k)
        f = np.ones(time.shape, dtype=np.float64)
        if sel.any():
            f[sel] = quad_ld_flux(z[sel], k, u1, u2)
        acc += f
    out = acc / len(offs)
    out[~fin] = 1.0
    return out


# ---------------------------------------------------------------- truth
@dataclass
class Truth:
    truth_id: str
    tic_id: str
    variant: str
    draw_index: int
    seed: int
    source: str                  # spoc | qlp | tglc
    period: float
    t0: float
    rp_rearth: float
    rstar_rsun: float
    k: float
    b: float
    a_rs: float
    inc_deg: float
    t14: float
    depth_ppm: float             # model maximum depth (1 - min flux), ppm
    depth_mean_ppm: float        # mean depth over |t-T0| < T14/2 (what a box fit sees)
    u1: float
    u2: float
    ld_fallback: bool
    rho_cgs: float
    rho_source: str
    grazing: bool
    n_ss: int
    exp_days: float
    teff: float | None = None
    logg: float | None = None
    mstar_msun: float | None = None
    tmag: float | None = None
    stellar_flag: str = ""
    n_transits_observed: int | None = None
    snr_exp: float | None = None
    sigma_pt: float | None = None
    span_days: float | None = None
    by_design_unrecoverable: bool | None = None
    unrecoverable_reason: str = ""

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True)


def opaque_hash(*parts) -> str:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return h[:16]


def rng_for(seed, tic_id, draw_index, variant_code=0):
    """Counter-based RNG: independent of how many other hosts/draws exist."""
    ctr = np.array([int(tic_id), int(draw_index), int(variant_code), 0], dtype=np.uint64)
    return np.random.Generator(np.random.Philox(key=int(seed), counter=ctr))


def draw_physical(seed, tic_id, draw_index, rstar_rsun, rho_cgs, t_first,
                  variant_code=0):
    """Draw (P, Rp, b, T0) for one injection and derive the geometry.

    Fixed draw order: logP, logRp (redrawn until k < K_MAX, max 200 times,
    sequential on the same stream), b, T0.  Returns a dict of physical params.
    """
    g = rng_for(seed, tic_id, draw_index, variant_code)
    period = float(math.exp(g.uniform(math.log(P_MIN), math.log(P_MAX))))
    for _ in range(200):
        rp = float(math.exp(g.uniform(math.log(RP_MIN), math.log(RP_MAX))))
        k = rp * REARTH_RSUN / rstar_rsun
        if k < K_MAX:
            break
    else:
        raise ValueError("no Rp with k < 0.5 for this star")
    b = float(g.uniform(0.0, 1.0 + k))
    t0 = float(t_first + g.uniform(0.0, period))
    a_rs = a_over_rstar(period, rho_cgs)
    return dict(period=period, rp_rearth=rp, k=k, b=b, t0=t0, a_rs=a_rs)


# ---------------------------------------------------------------- FITS columns
FAMILIES = {
    # Flux columns the production readers use + siblings.  Worker (lk.read):
    # PDCSAP_FLUX (SPOC); SAP_FLUX for QLP (lightkurve's QLP reader default --
    # NOT KSPSAP; verified 2026-10-15, lc.meta FLUX_ORIGIN = sap_flux); and
    # cal_aper_flux (TGLC, manual branch).  Postprocess load_lc prefers
    # kspsap_flux, sap_flux, cal_aper_flux.  Columns absent in a given file
    # version (QLP v02 has DET_* instead of KSPSAP_*) are skipped.
    "spoc": dict(time="TIME", flux=["SAP_FLUX", "PDCSAP_FLUX"],
                 err_abs=["SAP_FLUX_ERR", "PDCSAP_FLUX_ERR"], quality="QUALITY",
                 prod_flux="PDCSAP_FLUX"),
    "qlp": dict(time="TIME",
                flux=["SAP_FLUX", "KSPSAP_FLUX", "KSPSAP_FLUX_SML", "KSPSAP_FLUX_LAG",
                      "DET_FLUX", "DET_FLUX_SML", "DET_FLUX_LAG", "SYS_RM_FLUX"],
                err_abs=[], quality="QUALITY", prod_flux="SAP_FLUX"),
    "tglc": dict(time="time", flux=["psf_flux", "aperture_flux", "cal_psf_flux", "cal_aper_flux"],
                 err_abs=[], quality=None, prod_flux="cal_aper_flux"),
}


def detect_family(colnames):
    low = {c.lower() for c in colnames}
    if "cal_aper_flux" in low:
        return "tglc"
    if "kspsap_flux" in low or "det_flux" in low:
        return "qlp"
    if "pdcsap_flux" in low:
        return "spoc"
    raise ValueError(f"unknown FITS flavour, columns={sorted(low)[:12]}")


def _col(names, wanted):
    m = {n.lower(): n for n in names}
    return m.get(wanted.lower())


def valid_cadence_mask(hdu_data, family):
    """Cadences the production quality mask keeps (used for evaluation only)."""
    names = hdu_data.columns.names
    fam = FAMILIES[family]
    t = np.asarray(hdu_data[_col(names, fam["time"])], dtype=np.float64)
    f = np.asarray(hdu_data[_col(names, fam["prod_flux"])], dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(f)
    if family == "tglc":
        qa = np.asarray(hdu_data[_col(names, "TESS_flags")], dtype=np.int64)
        qb = np.asarray(hdu_data[_col(names, "TGLC_flags")], dtype=np.int64)
        ok &= (qa == 0) & (qb == 0) & (f > 0)
    else:
        q = np.asarray(hdu_data[_col(names, "QUALITY")], dtype=np.int64)
        # lightkurve 'default' bitmask is 175; QLP uses bit 4096 benignly
        ok &= (q & 175) == 0
    return ok


def scale_columns(hdu_data, family, m, include_err=True):
    """Multiply flux (and absolute-error) columns by the model ``m`` in place."""
    names = hdu_data.columns.names
    fam = FAMILIES[family]
    done = []
    for key in ("flux", "err_abs") if include_err else ("flux",):
        for cname in fam[key]:
            real = _col(names, cname)
            if real is None:
                continue
            arr = hdu_data[real]
            hdu_data[real] = (np.asarray(arr, dtype=np.float64) * m).astype(arr.dtype)
            done.append(real)
    return done


def read_time(hdu_data, family):
    names = hdu_data.columns.names
    return np.asarray(hdu_data[_col(names, FAMILIES[family]["time"])], dtype=np.float64)


def inject_hdu_data(hdu_data, family, ephem):
    """Compute the model on one file and scale its columns.  ``ephem`` = Truth
    fields (period, t0, k, b, a_rs, u1, u2, t14; n_ss optional -> derived from
    this file's cadence).  Returns per-file stats."""
    t = read_time(hdu_data, family)
    fin = t[np.isfinite(t)]
    cad = float(np.median(np.diff(fin))) if fin.size > 2 else 0.0
    n_ss = ephem.get("n_ss") or n_supersample(cad, ephem["k"], ephem["t14"])
    m = model_flux(t, ephem["period"], ephem["t0"], ephem["k"], ephem["b"], ephem["a_rs"],
                   ephem["u1"], ephem["u2"], exp_days=cad, n_ss=n_ss)
    valid = valid_cadence_mask(hdu_data, family)
    scale_columns(hdu_data, family, m)
    return dict(cadence_days=cad, n_ss=n_ss, n_cadences=int(t.size), n_valid=int(valid.sum()),
                n_intransit_valid=int(((m < 1.0) & valid).sum()),
                min_flux=float(m.min()))


def mean_depth_ppm(period, t0, k, b, a_rs, u1, u2, t14, n=401):
    """Mean fractional depth over the central T14 window (box-fit equivalent)."""
    t = t0 + np.linspace(-0.5, 0.5, n) * t14
    m = model_flux(t, period, t0, k, b, a_rs, u1, u2)
    return float((1.0 - m).mean() * 1e6), float((1.0 - m.min()) * 1e6)


def file_cadence_info(hdu_data, family):
    """Times of the cadences the production quality mask keeps, the median
    cadence, and the normalised production flux of those cadences (for the
    robust point-to-point scatter)."""
    names = hdu_data.columns.names
    t = read_time(hdu_data, family)
    valid = valid_cadence_mask(hdu_data, family)
    f = np.asarray(hdu_data[_col(names, FAMILIES[family]["prod_flux"])], dtype=np.float64)
    tf = t[np.isfinite(t)]
    cad = float(np.median(np.diff(tf))) if tf.size > 2 else 0.0
    tv, fv = t[valid], f[valid]
    med = np.median(fv) if fv.size else 1.0
    return tv, fv / med, cad
