"""Native FPP calculation — no triceratops/pytransit dependency.

Reimplements the False Positive Probability calculation from
Giacalone & Dressing (2020) / TRICERATOPS using only:
  numpy, scipy, numba, requests

Analytic Mandel–Agol transit model and exact Kepler geometry, CPU f64.

Background handling: observed TIC neighbours for NTP/NEB, plus the immutable
server-supplied Galaxia population for DTP/DEB/BTP/BEB. No worker catalogue
query, TRILEGAL CSV loader or population synthesis. Missing/invalid/alarmed
populations refuse FPP and request requeue. cc1 is applied only server-side.

FPP thresholds (standard in literature):
  - FPP < 0.015 → "validated"
  - FPP < 0.5   → "candidate"
  - FPP >= 0.5  → "likely_fp"

Backend: CPU only (numba, f64 reference). The GPU dispatch that used to
probe the CUDA / Metal / OpenCL kernel modules is retired — those kernels
carry the OLD geometry and cannot be made bit-exact, so no code path here
can reach a GPU (plan decision (e)).
"""


import os
import sys

import math
import json
import hashlib
import logging
import numpy as np

from scipy.stats import beta as beta_dist, powerlaw as powerlaw_dist
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import erf

from numba_serial import numba_parallel_guard
from phase_fold import phase_fold

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Pure-python Monte Carlo is orders of magnitude slower — never silent.
    logging.getLogger("exonit").warning(
        "numba not available: FPP Monte Carlo will run in PURE PYTHON "
        "(orders of magnitude slower). Install numba in the worker venv.")
    # Create no-op decorators mimicking numba.njit
    class _FakeNumba:
        @staticmethod
        def njit(*args, **kwargs):
            if len(args) == 1 and callable(args[0]):
                return args[0]
            def decorator(func):
                return func
            return decorator
        @staticmethod
        def prange(*a, **kw):
            return range(*a)
    numba = _FakeNumba()

# ============================================================
# FPP backend: CPU only (plan decision (e), step 5.3)
# ============================================================
# FPP is a CPU workload BY DESIGN. The GPU kernel modules (CUDA / Metal
# kernel / legacy Metal array / OpenCL) carry the OLD transit/eclipse
# geometry and cannot be made bit-exact with the corrected native model, so
# every dispatch path to them is removed here: no code path in this module can
# reach a GPU kernel. The kernel modules themselves stay on disk untouched.
# The two globals below are kept (always None) only so external callers that
# read them keep working.

_fpp_gpu_backend = None
_fpp_gpu_info = None


def _init_fpp_gpu():
    """No-op: FPP never dispatches to a GPU backend (plan decision (e)).
    Runs at import time like before, but imports NO GPU kernel module. The
    only thing it does is warn once when the operator asks for GPU backends
    via EXONIT_FPP_GPU_BACKENDS — the variable no longer has any effect, and
    silently ignoring it would hide that from the operator.

    """
    backends_env = os.environ.get("EXONIT_FPP_GPU_BACKENDS", "").strip().lower()
    if backends_env not in ("", "none"):
        logging.getLogger("exonit").warning(
            "EXONIT_FPP_GPU_BACKENDS is ignored: FPP is CPU-only by design "
            "(fix plan decision (e))")


def _configure_fpp_cpu_cores(n_cores=None):
    """Set numba thread count for FPP computation.

    If n_cores is None, defaults to floor(0.75 * cpu_count) — the same
    75%-of-CPUs formula agent.py:207-223 already uses to honour the project
    rule 2 ("never use all of a worker's CPUs, leave 25% free"). The previous default of max(1, cpu_count - 2) only
    leaves 2 fixed cores free regardless of host size, which violates the
    25%-free rule on anything with >8 cores (87.5% used at 16 cores,
    93.75% at 32) — found in the 2026-07-02 cleanup review (B7). Explicit
    config values (``fpp_cpu_cores``) remain authoritative — this default
    only applies when the caller passes n_cores=None.

    The effective value is clamped to numba's thread pool size
    (``NUMBA_NUM_THREADS``, fixed at numba import time from the env
    var worker.py sets at boot from ``config.cpu_threads``). That cap
    is hard — ``numba.set_num_threads`` rejects anything above it with
    ``ValueError: The number of threads must be between 1 and N``.
    Before the clamp was added, a config with ``fpp_cpu_cores > cpu_threads``
    silently fell back to numba's default and logged a confusing
    "FPP CPU core config failed" warning.

    Returns the thread count actually applied, or ``None`` if numba is
    unavailable so the caller can log accurately.
    """
    if not HAS_NUMBA:
        return None
    total = os.cpu_count() or 4
    numba_cap = getattr(numba.config, "NUMBA_NUM_THREADS", total)
    if n_cores is None:
        n_cores = max(1, math.floor(0.75 * total))
    effective = max(1, min(n_cores, numba_cap, total))
    numba.set_num_threads(effective)
    return effective


def get_fpp_backend_info():
    """Return (backend_name, info_string) for UI display.

    FPP is CPU-only (plan decision (e)), so this always reports the numba CPU
    path — there is no backend probing and no GPU to annotate.
    """
    if GL_TABLES_ERROR:
        return "unavailable", f"FPP unavailable: {GL_TABLES_ERROR}"
    cores = numba.config.NUMBA_NUM_THREADS if HAS_NUMBA else 1
    return "cpu", f"CPU ({cores} cores, numba, f64 reference)"


def disable_gpu():
    """No-op kept for its name: external code (worker configs, tests) calls it.

    There is nothing to disable any more — FPP never dispatches to a GPU
    (plan decision (e)) — so this only exists so those call sites keep working.
    """


# ============================================================
# Constants
# ============================================================
# Exact binary literals (float.fromhex) of the astropy 7.2.2 cgs values, as
# used by upstream TRICERATOPS — see plan fix N1. Using the hex form pins the
# exact double representation instead of a rounded decimal literal.
G_cgs = float.fromhex('0x1.1ea8b45edf98bp-24')      # cm³/(g·s²)
Msun = float.fromhex('0x1.88250b3b66e99p+110')      # g
Rsun = float.fromhex('0x1.032b1f4800000p+36')       # cm
Rearth = float.fromhex('0x1.3021aa8000000p+29')     # cm
au = float.fromhex('0x1.b363196a46000p+43')         # cm (astronomical unit) — used by _lnprior_bound_TP/EB
pi = np.pi
ln2pi = np.log(2 * pi)


def _check_constants_vs_astropy():
    """Assert the constants above match astropy's cgs values bit-for-bit.

    astropy is an optional dependency here, so it is imported lazily and the
    check is skipped silently when it is absent.
    """
    try:
        from astropy.constants import G, M_sun, R_earth, R_sun, au as au_const
    except ImportError:
        return
    assert G_cgs == float(G.cgs.value), (G_cgs, float(G.cgs.value))
    assert Msun == float(M_sun.cgs.value), (Msun, float(M_sun.cgs.value))
    assert Rearth == float(R_earth.cgs.value), (Rearth, float(R_earth.cgs.value))
    assert Rsun == float(R_sun.cgs.value), (Rsun, float(R_sun.cgs.value))
    assert au == float(au_const.cgs.value), (au, float(au_const.cgs.value))


_check_constants_vs_astropy()


# ============================================================
# FPP switches
# ============================================================
# Plan decision (o): the model is a switch set, not a single behaviour. The
# parity set reproduces every documented upstream quirk (needed to certify
# against official TRICERATOPS); the production set is the corrected
# behaviour and is the default. The full set is stored with every result
# (see calc_fpp_native) together with FPP_MODEL_VERSION, so a stored FPP can
# always be attributed to the model that produced it.
# 3 = FFI geometry v3 (docs/fpp-ffi-geometry-v3-design.md): per-product dilution convention (D8), product apertures,
# K-epoch time-averaged geometry, smooth flux ratios and neighbour viability weights.  The server gates every row with
# fpp_model_version >= 2 as "new model" (exonit/fpp_model.py), so 3 needs no new gate.
FPP_MODEL_VERSION = 3   # integer; the shipped model was 1

_SWITCH_NAMES = ("FPP_SECONDARY_GEOMETRY", "FPP_ORBIT_PHYSICS", "FPP_K1_NUDGE",
                 "FPP_Q_BRANCH", "FPP_EXPOSURE", "FPP_NORM", "FPP_BACKGROUND_INDEX", "FPP_BACKGROUND_PRIOR", "FPP_BACKGROUND_Q",
                 "FPP_GEOMETRY", "FPP_ADAPTIVE_N", "FPP_EPHEM_REFINE", "FPP_UNSTABLE_FLAG")

_PARITY_SET = {"FPP_SECONDARY_GEOMETRY": "upstream",
               "FPP_ORBIT_PHYSICS": "upstream",
               "FPP_K1_NUDGE": "upstream",
               "FPP_Q_BRANCH": "upstream",
               "FPP_EXPOSURE": "upstream_2min_x20",
               "FPP_NORM": "upstream", "FPP_BACKGROUND_INDEX": "upstream",
               "FPP_BACKGROUND_PRIOR": "upstream", "FPP_BACKGROUND_Q": "target",
               "FPP_GEOMETRY": "upstream", "FPP_ADAPTIVE_N": "off", "FPP_EPHEM_REFINE": "off",
               "FPP_UNSTABLE_FLAG": "off"}

_PRODUCTION_SET = {"FPP_SECONDARY_GEOMETRY": "consistent",
                   "FPP_ORBIT_PHYSICS": "roche",
                   "FPP_K1_NUDGE": "exact",
                   "FPP_Q_BRANCH": "correct",
                   "FPP_EXPOSURE": "cadence",
                   # FPP_NORM stays "upstream" in production too: the corrected
                   # data-space normalisation ("data", plan (r) / D16) is
                   # conditional on its two exit criteria (fr_i sensitivity
                   # analysis + adaptive N per decision), which are not met yet.
                   "FPP_NORM": "upstream", "FPP_BACKGROUND_INDEX": "correct",
                   "FPP_BACKGROUND_PRIOR": "circular", "FPP_BACKGROUND_Q": "target",
                   # smooth D5 geometry (geometry v3, the owner 2026-10-17 "4A"): product apertures, K-epoch average, flux
                   # ratios integrated over the calibration uncertainty, neighbour viability as a weight in [0, 1]
                   "FPP_GEOMETRY": "smooth",
                   # MC-noise policy of decision (g): sequential refinement + 3-sigma `validated` + `borderline`
                   "FPP_ADAPTIVE_N": "on",
                   # bounded local (P, t0) refinement of a box-start BLS ephemeris before the fold (section 8b,
                   # docs/reviews/2026-10-24-fpp-ephem-refine-report.md); "off" passes P / t0 through untouched
                   "FPP_EPHEM_REFINE": "on",
                   # honesty flag: dominant-scenario ESS of the reported stage < MC_FUTILE_ESS_DOM -> `fpp_unstable` + verdict
                   # cap (likely_fp / validated -> candidate); the FPP value itself is untouched (docs/superpowers/specs/2026-10-30-fpp-unstable-flag.md)
                   "FPP_UNSTABLE_FLAG": "on"}

_ALLOWED = {"FPP_SECONDARY_GEOMETRY": {"upstream", "consistent"},
            "FPP_ORBIT_PHYSICS": {"upstream", "roche"},
            "FPP_K1_NUDGE": {"upstream", "exact"},
            "FPP_Q_BRANCH": {"upstream", "correct"},
            "FPP_EXPOSURE": {"upstream_2min_x20", "cadence"},
            "FPP_NORM": {"upstream", "data"},
            "FPP_BACKGROUND_INDEX": {"upstream", "correct"},
            "FPP_BACKGROUND_PRIOR": {"upstream", "circular"},
            "FPP_BACKGROUND_Q": {"target", "host"},
            "FPP_GEOMETRY": {"upstream", "smooth"},
            "FPP_ADAPTIVE_N": {"off", "on"},
            "FPP_EPHEM_REFINE": {"off", "on"},
            "FPP_UNSTABLE_FLAG": {"off", "on"}}

_SWITCHES = {}

# Per-switch env vars honoured on top of the FPP_PARITY mode default (same semantics as the Rust / Swift ports,
# task_processor.rs parse_ephem_refine_switch / FPPV3Dispatch.swift): when the variable is SET it must be exactly
# "on" or "off" and overrides the mode default; anything else (incl. "1", "true", "ON", "") is a typed error.
_ENV_OVERRIDE_SWITCHES = ("FPP_EPHEM_REFINE", "FPP_UNSTABLE_FLAG")


class FPPSwitchEnvError(ValueError):
    """An FPP_* per-switch environment variable holds a value other than exactly "on" / "off".

    Raised while the switch set is built.  At worker start-up that is the `import fpp_native` in worker.py
    (not an ImportError, so it is NOT swallowed by the legacy-triceratops fallback): the worker refuses to start
    and the traceback names the variable.  Never a silent fallback to the mode default.
    """


def refuse_switch_env(who):
    """Golden / gate / A-B harness guard: refuse to run when FPP_EPHEM_REFINE or FPP_UNSTABLE_FLAG is exported.

    A stray export would silently change the switch set of generated or replayed vectors (set_fpp_switches applies
    the env on top of the mode default).  Harnesses that really intend the override opt in with
    FPP_ALLOW_SWITCH_ENV=1.  Harnesses that choose switches explicitly do so via keyword args, not env.
    """
    if os.environ.get("FPP_ALLOW_SWITCH_ENV") == "1":
        return
    set_ = [n for n in _ENV_OVERRIDE_SWITCHES if n in os.environ]
    if set_:
        raise FPPSwitchEnvError(
            f"{who}: refusing to run with {', '.join(f'{n}={os.environ[n]!r}' for n in set_)} in the environment "
            f"(it would change the switch set of generated/replayed vectors); unset it, or export "
            f"FPP_ALLOW_SWITCH_ENV=1 if the override is intended")


def _apply_env_overrides(candidate):
    for name in _ENV_OVERRIDE_SWITCHES:
        if name in os.environ:
            value = os.environ[name]
            if value not in ("on", "off"):
                raise FPPSwitchEnvError(
                    f"invalid switch value {name}={value!r}: must be exactly 'on' or 'off' "
                    f"(unset the variable to use the FPP_PARITY mode default)")
            candidate[name] = value


def _validate_switch(name, value):
    """Raise ValueError for an unknown switch name or a value outside its mode set."""
    if name not in _SWITCH_NAMES:
        raise ValueError(f"unknown FPP switch: {name} (known: {', '.join(_SWITCH_NAMES)})")
    if value not in _ALLOWED[name]:
        raise ValueError(f"invalid value {value!r} for {name} "
                         f"(allowed: {', '.join(sorted(_ALLOWED[name]))})")


def get_fpp_switches():
    """Copy of the active switch set — mutating it never changes module state."""
    return dict(_SWITCHES)


def switch_is(name, value):
    """True when the active mode of `name` is `value` (used by the model code)."""
    return _SWITCHES[name] == value


def set_fpp_switches(parity=None, **overrides):
    """Select the switch set. Returns get_fpp_switches().

    parity=True/False replaces the active set by a copy of the parity /
    production set. With parity=None the current set is kept and the
    per-switch environment variables (FPP_SECONDARY_GEOMETRY=... etc.) are
    applied on top of it. Keyword overrides win over both. Unknown names or
    values outside the switch's mode set raise ValueError.
    """
    global _SWITCHES

    candidate = (dict(_PARITY_SET if parity else _PRODUCTION_SET)
                 if parity is not None else dict(_SWITCHES))
    _apply_env_overrides(candidate)                       # strict, both branches (FPPSwitchEnvError)
    if parity is None:
        for name in _SWITCH_NAMES:
            if name in _ENV_OVERRIDE_SWITCHES:
                continue                                  # already applied strictly above
            value = (os.environ.get(name) or "").strip()
            if value:
                _validate_switch(name, value)
                candidate[name] = value
    for name, value in overrides.items():
        _validate_switch(name, value)
        candidate[name] = value
    _SWITCHES = candidate
    return get_fpp_switches()


# Production unless FPP_PARITY=1 (plan 0.3 (o): parity mode is one switch).
set_fpp_switches(parity=(os.environ.get("FPP_PARITY", "0") == "1"))


# Thresholds live in fpp_thresholds.py (shared with legacy triceratops_fpp);
# re-exported here for backwards compatibility of existing imports.
from fpp_thresholds import (  # noqa: F401
    FPP_VALIDATED, FPP_CANDIDATE, NFPP_VALIDATED,
    should_run_triceratops,
)

# ============================================================
# 1. Transit model — numba-accelerated ring integration
# ============================================================

# Kepler solver: Newton from E = M exactly as before, SAFEGUARDED by a bracket
# and a bisection fallback.  Plain f64 scalar arithmetic in a fixed order, only
# math.sin / math.cos, so the Swift / Rust / Metal / CUDA ports reproduce it bit
# for bit.  Exact sequence of operations: progress/impl-fpp-kepler-fix.md.
# The old undamped 30-step Newton did not converge for e >~ 0.97 at small M
# (|E| ~ 1e3 after 30 steps; 4.76 nat spurious lnZ on TIC 4897275).  Every draw
# on which it converged and no step left the bracket is reproduced bit for bit.
_KEPLER_NEWTON_TOL = 1.0e-10   # Newton stop |dE| < tol: as before, exact to ~1e-18 (quadratic)
_KEPLER_BRACKET_TOL = 1.0e-15  # bisection stop: hi - lo <= tol * max(1, |E|)
_KEPLER_NEWTON_ITERS = 30      # Newton iterations allowed (as before); then bisection only
_KEPLER_MAXIT = 100            # total; 30 + ~54 halvings of [0, 2*pi] always suffice
_KEPLER_M_SLACK = 1.0e-9       # callers' M reduction can land a few ulp outside [0, 2*pi]; clamp within this


@numba.njit(cache=True)
def _kepler_solve(M, ecc, maxit):
    """Solve M = E - ecc*sin(E) for M in [0, 2*pi], 0 <= ecc < 1.

    Returns (E, converged).  f(E) = E - ecc*sin(E) - M is increasing, with
    f(0) = -M <= 0 and f(2*pi) = 2*pi - M >= 0, so [lo, hi] = [0, 2*pi] always
    brackets the root.  Start E = M.  Each iteration: f = f(E); f == 0 -> done;
    f > 0: hi = E, else lo = E; Newton step En = E - f/(1 - ecc*cos E).  The
    step is taken (and stops the loop when |f/f'| < 1e-10) only while it is one
    of the first 30 iterations and either En lies strictly inside (lo, hi) or
    |f/f'| < 1e-10 (a sub-ulp step may land ON a bracket end); otherwise
    bisect: if hi - lo <= 1e-15*max(1, |E|) return E, else E = lo + 0.5*(hi-lo).
    Domain: M in [-1e-9, 2*pi + 1e-9] is first clamped into [0, 2*pi]
    (M < 0 -> 0.0, M > 2*pi -> 2*pi): the callers' reduction
    M -= 2*pi*floor(M/2*pi) can return e.g. -2.8e-14 when M/2*pi rounds up to
    an integer.  Inputs outside that slack (including NaN) or `maxit`
    exhausted -> converged = False.
    """
    two_pi = 2.0 * math.pi
    if M < 0.0 and M >= -_KEPLER_M_SLACK:
        M = 0.0
    elif M > two_pi and M <= two_pi + _KEPLER_M_SLACK:
        M = two_pi
    if not (M >= 0.0 and M <= two_pi and ecc >= 0.0 and ecc < 1.0):
        return (math.nan, False)
    lo = 0.0
    hi = two_pi
    E = M
    for it in range(maxit):
        f = E - ecc * math.sin(E) - M
        if f == 0.0:
            return (E, True)
        if f > 0.0:
            hi = E
        else:
            lo = E
        dE = f / (1.0 - ecc * math.cos(E))
        En = E - dE
        tiny = abs(dE) < _KEPLER_NEWTON_TOL
        if it < _KEPLER_NEWTON_ITERS and ((En > lo and En < hi) or tiny):
            E = En
            if tiny:
                return (E, True)
        else:
            if hi - lo <= _KEPLER_BRACKET_TOL * max(1.0, abs(E)):
                return (E, True)
            E = lo + 0.5 * (hi - lo)
    return (E, False)


@numba.njit(cache=True)
def _kepler_eq(M, ecc):
    """Eccentric anomaly E for M in [0, 2*pi]; NaN when the solver did not
    converge (the draw's lnL is then NaN and counted, see `_dispatch_lnL`)."""
    E, ok = _kepler_solve(M, ecc, _KEPLER_MAXIT)
    if ok:
        return E
    return math.nan


@numba.njit(cache=True)
def _true_anomaly(M, ecc):
    """True anomaly from mean anomaly and eccentricity."""
    if ecc < 1e-10:
        return M
    E = _kepler_eq(M, ecc)
    cosE = math.cos(E)
    nu = math.atan2(math.sqrt(1 - ecc * ecc) * math.sin(E),
                     cosE - ecc)
    return nu


@numba.njit(cache=True)
def _mid_transit_mean_anomaly(ecc, w_rad):
    """Mean anomaly at mid-transit, M_t = E_t - ecc*sin(E_t).

    E_t comes from the true anomaly of conjunction nu_t = pi/2 - w_rad (w_rad =
    90 deg - argument of periastron, as elsewhere in this module). Anchoring
    M = 2*pi*(time - t0)/period + M_t on it puts the transit centre at
    time = t0 for any eccentricity (D1).
    """
    nu_t = math.pi / 2.0 - w_rad
    E_t = 2.0 * math.atan2(math.sqrt(1.0 - ecc) * math.sin(nu_t / 2.0),
                           math.sqrt(1.0 + ecc) * math.cos(nu_t / 2.0))
    return E_t - ecc * math.sin(E_t)


@numba.njit(cache=True)
def _z_one(t, t0, period, a_Rs, cos_i, ecc, w_rad, M_t):
    """Projected separation z at one time, in units of R_star.

    Points where the companion is behind the star (cos f < 0 with
    f = nu + w_rad - pi/2) return z = 1e3 instead of being folded into a
    mirror-image dip at phase 0.5 (D2).
    """
    two_pi = 2.0 * math.pi
    phase = (t - t0) / period
    # Kepler's equation is solved on the principal branch: reduce M into
    # [0, 2*pi) first, then nu = _true_anomaly(M, ecc) is in (-pi, pi] and
    # belongs to the same branch (only sin/cos of it are used below).
    M = two_pi * phase + M_t
    M -= two_pi * math.floor(M / two_pi)
    nu = _true_anomaly(M, ecc)
    r_factor = (1.0 - ecc * ecc) / (1.0 + ecc * math.cos(nu))
    # projected separation, front side of the orbit only
    f = nu + w_rad - math.pi / 2.0
    sin_f = math.sin(f)
    cos_f = math.cos(f)
    if cos_f < 0.0:
        return 1.0e3  # companion behind the star: no transit
    return a_Rs * r_factor * math.sqrt(
        sin_f * sin_f + cos_i * cos_i * cos_f * cos_f)


@numba.njit(cache=True)
def _compute_z_arr(time, t0, period, a_Rs, inc_rad, ecc, w_rad):
    """Projected separation z(t) in units of R_star.

    The orbit is anchored on mid-transit (see _mid_transit_mean_anomaly) and
    flags the far side of the orbit with z = 1e3 (see _z_one).
    """
    n = len(time)
    z = np.empty(n)
    cos_i = math.cos(inc_rad)
    M_t = _mid_transit_mean_anomaly(ecc, w_rad)
    for i in range(n):
        z[i] = _z_one(time[i], t0, period, a_Rs, cos_i, ecc, w_rad, M_t)
    return z


# Complete elliptic integrals K, E, Pi through Carlson's symmetric forms R_F, R_C, R_J
# (B. C. Carlson, "Numerical computation of real or complex elliptic integrals", Numer. Algorithms 10, 13, 1995;
# DLMF section 19.36). Written from the published duplication theorems and series, not from any library or book code.
# Relations used (DLMF 19.25): K(k) = R_F(0, k'^2, 1); E(k) = K - (k^2/3) R_D(0, k'^2, 1), R_D(x,y,z) = R_J(x,y,z,z);
# Pi(n, k) = K + (n/3) R_J(0, k'^2, 1, 1 - n) for n < 1; for n > 1 the Cauchy principal value is
# Pi(n, k) = K - Pi(k^2/n, k) (DLMF 19.6.5) = -(k^2/(3 n)) R_J(0, k'^2, 1, 1 - k^2/n).

_CARLSON_TOL = 1.0e-3      # series truncation: error ~ tol^6 ~ 1e-18


@numba.njit(cache=True)
def _carlson_rf(x, y, z):
    """R_F(x, y, z) by the duplication theorem and the 5th-order series."""
    for _ in range(1100):
        sx = math.sqrt(x)
        sy = math.sqrt(y)
        sz = math.sqrt(z)
        lam = sx * sy + sy * sz + sz * sx
        x = 0.25 * (x + lam)
        y = 0.25 * (y + lam)
        z = 0.25 * (z + lam)
        mu = (x + y + z) / 3.0
        dx = 1.0 - x / mu
        dy = 1.0 - y / mu
        dz = 1.0 - z / mu
        if max(abs(dx), abs(dy), abs(dz)) < _CARLSON_TOL:
            break
    e2 = dx * dy - dz * dz
    e3 = dx * dy * dz
    return (1.0 - e2 / 10.0 + e3 / 14.0 + e2 * e2 / 24.0 - 3.0 * e2 * e3 / 44.0) / math.sqrt(mu)


@numba.njit(cache=True)
def _carlson_rc(x, y):
    """R_C(x, y) = R_F(x, y, y) for x >= 0, y > 0."""
    for _ in range(200):
        lam = 2.0 * math.sqrt(x) * math.sqrt(y) + y
        x = 0.25 * (x + lam)
        y = 0.25 * (y + lam)
        mu = (x + 2.0 * y) / 3.0
        s = (y - mu) / mu
        if abs(s) < _CARLSON_TOL:
            break
    return (1.0 + s * s * (0.3 + s * (1.0 / 7.0 + s * (0.375 + s * 9.0 / 22.0)))) / math.sqrt(mu)


@numba.njit(cache=True)
def _carlson_rj(x, y, z, p):
    """R_J(x, y, z, p) for p > 0 by the duplication theorem with the R_C sum and the 5th-order series."""
    total = 0.0
    fac = 1.0
    for _ in range(1100):
        sx = math.sqrt(x)
        sy = math.sqrt(y)
        sz = math.sqrt(z)
        sp = math.sqrt(p)
        lam = sx * sy + sy * sz + sz * sx
        alpha = (p * (sx + sy + sz) + sx * sy * sz) ** 2
        beta = p * (p + lam) ** 2
        total += fac * _carlson_rc(alpha, beta)
        fac *= 0.25
        x = 0.25 * (x + lam)
        y = 0.25 * (y + lam)
        z = 0.25 * (z + lam)
        p = 0.25 * (p + lam)
        mu = (x + y + z + 2.0 * p) / 5.0
        dx = (mu - x) / mu
        dy = (mu - y) / mu
        dz = (mu - z) / mu
        dp = (mu - p) / mu
        if max(max(abs(dx), abs(dy)), max(abs(dz), abs(dp))) < _CARLSON_TOL:
            break
    e2 = dx * dy + dx * dz + dy * dz - 3.0 * dp * dp
    e3 = dx * dy * dz + 2.0 * e2 * dp + 4.0 * dp ** 3
    e4 = (2.0 * dx * dy * dz + e2 * dp + 3.0 * dp ** 3) * dp
    e5 = dx * dy * dz * dp * dp
    series = (1.0 - 3.0 * e2 / 14.0 + e3 / 6.0 + 9.0 * e2 * e2 / 88.0 - 3.0 * e4 / 22.0
              - 9.0 * e2 * e3 / 52.0 + 3.0 * e5 / 26.0)
    return 3.0 * total + fac * mu ** (-1.5) * series


@numba.njit(cache=True)
def _ellip_ke(kc2):
    """(K, E) of the modulus whose complementary parameter is kc2 = 1 - m (0 <= kc2 <= 1).

    kc2 is floored at 1e-300 (K diverges logarithmically at kc2 = 0; the callers' combinations are finite there)."""
    kc2 = max(kc2, 1.0e-300)
    K = _carlson_rf(0.0, kc2, 1.0)
    E = K - ((1.0 - kc2) / 3.0) * _carlson_rj(0.0, kc2, 1.0, 1.0)
    return K, E


@numba.njit(cache=True)
def _ellip_kep(kc2, n):
    """(K, E, Pi(n, m)) with m = 1 - kc2; n > 1 is the Cauchy principal value (DLMF 19.6.5); n != 1."""
    kc2 = max(kc2, 1.0e-300)
    K, E = _ellip_ke(kc2)
    m = 1.0 - kc2
    if n < 1.0:
        Pi = K + (n / 3.0) * _carlson_rj(0.0, kc2, 1.0, 1.0 - n) if n != 0.0 else K
    else:
        Pi = -(m / (3.0 * n)) * _carlson_rj(0.0, kc2, 1.0, 1.0 - m / n)
    return K, E, Pi


@numba.njit(cache=True)
def _ellipk(k2):
    """Complete elliptic integral of the first kind K(m) with parameter m = k2."""
    return _carlson_rf(0.0, 1.0 - k2, 1.0)


@numba.njit(cache=True)
def _ellipe(k2):
    """Complete elliptic integral of the second kind E(m) with parameter m = k2."""
    return _ellip_ke(1.0 - k2)[1]


@numba.njit(cache=True)
def _ellippi(n, k2):
    """Complete elliptic integral of the third kind Pi(n, m = k2); n > 1 is a Cauchy principal value."""
    return _ellip_kep(1.0 - k2, n)[2]


@numba.njit(cache=True)
def _transit_flux_core(z, p, u1, u2):
    """Mandel & Agol (2002) quadratic-limb-darkening occultation flux (see _transit_flux).

    Clean-room: the published closed forms, with K / E / Pi from the module's
    Carlson-form `_ellip_kep`; every acos of the paper is replaced by an atan2 of
    the factored root S so that near-tangent geometries keep full precision.
    """
    c2 = u1 + 2.0 * u2
    c4 = -u2
    fo = 1.0 - c2 / 3.0 - c4 / 2.0
    if p <= 0.0 or z >= 1.0 + p:
        return 1.0
    if p >= 1.0 and z <= p - 1.0:
        return 0.0
    a = (z - p) * (z - p)
    b = (z + p) * (z + p)
    q = p * p - z * z
    theta = 1.0 if p > z else 0.0
    if p < 1.0 and z <= 1.0 - p:
        # planet fully inside the stellar disc
        lam_e = p * p
        eta_d = 0.5 * p * p * (p * p + 2.0 * z * z)
        if z == 0.0:
            lam_d = -(2.0 / 3.0) * (1.0 - p * p) ** 1.5
        elif z == p and p < 0.5:
            kc = math.sqrt(1.0 - 4.0 * p * p)
            K, E = _ellip_ke(kc * kc)
            lam_d = (1.0 / 3.0 + (2.0 / (9.0 * math.pi))
                     * (4.0 * (2.0 * p * p - 1.0) * E + (1.0 - 4.0 * p * p) * K))
            theta = 0.0
        elif z == p and p == 0.5:
            lam_d = 1.0 / 3.0 - 4.0 / (9.0 * math.pi)
            theta = 0.0
        elif z == 1.0 - p:
            lam_d = ((2.0 / (3.0 * math.pi)) * math.acos(1.0 - 2.0 * p)
                     - (4.0 / (9.0 * math.pi)) * (3.0 + 2.0 * p - 8.0 * p * p)
                     * math.sqrt(p * (1.0 - p))
                     - (2.0 / 3.0) * (1.0 if p > 0.5 else 0.0))
        else:
            kc = math.sqrt(((1.0 - z - p) * (1.0 + z + p))
                           / ((1.0 - z + p) * (1.0 + z - p)))
            n = (a - b) / a
            K, E, Pi = _ellip_kep(kc * kc, n)
            lam_d = ((2.0 / (9.0 * math.pi * math.sqrt(1.0 - a)))
                     * ((1.0 - 5.0 * z * z + p * p + q * q) * K
                        + (1.0 - a) * (z * z + 7.0 * p * p - 4.0) * E
                        - 3.0 * (q / a) * Pi))
    else:
        # partial overlap, |1 - p| < z < 1 + p (p > 1 allowed)
        S = math.sqrt(max(0.0, (1.0 - z + p) * (1.0 + z - p)
                          * (z + p - 1.0) * (z + p + 1.0)))
        k0 = math.atan2(S, p * p + z * z - 1.0)
        k1 = math.atan2(S, 1.0 - p * p + z * z)
        lam_e = (p * p * k0 + k1 - 0.5 * S) / math.pi
        eta2 = p * p * (p * p + 2.0 * z * z) / 2.0
        eta_d = (k1 + 2.0 * eta2 * k0
                 - 0.25 * (1.0 + 5.0 * p * p + z * z) * S) / (2.0 * math.pi)
        if z == p:
            if p >= 0.5:
                kc = math.sqrt(1.0 - 1.0 / (4.0 * p * p))
                K, E = _ellip_ke(kc * kc)
                lam_d = (1.0 / 3.0 + 16.0 * p / (9.0 * math.pi)
                         * (2.0 * p * p - 1.0) * E
                         - (1.0 - 4.0 * p * p) * (3.0 - 8.0 * p * p)
                         / (9.0 * math.pi * p) * K)
                theta = 0.0
            else:
                # unreachable (z == p < 0.5 is always inside); keep total
                lam_d = 0.0
        else:
            kc = math.sqrt(((z + p - 1.0) * (z + p + 1.0)) / (4.0 * p * z))
            n = (a - 1.0) / a
            K, E, Pi = _ellip_kep(kc * kc, n)
            lam_d = ((1.0 / (9.0 * math.pi * math.sqrt(p * z)))
                     * (((1.0 - b) * (2.0 * b + a - 3.0) - 3.0 * q * (b - 2.0)) * K
                        + 4.0 * p * z * (z * z + 7.0 * p * p - 4.0) * E
                        - 3.0 * (q / a) * Pi))
    return 1.0 - ((1.0 - c2) * lam_e + c2 * (lam_d + (2.0 / 3.0) * theta)
                  - c4 * eta_d) / fo


@numba.njit(cache=True)
def _transit_flux(z, p, u1, u2):
    """Relative flux of a quadratic-limb-darkened star occulted by a disc.

    z: centre separation, p: radius ratio (occulter / occulted star, may exceed
    1 for EB secondaries), in units of the occulted star's radius.
    Mandel & Agol (2002); accuracy <= 2e-10 absolute against adaptive
    quadrature on the golden grid. The removable singularity of the closed
    form at z == p (1/a cancellation) is bridged by linear interpolation
    between the exact z == p value and a point 1e-5 away for |z - p| < 1e-6.
    """
    if p > 0.0 and z > 0.0 and p != z and abs(z - p) < 1.0e-6 \
            and z < 1.0 + p and not (p >= 1.0 and z <= p - 1.0):
        h = 1.0e-5
        f0 = _transit_flux_core(p, p, u1, u2)
        z1 = p + (h if z > p else -h)
        f1 = _transit_flux_core(z1, p, u1, u2)
        return f0 + (z - p) / (z1 - p) * (f1 - f0)
    return _transit_flux_core(z, p, u1, u2)


# Parity-mode exposure supersampling: upstream TRICERATOPS integrates each
# cadence point over the 2-minute TESS exposure with 20 samples
# (FPP_EXPOSURE = "upstream_2min_x20"); production uses the Gauss-Legendre
# integrator over the light curve's cadence (plan 0.3 (j'')).
PARITY_EXPTIME_D = 0.00139   # 2 minutes, in days
PARITY_NSUB = 20


@numba.njit(cache=True)
def _lc_core(time, k, t0, period, a_Rs, inc_rad, ecc, w_rad, u1, u2,
             exptime, nsub):
    """Transit light curve for one parameter set, with exposure supersampling.

    nsub <= 1 or exptime <= 0 evaluates the analytic kernel at the cadence
    times. Otherwise each point is the mean of `nsub` samples spread over the
    exposure window [time[i] - exptime/2, time[i] + exptime/2], PARITY MODE
    ONLY: sample j = 0..nsub-1 at time[i] + exptime*(j/(nsub-1) - 1/2),
    END POINTS INCLUDED (batman's supersampling, the independent T3 reference
    of the golden fixtures). PyTransit (upstream) places the samples at the
    cell MIDPOINTS ((j + 1/2)/nsub); the two schemes differ only when the
    transit is shorter than the exposure (e.g. an e = 0.97 draw: 0.68 nat
    between them), i.e. exactly where PyTransit itself is not valid (U7e) and
    the case is gated against batman. Production uses the Gauss-Legendre
    integrator instead (plan 0.3 (j'')).
    """
    n = len(time)
    flux = np.empty(n)
    cos_i = math.cos(inc_rad)
    M_t = _mid_transit_mean_anomaly(ecc, w_rad)
    if nsub <= 1 or exptime <= 0.0:
        for i in range(n):
            z = _z_one(time[i], t0, period, a_Rs, cos_i, ecc, w_rad, M_t)
            flux[i] = _transit_flux(z, k, u1, u2)
        return flux
    for i in range(n):
        acc = 0.0
        for j in range(nsub):
            dt = exptime * (j / (nsub - 1.0) - 0.5)
            z = _z_one(time[i] + dt, t0, period, a_Rs, cos_i, ecc, w_rad, M_t)
            acc += _transit_flux(z, k, u1, u2)
        flux[i] = acc / nsub
    return flux


def _compute_transit_lc(time, k, t0, period, a_Rs, inc_rad, ecc, w_rad,
                        u1, u2):
    """Compute full transit light curve for one parameter set.

    Plain Python (not njit) because the exposure mode is chosen from the
    switch set; the numerics live in the njit `_lc_core`.
    """
    if switch_is("FPP_EXPOSURE", "upstream_2min_x20"):
        exptime, nsub = PARITY_EXPTIME_D, PARITY_NSUB
    else:
        exptime, nsub = 0.0, 1
    return _lc_core(time, k, t0, period, a_Rs, inc_rad, ecc, w_rad, u1, u2,
                    exptime, nsub)


@numba.njit(cache=True)
def _half_dz2_dt(t, period, a_Rs, inc_rad, ecc, w_rad, M_t):
    """(1/2) d(z^2)/dt = X dX/dt + Y dY/dt at time t, in R_s^2 / day.

    Analytic Keplerian velocities of the same geometry as _z_one (t measured
    from mid-transit, M = 2*pi*t/period + M_t reduced to [0, 2*pi) before
    _true_anomaly).  Sky-plane coordinates X = -r cos(w+nu),
    Y = -r sin(w+nu) cos i, so the root of this function is the time of
    minimum projected separation — the tangent point of a grazing transit —
    which is far better conditioned than argmin z (plan decision 0.3 (j'')).
    """
    two_pi = 2.0 * math.pi
    n = two_pi / period
    M = two_pi * t / period + M_t
    M -= two_pi * math.floor(M / two_pi)
    nu = _true_anomaly(M, ecc)
    one_minus_e2 = 1.0 - ecc * ecc
    q = 1.0 + ecc * math.cos(nu)
    r = a_Rs * one_minus_e2 / q
    nu_dot = n * q * q / one_minus_e2 ** 1.5
    r_dot = n * a_Rs * ecc * math.sin(nu) / math.sqrt(one_minus_e2)
    th = w_rad + nu
    cos_th = math.cos(th)
    sin_th = math.sin(th)
    cos_i = math.cos(inc_rad)
    X = -r * cos_th
    Y = -r * sin_th * cos_i
    Xd = -r_dot * cos_th + r * sin_th * nu_dot
    Yd = (-r_dot * sin_th - r * cos_th * nu_dot) * cos_i
    return X * Xd + Y * Yd


@numba.njit(cache=True)
def _bisect(flo, fhi, tlo, thi, period, a_Rs, inc_rad, ecc, w_rad, M_t,
            level, is_z):
    """Bisection of f(t) on [tlo, thi], where f is z(t) - level (is_z) or
    (1/2) d(z^2)/dt (level unused).  flo/fhi are f at the two ends and must
    have opposite signs (or be exactly zero).  Stops when the bracket is
    narrower than `tol` (1e-13 d) or after 100 halvings; returns NaN when the
    bracket does not straddle a root.
    """
    tol = 1e-13
    if flo == 0.0:
        return tlo
    if fhi == 0.0:
        return thi
    if flo * fhi > 0.0:
        return np.nan
    for _ in range(100):
        if thi - tlo < tol:
            break
        tm = 0.5 * (tlo + thi)
        if is_z:
            fm = _z_one(tm, 0.0, period, a_Rs, math.cos(inc_rad), ecc,
                        w_rad, M_t) - level
        else:
            fm = _half_dz2_dt(tm, period, a_Rs, inc_rad, ecc, w_rad, M_t)
        if fm == 0.0:
            return tm
        if (fm > 0.0) == (flo > 0.0):
            tlo, flo = tm, fm
        else:
            thi, fhi = tm, fm
    return 0.5 * (tlo + thi)


@numba.njit(cache=True)
def _contact_times_nb(period, a_Rs, inc_rad, ecc, w_rad, k):
    """Exact contact times of one transit, in days relative to mid-transit.

    Returns float64[5] = [t1, t2, t3, t4, t_min], NaN where a contact is
    absent.  t1/t4 are the outer contacts (roots of z = 1 + k), t2/t3 the
    inner contacts (roots of z = |1 - k|, only for a full transit or
    occultation), t_min the tangent point (root of _half_dz2_dt).  The transit
    is the interval around t = 0 where the companion is in FRONT of the star
    (far-side points carry z = 1e3 in _z_one, which is > any contact level and
    therefore consistent with the brackets).
    """
    out = np.full(5, np.nan)
    cos_i = math.cos(inc_rad)
    M_t = _mid_transit_mean_anomaly(ecc, w_rad)

    # Longest possible transit scale (apastron geometry), never past half an
    # orbit and never shorter than 1e-3 d.
    W = 2.5 * (1.0 + k) * period / (math.pi * a_Rs) / (1.0 - ecc)
    if W > 0.5 * period:
        W = 0.5 * period
    if W < 1e-3:
        W = 1e-3

    ngrid = 2001
    step = 2.0 * W / (ngrid - 1)
    best_z = 1.0e3
    i0 = -1
    for i in range(ngrid):
        z = _z_one(-W + step * i, 0.0, period, a_Rs, cos_i, ecc, w_rad, M_t)
        if z < 1.0e3 and z < best_z:
            best_z = z
            i0 = i
    if i0 < 0 or best_z >= 1.0 + k + 1e-3:
        return out  # no transit anywhere in the window

    # t_min: the derivative changes sign from - to + across the minimum grid
    # point; a bracket without a sign change means the minimum sits at the
    # window edge, i.e. the window is not wide enough.
    lo_i = i0 - 1
    if lo_i < 0:
        lo_i = 0
    hi_i = i0 + 1
    if hi_i > ngrid - 1:
        hi_i = ngrid - 1
    tlo = -W + step * lo_i
    thi = -W + step * hi_i
    t_min = _bisect(_half_dz2_dt(tlo, period, a_Rs, inc_rad, ecc, w_rad, M_t),
                    _half_dz2_dt(thi, period, a_Rs, inc_rad, ecc, w_rad, M_t),
                    tlo, thi, period, a_Rs, inc_rad, ecc, w_rad, M_t, 0.0,
                    False)
    if math.isnan(t_min):
        return out
    z_min = _z_one(t_min, 0.0, period, a_Rs, cos_i, ecc, w_rad, M_t)

    L = 1.0 + k
    if z_min >= L:
        return out  # the companion never touches the disc

    # Outer contacts.  z(t_min -/+ W) must lie above the level; if it does
    # not, widen the window (x2, at most 4 times) before giving up.
    hw = W
    t1 = np.nan
    for _ in range(5):
        a = t_min - hw
        fa = _z_one(a, 0.0, period, a_Rs, cos_i, ecc, w_rad, M_t) - L
        fb = z_min - L
        t1 = _bisect(fa, fb, a, t_min, period, a_Rs, inc_rad, ecc, w_rad,
                     M_t, L, True)
        if not math.isnan(t1):
            break
        hw *= 2.0
    hw = W
    t4 = np.nan
    for _ in range(5):
        b = t_min + hw
        fa = z_min - L
        fb = _z_one(b, 0.0, period, a_Rs, cos_i, ecc, w_rad, M_t) - L
        t4 = _bisect(fa, fb, t_min, b, period, a_Rs, inc_rad, ecc, w_rad,
                     M_t, L, True)
        if not math.isnan(t4):
            break
        hw *= 2.0

    # Inner contacts: only when the companion is fully inside (or fully in
    # front of) the disc at mid-transit.  For k > 1 the level is k - 1.
    t2 = np.nan
    t3 = np.nan
    Lin = abs(1.0 - k)
    if z_min < Lin:
        if not math.isnan(t1):
            t2 = _bisect(L - Lin, z_min - Lin, t1, t_min, period, a_Rs,
                         inc_rad, ecc, w_rad, M_t, Lin, True)
        if not math.isnan(t4):
            t3 = _bisect(z_min - Lin, L - Lin, t_min, t4, period, a_Rs,
                         inc_rad, ecc, w_rad, M_t, Lin, True)

    out[0] = t1
    out[1] = t2
    out[2] = t3
    out[3] = t4
    out[4] = t_min
    return out


def contact_times(P, aR, inc_rad, ecc, w_rad, k):
    """Exact contact times of a transit as a dict (plan decision 0.3 (j'')).

    Keys "t1", "t2", "t3", "t4", "t_min" in days relative to mid-transit
    (t = 0 at the tangent point); float('nan') where a contact is absent.
    Plain Python wrapper around the njit solver `_contact_times_nb`.
    """
    t1, t2, t3, t4, t_min = _contact_times_nb(float(P), float(aR),
                                             float(inc_rad), float(ecc),
                                             float(w_rad), float(k))
    return {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "t_min": t_min}


_contact_times = contact_times



# ============================================================
# Exposure integration (plan 0.3 (j''))
# ============================================================
# A cadence point is the mean of the model flux over its exposure window
# [tc - W/2, tc + W/2].  The window is cut at the contact times that fall
# strictly inside it and every piece is integrated with an m-point
# Gauss-Legendre rule on [-1, 1] after removing the endpoint singularity: the
# flux behaves as Delta^(3/2) at a contact (F'' ~ Delta^(-1/2)), so
#   contact at the LOWER end lo:  t = lo + s^2,              s in [0, sqrt(L)]
#   contact at the UPPER end hi:  t = hi - s^2,
#   contacts at BOTH ends:        t = lo + L u^2 (3 - 2u),   u in [0, 1]
#   window edges (not contacts):  plain Gauss-Legendre.
# The substitution makes the integrand analytic at the piece ends, so a small
# m already matches the m = 128 oracle; the order is picked ONCE per draw
# (_choose_order) and reused for all of its exposures.

GL_ORDERS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128)
GL_MAX_M = 128
GL_ORACLE_IDX = 10      # index of m = 128 in GL_ORDERS
GL_SMOOTH_IDX = 2       # m = 4: enough for a window with no contact inside


GL_TABLES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fpp_gl_tables.json")
# pinned hash of the committed table (LF-normalised); regenerate with scripts/fpp/gen_gl_tables.py and update HERE
GL_TABLES_SHA256 = "ce6ea74701efa0c69a562fdd5c0af4a22ed9c74e5a0219376baee6c32173f71f"


def _build_gl_tables():
    """Gauss-Legendre nodes/weights on [-1, 1] for each order, zero padded, READ from the committed data file.

    The tables are data (C99 hex floats written by scripts/fpp/gen_gl_tables.py), not computed: numpy `leggauss` goes
    through LAPACK and is not reproducible across platforms or in the Swift / Rust ports.  A missing or inconsistent file
    raises (no silent recomputation).
    """
    import hashlib
    import json
    with open(GL_TABLES_FILE, "rb") as fh:
        raw = fh.read()
    # CRLF-normalised (a Windows checkout may convert line endings; same convention as the worker manifest hash)
    digest = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    if digest != GL_TABLES_SHA256:
        raise RuntimeError(f"{GL_TABLES_FILE}: sha256 {digest} != pinned {GL_TABLES_SHA256}")
    data = json.loads(raw.decode("ascii"))
    if tuple(data["orders"]) != GL_ORDERS:
        raise RuntimeError(f"{GL_TABLES_FILE}: orders {data['orders']} != GL_ORDERS {list(GL_ORDERS)}")
    x = np.zeros((len(GL_ORDERS), GL_MAX_M), dtype=np.float64)
    w = np.zeros((len(GL_ORDERS), GL_MAX_M), dtype=np.float64)
    for i, m in enumerate(GL_ORDERS):
        t = data["tables"][str(m)]
        if len(t["x"]) != m or len(t["w"]) != m:
            raise RuntimeError(f"{GL_TABLES_FILE}: order {m} has {len(t['x'])} nodes")
        x[i, :m] = [float.fromhex(v) for v in t["x"]]
        w[i, :m] = [float.fromhex(v) for v in t["w"]]
    return x, w


# A missing / mismatched table must NOT crash the worker at import (FileNotFoundError is not an ImportError, and an
# auto-update can deliver this module before the data file): FPP is declared unavailable instead and calc_fpp_native
# refuses to run (no silent computation without the table, no fallback to another model).
try:
    GL_X, GL_W = _build_gl_tables()
    GL_TABLES_ERROR = None
except (OSError, RuntimeError, ValueError, KeyError) as _gl_e:
    GL_X = np.zeros((len(GL_ORDERS), GL_MAX_M), dtype=np.float64)
    GL_W = np.zeros((len(GL_ORDERS), GL_MAX_M), dtype=np.float64)
    GL_TABLES_ERROR = f"{type(_gl_e).__name__}: {_gl_e}"
    logging.getLogger("exonit").error(
        "FPP UNAVAILABLE: Gauss-Legendre table %s (%s); FPP tasks will not be computed on this host", GL_TABLES_FILE,
        GL_TABLES_ERROR)


class FPPUnavailable(RuntimeError):
    """FPP cannot run on this host (missing / corrupt data file); the caller must requeue, never compute without it."""


class GeometryUnavailable(FPPUnavailable):
    """The smooth-geometry node table is missing or edited on this host (not a population problem)."""
    skip_reason = "geometry_nodes_unavailable"


GL_M = np.array(GL_ORDERS, dtype=np.int64)


@numba.njit(cache=True)
def _flux_at(t, P, a_Rs, cos_i, ecc, w_rad, M_t, k, u1, u2, dil_a, dil_b):
    """Model flux at one time, with the affine dilution dil_a * F + dil_b.

    dil_a = 1.0, dil_b = 0.0 for no dilution (see the dilution applied to the
    light curve in _batch_transit_lnL: (F + F_dilute) / (1 + F_dilute) is
    exactly this affine map).
    """
    z = _z_one(t, 0.0, P, a_Rs, cos_i, ecc, w_rad, M_t)
    return dil_a * _transit_flux(z, k, u1, u2) + dil_b


@numba.njit(cache=True)
def _window_mean(tc, W, ct, order_idx, P, a_Rs, cos_i, ecc, w_rad, M_t,
                 k, u1, u2, dil_a, dil_b):
    """Mean model flux over the exposure window [tc - W/2, tc + W/2].

    `ct` holds the SORTED finite contact times of the transit (may be empty);
    only the ones strictly inside the window cut it.  `order_idx` indexes
    GL_ORDERS.  Each piece of length L = hi - lo is integrated with the m-point
    rule of that order, with the endpoint substitution applied when the piece
    END is an interior cut (a contact): the first piece's lo and the last
    piece's hi are window edges and stay plain.  Weights sum to L, so the
    result is the weighted sum divided by W.  No arrays are allocated.
    """
    a = tc - 0.5 * W
    b = tc + 0.5 * W
    m = GL_M[order_idx]
    xs = GL_X[order_idx]
    ws = GL_W[order_idx]
    nct = len(ct)
    acc = 0.0
    jc = 0
    while jc < nct and ct[jc] <= a:
        jc += 1                     # contacts at or below the lower edge
    lo = a
    lo_c = False
    while True:
        if jc < nct and ct[jc] < b:
            hi = ct[jc]             # interior cut (a contact)
            hi_c = True
            jc += 1
        else:
            hi = b                  # upper window edge
            hi_c = False

        L = hi - lo
        if L > 0.0:
            if not (lo_c or hi_c):
                half_L = 0.5 * L
                mid = 0.5 * (hi + lo)
                for i in range(m):
                    t = half_L * xs[i] + mid
                    wt = half_L * ws[i]
                    acc += wt * _flux_at(t, P, a_Rs, cos_i, ecc, w_rad, M_t,
                                         k, u1, u2, dil_a, dil_b)
            elif lo_c and hi_c:
                for i in range(m):
                    u = 0.5 * (xs[i] + 1.0)
                    t = lo + L * u * u * (3.0 - 2.0 * u)
                    wt = 0.5 * ws[i] * 6.0 * L * u * (1.0 - u)
                    acc += wt * _flux_at(t, P, a_Rs, cos_i, ecc, w_rad, M_t,
                                         k, u1, u2, dil_a, dil_b)
            else:
                sL = math.sqrt(L)
                for i in range(m):
                    s = 0.5 * sL * (xs[i] + 1.0)
                    wt = 0.5 * sL * ws[i] * 2.0 * s
                    if lo_c:
                        t = lo + s * s
                    else:
                        t = hi - s * s
                    acc += wt * _flux_at(t, P, a_Rs, cos_i, ecc, w_rad, M_t,
                                         k, u1, u2, dil_a, dil_b)
        if not hi_c:
            break                   # the piece reached the upper window edge
        lo = hi
        lo_c = True
    return acc / W



@numba.njit(cache=True)
def _choose_order(ct, W, eps, P, a_Rs, cos_i, ecc, w_rad, M_t, k, u1, u2,
                  dil_a, dil_b):
    """Pick the GL order index once per draw: returns (order_idx, status).

    status 0 = converged by an order <= 32, 1 = the smallest-order search
    failed and the oracle order 128 is used, 2 = even orders 64 vs 128
    disagree by more than eps (unconverged: the caller must flag it, never
    'validated').  The worst case is the exposure straddling a contact, so
    each contact c is tested with the window centred on it: the smallest i in
    0..8 (m = 2..32) with |I_i - I_{i+1}| <= eps and |I_i - I_{i+2}| <= eps,
    where i <= 6 keeps i+1 and i+2 inside the first nine orders and i = 7, 8
    use min(i+1, 10) and min(i+2, 10) as the next two.  The draw takes the
    MAXIMUM order over its contacts, and the worst status; a floor of m = 4
    (GL_SMOOTH_IDX) keeps the smooth exposures of the draw (windows with no
    contact inside) covered too.
    """
    nct = len(ct)
    if nct == 0:
        return (GL_SMOOTH_IDX, 0)
    chosen = GL_SMOOTH_IDX
    for j in range(nct):
        c = ct[j]

        v0 = _window_mean(c, W, ct, 0, P, a_Rs, cos_i, ecc, w_rad, M_t,
                          k, u1, u2, dil_a, dil_b)
        v1 = _window_mean(c, W, ct, 1, P, a_Rs, cos_i, ecc, w_rad, M_t,
                          k, u1, u2, dil_a, dil_b)
        v2 = _window_mean(c, W, ct, 2, P, a_Rs, cos_i, ecc, w_rad, M_t,
                          k, u1, u2, dil_a, dil_b)
        found = -1
        for i in range(7):
            if abs(v0 - v1) <= eps and abs(v0 - v2) <= eps:
                found = i
                break
            v0, v1, v2 = v1, v2, _window_mean(c, W, ct, i + 3, P, a_Rs, cos_i,
                                              ecc, w_rad, M_t, k, u1, u2,
                                              dil_a, dil_b)
        if found < 0:
            # v0, v1, v2 now hold I_7, I_8, I_9 (m = 24, 32, 64); the remaining
            # candidates i = 7 and 8 take min(i+1, 10) and min(i+2, 10) as the
            # next two orders, i.e. the oracle I_10 (m = 128) enters the test.
            v3 = _window_mean(c, W, ct, GL_ORACLE_IDX, P, a_Rs, cos_i, ecc,
                              w_rad, M_t, k, u1, u2, dil_a, dil_b)
            if abs(v0 - v1) <= eps and abs(v0 - v2) <= eps:
                found = 7
            elif abs(v1 - v2) <= eps and abs(v1 - v3) <= eps:
                found = 8
            elif abs(v2 - v3) <= eps:
                return (GL_ORACLE_IDX, 1)   # plateau holds: oracle fallback
            else:
                return (GL_ORACLE_IDX, 2)   # 64 vs 128 disagree: unconverged
        if found > chosen:
            chosen = found
    return (chosen, 0)


# ============================================================
# Fold-bin model ladder (plan 0.3 (j'), N5; every node integrated by (j''))
# ============================================================
# One draw, one transit: the model of a fold bin is the bin average of the cadence-box average of
# the flux, M_b = (1/n_b) sum_i G(phi_i P) with G the box average of _window_mean.  It is
# evaluated by the quantile-node ladder of 0.3 (j'): level 0 is the second-order single-node model
# (one node at the mean phase, box widened to W_eff^2 = W^2 + 12 Var_b), levels 1..6 average
# K = 4, 8, 16, 32, 64, 128 quantile nodes of the bin's own phase distribution over the cadence
# box (a bin with more than 256 points is reduced to its 256-point quantile table, N5), and level
# 7 is the exact per-point mean.  Level c is accepted when it agrees with the NEXT TWO within
# eps_b = sigma_b sqrt(2 delta / N_aff) and the G1 guard of round 6 (R6-4) holds: the two
# successive differences decrease (d2 <= d1), or both are far below eps_b.  Rungs 6 and 7 are
# reached but never accepted; the exact level is the fallback of a bin that never converges.

FPP_LADDER_DELTA_NAT = 0.02        # delta of 0.3 (j''): eps_b = sigma_b sqrt(2 delta / N_aff)
LADDER_K_QUANT = 256               # N5: a bin with more points than this uses its quantile table
LADDER_MAX_K = 128                 # largest quantile rung (level 6); level 7 is the exact one
LADDER_EXACT_LEVEL = 7             # index of the exact per-point rung
LADDER_MIN_POINTS = 8              # a bin with at most this many points is evaluated exactly
LADDER_ACCEPT_MAX_LEVEL = 5        # the highest rung that can be accepted (its next two are 6 and 7)


@numba.njit(cache=True)
def _ladder_bin(b, W, eps_b, order_idx, n_b, tmean, var_t, node_tab, pts_off, pts_t,
                ct, P, a_Rs, cos_i, ecc, w_rad, M_t, k, u1, u2, dil_a, dil_b):
    """Model M_b of one fold bin by the quantile-node ladder of 0.3 (j').

    Returns (M_b, number of _window_mean calls, level used). `n_b` is the bin's point count,
    `tmean[b]` / `var_t[b]` its mean time and time variance (days), `node_tab[b, l - 1, :K]` the
    K = 4 * 2**(l - 1) quantile-node times of level l, pts_t[pts_off[b]:pts_off[b + 1]] its point
    times (the exact rung). Levels are evaluated progressively; level c (c <= LADDER_ACCEPT_MAX_LEVEL)
    is accepted when d1 = |M_c - M_{c+1}| <= eps_b, d2 = |M_{c+1} - M_{c+2}| <= eps_b and
    (d2 <= d1 or both below eps_b / 20) (guard G1). The first accepted level's value is returned,
    otherwise the exact level. A bin with at most LADDER_MIN_POINTS points goes straight to it.
    """
    lo_p = pts_off[b]
    hi_p = pts_off[b + 1]
    npts = hi_p - lo_p
    nwin = 0
    if n_b <= LADDER_MIN_POINTS:
        acc = 0.0
        for i in range(npts):
            acc += _window_mean(pts_t[lo_p + i], W, ct, order_idx, P, a_Rs, cos_i, ecc,
                                w_rad, M_t, k, u1, u2, dil_a, dil_b)
        return (acc / npts, nwin + npts, LADDER_EXACT_LEVEL)

    vals = np.empty(LADDER_EXACT_LEVEL + 1)
    w_eff = math.sqrt(W * W + 12.0 * var_t[b])
    vals[0] = _window_mean(tmean[b], w_eff, ct, order_idx, P, a_Rs, cos_i, ecc, w_rad,
                           M_t, k, u1, u2, dil_a, dil_b)
    nwin += 1
    for lvl in range(1, LADDER_EXACT_LEVEL):
        K = 4 << (lvl - 1)
        acc = 0.0
        for j in range(K):
            acc += _window_mean(node_tab[b, lvl - 1, j], W, ct, order_idx, P, a_Rs, cos_i,
                                ecc, w_rad, M_t, k, u1, u2, dil_a, dil_b)
        vals[lvl] = acc / K
        nwin += K
        if 2 <= lvl <= LADDER_ACCEPT_MAX_LEVEL + 2:
            c = lvl - 2
            d1 = abs(vals[c] - vals[c + 1])
            d2 = abs(vals[c + 1] - vals[c + 2])
            if d1 <= eps_b and d2 <= eps_b and (d2 <= d1 or (d1 < 0.05 * eps_b and
                                                              d2 < 0.05 * eps_b)):
                return (vals[c], nwin, c)

    acc = 0.0
    for i in range(npts):
        acc += _window_mean(pts_t[lo_p + i], W, ct, order_idx, P, a_Rs, cos_i, ecc, w_rad,
                            M_t, k, u1, u2, dil_a, dil_b)
    nwin += npts
    vals[LADDER_EXACT_LEVEL] = acc / npts
    # the last candidate rung (K = 64) is tested against levels 6 and 7, i.e. against the exact
    # value that was just evaluated; if it does not pass either, the exact level is the answer.
    c = LADDER_ACCEPT_MAX_LEVEL
    d1 = abs(vals[c] - vals[c + 1])
    d2 = abs(vals[c + 1] - vals[c + 2])
    if d1 <= eps_b and d2 <= eps_b and (d2 <= d1 or (d1 < 0.05 * eps_b and d2 < 0.05 * eps_b)):
        return (vals[c], nwin, c)
    return (vals[LADDER_EXACT_LEVEL], nwin, LADDER_EXACT_LEVEL)



@numba.njit(cache=True)
def _draw_bin_models(W, delta, nb, sig, n, tmean, var_t, tmin, tmax, node_tab, pts_off,
                     pts_t, P, a_Rs, inc_rad, ecc, w_rad, k, u1, u2, dil_a, dil_b,
                     out_idx, out_M):
    """One draw: the ladder model of every fold bin the transit can touch (0.3 (j'), (j'')).

    The bins are the fold rows in TIME units (days from mid-transit, the transit at t = 0) as
    built by `prod_fold_arrays`; `sig[b]` is the bin's likelihood sigma.  Returns
    (N_aff, status, windows evaluated, bins that fell through to the exact level); the affected
    bin indices and their models are written to out_idx[:N_aff] / out_M[:N_aff].  status is the
    exposure-order status of `_choose_order` (2 = unconverged), and N_aff = 0 (status 0) when the
    draw has no transit with both outer contacts.
    """
    ct5 = _contact_times_nb(P, a_Rs, inc_rad, ecc, w_rad, k)
    t1 = ct5[0]
    t4 = ct5[3]
    if math.isnan(t1) or math.isnan(t4):
        return (0, 0, 0, 0)
    ct = np.empty(4)
    nct = 0
    for j in range(4):
        v = ct5[j]
        if not math.isnan(v):
            ct[nct] = v
            nct += 1
    ct = ct[:nct]
    ct.sort()

    hw = 0.5 * W
    aff = np.empty(nb, dtype=np.int64)
    n_aff = 0
    for b in range(nb):
        if tmax[b] + hw > t1 and tmin[b] - hw < t4:
            aff[n_aff] = b
            n_aff += 1
    if n_aff == 0:
        return (0, 0, 0, 0)

    eps_ref = math.sqrt(2.0 * delta / n_aff)
    eps_min = sig[aff[0]] * eps_ref
    for i in range(1, n_aff):
        e = sig[aff[i]] * eps_ref
        if e < eps_min:
            eps_min = e
    M_t = _mid_transit_mean_anomaly(ecc, w_rad)
    cos_i = math.cos(inc_rad)
    order_idx, status = _choose_order(ct, W, eps_min, P, a_Rs, cos_i, ecc, w_rad, M_t,
                                      k, u1, u2, dil_a, dil_b)
    n_eval = 0
    n_fb = 0
    for i in range(n_aff):
        b = aff[i]
        M_b, nwin, level = _ladder_bin(b, W, sig[b] * eps_ref, order_idx, n[b], tmean, var_t,
                                       node_tab, pts_off, pts_t, ct, P, a_Rs, cos_i, ecc,
                                       w_rad, M_t, k, u1, u2, dil_a, dil_b)
        out_idx[i] = b
        out_M[i] = M_b
        n_eval += nwin
        if level == LADDER_EXACT_LEVEL:
            n_fb += 1
    return (n_aff, status, n_eval, n_fb)


def prod_fold_arrays(rows, period, sigma_b):
    """The production fold of `fold_production(..., keep_phases=True)` in TIME units (days).

    `sigma_b` is one sigma per row (the per-bin likelihood sigma of D15, or the scalar sigma_L
    repeated).  Returns the dict consumed by `_draw_bin_models`: n, flux (the bin mean flux),
    sigma, tmean = phase_mean * period, var_t = var_b * period**2, tmin / tmax (first / last
    phase * period), node_tab[b, l, :] = period * fold_quantiles(SRC_b, K = 4 * 2**l) for
    l = 0..5 (SRC_b = the bin's sorted phases, or its 256-point quantile table when the bin holds
    more points than LADDER_K_QUANT — plan N5 'reduced to the 256-point table'), and the exact
    rung: the bin's points (phase * period, ascending) concatenated in pts_t with offsets
    pts_off.
    """
    period = float(period)
    nb = len(rows)
    sig_in = np.asarray(sigma_b, dtype=np.float64).ravel()
    if sig_in.size == 1 and nb > 1:
        sig_in = np.full(nb, float(sig_in[0]), dtype=np.float64)
    if sig_in.size != nb:
        raise ValueError("sigma_b must hold one value per fold row")

    n = np.fromiter((int(r["n"]) for r in rows), dtype=np.int64, count=nb)
    flux = np.fromiter((float(r["flux_mean"]) for r in rows), dtype=np.float64, count=nb)
    sigma = np.ascontiguousarray(sig_in[:nb], dtype=np.float64)
    tmean = np.fromiter((float(r["phase_mean"]) * period for r in rows),
                        dtype=np.float64, count=nb)
    var_t = np.fromiter((float(r["var_b"]) * period * period for r in rows),
                        dtype=np.float64, count=nb)

    src = []
    pts = []
    for r in rows:
        ph = np.sort(np.asarray(r["phases"], dtype=np.float64).ravel(), kind="stable")
        pts.append(ph)                                    # the exact rung: every point
        if int(r["n"]) > LADDER_K_QUANT:                  # N5: reduced to the 256-point table
            src.append(np.asarray(r["quantiles"], dtype=np.float64).ravel())
        else:
            src.append(ph)
    tmin = np.fromiter((p[0] * period for p in pts), dtype=np.float64, count=nb)
    tmax = np.fromiter((p[-1] * period for p in pts), dtype=np.float64, count=nb)

    pts_off = np.zeros(nb + 1, dtype=np.int64)
    for i in range(nb):
        pts_off[i + 1] = pts_off[i] + pts[i].size
    pts_t = np.empty(max(int(pts_off[nb]), 1), dtype=np.float64)
    node_tab = np.zeros((nb, LADDER_EXACT_LEVEL - 1, LADDER_MAX_K), dtype=np.float64)
    for i in range(nb):
        pts_t[pts_off[i]:pts_off[i + 1]] = pts[i] * period
        for l in range(LADDER_EXACT_LEVEL - 1):
            K = 4 << l
            node_tab[i, l, :K] = period * fold_quantiles(src[i], K)
    return dict(n=n, flux=flux, sigma=sigma, tmean=tmean, var_t=var_t, tmin=tmin, tmax=tmax,
                node_tab=node_tab, pts_off=pts_off, pts_t=pts_t)


def build_prod_fold(rows, period, sigma_b, flux_b, cadence_s):
    """The production fold (plan 0.3 (j'), (j'')) as the production likelihood wants it.

    `rows` are the rows of `fold_production(..., keep_phases=True)`, `sigma_b` the per-bin
    likelihood sigma of D15 (plan decision 0.3 (q')), `flux_b` the bin mean fluxes the model is
    fitted to (already renormalised to the star of interest — see `prod_fold_renorm`) and
    `cadence_s` the cadence in seconds, which sets the exposure box W = cadence_s / 86400 days.

    On top of `prod_fold_arrays`'s time / node arrays this carries the arguments
    `_batch_transit_lnL_prod` / `_batch_eb_transit_lnL_prod` consume: f (the bin fluxes), W,
    delta = FPP_LADDER_DELTA_NAT, nb, and the two fold-wide constants of the bin likelihood —
    base_chi2 = sum(((f - 1)/sigma)**2), the chi2 of a flat 1.0 model accumulated sequentially,
    and lnnorm = -0.5*ln(2pi) - mean(ln sigma), upstream's one -ln sigma per curve (plan D9).
    """
    fold = prod_fold_arrays(rows, period, sigma_b)
    nb = len(rows)
    if nb == 0:
        raise ValueError("build_prod_fold: the production fold has no rows")
    f = np.ascontiguousarray(flux_b, dtype=np.float64).ravel()
    if f.size != nb:
        raise ValueError(f"flux_b must hold one value per fold row ({f.size} != {nb})")
    cadence_s = float(cadence_s)
    if not cadence_s > 0.0:
        raise ValueError("cadence_s must be positive")

    sig = fold["sigma"]
    base_chi2 = 0.0
    mean_lnsigma = 0.0
    for i in range(nb):
        d = (f[i] - 1.0) / sig[i]
        base_chi2 += d * d
        mean_lnsigma += math.log(sig[i])
    fold.update(f=f, W=cadence_s / 86400.0, delta=FPP_LADDER_DELTA_NAT, nb=nb,
                base_chi2=base_chi2, lnnorm=-0.5 * ln2pi - mean_lnsigma / nb)
    return fold


def prod_fold_renorm(fold, fr):
    """Shallow copy of a production fold renormalised to a star of flux fraction `fr`.

    The flux renormalisation of `renorm_flux` applied per bin: f' = (f - (1 - fr))/fr and
    sigma' = sigma/fr, with base_chi2 and lnnorm recomputed for the renormalised fold.  Only
    those three are new arrays: the time / node arrays (n, tmean, var_t, tmin, tmax, node_tab,
    pts_off, pts_t) and the raw bin means `flux` of `prod_fold_arrays` are shared with `fold`
    (the production kernels read `f`, never `flux`).
    """
    fr = float(fr)
    if not fr > 0.0:
        raise ValueError("prod_fold_renorm: fr must be positive")
    out = dict(fold)
    f = (np.asarray(fold["f"], dtype=np.float64) - (1.0 - fr)) / fr
    sig = np.asarray(fold["sigma"], dtype=np.float64) / fr
    nb = int(out["nb"])
    base_chi2 = 0.0
    mean_lnsigma = 0.0
    for i in range(nb):
        d = (f[i] - 1.0) / sig[i]
        base_chi2 += d * d
        mean_lnsigma += math.log(sig[i])
    out["f"] = f
    out["sigma"] = sig
    out["base_chi2"] = base_chi2
    out["lnnorm"] = -0.5 * ln2pi - mean_lnsigma / nb
    return out

def production_bin_model(P, k, aR, inc_rad, ecc, w_rad, u1, u2, cadence_s, bin_lo, bin_hi,
                         sigma_L=None):
    """Production model of the uniformly filled fold bins [bin_lo[i], bin_hi[i]) (days from
    mid-transit) at cadence `cadence_s` seconds — the ladder of 0.3 (j') with the exposure
    integrator of (j''), run ONCE for the whole set of bins.

    A uniform bin is the model the production-exposure reference integrates: n = 1000 points,
    mean time = the middle, var_t = (hi - lo)^2 / 12, level rungs = the uniform quantile times
    lo + (j + 0.5) / K * (hi - lo), exact rung = LADDER_K_QUANT uniform midpoints.  Returns a
    float64 array in the input order, 1.0 for the bins the transit does not touch.  sigma_L is
    the likelihood sigma (default 2e-5, the stress value of the plan).
    """
    P = float(P)
    lo = np.asarray(bin_lo, dtype=np.float64).ravel()
    hi = np.asarray(bin_hi, dtype=np.float64).ravel()
    nb = lo.size
    if nb == 0:
        return np.zeros(0, dtype=np.float64)

    n = np.full(nb, 1000, dtype=np.int64)
    tmean = 0.5 * (lo + hi)
    var_t = (hi - lo) * (hi - lo) / 12.0
    width = (hi - lo)[:, None]
    node_tab = np.zeros((nb, LADDER_EXACT_LEVEL - 1, LADDER_MAX_K), dtype=np.float64)
    for l in range(LADDER_EXACT_LEVEL - 1):
        K = 4 << l
        u = (np.arange(K, dtype=np.float64) + 0.5)[None, :] / K
        node_tab[:, l, :K] = lo[:, None] + u * width
    pts_off = np.arange(nb + 1, dtype=np.int64) * LADDER_K_QUANT
    u = (np.arange(LADDER_K_QUANT, dtype=np.float64)[None, :] + 0.5) / LADDER_K_QUANT
    pts_t = np.ascontiguousarray((lo[:, None] + u * width).ravel())
    sig = np.full(nb, 2.0e-5 if sigma_L is None else float(sigma_L), dtype=np.float64)

    out_idx = np.empty(nb, dtype=np.int64)
    out_M = np.empty(nb, dtype=np.float64)
    n_aff, _status, _n_eval, _n_fb = _draw_bin_models(float(cadence_s) / 86400.0,
                                                       FPP_LADDER_DELTA_NAT, nb, sig, n, tmean,
                                                       var_t, lo, hi, node_tab, pts_off, pts_t,
                                                       P, float(aR), float(inc_rad), float(ecc),
                                                       float(w_rad), float(k), float(u1),
                                                       float(u2), 1.0, 0.0, out_idx, out_M)
    M = np.ones(nb, dtype=np.float64)
    for i in range(int(n_aff)):
        M[out_idx[i]] = out_M[i]
    return M



# ============================================================
# 2. Batch transit evaluation for MC (numba parallel)
# ============================================================

@numba.njit(parallel=True, cache=True)
def _batch_transit_lnL(time, flux, sigma, ks, P_orbs, incs_rad,
                        a_Rs_arr, eccs, ws_rad, u1, u2,
                        companion_fluxratio, companion_is_host,
                        exptime, nsub):
    """Evaluate log-likelihood for N parameter sets in parallel.

    ``sigma`` is a per-bin float64 array the same length as ``time`` (a
    constant array reproduces the scalar case). Upstream's likelihood uses one
    -ln sigma per curve, i.e. the mean of ln sigma over the cadence bins:
    lnL = -0.5*ln(2pi) - mean(ln sigma) - 0.5*sum(((flux-model)/sigma)^2).
    That mean is accumulated sequentially ONCE, before the parallel loop.

    Returns array of lnL values (one per parameter set), -inf when a draw is
    not evaluated.
    """
    N = len(ks)
    T = len(time)
    lnL = np.full(N, -np.inf)

    mean_lnsigma = 0.0
    for t in range(T):
        mean_lnsigma += math.log(sigma[t])
    mean_lnsigma /= T

    for j in numba.prange(N):
        model = _lc_core(
            time, ks[j], 0.0, P_orbs[j], a_Rs_arr[j],
            incs_rad[j], eccs[j], ws_rad[j], u1, u2, exptime, nsub)

        # Apply flux dilution from companion/background star
        fr = companion_fluxratio[j]

        if fr > 0.0:
            if companion_is_host:
                F_dilute = 1.0 / (fr / (1.0 - fr))
            else:
                F_dilute = fr / (1.0 - fr)
            for t in range(T):
                model[t] = (model[t] + F_dilute) / (1.0 + F_dilute)

        chi2 = 0.0
        for t in range(T):
            diff = (flux[t] - model[t]) / sigma[t]
            chi2 += diff * diff
        lnL[j] = -0.5 * ln2pi - mean_lnsigma - 0.5 * chi2

    return lnL


@numba.njit(cache=True)
def _eb_secondary_min_flux(k_raw, P, a_Rs, cos_i, ecc, w_rad,
                           u1, u2, u1_sec, u2_sec,
                           sec_consistent, k_nudge_upstream):
    """Minimum in-eclipse flux of the SECONDARY eclipse of an EB pair.

    Swapped geometry: the occulting disc is the primary with radius ratio
    k_s = 1/k_raw relative to the (occulted) secondary. Sampled on 25 times
    spanning +/-0.05 d around the secondary eclipse centre, with no exposure
    supersampling (upstream's fixed 25-point secondary window).

    ``sec_consistent`` picks the two conventions:
      * True  — self-consistent: z is rescaled into units of the occulted
        (secondary) radius and the secondary's own limb darkening is used.
      * False — upstream's quirk: z stays in units of the primary radius and
        the host (target) coefficients u1, u2 are reused for the secondary.
    """
    n_samp = 25
    half_span = 0.05

    k_s = 1.0 / k_raw
    if k_nudge_upstream and abs(k_s - 1.0) < 1e-6:
        k_s = k_s * 0.999

    w_sec = w_rad + pi
    M_t_s = _mid_transit_mean_anomaly(ecc, w_sec)

    sec_flux = np.inf
    for i in range(n_samp):
        t_i = -half_span + 2.0 * half_span * i / (n_samp - 1)
        z = _z_one(t_i, 0.0, P, a_Rs, cos_i, ecc, w_sec, M_t_s)
        if sec_consistent:
            z = z * k_s
            uu1 = u1_sec
            uu2 = u2_sec
        else:
            uu1 = u1
            uu2 = u2
        f = _transit_flux(z, k_s, uu1, uu2)
        if f < sec_flux:
            sec_flux = f
    return sec_flux


@numba.njit(parallel=True, cache=True)
def _batch_eb_transit_lnL(time, flux, sigma, sigma_veto, ks, eb_fluxratios,
                           P_orbs, incs_rad, a_Rs_arr, eccs, ws_rad, u1, u2,
                           companion_fluxratio, companion_is_host,
                           exptime, nsub,
                           reject_sec, sec_consistent, k_nudge_upstream,
                           u1_sec, u2_sec):
    """Evaluate log-likelihood for N EB parameter sets.

    Port of upstream TRICERATOPS ``simulate_EB_transit_p`` + ``lnL_EB_p``:
    primary eclipse light curve, secondary eclipse evaluated with swapped
    geometry, then the two-step upstream dilution (EB pair, then third
    light). ``sigma`` is a per-bin float64 array (see _batch_transit_lnL);
    ``sigma_veto`` is the scalar depth a detectable secondary must reach.
    ``reject_sec`` applies the secondary cut — the twin-EB call passes False
    (upstream has no secondary cut there, defect D3). ``u1_sec``/``u2_sec``
    are per-draw limb-darkening coefficients of the secondary star.
    """
    N = len(ks)
    T = len(time)
    lnL = np.full(N, -np.inf)

    mean_lnsigma = 0.0
    for t in range(T):
        mean_lnsigma += math.log(sigma[t])
    mean_lnsigma /= T

    F_target = 1.0

    for j in numba.prange(N):
        k_raw = ks[j]
        k = k_raw
        if k_nudge_upstream and abs(k - 1.0) < 1e-6:
            k = k * 0.999

        # Primary eclipse model
        flux_m = _lc_core(
            time, k, 0.0, P_orbs[j], a_Rs_arr[j],
            incs_rad[j], eccs[j], ws_rad[j], u1, u2, exptime, nsub)

        # Secondary eclipse (swapped geometry), minimum in-eclipse flux
        sec_flux = _eb_secondary_min_flux(
            k_raw, P_orbs[j], a_Rs_arr[j], math.cos(incs_rad[j]),
            eccs[j], ws_rad[j], u1, u2, u1_sec[j], u2_sec[j],
            sec_consistent, k_nudge_upstream)

        # Flux ratios: EB pair and third light
        fr_eb = eb_fluxratios[j]
        F_EB = fr_eb / (1.0 - fr_eb) if fr_eb < 1.0 else 1e6
        fc = companion_fluxratio[j]
        F_comp = fc / (1.0 - fc) if fc > 0.0 else 0.0

        # Upstream dilution: first the EB pair against the target flux, then
        # the third light against the combined pair.
        if companion_is_host:
            Fc = max(F_comp, 1e-30)
            ratio = F_EB / Fc
            for t in range(T):
                flux_m[t] = (flux_m[t] + ratio) / (1.0 + ratio)
            sec_flux = (sec_flux + Fc / F_EB) / (1.0 + Fc / F_EB)
            F_dilute = F_target / (Fc + F_EB)
        else:
            ratio = F_EB / F_target
            for t in range(T):
                flux_m[t] = (flux_m[t] + ratio) / (1.0 + ratio)
            sec_flux = (sec_flux + F_target / F_EB) / (1.0 + F_target / F_EB)
            F_dilute = F_comp / (F_target + F_EB)

        for t in range(T):
            flux_m[t] = (flux_m[t] + F_dilute) / (1.0 + F_dilute)
        secdepth = 1.0 - (sec_flux + F_dilute) / (1.0 + F_dilute)

        # Reject if secondary eclipse is detectable
        if reject_sec and secdepth >= 1.5 * sigma_veto:
            continue

        chi2 = 0.0
        for t in range(T):
            diff = (flux[t] - flux_m[t]) / sigma[t]
            chi2 += diff * diff
        lnL[j] = -0.5 * ln2pi - mean_lnsigma - 0.5 * chi2

    return lnL


# ---- production twins: same likelihood, primary curve from the bin ladder ----
# The two functions below are the production-exposure counterparts of
# _batch_transit_lnL / _batch_eb_transit_lnL: the fold bins (prod_fold_arrays,
# TIME units) replace the cadence time array, and the primary model of a draw
# comes from _draw_bin_models (0.3 (j'), (j'')).  Only the bins the transit
# touches are evaluated: the likelihood sum is base_chi2 (the unaffected bins,
# whose model is exactly 1.0) with the affected bins swapped in, so the caller
# precomputes lnnorm = -0.5*ln(2pi) - mean(ln sigma) and
# base_chi2 = sum(((f - 1)/sigma)^2) once for the whole fold.

@numba.njit(parallel=True, cache=True)
def _batch_transit_lnL_prod(nb, f, sig, base_chi2, lnnorm, n, tmean, var_t,
                            tmin, tmax, node_tab, pts_off, pts_t, W, delta,
                            ks, P_orbs, incs_rad, a_Rs_arr, eccs, ws_rad,
                            u1, u2, companion_fluxratio, companion_is_host):
    """lnL of N planet draws on a production fold, in parallel.

    Dilution is the affine form of the one applied to the light curve in
    _batch_transit_lnL: model -> (model + F) / (1 + F) = dil_a * model + dil_b
    with F the companion flux ratio relative to the target (the host case
    inverts fr/(1-fr) exactly as upstream), so it is carried into
    _draw_bin_models instead of being applied to a curve.

    Returns (lnL, status, n_aff): status is the exposure-order status of the
    draw (2 = unconverged, never 'validate' it) and n_aff the number of bins
    the transit touched (0 = no transit with both outer contacts, lnL = the
    baseline of a flat 1.0 model, not -inf).
    """
    N = len(ks)
    lnL = np.full(N, -np.inf, dtype=np.float64)
    status = np.zeros(N, dtype=np.int64)
    n_aff_out = np.zeros(N, dtype=np.int64)

    for j in numba.prange(N):
        fr = companion_fluxratio[j]
        dil_a = 1.0
        dil_b = 0.0
        if fr > 0.0:
            if companion_is_host:
                F = 1.0 / (fr / (1.0 - fr))
            else:
                F = fr / (1.0 - fr)
            dil_a = 1.0 / (1.0 + F)
            dil_b = F / (1.0 + F)

        out_idx = np.empty(nb, dtype=np.int64)
        out_M = np.empty(nb, dtype=np.float64)
        n_aff, st, _n_eval, _n_fb = _draw_bin_models(
            W, delta, nb, sig, n, tmean, var_t, tmin, tmax, node_tab, pts_off,
            pts_t, P_orbs[j], a_Rs_arr[j], incs_rad[j], eccs[j], ws_rad[j],
            ks[j], u1, u2, dil_a, dil_b, out_idx, out_M)

        chi2 = float(base_chi2)
        for i in range(n_aff):
            b = out_idx[i]
            d = (f[b] - out_M[i]) / sig[b]
            chi2 += d * d
            d0 = (f[b] - 1.0) / sig[b]
            chi2 -= d0 * d0

        lnL[j] = float(lnnorm) - 0.5 * chi2

        status[j] = st
        n_aff_out[j] = n_aff

    return (lnL, status, n_aff_out)


@numba.njit(parallel=True, cache=True)
def _batch_eb_transit_lnL_prod(nb, f, sig, base_chi2, lnnorm, n, tmean, var_t,
                               tmin, tmax, node_tab, pts_off, pts_t, W, delta,
                               sigma_veto, ks, eb_fluxratios, P_orbs, incs_rad,
                               a_Rs_arr, eccs, ws_rad, u1, u2,
                               companion_fluxratio, companion_is_host,
                               reject_sec, sec_consistent, k_nudge_upstream,
                               u1_sec, u2_sec):
    """lnL of N EB draws on a production fold; the primary curve is the ladder.

    Per-draw logic is that of _batch_eb_transit_lnL (k nudge, the swapped-
    geometry secondary, the two-step upstream dilution, the secondary veto),
    but the two dilution steps of the PRIMARY are composed into one affine map
    applied inside _draw_bin_models: step 1 flux -> (flux + x)/(1 + x) with
    x = F_EB/Fc (host) or F_EB/F_target, step 2 -> (flux + F_dilute)/(1 +
    F_dilute), so dil_a = a2*a1 and dil_b = a2*b1 + b2 with a_i = 1/(1 + x_i),
    b_i = x_i/(1 + x_i).  The secondary and the veto are computed FIRST: a
    rejected draw never pays for the (expensive) bin models and keeps -inf.

    Returns (lnL, status, n_aff) as in _batch_transit_lnL_prod.
    """
    N = len(ks)
    lnL = np.full(N, -np.inf, dtype=np.float64)
    status = np.zeros(N, dtype=np.int64)
    n_aff_out = np.zeros(N, dtype=np.int64)

    F_target = 1.0

    for j in numba.prange(N):
        k_raw = ks[j]
        k = k_raw
        if k_nudge_upstream and abs(k - 1.0) < 1e-6:
            k = k * 0.999

        # Secondary eclipse (swapped geometry) and the flux ratios: cheap, and
        # they decide whether the draw is worth any bin model at all.
        sec_flux = _eb_secondary_min_flux(
            k_raw, P_orbs[j], a_Rs_arr[j], math.cos(incs_rad[j]),
            eccs[j], ws_rad[j], u1, u2, u1_sec[j], u2_sec[j],
            sec_consistent, k_nudge_upstream)

        fr_eb = eb_fluxratios[j]
        F_EB = fr_eb / (1.0 - fr_eb) if fr_eb < 1.0 else 1e6
        fc = companion_fluxratio[j]
        F_comp = fc / (1.0 - fc) if fc > 0.0 else 0.0

        if companion_is_host:
            Fc = max(F_comp, 1e-30)
            x = F_EB / Fc
            sec_flux = (sec_flux + Fc / F_EB) / (1.0 + Fc / F_EB)
            F_dilute = F_target / (Fc + F_EB)
        else:
            x = F_EB / F_target
            sec_flux = (sec_flux + F_target / F_EB) / (1.0 + F_target / F_EB)
            F_dilute = F_comp / (F_target + F_EB)

        secdepth = 1.0 - (sec_flux + F_dilute) / (1.0 + F_dilute)
        if reject_sec and secdepth >= 1.5 * sigma_veto:
            continue

        a1 = 1.0 / (1.0 + x)
        b1 = x / (1.0 + x)
        a2 = 1.0 / (1.0 + F_dilute)
        b2 = F_dilute / (1.0 + F_dilute)
        dil_a = a2 * a1
        dil_b = a2 * b1 + b2

        out_idx = np.empty(nb, dtype=np.int64)
        out_M = np.empty(nb, dtype=np.float64)
        n_aff, st, _n_eval, _n_fb = _draw_bin_models(
            W, delta, nb, sig, n, tmean, var_t, tmin, tmax, node_tab, pts_off,
            pts_t, P_orbs[j], a_Rs_arr[j], incs_rad[j], eccs[j], ws_rad[j],
            k, u1, u2, dil_a, dil_b, out_idx, out_M)

        chi2 = float(base_chi2)
        for i in range(n_aff):
            b = out_idx[i]
            d = (f[b] - out_M[i]) / sig[b]
            chi2 += d * d
            d0 = (f[b] - 1.0) / sig[b]
            chi2 -= d0 * d0

        lnL[j] = float(lnnorm) - 0.5 * chi2
        status[j] = st
        n_aff_out[j] = n_aff

    return (lnL, status, n_aff_out)


# ============================================================
# 3. Stellar relations (mass → radius, Teff, flux)
# ============================================================

# Spline nodes from Torres (2010) for M > 0.63 Msun
_Mass_Torres = np.array([0.26, 0.47, 0.59, 0.69, 0.87, 0.98, 1.085,
                          1.4, 1.65, 2.0, 2.5, 3.0, 4.4, 15.0, 40.0])
_Rad_Torres = np.array([0.28, 0.47, 0.60, 0.72, 0.9, 1.05, 1.2, 1.55,
                         1.8, 2.1, 2.4, 2.6, 3.0, 6.2, 11.0])
_Teff_Torres = np.array([3170, 3520, 3840, 4410, 5150, 5560, 5940, 6650,
                          7300, 8180, 9790, 11400, 15200, 30000, 42000])

# Cool dwarf spline for M <= 0.63 Msun
_Mass_cdwrf = np.array([0.1, 0.135, 0.2, 0.35, 0.48, 0.58, 0.63])
_Rad_cdwrf = np.array([0.12, 0.165, 0.23, 0.36, 0.48, 0.585, 0.6])
_Teff_cdwrf = np.array([2800, 3000, 3200, 3400, 3600, 3800, 4000])

_spl_R_torres = InterpolatedUnivariateSpline(_Mass_Torres, _Rad_Torres)
_spl_T_torres = InterpolatedUnivariateSpline(_Mass_Torres, _Teff_Torres)
_spl_R_cdwrf = InterpolatedUnivariateSpline(_Mass_cdwrf, _Rad_cdwrf)
_spl_T_cdwrf = InterpolatedUnivariateSpline(_Mass_cdwrf, _Teff_cdwrf)

# TESS flux relation: log10(flux) vs mass
_Mass_flux = np.array([0.1, 0.15, 0.23, 0.4, 0.58, 0.7, 0.9, 1.15,
                        1.45, 2.2, 2.8])
_logflux_nodes = np.array([-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1,
                            1.5, 2])
_spl_flux = InterpolatedUnivariateSpline(_Mass_flux, _logflux_nodes)


def stellar_relations(masses):
    """Estimate radii and Teffs from masses (Solar units)."""
    masses = np.atleast_1d(masses).copy()
    R = np.empty_like(masses)
    T = np.empty_like(masses)
    hot = masses > 0.63
    cool = ~hot
    if np.any(hot):
        R[hot] = _spl_R_torres(masses[hot])
        T[hot] = _spl_T_torres(masses[hot])
    if np.any(cool):
        R[cool] = _spl_R_cdwrf(np.clip(masses[cool], 0.1, 0.63))
        T[cool] = _spl_T_cdwrf(np.clip(masses[cool], 0.1, 0.63))
    R = np.clip(R, 0.1, None)
    T = np.clip(T, 2800, None)
    return R, T


def flux_relation(masses):
    """Estimate relative TESS-band flux from mass."""
    return 10.0 ** _spl_flux(np.atleast_1d(masses))


# ============================================================
# 4. Limb darkening coefficients
# ============================================================

_ldc_data = None


class LDCTableMissing(FileNotFoundError):
    """No limb-darkening table found: never fall back to solar defaults."""
    skip_reason = "ldc_table_missing"


class LDCNoRow(LookupError):
    """No limb-darkening row for the requested (Z, Teff, logg) combination."""
    skip_reason = "ldc_no_row"


def _load_ldc():
    """Load LDC table from bundled CSV or triceratops package.

    Raises LDCTableMissing if no candidate path exists — the caller must skip
    the candidate rather than get silently solar-like coefficients. A failure
    is never cached (_ldc_data stays None).
    """
    global _ldc_data
    if _ldc_data is not None:
        return

    # Try local copy first, then triceratops package data
    candidates = [
        os.path.join(os.path.dirname(__file__), "ldc_tess.csv"),
    ]
    try:
        from pkg_resources import resource_filename
        candidates.append(resource_filename('triceratops', 'data/ldc_tess.csv'))
    except Exception:
        pass

    for path in candidates:
        if os.path.exists(path):
            from pandas import read_csv
            df = read_csv(path)
            _ldc_data = {
                'Z': np.array(df.Z, dtype=float),
                'Teff': np.array(df.Teff, dtype=int),
                'logg': np.array(df.logg, dtype=float),
                'u1': np.array(df.aLSM, dtype=float),
                'u2': np.array(df.bLSM, dtype=float),
            }
            return

    raise LDCTableMissing(
        "no limb-darkening table found, tried: " + ", ".join(candidates))


def get_ldc(Teff, logg, Z=0.0):
    """Look up quadratic limb darkening coefficients for TESS."""
    _load_ldc()
    d = _ldc_data
    best_Z = d['Z'][np.argmin(np.abs(d['Z'] - Z))]
    best_T = d['Teff'][np.argmin(np.abs(d['Teff'] - Teff))]
    best_g = d['logg'][np.argmin(np.abs(d['logg'] - logg))]
    mask = (d['Z'] == best_Z) & (d['Teff'] == best_T) & (d['logg'] == best_g)
    if np.any(mask):
        return float(d['u1'][mask].flat[0]), float(d['u2'][mask].flat[0])
    raise LDCNoRow(f"no limb-darkening row for Z={best_Z} Teff={best_T} "
                    f"logg={best_g}")


# ============================================================
# 5. Prior sampling (Giacalone & Dressing 2020 / Moe & Di Stefano 2017)
# ============================================================

def _sample_rp(N, M_s, rng=np.random):
    """Sample planet radii from broken power law prior (Earth radii).

    ``M_s`` is the host mass: a scalar (target / neighbour host) or, for the S family (planet around the drawn bound
    companion, plan step 4), an array of N host masses: each draw then uses the exponents of ITS host
    (M > 0.45: p2 = -4, else -7), exactly upstream's `sample_rp(x, M_s_array)`.
    """
    x = rng.rand(N)
    R_break1, R_break2, R_min, R_max = 3.0, 6.0, 0.5, 20.0
    if np.ndim(M_s) != 0:
        M_arr = np.asarray(M_s, dtype=np.float64)
        if M_arr.shape != (N,):
            raise ValueError(f"host mass array must have shape ({N},), got {M_arr.shape}")
        out = np.empty(N)
        for heavy, p2 in ((True, -4.0), (False, -7.0)):
            sel = (M_arr > 0.45) == heavy
            if not np.any(sel):
                continue
            p1, p3 = 0.0, -0.5
            A1 = R_break1**p1 / R_break1**p2
            A2 = R_break2**p2 / R_break2**p3
            I1 = (R_break1**(p1 + 1) - R_min**(p1 + 1)) / (p1 + 1)
            I2 = A1 * (R_break2**(p2 + 1) - R_break1**(p2 + 1)) / (p2 + 1)
            I3 = A2 * A1 * (R_max**(p3 + 1) - R_break2**(p3 + 1)) / (p3 + 1)
            Norm = 1.0 / (I1 + I2 + I3)
            xs = x[sel]
            m1 = xs <= Norm * I1
            m2 = (xs > Norm * I1) & (xs <= Norm * (I1 + I2))
            m3 = xs > Norm * (I1 + I2)
            xs[m1] = (xs[m1] / Norm * (p1 + 1) + R_min**(p1 + 1))**(1 / (p1 + 1))
            xs[m2] = ((xs[m2] / Norm - I1) * (p2 + 1) / A1 + R_break1**(p2 + 1))**(1 / (p2 + 1))
            xs[m3] = ((xs[m3] / Norm - I1 - I2) * (p3 + 1) / (A1 * A2) + R_break2**(p3 + 1))**(1 / (p3 + 1))
            out[sel] = xs
        return out

    if M_s > 0.45:
        p1, p2, p3 = 0.0, -4.0, -0.5
    else:
        p1, p2, p3 = 0.0, -7.0, -0.5

    A1 = R_break1**p1 / R_break1**p2
    A2 = R_break2**p2 / R_break2**p3
    I1 = (R_break1**(p1 + 1) - R_min**(p1 + 1)) / (p1 + 1)
    I2 = A1 * (R_break2**(p2 + 1) - R_break1**(p2 + 1)) / (p2 + 1)
    I3 = A2 * A1 * (R_max**(p3 + 1) - R_break2**(p3 + 1)) / (p3 + 1)
    Norm = 1.0 / (I1 + I2 + I3)

    m1 = x <= Norm * I1
    m2 = (x > Norm * I1) & (x <= Norm * (I1 + I2))
    m3 = (x > Norm * (I1 + I2))

    x[m1] = (x[m1] / Norm * (p1 + 1) + R_min**(p1 + 1))**(1 / (p1 + 1))
    x[m2] = ((x[m2] / Norm - I1) * (p2 + 1) / A1
             + R_break1**(p2 + 1))**(1 / (p2 + 1))
    x[m3] = ((x[m3] / Norm - I1 - I2) * (p3 + 1) / (A1 * A2)
             + R_break2**(p3 + 1))**(1 / (p3 + 1))
    return x


def _sample_inc(N, rng=np.random):
    """Sample inclinations from sin(i) prior (degrees)."""
    x = rng.rand(N)
    Norm = 1.0  # cos(0) - cos(90) = 1
    return np.arccos(1.0 - x / Norm) * 180.0 / pi


def _sample_ecc(N, planet=True, P_orb=10.0, rng=np.random):
    """Sample eccentricities from Beta (planets) or power law (binaries)."""
    if planet:
        return beta_dist.rvs(0.867, 3.030, size=N, random_state=rng)
    else:
        nu_plus_1 = 0.2 if P_orb <= 10 else 0.6
        return powerlaw_dist.rvs(nu_plus_1, size=N, random_state=rng)


def _sample_w(N, rng=np.random):
    """Sample argument of periastron uniformly (degrees)."""
    return rng.rand(N) * 360.0


def _sample_q(N, M_s, rng=np.random):
    """Sample mass ratios for short-period binaries (Moe & Di Stefano 2017)."""
    x = rng.rand(N)
    if M_s >= 1.0:
        q_min = 0.1
    elif M_s >= 0.3:
        q_min = 0.1 / M_s
    elif M_s > 0.1:
        q_min = 0.1 / M_s
    else:
        return np.ones(N)

    p1, p2 = 0.3, -0.5
    F_twin = 0.30
    # U9: upstream branches on M_s, not on q_min, so for 0.3 <= M_s < 1/3 it
    # keeps the 3-segment form even though q_min = 0.1/M_s > 0.3 (I1 < 0).
    upstream = switch_is("FPP_Q_BRANCH", "upstream")
    three_seg = (M_s >= 0.3) if upstream else (q_min < 0.3)
    A1 = (0.3**p1) / (0.3**p2) if three_seg else 1.0

    if three_seg:
        A2 = 1 + (F_twin / (1 - F_twin)) * (
            (1.0**(p2 + 1) - 0.3**(p2 + 1)) / (p2 + 1)) / (
            (1.0**(p2 + 1) - 0.95**(p2 + 1)) / (p2 + 1))
        I1 = (0.3**(p1 + 1) - q_min**(p1 + 1)) / (p1 + 1)
        I2 = A1 * (0.95**(p2 + 1) - 0.3**(p2 + 1)) / (p2 + 1)
        I3 = A2 * A1 * (1.0**(p2 + 1) - 0.95**(p2 + 1)) / (p2 + 1)
        Norm = 1.0 / (I1 + I2 + I3)

        m1 = x <= Norm * I1
        m2 = (x > Norm * I1) & (x <= Norm * (I1 + I2))
        m3 = x > Norm * (I1 + I2)
        x[m1] = (x[m1] / Norm * (p1 + 1) + q_min**(p1 + 1))**(1 / (p1 + 1))
        x[m2] = ((x[m2] / Norm - I1) * (p2 + 1) / A1
                  + 0.3**(p2 + 1))**(1 / (p2 + 1))
        x[m3] = ((x[m3] / Norm - I1 - I2) * (p2 + 1) / (A1 * A2)
                  + 0.95**(p2 + 1))**(1 / (p2 + 1))
    else:
        A2 = 1 + (F_twin / (1 - F_twin)) * (
            (1.0**(p2 + 1) - q_min**(p2 + 1)) / (p2 + 1)) / (
            (1.0**(p2 + 1) - 0.95**(p2 + 1)) / (p2 + 1))
        I2 = (0.95**(p2 + 1) - q_min**(p2 + 1)) / (p2 + 1)
        I3 = A2 * (1.0**(p2 + 1) - 0.95**(p2 + 1)) / (p2 + 1)
        Norm = 1.0 / (I2 + I3)

        m2 = x <= Norm * I2
        m3 = x > Norm * I2
        x[m2] = (x[m2] / Norm * (p2 + 1) + q_min**(p2 + 1))**(1 / (p2 + 1))
        x[m3] = ((x[m3] / Norm - I2) * (p2 + 1) / A2
                  + 0.95**(p2 + 1))**(1 / (p2 + 1))

    if upstream:
        return x   # U9: upstream never clips the drawn q back into [q_min, 1]
    return np.clip(x, q_min, 1.0)


def _sample_q_companion(N, M_s, rng=np.random):
    """Sample mass ratios for long-period companions."""
    x = rng.rand(N)
    q_min = max(0.1 / M_s, 0.1) if M_s > 0.1 else 1.0

    if M_s <= 0.1:
        return np.ones(N)

    p1, p2 = 0.3, -0.95
    F_twin = 0.05

    # U9: same M_s-based branch as _sample_q (see the note there).
    upstream = switch_is("FPP_Q_BRANCH", "upstream")
    three_seg = (M_s >= 0.3) if upstream else (q_min < 0.3)

    if three_seg:
        A1 = (0.3**p1) / (0.3**p2)
        A2 = 1 + (F_twin / (1 - F_twin)) * (
            (1.0**(p2 + 1) - 0.3**(p2 + 1)) / (p2 + 1)) / (
            (1.0**(p2 + 1) - 0.95**(p2 + 1)) / (p2 + 1))
        I1 = (0.3**(p1 + 1) - q_min**(p1 + 1)) / (p1 + 1)
        I2 = A1 * (0.95**(p2 + 1) - 0.3**(p2 + 1)) / (p2 + 1)
        I3 = A2 * A1 * (1.0**(p2 + 1) - 0.95**(p2 + 1)) / (p2 + 1)
        Norm = 1.0 / (I1 + I2 + I3)

        m1 = x <= Norm * I1
        m2 = (x > Norm * I1) & (x <= Norm * (I1 + I2))
        m3 = x > Norm * (I1 + I2)
        x[m1] = (x[m1] / Norm * (p1 + 1) + q_min**(p1 + 1))**(1 / (p1 + 1))
        x[m2] = ((x[m2] / Norm - I1) * (p2 + 1) / A1
                  + 0.3**(p2 + 1))**(1 / (p2 + 1))
        x[m3] = ((x[m3] / Norm - I1 - I2) * (p2 + 1) / (A1 * A2)
                  + 0.95**(p2 + 1))**(1 / (p2 + 1))
    else:
        A2 = 1 + (F_twin / (1 - F_twin)) * (
            (1.0**(p2 + 1) - q_min**(p2 + 1)) / (p2 + 1)) / (
            (1.0**(p2 + 1) - 0.95**(p2 + 1)) / (p2 + 1))
        I2 = (0.95**(p2 + 1) - q_min**(p2 + 1)) / (p2 + 1)
        I3 = A2 * (1.0**(p2 + 1) - 0.95**(p2 + 1)) / (p2 + 1)
        Norm = 1.0 / (I2 + I3)

        m2 = x <= Norm * I2
        m3 = x > Norm * I2
        x[m2] = (x[m2] / Norm * (p2 + 1) + q_min**(p2 + 1))**(1 / (p2 + 1))
        x[m3] = ((x[m3] / Norm - I2) * (p2 + 1) / A2
                  + 0.95**(p2 + 1))**(1 / (p2 + 1))

    if upstream:
        return x   # U9: upstream does not clip q back into [q_min, 1]
    return np.clip(x, q_min, 1.0)


# ============================================================
# 5b. Bound-companion prior (PTP/PEB only)
# ============================================================
#
# Upstream TRICERATOPS folds a per-draw ``lnprior_companion`` term into the
# PTP/PEB Monte-Carlo average (marginal_likelihoods.py:lnZ_PTP/lnZ_PEB) that
# this port did not implement until now (see 78fa512's docstring note). It
# is the log probability, from the Moe & Di Stefano (2017) bound-companion
# rate, that a companion drawn with the given mass ratio/parallax/contrast
# is actually bound to the target — everything else in PTP/PEB (the
# transit/eclipse geometry, the flux dilution) is unchanged by this term;
# it only reweights the Monte-Carlo average. It does NOT apply to TTP/TEB
# (no companion in that scenario, upstream's lnZ_TTP/lnZ_TEB have no such
# term at all) nor to this port's NTP/NEB (nearby-star scenarios, which
# reuse _lnZ_TTP/_lnZ_TEB with the default lnprior_companion=0.0) — upstream
# NTP/NEB use a DIFFERENT function, lnprior_background, for an unrelated
# "is this a real background star" prior that this port has never
# implemented; that remains a documented residual difference, out of scope
# here. S uses this bound-companion prior; D/B use the field-density prior below.
#
# Ported from triceratops/priors.py:lnprior_bound_TP / lnprior_bound_EB and
# triceratops/funcs.py:separation_at_contrast. This port never supplies a
# contrast-curve file (matches upstream's own contrast_curve_file=None /
# molusc_file=None default path — the only one this port's calc_fpp_native
# ever exercises), so separations/contrasts are hardcoded to upstream's own
# no-contrast-curve stand-in ([2.2 arcsec] -> [1.0 mag]), and molusc_file's
# lnprior_companion=np.zeros(N) branch is simply never reached, exactly as
# upstream never reaches it without a molusc_file either.

def _separation_at_contrast(delta_mags):
    """Limiting separation (arcsec) at delta_mags, from a contrast curve.

    Mirrors triceratops/funcs.py:separation_at_contrast(delta_mags,
    separations, contrasts) = np.interp(delta_mags, contrasts,
    separations), with separations/contrasts fixed at upstream's
    no-contrast-curve default [2.2] / [1.0]. With a single-point xp,
    np.interp returns that one fp value (1.0 arcsec) for every delta_mag —
    this is upstream's actual behavior, not a simplification made here.
    """
    return np.interp(delta_mags, np.array([1.0]), np.array([2.2]))


def _lnprior_bound_TP(M_s, plx, delta_mags):
    """Log prior of a bound companion, PTP convention (Porb_comp > 2500 d).

    Mirrors triceratops/priors.py:lnprior_bound_TP exactly (Moe & Di
    Stefano 2017). ``delta_mags`` must already be ``np.abs()``-ed by the
    caller, matching upstream's call site.
    """
    if np.isnan(plx):
        plx = 0.1
    d = 1000.0 / plx
    seps = d * _separation_at_contrast(delta_mags)

    if M_s >= 1.0:
        f1 = 0.020 + 0.04 * np.log10(M_s) + 0.07 * (np.log10(M_s))**2
        f2 = 0.039 + 0.07 * np.log10(M_s) + 0.01 * (np.log10(M_s))**2
        f3 = 0.078 - 0.05 * np.log10(M_s) + 0.04 * (np.log10(M_s))**2
        alpha = 0.018
        dlogP = 0.7
        max_Porbs = ((4 * pi**2) / (G_cgs * M_s * Msun)
                     * (seps * au)**3)**(1 / 2) / 86400
        log_mP = np.log10(max_Porbs)

        t4_partial = (
            alpha * dlogP * (log_mP - 3.4)
            + f2 * (log_mP - 3.4)
            + (f3 - f2 - alpha * dlogP)
            * (0.238095 * log_mP**2 - 0.952381 * log_mP + 0.485714))
        t4 = (alpha * dlogP * (5.5 - 3.4) + f2 * (5.5 - 3.4)
              + (f3 - f2 - alpha * dlogP)
              * (0.238095 * 5.5**2 - 0.952381 * 5.5 + 0.485714))
        t5_partial = f3 * (3.33333 - 17.3566 * np.exp(-0.3 * log_mP))
        t5 = f3 * (3.33333 - 17.3566 * np.exp(-0.3 * 8.0))

        f_comp = np.zeros(len(seps))
        m = (log_mP >= 3.4) & (log_mP < 5.5)
        f_comp[m] = t4_partial[m]
        m = (log_mP >= 5.5) & (log_mP < 8.0)
        f_comp[m] = t4 + t5_partial[m]
        m = (log_mP >= 8.0)
        f_comp[m] = t4 + t5
        lnprior_bound = np.log(f_comp)
    else:
        M_act = M_s
        M_s = 1.0
        f1 = 0.020 + 0.04 * np.log10(M_s) + 0.07 * (np.log10(M_s))**2
        f2 = 0.039 + 0.07 * np.log10(M_s) + 0.01 * (np.log10(M_s))**2
        f3 = 0.078 - 0.05 * np.log10(M_s) + 0.04 * (np.log10(M_s))**2
        alpha = 0.018
        dlogP = 0.7
        max_Porbs = ((4 * pi**2) / (G_cgs * M_s * Msun)
                     * (seps * au)**3)**(1 / 2) / 86400
        log_mP = np.log10(max_Porbs)

        t4_partial = (
            alpha * dlogP * (log_mP - 3.4)
            + f2 * (log_mP - 3.4)
            + (f3 - f2 - alpha * dlogP)
            * (0.238095 * log_mP**2 - 0.952381 * log_mP + 0.485714))
        t4 = (alpha * dlogP * (5.5 - 3.4) + f2 * (5.5 - 3.4)
              + (f3 - f2 - alpha * dlogP)
              * (0.238095 * 5.5**2 - 0.952381 * 5.5 + 0.485714))
        t5_partial = f3 * (3.33333 - 17.3566 * np.exp(-0.3 * log_mP))
        t5 = f3 * (3.33333 - 17.3566 * np.exp(-0.3 * 8.0))

        f_comp = np.zeros(len(seps))
        m = (log_mP >= 3.4) & (log_mP < 5.5)
        f_comp[m] = t4_partial[m]
        m = (log_mP >= 5.5) & (log_mP < 8.0)
        f_comp[m] = t4 + t5_partial[m]
        m = (log_mP >= 8.0)
        f_comp[m] = t4 + t5
        f_act = 0.65 * f_comp + 0.35 * f_comp * M_act
        f_act[f_act < 0.0] = 0.0
        lnprior_bound = np.log(f_act)

    return lnprior_bound


def _lnprior_bound_EB(M_s, plx, delta_mags):
    """Log prior of a bound companion, PEB convention (Porb_comp > 10 d).

    Mirrors triceratops/priors.py:lnprior_bound_EB exactly (same Moe & Di
    Stefano 2017 rate as _lnprior_bound_TP, but the tertiary-companion
    period floor is 10 days rather than 2500 days, so the non-zero mask
    ranges start two steps earlier). ``delta_mags`` must already be
    ``np.abs()``-ed by the caller, matching upstream's call site.
    """
    if np.isnan(plx):
        plx = 0.1
    d = 1000.0 / plx
    seps = d * _separation_at_contrast(delta_mags)

    if M_s >= 1.0:
        f1 = 0.020 + 0.04 * np.log10(M_s) + 0.07 * (np.log10(M_s))**2
        f2 = 0.039 + 0.07 * np.log10(M_s) + 0.01 * (np.log10(M_s))**2
        f3 = 0.078 - 0.05 * np.log10(M_s) + 0.04 * (np.log10(M_s))**2
        alpha = 0.018
        dlogP = 0.7
        max_Porbs = ((4 * pi**2) / (G_cgs * M_s * Msun)
                     * (seps * au)**3)**(1 / 2) / 86400
        log_mP = np.log10(max_Porbs)

        t2_partial = (
            0.5 * (log_mP - 1.0)
            * (2.0 * f1 + (f2 - f1 - alpha * dlogP) * (log_mP - 1.0)))
        t2 = (0.5 * (2.0 - 1.0)
              * (2.0 * f1 + (f2 - f1 - alpha * dlogP) * (2.0 - 1.0)))
        t3_partial = (
            0.5 * alpha * (log_mP**2 - 5.4 * log_mP + 6.8)
            + f2 * (log_mP - 2.0))
        t3 = 0.5 * alpha * (3.4**2 - 5.4 * 3.4 + 6.8) + f2 * (3.4 - 2.0)
        t4_partial = (
            alpha * dlogP * (log_mP - 3.4)
            + f2 * (log_mP - 3.4)
            + (f3 - f2 - alpha * dlogP)
            * (0.238095 * log_mP**2 - 0.952381 * log_mP + 0.485714))
        t4 = (alpha * dlogP * (5.5 - 3.4) + f2 * (5.5 - 3.4)
              + (f3 - f2 - alpha * dlogP)
              * (0.238095 * 5.5**2 - 0.952381 * 5.5 + 0.485714))
        t5_partial = f3 * (3.33333 - 17.3566 * np.exp(-0.3 * log_mP))
        t5 = f3 * (3.33333 - 17.3566 * np.exp(-0.3 * 8.0))

        f_comp = np.zeros(len(seps))
        m = (log_mP >= 1.0) & (log_mP < 2.0)
        f_comp[m] = t2_partial[m]
        m = (log_mP >= 2.0) & (log_mP < 3.4)
        f_comp[m] = t2 + t3_partial[m]
        m = (log_mP >= 3.4) & (log_mP < 5.5)
        f_comp[m] = t2 + t3 + t4_partial[m]
        m = (log_mP >= 5.5) & (log_mP < 8.0)
        f_comp[m] = t2 + t3 + t4 + t5_partial[m]
        m = (log_mP >= 8.0)
        f_comp[m] = t2 + t3 + t4 + t5
        lnprior_bound = np.log(f_comp)
    else:
        M_act = M_s
        M_s = 1.0
        f1 = 0.020 + 0.04 * np.log10(M_s) + 0.07 * (np.log10(M_s))**2
        f2 = 0.039 + 0.07 * np.log10(M_s) + 0.01 * (np.log10(M_s))**2
        f3 = 0.078 - 0.05 * np.log10(M_s) + 0.04 * (np.log10(M_s))**2
        alpha = 0.018
        dlogP = 0.7
        max_Porbs = ((4 * pi**2) / (G_cgs * M_s * Msun)
                     * (seps * au)**3)**(1 / 2) / 86400
        log_mP = np.log10(max_Porbs)

        t2_partial = (
            0.5 * (log_mP - 1.0)
            * (2.0 * f1 + (f2 - f1 - alpha * dlogP) * (log_mP - 1.0)))
        t2 = (0.5 * (2.0 - 1.0)
              * (2.0 * f1 + (f2 - f1 - alpha * dlogP) * (2.0 - 1.0)))
        t3_partial = (
            0.5 * alpha * (log_mP**2 - 5.4 * log_mP + 6.8)
            + f2 * (log_mP - 2.0))
        t3 = 0.5 * alpha * (3.4**2 - 5.4 * 3.4 + 6.8) + f2 * (3.4 - 2.0)
        t4_partial = (
            alpha * dlogP * (log_mP - 3.4)
            + f2 * (log_mP - 3.4)
            + (f3 - f2 - alpha * dlogP)
            * (0.238095 * log_mP**2 - 0.952381 * log_mP + 0.485714))
        t4 = (alpha * dlogP * (5.5 - 3.4) + f2 * (5.5 - 3.4)
              + (f3 - f2 - alpha * dlogP)
              * (0.238095 * 5.5**2 - 0.952381 * 5.5 + 0.485714))
        t5_partial = f3 * (3.33333 - 17.3566 * np.exp(-0.3 * log_mP))
        t5 = f3 * (3.33333 - 17.3566 * np.exp(-0.3 * 8.0))

        f_comp = np.zeros(len(seps))
        m = (log_mP >= 1.0) & (log_mP < 2.0)
        f_comp[m] = t2_partial[m]
        m = (log_mP >= 2.0) & (log_mP < 3.4)
        f_comp[m] = t2 + t3_partial[m]
        m = (log_mP >= 3.4) & (log_mP < 5.5)
        f_comp[m] = t2 + t3 + t4_partial[m]
        m = (log_mP >= 5.5) & (log_mP < 8.0)
        f_comp[m] = t2 + t3 + t4 + t5_partial[m]
        m = (log_mP >= 8.0)
        f_comp[m] = t2 + t3 + t4 + t5
        f_act = 0.65 * f_comp + 0.35 * f_comp * M_act
        f_act[f_act < 0.0] = 0.0
        lnprior_bound = np.log(f_act)

    return lnprior_bound


def _lnprior_companion(M_s, plx, fluxratios_comp, scenario):
    """Per-draw ``lnprior_companion`` array fed into PTP/PEB's Z average.

    Mirrors the "calculate priors for companions" block shared by upstream
    lnZ_PTP/lnZ_PEB (contrast_curve_file=None branch): computes delta_mags
    from the already-drawn ``fluxratios_comp``, looks up the bound rate
    (``scenario`` picks TP vs EB convention), then clips it to a proper log
    probability (``<= 0``) and vetoes companions brighter than the target
    (``delta_mags > 0`` -> ``-inf``, matches upstream's post-hoc mask using
    the SIGNED delta_mags even though the prior itself was computed from
    ``np.abs(delta_mags)``).
    """
    delta_mags = 2.5 * np.log10(fluxratios_comp / (1 - fluxratios_comp))
    prior_fn = _lnprior_bound_TP if scenario == "TP" else _lnprior_bound_EB
    lnprior_companion = prior_fn(M_s, plx, np.abs(delta_mags))
    lnprior_companion[lnprior_companion > 0.0] = 0.0
    lnprior_companion[delta_mags > 0.0] = -np.inf
    return lnprior_companion


# ============================================================
# 6. Marginal likelihood for each scenario
# ============================================================

#: Telemetry snapshot kept for the worker payload: FPP has no GPU path any
#: more (plan decision (e)), so the fallback counters are constants. Callers
#: (worker telemetry) may still import gpu_fallback_stats().

_GPU_FALLBACK_STATS = {
    "consecutive": 0,
    "total": 0,
    "disabled": False,
    "last_error": None,
    "note": "FPP is CPU-only; GPU dispatch removed",
}


def gpu_fallback_stats() -> dict:
    """Snapshot for the worker's telemetry payload — always the constant dict.

    Used to count GPU->CPU fallbacks so a worker silently running on the CPU
    was visible (project rule 2). Now the CPU is the only FPP backend by
    design, so there is nothing to count and the dict is constant.
    """
    return dict(_GPU_FALLBACK_STATS)


def _dispatch_lnL(cpu_fn, args):
    """Run one batched lnL kernel on the CPU (numba), under the parallel guard.

    Used to be a GPU-try/CPU-fallback dispatch shared by the three batched
    lnL call sites (TTP transit, TEB primary, TEB twin — extracted from 3
    verbatim copies in the 2026-07-02 cleanup review, D7). The GPU branch is
    gone: the GPU kernels carry the OLD geometry and cannot be made
    bit-exact, so FPP is CPU-only (plan decision (e)).

    ``numba_parallel_guard()`` is still required: on Apple Silicon numba's
    only threading layer (workqueue) is not thread-safe against the
    background injection-recovery thread also entering a parallel region
    (see 31383add).
    """
    with numba_parallel_guard():
        out = cpu_fn(*args)
    # Draws whose Kepler solve did not converge carry lnL = NaN (`_kepler_eq`);
    # count them per candidate (reported as kepler_unconverged_draws).
    lnL = out[0] if isinstance(out, tuple) else out
    _EXPOSURE_STATS["kepler_unconverged_draws"] += int(
        np.count_nonzero(np.isnan(np.asarray(lnL))))
    return out


#: Exposure diagnostics of the PRODUCTION fold likelihood (plan 0.3 (j'), (j'')):
#: the flux-weighted share of draws whose bin ladder fell through to the exact
#: rung (status 1 = the oracle fallback of `_choose_order`) and whose ladder never
#: converged (status 2).  Both are shares of the weight of the draws the batch
#: actually evaluated (finite lnL), so a scenario whose draws are all vetoes
#: contributes nothing.  Reset by `calc_fpp_native`, kept as the MAX over the
#: scenario calls of one candidate: a candidate whose ladder is unreliable
#: anywhere in its scenario set is an unreliable candidate.

_EXPOSURE_STATS = {"fallback_share": 0.0, "unconverged_share": 0.0,
                   "kepler_unconverged_draws": 0}


def _note_exposure_stats(lnL, status):
    """Fold one production batch's exposure status into `_EXPOSURE_STATS` (max).

    ``lnL`` / ``status`` are the per-draw outputs of `_batch_transit_lnL_prod` /
    `_batch_eb_transit_lnL_prod` on the draws it was called with.  The weights
    are w = exp(lnL - max(lnL)) over the finite ones, so the share is the
    probability a draw of this scenario pays for the exact rung (or is
    unconverged) rather than a plain count.
    """
    lnL = np.asarray(lnL, dtype=np.float64)
    status = np.asarray(status, dtype=np.int64)
    keep = np.isfinite(lnL)
    if not np.any(keep):
        return
    lw = lnL[keep]
    m = float(np.max(lw))
    w = np.exp(lw - m)
    tot = float(np.sum(w))
    if not tot > 0.0:
        return
    st = status[keep]
    fb = float(np.sum(w[st == 1])) / tot
    unc = float(np.sum(w[st == 2])) / tot
    if fb > _EXPOSURE_STATS["fallback_share"]:
        _EXPOSURE_STATS["fallback_share"] = fb
    if unc > _EXPOSURE_STATS["unconverged_share"]:
        _EXPOSURE_STATS["unconverged_share"] = unc



def nearby_lnz_offset(fr_t, fr_i):
    """lnZ offset added to every nearby-star scenario (NTP, NEB, NEBx2P) of neighbour i (plan D16, decision (r)).

    FPP_NORM = "upstream": 0 (upstream's convention: the neighbour's evidence in its own frame).
    FPP_NORM = "data": ln(fr_t / fr_i), the data-space normalisation (frame invariance of the evidence).
    """
    if switch_is("FPP_NORM", "data"):
        return math.log(float(fr_t) / float(fr_i))
    return 0.0


def _lnZ_from_logw(logw):
    """True log of the mean of exp(logw) over ALL draws (log-sum-exp, plan D9).

    ``logw`` is the per-draw log-weight lnL + lnprior_companion, length N, with
    -inf for draws that were never evaluated (non-transiting / rejected). The
    mean is over the full draw set, exactly like upstream's
    ``np.log(np.mean(np.exp(...)))`` — but computed in log space, so it cannot
    underflow. The old code returned log(mean(exp(lnL + lnprior + 600))), i.e.
    the true lnZ shifted by +600, and still underflowed for the deep-chi2
    draws that dominate an EB scenario (defect D9).
    """
    w = np.asarray(logw, dtype=np.float64)
    w = np.where(np.isnan(w), -np.inf, w)
    if not np.any(np.isfinite(w)):
        if _DRAW_DIAG is not None:
            _DRAW_DIAG.append({"ess": 0.0, "n": int(len(w)), "max_share": 0.0, "tail_scale": float("nan")})
        return -np.inf
    m = float(np.max(w))
    e = np.exp(w - m)
    s1 = float(np.sum(e))
    if _DRAW_DIAG is not None:
        # draw-weight diagnostics of this scenario (plan decision (g), step 7): ESS = (sum w)^2 / sum w^2 and the
        # share of the heaviest draw, both on the weights w = exp(logw) over the N draws (the same set whose mean is Z)
        s2 = float(np.sum(e * e))
        _DRAW_DIAG.append({"ess": (s1 * s1) / s2, "n": int(len(w)), "max_share": 1.0 / s1,
                           "tail_scale": _tail_scale(w)})
    return float(m + np.log(s1) - np.log(len(w)))


def _tail_scale(w):
    """Extreme-value scale beta (nats) of the upper tail of the log-weights, from the top order statistics of THIS run.

    For an exponential upper tail the spacings obey E[x_(i) - x_(i+1)] = beta / i, so beta = (x_(1) - x_(k)) / H_(k-1)
    over the top k = MC_TAIL_K_TOP finite draws; the maximum then has sd ~ 1.28 beta, and 10x more draws lift it by
    ~ beta ln 10.  NaN if fewer than 3 finite draws.
    """
    fin = w[np.isfinite(w)]
    kk = min(MC_TAIL_K_TOP, fin.size)
    if kk < 3:
        return float("nan")
    top = np.partition(fin, fin.size - kk)[fin.size - kk:]
    h = sum(1.0 / i for i in range(1, kk))
    return float((np.max(top) - np.min(top)) / h)


# Sink of the per-scenario draw-weight diagnostics of ONE Monte-Carlo pass (None = not recording).  calc_fpp_native
# sets it for the duration of a pass; the scenario functions append one entry per lnZ they return (EB/twin in order).
_DRAW_DIAG = None


def _exposure_args():
    """(exptime, nsub) for the active FPP_EXPOSURE mode, as the batch kernels want it.

    Parity: upstream's 2-minute TESS exposure integrated with 20 midpoint
    samples per cadence point. Production: the cadence points evaluated with
    no exposure integration — the production cadence path with the bin model
    is added by a later task.
    """
    if switch_is("FPP_EXPOSURE", "upstream_2min_x20"):
        return PARITY_EXPTIME_D, PARITY_NSUB
    return 0.0, 1


def _sigma_array(sigma, n):
    """Per-bin float64 sigma array of length n, from a scalar or an array.

    The batch kernels take one sigma per cadence bin (upstream's single
    -ln sigma per curve is reproduced as the mean of ln sigma over the bins);
    a constant array reproduces the old scalar behaviour exactly.
    """
    if np.ndim(sigma) == 0:
        return np.full(n, float(sigma), dtype=np.float64)
    arr = np.asarray(sigma, dtype=np.float64).ravel()
    if arr.size != n:
        raise ValueError(f"sigma array has {arr.size} entries, expected {n}")
    return arr.copy()


_ldc_row_cache = None


def _ldc_row_table():
    """Cached (Z grid, Teff grid, logg grid, {(Z, Teff, logg): (u1, u2)}) of the LDC table.

    Built once from ``_ldc_data`` (which ``_load_ldc`` fills); the joint row
    keys are exactly the rows the table contains, so a lookup miss means the
    nearest (Z, Teff, logg) combination genuinely has no row — the same
    condition ``get_ldc`` reports as LDCNoRow.
    """
    global _ldc_row_cache
    if _ldc_row_cache is not None:
        return _ldc_row_cache
    _load_ldc()
    d = _ldc_data
    rows = {}
    for z, t, g, u1, u2 in zip(d['Z'], d['Teff'], d['logg'], d['u1'], d['u2']):
        rows[(float(z), int(t), float(g))] = (float(u1), float(u2))
    _ldc_row_cache = (np.unique(d['Z']), np.unique(d['Teff']),
                      np.unique(d['logg']), rows)
    return _ldc_row_cache


def get_ldc_arrays(Teff, logg, Z=0.0):
    """Vectorised get_ldc: arrays of Teff/logg -> arrays of u1/u2 of the same length.

    Same semantics as ``get_ldc``: the nearest table Z, Teff and logg are
    chosen INDEPENDENTLY per draw, and the joint row must exist, otherwise
    LDCNoRow. Used by the production secondary-eclipse limb darkening
    (FPP_SECONDARY_GEOMETRY = "consistent"), where every EB draw has its own
    secondary star with its own Teff/logg.
    """
    Zs, Ts, gs, rows = _ldc_row_table()
    Teff = np.atleast_1d(np.asarray(Teff, dtype=np.float64))
    logg = np.atleast_1d(np.asarray(logg, dtype=np.float64))
    if Teff.shape != logg.shape:
        raise ValueError(f"Teff ({Teff.shape}) and logg ({logg.shape}) "
                         f"must have the same shape")
    iZ = int(np.argmin(np.abs(Zs - Z)))
    iTs = np.abs(Teff[:, None] - Ts[None, :]).argmin(axis=1)
    iGs = np.abs(logg[:, None] - gs[None, :]).argmin(axis=1)

    u1_arr = np.empty(Teff.size, dtype=np.float64)
    u2_arr = np.empty(Teff.size, dtype=np.float64)
    hit = {}
    for i in range(Teff.size):
        best_Z = float(Zs[iZ])
        best_T = int(Ts[iTs[i]])
        best_g = float(gs[iGs[i]])
        try:
            row = hit[(best_Z, best_T, best_g)]
        except KeyError:
            try:
                row = hit[(best_Z, best_T, best_g)] = rows[(best_Z, best_T, best_g)]
            except KeyError:
                raise LDCNoRow(f"no limb-darkening row for Z={best_Z} "
                               f"Teff={best_T} logg={best_g}") from None
        u1_arr[i], u2_arr[i] = row
    return u1_arr, u2_arr


def _roche_ok(a_cm, ecc, M_s, R_s, rho_comp):
    """Boolean mask: True where the companion clears its Roche limit at periastron.

    A draw is impossible when the companion's periastron distance
    a*(1-ecc) falls inside the Roche limit
    2.44 * R_host * (rho_host / rho_comp)^(1/3) — the host's mean density
    rho_host = M_s*Msun / ((4/3)*pi*(R_s*Rsun)^3), ``rho_comp`` in g/cm^3
    (scalar or array). ``FPP_ORBIT_PHYSICS = "roche"`` adds this to the
    geometric collision check, which only tests the sum of the radii.
    """
    R_host_cm = R_s * Rsun
    rho_host = (M_s * Msun) / ((4.0 / 3.0) * pi * R_host_cm**3)
    periastron = (np.asarray(a_cm, dtype=np.float64)
                  * (1.0 - np.asarray(ecc, dtype=np.float64)))
    limit = 2.44 * R_host_cm * (rho_host / np.asarray(rho_comp,
                                                      dtype=np.float64))**(1.0 / 3.0)
    return periastron >= limit


def _lnZ_TTP(time, flux, sigma, P_orb, M_s, R_s, Teff, N,
             u1, u2, companion_fr=0.0, companion_is_host=False,
             lnprior_companion=0.0, rng=np.random, sigma_veto=None,
             fold=None):
    """Marginal likelihood of Target Transiting Planet scenario.

    ``companion_fr`` may be a scalar (TTP: no companion, 0.0 for every
    draw) or an N-length array of per-draw companion flux ratios (PTP,
    via _lnZ_PTP below) — matches upstream TRICERATOPS, which indexes
    ``fluxratios_comp[mask]`` per Monte-Carlo draw rather than collapsing
    it to a single representative value (marginal_likelihoods.py:lnZ_PTP).

    ``lnprior_companion`` is likewise a scalar 0.0 for TTP/NTP (upstream's
    lnZ_TTP has no such term at all) or an N-length array (PTP, see
    _lnprior_companion) folded into the Monte-Carlo average exactly where
    upstream folds it — inside the per-draw log-weight lnL + lnprior_companion
    averaged by _lnZ_from_logw — not as a separate additive constant on lnZ.

    ``sigma_veto`` is accepted for signature symmetry with _lnZ_TEB and
    ignored: upstream's lnZ_TTP has no secondary-eclipse cut at all.

    ``fold`` (plan 0.3 (j'), (j'')) switches the likelihood to the PRODUCTION
    one: a `build_prod_fold` dict replaces the cadence arrays, so ``time``,
    ``flux`` and ``sigma`` are IGNORED (the caller passes None / np.empty(0))
    and the batch kernel is `_batch_transit_lnL_prod`, whose primary curve is
    the fold-bin ladder instead of the per-point light curve.  The sampling,
    the transit / collision / Roche masks and the D14 cap are unchanged.
    """
    companion_fr_arr = np.broadcast_to(
        np.asarray(companion_fr, dtype=np.float64), (N,))
    lnprior_arr = np.broadcast_to(
        np.asarray(lnprior_companion, dtype=np.float64), (N,))
    sigma_arr = None if fold is not None else _sigma_array(sigma, len(time))
    P_arr = np.full(N, P_orb)
    a_cm = ((G_cgs * M_s * Msun) / (4 * pi**2) * (P_arr * 86400)**2)**(1 / 3)
    a_Rs = a_cm / (R_s * Rsun)

    rps = _sample_rp(N, M_s, rng=rng)
    incs = _sample_inc(N, rng=rng)
    eccs = _sample_ecc(N, planet=True, P_orb=P_orb, rng=rng)
    argps = _sample_w(N, rng=rng)

    ks = rps * Rearth / (R_s * Rsun)
    incs_rad = incs * pi / 180.0
    ws_rad = (90.0 - argps) * pi / 180.0

    # Transit probability filter
    e_corr = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    Ptra = (rps * Rearth + R_s * Rsun) / a_cm * e_corr
    inc_min = np.full(N, pi / 2)
    can_transit = Ptra <= 1.0
    inc_min[can_transit] = np.arccos(Ptra[can_transit])

    # Filter: transiting and no collision
    coll = (rps * Rearth + R_s * Rsun) > a_cm * (1 - eccs)
    mask = (incs_rad >= inc_min) & (~coll)
    if switch_is("FPP_ORBIT_PHYSICS", "roche"):
        # Planet companion: the LEAST restrictive density (rocky 5.51 g/cm3),
        # so this rejects only draws that are impossible for ANY planet.
        mask &= _roche_ok(a_cm, eccs, M_s, R_s, 5.51)

    if not np.any(mask):
        return -np.inf

    comp_fr = companion_fr_arr[mask]
    exptime, nsub = _exposure_args()

    lnL = np.full(N, -np.inf)
    if fold is None:
        lnL[mask] = _dispatch_lnL(
            _batch_transit_lnL,
            (time, flux, sigma_arr,
             ks[mask], P_arr[mask], incs_rad[mask], a_Rs[mask],
             eccs[mask], ws_rad[mask], u1, u2,
             comp_fr, companion_is_host, exptime, nsub))
    else:
        lnL_m, status_m, _n_aff_m = _dispatch_lnL(
            _batch_transit_lnL_prod,
            (fold["nb"], fold["f"], fold["sigma"], fold["base_chi2"],
             fold["lnnorm"], fold["n"], fold["tmean"], fold["var_t"],
             fold["tmin"], fold["tmax"], fold["node_tab"], fold["pts_off"],
             fold["pts_t"], fold["W"], fold["delta"],
             ks[mask], P_arr[mask], incs_rad[mask], a_Rs[mask],
             eccs[mask], ws_rad[mask], u1, u2,
             comp_fr, companion_is_host))
        lnL[mask] = lnL_m
        _note_exposure_stats(lnL_m, status_m)

    # lnL is -inf for the draws the masks rejected: they still count in the
    # mean, exactly as upstream's np.mean over the full N (plan D9).
    return _lnZ_from_logw(lnL + lnprior_arr)



def _lnZ_TEB(time, flux, sigma, P_orb, M_s, R_s, Teff, N, u1, u2,
             companion_fr=0.0, companion_is_host=False,
             lnprior_companion=0.0, rng=np.random, sigma_veto=None,
             fold=None):
    """Marginal likelihood of Target Eclipsing Binary (and twin EB) scenarios.

    Returns (lnZ_EB, lnZ_EB_twin).

    ``companion_fr`` may be a scalar (TEB: no companion, 0.0 for every
    draw) or an N-length array of per-draw companion flux ratios (PEB,
    via _lnZ_PEB below) — see the matching note on _lnZ_TTP.

    ``lnprior_companion`` (see _lnZ_TTP's matching note) is the SAME
    N-length array for both the EB and twin-EB branches below — upstream
    lnZ_PEB computes ``lnprior_companion`` once and folds it into both
    ``Z`` and ``Z_twin``'s Monte-Carlo average (it is indexed by the
    companion draw, independent of which of the two EB masks the target's
    own eclipsing pair falls into).

    ``sigma_veto`` is the scalar depth a detectable secondary eclipse must
    reach for the draw to be rejected (upstream's 1.5*sigma cut). It defaults
    to the scalar ``sigma``; when ``sigma`` is a per-bin array it must be
    given explicitly, and so must it be when ``fold`` is given (there is no
    scalar sigma to fall back on).

    ``fold`` (plan 0.3 (j'), (j'')) switches the likelihood to the PRODUCTION
    one: a `build_prod_fold` dict replaces the cadence arrays, so ``time``,
    ``flux`` and ``sigma`` are IGNORED (the caller passes None / np.empty(0))
    and both batch kernels are the ``*_prod`` twins, whose primary curve is
    the fold-bin ladder instead of the per-point light curve.  The sampling,
    the EB / twin masks (incl. the Roche mask), the D14 radius / Teff cap and
    the secondary-eclipse limb-darkening logic are unchanged.
    """
    companion_fr_arr = np.broadcast_to(
        np.asarray(companion_fr, dtype=np.float64), (N,))
    lnprior_arr = np.broadcast_to(
        np.asarray(lnprior_companion, dtype=np.float64), (N,))
    sigma_arr = None if fold is not None else _sigma_array(sigma, len(time))
    if sigma_veto is None and fold is not None:
        raise ValueError("sigma_veto is required when a production fold is given")
    if sigma_veto is None and np.ndim(sigma) == 0:
        sigma_veto = float(sigma)
    elif sigma_veto is None:
        raise ValueError("sigma_veto is required when sigma is a per-bin array")
    sigma_veto = float(sigma_veto)
    P_arr = np.full(N, P_orb)

    incs = _sample_inc(N, rng=rng)
    qs = _sample_q(N, M_s, rng=rng)
    eccs = _sample_ecc(N, planet=False, P_orb=P_orb, rng=rng)
    argps = _sample_w(N, rng=rng)

    masses_eb = qs * M_s
    radii_eb, teff_eb = stellar_relations(masses_eb)
    # D14: upstream's funcs.py caps the companion radius / Teff at the host's
    radii_eb = np.minimum(radii_eb, R_s)
    teff_eb = np.minimum(teff_eb, Teff)
    fluxratios_eb = flux_relation(masses_eb) / (
        flux_relation(masses_eb) + flux_relation(np.array([M_s])))

    incs_rad = incs * pi / 180.0
    ws_rad = (90.0 - argps) * pi / 180.0

    # Semi-major axes
    a_cm = ((G_cgs * (M_s + masses_eb) * Msun) / (4 * pi**2)
            * (P_arr * 86400)**2)**(1 / 3)
    a_twin_cm = ((G_cgs * (M_s + masses_eb) * Msun) / (4 * pi**2)
                 * (2 * P_arr * 86400)**2)**(1 / 3)

    a_Rs = a_cm / (R_s * Rsun)
    a_twin_Rs = a_twin_cm / (R_s * Rsun)

    ks = radii_eb / R_s

    # Transit probability
    e_corr = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    Ptra = (radii_eb * Rsun + R_s * Rsun) / a_cm * e_corr
    Ptra_twin = (radii_eb * Rsun + R_s * Rsun) / a_twin_cm * e_corr

    inc_min = np.full(N, pi / 2)
    can = Ptra <= 1.0
    inc_min[can] = np.arccos(Ptra[can])
    inc_min_twin = np.full(N, pi / 2)
    can_tw = Ptra_twin <= 1.0
    inc_min_twin[can_tw] = np.arccos(Ptra_twin[can_tw])

    coll = (radii_eb * Rsun + R_s * Rsun) > a_cm * (1 - eccs)
    coll_twin = (2 * R_s * Rsun) > a_twin_cm * (1 - eccs)

    # q < 0.95 for EB
    mask_eb = (incs_rad >= inc_min) & (~coll) & (qs < 0.95)
    # q >= 0.95 for twin EB
    mask_twin = (incs_rad >= inc_min_twin) & (~coll_twin) & (qs >= 0.95)

    if switch_is("FPP_ORBIT_PHYSICS", "roche"):
        # Roche limit of the EB companion at periastron, relative to the host:
        # the eclipsing pair is the companion here, so its own mean density
        # (from the sampled mass and the capped radius) sets the limit.
        rho_comp = ((masses_eb * Msun)
                    / ((4.0 / 3.0) * pi * (radii_eb * Rsun)**3))
        mask_eb &= _roche_ok(a_cm, eccs, M_s, R_s, rho_comp)
        mask_twin &= _roche_ok(a_twin_cm, eccs, M_s, R_s, rho_comp)

    comp_arr = companion_fr_arr
    exptime, nsub = _exposure_args()
    sec_consistent = switch_is("FPP_SECONDARY_GEOMETRY", "consistent")
    k_nudge_upstream = switch_is("FPP_K1_NUDGE", "upstream")

    # Secondary-eclipse limb darkening: production uses the secondary star's
    # own coefficients (computed only on the surviving draws); parity reuses
    # the host's u1/u2 for the secondary — upstream's quirk.
    u1_sec = np.full(N, u1, dtype=np.float64)
    u2_sec = np.full(N, u2, dtype=np.float64)
    if sec_consistent:
        sel = mask_eb | mask_twin
        if np.any(sel):
            logg_eb = np.log10(G_cgs * masses_eb[sel] * Msun
                               / (radii_eb[sel] * Rsun)**2)
            u1_sec[sel], u2_sec[sel] = get_ldc_arrays(teff_eb[sel], logg_eb)

    lnL_eb = np.full(N, -np.inf)
    if np.any(mask_eb):
        if fold is None:
            lnL_eb[mask_eb] = _dispatch_lnL(
                _batch_eb_transit_lnL,
                (time, flux, sigma_arr, sigma_veto,
                 ks[mask_eb], fluxratios_eb[mask_eb], P_arr[mask_eb],
                 incs_rad[mask_eb], a_Rs[mask_eb], eccs[mask_eb], ws_rad[mask_eb],
                 u1, u2, comp_arr[mask_eb], companion_is_host, exptime, nsub,
                 True, sec_consistent, k_nudge_upstream,
                 u1_sec[mask_eb], u2_sec[mask_eb]))
        else:
            lnL_m, status_m, _n_aff_m = _dispatch_lnL(
                _batch_eb_transit_lnL_prod,
                (fold["nb"], fold["f"], fold["sigma"], fold["base_chi2"],
                 fold["lnnorm"], fold["n"], fold["tmean"], fold["var_t"],
                 fold["tmin"], fold["tmax"], fold["node_tab"], fold["pts_off"],
                 fold["pts_t"], fold["W"], fold["delta"], sigma_veto,
                 ks[mask_eb], fluxratios_eb[mask_eb], P_arr[mask_eb],
                 incs_rad[mask_eb], a_Rs[mask_eb], eccs[mask_eb], ws_rad[mask_eb],
                 u1, u2, comp_arr[mask_eb], companion_is_host,
                 True, sec_consistent, k_nudge_upstream,
                 u1_sec[mask_eb], u2_sec[mask_eb]))
            lnL_eb[mask_eb] = lnL_m
            _note_exposure_stats(lnL_m, status_m)

    lnL_twin = np.full(N, -np.inf)
    if np.any(mask_twin):
        # D3: upstream's lnL_EB_twin_p has no secondary-eclipse cut at all
        if fold is None:
            lnL_twin[mask_twin] = _dispatch_lnL(
                _batch_eb_transit_lnL,
                (time, flux, sigma_arr, sigma_veto,
                 ks[mask_twin], fluxratios_eb[mask_twin], 2 * P_arr[mask_twin],
                 incs_rad[mask_twin], a_twin_Rs[mask_twin],
                 eccs[mask_twin], ws_rad[mask_twin],
                 u1, u2, comp_arr[mask_twin], companion_is_host, exptime, nsub,
                 False, sec_consistent, k_nudge_upstream,
                 u1_sec[mask_twin], u2_sec[mask_twin]))
        else:
            # The same fold: the twin's transit sits at t = 0 of the fold's
            # time axis with its own period 2*P_orb.
            lnL_m, status_m, _n_aff_m = _dispatch_lnL(
                _batch_eb_transit_lnL_prod,
                (fold["nb"], fold["f"], fold["sigma"], fold["base_chi2"],
                 fold["lnnorm"], fold["n"], fold["tmean"], fold["var_t"],
                 fold["tmin"], fold["tmax"], fold["node_tab"], fold["pts_off"],
                 fold["pts_t"], fold["W"], fold["delta"], sigma_veto,
                 ks[mask_twin], fluxratios_eb[mask_twin], 2 * P_arr[mask_twin],
                 incs_rad[mask_twin], a_twin_Rs[mask_twin],
                 eccs[mask_twin], ws_rad[mask_twin],
                 u1, u2, comp_arr[mask_twin], companion_is_host,
                 False, sec_consistent, k_nudge_upstream,
                 u1_sec[mask_twin], u2_sec[mask_twin]))
            lnL_twin[mask_twin] = lnL_m
            _note_exposure_stats(lnL_m, status_m)

    lnZ_eb = _lnZ_from_logw(lnL_eb + lnprior_arr)
    lnZ_twin = _lnZ_from_logw(lnL_twin + lnprior_arr)
    return lnZ_eb, lnZ_twin



def _lnZ_PTP(time, flux, sigma, P_orb, M_s, R_s, Teff, N, u1, u2, plx, rng=np.random,
             fold=None, sigma_veto=None):
    """Marginal likelihood of Planet around physical companion (PTP).

    Mirrors triceratops/marginal_likelihoods.py:lnZ_PTP exactly: the
    physically-sampled companion (masses_comp, via the binary mass-ratio
    prior) only sets the per-draw flux-dilution ratio fed to the TTP
    likelihood as ``companion_fr`` — one value per Monte-Carlo draw, not
    a single scalar collapsed across draws. The transit geometry itself
    (a_cm, ks, Ptra, collision check, `_sample_rp`) and the limb-darkening
    coefficients (u1, u2, passed straight through) stay the TARGET star's,
    exactly as upstream does; `companion_is_host=False` matches upstream's
    call (the target, not the companion, is modeled as the transit host,
    with the companion adding only third light). Upstream additionally
    folds a ``lnprior_companion`` term (bound-companion probability from
    parallax + Δmag + the Moe & Di Stefano mass-ratio prior,
    `lnprior_bound_TP`) into Z — see `_lnprior_companion`.
    """
    qs_comp = _sample_q_companion(N, M_s, rng=rng)
    masses_comp = qs_comp * M_s
    fluxratios_comp = flux_relation(masses_comp) / (
        flux_relation(masses_comp) + flux_relation(np.array([M_s])))
    lnprior_companion = _lnprior_companion(M_s, plx, fluxratios_comp, "TP")

    return _lnZ_TTP(time, flux, sigma, P_orb, M_s, R_s, Teff, N,
                     u1, u2, companion_fr=fluxratios_comp,
                     companion_is_host=False,
                     lnprior_companion=lnprior_companion, rng=rng,
                     sigma_veto=sigma_veto, fold=fold)


def _lnZ_PEB(time, flux, sigma, P_orb, M_s, R_s, Teff, N, u1, u2, plx, rng=np.random,
             fold=None, sigma_veto=None):
    """Marginal likelihood of EB around physical companion (PEB).

    Same convention as _lnZ_PTP above (see its docstring) — mirrors
    triceratops/marginal_likelihoods.py:lnZ_PEB. Note this also fixes a
    second, independent bug: _lnZ_TEB used to hardcode its dilution array
    to zero (``comp_arr = np.zeros(N)``), silently discarding whatever
    ``companion_fr`` was passed in — PEB's third-light dilution was a
    no-op even before this change reached the M_s/R_s question at all.

    ``fold`` / ``sigma_veto`` are passed straight to _lnZ_TEB (see its note):
    with a production fold the caller gives the veto depth explicitly.
    """
    qs_comp = _sample_q_companion(N, M_s, rng=rng)
    masses_comp = qs_comp * M_s
    fluxratios_comp = flux_relation(masses_comp) / (
        flux_relation(masses_comp) + flux_relation(np.array([M_s])))
    lnprior_companion = _lnprior_companion(M_s, plx, fluxratios_comp, "EB")

    return _lnZ_TEB(time, flux, sigma, P_orb, M_s, R_s, Teff, N,
                     u1, u2, companion_fr=fluxratios_comp,
                     companion_is_host=False,
                     lnprior_companion=lnprior_companion, rng=rng,
                     sigma_veto=sigma_veto, fold=fold)


# ============================================================
# 6b. D6 S family: STP, SEB, SEBx2P (plan step 4)
# ============================================================
# The transit / eclipse is hosted by the unresolved BOUND companion of the target (mass q_comp * M_s, q_comp from
# Moe & Di Stefano 2017 with the twin excess), the target dilutes it (`companion_is_host = True`). Ported from
# triceratops 1.1.0 marginal_likelihoods.py:892 (lnZ_STP) and :1116 (lnZ_SEB). What the PTP / PEB machinery above
# does not have: the host properties are PER DRAW (radius / Teff from the capped stellar relations, per-draw limb
# darkening from the rounded companion lookup, planet radii drawn with the companion's mass, array semi-major axes).
# The batch kernels take scalar u1 / u2, so the draws are processed in groups of equal (u1, u2); the per-draw
# arithmetic is the kernels' own and does not depend on the grouping. The ports (Swift / Rust) take per-draw arrays.

def stellar_relations_capped(masses, max_R, max_T):
    """Upstream `funcs.stellar_relations(Masses, max_Radii, max_Teffs)`: spline radius / Teff (Torres above 0.63 Msun, cool-dwarf
    spline below, both EXTRAPOLATED outside their nodes), then the host caps (R > max_R -> max_R, T > max_T -> max_T) and only
    THEN the floors (R >= 0.1, T >= 2800 K): the order of upstream, which matters when a cap is below a floor."""
    masses = np.atleast_1d(np.asarray(masses, dtype=np.float64))
    max_R = np.broadcast_to(np.asarray(max_R, dtype=np.float64), masses.shape)
    max_T = np.broadcast_to(np.asarray(max_T, dtype=np.float64), masses.shape)
    R = np.zeros(masses.shape)
    T = np.zeros(masses.shape)
    hot = masses > 0.63
    cool = ~hot
    if np.any(hot):
        R[hot] = _spl_R_torres(masses[hot])
        T[hot] = _spl_T_torres(masses[hot])
    if np.any(cool):
        R[cool] = _spl_R_cdwrf(masses[cool])
        T[cool] = _spl_T_cdwrf(masses[cool])
    over = R > max_R
    R[over] = max_R[over]
    over = T > max_T
    T[over] = max_T[over]
    R[R < 0.1] = 0.1
    T[T < 2800.0] = 2800.0
    return R, T


def ldc_companion_arrays(Teff, logg, teff_max, Z=0.0):
    """Per-draw limb darkening of the S family's host (upstream lnZ_STP / lnZ_SEB): Teff rounded to 250 K and clipped to
    [3500, teff_max], logg rounded to 0.5 and clipped to [3.5, 5.0], EXACT row of the table at the nearest Z.  `teff_max` is
    10000 K in lnZ_STP and 13000 K in lnZ_SEB (an inconsistency of upstream that is reproduced).  A missing row raises
    LDCNoRow (upstream: `.item()` on an empty mask), never a default."""
    Zs, _Ts, _gs, rows = _ldc_row_table()
    Teff = np.atleast_1d(np.asarray(Teff, dtype=np.float64))
    logg = np.atleast_1d(np.asarray(logg, dtype=np.float64))
    rl = np.round(logg / 0.5) * 0.5
    rl[rl < 3.5] = 3.5
    rl[rl > 5.0] = 5.0
    rt = np.round(Teff / 250.0) * 250.0
    rt[rt < 3500.0] = 3500.0
    rt[rt > float(teff_max)] = float(teff_max)
    zbest = float(Zs[int(np.argmin(np.abs(Zs - Z)))])
    u1 = np.empty(Teff.size)
    u2 = np.empty(Teff.size)
    seen = {}
    for i in range(Teff.size):
        key = (zbest, int(rt[i]), float(rl[i]))
        if key not in seen:
            try:
                seen[key] = rows[key]
            except KeyError:
                raise LDCNoRow(f"no limb-darkening row for Z={key[0]} Teff={key[1]} logg={key[2]} (S family)") from None
        u1[i], u2[i] = seen[key]
    return u1, u2


def _bound_companion_block(N, M_s, R_s, Teff, rng, teff_max):
    """Draw the bound companions and derive their properties (shared by STP and SEB)."""
    qs_comp = _sample_q_companion(N, M_s, rng=rng)
    mc = qs_comp * M_s
    Rc, Tc = stellar_relations_capped(mc, np.full(N, R_s), np.full(N, Teff))
    loggc = np.log10(G_cgs * (mc * Msun) / (Rc * Rsun)**2)
    frc = flux_relation(mc) / (flux_relation(mc) + flux_relation(np.array([M_s])))
    u1c, u2c = ldc_companion_arrays(Tc, loggc, teff_max)
    return dict(qs_comp=qs_comp, mc=mc, Rc=Rc, Tc=Tc, loggc=loggc, frc=frc, u1c=u1c, u2c=u2c)


def _call_by_ldc(call, u1, u2, n, fold):
    """Run a batch kernel over draws grouped by equal (u1, u2).

    ``call(sel, u1_scalar, u2_scalar)`` evaluates the draws ``sel`` (indices into the masked arrays) and returns lnL (cadence
    path) or (lnL, status, n_aff) (production fold). Returns the lnL array of length n (and the status array with the fold).
    The per-draw results are independent of the grouping."""
    lnL = np.full(n, -np.inf)
    status = np.zeros(n, dtype=np.int64)
    if n == 0:
        return (lnL, status) if fold is not None else lnL
    pairs, inv = np.unique(np.stack([u1, u2], axis=1), axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    for g in range(len(pairs)):
        sel = np.nonzero(inv == g)[0]
        r = call(sel, float(pairs[g, 0]), float(pairs[g, 1]))
        if fold is None:
            lnL[sel] = r
        else:
            lnL[sel] = r[0]
            status[sel] = r[1]
    return (lnL, status) if fold is not None else lnL


def _lnZ_STP(time, flux, sigma, P_orb, M_s, R_s, Teff, N, plx, rng=np.random, fold=None, sigma_veto=None):
    """Marginal likelihood of the STP scenario: a planet around the unresolved bound companion (plan D6, step 4).

    Port of triceratops lnZ_STP (marginal_likelihoods.py:892). Sampler order as upstream: q_comp, planet radius (with the
    companion mass), inclination, eccentricity, argument of periastron. Prior: `lnprior_bound_TP` of the companion
    (2.2 arcsec, P > 2500 d), -inf when the companion is brighter than the target. Masks: transit, collision and (production)
    the Roche limit with the companion as host and a rocky planet (5.51 g/cm3, plan (h')). `sigma_veto` is accepted for
    symmetry and unused (a planet has no secondary-eclipse cut)."""
    sigma_arr = None if fold is not None else _sigma_array(sigma, len(time))
    blk = _bound_companion_block(N, M_s, R_s, Teff, rng, 10000.0)
    mc, Rc, frc, u1c, u2c = blk["mc"], blk["Rc"], blk["frc"], blk["u1c"], blk["u2c"]
    lnprior = _lnprior_companion(M_s, plx, frc, "TP")

    rps = _sample_rp(N, mc, rng=rng)
    incs = _sample_inc(N, rng=rng)
    eccs = _sample_ecc(N, planet=True, P_orb=P_orb, rng=rng)
    argps = _sample_w(N, rng=rng)

    P_arr = np.full(N, P_orb)
    a_cm = ((G_cgs * mc * Msun) / (4 * pi**2) * (P_arr * 86400)**2)**(1 / 3)
    a_Rs = a_cm / (Rc * Rsun)
    ks = rps * Rearth / (Rc * Rsun)
    incs_rad = incs * pi / 180.0
    ws_rad = (90.0 - argps) * pi / 180.0

    e_corr = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    Ptra = (rps * Rearth + Rc * Rsun) / a_cm * e_corr
    inc_min = np.full(N, pi / 2)
    can = Ptra <= 1.0
    inc_min[can] = np.arccos(Ptra[can])
    coll = (rps * Rearth + Rc * Rsun) > a_cm * (1 - eccs)
    mask = (incs_rad >= inc_min) & (~coll)
    if switch_is("FPP_ORBIT_PHYSICS", "roche"):
        mask &= _roche_ok(a_cm, eccs, mc, Rc, 5.51)
    if not np.any(mask):
        return -np.inf

    im = np.nonzero(mask)[0]
    exptime, nsub = _exposure_args()
    lnL = np.full(N, -np.inf)

    if fold is None:
        def call(sel, u1, u2):
            j = im[sel]
            return _dispatch_lnL(_batch_transit_lnL, (
                time, flux, sigma_arr, ks[j], P_arr[j], incs_rad[j], a_Rs[j], eccs[j], ws_rad[j], u1, u2,
                frc[j], True, exptime, nsub))
        lnL[im] = _call_by_ldc(call, u1c[im], u2c[im], len(im), None)
    else:
        def call(sel, u1, u2):
            j = im[sel]
            return _dispatch_lnL(_batch_transit_lnL_prod, (
                fold["nb"], fold["f"], fold["sigma"], fold["base_chi2"], fold["lnnorm"], fold["n"], fold["tmean"],
                fold["var_t"], fold["tmin"], fold["tmax"], fold["node_tab"], fold["pts_off"], fold["pts_t"],
                fold["W"], fold["delta"], ks[j], P_arr[j], incs_rad[j], a_Rs[j], eccs[j], ws_rad[j], u1, u2,
                frc[j], True))
        l_m, st_m = _call_by_ldc(call, u1c[im], u2c[im], len(im), fold)
        lnL[im] = l_m
        _note_exposure_stats(l_m, st_m)
    return _lnZ_from_logw(lnL + lnprior)


def _lnZ_SEB(time, flux, sigma, P_orb, M_s, R_s, Teff, N, plx, rng=np.random, fold=None, sigma_veto=None):
    """Marginal likelihoods of SEB and SEBx2P: an eclipsing binary around the unresolved bound companion (plan D6, step 4).

    Port of triceratops lnZ_SEB (marginal_likelihoods.py:1116). Sampler order as upstream: inclination, q, eccentricity,
    argument of periastron, q_comp. The EB's mass is q * companion mass, its radius / Teff are capped at the companion's;
    its flux ratio is relative to the TARGET. Prior: `lnprior_bound_EB` with the summed companion + EB magnitude difference,
    -inf if that sum is brighter than the target. q < 0.95 -> EB at P (with the secondary-eclipse cut), q >= 0.95 -> twin at
    2P (no cut, D3). Returns (lnZ_SEB, lnZ_SEBx2P)."""
    sigma_arr = None if fold is not None else _sigma_array(sigma, len(time))
    if sigma_veto is None and fold is not None:
        raise ValueError("sigma_veto is required when a production fold is given")
    if sigma_veto is None and np.ndim(sigma) == 0:
        sigma_veto = float(sigma)
    elif sigma_veto is None:
        raise ValueError("sigma_veto is required when sigma is a per-bin array")
    sigma_veto = float(sigma_veto)

    incs = _sample_inc(N, rng=rng)
    qs = _sample_q(N, M_s, rng=rng)
    eccs = _sample_ecc(N, planet=False, P_orb=P_orb, rng=rng)
    argps = _sample_w(N, rng=rng)
    blk = _bound_companion_block(N, M_s, R_s, Teff, rng, 13000.0)
    mc, Rc, Tc, frc, u1c, u2c = blk["mc"], blk["Rc"], blk["Tc"], blk["frc"], blk["u1c"], blk["u2c"]

    masses = qs * mc
    Reb, Teb = stellar_relations_capped(masses, Rc, Tc)
    freb = flux_relation(masses) / (flux_relation(masses) + flux_relation(np.array([M_s])))
    dm = 2.5 * np.log10((frc / (1 - frc)) + (freb / (1 - freb)))
    lnprior = _lnprior_bound_EB(M_s, plx, np.abs(dm))
    lnprior[lnprior > 0.0] = 0.0
    lnprior[dm > 0.0] = -np.inf

    P_arr = np.full(N, P_orb)
    e_corr = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    a_cm = ((G_cgs * (mc + masses) * Msun) / (4 * pi**2) * (P_arr * 86400)**2)**(1 / 3)
    a_tw = ((G_cgs * (mc + masses) * Msun) / (4 * pi**2) * (2 * P_arr * 86400)**2)**(1 / 3)
    Ptra = (Reb * Rsun + Rc * Rsun) / a_cm * e_corr
    Ptra_tw = (Reb * Rsun + Rc * Rsun) / a_tw * e_corr
    inc_min = np.full(N, pi / 2)
    can = Ptra <= 1.0
    inc_min[can] = np.arccos(Ptra[can])
    inc_min_tw = np.full(N, pi / 2)
    can = Ptra_tw <= 1.0
    inc_min_tw[can] = np.arccos(Ptra_tw[can])
    coll = (Reb * Rsun + Rc * Rsun) > a_cm * (1 - eccs)
    coll_tw = (2 * Rc * Rsun) > a_tw * (1 - eccs)
    incs_rad = incs * pi / 180.0
    ws_rad = (90.0 - argps) * pi / 180.0
    mask_eb = (incs_rad >= inc_min) & (~coll) & (qs < 0.95)
    mask_tw = (incs_rad >= inc_min_tw) & (~coll_tw) & (qs >= 0.95)
    if switch_is("FPP_ORBIT_PHYSICS", "roche"):
        rho_eb = (masses * Msun) / ((4.0 / 3.0) * pi * (Reb * Rsun)**3)
        mask_eb &= _roche_ok(a_cm, eccs, mc, Rc, rho_eb)
        mask_tw &= _roche_ok(a_tw, eccs, mc, Rc, rho_eb)

    ks = Reb / Rc
    a_Rs = a_cm / (Rc * Rsun)
    a_tw_Rs = a_tw / (Rc * Rsun)
    exptime, nsub = _exposure_args()
    sec_consistent = switch_is("FPP_SECONDARY_GEOMETRY", "consistent")
    k_nudge_upstream = switch_is("FPP_K1_NUDGE", "upstream")

    # secondary-eclipse limb darkening: production = the EB star's own coefficients (nearest table row, as TEB);
    # parity = the host's (here the companion's) coefficients, upstream's quirk
    u1_sec, u2_sec = u1c.copy(), u2c.copy()
    if sec_consistent:
        sel = mask_eb | mask_tw
        if np.any(sel):
            logg_eb = np.log10(G_cgs * masses[sel] * Msun / (Reb[sel] * Rsun)**2)
            u1_sec[sel], u2_sec[sel] = get_ldc_arrays(Teb[sel], logg_eb)

    def run(mask, Pv, aRs, with_sec):
        lnL = np.full(N, -np.inf)
        if not np.any(mask):
            return lnL
        im = np.nonzero(mask)[0]
        if fold is None:
            def call(sel, u1, u2):
                j = im[sel]
                return _dispatch_lnL(_batch_eb_transit_lnL, (
                    time, flux, sigma_arr, sigma_veto, ks[j], freb[j], Pv[j], incs_rad[j], aRs[j], eccs[j],
                    ws_rad[j], u1, u2, frc[j], True, exptime, nsub, with_sec, sec_consistent, k_nudge_upstream,
                    u1_sec[j], u2_sec[j]))
            lnL[im] = _call_by_ldc(call, u1c[im], u2c[im], len(im), None)
        else:
            def call(sel, u1, u2):
                j = im[sel]
                return _dispatch_lnL(_batch_eb_transit_lnL_prod, (
                    fold["nb"], fold["f"], fold["sigma"], fold["base_chi2"], fold["lnnorm"], fold["n"],
                    fold["tmean"], fold["var_t"], fold["tmin"], fold["tmax"], fold["node_tab"], fold["pts_off"],
                    fold["pts_t"], fold["W"], fold["delta"], sigma_veto, ks[j], freb[j], Pv[j], incs_rad[j], aRs[j],
                    eccs[j], ws_rad[j], u1, u2, frc[j], True, with_sec, sec_consistent, k_nudge_upstream,
                    u1_sec[j], u2_sec[j]))
            l_m, st_m = _call_by_ldc(call, u1c[im], u2c[im], len(im), fold)
            lnL[im] = l_m
            _note_exposure_stats(l_m, st_m)
        return lnL

    lnL_eb = run(mask_eb, P_arr, a_Rs, True)
    lnL_tw = run(mask_tw, 2 * P_arr, a_tw_Rs, False)       # D3: the twin has no secondary-eclipse cut
    return _lnZ_from_logw(lnL_eb + lnprior), _lnZ_from_logw(lnL_tw + lnprior)


# ============================================================
# 6c. D6 D/B families: task-supplied background population (plan step 5)
# ============================================================

class BackgroundUnavailable(FPPUnavailable):
    """Missing/unverifiable field population: refuse and requeue, never omit D/B."""
    def __init__(self, message, skip_reason="population_invalid"):
        super().__init__(message)
        self.skip_reason = skip_reason


# Galaxia generators whose served documents the worker accepts: the C++ install and the galaxia-rs port
# (C++ parity approved 2026-10-20, docs/reviews/2026-10-20-ext-galaxia-rs-parity-close-out-review.md).
GALAXIA_SOURCES = ("galaxia-0.7.2", "galaxia-rs-0.2.2")


def background_population(population, target_tmag, expected_sha256=None, expected_version=None,
                          expected_tic_id=None):
    """Validate the immutable server document and prepare its target-fainter rows.

    No Galaxia/TRILEGAL/network dependency in the worker. cc1 magnitudes have
    ALREADY been mapped server-side; never apply the correction twice. The
    count in the geometric prior is the filtered population, as trilegal_results.
    """
    if population is None:
        raise BackgroundUnavailable("background population is missing", "population_missing")
    try:
        if not isinstance(population, dict):
            raise ValueError("population must be a document")
        header = population["header"]
        if expected_tic_id is not None and str(header.get("tic_id")) != str(expected_tic_id):
            raise ValueError("population belongs to a different TIC")
        if header["source"] not in GALAXIA_SOURCES + ("gaia",):
            raise ValueError("unsupported population source")
        if header["columns"] != ["Tmag", "Mact", "logg", "logTe", "[M/H]"]:
            raise ValueError("population columns differ from contract")
        version = header["population_version"]
        if not isinstance(version, str) or not version:
            raise ValueError("missing population version")
        mapping = header["mapping_version"]
        if not isinstance(mapping, str) or not mapping:
            raise ValueError("missing population mapping version")
        if header["source"] in GALAXIA_SOURCES and mapping not in ("tmag-stassun2018-jks-cc1", "tmag-stassun2018-jks-v1"):
            raise ValueError("unknown Tmag mapping")
        if float(header["field_area_deg2"]) != 0.1 or float(header["tmag_limit"]) != 21.0:
            raise ValueError("unsupported field area or magnitude limit")
        if expected_version is not None and version != expected_version:
            raise ValueError("population version mismatch")
        payload = json.dumps(population, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("population sha256 mismatch")
        a = np.asarray(population["rows"], dtype=np.float64)
        count = header["n_comp"]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("empty/invalid population count")
        if a.shape != (count, 5) or not np.isfinite(a).all():
            raise ValueError("invalid population rows")
        if not np.isfinite(target_tmag) or np.any(a[:, 1] <= 0) or np.any(a[:, 0] > 21.0):
            raise ValueError("invalid magnitude or stellar mass")
        # Preserve canonical row order, including duplicate rows (they carry density).
        if any(tuple(a[i]) > tuple(a[i+1]) for i in range(len(a)-1)):
            raise ValueError("population is not canonically sorted")
        a = a[a[:, 0] >= float(target_tmag)]
        if len(a) == 0:
            raise ValueError("no target-fainter population rows")
        teff = 10.0 ** a[:, 3]
        radius = np.sqrt(G_cgs * a[:, 1] * Msun / 10.0 ** a[:, 2]) / Rsun
        if (not np.isfinite(teff).all() or not np.isfinite(radius).all()
                or np.any(teff <= 0) or np.any(radius <= 0)):
            raise ValueError("invalid population stellar properties")
        relative = 10.0 ** ((float(target_tmag) - a[:, 0]) / 2.5)
        fr = relative / (1.0 + relative)
        info = dict(population_version=version, population_sha256=digest,
                    population_source="gaia" if header["source"] == "gaia" else "galaxia",
                    population_generator=header["source"], population_mapping_version=mapping,
                    n_comp=count, n_comp_fainter=len(a))
        return dict(Tmag=a[:, 0], M=a[:, 1], logg=a[:, 2], Teff=teff, Z=a[:, 4],
                    R=radius, fr=fr, n_comp=len(a), info=info)
    except BackgroundUnavailable:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise BackgroundUnavailable(f"invalid background population: {exc}") from exc


def population_provenance_missing(population_sha256, population_version):
    """True when the task's immutable population provenance is absent or empty.

    Production refuses such a task (``population_invalid``, the reason the Swift
    port throws) instead of skipping the digest/version checks; decision 2026-10-25.
    """
    return not (isinstance(population_sha256, str) and population_sha256
                and isinstance(population_version, str) and population_version)


def _sample_background_indices(N, n_comp, rng=np.random, legacy=False):
    high = n_comp - 1 if legacy and switch_is("FPP_BACKGROUND_INDEX", "upstream") else n_comp
    if high <= 0:
        raise BackgroundUnavailable("population too small for upstream D-family indexing")
    return rng.randint(0, high, N)


def _background_prior(population, idxs, flux_eb=None):
    count = population["n_comp"]
    area = (count / 0.1) * (1 / 3600)**2 * 2.2**2
    if switch_is("FPP_BACKGROUND_PRIOR", "circular"):
        area *= pi
    prior = np.full(len(idxs), min(0.0, float(np.log(area))))
    fr = population["fr"][idxs]
    ratio = fr / (1 - fr)
    if flux_eb is not None:
        ratio = ratio + flux_eb / (1 - flux_eb)
    with np.errstate(invalid="ignore", divide="ignore"):
        dm = 2.5 * np.log10(ratio)
    prior[(dm > 0.0) | ~np.isfinite(dm)] = -np.inf
    return prior


def ldc_background_arrays(Teff, logg, Z):
    """Upstream B-family lookup: nearest Teff and logg, THEN nearest Z at that row.

    Use the original CSV ordering for argmin ties, not sorted unique axes.
    Unlike S, no 250-K/0.5 rounding or temperature ceiling before lookup.
    """
    _load_ldc()
    d = _ldc_data
    seen = {}
    u1, u2 = np.empty(len(Teff)), np.empty(len(Teff))
    for i, (t, g, z) in enumerate(zip(Teff, logg, Z)):
        tt = d['Teff'][np.argmin(np.abs(d['Teff'] - t))]
        gg = d['logg'][np.argmin(np.abs(d['logg'] - g))]
        select = (d['Teff'] == tt) & (d['logg'] == gg)
        zs = d['Z'][select]
        if not len(zs):
            raise BackgroundUnavailable(f"background LDC lacks Teff={tt}, logg={gg}", "population_ldc_invalid")
        zz = zs[np.argmin(np.abs(zs - z))]
        key = (float(tt), float(gg), float(zz))
        if key not in seen:
            row = select & (d['Z'] == zz)
            if np.sum(row) != 1:
                raise BackgroundUnavailable(f"background LDC row is not unique: {key}", "population_ldc_invalid")
            seen[key] = (float(d['u1'][row][0]), float(d['u2'][row][0]))
        u1[i], u2[i] = seen[key]
    return u1, u2


def _lnZ_DTP(time, flux, sigma, P_orb, M_s, R_s, Teff, N, u1, u2, population,
             rng=np.random, fold=None, sigma_veto=None):
    idx = _sample_background_indices(N, population["n_comp"], rng, legacy=True)
    return _lnZ_TTP(time, flux, sigma, P_orb, M_s, R_s, Teff, N, u1, u2,
                    companion_fr=population["fr"][idx], companion_is_host=False,
                    lnprior_companion=_background_prior(population, idx), rng=rng,
                    fold=fold, sigma_veto=sigma_veto)


def _db_planet(time, flux, sigma, P_orb, N, population, idx, rps, incs, eccs, argps,
               fold=None):
    mh, rh = population["M"][idx], population["R"][idx]
    u1h, u2h = ldc_background_arrays(population["Teff"], population["logg"], population["Z"])
    u1h, u2h = u1h[idx], u2h[idx]
    frc = population["fr"][idx]
    prior = _background_prior(population, idx)
    P_arr = np.full(N, P_orb)
    a_cm = ((G_cgs * mh * Msun) / (4 * pi**2) * (P_arr * 86400)**2)**(1 / 3)
    aRs = a_cm / (rh * Rsun)
    ks = rps * Rearth / (rh * Rsun)
    incs_rad = incs * pi / 180.0
    ws_rad = (90.0 - argps) * pi / 180.0
    ec = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    ptra = (rps * Rearth + rh * Rsun) / a_cm * ec
    lower = np.full(N, pi / 2)
    can = ptra <= 1.0
    lower[can] = np.arccos(ptra[can])
    coll = (rps * Rearth + rh * Rsun) > a_cm * (1 - eccs)
    mask = ((incs_rad >= lower) & ~coll & (population["logg"][idx] >= 3.5)
            & (population["Teff"][idx] <= 10000.0))
    if switch_is("FPP_ORBIT_PHYSICS", "roche"):
        mask &= _roche_ok(a_cm, eccs, mh, rh, 5.51)
    im = np.nonzero(mask)[0]
    lnL = np.full(N, -np.inf)
    exptime, nsub = _exposure_args()
    sig = None if fold is not None else _sigma_array(sigma, len(time))
    if fold is None:
        def call(sel, u1, u2):
            j = im[sel]
            return _dispatch_lnL(_batch_transit_lnL, (time, flux, sig, ks[j], P_arr[j], incs_rad[j], aRs[j],
                eccs[j], ws_rad[j], u1, u2, frc[j], True, exptime, nsub))
        lnL[im] = _call_by_ldc(call, u1h[im], u2h[im], len(im), None)
    else:
        def call(sel, u1, u2):
            j = im[sel]
            return _dispatch_lnL(_batch_transit_lnL_prod, (fold['nb'], fold['f'], fold['sigma'], fold['base_chi2'],
                fold['lnnorm'], fold['n'], fold['tmean'], fold['var_t'], fold['tmin'], fold['tmax'], fold['node_tab'],
                fold['pts_off'], fold['pts_t'], fold['W'], fold['delta'], ks[j], P_arr[j], incs_rad[j], aRs[j],
                eccs[j], ws_rad[j], u1, u2, frc[j], True))
        lm, st = _call_by_ldc(call, u1h[im], u2h[im], len(im), fold)
        lnL[im] = lm
        _note_exposure_stats(lm, st)
    return _lnZ_from_logw(lnL + prior)


def _lnZ_BTP(time, flux, sigma, P_orb, M_s, R_s, Teff, N, population,
             rng=np.random, fold=None, sigma_veto=None):
    idx = _sample_background_indices(N, population["n_comp"], rng)
    rps = _sample_rp(N, population['M'][idx], rng=rng)
    incs = _sample_inc(N, rng=rng)
    eccs = _sample_ecc(N, planet=True, P_orb=P_orb, rng=rng)
    argps = _sample_w(N, rng=rng)
    return _db_planet(time, flux, sigma, P_orb, N, population, idx, rps, incs, eccs, argps, fold)


def _binary_background_lnz(time, flux, sigma, P_orb, M_s, R_s, Teff, N, population, idx,
                           incs, qs, eccs, argps, background_host, u1=None, u2=None,
                           fold=None, sigma_veto=None):
    if sigma_veto is None:
        if fold is not None or np.ndim(sigma) != 0:
            raise ValueError("sigma_veto is required with a fold or per-bin sigma")
        sigma_veto = float(sigma)
    sig = None if fold is not None else _sigma_array(sigma, len(time))
    frc = population['fr'][idx]
    if background_host:
        mh, rh, th = population['M'][idx], population['R'][idx], population['Teff'][idx]
        u1h, u2h = ldc_background_arrays(population['Teff'], population['logg'], population['Z'])
        u1h, u2h = u1h[idx], u2h[idx]
    else:
        mh, rh, th = np.full(N, M_s), np.full(N, R_s), np.full(N, Teff)
        u1h, u2h = np.full(N, u1), np.full(N, u2)
    meb = qs * mh
    reb, teb = stellar_relations_capped(meb, rh, th)
    freb = flux_relation(meb) / (flux_relation(meb) + flux_relation(np.array([M_s])))
    if background_host:
        fr_bound = flux_relation(mh) / (flux_relation(mh) + flux_relation(np.array([M_s])))
        freb = freb * (frc / fr_bound)  # upstream's distance correction, not a redefined flux ratio
    prior = _background_prior(population, idx, freb if background_host else None)
    P_arr = np.full(N, P_orb)
    ec = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    acm = ((G_cgs * (mh + meb) * Msun) / (4 * pi**2) * (P_arr * 86400)**2)**(1 / 3)
    atw = ((G_cgs * (mh + meb) * Msun) / (4 * pi**2) * (2 * P_arr * 86400)**2)**(1 / 3)
    pt, ptw = (reb * Rsun + rh * Rsun) / acm * ec, (reb * Rsun + rh * Rsun) / atw * ec
    imin, imint = np.full(N, pi/2), np.full(N, pi/2)
    can = pt <= 1.0; imin[can] = np.arccos(pt[can])
    can = ptw <= 1.0; imint[can] = np.arccos(ptw[can])
    ir, wr = incs * pi / 180.0, (90.0 - argps) * pi / 180.0
    mask = (ir >= imin) & ~((reb*Rsun+rh*Rsun) > acm*(1-eccs)) & (qs < .95)
    twin = (ir >= imint) & ~((2*rh*Rsun) > atw*(1-eccs)) & (qs >= .95)
    if background_host:
        host_ok = (population['logg'][idx] >= 3.5) & (th <= 10000.)
        mask &= host_ok; twin &= host_ok
    if switch_is('FPP_ORBIT_PHYSICS', 'roche'):
        rho = meb * Msun / ((4./3.) * pi * (reb*Rsun)**3)
        mask &= _roche_ok(acm, eccs, mh, rh, rho)
        twin &= _roche_ok(atw, eccs, mh, rh, rho)
    secondary = switch_is('FPP_SECONDARY_GEOMETRY', 'consistent')
    u1s, u2s = u1h.copy(), u2h.copy()
    if secondary and np.any(mask | twin):
        sel = mask | twin
        gg = np.log10(G_cgs * meb[sel] * Msun / (reb[sel] * Rsun)**2)
        if background_host:
            u1s[sel], u2s[sel] = ldc_background_arrays(teb[sel], gg, population['Z'][idx[sel]])
        else:
            u1s[sel], u2s[sel] = get_ldc_arrays(teb[sel], gg)
    k = reb / rh
    exposure, nsub = _exposure_args()
    nudge = switch_is('FPP_K1_NUDGE', 'upstream')
    def run(keep, period, axes, reject_sec):
        lnL = np.full(N, -np.inf)
        im = np.nonzero(keep)[0]
        if fold is None:
            def call(sel, lu1, lu2):
                j = im[sel]
                return _dispatch_lnL(_batch_eb_transit_lnL, (time, flux, sig, float(sigma_veto), k[j], freb[j], period[j],
                    ir[j], axes[j], eccs[j], wr[j], lu1, lu2, frc[j], background_host, exposure, nsub, reject_sec,
                    secondary, nudge, u1s[j], u2s[j]))
            lnL[im] = _call_by_ldc(call, u1h[im], u2h[im], len(im), None)
        else:
            def call(sel, lu1, lu2):
                j = im[sel]
                return _dispatch_lnL(_batch_eb_transit_lnL_prod, (fold['nb'],fold['f'],fold['sigma'],fold['base_chi2'],
                    fold['lnnorm'],fold['n'],fold['tmean'],fold['var_t'],fold['tmin'],fold['tmax'],fold['node_tab'],
                    fold['pts_off'],fold['pts_t'],fold['W'],fold['delta'],float(sigma_veto),k[j],freb[j],period[j],
                    ir[j],axes[j],eccs[j],wr[j],lu1,lu2,frc[j],background_host,reject_sec,secondary,nudge,u1s[j],u2s[j]))
            lm, st = _call_by_ldc(call, u1h[im], u2h[im], len(im), fold)
            lnL[im] = lm; _note_exposure_stats(lm, st)
        return _lnZ_from_logw(lnL + prior)
    return run(mask, P_arr, acm/(rh*Rsun), True), run(twin, 2*P_arr, atw/(rh*Rsun), False)


def _lnZ_DEB(time, flux, sigma, P_orb, M_s, R_s, Teff, N, u1, u2, population,
             rng=np.random, fold=None, sigma_veto=None):
    incs = _sample_inc(N, rng=rng)
    qs = _sample_q(N, M_s, rng=rng)
    eccs = _sample_ecc(N, planet=False, P_orb=P_orb, rng=rng)
    argps = _sample_w(N, rng=rng)
    idx = _sample_background_indices(N, population['n_comp'], rng, legacy=True)
    return _binary_background_lnz(time,flux,sigma,P_orb,M_s,R_s,Teff,N,population,idx,incs,qs,eccs,argps,False,
                                   u1,u2,fold,sigma_veto)


def _lnZ_BEB(time, flux, sigma, P_orb, M_s, R_s, Teff, N, population,
             rng=np.random, fold=None, sigma_veto=None):
    incs = _sample_inc(N, rng=rng)
    # Preserve upstream's draw stream including the unused bound-companion draw.
    if switch_is('FPP_BACKGROUND_Q', 'target'):
        qs = _sample_q(N, M_s, rng=rng)
        q_uniform = None
    else:
        q_uniform = rng.rand(N)
    _sample_q_companion(N, M_s, rng=rng)
    eccs = _sample_ecc(N, planet=False, P_orb=P_orb, rng=rng)
    argps = _sample_w(N, rng=rng)
    idx = _sample_background_indices(N, population['n_comp'], rng)
    if q_uniform is not None:
        # One uniform per draw, with the prior transformed using ITS background host mass.
        qs = np.empty(N)
        host_mass = population['M'][idx]
        order = np.argsort(host_mass, kind='stable')
        edges = np.r_[0, np.nonzero(np.diff(host_mass[order]))[0] + 1, N]
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = order[lo:hi]
            class Uniforms:
                def rand(self, n):
                    return q_uniform[sel].copy()
            qs[sel] = _sample_q(len(sel), float(host_mass[sel[0]]), rng=Uniforms())
    return _binary_background_lnz(time,flux,sigma,P_orb,M_s,R_s,Teff,N,population,idx,incs,qs,eccs,argps,True,
                                   fold=fold,sigma_veto=sigma_veto)



# ============================================================
# 7. TIC query and flux ratio estimation
# ============================================================

def _query_tic_stars(tic_id, server_url=None, token=None, timeout=15):
    """Fetch nearby TIC stars from the ExoNit server.

    The server handles MAST queries and caching centrally.
    Returns (stars_list, gal_b) or None on failure.
    """
    if not server_url:
        logging.getLogger("exonit").debug(f"FPP: no server_url, cannot query nearby stars")
        return None

    import requests

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = requests.get(f"{server_url}/api/nearby/{tic_id}",
                            headers=headers, timeout=timeout)
        if resp.status_code != 200:
            logging.getLogger("exonit").debug(f"FPP: nearby query failed for TIC {tic_id}: "
                  f"HTTP {resp.status_code}")
            return None

        data = resp.json()
        stars = data.get("stars", [])
        gal_b = data.get("gal_b", 0.0)

        # Promote the target TIC to stars[0]. Catalogs.query_region does
        # not sort by distance from the query center, so the natural
        # position 0 is often an arbitrary faint neighbour with missing
        # catalog data — see the "FPP missing stellar params" incident
        # (2026-04-20): 39/98 baseline-100 TICs lost FPP because the
        # code further down takes `stars[0]` as the host star.
        target_idx = next(
            (i for i, s in enumerate(stars)
             if str(s.get("ID")) == str(tic_id)),
            None,
        )
        if target_idx is None:
            logging.getLogger("exonit").warning(
                f"FPP: TIC {tic_id} not found in its own /api/nearby "
                f"result ({len(stars)} stars); using stars[0] as fallback")
        elif target_idx != 0:
            stars.insert(0, stars.pop(target_idx))

        # Convert null → NaN for numeric fields
        for s in stars:
            for k in ('mass', 'rad', 'Teff', 'Tmag', 'plx'):
                if s.get(k) is None:
                    s[k] = np.nan

        source = data.get("source", "unknown")
        logging.getLogger("exonit").debug(f"FPP: TIC {tic_id} nearby stars from {source} "
              f"({len(stars)} stars)")
        return (stars, gal_b)

    except Exception as e:
        logging.getLogger("exonit").debug(f"FPP: nearby query failed for TIC {tic_id}: {e}")
        return None


PSF_SIGMA_PIX = 0.75
SYNTH_PIX_ARCSEC = 20.25
SYNTH_OFFSETS = tuple((ox, oy) for ox in (-0.4, 0.0, 0.4) for oy in (-0.4, 0.0, 0.4))
# 1 / (sigma * sqrt(2)): Phi(a, b) = 0.5 * (erf(b K) - erf(a K)); sqrt and * / are IEEE correctly rounded -> portable
_PSF_K = 1.0 / (PSF_SIGMA_PIX * math.sqrt(2.0))
GEOMETRY_VERSION = 2        # 1: v1 task geometry; 2: product apertures, epochs, sigma_cal (geometry v3)
FFI_REFERENCE_VALIDATABLE = False     # flip only in a commit that cites the passed v2 pre-registration


def _psf_pixel_weight(p, mu):
    """Integral of the 1-D Gaussian (sigma 0.75 px, mean mu) over the pixel [p - 1/2, p + 1/2]."""
    return 0.5 * (math.erf((p + 0.5 - mu) * _PSF_K) - math.erf((p - 0.5 - mu) * _PSF_K))


def _psf_aperture_flux(mu_x, mu_y, pixels):
    """Unit-amplitude PSF flux of a star at (mu_x, mu_y) in an aperture (sequence of (col, row)); sequential sum."""
    acc = 0.0
    for px, py in pixels:
        acc += _psf_pixel_weight(px, mu_x) * _psf_pixel_weight(py, mu_y)
    return acc


def tan_world_to_pixel(wcs, ra_deg, dec_deg):
    """0-based pixel (x, y) of the image grid of a TAN WCS (astropy `all_world2pix(ra, dec, 0)` equivalent).

    `wcs` = {"crpix": [x, y] (FITS, 1-based), "crval": [ra0, dec0] (deg), "pc": [p11, p12, p21, p22],
    "cdelt": [d1, d2] (deg/pix), "ctype": ["RA---TAN", "DEC--TAN"]}.  Gnomonic projection (FITS Paper II, TAN with the
    default pole); the intermediate world coordinates are (xi, eta) in degrees; [xi/d1, eta/d2] = PC [dx, dy].
    """
    ct = wcs.get("ctype", ["RA---TAN", "DEC--TAN"])
    if ct[0] != "RA---TAN" or ct[1] != "DEC--TAN":
        raise ValueError(f"unsupported WCS projection {ct}")
    a0 = math.radians(wcs["crval"][0])
    d0 = math.radians(wcs["crval"][1])
    a = math.radians(ra_deg)
    d = math.radians(dec_deg)
    da = a - a0
    sd, cd, sd0, cd0 = math.sin(d), math.cos(d), math.sin(d0), math.cos(d0)
    cda = math.cos(da)
    cosc = sd0 * sd + cd0 * cd * cda
    if cosc <= 0.0:
        raise ValueError("star is on the far side of the tangent point")
    xi = math.degrees(cd * math.sin(da) / cosc)
    eta = math.degrees((cd0 * sd - sd0 * cd * cda) / cosc)
    u = xi / wcs["cdelt"][0]
    v = eta / wcs["cdelt"][1]
    p11, p12, p21, p22 = wcs["pc"]
    det = p11 * p22 - p12 * p21
    dx = (p22 * u - p12 * v) / det
    dy = (-p21 * u + p11 * v) / det
    return wcs["crpix"][0] - 1.0 + dx, wcs["crpix"][1] - 1.0 + dy


def _default_aperture(target_pix):
    """Upstream's default: 5x5 box centred on round(target pixel) (np.round = half to even = Python round)."""
    cx, cy = round(target_pix[0]), round(target_pix[1])
    return [(float(cx + i), float(cy + j)) for i in range(-2, 3) for j in range(-2, 3)]


def _sector_flux_ratios(amp, mus, aperture):
    """Flux ratios of all stars in ONE aperture: amp[i] * PSF flux, normalised by the sequential total."""
    flux = [amp[i] * _psf_aperture_flux(mus[i][0], mus[i][1], aperture) for i in range(len(amp))]
    tot = 0.0
    for f in flux:
        tot += f
    return [f / tot for f in flux]


def _geometry_apertures(stars, geometry):
    """List of (mus, aperture) pairs, one per sector (real) or per sub-pixel offset (synthetic)."""
    out = []
    if geometry.get("kind") == "real":
        for sec in geometry["sectors"]:
            col0, row0 = float(sec["col0"]), float(sec["row0"])
            mus = []
            for s in stars:
                x, y = tan_world_to_pixel(sec["wcs"], s["ra"], s["dec"])
                mus.append((col0 + x, row0 + y))
            ap = sec.get("aperture")
            ap = _default_aperture(mus[0]) if not ap else [(float(c), float(r)) for c, r in ap]
            out.append((mus, ap))
        return out
    t = stars[0]
    cosd0 = math.cos(math.radians(t["dec"]))
    base = []
    for s in stars:
        dx = (s["ra"] - t["ra"]) * cosd0 * 3600.0 / SYNTH_PIX_ARCSEC
        dy = (s["dec"] - t["dec"]) * 3600.0 / SYNTH_PIX_ARCSEC
        base.append((dx, dy))
    ap = [(float(i), float(j)) for i in range(-2, 3) for j in range(-2, 3)]
    for ox, oy in geometry.get("offsets", SYNTH_OFFSETS):
        out.append(([(dx - ox, dy - oy) for dx, dy in base], ap))
    return out


def compute_flux_ratios_d5(stars, depth_ppm, geometry=None, dilution_corrected=False):
    """Plan D5: flux ratios, required depths and the viable-neighbour set.  Returns (stars, info).

    stars[0] is the target.  Every star gets `fluxratio` (mean over the geometry's apertures), `tdepth` (ppm; 0 if the
    star cannot produce the observed dip) and `viable` (= will be evaluated; the target is always evaluated).
    Real geometry: viability from the mean flux ratios exactly as upstream (`tdepth > 1 -> 0`).  Synthetic
    sub-pixel grid: viable = viable at ANY offset (conservative union), flux ratio = mean over offsets.
    `dilution_corrected` is upstream's calc_depths switch (aperture_depth = depth * fr_target when True).
    """
    if not stars:
        return [], {"geometry_kind": None}
    geometry = geometry if geometry else {"kind": "synthetic"}
    kind = geometry.get("kind", "synthetic")
    if kind == "real" and not geometry.get("sectors"):
        kind = "synthetic"
        geometry = {"kind": "synthetic"}
    n = len(stars)
    Tmin = min(s["Tmag"] for s in stars)
    amp = [10.0 ** ((Tmin - s["Tmag"]) / 2.5) for s in stars]
    per = [_sector_flux_ratios(amp, mus, ap) for mus, ap in _geometry_apertures(stars, geometry)]
    K = len(per)
    fr = []
    for i in range(n):
        acc = 0.0
        for k in range(K):
            acc += per[k][i]
        fr.append(acc / K)
    depth = float(depth_ppm)

    def required(frs, frt):
        ap_depth = depth * frt if dilution_corrected else depth
        td = [(ap_depth / f) if f != 0 else 0.0 for f in frs]
        return [0.0 if t > 1.0e6 else t for t in td]       # upstream: tdepth (fractional) > 1 -> 0

    td_mean = required(fr, fr[0])
    viable = [t > 0 for t in td_mean]
    if kind == "synthetic" and K > 1:
        for k in range(K):
            tdk = required(per[k], per[k][0])
            viable = [v or (t > 0) for v, t in zip(viable, tdk)]
    out = []
    for i, s in enumerate(stars):
        d = dict(s)
        d["fluxratio"] = fr[i]
        d["tdepth"] = td_mean[i]
        d["viable"] = bool(viable[i]) if i > 0 else True
        d["sep_arcsec"] = math.hypot((s["ra"] - stars[0]["ra"]) * math.cos(math.radians(stars[0]["dec"])),
                                     s["dec"] - stars[0]["dec"]) * 3600.0
        out.append(d)
    info = {"geometry_kind": kind, "geometry_n_sectors": K if kind == "real" else 0,
            "geometry_n_task_sectors": int(geometry.get("n_task_sectors", K)) if kind == "real" else 0,
            # fraction of the candidate's sectors that the real geometry covers (upstream averages over ALL sectors);
            # < 1 means a partial average: calc_fpp_native never validates it
            "geometry_coverage": (min(1.0, K / max(1, int(geometry.get("n_task_sectors", K)))) if kind == "real" else 0.0),
            # provenance of the real WCS per sector (spoc_lc_aperture / tesscut_aperture / ffi_reference / unknown)
            "geometry_source": sorted({str(sec.get("source", "unknown")) for sec in geometry["sectors"]}) if kind == "real" else ["synthetic"],
            # upstream operator: the SPOC mask where the sector has one, upstream's 5x5 at round(target) otherwise
            "geometry_mode": "upstream",
            "aperture_source": (sorted({"spoc_lc_mask" if sec.get("aperture") else "default_5x5" for sec in geometry["sectors"]})
                                if kind == "real" else ["synthetic"]),
            "geometry_path": (sorted({"spoc_mask" if sec.get("aperture") else "u_default" for sec in geometry["sectors"]})
                              if kind == "real" else ["synthetic"]),
            "geometry_path_emulated": False,
            "n_viable_neighbours": int(sum(1 for d in out[1:] if d["viable"])),
            "fluxratio_target": fr[0]}
    return out, info


def _compute_flux_ratios(stars, depth_ppm, geometry=None, dilution_corrected=False):
    """Stars with `fluxratio` / `tdepth` / `viable` (see compute_flux_ratios_d5); drops the geometry info."""
    return compute_flux_ratios_d5(stars, depth_ppm, geometry, dilution_corrected)[0]


# ============================================================
# 7a. D5 geometry v3: product apertures, K-epoch time average, smooth flux ratios (docs/fpp-ffi-geometry-v3-design.md)
# ============================================================
# What a port (Swift / Rust) must mirror, in the order the arithmetic runs (all plain f64, fixed order, no FMA contraction):
#  1. `_geom_resolve_sector`: the per-sector kind (mask / tglc_box / qlp_circle / default), the fixed pixel list, the
#     epoch offsets e_k (list order is part of the task), sigma_cal, c_shift.
#  2. positions: base_i = (col0 + x_i, row0 + y_i) from `tan_world_to_pixel` (scalar libm sin/cos/atan-free TAN, as D5),
#     then star position at node m, epoch k: ((base_i + c_shift) + d_m) + e_k with d_m = sigma_s * (a, b), a, b integers.
#  3. aperture per node: mask / tglc_box fixed; default = `_default_aperture` at round-half-even of (base_0 + c_shift + d_m);
#     qlp_circle = `qlp_circle_pixels` centred at d_m with the TARGET on a pixel centre: star positions are
#     (base_i + c_shift) - (base_0 + c_shift) (the circle follows the star; no epochs, no absolute WCS).
#  4. F_i = (sum_k amp_i * sum_p a_p * Wx(p_x, mu_x) * Wy(p_y, mu_y)) / K  (pixel sum sequential in list order; the weight
#     is `_psf_pixel_weight`, libm erf; k outer, then pixels; the sum over k starts at 0.0).
#  5. fr_i(s, m) = F_i / (sequential sum of F_j over stars in task order); fr_i(m) = (sequential sum over sectors) / S.
#  6. td_i(m) = ap_depth / fr_i(m) (0 if fr_i == 0), ap_depth = depth * fr_0(m) (True) or depth; viable at the node iff
#     0 < td <= 1e6 (ppm).  fr_i = sequential sum_m w_m fr_i(m), v_i = sequential sum_m w_m [viable at m]; nodes in the
#     order a outer, b inner (`_GEOM_W_HEX`, SHA-pinned, hex floats); M = 1 (w = 1.0) when every sigma_s is 0.
#  7. neighbour i is evaluated iff v_i >= V_MIN; its NTP / NEB / NEBx2P lnZ get + ln v_i (only when v_i != 1).
# libm calls in the region: erf (pixel weights), sin/cos/pow only in the TAN projection and in amp (as D5); no exp, no log
# (ln v_i is applied outside the region, in the scenario combination).

V_MIN = 0.01
SIGMA_CAL_DEFAULT = {"mask": 0.02, "tglc_box": 0.035, "qlp_circle": 0.05, "default": 0.035}      # px per axis, design 3
QLP_DEFAULT_RADIUS = 3.0          # px; BESTAP is 2.5 / 3.0 / 3.5 (47 / 128 / 17 of 192 bench files)
QLP_SUBGRID = 32                  # coverage = fraction of QLP_SUBGRID^2 midpoint samples of a pixel inside the circle
# tglc_box: True 2026-10-18 (the owner's decision) on the pre-registered v3 run, path T (TGLC 3x3 box, K-epoch FFI WCS) PASS on
# P1-P4 and P2R, docs/fpp-ffi-geometry-v3-data/result_v3/REPORT.md, script sha256 a7f6d647 (a7f6d6470cad41d5...).  Path U
# (default 5x5) FAILED, so u_default stays False; qlp_circle was not part of the run, so it stays False.
# A near-edge TGLC star (c = 1.27, provisional) is demoted in calc_fpp_by_source; an unknown NEAREDGE stays capped there too.
GEOMETRY_PATH_VALIDATED = {"spoc_mask": True, "tglc_box": True, "qlp_circle": False, "u_default": False}
# flip a path ONLY in a commit that cites a passed v3 report (docs/fpp-ffi-wcs-reference-preregistration-v3.md)
_GEOM_PATH_OF_KIND = {"mask": "spoc_mask", "tglc_box": "tglc_box", "qlp_circle": "qlp_circle", "default": "u_default"}

_GEOM_NODE_A = (-2, -1, 0, 1, 2)
_GEOM_W_HEX = (
    '0x1.8527aba212071p-9', '0x1.b404942866157p-7', '0x1.676f9c7838aa7p-6',
    '0x1.b404942866157p-7', '0x1.8527aba212071p-9', '0x1.b404942866157p-7',
    '0x1.e8862e93c9e5cp-5', '0x1.92b856facd857p-4', '0x1.e8862e93c9e5cp-5',
    '0x1.b404942866157p-7', '0x1.676f9c7838aa7p-6', '0x1.92b856facd857p-4',
    '0x1.4bfc906c2a8ebp-3', '0x1.92b856facd857p-4', '0x1.676f9c7838aa7p-6',
    '0x1.b404942866157p-7', '0x1.e8862e93c9e5cp-5', '0x1.92b856facd857p-4',
    '0x1.e8862e93c9e5cp-5', '0x1.b404942866157p-7', '0x1.8527aba212071p-9',
    '0x1.b404942866157p-7', '0x1.676f9c7838aa7p-6', '0x1.b404942866157p-7',
    '0x1.8527aba212071p-9',
)
# sha256 of "\n".join(_GEOM_W_HEX); table written by scripts/fpp/gen_geometry_nodes.py, fixture for the ports in
# tests/fpp/step3/fixtures/fpp_geometry_nodes.json
GEOM_NODES_SHA256 = "fa1c1df76c1a1e7b5357c7fa1e05131d5c124b09d0981dddccedebe11facc227"


def _load_geom_nodes():
    import hashlib
    digest = hashlib.sha256("\n".join(_GEOM_W_HEX).encode("ascii")).hexdigest()
    if digest != GEOM_NODES_SHA256 or len(_GEOM_W_HEX) != 25:
        raise RuntimeError(f"geometry node table sha256 {digest} != pinned {GEOM_NODES_SHA256}")
    return tuple(float.fromhex(h) for h in _GEOM_W_HEX)


try:
    _GEOM_W = _load_geom_nodes()
    GEOM_NODES_ERROR = None
except RuntimeError as _gn_e:                    # an edited table: smooth geometry refuses to run (no silent recomputation)
    _GEOM_W = (0.0,) * 25
    GEOM_NODES_ERROR = str(_gn_e)


def _norm_lc_source(lc_source):
    """'spoc' / 'tglc' / 'qlp' / 'mixed' / 'unknown' (anything else, None and the worker's FFI fallback included)."""
    s = str(lc_source).strip().lower() if lc_source is not None else ""
    return s if s in ("spoc", "tglc", "qlp", "mixed") else "unknown"


def qlp_circle_pixels(cx, cy, r, sub=QLP_SUBGRID):
    """[(col, row, a_p)] of a circular aperture of radius r centred at (cx, cy) (CCD pixel coordinates, pixel p spans
    [p - 1/2, p + 1/2]); a_p = fraction of the sub x sub midpoint samples of the pixel with dx^2 + dy^2 <= r^2.
    Column-major (col outer, row inner), only pixels with a_p > 0."""
    r = float(r)
    r2 = r * r
    inv = 1.0 / sub
    nn = float(sub * sub)
    out = []
    for px in range(int(math.floor(cx - r - 0.5)), int(math.ceil(cx + r + 0.5)) + 1):
        x_lo, x_hi = px - 0.5 - cx, px + 0.5 - cx
        xs2 = None
        for py in range(int(math.floor(cy - r - 0.5)), int(math.ceil(cy + r + 0.5)) + 1):
            y_lo, y_hi = py - 0.5 - cy, py + 0.5 - cy
            near_x = 0.0 if x_lo <= 0.0 <= x_hi else min(abs(x_lo), abs(x_hi))
            near_y = 0.0 if y_lo <= 0.0 <= y_hi else min(abs(y_lo), abs(y_hi))
            if near_x * near_x + near_y * near_y > r2:
                continue                                           # no sample can be inside
            far_x, far_y = max(abs(x_lo), abs(x_hi)), max(abs(y_lo), abs(y_hi))
            if far_x * far_x + far_y * far_y <= r2 * (1.0 - 1e-9):
                out.append((float(px), float(py), 1.0))            # every sample is inside
                continue
            if xs2 is None:
                xs2 = [((px - 0.5 + (i + 0.5) * inv) - cx) ** 2 for i in range(sub)]
            ys2 = [((py - 0.5 + (j + 0.5) * inv) - cy) ** 2 for j in range(sub)]
            cnt = 0
            for a in xs2:
                for b in ys2:
                    if a + b <= r2:
                        cnt += 1
            if cnt:
                out.append((float(px), float(py), cnt / nn))
    return out


def _tglc_box_pixels(cx, cy):
    return [(float(cx + i), float(cy + j), 1.0) for i in (-1, 0, 1) for j in (-1, 0, 1)]


def _geom_resolve_sector(sec, lc_source):
    """Per-sector geometry spec of the smooth operator (see the port list above).

    Task keys (all optional, server: exonit/fpp_geometry.py): `aperture_kind` in {mask, tglc_box, qlp_circle, default},
    `aperture` (pixel list; mask, or a TGLC box clipped to the detector), `box_centre` [col, row] (TGLC header),
    `circle_radius` (QLP BESTAP), `c_shift` [dx, dy] (common-mode target shift = epoch mean minus the WCS position),
    `epochs` [[ex, ey], ...] (offsets from the mean, order is part of the task), `sigma_cal` (px per axis).
    A product that is known (lc_source) but whose aperture the task does not carry is EMULATED from our own position
    (tglc_box_emulated / qlp_circle_assumed): flagged, and never validated.
    """
    lc = _norm_lc_source(lc_source) if lc_source is not None else None
    kind = sec.get("aperture_kind")
    ap = sec.get("aperture")
    emulated = False
    if kind is None:
        if lc == "tglc":
            kind, emulated = "tglc_box", True
        elif lc == "qlp":
            kind, emulated = "qlp_circle", True
        else:
            kind = "mask" if ap else "default"
    elif kind == "tglc_box":
        emulated = not (sec.get("box_centre") or ap)
    elif kind == "qlp_circle":
        emulated = sec.get("circle_radius") is None
    elif kind == "mask":
        if not ap:
            raise ValueError("aperture_kind 'mask' without an aperture")
    elif kind != "default":
        raise ValueError(f"unknown aperture_kind {kind!r}")
    pix = None
    radius = None
    box_centre = None
    if kind == "mask":
        pix = [(float(c), float(r), 1.0) for c, r in ap]
        src = str(sec.get("aperture_source") or ("spoc_lc_mask"))
    elif kind == "tglc_box":
        if ap and not emulated:
            pix = [(float(c), float(r), 1.0) for c, r in ap]
        elif sec.get("box_centre"):
            bc = sec["box_centre"]
            pix = _tglc_box_pixels(int(bc[0]), int(bc[1]))
        src = "tglc_box_emulated" if emulated else "tglc_box_header"
        # emulated: the centre is round(base_0 + c_shift), known only once the positions are (resolved by the caller)
    elif kind == "qlp_circle":
        radius = float(sec["circle_radius"]) if sec.get("circle_radius") is not None else QLP_DEFAULT_RADIUS
        if not (0.5 <= radius <= 8.0):
            raise ValueError(f"circle_radius {radius} outside the plausible 0.5-8 px (BESTAP is 2.5-3.5)")
        src = "qlp_circle_assumed" if emulated else "qlp_bestap_circle"
    else:
        src = "default_5x5"
    cs = sec.get("c_shift") or (0.0, 0.0)
    eps = sec.get("epochs") or ((0.0, 0.0),)
    if len(cs) != 2 or any(len(e) != 2 for e in eps):
        raise ValueError("c_shift and every epoch must be [dx, dy]")
    if not all(math.isfinite(float(v)) for v in (*cs, *(x for e in eps for x in e))):
        raise ValueError("c_shift / epochs must be finite")
    if kind == "qlp_circle":
        eps = ((0.0, 0.0),)                                   # the aperture follows the star: no drift
    sig = sec.get("sigma_cal")
    sig = float(SIGMA_CAL_DEFAULT[kind] if sig is None else sig)
    if not (sig >= 0.0 and math.isfinite(sig)):
        raise ValueError("sigma_cal must be a finite number >= 0")
    return {"kind": kind, "pix": pix, "radius": radius, "emulated": bool(emulated), "aperture_source": src,
            "path": _GEOM_PATH_OF_KIND[kind], "cshift": (float(cs[0]), float(cs[1])),
            "eps": tuple((float(e[0]), float(e[1])) for e in eps), "sigma": sig, "source": str(sec.get("source", "unknown"))}


def _star_aperture_flux(mu_x, mu_y, pix):
    """sum_p a_p * Wx * Wy over the (col, row, a_p) list; the column / row weights are cached per call (same floats as
    a direct `_psf_pixel_weight` call, so the sequential pixel sum is bit-identical to `_psf_aperture_flux` for a_p = 1)."""
    wxd, wyd = {}, {}
    acc = 0.0
    for px, py, a in pix:
        wx = wxd.get(px)
        if wx is None:
            wx = wxd[px] = _psf_pixel_weight(px, mu_x)
        wy = wyd.get(py)
        if wy is None:
            wy = wyd[py] = _psf_pixel_weight(py, mu_y)
        acc += a * wx * wy
    return acc


def compute_flux_ratios_smooth(stars, depth_ppm, geometry, dilution_corrected=False, lc_source=None, v_min=V_MIN):
    """Plan D5 with geometry v3 (design section 5): flux ratios of the epoch-averaged geometry integrated over the Gaussian
    calibration uncertainty, and a viability WEIGHT per neighbour.  Returns (stars, info); stars[0] is the target.

    Every star gets `fluxratio` (the expectation over the calibration nodes), `viability` (probability in [0, 1] that the
    star can produce the observed dip), `tdepth` (ppm, from the expected ratios), `viable` (= viability >= v_min; the target
    is always viable).  With every sigma_cal = 0 and no epochs this is D5 byte for byte (one node of weight 1.0).
    """
    if GEOM_NODES_ERROR:
        raise GeometryUnavailable(f"FPP geometry unavailable: {GEOM_NODES_ERROR}")
    n = len(stars)
    secs = [_geom_resolve_sector(sec, lc_source) for sec in geometry["sectors"]]
    S = len(secs)
    Tmin = min(s["Tmag"] for s in stars)
    amp = [10.0 ** ((Tmin - s["Tmag"]) / 2.5) for s in stars]
    base = []
    for sec, sr in zip(geometry["sectors"], secs):
        col0, row0 = float(sec["col0"]), float(sec["row0"])
        bx, by = [], []
        for s in stars:
            x, y = tan_world_to_pixel(sec["wcs"], s["ra"], s["dec"])
            bx.append((col0 + x) + sr["cshift"][0])
            by.append((row0 + y) + sr["cshift"][1])
        base.append((bx, by))
        if sr["kind"] == "tglc_box" and sr["pix"] is None:       # emulated header: the box around round(target)
            sr["pix"] = _tglc_box_pixels(round(bx[0]), round(by[0]))
    if all(sr["sigma"] == 0.0 for sr in secs):
        nodes = [(0, 0, 1.0)]
    else:
        nodes = [(a, b, _GEOM_W[ia * 5 + ib]) for ia, a in enumerate(_GEOM_NODE_A) for ib, b in enumerate(_GEOM_NODE_A)]
    depth = float(depth_ppm)
    fr_exp = [0.0] * n
    v_exp = [0.0] * n
    for a, b, w in nodes:
        per = []
        for (bx, by), sr in zip(base, secs):
            dx, dy = sr["sigma"] * a, sr["sigma"] * b
            if sr["kind"] == "qlp_circle":
                # the circle follows the target in every frame: only the RELATIVE geometry matters, so the target sits on the
                # centre of a pixel (canonical grid phase, no dependence on the absolute WCS) and the circle centre is d
                pix = qlp_circle_pixels(dx, dy, sr["radius"])
                sdx = sdy = 0.0
            else:
                if sr["kind"] == "default":
                    pix = [(x, y, 1.0) for x, y in _default_aperture((bx[0] + dx, by[0] + dy))]
                else:
                    pix = sr["pix"]
                sdx, sdy = dx, dy
            eps = sr["eps"]
            K = len(eps)
            F = []
            for i in range(n):
                acc = 0.0
                for ex, ey in eps:
                    if sr["kind"] == "qlp_circle":
                        acc += amp[i] * _star_aperture_flux(bx[i] - bx[0], by[i] - by[0], pix)
                    else:
                        acc += amp[i] * _star_aperture_flux(bx[i] + sdx + ex, by[i] + sdy + ey, pix)
                F.append(acc / K)
            tot = 0.0
            for f in F:
                tot += f
            per.append([f / tot for f in F])
        frm = []
        for i in range(n):
            acc = 0.0
            for k in range(S):
                acc += per[k][i]
            frm.append(acc / S)
        ap_depth = depth * frm[0] if dilution_corrected else depth
        for i in range(n):
            td = (ap_depth / frm[i]) if frm[i] != 0 else 0.0
            ind = 1.0 if (0.0 < td <= 1.0e6) else 0.0
            fr_exp[i] += w * frm[i]
            v_exp[i] += w * ind
    ap_depth = depth * fr_exp[0] if dilution_corrected else depth
    out = []
    skipped = 0.0
    weighted = 0
    for i, s in enumerate(stars):
        d = dict(s)
        d["fluxratio"] = fr_exp[i]
        td = (ap_depth / fr_exp[i]) if fr_exp[i] != 0 else 0.0
        d["tdepth_raw"] = td                                # before the upstream cut (used by the v3 validation, P2 band)
        d["tdepth"] = 0.0 if td > 1.0e6 else td
        if i == 0:
            d["viability"], d["viable"] = 1.0, True
        else:
            v = min(1.0, max(0.0, v_exp[i]))
            if v > 1.0 - 1e-12:
                v = 1.0                                            # the 25 weights sum to 1 within rounding
            d["viability"] = v
            d["viable"] = bool(v >= v_min)
            if d["viable"] and v < 1.0:
                weighted += 1
            if not d["viable"]:
                skipped += v
        d["sep_arcsec"] = math.hypot((s["ra"] - stars[0]["ra"]) * math.cos(math.radians(stars[0]["dec"])),
                                     s["dec"] - stars[0]["dec"]) * 3600.0
        out.append(d)
    info = {"geometry_kind": "real", "geometry_mode": "smooth", "geometry_n_sectors": S,
            "geometry_n_task_sectors": int(geometry.get("n_task_sectors", S)),
            "geometry_coverage": min(1.0, S / max(1, int(geometry.get("n_task_sectors", S)))),
            "geometry_source": sorted({sr["source"] for sr in secs}),
            "aperture_source": sorted({sr["aperture_source"] for sr in secs}),
            "geometry_path": sorted({sr["path"] for sr in secs}),
            "geometry_path_emulated": any(sr["emulated"] for sr in secs),
            "geometry_n_epochs": [len(sr["eps"]) for sr in secs],
            "geometry_sigma_cal": [sr["sigma"] for sr in secs],
            "n_nodes": len(nodes),
            "n_viable_neighbours": int(sum(1 for d in out[1:] if d["viable"])),
            "n_viability_weighted": weighted, "viability_mass_skipped": skipped,
            "fluxratio_target": fr_exp[0]}
    return out, info


def _ffi_reference_tglc_box_k_epoch(geo_info):
    """True iff the ffi_reference geometry is exactly what the v3 pre-registered run validated for path T: smooth mode,
    every sector a header-given TGLC box (not emulated), each with a K-epoch (>= 2 epochs) FFI WCS cache entry."""
    n_ep = geo_info.get("geometry_n_epochs") or []
    return bool(geo_info.get("geometry_mode") == "smooth"
                and geo_info.get("geometry_path") == ["tglc_box"]
                and not geo_info.get("geometry_path_emulated")
                and geo_info.get("aperture_source") == ["tglc_box_header"]
                and n_ep and all(int(k) >= 2 for k in n_ep))


def _flux_ratios_for_mode(stars, depth_ppm, geometry, dilution_corrected, lc_source):
    """D5 flux ratios under the active FPP_GEOMETRY switch: smooth (real geometry only) or the upstream operator."""
    real = bool(geometry) and geometry.get("kind") == "real" and bool(geometry.get("sectors"))
    if real and switch_is("FPP_GEOMETRY", "smooth"):
        return compute_flux_ratios_smooth(stars, depth_ppm, geometry, dilution_corrected, lc_source)
    return compute_flux_ratios_d5(stars, depth_ppm, geometry, dilution_corrected)


# ============================================================
# 7a'. Dilution convention per light-curve product (plan D8, docs/reviews/2026-10-17-tglc-qlp-dilution.md)
# ============================================================
# The convention is a property of the DATA: SPOC PDCSAP and TGLC cal_aper_flux are target-referenced (True); QLP and every
# unverifiable product run BOTH conventions and report the higher FPP (never validated).  `dilution_corrected=True` means the
# light curve is already on-target: the target scenarios use it as is and a neighbour's frame is 1 + (fr_t / fr_i)(f - 1).

TGLC_DEPTH_C = 1.09              # real flux / Gaia-predicted flux (docs/reviews/2026-10-15-tglc-depth-excess.md); exonit/physical_sanity.py
TGLC_NEAR_EDGE_C = 1.27          # NEAREDGE stars (provisional)


def frame_fluxratio(fr_i, fr_t, dilution_corrected):
    """Flux fraction by which the light curve is renormalised to put it in star i's frame.

    False (raw aperture flux): fr_i.  True (already target-referenced): fr_i / fr_t, so the target itself is 1.0 (not
    renormalised).  The ONE place that encodes the convention: the target, the neighbour (NTP / NEB) and any
    family that lives in a neighbour's frame must use it.
    """
    return (fr_i / fr_t) if dilution_corrected else fr_i


def dilution_plan(lc_source):
    """{'conventions': tuple of dilution_corrected values to run, 'unverified': bool} for a light-curve product."""
    s = _norm_lc_source(lc_source) if lc_source is not None else "unknown"
    if s in ("spoc", "tglc"):
        return {"conventions": (True,), "unverified": False}
    return {"conventions": (True, False), "unverified": True}


def lc_depth_scale(lc_source, near_edge=None):
    """Factor applied to the deviation of the light curve from 1 (and to the depth): TGLC 1/c, everything else 1."""
    if lc_source is not None and _norm_lc_source(lc_source) == "tglc":
        return 1.0 / (TGLC_NEAR_EDGE_C if near_edge else TGLC_DEPTH_C)
    return 1.0


# ============================================================
# 7b. Production fold (mean bins, plan N5)
# ============================================================
# Normative reference: tests/fpp/upstream_golden/fold_spec.py — plain sequential f64 with an
# operation order that IS the spec (decision 0.3 (j'), ladder acceptance N5), so Swift/Rust and
# this module can reproduce the fixtures bit-exactly. Therefore: no numpy pairwise reductions
# (np.sum / np.mean) and no np.quantile (its lerp differs by 1 ulp when g >= 0.5). The heavy
# loops are numba njit over float64 arrays; the arithmetic order is unchanged.
# The median-bin fold `_phase_fold_bin` below stays as it is — the parity replays use it.

FOLD_N_BINS = 1000
FOLD_K_QUANT = 256


@numba.njit(cache=True)
def _fold_quantiles_nb(s, K, out):
    """Type-7 quantile spec of fold_spec.quantiles over an ascending-sorted float64 array.

    p = (i + 0.5) / K; h = (n - 1) * p; j = floor(h); g = h - j;
    q = s[j] if j + 1 >= n else s[j] + g * (s[j + 1] - s[j]).
    """
    n = s.shape[0]
    nm1 = n - 1
    for i in range(K):
        p = (i + 0.5) / K
        h = nm1 * p
        j = int(math.floor(h))
        g = h - j
        if j + 1 >= n:
            out[i] = s[j]
        else:
            out[i] = s[j] + g * (s[j + 1] - s[j])


def fold_quantiles(x, K):
    """K quantiles of the sample x with the fold-ladder type-7 spec (float64 array of K values).

    `np.sort(..., kind="stable")` is the ascending sort of the spec's `sorted(...)`; the
    arithmetic itself runs in `_fold_quantiles_nb` in the spec's order.
    """
    K = int(K)
    if K <= 0:
        return np.empty(0, dtype=np.float64)
    s = np.sort(np.asarray(x, dtype=np.float64).ravel(), kind="stable")
    if s.size == 0:
        return np.full(K, np.nan, dtype=np.float64)
    out = np.empty(K, dtype=np.float64)
    _fold_quantiles_nb(s, K, out)
    return out


@numba.njit(cache=True)
def _fold_phase_nb(t, t0, period, out):
    """phase = ((t - t0) / period) % 1.0, then phase -= 1.0 when phase > 0.5 (fold_spec order).

    Numba's float % is CPython's float remainder (fmod plus the sign correction), so the bits
    match the Python formula of fold_spec.py — verified by tests/fpp/upstream_golden/test_f_fold.py.
    """
    for i in range(t.shape[0]):
        p = ((t[i] - t0) / period) % 1.0
        if p > 0.5:
            p -= 1.0
        out[i] = p


@numba.njit(cache=True)
def _fold_bins_nb(ph, fs, starts, counts, n_bins, k_quant, scratch,
                  flux_mean, flux_m2, phase_mean, phase_m2, var_b, quant_out):
    """Per-bin sequential sums (mean, second moment, Var_b) and the bin's phase quantiles.

    Every bin holds its points contiguously in ascending input order, so the sums run in the
    same order as the sequential reference. `scratch` is reused as the sort buffer of the
    quantile spec (it must be at least max(counts) long).
    """
    for b in range(n_bins):
        n = counts[b]
        if n == 0:
            continue
        st = starts[b]
        s = 0.0
        for i in range(n):
            s += fs[st + i]
        fmean = s / n
        m2 = 0.0
        for i in range(n):
            d = fs[st + i] - fmean
            m2 += d * d
        s = 0.0
        for i in range(n):
            s += ph[st + i]
        pmean = s / n
        pm2 = 0.0
        for i in range(n):
            d = ph[st + i] - pmean
            pm2 += d * d
        flux_mean[b] = fmean
        flux_m2[b] = m2
        phase_mean[b] = pmean
        phase_m2[b] = pm2
        var_b[b] = pm2 / n
        for i in range(n):
            scratch[i] = ph[st + i]
        scratch[:n].sort()
        _fold_quantiles_nb(scratch[:n], k_quant, quant_out[b])


def fold_production(time, flux, t0, period, n_bins=FOLD_N_BINS, k_quant=FOLD_K_QUANT,
                    keep_phases=False):
    """Production fold of plan 0.3 (j'): mean bins + each bin's phase distribution (N5).

    Same filtering as the reference (non-finite time/flux dropped, input order kept), same bin
    edges np.linspace(-0.5, 0.5, n_bins + 1) and the same digitize-then-clip binning. Returns the
    list of non-empty-bin dicts with the keys of fold_spec.fold: bin, n, flux_mean, flux_m2,
    phase_mean, phase_m2, var_b (= phase_m2 / n, population variance) and quantiles (k_quant
    values of the bin's phases). With keep_phases=True every row also carries "phases", the
    bin's phases SORTED ASCENDING — the exact rung of the ladder of 0.3 (j'), consumed by
    `prod_fold_arrays`. No sigma is derived here (plan D15 pending): flux_m2 and n are the
    inputs of any estimator.
    """
    t = np.asarray(time, dtype=np.float64).ravel()
    f = np.asarray(flux, dtype=np.float64).ravel()
    keep = np.flatnonzero(np.isfinite(t) & np.isfinite(f))
    if keep.size == 0:
        return []
    tk = np.ascontiguousarray(t[keep])
    fk = np.ascontiguousarray(f[keep])

    ph = np.empty(tk.size, dtype=np.float64)
    _fold_phase_nb(tk, float(t0), float(period), ph)

    edges = np.linspace(-0.5, 0.5, n_bins + 1)
    b_idx = np.clip(np.digitize(ph, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(b_idx, minlength=n_bins).astype(np.int64)
    order = np.argsort(b_idx, kind="stable")          # input order kept inside every bin
    ph_s = np.ascontiguousarray(ph[order])
    fs_s = np.ascontiguousarray(fk[order])
    starts = np.empty(n_bins + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(counts, out=starts[1:])

    filled = np.flatnonzero(counts)
    flux_mean = np.zeros(n_bins, dtype=np.float64)
    flux_m2 = np.zeros(n_bins, dtype=np.float64)
    phase_mean = np.zeros(n_bins, dtype=np.float64)
    phase_m2 = np.zeros(n_bins, dtype=np.float64)
    var_b = np.zeros(n_bins, dtype=np.float64)
    quant_out = np.empty((n_bins, int(k_quant)), dtype=np.float64)
    scratch = np.empty(int(counts.max()), dtype=np.float64)
    _fold_bins_nb(ph_s, fs_s, starts, counts, int(n_bins), int(k_quant), scratch,
                  flux_mean, flux_m2, phase_mean, phase_m2, var_b, quant_out)

    rows = []
    for b in filled:
        row = dict(bin=int(b), n=int(counts[b]),
                   flux_mean=float(flux_mean[b]), flux_m2=float(flux_m2[b]),
                   phase_mean=float(phase_mean[b]), phase_m2=float(phase_m2[b]),
                   var_b=float(var_b[b]), quantiles=quant_out[b])
        if keep_phases:
            row["phases"] = np.sort(ph_s[starts[b]:starts[b + 1]],
                                    kind="stable").tolist()
        rows.append(row)
    return rows


# ============================================================
# 7c. Likelihood sigma (D15, plan 0.3 (q'))
# ============================================================
# Normative reference: tests/fpp/upstream_golden/sigma_estimators.py — the verbatim port of the
# bench prototype HELPER_SIGMA (`sigbin` ablation). The arithmetic and the operation order below
# are that reference's, with ONE change (step 3): the phase-curve fit is the specified Householder QR (no LAPACK lstsq, agreement ~6e-11 with the prototype), so the fixtures fixtures/s_q_sigma.{json,npz} are
# reproduced to numerical precision. The mean-fold inputs come from the production fold above
# (mean bins of plan 0.3 (j'), decision 0.3 (q')): bin centre * period as the bin time, the bin's
# mean flux and its point count, then the reference's filter of the bins with a non-finite time or
# flux. The reference uses np.nanmean per bin and this module the sequential mean of fold_produc-
# tion, so the agreement is to ~1e-15 relative, not bit-exact.

def _mad(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


FIT_ILL_CONDITIONED_RCOND = 1e-8     # (q') phase-curve fit: below this the design is near-singular (clustered phases)


@numba.njit(cache=True)
def _phase_resid_householder(xb, y):
    """Residual of the low-order phase-curve fit of the (q') beta factors, by an explicitly specified Householder QR.

    NORMATIVE SPEC (Swift / Rust reproduce it bit for bit; replaces numpy lstsq, which goes through LAPACK):
      design row i = [1, cos(t), sin(t), cos(2t), sin(2t)] with t = 2.0 * pi * xb[i] (scalar libm cos / sin of
      `4.0 * pi * xb[i]` for the second harmonic);
      for j = 0 .. 4: norm = sqrt(sum_{i >= j} R[i,j]^2) (sequential i); alpha = -norm if R[j,j] >= 0 else norm;
        v_j = R[j,j] - alpha, v_i = R[i,j] (i > j); vn2 = sum_{i >= j} v_i^2 (sequential); if vn2 == 0 skip;
        for every column c >= j and for b: s = sum_{i >= j} v_i R[i,c] (sequential), f = 2.0 * s / vn2,
        R[i,c] -= f * v_i;
      back substitution j = 4 .. 0: coef_j = (b_j - sum_{c > j} R[j,c] coef_c, accumulated by subtraction in
        increasing c) / R[j,j] (coef_j = 0 when R[j,j] == 0);
      residual r_i = y_i - s_i with s_i = (((a0 coef0 + a1 coef1) + a2 coef2) + a3 coef3) + a4 coef4, a = the
        UNTRANSFORMED design row;
      conditioning proxy rcond = min_j |R[j,j]| / max_j |R[j,j]| (0.0 if max is 0), returned with the residual; a fit
        with rcond < FIT_ILL_CONDITIONED_RCOND (phases clustered in too few bins) is flagged by the caller.
    No pivoting, no fused multiply-add.
    """
    n = xb.shape[0]
    p = 5
    A = np.empty((n, p))
    for i in range(n):
        th = 2.0 * math.pi * xb[i]
        th2 = 4.0 * math.pi * xb[i]
        A[i, 0] = 1.0
        A[i, 1] = math.cos(th)
        A[i, 2] = math.sin(th)
        A[i, 3] = math.cos(th2)
        A[i, 4] = math.sin(th2)
    R = A.copy()
    b = y.copy()
    v = np.empty(n)
    for j in range(p):
        s = 0.0
        for i in range(j, n):
            s += R[i, j] * R[i, j]
        norm = math.sqrt(s)
        alpha = -norm if R[j, j] >= 0.0 else norm
        v[j] = R[j, j] - alpha
        for i in range(j + 1, n):
            v[i] = R[i, j]
        vn2 = 0.0
        for i in range(j, n):
            vn2 += v[i] * v[i]
        if vn2 == 0.0:
            continue
        for c in range(j, p):
            s = 0.0
            for i in range(j, n):
                s += v[i] * R[i, c]
            f = 2.0 * s / vn2
            for i in range(j, n):
                R[i, c] -= f * v[i]
        s = 0.0
        for i in range(j, n):
            s += v[i] * b[i]
        f = 2.0 * s / vn2
        for i in range(j, n):
            b[i] -= f * v[i]
    coef = np.zeros(p)
    for j in range(p - 1, -1, -1):
        s = b[j]
        for c in range(j + 1, p):
            s -= R[j, c] * coef[c]
        coef[j] = s / R[j, j] if R[j, j] != 0.0 else 0.0
    r = np.empty(n)
    for i in range(n):
        s = A[i, 0] * coef[0]
        for c in range(1, p):
            s += A[i, c] * coef[c]
        r[i] = y[i] - s
    dmin = abs(R[0, 0])
    dmax = dmin
    for j in range(1, p):
        a = abs(R[j, j])
        if a < dmin:
            dmin = a
        if a > dmax:
            dmax = a
    return r, (dmin / dmax if dmax > 0.0 else 0.0)


def _sigma_components_from_fold(time, flux, period, t0, duration, time_pf, flux_pf, counts):
    """The intermediate values of the (q') production noise model, in the prototype's operation order.

    Returns dict(sigma_pt, beta_bin, beta_T14, beta, sigma_oot, sigma_b):
      sigma_pt  1.4826 MAD of the FIRST DIFFERENCES of the time-ordered out-of-transit points (/ sqrt 2), robust
                against phase signals; the primary and the secondary-eclipse windows are excluded
      sigma_oot scalar: 1.4826 MAD of the ADJACENT-BIN differences of the out-of-transit (mean) bins / sqrt 2
                (robust against slow phase curves)
      beta_bin  bin-scale factor on MEAN bins (the median-bin sqrt(pi/2) is not part of it), from the residuals of
                the out-of-transit mean bins after removing the low-order (up to 2nd harmonic) phase curve
      beta_T14  time-averaging factor at the T14 scale (Pont et al. 2006): scatter of block means of ~T14 against
                the white expectation from the bin-level scatter
      beta      the combined factor min(max(1, beta_bin, beta_T14), 5)
      sigma_b   per-bin sigma_pt / sqrt(n_b) * beta (the prototype's `per_bin = True` return value)
    beta_bin / beta_T14 / beta are NaN when the prototype falls back to the scalar sigma_oot in every bin (fewer
    than 20 out-of-transit bins with n_b >= 5); sigma_b is then that fallback array.
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    time_pf = np.asarray(time_pf, dtype=np.float64)
    flux_pf = np.asarray(flux_pf, dtype=np.float64)
    guard = 0.75 * duration + 0.02
    ph = ((time - t0) / period + 0.5) % 1.0 - 0.5
    dt = ph * period
    dsec = (np.abs(ph) - 0.5) * period
    ok = (np.abs(dt) > guard) & (np.abs(dsec) > guard) & np.isfinite(flux)
    order = np.argsort(time)
    tt, ff, oo = time[order], flux[order], ok[order]
    cad = float(np.median(np.diff(tt))) if len(tt) > 2 else 1.0
    pair = oo[1:] & oo[:-1] & (np.diff(tt) < 3.0 * cad)
    d = np.diff(ff)[pair]
    if len(d) >= 50:
        sig_pt = 1.4826 * float(np.median(np.abs(d - np.median(d)))) / math.sqrt(2.0)
    elif ok.sum() >= 20:
        sig_pt = _mad(flux[ok])
    else:
        sig_pt = float(np.std(flux))
    ob = (np.abs(time_pf) > guard) & (np.abs(np.abs(time_pf) - 0.5 * period) > guard)
    io = np.where(ob)[0]
    ia = io[1:][(np.diff(io) == 1) & (np.diff(time_pf[io]) < 1.5 * period / 1000.0)]
    if len(ia) >= 20:
        d_ = flux_pf[ia] - flux_pf[ia - 1]
        sig_oot = 1.4826 * float(np.median(np.abs(d_ - np.median(d_)))) / math.sqrt(2.0)
    else:
        sig_oot = _mad(flux_pf[ob]) if ob.sum() >= 20 else float(np.std(flux_pf))
    sig_oot = max(sig_oot, 1e-9)
    n = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    white = sig_pt / np.sqrt(n)
    okb = ob & (n >= 5)
    if okb.sum() < 20:
        return dict(sigma_pt=sig_pt, beta_bin=float("nan"), beta_T14=float("nan"), beta=float("nan"),
                    sigma_oot=sig_oot, sigma_b=np.full(len(flux_pf), sig_oot), fit_rcond=1.0)
    w_med = float(np.median(white[okb]))
    # low-order phase curve (up to the 2nd harmonic) removed from the out-of-transit mean bins before the noise
    # factors (ellipsoidal / reflection signals)
    xb = time_pf[okb] / period
    r, fit_rcond = _phase_resid_householder(np.ascontiguousarray(xb, dtype=np.float64),
                                            np.ascontiguousarray(flux_pf[okb], dtype=np.float64))
    mad_r = _mad(r)
    beta_bin = mad_r / w_med
    Nb = max(2, int(round(duration / (period / 1000.0))))
    k = len(r) // Nb
    if k >= 8:
        beta_t14 = _mad(r[: k * Nb].reshape(k, Nb).mean(1)) / (mad_r / math.sqrt(Nb))
    else:
        beta_t14 = 1.0
    beta = min(max(1.0, beta_bin, beta_t14), 5.0)
    return dict(sigma_pt=sig_pt, beta_bin=beta_bin, beta_T14=beta_t14, beta=beta, sigma_oot=sig_oot,
                sigma_b=white * beta, fit_rcond=float(fit_rcond))


def sigma_components(time, flux, period, t0, duration):
    """The (q') production likelihood sigma components of a raw light curve (D15, plan decision 0.3 (q')).

    The mean-fold inputs are built the way the reference `mean_fold_inputs` builds them, but from the
    production fold rows (mean bins, plan 0.3 (j')): bin centre * period as the bin time, the bin's
    mean flux, the bin's point count, non-finite bins dropped. Returns the reference dict, with
    sigma_b one entry per KEPT bin.
    """
    period, t0, duration = float(period), float(t0), float(duration)
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    # Mean fold exactly as the reference `mean_fold_inputs` (np.nanmean per bin): the sigma estimators are
    # specified on it to 1e-12; the sequential-sum fold of `fold_production` differs from it at the 1e-11 level.
    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    edges = np.linspace(-0.5, 0.5, FOLD_N_BINS + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(phase, edges) - 1, 0, FOLD_N_BINS - 1)
    # One stable sort instead of a full-length boolean mask per bin (O(n log n) instead of O(1000 n), review
    # finding 7): a stable argsort keeps the input order inside each bin, so every slice is exactly the array
    # `flux[bin_idx == i]` and np.nanmean returns the identical bits.
    order = np.argsort(bin_idx, kind="stable")
    flux_sorted = flux[order]
    bounds = np.searchsorted(bin_idx[order], np.arange(FOLD_N_BINS + 1))
    t_out, f_out, cnt = [], [], []
    for i in range(FOLD_N_BINS):
        n_i = int(bounds[i + 1] - bounds[i])
        if n_i > 0:
            t_out.append(centres[i] * period)
            f_out.append(np.nanmean(flux_sorted[bounds[i]:bounds[i + 1]]))
            cnt.append(n_i)
    if not t_out:
        return None
    time_pf, flux_pf = np.array(t_out), np.array(f_out)
    counts = np.asarray(cnt, dtype=np.float64)
    g = np.isfinite(time_pf) & np.isfinite(flux_pf)
    return _sigma_components_from_fold(time, flux, period, t0, duration, time_pf[g], flux_pf[g], counts[g])


def likelihood_sigma(time, flux, period, t0, duration):
    """The (q') likelihood sigma of a raw light curve: (per-bin sigma_b array, scalar sigma_oot)."""
    c = sigma_components(time, flux, period, t0, duration)
    return None if c is None else (c["sigma_b"], c["sigma_oot"])


# ============================================================
# 8. Phase-fold helper
# ============================================================


def _phase_fold_bin(time, flux, period, t0, n_bins=1000):
    """Phase-fold and bin light curve, centered on transit."""
    # See docs/reviews/2026-07-02-opus48-cleanup-review.md item D5 and
    # tests/worker/test_phase_fold_golden.py for the bit-identical proof.
    phase = phase_fold(time, t0, period, fold_order="div_mod", center="mask")
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.digitize(phase, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    t_out, f_out = [], []
    for i in range(n_bins):
        mask = bin_idx == i
        if np.sum(mask) > 0:
            t_out.append(bin_centers[i] * period)  # time relative to t0
            f_out.append(np.nanmedian(flux[mask]))

    return np.array(t_out), np.array(f_out)


def renorm_flux(flux, flux_err, star_fluxratio):
    """Renormalize flux for a specific star's contribution."""
    renormed = (flux - (1 - star_fluxratio)) / star_fluxratio
    renormed_err = flux_err / star_fluxratio
    return renormed, renormed_err


# ============================================================
# 8b. Ephemeris refinement before FPP (FPP_EPHEM_REFINE)
# ============================================================
#
# Why (docs/reviews/2026-10-24-fpp-ephemeris-smearing-measurement.md): the production epoch is the BLS box START
# (worker/bls_fast.py: t0 = phase_bin_start * P + t0_ref, quantised to P/200; duration_refine is OFF fleet-wide) and P
# carries the BLS grid precision plus the 1e-4 d rounding of the candidate dict.  The FPP fold treats t0 as the transit
# CENTRE and P as exact, so the folded dip is shifted / smeared (45-89 % of the TP-fit dchi2 lost on the measured TOIs).
#
# refine_ephemeris is a bounded, deterministic local (P, t0) fit on the UNFOLDED light curve given to FPP.  The worker
# passes the STORED (rounded) row on both FPP paths (in-task and backfill), so the fold is a function of the DB row.
#   objective  dchi2(P, tc) = max over a fixed trapezoid shape grid of (sum m r)^2 / (sum m^2) / sigma^2, r = baseline - flux,
#              dips only (the Gaussian likelihood ratio of a TP-like dip against a flat curve, no fold binning; the same
#              statistic as scripts/fpp/ephem_smear/ephem_common.Unfolded).  The template m is the trapezoid AVERAGED over
#              the exposure box W = cadence (exact, analytic: the difference of the trapezoid's running integral at
#              dt +- W/2, divided by W) -- the same exposure model as the production FPP likelihood (FPP_EXPOSURE =
#              "cadence", box W = cadence_s / 86400, cadence_s = median point spacing rounded to 1 ms when not given).
#              At 30-min FFI cadence a 0.8-1 h transit is 1-2 exposures, so a point-sampled template would make the
#              objective a step function of tc.  sigma and the baseline are ONE robust value (1.4826 MAD, median) of the
#              points outside the searched window, so values are comparable between trials;
#   parameters (P, tc) with tc the transit nearest the middle of the data (decorrelates P and t0);
#   coverage   an epoch is "covered" when it has >= n_req points within +-T14/2 of the guess, n_req =
#              min(EPHEM_COVER_POINTS, max(1, floor(T14 / W))): 3 at 2-min cadence, but 1 when T14 spans < 2 exposures
#              (a 0.8 h transit at 30 min has 1-2 points per epoch).  The significance of the dip is decided by the total
#              dchi2 over all epochs (EPHEM_MIN_DCHI2), not by the per-epoch count;
#   search     P fixed at the input: coarse EPHEM_N_COARSE grid over tc +- dt_half, then EPHEM_N_ZOOMS nested EPHEM_N_ZOOM
#              grids, each 1/EPHEM_ZOOM of the previous half-range, centred on the running best.  With >= 2 covered
#              epochs the same nested search is also run over (P, tc) (EPHEM_N_COARSE^2, P +- dP_half) and the P-free
#              optimum is ACCEPTED only when it beats the P-fixed optimum by EPHEM_P_FREE_DCHI2 (period freedom test,
#              below); otherwise P stays the input and only tc moves.  No optimiser, no RNG, no clock: fixed loops,
#              first-maximum tie rule in row-major (P outer, t inner) order;
#   bounds     dt_half = min(EPHEM_T0_HALF * T14, (EPHEM_REACH_MAX_P * P - widest half shape - W/2) / 1.25): every trial
#              window (zooms reach at most 1.25 x the coarse range) stays within +-EPHEM_REACH_MAX_P * P of the guess at
#              the reference epoch, i.e. never reaches a secondary eclipse at phase 0.5 even for T14 = 0.15 P (the BLS
#              maximum).  dP_half = max(EPHEM_DRIFT_HALF * T14 / n_max, step_BLS(P) + EPHEM_P_ROUND), capped at
#              EPHEM_P_FRAC_MAX * P.  n_max = largest |epoch number| of the data relative to the reference transit, so the
#              first term is the P offset whose accumulated drift at the baseline ends is EPHEM_DRIFT_HALF * T14: a larger
#              P error would have smeared the BLS fold by more than the transit itself and BLS could not have detected it
#              there.  The floor covers the BLS period resolution independently of the drift argument: step_BLS(P) =
#              max(2e-5, 4e-6 P) is the pass-2 grid step of bls_two_pass (worker/bls_fast.py:441-480, n_fine = 10000 over
#              +-max(0.099 d, 2 % P)), EPHEM_P_ROUND = 1e-4 d is the full rounding quantum of the candidate dict (worker.py
#              round(P, 4)); the cap keeps the search inside the BLS pass-2 window (+-2 % P), away from other peaks;
#   period freedom  freeing P adds one parameter to the tc-only fit, so under "the input P is right" the gain
#              dchi2(P free) - dchi2(P fixed) is a likelihood-ratio statistic of ~1 dof, enlarged by the look-elsewhere of
#              the P grid.  EPHEM_P_FREE_DCHI2 = 9 is the 3-sigma 1-dof threshold (p = 2.7e-3); measured on noise-only
#              periods (tests/fpp/ephem, two-transit and many-transit synthetic cases with the guess AT the true P) the
#              null gain stays below it, while a real P error that moves the baseline-end transits by a sizeable fraction
#              of T14 gains far more.  A refused P change costs nothing measurable by definition (its gain is within noise);
#   refusal    status "edge" when the coarse optimum of the accepted search lies within EPHEM_EDGE_CELLS cells of a bound
#              (no expansion: the bound is the claim); "failed" for unusable input, too few points, best dip below
#              EPHEM_MIN_DCHI2, or a fold displaced by more than EPHEM_MAX_FREE_SHIFT * T14 anywhere in the data for a gain
#              over the guess below EPHEM_P_FREE_DCHI2 (a large move must pay for itself).  The caller then keeps
#              (P, t0 centre guess) and caps the verdict.
# The coarse grids contain the centre guess exactly (offset 0), so dchi2_refined >= dchi2_initial by construction.
# Ports (Rust / Swift) must mirror: the constants, the floor-mod phase x - P*floor(x/P), the running-integral template
# (_ephem_cum, branch order), the loop order of the sums, the shape order (T14 outer, ingress inner), both first-maximum
# tie rules, the P-fixed-then-P-free order and the acceptance test (>=).

T0_CONVENTIONS = ("box_start", "centre")
EPHEM_T14_FACTORS = (0.5, 0.65, 0.8, 1.0, 1.2, 1.45, 1.75, 2.1, 2.5)   # x input duration (BLS durations are coarse, 0.01 P steps)
EPHEM_INGRESS = (0.08, 0.2, 0.4)                                        # ingress fraction T12 / T14 (<= 0.5: template stays a trapezoid)
EPHEM_T0_HALF = 1.5            # t0 half-range, x duration
EPHEM_REACH_MAX_P = 0.25       # every trial window stays within +- this fraction of P of the guess (reference epoch)
EPHEM_DRIFT_HALF = 1.5         # half-range of the accumulated drift at the baseline ends, x duration
EPHEM_P_ROUND = 1e-4           # rounding quantum of P in the candidate dict (d)
EPHEM_P_FRAC_MAX = 0.02        # cap of the P half-range, fraction of P (BLS pass-2 window)
EPHEM_N_COARSE = 41
EPHEM_N_ZOOM = 31
EPHEM_N_ZOOMS = 3
EPHEM_ZOOM = 5.0
EPHEM_EDGE_CELLS = 2
EPHEM_MIN_DCHI2 = 25.0         # best dip below 5 sigma: not a signal to refine on (the max over ~1e3 trials of pure noise reaches ~10-15)
EPHEM_P_FREE_DCHI2 = 9.0       # gain of freeing P over P fixed needed to accept a period change (3 sigma, 1 dof)
EPHEM_MAX_FREE_SHIFT = 0.25    # x duration: a larger fold displacement needs a gain >= EPHEM_P_FREE_DCHI2 over the guess
EPHEM_COVER_POINTS = 3         # points within +-T14/2 for an epoch to count as covered (fewer when T14 < 3 exposures)
EPHEM_MIN_POINTS = 100
EPHEM_MIN_WINDOW_POINTS = 5    # points within the widest trial window (all epochs) for a trial to be evaluated


@numba.njit(cache=True)
def _ephem_cum(x, hw, L):
    """Running integral from -inf to x of the unit trapezoid of half-width hw and ingress length L (0 < L <= hw)."""
    y = -x if x > 0.0 else x
    if y <= -hw:
        c = 0.0
    elif y <= L - hw:
        u = y + hw
        c = u * u / (2.0 * L)
    else:
        c = 0.5 * L + (y + hw - L)
    if x > 0.0:
        return (2.0 * hw - L) - c
    return c


@numba.njit(cache=True)
def _ephem_dchi2_nb(t, r, P, tc, t14, ing, W, inv_s2, min_pts, smr, smm):
    """Best trapezoid-dip dchi2 at one (P, tc) on the unfolded data (see section 8b); returns (dchi2, shape index, n in window).

    W > 0: the template is the trapezoid averaged over the exposure box [dt - W/2, dt + W/2]; W <= 0: point-sampled."""
    ns = t14.shape[0]
    ng = ing.shape[0]
    for s in range(ns * ng):
        smr[s] = 0.0
        smm[s] = 0.0
    hW = 0.5 * W if W > 0.0 else 0.0
    half = 0.5 * t14[ns - 1] + hW
    npts = 0
    for i in range(t.shape[0]):
        x = t[i] - tc + 0.5 * P
        dt = x - P * math.floor(x / P) - 0.5 * P
        a = abs(dt)
        if a >= half:
            continue
        npts += 1
        for k in range(ns):
            hw = 0.5 * t14[k]
            if a >= hw + hW:
                continue
            for g in range(ng):
                if W > 0.0:
                    L = ing[g] * t14[k]
                    m = (_ephem_cum(a + hW, hw, L) - _ephem_cum(a - hW, hw, L)) / W
                else:
                    m = (hw - a) / (ing[g] * t14[k])
                    if m > 1.0:
                        m = 1.0
                smr[k * ng + g] += m * r[i]
                smm[k * ng + g] += m * m
    best = 0.0
    bi = -1
    if npts >= min_pts:
        for s in range(ns * ng):
            if smm[s] > 0.0 and smr[s] > 0.0:
                d = smr[s] * smr[s] / smm[s] * inv_s2
                if d > best:
                    best = d
                    bi = s
    return best, bi, npts


@numba.njit(cache=True)
def _ephem_grid_nb(t, r, P0, stepP, nP, tc0, stept, nt, t14, ing, W, inv_s2, min_pts):
    """Grid P0 + (i - (nP-1)/2) stepP, tc0 + (j - (nt-1)/2) stept; returns (best i, best j, best dchi2, shape, dchi2 at centre, n)."""
    smr = np.empty(t14.shape[0] * ing.shape[0])
    smm = np.empty(t14.shape[0] * ing.shape[0])
    ci = (nP - 1) // 2
    cj = (nt - 1) // 2
    bi = ci
    bj = cj
    best = -1.0
    bshape = -1
    centre = 0.0
    bn = 0
    for i in range(nP):
        P = P0 + (i - ci) * stepP
        for j in range(nt):
            tc = tc0 + (j - cj) * stept
            d, sh, n = _ephem_dchi2_nb(t, r, P, tc, t14, ing, W, inv_s2, min_pts, smr, smm)
            if i == ci and j == cj:
                centre = d
            if d > best:
                best = d
                bi = i
                bj = j
                bshape = sh
                bn = n
    return bi, bj, best, bshape, centre, bn


def _ephem_median(x):
    """Median of a float64 array: ascending stable sort, middle element or mean of the two middle ones (port spec)."""
    s = np.sort(np.asarray(x, dtype=np.float64), kind="stable")
    n = s.size
    h = n // 2
    return float(s[h]) if n % 2 == 1 else float(0.5 * (s[h - 1] + s[h]))


def _ephem_search(ts, rs, P0, dP_half, tref0, dt_half, t14, ing, W, inv_s2, p_free):
    """Coarse grid + nested zooms around (P0, tref0); P varies only when p_free.  Returns the optimum and its edge flags."""
    nP = EPHEM_N_COARSE if p_free else 1
    nt = EPHEM_N_COARSE
    stepP = 2.0 * dP_half / (nP - 1) if p_free else 0.0
    stept = 2.0 * dt_half / (nt - 1)
    bi, bj, best, bshape, centre, _bn = _ephem_grid_nb(ts, rs, P0, stepP, nP, tref0, stept, nt, t14, ing, W, inv_s2,
                                                       EPHEM_MIN_WINDOW_POINTS)
    ci, cj = (nP - 1) // 2, (nt - 1) // 2
    cP = P0 + (bi - ci) * stepP
    ct = tref0 + (bj - cj) * stept
    edge_P = bool(p_free and (bi < EPHEM_EDGE_CELLS or bi > nP - 1 - EPHEM_EDGE_CELLS))
    edge_t = bool(bj < EPHEM_EDGE_CELLS or bj > nt - 1 - EPHEM_EDGE_CELLS)
    hP, ht = dP_half, dt_half
    for _z in range(EPHEM_N_ZOOMS):
        hP /= EPHEM_ZOOM
        ht /= EPHEM_ZOOM
        nPz = EPHEM_N_ZOOM if p_free else 1
        sPz = 2.0 * hP / (nPz - 1) if p_free else 0.0
        stz = 2.0 * ht / (EPHEM_N_ZOOM - 1)
        zi, zj, zb, zs, _zc, _zn = _ephem_grid_nb(ts, rs, cP, sPz, nPz, ct, stz, EPHEM_N_ZOOM, t14, ing, W, inv_s2,
                                                  EPHEM_MIN_WINDOW_POINTS)
        if zb > best:          # the zoom grid contains its centre (offset 0): zb >= best up to the identical value
            best, bshape = zb, zs
            cP = cP + (zi - (nPz - 1) // 2) * sPz
            ct = ct + (zj - (EPHEM_N_ZOOM - 1) // 2) * stz
    return dict(P=float(cP), t=float(ct), best=float(best), shape=int(bshape), centre=float(centre),
                edge_P=edge_P, edge_t=edge_t)


def refine_ephemeris(time, flux, period, t0_centre_guess, duration, cadence_s=None):
    """Bounded deterministic local (P, transit-centre) refinement on the unfolded light curve (section 8b).

    Pure function of its inputs (no RNG, no clock, no global state).  `t0_centre_guess` must already be a CENTRE estimate
    (a box-start caller adds duration / 2 first).  `cadence_s` sets the exposure box of the template (None = median point
    spacing, rounded to 1 ms, as calc_fpp_native does).  Returns a dict:
      status       "ok" | "edge" (coarse optimum of the accepted search within EPHEM_EDGE_CELLS cells of a bound) |
                   "failed" (unusable input, too few points, no covered epoch, best dip < EPHEM_MIN_DCHI2, or a large
                   displacement that does not pay for itself);
      P, t0        refined period and transit centre (t0 = the transit nearest the middle of the data) when status ok,
                   otherwise the inputs (period, t0_centre_guess) unchanged;
      P_in, t0_in, t_ref_in   inputs and the centre guess propagated to the reference epoch;
      cadence_s, n_req        exposure used by the template and the per-epoch coverage requirement;
      n_orbits     data span / P; n_transits (covered epochs); p_searched (P-free search run: >= 2 covered epochs);
      p_free_accepted / delta_chi2_p (dchi2 P free - dchi2 P fixed) / dchi2_p_fixed / dchi2_p_free; p_fixed (final P is
                   the input);
      shift_h      refined minus guessed centre at the reference epoch (hours); dP; max_offset_h (largest |shift| over the
                   data, i.e. including the P drift at the baseline ends);
      dchi2_initial / dchi2_refined / delta_chi2, sigma, baseline, t14_factor / ingress of the best shape, edge flags, ranges.
    """
    P_in = float(period)
    t0_in = float(t0_centre_guess)
    D = float(duration)
    out = dict(status="failed", reason=None, P=P_in, t0=t0_in, P_in=P_in, t0_in=t0_in, t_ref_in=t0_in,
               cadence_s=None, n_req=None, n_orbits=None, n_transits=0, p_searched=False, p_free_accepted=False,
               delta_chi2_p=None, dchi2_p_fixed=None, dchi2_p_free=None, p_fixed=None, shift_h=0.0, dP=0.0,
               max_offset_h=0.0, dchi2_initial=None, dchi2_refined=None, delta_chi2=None, sigma=None, baseline=None,
               t14_factor=None, ingress=None, edge_P=False, edge_t0=False, dP_half=None, dt_half=None)
    if not (math.isfinite(P_in) and P_in > 0 and math.isfinite(t0_in) and math.isfinite(D) and D > 0):
        out["reason"] = "invalid_ephemeris"
        return out
    t = np.asarray(time, dtype=np.float64).ravel()
    f = np.asarray(flux, dtype=np.float64).ravel()
    ok = np.isfinite(t) & np.isfinite(f)
    t = np.ascontiguousarray(t[ok])
    f = np.ascontiguousarray(f[ok])
    if t.size < EPHEM_MIN_POINTS:
        out["reason"] = "too_few_points"
        return out
    if cadence_s is None:
        cadence_s = float(round(float(np.median(np.diff(np.sort(t)))) * 86400.0, 3))
    cadence_s = float(cadence_s)
    out["cadence_s"] = cadence_s
    if not (math.isfinite(cadence_s) and cadence_s > 0.0):
        out["reason"] = "cadence_unusable"
        return out
    W = cadence_s / 86400.0
    hW = 0.5 * W
    tmin, tmax = float(np.min(t)), float(np.max(t))
    out["n_orbits"] = (tmax - tmin) / P_in
    # reference epoch: the transit of the guess nearest the middle of the data
    n_ref = math.floor((0.5 * (tmin + tmax) - t0_in) / P_in + 0.5)
    tref0 = t0_in + n_ref * P_in
    out["t_ref_in"] = tref0
    ep = np.floor((t - tref0) / P_in + 0.5)                     # epoch number of every point (relative to tref0)
    x = t - tref0 + 0.5 * P_in
    dt0 = x - P_in * np.floor(x / P_in) - 0.5 * P_in            # time from the nearest guessed transit
    # covered epochs: >= n_req points within +-D/2 of the guess, n_req cadence-aware (a 0.8 h transit at 30 min has 1-2)
    n_req = min(EPHEM_COVER_POINTS, max(1, int(math.floor(D / W))))
    out["n_req"] = n_req
    ins = np.abs(dt0) < 0.5 * D
    _ep_in, cnt = np.unique(ep[ins], return_counts=True)
    n_tr = int(np.sum(cnt >= n_req))
    out["n_transits"] = n_tr
    if n_tr < 1:
        out["reason"] = "no_transit_covered"
        return out
    n_max = float(np.max(np.abs(ep)))
    p_search = n_tr >= 2 and n_max >= 1.0
    out["p_searched"] = bool(p_search)
    t14 = np.array([k * D for k in EPHEM_T14_FACTORS], dtype=np.float64)
    ing = np.array(EPHEM_INGRESS, dtype=np.float64)
    half_shape = 0.5 * t14[-1] + hW
    dt_half = min(EPHEM_T0_HALF * D, (EPHEM_REACH_MAX_P * P_in - half_shape) / 1.25)
    if not dt_half > 0.0:
        out["reason"] = "duration_too_long_for_window"
        return out
    if p_search:
        step_bls = max(2e-5, 4e-6 * P_in)
        dP_half = max(EPHEM_DRIFT_HALF * D / n_max, step_bls + EPHEM_P_ROUND)
        dP_half = min(dP_half, EPHEM_P_FRAC_MAX * P_in)
    else:
        dP_half = 0.0
    out["dP_half"], out["dt_half"] = dP_half, dt_half
    # every point any trial window can reach (zooms stay within 1.25 x the coarse range; 1.5 x is a strict superset)
    reach = half_shape + 1.5 * (dt_half + np.abs(ep) * dP_half)
    sub = np.abs(dt0) < reach
    oot = ~(np.abs(dt0) < half_shape + dt_half + np.abs(ep) * dP_half)
    ref = f[oot] if int(np.sum(oot)) >= EPHEM_MIN_POINTS else f
    base = _ephem_median(ref)
    sigma = 1.4826 * _ephem_median(np.abs(ref - base))
    out["baseline"], out["sigma"] = base, sigma
    if not (sigma > 0 and math.isfinite(sigma)):
        out["reason"] = "sigma_unusable"
        return out
    ts = np.ascontiguousarray(t[sub])
    rs = np.ascontiguousarray(base - f[sub])
    inv_s2 = 1.0 / (sigma * sigma)
    fixed = _ephem_search(ts, rs, P_in, 0.0, tref0, dt_half, t14, ing, W, inv_s2, False)
    out["dchi2_p_fixed"] = fixed["best"]
    best_s = fixed
    if p_search:
        free = _ephem_search(ts, rs, P_in, dP_half, tref0, dt_half, t14, ing, W, inv_s2, True)
        gain = free["best"] - fixed["best"]
        out["dchi2_p_free"], out["delta_chi2_p"] = free["best"], float(gain)
        if gain >= EPHEM_P_FREE_DCHI2:
            best_s = free
            out["p_free_accepted"] = True
    p_fixed = best_s is fixed
    out["p_fixed"] = bool(p_fixed)
    centre = fixed["centre"]                  # dchi2 at (P_in, tref0): the same trial in both searches
    best = best_s["best"]
    out["dchi2_initial"] = float(centre)
    out["dchi2_refined"] = float(best)
    out["delta_chi2"] = float(best - centre)
    if best_s["shape"] >= 0:
        out["t14_factor"] = float(EPHEM_T14_FACTORS[best_s["shape"] // len(EPHEM_INGRESS)])
        out["ingress"] = float(EPHEM_INGRESS[best_s["shape"] % len(EPHEM_INGRESS)])
    out["edge_P"], out["edge_t0"] = best_s["edge_P"], best_s["edge_t"]
    if best < EPHEM_MIN_DCHI2:
        out["reason"] = "no_significant_dip"
        return out
    if best_s["edge_P"] or best_s["edge_t"]:
        out["status"] = "edge"
        out["reason"] = "optimum_at_search_edge"
        return out
    cP, ct = best_s["P"], best_s["t"]
    dP = cP - P_in
    shift = ct - tref0
    max_off = max(abs(shift + float(np.min(ep)) * dP), abs(shift + float(np.max(ep)) * dP))
    if max_off > EPHEM_MAX_FREE_SHIFT * D and best - centre < EPHEM_P_FREE_DCHI2:
        out["max_offset_h"] = float(max_off * 24.0)
        out["reason"] = "insignificant_large_shift"
        return out
    out.update(status="ok", P=float(cP), t0=float(ct), dP=float(dP), shift_h=float(shift * 24.0),
               max_offset_h=float(max_off * 24.0))
    return out


# ============================================================
# 9. Main FPP calculation
# ============================================================

# ============================================================
# 9. Deterministic per-candidate seed
# ============================================================
#
# Normative scheme shared with the Rust (fpp/rng.rs::tic_seed) and Swift
# (FPPRNG.ticSeed) ports — see exonit-rust-worker-fpp/parity/fpp/README.md
# "Deterministic per-TIC seed" and fixtures/fpp/rng_golden.json. Any change
# here must be mirrored in both ports or FPP/NFPP will silently diverge
# between the Python and native workers for the same candidate.


def _fnv1a64(data: bytes) -> int:
    """FNV-1a 64-bit hash (offset basis 0xcbf29ce484222325, prime 0x100000001b3)."""
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def tic_seed(tic_id, period, t0) -> int:
    """Deterministic 32-bit seed for one (tic_id, period, t0) candidate.

    key = f"{tic_id}:{period:.10e}:{t0:.10e}" (ASCII), seed = fnv1a64(key) & 0xFFFFFFFF.
    Must match Rust `tic_seed()` and Swift `FPPRNG.ticSeed()` byte-for-byte.
    """
    key = f"{tic_id}:{period:.10e}:{t0:.10e}"
    return _fnv1a64(key.encode("ascii")) & 0xFFFFFFFF


class FPPAborted(Exception):
    """Raised when FPP computation is aborted by user."""
    pass


# ============================================================
# 9b. MC-noise policy: sequential refinement with a standard-error rule (plan decision (g), step 7)
# ============================================================
#
# Estimator.  Scenario s has the evidence Z_s = mean_j w_j over its N iid draws (w_j = exp(lnL_j + ln prior_j), zero for
# rejected draws).  The relative standard error of Z_s is
#       r_s = sqrt(var(w) / N) / Z_s = sqrt(1/ESS_s - 1/N),    ESS_s = (sum w)^2 / sum w^2,
# so one number per scenario carries both the error and the weight degeneracy.  The scenario probabilities are
# p_s = Z_s / sum Z and any gate quantity is a fraction f = sum_{s in U} p_s (FPP: U = every scenario but TP / PTP / DTP;
# NFPP: U = the nearby-star scenarios).  The scenarios are independent (disjoint draws), so the delta method gives
#       var(f) = sum_s (1[s in U] - f)^2 p_s^2 r_s^2.
# The SE of a single run can be tiny exactly when the estimate is wrong (a heavy-tailed scenario whose dominant draw was
# never sampled).  Three guards, none of which trusts the nominal SE alone:
#   1. a scenario with ESS < MC_ESS_MIN is not trusted to its nominal SE: its lnZ gets the extreme-value uncertainty of the
#      best draws (+-3 * 1.2825 beta, beta from the top-10 order statistics of the SAME run) and the stage is unresolved if
#      a gate flips at the corners of that box (a scenario with no usable scale is unresolved if its share is >= MC_HEAVY_P_MIN);
#   2. a stage after the first must agree with the previous one within 3 combined SE (if not, the earlier SE was wrong
#      and the candidate stays unresolved until the cap);
#   3. validated additionally needs FPP + 3 sigma < 0.015 and NFPP + 3 sigma < 0.001.
# Stages: N, 10 N, 100 N, capped at MC_N_CAP = 1e6 (decision C of the owner, 2026-10-24: ladder 1e5 -> 1e6, the 1e7 stage is dropped; an
# unresolved candidate at the cap stays `borderline`).  A futility stop after stage 0 (MC_FUTILE_ESS_DOM, below) skips stage 1 when
# more draws cannot resolve the candidate.  Stage 0 uses the candidate's own seed (identical to the
# pre-step-7 single pass); stage k >= 1 uses stage_seed().  Calls with N < MC_MIN_REFINE_N (unit tests, diagnostics) are a
# single pass.  Parity mode (FPP_ADAPTIVE_N = off) does none of this.

MC_SIGMA = 3.0                    # decidability margin of decision (g) / 2.2
MC_GATES_FPP = (FPP_VALIDATED, FPP_CANDIDATE)
MC_GATE_NFPP = NFPP_VALIDATED
MC_STAGE_FACTORS = (1, 10, 100)
MC_N_CAP = 1_000_000              # decision C: cap of the draws per scenario of the LAST stage (ladder 1e5 -> 1e6)
MC_MIN_REFINE_N = 100_000
MC_ESS_MIN = 10.0                 # below this ESS the nominal SE is replaced by the extreme-value box.  Measured on 1268 seed pairs: the
                                  # nominal |z|<3 holds for 43 % (ESS < 1.5), 83 % (1.5-3), 94 % (3-10), 99-100 % (>= 10); the PAIR statistic of the
                                  # tail estimator (hypot of both runs' beta) stays below 2.0 in every ESS < 100 bin
                                  # (scripts/fpp/analyze_step7_calibration.py, report).  A box of +-3 sd around ONE run has only ~6 % margin
                                  # against that worst pair (2.0 x sqrt 2 = 2.83 sd) and misses the other run's evidence in 2.3 % of the
                                  # 605 low-ESS seed pairs (one-sided, report section 3); MC_TAIL_SIGMAS was NOT changed (see report).
MC_HEAVY_P_MIN = 1e-4             # scenario with no usable tail scale (< 3 finite draws): unresolved if it carries >= this share
MC_TAIL_K_TOP = 10                # order statistics used for the extreme-value scale of a degenerate scenario
MC_TAIL_SIGMAS = 3.0              # (decision C: unchanged, not re-measured) half-width of the lnZ box of a degenerate scenario, in extreme-value sd (1.2825 beta)
MC_EV_SD = math.pi / math.sqrt(6.0)
MC_ABS_TOL = 1e-9                 # absolute slack of the stage-agreement test (exact zeros)
# Futility stop (decision C).  After an unresolved stage 0 the scenario with the largest probability (the dominant one) decides
# whether 10x more draws can help: if its ESS is below MC_FUTILE_ESS_DOM its evidence is (nearly) the single best draw of
# the run, every other scenario is compared with that one lucky draw, and 10x more draws only move the best draw by ~ beta ln 10
# (they do not close the gap).  Calibrated on the 116 measured runs (scripts/fpp/calibrate_step7_futility.py): the 10 runs that
# resolved at stage 1 have a dominant ESS of 32.6 .. 227 at stage 0; MC_FUTILE_ESS_DOM = 25 stops 86 of the 103 runs that did not
# resolve and none of the 10 that did (smallest resolved dominant ESS / constant = 1.30).  A function of the stage-0 summary only
# (no clock, no host), so the result stays bit-reproducible.
MC_FUTILE_ESS_DOM = 25.0


def stage_seed(tic_id, period, t0, stage, seed0=None) -> int:
    """Deterministic 32-bit seed of refinement stage `stage` >= 1 of one candidate (plan decision (g)).

    key = f"{tic_id}:{period:.10e}:{t0:.10e}:stage{stage}:{seed0}" (ASCII), seed = fnv1a64(key) & 0xFFFFFFFF, where seed0 is
    the candidate's stage-0 seed (tic_seed(); the certification harness shifts it to get independent replicates, so the
    stage seeds of two replicates differ too).  Same hash as tic_seed(); ports must mirror it byte for byte.
    """
    if int(stage) < 1:
        raise ValueError("stage_seed is for refinement stages >= 1; stage 0 uses tic_seed()")
    if seed0 is None:
        seed0 = tic_seed(tic_id, period, t0)
    key = f"{tic_id}:{period:.10e}:{t0:.10e}:stage{int(stage)}:{int(seed0)}"
    return _fnv1a64(key.encode("ascii")) & 0xFFFFFFFF


def _mc_stage_ladder(N, refine, n_cap=None):
    """Draws per scenario of each stage: [N] (no refinement) or N, 10 N, 100 N up to the cap."""
    N = int(N)
    cap = MC_N_CAP if n_cap is None else int(n_cap)
    if not refine or N < MC_MIN_REFINE_N:
        return [N]
    ladder = [N]
    for f in MC_STAGE_FACTORS[1:]:
        if N * f <= cap:
            ladder.append(N * f)
    return ladder


def _delta_se(probs, in_set, rel_var):
    """Delta-method SE of f = sum_{s in U} p_s from the per-scenario relative variances of Z_s."""
    f = float(np.sum(probs[in_set]))
    ind = in_set.astype(np.float64)
    return float(np.sqrt(np.sum(((ind - f) ** 2) * probs ** 2 * rel_var)))


def _mc_heavy_scenarios(names, probs, ess):
    """Scenarios whose draw weights are too degenerate to trust the nominal SE.

    ESS < MC_ESS_MIN and a probability share >= MC_HEAVY_P_MIN.  The share floor keeps a hopeless but irrelevant
    scenario (p ~ 1e-12) from blocking every candidate; it is deliberately low because the share of a degenerate scenario
    is itself underestimated when its dominant draw was not sampled.
    """
    return [n for n, p, e in zip(names, probs, ess) if (not e >= MC_ESS_MIN) and p >= MC_HEAVY_P_MIN]   # NaN ESS (no record) counts


def _mc_summarise(lnZ, nearby_scen, diag, N):
    """Probabilities, FPP / NFPP and their MC standard errors from one pass.  None if every lnZ is non-finite."""
    all_scen = dict(lnZ, **nearby_scen)
    names = list(all_scen)
    all_lnZ = np.array([x if np.isfinite(x) else -np.inf for x in all_scen.values()])
    if not np.any(all_lnZ > -np.inf):
        return None
    max_lnZ = np.max(all_lnZ[all_lnZ > -np.inf])
    probs = np.exp(all_lnZ - max_lnZ)
    total = np.sum(probs)
    if total <= 0:
        return None
    probs /= total
    target_probs = dict(zip(lnZ, probs[:len(lnZ)]))
    fpp = 1.0 - target_probs.get('TP', 0.0) - target_probs.get('PTP', 0.0) - target_probs.get('DTP', 0.0)
    nfpp = float(np.sum(probs[len(lnZ):]))
    fpp = float(np.clip(fpp, 0, 1))
    nfpp = float(np.clip(nfpp, 0, 1))

    ess = np.array([float(diag[n]["ess"]) if n in diag else (np.nan if np.isfinite(all_scen[n]) else 0.0) for n in names])
    n_draws = np.array([float(diag[n]["n"]) if n in diag else float(N) for n in names])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_var = np.where(ess > 0.0, np.maximum(1.0 / ess - 1.0 / n_draws, 0.0), 0.0)     # NaN (unknown) compares False -> 0
    rel_var = np.where(probs > 0.0, rel_var, 0.0)
    false_set = np.array([not (n in ("TP", "PTP", "DTP")) for n in names])
    nearby_set = np.array([n not in lnZ for n in names])
    fpp_se = _delta_se(probs, false_set, rel_var)
    nfpp_se = _delta_se(probs, nearby_set, rel_var)
    # degenerate scenarios (ESS < MC_ESS_MIN, or ESS unknown): their Z is only known to within the extreme-value spread of
    # the best draws.  Known scale -> lnZ box +-MC_TAIL_SIGMAS sd, and the gates are re-evaluated at the box corners;
    # unknown scale -> unresolved if the scenario carries >= MC_HEAVY_P_MIN of the probability.
    beta = np.array([float(diag[n].get("tail_scale", np.nan)) if n in diag else np.nan for n in names])
    finite_lnz = np.isfinite(all_lnZ)
    low = finite_lnz & ~(ess >= MC_ESS_MIN)
    known = low & np.isfinite(beta)
    heavy = _mc_heavy_scenarios([n for n, k in zip(names, low & ~known) if k], probs[low & ~known], ess[low & ~known])
    half = np.where(known, MC_TAIL_SIGMAS * MC_EV_SD * np.where(known, beta, 0.0), 0.0)

    def _fractions(lnz):
        m = np.max(lnz[np.isfinite(lnz)])
        q = np.exp(lnz - m)
        q = q / np.sum(q)
        return float(np.sum(q[false_set])), float(np.sum(q[nearby_set]))

    up = np.where(false_set, half, -half)                     # shifts that raise FPP: false up, good down
    up_n = np.where(nearby_set, half, -half)                  # shifts that raise NFPP: nearby up, the rest down
    fpp_hi, _ = _fractions(all_lnZ + up)
    fpp_lo, _ = _fractions(all_lnZ - up)
    _, nfpp_hi = _fractions(all_lnZ + up_n)
    _, nfpp_lo = _fractions(all_lnZ - up_n)
    live = probs > 0.0
    ess_live = np.where(live & ~np.isnan(ess), ess, np.inf)
    i_min = int(np.argmin(ess_live)) if np.any(np.isfinite(ess_live)) else 0
    return {"names": names, "probs": probs, "fpp": fpp, "nfpp": nfpp, "fpp_se": fpp_se, "nfpp_se": nfpp_se,
            "heavy": heavy, "min_ess": float(ess[i_min]) if np.any(np.isfinite(ess_live)) else 0.0, "min_ess_scenario": names[i_min],
            "N": int(N), "ess": ess, "rel_var": rel_var, "tail_box_fpp": (fpp_lo, fpp_hi), "tail_box_nfpp": (nfpp_lo, nfpp_hi),
            "n_low_ess": int(np.sum(low)), "tail_scale": beta}


def _mc_assess(summ, prev):
    """Is the decision of this stage resolved?  Returns {"resolved": bool, "reasons": [...]}."""
    reasons = []
    for g in MC_GATES_FPP:
        if abs(summ["fpp"] - g) < MC_SIGMA * summ["fpp_se"]:
            reasons.append(f"fpp_gate_{g:g}")
    if abs(summ["nfpp"] - MC_GATE_NFPP) < MC_SIGMA * summ["nfpp_se"]:
        reasons.append(f"nfpp_gate_{MC_GATE_NFPP:g}")
    for n in summ["heavy"]:
        reasons.append(f"heavy_tail:{n}")
    lo, hi = summ["tail_box_fpp"]
    for g in MC_GATES_FPP:
        if lo < g < hi:
            reasons.append(f"tail_box_fpp_{g:g}")
    lo, hi = summ["tail_box_nfpp"]
    if lo < MC_GATE_NFPP < hi:
        reasons.append(f"tail_box_nfpp_{MC_GATE_NFPP:g}")
    if prev is not None:
        for key, se in (("fpp", "fpp_se"), ("nfpp", "nfpp_se")):
            tol = MC_SIGMA * float(np.hypot(summ[se], prev[se])) + MC_ABS_TOL
            if abs(summ[key] - prev[key]) > tol:
                reasons.append(f"stage_disagreement_{key}")
    return {"resolved": not reasons, "reasons": reasons}


def _mc_futile(summ):
    """Futility rule of decision C on a stage summary: True when the scenario of largest probability has ESS < MC_FUTILE_ESS_DOM.

    The dominant scenario is the first of maximal probability (names order); an unknown ESS (NaN, no weight record) is futile,
    i.e. fail-closed like the heavy-tail rule.  Uses only the draws of the stage itself.
    """
    return bool(not _mc_dominant_ess(summ)[1] >= MC_FUTILE_ESS_DOM)


def _mc_dominant_ess(summ):
    """(index, ESS) of the dominant scenario of a stage summary: the first of maximal probability (names order).

    The ESS is the one `_mc_summarise` stored (NaN = unknown).  Shared by the futility rule and the `fpp_unstable` flag.
    """
    probs = np.asarray(summ["probs"], dtype=np.float64)
    ess = np.asarray(summ["ess"], dtype=np.float64)
    i = int(np.argmax(probs))
    return i, float(ess[i])


def _mc_dominant(summ):
    """(name, ESS) of the dominant scenario (see `_mc_dominant_ess`)."""
    i, ess = _mc_dominant_ess(summ)
    return summ["names"][i], ess


def mc_is_unstable(summ):
    """`fpp_unstable` of the REPORTED stage: dominant-scenario ESS < MC_FUTILE_ESS_DOM (strict; NaN / unknown ESS is unstable).

    The same predicate as the futility stop, applied to the reported stage, so it also covers a final stage (cap, or a
    resolved-looking stage) whose dominant ESS is still below the threshold.  Pure function of the stage summary.
    """
    return bool(not _mc_dominant_ess(summ)[1] >= MC_FUTILE_ESS_DOM)


def _mc_stage_record(stage, N, seed, summ, assess):
    # the five scenarios with the largest probability share, with their ESS and relative SE of Z (what the SE is made of)
    order = np.argsort(-summ["probs"], kind="stable")[:5]
    top = [{"scenario": summ["names"][i], "p": float(summ["probs"][i]), "ess": float(summ["ess"][i]),
            "rel_se": float(np.sqrt(summ["rel_var"][i]))} for i in order]
    return {"stage": int(stage), "n": int(N), "seed": int(seed), "fpp": summ["fpp"], "nfpp": summ["nfpp"],
            "fpp_mc_se": summ["fpp_se"], "nfpp_mc_se": summ["nfpp_se"], "min_ess": summ["min_ess"],
            "min_ess_scenario": summ["min_ess_scenario"], "heavy_scenarios": list(summ["heavy"]), "n_low_ess": summ["n_low_ess"],
            "tail_box_fpp": [float(x) for x in summ["tail_box_fpp"]], "tail_box_nfpp": [float(x) for x in summ["tail_box_nfpp"]],
            "resolved": bool(assess["resolved"]), "reasons": list(assess["reasons"]), "top_scenarios": top}


def _mc_run_stages(pass_fn, ladder, seed_for_stage, futility=True):
    """Run the stage ladder: stop at the first stage whose gate decisions are resolved, or at the last stage.

    futility: after an unresolved stage 0, stop (candidate stays unresolved = borderline) when `_mc_futile` says more draws
    cannot resolve it; the assessment then carries the reason "futility_stop".

    pass_fn(N, seed) -> (lnZ, nearby_lnZ, nearby_scen, diag).  Returns (pass output, summary, assessment, stage log)
    of the LAST stage run, or None if a pass has no finite evidence.  The reported result is that stage alone.
    """
    log = []
    prev = None
    for idx, N_k in enumerate(ladder):
        seed_k = seed_for_stage(idx)
        out = pass_fn(N_k, seed_k)
        summ = _mc_summarise(out[0], out[2], out[3], N_k)
        if summ is None:
            return None
        assess = _mc_assess(summ, prev)
        futile = bool(futility and idx == 0 and not assess["resolved"] and idx + 1 < len(ladder) and _mc_futile(summ))
        if futile:
            assess["reasons"].append("futility_stop")
        log.append(_mc_stage_record(idx, N_k, seed_k, summ, assess))
        prev = summ
        if assess["resolved"] or futile:
            break
    return out, summ, assess, log


def mc_validated_3sigma(fpp, fpp_se, nfpp, nfpp_se):
    """Decision (g): validated needs FPP + 3 sigma < 0.015 and NFPP + 3 sigma < 0.001."""
    return bool(fpp + MC_SIGMA * fpp_se < FPP_VALIDATED and nfpp + MC_SIGMA * nfpp_se < NFPP_VALIDATED)


def calc_fpp_native(tic_id, period, t0, duration, depth_ppm, sectors,
                     time, flux, N=100000, workdir=".",
                     server_url=None, token=None, abort_check=None,
                     force_cpu=False, cadence_s=None, allow_prefolded=False,
                     geometry=None, dilution_corrected=False,
                     population=None, population_sha256=None, population_version=None,
                     population_crosscheck=None, lc_source=None, depth_scale=1.0,
                     adaptive_n=None, n_cap=None, futility_stop=None, seed_ephemeris=None):
    """Calculate False Positive Probability without triceratops/pytransit.

    Implements Giacalone & Dressing (2020) FPP framework:
    - Scenarios: target, bound-companion, D/B background and viable nearby families
    - Monte Carlo marginal likelihoods with N draws per scenario
    - FPP = 1 - P(TP) - P(PTP) - P(DTP)

    Args:
        tic_id: TIC identifier.
        period: Orbital period (days).
        t0: Transit epoch (BJD - 2457000).
        duration: Transit duration (days).
        depth_ppm: Transit depth (ppm).
        sectors: Array of TESS sector numbers.
        time: Light curve time array.
        flux: Light curve flux array (normalized).
        N: MC draws per scenario (default 100000).
        workdir: Directory for cache files.
        server_url: ExoNit server URL for nearby star queries.
        token: Authentication token for server API.
        abort_check: Optional callable returning True to abort computation.
        force_cpu: Accepted and IGNORED — FPP is CPU-only by design (plan
            decision (e)), so there is no GPU path left to force off. The
            parameter stays so existing call sites keep working.
        cadence_s: Cadence of the raw light curve in seconds — the exposure
            box W = cadence_s / 86400 of the production fold likelihood (plan
            0.3 (j'), (j'')). Only used when FPP_EXPOSURE = "cadence" and the
            raw light curve is usable; None (the default) derives it from the
            median spacing of the finite times.
        geometry: D5 neighbour geometry shipped in the task (plan 1.5, decision
            (f)): {"kind": "real", "sectors": [{"wcs", "col0", "row0",
            "aperture"}, ...], "n_task_sectors": n} built server-side from the
            SPOC / TESSCut headers (exonit/fpp_geometry.py); None or a
            geometry without sectors = synthetic (3x3 sub-pixel union), which
            is counted in the result and can never be `validated`.
        dilution_corrected: upstream's calc_depths switch.  True = the light
            curve is already target-referenced (SPOC PDCSAP, TGLC cal_aper_flux):
            the target is not renormalised and a neighbour's frame is
            1 + (fr_t / fr_i)(f - 1); False = raw aperture flux.  The choice per
            light-curve product is made by calc_fpp_by_source (plan D8).
        lc_source: 'spoc' / 'tglc' / 'qlp' / ... of the light curve.  Recorded in
            the result and used by the smooth geometry to pick the product's
            aperture when the task does not carry one (emulated, flagged,
            never validated); None keeps the sector's own aperture.
        adaptive_n: MC-noise refinement (plan decision (g), step 7).  None = follow the FPP_ADAPTIVE_N switch (on in
            production, off in parity mode); True / False force the sequential refinement (N, 10 N, 100 N up to n_cap, new
            deterministic seed per stage) on / off.  The SE-based verdict rule (3 sigma `validated`, `borderline`) is
            applied whenever the switch is on or refinement is forced; parity mode with adaptive_n=None is untouched.
        n_cap: cap of the draws per scenario of the last stage (default MC_N_CAP = 1e6, decision C).
        futility_stop: None / True = stop after an unresolved stage 0 when the dominant scenario has ESS < MC_FUTILE_ESS_DOM (the
            candidate stays `borderline`); False runs the whole ladder (measurement only).
        seed_ephemeris: (period, t0) the MC seeds (tic_seed / stage_seed) are derived from; None = (period, t0).
            calc_fpp_by_source passes the ORIGINAL input ephemeris here when `period` / `t0` are the refined values
            (FPP_EPHEM_REFINE), so the seed never depends on the refinement numerics.
        depth_scale: factor applied to the deviation of the light curve from 1
            and to depth_ppm before anything else (TGLC: 1/1.09, see
            lc_depth_scale); 1.0 leaves the input untouched.

    Returns:

        Dict with fpp, nfpp, verdict, per-scenario evidences/probabilities and
        population provenance, or None for non-retryable input failures.
        Raises FPPUnavailable for missing/invalid/alarmed population (requeue),
        and FPPAborted if abort_check() returns True.

        population: canonical task document header+rows [Tmag,Mact,logg,logTe,[M/H]].
        population_sha256/version: expected immutable task provenance. REQUIRED whenever a
            population is supplied: a missing/empty value refuses with ``population_invalid``
            before any computation (same as the Swift port; decision 2026-10-25).
        population_crosscheck: R8 metadata separate from the hashed document;
            an alarm requires a genuine server-supplied Gaia fallback, otherwise refuse.
            An absent/unavailable cross-check cannot produce a validated result.
    """
    def _check_abort():
        if abort_check and abort_check():
            raise FPPAborted(f"FPP aborted for TIC {tic_id}")

    if GL_TABLES_ERROR:
        calc_fpp_native.last_skip_reason = "gl_tables_unavailable"
        raise FPPUnavailable(f"FPP unavailable on this host: {GL_TABLES_ERROR}")
    dilution_corrected = bool(dilution_corrected)
    depth_scale = float(depth_scale)
    if not (depth_scale > 0.0 and math.isfinite(depth_scale)):
        raise ValueError("depth_scale must be a positive finite number")
    calc_fpp_native.last_skip_reason = None
    # Exposure diagnostics are per-candidate: the shares below are the MAX over
    # this candidate's scenario calls (see _note_exposure_stats).
    _EXPOSURE_STATS["fallback_share"] = 0.0
    _EXPOSURE_STATS["unconverged_share"] = 0.0
    _EXPOSURE_STATS["kepler_unconverged_draws"] = 0

    # Deterministic per-candidate seed (normative scheme, see tic_seed()
    # docstring) — a private RandomState (not the process-global np.random)
    # so two candidates never perturb each other's draw stream even if
    # calc_fpp_native is ever called from more than one thread.
    seed_P, seed_t0 = (period, t0) if seed_ephemeris is None else (float(seed_ephemeris[0]), float(seed_ephemeris[1]))
    seed_u32 = tic_seed(tic_id, seed_P, seed_t0)
    mc_switch = switch_is("FPP_ADAPTIVE_N", "on")
    refine = mc_switch if adaptive_n is None else bool(adaptive_n)
    mc_policy = mc_switch or refine

    try:
        # Provenance gate first, before any CPU work: a supplied population whose digest or
        # version is missing is unverifiable and refused (population_invalid), never computed.
        # A missing population keeps its own population_missing refusal below.
        if population is not None and population_provenance_missing(population_sha256, population_version):
            raise BackgroundUnavailable("population provenance is incomplete "
                                        "(population_sha256/population_version missing)",
                                        "population_invalid")
        # Clean the raw light curve first (plan 0.3 (q')): the likelihood sigma
        # comes from the RAW curve, not from a fold that was already binned.
        time = np.asarray(time, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float64)
        if depth_scale != 1.0:
            # product depth normaliser (TGLC: depths are too deep by c = 1.09): rescale the deviation from 1 (signal AND
            # noise, both are in the wrong unit) and the depth the neighbours must reproduce
            flux = 1.0 + (flux - 1.0) * depth_scale
            depth_ppm = float(depth_ppm) * depth_scale
        finite = np.isfinite(time) & np.isfinite(flux)
        raw_usable = int(finite.sum()) >= 100

        sigma_bins = None       # per-bin likelihood sigma of D15 (production)
        sigma_scalar = None     # scalar sigma_oot: the veto depth of the renormed curve
        sigma_source = "folded_std_fallback"
        fit_rcond = 1.0         # conditioning of the (q') phase-curve fit (1.0 = no fit / not applicable)

        if raw_usable:
            comps = sigma_components(time[finite], flux[finite], period, t0, duration)
            if comps is None:
                raw_usable = False
            else:
                sigma_scalar = float(comps["sigma_oot"])
                fit_rcond = float(comps.get("fit_rcond", 1.0))
                sigma_bins = np.asarray(comps["sigma_b"], dtype=np.float64)
                sigma_source = "q_prime"

        if not raw_usable and not allow_prefolded:
            # No silent downgrade (CLAUDE.md): an unusable raw curve never falls back to the old
            # model (std sigma, median fold, no exposure integration). The pre-folded legacy path
            # exists ONLY for the golden harness, which passes allow_prefolded=True explicitly.
            logging.getLogger("exonit").warning(
                f"FPP skip TIC {tic_id}: raw light curve unusable (< 100 finite points or no "
                f"sigma estimate); skip_reason=raw_lc_unusable")
            calc_fpp_native.last_skip_reason = "raw_lc_unusable"
            return None

        if raw_usable and switch_is("FPP_EXPOSURE", "cadence"):
            # PRODUCTION (plan 0.3 (j'), (j'')): the mean-plus-phase production
            # fold of the raw curve, binned sigma per row, exposure box W from
            # the cadence. `_phase_fold_bin` is not used on this path at all.
            if cadence_s is None:
                cadence_s = float(round(float(np.median(np.diff(np.sort(time[finite])))) * 86400.0, 3))
            cadence_s = float(cadence_s)

            rows = fold_production(time[finite], flux[finite], t0, period,
                                   keep_phases=True)
            if len(rows) != len(sigma_bins):
                raise RuntimeError(
                    f"production fold has {len(rows)} non-empty bins but the "
                    f"(q') sigma has {len(sigma_bins)}")
            fold = build_prod_fold(rows, period, sigma_bins,
                                   [r["flux_mean"] for r in rows], cadence_s)
            time_pf = flux_pf = None
            sigma = sigma_scalar
        else:
            # PARITY / degenerate input (the golden harness injects a pre-folded
            # curve): the median-bin fold `_phase_fold_bin` and the time-array
            # likelihood, exactly as before.
            cadence_s = None
            fold = None

            # Phase-fold and bin
            time_pf, flux_pf = _phase_fold_bin(time, flux, period, t0)
            if len(time_pf) < 10:
                logging.getLogger("exonit").warning(

                    f"FPP skip TIC {tic_id}: too few phase-folded points "
                    f"({len(time_pf)})")
                return None

            # Remove NaNs
            good = ~np.isnan(time_pf) & ~np.isnan(flux_pf)
            time_pf = np.ascontiguousarray(time_pf[good], dtype=np.float64)
            flux_pf = np.ascontiguousarray(flux_pf[good], dtype=np.float64)
            if raw_usable:
                # PARITY mode with a usable raw curve: the same fold and the
                # same time-array likelihood, but the SCALAR (q') sigma_oot
                # instead of np.std — the parity fixtures feed both sides the
                # same scalar sigma_oot.
                sigma = sigma_scalar
                sigma_source = "q_prime"
            else:
                sigma = float(np.std(flux_pf))
                if sigma <= 0:
                    sigma = 1e-6


        # Query TIC for nearby stars (via server API)
        tic_result = _query_tic_stars(tic_id, server_url=server_url,
                                       token=token)
        if not tic_result:
            logging.getLogger("exonit").warning(
                f"FPP skip TIC {tic_id}: no TIC neighbor data")
            return None

        stars, gal_b = tic_result

        # Crowding assessment: |b| < 10° or >15 TIC neighbors
        n_nearby = len(stars) - 1
        high_crowding = abs(gal_b) < 10.0 or n_nearby > 15

        stars, geo_info = _flux_ratios_for_mode(stars, depth_ppm, geometry, dilution_corrected, lc_source)
        target = stars[0]
        bg = background_population(population, target['Tmag'], population_sha256, population_version,
                                   expected_tic_id=tic_id)
        if population_crosscheck is not None and not isinstance(population_crosscheck, dict):
            raise BackgroundUnavailable('invalid population cross-check')
        if (population_crosscheck or {}).get('alarm') and bg['info']['population_source'] != 'gaia':
            # The payload must contain a real Gaia fallback table. Never fabricate
            # stellar properties or rescale Galaxia to make an alarm disappear.
            raise BackgroundUnavailable('R8 alarm requires a server-supplied Gaia population', 'population_alarm')
        crosscheck_ok = bool(population_crosscheck and population_crosscheck.get('gaia_available') is True)
        bg['info']['population_crosscheck_status'] = 'passed' if crosscheck_ok else 'unverified'

        # Validate target star properties
        M_s = target.get('mass', np.nan)
        R_s = target.get('rad', np.nan)
        Teff = target.get('Teff', np.nan)
        plx = target.get('plx', np.nan)

        if np.isnan(M_s) or np.isnan(R_s) or np.isnan(Teff):
            logging.getLogger("exonit").warning(
                f"FPP skip TIC {tic_id}: missing stellar params "
                f"(M={M_s} R={R_s} Teff={Teff}, "
                f"mass_source={target.get('mass_source')}); skip_reason=giant_host|stellar_params_missing")
            # Explicit, countable outcome: the caller reads last_skip_reason and reports it to the
            # server (candidates.fpp_skip_reason). A mass is never invented here.
            # The server classifies WHY there is no mass (giant_host / stellar_params_missing); an older server
            # sends no hint: then a large radius still means a giant host.
            _hint = target.get("mass_unavailable_reason")
            if _hint not in ("giant_host", "stellar_params_missing"):
                _hint = "giant_host" if (np.isfinite(R_s) and R_s > 2.5) else "stellar_params_missing"
            calc_fpp_native.last_skip_reason = _hint
            return None

        # Limb darkening
        logg = np.log10(G_cgs * M_s * Msun / (R_s * Rsun)**2)
        u1, u2 = get_ldc(Teff, logg)

        # Frame of the target's light curve (dilution convention, plan D8): False = raw aperture flux, renormalised by the
        # target's own flux fraction; True = already target-referenced, used as is (fr_frame_t == 1.0 exactly).
        fr_frame_t = frame_fluxratio(target['fluxratio'], target['fluxratio'], dilution_corrected)
        if fold is None:
            flux_t, sigma_t = renorm_flux(flux_pf, sigma, fr_frame_t)
            sigma_t = float(sigma_t)
            fold_t = None

        else:
            # Production: the fold renormalised to the target's flux fraction;
            # the veto depth is the SCALAR noise level of that renormalised
            # curve (the fold carries a per-bin sigma array).  The target's
            # scenarios all share one renormalised fold.
            flux_t = None
            sigma_t = sigma_scalar / fr_frame_t
            fold_t = fold if dilution_corrected else prod_fold_renorm(fold, fr_frame_t)


        P_orb = float(period)

        crowding_tag = " HIGH_CROWDING" if high_crowding else ""
        trilegal_tag = ""
        logging.getLogger("exonit").debug(f"FPP: TIC {tic_id} — M={M_s:.2f} R={R_s:.2f} "
              f"Teff={Teff:.0f} u1={u1:.3f} u2={u2:.3f} "
              f"N_nearby={n_nearby} b={gal_b:.1f}°"
              f"{crowding_tag}{trilegal_tag}")

        def _mc_pass(N, rng):
            """One Monte-Carlo pass of every scenario with N draws per scenario and the given RandomState.

            Returns (lnZ, nearby_lnZ, nearby_scen, diag): the target-family evidences, the nearby-star evidences (list and
            by name) and the per-scenario draw-weight diagnostics {name: {"ess", "n", "max_share"}} (step 7).
            """
            global _DRAW_DIAG
            sink = []
            diag = {}

            def _sc(names, fn, *a, **k):
                n0 = len(sink)
                out = fn(*a, **k)
                got = sink[n0:]
                if got and len(got) != len(names):
                    raise RuntimeError(f"{fn.__name__}: {len(got)} draw-weight records for {len(names)} scenarios")
                diag.update(zip(names, got))          # no record at all (a stubbed scenario): ESS unknown -> unresolved
                return out

            _DRAW_DIAG = sink
            try:
                # Compute marginal likelihoods for each scenario
                # _check_abort() between each heavy computation so Q/Ctrl+C can stop us
                lnZ = {}

                # Target star scenarios
                # Target star scenarios
                _check_abort()
                lnZ['TP'] = _sc(['TP'], _lnZ_TTP, time_pf, flux_t, sigma_t, P_orb,
                                       M_s, R_s, Teff, N, u1, u2, rng=rng,
                                       fold=fold_t,
                                       sigma_veto=sigma_t if fold is not None else None)
                _check_abort()
                lnZ_eb, lnZ_eb_twin = _sc(['EB', 'EBx2P'], _lnZ_TEB, time_pf, flux_t, sigma_t, P_orb,
                                                  M_s, R_s, Teff, N, u1, u2, rng=rng,
                                                  fold=fold_t,
                                                  sigma_veto=sigma_t if fold is not None else None)
                lnZ['EB'] = lnZ_eb
                lnZ['EBx2P'] = lnZ_eb_twin

                # Physical companion scenarios
                _check_abort()
                if not np.isnan(plx):
                    lnZ['PTP'] = _sc(['PTP'], _lnZ_PTP, time_pf, flux_t, sigma_t, P_orb,
                                            M_s, R_s, Teff, N, u1, u2, plx, rng=rng,
                                            fold=fold_t,
                                            sigma_veto=sigma_t if fold is not None else None)
                    _check_abort()
                    lnZ_peb, lnZ_peb_twin = _sc(['PEB', 'PEBx2P'], _lnZ_PEB, time_pf, flux_t, sigma_t,
                                                        P_orb, M_s, R_s, Teff,
                                                        N, u1, u2, plx, rng=rng,
                                                        fold=fold_t,
                                                        sigma_veto=sigma_t if fold is not None else None)
                    lnZ['PEB'] = lnZ_peb
                    lnZ['PEBx2P'] = lnZ_peb_twin
                else:
                    lnZ['PTP'] = -np.inf
                    lnZ['PEB'] = -np.inf
                    lnZ['PEBx2P'] = -np.inf

                # D6 S family: planet / EB around the unresolved bound companion (plan step 4); needs the parallax like PTP / PEB
                _check_abort()
                if not np.isnan(plx):
                    lnZ['STP'] = _sc(['STP'], _lnZ_STP, time_pf, flux_t, sigma_t, P_orb, M_s, R_s, Teff, N, plx, rng=rng, fold=fold_t,
                                          sigma_veto=sigma_t if fold is not None else None)
                    _check_abort()
                    lnZ['SEB'], lnZ['SEBx2P'] = _sc(['SEB', 'SEBx2P'], _lnZ_SEB, time_pf, flux_t, sigma_t, P_orb, M_s, R_s, Teff, N, plx, rng=rng,
                                                         fold=fold_t, sigma_veto=sigma_t if fold is not None else None)
                else:
                    lnZ['STP'] = -np.inf
                    lnZ['SEB'] = -np.inf
                    lnZ['SEBx2P'] = -np.inf

                # D6 D/B families: same target-renormalised frame as the other scenarios.
                _check_abort()
                lnZ['DTP'] = _sc(['DTP'], _lnZ_DTP, time_pf,flux_t,sigma_t,P_orb,M_s,R_s,Teff,N,u1,u2,bg,rng=rng,fold=fold_t,
                                       sigma_veto=sigma_t if fold is not None else None)
                _check_abort()
                lnZ['DEB'],lnZ['DEBx2P'] = _sc(['DEB', 'DEBx2P'], _lnZ_DEB, time_pf,flux_t,sigma_t,P_orb,M_s,R_s,Teff,N,u1,u2,bg,rng=rng,fold=fold_t,
                                       sigma_veto=sigma_t if fold is not None else None)
                _check_abort()
                lnZ['BTP'] = _sc(['BTP'], _lnZ_BTP, time_pf,flux_t,sigma_t,P_orb,M_s,R_s,Teff,N,bg,rng=rng,fold=fold_t,
                                       sigma_veto=sigma_t if fold is not None else None)
                _check_abort()
                lnZ['BEB'],lnZ['BEBx2P'] = _sc(['BEB', 'BEBx2P'], _lnZ_BEB, time_pf,flux_t,sigma_t,P_orb,M_s,R_s,Teff,N,bg,rng=rng,fold=fold_t,
                                       sigma_veto=sigma_t if fold is not None else None)

                # Nearby star scenarios (NTP, NEB for each neighbor)
                _check_abort()
                nearby_lnZ = []
                nearby_scen = {}
                # D5 (plan 1.5): EVERY viable star (upstream `tdepth > 0`) gets its scenarios with the full N draws
                # (upstream triceratops.py:924-925); no cap on the number of neighbours.
                # Legacy TRILEGAL pseudo-stars are placed AT the target position (no real position): they are not nearby stars
                # and must not get NTP/NEB scenarios (the background population is step 5, D6).
                filtered_nearby = [s for s in stars[1:] if s['viable'] and not s.get('is_trilegal')]
                # Plan decision (r) / D16: FPP_NORM = "data" puts the nearby-star
                # scenarios in the data space of the target (the flux ratio between the
                # two stars is a prior factor on the scenario, not on the light curve).
                norm_data = switch_is("FPP_NORM", "data")
                for nearby_index, ns in enumerate(filtered_nearby):
                    ns_M = ns.get('mass', 1.0)
                    ns_R = ns.get('rad', 1.0)
                    ns_Teff = ns.get('Teff', 5777)
                    if np.isnan(ns_M):
                        ns_M = 1.0
                    if np.isnan(ns_R):
                        ns_R = 1.0
                    if np.isnan(ns_Teff):
                        ns_Teff = 5777

                    ns_logg = np.log10(G_cgs * ns_M * Msun / (ns_R * Rsun)**2)
                    ns_u1, ns_u2 = get_ldc(ns_Teff, ns_logg)

                    fr_frame_ns = frame_fluxratio(ns['fluxratio'], target['fluxratio'], dilution_corrected)
                    if fold is None:
                        flux_ns, sigma_ns = renorm_flux(flux_pf, sigma, fr_frame_ns)
                        ns_fold = None
                        ns_veto = None
                    else:
                        flux_ns = None
                        sigma_ns = sigma_scalar / fr_frame_ns
                        ns_fold = prod_fold_renorm(fold, fr_frame_ns)
                        ns_veto = sigma_ns

                    _check_abort()
                    tag = f"{ns.get('ID', '?')}:{nearby_index}"
                    ntp = _sc([f'NTP:{tag}'], _lnZ_TTP, time_pf, flux_ns, float(sigma_ns), P_orb,
                                    ns_M, ns_R, ns_Teff, N, ns_u1, ns_u2, rng=rng,
                                    fold=ns_fold, sigma_veto=ns_veto)
                    _check_abort()
                    neb, neb_twin = _sc([f'NEB:{tag}', f'NEBx2P:{tag}'], _lnZ_TEB, time_pf, flux_ns, float(sigma_ns),
                                               P_orb, ns_M, ns_R, ns_Teff,
                                               N, ns_u1, ns_u2, rng=rng,
                                               fold=ns_fold, sigma_veto=ns_veto)
                    if norm_data:
                        ln_norm = nearby_lnz_offset(target['fluxratio'], ns['fluxratio'])
                        ntp += ln_norm
                        neb += ln_norm
                        neb_twin += ln_norm
                    # geometry v3: the neighbour's viability is a probability in (0, 1], the prior of its scenarios is scaled by it
                    # (first-order marginalisation over the calibration uncertainty; v = 1 adds nothing, parity mode has v in {0, 1})
                    v_ns = float(ns.get('viability', 1.0))
                    if v_ns != 1.0:
                        ln_v = math.log(v_ns)
                        ntp += ln_v
                        neb += ln_v
                        neb_twin += ln_v
                    nearby_lnZ.extend([ntp, neb, neb_twin])
                    nearby_scen.update({f'NTP:{tag}':ntp, f'NEB:{tag}':neb, f'NEBx2P:{tag}':neb_twin})

            finally:
                _DRAW_DIAG = None
            return lnZ, nearby_lnZ, nearby_scen, diag

        # Stage loop of the MC-noise policy (plan decision (g), step 7).  Stage 0 is the candidate's own seed and N, i.e.
        # exactly the pre-step-7 single pass; stages 1.. are re-runs at 10x / 100x the draws with new deterministic seeds,
        # taken only while the gate decision is unresolved.  The reported result is the LAST stage alone (a fresh
        # independent sample; pooling stages after a data-dependent stopping rule would bias it).
        stage_ladder = _mc_stage_ladder(N, refine, n_cap)

        def _stage_seed(idx):
            return seed_u32 if idx == 0 else stage_seed(tic_id, seed_P, seed_t0, idx, seed_u32)

        def _stage_pass(N_k, seed_k):
            _check_abort()
            # exposure diagnostics describe the stage that is reported
            _EXPOSURE_STATS["fallback_share"] = 0.0
            _EXPOSURE_STATS["unconverged_share"] = 0.0
            _EXPOSURE_STATS["kepler_unconverged_draws"] = 0
            return _mc_pass(N_k, np.random.RandomState(seed_k))

        staged = _mc_run_stages(_stage_pass, stage_ladder, _stage_seed, futility=(True if futility_stop is None else bool(futility_stop)))
        if staged is None:
            logging.getLogger("exonit").warning(
                f"FPP skip TIC {tic_id}: all lnZ scenarios non-finite")
            return None
        (lnZ, nearby_lnZ, nearby_scen, mc_diag), summ, assess, stage_log = staged

        probs = summ["probs"]
        # Keep scenario names authoritative: inserting a family never shifts TP indices.
        all_scen = dict(lnZ, **nearby_scen)
        scen_probs = dict(zip(all_scen, probs))
        fpp = summ["fpp"]
        nfpp = summ["nfpp"]
        borderline = bool(mc_policy and not assess["resolved"])

        verdict = fpp_verdict(fpp, nfpp=nfpp, high_crowding=high_crowding)
        if mc_policy and verdict == "validated" and (borderline or not mc_validated_3sigma(fpp, summ["fpp_se"], nfpp, summ["nfpp_se"])):
            # 3-sigma rule of decision (g): FPP + 3 sigma_MC < 0.015 and NFPP + 3 sigma_MC < 0.001; an unresolved
            # candidate (gate within 3 sigma / heavy-tailed weights at the cap) is never validated
            verdict = "candidate"
        # Honesty cap (FPP_UNSTABLE_FLAG): the dominant scenario's evidence of the reported stage is (nearly) one draw, so the FPP is
        # a lottery value; neither an FP claim nor a validation rests on it.  After the step-7 cap, before the geometry caps.
        unstable_on = switch_is("FPP_UNSTABLE_FLAG", "on")
        unstable = bool(unstable_on and mc_is_unstable(summ))
        if unstable and verdict in ("likely_fp", "validated"):
            verdict = "candidate"
        # Exposure diagnostics of the production fold likelihood: a candidate
        # whose bin ladder needed the oracle fallback / never converged cannot
        # be validated on that likelihood.
        fb_share = float(_EXPOSURE_STATS["fallback_share"])
        unc_share = float(_EXPOSURE_STATS["unconverged_share"])
        unconverged = unc_share > 0.01
        geo_real = geo_info["geometry_kind"] == "real" and geo_info["geometry_coverage"] >= 1.0
        # reference-FFI WCS (FFI-only candidates) is never validated until the amended pre-registration
        # (docs/fpp-ffi-wcs-reference-preregistration-v2.md) has passed on fresh sectors
        # The one exception is what the v3 run validated: TGLC 3x3 box (path T) with neighbour positions from the K-epoch FFI
        # WCS cache.  FFI_REFERENCE_VALIDATABLE itself stays False, so every other ffi_reference geometry (default 5x5,
        # QLP circle, a v1 single-epoch entry, an emulated box) remains capped.
        if ("ffi_reference" in geo_info["geometry_source"] and not FFI_REFERENCE_VALIDATABLE
                and not _ffi_reference_tglc_box_k_epoch(geo_info)):
            geo_real = False
        # smooth geometry: every path must have passed its own v3 report (GEOMETRY_PATH_VALIDATED); a product aperture
        # emulated from our own position (no header / BESTAP in the task) is never validated
        if geo_info.get("geometry_mode") == "smooth" and (
                geo_info.get("geometry_path_emulated")
                or not all(GEOMETRY_PATH_VALIDATED.get(p, False) for p in geo_info["geometry_path"])):
            geo_real = False
        # near-singular phase-curve fit (out-of-transit phases clustered in too few bins): the beta factors are
        # not trustworthy, so the candidate cannot be validated
        fit_ill = fit_rcond < FIT_ILL_CONDITIONED_RCOND
        # a non-zero Kepler non-convergence count means draws were silently dropped (weight 0): the FPP is biased
        kepler_unc = int(_EXPOSURE_STATS["kepler_unconverged_draws"]) > 0
        # giant host that still has a (TIC/Berger) mass: computed, flagged, never validated. The server flag covers
        # logg < 3.5; R > 2.5 is checked here too for an older server.
        giant_host = bool(target.get("giant_host")) or bool(np.isfinite(R_s) and R_s > 2.5)
        if (unconverged or kepler_unc or sigma_source != "q_prime" or not geo_real or fit_ill
                or not crosscheck_ok or giant_host) and verdict == "validated":
            verdict = "candidate"

        result = {
            "fpp": fpp,
            "nfpp": nfpp,
            "verdict": verdict,
            "high_crowding": high_crowding,
            "gal_b": round(gal_b, 2),
            "n_nearby": n_nearby,
            "trilegal_used": False,
            "fpp_backend": "cpu-python",
            "fpp_flags": (["population_crosscheck_unverified"] if not crosscheck_ok else [])
                         + (["background_q_target_quirk"] if switch_is("FPP_BACKGROUND_Q", "target") else [])
                         + (["mass_from_relation"] if target.get("mass_source") == "teff_relation_tic8fit" else [])
                         + (["giant_host"] if giant_host else [])
                         + (["mc_borderline"] if borderline else [])
                         + (["fpp_unstable"] if unstable else []),
            **bg["info"],
            "fpp_scen": {name: {"lnZ": float(z) if np.isfinite(z) else None, "prob": float(scen_probs[name])}
                         for name,z in all_scen.items()},
            "fpp_seed": int(seed_u32),
            # MC noise (decision (g), step 7): SE of the reported (last-stage) FPP / NFPP, draws spent, stages, weight degeneracy
            "fpp_mc_se": summ["fpp_se"],
            "nfpp_mc_se": summ["nfpp_se"],
            "n_draws_final": int(summ["N"]),
            "n_draws_total": int(sum(r["n"] for r in stage_log)),
            "refinement_stages": stage_log,
            # per scenario of the reported stage: [ESS of its draw weights, relative SE of its Z, extreme-value scale beta of its top draws (nats)]
            "mc_scen": {n: [float(e), float(np.sqrt(v)), float(b)] for n, e, v, b in zip(summ["names"], summ["ess"], summ["rel_var"], summ["tail_scale"])},
            "mc_min_ess": summ["min_ess"],
            "mc_min_ess_scenario": summ["min_ess_scenario"],
            "mc_policy": bool(mc_policy),
            "borderline": borderline,
            "borderline_reasons": list(assess["reasons"]) if borderline else [],
            "fpp_model_version": FPP_MODEL_VERSION,
            "fpp_switches": get_fpp_switches(),
            "sigma_source": sigma_source,
            "sigma_fit_ill_conditioned": bool(fit_ill),
            # D8 (geometry v3): what convention the light curve was treated with and how it was rescaled
            "dilution_corrected": dilution_corrected,
            "dilution_convention": "true" if dilution_corrected else "false",
            "depth_scale": depth_scale,
            "lc_source": _norm_lc_source(lc_source) if lc_source is not None else None,
            "geometry_model_version": GEOMETRY_VERSION,
            "geometry_version": GEOMETRY_VERSION,      # the name the server stores (candidates.geometry_version)
            **geo_info,
            "exposure_gl_fallback_share": fb_share,
            "exposure_unconverged_share": unc_share,
            "exposure_unconverged": unconverged,
            "kepler_unconverged_draws": int(_EXPOSURE_STATS["kepler_unconverged_draws"]),
        }
        if unstable_on:
            dom_name, dom_ess = _mc_dominant(summ)
            result.update({"fpp_unstable": unstable, "mc_ess_dom": dom_ess if math.isfinite(dom_ess) else None, "mc_dominant_scenario": dom_name,
                           "fpp_interval": [float(x) for x in summ["tail_box_fpp"]]})
        if cadence_s is not None:
            result["cadence_s"] = cadence_s
        return result

    except FPPUnavailable as exc:
        calc_fpp_native.last_skip_reason = getattr(exc, "skip_reason", "population_invalid")
        raise
    except FPPAborted:
        logging.getLogger("exonit").info(f"FPP aborted for TIC {tic_id}")
        raise
    except (LDCTableMissing, LDCNoRow) as e:
        logging.getLogger("exonit").warning(
            f"FPP skip TIC {tic_id}: skip_reason={e.skip_reason}: {e}")
        calc_fpp_native.last_skip_reason = e.skip_reason
        return None
    except Exception as e:
        logging.getLogger("exonit").warning(
            f"FPP failed for TIC {tic_id}: {type(e).__name__}: {e}")
        import traceback
        logging.getLogger("exonit").debug(traceback.format_exc())
        return None


def fpp_ephemeris(time, flux, period, t0, duration, t0_convention, cadence_s=None):
    """The (P, t0 centre) the FPP fold uses, plus the provenance fields of the result (section 8b).

    t0_convention is EXPLICIT, never inferred: "box_start" = raw BLS epoch (start of the best box; every production
    candidate while duration_refine is OFF), "centre" = a transit-centre epoch (TOI catalogue harnesses, duration_refine).
    FPP_EPHEM_REFINE = "off" (parity set): (period, t0) pass through untouched and no field is added (pre-change
    behaviour).  "on": box_start -> refine_ephemeris from the centre guess t0 + duration / 2; on status edge / failed
    the guess (period, t0 + duration / 2) is used and refine_failed is True (the caller flags and caps).  centre -> used
    as given (a catalogue reference is not re-fitted).  `cadence_s` = the exposure box of the template (None = from the
    point spacing, as calc_fpp_native derives its own).
    Returns (P_use, t0_use, provenance dict or None, refine_failed).
    """
    if t0_convention not in T0_CONVENTIONS:
        raise ValueError(f"t0_convention must be one of {T0_CONVENTIONS}, got {t0_convention!r}")
    if switch_is("FPP_EPHEM_REFINE", "off"):
        return period, t0, None, False
    if t0_convention == "centre":
        prov = {"ephem_refined": False, "ephem_status": "not_requested", "ephem_t0_convention": "centre",
                "ephem_P": float(period), "ephem_t0": float(t0), "ephem_shift_h": 0.0}
        return period, t0, prov, False
    guess = float(t0) + 0.5 * float(duration)
    ref = refine_ephemeris(time, flux, period, guess, duration, cadence_s=cadence_s)
    ok = ref["status"] == "ok"
    P_use, t0_use = (ref["P"], ref["t0"]) if ok else (float(period), guess)
    prov = {"ephem_refined": bool(ok), "ephem_status": ref["status"], "ephem_t0_convention": "box_start",
            "ephem_P": float(P_use), "ephem_t0": float(t0_use),
            "ephem_shift_h": float(ref["shift_h"]) if ok else 0.0,
            "ephem_refine": {k: ref[k] for k in ("reason", "P_in", "t0_in", "t_ref_in", "cadence_s", "n_req", "n_orbits",
                                                  "n_transits", "p_searched", "p_free_accepted", "delta_chi2_p",
                                                  "dchi2_p_fixed", "dchi2_p_free", "p_fixed", "dP", "max_offset_h",
                                                  "dchi2_initial", "dchi2_refined", "delta_chi2", "sigma", "t14_factor",
                                                  "ingress", "edge_P", "edge_t0", "dP_half", "dt_half")}}
    return P_use, t0_use, prov, not ok


def calc_fpp_by_source(tic_id, period, t0, duration, depth_ppm, sectors, time, flux, *, lc_source, t0_convention,
                       tglc_near_edge=None, depth_scale=None, seed_ephemeris=None, **kwargs):
    """FPP of a candidate with the dilution convention decided by the light-curve product (plan D8, decision 1A).

    SPOC PDCSAP and TGLC cal_aper_flux: one run with dilution_corrected=True (TGLC also with the depth scale 1/c, c = 1.09,
    or 1.27 for a NEAREDGE star).  QLP and every product that cannot be verified (mixed, unknown, the worker's FFI fallback):
    both conventions with the SAME seed, the HIGHER FPP is reported (True on a tie), `dilution_unverified` is set and the
    verdict is never `validated`.  `dilution_alternatives` records both runs.  `kwargs` go to calc_fpp_native
    (geometry, N, cadence_s, ...); `dilution_corrected` must not be among them.
    `t0_convention` ("box_start" for the worker's BLS / DB epochs, "centre" for catalogue ephemerides) is required: with
    FPP_EPHEM_REFINE = "on" a box-start ephemeris is refined ONCE here (fpp_ephemeris) and both dilution runs fold at the
    refined (P, t0); the MC seeds stay those of the ORIGINAL (tic, period, t0).  A refinement at the search edge / failed
    keeps the box centre, sets `ephem_refine_failed` and is never `validated`.
    `seed_ephemeris` (P, t0): what the seeds are derived from when it is not the passed (period, t0) (measurement
    harnesses only; the worker passes the stored row on both paths and no seed_ephemeris).  None = (period, t0).
    The refinement uses kwargs["cadence_s"] (when given) as the exposure box of its template, like the FPP likelihood.
    Returns the chosen run's result dict (plus the fields above), or None if a run was skipped (the reason is in
    calc_fpp_by_source.last_skip_reason).
    """
    if "dilution_corrected" in kwargs:
        raise TypeError("calc_fpp_by_source decides dilution_corrected from lc_source")
    P_use, t0_use, ephem_prov, ephem_failed = fpp_ephemeris(time, flux, period, t0, duration, t0_convention,
                                                            cadence_s=kwargs.get("cadence_s"))
    if seed_ephemeris is not None:
        kwargs["seed_ephemeris"] = (float(seed_ephemeris[0]), float(seed_ephemeris[1]))
    elif ephem_prov is not None:
        kwargs["seed_ephemeris"] = (period, t0)
    src = _norm_lc_source(lc_source) if lc_source is not None else "unknown"
    plan = dilution_plan(src)
    ds = float(depth_scale) if depth_scale is not None else lc_depth_scale(src, tglc_near_edge)
    runs = {}
    calc_fpp_by_source.last_skip_reason = None
    for dc in plan["conventions"]:
        r = calc_fpp_native(tic_id, P_use, t0_use, duration, depth_ppm, sectors, time, flux,
                            dilution_corrected=dc, lc_source=src, depth_scale=ds, **kwargs)
        calc_fpp_by_source.last_skip_reason = getattr(calc_fpp_native, "last_skip_reason", None)
        if r is None:
            return None
        runs[dc] = r
    chosen_dc = True if True in runs else False
    if len(runs) == 2 and runs[False]["fpp"] > runs[True]["fpp"]:
        chosen_dc = False
    out = dict(runs[chosen_dc])
    out["n_draws_all_runs"] = int(sum(r.get("n_draws_total", 0) for r in runs.values()))     # compute actually spent (both conventions)
    out["dilution_corrected"] = chosen_dc
    out["dilution_convention"] = "true" if chosen_dc else "false"
    out["dilution_unverified"] = bool(plan["unverified"])
    flags = list(out.get("fpp_flags") or [])
    if plan["unverified"]:
        flags.append("dilution_unverified")
    if src == "tglc":
        out["tglc_near_edge"] = None if tglc_near_edge is None else bool(tglc_near_edge)
        if tglc_near_edge:
            flags.append("tglc_near_edge")                      # c = 1.27 (provisional) instead of 1.09
            if out.get("verdict") == "validated":
                out["verdict"] = "candidate"                    # the provisional constant is never validated
        elif tglc_near_edge is None and depth_scale is None:
            # NEAREDGE unknown: the depth constant (1.09 vs 1.27) is a guess. Flagged and never validated,
            # never a silent 1.09 (geometry review N4).
            flags.append("tglc_near_edge_unknown")
            out["tglc_near_edge_unknown"] = True
            if out.get("verdict") == "validated":
                out["verdict"] = "candidate"
    if ephem_prov is not None:
        out.update(ephem_prov)
        if ephem_failed:
            flags.append("ephem_refine_failed")
            out["ephem_refine_failed"] = True
            if out.get("verdict") == "validated":
                out["verdict"] = "candidate"                    # the fold ephemeris is not established
    out["fpp_flags"] = flags
    if len(runs) == 2:
        out["dilution_alternatives"] = {("true" if dc else "false"): {"fpp": r["fpp"], "nfpp": r["nfpp"]}
                                        for dc, r in sorted(runs.items(), reverse=True)}
    if plan["unverified"] and out.get("verdict") == "validated":
        out["verdict"] = "candidate"
    return out


calc_fpp_by_source.last_skip_reason = None


# ============================================================

# 10. Utilities
# ============================================================

def fpp_verdict(fpp, nfpp=None, high_crowding=False):
    # Validation rule (Giacalone+ 2020): FPP<0.015 AND NFPP<0.001.
    # high_crowding downgrades validated to candidate because the TIC-only
    # background catalog undercounts BEB contributors past Tmag~16.
    if fpp >= FPP_CANDIDATE:
        return "likely_fp"
    if fpp < FPP_VALIDATED and nfpp is not None and nfpp < NFPP_VALIDATED:
        return "candidate" if high_crowding else "validated"
    return "candidate"


# ============================================================
# Numba warm-up on import (compile transit model)
# ============================================================

def _warmup():
    """Pre-compile numba functions with small arrays."""
    t = np.linspace(-0.05, 0.05, 10)
    sig = np.full(10, 0.001)
    # both _lc_core branches: no exposure, and the parity supersampling loop
    _lc_core(t, 0.1, 0.0, 3.0, 15.0, 89.0 * pi / 180, 0.0, pi / 2,
             0.3, 0.2, 0.0, 1)
    _lc_core(t, 0.1, 0.0, 3.0, 15.0, 89.0 * pi / 180, 0.0, pi / 2,
             0.3, 0.2, PARITY_EXPTIME_D, PARITY_NSUB)
    _compute_transit_lc(t, 0.1, 0.0, 3.0, 15.0, 89.0 * pi / 180,
                        0.0, pi / 2, 0.3, 0.2)

    # both batch kernels, in both exposure modes and both companion geometries
    _batch_transit_lnL(t, np.ones(10), sig,
                        np.array([0.1]), np.array([3.0]),
                        np.array([89.0 * pi / 180]),
                        np.array([15.0]),
                        np.array([0.0]), np.array([pi / 2]),
                        0.3, 0.2, np.array([0.0]), False, 0.0, 1)
    _batch_transit_lnL(t, np.ones(10), sig,
                        np.array([0.1]), np.array([3.0]),
                        np.array([89.0 * pi / 180]),
                        np.array([15.0]),
                        np.array([0.0]), np.array([pi / 2]),
                        0.3, 0.2, np.array([0.3]), True,
                        PARITY_EXPTIME_D, PARITY_NSUB)

    _batch_eb_transit_lnL(t, np.ones(10), sig, 0.001,
                           np.array([0.1]), np.array([0.3]), np.array([3.0]),
                           np.array([89.0 * pi / 180]),
                           np.array([15.0]),
                           np.array([0.0]), np.array([pi / 2]),
                           0.3, 0.2, np.array([0.0]), False, 0.0, 1,
                           True, True, True,
                           np.array([0.3]), np.array([0.2]))
    _batch_eb_transit_lnL(t, np.ones(10), sig, 0.001,
                           np.array([1.0]), np.array([0.95]),
                           np.array([3.0]),
                           np.array([89.0 * pi / 180]),
                           np.array([15.0]),
                           np.array([0.1]), np.array([0.7]),
                           0.3, 0.2, np.array([0.3]), True,
                           PARITY_EXPTIME_D, PARITY_NSUB,
                           False, False, False,
                           np.array([0.3]), np.array([0.2]))

_init_fpp_gpu()

try:
    _warmup()
except Exception:
    pass
