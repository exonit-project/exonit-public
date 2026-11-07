"""Deterministic replacement for lightkurve's unseeded ``LightCurve.fill_gaps()``.

lightkurve 2.x fills gaps with ``np.random.normal(mean, CDPP, n)`` drawn from the
PROCESS-GLOBAL numpy RNG, so any light curve with an intra-sector gap < 5 d gave a
different flux series (and sometimes a different candidate list) on every run.
The injection-recovery pilot (docs/reviews/2026-10-15-ir-pilot-report.md) measured
20/100 hosts with different candidate lists on identical repeats, all of them
fill_gaps hosts.

``fill_gaps_seeded`` keeps lightkurve's gap geometry, flux_err interpolation and
quality flags verbatim (it calls ``lc.fill_gaps()``) and only REPLACES the noise
values of the synthetic points by ``RandomState(seed).normal(mean, std, n)``
with the SAME distribution lightkurve uses, N(mean(flux), CDPP-or-std).

Seed scheme (normative, ported to the Rust/Swift workers): key
``"{tic_id}:{s1},{s2},..."`` (sectors sorted, de-duplicated; empty list -> empty
string), FNV-1a 64 over the UTF-8 bytes, truncated to u32 -- the same family as
``fpp_native.tic_seed``.  A private RandomState (legacy MT19937 + polar
Box-Muller ``legacy_gauss``) is used, not the global RNG, so the draw is
thread-safe w.r.t. the FPP prefetch thread and independent across stars.

``gap_fill_detrend_only`` (task config flag, default False; see
docs/reviews/2026-10-15-fill-gaps-sigma-parity.md option A): the synthetic
points exist ONLY to bridge intra-sector gaps for the wotan detrender (so its
``break_tolerance`` split + ``edge_cutoff`` do not eat real cadences next to
every gap). With the flag on, ``fill_gaps_seeded(..., return_synthetic=True)``
also returns the boolean ``synthetic`` mask; the worker detrends on the
real+synthetic series and then DROPS the synthetic points, so BLS, vetting,
the multi-planet loop, CNN views and injection see only real cadences.
Every later re-detrend goes through ``DetrendFrame`` so it sees the same
real+synthetic series as the first detrend (consistent edge handling).
Transit masks apply to real points only; synthetic points (noise around the
global mean, no transit signal) always stay in the trend fit.

Deviation from flag OFF (deliberate, accepted 2026-10-15): with the flag off
the phase-based transit masks also cover synthetic points that fall inside a
transit window (they are part of ``time_arr``); with the flag on they never
do. Everything else the detrender sees is identical (the DetrendFrame series
is bit-identical to the flag-off ``time_arr``/``raw_flux_arr``). Also with the
flag on, the multi-planet 30 % mask-coverage gate counts real points only.

Flag resolution (worker.py ``_gap_fill_detrend_only_enabled``): task config
key if present, else worker config key, else False -- the FPP backfill has no
task config, so the worker key keeps in-task and backfill FPP consistent.
"""
import numpy as np


def gap_fill_seed_key(tic_id, sectors=None):
    secs = sorted({int(x) for x in (sectors or [])})
    return f"{int(tic_id)}:{','.join(map(str, secs))}"


def gap_fill_seed(tic_id, sectors=None):
    """FNV-1a 64 of the seed key, truncated to u32."""
    h = 0xCBF29CE484222325
    for byte in gap_fill_seed_key(tic_id, sectors).encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h & 0xFFFFFFFF


def fill_gaps_seeded(lc, tic_id, sectors=None, return_synthetic=False):
    """Return ``lc.fill_gaps()`` with deterministic noise (see module docstring).

    With ``return_synthetic=True`` returns ``(filled, synthetic)`` where
    ``synthetic`` is a boolean array aligned with ``filled`` (True = point
    created by the gap fill). The filled light curve is identical either way.
    """
    base = lc.copy().remove_nans()
    filled = lc.fill_gaps()          # consumes global RNG; those draws are discarded
    if hasattr(base, "cadenceno"):
        synth = ~np.isin(np.asarray(filled.cadenceno.value),
                         np.asarray(base.cadenceno.value))
    else:
        synth = ~np.isin(np.asarray(filled.time.value),
                         np.asarray(base.time.value))
    n = int(synth.sum())
    if n == 0:
        return (filled, synth) if return_synthetic else filled
    try:
        std = base.estimate_cdpp().to(base.flux.unit).value
    except Exception:
        std = np.nanstd(base.flux.value)
    rng = np.random.RandomState(gap_fill_seed(tic_id, sectors))
    noise = rng.normal(np.nanmean(base.flux.value), std, n)
    flux = np.array(filled.flux.value, dtype=np.float64)
    flux[synth] = noise
    filled.flux = flux * filled.flux.unit
    return (filled, synth) if return_synthetic else filled


class DetrendFrame:
    """Real+synthetic pre-detrend series for ``gap_fill_detrend_only``.

    ``time``/``raw_flux`` are the full (sorted, finite) arrays the first
    detrend saw; ``real_idx`` indexes the real points, i.e. the arrays the
    worker hands to BLS. ``detrend`` re-runs a detrender on the full series
    with a real-point transit mask scattered onto it (synthetic -> False) and
    returns the result at the real points only.
    """

    def __init__(self, time, raw_flux, synthetic):
        self.time = np.asarray(time, dtype=np.float64)
        self.raw_flux = np.asarray(raw_flux, dtype=np.float64)
        synthetic = np.asarray(synthetic, dtype=bool)
        if not (len(self.time) == len(self.raw_flux) == len(synthetic)):
            raise ValueError("DetrendFrame: time/raw_flux/synthetic lengths differ")
        self.real_idx = np.flatnonzero(~synthetic)
        self.n_synthetic = int(synthetic.sum())

    def expand_mask(self, mask_real):
        mask_real = np.asarray(mask_real, dtype=bool)
        if mask_real.shape != self.real_idx.shape:
            raise ValueError(
                f"DetrendFrame: mask shape {mask_real.shape} != "
                f"{self.real_idx.shape} real points")
        full = np.zeros(len(self.time), dtype=bool)
        full[self.real_idx] = mask_real
        return full

    def detrend(self, detrend_fn, transit_mask=None, **kwargs):
        """``detrend_fn(time_full, raw_full, transit_mask=..., **kwargs)[real]``."""
        full_mask = None if transit_mask is None else self.expand_mask(transit_mask)
        flat = detrend_fn(self.time, self.raw_flux, transit_mask=full_mask, **kwargs)
        return np.asarray(flat)[self.real_idx]
