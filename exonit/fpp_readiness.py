"""Fail-closed population readiness and counted parking for FPP dispatch.

Counts live in SharedState (Redis in production), not candidate FPP values.
A changed document or R8 result creates a new refusal identity and permits
recovery. No producer of Gaia stellar tables exists yet: unresolved alarms
are terminal for that input, never a fabricated FPP or a -1 target sentinel.
"""
from __future__ import annotations

import hashlib
import json

REFUSALS_KEY = "fpp:population_refusals"
LEGACY_RELEASES_KEY = "fpp:legacy_releases"
MAX_REFUSALS = 3

#: Refusal reasons that can never resolve on their own (no Gaia stellar-table producer
#: exists): dispatch parks the candidate. They are reported by the readiness gate but
#: must not block the other candidates of a run.
PARKED_REASONS = frozenset({"population_alarm_unresolved"})


def population_refusal(task):
    population = task.get("population")
    if task.get("population_missing") or not isinstance(population, dict):
        return "population_missing"
    crosscheck = task.get("population_crosscheck")
    if not isinstance(crosscheck, dict):
        return "population_crosscheck_unverified"
    if crosscheck.get("alarm") and population.get("header", {}).get("source") != "gaia":
        return "population_alarm_unresolved"
    if crosscheck.get("gaia_available") is not True:
        return "population_crosscheck_unverified"
    return None


def field_refusal(row):
    """Refusal reason of ONE field from its ``fpp_background`` row (``population_version``, ``source``,
    ``crosscheck_json``; a missing store row has ``population_version`` None), or None when servable.

    This is the single place that turns the stored row into a reason; the readiness gate and the
    ``/api/health/work`` drain counter both go through it (via :func:`parked_tics`), so the two can
    never disagree about which fields are parked."""
    try:
        task = {"population": {"header": {"source": row["source"]}} if row["population_version"] else None,
                "population_crosscheck": json.loads(row["crosscheck_json"]) if row["crosscheck_json"] else None}
        return population_refusal(task)
    except Exception:
        return "population_lookup_failed"


def parked_tics(rows):
    """TICs whose field is parked (reason in :data:`PARKED_REASONS`): never served, never resolved."""
    return {str(r["tic_id"]) for r in rows if field_refusal(r) in PARKED_REASONS}


def record_refusal(state, candidate_id, task, reason):
    """Count up to three refusals per candidate/input, surviving releases.

    Keep a visible, terminal reason even if a reconcile rebuilds the stream.
    Concurrent checks can exceed the display cap internally, but can never
    serve a refused input; the recorded counter is atomic across processes.
    """
    identity = json.dumps([task.get("population_version"),
                           task.get("population_sha256"),
                           task.get("population_crosscheck"), reason], sort_keys=True)
    signature = hashlib.sha256(identity.encode()).hexdigest()
    field = f"{candidate_id}:{signature}"
    entry = state.hget_json(REFUSALS_KEY, field) or {}
    if entry.get("attempts", 0) < MAX_REFUSALS:
        state.hupdate_json(REFUSALS_KEY, field, {
            "candidate_id": candidate_id, "reason": reason,
            "input_sha256": signature, "outcome": "parked"})
        state.hincr_json_field(REFUSALS_KEY, field, "attempts", 1)
    return min(MAX_REFUSALS, (state.hget_json(REFUSALS_KEY, field) or {}).get("attempts", 0))


def input_signature(task):
    """Digest of the population input a task was built from (version, document sha, R8 counts).

    A changed document or R8 result changes it, so every counter keyed by it restarts at zero
    and a candidate that was refused for a missing/defective population is served again.
    """
    identity = json.dumps([task.get("population_version"), task.get("population_sha256"),
                           task.get("population_crosscheck")], sort_keys=True)
    return hashlib.sha256(identity.encode()).hexdigest()


def legacy_release_field(candidate_id, signature):
    """Field of the legacy (non-stream) release counter: candidate AND population input."""
    return f"{candidate_id}:{signature or 'unknown'}"
