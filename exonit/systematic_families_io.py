"""Serialize / deserialize the systematic-families structure to Parquet.

``exonit.systematic_veto.find_systematic_periods`` returns a list of dicts
(the ``sysp`` list consumed by ``is_vetoed`` / ``find_deep_eclipse_hosts``):

    {"period": float, "n_stars": int, "epoch_agnostic": bool,
     "epoch_phase": float,       # present only for epoch_agnostic=False entries
     "concentration": float}     # present only for epoch_agnostic=False entries

This module round-trips that exact structure (including which optional keys
are present/absent — ``is_vetoed`` branches on ``s.get("epoch_agnostic")``
and only reads ``epoch_phase``/``concentration`` on the non-agnostic path, so
a missing key there is never spurious). Written for
docs/superpowers/specs/2026-07-11-injection-recovery-e2e.md deliverable 2
(``discovery_postprocess.py --systematic-periods``): the injection-recovery
harness freezes prod's systematic families at export time
(``scripts/eval/export_systematic_families.py``) so a synthetic-candidate run
against an isolated DB evaluates against the SAME campaign-wide veto set as
production, without re-running ``find_systematic_periods`` over a population
contaminated by synthetics.

Order is preserved via an explicit ``_order`` column rather than relying on
row order surviving a Parquet round-trip (not guaranteed by the format/engine
in general, even though pyarrow/pandas usually do preserve it) — this matters
because ``is_vetoed`` returns the FIRST matching systematic in list order, so
two systematics that could both match the same (period, t0) must keep their
relative order for a byte-identical veto decision after reload.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd

# Column order in the persisted Parquet file. ``_order`` is dropped from the
# reconstructed dicts (it is bookkeeping, not part of the sysp entry shape).
_COLUMNS = ["_order", "period", "n_stars", "epoch_agnostic", "epoch_phase", "concentration"]
# Veto-v2 (EXONIT_VETO_V2) families carry extra optional keys; the extra columns are
# written ONLY when some entry has ``"v2"`` so v1 files stay byte-identical.
_V2_FLOAT_KEYS = ["sector", "t0_k", "ratio", "p_lo", "p_hi", "t0_mass", "dur_lo", "dur_hi",
                  "depth_lo", "depth_hi"]


def dump_families(sysp: List[Dict[str, Any]], path: str) -> None:
    """Write *sysp* (output of ``find_systematic_periods``) to a Parquet file at *path*.

    Preserves list order (via ``_order``) and which of the optional
    ``epoch_phase``/``concentration`` keys are present on each entry (absent
    -> NaN in the Parquet column, restored to "key absent" by ``load_families``).
    """
    rows = []
    is_v2 = any(s.get("v2") for s in sysp)
    for i, s in enumerate(sysp):
        rows.append({
            "_order": i,
            "period": float(s["period"]),
            "n_stars": int(s["n_stars"]),
            "epoch_agnostic": bool(s.get("epoch_agnostic", False)),
            "epoch_phase": float(s["epoch_phase"]) if "epoch_phase" in s and s["epoch_phase"] is not None else None,
            "concentration": float(s["concentration"]) if "concentration" in s and s["concentration"] is not None else None,
        })
        if is_v2:
            rows[-1]["v2"] = bool(s.get("v2", False))
            for k in _V2_FLOAT_KEYS:
                rows[-1][k] = float(s[k]) if s.get(k) is not None else None
            rows[-1]["t0_modes"] = json.dumps(s["t0_modes"]) if s.get("t0_modes") else None
            rows[-1]["kind"] = s.get("kind")
            rows[-1]["t0_cells"] = json.dumps(s["t0_cells"]) if s.get("t0_cells") else None
    df = pd.DataFrame(rows, columns=_COLUMNS + (["v2"] + _V2_FLOAT_KEYS + ["t0_modes", "kind", "t0_cells"] if is_v2 else []))
    # Explicit dtypes so an empty *sysp* still produces a schema-stable file
    # (pandas infers "object" for all-NaN/empty columns otherwise, which some
    # Parquet engines round-trip fine but is not worth relying on).
    if not df.empty:
        df["_order"] = df["_order"].astype("int64")
        df["n_stars"] = df["n_stars"].astype("int64")
        df["epoch_agnostic"] = df["epoch_agnostic"].astype("bool")
    df.to_parquet(path, index=False)


def load_families(path: str) -> List[Dict[str, Any]]:
    """Read a Parquet file written by ``dump_families`` back into a sysp list.

    Returns entries in the ORIGINAL order (see ``_order`` note above), with
    ``epoch_phase``/``concentration`` present iff they were present at dump
    time — matching ``find_systematic_periods``'s own output shape exactly so
    ``is_vetoed`` / ``find_deep_eclipse_hosts`` behave identically whether fed
    a live or a frozen (loaded) family set.
    """
    df = pd.read_parquet(path)
    if "_order" in df.columns:
        df = df.sort_values("_order")
    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        entry: Dict[str, Any] = {
            "period": float(row["period"]),
            "n_stars": int(row["n_stars"]),
            "epoch_agnostic": bool(row["epoch_agnostic"]),
        }
        ep = row.get("epoch_phase")
        if ep is not None and pd.notna(ep):
            entry["epoch_phase"] = float(ep)
        conc = row.get("concentration")
        if conc is not None and pd.notna(conc):
            entry["concentration"] = float(conc)
        if "v2" in df.columns and bool(row.get("v2")):
            entry["v2"] = True
            for k in _V2_FLOAT_KEYS:
                v = row.get(k)
                if v is not None and pd.notna(v):
                    entry[k] = int(v) if k == "sector" else float(v)
            tm = row.get("t0_modes")
            if tm is not None and isinstance(tm, str):
                entry["t0_modes"] = json.loads(tm)
            kd = row.get("kind")
            if isinstance(kd, str):
                entry["kind"] = kd
            tc = row.get("t0_cells")
            if isinstance(tc, str):
                entry["t0_cells"] = json.loads(tc)
        out.append(entry)
    return out
