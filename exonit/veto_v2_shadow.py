"""EXONIT_VETO_V2=shadow: side report comparing the v1-driven strong/finalist lists with what
veto v2 would have produced.  Pure functions (no DB, no network); the postprocess glue lives in
scripts/discovery_postprocess.py::_veto_v2_shadow_step and wraps everything in try/except so a
failure here can never fail the postprocess.

Never touches the DB or the dossier: it only writes ``<stem>_veto_v2_shadow.{json,md}``.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional

from exonit.systematic_veto import is_vetoed
import exonit.systematic_veto_v2 as v2

BANDS = [(0.0, 1.0, "<1 d"), (1.0, 2.0, "1-2 d"), (2.0, 5.0, "2-5 d"), (5.0, 10.0, "5-10 d"),
         (10.0, 20.0, "10-20 d"), (20.0, 30.0, "20-30 d"), (30.0, 1e9, ">=30 d")]


def band_of(p: Optional[float]) -> str:
    if p is None or not math.isfinite(p):
        return "?"
    for lo, hi, name in BANDS:
        if lo <= p < hi:
            return name
    return "?"


def _kw(row: Dict[str, Any]) -> Dict[str, Any]:
    return dict(duration=row.get("duration_days"), depth=row.get("depth_ppm"),
                snr=row.get("snr"), sector=row.get("sector"))


def _layer_kind(s: Dict[str, Any]) -> str:
    if s.get("kind") == "t0_events":
        return "t0_event"
    return "density_family_agnostic" if s.get("epoch_agnostic") else "density_family_epoch_cluster"


def explain_v2(row: Dict[str, Any], sysp: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Every v2 layer that vetoes or soft-tags this row (not just the first).

    ``decision``/``kind``/``family_period`` describe the first veto layer (soft if no veto), for
    compatibility; ``layers`` lists ALL matching layers, ``veto_kinds`` the distinct vetoing
    kinds (density_family_agnostic / density_family_epoch_cluster / t0_event) and ``soft`` the
    soft-tag kinds.
    """
    layers: List[Dict[str, Any]] = []
    for s in sysp:
        if not s.get("v2"):
            continue
        d = v2.family_decision_v2(s, row["period_days"], row.get("t0"), row.get("duration_days"),
                                  row.get("depth_ppm"), row.get("snr"), row.get("sector"))
        if d is None:
            continue
        layers.append({"decision": d, "kind": _layer_kind(s), "family_period": s.get("period"),
                       "family_sector": s.get("sector"), "ratio": s.get("ratio")})
    vetoes = [x for x in layers if x["decision"] == "veto"]
    softs = [x for x in layers if x["decision"] == "soft"]
    first = (vetoes or softs or [None])[0]
    out = dict(first) if first else {"decision": "pass", "kind": None}
    out["layers"] = layers
    out["veto_kinds"] = sorted({x["kind"] for x in vetoes})
    out["soft_kinds"] = sorted({x["kind"] for x in softs})
    return out


def explain_v1(row: Dict[str, Any], sysp: List[Dict[str, Any]]) -> Dict[str, Any]:
    f = is_vetoed(row["period_days"], row.get("t0"), sysp, **_kw(row))
    if f is None:
        return {"decision": "pass", "kind": None}
    return {"decision": "veto", "kind": "v1_family_epoch_agnostic" if f.get("epoch_agnostic") else "v1_family_epoch_phase",
            "family_period": f.get("period")}


def _brief(r: Dict[str, Any]) -> Dict[str, Any]:
    return {k: r.get(k) for k in ("id", "tic_id", "sector", "period_days", "t0", "snr", "depth_ppm",
                                  "exominer_ffi_score", "claude_verdict", "cross_source_name")}


def build_report(run_id: str, best_v1: Dict[str, Dict], best_v2: Dict[str, Dict], final_v1: Iterable[Dict],
                 sysp_v1: List[Dict], sysp_v2: List[Dict], *, n_real_survivors: Optional[int] = None,
                 note: str = "", meta: Optional[Dict[str, Any]] = None,
                 best_v2_noT0: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
    """best_*: {tic_id: strong row} under v1 / v2 on the SAME pool; final_v1: the real finalists."""
    gained, lost = [], []
    # a TIC strong in both arms but with a different representative row is a REPLACEMENT, not a gain/loss
    n_replaced = sum(1 for t, r in best_v2.items() if t in best_v1 and best_v1[t]["id"] != r["id"])
    for t, r in best_v2.items():
        if t not in best_v1:
            e = dict(_brief(r), band=band_of(r["period_days"]),
                     v1_reason=explain_v1(r, sysp_v1), v2_soft=explain_v2(r, sysp_v2).get("decision") == "soft",
                     replaces_row=None)
            gained.append(e)
    for t, r in best_v1.items():
        if t not in best_v2:
            e = dict(_brief(r), band=band_of(r["period_days"]), v2_reason=explain_v2(r, sysp_v2),
                     replaced_by_row=None)
            lost.append(e)
    lost_by_layer: Dict[str, int] = {}
    lost_only_by_layer: Dict[str, int] = {}
    for e in lost:
        vk = e["v2_reason"].get("veto_kinds", [])
        for k in vk:
            lost_by_layer[k] = lost_by_layer.get(k, 0) + 1
        if len(vk) == 1:
            lost_only_by_layer[vk[0]] = lost_only_by_layer.get(vk[0], 0) + 1
    marginal = None
    if best_v2_noT0 is not None:
        # TICs the T0-event layer alone changes: strong without the layer but not with it (it vetoes
        # them), and the reverse (dedup/representative effects)
        removed = [r for t, r in best_v2_noT0.items() if t not in best_v2]
        added = [r for t, r in best_v2.items() if t not in best_v2_noT0]
        def _pb(rows):
            out: Dict[str, int] = {name: 0 for _, _, name in BANDS}
            for r in rows:
                out[band_of(r["period_days"])] = out.get(band_of(r["period_days"]), 0) + 1
            return out
        marginal = {"n_strong_v2_noT0": len(best_v2_noT0), "n_strong_v2_full": len(best_v2),
                    "removed_by_t0_layer": len(removed), "added_by_t0_layer": len(added),
                    "per_band_removed_by_t0_layer": _pb(removed), "per_band_added_by_t0_layer": _pb(added)}
    ev = [x for x in sysp_v2 if x.get("kind") == "t0_events"]
    calib = ({"t0_k": ev[0].get("t0_k"), "t0_event_cells": ev[0].get("n_stars"),
              "t0_event_classes": len(ev[0].get("t0_cells", {}))} if ev else {"t0_k": None})
    fin_ids = {f["id"] for f in final_v1}
    fin_tics = {f["tic_id"] for f in final_v1}
    fin_lost = [e for e in lost if e["id"] in fin_ids]
    # finalists whose TIC is no longer strong under v2 at all
    fin_removed_tics = sorted(t for t in fin_tics if t not in best_v2)
    def per_band(rows):
        out: Dict[str, int] = {}
        for e in rows:
            out[e["band"]] = out.get(e["band"], 0) + 1
        return {name: out.get(name, 0) for _, _, name in BANDS}
    def per_reason(rows, key):
        out: Dict[str, int] = {}
        for e in rows:
            k = (e[key] or {}).get("kind") or "none"
            out[k] = out.get(k, 0) + 1
        return out
    return {
        "run_id": run_id, "mode": "shadow (v1 drives; nothing written to DB or dossier)",
        "note": note,
        "n_strong_v1_recomputed": len(best_v1), "n_strong_v2": len(best_v2),
        "n_strong_v1_real_survivors": n_real_survivors, "n_finalists_v1": len(fin_tics),
        "gained_strong_tics": len(gained), "lost_strong_tics": len(lost), "replaced_representative_row": n_replaced,
        "calibration": calib, "cost": meta or {},
        "arms": {"n_strong_v1": len(best_v1), "n_strong_v2": len(best_v2),
                 "n_strong_v2_noT0": None if best_v2_noT0 is None else len(best_v2_noT0)},
        "t0_layer_marginal": marginal,
        "lost_by_layer": lost_by_layer, "lost_only_by_layer": lost_only_by_layer,
        "finalists_v1_that_v2_would_lose": len(fin_lost), "finalist_tics_no_longer_strong_in_v2": fin_removed_tics,
        "gained_per_band": per_band(gained), "lost_per_band": per_band(lost),
        "gained_by_v1_reason": per_reason(gained, "v1_reason"), "lost_by_v2_reason": per_reason(lost, "v2_reason"),
        "gained": gained, "lost": lost,
        "finalists_note": "v2 finalists are not recomputed (layer 3 / FPP / MAST are not re-run in shadow mode); "
                          "'gained' rows have unknown finalist status.",
    }


def render_md(rep: Dict[str, Any], limit: int = 60) -> str:
    L = [f"# Veto v2 shadow report: {rep['run_id']}", "",
         f"Mode: {rep['mode']}.", rep.get("note") or "", "",
         f"Strong TICs: v1 (recomputed on the same pool) {rep['n_strong_v1_recomputed']}, v2 {rep['n_strong_v2']}; "
         f"real v1 survivors {rep['n_strong_v1_real_survivors']}; v1 finalists {rep['n_finalists_v1']}.",
         f"v2 would **add {rep['gained_strong_tics']}** and **remove {rep['lost_strong_tics']}** strong TICs "
         f"({rep['replaced_representative_row']} TICs stay strong with another representative row); "
         f"{rep['finalists_v1_that_v2_would_lose']} v1 finalist row(s) would be lost "
         f"(finalist TICs no longer strong: {', '.join(rep['finalist_tics_no_longer_strong_in_v2']) or 'none'}).", "",
         rep["finalists_note"], "",
         f"Calibration used by this report: {rep['calibration']}; cost: {rep['cost']}.", "", "## Per period band", "", "| band | gained | lost |", "|---|---|---|"]
    for _, _, name in BANDS:
        L.append(f"| {name} | {rep['gained_per_band'][name]} | {rep['lost_per_band'][name]} |")
    L += ["", f"Gained by why v1 vetoed them: {rep['gained_by_v1_reason']}", f"Lost by v2 reason: {rep['lost_by_v2_reason']}",
          f"Lost rows by ANY vetoing layer: {rep['lost_by_layer']}; by exactly one layer: {rep['lost_only_by_layer']}", ""]
    m = rep.get("t0_layer_marginal")
    if m:
        L += ["## T0-event layer, marginal effect (arm v2 without the layer vs full v2)", "",
              f"Strong TICs: v2 without T0 layer {m['n_strong_v2_noT0']}, full v2 {m['n_strong_v2_full']}; "
              f"the layer removes {m['removed_by_t0_layer']} TICs and adds {m['added_by_t0_layer']}.", "",
              "| band | removed by T0 layer | added |", "|---|---|---|"]
        for _, _, name in BANDS:
            L.append(f"| {name} | {m['per_band_removed_by_t0_layer'].get(name, 0)} | {m['per_band_added_by_t0_layer'].get(name, 0)} |")
        L.append("")
    for title, key, rkey in (("Removed by v2", "lost", "v2_reason"), ("Added by v2", "gained", "v1_reason")):
        rows = sorted(rep[key], key=lambda e: -(e.get("exominer_ffi_score") or 0))
        L += [f"## {title} ({len(rows)}; top {min(limit, len(rows))} by em_ffi)", "",
              "| TIC | sector | P (d) | SNR | depth (ppm) | em_ffi | reason |", "|---|---|---|---|---|---|---|"]
        for e in rows[:limit]:
            rs = e.get(rkey) or {}
            why = (",".join(rs.get("veto_kinds") or []) or f"{rs.get('kind')}") + (f" P~{rs['family_period']}" if rs.get("family_period") else "") + (" (v2 soft)" if e.get("v2_soft") else "")
            L.append(f"| {e['tic_id']} | {e['sector']} | {e['period_days']:.4f} | {e['snr']} | {e['depth_ppm']} | "
                     f"{(e.get('exominer_ffi_score') or 0):.2f} | {why} |")
        L.append("")
    return "\n".join(L) + "\n"


def write_report(rep: Dict[str, Any], stem: str) -> List[str]:
    jp, mp = f"{stem}_veto_v2_shadow.json", f"{stem}_veto_v2_shadow.md"
    with open(jp, "w") as fh:
        json.dump(rep, fh, indent=1, default=str)
    with open(mp, "w") as fh:
        fh.write(render_md(rep))
    return [jp, mp]


def failure_stub(stem: str, stage: str, exc: BaseException) -> List[str]:
    """Written when the shadow code fails: the run keeps going, but the missing evidence is visible."""
    jp, mp = f"{stem}_veto_v2_shadow.json", f"{stem}_veto_v2_shadow.md"
    rep = {"error": repr(exc), "stage": stage,
           "note": "EXONIT_VETO_V2=shadow FAILED for this run; postprocess result (DB, dossier) is unaffected."}
    with open(jp, "w") as fh:
        json.dump(rep, fh, indent=1)
    with open(mp, "w") as fh:
        fh.write(f"# Veto v2 shadow report: FAILED\n\nStage `{stage}`: `{exc!r}`. "
                 "The postprocess result (DB, dossier) is unaffected; this lot has no shadow evidence.\n")
    return [jp, mp]
