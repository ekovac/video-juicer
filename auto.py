"""`vj auto`: script the common verb chain end to end.

  align → resolve → (OCR-escalate unverifiable discs, if a VLM is up AND the
  show captions titles) → resolve → elimination → resolve

Writes only evidence and *proposals* — it never confirms. A human/agent still
adjudicates whatever `gaps` surfaces afterward. The OCR-escalation gate mirrors
the old --auto policy: escalate a disc only when its episode ORDER can't be
verified from metadata (`assess_ordering`), a VLM is reachable, and a cheap
2-playlist probe finds on-screen titles. Verifiable discs (DVD with disc hints,
runtime-separable, play-all-corroborated) never pay for OCR.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import compute
import review
import state
from discs import Assignment
from identify import assess_ordering, probe_card_presence, vlm_available


def _assignments_by_disc(conn):
    """{disc_id: [Assignment]} (sorted by play order) from current proposals,
    with per-title delta pulled from runtime-align evidence so assess_ordering
    sees real alignment deltas."""
    discs = {r["id"]: state.load_disc(conn, r["id"]) for r in state.list_discs(conn)}
    out: dict[int, list[Assignment]] = {did: [] for did in discs}
    rows = conn.execute(
        "SELECT a.episode_ids_json, e.payload_json, t.disc_id, t.title_number "
        "FROM assignment a JOIN title t ON t.id=a.title_id "
        "LEFT JOIN evidence e ON e.title_id=t.id AND e.category='runtime-align' "
        "WHERE a.status IN ('proposed','confirmed')").fetchall()
    for r in rows:
        eids = json.loads(r["episode_ids_json"])
        if not eids:
            continue
        eps = [state.episode_by_id(conn, x) for x in eids]
        disc = discs[r["disc_id"]]
        title = next(t for t in disc.titles if t.id == r["title_number"])
        delta = (json.loads(r["payload_json"]).get("delta", 0.0)
                 if r["payload_json"] else 0.0)
        out[r["disc_id"]].append(Assignment(disc, title, eps, delta, "medium"))
    for did in out:
        out[did].sort(key=lambda a: a.title.order_key)
    return discs, out


def _ocr_plan(conn, args) -> tuple[list[int], str]:
    """Which disc ids to OCR, and why (the escalation decision)."""
    disc_ids = [r["id"] for r in state.list_discs(conn)]
    if args.ocr == "always":
        return disc_ids, "ocr=always"
    if args.ocr == "never":
        return [], "ocr=never"
    # auto: escalate only unverifiable-order discs, gated on VLM + card presence
    discs, by_disc = _assignments_by_disc(conn)
    unverifiable = []
    for did, asgs in by_disc.items():
        if not asgs:
            continue
        ok, why = assess_ordering(discs[did], asgs)
        if not ok:
            unverifiable.append((did, why))
    if not unverifiable:
        return [], "all disc orders verifiable from metadata"
    if not vlm_available(args.vlm_model, args.ollama_host):
        return [], f"{len(unverifiable)} disc(s) order-unverifiable but no VLM at " \
                   f"{args.ollama_host} — kept metadata mapping (flagged)"
    seasons, _ = compute._season_pools(conn)
    with tempfile.TemporaryDirectory(prefix="vj-probe-", dir=args.scratch_dir) as tmp:
        present = probe_card_presence(list(discs.values()), seasons, args, Path(tmp))
    if not present:
        return [], f"{len(unverifiable)} disc(s) order-unverifiable but no " \
                   f"on-screen titles found — kept metadata mapping (flagged)"
    return [did for did, _ in unverifiable], \
        f"escalating {len(unverifiable)} unverifiable disc(s); cards present"


def run_auto(conn, args) -> dict:
    if not state.list_discs(conn):
        return {"ok": False, "error": "no-discs",
                "message": "no discs scanned yet (run `scan`)"}

    steps = []
    al = compute.run_align(conn, args)
    steps.append({"step": "align", "evidence": al.get("evidence"),
                  "leftovers": al.get("leftovers")})
    steps.append({"step": "resolve", **review.resolve(conn, args.threshold)})

    ocr_discs, reason = _ocr_plan(conn, args)
    steps.append({"step": "ocr-policy", "reason": reason, "discs": ocr_discs})
    if ocr_discs:
        for did in ocr_discs:
            args.title, args.disc = None, did
            oc = compute.run_ocr(conn, args)
            steps.append({"step": "ocr", "disc": did,
                          "titles": len(oc.get("ocr", []))})
        steps.append({"step": "resolve", **review.resolve(conn, args.threshold)})

    el = compute.run_elimination(conn, args)
    steps.append({"step": "elimination", "recovered": el.get("recovered", 0)})
    if el.get("recovered"):
        steps.append({"step": "resolve", **review.resolve(conn, args.threshold)})

    summary = review.summarize(conn)
    g = review.gaps(conn, args.threshold)
    return {"ok": True, "steps": steps, "seasons": summary["seasons"],
            "assignments_by_status": summary["assignments_by_status"],
            "n_gaps": g["n_gaps"], "n_missing": g["n_missing"]}
