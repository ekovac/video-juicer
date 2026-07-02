"""Inspect, adjudicate, and resolve — the human/agent-facing layer.

- inspect (read-only): `status` summary, `gaps` worklist, `show` one thing.
  Conflict is COMPUTED here (two evidence categories naming different episodes),
  never stored — see DESIGN.md.
- resolve: materialise `proposed` assignments from current evidence, per an
  explainable policy. This is the bridge between pure evidence and the
  assignment layer; it never overwrites a human/agent decision.
- adjudicate: `assign` / `confirm` / `reject` set the sticky decided answer.
"""
from __future__ import annotations

import json
import sqlite3

import state
from discs import Assignment
from identify import assess_ordering


def _disc_assignments(conn):
    """{disc_id: (Disc, [Assignment sorted by play order])} from current
    proposed/confirmed assignments, with per-title delta from runtime-align
    evidence — the shape `assess_ordering` needs."""
    discs = {r["id"]: state.load_disc(conn, r["id"]) for r in state.list_discs(conn)}
    out = {did: [] for did in discs}
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
        title = next((t for t in disc.titles if t.id == r["title_number"]), None)
        if title is None:
            continue
        delta = (json.loads(r["payload_json"]).get("delta", 0.0)
                 if r["payload_json"] else 0.0)
        out[r["disc_id"]].append(Assignment(disc, title, eps, delta, "medium"))
    for did in out:
        out[did].sort(key=lambda a: a.title.order_key)
    return discs, out


def order_warnings(conn) -> list[dict]:
    """Discs whose episode ORDER can't be corroborated from metadata alone
    (`assess_ordering`) — so their proposals are position guesses that a second
    evidence source (OCR) should confirm. Surfaced, not acted on."""
    discs, by_disc = _disc_assignments(conn)
    warns = []
    for did, asgs in by_disc.items():
        if len(asgs) <= 1:
            continue
        ok, reason = assess_ordering(discs[did], asgs)
        if not ok:
            warns.append({"disc_id": did, "disc": discs[did].label,
                          "titles": len(asgs), "reason": reason})
    return warns


def _season_medians(conn) -> dict[int, float]:
    """Median episode runtime per season (for the episode-length anomaly)."""
    med = {}
    for s in conn.execute("SELECT DISTINCT season FROM episode WHERE season>0"):
        rts = [r["runtime"] for r in conn.execute(
            "SELECT runtime FROM episode WHERE season=? AND runtime IS NOT NULL",
            (s["season"],))]
        if rts:
            med[s["season"]] = sorted(rts)[len(rts) // 2]
    return med


def _episode_length(duration: float, medians: dict[int, float],
                    tol: float = 0.15) -> bool:
    """Is a title's duration ~a single episode (or clean multiple) of some
    season's median? Catches a dropped/unmatched episode masquerading as an
    extra."""
    for m in medians.values():
        for k in (1, 2, 3):
            if abs(duration - k * m) <= tol * m:
                return True
    return False


def _sxxeyy(season, number) -> str:
    return f"S{season:02d}E{number:02d}"


def _title_label(conn, tid: int) -> dict:
    r = conn.execute(
        "SELECT t.id, t.title_number, t.duration, t.kind, d.id AS disc_id, "
        "d.label, d.path FROM title t JOIN disc d ON d.id=t.disc_id "
        "WHERE t.id=?", (tid,)).fetchone()
    return {"title_id": r["id"], "disc_id": r["disc_id"], "disc": r["label"],
            "title_number": r["title_number"], "kind": r["kind"],
            "minutes": round(r["duration"] / 60, 1)}


def _evidence_view(conn, tid: int) -> list[dict]:
    out = []
    for e in state.evidence_for_title(conn, tid):
        out.append({
            "category": e["category"],
            "episode": (_sxxeyy(e["ep_season"], e["ep_number"])
                        if e["episode_id"] else None),
            "verdict": e["verdict"], "confidence": e["confidence"],
        })
    return out


def _conflict(conn, tid: int, threshold: float) -> tuple[bool, set]:
    """Do two evidence categories name different episodes above threshold?"""
    eps = {e["episode_id"] for e in state.evidence_for_title(conn, tid)
           if e["episode_id"] is not None
           and (e["confidence"] or 0) >= threshold}
    return (len(eps) > 1, eps)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def summarize(conn) -> dict:
    proj = state.get_project(conn)
    # per-season episode coverage (an episode is "covered" if some assignment,
    # proposed or confirmed, claims it)
    covered: dict[int, set] = {}
    by_status: dict[str, int] = {}
    for a in conn.execute("SELECT * FROM assignment"):
        by_status[a["status"]] = by_status.get(a["status"], 0) + 1
        if a["status"] in ("proposed", "confirmed"):
            for eid in json.loads(a["episode_ids_json"]):
                ep = conn.execute("SELECT season,number FROM episode WHERE id=?",
                                  (eid,)).fetchone()
                if ep:
                    covered.setdefault(ep["season"], set()).add(ep["number"])
    totals: dict[int, int] = {}
    for ep in conn.execute("SELECT season, COUNT(*) c FROM episode GROUP BY season"):
        totals[ep["season"]] = ep["c"]
    seasons = [{"season": s, "matched": len(covered.get(s, set())), "total": totals[s]}
               for s in sorted(totals)]
    n_titles = conn.execute("SELECT COUNT(*) c FROM title").fetchone()["c"]
    return {"ok": True, "show": proj.get("show_name"),
            "titles": n_titles, "assignments_by_status": by_status,
            "seasons": seasons, "order_warnings": order_warnings(conn)}


# ---------------------------------------------------------------------------
# gaps — the worklist
# ---------------------------------------------------------------------------


def gaps(conn, threshold: float = 0.5) -> dict:
    """Titles needing attention + episodes still missing.

    A title is a gap if it has no sticky/proposed assignment, or its evidence
    conflicts. Confirmed titles with agreeing evidence drop out. An episode-
    length title left unclaimed is flagged as an anomaly (a dropped/unmatched
    episode — the tell of a same-runtime alignment shift)."""
    medians = _season_medians(conn)
    title_gaps = []
    for t in conn.execute("SELECT id FROM title ORDER BY id"):
        tid = t["id"]
        ev = _evidence_view(conn, tid)
        a = state.get_assignment(conn, tid)
        status = a["status"] if a else "unresolved"
        conflict, eps = _conflict(conn, tid, threshold)
        if status == "confirmed" and not conflict:
            continue
        if status == "rejected":
            continue
        if status == "proposed" and not conflict:
            continue          # a clean proposal isn't a gap until reviewed
        lbl = _title_label(conn, tid)
        if not ev and status == "unresolved":
            # no evidence at all: only a gap if it's a plausible episode title
            if lbl["kind"] not in ("episode-candidate", "unknown"):
                continue
        # an unclaimed episode-length candidate is a high-signal anomaly
        unclaimed = status in ("unresolved",) and not (
            a and json.loads(a["episode_ids_json"]))
        anomaly = (unclaimed and lbl["kind"] == "episode-candidate"
                   and _episode_length(lbl["minutes"] * 60, medians))
        if conflict:
            reason = "conflict: evidence names >1 episode"
        elif anomaly:
            reason = ("episode-length title unclaimed — likely a dropped/shifted "
                      "episode; corroborate this disc with `run ocr`")
        elif not ev:
            reason = "no evidence yet"
        else:
            reason = f"{status}, awaiting decision"
        lbl.update(status=status, conflict=conflict, anomaly=bool(anomaly),
                   reason=reason, evidence=ev)
        title_gaps.append(lbl)

    # episodes claimed by no assignment
    claimed: set = set()
    for a in conn.execute("SELECT episode_ids_json FROM assignment WHERE status "
                          "IN ('proposed','confirmed')"):
        claimed.update(json.loads(a["episode_ids_json"]))
    missing = [_sxxeyy(e["season"], e["number"]) for e in conn.execute(
        "SELECT id,season,number FROM episode WHERE season>0 ORDER BY season,number")
        if e["id"] not in claimed]

    warns = order_warnings(conn)
    # sort anomalies + conflicts to the top of the worklist
    title_gaps.sort(key=lambda g: (not g.get("anomaly"), not g["conflict"], g["title_id"]))
    return {"ok": True, "gaps": title_gaps, "missing_episodes": missing,
            "order_warnings": warns, "n_gaps": len(title_gaps),
            "n_missing": len(missing)}


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


def show_title(conn, tid: int) -> dict:
    if conn.execute("SELECT 1 FROM title WHERE id=?", (tid,)).fetchone() is None:
        return {"ok": False, "error": "no-title", "message": f"no title {tid}"}
    lbl = _title_label(conn, tid)
    a = state.get_assignment(conn, tid)
    assignment = None
    if a:
        eps = [_sxxeyy(*conn.execute("SELECT season,number FROM episode WHERE id=?",
                                     (eid,)).fetchone())
               for eid in json.loads(a["episode_ids_json"])]
        assignment = {"episodes": eps, "status": a["status"],
                      "decided_by": a["decided_by"], "note": a["note"]}
    lbl.update(assignment=assignment, evidence=_evidence_view(conn, tid),
               frames=state.frame_categories(conn, tid))
    lbl["ok"] = True
    return lbl


def show_episode(conn, season: int, number: int) -> dict:
    eid = state.episode_id(conn, season, number)
    if eid is None:
        return {"ok": False, "error": "no-episode",
                "message": f"no episode {_sxxeyy(season, number)}"}
    ep = conn.execute("SELECT * FROM episode WHERE id=?", (eid,)).fetchone()
    # titles with evidence or an assignment pointing at this episode
    titles = []
    for t in conn.execute("SELECT DISTINCT title_id FROM evidence WHERE episode_id=?",
                          (eid,)):
        titles.append(show_title(conn, t["title_id"]))
    return {"ok": True, "episode": _sxxeyy(season, number), "name": ep["name"],
            "runtime": ep["runtime"], "titles": titles}


# ---------------------------------------------------------------------------
# resolve — evidence -> proposed assignments (never overwrites a decision)
# ---------------------------------------------------------------------------


def resolve(conn, threshold: float = 0.5) -> dict:
    """Propose an assignment for each title whose evidence agrees.

    Policy: among a title's evidence rows that name an episode at confidence
    >= threshold, if they all name the SAME episode, propose it (decided_by
    consensus, or the single source). If they disagree it's a conflict — leave
    it unresolved for `gaps` to surface. Sticky human/agent/confirmed/rejected
    assignments are never touched."""
    proposed = 0
    skipped_conflict = 0
    for t in conn.execute("SELECT id FROM title ORDER BY id"):
        tid = t["id"]
        a = state.get_assignment(conn, tid)
        if a and (a["decided_by"] in ("human", "agent")
                  or a["status"] in ("confirmed", "rejected")):
            continue  # sticky decision — hands off
        rows = [e for e in state.evidence_for_title(conn, tid)
                if e["episode_id"] is not None
                and (e["confidence"] or 0) >= threshold]
        if not rows:
            continue
        eps = {e["episode_id"] for e in rows}
        if len(eps) > 1:
            skipped_conflict += 1
            continue
        best = max(rows, key=lambda e: e["confidence"] or 0)
        episode_ids = [best["episode_id"]]
        # two-parter carried in a runtime-align payload
        payload = json.loads(best["payload_json"] or "{}")
        if best["category"] == "runtime-align" and len(payload.get("episodes", [])) == 2:
            season = payload["season"]
            episode_ids = [state.episode_id(conn, season, n)
                           for n in payload["episodes"]]
            episode_ids = [e for e in episode_ids if e]
        cats = {e["category"] for e in rows}
        decided = ("heuristic:consensus" if len(cats) > 1
                   else f"heuristic:{best['category']}")
        state.set_assignment(conn, tid, episode_ids, status="proposed",
                             decided_by=decided)
        proposed += 1
    return {"ok": True, "proposed": proposed, "conflicts": skipped_conflict}
