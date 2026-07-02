"""Compute verbs: the heuristics reframed as on-demand evidence producers.

Each `vj run <heuristic>` here loads state, runs a library heuristic from
`identify` / `discs` / `synopsis`, and records its finding as *evidence* — one
upserted row per (title, category). Compute verbs decide nothing: they never
touch the `assignment` table. Turning evidence into proposals is the job of
`resolve` / the adjudicate verbs (see DESIGN.md).

Categories written here:
  runtime-align   <- align()          (metadata DP alignment over a season)
  title-card-ocr  <- verify_title()   (+ retained frame for review)
  synopsis        <- identify_by_synopsis()
  elimination     <- recover_by_elimination()  (reads assignments; run after resolve)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import state
from discs import Assignment, Disc, Episode, group_discs, log
from identify import align, classify_disc, recover_by_elimination, verify_title
from synopsis import identify_by_synopsis

# Map align's confidence label to a numeric evidence confidence.
_CONF = {"high": 0.9, "medium": 0.6, "low": 0.3}

# Heuristics exposed by `vj run` (name -> one-line description), for --list.
HEURISTICS = {
    "align": "metadata runtime alignment over each season (DP)",
    "ocr": "OCR each title's title card; keeps the winning frame",
    "synopsis": "judge sampled dialogue against TMDB synopses",
    "elimination": "pin a disc's lone unmatched candidate to its one missing "
                   "adjacent episode (run after resolve)",
}


# ---------------------------------------------------------------------------
# loading helpers
# ---------------------------------------------------------------------------


def _load_discs(conn) -> list[tuple[int, Disc]]:
    """(disc_id, Disc) for every scanned disc."""
    return [(r["id"], state.load_disc(conn, r["id"]))
            for r in state.list_discs(conn)]


def _season_pools(conn) -> tuple[dict[int, list[Episode]], list[Episode]]:
    """Episodes grouped by season (>0), plus specials (season 0)."""
    seasons: dict[int, list[Episode]] = {}
    specials: list[Episode] = []
    for e in state.load_episodes(conn):
        (specials if e.season == 0 else seasons.setdefault(e.season, [])).append(e)
    return seasons, specials


def _title_db_id(conn, path2id: dict[str, int], disc: Disc, title) -> int:
    return state.title_id(conn, path2id[str(disc.path)], title.id)


def _pool_for(disc: Disc, seasons, specials, all_eps, include_specials: bool):
    """Episodes a title on `disc` could be. Scoped to the disc's season when it
    has a hint (efficient, and what you want for episodes); widened with the S00
    specials when `include_specials` (for identifying a leftover as a special).
    No hint → the whole series (already includes specials)."""
    base = seasons.get(disc.season_hint)
    if base is None:
        return all_eps
    return list(base) + (specials if include_specials else [])


# ---------------------------------------------------------------------------
# run align — metadata path -> runtime-align evidence
# ---------------------------------------------------------------------------


def run_align(conn, args) -> dict:
    """Classify + align each season's discs together, recording per-title
    runtime-align evidence. Aligns the WHOLE scanned set per season (never a
    subset — a per-disc align double-claims episodes; see CLAUDE.md)."""
    loaded = _load_discs(conn)
    if not loaded:
        return {"ok": False, "error": "no-discs",
                "message": "no discs scanned yet (run `scan`)"}
    path2id = {str(d.path): did for did, d in loaded}
    discs = [d for _, d in loaded]
    seasons, _ = _season_pools(conn)
    all_eps = [e for n in sorted(seasons) for e in seasons[n]]

    n_ev = 0
    n_left = 0
    for season, group in group_discs(discs):
        pool = seasons.get(season) or all_eps
        if not pool:
            continue
        rts = sorted(e.runtime for e in pool if e.runtime)
        expected = rts[len(rts) // 2] if rts else 1320.0
        cands: list[tuple[Disc, object]] = []
        for d in group:
            for t in classify_disc(d, expected):
                cands.append((d, t))
            # persist structural verdicts (kind/order_key) onto title rows
            state.set_classification(conn, path2id[str(d.path)], d.titles)
        if not cands:
            continue
        assignments, leftovers, _missed = align(cands, pool)

        for a in assignments:
            tid = _title_db_id(conn, path2id, a.disc, a.title)
            e0 = a.episodes[0]
            ep_id = state.episode_id(conn, e0.season, e0.number)
            nums = "".join(f"E{e.number:02d}" for e in a.episodes)
            state.put_evidence(
                conn, tid, "runtime-align", episode_id=ep_id,
                verdict=f"S{e0.season:02d}{nums} (Δ{a.delta:.0f}s, {a.confidence})",
                confidence=_CONF.get(a.confidence, 0.3),
                payload={
                    "season": e0.season,
                    "episodes": [e.number for e in a.episodes],
                    "delta": round(a.delta, 1),
                    "confidence_label": a.confidence,
                    "title_kind": a.title.kind,
                },
            )
            n_ev += 1
        # a leftover is a finding too: "align thinks this title is not an episode"
        for d, t in leftovers:
            tid = _title_db_id(conn, path2id, d, t)
            state.put_evidence(
                conn, tid, "runtime-align", episode_id=None,
                verdict="not an episode (leftover)", confidence=0.0,
                payload={"title_kind": t.kind,
                         "duration": round(t.duration, 1)},
            )
            n_left += 1

    return {"ok": True, "evidence": n_ev, "leftovers": n_left,
            "discs": len(discs)}


# ---------------------------------------------------------------------------
# run ocr — title-card OCR -> title-card-ocr evidence (+ retained frame)
# ---------------------------------------------------------------------------


def _ocr_targets(conn, args) -> list[tuple[int, int]]:
    """Resolve --title/--disc/--all into [(disc_id, title_db_id)] to OCR,
    ordered by disc then play order so the adaptive card anchor is learned once
    per disc (grouping --all by disc keeps that per-disc anchor threading)."""
    targets: list[tuple[int, int]] = []
    if args.title:
        for tid in args.title:
            r = conn.execute("SELECT id,disc_id FROM title WHERE id=?", (tid,)).fetchone()
            if r is None:
                raise KeyError(f"no title with id {tid}")
            targets.append((r["disc_id"], r["id"]))
    elif getattr(args, "all", False):
        rows = conn.execute(
            "SELECT id, disc_id FROM title WHERE kind='episode-candidate' "
            "ORDER BY disc_id, order_key, title_number").fetchall()
        targets = [(r["disc_id"], r["id"]) for r in rows]
    elif args.disc is not None:
        rows = conn.execute(
            "SELECT id FROM title WHERE disc_id=? AND kind='episode-candidate' "
            "ORDER BY order_key, title_number", (args.disc,)).fetchall()
        if not rows:   # not classified yet -> fall back to all non-junk titles
            rows = conn.execute(
                "SELECT id FROM title WHERE disc_id=? AND kind!='junk' "
                "ORDER BY order_key, title_number", (args.disc,)).fetchall()
            log.warning("disc %d has no classified candidates; OCRing all "
                        "non-junk titles (run `align` first to narrow)", args.disc)
        targets = [(args.disc, r["id"]) for r in rows]
    return targets


def run_ocr(conn, args) -> dict:
    targets = _ocr_targets(conn, args)
    if not targets:
        return {"ok": False, "error": "no-targets",
                "message": "specify --title <id> (repeatable), --disc <basename|id>, or --all"}
    seasons, specials = _season_pools(conn)
    all_eps = [e for n in sorted(seasons) for e in seasons[n]] + specials

    results = []
    anchors: dict[int, float] = {}   # disc_id -> learned card location
    with tempfile.TemporaryDirectory(prefix="vj-ocr-", dir=args.scratch_dir) as tmp:
        workdir = Path(tmp)
        for disc_id, tid in targets:
            disc = state.load_disc(conn, disc_id)
            title = next(t for t in disc.titles
                         if t.id == conn.execute(
                             "SELECT title_number FROM title WHERE id=?",
                             (tid,)).fetchone()["title_number"])
            pool = _pool_for(disc, seasons, specials, all_eps,
                             getattr(args, "include_specials", False))
            capture: dict = {}
            ep, score, card_t = verify_title(
                disc, title, pool, args.vlm_model, args.ollama_host, workdir,
                accept=args.ocr_accept, anchor=anchors.get(disc_id),
                engine=args.ocr_engine, capture=capture,
                text_filter=getattr(args, "text_filter", True))
            if card_t is not None:
                anchors[disc_id] = card_t

            matched = ep is not None and score >= args.ocr_accept
            ep_id = (state.episode_id(conn, ep.season, ep.number)
                     if matched else None)
            if matched:
                verdict = (f"read {capture.get('text','')!r} -> "
                           f"S{ep.season:02d}E{ep.number:02d} ({score:.2f})")
            elif ep is not None:
                verdict = (f"below accept: best S{ep.season:02d}E{ep.number:02d} "
                           f"({score:.2f})")
            else:
                verdict = "no title text read"
            state.put_evidence(
                conn, tid, "title-card-ocr", episode_id=ep_id, verdict=verdict,
                confidence=round(score, 3),
                payload={
                    "text": capture.get("text"),
                    "card_seconds": round(card_t, 1) if card_t else None,
                    "best_guess": (f"S{ep.season:02d}E{ep.number:02d}"
                                   if ep else None),
                    "accepted": matched, "engine": args.ocr_engine,
                    "has_frame": capture.get("image") is not None,
                },
            )
            if capture.get("image"):   # keep the frame for eyeball / VLM re-check
                state.put_frame(conn, tid, "title-card-ocr", capture["image"],
                                source_time=capture.get("time"),
                                ocr_text=capture.get("text"))
            results.append({"title_id": tid, "verdict": verdict,
                            "confidence": round(score, 3),
                            "frame": capture.get("image") is not None})

    return {"ok": True, "ocr": results}


# ---------------------------------------------------------------------------
# run synopsis — dialogue vs TMDB synopses -> synopsis evidence
# ---------------------------------------------------------------------------


def run_synopsis(conn, args) -> dict:
    targets = _ocr_targets(conn, args)   # same target resolution as OCR
    if not targets:
        return {"ok": False, "error": "no-targets",
                "message": "specify --title <id> (repeatable), --disc <basename|id>, or --all"}
    seasons, specials = _season_pools(conn)
    all_eps = [e for n in sorted(seasons) for e in seasons[n]] + specials

    results = []
    with tempfile.TemporaryDirectory(prefix="vj-syn-", dir=args.scratch_dir) as tmp:
        workdir = Path(tmp)
        for disc_id, tid in targets:
            disc = state.load_disc(conn, disc_id)
            tn = conn.execute("SELECT title_number FROM title WHERE id=?",
                              (tid,)).fetchone()["title_number"]
            title = next(t for t in disc.titles if t.id == tn)
            pool = _pool_for(disc, seasons, specials, all_eps,
                             getattr(args, "include_specials", False))
            ep, conf, detail = identify_by_synopsis(
                disc, title, pool, workdir, args.vlm_model, args.ollama_host)
            ep_id = state.episode_id(conn, ep.season, ep.number) if ep else None
            verdict = (f"S{ep.season:02d}E{ep.number:02d} ({conf:.2f}): {detail}"
                       if ep else f"abstained: {detail}")
            state.put_evidence(
                conn, tid, "synopsis", episode_id=ep_id, verdict=verdict,
                confidence=round(conf, 3),
                payload={"detail": detail,
                         "best_guess": (f"S{ep.season:02d}E{ep.number:02d}"
                                        if ep else None)},
            )
            results.append({"title_id": tid, "verdict": verdict,
                            "confidence": round(conf, 3)})
    return {"ok": True, "synopsis": results}


# ---------------------------------------------------------------------------
# run elimination — recover title-card-less episodes by adjacency -> evidence
# ---------------------------------------------------------------------------


def run_elimination(conn, args) -> dict:
    """Reconstruct the assignment picture, then pin each disc's lone unmatched
    candidate to the one still-missing episode adjacent to its matched run.

    Unlike the other verbs this READS the assignment layer (proposed/confirmed),
    so run it after `resolve`. It still only WRITES evidence (category
    'elimination'); a subsequent `resolve` turns that into a proposal."""
    loaded = _load_discs(conn)
    if not loaded:
        return {"ok": False, "error": "no-discs",
                "message": "no discs scanned yet (run `scan`)"}
    path2id = {str(d.path): did for did, d in loaded}
    disc_by_id = {did: d for did, d in loaded}
    title_of = {(did, t.id): t for did, d in loaded for t in d.titles}

    # current assignments -> `final`; also track which episodes are claimed
    final: list[Assignment] = []
    claimed: set = set()
    rows = conn.execute(
        "SELECT a.episode_ids_json, t.disc_id, t.title_number FROM assignment a "
        "JOIN title t ON t.id=a.title_id "
        "WHERE a.status IN ('proposed','confirmed')").fetchall()
    for a in rows:
        eids = json.loads(a["episode_ids_json"])
        if not eids:
            continue
        eps = [state.episode_by_id(conn, e) for e in eids]
        final.append(Assignment(disc_by_id[a["disc_id"]],
                                title_of[(a["disc_id"], a["title_number"])],
                                eps, 0.0, "medium"))
        claimed.update((e.season, e.number) for e in eps)

    # unmatched episode candidates -> `leftovers`
    left_rows = conn.execute(
        "SELECT t.disc_id, t.title_number FROM title t "
        "LEFT JOIN assignment a ON a.title_id=t.id "
        "WHERE t.kind='episode-candidate' "
        "AND (a.status IS NULL OR a.status='unresolved')").fetchall()
    leftovers = [(disc_by_id[r["disc_id"]],
                  title_of[(r["disc_id"], r["title_number"])]) for r in left_rows]

    # still-missing episodes -> `missed`
    missed = [e for e in state.load_episodes(conn)
              if e.season > 0 and (e.season, e.number) not in claimed]

    if not (final and leftovers and missed):
        return {"ok": True, "recovered": 0, "titles": [],
                "note": "need proposed assignments + unmatched candidates + "
                        "missing episodes (run align + resolve first)"}

    final2, _left, _missed = recover_by_elimination(final, leftovers, missed)
    results = []
    for a in (x for x in final2 if x.method == "elimination"):
        did = path2id[str(a.disc.path)]
        tid = state.title_id(conn, did, a.title.id)
        e0 = a.episodes[0]
        eid = state.episode_id(conn, e0.season, e0.number)
        verdict = (f"S{e0.season:02d}E{e0.number:02d} by elimination "
                   f"(lone candidate adjacent to matched run)")
        state.put_evidence(
            conn, tid, "elimination", episode_id=eid, verdict=verdict,
            confidence=0.6,
            payload={"season": e0.season, "episode": e0.number,
                     "method": "adjacency+constraint-propagation"})
        results.append({"title_id": tid, "verdict": verdict})
    return {"ok": True, "recovered": len(results), "titles": results}


DISPATCH = {"align": run_align, "ocr": run_ocr, "synopsis": run_synopsis,
            "elimination": run_elimination}
