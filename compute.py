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
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

import state
from discs import Assignment, Disc, Episode, group_discs, log
from identify import (align, classify_disc, recover_by_elimination,
                      stream_signature, verify_title)
import synopsis
from synopsis import (assign_by_synopsis, full_transcript, rank_candidates,
                      sample_transcript, subtitle_cc, subtitle_ocr,
                      transcribe_and_rank)  # noqa: F401 (public re-export)

# Map align's confidence label to a numeric evidence confidence.
_CONF = {"high": 0.9, "medium": 0.6, "low": 0.3}

# Heuristics exposed by `vj run` (name -> one-line description), for --list.
HEURISTICS = {
    "align": "metadata runtime alignment over each season (DP)",
    "streams": "flag episode-length titles whose audio/subtitle layout is "
               "unlike their disc's episodes (likely extras)",
    "ocr": "OCR each title's title card; keeps the winning frame",
    "synopsis": "identify titles by content: whole-episode dialogue (subtitles, "
                "else whisper) judged against episode synopses, one episode per title",
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

    # adjudicated titles constrain the aligner: a confirmed title is a hard
    # ANCHOR (pin its candidate to its episode and re-frame the rest around it);
    # a rejected title is dropped from the candidate set entirely. So confirming
    # one fix and re-running align propagates it, no hand-bumping downstream.
    adjudicated = {}   # title_id -> (status, episode_id | None)
    for a in conn.execute("SELECT title_id, status, episode_ids_json FROM "
                          "assignment WHERE status IN ('confirmed','rejected')"):
        eids = json.loads(a["episode_ids_json"])
        adjudicated[a["title_id"]] = (a["status"], eids[0] if eids else None)

    n_ev = 0
    n_left = 0
    n_anchor = 0
    for season, group in group_discs(discs):
        pool = seasons.get(season) or all_eps
        if not pool:
            continue
        rts = sorted(e.runtime for e in pool if e.runtime)
        expected = rts[len(rts) // 2] if rts else 1320.0
        ep_index = {(e.season, e.number): idx for idx, e in enumerate(pool)}
        cands: list[tuple[Disc, object]] = []
        cand_dbids: list[int] = []
        for d in group:
            ordered = classify_disc(d, expected)
            # persist structural verdicts (kind/order_key) onto title rows
            state.set_classification(conn, path2id[str(d.path)], d.titles)
            for t in ordered:
                dbid = state.title_id(conn, path2id[str(d.path)], t.id)
                if adjudicated.get(dbid, (None,))[0] == "rejected":
                    continue                      # human said: not an episode
                cands.append((d, t))
                cand_dbids.append(dbid)
        if not cands:
            continue
        anchors = {}
        for ci, dbid in enumerate(cand_dbids):
            st = adjudicated.get(dbid)
            if st and st[0] == "confirmed" and st[1] is not None:
                ep = state.episode_by_id(conn, st[1])
                j = ep_index.get((ep.season, ep.number))
                if j is not None:
                    anchors[ci] = j
        n_anchor += len(anchors)

        # soft packaging constraint: each candidate on a disc with an asserted
        # episode set may only route (cheaply) onto those episodes' pool indices.
        background = {}
        for ci, (d, _t) in enumerate(cands):
            asserted = state.get_background(conn, state.disc_name(d.path))
            if asserted:
                allowed = {ep_index[key] for key in asserted if key in ep_index}
                if allowed:
                    background[ci] = allowed
        if background:
            log.info("season %s: %d candidate(s) constrained by packaging hints",
                     season, len(background))

        assignments, leftovers, _missed = align(cands, pool, anchors=anchors,
                                                 background=background)

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
            "anchors": n_anchor, "discs": len(discs)}


# ---------------------------------------------------------------------------
# run streams — audio/subtitle stream layout -> stream-signature evidence
# ---------------------------------------------------------------------------

# Titles shorter than this can't be episodes; excluding them keeps menus and
# short extras from diluting the per-disc majority layout.
STREAM_FLOOR = 300.0   # seconds


def run_streams(conn, args) -> dict:
    """Record a `stream-signature` evidence row per episode-length title: real
    episodes on a disc share an audio/subtitle layout, so a title whose streams
    disagree with its peers' is likely an episode-length *extra* (a featurette
    or alternate cut that fools runtime matching). This carries NO episode
    identity (episode_id is NULL) — it corroborates or contradicts *episode-
    hood*, which `gaps` folds into its worklist. DVD counts come from lsdvd,
    Blu-ray from the HandBrake scan; a disc with no counts yields no evidence.
    """
    loaded = _load_discs(conn)
    if not loaded:
        return {"ok": False, "error": "no-discs",
                "message": "no discs scanned yet (run `scan`)"}
    only = getattr(args, "disc", None)
    n_ev = 0
    results = []
    for did, disc in loaded:
        if only is not None and did != only:
            continue
        # Cluster over episode-length titles only. Exclude play-alls (they carry
        # a legitimately richer/leaner layout — e.g. an added commentary track —
        # so they'd read as the odd one out). Then drop any concatenation far
        # longer than its peers (a whole-disc monolith) — but base that length
        # cut on titles that actually CARRY stream counts, so a disc with many
        # short count-less extra playlists (TNG: six 18-min menu loops) can't
        # drag the median down and exclude the real episodes.
        cands = [t for t in disc.titles
                 if t.duration >= STREAM_FLOOR and t.kind != "play-all"]
        counted = sorted(t.duration for t in cands if t.n_audio or t.n_sub)
        if len(counted) >= 2:
            med = counted[len(counted) // 2]
            cands = [t for t in cands if t.duration <= 1.6 * med]
        maj, verdict = stream_signature(cands)
        if not verdict:
            continue                      # unanimous / no usable split on disc
        ma, ms = maj
        for t in cands:
            v = verdict.get(t.id)
            if v is None:
                continue                  # title carried no stream count
            tid = state.title_id(conn, did, t.id)
            a, s = t.n_audio, t.n_sub
            if v == "episode":
                txt = f"episode stream layout {a}A/{s}S (disc majority)"
                conf = 0.8
            else:
                txt = f"extra-like {a}A/{s}S vs {ma}A/{ms}S disc majority"
                conf = 0.2
            state.put_evidence(conn, tid, "stream-signature", episode_id=None,
                               verdict=txt, confidence=conf,
                               payload={"n_audio": a, "n_sub": s,
                                        "majority": [ma, ms], "class": v})
            n_ev += 1
            results.append({"disc": state.disc_name(str(disc.path)),
                            "title": t.id, "class": v, "sig": [a, s]})
    return {"ok": True, "evidence": n_ev, "flagged":
            sum(1 for r in results if r["class"] == "extra"), "titles": results}


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

            # one clean, interpretable line per title (the read text, the matched
            # episode + its NAME, engine, and where the card was) — not the old
            # first-line-of-a-garbled-read dump
            read = " ".join((capture.get("text") or "").split())[:48]
            eng = capture.get("engine") or "—"
            loc = f" @{card_t:.0f}s" if card_t else ""
            where = f"{state.disc_name(str(disc.path))} t{title.id} (id {tid})"
            if matched:
                log.info("ocr %s  →  S%02dE%02d %r  %.2f [%s%s]  read: %r",
                         where, ep.season, ep.number, ep.name, score, eng, loc, read)
            elif ep is not None:
                log.info("ocr %s  →  ? best S%02dE%02d %r  %.2f (below %.2f) [%s]  "
                         "read: %r", where, ep.season, ep.number, ep.name, score,
                         args.ocr_accept, eng, read)
            else:
                log.info("ocr %s  →  no title-card text found", where)

            state.put_evidence(
                conn, tid, "title-card-ocr", episode_id=ep_id, verdict=verdict,
                confidence=round(score, 3),
                payload={
                    "text": capture.get("text"),
                    "card_seconds": round(card_t, 1) if card_t else None,
                    "best_guess": (f"S{ep.season:02d}E{ep.number:02d}"
                                   if ep else None),
                    "accepted": matched,
                    "engine": capture.get("engine") or args.ocr_engine,
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
# run synopsis — dialogue vs episode synopses -> synopsis evidence
# ---------------------------------------------------------------------------


def run_synopsis(conn, args) -> dict:
    """Identify titles by dialogue-vs-synopsis in two phases: (1) transcribe +
    rank each title against its season's synopses — one judge call per title,
    returning a ranked shortlist; (2) a per-SEASON global assignment (bijection)
    over those rankings, so each episode is claimed by at most one title. Phase 2
    is what kills the magnet failure mode of independent per-title judging (many
    titles picking one arc-heavy episode) — see synopsis.assign_by_synopsis."""
    targets = _ocr_targets(conn, args)   # same target resolution as OCR
    if not targets:
        return {"ok": False, "error": "no-targets",
                "message": "specify --title <id> (repeatable), --disc <basename|id>, or --all"}
    seasons, specials = _season_pools(conn)
    all_eps = [e for n in sorted(seasons) for e in seasons[n]] + specials

    # which synopsis text the judge sees: 'auto' prefers the richer Wikipedia
    # summary (Episode.synopsis already does), 'wikipedia'/'tmdb' force one. Apply
    # once to the shared episode objects up front.
    src = getattr(args, "synopsis_source", "auto") or "auto"
    for e in all_eps:
        if src == "wikipedia":
            e.overview = ""
        elif src == "tmdb":
            e.wiki_overview = ""

    model = getattr(args, "judge_model", None) or synopsis.JUDGE_MODEL
    retranscribe = getattr(args, "retranscribe", False)
    # Refresh the transcript cache WITHOUT judging (no evidence written) — e.g.
    # to backfill cue timings. Also lifts the multi-episode guard: that guard
    # exists to avoid judging a play-all, and here nothing is judged.
    transcribe_only = getattr(args, "transcribe_only", False)
    # Fail FAST if the judge's local server is down — before transcript
    # extraction, which can take hours of subtitle OCR. Never fall back to a
    # weaker judge silently.
    if not transcribe_only:
        why = synopsis.judge_unreachable(model)
        if why:
            return {"ok": False, "error": "judge-unreachable", "message": why}
    # Where the dialogue TEXT comes from. 'auto' prefers SUBTITLES (DVD closed
    # captions — exact words, whole episode, near-instant, no OCR) and falls back
    # to whisper audio when a title has none; 'subtitle'/'audio' force one.
    tsrc = getattr(args, "transcript_source", "auto") or "auto"
    # Default: transcribe the WHOLE episode (identifying dialogue is strewn
    # throughout, so windowing can miss it). `--synopsis-windows N` opts into the
    # faster sampled path. The cache key records which was used — full mode is
    # (0, 0) — so a later judge-swap run reuses the right transcript and a switch
    # between modes is a miss that re-transcribes.
    win_arg = getattr(args, "synopsis_windows", None)
    full = win_arg is None
    if full:
        cache_windows, cache_length, fractions = 0, 0.0, None
    else:
        cache_windows = win_arg
        cache_length = getattr(args, "synopsis_length", None) or synopsis.SAMPLE_LENGTH
        fractions = synopsis.spread_fractions(cache_windows)

    def _audio(d, t, w):
        return (full_transcript(d, t, w) if full
                else sample_transcript(d, t, w, fractions=fractions, length=cache_length))

    # Preference-ordered transcript methods for the requested source, each tagged
    # with its cache key (source, sampling params). 'auto' tries closed captions
    # (DVD text, instant) → bitmap-subtitle OCR (PGS/VOBSUB via PP-OCR, exact) →
    # whisper audio. Each extractor self-selects by disc format, returning "" when
    # it doesn't apply so the chain falls through.
    methods = []   # (source, cache_windows, cache_length, extractor)
    if tsrc in ("auto", "subtitle"):
        # subtitle extractors return (text, timed cues); audio returns text
        methods.append(("cc", 0, 0.0, subtitle_cc))
        methods.append(("subtitle-ocr", 0, 0.0, subtitle_ocr))
    if tsrc in ("auto", "audio"):
        methods.append(("audio", cache_windows, cache_length, _audio))

    # Resolve every target to its (disc, title) up front so we can compute a
    # per-disc multi-episode guard before spending any whisper time.
    disc_cache: dict = {}
    resolved = []      # (disc_id, tid, disc, title)
    for disc_id, tid in targets:
        disc = disc_cache.get(disc_id) or state.load_disc(conn, disc_id)
        disc_cache[disc_id] = disc
        tn = conn.execute("SELECT title_number FROM title WHERE id=?",
                          (tid,)).fetchone()["title_number"]
        title = next(t for t in disc.titles if t.id == tn)
        resolved.append((disc_id, tid, disc, title))

    # Multi-episode guard: a title far longer than the disc's typical candidate is
    # a play-all / concatenation (it holds several episodes' dialogue at once), so
    # it can't be matched to ONE episode — skip it rather than transcribe/judge it.
    # Threshold = the disc-local median candidate duration ×1.5 (TMDB-independent).
    # A single episode runs ≤ ~1.3× the typical one (a premiere/finale); a 2-part
    # concatenation runs ~1.7-1.9× — so 1.5× separates them. The factor is tighter
    # than run_streams' 1.6× on purpose: run_streams excludes play-alls BEFORE
    # taking its median, but here they're still in the sample and inflate it, so a
    # looser cut would let a 2-episode title slip through. Assumes real episodes
    # are the majority (as run_streams does); needs ≥3 targets for a stable median,
    # with fewer there's nothing to compare against so no guard.
    disc_cap: dict = {}
    by_disc: dict = defaultdict(list)
    for disc_id, _tid, _disc, title in resolved:
        by_disc[disc_id].append(title.duration)
    for disc_id, durs in by_disc.items():
        disc_cap[disc_id] = (statistics.median(durs) * 1.5
                             if len(durs) >= 3 else float("inf"))

    # --- phase 1: get dialogue text + rank each title (one judge call each) ---
    # The transcript (subtitle or whisper) is the expensive part and is
    # judge-independent, so it's cached in the DB keyed by (source, sampling
    # params): a second run — e.g. to swap the judge model — reuses it and only
    # re-does the cheap judge call.
    recs = []          # (tid, season_key, pool, ranked, evidence, source)
    with tempfile.TemporaryDirectory(prefix="vj-syn-", dir=args.scratch_dir) as tmp:
        workdir = Path(tmp)
        for disc_id, tid, disc, title in resolved:
            pool = _pool_for(disc, seasons, specials, all_eps,
                             getattr(args, "include_specials", False))
            season_key = disc.season_hint or 0

            if title.duration > disc_cap.get(disc_id, float("inf")) \
                    and not transcribe_only:
                log.info("synopsis: skipping title %d (%.0fm) — multi-episode "
                         "(> 1.5× disc median), not a single-episode target",
                         tid, title.duration / 60)
                recs.append((tid, season_key, pool, [],
                             f"multi-episode title ({title.duration/60:.0f}m) — "
                             "not a single-episode synopsis target", None))
                continue

            transcript, used = None, None
            if not retranscribe:   # reuse a cached transcript for the wanted source
                for src, cw, cl, _fn in methods:
                    c = state.get_transcript(conn, tid, cw, cl, src)
                    if c is not None:
                        transcript, used = c, src
                        break
            if transcript is None:   # extract fresh, walking the fallback chain
                for src, cw, cl, fn in methods:
                    res = fn(disc, title, workdir)
                    txt, cues = res if isinstance(res, tuple) else (res, None)
                    if txt:
                        transcript, used = txt, src
                        state.put_transcript(conn, tid, txt, cw, cl, src, cues)
                        break
                else:   # nothing produced text — cache empty so re-runs don't retry,
                    # but never clobber a good transcript (a --retranscribe that
                    # failed transiently would otherwise wipe it)
                    if not state.get_transcript(conn, tid):
                        state.put_transcript(conn, tid, "", cache_windows,
                                             cache_length, "audio")
                    else:
                        log.warning("synopsis: re-extraction of title %d produced "
                                    "nothing; keeping the cached transcript", tid)

            if transcribe_only:
                recs.append((tid, season_key, pool, [], "", used))
                continue
            if transcript:
                ranked, evidence = rank_candidates(
                    transcript, pool, model, args.ollama_host)
            else:
                ranked, evidence = [], ("no subtitles found" if tsrc == "subtitle"
                                        else "no dialogue transcribed")
            recs.append((tid, season_key, pool, ranked, evidence, used))

    if transcribe_only:   # extraction only: no judge calls, no evidence written
        return {"ok": True, "transcribed": [
            {"title_id": tid, "source": used} for tid, *_rest, used in recs]}

    # --- phase 2: bijection assignment over the rankings ---
    # Real episodes get a per-SEASON bijection (each disc pair aligns within its
    # season). Specials (season 0) sit in EVERY season's pool, so a per-season pass
    # would let one special be claimed by several seasons (VB: Gargantua-2 grabbed
    # by S4 AND S6). Instead they get ONE GLOBAL bijection over all titles no season
    # episode claimed — so a special is assigned at most once across the series.
    groups: dict = defaultdict(list)
    for r in recs:
        groups[r[1]].append(r)

    season_assigned: dict = {}       # tid -> (episode, score, rank)
    special_claims: list = []        # (tid, [(special, score), …]) for leftovers
    for season_key, group in groups.items():
        season_pool = [e for e in group[0][2] if e.season != 0]
        rows = [(tid, [(e, s) for e, s in ranked if e.season != 0])
                for tid, _, _, ranked, _, _ in group]
        season_assigned.update(assign_by_synopsis(rows, season_pool))
        for tid, _, _, ranked, _, _ in group:
            if tid not in season_assigned:
                spec = [(e, s) for e, s in ranked if e.season == 0]
                if spec:
                    special_claims.append((tid, spec))
    specials_assigned = (assign_by_synopsis(special_claims, specials)
                         if special_claims else {})
    assigned = {**season_assigned, **specials_assigned}

    results = []
    for tid, _sk, _pool, ranked, evidence, src_used in recs:
        shortlist = [f"S{e.season:02d}E{e.number:02d}" for e, _ in ranked]
        if tid in assigned:
            ep, score, rank = assigned[tid]
            ep_id = state.episode_id(conn, ep.season, ep.number)
            # relative to this title's own best score, so both judge kinds land
            # on one scale: Borda ranks (LLMs: 6 for 1st) and probabilities
            # (Jev/Kev: ~0.2 even for a correct pick). 1.0 = its first choice;
            # lower = the bijection moved it down its shortlist. (score/RANK_TOP_K
            # put every Jev/Kev pick under resolve's 0.5 threshold.)
            conf = round(score / ranked[0][1], 3) if ranked and ranked[0][1] else 0.0
            verdict = (f"S{ep.season:02d}E{ep.number:02d} "
                       f"(rank {rank}/{len(ranked)}): {evidence}")
            payload = {"rank": rank, "assigned": True, "shortlist": shortlist,
                       "evidence": evidence, "source": src_used}
        else:
            ep_id, conf = None, 0.0
            verdict = (f"abstained — shortlist {shortlist} taken by better "
                       "fits" if ranked else f"abstained: {evidence}")
            payload = {"assigned": False, "shortlist": shortlist, "source": src_used}
        state.put_evidence(conn, tid, "synopsis", episode_id=ep_id,
                           verdict=verdict, confidence=conf, payload=payload)
        results.append({"title_id": tid, "verdict": verdict,
                        "confidence": conf, "source": src_used})
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


DISPATCH = {"align": run_align, "streams": run_streams, "ocr": run_ocr,
            "synopsis": run_synopsis, "elimination": run_elimination}
