#!/usr/bin/env python3
"""video-juicer — evidence-first disc→episode identification (Picard-style).

Heuristics are on-demand evidence producers; a human or agent adjudicates the
result. All state lives in a SQLite project file (see DESIGN.md). Verbs:

  ingest:      init, scan
  compute:     run <heuristic>            (align, ocr, synopsis)
  inspect:     board, status, gaps, show, frame
  resolve:     resolve                    (evidence -> proposed assignments)
  adjudicate:  assign, confirm, reject
  auto:        auto                       (scripts align→resolve→ocr→…→resolve)
  inspect:     play                       (launch a player on a title)
  export:      export

Every verb takes an explicit <state.db> and emits JSON when stdout is not a TTY
(pass --json to force it), human-readable text otherwise. Errors are structured.

Requires on PATH: lsdvd, 7z (scan). OCR verbs also need ffmpeg/mencoder + Ollama.
TMDB_API_KEY in the environment (or --tmdb-api-key) for `init`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Optional

import auto as auto_mod
import compute
import export as export_mod
import review
import state
import wiki
from discs import (
    Tmdb, grouped_seasons, log, scan_disc,
)

# ---------------------------------------------------------------------------
# output: JSON for machines/agents, text for humans
# ---------------------------------------------------------------------------


def _json_mode(args) -> bool:
    return bool(getattr(args, "json", False)) or not sys.stdout.isatty()


def emit(args, payload: dict, human: str = "") -> None:
    """Print a verb's result: JSON when non-interactive, else `human` text."""
    if _json_mode(args):
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(human.rstrip("\n") + "\n" if human else "")


def fail(args, code: str, message: str, **extra) -> int:
    """Structured error to stderr; JSON when non-interactive. Returns exit 1."""
    payload = {"ok": False, "error": code, "message": message, **extra}
    if _json_mode(args):
        json.dump(payload, sys.stderr, indent=2, default=str)
        sys.stderr.write("\n")
    else:
        sys.stderr.write(f"error [{code}]: {message}\n")
    return 1


# ---------------------------------------------------------------------------
# ingest: init
# ---------------------------------------------------------------------------


def cmd_init(args) -> int:
    api_key = args.tmdb_api_key or os.environ.get("TMDB_API_KEY")
    if not api_key:
        return fail(args, "no-api-key",
                    "TMDB_API_KEY not set and --tmdb-api-key not given")

    tmdb = Tmdb(api_key, args.cache_dir / str(args.tmdb_id))
    series = tmdb.series(args.tmdb_id)
    show = series["name"]
    fad = series.get("first_air_date") or ""
    year = int(fad[:4]) if fad[:4].isdigit() else None
    series_rt = series.get("episode_run_time") or []
    fallback_rt = series_rt[0] * 60.0 if series_rt else None

    if args.episode_order != "aired":
        seasons, specials = grouped_seasons(
            tmdb, args.tmdb_id, args.episode_order, fallback_rt)
    else:
        nums = [s["season_number"] for s in series["seasons"]
                if s["season_number"] > 0]
        seasons = {n: tmdb.season_episodes(args.tmdb_id, n, fallback_rt)
                   for n in nums}
        specials = (tmdb.season_episodes(args.tmdb_id, 0, fallback_rt)
                    if any(s["season_number"] == 0 for s in series["seasons"])
                    else [])

    episodes = [e for n in sorted(seasons) for e in seasons[n]] + specials

    conn = state.connect(args.db)
    state.set_project(conn, tmdb_id=args.tmdb_id, show_name=show, year=year,
                      episode_order=args.episode_order)
    state.upsert_episodes(conn, episodes)
    conn.close()

    n_specials = len(specials)
    emit(args,
         {"ok": True, "db": str(args.db), "show": show, "year": year,
          "tmdb_id": args.tmdb_id, "seasons": len(seasons),
          "episodes": len(episodes), "specials": n_specials},
         human=(f"initialised {args.db}\n"
                f"  {show} ({year}) [tmdb {args.tmdb_id}]\n"
                f"  {len(seasons)} seasons, {len(episodes)} episodes"
                f"{f' (+{n_specials} specials)' if n_specials else ''}"))
    return 0


# ---------------------------------------------------------------------------
# enrich: richer episode synopses from a local Wikipedia dump
# ---------------------------------------------------------------------------


def cmd_enrich(args) -> int:
    if not Path(args.db).exists():
        return fail(args, "no-db", f"state file not found: {args.db} (run `init`)")
    conn = state.connect(args.db)
    page = args.page or state.get_project(conn).get("wikipedia_page")
    if not page:
        show = state.get_project(conn).get("show_name", "")
        guess = f"List of {show} episodes" if show else ""
        conn.close()
        return fail(args, "no-page",
                    "no Wikipedia page given; pass --page \"List of <Show> "
                    f"episodes\"{f' (try: {guess!r})' if guess else ''}. It is "
                    "remembered for next time.")
    try:
        snap = wiki.MultistreamSnapshot(args.snapshot, args.index)
        summaries = wiki.episode_summaries(snap, page)
    except wiki.SnapshotError as e:
        conn.close()
        return fail(args, "snapshot-error", str(e))

    by_key = {k: text for k, (_name, text) in summaries.items()}
    n = state.set_wiki_overviews(conn, by_key)
    state.set_project(conn, wikipedia_page=page)   # remember for re-runs
    total = len(state.load_episodes(conn))
    conn.close()

    emit(args,
         {"ok": True, "page": page, "parsed": len(summaries),
          "updated": n, "episodes": total},
         human=(f"enriched {n}/{total} episodes from Wikipedia\n"
                f"  page: {page}\n"
                f"  parsed {len(summaries)} summaries from the dump"))
    return 0


def _parse_ranges(spec: str) -> list[int]:
    """'1-4,6,8-9' -> [1,2,3,4,6,8,9] (sorted, deduped). Raises ValueError on junk."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                raise ValueError(f"backwards range {part!r}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _match_ep_title(eps, needle: str, season):
    """Resolve a packaging title string to an Episode by name (casefold exact,
    then unique substring), optionally scoped to a season. Returns the Episode or
    raises ValueError naming the ambiguity so a typo isn't silently mismapped."""
    pool = [e for e in eps if season is None or e.season == season]
    key = needle.strip().casefold()
    exact = [e for e in pool if (e.name or "").strip().casefold() == key]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(f"title {needle!r} matches several episodes")
    sub = [e for e in pool if key in (e.name or "").strip().casefold()]
    if len(sub) == 1:
        return sub[0]
    if len(sub) > 1:
        raise ValueError(f"title {needle!r} is ambiguous "
                         f"({', '.join(f'S{e.season:02d}E{e.number:02d}' for e in sub)})")
    raise ValueError(f"no episode matching title {needle!r}")


def cmd_hint(args) -> int:
    """Record box-packaging knowledge: which episodes a disc holds. A SOFT signal
    the aligner prefers but can override on strong runtime disagreement."""
    if not Path(args.db).exists():
        return fail(args, "no-db", f"state file not found: {args.db} (run `init`)")
    conn = state.connect(args.db)

    # resolve --disc to a canonical basename; allow a not-yet-scanned disc
    did, derr = _resolve_disc(conn, args.disc)
    prescan = False
    if did is not None:
        name = next(state.disc_name(r["path"]) for r in state.list_discs(conn)
                    if r["id"] == did)
    elif derr and "ambiguous" in derr:
        conn.close()
        return fail(args, "ambiguous-disc", derr)
    else:
        name, prescan = state.disc_name(args.disc), True

    eps = state.load_episodes(conn)
    known = {(e.season, e.number) for e in eps}
    pairs: list[tuple[int, int]] = []
    try:
        if args.episodes:
            if args.season is None:
                raise ValueError("--episodes needs --season")
            pairs += [(args.season, n) for n in _parse_ranges(args.episodes)]
        for t in (args.titles or []):
            e = _match_ep_title(eps, t, args.season)
            pairs.append((e.season, e.number))
    except ValueError as e:
        conn.close()
        return fail(args, "bad-hint", str(e))

    pairs = sorted(set(pairs))
    if not pairs:
        conn.close()
        return fail(args, "empty-hint",
                    "nothing to record; pass --episodes (with --season) and/or --titles")
    unknown = [p for p in pairs if p not in known]
    if unknown:
        conn.close()
        return fail(args, "unknown-episode",
                    "not in this show's episode list (typo?): "
                    + ", ".join(f"S{s:02d}E{n:02d}" for s, n in unknown))

    state.set_background(conn, name, pairs, source=args.source)
    conn.close()
    labels = ", ".join(f"S{s:02d}E{n:02d}" for s, n in pairs)
    emit(args,
         {"ok": True, "disc": name, "prescan": prescan,
          "episodes": [list(p) for p in pairs], "source": args.source},
         human=(f"recorded packaging hint for {name}: {labels}"
                + ("\n  (disc not scanned yet — will apply once it is)"
                   if prescan else "")))
    return 0


# ---------------------------------------------------------------------------
# ingest: scan
# ---------------------------------------------------------------------------


def cmd_run(args) -> int:
    if args.list or args.heuristic is None:
        w = max((len(k) for k in compute.HEURISTICS), default=0)
        emit(args, {"ok": True, "heuristics": compute.HEURISTICS},
             human="\n".join(f"  {k:<{w}}  {v}"
                             for k, v in compute.HEURISTICS.items()))
        return 0
    if not Path(args.db).exists():
        return fail(args, "no-db", f"state file not found: {args.db}")
    if args.heuristic not in compute.DISPATCH:
        return fail(args, "unknown-heuristic",
                    f"unknown heuristic {args.heuristic!r}; "
                    f"expected one of {sorted(compute.DISPATCH)}")

    conn = state.connect(args.db)
    if getattr(args, "disc", None) is not None:
        did, derr = _resolve_disc(conn, args.disc)
        if derr:
            conn.close()
            return fail(args, "no-disc", derr)
        args.disc = did
    try:
        result = compute.DISPATCH[args.heuristic](conn, args)
    finally:
        conn.close()
    if not result.get("ok"):
        return fail(args, result.get("error", "run-failed"),
                    result.get("message", "heuristic failed"))
    emit(args, result, human=_run_human(args.heuristic, result))
    return 0


def _run_human(heuristic: str, r: dict) -> str:
    if heuristic == "align":
        anc = f", {r['anchors']} confirmed anchor(s)" if r.get("anchors") else ""
        return (f"align: {r['evidence']} episode evidence rows, "
                f"{r['leftovers']} leftovers across {r['discs']} disc(s){anc}")
    if heuristic == "elimination":
        lines = [f"elimination: recovered {r['recovered']} title(s)"
                 + (f" — {r['note']}" if r.get("note") else "")]
        for x in r.get("titles", []):
            lines.append(f"  title {x['title_id']}: {x['verdict']}")
        return "\n".join(lines)
    if heuristic == "streams":
        lines = [f"streams: {r['evidence']} stream-signature row(s), "
                 f"{r['flagged']} flagged extra-like"]
        for x in r.get("titles", []):
            if x["class"] == "extra":
                lines.append(f"  disc {x['disc']} title {x['title']}: "
                             f"extra-like ({x['sig'][0]}A/{x['sig'][1]}S)")
        return "\n".join(lines)
    key = {"ocr": "ocr", "synopsis": "synopsis"}[heuristic]
    lines = [f"{heuristic}: {len(r[key])} title(s)"]
    for x in r[key]:
        lines.append(f"  title {x['title_id']}: {x['verdict']}"
                     + ("  [frame kept]" if x.get("frame") else ""))
    return "\n".join(lines)


def cmd_scan(args) -> int:
    for tool in ("lsdvd", "7z"):
        if not shutil.which(tool):
            return fail(args, "missing-tool", f"required tool not on PATH: {tool}")
    if not Path(args.db).exists():
        return fail(args, "no-db", f"state file not found: {args.db} (run `init` first)")

    conn = state.connect(args.db)
    scanned = []
    for p in args.images:
        if not p.exists():
            conn.close()
            return fail(args, "no-image", f"disc image not found: {p}")
        log.info("scanning %s", p)
        disc = scan_disc(p)
        disc_id = state.add_disc(conn, disc)
        scanned.append({
            "disc_id": disc_id, "path": str(disc.path), "format": disc.format,
            "label": disc.label, "season_hint": disc.season_hint,
            "disc_hint": disc.disc_hint, "titles": len(disc.titles),
        })
    conn.close()

    human = "\n".join(
        f"scanned disc {d['disc_id']}: {state.disc_name(d['path'])} "
        f"[{d['format']}] {d['titles']} titles"
        + (f" (S{d['season_hint']}" if d['season_hint'] else "")
        + (f"D{d['disc_hint']})" if d['disc_hint'] else
           (")" if d['season_hint'] else ""))
        for d in scanned)
    emit(args, {"ok": True, "scanned": scanned}, human=human)
    return 0


# ---------------------------------------------------------------------------
# inspect / adjudicate / resolve
# ---------------------------------------------------------------------------


def _compress_eps(eps: list[str]) -> str:
    """"S02E01".."S02E13" -> "S02E01-E13"; collapse consecutive runs per season."""
    parsed = []
    for s in eps:
        try:
            se, ep = s.upper().lstrip("S").split("E")
            parsed.append((int(se), int(ep)))
        except ValueError:
            return ", ".join(eps)   # unexpected format: don't mangle it
    parsed.sort()
    out, i = [], 0
    while i < len(parsed):
        se, lo = parsed[i]
        j = i
        while j + 1 < len(parsed) and parsed[j + 1] == (se, parsed[j][1] + 1):
            j += 1
        hi = parsed[j][1]
        out.append(f"S{se:02d}E{lo:02d}" if lo == hi
                   else f"S{se:02d}E{lo:02d}-E{hi:02d}")
        i = j + 1
    return ", ".join(out)


def _open(args):
    if not Path(args.db).exists():
        return None, fail(args, "no-db", f"state file not found: {args.db}")
    return state.connect(args.db), None


def _resolve_disc(conn, value):
    """Resolve a --disc argument (a disc basename, an integer id, or a unique
    substring) to a disc id. Returns (disc_id, error_message)."""
    if value is None:
        return None, None
    rows = [(r["id"], state.disc_name(r["path"])) for r in state.list_discs(conn)]
    v = str(value).strip()
    exact = [i for i, n in rows if n.lower() == v.lower()]
    if len(exact) == 1:
        return exact[0], None
    if v.isdigit() and any(i == int(v) for i, _ in rows):
        return int(v), None
    sub = [(i, n) for i, n in rows if v.lower() in n.lower()]
    if len(sub) == 1:
        return sub[0][0], None
    if len(sub) > 1:
        return None, (f"disc {value!r} is ambiguous — matches: "
                      + ", ".join(n for _, n in sub))
    return None, (f"no disc matching {value!r}; scanned discs: "
                  + ", ".join(n for _, n in rows) or "(none)")


def cmd_status(args) -> int:
    conn, err = _open(args)
    if err:
        return err
    r = review.summarize(conn)
    conn.close()
    lines = [f"{r['show']}: {r['titles']} titles scanned",
             f"  assignments: {r['assignments_by_status'] or '(none)'}"]
    for s in r["seasons"]:
        lines.append(f"  S{s['season']:02d}: {s['matched']}/{s['total']} episodes matched")
    for w in r.get("order_warnings", []):
        lines.append(f"  ⚠ {w['disc']}: order unverified — {w['reason']}")
    for p in r.get("packaging", []):
        n = len(p["asserted"])
        tag = "" if p["scanned"] else " (not scanned yet)"
        lines.append(f"  📦 {p['disc']}: packaging lists {n} episode(s){tag}")
        if p.get("outside"):
            eps = ", ".join(f"S{s:02d}E{e:02d}" for s, e in p["outside"])
            lines.append(f"     ‼ assigned but NOT on the box: {eps}")
        if p.get("missing"):
            eps = ", ".join(f"S{s:02d}E{e:02d}" for s, e in p["missing"])
            lines.append(f"     · listed but unassigned: {eps}")
    emit(args, r, human="\n".join(lines))
    return 0


def cmd_gaps(args) -> int:
    conn, err = _open(args)
    if err:
        return err
    r = review.gaps(conn, args.threshold)
    conn.close()
    lines = [f"{r['n_gaps']} title(s) need attention, "
             f"{r['n_missing']} episode(s) missing"]
    for w in r.get("order_warnings", []):
        lines.append(f"  ⚠ order unverified: {w['disc']} ({w['titles']} titles) "
                     f"— {w['reason']}")
    _sig = {"assign": "→ ASSIGN", "reject": "→ REJECT",
            "run-ocr": "→ run ocr", "review": "→ review"}
    for g in r["gaps"]:
        s = g["suggestion"]
        act = _sig.get(s["action"], s["action"])
        if s["action"] == "assign":
            act += f" {s['episode']}"
        lines.append(f"  title {g['title_id']} ({g['disc']} t{g['title_number']}, "
                     f"{g['minutes']}m)  {act}  ({s['why']})")
        for e in g["evidence"]:
            lines.append(f"      [{e['category']}] {e['episode'] or '—'} "
                         f"{e['verdict'] or ''} ({e['confidence']})")
    if r["missing_episodes"]:
        lines.append("  missing: " + _compress_eps(r["missing_episodes"]))
    emit(args, r, human="\n".join(lines))
    return 0


def cmd_show(args) -> int:
    conn, err = _open(args)
    if err:
        return err
    if args.episode:
        try:
            s, e = args.episode.upper().lstrip("S").split("E")
            r = review.show_episode(conn, int(s), int(e))
        except ValueError:
            conn.close()
            return fail(args, "bad-episode", "use --episode S01E02")
    elif args.title is not None:
        r = review.show_title(conn, args.title)
    else:
        conn.close()
        return fail(args, "no-target", "give --title <id> or --episode S01E02")
    conn.close()
    if not r.get("ok"):
        return fail(args, r["error"], r["message"])
    emit(args, r, human=_show_human(r))
    return 0


def _show_title_human(t: dict) -> list[str]:
    a = t.get("assignment")
    head = (f"title {t['title_id']}  ({t['disc']} t{t['title_number']}, "
            f"{t['minutes']}m, {t['kind']})")
    lines = [head]
    if a:
        lines.append(f"  assignment: {'+'.join(a['episodes']) or '(none)'} "
                     f"[{a['status']} by {a['decided_by']}]"
                     + (f" — {a['note']}" if a.get("note") else ""))
    else:
        lines.append("  assignment: (none)")
    if t["evidence"]:
        lines.append("  evidence:")
        for e in t["evidence"]:
            lines.append(f"    [{e['category']}] {e['episode'] or '—'}  "
                         f"{e['verdict'] or ''} ({e['confidence']})")
    if t.get("frames"):
        lines.append(f"  frames on file: {', '.join(t['frames'])} "
                     f"(dump with: vj frame <db> --title {t['title_id']})")
    return lines


def _show_human(r: dict) -> str:
    if "episode" in r:   # episode view
        lines = [f"{r['episode']}  {r['name']}  "
                 f"({(r['runtime'] or 0)/60:.0f}m)"]
        if not r["titles"]:
            lines.append("  no titles carry evidence for this episode")
        for t in r["titles"]:
            lines.append("")
            lines += ["  " + x for x in _show_title_human(t)]
        return "\n".join(lines)
    return "\n".join(_show_title_human(r))   # title view


def cmd_frame(args) -> int:
    conn, err = _open(args)
    if err:
        return err
    f = state.get_frame(conn, args.title, args.category)
    conn.close()
    if f is None or f["image"] is None:
        return fail(args, "no-frame", f"no stored frame for title {args.title}")
    ext = ".jpg" if "jpeg" in (f["mime"] or "") else ".png"
    out = args.out or Path(f"/tmp/vj-frame-t{args.title}{ext}")
    out.write_bytes(f["image"])
    r = {"ok": True, "title_id": args.title, "out": str(out),
         "category": f["category"], "source_time": f["source_time"],
         "ocr_text": f["ocr_text"]}
    emit(args, r, human=f"wrote frame -> {out}\n  category: {f['category']}  "
                        f"@{f['source_time']}s  read: {f['ocr_text']!r}")
    return 0


def cmd_resolve(args) -> int:
    conn, err = _open(args)
    if err:
        return err
    r = review.resolve(conn, args.threshold)
    conn.close()
    emit(args, r, human=f"proposed {r['proposed']} assignment(s); "
                        f"{r['conflicts']} left as conflicts")
    return 0


def _parse_ep(conn, spec: str):
    s, e = spec.upper().lstrip("S").split("E")
    return state.episode_id(conn, int(s), int(e))


def cmd_assign(args) -> int:
    conn, err = _open(args)
    if err:
        return err
    if conn.execute("SELECT 1 FROM title WHERE id=?", (args.title,)).fetchone() is None:
        conn.close()
        return fail(args, "no-title", f"no title {args.title}")
    try:
        ids = [_parse_ep(conn, s) for s in args.episode]
    except ValueError:
        conn.close()
        return fail(args, "bad-episode", "use SxxEyy, e.g. --episode S01E02")
    if any(i is None for i in ids):
        conn.close()
        return fail(args, "no-episode", "one or more episodes not found")
    by = "agent" if args.agent else "human"
    state.set_assignment(conn, args.title, ids, status="confirmed",
                         decided_by=by, note=args.note)
    conn.close()
    emit(args, {"ok": True, "title_id": args.title, "episodes": args.episode,
                "status": "confirmed", "decided_by": by},
         human=f"title {args.title} -> {'+'.join(args.episode)} [confirmed by {by}]")
    return 0


def cmd_confirm(args) -> int:
    conn, err = _open(args)
    if err:
        return err
    a = state.get_assignment(conn, args.title)
    if a is None or not json.loads(a["episode_ids_json"]):
        conn.close()
        return fail(args, "no-proposal",
                    f"title {args.title} has no proposal to confirm")
    by = "agent" if args.agent else "human"
    state.set_assignment(conn, args.title, json.loads(a["episode_ids_json"]),
                         status="confirmed", decided_by=by, note=a["note"])
    conn.close()
    emit(args, {"ok": True, "title_id": args.title, "status": "confirmed",
                "decided_by": by}, human=f"title {args.title} confirmed by {by}")
    return 0


def cmd_reject(args) -> int:
    conn, err = _open(args)
    if err:
        return err
    if conn.execute("SELECT 1 FROM title WHERE id=?", (args.title,)).fetchone() is None:
        conn.close()
        return fail(args, "no-title", f"no title {args.title}")
    by = "agent" if args.agent else "human"
    state.set_assignment(conn, args.title, [], status="rejected",
                         decided_by=by, note=args.note)
    conn.close()
    emit(args, {"ok": True, "title_id": args.title, "status": "rejected",
                "decided_by": by},
         human=f"title {args.title} rejected (not an episode) by {by}")
    return 0


def cmd_auto(args) -> int:
    conn, err = _open(args)
    if err:
        return err
    r = auto_mod.run_auto(conn, args)
    conn.close()
    if not r.get("ok"):
        return fail(args, r["error"], r["message"])
    lines = ["auto pipeline:"]
    for s in r["steps"]:
        if s["step"] == "align":
            lines.append(f"  align      → {s['evidence']} evidence, {s['leftovers']} leftovers")
        elif s["step"] == "resolve":
            lines.append(f"  resolve    → {s['proposed']} proposed, {s['conflicts']} conflicts")
        elif s["step"] == "ocr-policy":
            lines.append(f"  ocr        → {s['reason']}")
        elif s["step"] == "ocr":
            lines.append(f"               disc {s['disc']}: {s['titles']} title(s)")
        elif s["step"] == "elimination":
            lines.append(f"  eliminate  → {s['recovered']} recovered")
    lines.append(f"\nstatus: {r['assignments_by_status'] or '(none)'}; "
                 f"{r['n_gaps']} gap(s), {r['n_missing']} episode(s) missing")
    for s in r["seasons"]:
        if s["total"]:
            lines.append(f"  S{s['season']:02d}: {s['matched']}/{s['total']}")
    emit(args, r, human="\n".join(lines))
    return 0


def _start_args(player: str, sec: float) -> list[str]:
    """Seek flag, per player (vlc: --start-time N; mpv: --start=N)."""
    if not sec:
        return []
    mpv = "mpv" in os.path.basename(player).lower()
    return [f"--start={int(sec)}"] if mpv else ["--start-time", str(int(sec))]


def _play_argv(player: str, row, start: float) -> tuple[list[str], str]:
    """Build the player command for one title, plus a note.

    Blu-ray title selection isn't reliably expressible as an MRL — the disc's
    libbluray title index doesn't match our `.mpls` id (same mismatch as
    HandBrake), and a disc's "Play All" may not even be a playlist. But for a
    BDMV backup DIR we already know the title's ordered CLIPS, so we play the
    `.m2ts` files directly — exact, and works in any player (vlc or mpv). DVD
    uses the reliable `dvd://…#N` title fragment; a Blu-ray IMAGE (no BDMV dir)
    falls back to the disc's main title."""
    fmt, path = row["format"], row["path"]
    if fmt == "dvd":
        return [player, f"dvd://{path}#{row['title_number']}"] + _start_args(player, start), ""
    stream = Path(path) / "BDMV" / "STREAM"
    clips = [str(stream / f"{c}.m2ts")
             for c in json.loads(row["clips_json"] or "[]")]
    clips = [c for c in clips if Path(c).is_file()]
    if clips:
        return ([player] + clips + _start_args(player, start),
                f"Blu-ray: playing the title's {len(clips)} .m2ts clip(s) directly")
    return ([player, f"bluray://{path}"] + _start_args(player, start),
            "Blu-ray image (no BDMV dir): opens the disc's main title — can't "
            "select this title from an image")


def cmd_play(args) -> int:
    conn, err = _open(args)
    if err:
        return err
    # resolve target -> a title row
    if args.title is not None:
        row = conn.execute(
            "SELECT t.id, t.title_number, t.clips_json, d.path, d.format FROM title t "
            "JOIN disc d ON d.id=t.disc_id WHERE t.id=?", (args.title,)).fetchone()
        if row is None:
            conn.close()
            return fail(args, "no-title", f"no title {args.title}")
        tid = args.title
    elif args.target:
        try:
            eid = _parse_ep(conn, args.target)
        except ValueError:
            conn.close()
            return fail(args, "bad-episode", "use an episode like S01E03, or --title N")
        if eid is None:
            conn.close()
            return fail(args, "no-episode", f"no episode {args.target}")
        tid = None
        for a in conn.execute("SELECT title_id, episode_ids_json FROM assignment "
                              "WHERE status IN ('proposed','confirmed')"):
            if eid in json.loads(a["episode_ids_json"]):
                tid = a["title_id"]
                break
        if tid is None:
            conn.close()
            return fail(args, "no-assignment",
                        f"no title is assigned to {args.target} yet "
                        f"(try --title N to preview a candidate)")
        row = conn.execute(
            "SELECT t.id, t.title_number, d.path, d.format FROM title t "
            "JOIN disc d ON d.id=t.disc_id WHERE t.id=?", (tid,)).fetchone()
    else:
        conn.close()
        return fail(args, "no-target", "give an episode (S01E03) or --title N")

    start = args.start
    if args.at_card:                     # jump to the retained title-card frame
        f = state.get_frame(conn, tid, "title-card-ocr")
        if f and f["source_time"] is not None:
            start = max(0.0, f["source_time"] - 5.0)
    conn.close()

    argv, note = _play_argv(args.player, row, start)
    tail = f"  ({note})" if note else ""
    if args.print:
        emit(args, {"ok": True, "command": argv, "launched": False,
                    "title_id": tid, "note": note or None},
             human="would run: " + " ".join(shlex.quote(a) for a in argv) + tail)
        return 0
    if not shutil.which(args.player):
        return fail(args, "no-player", f"player not on PATH: {args.player}")
    import subprocess
    subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    emit(args, {"ok": True, "command": argv, "launched": True,
                "title_id": tid, "note": note or None},
         human=f"launched {args.player}: title {row['title_number']} on "
               f"{state.disc_name(row['path'])}"
               + (f" @{int(start)}s" if start else "") + tail)
    return 0


_KIND_ABBR = {"episode-candidate": "episode", "play-all": "play-all",
              "extra": "extra", "unknown": "unknown"}
_CAT_ABBR = {"runtime-align": "align", "title-card-ocr": "ocr",
             "synopsis": "syn", "elimination": "elim", "play-all": "pa"}
_STATUS_SYM = {"confirmed": "✓", "proposed": "●", "rejected": "✗"}


def _ev_detail(cat: str, e: dict) -> str:
    """The right-hand detail for an evidence line: the OCR read text, or the
    align delta/label parenthetical, else the verdict."""
    if cat == "title-card-ocr":
        read = " ".join((e.get("read") or "").split()).strip("\"“”' ")
        return f'"{read}"' if read else "(no text read)"
    v = e.get("verdict") or ""
    return v[v.index("("):] if "(" in v else v


def _evidence_lines(ev: dict, indent: str, width: int = 92,
                    max_lines: int = 3) -> list[str]:
    """One line per evidence source under the title; a long read (a garbage
    scene dump) WRAPS across continuation lines instead of truncating, capped so
    it can't become a wall. A normal title card is a single line."""
    import textwrap
    out = []
    for cat, e in ev.items():
        ab = _CAT_ABBR.get(cat, cat)
        ep = e["episode"] or "—"
        conf = f"{e['confidence']:.2f}" if e["confidence"] is not None else " — "
        prefix = f"{indent}{ab:<6} {ep:<7} {conf}   "
        segs = textwrap.wrap(_ev_detail(cat, e), width=width) or [""]
        cont = " " * len(prefix)
        for i, seg in enumerate(segs[:max_lines]):
            out.append((prefix if i == 0 else cont) + seg)
        if len(segs) > max_lines:
            out[-1] += " …"
    return out


def _board_human(r: dict, show_all: bool = False) -> str:
    from collections import OrderedDict
    lines = []
    head = f"{r['show']} ({r['year']})"
    if r.get("tmdb_id"):
        head += f"  {{tmdb-{r['tmdb_id']}}}"
    lines.append(head)
    lines.append("  ✓ confirmed  ● proposed  ⚠ conflict  ✗ rejected  · undecided"
                 "  ◆ frame kept")
    seasons = {s["season"]: s for s in r["seasons"]}
    by_season = OrderedDict()
    for d in r["discs"]:
        by_season.setdefault(d["season"], []).append(d)
    for snum, discs in by_season.items():
        s = seasons.get(snum)
        cov = f"{s['matched']}/{s['total']} matched" if s else ""
        title = f"Season {snum:02d}" if snum is not None else "(no season hint)"
        lines.append(f"\n{title}  ·  {cov}")
        for d in discs:
            warn = f"   ⚠ order: {d['order_warning']}" if d["order_warning"] else ""
            lines.append(f"  ▸ {d['disc']}{warn}")
            hidden = 0
            for t in d["titles"]:
                a = t["assignment"]
                assigned = a and (a["episodes"] or a["status"] == "rejected")
                # collapse pure disc clutter (unassigned extras) unless --all
                if not show_all and not assigned and t["kind"] == "extra":
                    hidden += 1
                    continue
                if a and a["episodes"]:
                    sym = ("⚠" if t["conflict"] and a["status"] == "proposed"
                           else _STATUS_SYM.get(a["status"], "·"))
                    by = (a["decided_by"] or "").split(":")[-1]
                    name = " & ".join(n for n in (a.get("names") or []) if n)
                    eps = "+".join(a["episodes"])
                    asg = (f"{sym} {eps} {name!r} ({by})" if name
                           else f"{sym} {eps} ({by})")
                elif a and a["status"] == "rejected":
                    asg = "✗ rejected"
                else:
                    asg = ("⚠ conflict" if t["conflict"] else "· undecided")
                fr = "  ◆ frame" if t["frames"] else ""
                lines.append(f"      pl{t['pl']:<3} {t['minutes']:>3.0f}m  "
                             f"{_KIND_ABBR.get(t['kind'], t['kind']):<8}  "
                             f"{asg}{fr}")
                lines.extend(_evidence_lines(t["evidence"], "          "))
            if hidden:
                lines.append(f"      … +{hidden} extra title(s) (--all to show)")
    return "\n".join(lines)


def cmd_board(args) -> int:
    conn, err = _open(args)
    if err:
        return err
    disc_id = None
    if args.disc is not None:
        disc_id, derr = _resolve_disc(conn, args.disc)
        if derr:
            conn.close()
            return fail(args, "no-disc", derr)
    season = None
    if args.season is not None:
        season = int(str(args.season).upper().lstrip("S"))
    r = review.board(conn, season=season, disc_id=disc_id)
    conn.close()
    emit(args, r, human=_board_human(r, show_all=args.all))
    return 0


def cmd_export(args) -> int:
    conn, err = _open(args)
    if err:
        return err
    if not args.manifest and not args.rip_script:
        conn.close()
        return fail(args, "no-output", "give --manifest FILE and/or --rip-script FILE")
    r = export_mod.run_export(conn, args)
    conn.close()
    if not r.get("ok"):
        return fail(args, r["error"], r["message"])
    emit(args, r, human=f"exported {r['records']} record(s): "
                        + ", ".join(f"{k}={v}" for k, v in r["written"].items()))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vj", description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true",
                    help="force JSON output even on a TTY")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="create the state file + pull TMDB episodes")
    p_init.add_argument("db", type=Path, help="state file to create")
    p_init.add_argument("--tmdb-id", type=int, required=True, help="TMDB series id")
    p_init.add_argument("--tmdb-api-key", default=None)
    p_init.add_argument("--cache-dir", type=Path, default=Path(".tmdb_cache"))
    p_init.add_argument("--episode-order", default="aired", metavar="ORDER",
                        help="'aired' (default), an alias (dvd, digital, "
                             "absolute, production, story, tv), or a TMDB "
                             "episode-group id")
    p_init.set_defaults(func=cmd_init)

    p_scan = sub.add_parser("scan", help="scan disc images into the state file")
    p_scan.add_argument("db", type=Path, help="existing state file")
    p_scan.add_argument("images", nargs="+", type=Path,
                        help="disc images (.iso) or backup directories")
    p_scan.set_defaults(func=cmd_scan)

    p_enrich = sub.add_parser(
        "enrich", help="add richer episode synopses from a local Wikipedia dump")
    p_enrich.add_argument("source", choices=["wikipedia"],
                          help="synopsis source (only 'wikipedia' for now)")
    p_enrich.add_argument("db", type=Path, help="existing state file")
    p_enrich.add_argument("--snapshot", type=Path, required=True,
                          help="enwiki-<date>-pages-articles-multistream.xml.bz2")
    p_enrich.add_argument("--index", type=Path, required=True,
                          help="the matching …-multistream-index.txt.bz2")
    p_enrich.add_argument("--page", default=None,
                          help="article title (e.g. 'List of <Show> "
                               "(American TV series) episodes'); remembered")
    p_enrich.set_defaults(func=cmd_enrich)

    p_hint = sub.add_parser(
        "hint", help="record box-packaging knowledge (which episodes a disc holds)")
    p_hint.add_argument("what", choices=["disc"],
                        help="what the hint is about (only 'disc' for now)")
    p_hint.add_argument("db", type=Path, help="existing state file")
    p_hint.add_argument("--disc", required=True,
                        help="disc basename or id (may be entered before scanning)")
    p_hint.add_argument("--season", type=int, default=None,
                        help="season the --episodes numbers belong to")
    p_hint.add_argument("--episodes", default=None,
                        help="episode numbers on the disc, e.g. '1-4,6' (needs --season)")
    p_hint.add_argument("--titles", nargs="+", default=None,
                        help="episode titles on the disc (resolved to numbers via TMDB)")
    p_hint.add_argument("--source", default="packaging",
                        help="provenance label (default: packaging)")
    p_hint.set_defaults(func=cmd_hint)

    p_run = sub.add_parser("run", help="run a heuristic as an evidence producer")
    p_run.add_argument("heuristic", nargs="?", default=None,
                       help="align | ocr | synopsis (or --list)")
    p_run.add_argument("db", type=Path, nargs="?", help="existing state file")
    p_run.add_argument("--list", action="store_true",
                       help="list available heuristics and exit")
    p_run.add_argument("--disc", help="restrict to one disc (basename or id)")
    p_run.add_argument("--all", action="store_true",
                       help="ocr/synopsis: every episode-candidate title on all "
                            "discs (fast now that the text-region gate prunes "
                            "scene frames)")
    p_run.add_argument("--title", type=int, action="append",
                       help="restrict to a title (id); repeatable (ocr/synopsis)")
    p_run.add_argument("--vlm-model", default="qwen3-vl:2B")
    p_run.add_argument("--ocr-engine", choices=("auto", "tesseract", "vlm"),
                       default="auto")
    p_run.add_argument("--ocr-accept", type=float, default=0.8,
                       help="min score to accept an OCR/synopsis match (0-1)")
    p_run.add_argument("--include-specials", action="store_true",
                       help="add S00 specials to the match pool (for "
                            "identifying a leftover title as a special)")
    p_run.add_argument("--no-text-filter", dest="text_filter",
                       action="store_false",
                       help="disable the EAST text-region gate on the VLM pass. "
                            "Turn OFF for shows whose title is painted into the "
                            "scene art (Adventure Time), where EAST may not see "
                            "it as text and could prune the real card")
    p_run.add_argument("--scratch-dir", type=Path, default=None,
                       help="dir for temp rips/frames (real disk, not tmpfs)")
    p_run.add_argument("--ollama-host",
                       default=os.environ.get("OLLAMA_HOST",
                                              "http://localhost:11434"))
    p_run.add_argument("--synopsis-windows", type=int, default=None,
                       help="synopsis: opt into SAMPLED transcription with N "
                            "dialogue windows instead of the default whole-episode "
                            "pass (faster, but can miss identifying lines between "
                            "windows)")
    p_run.add_argument("--synopsis-length", type=float, default=None,
                       help="synopsis: seconds of audio per window when "
                            "--synopsis-windows is set (default 40)")
    p_run.add_argument("--judge-model", default=None,
                       help="synopsis: TEXT model for the synopsis judge (default "
                            "a text model, NOT the --vlm-model). A `claude-*` id "
                            "(e.g. claude-sonnet-5) routes to the Anthropic API "
                            "via ANTHROPIC_API_KEY instead of Ollama.")
    p_run.add_argument("--retranscribe", action="store_true",
                       help="synopsis: force fresh transcript extraction, ignoring "
                            "any cached transcript (default: reuse the stored one)")
    p_run.add_argument("--transcript-source", choices=["auto", "subtitle", "audio"],
                       default="auto",
                       help="synopsis: dialogue source — auto prefers DVD closed "
                            "captions (exact, whole-episode, near-instant) and "
                            "falls back to whisper audio; subtitle/audio force one")
    p_run.add_argument("--synopsis-source", choices=["auto", "wikipedia", "tmdb"],
                       default="auto",
                       help="synopsis: which plot summary to judge against "
                            "(auto = Wikipedia if enriched, else TMDB)")
    p_run.set_defaults(func=cmd_run)

    p_status = sub.add_parser("status", help="coverage summary")
    p_status.add_argument("db", type=Path)
    p_status.set_defaults(func=cmd_status)

    p_gaps = sub.add_parser("gaps", help="titles needing attention + missing eps")
    p_gaps.add_argument("db", type=Path)
    p_gaps.add_argument("--threshold", type=float, default=0.5,
                        help="min confidence for conflict detection")
    p_gaps.set_defaults(func=cmd_gaps)

    p_board = sub.add_parser("board", help="rich overview: every disc's titles + "
                             "assignment + evidence in one table")
    p_board.add_argument("db", type=Path)
    p_board.add_argument("--season", help="limit to one season (N or Sxx)")
    p_board.add_argument("--disc", help="limit to one disc (basename or id)")
    p_board.add_argument("--all", action="store_true",
                         help="show every title incl. unassigned extras")
    p_board.set_defaults(func=cmd_board)

    p_show = sub.add_parser("show", help="all evidence + assignment for one thing")
    p_show.add_argument("db", type=Path)
    p_show.add_argument("--title", type=int)
    p_show.add_argument("--episode", help="SxxEyy")
    p_show.set_defaults(func=cmd_show)

    p_frame = sub.add_parser("frame", help="dump a title's retained OCR frame")
    p_frame.add_argument("db", type=Path)
    p_frame.add_argument("--title", type=int, required=True)
    p_frame.add_argument("--category", default="title-card-ocr")
    p_frame.add_argument("--out", type=Path, help="output image path")
    p_frame.set_defaults(func=cmd_frame)

    p_resolve = sub.add_parser("resolve",
                               help="propose assignments from agreeing evidence")
    p_resolve.add_argument("db", type=Path)
    p_resolve.add_argument("--threshold", type=float, default=0.5)
    p_resolve.set_defaults(func=cmd_resolve)

    p_assign = sub.add_parser("assign", help="confirm a title -> episode(s)")
    p_assign.add_argument("db", type=Path)
    p_assign.add_argument("--title", type=int, required=True)
    p_assign.add_argument("--episode", action="append", required=True,
                          help="SxxEyy; repeat for a two-parter")
    p_assign.add_argument("--note")
    p_assign.add_argument("--agent", action="store_true",
                          help="record decided_by=agent (default: human)")
    p_assign.set_defaults(func=cmd_assign)

    p_confirm = sub.add_parser("confirm", help="accept a title's standing proposal")
    p_confirm.add_argument("db", type=Path)
    p_confirm.add_argument("--title", type=int, required=True)
    p_confirm.add_argument("--agent", action="store_true")
    p_confirm.set_defaults(func=cmd_confirm)

    p_reject = sub.add_parser("reject", help="mark a title as not an episode")
    p_reject.add_argument("db", type=Path)
    p_reject.add_argument("--title", type=int, required=True)
    p_reject.add_argument("--note")
    p_reject.add_argument("--agent", action="store_true")
    p_reject.set_defaults(func=cmd_reject)

    p_auto = sub.add_parser("auto", help="run the whole chain: align→resolve→"
                            "(escalate OCR)→resolve→elimination→resolve")
    p_auto.add_argument("db", type=Path)
    p_auto.add_argument("--threshold", type=float, default=0.5,
                        help="resolve/conflict confidence threshold")
    p_auto.add_argument("--ocr", choices=("auto", "always", "never"),
                        default="auto",
                        help="OCR escalation: 'auto' (only order-unverifiable "
                             "discs when a VLM + on-screen titles are present), "
                             "'always', or 'never'")
    p_auto.add_argument("--vlm-model", default="qwen3-vl:2B")
    p_auto.add_argument("--ocr-engine", choices=("auto", "tesseract", "vlm"),
                        default="auto")
    p_auto.add_argument("--ocr-accept", type=float, default=0.8)
    p_auto.add_argument("--no-text-filter", dest="text_filter",
                        action="store_false",
                        help="disable the EAST text-region gate on OCR "
                             "(for scene-art title cards, e.g. Adventure Time)")
    p_auto.add_argument("--include-specials", action="store_true",
                        help="add S00 specials to the OCR match pool")
    p_auto.add_argument("--scratch-dir", type=Path, default=None,
                        help="dir for temp rips/frames (real disk, not tmpfs)")
    p_auto.add_argument("--ollama-host",
                        default=os.environ.get("OLLAMA_HOST",
                                               "http://localhost:11434"))
    p_auto.set_defaults(func=cmd_auto)

    p_play = sub.add_parser("play", help="launch a player on a candidate title")
    p_play.add_argument("db", type=Path)
    p_play.add_argument("target", nargs="?",
                        help="episode to play (SxxEyy) — the title assigned to it")
    p_play.add_argument("--title", type=int, help="play a title by id instead")
    p_play.add_argument("--player", default="vlc", help="player command (default vlc)")
    p_play.add_argument("--start", type=float, default=0.0,
                        help="start N seconds in")
    p_play.add_argument("--at-card", action="store_true",
                        help="start near the retained OCR title-card frame")
    p_play.add_argument("--print", action="store_true",
                        help="print the command instead of launching it")
    p_play.set_defaults(func=cmd_play)

    p_exp = sub.add_parser("export", help="manifest + rip script from assignments")
    p_exp.add_argument("db", type=Path)
    p_exp.add_argument("--manifest", type=Path, help="write manifest JSON here")
    p_exp.add_argument("--rip-script", type=Path, help="write bash rip script here")
    p_exp.add_argument("--include-proposed", action="store_true",
                       help="also export proposed (not just confirmed) assignments")
    p_exp.add_argument("--handbrake-preset", default="Fast 1080p30")
    p_exp.add_argument("--output-prefix", type=Path,
                       help="rip-script PREFIX (output root)")
    p_exp.set_defaults(func=cmd_export)

    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
