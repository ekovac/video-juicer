#!/usr/bin/env python3
"""Map DVD/Blu-ray disc-image titles to TMDB episodes.

Metadata-first pipeline (see IMPLEMENTATION_PLAN.md):
  1. Scan each disc image's title table (lsdvd for DVD, MPLS parsing for
     Blu-ray) — metadata only, the multi-GB video payload is never read.
  2. Fetch episode lists with runtimes from TMDB (cached locally).
  3. Classify titles into episode candidates with structural evidence
     (play-all chapter match, stream-signature clustering), then align the
     candidate sequence against the TMDB episode sequence with a monotonic
     DP alignment.
  4. Emit manifest.json plus a human-readable report; optionally verify
     low-confidence matches by OCRing title cards with a local VLM (Ollama).

Requires on PATH: lsdvd, 7z. Optional: mencoder + ffmpeg + ollama (--verify).
TMDB_API_KEY must be in the environment (or pass --tmdb-api-key).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Re-export the pipeline so `import identify_episodes` (and the CLI) sees the
# whole API in one namespace; the implementation lives in discs.py / identify.py.
from discs import *          # noqa: F401,F403
from discs import _parse_hb_titles   # noqa: F401  (underscore: not via *)
from identify import *       # noqa: F401,F403


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="*", type=Path,
                    help="disc images (.iso) or backup directories")
    ap.add_argument("--tv-id", type=int, help="TMDB series id")
    ap.add_argument("--tmdb-api-key", default=os.environ.get("TMDB_API_KEY"))
    ap.add_argument("--out", type=Path, default=Path("manifest.json"))
    ap.add_argument("--cache-dir", type=Path, default=Path(".tmdb_cache"))
    ap.add_argument("--episode-order", default="aired", metavar="ORDER",
                    help="episode ordering to match against: 'aired' "
                         "(default), an alias (dvd, digital, absolute, "
                         "production, story, tv), or an explicit TMDB "
                         "episode-group id")
    ap.add_argument("--emit-rip-commands", action="store_true")
    ap.add_argument("--from-manifest", type=Path, metavar="FILE",
                    help="emit rip commands from an existing manifest and "
                         "exit — no disc scanning or OCR")
    ap.add_argument("--merge", action="store_true",
                    help="merge this run's results into the existing --out "
                         "manifest (replace only the discs processed now)")
    ap.add_argument("--handbrake-preset", default="Fast 1080p30",
                    metavar="PRESET",
                    help='HandBrake preset for rip commands (default: "Fast 1080p30")')
    ap.add_argument("--output-prefix", type=Path, metavar="DIR",
                    help="prepend this path to rip-command output files (e.g. a "
                         "target disk); the Plex/Jellyfin tree is built under it")
    ap.add_argument("--verify", action="store_true",
                    help="OCR title cards of low-confidence matches via Ollama")
    ap.add_argument("--verify-all", action="store_true",
                    help="OCR every matched title, not just low-confidence ones")
    ap.add_argument("--spot-check", action="store_true",
                    help="OCR each disc's FIRST and LAST matched episode as a "
                         "cheap check; if either disagrees with the alignment, "
                         "fully verify that disc. The boundary episodes bracket "
                         "the disc, so a dropped/shifted title (which moves the "
                         "whole run) shows up there")
    ap.add_argument("--ocr-identify", action="store_true",
                    help="identify every candidate playlist by OCRing its "
                         "title card instead of metadata alignment — for "
                         "irregular discs (duplicate/combined playlists)")
    ap.add_argument("--auto", action="store_true",
                    help="run the metadata path, and if a disc's episode order "
                         "is unverifiable, auto-escalate to --ocr-identify "
                         "(gated on a 2-playlist card-presence probe + an "
                         "available VLM)")
    ap.add_argument("--ocr-accept", type=float, default=0.8,
                    help="min fuzzy score to accept an OCR title match (0-1)")
    ap.add_argument("--scratch-dir", type=Path, default=None,
                    help="directory for temporary rips/frames (put on disk, "
                         "not tmpfs, for large OCR runs)")
    ap.add_argument("--vlm-model", default="qwen3-vl:2B")
    ap.add_argument("--ocr-engine", choices=("auto", "tesseract", "vlm"),
                    default="auto",
                    help="OCR engine for title cards: 'auto' (fast Tesseract "
                         "first, VLM fallback for stylized cards — default), "
                         "'tesseract' only, or 'vlm' only")
    ap.add_argument("--ollama-host",
                    default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    # Pure transform: emit rip commands from a saved manifest, no scanning.
    if args.from_manifest:
        records = json.loads(args.from_manifest.read_text())
        emit_rip_commands(records, args.handbrake_preset, args.output_prefix)
        return 0

    if not args.images:
        ap.error("no disc images given (required unless --from-manifest)")
    if not args.tv_id:
        ap.error("--tv-id is required")
    if not args.tmdb_api_key:
        ap.error("TMDB_API_KEY not set and --tmdb-api-key not given")
    for tool in ("lsdvd", "7z"):
        if not shutil.which(tool):
            ap.error(f"required tool not on PATH: {tool}")

    tmdb = Tmdb(args.tmdb_api_key, args.cache_dir / str(args.tv_id))
    series = tmdb.series(args.tv_id)
    show = series["name"]
    fad = series.get("first_air_date") or ""
    year = int(fad[:4]) if fad[:4].isdigit() else None
    series_rt = series.get("episode_run_time") or []
    fallback_rt = series_rt[0] * 60.0 if series_rt else None
    if args.episode_order != "aired":
        seasons, specials = grouped_seasons(tmdb, args.tv_id,
                                            args.episode_order, fallback_rt)
    else:
        season_numbers = [s["season_number"] for s in series["seasons"]
                          if s["season_number"] > 0]
        seasons = {n: tmdb.season_episodes(args.tv_id, n, fallback_rt)
                   for n in season_numbers}
        specials = (tmdb.season_episodes(args.tv_id, 0, fallback_rt)
                    if any(s["season_number"] == 0 for s in series["seasons"]) else [])
    log.info("%s: %d seasons, %d episodes", show, len(seasons),
             sum(len(v) for v in seasons.values()))

    discs = [scan_disc(p) for p in args.images]

    all_assignments: list[Assignment] = []
    all_leftovers: list[tuple[Disc, Title]] = []
    all_missed: list[Episode] = []
    if args.ocr_identify:
        log.info("OCR-identify mode: matching each candidate playlist by its "
                 "on-screen title card via %s", args.vlm_model)
        with tempfile.TemporaryDirectory(prefix="identify-eps-ocr-",
                                         dir=args.scratch_dir) as tmp:
            all_assignments, all_leftovers, all_missed = ocr_identify(
                discs, seasons, args, Path(tmp))
    else:
        for season, group in group_discs(discs):
            pool = (seasons.get(season) or
                    [e for n in sorted(seasons) for e in seasons[n]])
            runtimes = sorted(e.runtime for e in pool if e.runtime)
            expected = runtimes[len(runtimes) // 2] if runtimes else 1320.0
            cands: list[tuple[Disc, Title]] = []
            for d in group:
                for t in classify_disc(d, expected):
                    cands.append((d, t))
            assignments, leftovers, missed = align(cands, pool)
            all_assignments += assignments
            all_leftovers += leftovers
            all_missed += missed

        # --auto: if any disc's order is unverifiable, escalate to OCR — but
        # only if a VLM is up and a quick probe shows the show captions titles.
        per_disc: dict[Path, list[Assignment]] = {}
        for a in all_assignments:
            per_disc.setdefault(a.disc.path, []).append(a)
        unverifiable = [
            asgs[0].disc for asgs in per_disc.values()
            if not assess_ordering(asgs[0].disc,
                                   sorted(asgs, key=lambda a: a.title.order_key))[0]]
        if args.auto and unverifiable:
            with tempfile.TemporaryDirectory(prefix="identify-eps-auto-",
                                             dir=args.scratch_dir) as tmp:
                if not vlm_available(args.vlm_model, args.ollama_host):
                    log.warning("%d disc(s) unverifiable but no VLM at %s — "
                                "keeping the metadata mapping (flagged)",
                                len(unverifiable), args.ollama_host)
                elif probe_card_presence(discs, seasons, args, Path(tmp)):
                    log.warning("episode order unverifiable; on-screen titles "
                                "present — escalating to OCR-identify")
                    args.ocr_identify = True   # records get title-card provenance
                    all_assignments, all_leftovers, all_missed = ocr_identify(
                        discs, seasons, args, Path(tmp))
                else:
                    log.warning("episode order unverifiable and no on-screen "
                                "titles found — keeping the metadata mapping "
                                "(flagged); neither path can confirm it")

    # leftovers may be TMDB specials: best-effort runtime match against S0
    special_notes: dict[int, str] = {}
    for idx, (d, t) in enumerate(all_leftovers):
        hits = [e for e in specials
                if e.runtime and abs(e.runtime - t.duration) <= CLEAN_MATCH_TOL]
        if len(hits) == 1:
            special_notes[idx] = f"possible special S00E{hits[0].number:02d} {hits[0].name!r}"

    # optional VLM verification of weak matches (already done in ocr-identify)
    if (args.verify or args.verify_all) and not args.ocr_identify:
        targets = [a for a in all_assignments
                   if args.verify_all or a.confidence == "low"]
        log.info("verifying %d title(s) via %s", len(targets), args.vlm_model)
        with tempfile.TemporaryDirectory(prefix="identify-eps-",
                                         dir=args.scratch_dir) as tmp:
            for a in targets:
                verify_assignment(a, seasons, args.vlm_model, args.ollama_host,
                                  Path(tmp), args.ocr_accept, args.ocr_engine)
    # cheap spot-check: OCR each disc's first and last matched episode; if
    # either disagrees with the alignment, fully verify that disc. A dropped/
    # shifted title renumbers the whole contiguous run, so the boundary episodes
    # reveal it (Sonic SatAM) — without OCRing every episode.
    elif args.spot_check and not args.ocr_identify:
        from collections import defaultdict
        per_disc: dict[Path, list] = defaultdict(list)
        for a in all_assignments:
            per_disc[a.disc.path].append(a)
        log.info("spot-check (first+last/disc) across %d disc(s) via %s",
                 len(per_disc), args.vlm_model)
        with tempfile.TemporaryDirectory(prefix="identify-eps-",
                                         dir=args.scratch_dir) as tmp:
            for path, asgs in per_disc.items():
                ordered = sorted(asgs, key=lambda a: a.title.order_key)
                sample = [ordered[i] for i in sorted({0, len(ordered) - 1})]
                results = [verify_assignment(a, seasons, args.vlm_model,
                                             args.ollama_host, Path(tmp),
                                             args.ocr_accept, args.ocr_engine)
                           for a in sample]
                # Escalate UNLESS both boundaries positively confirmed. A
                # disagreement is an error; a no-readable-card boundary means we
                # couldn't verify the disc cheaply (Sonic disc 3's boundary
                # episodes truncate while its middle reads) — either way, do the
                # full check rather than silently trust the alignment.
                if all(r is True for r in results):
                    continue
                why = ("DISAGREED" if any(r is False for r in results)
                       else "couldn't confirm (no readable card at first/last)")
                log.warning("%s: spot-check %s — verifying the whole disc",
                            path.name, why)
                for a in ordered:
                    if a not in sample:
                        verify_assignment(a, seasons, args.vlm_model,
                                          args.ollama_host, Path(tmp),
                                          args.ocr_accept, args.ocr_engine)

    # ---- report ----
    records = []
    for a in sorted(all_assignments,
                    key=lambda a: (a.episodes[0].season, a.episodes[0].number)):
        e0 = a.episodes[0]
        records.append({
            "image": str(a.disc.path),
            "title": rip_title_number(a.disc, a.title), "kind": "episode",
            "season": e0.season,
            "episodes": [e.number for e in a.episodes],
            "episode_name": " & ".join(e.name for e in a.episodes),
            "title_seconds": round(a.title.duration, 1),
            "tmdb_seconds": sum(e.runtime or 0 for e in a.episodes),
            "delta_seconds": round(a.delta, 1),
            "confidence": a.confidence,
            "verified_by_titlecard": a.verified_name is not None,
            "identified_by": a.method or (
                "title-card" if args.ocr_identify else "runtime-align"),
            "video_format": a.title.video_format,
            "audio_format": a.title.audio_format,
            "suggested_filename": suggested_filename(show, a.episodes, year,
                                                     args.tv_id),
        })
        if e0.aired_season is not None:
            records[-1]["aired"] = [
                f"S{e.aired_season:02d}E{e.aired_number:02d}" for e in a.episodes]
    for idx, (d, t) in enumerate(all_leftovers):
        records.append({
            "image": str(d.path), "title": rip_title_number(d, t),
            "kind": "extra",
            "title_seconds": round(t.duration, 1),
            "note": special_notes.get(idx, ""),
        })
    for d in discs:
        for t in d.titles:
            if t.kind == "play-all":
                records.append({"image": str(d.path),
                                "title": rip_title_number(d, t),
                                "kind": "play_all",
                                "title_seconds": round(t.duration, 1)})
    before = len(records)
    total = write_manifest(args.out, records, args.merge, args.images)
    if args.merge:
        log.info("merged %d new record(s) into %s (%d total)",
                 before, args.out, total)
    else:
        log.info("wrote %s (%d records)", args.out, total)
    # re-read for the validation summary so it reflects the merged file
    records = json.loads(args.out.read_text())

    # human-readable table
    print(f"\n{show} — {len(all_assignments)} titles matched")
    for a in sorted(all_assignments,
                    key=lambda a: (a.episodes[0].season, a.episodes[0].number)):
        e0 = a.episodes[0]
        nums = "".join(f"E{e.number:02d}" for e in a.episodes)
        flag = {"high": " ", "medium": " ", "low": "?"}[a.confidence]
        ver = " [verified]" if a.verified_name else ""
        print(f"  S{e0.season:02d}{nums} {flag} {a.disc.path.name} "
              f"title {rip_title_number(a.disc, a.title):>2} "
              f"Δ{a.delta:5.1f}s  {' & '.join(e.name for e in a.episodes)}{ver}")
    if all_missed:
        print("\nMISSING EPISODES (not found on any disc):")
        for e in all_missed:
            print(f"  S{e.season:02d}E{e.number:02d} {e.name}")
    if all_leftovers:
        print(f"\nExtras / unmatched titles: {len(all_leftovers)}")
        for idx, (d, t) in enumerate(all_leftovers):
            note = f"  ({special_notes[idx]})" if idx in special_notes else ""
            print(f"  {d.path.name} title {rip_title_number(d, t):>2} "
                  f"{t.duration/60:6.1f} min{note}")

    if args.emit_rip_commands:
        print("\n# rip commands")
        emit_rip_commands(records, args.handbrake_preset, args.output_prefix)

    # validation summary — count from the written manifest (so a --merge run
    # reflects total coverage across all discs, not just the ones re-run)
    by_season: dict[int, int] = {}
    for r in records:
        if r.get("kind") == "episode":
            for n in r.get("episodes", []):
                by_season[r["season"]] = by_season.get(r["season"], 0) + 1
    covered_seasons = {r["season"] for r in records if r.get("kind") == "episode"}
    problems = [f"S{s:02d}: matched {by_season.get(s, 0)}/{len(eps)}"
                for s, eps in sorted(seasons.items())
                if by_season.get(s, 0) != len(eps) and s in covered_seasons]

    # Cross-disc duplicates: the same episode claimed by records on different
    # discs. (A single + its combined double live on one disc, so same image —
    # those are fine.) Per-run collision resolution can't see across discs, so
    # a --merge can leave two discs both claiming an episode; flag it.
    ep_imgs: dict[tuple, set] = {}
    for r in records:
        if r.get("kind") == "episode":
            for n in r.get("episodes", []):
                ep_imgs.setdefault((r["season"], n), set()).add(r["image"])
    crossdisc = {k: v for k, v in ep_imgs.items() if len(v) > 1}

    # Orderability: in the metadata path, can each disc's episode ORDER be
    # trusted, or does identity rest on an unreliable playlist position?
    # (OCR-identify pins identity by content, so it's exempt.)
    unverifiable = []
    if not args.ocr_identify:
        per_disc: dict[Path, list[Assignment]] = {}
        for a in all_assignments:
            per_disc.setdefault(a.disc.path, []).append(a)
        for path, asgs in per_disc.items():
            asgs.sort(key=lambda a: a.title.order_key)
            ok, reason = assess_ordering(asgs[0].disc, asgs)
            if not ok:
                unverifiable.append((path.name, reason))

    if problems:
        print("\nWARNING: incomplete coverage: " + "; ".join(problems))
    if crossdisc:
        print("\nWARNING: episode claimed on multiple discs (pick one):")
        for (s, n), imgs in sorted(crossdisc.items()):
            print(f"  S{s:02d}E{n:02d}: " + ", ".join(
                Path(i).name for i in sorted(imgs)))
    if unverifiable:
        print("\nWARNING: episode ORDER unverifiable from metadata — the "
              "mapping below is a guess. Re-run with --ocr-identify to pin "
              "identities by title card:")
        for name, reason in unverifiable:
            print(f"  {name}: {reason}")

    # Video-format outliers: episodes authored at a lower quality than the rest
    # (e.g. Avatar's 480i Sozin's Comet finale among a 1080p season). Correctly
    # identified, but the selected source is inferior — flag before ripping.
    maj_fmt, outliers = format_outliers(all_assignments)
    if outliers:
        print(f"\nWARNING: video format differs from the majority ({maj_fmt}) — "
              "these episodes are a lower-quality source on the disc:")
        for a in sorted(outliers, key=lambda a: (a.episodes[0].season,
                                                 a.episodes[0].number)):
            e0 = a.episodes[0]
            af = f", {a.title.audio_format}" if a.title.audio_format else ""
            print(f"  S{e0.season:02d}E{e0.number:02d} {a.title.video_format}{af}"
                  f"  ({a.disc.path.name} title "
                  f"{rip_title_number(a.disc, a.title)})  "
                  f"{' & '.join(e.name for e in a.episodes)}")
    return 1 if (problems or crossdisc or unverifiable or outliers) else 0


if __name__ == "__main__":
    sys.exit(main())
