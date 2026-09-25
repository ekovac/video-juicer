"""Export: turn resolved assignments into a manifest + a HandBrake rip script.

Reads the adjudicated `assignment` layer (confirmed by default; add proposed
with --include-proposed) and reuses the existing output builders
(`suggested_filename`, `emit_rip_commands`, incl. the video-format-outlier
split) so the downstream rip workflow is unchanged.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

import state
from discs import Episode, rip_title_number
from identify import emit_rip_commands, suggested_filename


def _episode(conn, eid: int) -> Episode:
    r = conn.execute("SELECT * FROM episode WHERE id=?", (eid,)).fetchone()
    return Episode(season=r["season"], number=r["number"], name=r["name"],
                   runtime=r["runtime"], overview=r["overview"] or "")


def build_records(conn, include_proposed: bool = False) -> list[dict]:
    proj = state.get_project(conn)
    show = proj.get("show_name") or "Show"
    year = int(proj["year"]) if proj.get("year") not in (None, "None") else None
    tmdb_id = int(proj["tmdb_id"]) if proj.get("tmdb_id") else None

    # Every record says which numbering its SxxEyy are in: "S01E03" names a
    # different episode in TMDB aired vs DVD order (Venture Bros), and a media
    # server matches files by number — so the ordering must travel with it.
    order_id = proj.get("episode_order") or "aired"
    order_label = state.order_label(conn)

    wanted = ("confirmed", "proposed") if include_proposed else ("confirmed",)
    disc_cache: dict[int, object] = {}
    records = []
    rows = conn.execute(
        "SELECT a.*, t.disc_id, t.title_number FROM assignment a "
        "JOIN title t ON t.id=a.title_id "
        "WHERE a.status IN (%s) ORDER BY t.disc_id, t.title_number"
        % ",".join("?" * len(wanted)), wanted).fetchall()

    for a in rows:
        eids = json.loads(a["episode_ids_json"])
        if not eids:
            continue
        eps = [_episode(conn, e) for e in eids]
        disc = disc_cache.setdefault(a["disc_id"],
                                     state.load_disc(conn, a["disc_id"]))
        title = next(t for t in disc.titles if t.id == a["title_number"])
        # was this title's identity confirmed by a title card?
        oc = state.get_frame(conn, a["title_id"], "title-card-ocr")
        e0 = eps[0]
        aired = [r for r in (conn.execute(
            "SELECT aired_season, aired_number FROM episode WHERE id=?",
            (e,)).fetchone() for e in eids) if r["aired_season"] is not None]
        records.append({
            "image": str(disc.path),
            "title": rip_title_number(disc, title),
            "kind": "episode",
            "season": e0.season,
            "episodes": [e.number for e in eps],
            "episode_name": " & ".join(e.name for e in eps),
            "episode_order": order_label,
            "episode_order_id": order_id,
            # the same episodes in TMDB aired numbering, when the project isn't
            # aired (a cross-reference for libraries set to aired order)
            **({"aired": [f"S{r['aired_season']:02d}E{r['aired_number']:02d}"
                          for r in aired]} if order_id != "aired" and aired else {}),
            "title_seconds": round(title.duration, 1),
            "tmdb_seconds": sum(e.runtime or 0 for e in eps),
            "confidence": a["status"],
            "identified_by": a["decided_by"],
            "verified_by_titlecard": oc is not None,
            "video_format": title.video_format,
            "audio_format": title.audio_format,
            "suggested_filename": suggested_filename(show, eps, year, tmdb_id),
        })
    return records


def run_export(conn, args) -> dict:
    records = build_records(conn, include_proposed=args.include_proposed)
    if not records:
        return {"ok": False, "error": "nothing-to-export",
                "message": ("no confirmed assignments"
                            + ("" if args.include_proposed
                               else " (try --include-proposed)"))}
    written = {}
    if args.manifest:
        Path(args.manifest).write_text(json.dumps(records, indent=2))
        written["manifest"] = str(args.manifest)
    if args.rip_script:
        with open(args.rip_script, "w") as fh, contextlib.redirect_stdout(fh):
            emit_rip_commands(records, args.handbrake_preset, args.output_prefix)
        written["rip_script"] = str(args.rip_script)
    return {"ok": True, "records": len(records), "written": written,
            "include_proposed": args.include_proposed}
