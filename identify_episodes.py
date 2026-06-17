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
import ast
import base64
import fcntl
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("identify-episodes")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Title:
    """One playable title (DVD title / Blu-ray playlist) on a disc."""

    id: int                      # DVD title number or BD playlist number
    duration: float              # seconds
    chapters: list[float]        # chapter durations, seconds
    n_audio: int = 0
    n_sub: int = 0
    cells: int = 0               # DVD cells / BD play items
    clips: tuple = ()            # BD: referenced .m2ts clip ids (for dedup)
    # classification results
    kind: str = "unknown"        # episode-candidate | play-all | extra | junk
    evidence: float = 0.0        # [-1, 1]; >0 favors episode
    order_key: int = 0           # play order within the disc


@dataclass
class Disc:
    path: Path
    format: str                  # "dvd" | "bluray"
    label: str
    titles: list[Title] = field(default_factory=list)
    season_hint: Optional[int] = None
    disc_hint: Optional[int] = None
    # Blu-ray only: maps a playlist's .mpls id (Title.id, used internally by
    # ffmpeg --playlist) to the HandBrake title index the user rips with.
    hb_map: dict = field(default_factory=dict)


@dataclass
class Episode:
    season: int
    number: int
    name: str
    runtime: Optional[float]     # seconds, None if TMDB has no runtime
    # set when an alternate episode ordering (TMDB episode group) is in use
    aired_season: Optional[int] = None
    aired_number: Optional[int] = None


@dataclass
class Assignment:
    disc: Disc
    title: Title
    episodes: list[Episode]      # usually 1; 2 for a merged two-parter
    delta: float
    confidence: str              # high | medium | low
    verified_name: Optional[str] = None
    method: str = ""             # provenance, e.g. "elimination"


def run(cmd: list[str], timeout: int = 300, **kw) -> subprocess.CompletedProcess:
    log.debug("run: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)


# ---------------------------------------------------------------------------
# Stage 1: disc scanning (metadata only)
# ---------------------------------------------------------------------------


def detect_format(path: Path) -> str:
    if path.is_dir():
        if (path / "BDMV").is_dir():
            return "bluray"
        if (path / "VIDEO_TS").is_dir():
            return "dvd"
        raise ValueError(f"{path}: directory has neither BDMV/ nor VIDEO_TS/")
    # Image file: list the filesystem root without extracting anything.
    proc = run(["7z", "l", "-ba", "-slt", str(path)])
    names = re.findall(r"^Path = (.+)$", proc.stdout, re.M)
    tops = {n.split("/")[0].upper() for n in names}
    if "BDMV" in tops:
        return "bluray"
    if "VIDEO_TS" in tops:
        return "dvd"
    raise ValueError(f"{path}: no BDMV/ or VIDEO_TS/ in image root")


def scan_dvd(path: Path) -> Disc:
    """Read the DVD title table via lsdvd (IFO metadata only)."""
    proc = run(["lsdvd", "-Oy", "-c", "-a", "-s", str(path)])
    out = proc.stdout
    # lsdvd prints libdvdread warnings on stdout before the dict.
    idx = out.find("lsdvd = {")
    if idx < 0:
        raise RuntimeError(f"lsdvd failed on {path}: {proc.stderr.strip() or out.strip()}")
    data = ast.literal_eval(out[idx + len("lsdvd = "):])
    titles = []
    for t in data.get("track", []):
        titles.append(Title(
            id=t["ix"],
            duration=float(t["length"]),
            chapters=[float(c["length"]) for c in t.get("chapter", [])],
            n_audio=len(t.get("audio", [])),
            n_sub=len(t.get("subp", [])),
            cells=len(t.get("cell", [])) or len(t.get("chapter", [])),
            order_key=t["ix"],
        ))
    return Disc(path=path, format="dvd", label=data.get("title", path.stem), titles=titles)


# --- Blu-ray: parse BDMV/PLAYLIST/*.mpls without touching the m2ts payload ---


def parse_mpls(buf: bytes) -> Optional[dict]:
    """Parse one .mpls playlist. Returns {duration, chapters, clips, n_audio, n_sub}."""
    if len(buf) < 40 or buf[:4] != b"MPLS":
        return None
    playlist_start, mark_start = struct.unpack_from(">II", buf, 8)
    # PlayList block
    n_items, _n_subpaths = struct.unpack_from(">HH", buf, playlist_start + 6)
    pos = playlist_start + 10
    clips, in_times, durations = [], [], []
    n_audio = n_sub = 0
    for _ in range(n_items):
        item_len = struct.unpack_from(">H", buf, pos)[0]
        clip = buf[pos + 2:pos + 7].decode("ascii", "replace")
        in_t, out_t = struct.unpack_from(">II", buf, pos + 14)
        clips.append(clip)
        in_times.append(in_t)
        durations.append((out_t - in_t) / 45000.0)
        pos += 2 + item_len
    # Stream counts live in the STN table at a variable offset; parsing it is
    # fragile, so Blu-ray keeps counts at 0 and relies on duration/chapter
    # evidence only.
    duration = sum(durations)
    # PlaylistMark block: type 1 == chapter mark
    n_marks = struct.unpack_from(">H", buf, mark_start + 4)[0]
    mpos = mark_start + 6
    marks = []  # (playitem_ref, tick)
    for _ in range(n_marks):
        mark_type = buf[mpos + 1]
        ref, tick = struct.unpack_from(">HI", buf, mpos + 2)
        if mark_type == 1:
            marks.append((ref, tick))
        mpos += 14
    # Convert chapter marks to durations within the playlist timeline.
    starts = []
    base = 0.0
    item_base = []
    for d in durations:
        item_base.append(base)
        base += d
    for ref, tick in marks:
        if ref < len(item_base):
            starts.append(item_base[ref] + (tick - in_times[ref]) / 45000.0)
    starts = sorted(s for s in starts if 0 <= s < duration + 1)
    chapters = []
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else duration
        chapters.append(max(0.0, end - s))
    return {"duration": duration, "chapters": chapters, "clips": tuple(clips),
            "n_audio": n_audio, "n_sub": n_sub, "n_items": n_items}


def dedup_subset_playlists(titles: list[Title]) -> list[Title]:
    """Drop playlists that are a near-duplicate of a fuller one.

    TV Blu-rays often author each episode as two playlists: the body alone,
    and the body with a logo/recap clip prepended (e.g. Avatar: clips
    (01094,) vs (01100, 01088, 01094)). The first is a strict subset of the
    second and the same episode. Drop the subset, keep the fuller version
    (which carries the title/recap). The 1.5x length guard stops a long
    play-all (a superset of many episodes) from swallowing the episodes it
    contains."""
    drop = set()
    for a in titles:
        if not a.clips or id(a) in drop:
            continue
        sa = set(a.clips)
        for b in titles:
            if a is b or not b.clips:
                continue
            sb = set(b.clips)
            if sa < sb and b.duration <= a.duration * 1.5:
                drop.add(id(a))     # a is the subset (shorter); b is fuller
                break
    return [t for t in titles if id(t) not in drop]


def order_by_playall(titles: list[Title]) -> Optional[Title]:
    """Use a play-all's clip sequence to set episode order_keys exactly.

    A Blu-ray "play all" is one playlist whose clips are the ordered union of
    the episode clips. Its clip order IS broadcast order — so it pins the
    episodes' sequence even when they're all the same runtime and the .mpls
    numbering is scrambled (Avatar), the one case duration-based ordering and
    the .mpls order both fail. Sets each covered episode's order_key to its
    rank in the play-all and marks the play-all kind="play-all" (so
    assess_ordering trusts the order). Returns the play-all, or None.

    Robust to shared intro/outro clips: a clip in >1 covered episode is shared,
    so episodes are ranked by their *distinguishing* clip's position."""
    from collections import Counter
    if len(titles) < 4:
        return None
    cands = []
    for pa in titles:
        clipset = set(pa.clips)
        cov = [t for t in titles if t is not pa and t.clips
               and t.duration > 600 and set(t.clips) <= clipset]
        if len(cov) >= 3:
            cands.append((pa, cov))
    if not cands:
        return None
    pa, cov = max(cands, key=lambda pc: (len(pc[1]), -pc[0].duration))
    pos = {c: i for i, c in enumerate(pa.clips)}
    freq = Counter(c for t in cov for c in set(t.clips))
    def rank(t):
        uniq = [pos[c] for c in t.clips if freq[c] == 1 and c in pos]
        return min(uniq) if uniq else min(pos[c] for c in t.clips if c in pos)
    for k, t in enumerate(sorted(cov, key=rank)):
        t.order_key = k
    pa.kind = "play-all"
    log.info("play-all clip order: %d episodes ordered from playlist %d",
             len(cov), pa.id)
    return pa


def scan_bluray(path: Path) -> Disc:
    """Read playlists from BDMV/PLAYLIST. Only the small .mpls files are read."""
    playlists: dict[int, bytes] = {}
    if path.is_dir():
        for f in sorted((path / "BDMV" / "PLAYLIST").glob("*.mpls")):
            playlists[int(f.stem)] = f.read_bytes()
        label = path.name
    else:
        proc = run(["7z", "l", "-ba", "-slt", str(path)])
        names = [n for n in re.findall(r"^Path = (.+)$", proc.stdout, re.M)
                 if re.search(r"BDMV[/\\]PLAYLIST[/\\]\d+\.mpls$", n, re.I)]
        for n in sorted(names):
            num = int(Path(n).stem)
            ex = subprocess.run(["7z", "e", "-so", str(path), n],
                                capture_output=True, timeout=120)
            playlists[num] = ex.stdout
        label = path.stem
    titles, seen_clips = [], set()
    for num, buf in sorted(playlists.items()):
        info = parse_mpls(buf)
        if not info:
            continue
        if info["clips"] in seen_clips:   # duplicate/obfuscation playlist
            continue
        seen_clips.add(info["clips"])
        titles.append(Title(
            id=num, duration=info["duration"], chapters=info["chapters"],
            n_audio=info["n_audio"], n_sub=info["n_sub"],
            cells=info["n_items"], clips=info["clips"], order_key=num,
        ))
    titles = dedup_subset_playlists(titles)
    order_by_playall(titles)   # exact ordering when a play-all is present
    disc = Disc(path=path, format="bluray", label=label, titles=titles)
    disc.hb_map = handbrake_title_map(path)
    return disc


def _parse_hb_titles(stdout: str) -> dict[int, int]:
    """Extract {playlist_id: handbrake_title_index} from HandBrakeCLI --json."""
    i = stdout.find("JSON Title Set:")
    if i < 0:
        return {}
    start = stdout.find("{", i)
    depth, end = 0, -1
    for j in range(start, len(stdout)):
        if stdout[j] == "{":
            depth += 1
        elif stdout[j] == "}":
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    if end < 0:
        return {}
    try:
        data = json.loads(stdout[start:end])
    except json.JSONDecodeError:
        return {}
    mapping = {}
    for t in data.get("TitleList", []):
        pl, idx = t.get("Playlist"), t.get("Index")
        if pl is None or idx is None:
            continue
        try:
            mapping[int(pl)] = int(idx)
        except (ValueError, TypeError):
            continue
    return mapping


def handbrake_title_map(path: Path) -> dict[int, int]:
    """Map BD playlist .mpls id -> HandBrake title index (the `-t N` to rip).

    HandBrake enumerates relevant playlists; that numbering differs from raw
    .mpls ids and from a player's title-object list (e.g. VLC). Best-effort:
    returns {} (callers warn and fall back) if HandBrakeCLI is missing or the
    scan fails.
    """
    if not shutil.which("HandBrakeCLI"):
        log.warning("HandBrakeCLI not on PATH; Blu-ray output will use raw "
                    ".mpls ids, which do NOT match HandBrake's -t numbers")
        return {}
    try:
        proc = run(["HandBrakeCLI", "-i", str(path), "-t", "0", "--scan",
                    "--json"], timeout=300)
    except subprocess.TimeoutExpired:
        log.warning("HandBrake scan timed out on %s", path.name)
        return {}
    mapping = _parse_hb_titles(proc.stdout)
    if not mapping:
        log.warning("HandBrake scan yielded no titles for %s", path.name)
    return mapping


def rip_title_number(disc: Disc, title: Title) -> int:
    """The title number to pass to a ripper. For Blu-ray, translate the
    internal .mpls id to HandBrake's title index; for DVD the id already is
    the rip title."""
    if disc.format == "bluray" and disc.hb_map:
        hb = disc.hb_map.get(title.id)
        if hb is not None:
            return hb
        log.warning("no HandBrake title for playlist %d on %s; using .mpls id",
                    title.id, disc.path.name)
    return title.id


_SEASON_WORD = r"(?:season|book|vol(?:ume)?|part|series|chapter)"
SEASON_DISC_RE = [
    re.compile(r"[Ss](?:eason[ ._]?)?(\d{1,2})[ ._-]?[Dd](?:isc)?[ ._]?(\d{1,2})"),
    # season-word N ... disc N — e.g. Avatar "Book_1_Disc_1", "VOLUME 2 DISC 3"
    re.compile(_SEASON_WORD + r"[ ._]?(\d{1,2}).*?dis[ck][ ._]?(\d{1,2})", re.I),
    re.compile(r"[Ss](?:eason)?[ ._]?(\d{1,2})"),
    # bare season-word N label, no disc number
    re.compile(_SEASON_WORD + r"[ ._]?(\d{1,2})", re.I),
]


def parse_hints(disc: Disc) -> None:
    for text in (disc.path.name, disc.label):
        for rx in SEASON_DISC_RE:
            m = rx.search(text)
            if m:
                disc.season_hint = int(m.group(1))
                if m.lastindex and m.lastindex >= 2:
                    disc.disc_hint = int(m.group(2))
                return


def scan_disc(path: Path) -> Disc:
    fmt = detect_format(path)
    disc = scan_dvd(path) if fmt == "dvd" else scan_bluray(path)
    parse_hints(disc)
    log.info("%s [%s] %s: %d titles, hint S%sD%s", path.name, fmt, disc.label,
             len(disc.titles), disc.season_hint, disc.disc_hint)
    return disc


# ---------------------------------------------------------------------------
# Stage 2: TMDB metadata
# ---------------------------------------------------------------------------


class Tmdb:
    BASE = "https://api.themoviedb.org/3"

    def __init__(self, api_key: str, cache_dir: Path):
        self.key = api_key
        self.cache = cache_dir
        self.cache.mkdir(parents=True, exist_ok=True)

    def get(self, route: str) -> dict:
        cache_file = self.cache / (route.strip("/").replace("/", "_") + ".json")
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        url = f"{self.BASE}{route}?api_key={self.key}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        cache_file.write_text(json.dumps(data))
        return data

    def series(self, tv_id: int) -> dict:
        return self.get(f"/tv/{tv_id}")

    def season_episodes(self, tv_id: int, season: int,
                        fallback_runtime: Optional[float]) -> list[Episode]:
        data = self.get(f"/tv/{tv_id}/season/{season}")
        eps = []
        runtimes = [e["runtime"] for e in data["episodes"] if e.get("runtime")]
        median = sorted(runtimes)[len(runtimes) // 2] * 60.0 if runtimes else fallback_runtime
        for e in data["episodes"]:
            rt = e.get("runtime")
            eps.append(Episode(
                season=season, number=e["episode_number"], name=e["name"],
                runtime=rt * 60.0 if rt else median,
            ))
        return eps

    def episode_groups(self, tv_id: int) -> list[dict]:
        return self.get(f"/tv/{tv_id}/episode_groups")["results"]

    def episode_group(self, group_id: str) -> dict:
        return self.get(f"/tv/episode_group/{group_id}")


# TMDB episode-group types. Note: "DVD Order" groups are type 3 (verified on
# live data for multiple series); type 4 is *Digital* order, 6 is Production.
GROUP_TYPE_ALIASES = {
    "absolute": 2, "dvd": 3, "digital": 4, "story": 5, "production": 6, "tv": 7,
}


def grouped_seasons(tmdb: Tmdb, tv_id: int, selector: str,
                    fallback_rt: Optional[float]
                    ) -> tuple[dict[int, list[Episode]], list[Episode]]:
    """Build season pools from a TMDB episode group instead of aired order.

    `selector` is an alias from GROUP_TYPE_ALIASES or an explicit group id.
    Returns ({season_number: episodes}, specials) where season numbers are
    the group's own ordering (group order 0 = Specials).
    """
    if selector in GROUP_TYPE_ALIASES:
        wanted = GROUP_TYPE_ALIASES[selector]
        meta = tmdb.episode_groups(tv_id)
        hits = [g for g in meta if g["type"] == wanted]
        if not hits:
            listing = "; ".join(
                f"{g['name']!r} (type {g['type']}, id {g['id']})" for g in meta)
            raise SystemExit(
                f"TMDB has no episode group of type {selector!r} for this "
                f"series. Available groups: {listing or 'none'} — pass the "
                f"group id directly via --episode-order <id>.")
        group_id = hits[0]["id"]
        log.info("using episode group %r (%s)", hits[0]["name"], group_id)
    else:
        group_id = selector
    detail = tmdb.episode_group(group_id)
    pools: dict[int, list[Episode]] = {}
    for g in detail["groups"]:
        rts = [e["runtime"] for e in g["episodes"] if e.get("runtime")]
        median = sorted(rts)[len(rts) // 2] * 60.0 if rts else fallback_rt
        eps = []
        for e in sorted(g["episodes"], key=lambda e: e["order"]):
            rt = e.get("runtime")
            eps.append(Episode(
                season=g["order"], number=e["order"] + 1, name=e["name"],
                runtime=rt * 60.0 if rt else median,
                aired_season=e["season_number"],
                aired_number=e["episode_number"],
            ))
        pools[g["order"]] = eps
    specials = pools.pop(0, [])
    return pools, specials


# ---------------------------------------------------------------------------
# Stage 3a: per-disc classification (permissive candidates + evidence)
# ---------------------------------------------------------------------------

PLAYALL_CHAPTER_TOL = 2.0     # observed agreement on real discs is 0.1 s
CLEAN_MATCH_TOL = 90.0        # TMDB runtimes round to whole minutes
HARD_MATCH_TOL = 420.0


def detect_play_all(titles: list[Title]) -> Optional[tuple[Title, list[Title]]]:
    """Find a title whose chapters segment into the durations of other titles.

    Handles both one-chapter-per-episode play-alls (S1: 8 chapters <-> 8
    titles) and multi-chapter-per-episode ones (S7: 25 chapters spanning 5
    titles) by greedily accumulating consecutive chapters until the running
    sum matches an unused title.
    """
    best = None
    for cand in titles:
        if len(cand.chapters) < 2:
            continue
        others = [t for t in titles if t is not cand and t.duration > 60]
        if sum(o.duration for o in others) < cand.duration * 0.5:
            continue
        unused = list(others)
        matched: list[Title] = []
        acc = 0.0
        ok = True
        for ch in cand.chapters:
            acc += ch
            hit = next((o for o in unused
                        if abs(o.duration - acc) <= PLAYALL_CHAPTER_TOL), None)
            if hit:
                matched.append(hit)
                unused.remove(hit)
                acc = 0.0
        # Allow a small unmatched tail (credits/logo chapter).
        if acc > 30.0:
            ok = False
        if ok and len(matched) >= 2 and (best is None or len(matched) > len(best[1])):
            best = (cand, matched)
    return best


def classify_disc(disc: Disc, expected_runtime: float) -> list[Title]:
    """Mark titles and return episode candidates in play order."""
    # 1. duration band kills menus/bumpers but keeps double-length episodes
    lo, hi = expected_runtime * 0.5, expected_runtime * 2.5
    # 2. drop exact duplicates (DRM duplicate-title obfuscation)
    seen: dict[tuple, Title] = {}
    for t in disc.titles:
        key = (round(t.duration, 1), t.cells, tuple(round(c, 1) for c in t.chapters))
        if key in seen:
            t.kind = "junk"
        else:
            seen[key] = t

    play_all = detect_play_all([t for t in disc.titles if t.kind != "junk"])
    ordered: list[Title] = []
    if play_all:
        pa, matched = play_all
        pa.kind = "play-all"
        for order, t in enumerate(matched):
            t.evidence += 1.0
            t.order_key = order
        log.info("%s: play-all is title %d (%d episodes by chapter match)",
                 disc.path.name, pa.id, len(matched))

    in_band = [t for t in disc.titles
               if t.kind == "unknown" and lo <= t.duration <= hi]
    # 3. stream-signature clustering: episodes share audio/subtitle layout
    if in_band:
        sigs: dict[tuple, int] = {}
        for t in in_band:
            sigs[(t.n_audio, t.n_sub)] = sigs.get((t.n_audio, t.n_sub), 0) + 1
        majority = max(sigs, key=lambda s: sigs[s])
        if sigs[majority] >= 2 and any(majority != s for s in sigs):
            for t in in_band:
                t.evidence += 0.4 if (t.n_audio, t.n_sub) == majority else -0.4

    for t in disc.titles:
        if t.kind == "unknown":
            t.kind = "episode-candidate" if lo <= t.duration <= hi else "extra"

    ordered = sorted([t for t in disc.titles if t.kind == "episode-candidate"],
                     key=lambda t: t.order_key)
    return ordered


def assess_ordering(disc: Disc, assignments: list["Assignment"],
                    tol: float = CLEAN_MATCH_TOL) -> tuple[bool, str]:
    """Whether a disc's episode ORDER can be trusted from the metadata path.

    Identity there rests on title/playlist position plus a runtime match,
    aligned monotonically (the aligner can't reorder). Reliable for DVD
    (lsdvd title order) but not Blu-ray, where .mpls order has been scrambled
    vs broadcast (Avatar, MOTU). Signals, in order:
      - DVD, or a play-all whose chapters matched the titles, or multi-part
        "(1)/(2)" names in sequence  -> order corroborated;
      - episodes ~all one length     -> runtime can't order them at all;
      - large alignment deltas        -> the monotonic order conflicts with
        the TMDB runtimes, i.e. the playlist order is likely scrambled;
      - otherwise the runtimes fit the order -> trust it.
    Returns (verifiable, reason)."""
    if len(assignments) <= 1:
        return True, "single title"
    if disc.format == "dvd":
        return True, "DVD title order"
    if any(t.kind == "play-all" for t in disc.titles):
        return True, "play-all corroborates order"
    eps = [e for a in assignments for e in a.episodes]
    if sum(1 for e in eps if re.search(r"\(\d+\)\s*$", e.name)) >= 2:
        return True, "multi-part titles corroborate order"
    rts = sorted(e.runtime for e in eps if e.runtime)
    min_gap = min((rts[i + 1] - rts[i] for i in range(len(rts) - 1)),
                  default=tol + 1)
    if min_gap <= tol:                 # two episodes runtime-indistinguishable
        return False, (f"Blu-ray, episodes not runtime-separable (two within "
                       f"{min_gap:.0f}s) — order can't be verified")
    worst = max((a.delta for a in assignments), default=0.0)
    if worst > 1.5 * tol:              # monotonic order fights the runtimes
        return False, (f"Blu-ray playlist order conflicts with TMDB runtimes "
                       f"(worst delta {worst:.0f}s) — likely scrambled")
    return True, f"runtimes uniquely fit the order (worst delta {worst:.0f}s)"


# ---------------------------------------------------------------------------
# Stage 3c: monotonic DP alignment (Needleman-Wunsch with merge moves)
# ---------------------------------------------------------------------------

GAP_EPISODE = 800.0      # leaving a TMDB episode unmatched: loud failure


def runtime_scale(cands: list[tuple["Disc", Title]],
                  episodes: list[Episode]) -> float:
    """Estimate the systematic disc-duration / TMDB-runtime ratio.

    TMDB frequently records the broadcast slot (e.g. 30 min with ads) while
    the disc carries the actual ~22 min episode.  A per-season median ratio
    catches that; matching then accepts whichever of raw/scaled runtime fits.
    """
    durs = sorted(t.duration for _, t in cands)
    rts = sorted(e.runtime for e in episodes if e.runtime)
    if not durs or not rts:
        return 1.0
    scale = durs[len(durs) // 2] / rts[len(rts) // 2]
    if 0.5 <= scale <= 2.0 and abs(scale - 1.0) > 0.08:
        return scale
    return 1.0


def _delta(dur: float, runtime: Optional[float], scale: float) -> float:
    if runtime is None:
        return 0.0
    return min(abs(dur - runtime), abs(dur - runtime * scale))


def _match_cost(dur: float, runtime: Optional[float], evidence: float,
                scale: float) -> float:
    if runtime is None:
        return -50.0  # order-only match
    delta = _delta(dur, runtime, scale)
    if delta > HARD_MATCH_TOL:
        return float("inf")
    return max(0.0, delta - CLEAN_MATCH_TOL) - 50.0 - 100.0 * evidence


def _gap_title_cost(t: Title) -> float:
    return 150.0 + 300.0 * max(0.0, t.evidence)


def align(cands: list[tuple[Disc, Title]], episodes: list[Episode]
          ) -> tuple[list[Assignment], list[tuple[Disc, Title]], list[Episode]]:
    """Monotonic alignment. Returns (assignments, leftover_titles, missed_eps)."""
    scale = runtime_scale(cands, episodes)
    if scale != 1.0:
        log.info("runtime calibration: TMDB runtimes scaled by %.2f "
                 "(broadcast-slot vs actual runtime)", scale)
    m, n = len(cands), len(episodes)
    INF = float("inf")
    dp = [[INF] * (n + 1) for _ in range(m + 1)]
    bt: list[list[Optional[str]]] = [[None] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0
    for i in range(m + 1):
        for j in range(n + 1):
            cur = dp[i][j]
            if cur == INF:
                continue
            if i < m:  # disc title is an extra
                c = cur + _gap_title_cost(cands[i][1])
                if c < dp[i + 1][j]:
                    dp[i + 1][j], bt[i + 1][j] = c, "gapA"
            if j < n:  # TMDB episode missing from discs
                c = cur + GAP_EPISODE
                if c < dp[i][j + 1]:
                    dp[i][j + 1], bt[i][j + 1] = c, "gapB"
            if i < m and j < n:
                t = cands[i][1]
                c = cur + _match_cost(t.duration, episodes[j].runtime,
                                      t.evidence, scale)
                if c < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1], bt[i + 1][j + 1] = c, "match"
                if j + 1 < n and episodes[j].runtime and episodes[j + 1].runtime:
                    # one disc title covering two episodes (two-parter)
                    rt = episodes[j].runtime + episodes[j + 1].runtime
                    c = cur + _match_cost(t.duration, rt, t.evidence, scale) + 30.0
                    if c < dp[i + 1][j + 2]:
                        dp[i + 1][j + 2], bt[i + 1][j + 2] = c, "merge2"
    # backtrack
    i, j = m, n
    assignments, leftovers, missed = [], [], []
    while i > 0 or j > 0:
        move = bt[i][j]
        if move == "match":
            i, j = i - 1, j - 1
            d, t = cands[i]
            ep = episodes[j]
            delta = _delta(t.duration, ep.runtime, scale)
            conf = ("high" if delta <= CLEAN_MATCH_TOL and t.evidence > 0.5
                    else "medium" if delta <= CLEAN_MATCH_TOL else "low")
            assignments.append(Assignment(d, t, [ep], delta, conf))
        elif move == "merge2":
            i, j = i - 1, j - 2
            d, t = cands[i]
            eps = [episodes[j], episodes[j + 1]]
            delta = _delta(t.duration, sum(e.runtime for e in eps), scale)
            conf = "medium" if delta <= CLEAN_MATCH_TOL else "low"
            assignments.append(Assignment(d, t, eps, delta, conf))
        elif move == "gapA":
            i -= 1
            leftovers.append(cands[i])
        elif move == "gapB":
            j -= 1
            missed.append(episodes[j])
        else:
            raise RuntimeError("alignment backtrack failed")
    assignments.reverse()
    leftovers.reverse()
    missed.reverse()
    return assignments, leftovers, missed


# ---------------------------------------------------------------------------
# Stage 5 (optional): VLM title-card verification
# ---------------------------------------------------------------------------

VLM_PROMPT = (
    "Transcribe ALL text visible in this image VERBATIM, exactly as written, "
    "in its original language. Do not translate, do not paraphrase. If there "
    "is no text reply with NONE. Reply with only the transcription."
)


def ollama_chat(model: str, prompt: str, image_path: Path,
                host: str, retries: int = 4) -> str:
    """One VLM request with retries.

    Retries cover ollama getting OOM-killed mid-request (known memory leak):
    the daemon restarts but in-flight requests die with connection errors,
    and the first retry pays a model reload, hence the generous timeout.
    """
    img_b64 = base64.b64encode(image_path.read_bytes()).decode()
    # Thinking models (qwen3-vl) spend tokens reasoning before answering; a
    # tight num_predict exhausts the budget mid-think and yields empty content.
    body = json.dumps({
        "model": model, "stream": False,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "options": {"num_predict": 2048, "temperature": 0},
    }).encode()
    delays = [5, 15, 30, 60]
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                f"{host}/api/chat", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
            msg = data["message"]
            # If the answer still got cut off mid-think, the transcription
            # often appears inside the thinking text — better than nothing.
            return (msg.get("content") or msg.get("thinking") or "").strip()
        except Exception as e:  # noqa: BLE001 - URLError, timeout, bad JSON
            if attempt >= retries:
                raise
            wait = delays[min(attempt, len(delays) - 1)]
            log.warning("VLM request failed (%s); retry %d/%d in %ds",
                        e, attempt + 1, retries, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


_PARTNUM = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5",
            "vi": "6"}


def canon_parts(norm: str) -> str:
    """Canonicalize part markers so a card's wording matches TMDB's.

    TMDB writes two-parters as "Storm Front (1)" -> normalizes to
    "storm front 1", but the on-screen card may read "PART ONE" / "PART I" /
    "PART 1". Collapse all of those to the digit so the part still aligns
    (and still discriminates part 1 from part 2)."""
    return re.sub(
        r"\bpart\s+(one|two|three|four|five|six|i{1,3}|iv|vi?|\d+)\b",
        lambda m: _PARTNUM.get(m.group(1), m.group(1)), norm)


# Text that means a short word-boundary hit is probably incidental, not a
# title: opening-credits roles, and the VLM's own reasoning preamble.
_INCIDENTAL_MARKERS = (
    "producer", "directed", "director", "written", "writer", "teleplay",
    "story by", "music", "edited", "editor", "starring", "executive",
    "casting", "narrat",
    "got it", "let s", "the image", "i need", "looking at", "transcribe",
)


def fuzzy_best(text: str, episodes: list[Episode]) -> tuple[Optional[Episode], float]:
    """Best episode-name match for transcribed frame text (closed set)."""
    import difflib
    norm = canon_parts(normalize_text(text))
    if not norm:
        return None, 0.0
    incidental = any(m in norm for m in _INCIDENTAL_MARKERS)
    best, best_score = None, 0.0
    for ep in episodes:
        name = canon_parts(normalize_text(ep.name))
        if not name:
            continue
        # Word-boundary substring: a real title card is dominated by the title.
        # A distinctive multi-word/long title appearing verbatim is conclusive
        # (1.0) even amid branding. A short single-word title (Dawn, Jet) is
        # conclusive ONLY when the rest of the frame looks like a title card,
        # not credits/reasoning: "CHAPTER TEN: JET" matches Jet, but "PRODUCER
        # DAWN ..." or a VLM reasoning dump must not. When the frame carries
        # incidental markers we fall back to coverage (title must dominate).
        if re.search(rf"\b{re.escape(name)}\b", norm):
            distinctive = len(name.split()) >= 2 or len(name.replace(" ", "")) >= 10
            if distinctive or not incidental:
                score = 1.0
            else:
                score = len(name) / len(norm)
        else:
            score = difflib.SequenceMatcher(None, name, norm).ratio()
            # also try the best window of the transcription
            words = norm.split()
            target_len = len(name.split())
            for k in range(max(1, len(words) - target_len + 1)):
                window = " ".join(words[k:k + target_len + 1])
                score = max(score, difflib.SequenceMatcher(None, name, window).ratio())
        if score > best_score:
            best, best_score = ep, score
    return best, best_score


def rip_window(disc: Disc, title: Title, start: float, length: float,
               workdir: Path) -> Optional[Path]:
    """Copy a bounded window of one title to a local file. Never re-encodes."""
    out = workdir / "win.avi"
    out.unlink(missing_ok=True)
    if disc.format == "dvd":
        cmd = ["mencoder", f"dvd://{title.id}", "-dvd-device", str(disc.path),
               "-ovc", "copy", "-nosound", "-quiet",
               "-ss", str(int(start)), "-endpos", str(int(length)),
               "-o", str(out)]
    else:
        out = workdir / "win.ts"
        src = f"bluray:{disc.path}"
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-playlist", str(title.id), "-ss", str(int(start)),
               "-i", src, "-t", str(int(length)),
               "-map", "0:v:0", "-c", "copy", str(out)]
    try:
        run(cmd, timeout=int(length) * 4 + 120)
    except subprocess.TimeoutExpired:
        log.warning("rip timed out: %s title %d", disc.path.name, title.id)
        return None
    return out if out.exists() and out.stat().st_size > 0 else None


FRAME_INTERVAL = 1.5   # seconds between sampled frames (cards show ~2-4 s)


def extract_frames(video: Path, workdir: Path,
                   interval: float = FRAME_INTERVAL) -> list[Path]:
    # Title cards are only on screen ~2-4 s; a coarse stride (4+ s) phase-skips
    # right over them. Sample at <=2 s. The windows verify_title rips are
    # bounded, so the extra frames are cheap.
    pattern = workdir / "frame_%04d.jpg"
    for f in workdir.glob("frame_*.jpg"):
        f.unlink()
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
         "-vf", f"fps=1/{interval},scale=960:-2", "-qscale:v", "3",
         str(pattern)], timeout=600)
    return sorted(workdir.glob("frame_*.jpg"))


ANCHOR_RADIUS = 90.0   # seconds either side of a learned card location


def verify_title(disc: Disc, title: Title, episodes: list[Episode],
                 model: str, host: str, workdir: Path,
                 window: float = 150.0, front_window: float = 280.0,
                 fallback_window: float = 720.0, accept: float = 0.8,
                 anchor: Optional[float] = None
                 ) -> tuple[Optional[Episode], float, Optional[float]]:
    """OCR a title's title-card window against the season's episode names.

    Returns (episode, score, card_seconds) — card_seconds is where the
    matching card was found, so a caller can learn the per-disc location.

    Scanning is cheap-first. If `anchor` (an absolute second offset, learned
    from earlier episodes on the same disc) is given, a tight window around it
    is scanned first, frames ordered outward from the anchor so the card is
    usually the first few tried. On a miss it falls back to a broad pass — the
    front (0..front_window, cold-open shows) and the end (window before the
    end, Venture Bros. style) — and finally the NO-CARD widen of the front to
    fallback_window for recap-delayed premieres. So the first episode or two
    on a disc pay full cost; once the location is known the rest are cheap."""
    state = {"ep": None, "score": 0.0, "time": None}

    def scan(start, length, anchor_time=None):
        start = max(0.0, start)
        length = min(length, title.duration - start)
        if length <= 0:
            return None
        video = rip_window(disc, title, start, length, workdir)
        if not video:
            return None
        try:
            frames = extract_frames(video, workdir)
            timed = [(f, start + i * FRAME_INTERVAL) for i, f in enumerate(frames)]
            if anchor_time is not None:   # try frames nearest the anchor first
                timed.sort(key=lambda ft: abs(ft[1] - anchor_time))
            for frame, ts in timed:
                try:
                    text = ollama_chat(model, VLM_PROMPT, frame, host)
                except Exception as e:  # noqa: BLE001
                    log.error("VLM permanently failed on %s: %s", frame.name, e)
                    continue
                ep, score = fuzzy_best(text, episodes)
                if score > state["score"]:
                    state["ep"], state["score"], state["time"] = ep, score, ts
                if score >= accept:
                    log.info("%s title %d: verified %r -> S%02dE%02d (%.2f) @%.0fs",
                             disc.path.name, title.id,
                             text.splitlines()[0][:60] if text else "",
                             ep.season, ep.number, score, ts)
                    return ep, score
        finally:
            video.unlink(missing_ok=True)
        return None

    result = lambda: (state["ep"], state["score"], state["time"])

    if anchor is not None:
        if scan(anchor - ANCHOR_RADIUS, 2 * ANCHOR_RADIUS, anchor_time=anchor):
            return result()
        log.info("%s title %d: anchor @%.0fs missed; broad scan",
                 disc.path.name, title.id, anchor)

    for start, length in [(0.0, front_window),
                          (title.duration - window, window + 60.0)]:
        if scan(start, length):
            return result()
    # NO-CARD fallback: nothing in the primary band — widen the front.
    if title.duration > front_window + 5:
        log.info("%s title %d: no card in primary band; widening front scan",
                 disc.path.name, title.id)
        if scan(front_window, min(fallback_window, title.duration) - front_window):
            return result()
    return result()


def valid_episode_lengths(pool: list[Episode], max_parts: int = 4
                          ) -> Optional[tuple[list[float], float]]:
    """Plausible playlist durations (seconds) for episode candidates.

    Instead of a wide multiplier band around the median (which admits
    gap-length extras like a 35-min featurette on a 24-min show), build a
    discrete set of real lengths:
      - each TMDB per-episode runtime (handles a feature-length pilot/finale
        that TMDB reports correctly, e.g. Broken Bow = 86 min);
      - small integer multiples of the median runtime — robust to TMDB
        reporting a wrong/null runtime for a long episode, since a 2x-length
        episode still lands on 2*median;
      - sums of consecutive episodes, for combined multi-part playlists.
    Returns (sorted_lengths, tolerance), or None when TMDB gives no runtimes
    at all (caller falls back to the wide band)."""
    rts = [e.runtime for e in pool if e.runtime]
    if not rts:
        return None
    median = sorted(rts)[len(rts) // 2]
    lengths = set(rts)
    lengths |= {k * median for k in range(1, max_parts + 1)}
    for n in range(2, max_parts + 1):       # combined multi-parters
        for i in range(len(pool) - n + 1):
            seg = [pool[i + j].runtime for j in range(n)]
            if all(seg):
                lengths.add(sum(seg))
    return sorted(lengths), max(120.0, 0.1 * median)


def episode_candidates(disc: Disc, pool: list[Episode]) -> list[Title]:
    """Episode-length playlists on a disc, in play order — the candidates an
    OCR pass should consider. Uses the TMDB runtime set when available, else a
    wide band around the median."""
    bands = valid_episode_lengths(pool)
    if bands:
        lengths, tol = bands
        ok = lambda dur: min(abs(dur - v) for v in lengths) <= tol
    else:
        rts = [e.runtime for e in pool if e.runtime]
        expected = sorted(rts)[len(rts) // 2] if rts else 1320.0
        ok = lambda dur: expected * 0.6 <= dur <= expected * 2.6
    return sorted([t for t in disc.titles if ok(t.duration)],
                  key=lambda t: t.order_key)


def ocr_identify(discs: list[Disc], seasons: dict[int, list[Episode]],
                 args, workdir: Path
                 ) -> tuple[list[Assignment], list[tuple[Disc, Title]], list[Episode]]:
    """Identify episode-band playlists by OCRing their title cards directly.

    For discs whose metadata ordering can't be trusted — multiple playlists
    per episode, combined two-parters, alternate intro clips (Avatar) — this
    bypasses the DP alignment: every episode-length candidate is OCR-matched
    to the season pool, so each playlist's identity comes from its on-screen
    title, not its position. Slower (a rip + VLM pass per candidate) but
    robust to irregular authoring."""
    raw: list[Assignment] = []
    leftovers: list[tuple[Disc, Title]] = []
    for season, group in group_discs(discs):
        pool = (seasons.get(season) or
                [e for n in sorted(seasons) for e in seasons[n]])
        for d in group:
            cands = episode_candidates(d, pool)
            log.info("%s: OCR-identifying %d candidate playlist(s)",
                     d.path.name, len(cands))
            # learn where this disc puts its card; anchor later episodes there
            card_ends: list[bool] = []
            card_offsets: list[float] = []
            for t in cands:
                anchor = None
                if card_offsets:
                    from_end = sum(card_ends) * 2 >= len(card_ends)  # majority
                    off = sorted(card_offsets)[len(card_offsets) // 2]  # median
                    anchor = (t.duration - off) if from_end else off
                ep, score, card_time = verify_title(
                    d, t, pool, args.vlm_model, args.ollama_host, workdir,
                    accept=args.ocr_accept, anchor=anchor)
                if card_time is not None and ep and score >= args.ocr_accept:
                    from_end = card_time > t.duration / 2
                    card_ends.append(from_end)
                    card_offsets.append(t.duration - card_time if from_end
                                        else card_time)
                if not ep or score < args.ocr_accept:
                    leftovers.append((d, t))
                    log.info("%s pl %d: unmatched (best %.2f)",
                             d.path.name, t.id, score)
                    continue
                eps = [ep]
                # combined two-parter playlist (~2x runtime): claim next too
                if ep.runtime and t.duration >= ep.runtime * 1.6:
                    nxt = next((e for e in pool if e.number == ep.number + 1), None)
                    if nxt:
                        eps.append(nxt)
                delta = abs(t.duration - sum(e.runtime or 0 for e in eps))
                raw.append(Assignment(d, t, eps, delta, "high", ep.name))

    # Resolve collisions: each episode claimed once. Prefer single-episode
    # assignments over doubles, then the longest (most complete) playlist. A
    # combined double is kept if it carries at least one *unclaimed* episode
    # (e.g. a finale where E19 exists only inside the E19+E20 double while E20
    # also has a single) — only dropped when fully redundant.
    final: list[Assignment] = []
    claimed: dict[tuple, Assignment] = {}
    for a in sorted(raw, key=lambda a: (len(a.episodes), -a.title.duration)):
        keys = [(e.season, e.number) for e in a.episodes]
        unclaimed = [k for k in keys if k not in claimed]
        if not unclaimed:
            leftovers.append((a.disc, a.title))
            log.warning("%s pl %d (%r) fully duplicates already-identified "
                        "episode(s); treating as extra", a.disc.path.name,
                        a.title.id, a.episodes[0].name)
            continue
        for k in unclaimed:
            claimed[k] = a
        final.append(a)

    hinted = {d.season_hint for d in discs if d.season_hint}
    missed = [e for n in sorted(seasons) for e in seasons[n]
              if (e.season, e.number) not in claimed
              and (not hinted or e.season in hinted)]
    return recover_by_elimination(final, leftovers, missed)


def recover_by_elimination(final: list[Assignment],
                           leftovers: list[tuple[Disc, Title]],
                           missed: list[Episode]
                           ) -> tuple[list[Assignment], list[tuple[Disc, Title]], list[Episode]]:
    """Pin an unmatched candidate to the one episode it must be.

    Some episodes show no on-screen title (a premiere whose card is the
    series-logo sequence, e.g. MOTU E01/E06), so OCR leaves them as leftovers.
    But when a disc has exactly one unmatched candidate and exactly one of the
    still-missing episodes is adjacent to that disc's matched run, the leftover
    must be that episode. Constraint-propagate so a disc with only one option
    (D2: only E06 borders E07-E10) resolves first and frees the ambiguous one
    (D1: E01 or E06 -> E01 once E06 is taken)."""
    missing = {(e.season, e.number): e for e in missed}
    left_by_disc: dict[Path, list[tuple[Disc, Title]]] = {}
    for d, t in leftovers:
        left_by_disc.setdefault(d.path, []).append((d, t))
    matched_by_disc: dict[Path, set] = {}
    for a in final:
        s = matched_by_disc.setdefault(a.disc.path, set())
        s.update((e.season, e.number) for e in a.episodes)

    recovered: list[tuple[Disc, Title, Episode]] = []
    changed = True
    while changed and missing:
        changed = False
        for path, lefts in left_by_disc.items():
            if len(lefts) != 1:           # only unambiguous single-leftover discs
                continue
            matched = matched_by_disc.get(path, set())
            opts = [k for k in missing
                    if (k[0], k[1] - 1) in matched or (k[0], k[1] + 1) in matched]
            if len(opts) == 1:
                d, t = lefts[0]
                ep = missing.pop(opts[0])
                matched.add(opts[0])
                left_by_disc[path] = []
                recovered.append((d, t, ep))
                changed = True

    consumed = set()
    for d, t, ep in recovered:
        delta = abs(t.duration - (ep.runtime or t.duration))
        final.append(Assignment(d, t, [ep], delta, "medium", method="elimination"))
        consumed.add((str(d.path), t.id))
        log.info("%s pl %d: no title card — recovered S%02dE%02d %r by "
                 "elimination", d.path.name, t.id, ep.season, ep.number, ep.name)
    leftovers = [(d, t) for d, t in leftovers
                 if (str(d.path), t.id) not in consumed]
    return final, leftovers, list(missing.values())


def vlm_available(model: str, host: str) -> bool:
    """Is the Ollama VLM reachable and the model pulled?"""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=10) as r:
            names = {m.get("name", "") for m in json.loads(r.read()).get("models", [])}
    except Exception:  # noqa: BLE001 — daemon down / network / bad JSON
        return False
    base = model.split(":")[0]
    return model in names or any(n.split(":")[0] == base for n in names)


def probe_card_presence(discs: list[Disc], seasons: dict[int, list[Episode]],
                        args, workdir: Path, n: int = 2) -> bool:
    """Cheap gate before a full OCR escalation: do episodes caption their
    titles on screen at all? OCR up to n candidate playlists (skipping each
    disc's first/last to dodge premiere/finale intro quirks) and return True
    as soon as one matches an episode title."""
    probed = 0
    for season, group in group_discs(discs):
        pool = (seasons.get(season) or
                [e for s in sorted(seasons) for e in seasons[s]])
        for d in group:
            cands = episode_candidates(d, pool)
            mids = cands[1:-1] or cands          # avoid first/last
            for t in mids:
                ep, score, _ = verify_title(d, t, pool, args.vlm_model,
                                            args.ollama_host, workdir,
                                            accept=args.ocr_accept)
                probed += 1
                log.info("card probe: %s pl %d -> %.2f", d.path.name, t.id, score)
                if ep and score >= args.ocr_accept:
                    return True
                if probed >= n:
                    return False
    return False


# ---------------------------------------------------------------------------
# Stage 4: orchestration + reporting
# ---------------------------------------------------------------------------


def sanitize_filename(s: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "", s).strip()


def suggested_filename(show: str, eps: list[Episode]) -> str:
    nums = "".join(f"E{e.number:02d}" for e in eps)
    names = " & ".join(e.name for e in eps)
    return sanitize_filename(f"{show} - S{eps[0].season:02d}{nums} - {names}.mkv")


def group_discs(discs: list[Disc]) -> list[tuple[Optional[int], list[Disc]]]:
    """Group discs by season hint; fall back to one global group."""
    if all(d.season_hint is not None for d in discs):
        groups: dict[int, list[Disc]] = {}
        for d in discs:
            groups.setdefault(d.season_hint, []).append(d)
        return [(s, sorted(ds, key=lambda d: (d.disc_hint or 0, d.path.name)))
                for s, ds in sorted(groups.items())]
    return [(None, sorted(discs, key=lambda d: d.path.name))]


def emit_rip_commands(records: list[dict], preset: str) -> None:
    """Print a HandBrakeCLI line per episode record."""
    for r in records:
        if r.get("kind") != "episode":
            continue
        print(f'HandBrakeCLI -i "{r["image"]}" -t {r["title"]} '
              f'--preset "{preset}" -o "{r["suggested_filename"]}"')


def merge_records(existing: list[dict], new: list[dict],
                  images: list[Path]) -> list[dict]:
    """Merge a fresh run's records into an existing manifest.

    Records for the discs processed this run (matched by `image`) are
    replaced wholesale by the new ones; records for every other disc are
    kept. This lets a re-run of a few discs refine the manifest in place."""
    touched = {str(p) for p in images}
    kept = [r for r in existing if r.get("image") not in touched]
    merged = kept + new
    merged.sort(key=lambda r: (r.get("season", 99), r.get("episodes", [999]),
                               r.get("image", ""), r.get("kind", "")))
    return merged


def resolve_cross_disc(records: list[dict]) -> list[dict]:
    """Demote cross-disc duplicate episode claims to extras.

    The same episode claimed by records on different discs (a featurette that
    names an episode, a recap, a redundant playlist) can't be caught by length
    or per-disc collision resolution. Keep the claim on the disc with the most
    of that episode's neighbours present — the real home sits in a contiguous
    run; an impostor is alone amid a different stretch — and demote the others.
    Ties (no clear contiguous winner) are left for the validation warning."""
    eps = [r for r in records if r.get("kind") == "episode"]
    disc_eps: dict[str, set] = {}
    for r in eps:
        for n in r.get("episodes", []):
            disc_eps.setdefault(r["image"], set()).add((r["season"], n))

    claims: dict[tuple, list] = {}
    for r in eps:
        for n in r.get("episodes", []):
            claims.setdefault((r["season"], n), []).append(r)

    def neighbours(r, s, n):
        return sum((s, n + dd) in disc_eps[r["image"]] for dd in (-2, -1, 1, 2))

    demote: set = set()
    for (s, n), rs in claims.items():
        if len({r["image"] for r in rs}) <= 1:
            continue                       # same disc (single+double) is fine
        ranked = sorted(rs, key=lambda r: (neighbours(r, s, n), -len(r["episodes"])),
                        reverse=True)
        if neighbours(ranked[0], s, n) == neighbours(ranked[1], s, n):
            continue                       # tie -> leave for the warning
        for r in ranked[1:]:
            if r["image"] != ranked[0]["image"]:
                demote.add(id(r))

    out = []
    for r in records:
        if id(r) in demote:
            log.warning("demoting cross-disc duplicate to extra: %s title %s "
                        "(S%02dE%02d %r) — kept the disc with the contiguous run",
                        Path(r["image"]).name, r["title"], r["season"],
                        r["episodes"][0], r["episode_name"])
            r = {"image": r["image"], "title": r["title"], "kind": "extra",
                 "title_seconds": r.get("title_seconds"),
                 "note": f"cross-disc duplicate of S{r['season']:02d}"
                         f"E{r['episodes'][0]:02d} {r['episode_name']!r}"}
        out.append(r)
    return out


def write_manifest(out: Path, records: list[dict], merge: bool,
                   images: list[Path]) -> int:
    """Write the manifest under an exclusive lock, atomically.

    The lock serializes the read-merge-write so two runs targeting the same
    --out can't interleave and corrupt it (a concurrent --merge race once
    silently dropped already-identified episodes). The temp+rename makes the
    final file appear atomically. Returns the record count written."""
    lock = out.with_suffix(out.suffix + ".lock")
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        if merge and out.exists():
            records = merge_records(json.loads(out.read_text()), records, images)
        records = resolve_cross_disc(records)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(records, indent=2))
        os.replace(tmp, out)
    return len(records)


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
    ap.add_argument("--verify", action="store_true",
                    help="OCR title cards of low-confidence matches via Ollama")
    ap.add_argument("--verify-all", action="store_true",
                    help="OCR every matched title, not just low-confidence ones")
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
    ap.add_argument("--ollama-host",
                    default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    # Pure transform: emit rip commands from a saved manifest, no scanning.
    if args.from_manifest:
        records = json.loads(args.from_manifest.read_text())
        emit_rip_commands(records, args.handbrake_preset)
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
                season_pool = seasons.get(a.episodes[0].season, [])
                ep, score, _ = verify_title(a.disc, a.title, season_pool,
                                            args.vlm_model, args.ollama_host,
                                            Path(tmp))
                if ep and score >= 0.8:
                    a.verified_name = ep.name
                    if [ep.number] != [e.number for e in a.episodes]:
                        log.warning(
                            "%s title %d: VLM says S%02dE%02d %r, alignment said %s — using VLM",
                            a.disc.path.name, a.title.id, ep.season, ep.number,
                            ep.name, [e.number for e in a.episodes])
                        a.episodes = [ep]
                    a.confidence = "high"
                else:
                    log.warning(
                        "%s title %d: no title card found (best fuzzy score "
                        "%.2f%s) — keeping alignment result %s",
                        a.disc.path.name, a.title.id, score,
                        f" vs {ep.name!r}" if ep else "",
                        [e.number for e in a.episodes])

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
            "suggested_filename": suggested_filename(show, a.episodes),
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
        emit_rip_commands(records, args.handbrake_preset)

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
    return 1 if (problems or crossdisc or unverifiable) else 0


if __name__ == "__main__":
    sys.exit(main())
