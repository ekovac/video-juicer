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
            cells=info["n_items"], order_key=num,
        ))
    return Disc(path=path, format="bluray", label=label, titles=titles)


SEASON_DISC_RE = [
    re.compile(r"[Ss](?:eason[ ._]?)?(\d{1,2})[ ._-]?[Dd](?:isc)?[ ._]?(\d{1,2})"),
    re.compile(r"VOL(?:UME)?[ ._]?(\d{1,2}).*?DIS[CK][ ._]?(\d{1,2})", re.I),
    re.compile(r"[Ss](?:eason)?[ ._]?(\d{1,2})"),
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


def fuzzy_best(text: str, episodes: list[Episode]) -> tuple[Optional[Episode], float]:
    """Best episode-name match for transcribed frame text (closed set)."""
    import difflib
    norm = normalize_text(text)
    if not norm:
        return None, 0.0
    best, best_score = None, 0.0
    for ep in episodes:
        name = normalize_text(ep.name)
        if not name:
            continue
        # substring hit beats ratio: frames carry extra text around the title
        if name in norm:
            return ep, 1.0
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


def extract_frames(video: Path, workdir: Path, interval: float = 4.0) -> list[Path]:
    pattern = workdir / "frame_%03d.jpg"
    for f in workdir.glob("frame_*.jpg"):
        f.unlink()
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
         "-vf", f"fps=1/{interval},scale=960:-2", "-qscale:v", "3",
         str(pattern)], timeout=600)
    return sorted(workdir.glob("frame_*.jpg"))


def verify_title(disc: Disc, title: Title, episodes: list[Episode],
                 model: str, host: str, workdir: Path,
                 window: float = 150.0, accept: float = 0.8
                 ) -> tuple[Optional[Episode], float]:
    """OCR a title's both-end windows against the season's episode names."""
    # Title cards can be at the start or the end (Venture Bros: end) —
    # overshoot the end window so post-credits cards aren't cut off.
    windows = [(max(0.0, title.duration - window), window + 60.0), (0.0, window)]
    best_ep, best_score = None, 0.0
    for start, length in windows:
        video = rip_window(disc, title, start, length, workdir)
        if not video:
            continue
        for frame in extract_frames(video, workdir):
            try:
                text = ollama_chat(model, VLM_PROMPT, frame, host)
            except Exception as e:  # noqa: BLE001
                log.error("VLM permanently failed on %s: %s", frame.name, e)
                continue
            ep, score = fuzzy_best(text, episodes)
            if score > best_score:
                best_ep, best_score = ep, score
            if score >= accept:
                log.info("%s title %d: verified %r -> S%02dE%02d (%.2f)",
                         disc.path.name, title.id, text.splitlines()[0][:60]
                         if text else "", ep.season, ep.number, score)
                return ep, score
        video.unlink(missing_ok=True)
    return best_ep, best_score


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


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="+", type=Path,
                    help="disc images (.iso) or backup directories")
    ap.add_argument("--tv-id", type=int, required=True, help="TMDB series id")
    ap.add_argument("--tmdb-api-key", default=os.environ.get("TMDB_API_KEY"))
    ap.add_argument("--out", type=Path, default=Path("manifest.json"))
    ap.add_argument("--cache-dir", type=Path, default=Path(".tmdb_cache"))
    ap.add_argument("--episode-order", default="aired", metavar="ORDER",
                    help="episode ordering to match against: 'aired' "
                         "(default), an alias (dvd, digital, absolute, "
                         "production, story, tv), or an explicit TMDB "
                         "episode-group id")
    ap.add_argument("--emit-rip-commands", action="store_true")
    ap.add_argument("--handbrake-preset", default="Fast 1080p30",
                    metavar="PRESET",
                    help='HandBrake preset for --emit-rip-commands (default: "Fast 1080p30")')
    ap.add_argument("--verify", action="store_true",
                    help="OCR title cards of low-confidence matches via Ollama")
    ap.add_argument("--verify-all", action="store_true",
                    help="OCR every matched title, not just low-confidence ones")
    ap.add_argument("--vlm-model", default="qwen3-vl:2B")
    ap.add_argument("--ollama-host",
                    default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
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

    # leftovers may be TMDB specials: best-effort runtime match against S0
    special_notes: dict[int, str] = {}
    for idx, (d, t) in enumerate(all_leftovers):
        hits = [e for e in specials
                if e.runtime and abs(e.runtime - t.duration) <= CLEAN_MATCH_TOL]
        if len(hits) == 1:
            special_notes[idx] = f"possible special S00E{hits[0].number:02d} {hits[0].name!r}"

    # optional VLM verification of weak matches
    if args.verify or args.verify_all:
        targets = [a for a in all_assignments
                   if args.verify_all or a.confidence == "low"]
        log.info("verifying %d title(s) via %s", len(targets), args.vlm_model)
        with tempfile.TemporaryDirectory(prefix="identify-eps-") as tmp:
            for a in targets:
                season_pool = seasons.get(a.episodes[0].season, [])
                ep, score = verify_title(a.disc, a.title, season_pool,
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
            "image": str(a.disc.path), "title": a.title.id, "kind": "episode",
            "season": e0.season,
            "episodes": [e.number for e in a.episodes],
            "episode_name": " & ".join(e.name for e in a.episodes),
            "title_seconds": round(a.title.duration, 1),
            "tmdb_seconds": sum(e.runtime or 0 for e in a.episodes),
            "delta_seconds": round(a.delta, 1),
            "confidence": a.confidence,
            "verified_by_titlecard": a.verified_name is not None,
            "suggested_filename": suggested_filename(show, a.episodes),
        })
        if e0.aired_season is not None:
            records[-1]["aired"] = [
                f"S{e.aired_season:02d}E{e.aired_number:02d}" for e in a.episodes]
    for idx, (d, t) in enumerate(all_leftovers):
        records.append({
            "image": str(d.path), "title": t.id, "kind": "extra",
            "title_seconds": round(t.duration, 1),
            "note": special_notes.get(idx, ""),
        })
    for d in discs:
        for t in d.titles:
            if t.kind == "play-all":
                records.append({"image": str(d.path), "title": t.id,
                                "kind": "play_all",
                                "title_seconds": round(t.duration, 1)})
    args.out.write_text(json.dumps(records, indent=2))
    log.info("wrote %s (%d records)", args.out, len(records))

    # human-readable table
    print(f"\n{show} — {len(all_assignments)} titles matched")
    for a in sorted(all_assignments,
                    key=lambda a: (a.episodes[0].season, a.episodes[0].number)):
        e0 = a.episodes[0]
        nums = "".join(f"E{e.number:02d}" for e in a.episodes)
        flag = {"high": " ", "medium": " ", "low": "?"}[a.confidence]
        ver = " [verified]" if a.verified_name else ""
        print(f"  S{e0.season:02d}{nums} {flag} {a.disc.path.name} title {a.title.id:>2} "
              f"Δ{a.delta:5.1f}s  {' & '.join(e.name for e in a.episodes)}{ver}")
    if all_missed:
        print("\nMISSING EPISODES (not found on any disc):")
        for e in all_missed:
            print(f"  S{e.season:02d}E{e.number:02d} {e.name}")
    if all_leftovers:
        print(f"\nExtras / unmatched titles: {len(all_leftovers)}")
        for idx, (d, t) in enumerate(all_leftovers):
            note = f"  ({special_notes[idx]})" if idx in special_notes else ""
            print(f"  {d.path.name} title {t.id:>2} {t.duration/60:6.1f} min{note}")

    if args.emit_rip_commands:
        print("\n# rip commands")
        for r in records:
            if r["kind"] != "episode":
                continue
            print(f'HandBrakeCLI -i "{r["image"]}" -t {r["title"]} '
                  f'--preset "{args.handbrake_preset}" -o "{r["suggested_filename"]}"')

    # validation summary
    by_season: dict[int, int] = {}
    for a in all_assignments:
        for e in a.episodes:
            by_season[e.season] = by_season.get(e.season, 0) + 1
    problems = [f"S{s:02d}: matched {by_season.get(s, 0)}/{len(eps)}"
                for s, eps in sorted(seasons.items())
                if by_season.get(s, 0) != len(eps)
                and any(d.season_hint == s for d in discs)]
    if problems or all_missed:
        print("\nWARNING: incomplete coverage: " + "; ".join(problems))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
