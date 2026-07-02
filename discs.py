"""Disc scanning (DVD/Blu-ray), TMDB metadata, and the shared data model."""
from __future__ import annotations

import ast
import json
import logging
import re
import shutil
import struct
import subprocess
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
    video_format: Optional[str] = None   # e.g. "1080p", "480i" (resolution+scan)
    audio_format: Optional[str] = None   # e.g. "DTS-HDMA", "AC3" (best-effort)
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
    overview: str = ""           # TMDB synopsis (for the dialogue/synopsis judge)
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


# MPLS STN stream-attribute codes.
_VIDEO_FORMAT = {1: "480i", 2: "576i", 3: "480p", 4: "1080i", 5: "720p",
                 6: "1080p", 7: "576p", 8: "2160p"}
_AUDIO_CODING = {0x80: "LPCM", 0x81: "AC3", 0x82: "DTS", 0x83: "TrueHD",
                 0x84: "EAC3", 0x85: "DTS-HD", 0x86: "DTS-HDMA"}


def _stn_formats(buf: bytes, item_pos: int, item_len: int):
    """Read the first PlayItem's primary video (and best-effort audio) format
    from its STN stream table. Pure metadata — no payload. Returns
    (video_format, audio_format) as strings, or (None, None) if the layout
    doesn't validate (parsing the STN is fragile; fail soft rather than guess).

    The non-multi-angle PlayItem header is 32 bytes before the STN table:
    clip(5)+codec(4)+flags(2)+stc(1)+in(4)+out(4)+UO(8)+ra(1)+still(1)+
    still_time(2). The STN table then opens with its own length, which must
    end exactly at the PlayItem boundary — used here as a sanity anchor."""
    data = item_pos + 2
    try:
        if (struct.unpack_from(">H", buf, data + 9)[0] >> 4) & 1:
            return None, None                 # multi-angle: header differs
        stn = data + 32
        stn_len = struct.unpack_from(">H", buf, stn)[0]
        if stn + 2 + stn_len != data + item_len:
            return None, None                 # end-anchor failed
        n_video, n_audio = buf[stn + 4], buf[stn + 5]
        p = stn + 16                          # past counts(7) + reserved(5)
        vfmt = afmt = None
        for i in range(n_video):
            attr = p + 1 + buf[p]
            if i == 0:
                vfmt = _VIDEO_FORMAT.get(buf[attr + 2] >> 4)
            p = attr + 1 + buf[attr]
        for i in range(n_audio):
            attr = p + 1 + buf[p]
            if i == 0:
                afmt = _AUDIO_CODING.get(buf[attr + 1], f"a{buf[attr + 1]:02x}")
            p = attr + 1 + buf[attr]
        return vfmt, afmt
    except (struct.error, IndexError):
        return None, None


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
    video_format = audio_format = None
    for idx in range(n_items):
        item_len = struct.unpack_from(">H", buf, pos)[0]
        clip = buf[pos + 2:pos + 7].decode("ascii", "replace")
        in_t, out_t = struct.unpack_from(">II", buf, pos + 14)
        clips.append(clip)
        in_times.append(in_t)
        durations.append((out_t - in_t) / 45000.0)
        if idx == 0:        # the playlist's format is its first item's
            video_format, audio_format = _stn_formats(buf, pos, item_len)
        pos += 2 + item_len
    # Stream counts live in the STN table at a variable offset; parsing it is
    # fragile, so the MPLS parse leaves counts at 0 — scan_bluray fills them in
    # from the (already-run) HandBrake scan, which demuxes them reliably.
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
            "n_audio": n_audio, "n_sub": n_sub, "n_items": n_items,
            "video_format": video_format, "audio_format": audio_format}


def dedup_identical_clips(titles: list[Title],
                          lossless: "frozenset[int]" = frozenset()) -> list[Title]:
    """Collapse playlists that reference the identical clip sequence — Blu-rays
    carry duplicate/obfuscation playlists, and often two authorings of one
    episode (a lossless/multi-language master and a stripped stereo copy).

    Keep the RICHEST of each group by, in order: most audio+subtitle streams;
    then lossless audio present (`lossless` = the set of playlist ids HandBrake
    reported a lossless track for) so a lossless master beats a same-count lossy
    twin; then lowest playlist id (stable, and matches the old first-seen-by-
    filename behaviour when no stream counts are available)."""
    def rank(t: Title) -> tuple:
        return (t.n_audio + t.n_sub, t.id in lossless, -t.id)
    best: dict[tuple, Title] = {}
    for t in titles:
        cur = best.get(t.clips)
        if cur is None or rank(t) > rank(cur):
            best[t.clips] = t
    return sorted(best.values(), key=lambda t: t.id)


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
    # Build every playlist first, then dedup — so when two playlists reference
    # identical clips we keep the BETTER one. Blu-rays routinely author an
    # episode twice: a full playlist (lossless audio + subtitles) and a lean one
    # (stereo AC3, no subs). Dropping by filename order silently kept the
    # inferior master (Avatar B1D3: pl 254 = 1A/0S beat pl 601 = DTS-HD MA 4A/1S
    # over the same clips). The HandBrake scan's stream counts are the
    # discriminator, so run it before dedup rather than after.
    hb = handbrake_scan(path)
    all_titles = []
    for num, buf in sorted(playlists.items()):
        info = parse_mpls(buf)
        if not info:
            continue
        v = hb.get(num)   # HandBrake demuxed counts (Blu-ray) override the 0s
        all_titles.append(Title(
            id=num, duration=info["duration"], chapters=info["chapters"],
            n_audio=v["n_audio"] if v else info["n_audio"],
            n_sub=v["n_sub"] if v else info["n_sub"],
            cells=info["n_items"], clips=info["clips"], order_key=num,
            video_format=info["video_format"], audio_format=info["audio_format"],
        ))
    lossless = frozenset(pl for pl, v in hb.items() if v.get("lossless"))
    titles = dedup_identical_clips(all_titles, lossless)
    titles = dedup_subset_playlists(titles)
    order_by_playall(titles)   # exact ordering when a play-all is present
    disc = Disc(path=path, format="bluray", label=label, titles=titles)
    disc.hb_map = {pl: v["index"] for pl, v in hb.items()}
    return disc


# Lossless audio codec markers as they appear in HandBrake's audio Description
# (e.g. "English (DTS-HD MA, 2.0 ch)"). "DTS-HD MA" only — plain "DTS" and
# "DTS-HD HRA" are lossy; matched lowercase.
_LOSSLESS_AUDIO = ("truehd", "dts-hd ma", "flac", "lpcm", "pcm", "alac")


def _has_lossless(audio_list: list) -> bool:
    for a in audio_list:
        desc = f"{a.get('Description', '')} {a.get('CodecName', '')}".lower()
        if any(m in desc for m in _LOSSLESS_AUDIO):
            return True
    return False


def _parse_hb_scan(stdout: str) -> dict[int, dict]:
    """Parse HandBrakeCLI --json scan output into
    {playlist_id: {"index", "n_audio", "n_sub", "lossless"}}.

    HandBrake reports each Blu-ray title's `.mpls` id (`Playlist`), its own
    title index (`Index`, the `-t N` to rip), and the demuxed `AudioList` /
    `SubtitleList` — an accurate stream count without our parsing the fragile
    MPLS STN table ourselves, plus whether any audio track is lossless. One scan
    yields the rip index, the stream-richness signal, and the master/duplicate
    tiebreak."""
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
    scan = {}
    for t in data.get("TitleList", []):
        pl, idx = t.get("Playlist"), t.get("Index")
        if pl is None or idx is None:
            continue
        try:
            scan[int(pl)] = {"index": int(idx),
                             "n_audio": len(t.get("AudioList", [])),
                             "n_sub": len(t.get("SubtitleList", [])),
                             "lossless": _has_lossless(t.get("AudioList", []))}
        except (ValueError, TypeError):
            continue
    return scan


def _parse_hb_titles(stdout: str) -> dict[int, int]:
    """{playlist_id: handbrake_title_index} — the index-only view for ripping."""
    return {pl: v["index"] for pl, v in _parse_hb_scan(stdout).items()}


def handbrake_scan(path: Path) -> dict[int, dict]:
    """Scan a Blu-ray once with HandBrake, returning per-playlist
    {"index", "n_audio", "n_sub"} (see `_parse_hb_scan`). One subprocess yields
    both the rip title index and the stream-richness signal.

    HandBrake enumerates relevant playlists; that numbering differs from raw
    .mpls ids and from a player's title-object list (e.g. VLC). Best-effort:
    returns {} (callers warn and fall back) if HandBrakeCLI is missing or the
    scan fails.
    """
    if not shutil.which("HandBrakeCLI"):
        log.warning("HandBrakeCLI not on PATH; Blu-ray output will use raw "
                    ".mpls ids, which do NOT match HandBrake's -t numbers, and "
                    "titles carry no audio/subtitle stream counts")
        return {}
    try:
        proc = run(["HandBrakeCLI", "-i", str(path), "-t", "0", "--scan",
                    "--json"], timeout=300)
    except subprocess.TimeoutExpired:
        log.warning("HandBrake scan timed out on %s", path.name)
        return {}
    scan = _parse_hb_scan(proc.stdout)
    if not scan:
        log.warning("HandBrake scan yielded no titles for %s", path.name)
    return scan


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
                overview=e.get("overview", ""),
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
                overview=e.get("overview", ""),
                aired_season=e["season_number"],
                aired_number=e["episode_number"],
            ))
        pools[g["order"]] = eps
    specials = pools.pop(0, [])
    return pools, specials




def group_discs(discs: list[Disc]) -> list[tuple[Optional[int], list[Disc]]]:
    """Group discs by season hint; fall back to one global group."""
    if all(d.season_hint is not None for d in discs):
        groups: dict[int, list[Disc]] = {}
        for d in discs:
            groups.setdefault(d.season_hint, []).append(d)
        return [(s, sorted(ds, key=lambda d: (d.disc_hint or 0, d.path.name)))
                for s, ds in sorted(groups.items())]
    return [(None, sorted(discs, key=lambda d: d.path.name))]

