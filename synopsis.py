"""Fourth-tier identity source: match spoken dialogue against TMDB synopses.

When a disc has no readable title card (The Magicians; the Broken Saints finale
fragments) the OCR path yields nothing. Dialogue can still identify an episode —
not by echoing the synopsis verbatim (it never does) but by *plausibility*: a
mid-size LLM can judge whether a transcript could belong to the episode a
synopsis describes ("synopsis: Aang visits a fortuneteller" vs dialogue arguing
about whether the future is fixed — consistent). Embedding cosine can't bridge
that summary-vs-dialogue gap; an entailment judge can.

The same transcript also flags a director's-commentary title: commentary
narration ("when we were animating this scene") matches no synopsis as in-world
dialogue, so the judge abstains on it.

Cheap *in context*: this only runs once we've already fallen back to the
expensive OCR path (mencoder rips + a thinking-VLM frame sweep), so a CPU whisper
pass plus one text-LLM call is marginal next to what we've already spent.

Gated hard, same discipline as the OCR side: forced choice against the whole
season pool WITH an explicit abstention, a top-1/top-2 margin, and a second
"cite the specific shared event" confirmation pass. A missing identity is
recoverable by position/elimination; a confident-wrong one corrupts the mapping,
so when in doubt we return nothing.
"""
from __future__ import annotations

import html
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Optional

from discs import Disc, Episode, Title, log, run

# whisper model: base.en is ~74 MB, int8 on CPU runs faster-than-realtime and
# reads TV dialogue cleanly. Bump to small.en for noisy mixes. Cached once.
WHISPER_MODEL = "base.en"
_whisper = None

# Sample a few ~40 s windows away from the cold open and end credits, where
# plot-distinctive dialogue lives (a title's first minute is often a recap /
# "previously on", its last is credits music). More/longer windows see more of
# the synopsis's plot beats at more whisper cost — tunable per run via
# `vj run synopsis --synopsis-windows N --synopsis-length SEC`.
def spread_fractions(n: int, lo: float = 0.2, hi: float = 0.8) -> tuple:
    """`n` sampling positions evenly spread across a title's interior BAND
    [lo, hi] (default 20%–80%), so no window falls in the cold-open "previously
    on" recap or the end credits regardless of how many are requested:
    n=1 -> (0.5,); n=3 -> (0.2, 0.5, 0.8); n=6 -> (0.2, 0.32, …, 0.8)."""
    n = max(1, n)
    if n == 1:
        return ((lo + hi) / 2,)
    return tuple(lo + (hi - lo) * i / (n - 1) for i in range(n))


SAMPLE_WINDOWS = 3
SAMPLE_FRACTIONS = spread_fractions(SAMPLE_WINDOWS)
SAMPLE_LENGTH = 40.0

# A NON-thinking instruct model: a thinking judge (e.g. gemma4) spends its token
# budget reasoning and can return empty content on long synopsis prompts, which
# reads as an abstention. qwen2.5:14b answers directly with the JSON verdict.
JUDGE_MODEL = "qwen2.5:14b-instruct"


def _load_whisper(model_size: str = WHISPER_MODEL):
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        log.info("loading whisper %s (cpu/int8)", model_size)
        _whisper = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _whisper


def rip_audio(disc: Disc, title: Title, start: float, length: float,
              out: Path) -> Optional[Path]:
    """Rip one bounded audio-only window to 16 kHz mono wav (whisper's format).

    Blu-ray reads via ffmpeg's `bluray:` protocol (same as rip_window's video).
    DVD has no ffmpeg protocol (libdvdread), so it goes through mplayer's DVD
    reader — the same `dvd://<id> -dvd-device <path>` addressing rip_window uses
    for video — decoding the primary audio track straight to a 16 kHz mono wav
    (`-vc null -vo null` skips the video work). `-endpos` is ABSOLUTE in mplayer
    (unlike mencoder, where it's a length), so it's start+length here."""
    out.unlink(missing_ok=True)
    if disc.format == "dvd":
        cmd = ["mplayer", f"dvd://{title.id}", "-dvd-device", str(disc.path),
               "-ss", str(int(start)), "-endpos", str(int(start + length)),
               "-vc", "null", "-vo", "null",
               "-ao", f"pcm:fast:file={out}",
               "-format", "s16le", "-channels", "1", "-srate", "16000",
               "-noconfig", "all", "-really-quiet"]
    else:
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-playlist", str(title.id), "-ss", str(int(start)),
               "-i", f"bluray:{disc.path}", "-t", str(int(length)),
               "-map", "0:a:0", "-ac", "1", "-ar", "16000", str(out)]
    try:
        run(cmd, timeout=int(length) * 4 + 120)
    except subprocess.TimeoutExpired:
        log.warning("synopsis: audio rip timed out %s title %d",
                    disc.path.name, title.id)
        return None
    return out if out.exists() and out.stat().st_size > 0 else None


def transcribe(wav: Path, model_size: str = WHISPER_MODEL) -> str:
    """Transcribe a wav to plain text via faster-whisper (CPU)."""
    model = _load_whisper(model_size)
    segments, _ = model.transcribe(str(wav), language="en", beam_size=1)
    return " ".join(s.text.strip() for s in segments).strip()


_SRT_TAG = re.compile(r"<[^>]+>|\{[^}]*\}")   # <font…>, ASS {\an7}, etc.


def srt_to_text(srt: str) -> str:
    """Flatten an SRT into plain dialogue: drop index+timestamp lines, strip
    markup, unescape entities, and collapse the roll-up caption repetition (CC
    re-emits each line across several cues) by dropping a line identical to the
    one before it."""
    out: list[str] = []
    for block in srt.replace("\r", "").split("\n\n"):
        for row in block.splitlines():
            if "-->" in row or row.strip().isdigit():
                continue
            text = html.unescape(_SRT_TAG.sub("", row)).strip()
            if text and (not out or text != out[-1]):
                out.append(text)
    return " ".join(out).strip()


def subtitle_transcript(disc: Disc, title: Title, workdir: Path) -> str:
    """Pull the episode's dialogue from its SUBTITLES — better than whisper (exact
    words, whole episode) and near-instant (no audio decode, no transcription).

    DVD path: the MPEG-2 video carries EIA-608 closed captions as TEXT (no OCR).
    Stream-copy the title to a local mpg (preserving the video user-data;
    `-nosound` keeps it small) then let ffmpeg's `subcc` decoder emit an SRT.
    Returns "" when there are no captions (→ caller falls back to audio) or on a
    non-DVD disc — Blu-ray subtitles are PGS bitmaps (would need OCR), a follow-up."""
    if disc.format != "dvd":
        return ""
    mpg = workdir / f"cc_{title.id}.mpg"
    srt = workdir / f"cc_{title.id}.srt"
    for p in (mpg, srt):
        p.unlink(missing_ok=True)
    try:
        run(["mencoder", f"dvd://{title.id}", "-dvd-device", str(disc.path),
             "-nosound", "-ovc", "copy", "-of", "mpeg", "-o", str(mpg),
             "-really-quiet"], timeout=900)
        if not (mpg.exists() and mpg.stat().st_size > 0):
            return ""
        # movie source exposes closed captions as a second output (subcc)
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", f"movie={mpg}[out+subcc]", "-map", "0:1", str(srt)],
            timeout=900)
        if not srt.exists() or srt.stat().st_size == 0:
            return ""
        return srt_to_text(srt.read_text(errors="replace"))
    except (subprocess.TimeoutExpired, OSError) as e:
        log.warning("synopsis: subtitle extract failed (%s) %s title %d",
                    e, disc.path.name, title.id)
        return ""
    finally:
        for p in (mpg, srt):
            p.unlink(missing_ok=True)


def _probe_video_size(source: list) -> Optional[tuple]:
    """(width, height) of the first video stream, via ffprobe (header read only).
    The Blu-ray demuxer prints the resolution once PER CLIP (a multi-clip playlist
    yields `1920x1080\\n1920x1080\\n…`), so match the FIRST WxH rather than parsing
    the whole stdout — otherwise the size read is garbage and OCR silently dies."""
    try:
        r = run(["ffprobe", "-hide_banner", "-v", "error", "-select_streams",
                 "v:0", "-show_entries", "stream=width,height",
                 "-of", "csv=p=0:s=x", *source], timeout=120)
    except (subprocess.TimeoutExpired, OSError):
        return None
    m = re.search(r"(\d+)x(\d+)", r.stdout)
    return (int(m.group(1)), int(m.group(2))) if m else None


def subtitle_ocr_transcript(disc: Disc, title: Title, workdir: Path) -> str:
    """OCR the BITMAP subtitle track (Blu-ray PGS or DVD VOBSUB) into dialogue —
    the fallback when there are no closed captions (BD never carries CC; some DVD
    sets don't either). ffmpeg renders ONLY the subtitle stream onto a black
    canvas (the video is never decoded, so it's fast), `mpdecimate` keeps just the
    frames where the caption changed, and PP-OCR (RapidOCR) reads each — markedly
    more accurate than tesseract, especially on low-res VOBSUB.

    Blu-ray reads via the `bluray:` protocol directly; DVD has no ffmpeg protocol
    so mplayer first dumps the title's raw program stream (which carries the
    subpicture) to a local file. Returns "" if there's no subtitle track / OCR
    backend, so the caller falls through to whisper."""
    from text_region import ocr_texts, paddle_available
    if disc.format not in ("bluray", "dvd") or not paddle_available():
        return ""
    frames = workdir / f"subf_{title.id}"
    frames.mkdir(exist_ok=True)
    dump = workdir / f"sub_{title.id}.{'mkv' if disc.format == 'bluray' else 'vob'}"
    dur = int(title.duration) + 2 if title.duration else 3600
    size = None
    try:
        dump.unlink(missing_ok=True)
        # BOTH formats render from a LOCAL file, not the disc directly — rendering
        # straight off the `bluray:` protocol stitches the multi-clip playlist live
        # at ~1x realtime (~20 min/episode!).
        if disc.format == "bluray":
            # canvas size (PGS positions are absolute) via a cheap ffprobe header
            # read — the subtitle-only copy below carries no video stream.
            size = _probe_video_size(["-playlist", str(title.id),
                                      f"bluray:{disc.path}"])
            # copy ONLY the PGS subtitle stream (~1 s, a few MB). A full `-map 0`
            # remux both hits an unmuxable data stream and stitches the whole title.
            run(["ffmpeg", "-hide_banner", "-y", "-playlist", str(title.id),
                 "-i", f"bluray:{disc.path}", "-map", "0:s:0", "-c", "copy",
                 str(dump)], timeout=dur * 2 + 300)
        else:   # DVD has no ffmpeg protocol — mplayer dumps the program stream
            run(["mplayer", f"dvd://{title.id}", "-dvd-device", str(disc.path),
                 "-dumpstream", "-dumpfile", str(dump), "-really-quiet"],
                timeout=dur * 2 + 300)
        if not (dump.exists() and dump.stat().st_size > 0):
            return ""
        if size is None:                       # DVD: probe the dumped VOB
            size = _probe_video_size([str(dump)])
        if not size:
            return ""
        w, h = size
        # render subs on black, keep only changed frames (one per distinct caption),
        # and DOWNSCALE to ≤960px wide — PGS text is large, OCRs fine at half res,
        # and det cost scales with pixels (1080p→960 is ~4x fewer). DVD (720) is
        # left as-is by the min().
        # The color source MUST be duration-bounded (`d=`): on a video-less dump
        # (the BD subtitle-only copy) sub2video never signals EOF, so an infinite
        # color + overlay=shortest=1 renders forever, mpdecimate drops the
        # identical frames, output PTS never reaches -t, and ffmpeg spins until
        # the outer timeout (~76 min/title, observed on Avatar). A finite bg ends
        # the graph at `dur` regardless. (The DVD .vob path has video and never
        # hit this, but the bound is correct there too.)
        run(["ffmpeg", "-hide_banner", "-y", "-i", str(dump), "-t", str(dur),
             "-filter_complex",
             f"color=black:s={w}x{h}:r=2:d={dur}[bg];[bg][0:s:0]overlay=shortest=1,"
             "mpdecimate,scale='min(960,iw)':-2",
             "-fps_mode", "vfr", str(frames / "f%05d.png")],
            timeout=dur * 3 + 300)
        out: list[str] = []
        for text in ocr_texts(sorted(frames.glob("f*.png"))):   # parallel OCR
            text = " ".join(text.split())
            if text and (not out or text != out[-1]):
                out.append(text)
        return " ".join(out).strip()
    except (subprocess.TimeoutExpired, OSError) as e:
        log.warning("synopsis: subtitle OCR failed (%s) %s title %d",
                    e, disc.path.name, title.id)
        return ""
    finally:
        dump.unlink(missing_ok=True)
        for png in frames.glob("f*.png"):
            png.unlink(missing_ok=True)
        try:
            frames.rmdir()
        except OSError:
            pass


def full_transcript(disc: Disc, title: Title, workdir: Path) -> str:
    """Rip + transcribe the ENTIRE title audio — the default, and preferred over
    windowing. Plot-distinctive dialogue is strewn throughout an episode, so a few
    sampled windows can phase-skip the very lines that identify it; whisper
    base.en on CPU runs faster than realtime, and we only reach the synopsis path
    after the (far costlier) OCR fallback has already failed, so a whole-episode
    pass is affordable. (The cold-open "previously on" recap is a minor risk — it
    injects a little prior-episode plot — but it's tiny next to a full episode and
    the judge keys on specific in-episode events, not stray recap lines.)"""
    wav = workdir / f"aud_{title.id}_full.wav"
    if not rip_audio(disc, title, 0.0, title.duration, wav):
        return ""
    try:
        return transcribe(wav)
    finally:
        wav.unlink(missing_ok=True)


def sample_transcript(disc: Disc, title: Title, workdir: Path,
                      fractions=SAMPLE_FRACTIONS, length=SAMPLE_LENGTH) -> str:
    """Rip + transcribe a few windows spread across a title; join the text. The
    OPT-IN fast path (`--synopsis-windows N`); the default is `full_transcript`,
    which doesn't risk missing the identifying lines between windows.

    Spreading the samples (not one long block) captures plot-distinctive
    dialogue from different acts, which is what the synopsis judge keys on."""
    parts: list[str] = []
    for i, frac in enumerate(fractions):
        start = max(0.0, min(title.duration - length, title.duration * frac))
        wav = workdir / f"aud_{title.id}_{i}.wav"
        if not rip_audio(disc, title, start, length, wav):
            continue
        try:
            text = transcribe(wav)
        finally:
            wav.unlink(missing_ok=True)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _anthropic_text(model: str, prompt: str) -> str:
    """One text-only Anthropic Messages call. A quick swap for the Ollama judge:
    a Claude-id model routes here (via `_ollama_text`) so `run synopsis
    --judge-model claude-sonnet-5` uses a frontier judge with no other changes.
    Needs ANTHROPIC_API_KEY in the env; same text→_extract_json output contract."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set (needed for a claude judge)")
    # No `temperature`: it's deprecated on current Claude models (a 400), and
    # they're near-deterministic at greedy defaults anyway.
    body = json.dumps({
        "model": model, "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"content-type": "application/json", "x-api-key": key,
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    parts = [b.get("text", "") for b in data.get("content", [])
             if b.get("type") == "text"]
    return "".join(parts).strip()


def _ollama_text(model: str, prompt: str, host: str, retries: int = 3) -> str:
    """One text-only chat with retries (mirrors identify.ollama_chat's resilience
    to the daemon's OOM-restart, minus the image payload). A `claude-*` model id
    routes to the Anthropic backend instead — the retry/backoff also rides out a
    429/503/529 rate-limit or overload there."""
    is_claude = model.startswith("claude")
    # Ollama defaults num_ctx to 2048 and SILENTLY truncates a longer prompt to
    # its TAIL — on a full-episode transcript (~5k+ tokens) that drops most of the
    # dialogue the judge needs, and it abstains on every title (observed on the
    # Magicians full-transcript run). Size the window to the prompt + response
    # budget so the whole transcript is seen; cap at 32k (qwen2.5's native ctx).
    # num_predict caps TOTAL output tokens (thinking + answer). 2048 suffices for
    # a non-thinking judge's JSON, but a THINKING model (gemma4) spends most of it
    # reasoning and returns empty content if cut off mid-thought — bump it via
    # VJ_JUDGE_NUM_PREDICT for those. It's a cap not a target, so a larger value is
    # near-free for the non-thinking path (it stops as soon as the JSON is done).
    num_predict = int(os.environ.get("VJ_JUDGE_NUM_PREDICT", "2048"))
    approx = len(prompt) // 4 + num_predict         # ~4 chars/token + response room
    num_ctx = min(32768, max(4096, -(-approx // 4096) * 4096))  # round up to 4k
    payload = {
        "model": model, "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": num_ctx},
    }
    # VJ_JUDGE_THINK=false runs a thinking-capable model (gemma4) in NON-thinking
    # mode — worth it because for this task thinking reasons its way to confident-
    # wrong recurring-arc matches (observed: gemma4 thinking 1/4, worse than a
    # non-thinking judge). Only sent when set, so the default path is unchanged.
    think = os.environ.get("VJ_JUDGE_THINK")
    if think is not None:
        payload["think"] = think.strip().lower() not in ("false", "0", "no", "off")
    body = json.dumps(payload).encode()
    delays = [5, 15, 30]
    for attempt in range(retries + 1):
        try:
            if is_claude:
                return _anthropic_text(model, prompt)
            req = urllib.request.Request(
                f"{host}/api/chat", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
            return (data["message"].get("content") or "").strip()
        except Exception as e:  # noqa: BLE001
            if attempt >= retries:
                raise
            wait = delays[min(attempt, len(delays) - 1)]
            log.warning("judge request failed (%s); retry %d/%d in %ds",
                        e, attempt + 1, retries, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of an LLM reply (they wrap it in prose)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# One call ranks the WHOLE candidate row (the model already sees every synopsis
# and picks) — so a per-title ranked shortlist costs one call, not one per pair.
# The rank is what downstream assignment keys on; the model's confidence number
# is unreliable (0.85–0.95 on flat-wrong picks), the ordering far less so.
RANK_TOP_K = 6

_STAGE1 = """You are given a dialogue transcript sampled from one unknown TV \
episode, and a numbered list of candidate episodes with their plot synopses.

Rank the candidates the transcript most plausibly belongs to, best first, up to \
{k}. The dialogue will NOT quote the synopsis — judge by whether the events, \
places and situations in the transcript are CONSISTENT with a synopsis. Even \
when one candidate clearly fits best, ALSO list your next 2-3 most plausible as \
lower-ranked entries — a later step assigns each episode to at most one title, \
so a ranked backup lets a title recover when its top pick is claimed by a better \
match. Only list candidates with a genuine episode-specific match; return [] if \
none fits.

CRITICAL: series regulars (main characters in every episode) are NOT evidence, \
and neither is the show's recurring premise or its season-long arc — every \
episode shares those, so they cannot rank one candidate above another. Key ONLY \
on details unique to a single episode: distinctive plot events, guest \
characters, specific named locations, one-off objects or situations. If the \
transcript is production commentary, or surfaces no episode-specific detail at \
all, return an empty ranking [].

Reply with ONLY a JSON object:
{{"ranking": [<candidate numbers, best first, at most {k}>], \
"confidence": <0.0-1.0 for the top pick>, \
"evidence": "<the episode-specific detail behind the top pick>"}}

CANDIDATES:
{candidates}

TRANSCRIPT:
{transcript}
"""


def rank_candidates(transcript: str, candidates: list[Episode],
                    model: str = JUDGE_MODEL, host: str = "http://localhost:11434",
                    top_k: int = RANK_TOP_K) -> tuple[list, str]:
    """One judge call → the candidates whose synopsis best fits the transcript,
    ranked best-first. Returns ([(episode, borda_score)], evidence): score is
    rank-derived (top_k for 1st, top_k-1 for 2nd, …) so downstream assignment
    keys on the reliable ORDER, not the model's confidence. ([], reason) when it
    abstains (empty ranking / no transcript / <2 synopses)."""
    pool = [e for e in candidates if e.synopsis]
    if not transcript:
        return [], "no transcript"
    if len(pool) < 2:
        return [], "need >=2 synopses to discriminate"
    listing = "\n".join(
        f"{i+1}. {e.name}: {e.synopsis}" for i, e in enumerate(pool))
    reply = _ollama_text(
        model, _STAGE1.format(candidates=listing, transcript=transcript,
                              k=top_k), host)
    obj = _extract_json(reply) or {}
    ranked, seen = [], set()
    for pos, num in enumerate(obj.get("ranking") or []):
        try:
            idx = int(num) - 1
        except (ValueError, TypeError):
            continue
        if 0 <= idx < len(pool) and idx not in seen:
            seen.add(idx)
            ranked.append((pool[idx], max(1, top_k - len(ranked))))
        if len(ranked) >= top_k:
            break
    return ranked, (obj.get("evidence") or "").strip() or "judge abstained"


def transcribe_and_rank(disc: Disc, title: Title, candidates: list[Episode],
                        workdir: Path, model: str = JUDGE_MODEL,
                        host: str = "http://localhost:11434",
                        fractions=SAMPLE_FRACTIONS, length=SAMPLE_LENGTH,
                        top_k: int = RANK_TOP_K) -> tuple[list, str]:
    """Sample a title's dialogue and rank it against the pool (one judge call).
    Returns ([(episode, score)], evidence) — see `rank_candidates`."""
    transcript = sample_transcript(disc, title, workdir,
                                   fractions=fractions, length=length)
    if not transcript:
        return [], "no dialogue transcribed"
    return rank_candidates(transcript, candidates, model, host, top_k)


def assign_by_synopsis(rows: list, episodes: list[Episode]) -> dict:
    """Global one-episode-per-title assignment over the ranked judge scores — the
    fix for the MAGNET failure mode where many titles independently pick the same
    episode (its synopsis is arc-heavy). Solved as max-weight bipartite matching
    (Hungarian) so each episode is claimed at most once.

    `rows`: [(title_key, [(episode, score), …])]. Returns {title_key: (episode,
    score, rank)} for assigned titles — ONLY shortlisted (score>0) pairs, so a
    title whose whole shortlist is taken by better-fitting titles abstains rather
    than being forced onto a wrong episode. Order-agnostic (Hungarian), which
    suits the scrambled-Blu-ray case synopsis escalates for; a play-all-ordered
    disc could instead use a monotonic DP, but that's not wired yet."""
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    if not rows or not episodes:
        return {}
    ep_idx = {(e.season, e.number): j for j, e in enumerate(episodes)}
    score = np.zeros((len(rows), len(episodes)))
    rank_at = {}
    for i, (_key, ranked) in enumerate(rows):
        for pos, (ep, s) in enumerate(ranked):
            j = ep_idx.get((ep.season, ep.number))
            if j is not None:
                score[i, j] = s
                rank_at[(i, j)] = pos + 1
    out = {}
    for i, j in zip(*linear_sum_assignment(-score)):   # maximise total score
        if score[i, j] > 0:
            out[rows[i][0]] = (episodes[j], float(score[i, j]), rank_at[(i, j)])
    return out
