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

import json
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
    DVD title-audio extraction is a follow-up — libdvdread has no ffmpeg
    protocol, so it needs a mencoder path; return None for now so the caller
    skips DVD discs cleanly."""
    out.unlink(missing_ok=True)
    if disc.format != "bluray":
        log.warning("synopsis: DVD audio rip not implemented (%s title %d)",
                    disc.path.name, title.id)
        return None
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


def sample_transcript(disc: Disc, title: Title, workdir: Path,
                      fractions=SAMPLE_FRACTIONS, length=SAMPLE_LENGTH) -> str:
    """Rip + transcribe a few windows spread across a title; join the text.

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


def _ollama_text(model: str, prompt: str, host: str, retries: int = 3) -> str:
    """One text-only Ollama chat with retries (mirrors identify.ollama_chat's
    resilience to the daemon's OOM-restart, minus the image payload)."""
    body = json.dumps({
        "model": model, "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0, "num_predict": 2048},
    }).encode()
    delays = [5, 15, 30]
    for attempt in range(retries + 1):
        try:
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
