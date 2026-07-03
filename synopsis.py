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
def spread_fractions(n: int) -> tuple:
    """`n` positions evenly spread across a title's interior (avoiding the very
    start/end): n=3 -> (0.25, 0.5, 0.75)."""
    n = max(1, n)
    return tuple((i + 1) / (n + 1) for i in range(n))


SAMPLE_WINDOWS = 3
SAMPLE_FRACTIONS = spread_fractions(SAMPLE_WINDOWS)
SAMPLE_LENGTH = 40.0

JUDGE_MODEL = "gemma4:latest"
ACCEPT = 0.7          # min stage-1 confidence to bother running the contrast


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


_STAGE1 = """You are given a dialogue transcript sampled from one unknown TV \
episode, and a numbered list of candidate episodes with their plot synopses.

Decide which single candidate the transcript most plausibly belongs to. The \
dialogue will NOT quote the synopsis — judge by whether the events, places and \
situations in the transcript are CONSISTENT with a synopsis.

CRITICAL: series regulars (main characters who appear in every episode) are NOT \
evidence. A synopsis and transcript both mentioning a lead character means \
nothing — every episode has them. Key ONLY on details unique to one episode: \
distinctive plot events, guest characters, specific named locations, one-off \
objects or situations. If the transcript is production commentary (people \
discussing making the show), or fits no synopsis distinctly better than the \
rest on episode-specific details, choose null.

Also name the SECOND most plausible candidate (never null unless you chose \
null) — the nearest competitor — so it can be checked against the winner.

Reply with ONLY a JSON object:
{{"choice": <candidate number or null>, "runner_up": <second-best number>, \
"confidence": <0.0-1.0>, "evidence": "<the specific detail that decided it>"}}

CANDIDATES:
{candidates}

TRANSCRIPT:
{transcript}
"""

_STAGE2 = """A dialogue transcript from an unknown TV episode could plausibly \
belong to one of two candidate episodes. Cite ONE concrete detail in the \
TRANSCRIPT that fits candidate A AND rules OUT candidate B — a plot event, \
guest character, place, or object that belongs to A's story but not B's.

Anything the two episodes SHARE cannot rule anything out, so it does not count: \
a recurring main character, the show's usual setting, or generic dialogue is \
worthless here because it fits both A and B equally. If nothing in the \
transcript distinguishes A from B, answer false.

Reply with ONLY a JSON object:
{{"distinguishes": <true|false>, "detail": "<the detail that fits A but not B, or ''>"}}

CANDIDATE A: {a_name} — {a_overview}
CANDIDATE B: {b_name} — {b_overview}

TRANSCRIPT:
{transcript}
"""


def judge_by_synopsis(transcript: str, candidates: list[Episode],
                      model: str = JUDGE_MODEL, host: str = "http://localhost:11434",
                      accept: float = ACCEPT
                      ) -> tuple[Optional[Episode], float, str]:
    """Forced-choice + abstention identity from dialogue vs season synopses.

    Returns (episode, confidence, evidence) or (None, score, reason). Abstains
    unless: stage-1 names a candidate at confidence >= accept, and a contrastive
    stage-2 cites a detail that fits the winner AND rules out its nearest
    competitor (so recurring cast / usual setting / generic banter can't carry a
    match — they fit both and cancel). Only candidates with a synopsis are
    offered."""
    pool = [e for e in candidates if e.overview]
    if not transcript:
        return None, 0.0, "no transcript"
    if len(pool) < 2:
        return None, 0.0, "need >=2 synopses to discriminate"

    listing = "\n".join(
        f"{i+1}. {e.name}: {e.overview}" for i, e in enumerate(pool))
    reply = _ollama_text(
        model, _STAGE1.format(candidates=listing, transcript=transcript), host)
    obj = _extract_json(reply)
    if not obj or obj.get("choice") in (None, "null"):
        return None, 0.0, "judge abstained"
    try:
        idx = int(obj["choice"]) - 1
    except (ValueError, TypeError):
        return None, 0.0, "unparseable choice"
    if not 0 <= idx < len(pool):
        return None, 0.0, "choice out of range"
    conf = float(obj.get("confidence") or 0.0)
    chosen = pool[idx]
    if conf < accept:
        return None, conf, f"low confidence ({conf:.2f})"

    # find the runner-up (nearest competitor) to contrast against
    try:
        ridx = int(obj.get("runner_up")) - 1
    except (ValueError, TypeError):
        ridx = -1
    if not (0 <= ridx < len(pool)) or ridx == idx:
        return None, conf, "no distinct runner-up to contrast against"
    runner = pool[ridx]

    # stage 2: the pick must be justified by a detail that fits the winner AND
    # rules OUT the runner-up. Anything the two share (recurring cast, the usual
    # setting, generic banter) can't discriminate, so it can't pass here — no
    # exclusion list needed, it cancels by construction.
    reply2 = _ollama_text(model, _STAGE2.format(
        a_name=chosen.name, a_overview=chosen.overview,
        b_name=runner.name, b_overview=runner.overview,
        transcript=transcript), host)
    obj2 = _extract_json(reply2) or {}
    if not obj2.get("distinguishes") or not (obj2.get("detail") or "").strip():
        return None, conf, f"stage-2: nothing distinguishes {chosen.name} from {runner.name}"
    return chosen, conf, obj2["detail"].strip()


def identify_by_synopsis(disc: Disc, title: Title, candidates: list[Episode],
                         workdir: Path, model: str = JUDGE_MODEL,
                         host: str = "http://localhost:11434",
                         fractions=SAMPLE_FRACTIONS, length=SAMPLE_LENGTH
                         ) -> tuple[Optional[Episode], float, str]:
    """End-to-end: sample dialogue from a title and judge it against the pool.

    `fractions`/`length` set how many dialogue windows to sample and how long
    each is — more/longer sees more of the episode at more whisper cost."""
    transcript = sample_transcript(disc, title, workdir,
                                   fractions=fractions, length=length)
    if not transcript:
        return None, 0.0, "no dialogue transcribed"
    return judge_by_synopsis(transcript, candidates, model, host)
