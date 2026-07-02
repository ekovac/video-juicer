"""Episode matching: metadata alignment, title-card OCR, and manifest output."""
from __future__ import annotations

import base64
import fcntl
import json
import os
import re
import shlex
import subprocess
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from discs import (Title, Disc, Episode, Assignment, run, log, group_discs)

# ---------------------------------------------------------------------------
# Stage 3a: per-disc classification (permissive candidates + evidence)
# ---------------------------------------------------------------------------

PLAYALL_CHAPTER_TOL = 2.0     # observed agreement on real discs is 0.1 s
CLEAN_MATCH_TOL = 90.0        # TMDB runtimes round to whole minutes
HARD_MATCH_TOL = 420.0


def detect_play_all(titles: list[Title]) -> Optional[tuple[Title, list[Title]]]:
    """Find the DVD play-all and its episode titles in order, or None.

    Runtime-INDEPENDENT — keys on the disc's own structure, not TMDB runtimes
    (which can be wrong: Broken Saints reports a uniform 9 min for 9-49 min
    chapters). Two strategies, precise first:

    1. **Chapter-match**: the play-all's chapter marks segment into the other
       titles' durations. Handles one-chapter-per-episode (8 chapters <-> 8
       titles) and multi-chapter-per-episode (25 chapters spanning 5) by
       greedily accumulating chapters until the running sum hits an unused
       title. Exact when the marks fall on episode boundaries.
    2. **Duration-sum fallback**: the longest title whose runtime == the sum of
       its peers (peers share its audio layout; extras/menus have fewer
       streams). Catches discs whose play-all chapters DON'T align to title
       boundaries (Broken Saints D3/D4, where chapter-match finds nothing).
       Looser — can sweep in a same-audio extra — so it's the fallback. Inert
       on Blu-ray (n_audio is 0 there, so every title "matches" and the sum
       overshoots), where order_by_playall handles play-alls via clips instead.

    Returns (play_all, episodes_in_play_order).
    """
    # Strategy 1: chapter segmentation.
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
        for ch in cand.chapters:
            acc += ch
            hit = next((o for o in unused
                        if abs(o.duration - acc) <= PLAYALL_CHAPTER_TOL), None)
            if hit:
                matched.append(hit)
                unused.remove(hit)
                acc = 0.0
        # Allow a small unmatched tail (credits/logo chapter).
        if acc <= 30.0 and len(matched) >= 2 and (
                best is None or len(matched) > len(best[1])):
            best = (cand, matched)
    if best:
        return best

    # Strategy 2: duration-sum fallback (the longest title concatenates peers
    # of the same audio richness). The episodes' DVD title order is play order.
    if len(titles) < 4:
        return None
    pa = max(titles, key=lambda t: t.duration)
    rich = pa.n_audio >= 2          # commentary present -> match by ">= 2";
    peers = [t for t in titles if t is not pa and t.duration > 60
             and (t.n_audio >= 2 if rich else t.n_audio == pa.n_audio)]
    if len(peers) < 3:
        return None
    total = sum(t.duration for t in peers)
    if abs(total - pa.duration) > max(30.0, 0.03 * pa.duration):
        return None
    return pa, sorted(peers, key=lambda t: t.order_key)


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
    aligned monotonically (the aligner can't reorder). Signals, in order:
      - DVD with a disc hint, a play-all, or multi-part "(1)/(2)" names in
        sequence  -> order/numbering corroborated;
      - episodes ~all one length     -> runtime can't order them at all;
      - large alignment deltas        -> the monotonic order conflicts with
        the TMDB runtimes, i.e. the playlist order is likely scrambled;
      - otherwise the runtimes fit the order -> trust it.
    Returns (verifiable, reason)."""
    if len(assignments) <= 1:
        return True, "single title"
    eps = [e for a in assignments for e in a.episodes]
    rts = sorted(e.runtime for e in eps if e.runtime)
    min_gap = min((rts[i + 1] - rts[i] for i in range(len(rts) - 1)),
                  default=tol + 1)
    if disc.format == "dvd":
        # Within a disc, lsdvd title order IS broadcast order. But the aligner
        # decides which episode each disc *starts* on; with no season/disc hint
        # AND same-runtime episodes it has no anchor and can drop or shift a
        # title, renumbering the rest (Sonic SatAM: dropped a 22.7-min episode
        # that looked like its peers). A disc hint anchors the numbering.
        if disc.disc_hint is not None:
            return True, "DVD title order (disc hint anchors numbering)"
        if min_gap <= tol:
            return False, ("DVD, episodes ~same runtime and no disc hint — "
                           "cross-disc numbering is unanchored and can drop or "
                           "shift a title; recommend --ocr-identify")
        return True, "DVD title order"
    if any(t.kind == "play-all" for t in disc.titles):
        return True, "play-all corroborates order"
    if sum(1 for e in eps if re.search(r"\(\d+\)\s*$", e.name)) >= 2:
        return True, "multi-part titles corroborate order"
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

# Token cap for the VLM call. A thinking model needs room to finish reasoning
# AND emit the answer; 2048 truncated rare complex frames (leaking the partial
# chain-of-thought as a fake transcription). It's a cap, not a target.
VLM_NUM_PREDICT = 8192


def ollama_chat(model: str, prompt: str, image_path: Path,
                host: str, retries: int = 4) -> str:
    """One VLM request with retries.

    Retries cover ollama getting OOM-killed mid-request (known memory leak):
    the daemon restarts but in-flight requests die with connection errors,
    and the first retry pays a model reload, hence the generous timeout.
    """
    img_b64 = base64.b64encode(image_path.read_bytes()).decode()
    # Thinking models (qwen3-vl) spend tokens reasoning before the final answer;
    # if num_predict is exhausted mid-think, `content` comes back empty. It's a
    # CAP not a target (a frame that finishes early stops regardless), so set it
    # generously: near-free for normal frames, and it lets the rare complex
    # frame finish instead of truncating. think:false is a no-op for this model.
    body = json.dumps({
        "model": model, "stream": False,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "options": {"num_predict": VLM_NUM_PREDICT, "temperature": 0},
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
            content = (msg.get("content") or "").strip()
            if not content and data.get("done_reason") == "length":
                # Truncated mid-think: the leftover `thinking` is chain-of-thought,
                # NOT a transcription. Matching against it yields confident-wrong
                # hits, so drop it — a missing read is recoverable by position/
                # elimination; a wrong one silently corrupts the mapping.
                log.warning("%s: VLM truncated mid-think at num_predict=%d; "
                            "no transcription", image_path.name, VLM_NUM_PREDICT)
                # Persist the offending frame (content-hashed, so dups collapse)
                # when OCR_DEBUG_FRAMES names a dir — for diagnosing WHY these
                # frames make the model reason past the cap.
                dbg = os.environ.get("OCR_DEBUG_FRAMES")
                if dbg:
                    import hashlib
                    raw = image_path.read_bytes()
                    out = Path(dbg)
                    out.mkdir(parents=True, exist_ok=True)
                    (out / f"trunc_{hashlib.md5(raw).hexdigest()[:10]}.jpg"
                     ).write_bytes(raw)
            return content
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


# Markers that a word-boundary title hit is incidental, not a real card.
# Two kinds, handled differently in fuzzy_best:
#  - CREDITS: a genuine card can carry a credit line beside a distinctive
#    title, so a distinctive verbatim hit there is still trustworthy (1.0).
#  - REASONING: the VLM's own chain-of-thought (used when a thinking model
#    returns no `content` and we fall back to `thinking`). Any title it names
#    is the model *guessing*, not transcribing — it once scored a hallucinated
#    "The World in the Walls" at 1.0 and overrode correct metadata. A title
#    inside a reasoning dump must never win on verbatim presence alone.
_CREDIT_MARKERS = (
    "producer", "directed", "director", "written", "writer", "teleplay",
    "story by", "music", "edited", "editor", "starring", "executive",
    "casting", "narrat",
)
_REASONING_MARKERS = (
    "got it", "let s", "the image", "i need", "looking at", "transcribe",
    "appears to", "this is", "the text reads", "i can see", "the title",
)
_INCIDENTAL_MARKERS = _CREDIT_MARKERS + _REASONING_MARKERS


def fuzzy_best(text: str, episodes: list[Episode]) -> tuple[Optional[Episode], float]:
    """Best episode-name match for transcribed frame text (closed set)."""
    import difflib
    norm = canon_parts(normalize_text(text))
    if not norm:
        return None, 0.0
    reasoning = any(m in norm for m in _REASONING_MARKERS)
    incidental = reasoning or any(m in norm for m in _CREDIT_MARKERS)
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
            if reasoning:
                # A reasoning dump that merely *names* a title is the model
                # guessing — never let verbatim presence alone score 1.0. Use
                # coverage: in a long chain-of-thought the title is a tiny
                # fraction, so it lands well below ocr_accept and is rejected.
                score = len(name) / len(norm)
            elif distinctive or not incidental:
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



def tesseract_text(frame: Path) -> str:
    """Fast CPU OCR of a frame via Tesseract. ~60-95 ms/frame; returns "" on no
    text or failure. Plain block title cards (Sonic, Enterprise, Avatar) read
    cleanly and, crucially, a busy text-less scene returns "" instantly instead
    of a thinking VLM burning tokens on it. Stylized script cards (Venture Bros)
    don't read here — the VLM fallback in verify_title covers those."""
    try:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(frame)).strip()
    except Exception as e:  # noqa: BLE001 - tesseract/PIL missing or bad image
        log.debug("tesseract failed on %s: %s", frame.name, e)
        return ""


_VLM_OK: dict[tuple, bool] = {}


def _vlm_reachable(model: str, host: str) -> bool:
    """vlm_available, cached per (model, host) so the hybrid path checks once."""
    key = (model, host)
    if key not in _VLM_OK:
        _VLM_OK[key] = vlm_available(model, host)
    return _VLM_OK[key]


def verify_title(disc: Disc, title: Title, episodes: list[Episode],
                 model: str, host: str, workdir: Path,
                 accept: float = 0.8, anchor: Optional[float] = None,
                 engine: str = "auto", capture: Optional[dict] = None,
                 text_filter: bool = True, vlm_budget: int = 80
                 ) -> tuple[Optional[Episode], float, Optional[float]]:
    """Identify a title by OCRing its title card against the episode names.

    Returns (episode, score, card_seconds) — card_seconds is where the matching
    card was found, so a caller can learn the per-disc location.

    Windowless (SD MPEG-2 decodes at ~2 ms/frame, so extracting a whole 50-min
    title is ~10 s): rip+extract every frame once, then three tiers cheap-first:
      1. **Tesseract** over all frames — plain block cards read outright, ANYWHERE
         in the title (no window to miss them); returns on the first accept.
      2. an OCR-free **text-region gate** (`frame_has_text`, PaddleOCR/EAST)
         prunes text-less scene frames from the expensive VLM pass.
      3. the **VLM** reads the gate survivors, capped at `vlm_budget` calls so a
         card-less title can't run away (a missed read is recovered later by
         elimination).
    Frames are processed in a card-likely ORDER — nearest a known `anchor`
    (learned from an earlier episode on the disc), else nearest either END (cold
    opens and end-cards both land early) — so a card-bearing title early-exits
    after a few reads. The anchor is now only an ordering hint, not a window."""
    state = {"ep": None, "score": 0.0, "time": None, "frame": None, "text": None}
    use_tess = engine in ("auto", "tesseract")
    use_vlm = engine == "vlm" or (engine == "auto"
                                  and _vlm_reachable(model, host))
    has_text = None
    if text_filter and use_vlm:
        try:
            from text_region import frame_has_text as has_text
        except Exception:  # noqa: BLE001 — cv2/rapidocr missing
            has_text = None

    def result():
        if capture is not None:
            capture.update(image=state["frame"], time=state["time"],
                           text=state["text"], score=state["score"])
        return state["ep"], state["score"], state["time"]

    def ocr_pass(timed, ocr_fn, tag, gate=None, budget=None):
        """OCR frames in order, skipping gate failures; update the running best
        and return (ep, score) on the first accept. `budget` caps the OCR calls
        actually made (gate-skipped frames don't count)."""
        calls = 0
        for frame, ts in timed:
            if gate is not None and not gate(frame):
                continue
            if budget is not None and calls >= budget:
                log.info("%s title %d: %s budget (%d) reached; stopping",
                         disc.path.name, title.id, tag, budget)
                break
            try:
                text = ocr_fn(frame)
            except Exception as e:  # noqa: BLE001
                log.error("%s failed on %s: %s", tag, frame.name, e)
                continue
            calls += 1
            ep, score = fuzzy_best(text, episodes)
            if score > state["score"]:
                state["ep"], state["score"], state["time"] = ep, score, ts
                if capture is not None:   # keep the winning frame for review
                    try:
                        state["frame"], state["text"] = frame.read_bytes(), text
                    except OSError:
                        pass
            if score >= accept:
                log.info("%s title %d: verified %r -> S%02dE%02d (%.2f) @%.0fs "
                         "[%s]", disc.path.name, title.id,
                         text.splitlines()[0][:60] if text else "",
                         ep.season, ep.number, score, ts, tag)
                return ep, score
        return None

    # 1. extract every frame of the whole title, once (then free the rip)
    video = rip_window(disc, title, 0.0, title.duration, workdir)
    if not video:
        return result()
    try:
        frames = extract_frames(video, workdir)
    finally:
        video.unlink(missing_ok=True)
    timed = [(f, i * FRAME_INTERVAL) for i, f in enumerate(frames)]
    if anchor is not None:                       # nearest the learned card first
        timed.sort(key=lambda ft: abs(ft[1] - anchor))
    else:                                        # else nearest either end first
        timed.sort(key=lambda ft: min(ft[1], title.duration - ft[1]))

    # 2. cheap Tesseract sweep (plain cards, anywhere) -> return on accept
    if use_tess and ocr_pass(timed, tesseract_text, "tesseract"):
        return result()
    # 3. VLM the gate survivors, budget-capped
    if use_vlm and state["score"] < accept:
        ocr_pass(timed, lambda f: ollama_chat(model, VLM_PROMPT, f, host),
                 "vlm", gate=has_text, budget=vlm_budget)
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
    # A DVD play-all names the episode set structurally, independent of
    # runtimes — use it when present, since it survives wrong TMDB durations
    # (Broken Saints). On Blu-ray the play-all set isn't enumerated this way
    # (order_by_playall only fixes order), so fall through to the band there.
    if disc.format == "dvd":
        pa = detect_play_all([t for t in disc.titles if t.duration > 0])
        if pa:
            pa[0].kind = "play-all"
            for t in pa[1]:
                t.kind = "episode-candidate"   # ocr_identify skips 2x-doubling
            return list(pa[1])                 # for these (runtimes untrusted)
    bands = valid_episode_lengths(pool)
    if bands:
        lengths, tol = bands
        def ok(dur):
            return min(abs(dur - v) for v in lengths) <= tol
    else:
        rts = [e.runtime for e in pool if e.runtime]
        expected = sorted(rts)[len(rts) // 2] if rts else 1320.0
        def ok(dur):
            return expected * 0.6 <= dur <= expected * 2.6
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
                    accept=args.ocr_accept, anchor=anchor,
                    engine=getattr(args, "ocr_engine", "auto"))
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
                # combined two-parter playlist (~2x runtime): claim next too.
                # This trusts the TMDB runtime, so skip it for play-all-derived
                # candidates — the play-all is used precisely *because* runtimes
                # are unreliable (Broken Saints: every 9-min TMDB runtime vs real
                # 9-49 min titles made every single look like a 2x double, then
                # got the correct single demoted). The play-all already segments
                # the disc into one-episode units; its card gives the identity.
                if (t.kind != "episode-candidate" and ep.runtime
                        and t.duration >= ep.runtime * 1.6):
                    nxt = next((e for e in pool if e.number == ep.number + 1), None)
                    if nxt:
                        eps.append(nxt)
                delta = abs(t.duration - sum(e.runtime or 0 for e in eps))
                raw.append(Assignment(d, t, eps, delta, "high", ep.name))

    final, claimed = resolve_assignment_collisions(raw, leftovers)

    hinted = {d.season_hint for d in discs if d.season_hint}
    missed = [e for n in sorted(seasons) for e in seasons[n]
              if (e.season, e.number) not in claimed
              and (not hinted or e.season in hinted)]
    return recover_by_elimination(final, leftovers, missed)


def _format_quality(t: Title) -> tuple[int, int]:
    """Coarse (video, audio) quality rank; higher is better.

    When two same-disc playlists OCR to the *same* episode they share a title
    card, so identity can't separate them — but one may be a clean HD master and
    the other a lossy alternate. Avatar authors an audio-commentary playlist
    (480i / AC3) beside each episode's 1080p / DTS-HD master; the commentary one
    is often marginally longer, so a pure duration tiebreak picked it (S01E15
    "Bato" ripped from the SD commentary title). Rank by format so the master
    wins. Titles with no parsed format (all DVDs, unparsed STN) rank (0, 0) —
    equal — so this is inert unless the formats actually differ."""
    v = 0
    if t.video_format:
        m = re.match(r"(\d+)", t.video_format)
        if m:                       # progressive edges interlaced at equal res
            v = int(m.group(1)) * 2 + (0 if t.video_format.endswith("i") else 1)
    a = 0
    if t.audio_format:              # lossless master beats a lossy alternate
        af = t.audio_format.upper()
        a = 2 if any(k in af for k in
                     ("HDMA", "TRUEHD", "PCM", "FLAC")) else 1
    return v, a


def resolve_assignment_collisions(
        raw: list[Assignment],
        leftovers: list[tuple[Disc, Title]]
) -> tuple[list[Assignment], dict[tuple, Assignment]]:
    """Reduce overlapping OCR assignments to one claim per episode.

    Prefer single-episode assignments over doubles, then the higher-quality
    source (an HD/lossless master over a same-episode SD/commentary alternate),
    then the longest (most complete) playlist. A combined double is kept if it
    carries at least one *unclaimed* episode (e.g. a finale where E19 exists
    only inside the E19+E20 double while E20 also has a single) — only dropped
    when fully redundant. Demoted titles are appended to `leftovers`. Returns
    (final, claimed)."""
    final: list[Assignment] = []
    claimed: dict[tuple, Assignment] = {}
    for a in sorted(raw, key=lambda a: (len(a.episodes),
                                        tuple(-q for q in _format_quality(a.title)),
                                        -a.title.duration)):
        keys = [(e.season, e.number) for e in a.episodes]
        unclaimed = [k for k in keys if k not in claimed]
        if not unclaimed:
            leftovers.append((a.disc, a.title))
            log.warning("%s pl %d (%r) fully duplicates already-identified "
                        "episode(s); treating as extra", a.disc.path.name,
                        a.title.id, a.episodes[0].name)
            continue
        # A retained combined playlist supersedes any same-disc standalone
        # single it already contains. The double is kept because it is the sole
        # source of its *other* episode (e.g. E13 lives only inside the E12+E13
        # playlist); but it also carries E12, which a standalone single already
        # claimed. Ripping both would duplicate E12's content on disk and make
        # Plex see S02E12 and S02E12-E13 overlap, so demote the redundant single
        # to an extra — the merged file is now the source for both episodes.
        # (Cross-disc dups are resolve_cross_disc's job; this is same-disc only.)
        for k in keys:
            if k in unclaimed:
                continue
            prev = claimed[k]
            if (prev.disc is a.disc and len(prev.episodes) < len(a.episodes)
                    and any(p is prev for p in final)):
                final = [p for p in final if p is not prev]
                leftovers.append((prev.disc, prev.title))
                log.warning("%s pl %d (%r) is contained in combined playlist "
                            "pl %d (%r); demoting the standalone single to extra",
                            prev.disc.path.name, prev.title.id,
                            prev.episodes[0].name, a.title.id, a.episodes[0].name)
                claimed[k] = a
        for k in unclaimed:
            claimed[k] = a
        final.append(a)
    return final, claimed


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


def format_outliers(assignments: list[Assignment]
                    ) -> tuple[Optional[str], list[Assignment]]:
    """Episodes whose video format differs from the run's majority.

    A few episodes authored at a lower quality than the rest (Avatar's Sozin's
    Comet finale is 480i SD while the season is 1080p) won't be caught by
    runtime/title-card matching — they map correctly, just to an inferior
    source. Flag them so the user can decide before ripping. Returns
    (majority_format, outlier_assignments); empty when formats agree or are
    unknown (DVD / unparsed STN)."""
    from collections import Counter
    fmts = [a.title.video_format for a in assignments if a.title.video_format]
    if len(fmts) < 3 or len(set(fmts)) < 2:
        return None, []
    majority = Counter(fmts).most_common(1)[0][0]
    outliers = [a for a in assignments
                if a.title.video_format and a.title.video_format != majority]
    return majority, outliers


def verify_assignment(a: Assignment, seasons: dict[int, list[Episode]],
                      vlm_model: str, ollama_host: str, workdir: Path,
                      accept: float = 0.8, engine: str = "auto") -> Optional[bool]:
    """OCR one assignment's title card and reconcile it with the alignment.

    Mutates `a` (verified_name / episodes / confidence) when the OCR is
    confident. Returns True if the card CONFIRMED the alignment, False if it
    OVERRODE it (a real disagreement), None if no card was found (alignment
    kept). The True/False distinction is what a spot-check keys on."""
    season_pool = seasons.get(a.episodes[0].season, [])
    ep, score, _ = verify_title(a.disc, a.title, season_pool, vlm_model,
                                ollama_host, workdir, accept=accept, engine=engine)
    if ep and score >= accept:
        agreed = (ep.season == a.episodes[0].season
                  and [ep.number] == [e.number for e in a.episodes])
        a.verified_name = ep.name
        if not agreed:
            log.warning("%s title %d: OCR says S%02dE%02d %r, alignment said "
                        "%s — using OCR", a.disc.path.name, a.title.id,
                        ep.season, ep.number, ep.name,
                        [e.number for e in a.episodes])
            a.episodes = [ep]
        a.confidence = "high"
        return agreed
    log.warning("%s title %d: no title card found (best fuzzy score %.2f%s) — "
                "keeping alignment result %s", a.disc.path.name, a.title.id,
                score, f" vs {ep.name!r}" if ep else "",
                [e.number for e in a.episodes])
    return None


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
                                            accept=args.ocr_accept,
                                            engine=getattr(args, "ocr_engine", "auto"),
                                            text_filter=False)
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


def _episode_tag(eps: list[Episode]) -> str:
    """SxxEyy for a single episode; SxxEyy-Ezz for a multi-episode title.

    The hyphen-joined range is the form Plex and Jellyfin both parse as one
    file covering several episodes (vs the old E01E02 run, which neither does)."""
    s = eps[0].season
    if len(eps) == 1:
        return f"S{s:02d}E{eps[0].number:02d}"
    return f"S{s:02d}E{eps[0].number:02d}-E{eps[-1].number:02d}"


def suggested_filename(show: str, eps: list[Episode],
                       year: Optional[int] = None,
                       tmdb_id: Optional[int] = None) -> str:
    """Relative library path in the Plex/Jellyfin preferred layout:

        <Show (Year) {tmdb-ID}>/<Season NN>/<Show (Year)> - SxxEyy - Names.mkv

    The TMDB id is an agent-match hint that belongs on the show *folder* only
    (both servers read `{tmdb-NNN}` there); the file prefix stays clean.
    Season 0 lands in the "Specials" folder (recognised by both servers).
    Each path component is sanitised independently, then joined with "/" —
    the separators in the returned string are deliberate, not stray."""
    prefix = sanitize_filename(f"{show} ({year})" if year else show)
    show_dir = prefix + (f" {{tmdb-{tmdb_id}}}" if tmdb_id else "")
    season = eps[0].season
    folder = "Specials" if season == 0 else f"Season {season:02d}"
    names = " & ".join(e.name for e in eps)
    fname = sanitize_filename(f"{prefix} - {_episode_tag(eps)} - {names}.mkv")
    return f"{show_dir}/{folder}/{fname}"


def _record_video_majority(records: list[dict]) -> Optional[str]:
    """Majority video format among episode records, or None if uniform/unknown."""
    from collections import Counter
    fmts = [r.get("video_format") for r in records
            if r.get("kind") == "episode" and r.get("video_format")]
    if len(fmts) < 3 or len(set(fmts)) < 2:
        return None
    return Counter(fmts).most_common(1)[0][0]


def _emit_rip_line(r: dict, preset_var: str) -> None:
    out = r["suggested_filename"]
    parent = os.path.dirname(out)
    if parent:
        print(f'mkdir -p "$PREFIX/{parent}"')
    print(f'HandBrakeCLI "${{HANDBRAKE_OPTS[@]}}" '
          f'-i {shlex.quote(r["image"])} -t {r["title"]} '
          f'--preset "{preset_var}" -o "$PREFIX/{out}"')


def emit_rip_commands(records: list[dict], preset: str,
                      output_prefix: Optional[str] = None) -> None:
    """Emit a runnable bash rip script (one HandBrakeCLI call per episode).

    The common knobs are hoisted into shell variables at the top so the script
    can be tweaked after generation without touching every line:
      PREFIX          output root (e.g. a transcode disk)
      PRESET          HandBrake preset name
      PRESET_ALT      preset for format-outlier episodes (only when present)
      HANDBRAKE_OPTS  array of extra flags (e.g. --preset-import-gui to load
                      GUI-saved presets) — edit it to apply to every rip
    suggested_filename is a relative path including season folders, so each
    rip is preceded by an idempotent mkdir -p for its season directory.

    Episodes whose video format differs from the majority (Avatar's 480i finale
    among a 1080p show) are emitted in a separate, commented block that rips
    with $PRESET_ALT, so the user can give that lower-quality source a different
    encode (e.g. a deinterlacing profile) without touching the rest."""
    eps = [r for r in records if r.get("kind") == "episode"]
    majority = _record_video_majority(records)
    outliers = [r for r in eps
                if majority and r.get("video_format")
                and r["video_format"] != majority]
    out_ids = {id(r) for r in outliers}
    conforming = [r for r in eps if id(r) not in out_ids]

    print("#!/usr/bin/env bash")
    print("set -euo pipefail")
    print()
    print(f"PREFIX={shlex.quote(str(output_prefix) if output_prefix else '.')}")
    print(f"PRESET={shlex.quote(preset)}")
    if outliers:
        odd = sorted({r["video_format"] for r in outliers})
        print(f"# {len(outliers)} episode(s) are {'/'.join(odd)} on the disc "
              f"(the rest are {majority}); they rip with PRESET_ALT below so you")
        print("# can choose a different encode for them. Defaults to PRESET.")
        print('PRESET_ALT="$PRESET"')
    print("# Extra HandBrakeCLI flags applied to every rip, e.g.:")
    print("#   HANDBRAKE_OPTS=(--preset-import-gui)")
    print("HANDBRAKE_OPTS=()")
    print()
    for r in conforming:
        _emit_rip_line(r, "$PRESET")
    if outliers:
        odd = sorted({r["video_format"] for r in outliers})
        print()
        print(f"# {'=' * 70}")
        print(f"# NON-CONFORMING VIDEO FORMAT ({'/'.join(odd)}) — "
              f"lower-quality source on the disc.")
        print("# Reviewed separately so you can apply a different encode profile")
        print("# (edit PRESET_ALT above, e.g. a deinterlace/upscale preset).")
        print(f"# {'=' * 70}")
        for r in sorted(outliers, key=lambda r: (r.get("season", 0),
                                                 r.get("episodes", [0]))):
            af = f", {r['audio_format']}" if r.get("audio_format") else ""
            print(f"# {os.path.basename(r['suggested_filename'])}  "
                  f"[{r['video_format']}{af}]")
            _emit_rip_line(r, "$PRESET_ALT")


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

