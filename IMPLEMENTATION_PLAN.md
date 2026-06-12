# Implementation Plan: Disc-to-Episode Identification

Map each video title on a set of DVD/Blu-ray disc images to the TMDB episode it
contains, producing a machine-readable manifest suitable for driving a later
rip/rename step. Verified against the 14 Venture Bros. ISOs in `discs/`
(TMDB series 2418), but designed to generalize to any series and to Blu-ray.

## Findings from probing the actual discs (drives the design)

- `lsdvd` (libdvdread) reads ISOs directly — no mount needed — and touches only
  the IFO metadata files (a few MB), never the multi-GB VOB payload. It has a
  machine-readable mode (`lsdvd -Oy -c -a -s <iso>`) that emits a Python dict;
  the only wrinkle is a `libdvdread:` warning line printed *on stdout* before
  the dict, so the parser must slice from `'lsdvd = {'`.
- Disc layout pattern (S1D1): Title 1 is a **play-all** title (2:58:49, 8
  chapters); Titles 2–9 are the 8 individual episodes; Title 10 is a 24s extra.
  Crucially, the play-all's chapter durations match the individual episode
  titles to within **0.1 seconds** — this is the structural signal that
  identifies which titles are episodes *and* their on-disc order.
- Extras can be episode-length: S1D2 Title 8 is a 25:04 featurette and Title 16
  is a 21:23 extra, both confusable with ~22-minute episodes by runtime alone.
  But neither matches a play-all chapter, and both lack the subtitle streams
  (`Subpictures: 00`) that real episodes carry (3 subtitle tracks). Later discs
  (S7) differ in chapter counts and stream layout but follow the same
  play-all-plus-episodes pattern.
- TMDB runtimes are integer minutes (e.g. 22), so direct runtime matching needs
  ~±90s tolerance; some episodes may have `runtime: null`.

## Architecture

Single Python 3 script (stdlib + `requests` optional; `curl` via subprocess is
fine), four stages:

```
discs/*.iso ─► [1] Disc scan ─► titles per disc
                                      │
TMDB API    ─► [2] Metadata fetch ─► episode list per season
                                      │
               [3] Classify + align ─► (disc, title) → SxxEyy mapping
                                      │
               [4] Report ─► manifest.json + human-readable table
```

### Stage 1: Disc scanning (low-memory by construction)

1. **Format detection**: list the image's root directory without extraction
   (`7z l <iso>` or `pycdlib`/UDF): `VIDEO_TS/` → DVD, `BDMV/` → Blu-ray.
2. **DVD path**: run `lsdvd -Oy -c -a -s <iso>`, strip the warning prefix,
   `ast.literal_eval`-style parse (exec into a dict is what the format
   supports; sanitize by slicing from `lsdvd = {`). Capture per title:
   duration, chapter durations, audio stream count/langs, subtitle count,
   cells. Memory cost: IFO files only.
3. **Blu-ray path**: extract only `BDMV/PLAYLIST/*.mpls` (each a few KB) via
   `7z e -so` and parse MPLS in pure Python (fixed binary header; total
   duration = sum of PlayItem `out_time - in_time` in 45 kHz ticks; chapter
   marks from the PlaylistMark section). Dedupe playlists that reference the
   identical clip sequence (Blu-rays commonly carry duplicate/obfuscation
   playlists). Same downstream shape as the DVD path: a list of
   `{id, duration, chapters[], n_audio, n_sub}`.
4. Never open `*.VOB` / `*.m2ts`. Peak RSS stays in the tens of MB regardless
   of image size.

### Stage 2: TMDB metadata

- `GET /tv/{id}` for the season list, then `GET /tv/{id}/season/{n}` for
  episodes (name, episode_number, runtime). Auth via `TMDB_API_KEY` env var.
- Cache responses to `.tmdb_cache/{id}/...json` so reruns are offline.
- Runtime fallbacks for `runtime: null`: series `episode_run_time`, else the
  median runtime of the season.
- Also fetch season 0 (Specials) as a secondary match pool — disc extras are
  sometimes TMDB specials.

### Stage 3: Classification and alignment

**3a. Per-disc candidate selection (deliberately permissive):**

Not all boxsets have play-all titles, so no single structural heuristic can be
the decision-maker. Instead, build a *superset* of episode candidates per disc
and let the global alignment (3c) decide which ones are actually episodes —
unmatched candidates fall out as extras via gap moves. Candidate filtering
only removes titles that are clearly not episodes:

1. Drop exact-duplicate titles (same duration + cell/clip layout = DRM
   duplicate-title obfuscation).
2. Drop titles outside a generous duration band (0.5×–2.5× of the season's
   median TMDB runtime) — kills menus, trailers, and 30-second bumpers while
   keeping double-length episodes.
3. Detected play-all titles are excluded from candidates but kept as evidence.

Per-candidate **evidence scores** (each optional, combined into a prior that
biases the aligner rather than gating it):

- **Play-all chapter match** (when a play-all exists): a title with N≥2
  chapters whose duration ≈ the sum of N other titles, confirmed by
  multiset-matching its chapter durations against those titles (±2 s —
  observed agreement on these discs is 0.1 s). Matched titles get a strong
  episode prior and an authoritative on-disc order.
- **Stream-signature clustering**: episodes on a disc share audio/subtitle
  layout (here: 2 audio + 3 subs vs extras' 0 subs). The majority signature
  among in-band titles boosts members, penalizes outliers.
- **Structural similarity**: episodes tend to share chapter counts and VTS
  grouping (DVD) or clip layout (Blu-ray); extras are structurally ragged.

When no play-all exists, on-disc order falls back to title/playlist numbering,
which is the pressing order on every set observed in practice.

**3b. Disc ordering:** natural-sort the image filenames; additionally parse
`S(\d+)D(\d+)` / volume-label hints (`lsdvd` reports e.g.
`VENTURE_BROS_VOL_1_DISC_2`) as a *season hint*, used to narrow the alignment
pool when available but not required for correctness.

**3c. Sequence alignment:** concatenate detected episode titles across the
ordered discs (per season when hinted, else the whole series in season order)
and align against the TMDB episode sequence with a monotonic DP alignment
(Needleman–Wunsch style):

- match cost = `|title_seconds − tmdb_runtime*60|`, capped; tolerance ±90 s
  counts as a clean match (TMDB rounds to minutes); the 3a evidence scores
  enter as a per-candidate bonus/penalty on match moves
- gap on the disc side = unmatched candidate becomes an extra (cheap,
  cheaper still for candidates with weak evidence scores); gap on the TMDB
  side = missing episode (expensive — should be flagged)

Monotonicity encodes the one safe assumption (episodes appear on disc in
broadcast order) and resolves ambiguity between equal-runtime episodes, which
runtime alone cannot.

**3d. Validation:** every season's matched count must equal TMDB
`episode_count` (S1–S7 here: 13/13/13/16/8/8/10); per-match runtime delta
recorded; any delta >90 s or unmatched TMDB episode → loud warning, never a
silent guess. Unmatched disc titles are reported as extras (with a secondary
pass against season 0 specials by runtime).

### Stage 4: Output

`manifest.json`, one record per disc title:

```json
{
  "image": "discs/VENTURE_BROS_S1D1.iso",
  "title": 2,
  "kind": "episode",
  "season": 1, "episode": 1,
  "episode_name": "Dia de los Dangerous!",
  "title_seconds": 1330.2, "tmdb_seconds": 1320,
  "delta_seconds": 10.2, "confidence": "high",
  "suggested_filename": "The Venture Bros. - S01E01 - Dia de los Dangerous!.mkv"
}
```

plus `kind: "play_all" | "extra"` records, and a human-readable table on
stdout. Optionally `--emit-rip-commands` prints ready-to-run
`HandBrakeCLI --input <iso> --title <n> ...` lines (HandBrake reads title
numbers from ISOs directly).

## CLI

```
identify-episodes --tv-id 2418 [--season-map auto] [--out manifest.json]
                  [--emit-rip-commands] discs/*.iso
```

`TMDB_API_KEY` from the environment. `--tv-id` is the only required knob, so
the tool works for any series. A `--search "name"` convenience flag can resolve
the ID via TMDB search.

## Edge cases to handle explicitly

- **Double-length episodes**: one disc title may cover a TMDB two-parter (or
  one ~45 min TMDB episode spans two chapters). Allow a 1-title↔2-episode
  merge move in the DP when the durations sum correctly.
- **Null/wrong TMDB runtimes**: fall back as in Stage 2; alignment order still
  carries most of the signal.
- **CSS-encrypted DVDs**: these ISOs are decrypted; if libdvdread reports
  encryption and IFO reads fail, error out with a message about libdvdcss
  rather than producing garbage.
- **Blu-ray playlist obfuscation**: dedupe by clip-sequence; prefer playlists
  reachable from the index/movie object when ambiguous.
- **Specials interleaved on discs**: matched via the season-0 pool, emitted as
  `S00Exx`.
- **No play-all + episode-length extras**: the genuinely hard case — an extra
  that sits mid-sequence and matches an episode runtime can fool pure
  duration alignment. The aligner reports per-match confidence; for low-
  confidence discs an opt-in `--verify` mode does a bounded content check
  with real ground truth: decode a bounded window of each contested title at
  low resolution, sample frames, read on-screen text, and fuzzy-match it
  against the season's TMDB episode names (shows that display title cards
  make this decisive; absence of any name match is itself evidence the title
  is an extra). Scan **both ends** of the title, not just the start — some
  shows (Venture Bros. included) put the title card near the end of the
  episode. A/V-attribute comparison via ffprobe and candidate-vs-candidate
  audio comparison are limited to what they can actually prove — the latter
  only detects that two titles carry identical content (duplicate
  obfuscation), never which one is the episode. All verification reads are
  bounded; never the whole title.

  **Text-reading engine** (empirically tested on real title-card frames from
  these discs): classic OCR is inadequate — Tesseract read the episode title
  on only 1 of 8 frames, failing on stylized script fonts over textured
  backgrounds while happily reading the plain-font straplines. A local VLM
  via Ollama is the right tool: `qwen3-vl:2B` transcribed the failed card
  faithfully in ~5 s. Caution from the same test: larger models can
  "helpfully" translate non-English text (`qwen2.5vl:7b` rendered "Día de
  los Dangerous" as "Day of the Dangerous"), so the prompt must demand
  verbatim transcription and the fuzzy matcher must tolerate diacritics,
  case, and minor normalization. The closed candidate set (the season's
  episode names) is what makes noisy transcription workable — we match, we
  don't transcribe-perfectly. Scene-text OCR packages (RapidOCR/PaddleOCR/
  EasyOCR) are a possible non-LLM middle ground but unproven here; Tesseract
  remains useful only as a free instant pre-pass that occasionally
  short-circuits a VLM call.

## Testing

1. Run against all 14 ISOs; assert full coverage of 81 episodes across S1–S7
   with no >90 s deltas.
2. Unit-test the play-all detector and the DP aligner on synthetic title
   tables (no ISOs needed), including the S1D2 trap case (25-minute extra that
   must *not* match an episode) and a no-play-all variant of the same disc to
   prove the aligner gets the right answer from order + duration + stream
   signature alone.
3. MPLS parser unit-tested against a checked-in sample playlist file.
