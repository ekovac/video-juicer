# bitter-episode-collator

Maps the titles on DVD/Blu-ray disc images to TMDB episodes, without reading
the multi-GB video payload. Design rationale in `IMPLEMENTATION_PLAN.md`.

```bash
export TMDB_API_KEY=...
python3 identify_episodes.py --tv-id 2418 discs/*.iso
```

Outputs `manifest.json` (one record per title: episode mapping, runtime
delta, confidence, suggested Plex-style filename) plus a human-readable
table. `--emit-rip-commands` prints ready-to-run HandBrakeCLI lines.

How it works: reads only disc metadata (DVD IFO via `lsdvd`, Blu-ray
`.mpls` playlists via `7z`), detects play-all titles by chapter-duration
matching, then aligns episode candidates against TMDB episode runtimes with
a monotonic DP alignment (Needleman–Wunsch with two-parter merge moves).
Peak RSS ≈ 140 MB regardless of image sizes.

`--episode-order dvd` matches against a TMDB *episode group* instead of the
default aired order — essential for shows whose discs reorder episodes
(Firefly) or fold specials into seasons. Accepts an alias (`dvd`, `digital`,
`absolute`, `production`, `story`, `tv`) or an explicit TMDB episode-group
id; manifest records then carry the aired numbering as an `aired`
cross-reference. Note the TMDB type enum: DVD order is type 3 (verified on
live data), digital is 4, production is 6.

`--verify` (or `--verify-all`) rips bounded windows from both ends of
low-confidence titles (`mencoder` for DVD, `ffmpeg` for Blu-ray), samples
frames, and reads on-screen title cards with a local Ollama VLM, fuzzy-
matched against the season's episode names.

Requires: `lsdvd`, `7z`; for `--verify` also `mencoder`, `ffmpeg`, and a
running Ollama with a vision model (default `qwen3-vl:2B`).

Tests: `python3 -m unittest test_identify_episodes -v` (no discs or network
needed).

## Hard-won implementation notes

- **TMDB runtimes are often broadcast-slot lengths** (30 min with ads) while
  discs carry the actual ~22 min episode. A per-season median ratio between
  disc and TMDB durations calibrates this; matching accepts whichever of
  raw/scaled runtime fits. Without it, four of seven Venture Bros. seasons
  fail entirely.
- **Thinking VLMs (qwen3-vl) need token headroom**: a tight `num_predict`
  gets exhausted by the reasoning phase and `content` comes back empty with
  `done_reason: length`. Budget generously and fall back to the `thinking`
  text, which usually quotes the transcription.
- **Title cards may be at the end** of the episode (all Venture Bros.
  seasons), sometimes as an `EPISODE: <name>` line inside the credits.
  Scan both ends, and let the end window run all the way to the last frame.
- **Sample frames at <=2 s.** A title card is only on screen ~2-4 s, so a
  coarse stride (4-8 s) phase-skips straight over it — producing confident
  "no title found" misses that look like the show simply doesn't display
  titles. (It cost a wrong conclusion that Venture Bros. S3 had no on-screen
  titles; they were there at ~21:40, missed by an 8 s back-window stride.)
  `extract_frames` defaults to 1.5 s.
- **Title-card position varies by show; scan a wide front window.** Venture
  Bros. cards are at the end; Star Trek: Enterprise captions the title after
  the opening sequence, but a variable-length cold open floats it several
  minutes in (observed 2:38 and 4:10 on adjacent episodes). `verify_title`
  scans the first 8 min plus the tail. For all-same-runtime shows (every
  Enterprise episode is ~44 min), runtime can't order episodes within a
  disc — OCR is the way to confirm the playlist/title sequence is right.
- Ollama has a known memory leak and may be OOM-killed mid-request; VLM
  calls retry with backoff and a generous timeout to ride out the daemon
  restart and model reload.
