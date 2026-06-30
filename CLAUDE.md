# CLAUDE.md

Working notes for `identify_episodes.py` — the hard-won lessons, decision model,
and gotchas accumulated across real series (Venture Bros, Star Trek: Enterprise,
Avatar, MOTU Revelation, Legend of Korra). README.md has user-facing usage and
granular VLM notes; this file is the *why* and the traps. Keep it current.

## What this is

Maps the playable titles on DVD/Blu-ray disc images (or BDMV/VIDEO_TS backup
dirs) to TMDB episodes. Metadata-first; escalates to local-VLM title-card OCR
only when metadata can't be trusted.

Layout (split from one 1600-line script):
- `discs.py` — data model (Title/Disc/Episode/Assignment), disc scanning
  (DVD/Blu-ray, MPLS, dedup, play-all clip ordering, HandBrake titles, hints),
  TMDB client. `log` and `run()` live here.
- `identify.py` — matching: metadata alignment + orderability, title-card OCR
  (verify_title/ocr_identify/probe/elimination), manifest output. Imports the
  model from `discs`.
- `identify_episodes.py` — CLI/`main`; re-exports both modules, so
  `import identify_episodes` and `python3 identify_episodes.py` are unchanged. Reads disc *metadata* only (IFO via
`lsdvd`, `.mpls` playlists via `7z`/parse) — never the multi-GB payload, so peak
RSS stays ~140 MB. Output: `manifest.json` (one record per title) + Plex-style
filenames + `HandBrakeCLI` rip commands.

Run: `python3 identify_episodes.py --tv-id <id> --auto --scratch-dir <disk> --out <m.json> <discs...>`
Tests: `python3 -m unittest test_identify_episodes` (no discs or network needed).
Needs `lsdvd`,`7z`; OCR also needs `ffmpeg`/`mencoder` + Ollama; `TMDB_API_KEY` in env.

## Core idea: trust metadata, escalate to OCR only when ambiguous

Three sources of canonical episode ORDER, cheapest first:
1. **DVD (lsdvd) title order** — reliable.
2. **A play-all that concatenates the episodes** — order/set independent of
   (possibly wrong) TMDB runtimes. Two detectors, by disc format:
   - **Blu-ray: `order_by_playall`** — a play-all's clips are the ordered union
     of the episode clips, so its clip order IS broadcast order, *exact even for
     same-runtime scrambled discs* (Avatar has one; MOTU doesn't). Sets
     `order_key` + `kind="play-all"`.
   - **DVD: `detect_play_all`** — ONE function, two strategies: (a) *chapter
     match* — the play-all's chapter marks segment into the other titles'
     durations (exact; handles multi-chapter-per-episode); (b) *duration-sum
     fallback* — the longest title whose runtime == the sum of its same-audio
     peers (catches discs whose chapters don't align to title boundaries, e.g.
     Broken Saints D3/D4 where chapter-match finds nothing). Inert on Blu-ray
     (n_audio is 0 there). `classify_disc` (metadata path) and `episode_candidates`
     (OCR path) both call it; it's why Broken Saints (tmdb 111718, TMDB says
     9 min for 9-49 min chapters) picks the right 8/7/8/10 episodes/disc where
     the length band picked 6 wrong ones. (This unified the old `detect_play_all`
     + `mark_dvd_playall` pair into one — same results: BS candidates unchanged,
     Venture Bros 81/81 unchanged.)
3. **Title-card OCR** — when neither of the above is available.

`assess_ordering` decides per disc whether the episode ORDER can be trusted:
- **DVD (lsdvd) title order is broadcast order WITHIN a disc** — but the aligner
  decides which episode each disc *starts* on, and with no season/disc hint AND
  same-runtime episodes it has no anchor: it can drop or shift a title and
  renumber the rest. Sonic SatAM (tmdb 2404) — ~all 22.7 min, no `S?D?` in the
  ISO names — dropped a 22.7-min episode that looked like its peers, shifting
  E08→ onward by one (OCR caught it). So DVD is trusted only with a disc hint
  (Venture Bros has `S?D?`) OR runtime-separable episodes; else → unverifiable,
  recommend OCR. (DVD was previously trusted unconditionally — the bug was false
  confidence, not a missing aligner constraint.)
- **Blu-ray `.mpls` playlist order is NOT broadcast order** — scrambled on Avatar
  and MOTU (verified by OCR). Trust it only via a play-all, multi-part names,
  or runtime-separability; else escalate to OCR.
- Caveat: a play-all fixes ORDER, not identity — on a disc that also has the
  dup/double/featurette mess (Avatar), it makes the order trustworthy but the
  candidate set may still need OCR; `--ocr-identify` is the override.
- Verifiable iff: DVD, OR a play-all whose chapters matched the titles, OR
  multi-part "(1)/(2)" names appear in sequence, OR runtimes are uniquely
  separable (min pairwise gap > tol) with small alignment deltas. Else
  → "unverifiable", recommend OCR.
- `--auto` runs metadata, then escalates to `--ocr-identify` ONLY when
  unverifiable AND `vlm_available` AND a 2-playlist `probe_card_presence`
  (skips first/last to dodge premiere/finale quirks) finds on-screen titles.
  No VLM or no titles → keep the metadata mapping, flagged. Never silently
  wrong; never wasted OCR. Korra was correctly judged verifiable → no OCR
  needed; Avatar/MOTU unverifiable → OCR.
- **OCR verification tiers** (post-metadata; all share `verify_assignment`,
  which OCRs a title card and returns confirmed / overridden / no-card):
  `--verify` OCRs only low-confidence matches; `--verify-all` OCRs every match;
  **`--spot-check N`** (default 1) OCRs ~N matched episodes *per disc* at random
  (seeded via `--spot-check-seed`, stratified so every disc is sampled) and, if
  any sampled episode's card DISAGREES with the alignment, fully verifies that
  disc. It's the cheap middle ground — a stratified spot-check catches a
  dropped/shifted title that renumbered a whole disc (Sonic SatAM) in ~1 OCR
  call/disc instead of all 26.

## Failure modes and the defense for each

- **Scrambled BD playlist order** → `--ocr-identify`: identity comes from the
  on-screen title, not the position.
- **Same-runtime episodes** → metadata can't order them → unverifiable → OCR.
- **Two authoring versions per episode** (body alone vs body+logo/recap) →
  `dedup_subset_playlists`: drop a playlist whose clips are a strict subset of a
  similar-length one (1.5x guard stops a play-all swallowing episodes).
- **Combined two-parter "double" playlists** (one playlist = 2 episodes) →
  claim N and N+1; `resolve_assignment_collisions` (OCR path only — the
  metadata DP is monotonic and never double-claims) keeps a double only when it
  carries an *unclaimed* episode, so a finale present only inside an E19+E20
  double survives. When a kept double *also* contains an episode a same-disc
  standalone single already claimed (Avatar: single E12 + the E12+E13 double,
  where E13 lives only in the double), the redundant single is demoted to an
  extra — else the rip plan emits E12 twice and Plex sees S02E12 vs S02E12-E13
  overlap. The merged file becomes the source for both; the bundled twin (E12)
  has no standalone, which is expected for combined-on-disc two-parters.
  Cross-disc dups are a separate pass (`resolve_cross_disc`).
- **Featurette that runs episode-length AND names an episode on screen**
  (Avatar "Inside the Korean Animation Studios" said "Chapter 14, The
  Fortuneteller") → false OCR match at 1.0. **Length cannot catch this** — it is
  a valid episode length (the featurette was 6s from a real 26-min episode, even
  by precise per-episode TMDB runtime). `resolve_cross_disc` catches it: when an
  episode is claimed on two discs, keep the disc with the contiguous run, demote
  the impostor. Position/structure discriminates where length can't.
- **Title-card-less episodes** (a premiere whose card is the series-logo
  sequence — MOTU E01/E06) → `recover_by_elimination`: a disc's lone unmatched
  candidate is pinned to the one still-missing episode adjacent to its matched
  run; constraint-propagate so the constrained disc resolves first.
- **Candidate length filter** is `valid_episode_lengths`, NOT a wide band: TMDB
  per-episode runtimes ∪ multiples of the median (robust to a wrong/null TMDB
  runtime for a long episode — it still lands on k*median) ∪ consecutive sums.
  `episode_candidates` overrides it with the `detect_play_all` set when present
  (the play-all is used *because* runtimes are untrusted, so the length band is
  moot), and marks those titles `kind="episode-candidate"`. For the same reason
  the OCR two-parter heuristic (claim N+1 when a title is ~2x the runtime) is
  **skipped for `episode-candidate` titles** — with wrong runtimes every single
  looked like a 2x double, so it merged E01+E02 and the collision pass then
  demoted the correct standalone.
- **Known limitation (Broken Saints back discs):** a feature-length finale split
  across many DVD titles (E24 "Truth" = ~7 titles, only fragments carrying a
  card) does not map cleanly to one title; sub-segments open with epigraph
  quotes, and short common-word titles (Truth/Inside/Signals/Trinity) match any
  quote frame containing the word. Also a VLM *reasoning dump* can still
  fuzzy-match (~0.80) via the SequenceMatcher branch — the verbatim branch is
  guarded but the fuzzy branch is not. Unresolved; fixing it is delicate (on
  Broken Saints a reasoning-dump match is currently load-bearing for E19).

  **Reasoning-leak fix (implemented):** the leak is a thinking-model truncation,
  not a content problem. Experiment (2026-06): `think:false` is a no-op for
  qwen3-vl on Ollama 0.30.3 (still thinks); leaks are rare (1/60 frames in the
  leaky window) and happen only when thinking overruns `num_predict`
  (`done_reason=="length"` → empty `content`). `num_predict` is a CAP not a
  target, so a frame that finishes early stops regardless — raising it is
  near-free. Fix in `ollama_chat`: (1) `VLM_NUM_PREDICT=8192`; (2) on empty
  `content` with `done_reason=="length"` return "" instead of the partial
  `thinking` — `done_reason` is the deterministic signal, no classifier LLM
  needed. Caveat still open: cleanly rejecting leaks turns the load-bearing E19
  reasoning match into a leftover (its frame is a quote, not a card), handing
  E19 to the clean "Signals" extra — the position-vs-content question is
  unresolved, so a Broken Saints re-run may shift the finale-disc mapping.

## VLM / OCR specifics

- **Model: `qwen3-vl:2B`.** Benchmarked alternatives and rejected them:
  moondream *describes* images instead of transcribing (unusable); `glm-ocr`
  equally accurate but ~3x slower; `qwen2.5vl` translates non-English titles.
- qwen3-vl is a *thinking* model: it spends tokens reasoning before the final
  answer, so `num_predict` must cover both. `VLM_NUM_PREDICT=8192` (it's a cap,
  not a target — frames that finish early stop regardless, so a big cap is
  near-free and lets rare complex frames finish). If it still truncates
  (`done_reason == "length"`, empty `content`), `ollama_chat` returns "" — it
  does NOT fall back to the partial `thinking` text, which is chain-of-thought,
  not a transcription, and fuzzy-matches confident-wrong titles. A missing read
  is recoverable by position/elimination; a wrong one corrupts the mapping.
  (`think:false` is a no-op for this model on Ollama 0.30.3.)
- **Sample frames at ≤2 s** — cards are on screen ~2-4 s; a 4-8 s stride
  phase-skips them and looks like "show has no titles". `extract_frames`=1.5 s.
- **Card location varies by show**: VB at the end (~21:40, stylized script —
  Tesseract fails, VLM needed); Enterprise mid, cold-open-delayed (140-374 s,
  premieres later); Avatar/Korra/MOTU early (~15-80 s). `verify_title` learns
  the per-disc location (adaptive anchor) so only the first episode pays full
  cost; NO-CARD widens the front window as a fallback.
- `fuzzy_best`: word-boundary substring; a distinctive multi-word/long title is
  conclusive; a short single-word title (Dawn, Jet, ORB) matches only if the
  frame isn't dominated by credit/reasoning markers (`_INCIDENTAL_MARKERS`),
  else by coverage. `canon_parts` maps "Part One/I/1" ↔ TMDB "(1)".
  - Markers split into `_CREDIT_MARKERS` and `_REASONING_MARKERS`. A distinctive
    verbatim hit beside a *credit* line is still 1.0 (real card + credit), but a
    title named inside a *reasoning* dump never wins on presence alone — it
    scores by coverage (tiny in a long chain-of-thought) and is rejected. This
    fires when a thinking model returns no `content` and we fall back to
    `thinking`: on a no-title-card show (The Magicians) the VLM's reasoning
    ("Got it, let's look at the image…") once named a title and scored 1.0,
    overriding correct metadata (E01→E04). Cards-less shows must yield no OCR
    match, not a confident wrong one.

## TMDB notes

- Runtimes are sometimes the *broadcast slot* (30 min) not the actual episode
  (22 min) → per-season scale calibration in `align`.
- Alt ordering via episode groups: `--episode-order dvd`. **DVD order is TMDB
  type 3** (digital=4, production=6) — verified on live data; a prior session
  wrongly believed DVD was type 4.
- A 2-part pilot/finale may be one TMDB entry (Enterprise "Broken Bow" = S01E01,
  86 min, with no E02 — numbering jumps E01→E03).

## Output / rip workflow

- The manifest **`title` field is the number to pass to `HandBrakeCLI -t`.**
  DVD: the lsdvd title. **Blu-ray: HandBrake's own title index** (from a per-disc
  HandBrake scan) — NOT the `.mpls` id and NOT a player's HDMV title-object
  number. All three differ (one Enterprise episode was .mpls 1 = HandBrake t2 =
  VLC title 19, because index.bdmv defined 78 title objects).
- **`suggested_filename` is a relative path in the Plex/Jellyfin layout**:
  `<Show (Year) {tmdb-ID}>/<Season NN>/<Show (Year)> - SxxEyy - Names.mkv`. The
  `{tmdb-ID}` match hint goes on the show *folder* only (both servers read it
  there) — the file prefix stays clean. Season 0 → the `Specials` folder; a
  multi-episode title uses `SxxEyy-Ezz` (the hyphen range both servers parse —
  NOT a bare `E01E02` run). Each path component is sanitised independently; the
  `/`s are real separators. Year comes from TMDB `first_air_date`.
- **`emit_rip_commands` emits a runnable bash script**, not bare lines: a
  `#!/usr/bin/env bash` + `set -euo pipefail` header that hoists the common
  knobs into shell variables — `PREFIX` (output root), `PRESET`, and
  `HANDBRAKE_OPTS=()` (an array of extra flags, e.g. `--preset-import-gui` to
  load GUI-saved presets) — so the script is editable after generation without
  touching every line. Each rip is a `mkdir -p "$PREFIX/<season>"` then
  `HandBrakeCLI "${HANDBRAKE_OPTS[@]}" -i <img> -t N --preset "$PRESET" -o
  "$PREFIX/<path>"`. Image paths are `shlex.quote`d (disc dirs have spaces).
- `--output-prefix DIR` → sets the script's `PREFIX` (e.g. a target transcode
  disk); the whole Plex/Jellyfin tree is built under it. Default `PREFIX=.`.
- `--from-manifest FILE` → emit rip commands from a saved manifest, no scanning.
- `--merge` → merge a run into the existing `--out`, replacing only the discs
  processed this run.
- Every episode record has `identified_by`: `runtime-align` / `title-card` /
  `elimination`, plus `confidence` and `verified_by_titlecard`.
- **Video-format outlier warning.** Each record carries `video_format`
  (`1080p`/`480i`/…) and `audio_format`, read from the Blu-ray MPLS STN stream
  table (`_stn_formats` — pure metadata, no payload; end-anchored so a bad
  layout fails soft to None; audio is best-effort, video is exact). At report
  time `format_outliers` flags episodes whose video format differs from the
  run's majority — they matched correctly but point to an inferior source. The
  Avatar BD authors 11 of its 61 episodes (the Sozin's Comet finale, Day of
  Black Sun, a few S1/S2) at 480i SD among a 1080p show; the warning surfaces
  all of them. DVD titles have `video_format=None` (not parsed) → never flagged.
  - The **rip script** splits on this too: the conforming episodes rip with
    `$PRESET`, and the format-outliers go in a separate commented block that rips
    with `$PRESET_ALT` (defaults to `$PRESET`), so the user can give the SD
    source a different encode (e.g. a deinterlace/upscale preset) without
    touching the rest. Old manifests lacking `video_format` → one block, as
    before.

## Operational gotchas (Ollama OOM + orchestration)

- **Ollama leaks memory; the OOM-killer fires on long runs.** VLM calls retry
  with backoff to ride out the daemon restart + model reload. Reduce risk:
  `--scratch-dir` on a real disk (NOT `/tmp` tmpfs — rips there add memory
  pressure); the adaptive anchor cuts calls ~8x after the first episode.
- **For OCR runs, go per-disc with `--merge`** so each disc is written on
  completion — an OOM only loses the in-progress disc; re-run just that one.
- **BUT `--merge`/per-disc is OCR-ONLY. The metadata aligner needs all of a
  season's discs together.** Aligning a subset of a season against the full
  episode pool makes each disc independently claim overlapping ranges (hit on
  Korra: per-disc runs gave 7 unique episodes/season instead of 12-14). Run all
  discs (or at least a whole season) in one metadata invocation.
- Concurrent runs on the same `--out` race; `write_manifest` holds a file lock +
  atomic rename. Still, don't launch two runs at once — an OOM'd run can linger
  and a late write can clobber a good one (this corrupted an Avatar manifest).
- Manifests are local artifacts (gitignored); only the code/tests/docs are
  committed.

## Claude self-inflicted workflow traps (don't repeat)

- `pgrep -f identify_episodes` matches its **own** command line → false "already
  running". Use a specific pattern (e.g. `identify_episodes.py`).
- Inner `&`/`nohup` backgrounding inside a Bash tool call gets orphaned or torn
  down at the foreground command's exit. Use the tool's `run_in_background`.
- Running a script from `/tmp` puts `/tmp` (not the repo) on `sys.path`; set
  `sys.path.insert(0, repo)` in throwaway scripts, or run from the repo.
- `Title(...)` requires `chapters`; construct with `chapters=[]` in test scripts.
