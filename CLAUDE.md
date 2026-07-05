# CLAUDE.md

Working notes for video-juicer — the hard-won lessons, decision model, and
gotchas accumulated across real series (Venture Bros, Star Trek: Enterprise,
Avatar, MOTU Revelation, Legend of Korra). README.md has user-facing usage;
DESIGN.md has the architecture + SQLite schema; **this file is the *why* behind
the heuristics and their traps.** Keep it current.

## What this is

Maps the playable titles on DVD/Blu-ray disc images (or BDMV/VIDEO_TS backup
dirs) to TMDB episodes. Reads disc *metadata* only (IFO via `lsdvd`, `.mpls`
playlists via `7z`/parse) — never the multi-GB payload, so peak RSS stays
~140 MB.

## Architecture: Picard-style, evidence-first (see DESIGN.md)

The tool is modelled on MusicBrainz Picard. Heuristics are **on-demand evidence
producers**, not one committed pipeline; a human or agent adjudicates. State is a
SQLite **project file**, three layers: facts (`disc`/`title`/`episode`),
**evidence** (one upserted row per (title, category) — categories are the
sources `runtime-align`/`stream-signature`/`title-card-ocr`/`synopsis`/
`elimination`), and
**assignment** (the thin adjudicated answer). Conflict is *computed* by `gaps`,
not stored. Full schema + rationale: DESIGN.md.

**IMPORTANT for edits:** the heuristic sections below (core idea, failure modes,
VLM/OCR, TMDB, output) describe **library functions that the Picard rewrite left
unchanged** — `align`, `detect_play_all`, `assess_ordering`, `verify_title`,
`recover_by_elimination`, `suggested_filename`, `emit_rip_commands`, etc. What
changed is *how they're invoked* (verbs, each writing evidence) and *where output
lives* (state DB, then `vj export`). Where an old note mentions a flag
(`--auto`, `--ocr-identify`, `--verify`, `--spot-check`, `--merge`), read it as
describing the heuristic's *behaviour*, now reached through the verbs below.

Layout:
- `discs.py` — data model (Title/Disc/Episode/Assignment), disc scanning
  (DVD/Blu-ray, MPLS, dedup, play-all clip ordering, HandBrake titles, hints),
  TMDB client. `log` and `run()` live here.
- `identify.py` — the heuristics: metadata alignment + orderability, title-card
  OCR (verify_title/ocr_identify/probe/elimination), rip-command/filename output.
- `synopsis.py` — the dialogue→synopsis judge (a second OCR-free identifier):
  whisper-transcribe sampled audio windows, judge against episode synopses.
- `wiki.py` — read episode plot summaries from a local Wikipedia **multistream**
  dump (offline synopsis source; see the synopsis note below).
- `state.py` — SQLite data layer (schema, row⇄dataclass mappers, evidence upsert,
  frame BLOBs, assignment ops, `transcript` cache, `background` packaging hints).
- `compute.py` — `vj run <heuristic>`: wraps the heuristics as evidence producers.
- `review.py` — inspect (`status`/`gaps`/`show`), `resolve` (evidence→proposals),
  adjudicate (`assign`/`confirm`/`reject`). Conflict computed here.
- `export.py` — manifest + rip script from the adjudicated assignments.
- `vj.py` — the CLI entry point wiring all verbs.

Run (per-verb; see README for the full flow): `vj init db --tmdb-id <id>` →
`vj scan db <discs>` → (optional `vj hint disc db --disc … --season N --episodes
1-4` to feed box-packaging into align) → `vj run align db` → `vj run streams db`
(optional; flags episode-length extras by audio/subtitle layout) → `vj run ocr db
--disc N` (or, for a no-title-card show, `vj enrich wikipedia db --snapshot …
--index … --page …` then `vj run synopsis db`) → `vj resolve db` →
review/adjudicate → `vj export db --manifest … --rip-script …`.
Tests: `python3 -m unittest test_identify_episodes test_vj` (no discs/network).
Needs `lsdvd`,`7z`; OCR also needs `ffmpeg`/`mencoder` + Ollama; `TMDB_API_KEY`
in env for `init`.

## Disc-probing findings (drive the scan design)

- **lsdvd reads ISOs directly** (libdvdread, no mount) and touches only the IFO
  metadata (a few MB), never the VOB payload. Machine-readable mode
  `lsdvd -Oy -c -a -s <iso>` emits a Python dict, but prints a `libdvdread:`
  warning line *on stdout first* — the parser slices from `lsdvd = {`.
- **Blu-ray**: extract only `BDMV/PLAYLIST/*.mpls` (a few KB each) via `7z`; parse
  MPLS in pure Python — total duration = Σ PlayItem `out_time - in_time` in
  45 kHz ticks; chapter marks from PlaylistMark. Dedupe playlists referencing the
  identical clip sequence (Blu-rays carry duplicate/obfuscation playlists) —
  keeping the richest/lossless copy, see `dedup_identical_clips` in failure modes.
- **BD audio/subtitle stream counts come from the HandBrake scan, not the MPLS.**
  The STN table is too fragile to parse for counts (we only end-anchor it for the
  `video_format`/`audio_format` codes), so `n_audio`/`n_sub` are 0 out of
  `parse_mpls` and filled from `handbrake_scan` — the SAME scan already run for
  the rip title index (`AudioList`/`SubtitleList` lengths, plus a lossless flag).
  No HandBrake → counts stay 0 and the stream signals go quiet (fail-soft). DVD
  counts come from lsdvd directly.
- **Extras can be episode-length AND fool runtime alignment.** On Venture Bros
  S1D2, Title 8 is a 25:04 featurette and Title 16 a 21:23 extra, both confusable
  with ~22-min episodes; neither matches a play-all chapter, and (on these discs)
  extras lack the subtitle streams real episodes carry — a secondary discriminator
  behind the play-all/stream-signature logic. This is the canonical unit-test trap.
  The stream discriminator lives in two places off ONE helper
  (`identify.stream_signature`: the disc's majority `(n_audio, n_sub)` layout +
  each title's episode/extra verdict): (1) a ±0.4 candidate-score nudge inside
  `classify_disc` (unchanged), and (2) a first-class **`stream-signature`**
  evidence category via `vj run streams` — a per-title row (episode_id NULL; it
  attests episode-*hood*, not identity) that `gaps` folds in: an episode-length
  title proposed as an episode but with an extra-like layout is surfaced for
  review, and an unidentified episode-length leftover's dropped/-episode anomaly
  is strengthened or softened by whether its layout matches the disc's episodes.
  `run streams` excludes play-alls (legitimately richer — commentary track) and
  concatenations from the clustering, matching `classify_disc`; the concatenation
  length cut (>1.6×median) is taken over titles that CARRY counts, so a disc with
  many count-less short extra playlists (TNG: six 18-min menu loops per disc)
  can't drag the median down and exclude the real episodes.
  - **A title is flagged only when STRICTLY POORER than the majority (≤ audio and
    ≤ sub, < one), never merely different.** Real episodes vary in richness: TNG
    Blu-ray authors most S1 episodes 8A/11S but Farpoint (E01) and E14 at 8A/12S
    (a bonus subtitle) — a *richer* layout is an episode with extra tracks, not an
    extra, so it must not be flagged. Only a genuinely lean title (TNG S1D1 pl 4,
    a 1A/11S single-audio extra among 8A episodes) is.
  - **Stream layout does NOT separate a duplicate authoring from the episode it
    duplicates when both are full-quality.** TNG S1D4 pl 40 (the distinct-clip
    duplicate that gives the aligner slack — see the play-all notes) is 8A/12S,
    identical to the real E14 it copies, so streams can't tell them apart; the
    stream signal is orthogonal to that ordering bug, which stays on
    assess_ordering + confirmed-anchor re-align + OCR. (Contrast the Avatar case,
    where the duplicate was a *stripped* 1A/0S copy — there streams DO separate
    them, and `dedup_identical_clips` keeps the master.)
- **TMDB runtimes are integer minutes** (some `null`); direct runtime matching
  needs ~±90 s tolerance, hence the calibration + `valid_episode_lengths` band.
- **Volume labels are unreliable; the file basename is canonical.** Many box
  sets have blank, wrong, or *identical* filesystem volume labels across every
  disc (VB S2 labels both discs `VENTURE_BROS_SEASON_2`). So display and disc
  identity use the image filename / backup-dir basename (`state.disc_name`,
  `.iso` stripped), never `disc.label`. Hint parsing already agrees — it reads
  `disc.path.name` before `disc.label`. The volume label is kept only as
  secondary info in JSON.
- **CSS-encrypted DVDs**: these ISOs are expected pre-decrypted; if libdvdread
  reports encryption and IFO reads fail, error out pointing at libdvdcss rather
  than emitting garbage.

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
- **A play-all only vouches for the episodes it CONCATENATES.** On Blu-ray, if a
  disc has clip-bearing episode titles NOT covered by the play-all (a feature-
  length pilot authored separately — Star Trek: TNG "Encounter at Farpoint",
  clips 00000/00084 vs the play-all's 00001/00002/…), the play-all doesn't order
  them and the monotonic aligner can drop/shift a title around them (TNG S1 went
  +1 across the whole season, a title silently dropped). So `assess_ordering`
  treats a play-all as full corroboration only when it covers every clip-bearing
  episode title; a partial play-all → unverifiable → surfaced (below), not
  auto-trusted. (Clipless synthetic/DVD play-alls can't be checked → trusted as
  before.) Do NOT "fix" this by ordering on clip id — clip numbering is not
  broadcast order in general (the whole reason `order_by_playall` exists).
- **Do NOT try to detect the "full" play-all statically — it often isn't a
  playlist at all.** Investigated on the real TNG S1D1 disc (2026-07): there is
  NO 3-episode play-all `.mpls`. The six longest playlists are duplicate copies
  of Farpoint itself (91.4 min, clips 00000/00084); `mpls 0` (91.1 min, clips
  00001/00002/00064) is a genuine but PARTIAL play-all of only E02+E03. The
  disc's on-screen "Play All" (which does start with Farpoint) is implemented in
  the **navigation layer** (HDMV movie objects / menu button commands chaining
  playlists), invisible to an `.mpls` parser. So we aren't picking the wrong
  playlist — the full order simply isn't in one. And no static signal cleanly
  separates a **feature-length single episode outside the play-all** (Farpoint,
  ~2x, own clips) from a **whole-disc play-all authored as one monolithic clip**
  (Avatar B1D1 `mpls 1000` = 226 min, single clip `01010`, also disjoint) or a
  **combined two-parter** (~2x, own clips): duration and clip-disjointness look
  identical. Two attempts to fix it — a duration-band "stray" check and a
  clip-coverage check — both **regressed Avatar (57→55/61, scrambled) and fired
  on every Avatar/Korra disc** (their book-play-alls are 150-226 min single
  clips) and were reverted. Leave the play-all detector as-is; TNG is handled by
  the surfacing below + the confirmed-anchor re-align (`run align` pins confirmed
  titles) + OCR. Blu-ray authoring is a zoo — deliberate obfuscation and awkward
  mastering are indistinguishable and often both on one disc; don't chase a
  static heuristic here, it will re-regress the play-all discs.
- **How untrustworthy order surfaces (evidence-first, not auto):** `gaps`/
  `status` report per-disc "⚠ order unverified" from `assess_ordering`, and
  `gaps` flags an unclaimed *episode-length* candidate as a `‼` anomaly
  ("likely a dropped/shifted episode") — the tell of exactly this failure,
  promoted from a buried leftover to the top of the worklist. The human/agent
  then chooses to `run ocr` on that disc, whose title-card evidence conflicts
  with runtime-align → a normal `gaps` conflict to adjudicate. `vj auto` reads
  the same signals and acts; nothing is auto-decided in the manual flow.
- `--auto` runs metadata, then escalates to `--ocr-identify` ONLY when
  unverifiable AND `vlm_available` AND a 2-playlist `probe_card_presence`
  (skips first/last to dodge premiere/finale quirks) finds on-screen titles.
  No VLM or no titles → keep the metadata mapping, flagged. Never silently
  wrong; never wasted OCR. Korra was correctly judged verifiable → no OCR
  needed; Avatar/MOTU unverifiable → OCR.
- **OCR verification tiers** (post-metadata; all share `verify_assignment`,
  which OCRs a title card and returns confirmed / overridden / no-card):
  `--verify` OCRs only low-confidence matches; `--verify-all` OCRs every match;
  **`--spot-check`** OCRs each disc's *first and last* matched episode and
  **escalates to a full verify of that disc UNLESS both boundaries positively
  confirm** the alignment. The failure it targets is a dropped/shifted title
  that renumbers a whole contiguous run (Sonic SatAM) — so the *boundary*
  episodes are where it shows up. First+last beats random-N: deterministic,
  brackets the run, two shots at a readable card. The escalate-unless-confirmed
  policy matters: a *disagreement* is an error, but a *no-readable-card*
  boundary also escalates — Sonic disc 3's boundary episodes truncate while its
  middle reads, so "escalate only on disagree" would have silently kept it
  wrong. Cost: shows whose premieres/finales have no title card escalate every
  disc (acceptable — you'd want them verified). Caveat it still does NOT fix: a
  *dropped* title is a leftover, not an assignment, so it stays missing (Sonic
  E08 — only `recover_by_elimination` in the OCR path recovers those).

## Failure modes and the defense for each

- **Scrambled BD playlist order** → `--ocr-identify`: identity comes from the
  on-screen title, not the position.
- **Same-runtime episodes** → metadata can't order them → unverifiable → OCR.
- **Two authoring versions per episode** (body alone vs body+logo/recap) →
  `dedup_subset_playlists`: drop a playlist whose clips are a strict subset of a
  similar-length one (1.5x guard stops a play-all swallowing episodes).
- **Two playlists over the *identical* clips** (a lossless/multi-language master
  and a stripped stereo copy — Avatar B1D3 authored every episode as both pl
  60x = DTS-HD MA 4A/1S and pl 25x = AC3-stereo 1A/0S, byte-identical clips and
  in/out ticks) → `dedup_identical_clips`. The old scan dropped exact-clip
  duplicates by *filename order*, silently keeping the stripped version; now the
  BD scan runs HandBrake *before* dedup (it already runs it for the rip index)
  and keeps the RICHEST of each identical-clip group: most audio+subtitle
  streams, then lossless-audio-present (`_has_lossless` on the HB Description),
  then lowest id (= old first-seen, so no-HandBrake/DVD behaviour is unchanged).
  So the rip plan sources the lossless master, and OCR/align are unaffected
  (same clips, duration, and play-all order regardless of which id is kept). The
  stripped twins are commonly commentary or stereo-compat versions.
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

- **OCR engine is a hybrid, `--ocr-engine auto` (default): Tesseract first,
  VLM fallback.** A title card is a plain OCR task, not a reasoning one — so
  `tesseract_text` (pytesseract, ~60-95 ms/frame) reads plain block cards
  (Sonic, Enterprise, Avatar) outright and returns "" *instantly* on text-less
  scene frames. Only when Tesseract finds no accept-match in a whole window does
  `verify_title` fall back to the VLM, **over the whole window** (anchor-sorted,
  so a known anchor still hits the card first). On Sonic SatAM this cut a
  multi-hour VLM-only run to ~21 min (season 1 near-instant) with no accuracy
  loss on plain cards; stylized-script shows (Venture Bros) still read via the
  VLM fallback. `--ocr-engine {tesseract,vlm}` force either.
  - **Do NOT gate the VLM fallback on "Tesseract saw text here."** Tried it (to
    skip truncation-bait scene frames) — it broke VB: a stylized card Tesseract
    can't read returns "" just like a scene frame, so gating skips the card
    frame and misses the episode (VB got 0.56/wrong vs 1.00 correct un-gated).
    Tesseract-empty can't distinguish "no text" from "text I can't read"; only a
    text-*region* detector could, which is the real follow-up for wart 1.
  - **Wart 1 (fixed): scene frames flooding the VLM pass.** The VLM fallback ran
    on every frame of a window; on a long title with no card in the primary band
    it ground through hundreds (a 50-min VB special once took 44 min). Now
    `verify_title` gates the VLM pass with an **OCR-free text-region detector**
    (`text_region.py`): only frames a detector scores as bearing text reach the
    VLM. It's **recall-first**, **fails open** (no detector → no pruning), and
    **guarded** (would-prune-whole-window → keep the window). The seam
    (`frame_has_text`) is detector-agnostic; two backends, measured on a
    VB(stylized)+Enterprise(plain) card corpus vs scene negatives (both hold
    100% recall on real episode cards):
    - **PaddleOCR PP-OCRv3 det via RapidOCR/onnx (default)** — Apache-2.0, model
      bundled with the pip package, **~92% scene rejection** (88% on the pathological
      title-205 window), ~170 ms/frame.
    - **EAST (cv2.dnn, fallback)** — ~70% rejection, ~50 ms/frame; its ~96 MB
      model isn't shipped and is license-murky (GPL-3.0 upstream / ICDAR data),
      so it's fallback-only (`VJ_EAST_MODEL`). Force a backend with
      `VJ_TEXT_DETECTOR=paddle|east`.

    Classical detectors were tried and rejected first (morphological gate ~60%
    recall on stylized/textured cards; MSER ~30% scene rejection).
  - **Wart 1 caveat — scene-art title cards.** EAST detects *text*; a show whose
    title is painted INTO the scene art (Adventure Time — hand-lettered, more
    stylized than VB) may not read as text, so the gate could prune the real
    card. Default is on; disable per-run with `vj run ocr --no-text-filter`
    (also on `vj auto`). The escalation probe (`probe_card_presence`) always
    runs unfiltered so a false "no text" never wrongly skips OCR.
  - **TODO (wart 2): OCR overrides correct metadata on part-number two-parters.**
    "Blast to the Past (1)/(2)" show an identical card with no "(1)/(2)", so OCR
    matches both to part 1 and *overrode* the alignment's correct E05→part-2,
    losing it (Sonic S02E05). When the matched episode is a `(N)`-part title and
    the card lacks the part number, OCR should NOT override the alignment (the
    monotonic order distinguishes the parts; the card can't).
- **Model (VLM fallback): `qwen3-vl:2B`.** Benchmarked alternatives and rejected
  them: moondream *describes* images instead of transcribing (unusable);
  `glm-ocr` equally accurate but ~3x slower; `qwen2.5vl` translates titles.
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
- **Card location varies by show** and there are NO windows any more: VB at the
  end (~21:40, stylized — VLM needed); Enterprise mid, cold-open-delayed
  (140-374 s); TNG ~360 s (past the old 280 s front window — a concrete miss the
  windowless design fixed); Avatar/Korra/MOTU early (~15-80 s). `verify_title` is
  **windowless**: it rips+extracts the WHOLE title once (SD decodes ~2 ms/frame,
  so even a 50-min title is ~10 s), Tesseract-sweeps every frame, then gate+VLMs
  the survivors (budget-capped) — processing frames nearest a learned `anchor`
  first (else nearest either END), so a card-bearing title early-exits and only
  the first episode on a disc pays discovery cost. This replaced the old
  front/tail/widen windows, which could miss a card outside their bands (TNG).
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

## Synopsis judge + Wikipedia enrichment (the OCR-free identifier)

`vj run synopsis` identifies a title by *content*, not position — the escalation
for a **no-title-card, order-unverified** show (Blu-ray, same-runtime episodes;
OCR has nothing to read). It gets the episode's dialogue as text and a 2-stage
LLM judge matches it against each candidate's plot synopsis.

- **Dialogue comes from SUBTITLES first, whisper last (`--transcript-source`,
  default auto). The fallback chain, each tier self-selecting by disc format:**
  1. **Closed captions (`subtitle_transcript`, DVD)** — the MPEG-2 video carries
     EIA-608 CC as TEXT (no OCR): stream-copy the title (`mencoder -ovc copy
     -nosound`, preserves the video user-data), `ffmpeg movie=…[out+subcc]` emits
     an SRT, `srt_to_text` flattens it. Exact words, whole episode, ~4 s. No CC →
     "" → next tier. (Extras carry no CC — a free episode/extra tell.)
  2. **Bitmap-subtitle OCR (`subtitle_ocr_transcript`, DVD VOBSUB + BD PGS)** — for
     CC-less DVDs (VB S3) and ALL Blu-ray (PGS, no CC ever). ffmpeg renders ONLY
     the subtitle stream onto a black canvas (video never decoded, so fast),
     `mpdecimate` keeps one frame per distinct caption, and **PP-OCR (RapidOCR,
     the text-region detector's engine — `text_region.ocr_text`) reads each**.
     Blu-ray reads via `bluray:` directly; DVD has no ffmpeg protocol so mplayer
     `-dumpstream` first dumps the title's program stream (carries the subpicture).
     **Use RapidOCR, NOT tesseract:** on low-res 480p VOBSUB tesseract garbles it
     ("GIRLFRIEND"→"GIREERIEND", words mashed) while RapidOCR is near-exact; 1080p
     PGS is clean on both. Cost is the OCR loop: ~335 ms/frame CPU × ~385
     frames/episode ≈ **~2 min/episode** (vs whisper ~4-5 min, and far cleaner) —
     OCR is parallelized over a persistent spawn pool (`text_region.ocr_texts`,
     cpu_count-2 single-thread workers, angle classifier off, ≤960px downscale);
     Avatar B1D1 measured ~1:50/episode end-to-end.
     **The render's `color` source MUST be duration-bounded (`d={dur}`).** The BD
     dump is subtitle-ONLY (no video), and ffmpeg's sub2video never signals EOF
     for a bare subtitle stream — an infinite color + `overlay=shortest=1` then
     renders forever while `mpdecimate` drops the identical frames, so output PTS
     never reaches `-t` and ffmpeg spins at ~360% CPU until the outer timeout
     (~76 min/title, hit on Avatar). A finite `d=` ends the graph at the title
     length (6 s for a 24-min episode). The DVD path never hit this only because
     its `.vob` dump has video (a sub2video heartbeat).
  3. **Whisper audio (`full_transcript`/`sample_transcript`)** — last resort when a
     title has neither CC nor a subtitle track (bonus featurettes) — those usually
     abstain in the judge anyway.
  Cached in the `transcript` table keyed by `(source, windows, length)` — a source
  is 'cc'/'subtitle-ocr'/'audio'. Validation (Haiku judge): **VB CC path S1D1
  8/8, VB VOBSUB-OCR S3 13/13** (Wikipedia synopses), and the **full Avatar
  Blu-ray (all 9 discs, PGS-OCR, TMDB synopses)** — all vs the known-correct
  order. `--transcript-source {auto,subtitle,audio}` forces a tier.
  - **The full-Avatar run is the demonstration that synopsis CORRECTS a scrambled
    metadata alignment, not just corroborates it** — the Blu-ray-scramble failure
    mode this path exists for. S1 (unscrambled here) was 20/20 agreeing with
    runtime-align at 1.0. But on B2D2 the aligner mapped E10/E11/E15 onto titles
    whose *content* is E14/E10/E11 and dropped the real E15 as a leftover; synopsis
    got all four right and recovered the leftover. Same on B3D2 (E13/E17 titles are
    really E09/E13) and it recovered E17 "Ember Island Players" that the aligner had
    dropped on B3D3. **Every single title where synopsis DIFFERED from metadata,
    synopsis was right and metadata was the scramble victim; zero wrong synopsis
    calls, and the per-season bijection held with zero double-claims.** This is the
    contrast the magnet note predicts: on a show with DISTINCTIVE episodes (Avatar)
    synopsis+bijection is a reliable *order-verifier*; on a heavy-shared-arc
    ensemble (Magicians) it stays a corroborator. Frontier judge (Haiku) + full
    transcripts + bijection is what tips it over on the distinctive show.
  - **Synopsis also implicitly rejects stripped duplicate titles the aligner falls
    for.** The scramble-victim slots on B2D2/B3D2 were 1A/0S copies with NO PGS
    stream at all (byte-stripped alternates); the aligner's position DP grabbed one
    as "E14" but synopsis abstained ("no subtitles found") — the *real* episode is
    the full-PGS twin, which synopsis identified. No-subtitle abstain doubles as a
    junk-title filter.
  - **`--transcript-source subtitle` correctly abstains on genuinely sub-less
    titles — escalate to `auto`/`audio` for those.** Avatar's Sozin's Comet finale
    (B3D3, E18–E21) is authored with only H.264 + one AC3 track, no PGS at all (the
    480i format-outlier authoring), so the forced subtitle tier had nothing to OCR
    and abstained fail-soft (metadata kept its medium-confidence assignment). Those
    titles DO carry audio, so `auto` (or `--transcript-source audio`) whispers them
    — the subtitle-only run just skipped that tier by request.
- **Transcribe the WHOLE episode by default (2026-07); windowing is opt-in.**
  Identifying dialogue is strewn throughout an episode, so sampling a few windows
  can phase-skip the very lines that name it — proven on Magicians S1D1 title 165
  (E02): the windowed sample caught only "freeze our tits off" (Julia's freezer
  initiation) and Sonnet mis-read it as the Antarctica episode (E07, confidently
  1.0); the FULL transcript shows the whole E02 plot and Sonnet gets it right.
  `full_transcript` rips+whispers the whole title (whisper base.en is
  faster-than-realtime, and we only reach synopsis after the costlier OCR path
  already failed, so it's affordable). `--synopsis-windows N` opts back into the
  faster sampled path (`sample_transcript`). Recap risk (the cold-open "previously
  on" injects a little prior-episode plot) is tiny next to a full episode and the
  judge keys on specific in-episode events.
- **Transcripts are CACHED in the DB (`transcript` table) and inspectable.**
  Whisper is the expensive part and is judge-independent, so `run_synopsis` stores
  each title's transcript keyed by its sampling params `(windows, length)` — full
  mode is `(0,0)`. A second run (e.g. to swap the judge model) reuses it and only
  re-does the cheap judge call; `--retranscribe` forces a fresh pass. `vj show
  <title>` surfaces the transcript for a human/agent. Keyed by the surrogate
  title_id and cascades on re-scan (a re-scanned disc's dialogue may differ).
- **Multi-episode guard: don't full-transcribe a play-all.** `run_synopsis` skips
  any candidate longer than **1.5× the disc-local median** candidate duration — a
  play-all/concatenation holds several episodes' dialogue and can't match ONE
  episode (and full-ripping a 102-min title is pure waste). 1.5× (tighter than
  run_streams' 1.6×) because here the long titles are still IN the sample and
  inflate the median; assumes episodes are the majority, needs ≥3 candidates.
  This surfaced two long titles mis-classified as `episode-candidate` on Magicians
  S1D1 (76m, 102m) that the windowed path had silently sampled 120s of.
- **Ollama num_ctx/num_predict must be SIZED to the prompt for this path.** Ollama
  defaults `num_ctx` to 2048 and SILENTLY truncates a longer prompt to its TAIL —
  a full-episode transcript (~5k+ tokens) then loses the dialogue and the judge
  abstains on EVERY title (observed: full-transcript qwen abstained 4/4 until
  fixed). `_ollama_text` now sizes `num_ctx` to `len(prompt)//4 + num_predict`
  rounded up to 4k, capped 32k (qwen2.5's native ctx). `VJ_JUDGE_NUM_PREDICT`
  (default 2048) bumps the output cap for a THINKING judge (gemma4) whose thinking
  would otherwise exhaust it and return empty content; `VJ_JUDGE_THINK=false`
  sends Ollama's `think:false` to run a thinking model in non-thinking mode.

- **The judge is only as good as the synopsis, and TMDB's are often too generic.**
  On The Magicians, TMDB's E01 overview is a series-premise blurb
  ("twentysomethings studying magic in New York discover a fantasy world") that
  names none of the episode's events; the exam dialogue then matched *E06's*
  "The Trials" synopsis and the judge was confidently wrong. Result on S1D1
  E01–E04: **0/4 with TMDB**.
- **Wikipedia episode summaries fix this.** They're plot-specific ("Quentin and
  Julia are invited to a *test*… Julia *fails*… they *wipe her memory*"). Same
  titles: **3/4 with Wikipedia** (the one miss, E02→E01, is genuine adjacency —
  E02's dialogue is *about* E01's aftermath). `vj enrich wikipedia <db>
  --snapshot <…-multistream.xml.bz2> --index <…-multistream-index.txt.bz2>
  --page "List of <Show> (…) episodes"` parses the `{{Episode list}}` templates
  into `episode.wiki_overview` (kept alongside TMDB `overview`; `Episode.synopsis`
  prefers wiki). `--page` is remembered on the project. `run synopsis
  --synopsis-source {auto,wikipedia,tmdb}` picks the source (auto = wiki if
  enriched). Reader (`wiki.py`): the multistream dump is concatenated ~100-page
  bz2 streams; the index gives `offset:pageid:title`, so a lookup = find offset →
  seek → decompress ONE stream → pull the `<page>`. Index and data MUST be from
  the same dump run (offsets are file-specific) — a mismatch raises, not silent
  garbage. Fully offline, pure `bz2` (no deps).
- **Use a NON-thinking judge model** (default `qwen2.5:14b-instruct`, overridable
  with `--judge-model`). A thinking model (gemma4) spends its `num_predict`
  budget reasoning and returns empty `content` on the long Wikipedia prompt —
  which reads as an abstention. (Same thinking-truncation wart as the OCR VLM.)
  `run_synopsis` had defaulted the judge to `--vlm-model` (a 2B *vision* model) —
  fixed. Judge accuracy is also model-sensitive: qwen2.5:14b got E01 where a
  weaker model didn't.
- **The judge is the bottleneck — a frontier judge is the lever (built).** A
  `claude-*` `--judge-model` (e.g. `claude-sonnet-5`, `claude-haiku-4-5-20251001`)
  routes `synopsis._ollama_text` to the Anthropic Messages API via
  `ANTHROPIC_API_KEY` instead of Ollama (no `temperature` — deprecated on current
  Claude models). Controlled comparison on Magicians S1D1 (E01–E04), IDENTICAL
  cached full transcripts + Wikipedia synopses, only the judge swapped:
  **Sonnet 4/4, Haiku 4/4, qwen2.5:14b 2/4 (windowed 3/4), gemma4 1/4 thinking /
  0/4 non-thinking.** The tell: gemma's own evidence text often *describes the
  right episode* but ranks the wrong number — a ranking-fidelity gap that scales
  with instruct-model size, exactly what CLAUDE.md predicted ("ranking quality,
  not the assignment or the transcript, is the bottleneck"). More context (num_ctx
  fix) and full transcripts did NOT rescue the weak local judges; a stronger judge
  did. Haiku ties Sonnet here, so it's the recommended API judge.
- **Cost/latency: Haiku ≈ $1 per 96-episode series, run it SYNCHRONOUSLY.** ~8.5k
  input + ~250 output tokens/judge-call (measured), one call/episode, at Haiku's
  $1/$5 per-Mtok → ~$0.93 for 96 episodes (~1¢/episode; whisper is local/free).
  The Batch API is 50% off but async (typ. <1h, ceiling 24h) — **rejected for this
  workflow**: whisper already dominates wall-clock so the discount buys nothing,
  and the poll-cycle latency is annoying for the spot-check/experiment loop. Run
  the judge synchronously (~10 min for 96 titles, deterministic).
- **Local ceiling is 16 GB VRAM (see the hardware memory).** A 14B Q4 (~9-10 GB,
  qwen2.5:14b-instruct) fits and is the best offline judge tested; 32B (~20 GB) /
  72B (~40 GB) spill to CPU and are too slow, so "a bigger local model" is not the
  lever. The productive local lever is instead constraining the candidate pool
  (feed the packaging/background hint into the synopsis pool — NOT YET BUILT; it
  would have fenced qwen off its off-disc E05 pick and gemma off E09/E12).
- **Magnet failure mode — synopsis is a corroborator, not a reliable identifier
  on ensemble shows.** Full-series Magicians run (Wikipedia source, qwen2.5:14b):
  AGREE 38 / DIFFER 25 / abstain 30 vs the metadata order. But the DIFFERs are
  mostly noise: each title is judged INDEPENDENTLY, so nothing stops many titles
  claiming one episode — 14 episodes soaked up 37 titles (S03E09 alone picked by
  6), because episodes whose synopsis is heavy on shared season-arc vocabulary
  (Fillory, the Beast, the main quest) match lots of dialogue. So: trust an AGREE
  as a confidence boost (S1 order was effectively confirmed), but do NOT treat
  the raw DIFFER list as a worklist.
- **Bijection assignment (built) — necessary but NOT sufficient.** `run_synopsis`
  now runs two phases: (1) one judge call per title returns a RANKED shortlist
  (Borda-scored — the order is reliable, the model's confidence number isn't);
  (2) a per-season global assignment (scipy Hungarian, `assign_by_synopsis`) so
  each episode is claimed at most once, a title whose shortlist is all taken
  abstains. Re-run on Magicians: magnets → **zero** (was 14 episodes / S03E09
  ×6), and the output is finally a valid one-to-one mapping. BUT accuracy did NOT
  improve — AGREE 38→31, abstain 30→42. The magnet was a *symptom* of weak
  rankings, not the disease: forcing a coherent bijection over noisy rankings
  redistributes the wrongness into a different permutation and can even move a
  correct assignment (S1D1 pl803 was rightly E04 independently, bumped to E08 by
  the global optimum). **The bottleneck is per-title ranking QUALITY**, not the
  assignment — on a heavy-shared-arc show the rankings are too short/noisy for
  even a perfect assignment to recover the order. Keep the bijection (guarantees
  a duplicate-free rip plan, and it helps on shows with distinctive episodes);
  the next lever is a stronger judge (see the OpenAI/HF backend in Future
  features), not more assignment cleverness.

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
- **OCR is naturally crash-safe now.** `vj run ocr` commits each title's evidence
  (+ frame) as it finishes, so an OOM loses only the in-progress title — re-run
  the verb and completed titles are already recorded. No `--merge` needed; the
  state DB *is* incremental. Still point `--scratch-dir` at a real disk, not
  tmpfs.
- **BUT `vj run align` needs a whole season's discs scanned first.** Align reads
  every scanned disc and aligns each season's discs together; running it after
  scanning only a subset makes each disc independently claim overlapping ranges
  (hit on Korra: 7 unique episodes/season instead of 12-14). Scan the whole
  season (ideally the whole series) before `run align`. OCR, by contrast, is
  per-title and safe to run disc-by-disc.
- **`run align` honors adjudications: confirmed = anchor, rejected = excluded.**
  A `confirmed` title is a hard PIN in the DP (`align`'s `anchors`): the path
  must route through it regardless of runtime delta, and can't gap or re-match
  it. A `rejected` title is dropped from the candidate set. So the fix for a
  same-runtime alignment shift is: confirm ONE title correctly and re-run
  `align` — a `GAP_INTERIOR` tie-break makes the aligner prefer a *contiguous*
  episode run, so the anchor shifts the whole run instead of pinning one title
  and leaving holes. No hand-bumping every downstream episode. (`GAP_INTERIOR`
  is tiny — it only breaks ties among within-tolerance matches; a real
  mid-season missing episode is still gapped.)
- **Box-packaging hints: `vj hint disc` feeds a SOFT constraint into `align`.**
  Box sets print an episode→disc mapping; `vj hint disc <db> --disc <name>
  --season N --episodes 1-4` (and/or `--titles "A" "B"`, resolved to numbers via
  TMDB at entry) records it. Stored in the `background` table keyed by disc
  **basename** — NOT by disc.id and NOT cascaded, so it survives a re-scan (which
  DELETEs+recreates the disc row) and a re-scanned disc re-adopts it. `run_align`
  builds a per-candidate allowed-episode-index set and passes it to `align`, which
  charges `BG_OUT_PENALTY` (100, one-sided) for routing a disc's title onto an
  episode the box doesn't list. Soft/advisory by design: big enough to break the
  same-runtime ambiguity it exists for, but FINITE — a runtime-impossible in-set
  episode (cost ∞) still yields to an out-of-set match, and a `confirmed`
  adjudication is still the hard override. `vj status` shows a `📦` line per disc
  with `outside` (assigned but not on the box — align overrode the hint) and
  `missing` (listed but unassigned) so a mismatch is visible.
- **Don't launch two writers on the same DB at once.** SQLite is in WAL mode and
  each op is transactional, so a single writer is safe and reads never block —
  but two concurrent `vj run`s writing the same file still contend, and an OOM'd
  run lingering can interleave badly. One writer at a time.
- Project DBs, manifests, and rip scripts are local artifacts — keep them OUTSIDE
  the repo at `/run/media/ekovac/MediaScratc/video-juicer-artifacts/` (repo =
  code/tests/docs). `.gitignore` also blocks the usual artifact patterns
  (incl. `*.db`) so a stray run in the repo dir won't pollute it.

## Future features (backlog — not yet built)

- **Pluggable LLM backend (OpenAI-compatible) instead of only Ollama.** There are
  exactly two Ollama call sites, both POSTing `{host}/api/chat`:
  `synopsis._ollama_text` (text judge) and `identify.ollama_chat` (VLM/OCR, with a
  base64 image + `done_reason` truncation handling). The clean seam is NOT
  "an HF backend" but an **OpenAI-compatible chat backend** (`/v1/chat/completions`,
  `Authorization: Bearer …`, read `choices[0].message.content`) — that one adapter
  covers HuggingFace **Inference Providers** (`https://router.huggingface.co/v1`),
  OpenAI, Groq, Together, a local vLLM, and Ollama's own `/v1`. Select by
  flag/env (`--llm-backend {ollama,openai}` + `VJ_LLM_BASE`/`VJ_LLM_KEY`), default
  Ollama.
  - **Text judge: DONE for Anthropic (2026-07), OpenAI-compat still open.** A
    `claude-*` `--judge-model` already routes `_ollama_text` to the Anthropic
    Messages API (`ANTHROPIC_API_KEY`; retry/backoff widened for 429/503/529) and
    is the recommended judge (Haiku 4/4 on Magicians S1D1 for ~$1/series — see the
    synopsis section). The remaining backlog is the *generic* OpenAI-compatible
    adapter (`/v1/chat/completions`) for HF/Groq/vLLM/etc., which the Anthropic
    branch does NOT cover (different endpoint + response shape).
  - **VLM/OCR path — moderate/hard (~+1 day), medium risk.** The API glue is easy
    (OpenAI vision uses an `image_url` `data:` URI part), but the real cost is
    **re-validating a hosted vision model** against the title-card corpus —
    qwen3-vl:2B was hand-picked because moondream *described* images and qwen2.5vl
    *translated* titles (see VLM/OCR notes). Also redo the truncation guard
    (`finish_reason=="length"`, no separate `thinking` field), and mind that this
    is the call-heavy path → cloud latency/per-token cost/rate limits (mitigated
    by Tesseract-first + the text-region gate) and it ships disc frames to a third
    party.
  - Whisper (synopsis transcription) is local `faster-whisper`, not Ollama — leave
    it local (cheap/fast); no reason to route it through a remote backend.
- **Synopsis bijection constraint (kill the magnet failure mode).** `run synopsis`
  judges every title independently, so several titles can claim one episode
  (Magicians: S03E09 picked by 6). Add a per-disc/-season global assignment pass —
  each candidate episode claimed by at most one title, solved as a bijection over
  the judge's scores (Hungarian / DP), like `align` and the OCR collision
  resolution. This is what turns synopsis from a corroborator into a usable
  order-verifier on ensemble shows. See the synopsis note for the observed data.

## Claude self-inflicted workflow traps (don't repeat)

- `pgrep -f vj` (or `identify`) matches its **own** command line → false "already
  running". Use a specific pattern (e.g. `vj.py run`).
- Inner `&`/`nohup` backgrounding inside a Bash tool call gets orphaned or torn
  down at the foreground command's exit. Use the tool's `run_in_background`.
- Running a script from `/tmp` puts `/tmp` (not the repo) on `sys.path`; set
  `sys.path.insert(0, repo)` in throwaway scripts, or run from the repo.
- `Title(...)` requires `chapters`; construct with `chapters=[]` in test scripts.
