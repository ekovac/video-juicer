# video-juicer

Maps the titles on DVD/Blu-ray disc images (or `BDMV`/`VIDEO_TS` backup dirs) to
TMDB episodes, without reading the multi-GB video payload. Peak RSS ≈ 140 MB
regardless of image size.

Modelled on **MusicBrainz Picard**: heuristics are on-demand *evidence
producers*, not a black-box pipeline. Each run attaches evidence to a
title→episode hypothesis; a human or an agent adjudicates the result. All state
lives in a SQLite **project file**. Design rationale and schema: `DESIGN.md`.
The *why* behind the heuristics and their traps: `CLAUDE.md`.

## Workflow

```bash
export TMDB_API_KEY=...

# 1. create the project + pull the episode list from TMDB
vj init show.db --tmdb-id 2418

# 2. scan disc images (metadata only — never the payload)
vj scan show.db /path/to/*.iso

# 3. run heuristics; each records evidence, decides nothing
vj run align show.db                      # metadata runtime alignment (all seasons)
vj run ocr   show.db --disc 1             # OCR title cards (keeps the frame)

# 4. turn agreeing evidence into proposed assignments
vj resolve show.db

# 5. review + adjudicate what's uncertain
vj status show.db                         # coverage summary
vj gaps   show.db                         # the worklist: conflicts + missing eps
vj show   show.db --title 7               # all evidence for one title
vj frame  show.db --title 7 --out /tmp/card.jpg   # eyeball the OCR frame
vj play   show.db S01E03                  # watch the assigned title in VLC
vj play   show.db --title 7 --at-card     # preview a candidate at its title card
vj confirm show.db --title 7              # accept the proposal
vj assign  show.db --title 9 --episode S02E05     # or set it yourself
vj reject  show.db --title 12             # not an episode

# 6. emit outputs from the adjudicated state
vj export show.db --manifest m.json --rip-script rip.sh --output-prefix /mnt/media
```

Steps 3-4 can be run in one shot with **`vj auto show.db`**, which scripts
`align → resolve → (OCR-escalate only order-unverifiable discs, when a VLM is up
and the show captions titles) → resolve → elimination → resolve`. It only ever
*proposes* — you still review `gaps` and confirm. Verifiable discs (DVD with
disc hints, runtime-separable, play-all-corroborated) skip OCR entirely, so
`auto` on a clean DVD boxset is near-instant.

`vj <verb> --help` documents each verb. Every verb emits JSON when stdout is not
a TTY (or with `--json`) and human-readable text otherwise; errors are
structured. `vj run --list` lists the available heuristics.

Because the state file is SQLite, runs are incremental and transactional by
construction — no manifest merge/lock dance. Re-running a heuristic **upserts**
its finding (one row per title per category); re-scanning a disc replaces that
disc's rows. Human/agent decisions are sticky: `resolve` never overwrites a
`confirmed`/`rejected` title.

Project databases are artifacts, not repo content — keep them out of the repo
(this workspace uses `/run/media/ekovac/MediaScratc/video-juicer-artifacts/`).

## The evidence model in one paragraph

Three layers (see `DESIGN.md` for the schema): **facts** (`disc`/`title`/
`episode`) written at ingest; **evidence** — one upserted row per (title,
category), where categories are the sources `runtime-align`, `title-card-ocr`,
`synopsis`, `elimination`; and **assignment** — the thin adjudicated answer, one
per title, `unresolved`/`proposed`/`confirmed`/`rejected`. *Conflict* is not
stored — `gaps` computes it on the fly (two categories naming different
episodes). OCR keeps the actual frame it read (as a BLOB) so a reviewer — or a
stronger VLM on demand — can judge whether it's a real title card.

## Outputs

`vj export` regenerates, from the confirmed assignments (add `--include-proposed`
to include proposals):

- a **manifest** (`--manifest`): one record per title — episode mapping, runtime
  delta, provenance (`identified_by`: `human`/`agent`/`heuristic:<category>`),
  video/audio format, and a suggested Plex/Jellyfin path.
- a runnable **HandBrake rip script** (`--rip-script`): a `bash` file with the
  common knobs hoisted into shell variables (`PREFIX`, `PRESET`,
  `HANDBRAKE_OPTS`), one `HandBrakeCLI` call per episode, with format-outlier
  episodes split into a `$PRESET_ALT` block.

The manifest **`title` field is the number to pass to your ripper**
(`HandBrakeCLI -t N`). For DVD it's the lsdvd/HandBrake title number directly.
For **Blu-ray** the tool reports HandBrake's title index from a per-disc scan —
**not** the raw `.mpls` id and **not** a player's title number, which differ
(see `CLAUDE.md`). `--output-prefix DIR` sets the script's `PREFIX`; the whole
Plex/Jellyfin tree is built under it.

## Episode ordering

`vj init --episode-order dvd` matches against a TMDB *episode group* instead of
the default aired order — essential for shows whose discs reorder episodes or
fold specials into seasons. Accepts an alias (`dvd`, `digital`, `absolute`,
`production`, `story`, `tv`) or an explicit TMDB episode-group id; records then
carry the aired numbering as an `aired` cross-reference. The TMDB type enum:
**DVD order is type 3** (verified on live data), digital is 4, production is 6.

## Requirements

- `lsdvd`, `7z` — disc scanning (`init` also needs network for TMDB, cached to
  `.tmdb_cache/`).
- `ffmpeg`, `mencoder`, and a running **Ollama** with a vision model
  (default `qwen3-vl:2B`) — for `vj run ocr`.
- `HandBrakeCLI` — only for the Blu-ray title-number scan at export time.

## Tests

```bash
python3 -m unittest test_identify_episodes test_vj -v
```

`test_identify_episodes` covers the heuristic library (play-all detector, DP
aligner, MPLS parser, OCR fuzzy-matching) on synthetic tables and a checked-in
sample playlist. `test_vj` covers the state/compute/review/export layers. Neither
needs disc images or network.

## Hard-won implementation notes

These describe the heuristic library, which is unchanged by the Picard rewrite —
only how it's invoked changed (verbs, not one pipeline).

- **TMDB runtimes are often broadcast-slot lengths** (30 min with ads) while
  discs carry the actual ~22 min episode. A per-season median ratio between disc
  and TMDB durations calibrates this; matching accepts whichever of raw/scaled
  runtime fits. Without it, four of seven Venture Bros. seasons fail entirely.
- **Thinking VLMs (qwen3-vl) need token headroom**: a tight `num_predict` gets
  exhausted by the reasoning phase and `content` comes back empty with
  `done_reason: length`. `VLM_NUM_PREDICT=8192` (a cap, not a target). On
  truncation the code returns `""` — it does **not** fall back to the partial
  `thinking` text, which is chain-of-thought, not a transcription, and
  fuzzy-matches confident-wrong titles.
- **Title cards may be at the end** of the episode (all Venture Bros. seasons),
  sometimes as an `EPISODE: <name>` line inside the credits. Scan both ends, and
  let the end window run to the last frame.
- **Sample frames at ≤2 s.** A title card is on screen only ~2-4 s, so a coarse
  stride (4-8 s) phase-skips it — producing confident "no title found" misses
  that look like the show has no titles. `extract_frames` defaults to 1.5 s.
- **Title-card position varies by show; scan a primary band, widen on miss.**
  Venture Bros. cards are at the end; Star Trek: Enterprise captions the title
  after a variable-length cold open (140-274 s normally; 306-374 s on
  recap-delayed premieres). `verify_title` runs a cheap primary pass (front
  0-280 s + the tail) and only widens the front to 720 s on a NO-CARD result,
  learning a per-disc anchor so only the first episode pays full cost.
- **The closed candidate set makes noisy OCR workable.** We match a transcription
  against the season's episode names — we don't need a perfect read. Tesseract
  reads plain block cards instantly and returns `""` on scene frames; the VLM
  fallback covers stylized script cards Tesseract can't. Larger models can
  "helpfully" translate non-English titles (`qwen2.5vl` rendered "Día de los
  Dangerous" as "Day of the Dangerous"), so the prompt demands verbatim
  transcription and the matcher tolerates diacritics/case.
- **Match part numbers flexibly.** TMDB writes "Storm Front (1)" but the card may
  read "PART ONE"/"PART I"/"PART 1"; `canon_parts` collapses them to the digit
  so the part still aligns *and* still discriminates part 1 from part 2.
- **Ollama has a memory leak** and may be OOM-killed mid-request; VLM calls retry
  with backoff and a generous timeout to ride out the daemon restart + model
  reload. Point OCR scratch at a real disk, not tmpfs.
- **Blu-ray "title number" is not universal across tools.** Three schemes coexist:
  the raw `.mpls` playlist id (`ffmpeg -playlist`), HandBrake's filtered-playlist
  title index (what you rip with), and a player's HDMV title-object list from
  `index.bdmv` (VLC). On one Enterprise disc an episode was `.mpls` 1 = HandBrake
  title 2 = VLC title 19. The tool identifies by `.mpls` but **emits HandBrake's
  number**.
