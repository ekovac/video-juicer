# video-juicer

Maps the titles on DVD/Blu-ray disc images (or `BDMV`/`VIDEO_TS` backup dirs) to
TMDB episodes. Scanning reads disc *metadata* only — IFO tables via `lsdvd`,
`.mpls` playlists via `7z` — never the multi-GB video payload (peak RSS ≈ 140 MB
regardless of image size). The heavier identifiers (title-card OCR, dialogue
transcripts) read only the titles you point them at.

Modelled on **MusicBrainz Picard**: heuristics are on-demand *evidence
producers*, not a black-box pipeline. Each run attaches evidence to a
title→episode hypothesis; a human or an agent adjudicates. All state lives in a
SQLite **project file**. Architecture and schema: `DESIGN.md`. The *why* behind
every heuristic and the traps it avoids: `CLAUDE.md`.

`vj` below is `python3 vj.py`. Every verb emits JSON when stdout is not a TTY
(or with `--json`) and human-readable text otherwise; errors are structured.
`vj <verb> --help` documents each verb; `vj run --list` lists the heuristics.

## Workflow

```bash
export TMDB_API_KEY=...

# 1. create the project + pull the episode list (and TMDB's alternative orderings)
vj init show.db --tmdb-id 2418                  # add --episode-order dvd if the discs are in DVD order
vj scan show.db /path/to/*.iso                  # metadata only

# (optional) facts you know from the box: which episodes each disc holds
vj hint disc show.db --disc SHOW_S1D1 --season 1 --episodes 1-4

# 2. produce evidence — each run records findings, decides nothing
vj run align show.db                            # runtime alignment, every season
vj run streams show.db                          # flag episode-length extras by audio/sub layout
vj run ocr show.db --disc SHOW_S1D1             # read on-screen title cards (keeps the frame)
vj run ocr show.db --all                        # …or every candidate title
vj run synopsis show.db --all                   # identify by dialogue (no-title-card shows)

# 3. turn agreeing evidence into proposals
vj resolve show.db

# 4. review and adjudicate what's uncertain
vj board   show.db                              # every disc's titles + all evidence, one screen
vj status  show.db                              # coverage, order warnings, packaging checks
vj gaps    show.db                              # the worklist, each item with a suggested action
vj show    show.db --title 7                    # all evidence for one title (also --episode S01E03)
vj frame   show.db --title 7 --out card.jpg     # eyeball the title card OCR read
vj play    show.db S01E03                       # watch the title assigned to an episode
vj assign  show.db --title 9 --episode S02E05   # decide it yourself (repeat --episode for a two-parter)
vj confirm show.db --disc SHOW_S1D1 --playlist 4  # accept a proposal (titles by id or disc+number)
vj reject  show.db --title 12                   # not an episode
vj unassign show.db --title 12                  # undo a decision

# 5. outputs from the adjudicated state
vj export show.db --manifest m.json --rip-script rip.sh --output-prefix /mnt/media
vj transcode show.db --output-prefix /mnt/media  # or run the encodes directly (idempotent)
```

Steps 2–3 run in one shot with **`vj auto show.db`**:
`align → resolve → (OCR only the discs whose order can't be verified, when a VLM
is up and the show has title cards) → resolve → elimination → resolve`. It only
ever *proposes*; you still review `gaps` and confirm. A clean DVD box set with
disc hints never pays for OCR.

Runs are incremental and transactional: re-running a heuristic **upserts** its
finding (one row per title per heuristic), re-scanning a disc replaces that
disc's rows, and human/agent decisions are sticky — nothing automatic overwrites
a `confirmed`/`rejected` title or a human/agent assignment. **Re-run `resolve`
after any evidence producer**: it proposes what newly agrees and *withdraws* a
stale proposal that new evidence contradicts (it then shows up in `gaps`).

Keep project databases, manifests and rip scripts outside the repo (this
workspace uses `/run/media/ekovac/MediaScratc/video-juicer-artifacts/`).

## The evidence model

Three layers (`DESIGN.md` has the schema):

- **Facts** — `disc`/`title`/`episode`, written at `scan`/`init`.
- **Evidence** — one row per (title, heuristic), replaced when that heuristic
  re-runs: `runtime-align`, `stream-signature`, `title-card-ocr`, `synopsis`,
  `elimination`. A row may say "not an episode" or "no card readable".
- **Assignment** — the thin decided answer per title:
  `proposed` (by `resolve`), `confirmed`, or `rejected`, with who decided
  (`heuristic:<name>`, `human`, `agent`).

*Conflict* is never stored — `gaps` computes it (two heuristics naming different
episodes). OCR keeps the frame it read, and synopsis keeps the transcript, so a
reviewer can check the evidence itself.

## The heuristics

| `vj run …` | Evidence | Needs | Use when |
|---|---|---|---|
| `align` | disc order × runtimes → episodes (DP; calibrates TMDB's broadcast-slot runtimes) | nothing | always, first. Scan the whole season/series before running it |
| `streams` | titles whose audio/subtitle layout is poorer than the disc's episodes (likely extras) | HandBrake scan (Blu-ray) | episode-length extras confuse alignment |
| `ocr` | the on-screen title card, matched to the season's episode names | ffmpeg, mencoder, Tesseract, Ollama VLM | the show has title cards; order is unverified (Blu-ray) |
| `synopsis` | dialogue judged against episode plot summaries | subtitles or whisper; a judge model | no title cards, order unverified |
| `elimination` | a disc's lone unmatched title → its one missing adjacent episode | run after `resolve` | a premiere/finale without a card |

`align` honours decisions: a `confirmed` title is a hard anchor (confirm one
title of a shifted run and re-align to shift the whole run) and a `rejected`
title is excluded. `vj hint disc` adds the box's episode list as a soft
constraint.

### Title-card OCR

Tesseract reads plain cards first (tens of ms/frame); a vision model (Ollama,
default `qwen3-vl:2B`) is the fallback for stylized cards. A text-region detector
(PaddleOCR detection via RapidOCR) keeps text-less scene frames away from the
VLM (~92% pruned at 100% recall on real cards); if a show paints its title into
the scene art, pass `--no-text-filter`. `--include-specials` lets a leftover
title match an S00 special.

### Synopsis identification

For shows without title cards. Dialogue comes from, in order: DVD closed
captions (exact text, seconds), OCR of the bitmap subtitle track (Blu-ray PGS /
DVD VOBSUB; minutes per episode), then whisper audio. Transcripts — with caption
timings when they came from subtitles — are cached in the project, so swapping
the judge costs only the judge calls (`--transcribe-only` extracts without
judging; `--retranscribe` forces a fresh pass).

The judge compares each title's dialogue with every episode's plot summary in
its season, then a one-episode-per-title assignment (Hungarian) settles the
season. Plot summaries: TMDB's, or — much better — Wikipedia's, from a local
dump:

```bash
vj enrich wikipedia show.db --snapshot enwiki-…-multistream.xml.bz2 \
    --index enwiki-…-multistream-index.txt.bz2 --page "List of <Show> episodes"
```

Summaries are matched to episodes by title (Wikipedia's numbering can differ).

`--judge-model` picks the judge: an Ollama model (default
`qwen2.5:14b-instruct`), a `claude-*` model via `ANTHROPIC_API_KEY`, or
TypeSafe's Jev (`jev-latest`) via `TYPESAFE_API_KEY`. Measured on two full series
(`bench_synopsis.py`, below):

| Judge | The Expanse (60, serialized) | Venture Bros (81, episodic) | Cost for both |
|---|---|---|---|
| Opus 5.5 | 60/60 | 80/81 | ~$8 |
| Jev (chunked) | 57/60 | 81/81 | ~$0.10 |
| Sonnet 5 | 46/60 | 78/81 | ~$4.40 |
| Haiku 4.5 | 23/60 | 78/81 | ~$1.45 |

### Benchmarking judges

`bench_synopsis.py <db>` replays a project's cached transcripts through any set
of judges and scores them against the project's assignments: accuracy (wrong
counted separately from abstaining), false claims on non-episode titles, latency
and cost. Results are cached per judge, so runs resume and `--report-only`
re-scores for free. If every judge "misses" the same titles, audit the reference
project before believing the scores — that's how 12 bad Venture Bros assignments
were found.

## Episode ordering

"S01E03" means different episodes in different orderings — Venture Bros' DVDs
follow TMDB's *DVD Order*, which reshuffles seasons 1–3 against aired order. So:

- **Choose at `init`.** `--episode-order` takes `aired` (default), an alias
  (`dvd`, `digital`, `absolute`, `production`, `story`, `tv`) or a TMDB
  episode-group id. `init` lists the show's other orderings.
- **Detection.** `vj status`/`gaps` warn when a season's discs, in play order,
  follow a different TMDB ordering than the project (judged from title cards,
  synopsis and confirmed decisions — never from alignment's position guesses),
  and say which `--episode-order` would match. `vj orders show.db` shows the
  per-season fit and fetches TMDB's orderings for projects created before this.
- **Stamping.** Every output states its numbering: manifest `episode_order` (plus
  `aired` cross-references for a non-aired project), the rip script's header, and
  a `VJ_ORDER` Matroska tag. Media servers match files by SxxEyy — set the show's
  episode ordering in Plex/Jellyfin to the same one.

## Outputs

`vj export` writes, from the confirmed assignments (`--include-proposed` adds
proposals — run `gaps` first):

- a **manifest** (`--manifest`): one record per episode title — mapping, runtime
  delta, provenance (`identified_by`), numbering (`episode_order`), video/audio
  format, and a Plex/Jellyfin path
  (`Show (Year) {tmdb-ID}/Season NN/Show (Year) - SxxEyy - Name.mkv`);
- a runnable **HandBrake script** (`--rip-script`) with the knobs hoisted into
  variables (`PREFIX`, `PRESET`, `HANDBRAKE_OPTS`); episodes whose video format
  differs from the rest (Avatar's 480i finale in a 1080p set) rip with
  `$PRESET_ALT`.

**`vj transcode`** runs the encodes itself and writes Matroska tags (show,
season/episode, name, TMDB id, `VJ_*` provenance). It is idempotent: a
`VJ_RECIPE` tag hashes exactly the encode inputs, so a re-run re-encodes only
what changed, renames/re-tags on a metadata-only change, and skips the rest.
Encodes go to a `.part` file and move into place when complete. `--dry-run`
shows the plan; `--force` re-encodes everything.

The manifest's **`title` is the number to pass to `HandBrakeCLI -t`**. On DVD
it's the lsdvd title; on **Blu-ray** it's HandBrake's own title index — not the
`.mpls` id and not a player's title number, which all differ (one Enterprise
episode was `.mpls` 1 = HandBrake 2 = VLC 19).

## Requirements

- **Always:** Python 3, `lsdvd`, `7z`, and `TMDB_API_KEY` (responses cached in
  `.tmdb_cache/`).
- **Blu-ray / export:** `HandBrakeCLI` — the per-disc scan supplies stream counts
  and the rip title numbers, and runs the encodes for `vj transcode`.
- **OCR:** `ffmpeg`, `mencoder`, `pytesseract` + Tesseract, Ollama with a vision
  model; recommended `rapidocr-onnxruntime` for the text-region gate (missing →
  the gate is skipped, no risk).
- **Synopsis:** `ffmpeg`, `mplayer`/`mencoder` (DVD), `rapidocr-onnxruntime`
  (subtitle OCR), `faster-whisper` (audio fallback), `scipy` (the assignment),
  and a judge (Ollama, or `ANTHROPIC_API_KEY` / `TYPESAFE_API_KEY`). A Wikipedia
  multistream dump for `vj enrich` is optional but strongly recommended.
- **Transcode:** `mkvtoolnix` (`mkvpropedit`, `mkvextract`) for the tags.

## Tests

```bash
python3 -m unittest test_identify_episodes test_vj
```

`test_identify_episodes` covers the heuristic library (play-all detection, the
aligner, MPLS parsing, OCR matching) on synthetic data and a checked-in sample
playlist; `test_vj` covers the state/compute/review/export/transcode layers.
Neither needs disc images or network.

## See also

- `DESIGN.md` — architecture, schema, verb contracts, decisions.
- `CLAUDE.md` — every heuristic's rationale, measured results, and failure modes.
- `BDJ_NOTES.md` — parked experiments with Blu-ray Java navigation (discs whose
  episode order exists only in BD-J code).
