# video-juicer — Design

The architecture reference: state model, verb contracts, module layout, and the
decisions behind them. README.md is the user guide; CLAUDE.md holds each
heuristic's rationale, measurements and failure modes.

## Motivation: evidence first, decisions separate

The original tool (`identify_episodes.py`) was a pipeline that decided:
alignment, play-all detection, OCR escalation and collision resolution ran end
to end and emitted a manifest as a verdict. The reasoning behind each answer
("play-all matched", "OCR read *The Fortuneteller*") was spent internally, so a
wrong answer couldn't be explained or cheaply corrected.

The current design follows **MusicBrainz Picard**:

- Heuristics are **on-demand evidence producers**. Running one attaches findings
  to titles; it never decides.
- The tool surfaces **confidence, corroboration, conflicts and gaps** per title.
- A **human or an agent** adjudicates, and those decisions are sticky.
- Everything persists in a **project file**, mutated incrementally.

The heuristics themselves (`identify.py`, `discs.py`, `synopsis.py`) are a
library; the verbs are thin wrappers that call them and write evidence.

## The state model

SQLite: incremental, transactional writes (no manifest lock/rename dance),
queryable evidence, and a single portable file. Human-readable views come from
the inspect verbs, `vj export`, or `sqlite3`.

Three layers, deliberately separate:

1. **Facts** — what's on the discs and what TMDB says exists.
2. **Evidence** — the latest finding from each heuristic, per title.
3. **Assignment** — the thin decided answer that outputs read.

```
project            (key, value)
                     -- tmdb_id, show_name, year, episode_order (aired | alias |
                     --   group id), wikipedia_page, …

disc               (id, path, format, label, season_hint, disc_hint,
                    hb_map_json, scanned_at)
                     -- identity is the path BASENAME (state.disc_name), never the
                     --   volume label (blank/wrong/identical across box sets).
                     -- hb_map_json: .mpls id → HandBrake title index (Blu-ray).

title              (id, disc_id, title_number, duration, chapters_json,
                    n_audio, n_sub, cells, clips_json, video_format,
                    audio_format, kind, order_key, UNIQUE(disc_id, title_number))
                     -- one row per playable title (DVD title / BD playlist).
                     -- kind: episode-candidate | play-all | extra | junk | unknown.
                     -- order_key: play order (BD: set exactly by a play-all's
                     --   clip order when one exists; bare .mpls order is NOT
                     --   broadcast order).

episode            (id, season, number, name, runtime, overview,
                    wiki_overview, aired_season, aired_number,
                    UNIQUE(season, number))
                     -- the target space, in the PROJECT's ordering. aired_* hold
                     --   the aired numbering when the project uses a TMDB episode
                     --   group. wiki_overview comes from `vj enrich`.

episode_order_map  (group_id, group_name, group_type, aired_season,
                    aired_number, season, number)
                     -- TMDB's alternative orderings (DVD, production, …), keyed
                     --   by AIRED numbering, cached for the offline order check.

evidence           (id, title_id, category, episode_id, verdict, confidence,
                    payload_json, updated_at, UNIQUE(title_id, category))
                     -- ONE row per (title, category); re-running a heuristic
                     --   UPSERTS it. Categories = sources: runtime-align |
                     --   stream-signature | title-card-ocr | synopsis |
                     --   elimination.
                     -- episode_id NULL is a finding: "not an episode", "no card
                     --   readable", or (stream-signature) a claim about
                     --   episode-HOOD, not identity.

assignment         (title_id PK, episode_ids_json, status, decided_by, note,
                    decided_at)
                     -- the current answer, one per title. episode_ids_json is a
                     --   list: 2 for a merged two-parter (SxxEyy-Ezz).
                     -- status ∈ {unresolved, proposed, confirmed, rejected}; no
                     --   row reads as unresolved. Conflict is NOT stored
                     --   (derived by gaps).
                     -- decided_by ∈ {heuristic:<category>, heuristic:consensus,
                     --   human, agent}.

frame              (title_id, category, image BLOB, mime, source_time,
                    ocr_text, extracted_at, UNIQUE(title_id, category))
                     -- the frame an OCR read came from, so a reviewer (or a
                     --   stronger VLM, on demand) can judge it. Upserts with its
                     --   evidence row.

transcript         (title_id PK, text, windows, length, source, updated_at,
                    cues_json)
                     -- a title's dialogue, cached because extraction dominates
                     --   synopsis cost. source ∈ cc | subtitle-ocr | audio;
                     --   windows/length = the audio sampling (0/0 = whole
                     --   episode or subtitles). cues_json = timed captions
                     --   [[start, end, text], …] for subtitle sources.

background         (disc_name PK, episodes_json, source, updated_at)
                     -- human-entered packaging facts (`vj hint disc`). Keyed by
                     --   disc basename and NOT cascaded, so it survives a re-scan.
```

Why this shape:

- **Evidence is bounded.** Each source contributes at most one current finding
  per title; a re-run replaces it. A title's evidence is always a short, readable
  set (align says X, OCR says Y). Replay history is intentionally dropped.
- **Assignments are thin and sticky.** A heuristic writes only `proposed` (via
  `resolve`); a human or agent confirms, rejects or assigns, and nothing automatic
  overwrites that.
- **Keep what a reviewer needs to check the evidence.** OCR keeps its frame,
  synopsis keeps its transcript, align keeps its delta. A confident-wrong finding
  is only catchable if its input survives.
- **Human knowledge is keyed to stable identity.** Packaging hints key on the
  disc basename, not the surrogate `disc.id` that a re-scan recreates.

Project files are artifacts, not repo content — keep them outside the repo
(this workspace: `/run/media/ekovac/MediaScratc/video-juicer-artifacts/`).

## Verb contracts

An agent-facing tool is a UNIX tool with structured output and no hidden state:

1. Every verb takes **explicit targets** — no implicit "current disc". Titles are
   addressed by internal id (`--title`) or by disc basename + title number
   (`--disc SHOW_S1D1 --playlist 4`).
2. **JSON when stdout isn't a TTY** (or `--json`); human text otherwise.
3. **Idempotent**: compute verbs upsert their evidence; re-running never
   duplicates or corrupts.
4. **Errors are structured** (`{"ok": false, "error": …, "message": …}`).
5. **Compute verbs decide nothing.** Only `resolve` (proposals) and the
   adjudicate verbs touch `assignment`.
6. **Inspect verbs point at what's unresolved**, with a suggested next action,
   so a loop can proceed without outside reasoning.

### Ingest

- `vj init <db> --tmdb-id N [--episode-order ORDER]` — create the project, pull
  episodes (aired, or a TMDB episode group by alias/id), and cache TMDB's other
  orderings into `episode_order_map` (best effort).
- `vj scan <db> <disc…>` — scan images/backup dirs into `disc`/`title`: IFO via
  lsdvd, `.mpls` via 7z; Blu-ray also runs a HandBrake scan for stream counts and
  rip title numbers; play-all ordering and duplicate-playlist dedup happen here.
  Re-scanning a disc replaces its rows (and cascades its evidence).
- `vj orders <db> [--refresh]` — fetch/refresh TMDB's orderings (for projects
  created before `init` cached them) and report each season's fit.
- `vj enrich wikipedia <db> --snapshot … --index … --page …` — Wikipedia plot
  summaries from a local multistream dump into `episode.wiki_overview`, matched
  by title (Wikipedia's numbering may differ from the project's).
- `vj hint disc <db> --disc NAME (--season N --episodes 1-4 | --titles …)` —
  record which episodes the box says a disc holds; a soft constraint for `align`.

### Compute — evidence producers (`vj run <heuristic>`)

| Heuristic | Writes | Wraps |
|---|---|---|
| `align` | `runtime-align`: per season, a DP over disc order × runtimes (with a TMDB-runtime scale calibration). `confirmed` titles are hard anchors, `rejected` are excluded, packaging hints are a soft penalty. Aligns every scanned disc of a season together. | `classify_disc`, `align` |
| `streams` | `stream-signature`: a title's audio/subtitle layout vs its disc's majority; flags strictly-poorer titles as likely extras | `stream_signature` |
| `ocr` | `title-card-ocr` + its `frame`: Tesseract, then a VLM on frames a text-region detector passes; matched against the disc's season (`--include-specials` adds S00) | `verify_title` |
| `synopsis` | `synopsis`: transcript (cached) → judge ranking → per-season one-episode-per-title assignment | `synopsis.rank_candidates`, `assign_by_synopsis` |
| `elimination` | `elimination`: a disc's lone unmatched candidate → its one missing adjacent episode. Reads assignments, so run it after `resolve`. | `recover_by_elimination` |

`vj run --list` enumerates them. Targeting: `--disc`, `--title` (repeatable) or
`--all`.

**Synopsis pipeline.** Transcripts come from a tiered chain, each tier
self-selecting by disc format: DVD closed captions (text) → OCR of the bitmap
subtitle track (PGS/VOBSUB, RapidOCR) → whisper audio. `--transcribe-only`
refreshes the cache without judging or writing evidence; a failed re-extraction
never overwrites a cached transcript. Titles longer than 1.5× the disc median
are skipped as multi-episode. The judge is pluggable by model id: `kev`
(default — a local, open-weight Kev server), `jev-*` (TypeSafe's hosted System
One), `claude-*` (Anthropic Messages API), or any Ollama model. Kev and Jev share
one wire protocol and one question: a typed Choice over the season's episodes
plus a "none" option, whose probabilities feed the assignment directly (Kev
always on chunked dialogue, abstaining only when "none" is a clear majority). A
local judge's server is probed before any extraction; an unreachable default
stops the run rather than falling back. Evidence confidence is the assigned
episode's score relative to the title's best, so rank-scored (LLM) and
probability-scored (Jev/Kev) judges share `resolve`'s threshold.

### Resolve — evidence → proposals

`vj resolve <db> [--threshold C]` — for each title, take the evidence rows naming
an episode at confidence ≥ C:

- all agree → write `proposed` (`heuristic:consensus`, or `heuristic:<category>`
  for a single source);
- they disagree → leave it for `gaps`, and **withdraw** any existing heuristic
  proposal. Evidence arrives over time (align today, OCR tomorrow); a proposal
  made before the contradicting read existed must not keep claiming the episode.
- sticky decisions (human/agent, confirmed/rejected) are never touched.

It returns `proposed`, `conflicts`, and the `withdrawn` title ids.

### Inspect — read-only

- `vj board <db> [--season N] [--disc D] [--all]` — every disc's titles with
  assignment, status and all evidence inline; unassigned extras collapse unless
  `--all`.
- `vj status <db>` — per-season coverage, the project's episode ordering,
  order-unverified discs, order mismatches, packaging checks.
- `vj gaps <db> [--threshold C]` — the worklist. Each title needing a decision
  carries its evidence and a `suggestion`:
  - OCR names an episode, the duration corroborates, and it's the primary
    claimant → `assign` (noting which current assignment it REPLACES, if any);
  - OCR names an episode the duration doesn't fit (a featurette or play-all
    flashing a title) → `reject`;
  - a later-positioned duplicate of a corroborated read → `reject`;
  - two assignments on one episode → `review`, naming the other claimants;
  - an episode-length candidate with no identity → `run-ocr` (strengthened or
    softened by its stream signature);
  - otherwise → `review`.
  Plus the episodes nobody claims, order warnings, and order mismatches.
- `vj show <db> --title T | --episode SxxEyy` — all evidence, the assignment,
  whether a frame/transcript is on file.
- `vj frame <db> --title T [--out FILE]` — dump the retained OCR frame.
- `vj play <db> SxxEyy | --title T [--at-card] [--print]` — play the title in
  VLC/mpv. DVD selects the title exactly; Blu-ray plays the playlist's `.m2ts`
  clips from the BDMV dir (a player's title index doesn't match the `.mpls` id).

### Order checks

Two different questions:

- **Is the order within a disc trustworthy?** (`assess_ordering`, surfaced as
  "order unverified") — DVD title order is broadcast order within a disc; Blu-ray
  `.mpls` order is not, unless a play-all covering every episode title, multi-part
  names, or separable runtimes corroborate it.
- **Which numbering do the discs follow?** (`review.order_check`, surfaced as a
  mismatch) — per season, walk the assigned titles in disc order, take each one's
  *content* identity (accepted title-card read, else synopsis, else a confirmed
  decision — never runtime-align, which is the position guess under test), skip
  specials, and score each ordering (the project's and every cached TMDB group) by
  the share of consecutive titles that don't go backwards. The project's ordering
  out of sequence (≤0.9) while a TMDB group fits (≥0.85) and beats it by ≥0.05 →
  "discs follow TMDB '<group>'; re-init with `--episode-order <alias>`". Only
  discs whose play order is meaningful are used (DVDs, and Blu-rays with a
  play-all).

### Adjudicate

- `vj assign <db> <title> --episode SxxEyy [--episode …] [--note …] [--agent]` →
  `confirmed` (`decided_by` = human, or agent with `--agent`).
- `vj confirm <db> <title> [--agent]` → accept the standing proposal.
- `vj reject <db> <title> [--note …] [--agent]` → not an episode.
- `vj unassign <db> <title> | --all` → clear decisions.

### Outputs

Both read the adjudicated layer (confirmed; `--include-proposed` adds proposals)
through one record builder (`export.build_records`), so they always agree.

- `vj export <db> [--manifest F] [--rip-script F] [--output-prefix DIR]` — a JSON
  manifest (mapping, provenance, runtime delta, formats, the ordering the numbers
  are in, the Plex/Jellyfin path) and a bash HandBrake script (knobs hoisted into
  variables; format outliers rip with `$PRESET_ALT`; header states the ordering).
- `vj transcode <db> [--output-prefix DIR] [--dry-run] [--force]` — runs the
  encodes and tags each `.mkv`. Idempotent by construction: `VJ_RECIPE` hashes
  only encode inputs (source basename, HandBrake title, preset, options) and
  `VJ_META` hashes descriptive fields (names, numbers, ids, ordering). Outputs are
  indexed by their `VJ_EPISODES` tag, and each episode gets one action: recipe
  changed → re-encode; moved → rename + retag; metadata-only change → retag;
  unchanged → skip. Encodes write to `.part` and move into place when tagged.

The Blu-ray `title` in every output is HandBrake's title index (from the scan's
`hb_map_json`), not the `.mpls` id — the three numbering schemes (mpls,
HandBrake, player) differ.

### Automation

`vj auto <db>` scripts the loop:
`align → resolve → (OCR discs whose order is unverifiable, if a VLM is reachable
and a two-title probe finds on-screen titles) → resolve → elimination → resolve`.
It writes evidence and proposals and never confirms — the same verbs a human or
agent would run, with a fixed policy.

## Modules

| Module | Role |
|---|---|
| `vj.py` | CLI: argument parsing, verb dispatch, human/JSON rendering |
| `state.py` | schema, migrations, row⇄dataclass mapping, all DB operations |
| `discs.py` | data model (`Disc`/`Title`/`Episode`/`Assignment`), disc scanning, MPLS parsing, dedup, play-all ordering, HandBrake scan, TMDB client + episode groups |
| `identify.py` | heuristics: alignment, ordering assessment, title-card OCR, elimination, filenames, rip script |
| `text_region.py` | OCR-free text detection (VLM gate) and RapidOCR text reading |
| `synopsis.py` | transcript extraction (CC / subtitle OCR / whisper), judge backends (Kev / TypeSafe Jev / Anthropic / Ollama), ranking, the assignment |
| `wiki.py` | offline Wikipedia multistream reader; summary parsing and title matching |
| `compute.py` | `vj run` — heuristics as evidence producers |
| `review.py` | inspect (status/gaps/board/show), resolve, order checks |
| `export.py` | manifest + rip script from assignments |
| `transcode.py` | idempotent HandBrake runs + Matroska tags |
| `auto.py` | `vj auto` |
| `bench_synopsis.py` | offline benchmark of synopsis judges against a project's assignments (opens it read-only; per-judge JSONL cache) |
| `bdj_*.c` | parked Blu-ray Java navigation probes (`BDJ_NOTES.md`) |

## Decisions

- **Evidence is one bounded row per (title, category), upserted** — not an append
  log. The newest run of a source is what's true now. (2026-07-01)
- **Conflict is computed, not stored.** `gaps` derives it; `assignment` holds only
  decisions. (2026-07-01)
- **A single source records one finding per title**; extra detail (OCR text,
  deltas, shortlists) goes in `payload_json`, not a second row. (2026-07-01)
- **Disc identity is the path basename**, everywhere: display, packaging hints,
  and `--disc` resolution. The volume label is secondary. (2026-09-24)
- **`resolve` withdraws stale proposals** when evidence now conflicts, so a
  position guess never outlives the read that contradicts it. (2026-09-24)
- **Every episode number travels with its ordering.** The project's
  `episode_order` is stated in status, manifests, rip scripts and MKV tags, and
  the order check compares the discs against TMDB's alternatives. (2026-09-24)
- **Judges are pluggable by model id and benchmarked, not assumed.** The best
  judge differs by show (serialized vs episodic); `bench_synopsis.py` measures it.
  (2026-09-23)
- **The default judge is open-weight and local (Kev-4B)**, chosen by benchmark
  over the previous Ollama default; a missing server is an error, not a silent
  downgrade. (2026-09-25)

## History

- **2026-07-01** — the Picard rewrite replaced the `identify_episodes.py`
  pipeline; its heuristics became the library above. `IMPLEMENTATION_PLAN.md` and
  `identify_episodes.py` were removed (in git history).
- **July–September 2026** — added `streams`, `synopsis` (subtitle-first
  transcripts, frontier and Jev judges, the per-season assignment), Wikipedia
  enrichment, packaging hints, `play`, `transcode`, `auto`, the judge benchmark,
  cue timings, and the ordering checks.
- A recap-trimming experiment for the synopsis path (drop the "previously on"
  segment by shared n-grams) worked but wasn't worth its complexity; it lives on
  the `recap-trim-experiment` branch, summarized in CLAUDE.md.
