# video-juicer — Design: evidence-first, Picard-style rewrite

Status: **implemented** (2026-07-01). Replaced the earlier end-to-end "verdict"
architecture (a single `identify_episodes.py` pipeline). The hard-won heuristics
stayed as a library; the *orchestration and output model* changed. This doc is
the living reference for the design; README.md is user-facing, CLAUDE.md is the
heuristic *why*.

## Motivation

The old tool was a black box that decided. `identify_episodes.py` ran the whole
pipeline — metadata alignment, play-all detection, OCR escalation,
collision resolution — and emits `manifest.json` as a committed verdict. The
human only sees the final answer plus a few flags. The reasoning that produced
it (play-all matched, runtimes separable, OCR read "The Fortuneteller") is spent
internally and thrown away, so when the verdict is wrong the user can't see
*why*, and can't cheaply correct one title without re-running the pipeline.

We want the **MusicBrainz Picard** model instead:

- Heuristics are **on-demand evidence producers**, not steps of one committed
  pipeline. Each run *attaches evidence* to a title→episode hypothesis rather
  than deciding.
- The tool surfaces **confidence, corroboration, and gaps/conflicts** per title.
- A **human or an agent** is the final resolver — adjudicating ambiguous titles
  and filling holes.
- Everything persists in a **project state file** (Picard's "album" pane): the
  working document that runs mutate incrementally.

This reframes — does not discard — the logic in `identify.py` / `discs.py` /
`synopsis.py`.

## The state model

SQLite, not JSON. Rationale: incremental mutation without clunky full-file
rewrites; transactional writes replace the manifest's file-lock + atomic-rename
dance; querying over evidence is most of what the inspect verbs need; a future
GUI binds to it directly. Human-readability is recovered via `vj export` (JSON
dump) and the always-available `sqlite3` CLI.

### Core insight: evidence is a bounded set of categorized findings; assignments are adjudicated

Keep the two layers separate. Evidence is **one row per (title, category)** — the
latest finding from each source, like Picard showing the current result of each
lookup, *not* an unending log. Re-running a category (e.g. a second OCR pass with
a better model) **upserts** its row, replacing the stale finding. The
`assignment` row is the thin, mutable, resolved answer that `export` reads.

```
disc        (id, path, format, label, season_hint, disc_hint, hb_map_json,
             scanned_at)

title       (id, disc_id, title_number, duration, chapters_json,
             n_audio, n_sub, cells, clips_json,
             video_format, audio_format,
             kind, order_key)
              -- one row per playable title (DVD title / BD playlist).
              -- title_number is the HandBrakeCLI -t arg (BD: HandBrake index,
              --   NOT the .mpls id — see CLAUDE.md output notes).

episode     (id, season, number, name, runtime, overview,
             aired_season, aired_number)
              -- the target space, pulled from TMDB. aired_* set when an
              --   alternate episode-group ordering is in use.

evidence    (id, title_id, category, episode_id, verdict, confidence,
             payload_json, updated_at,
             UNIQUE(title_id, category))
              -- ONE row per (title, category). categories are the evidence
              --   SOURCES: 'runtime-align' | 'play-all' | 'title-card-ocr' |
              --   'synopsis' | 'elimination'. re-running a category UPSERTS
              --   (overwrites) its row — bounded, not a log.
              -- episode_id NULL is legal and meaningful: "this title is NOT an
              --   episode" (featurette/junk) or "no card readable" is a finding.
              -- payload_json holds category-specific detail (OCR text, delta,
              --   play-all clip order, chapter-match segments, …).

assignment  (title_id PK, episode_ids_json, status, decided_by, note,
             decided_at)
              -- the CURRENT resolved answer; ONE row per title.
              -- episode_ids_json is a list: 1 normally, 2 for a merged
              --   two-parter (SxxEyy-Ezz).
              -- status ∈ {unresolved, proposed, confirmed, rejected}
              --   (conflict is NOT stored — it's derived by `gaps`.)
              -- decided_by ∈ {heuristic:<name>, human, agent}

frame       (title_id, category, image BLOB, mime, source_time, ocr_text,
             extracted_at,
             UNIQUE(title_id, category))
              -- the actual frame an OCR-family source read, kept so a human
              --   (or a stronger VLM, on demand) can eyeball whether it's a
              --   real title card or a mis-fired scene frame. BLOB keeps the
              --   project file self-contained + portable; title cards are small
              --   JPEGs (tens of KB) so even a full show is a few MB. Upserts
              --   with its evidence row (re-running OCR replaces the frame too).
              --   source_time = seconds into the title the frame was sampled.

project     (kv: tmdb_id, show_name, year, episode_order, created_at, …)
```

Why this shape:

- **Evidence is categorized and bounded.** Each source contributes at most one
  finding per title; a fresh run of that source replaces its prior finding
  rather than piling up. At any moment the evidence for a title is a short,
  readable set (align says X, OCR says Y, …) — the Picard "look it over" view.
  We deliberately drop replay history; if a source disagrees with itself across
  runs, the newest run is what's true now.
- **`assignment` is the thin adjudicated layer.** A heuristic sets
  `status='proposed'`; a human/agent flips it to `confirmed`. Two categories
  pointing at different episodes → `conflict`, which is exactly what `gaps`
  surfaces.
- **`episode_id NULL` evidence** cleanly models "featurette, not an episode" and
  "no card readable" — cases the current code scatters across special-casing.
- **Two-parters** are a list in `episode_ids_json`, matching the existing
  `Assignment.episodes` (list; 2 for a merged double). (A single OCR/align
  category that itself implies two episodes stores both in its `payload_json`;
  the category→episode_id column holds the primary.)
- **OCR keeps its frame.** The evidence is only as trustworthy as the frame it
  read; a mis-fired scene frame can produce a confident-wrong title. Retaining
  the frame lets a reviewer confirm at a glance ("that's a title card" vs "that's
  a random shot"), and lets an agent escalate a *cheap* OCR finding to an
  *expensive* stronger-VLM re-check without re-ripping the disc. Stored as a BLOB
  so the project file stays a single portable artifact.

### State file location

Project databases (like manifests/rip scripts before them) are **artifacts, not
repo content** — write them to
`/run/media/ekovac/MediaScratc/video-juicer-artifacts/`. The repo holds only
code/tests/docs.

## The verbs

Design principle (this is our first agent-facing tool): **an agent tool is a
UNIX tool with structured output and no hidden state.** Concretely:

1. Every verb takes **explicit IDs** — no implicit "current disc."
2. **JSON output by default** when stdout is not a TTY (human-pretty when it is).
3. **Idempotent**: re-running a compute verb **upserts** its category's evidence
   row for each title (overwrites the stale finding), so state stays bounded and
   a re-run never corrupts or duplicates.
4. Every verb **ends by pointing at what's still unresolved**, so an agent can
   loop without guessing the next step.
5. **Errors are structured data**, not just an exit code.
6. Compute verbs **decide nothing** — they only append evidence. Only adjudicate
   verbs touch `assignment`.

### Ingest — populate the target + candidate space
- `vj init <state.db> --tmdb-id N [--episode-order dvd]` — create DB, pull
  episodes into `episode`, write `project` metadata.
- `vj scan <state.db> <disc…>` — `scan_disc` each image, populate `disc`/`title`.

### Compute — evidence producers (reframed heuristics; append only)
- `vj run align <state.db> [--disc D]` — runtime/DP alignment → proposed
  evidence (wraps `align`, `runtime_scale`).
- `vj run play-all <state.db> --disc D` — order via play-all (wraps
  `detect_play_all` / `order_by_playall`).
- `vj run ocr <state.db> --title T [--title …] [--include-specials]` —
  title-card OCR on specific titles (wraps `verify_title`). The match pool is
  scoped to the disc's season by default (efficient, right for episodes);
  `--include-specials` widens it with the S00 pool so a leftover title can be
  identified as a special (its card has something to match against).
- `vj run synopsis <state.db> --title T` — dialogue→synopsis judge (wraps
  `synopsis.identify_by_synopsis`).
- `vj run elimination <state.db> [--disc D]` — recover-by-elimination.
- `vj run --list` — enumerate available heuristics (agent discovery).

### Inspect — read-only
- `vj status <state.db>` — summary counts (confirmed / proposed / conflict /
  unresolved) per season.
- `vj gaps <state.db>` — **the agent worklist**: every title that's `conflict`
  or `unresolved`, with its competing evidence.
- `vj show <state.db> --title T | --episode E` — all evidence + current
  assignment for one thing; reports whether an OCR frame is on file.
- `vj frame <state.db> --title T [--out FILE]` — dump the retained OCR frame to
  a file (default: a temp path), so a human or a vision-capable agent can look
  at it and judge whether it's a real title card.
- `vj play <state.db> <SxxEyy> | --title T [--player vlc] [--at-card] [--print]`
  — launch a player on the title assigned to an episode (or any title by id, to
  preview a candidate before adjudicating). `--at-card` seeks to the retained
  OCR frame's timestamp. DVD title selection is exact (`dvd:///path#N`); Blu-ray
  opens at the disc's main title (the MRL can't select a playlist).

### Resolve — evidence → proposed assignments (the bridge)
- `vj resolve <state.db> [--threshold C]` — for each title whose evidence
  *agrees* (all categories above `C` name the same episode), write a `proposed`
  assignment (`decided_by=heuristic:consensus`, or `heuristic:<category>` for a
  lone source). Disagreement is left unresolved for `gaps` to surface as a
  conflict. **Never touches a sticky decision** (human/agent/confirmed/rejected).
  This reconciles principle 6 (compute verbs write only evidence) with the
  core-insight line that assignments can be heuristic-decided: the heuristic
  decides *proposals* here, from evidence, in one explainable step — not inside
  the compute verbs.

### Adjudicate — write the thin resolved layer (human or agent)
- `vj assign <state.db> --title T --episode E [--episode E2] [--note …]` →
  `confirmed` (repeat `--episode` for a two-parter). `--agent` records
  `decided_by=agent`.
- `vj reject <state.db> --title T` → not an episode.
- `vj confirm <state.db> --title T` → accept the standing proposal as-is.

### Export
- `vj export <state.db> [--manifest out.json] [--rip-script out.sh]
  [--output-prefix DIR]` — regenerate outputs from resolved assignments (reuse
  `suggested_filename`, `emit_rip_commands`, format-outlier split). Keeps the
  entire downstream rip workflow intact.

### The nice property
`--auto` stops being a special mode and becomes a *script over these verbs*:
`scan → run align → run play-all → for each gap: run ocr → confirm the
unambiguous`. The agent workflow is the same loop with an LLM making the
`assign`/`confirm` calls where rules can't. **Same verbs, two drivers** (human,
agent).

## Migration (done)

Full replacement of the CLI/output model; the heuristics survived as a library.

- **Kept as library, unchanged:** `discs.py` (data model + scanning + `Tmdb`),
  and the heuristic functions in `identify.py` / `synopsis.py` (`align`,
  `detect_play_all`, `assess_ordering`, `verify_title`, `ocr_identify`,
  `recover_by_elimination`, `resolve_*`, `format_outliers`, `suggested_filename`,
  `emit_rip_commands`, …). `verify_title` gained one optional `capture` out-param
  so the OCR verb can retain the winning frame; no other heuristic changed.
- **New:** `state.py` (schema + row⇄dataclass mappers + ops), `compute.py`
  (evidence-producer verbs), `review.py` (inspect/resolve/adjudicate),
  `export.py` (manifest + rip script), `vj.py` (entry point). Compute verbs are
  thin wrappers that call the heuristics and write `evidence` rows.
- **Removed:** `identify_episodes.py` (the committed end-to-end `main` +
  manifest-as-primary-output) and `IMPLEMENTATION_PLAN.md` (its durable findings
  folded into README/CLAUDE). Git retains both for earlier revisions.
- **Tests:** `test_identify_episodes.py` keeps the pure heuristic unit tests (now
  importing the library modules directly via an `ie` shim); `test_vj.py` adds
  state/compute/review/export coverage (evidence upsert, resolve consensus +
  conflict + sticky decisions, align mapping, export records).

- **`elimination` verb (built):** `vj run elimination` reads the assignment
  layer (proposed/confirmed), reconstructs `final`/`leftovers`/`missed`, runs
  `recover_by_elimination`, and writes `elimination` evidence — so it runs
  *after* `resolve`, and a subsequent `resolve` proposes what it recovered. It
  still writes only evidence (never assignments), keeping the compute/adjudicate
  split intact.

- **`auto` verb (built):** `vj auto` scripts the chain — `align → resolve →
  (OCR-escalate only order-unverifiable discs, gated on a reachable VLM + a
  card-presence probe) → resolve → elimination → resolve`. It writes evidence
  and *proposals* but never confirms; a human/agent still adjudicates what
  `gaps` surfaces. This realises the "same verbs, one driver" property above —
  `auto` is just the human/agent loop with a fixed policy. Verifiable discs
  (DVD with disc hints, runtime-separable, play-all-corroborated) skip OCR.

## Open questions / notes

- **Decided (2026-07-01):** evidence is one bounded row per (title, category),
  upserted on re-run — not an append log. Replay history is intentionally
  dropped.
- Category granularity: categories == evidence sources (`runtime-align`,
  `play-all`, `title-card-ocr`, `synopsis`, `elimination`). If a single source
  ever needs to record two distinct findings for one title, it goes in that
  category's `payload_json`, not a second row. Revisit only if a real case
  breaks this.
- **Decided (2026-07-01):** conflict is **computed, not persisted**. `gaps`
  derives conflict on the fly (two evidence categories naming different episodes
  above threshold); `assignment.status` therefore carries only the
  human/agent/heuristic-set values {unresolved, proposed, confirmed, rejected}.
  Keeps the assignment layer purely about *decisions*, not derived observations.
- (space for your notes)
```
