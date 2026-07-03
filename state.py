"""SQLite project-state layer for video-juicer.

The state file is the persistent working document (Picard's "album" pane). Three
layers live here:

- **facts** — `disc` / `title` / `episode`: what's on the discs and what TMDB
  says the target episodes are. Written once at ingest.
- **evidence** — one row per (title, category); each heuristic ("source")
  contributes at most one *current* finding per title. Re-running a category
  UPSERTS its row (bounded, not a log). See DESIGN.md.
- **assignment** — the thin adjudicated layer: one decided answer per title,
  set by a human/agent/heuristic. `conflict` is NOT stored here; it's derived.

This module is pure data-access (no heuristics, no TMDB, no disc scanning). The
`discs` dataclasses (Title/Disc/Episode) are the in-memory shape; helpers here
map them to/from rows so the heuristic library can consume loaded state.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from discs import Disc, Episode, Title

# Evidence categories == evidence sources. One row per (title, category).
CATEGORIES = (
    "runtime-align",
    "play-all",
    "stream-signature",
    "title-card-ocr",
    "synopsis",
    "elimination",
)

# Adjudicated statuses (conflict is derived by `gaps`, never stored).
STATUSES = ("unresolved", "proposed", "confirmed", "rejected")

SCHEMA = """
CREATE TABLE IF NOT EXISTS project (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS disc (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL,
    format      TEXT NOT NULL,          -- dvd | bluray
    label       TEXT,
    season_hint INTEGER,
    disc_hint   INTEGER,
    hb_map_json TEXT NOT NULL DEFAULT '{}',
    scanned_at  TEXT
);

CREATE TABLE IF NOT EXISTS title (
    id            INTEGER PRIMARY KEY,   -- surrogate; unique across discs
    disc_id       INTEGER NOT NULL REFERENCES disc(id) ON DELETE CASCADE,
    title_number  INTEGER NOT NULL,      -- disc-local id / HandBrakeCLI -t arg
    duration      REAL NOT NULL,
    chapters_json TEXT NOT NULL DEFAULT '[]',
    n_audio       INTEGER NOT NULL DEFAULT 0,
    n_sub         INTEGER NOT NULL DEFAULT 0,
    cells         INTEGER NOT NULL DEFAULT 0,
    clips_json    TEXT NOT NULL DEFAULT '[]',
    video_format  TEXT,
    audio_format  TEXT,
    kind          TEXT NOT NULL DEFAULT 'unknown',
    order_key     INTEGER NOT NULL DEFAULT 0,
    UNIQUE(disc_id, title_number)
);

CREATE TABLE IF NOT EXISTS episode (
    id            INTEGER PRIMARY KEY,
    season        INTEGER NOT NULL,
    number        INTEGER NOT NULL,
    name          TEXT,
    runtime       REAL,
    overview      TEXT NOT NULL DEFAULT '',      -- TMDB synopsis
    wiki_overview TEXT NOT NULL DEFAULT '',      -- richer Wikipedia plot summary
    aired_season  INTEGER,
    aired_number  INTEGER,
    UNIQUE(season, number)
);

CREATE TABLE IF NOT EXISTS evidence (
    id           INTEGER PRIMARY KEY,
    title_id     INTEGER NOT NULL REFERENCES title(id) ON DELETE CASCADE,
    category     TEXT NOT NULL,
    episode_id   INTEGER REFERENCES episode(id) ON DELETE SET NULL,
    verdict      TEXT,                  -- short human-readable finding
    confidence   REAL,                  -- [0,1]
    payload_json TEXT NOT NULL DEFAULT '{}',
    updated_at   TEXT,
    UNIQUE(title_id, category)          -- upsert target: bounded, not a log
);

CREATE TABLE IF NOT EXISTS assignment (
    title_id         INTEGER PRIMARY KEY REFERENCES title(id) ON DELETE CASCADE,
    episode_ids_json TEXT NOT NULL DEFAULT '[]',  -- 1, or 2 for a merged double
    status           TEXT NOT NULL DEFAULT 'unresolved',
    decided_by       TEXT,             -- heuristic:<name> | human | agent
    note             TEXT,
    decided_at       TEXT
);

CREATE TABLE IF NOT EXISTS background (
    disc_name     TEXT PRIMARY KEY,     -- canonical disc basename (state.disc_name)
    episodes_json TEXT NOT NULL,        -- [[season, number], …] the box says are here
    source        TEXT,                 -- provenance, e.g. 'packaging'
    updated_at    TEXT
    -- NOT keyed to disc.id and NOT cascaded: human-entered packaging knowledge
    -- must survive a re-scan (which DELETEs+recreates the disc row). Looked up by
    -- basename — the canonical disc identity — so a re-scanned disc re-adopts it.
);

CREATE TABLE IF NOT EXISTS transcript (
    title_id     INTEGER PRIMARY KEY REFERENCES title(id) ON DELETE CASCADE,
    text         TEXT NOT NULL,         -- joined whisper transcript of the samples
    windows      INTEGER NOT NULL,      -- sampling params it was produced with:
    length       REAL NOT NULL,         --   reuse only when both still match
    updated_at   TEXT
);

CREATE TABLE IF NOT EXISTS frame (
    title_id     INTEGER NOT NULL REFERENCES title(id) ON DELETE CASCADE,
    category     TEXT NOT NULL,        -- the OCR-family source that read it
    image        BLOB,
    mime         TEXT NOT NULL DEFAULT 'image/jpeg',
    source_time  REAL,                 -- seconds into the title
    ocr_text     TEXT,                 -- raw text read from this frame
    extracted_at TEXT,
    UNIQUE(title_id, category)         -- upserts alongside its evidence row
);

CREATE INDEX IF NOT EXISTS ix_title_disc ON title(disc_id);
CREATE INDEX IF NOT EXISTS ix_evidence_title ON evidence(title_id);
CREATE INDEX IF NOT EXISTS ix_evidence_episode ON evidence(episode_id);
CREATE INDEX IF NOT EXISTS ix_frame_title ON frame(title_id);
"""


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def disc_name(path: str | Path) -> str:
    """Canonical human/display name for a disc: its file basename. Volume labels
    (disc.label, read from the filesystem) are routinely blank, wrong, or
    identical across every disc in a set, so the image filename / backup-dir name
    — which the user controls — is the reliable identifier. Drops a disc-image
    extension; leaves backup-dir names as-is."""
    p = Path(path)
    return p.stem if p.suffix.lower() in (".iso", ".img", ".udf", ".nrg") else p.name


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (creating + migrating) the state DB. Rows come back as dict-likes."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")  # concurrent readers, safe writes
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive column migrations for DBs created before a column existed —
    CREATE IF NOT EXISTS never alters an existing table, so add missing columns
    here (idempotent; safe on fresh and old DBs alike)."""
    have = {r["name"] for r in conn.execute("PRAGMA table_info(episode)")}
    if "wiki_overview" not in have:
        conn.execute("ALTER TABLE episode ADD COLUMN "
                     "wiki_overview TEXT NOT NULL DEFAULT ''")


def _now(conn: sqlite3.Connection) -> str:
    """UTC timestamp from SQLite (avoids Date.now-style nondeterminism here)."""
    return conn.execute("SELECT strftime('%Y-%m-%dT%H:%M:%SZ','now')").fetchone()[0]


# ---------------------------------------------------------------------------
# project metadata (kv)
# ---------------------------------------------------------------------------


def set_project(conn: sqlite3.Connection, **kv) -> None:
    conn.executemany(
        "INSERT INTO project(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        [(k, None if v is None else str(v)) for k, v in kv.items()],
    )
    conn.commit()


def get_project(conn: sqlite3.Connection) -> dict[str, str]:
    return {r["key"]: r["value"] for r in conn.execute("SELECT key,value FROM project")}


# ---------------------------------------------------------------------------
# facts: episodes
# ---------------------------------------------------------------------------


def upsert_episodes(conn: sqlite3.Connection, episodes: list[Episode]) -> None:
    conn.executemany(
        "INSERT INTO episode(season,number,name,runtime,overview,"
        "aired_season,aired_number) VALUES(?,?,?,?,?,?,?) "
        "ON CONFLICT(season,number) DO UPDATE SET "
        "name=excluded.name, runtime=excluded.runtime, overview=excluded.overview, "
        "aired_season=excluded.aired_season, aired_number=excluded.aired_number",
        [
            (e.season, e.number, e.name, e.runtime, e.overview,
             e.aired_season, e.aired_number)
            for e in episodes
        ],
    )
    conn.commit()


def _episode_from_row(r: sqlite3.Row) -> Episode:
    return Episode(
        season=r["season"], number=r["number"], name=r["name"],
        runtime=r["runtime"], overview=r["overview"] or "",
        wiki_overview=(r["wiki_overview"] or "") if "wiki_overview" in r.keys() else "",
        aired_season=r["aired_season"], aired_number=r["aired_number"],
    )


def set_wiki_overviews(conn: sqlite3.Connection,
                       by_key: dict[tuple[int, int], str]) -> int:
    """Write Wikipedia plot summaries onto episodes, keyed by (season, number).
    Only touches episodes that already exist; returns the count updated."""
    n = 0
    for (season, number), text in by_key.items():
        cur = conn.execute(
            "UPDATE episode SET wiki_overview=? WHERE season=? AND number=?",
            (text, season, number))
        n += cur.rowcount
    conn.commit()
    return n


def load_episodes(conn: sqlite3.Connection) -> list[Episode]:
    rows = conn.execute(
        "SELECT * FROM episode ORDER BY season,number").fetchall()
    return [_episode_from_row(r) for r in rows]


def episode_id(conn: sqlite3.Connection, season: int, number: int) -> Optional[int]:
    r = conn.execute(
        "SELECT id FROM episode WHERE season=? AND number=?",
        (season, number)).fetchone()
    return r["id"] if r else None


def episode_by_id(conn: sqlite3.Connection, eid: int) -> Optional[Episode]:
    r = conn.execute("SELECT * FROM episode WHERE id=?", (eid,)).fetchone()
    return _episode_from_row(r) if r else None


# ---------------------------------------------------------------------------
# facts: discs + titles
# ---------------------------------------------------------------------------


def add_disc(conn: sqlite3.Connection, disc: Disc) -> int:
    """Insert a scanned disc and its titles; returns the disc's surrogate id.

    Re-scanning the same path replaces the prior disc row and its titles (which
    cascades to that disc's evidence/assignments — a re-scan is a fresh start
    for that disc, matching the old per-disc --merge semantics)."""
    old = conn.execute("SELECT id FROM disc WHERE path=?", (str(disc.path),)).fetchone()
    if old:
        conn.execute("DELETE FROM disc WHERE id=?", (old["id"],))  # cascades
    cur = conn.execute(
        "INSERT INTO disc(path,format,label,season_hint,disc_hint,hb_map_json,scanned_at) "
        "VALUES(?,?,?,?,?,?,?)",
        (str(disc.path), disc.format, disc.label, disc.season_hint,
         disc.disc_hint, json.dumps(disc.hb_map), _now(conn)),
    )
    disc_id = cur.lastrowid
    conn.executemany(
        "INSERT INTO title(disc_id,title_number,duration,chapters_json,n_audio,"
        "n_sub,cells,clips_json,video_format,audio_format,kind,order_key) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (disc_id, t.id, t.duration, json.dumps(t.chapters), t.n_audio,
             t.n_sub, t.cells, json.dumps(list(t.clips)), t.video_format,
             t.audio_format, t.kind, t.order_key)
            for t in disc.titles
        ],
    )
    conn.commit()
    return disc_id


def _title_from_row(r: sqlite3.Row) -> Title:
    return Title(
        id=r["title_number"], duration=r["duration"],
        chapters=json.loads(r["chapters_json"]), n_audio=r["n_audio"],
        n_sub=r["n_sub"], cells=r["cells"],
        clips=tuple(json.loads(r["clips_json"])),
        video_format=r["video_format"], audio_format=r["audio_format"],
        kind=r["kind"], order_key=r["order_key"],
    )


def list_discs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM disc ORDER BY id").fetchall()


def list_titles(conn: sqlite3.Connection, disc_id: int) -> list[sqlite3.Row]:
    """Raw title rows (carry both surrogate `id` and disc-local `title_number`)."""
    return conn.execute(
        "SELECT * FROM title WHERE disc_id=? ORDER BY title_number",
        (disc_id,)).fetchall()


def load_disc(conn: sqlite3.Connection, disc_id: int) -> Disc:
    """Reconstruct a `discs.Disc` (with Title objects) for the heuristic library.

    Note: `Title.id` is the disc-local title_number, as the heuristics expect.
    To map a heuristic's result back to a surrogate row id, use `list_titles`
    (or `title_id`)."""
    d = conn.execute("SELECT * FROM disc WHERE id=?", (disc_id,)).fetchone()
    if d is None:
        raise KeyError(f"no disc with id {disc_id}")
    disc = Disc(
        path=Path(d["path"]), format=d["format"], label=d["label"],
        season_hint=d["season_hint"], disc_hint=d["disc_hint"],
        hb_map={int(k): v for k, v in json.loads(d["hb_map_json"]).items()},
    )
    disc.titles = [_title_from_row(r) for r in list_titles(conn, disc_id)]
    return disc


def title_id(conn: sqlite3.Connection, disc_id: int, title_number: int) -> Optional[int]:
    r = conn.execute(
        "SELECT id FROM title WHERE disc_id=? AND title_number=?",
        (disc_id, title_number)).fetchone()
    return r["id"] if r else None


def set_classification(conn: sqlite3.Connection, disc_id: int,
                       titles: list[Title]) -> None:
    """Persist a classifier's structural verdicts (kind/order_key) back onto
    the disc's title rows. Keyed by the disc-local title_number the heuristic
    library operates on."""
    conn.executemany(
        "UPDATE title SET kind=?, order_key=? WHERE disc_id=? AND title_number=?",
        [(t.kind, t.order_key, disc_id, t.id) for t in titles],
    )
    conn.commit()


# ---------------------------------------------------------------------------
# background (soft packaging knowledge: which episodes a disc holds; per basename)
# ---------------------------------------------------------------------------


def set_background(conn: sqlite3.Connection, disc_name: str,
                   episodes: list[tuple[int, int]], source: str = "packaging") -> None:
    """Assert which (season, number) episodes a disc holds, per box packaging.
    Keyed by canonical basename so it survives a re-scan (see the table note)."""
    conn.execute(
        "INSERT INTO background(disc_name,episodes_json,source,updated_at) "
        "VALUES(?,?,?,?) ON CONFLICT(disc_name) DO UPDATE SET "
        "episodes_json=excluded.episodes_json, source=excluded.source, "
        "updated_at=excluded.updated_at",
        (disc_name, json.dumps([[s, n] for s, n in episodes]), source, _now(conn)))
    conn.commit()


def get_background(conn: sqlite3.Connection,
                   disc_name: str) -> Optional[list[tuple[int, int]]]:
    """The asserted (season, number) episodes for a disc basename, or None."""
    r = conn.execute("SELECT episodes_json FROM background WHERE disc_name=?",
                     (disc_name,)).fetchone()
    return [tuple(x) for x in json.loads(r["episodes_json"])] if r else None


def all_background(conn: sqlite3.Connection) -> dict:
    """{disc_name: [(season, number), …]} for every asserted disc."""
    return {r["disc_name"]: [tuple(x) for x in json.loads(r["episodes_json"])]
            for r in conn.execute("SELECT disc_name,episodes_json FROM background")}


# ---------------------------------------------------------------------------
# transcripts (one per title; the expensive whisper output, judge-independent)
# ---------------------------------------------------------------------------


def get_transcript(conn: sqlite3.Connection, title_id: int,
                   windows: Optional[int] = None,
                   length: Optional[float] = None) -> Optional[str]:
    """The stored transcript for a title, or None. When `windows`/`length` are
    given, only return a HIT whose sampling params match — a different sampling
    would read different dialogue, so a mismatch is a miss (re-transcribe)."""
    r = conn.execute("SELECT text,windows,length FROM transcript WHERE title_id=?",
                     (title_id,)).fetchone()
    if r is None:
        return None
    if windows is not None and r["windows"] != windows:
        return None
    if length is not None and abs(r["length"] - length) > 1e-6:
        return None
    return r["text"]


def put_transcript(conn: sqlite3.Connection, title_id: int, text: str,
                   windows: int, length: float) -> None:
    """Persist a title's whisper transcript (+ the sampling params it used) so a
    later run — or a human/agent — can reuse it without re-running whisper."""
    conn.execute(
        "INSERT INTO transcript(title_id,text,windows,length,updated_at) "
        "VALUES(?,?,?,?,?) ON CONFLICT(title_id) DO UPDATE SET "
        "text=excluded.text, windows=excluded.windows, length=excluded.length, "
        "updated_at=excluded.updated_at",
        (title_id, text, windows, length, _now(conn)))
    conn.commit()


# ---------------------------------------------------------------------------
# evidence (upsert: one row per (title, category))
# ---------------------------------------------------------------------------


def put_evidence(
    conn: sqlite3.Connection,
    title_id: int,
    category: str,
    *,
    episode_id: Optional[int] = None,
    verdict: Optional[str] = None,
    confidence: Optional[float] = None,
    payload: Optional[dict] = None,
) -> None:
    """Record (or replace) a source's current finding for a title.

    `episode_id=None` is a real finding ("not an episode" / "no card read")."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown evidence category {category!r}; "
                         f"expected one of {CATEGORIES}")
    conn.execute(
        "INSERT INTO evidence(title_id,category,episode_id,verdict,confidence,"
        "payload_json,updated_at) VALUES(?,?,?,?,?,?,?) "
        "ON CONFLICT(title_id,category) DO UPDATE SET "
        "episode_id=excluded.episode_id, verdict=excluded.verdict, "
        "confidence=excluded.confidence, payload_json=excluded.payload_json, "
        "updated_at=excluded.updated_at",
        (title_id, category, episode_id, verdict, confidence,
         json.dumps(payload or {}), _now(conn)),
    )
    conn.commit()


def evidence_for_title(conn: sqlite3.Connection, title_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT e.*, ep.season AS ep_season, ep.number AS ep_number, "
        "ep.name AS ep_name FROM evidence e "
        "LEFT JOIN episode ep ON ep.id=e.episode_id "
        "WHERE e.title_id=? ORDER BY e.category",
        (title_id,)).fetchall()


# ---------------------------------------------------------------------------
# assignment (the adjudicated layer)
# ---------------------------------------------------------------------------


def set_assignment(
    conn: sqlite3.Connection,
    title_id: int,
    episode_ids: list[int],
    *,
    status: str = "proposed",
    decided_by: str,
    note: Optional[str] = None,
) -> None:
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}; expected one of {STATUSES}")
    conn.execute(
        "INSERT INTO assignment(title_id,episode_ids_json,status,decided_by,"
        "note,decided_at) VALUES(?,?,?,?,?,?) "
        "ON CONFLICT(title_id) DO UPDATE SET "
        "episode_ids_json=excluded.episode_ids_json, status=excluded.status, "
        "decided_by=excluded.decided_by, note=excluded.note, "
        "decided_at=excluded.decided_at",
        (title_id, json.dumps(episode_ids), status, decided_by, note, _now(conn)),
    )
    conn.commit()


def get_assignment(conn: sqlite3.Connection, title_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM assignment WHERE title_id=?", (title_id,)).fetchone()


# ---------------------------------------------------------------------------
# frames (the image an OCR-family source read — kept for eyeball/VLM re-check)
# ---------------------------------------------------------------------------


def put_frame(
    conn: sqlite3.Connection,
    title_id: int,
    category: str,
    image: bytes,
    *,
    mime: str = "image/jpeg",
    source_time: Optional[float] = None,
    ocr_text: Optional[str] = None,
) -> None:
    """Store (or replace) the frame a source read for a title."""
    conn.execute(
        "INSERT INTO frame(title_id,category,image,mime,source_time,ocr_text,"
        "extracted_at) VALUES(?,?,?,?,?,?,?) "
        "ON CONFLICT(title_id,category) DO UPDATE SET "
        "image=excluded.image, mime=excluded.mime, "
        "source_time=excluded.source_time, ocr_text=excluded.ocr_text, "
        "extracted_at=excluded.extracted_at",
        (title_id, category, image, mime, source_time, ocr_text, _now(conn)),
    )
    conn.commit()


def get_frame(
    conn: sqlite3.Connection, title_id: int, category: Optional[str] = None
) -> Optional[sqlite3.Row]:
    """The stored frame for a title (of a given category, else the first)."""
    if category is not None:
        return conn.execute(
            "SELECT * FROM frame WHERE title_id=? AND category=?",
            (title_id, category)).fetchone()
    return conn.execute(
        "SELECT * FROM frame WHERE title_id=? ORDER BY category LIMIT 1",
        (title_id,)).fetchone()


def frame_categories(conn: sqlite3.Connection, title_id: int) -> list[str]:
    """Which categories have a stored frame (cheap — no BLOB fetched)."""
    return [r["category"] for r in conn.execute(
        "SELECT category FROM frame WHERE title_id=? ORDER BY category",
        (title_id,))]
