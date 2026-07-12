"""Run the HandBrake transcode jobs directly (the managed alternative to the
emitted bash rip script), tagging each output and skipping work that's already
up to date.

Why in the tool rather than a bash script (see README/DESIGN):
  * every output carries its identity as **Matroska tags** (episode name, show,
    season/episode, TMDB id) so a file is self-describing to Plex/Jellyfin and
    to us — no reliance on the filename alone;
  * a `VJ_RECIPE` tag records a hash of exactly the inputs that determine the
    encoded bytes (source disc + HandBrake title index + preset + extra opts),
    so a re-run **re-encodes only what actually changed** in the project DB and
    skips the rest — the idempotent-regeneration property.

The encode inputs (`VJ_RECIPE`) are kept separate from the descriptive metadata
(`VJ_META`: names/numbers/ids): a change to the former means a real re-encode; a
change to only the latter is a cheap rename + re-tag of the existing file, never
a multi-hour re-encode. Existing outputs are indexed by their `VJ_EPISODES` tag
(not their path), so a rename is detected as a rename, not a re-encode + orphan.

Tagging uses mkvtoolnix (`mkvpropedit` to write, `mkvextract` to read) — a
separate program invoked at arm's length, like ffmpeg/HandBrake. Outputs are
assumed to be Matroska (`.mkv`, as `suggested_filename` produces).
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from discs import log, run
from export import build_records
from identify import _record_video_majority

HANDBRAKE = "HandBrakeCLI"
MKVPROPEDIT = "mkvpropedit"
MKVEXTRACT = "mkvextract"


# --------------------------------------------------------------------------- #
# recipe + metadata identity
# --------------------------------------------------------------------------- #

def _sha(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def encode_recipe(record: dict, preset: str, opts: list[str]) -> dict:
    """The inputs that determine the encoded BYTES. A change here => re-encode.
    Deliberately does NOT include the tool version or the output name — only
    what the user controls about the source and the encode."""
    return {"source": os.path.basename(str(record["image"]).rstrip("/")),
            "title": record["title"], "preset": preset, "opts": list(opts)}


def meta_fields(record: dict, show: str, year: Optional[int],
                tmdb_id: Optional[int]) -> dict:
    """Descriptive metadata. A change here => rename/re-tag, never a re-encode."""
    return {"SHOW": show, "YEAR": str(year or ""), "TMDB": str(tmdb_id or ""),
            "SEASON": str(record["season"]),
            "EPISODES": _episode_tag_for(record),
            "TITLE": record["episode_name"]}


def _episode_tag_for(record: dict) -> str:
    # reuse identify._episode_tag, which wants Episode objects; we only have
    # numbers here, so build the SxxEyy / SxxEyy-Ezz string directly.
    s, nums = record["season"], record["episodes"]
    if len(nums) == 1:
        return f"S{s:02d}E{nums[0]:02d}"
    return f"S{s:02d}E{nums[0]:02d}-E{nums[-1]:02d}"


# --------------------------------------------------------------------------- #
# Matroska tags: write (mkvpropedit) + read (mkvextract)
# --------------------------------------------------------------------------- #

def tags_xml(simples: dict) -> str:
    """Build a Matroska global-tags XML doc from a flat {name: value} dict."""
    tags = ET.Element("Tags")
    tag = ET.SubElement(tags, "Tag")
    targets = ET.SubElement(tag, "Targets")
    ET.SubElement(targets, "TargetTypeValue").text = "50"   # episode/movie level
    for name, value in simples.items():
        if value in (None, ""):
            continue
        simple = ET.SubElement(tag, "Simple")
        ET.SubElement(simple, "Name").text = str(name)
        ET.SubElement(simple, "String").text = str(value)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE Tags SYSTEM "matroskatags.dtd">\n'
            + ET.tostring(tags, encoding="unicode"))


def parse_tags(xml_text: str) -> dict:
    """Flatten a Matroska tags XML doc to {name: value}."""
    out: dict = {}
    if not xml_text.strip():
        return out
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for simple in root.iter("Simple"):
        name = simple.findtext("Name")
        value = simple.findtext("String")
        if name is not None:
            out[name] = value or ""
    return out


def read_tags(path: Path) -> dict:
    """Read a file's Matroska tags via mkvextract; {} if none/unreadable."""
    try:
        r = subprocess.run([MKVEXTRACT, str(path), "tags", "-"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return {}
    return parse_tags(r.stdout) if r.returncode == 0 else {}


def write_tags(path: Path, simples: dict, workdir: Path) -> None:
    xmlf = workdir / (path.stem + ".tags.xml")
    xmlf.write_text(tags_xml(simples))
    res = run([MKVPROPEDIT, str(path), "--tags", f"global:{xmlf}"], timeout=120)
    xmlf.unlink(missing_ok=True)
    if res.returncode != 0:   # don't leave an untagged file looking done
        raise OSError(f"mkvpropedit failed ({res.returncode}): "
                      f"{(res.stderr or '').strip()[:200]}")


def _is_matroska(path: Path) -> bool:
    """EBML magic bytes. HandBrake picks its muxer from the output extension, so
    a wrong temp name (or a preset defaulting to MP4) can yield a non-MKV file;
    check the container so that fails LOUDLY instead of shipping an untagged MP4
    with a .mkv name (which then can't be tagged and never matches on re-run)."""
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == b"\x1a\x45\xdf\xa3"
    except OSError:
        return False


def _tool_missing() -> Optional[str]:
    for t in (HANDBRAKE, MKVPROPEDIT, MKVEXTRACT):
        if not shutil.which(t):
            return t
    return None


# --------------------------------------------------------------------------- #
# per-episode action decision (pure — unit-tested)
# --------------------------------------------------------------------------- #

def decide_action(recipe_hash: str, meta_hash: str, target: str,
                  existing: Optional[dict], force: bool = False) -> str:
    """encode | reencode | rename | retag | skip.

    `existing` is None, or {"path", "recipe", "meta"} read from the output tree.
    Split so a source/preset change re-encodes but a rename/rename-only-metadata
    change is cheap."""
    if existing is None:
        return "encode"
    if force or existing.get("recipe") != recipe_hash:
        return "reencode"
    if existing.get("path") != target:
        return "rename"
    if existing.get("meta") != meta_hash:
        return "retag"
    return "skip"


# --------------------------------------------------------------------------- #
# output-tree index (find existing outputs by their VJ_EPISODES tag)
# --------------------------------------------------------------------------- #

def index_outputs(prefix: Path) -> dict:
    """Map VJ_EPISODES tag -> {path, recipe, meta} for every tagged .mkv under
    `prefix`. Lets us match by identity, so a renamed episode is a rename (move)
    not a re-encode-and-orphan."""
    idx: dict = {}
    if not prefix.is_dir():
        return idx
    for mkv in prefix.rglob("*.mkv"):
        tags = read_tags(mkv)
        key = tags.get("VJ_EPISODES")
        if key:
            idx[key] = {"path": str(mkv), "recipe": tags.get("VJ_RECIPE"),
                        "meta": tags.get("VJ_META")}
    return idx


# --------------------------------------------------------------------------- #
# the verb
# --------------------------------------------------------------------------- #

def _plan(conn, args) -> tuple[list[dict], dict]:
    import state
    records = build_records(conn, include_proposed=args.include_proposed)
    proj = state.get_project(conn)
    show = proj.get("show_name") or "Show"
    year = int(proj["year"]) if proj.get("year") not in (None, "None") else None
    tmdb_id = int(proj["tmdb_id"]) if proj.get("tmdb_id") else None

    preset = args.handbrake_preset
    preset_alt = getattr(args, "handbrake_preset_alt", None) or preset
    raw_opts = getattr(args, "handbrake_opts", None)
    opts = shlex.split(raw_opts) if isinstance(raw_opts, str) else (raw_opts or [])
    prefix = Path(args.output_prefix or ".")

    majority = _record_video_majority(records)
    steps = []
    for r in records:
        if r.get("kind") != "episode":
            continue
        outlier = bool(majority and r.get("video_format")
                       and r["video_format"] != majority)
        eff_preset = preset_alt if outlier else preset
        recipe = encode_recipe(r, eff_preset, opts)
        recipe_hash = _sha(recipe)
        meta = meta_fields(r, show, year, tmdb_id)
        meta_hash = _sha(meta)
        target = str(prefix / r["suggested_filename"])
        steps.append({"record": r, "preset": eff_preset, "opts": opts,
                      "recipe": recipe, "recipe_hash": recipe_hash,
                      "meta": meta, "meta_hash": meta_hash, "target": target,
                      "ep_key": meta["EPISODES"], "outlier": outlier})
    ctx = {"show": show, "year": year, "tmdb_id": tmdb_id, "prefix": prefix}
    return steps, ctx


def run_transcode(conn, args) -> dict:
    missing = _tool_missing()
    if missing and not args.dry_run:
        return {"ok": False, "error": "missing-tool",
                "message": f"{missing} not on PATH — install HandBrake and "
                           "mkvtoolnix (dnf install mkvtoolnix)"}
    steps, ctx = _plan(conn, args)
    if not steps:
        return {"ok": False, "error": "nothing-to-transcode",
                "message": ("no confirmed assignments"
                            + ("" if args.include_proposed
                               else " (try --include-proposed)"))}
    idx = index_outputs(ctx["prefix"])
    for s in steps:
        s["action"] = decide_action(s["recipe_hash"], s["meta_hash"],
                                     s["target"], idx.get(s["ep_key"]),
                                     force=args.force)

    counts: dict = {}
    for s in steps:
        counts[s["action"]] = counts.get(s["action"], 0) + 1
    if args.dry_run:
        return {"ok": True, "dry_run": True, "plan": counts,
                "steps": [{"ep": s["ep_key"], "action": s["action"],
                           "target": s["target"]} for s in steps]}

    import tempfile
    done = {"encoded": 0, "renamed": 0, "retagged": 0, "skipped": 0, "failed": 0}
    with tempfile.TemporaryDirectory(prefix="vj-tc-", dir=args.scratch_dir) as tmp:
        workdir = Path(tmp)
        for s in steps:
            try:
                _run_step(s, idx.get(s["ep_key"]), ctx, workdir, done)
            except (OSError, subprocess.SubprocessError) as e:
                done["failed"] += 1
                log.warning("transcode: %s failed on %s (%s)",
                            s["action"], s["ep_key"], e)
    return {"ok": True, "plan": counts, "result": done}


def _simples(s: dict, ctx: dict) -> dict:
    r = s["record"]
    return {**s["meta"],                       # SHOW/YEAR/TMDB/SEASON/EPISODES/TITLE
            "VJ_RECIPE": s["recipe_hash"], "VJ_META": s["meta_hash"],
            "VJ_EPISODES": s["ep_key"],
            "VJ_SOURCE": f"{s['recipe']['source']} t{r['title']}",
            "VJ_PRESET": s["preset"]}


def _run_step(s: dict, existing: Optional[dict], ctx: dict, workdir: Path,
              done: dict) -> None:
    target = Path(s["target"])
    if s["action"] == "skip":
        done["skipped"] += 1
        return
    if s["action"] == "retag":
        write_tags(Path(existing["path"]), _simples(s, ctx), workdir)
        done["retagged"] += 1
        return
    if s["action"] == "rename":
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(existing["path"], target)          # same filesystem move
        write_tags(target, _simples(s, ctx), workdir)
        done["renamed"] += 1
        return
    # encode | reencode
    if s["action"] == "reencode" and existing and os.path.exists(existing["path"]):
        os.remove(existing["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    # The temp keeps a .mkv extension AND we force `--format av_mkv`: HandBrake
    # chooses its muxer from the output extension, so a bare ".part" makes it
    # fall back to the preset's container (MP4) and write MP4 bytes into a .mkv
    # name. Both guards ensure Matroska; _is_matroska below verifies it.
    part = target.with_name(target.stem + ".vjpart.mkv")
    r = s["record"]
    cmd = [HANDBRAKE, "--format", "av_mkv", *s["opts"], "-i", str(r["image"]),
           "-t", str(r["title"]), "--preset", s["preset"], "-o", str(part)]
    log.info("transcode: encoding %s -> %s", s["ep_key"], target.name)
    # Do NOT capture: a multi-hour HandBrake run streams continuous progress to
    # stderr; buffering it (as discs.run does) would grow unbounded. No timeout.
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if res.returncode != 0:
        raise OSError(f"HandBrake exited {res.returncode}")
    if not (part.exists() and part.stat().st_size > 0):
        raise OSError("HandBrake produced no output")
    if not _is_matroska(part):
        part.unlink(missing_ok=True)
        raise OSError("HandBrake did not produce a Matroska file (check preset/format)")
    write_tags(part, _simples(s, ctx), workdir)
    os.replace(part, target)
    done["encoded"] += 1
