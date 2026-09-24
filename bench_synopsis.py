#!/usr/bin/env python3
"""Benchmark synopsis judges against a project whose mapping is known-correct.

Replays the CACHED transcripts in a project DB (no disc access, no whisper/OCR)
through each judge, then scores the answers against the DB's assignments, which
are treated as golden. Measures what the judge choice actually changes:

  * accuracy — the judge's raw top pick, and the final answer after the
    per-season bijection (`assign_by_synopsis`, exactly as `run synopsis` does);
    wrong answers are split from abstentions (a confident-wrong is the costly
    failure — an abstention is recoverable by position/elimination)
  * false claims — titles with a transcript but NO golden assignment
    (featurettes, whisper noise) are run too, as distractors in the bijection;
    one of them claiming an episode is a false claim
  * wall-clock — per-call latency (serial Σ, median, p95)
  * cost — from each response's reported token usage × the price table

Judges: any `claude-*` model id (the production stage-1 prompt, unchanged), and
`jev` / `jev-chunked` (TypeSafe's Jev answering a typed Choice; see
synopsis.jev_rank). Results are cached per judge in --out as JSONL, so a run
resumes where it stopped and re-scoring is free; --fresh re-queries.

  python3 bench_synopsis.py expanse.db --out bench/        # all default judges
  python3 bench_synopsis.py expanse.db --judge jev --judge jev-chunked
  python3 bench_synopsis.py expanse.db --report-only        # re-score cached

Needs ANTHROPIC_API_KEY (claude judges) and TYPESAFE_API_KEY (jev judges).
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import state
import synopsis
from compute import _pool_for, _season_pools
from discs import Episode, log

DEFAULT_JUDGES = ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5-5",
                  "jev", "jev-chunked"]

# $ per million tokens (input, output), first-party list prices. Jev bills input
# only. No prompt caching is used (each title's prompt is unique past the
# instructions), so list price is the real price.
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "jev": (0.042, 0.0),
}

# transcript preference when a title has several cached: exact text first
TRANSCRIPT_ORDER = ("cc", "subtitle-ocr", "audio")


def label(e: Episode) -> str:
    return f"S{e.season:02d}E{e.number:02d}"


@dataclass
class Case:
    title_id: int
    disc: str
    title_number: int
    season_key: int
    gold: set            # {"S01E03"}; empty = a distractor (no golden episode)
    transcript: str
    source: str
    pool: list = field(repr=False)


def load_cases(conn, include_multi: bool = False) -> tuple[list[Case], dict]:
    """Golden titles (every assigned title — confirmed and proposed) plus
    distractors (titles with a non-empty transcript but no assignment), each with
    its cached transcript and the candidate pool `run synopsis` would use.
    Multi-episode golden titles are skipped by default: `run synopsis` never
    judges them (its >1.5× median guard), so they aren't part of the task."""
    seasons, specials = _season_pools(conn)
    all_eps = [e for n in sorted(seasons) for e in seasons[n]] + specials
    by_id = {r["id"]: r for r in conn.execute(
        "SELECT id, season, number FROM episode")}
    gold: dict[int, set] = {}
    for r in conn.execute("SELECT title_id, episode_ids_json FROM assignment "
                          "WHERE status IN ('confirmed','proposed')"):
        eps = [by_id[i] for i in json.loads(r["episode_ids_json"])]
        gold[r["title_id"]] = {f"S{e['season']:02d}E{e['number']:02d}" for e in eps}
    rows = conn.execute(
        "SELECT t.id, t.title_number, d.path, d.season_hint FROM title t "
        "JOIN disc d ON d.id = t.disc_id ORDER BY d.id, t.title_number").fetchall()
    cases, skipped = [], {"multi-episode": 0, "no-transcript": 0}
    for r in rows:
        tid = r["id"]
        text, src = None, None
        for s in TRANSCRIPT_ORDER:
            text = state.get_transcript(conn, tid, 0, 0.0, s)
            if text:
                src = s
                break
        if tid in gold and len(gold[tid]) > 1 and not include_multi:
            skipped["multi-episode"] += 1
            continue
        if not text:
            if tid in gold:
                skipped["no-transcript"] += 1
            continue
        pool = _pool_for(SimpleNamespace(season_hint=r["season_hint"]),
                         seasons, specials, all_eps, False)
        cases.append(Case(tid, state.disc_name(r["path"]), r["title_number"],
                          r["season_hint"] or 0, gold.get(tid, set()),
                          text, src, pool))
    return cases, skipped


# --- judges -----------------------------------------------------------------
# Each returns a result dict: ranked [[label, score], …] best-first, evidence,
# latency_s (successful call(s) only), input/output tokens, requests, retries,
# stop_reason, model (the version that answered), error.

def _price_key(judge: str) -> str:
    return "jev" if judge.startswith("jev") else judge


def judge_claude(case: Case, model: str, max_tokens: int, effort) -> dict:
    pool = [e for e in case.pool if e.synopsis]
    prompt = synopsis.rank_prompt(case.transcript, pool)
    retries = 0
    while True:
        t0 = time.monotonic()
        try:
            reply, meta = synopsis.anthropic_message(model, prompt, max_tokens,
                                                     effort)
            break
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 529) or retries >= 5:
                raise RuntimeError(
                    f"anthropic {e.code}: {e.read().decode(errors='replace')[:300]}")
            retries += 1
            wait = float(e.headers.get("retry-after") or 5 * 2 ** retries)
            log.warning("%s: %d, retry %d in %.0fs", model, e.code, retries, wait)
            time.sleep(wait)
    latency = time.monotonic() - t0
    ranked, evidence = synopsis.parse_ranking(reply, pool)
    u = meta["usage"]
    return {"ranked": [[label(e), s] for e, s in ranked], "evidence": evidence,
            "latency_s": latency, "requests": 1, "retries": retries,
            "input_tokens": u.get("input_tokens", 0)
            + u.get("cache_creation_input_tokens", 0)
            + u.get("cache_read_input_tokens", 0),
            "output_tokens": u.get("output_tokens", 0),
            "stop_reason": meta["stop_reason"], "model": model}


def judge_jev(case: Case, chunked: bool, model: str, show: str) -> dict:
    t0 = time.monotonic()
    ranked, evidence, meta = synopsis.jev_rank(
        case.transcript, case.pool, model, show,
        synopsis.JEV_CHUNK_CHARS if chunked else 0)
    return {"ranked": [[label(e), round(p, 4)] for e, p in ranked],
            "evidence": evidence, "latency_s": time.monotonic() - t0,
            "requests": meta["requests"], "retries": 0,
            "input_tokens": meta["input_tokens"],
            "output_tokens": meta["output_tokens"],
            "stop_reason": None, "model": meta["model"]}


def run_judge(judge: str, case: Case, args, show: str) -> dict:
    if judge.startswith("jev"):
        return judge_jev(case, judge.endswith("-chunked"), args.jev_model, show)
    if judge.startswith("claude"):
        effort = None if "haiku" in judge else args.effort   # Haiku 4.5: no effort
        return judge_claude(case, judge, args.max_tokens, effort)
    raise ValueError(f"unknown judge {judge!r} (want claude-* or jev[-chunked])")


# --- result cache -----------------------------------------------------------

def _results_path(out: Path, judge: str) -> Path:
    return out / f"{judge}.jsonl"


def load_results(out: Path, judge: str) -> dict[int, dict]:
    p = _results_path(out, judge)
    if not p.exists():
        return {}
    res = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            res[r["title_id"]] = r
    return res


def collect(judge: str, cases: list[Case], args, show: str) -> dict[int, dict]:
    """Query `judge` for every case not already cached (errors are retried on
    the next run — they're cached with an `error` and skipped only once OK)."""
    out = Path(args.out)
    done = {} if args.fresh else load_results(out, judge)
    todo = [c for c in cases if c.title_id not in done or done[c.title_id].get("error")]
    if args.fresh:
        _results_path(out, judge).unlink(missing_ok=True)
    if not todo:
        return done
    log.info("%s: %d titles to judge (%d cached)", judge, len(todo),
             len(cases) - len(todo))
    fh = _results_path(out, judge).open("a")
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(run_judge, judge, c, args, show): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001 — record, keep benchmarking
                r = {"error": str(e), "ranked": [], "latency_s": 0,
                     "input_tokens": 0, "output_tokens": 0, "requests": 0}
                log.warning("%s title %d: %s", judge, c.title_id, e)
            r["title_id"] = c.title_id
            done[c.title_id] = r
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            top = r["ranked"][0][0] if r["ranked"] else "—"
            log.info("%s [%d/%d] %s t%d → %s (gold %s) %.1fs", judge, i, len(todo),
                     c.disc, c.title_number, top, ",".join(sorted(c.gold)) or "none",
                     r["latency_s"])
    fh.close()
    log.info("%s: batch wall-clock %.0fs at concurrency %d", judge,
             time.monotonic() - t0, args.concurrency)
    return done


# --- scoring ----------------------------------------------------------------

def _verdict(pick, gold: set) -> str:
    if pick is None:
        return "abstain"
    if not gold:
        return "false-claim"
    return "correct" if pick in gold else "wrong"


def score(judge: str, cases: list[Case], results: dict[int, dict]) -> dict:
    """Top-1 and post-bijection verdicts per case, plus latency/cost totals."""
    eps = {label(e): e for c in cases for e in c.pool}
    # the bijection, per season, over golden + distractor titles alike
    assigned: dict[int, str] = {}
    for season in sorted({c.season_key for c in cases}):
        rows = []
        for c in cases:
            r = results.get(c.title_id)
            if c.season_key != season or not r or r.get("error"):
                continue
            rows.append((c.title_id,
                         [(eps[l], s) for l, s in r["ranked"] if l in eps]))
        pool = [e for c in cases if c.season_key == season for e in c.pool]
        uniq = list({label(e): e for e in pool}.values())
        for tid, (e, _s, _rank) in synopsis.assign_by_synopsis(rows, uniq).items():
            assigned[tid] = label(e)

    per_title, top1, final = [], {}, {}
    lat, cost_in, cost_out, tin, tout, errors, truncated = [], 0.0, 0.0, 0, 0, 0, 0
    pin, pout = PRICES.get(_price_key(judge), (0.0, 0.0))
    for c in cases:
        r = results.get(c.title_id)
        if r is None:
            continue
        if r.get("error"):
            errors += 1
        pick = r["ranked"][0][0] if r["ranked"] else None
        v1 = _verdict(pick, c.gold)
        vf = _verdict(assigned.get(c.title_id), c.gold)
        top1[v1] = top1.get(v1, 0) + 1
        final[vf] = final.get(vf, 0) + 1
        if r.get("latency_s"):
            lat.append(r["latency_s"])
        tin += r.get("input_tokens", 0)
        tout += r.get("output_tokens", 0)
        truncated += r.get("stop_reason") == "max_tokens"
        per_title.append({
            "title_id": c.title_id, "disc": c.disc, "title": c.title_number,
            "gold": ",".join(sorted(c.gold)) or "-", "top1": pick or "-",
            "final": assigned.get(c.title_id, "-"), "top1_verdict": v1,
            "final_verdict": vf, "evidence": r.get("evidence", r.get("error", ""))})
    cost_in, cost_out = tin * pin / 1e6, tout * pout / 1e6
    n_gold = sum(1 for c in cases if c.gold and c.title_id in results)
    lat_sorted = sorted(lat)
    return {
        "judge": judge, "n_golden": n_gold,
        "n_distractor": sum(1 for c in cases if not c.gold and c.title_id in results),
        "top1": top1, "final": final, "errors": errors, "truncated": truncated,
        "latency_serial_s": sum(lat),
        "latency_median_s": statistics.median(lat) if lat else 0.0,
        "latency_p95_s": (lat_sorted[max(0, math.ceil(0.95 * len(lat_sorted)) - 1)]
                          if lat else 0.0),   # nearest-rank
        "input_tokens": tin, "output_tokens": tout,
        "cost_usd": cost_in + cost_out,
        "cost_per_title_usd": (cost_in + cost_out) / max(1, len(per_title)),
        "models_seen": sorted({r.get("model") for r in results.values()
                               if r.get("model")}),
        "per_title": per_title,
    }


def _pct(n, d):
    return f"{100 * n / d:5.1f}%" if d else "  —  "


def render(summaries: list[dict]) -> str:
    """Markdown summary table + each judge's misses."""
    hdr = ("| judge | final acc | final wrong | final abstain | false claims "
           "| top-1 acc | Σ latency | median | p95 | cost | $/title | errors |")
    lines = [hdr, "|" + "---|" * 12]
    for s in summaries:
        g, f, t = s["n_golden"], s["final"], s["top1"]
        lines.append(
            f"| {s['judge']} | {f.get('correct', 0)}/{g} {_pct(f.get('correct', 0), g)} "
            f"| {f.get('wrong', 0)} | {sum(1 for p in s['per_title'] if p['gold'] != '-' and p['final_verdict'] == 'abstain')} "
            f"| {f.get('false-claim', 0)}/{s['n_distractor']} "
            f"| {t.get('correct', 0)}/{g} {_pct(t.get('correct', 0), g)} "
            f"| {s['latency_serial_s']:.0f}s | {s['latency_median_s']:.1f}s "
            f"| {s['latency_p95_s']:.1f}s | ${s['cost_usd']:.4f} "
            f"| ${s['cost_per_title_usd']:.5f} | {s['errors']}"
            + (f" ({s['truncated']} truncated)" if s["truncated"] else "") + " |")
    for s in summaries:
        miss = [p for p in s["per_title"]
                if p["final_verdict"] not in ("correct",)
                and not (p["gold"] == "-" and p["final_verdict"] == "abstain")]
        if not miss:
            continue
        lines += ["", f"**{s['judge']}** misses (final answer):", ""]
        for p in miss:
            lines.append(f"- {p['disc']} t{p['title']}: gold {p['gold']}, "
                         f"final {p['final']} ({p['final_verdict']}), top-1 "
                         f"{p['top1']} — {p['evidence'][:140]}")
    return "\n".join(lines)


def connect_readonly(path: str) -> sqlite3.Connection:
    """The golden DB, opened READ-ONLY — `state.connect` would run the schema
    script + migrations and flip WAL mode, i.e. write to the reference project."""
    conn = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("db", help="project DB whose assignments are golden")
    ap.add_argument("--judge", action="append",
                    help=f"judge to run (repeatable; default: {' '.join(DEFAULT_JUDGES)})")
    ap.add_argument("--out", default="bench-synopsis",
                    help="results dir (per-judge JSONL cache + summary.json/.md)")
    ap.add_argument("--fresh", action="store_true", help="ignore cached results")
    ap.add_argument("--report-only", action="store_true",
                    help="score cached results only; make no API calls")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="parallel calls per judge (latency is per-call, so "
                         "this changes the batch time, not the reported numbers)")
    ap.add_argument("--limit", type=int, help="only the first N cases (smoke test)")
    ap.add_argument("--include-multi", action="store_true",
                    help="also judge multi-episode golden titles (a hit = any member)")
    ap.add_argument("--max-tokens", type=int, default=16000,
                    help="claude output cap INCLUDING thinking (default 16000; "
                         "production uses 2048, which can truncate thinking models)")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"],
                    help="claude effort (not sent to Haiku); default = model default")
    ap.add_argument("--jev-model", default=synopsis.JEV_MODEL,
                    help="TypeSafe model id (pin e.g. jev-1.13.0 for repeatability)")
    args = ap.parse_args(argv)

    conn = connect_readonly(args.db)
    show = state.get_project(conn).get("show_name", "")
    cases, skipped = load_cases(conn, args.include_multi)
    if args.limit:
        cases = cases[:args.limit]
    n_gold = sum(1 for c in cases if c.gold)
    log.info("%d golden titles + %d distractors (skipped: %s)", n_gold,
             len(cases) - n_gold, skipped)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for judge in args.judge or DEFAULT_JUDGES:
        results = (load_results(out, judge) if args.report_only
                   else collect(judge, cases, args, show))
        if results:
            summaries.append(score(judge, cases, results))
    (out / "summary.json").write_text(json.dumps(
        {"db": str(args.db), "show": show, "skipped": skipped,
         "prices_per_mtok": PRICES, "judges": summaries}, indent=2))
    report = render(summaries)
    (out / "summary.md").write_text(report + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
