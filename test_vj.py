"""Unit tests for the vj state/compute/review/export layers.

No disc images or network: discs/titles are constructed in-memory and inserted
straight into a temp state DB, and the heuristics that need only metadata
(classify + align) run against those rows.

Run: python3 -m unittest test_vj -v
"""
import tempfile
import unittest
from pathlib import Path

import contextlib
import io
import json as _json
import types

import auto as auto_mod
import compute
import export as export_mod
import review
import state
import vj
from discs import Disc, Episode, Title


def auto_args(**kw):
    base = dict(threshold=0.5, ocr="never", vlm_model="x", ocr_accept=0.8,
                ocr_engine="auto", scratch_dir=None,
                ollama_host="http://localhost:1")
    base.update(kw)
    return types.SimpleNamespace(**base)


def ep(season, number, name, runtime):
    return Episode(season=season, number=number, name=name, runtime=runtime)


def title(tid, dur, chapters):
    return Title(id=tid, duration=float(dur), chapters=list(chapters),
                 n_audio=2, n_sub=1, cells=len(chapters) or 1)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "t.db")
        self.conn = state.connect(self.db)
        state.set_project(self.conn, tmdb_id=999, show_name="Test Show",
                          year=2020, episode_order="aired")

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def add_disc(self, titles, season_hint=1, disc_hint=1, path="/d/s1d1.iso"):
        d = Disc(path=Path(path), format="dvd", label="S1D1",
                 season_hint=season_hint, disc_hint=disc_hint)
        d.titles = titles
        return state.add_disc(self.conn, d)


class StateTests(Base):
    def test_episode_roundtrip(self):
        state.upsert_episodes(self.conn, [ep(1, 1, "A", 1320.0), ep(1, 2, "B", 1320.0)])
        eps = state.load_episodes(self.conn)
        self.assertEqual([(e.season, e.number) for e in eps], [(1, 1), (1, 2)])

    def test_evidence_is_bounded_per_category(self):
        state.upsert_episodes(self.conn, [ep(1, 1, "A", 1320.0)])
        did = self.add_disc([title(1, 1320, [660, 660])])
        tid = state.title_id(self.conn, did, 1)
        eid = state.episode_id(self.conn, 1, 1)
        state.put_evidence(self.conn, tid, "runtime-align", episode_id=eid, confidence=0.9)
        state.put_evidence(self.conn, tid, "title-card-ocr", episode_id=eid, confidence=1.0)
        state.put_evidence(self.conn, tid, "title-card-ocr", episode_id=eid,
                           confidence=0.7, verdict="reread")   # upsert, not append
        rows = state.evidence_for_title(self.conn, tid)
        self.assertEqual(len(rows), 2)
        ocr = [r for r in rows if r["category"] == "title-card-ocr"][0]
        self.assertEqual(ocr["verdict"], "reread")
        self.assertAlmostEqual(ocr["confidence"], 0.7)

    def test_frame_roundtrip_and_cascade(self):
        state.upsert_episodes(self.conn, [ep(1, 1, "A", 1320.0)])
        did = self.add_disc([title(1, 1320, [660, 660])])
        tid = state.title_id(self.conn, did, 1)
        state.put_frame(self.conn, tid, "title-card-ocr", b"\xff\xd8\xff\xe0",
                        source_time=1300.0, ocr_text="A")
        f = state.get_frame(self.conn, tid)
        self.assertEqual(f["ocr_text"], "A")
        self.assertEqual(state.frame_categories(self.conn, tid), ["title-card-ocr"])
        # re-scanning the disc cascades away its frames
        self.add_disc([title(1, 1320, [660, 660])])
        self.assertIsNone(state.get_frame(self.conn, tid))

    def test_bad_category_rejected(self):
        state.upsert_episodes(self.conn, [ep(1, 1, "A", 1320.0)])
        did = self.add_disc([title(1, 1320, [660, 660])])
        tid = state.title_id(self.conn, did, 1)
        with self.assertRaises(ValueError):
            state.put_evidence(self.conn, tid, "bogus", confidence=1.0)


class AlignEvidenceTests(Base):
    def test_align_writes_runtime_evidence(self):
        # three runtime-separable episodes so alignment is unambiguous
        state.upsert_episodes(self.conn, [
            ep(1, 1, "One", 1200.0), ep(1, 2, "Two", 1400.0),
            ep(1, 3, "Three", 1600.0)])
        self.add_disc([
            title(1, 1200, [600, 600]),
            title(2, 1400, [700, 700, 300]),
            title(3, 1600, [800, 800])])
        r = compute.run_align(self.conn, args=None)
        self.assertTrue(r["ok"])
        self.assertEqual(r["evidence"], 3)
        # each title got a runtime-align row pointing at the right episode
        got = {}
        for t in (1, 2, 3):
            tid = state.title_id(self.conn, 1, t)
            row = [x for x in state.evidence_for_title(self.conn, tid)
                   if x["category"] == "runtime-align"][0]
            got[t] = (row["ep_season"], row["ep_number"])
        self.assertEqual(got, {1: (1, 1), 2: (1, 2), 3: (1, 3)})


class CollisionTests(Base):
    def test_same_episode_collision_flagged(self):
        # two proposed assignments to the same episode (metadata collision, no
        # OCR) -> both surfaced as review, each pointing at the other
        state.upsert_episodes(self.conn, [ep(1, 1, "A", 1320.0), ep(1, 2, "B", 1320.0)])
        t1 = title(1, 1320, [660, 660]); t1.kind = "episode-candidate"
        t2 = title(2, 1320, [600, 720]); t2.kind = "episode-candidate"
        did = self.add_disc([t1, t2])
        a, b = state.title_id(self.conn, did, 1), state.title_id(self.conn, did, 2)
        e1 = state.episode_id(self.conn, 1, 1)
        for tid in (a, b):
            state.set_assignment(self.conn, tid, [e1], status="proposed",
                                 decided_by="heuristic:runtime-align")
        g = {x["title_id"]: x for x in review.gaps(self.conn)["gaps"]}
        self.assertTrue(g[a]["collision"] and g[b]["collision"])
        self.assertEqual(g[a]["suggestion"]["action"], "review")
        self.assertIn("also claimed", g[a]["suggestion"]["why"])

    def test_no_collision_when_distinct(self):
        state.upsert_episodes(self.conn, [ep(1, 1, "A", 1320.0), ep(1, 2, "B", 1320.0)])
        t1 = title(1, 1320, [660, 660]); t1.kind = "episode-candidate"
        t2 = title(2, 1320, [600, 720]); t2.kind = "episode-candidate"
        did = self.add_disc([t1, t2])
        a, b = state.title_id(self.conn, did, 1), state.title_id(self.conn, did, 2)
        state.set_assignment(self.conn, a, [state.episode_id(self.conn, 1, 1)],
                             status="proposed", decided_by="heuristic:runtime-align")
        state.set_assignment(self.conn, b, [state.episode_id(self.conn, 1, 2)],
                             status="proposed", decided_by="heuristic:runtime-align")
        # clean, distinct proposals -> not gaps at all
        self.assertEqual(review.gaps(self.conn)["gaps"], [])


class StreamSignatureComputeTests(Base):
    def _seed(self):
        state.upsert_episodes(
            self.conn, [ep(1, k, f"E{k}", 1320.0) for k in range(1, 5)])
        # four episodes at 5A/2S + one episode-length extra at 1A/0S
        ts = [Title(id=k, duration=1320.0, chapters=[1320.0],
                    n_audio=5, n_sub=2, kind="episode-candidate")
              for k in range(1, 5)]
        ts.append(Title(id=9, duration=1350.0, chapters=[1350.0],
                        n_audio=1, n_sub=0, kind="episode-candidate"))
        return self.add_disc(ts)

    def test_run_streams_flags_the_odd_layout(self):
        did = self._seed()
        res = compute.run_streams(self.conn, auto_args())
        self.assertTrue(res["ok"])
        self.assertEqual(res["evidence"], 5)     # all five titles get a row
        self.assertEqual(res["flagged"], 1)      # one is extra-like
        tid9 = state.title_id(self.conn, did, 9)
        ev = [e for e in state.evidence_for_title(self.conn, tid9)
              if e["category"] == "stream-signature"][0]
        self.assertIn("extra-like", ev["verdict"])
        self.assertIsNone(ev["episode_id"])      # no episode identity claimed

    def test_no_signal_when_counts_absent(self):
        # a Blu-ray scanned without HandBrake: every title has 0/0 -> no rows
        state.upsert_episodes(self.conn, [ep(1, 1, "A", 1320.0)])
        ts = [Title(id=k, duration=1320.0, chapters=[1320.0],
                    n_audio=0, n_sub=0) for k in range(1, 4)]
        self.add_disc(ts)
        res = compute.run_streams(self.conn, auto_args())
        self.assertEqual(res["evidence"], 0)

    def test_gaps_flags_extra_like_proposed_episode(self):
        did = self._seed()
        compute.run_streams(self.conn, auto_args())
        tid9 = state.title_id(self.conn, did, 9)
        # metadata-only align could propose the extra as an episode
        state.set_assignment(self.conn, tid9,
                             [state.episode_id(self.conn, 1, 4)],
                             status="proposed", decided_by="heuristic:align")
        row = [x for x in review.gaps(self.conn)["gaps"]
               if x["title_id"] == tid9][0]
        self.assertEqual(row["stream"]["class"], "extra")
        self.assertIn("extra", row["reason"])


class SynopsisGuardTests(Base):
    def test_multi_episode_title_is_skipped_not_transcribed(self):
        # three ~22-min episodes + one ~44-min play-all, all episode-candidates.
        # The play-all holds two episodes' dialogue, so the synopsis path must
        # skip it (not full-transcribe 44 min and match it to one episode).
        state.upsert_episodes(
            self.conn, [ep(1, k, f"E{k}", 1320.0) for k in range(1, 4)])
        ts = [Title(id=k, duration=1320.0, chapters=[1320.0], n_audio=2, n_sub=1,
                    kind="episode-candidate") for k in range(1, 4)]
        ts.append(Title(id=9, duration=2640.0, chapters=[2640.0], n_audio=2,
                        n_sub=1, kind="episode-candidate"))   # 2× play-all
        did = self.add_disc(ts)

        transcribed = []
        orig_ft, orig_rc = compute.full_transcript, compute.rank_candidates
        compute.full_transcript = lambda disc, title, wd: (
            transcribed.append(title.id) or f"dialogue {title.id}")
        compute.rank_candidates = lambda tr, pool, model, host: ([(pool[0], 6)], "ev")
        try:
            args = auto_args(disc=did, title=None, all=False, judge_model=None,
                             synopsis_windows=None, synopsis_length=None,
                             retranscribe=False, synopsis_source="tmdb",
                             transcript_source="audio", include_specials=False)
            res = compute.run_synopsis(self.conn, args)
        finally:
            compute.full_transcript, compute.rank_candidates = orig_ft, orig_rc

        self.assertTrue(res["ok"])
        self.assertEqual(sorted(transcribed), [1, 2, 3])   # play-all never ripped
        tid9 = state.title_id(self.conn, did, 9)
        ev = [e for e in state.evidence_for_title(self.conn, tid9)
              if e["category"] == "synopsis"][0]
        self.assertIn("multi-episode", ev["verdict"])
        self.assertIsNone(ev["episode_id"])                # no identity claimed

    def test_no_guard_below_three_targets(self):
        # with <3 candidates there's no stable median, so nothing is guarded
        state.upsert_episodes(
            self.conn, [ep(1, k, f"E{k}", 1320.0) for k in range(1, 3)])
        ts = [Title(id=1, duration=1320.0, chapters=[1320.0], n_audio=2, n_sub=1,
                    kind="episode-candidate"),
              Title(id=2, duration=5000.0, chapters=[5000.0], n_audio=2, n_sub=1,
                    kind="episode-candidate")]
        did = self.add_disc(ts)
        transcribed = []
        orig_ft, orig_rc = compute.full_transcript, compute.rank_candidates
        compute.full_transcript = lambda disc, title, wd: (
            transcribed.append(title.id) or f"d{title.id}")
        compute.rank_candidates = lambda tr, pool, model, host: ([(pool[0], 6)], "ev")
        try:
            compute.run_synopsis(self.conn, auto_args(
                disc=did, title=None, all=False, judge_model=None,
                synopsis_windows=None, synopsis_length=None, retranscribe=False,
                synopsis_source="tmdb", transcript_source="audio",
                include_specials=False))
        finally:
            compute.full_transcript, compute.rank_candidates = orig_ft, orig_rc
        self.assertEqual(sorted(transcribed), [1, 2])      # both transcribed


class BackgroundHintTests(Base):
    def test_parse_ranges(self):
        self.assertEqual(vj._parse_ranges("1-4,6,8-9"), [1, 2, 3, 4, 6, 8, 9])
        self.assertEqual(vj._parse_ranges("3"), [3])
        self.assertEqual(vj._parse_ranges(" 2 , 1 "), [1, 2])   # order/space-tolerant
        with self.assertRaises(ValueError):
            vj._parse_ranges("4-1")                             # backwards

    def test_background_roundtrip_and_survives_rescan(self):
        state.set_background(self.conn, "s1d1", [(1, 3), (1, 4)])
        self.assertEqual(state.get_background(self.conn, "s1d1"), [(1, 3), (1, 4)])
        # a re-scan DELETEs+recreates the disc row; the basename-keyed hint stays
        self.add_disc([title(1, 1320, [1320])], path="/d/s1d1.iso")
        self.add_disc([title(1, 1320, [1320])], path="/d/s1d1.iso")
        self.assertEqual(state.get_background(self.conn, "s1d1"), [(1, 3), (1, 4)])

    def test_hint_steers_run_align(self):
        # 4 same-runtime episodes; a 2-title disc the box says holds E3,E4
        state.upsert_episodes(
            self.conn, [ep(1, k, f"E{k}", 1320.0) for k in range(1, 5)])
        # same total runtime (so alignment is ambiguous) but distinct chapter
        # layouts (so they aren't taken for duplicates and junked)
        did = self.add_disc([title(1, 1320, [660, 660]),
                             title(2, 1320, [440, 440, 440])],
                            path="/d/THE_SHOW_S1D2.iso")
        state.set_background(self.conn, state.disc_name("/d/THE_SHOW_S1D2.iso"),
                            [(1, 3), (1, 4)])
        compute.run_align(self.conn, args=None)
        got = []
        for tno in (1, 2):
            tid = state.title_id(self.conn, did, tno)
            row = [x for x in state.evidence_for_title(self.conn, tid)
                   if x["category"] == "runtime-align"][0]
            got.append(row["ep_number"])
        self.assertEqual(sorted(got), [3, 4])

    def test_status_flags_assignment_outside_packaging(self):
        state.upsert_episodes(
            self.conn, [ep(1, k, f"E{k}", 1320.0) for k in range(1, 5)])
        did = self.add_disc([title(1, 1320, [1320])], path="/d/s1d1.iso")
        tid = state.title_id(self.conn, did, 1)
        # box says E3-E4, but an assignment lands on E1 -> surfaced as 'outside'
        state.set_background(self.conn, "s1d1", [(1, 3), (1, 4)])
        state.set_assignment(self.conn, tid, [state.episode_id(self.conn, 1, 1)],
                             status="proposed", decided_by="heuristic:align")
        pkg = {p["disc"]: p for p in review.summarize(self.conn)["packaging"]}
        self.assertEqual(pkg["s1d1"]["outside"], [[1, 1]])
        self.assertIn([1, 3], pkg["s1d1"]["missing"])


class SpecialsDedupTests(Base):
    def test_special_assigned_at_most_once_across_seasons(self):
        # one special (S00E01) ranked #1 by a leftover title in S1 AND one in S2.
        # A per-season bijection would let BOTH claim it (each season's pool holds
        # all specials); the global specials pass must give it to only one.
        state.upsert_episodes(self.conn, [
            ep(1, 1, "A", 1320.0), ep(2, 1, "B", 1320.0),
            ep(0, 1, "Special", 1320.0)])
        t1 = title(1, 1320, [1320]); t1.kind = "episode-candidate"
        t2 = title(1, 1320, [1320]); t2.kind = "episode-candidate"
        did1 = self.add_disc([t1], season_hint=1, path="/d/S1D1.iso")
        did2 = self.add_disc([t2], season_hint=2, path="/d/S2D1.iso")

        orig_ft, orig_rc = compute.full_transcript, compute.rank_candidates
        compute.full_transcript = lambda d, t, w: "dialogue"
        compute.rank_candidates = lambda tr, pool, model, host: (
            [(next(e for e in pool if e.season == 0), 6)], "ev")   # both pick the special
        try:
            compute.run_synopsis(self.conn, auto_args(
                all=True, title=None, disc=None, judge_model=None,
                synopsis_windows=None, synopsis_length=None, retranscribe=False,
                synopsis_source="tmdb", transcript_source="audio",
                include_specials=True))
        finally:
            compute.full_transcript, compute.rank_candidates = orig_ft, orig_rc

        sp = state.episode_id(self.conn, 0, 1)
        got = []
        for did in (did1, did2):
            tid = state.title_id(self.conn, did, 1)
            e = [x for x in state.evidence_for_title(self.conn, tid)
                 if x["category"] == "synopsis"][0]
            got.append(e["episode_id"])
        self.assertEqual(got.count(sp), 1)      # special claimed exactly once
        self.assertEqual(got.count(None), 1)    # the loser abstains, not double-claim


class PoolTests(Base):
    def test_include_specials_widens_season_pool(self):
        from discs import Disc
        seasons = {1: [ep(1, 1, "A", 1320.0)]}
        specials = [ep(0, 1, "Sp", 1400.0)]
        all_eps = seasons[1] + specials
        d = Disc(path=Path("/d.iso"), format="dvd", label="x", season_hint=1)
        without = compute._pool_for(d, seasons, specials, all_eps, False)
        withsp = compute._pool_for(d, seasons, specials, all_eps, True)
        self.assertEqual([(e.season, e.number) for e in without], [(1, 1)])
        self.assertEqual([(e.season, e.number) for e in withsp], [(1, 1), (0, 1)])

    def test_no_hint_uses_whole_series(self):
        from discs import Disc
        seasons = {1: [ep(1, 1, "A", 1320.0)]}
        specials = [ep(0, 1, "Sp", 1400.0)]
        all_eps = seasons[1] + specials
        d = Disc(path=Path("/d.iso"), format="dvd", label="x", season_hint=None)
        self.assertEqual(compute._pool_for(d, seasons, specials, all_eps, False),
                         all_eps)


class AlignAnchorComputeTests(Base):
    def test_confirmed_title_anchors_align_rejected_excluded(self):
        # 5 same-runtime episodes; a disc with 3 candidates + 1 rejected extra.
        state.upsert_episodes(self.conn, [ep(1, k, f"E{k}", 1320.0)
                                          for k in range(1, 6)])
        cand = [title(1, 1320, [660, 660]), title(2, 1320, [600, 720]),
                title(3, 1320, [700, 620])]
        for i, t in enumerate(cand):
            t.order_key = i
        rej = title(9, 1320, [500, 820]); rej.order_key = 9
        did = self.add_disc(cand + [rej], disc_hint=1)
        # human: the FIRST candidate is really E2 (anchor), and t9 is not an episode
        t1id = state.title_id(self.conn, did, 1)
        e2 = state.episode_id(self.conn, 1, 2)
        state.set_assignment(self.conn, t1id, [e2], status="confirmed",
                             decided_by="human")
        state.set_assignment(self.conn, state.title_id(self.conn, did, 9), [],
                             status="rejected", decided_by="human")

        r = compute.run_align(self.conn, args=None)
        self.assertGreaterEqual(r["anchors"], 1)
        # anchored title keeps its confirmed episode in the fresh evidence...
        a1 = [e for e in state.evidence_for_title(self.conn, t1id)
              if e["category"] == "runtime-align"][0]
        self.assertEqual((a1["ep_season"], a1["ep_number"]), (1, 2))
        # ...and the next candidate shifts to E3 (contiguous around the anchor)
        t2id = state.title_id(self.conn, did, 2)
        a2 = [e for e in state.evidence_for_title(self.conn, t2id)
              if e["category"] == "runtime-align"][0]
        self.assertEqual((a2["ep_season"], a2["ep_number"]), (1, 3))
        # rejected title was dropped from the candidate set -> no episode evidence
        rid = state.title_id(self.conn, did, 9)
        rev = [e for e in state.evidence_for_title(self.conn, rid)
               if e["category"] == "runtime-align" and e["episode_id"]]
        self.assertEqual(rev, [])


class ResolveTests(Base):
    def _one_title(self):
        state.upsert_episodes(self.conn, [ep(1, 1, "A", 1320.0), ep(1, 2, "B", 1320.0)])
        did = self.add_disc([title(1, 1320, [660, 660])])
        return state.title_id(self.conn, did, 1)

    def test_agreement_proposes_consensus(self):
        tid = self._one_title()
        e1 = state.episode_id(self.conn, 1, 1)
        state.put_evidence(self.conn, tid, "runtime-align", episode_id=e1, confidence=0.9)
        state.put_evidence(self.conn, tid, "title-card-ocr", episode_id=e1, confidence=1.0)
        r = review.resolve(self.conn)
        self.assertEqual(r["proposed"], 1)
        a = state.get_assignment(self.conn, tid)
        self.assertEqual(a["status"], "proposed")
        self.assertEqual(a["decided_by"], "heuristic:consensus")

    def test_conflict_is_not_proposed_and_shows_in_gaps(self):
        tid = self._one_title()
        e1 = state.episode_id(self.conn, 1, 1)
        e2 = state.episode_id(self.conn, 1, 2)
        state.put_evidence(self.conn, tid, "runtime-align", episode_id=e1, confidence=0.9)
        state.put_evidence(self.conn, tid, "title-card-ocr", episode_id=e2, confidence=1.0)
        r = review.resolve(self.conn)
        self.assertEqual(r["proposed"], 0)
        self.assertEqual(r["conflicts"], 1)
        g = review.gaps(self.conn)
        hit = [x for x in g["gaps"] if x["title_id"] == tid][0]
        self.assertTrue(hit["conflict"])

    def test_human_decision_is_sticky(self):
        tid = self._one_title()
        e1 = state.episode_id(self.conn, 1, 1)
        e2 = state.episode_id(self.conn, 1, 2)
        state.set_assignment(self.conn, tid, [e2], status="confirmed",
                             decided_by="human")
        state.put_evidence(self.conn, tid, "runtime-align", episode_id=e1, confidence=0.9)
        review.resolve(self.conn)
        a = state.get_assignment(self.conn, tid)
        self.assertEqual(a["decided_by"], "human")
        import json
        self.assertEqual(json.loads(a["episode_ids_json"]), [e2])


class ExportTests(Base):
    def test_confirmed_assignment_exports(self):
        state.upsert_episodes(self.conn, [ep(1, 1, "Pilot", 1320.0)])
        did = self.add_disc([title(1, 1320, [660, 660])])
        tid = state.title_id(self.conn, did, 1)
        e1 = state.episode_id(self.conn, 1, 1)
        state.set_assignment(self.conn, tid, [e1], status="confirmed",
                             decided_by="human")
        recs = export_mod.build_records(self.conn)
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual(r["title"], 1)                 # DVD rip title number
        self.assertIn("S01E01", r["suggested_filename"])
        self.assertIn("{tmdb-999}", r["suggested_filename"])
        self.assertEqual(r["identified_by"], "human")

    def test_proposed_excluded_unless_requested(self):
        state.upsert_episodes(self.conn, [ep(1, 1, "Pilot", 1320.0)])
        did = self.add_disc([title(1, 1320, [660, 660])])
        tid = state.title_id(self.conn, did, 1)
        e1 = state.episode_id(self.conn, 1, 1)
        state.set_assignment(self.conn, tid, [e1], status="proposed",
                             decided_by="heuristic:runtime-align")
        self.assertEqual(len(export_mod.build_records(self.conn)), 0)
        self.assertEqual(len(export_mod.build_records(self.conn, include_proposed=True)), 1)


class EliminationTests(Base):
    def test_recovers_lone_adjacent_missing(self):
        # E2/E3 matched on a disc; E1 missing and adjacent -> the lone
        # unmatched candidate (a title-card-less premiere) must be E1.
        state.upsert_episodes(self.conn, [
            ep(1, 1, "One", 1320.0), ep(1, 2, "Two", 1320.0),
            ep(1, 3, "Three", 1320.0)])
        t1, t2, t3 = (title(1, 1320, [660, 660]), title(2, 1320, [600, 720]),
                      title(3, 1320, [700, 620]))
        for t in (t1, t2, t3):
            t.kind = "episode-candidate"
        did = self.add_disc([t1, t2, t3])
        e2 = state.episode_id(self.conn, 1, 2)
        e3 = state.episode_id(self.conn, 1, 3)
        state.set_assignment(self.conn, state.title_id(self.conn, did, 2), [e2],
                             status="proposed", decided_by="heuristic:runtime-align")
        state.set_assignment(self.conn, state.title_id(self.conn, did, 3), [e3],
                             status="proposed", decided_by="heuristic:runtime-align")

        r = compute.run_elimination(self.conn, args=None)
        self.assertEqual(r["recovered"], 1)
        t1id = state.title_id(self.conn, did, 1)
        ev = [x for x in state.evidence_for_title(self.conn, t1id)
              if x["category"] == "elimination"][0]
        self.assertEqual((ev["ep_season"], ev["ep_number"]), (1, 1))
        # resolve then proposes the recovered episode
        review.resolve(self.conn)
        a = state.get_assignment(self.conn, t1id)
        self.assertEqual(a["status"], "proposed")

    def test_noop_without_gaps(self):
        state.upsert_episodes(self.conn, [ep(1, 1, "One", 1320.0)])
        t1 = title(1, 1320, [660, 660])
        t1.kind = "episode-candidate"
        self.add_disc([t1])
        r = compute.run_elimination(self.conn, args=None)
        self.assertEqual(r["recovered"], 0)


class AutoTests(Base):
    def _separable_dvd(self):
        # runtime-separable episodes on a disc-hinted DVD -> order verifiable
        state.upsert_episodes(self.conn, [
            ep(1, 1, "One", 1200.0), ep(1, 2, "Two", 1400.0),
            ep(1, 3, "Three", 1600.0)])
        self.add_disc([title(1, 1200, [600, 600]),
                       title(2, 1400, [700, 700]),
                       title(3, 1600, [800, 800])], disc_hint=1)

    def test_auto_chains_align_resolve(self):
        self._separable_dvd()
        r = auto_mod.run_auto(self.conn, auto_args(ocr="never"))
        self.assertTrue(r["ok"])
        self.assertEqual(r["assignments_by_status"].get("proposed"), 3)
        s1 = [s for s in r["seasons"] if s["season"] == 1][0]
        self.assertEqual((s1["matched"], s1["total"]), (3, 3))

    def test_ocr_plan_skips_verifiable_disc(self):
        # disc-hinted + runtime-separable -> assess_ordering verifiable, so the
        # 'auto' OCR policy skips OCR without needing a VLM at all
        self._separable_dvd()
        compute.run_align(self.conn, args=None)
        review.resolve(self.conn)
        discs, why = auto_mod._ocr_plan(self.conn, auto_args(ocr="auto"))
        self.assertEqual(discs, [])
        self.assertIn("verifiable", why)

    def test_no_discs_errors(self):
        r = auto_mod.run_auto(self.conn, auto_args())
        self.assertFalse(r["ok"])
        self.assertEqual(r["error"], "no-discs")


class TextGateTests(unittest.TestCase):
    def test_fails_open_when_backend_unavailable(self):
        import text_region as tr
        # an unrecognized/unavailable backend must never prune (recall-first)
        self.assertTrue(tr.frame_has_text("/any.jpg", backend="nope"))

    def test_verify_title_accepts_text_filter(self):
        import inspect
        from identify import verify_title
        self.assertIn("text_filter", inspect.signature(verify_title).parameters)

    def test_card_vs_scene_active_backend(self):
        import glob
        import text_region as tr
        base = "/run/media/ekovac/MediaScratc/video-juicer-artifacts/tr-eval"
        cards = glob.glob(f"{base}/pos_db/*.jpg")
        scenes = glob.glob(f"{base}/neg_ent/*.jpg")
        available = tr.paddle_available()
        if not (available and cards and scenes):
            self.skipTest("no text detector or fixture frames available")
        self.assertTrue(tr.frame_has_text(cards[0]))                 # card kept
        self.assertFalse(any(tr.frame_has_text(s) for s in scenes[:15]))  # scenes pruned


class AnomalyTests(Base):
    def test_episode_length_leftover_flagged(self):
        # an unclaimed episode-length candidate = the tell of a dropped/shifted
        # episode (TNG D1's Farpoint-displaced title) -> anomaly-flagged in gaps
        state.upsert_episodes(self.conn, [ep(1, 1, "A", 1320.0),
                                          ep(1, 2, "B", 1320.0), ep(1, 3, "C", 1320.0)])
        t = title(1, 1320, [660, 660])
        t.kind = "episode-candidate"
        did = self.add_disc([t])
        tid = state.title_id(self.conn, did, 1)
        state.put_evidence(self.conn, tid, "runtime-align", episode_id=None,
                           verdict="not an episode (leftover)", confidence=0.0)
        hit = [g for g in review.gaps(self.conn)["gaps"] if g["title_id"] == tid][0]
        self.assertTrue(hit["anomaly"])
        self.assertIn("episode-length", hit["reason"])

    def _ep_cand(self, tid, dur, disc_id=None, path="/d/x.iso", disc_hint=1):
        t = title(tid, dur, [dur / 2, dur / 2])
        t.kind = "episode-candidate"
        return self.add_disc([t], disc_hint=disc_hint, path=path)

    def test_suggest_assign_when_ocr_corroborates(self):
        state.upsert_episodes(self.conn, [ep(1, 1, "Pilot", 1320.0)])
        did = self._ep_cand(1, 1320)
        tid = state.title_id(self.conn, did, 1)
        e1 = state.episode_id(self.conn, 1, 1)
        state.put_evidence(self.conn, tid, "title-card-ocr", episode_id=e1,
                           verdict="read Pilot", confidence=1.0)
        g = [x for x in review.gaps(self.conn)["gaps"] if x["title_id"] == tid][0]
        self.assertEqual(g["suggestion"]["action"], "assign")
        self.assertEqual(g["suggestion"]["episode"], "S01E01")

    def test_suggest_reject_when_duration_mismatches(self):
        # a 5-min featurette whose card names a 22-min episode -> reject impostor
        state.upsert_episodes(self.conn, [ep(1, 1, "Pilot", 1320.0)])
        did = self._ep_cand(1, 300)                      # 5 min title
        tid = state.title_id(self.conn, did, 1)
        e1 = state.episode_id(self.conn, 1, 1)
        state.put_evidence(self.conn, tid, "title-card-ocr", episode_id=e1,
                           verdict="named Pilot", confidence=1.0)
        g = [x for x in review.gaps(self.conn)["gaps"] if x["title_id"] == tid][0]
        self.assertEqual(g["suggestion"]["action"], "reject")
        self.assertIn("doesn't fit", g["suggestion"]["why"])

    def test_suggest_reject_duplicate_keeps_primary(self):
        state.upsert_episodes(self.conn, [ep(1, 1, "Pilot", 1320.0)])
        # two titles both OCR-corroborate S01E01; earlier order_key wins
        t_lo = title(1, 1320, [660, 660]); t_lo.kind = "episode-candidate"; t_lo.order_key = 0
        t_hi = title(9, 1320, [660, 660]); t_hi.kind = "episode-candidate"; t_hi.order_key = 9
        did = self.add_disc([t_lo, t_hi])
        e1 = state.episode_id(self.conn, 1, 1)
        lo = state.title_id(self.conn, did, 1)
        hi = state.title_id(self.conn, did, 9)
        for x in (lo, hi):
            state.put_evidence(self.conn, x, "title-card-ocr", episode_id=e1,
                               verdict="Pilot", confidence=1.0)
        gs = {x["title_id"]: x["suggestion"] for x in review.gaps(self.conn)["gaps"]}
        self.assertEqual(gs[lo]["action"], "assign")     # primary kept
        self.assertEqual(gs[hi]["action"], "reject")     # duplicate rejected
        self.assertIn("duplicate", gs[hi]["why"])

    def test_short_extra_not_flagged(self):
        # a genuinely short extra is not episode-length -> no anomaly
        state.upsert_episodes(self.conn, [ep(1, 1, "A", 1320.0), ep(1, 2, "B", 1320.0)])
        t = title(1, 300, [300])           # 5 min featurette
        t.kind = "episode-candidate"
        did = self.add_disc([t])
        tid = state.title_id(self.conn, did, 1)
        state.put_evidence(self.conn, tid, "runtime-align", episode_id=None,
                           verdict="not an episode (leftover)", confidence=0.0)
        hits = [g for g in review.gaps(self.conn)["gaps"] if g["title_id"] == tid]
        self.assertFalse(hits and hits[0]["anomaly"])


class OcrTargetTests(Base):
    def test_all_covers_every_candidate_grouped_by_disc(self):
        import types
        t1 = title(1, 1320, [660, 660]); t1.kind = "episode-candidate"
        t2 = title(2, 1320, [660, 660]); t2.kind = "episode-candidate"
        t2.order_key = 1
        d1 = self.add_disc([t1, t2], path="/x/d1.iso")
        t3 = title(1, 1320, [660, 660]); t3.kind = "episode-candidate"
        tx = title(3, 300, [300]); tx.kind = "extra"        # not a candidate
        d2 = self.add_disc([t3, tx], path="/x/d2.iso")
        args = types.SimpleNamespace(title=None, all=True, disc=None)
        tgts = compute._ocr_targets(self.conn, args)
        self.assertEqual(len(tgts), 3)                       # 3 candidates, extra skipped
        self.assertEqual([d for d, _ in tgts], sorted(d for d, _ in tgts))  # disc-grouped


class DiscResolveTests(Base):
    def test_resolve_disc(self):
        import vj
        d1 = self.add_disc([title(1, 1320, [660, 660])], path="/x/VENTURE_BROS_S1D1.iso")
        d2 = self.add_disc([title(1, 1320, [660, 660])], path="/x/VENTURE_BROS_S1D2.iso")
        self.assertEqual(vj._resolve_disc(self.conn, "VENTURE_BROS_S1D1")[0], d1)
        self.assertEqual(vj._resolve_disc(self.conn, "s1d2")[0], d2)   # substr, ci
        self.assertEqual(vj._resolve_disc(self.conn, str(d2))[0], d2)  # integer id
        self.assertIsNotNone(vj._resolve_disc(self.conn, "nope")[1])   # not found
        self.assertIsNotNone(vj._resolve_disc(self.conn, "VENTURE")[1])  # ambiguous


class BoardTests(Base):
    def test_board_structure_and_render(self):
        import vj
        state.set_project(self.conn, tmdb_id=655, show_name="TNG", year=1987)
        state.upsert_episodes(self.conn, [ep(1, 1, "Pilot", 1320.0),
                                          ep(1, 2, "Two", 1320.0)])
        t1 = title(1, 1320, [660, 660]); t1.kind = "episode-candidate"
        t2 = title(2, 300, [300]); t2.kind = "extra"
        did = self.add_disc([t1, t2])
        tid = state.title_id(self.conn, did, 1)
        e1 = state.episode_id(self.conn, 1, 1)
        state.put_evidence(self.conn, tid, "runtime-align", episode_id=e1, confidence=0.9)
        state.put_evidence(self.conn, tid, "title-card-ocr", episode_id=e1,
                           verdict="read 'Pilot' -> S01E01 (1.00)", confidence=1.0)
        state.set_assignment(self.conn, tid, [e1], status="confirmed",
                             decided_by="agent")
        b = review.board(self.conn, season=1)
        self.assertEqual(b["discs"][0]["titles"][0]["assignment"]["episodes"], ["S01E01"])
        self.assertIn("runtime-align", b["discs"][0]["titles"][0]["evidence"])
        # renders without error; collapses the unassigned extra by default
        text = vj._board_human(b)
        self.assertIn("S01E01", text)
        self.assertIn("+1 extra", text)
        self.assertIn("Pilot", vj._board_human(b, show_all=True))


class PlayTests(Base):
    def test_dvd_argv(self):
        row = {"format": "dvd", "path": "/x.iso", "title_number": 4, "clips_json": "[]"}
        argv, note = vj._play_argv("vlc", row, 0)
        self.assertEqual(argv, ["vlc", "dvd:///x.iso#4"])

    def test_bluray_image_falls_back(self):
        # no BDMV dir -> can't select the title, falls back to the disc MRL
        row = {"format": "bluray", "path": "/no/such/bd", "title_number": 1,
               "clips_json": '["00000"]'}
        argv, note = vj._play_argv("mpv", row, 0)
        self.assertTrue(any("bluray:///no/such/bd" in a for a in argv))
        self.assertIn("main title", note)

    def test_start_flag_per_player(self):
        self.assertEqual(vj._start_args("vlc", 40), ["--start-time", "40"])
        self.assertEqual(vj._start_args("mpv", 40), ["--start=40"])
        self.assertEqual(vj._start_args("vlc", 0), [])

    def _play_json(self, *cli):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = vj.main(["--json", "play", self.db, *cli])
        # success prints to stdout; structured errors go to stderr
        return rc, _json.loads(out.getvalue() or err.getvalue())

    def test_play_by_title_print(self):
        state.upsert_episodes(self.conn, [ep(1, 1, "A", 1320.0)])
        did = self.add_disc([title(1, 1320, [660, 660])], path="/d/x.iso")
        tid = state.title_id(self.conn, did, 1)
        rc, out = self._play_json("--title", str(tid), "--print")
        self.assertEqual(rc, 0)
        self.assertIn("dvd:///d/x.iso#1", out["command"])
        self.assertFalse(out["launched"])

    def test_play_by_episode_uses_assignment(self):
        state.upsert_episodes(self.conn, [ep(1, 1, "A", 1320.0)])
        did = self.add_disc([title(1, 1320, [660, 660])], path="/d/x.iso")
        tid = state.title_id(self.conn, did, 1)
        e1 = state.episode_id(self.conn, 1, 1)
        state.set_assignment(self.conn, tid, [e1], status="confirmed",
                             decided_by="human")
        rc, out = self._play_json("S01E01", "--print")
        self.assertEqual(rc, 0)
        self.assertEqual(out["title_id"], tid)

    def test_play_unassigned_episode_errors(self):
        state.upsert_episodes(self.conn, [ep(1, 1, "A", 1320.0)])
        self.add_disc([title(1, 1320, [660, 660])])
        rc, out = self._play_json("S01E01", "--print")
        self.assertEqual(rc, 1)
        self.assertEqual(out["error"], "no-assignment")


import bz2
import wiki

_EPLIST = """Lead paragraph.
== Season 1 ==
{{Episode table}}
{{Episode list
|EpisodeNumber=1
|EpisodeNumber2=1
|Title=Pilot
|ShortSummary=Quentin takes a magic entrance exam and [[passes]] it.<ref>x</ref>
}}
{{Episode list
|EpisodeNumber=2
|EpisodeNumber2=2
|Title=Second
|ShortSummary=The students face the '''aftermath'''.
}}
== Season 2 ==
{{Episode list
|EpisodeNumber=3
|EpisodeNumber2=1
|Title=Return
|ShortSummary=They return to the world.
}}
"""


def _build_snapshot(dirpath):
    """A synthetic 2-stream multistream dump + index, mirroring Wikimedia's
    layout: independent bz2 streams concatenated, index = offset:pageid:title."""
    def page(title, text):
        return (f"<page><title>{title}</title><ns>0</ns><id>0</id><revision>"
                f'<text xml:space="preserve">{text}</text></revision></page>')
    streams = [
        [("Alpha", "alpha body"),
         ("List of Foo (TV series) episodes", _EPLIST)],
        [("Beta", "beta body"), ("Foo Redirect", "#REDIRECT [[Alpha]]")],
    ]
    data, index, pid = b"", [], 1
    for pages in streams:
        offset = len(data)
        xml = "".join(page(t, x) for t, x in pages)
        data += bz2.compress(xml.encode("utf-8"))
        for t, _ in pages:
            index.append(f"{offset}:{pid}:{t}")
            pid += 1
    dp = Path(dirpath)
    (dp / "data.xml.bz2").write_bytes(data)
    (dp / "index.txt.bz2").write_bytes(bz2.compress(("\n".join(index) + "\n").encode()))
    return dp / "data.xml.bz2", dp / "index.txt.bz2"


class BoardOrderTests(Base):
    def test_board_is_season_then_disc_order(self):
        state.upsert_episodes(self.conn, [ep(1, 1, "A", 1320.0)])
        # add discs out of season/disc order; scan (id) order would be S2D1,
        # S1D2, S1D1 — board must re-sort to S1D1, S1D2, S2D1
        self.add_disc([title(1, 1320, [660])], season_hint=2, disc_hint=1,
                      path="/d/s2d1")
        self.add_disc([title(1, 1320, [660])], season_hint=1, disc_hint=2,
                      path="/d/s1d2")
        self.add_disc([title(1, 1320, [660])], season_hint=1, disc_hint=1,
                      path="/d/s1d1")
        order = [(d["season"], d["disc"]) for d in review.board(self.conn)["discs"]]
        self.assertEqual(order, [(1, "s1d1"), (1, "s1d2"), (2, "s2d1")])


class WikiReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data, self.index = _build_snapshot(self.tmp.name)
        self.snap = wiki.MultistreamSnapshot(self.data, self.index)

    def tearDown(self):
        self.tmp.cleanup()

    def test_lookup_first_and_second_stream(self):
        self.assertEqual(self.snap.article("Alpha"), "alpha body")
        self.assertEqual(self.snap.article("Beta"), "beta body")   # 2nd stream

    def test_missing_title(self):
        self.assertIsNone(self.snap.article("Nonexistent"))

    def test_redirect_followed(self):
        self.assertEqual(self.snap.article("Foo Redirect"), "alpha body")

    def test_missing_files_raise(self):
        with self.assertRaises(wiki.SnapshotError):
            wiki.MultistreamSnapshot("/no/data.bz2", self.index)

    def test_parse_episode_summaries(self):
        s = wiki.episode_summaries(self.snap, "List of Foo (TV series) episodes")
        self.assertEqual(s[(1, 1)][0], "Pilot")
        self.assertIn("entrance exam", s[(1, 1)][1])
        self.assertNotIn("<ref>", s[(1, 1)][1])      # markup stripped
        self.assertNotIn("'''", s[(1, 2)][1])
        self.assertEqual(s[(2, 1)][0], "Return")     # season header tracked
        self.assertEqual(len(s), 3)


class TitleArgTests(Base):
    """--disc resolves by image basename; a shared volume label can't pick one."""

    def test_disc_name_not_label(self):
        d1 = self.add_disc([title(4, 1320, [1320])], path="/d/VB_S1D1.iso")
        d2 = self.add_disc([title(4, 1320, [1320])], disc_hint=2, path="/d/VB_S1D2.iso")
        # both discs carry the same volume label ("S1D1", from add_disc)
        args = types.SimpleNamespace(title=None, disc="VB_S1D2", playlist=4)
        self.assertEqual(vj._resolve_title_arg(self.conn, args),
                         state.title_id(self.conn, d2, 4))
        args.disc = "S1D1"                     # ambiguous label -> no guess
        self.assertIsNone(vj._resolve_title_arg(self.conn, args))
        self.assertNotEqual(d1, d2)


class WikiMatchTests(unittest.TestCase):
    """wiki.match_summaries: title-first, so a Wikipedia list numbered
    differently from TMDB still files each summary under the right episode."""

    def test_title_beats_number_and_parts_share_a_combined_entry(self):
        import wiki
        eps = [ep(1, 3, "Home Insecurity", None), ep(1, 8, "Mid-Life Chrysalis", None),
               ep(1, 9, "Are You There, God? It's Me, Dean", None),
               ep(2, 12, "Showdown at Cremation Creek (1)", None),
               ep(2, 13, "Showdown at Cremation Creek (2)", None),
               ep(7, 1, "The Venture Bros. and the Curse of the Haunted Problem", None),
               ep(3, 5, "No Title Match", None)]
        summaries = {(1, 3): ("Mid-life Chrysalis", "chrysalis plot"),
                     (1, 7): ("Home Insecurity", "insecurity plot"),
                     (1, 10): ("Are You There God, It's Me, Dean", "dean plot"),
                     (2, 12): ("Showdown at Cremation Creek", "showdown plot"),
                     (7, 1): ("The Venture Bros. & The Curse of the Haunted Problem",
                              "haunted plot"),
                     (3, 5): ("Retitled On Wikipedia", "fallback plot"),
                     (9, 9): ("Nowhere", "dropped")}
        m = wiki.match_summaries(summaries, eps)
        self.assertEqual(m[(1, 3)], "insecurity plot")      # NOT chrysalis (number)
        self.assertEqual(m[(1, 8)], "chrysalis plot")
        self.assertEqual(m[(1, 9)], "dean plot")
        self.assertEqual((m[(2, 12)], m[(2, 13)]), ("showdown plot", "showdown plot"))
        self.assertEqual(m[(7, 1)], "haunted plot")         # & == and
        self.assertEqual(m[(3, 5)], "fallback plot")        # number fallback
        self.assertNotIn((9, 9), m)

    def test_number_fallback_never_overwrites_a_title_match(self):
        import wiki
        eps = [ep(1, 1, "Alpha", None), ep(1, 2, "Beta", None)]
        m = wiki.match_summaries({(1, 1): ("Beta", "beta plot"),
                                  (1, 2): ("Gamma", "gamma plot")}, eps)
        self.assertEqual(m, {(1, 2): "beta plot"})


class EnrichTests(Base):
    def _run_enrich(self, **kw):
        data, index = _build_snapshot(self.tmp.name)
        args = types.SimpleNamespace(db=self.db, source="wikipedia",
                                     snapshot=data, index=index, json=True, **kw)
        out_buf, err_buf = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            rc = vj.cmd_enrich(args)
        text = out_buf.getvalue().strip() or err_buf.getvalue().strip()
        return rc, (_json.loads(text) if text else {})

    def test_enrich_populates_wiki_overview_and_remembers_page(self):
        state.upsert_episodes(self.conn, [
            ep(1, 1, "Pilot", 2640.0), ep(1, 2, "Second", 2640.0),
            ep(2, 1, "Return", 2640.0)])
        rc, out = self._run_enrich(page="List of Foo (TV series) episodes")
        self.assertEqual(rc, 0)
        self.assertEqual(out["updated"], 3)
        eps = {(e.season, e.number): e for e in state.load_episodes(self.conn)}
        self.assertIn("entrance exam", eps[(1, 1)].wiki_overview)
        self.assertEqual(eps[(1, 1)].synopsis, eps[(1, 1)].wiki_overview)  # prefers wiki
        # page is remembered so a re-run needs no --page
        self.assertEqual(state.get_project(self.conn).get("wikipedia_page"),
                         "List of Foo (TV series) episodes")

    def test_enrich_without_page_errors(self):
        state.upsert_episodes(self.conn, [ep(1, 1, "Pilot", 2640.0)])
        rc, out = self._run_enrich(page=None)
        self.assertEqual(rc, 1)
        self.assertEqual(out["error"], "no-page")


class TranscodeTests(unittest.TestCase):
    """Pure logic of the transcode verb (no HandBrake/mkvtoolnix needed)."""

    def _rec(self, image="/img/Book_1_Disc_1", title=62, season=1,
             episodes=(1,), name="The Boy in the Iceberg"):
        return {"image": image, "title": title, "kind": "episode",
                "season": season, "episodes": list(episodes),
                "episode_name": name, "suggested_filename": "x.mkv",
                "video_format": "1080p"}

    def test_recipe_hash_is_encode_only(self):
        import transcode as tc
        r = self._rec()
        base = tc._sha(tc.encode_recipe(r, "Fast 1080p30", []))
        # renaming the episode does NOT change the encode recipe
        r2 = self._rec(name="Totally Different Name")
        self.assertEqual(base, tc._sha(tc.encode_recipe(r2, "Fast 1080p30", [])))
        # changing preset / title / source / opts DOES
        self.assertNotEqual(base, tc._sha(tc.encode_recipe(r, "HQ 1080p30", [])))
        self.assertNotEqual(base, tc._sha(tc.encode_recipe(self._rec(title=7),
                                                           "Fast 1080p30", [])))
        self.assertNotEqual(base, tc._sha(tc.encode_recipe(r, "Fast 1080p30",
                                                           ["--deinterlace"])))

    def test_meta_hash_tracks_names_not_encode(self):
        import transcode as tc
        m1 = tc._sha(tc.meta_fields(self._rec(), "Avatar", 2005, 246))
        m2 = tc._sha(tc.meta_fields(self._rec(name="Renamed"), "Avatar", 2005, 246))
        self.assertNotEqual(m1, m2)

    def test_episode_tag_single_and_range(self):
        import transcode as tc
        self.assertEqual(tc._episode_tag_for(self._rec(episodes=(1,))), "S01E01")
        self.assertEqual(
            tc._episode_tag_for(self._rec(season=3, episodes=(18, 19, 20, 21))),
            "S03E18-E21")

    def test_decide_action(self):
        import transcode as tc
        self.assertEqual(tc.decide_action("r", "m", "/t.mkv", None), "encode")
        # recipe changed -> reencode
        self.assertEqual(tc.decide_action(
            "r2", "m", "/t.mkv",
            {"path": "/t.mkv", "recipe": "r", "meta": "m"}), "reencode")
        # same recipe, moved path -> rename
        self.assertEqual(tc.decide_action(
            "r", "m", "/new.mkv",
            {"path": "/old.mkv", "recipe": "r", "meta": "m"}), "rename")
        # same recipe+path, metadata changed -> retag
        self.assertEqual(tc.decide_action(
            "r", "m2", "/t.mkv",
            {"path": "/t.mkv", "recipe": "r", "meta": "m"}), "retag")
        # nothing changed -> skip
        self.assertEqual(tc.decide_action(
            "r", "m", "/t.mkv",
            {"path": "/t.mkv", "recipe": "r", "meta": "m"}), "skip")
        # --force re-encodes even an identical output
        self.assertEqual(tc.decide_action(
            "r", "m", "/t.mkv",
            {"path": "/t.mkv", "recipe": "r", "meta": "m"}, force=True), "reencode")

    def test_tags_xml_roundtrip(self):
        import transcode as tc
        simples = {"TITLE": "The Library", "SEASON": "2", "VJ_RECIPE": "abc123",
                   "VJ_EPISODES": "S02E10", "EMPTY": ""}
        parsed = tc.parse_tags(tc.tags_xml(simples))
        self.assertEqual(parsed["TITLE"], "The Library")
        self.assertEqual(parsed["VJ_RECIPE"], "abc123")
        self.assertEqual(parsed["VJ_EPISODES"], "S02E10")
        self.assertNotIn("EMPTY", parsed)   # empty values are dropped

    def test_parse_tags_empty(self):
        import transcode as tc
        self.assertEqual(tc.parse_tags(""), {})
        self.assertEqual(tc.parse_tags("not xml <<<"), {})

    def test_handbrake_opts_passthrough_after_dashdash(self):
        # the footgun fix: HandBrake flags (which start with --) go after `--`
        # so argparse's leading-dash trap can't fire. db is still captured.
        ns = vj.build_parser().parse_args([
            "transcode", "my.db", "--output-prefix", "/out",
            "--handbrake-preset", "AV1", "--",
            "--preset-import-gui", "--encoder-preset", "8"])
        self.assertEqual(str(ns.db), "my.db")
        self.assertEqual(ns.handbrake_opts,
                         ["--preset-import-gui", "--encoder-preset", "8"])
        # and none is fine
        ns2 = vj.build_parser().parse_args(
            ["transcode", "my.db", "--output-prefix", "/out"])
        self.assertEqual(ns2.handbrake_opts, [])

    def test_is_matroska_magic(self):
        import transcode as tc, tempfile, os
        with tempfile.TemporaryDirectory() as d:
            mkv = os.path.join(d, "a.mkv")
            with open(mkv, "wb") as f:
                f.write(b"\x1a\x45\xdf\xa3rest")          # EBML head
            self.assertTrue(tc._is_matroska(Path(mkv)))
            mp4 = os.path.join(d, "b.mkv")
            with open(mp4, "wb") as f:
                f.write(b"\x00\x00\x00\x20ftypmp42")       # MP4 ftyp box
            self.assertFalse(tc._is_matroska(Path(mp4)))
            self.assertFalse(tc._is_matroska(Path(d) / "missing.mkv"))


class JevJudgeTests(unittest.TestCase):
    """synopsis.jev_rank over a mocked TypeSafe endpoint: probabilities in,
    ranked (episode, p) out; slice-averaging; `none` as an abstention."""

    def setUp(self):
        import synopsis
        self.syn = synopsis
        self.pool = [Episode(1, n, f"Ep{n}", None, overview=f"plot {n}")
                     for n in (1, 2, 3)]
        self.calls = []
        self.orig = synopsis.typesafe_system_one

    def tearDown(self):
        self.syn.typesafe_system_one = self.orig

    def _mock(self, answers):
        it = iter(answers)

        def fake(state_, questions, model="jev-latest"):
            self.calls.append((state_, questions))
            probs = next(it)
            return {"model": "jev-1.13.0", "usage": {"input_tokens": 100,
                                                     "output_tokens": 5},
                    "answers": {"episode": {"type": "choice",
                                            "probabilities": probs}}}
        self.syn.typesafe_system_one = fake

    def test_ranks_by_probability_and_drops_tiny(self):
        self._mock([{"S01E01 Ep1": 0.1, "S01E02 Ep2": 0.89, "S01E03 Ep3": 0.01,
                     self.syn.JEV_NONE: 0.0}])
        ranked, ev, meta = self.syn.jev_rank("words", self.pool)
        self.assertEqual([(e.number, p) for e, p in ranked], [(2, 0.89), (1, 0.1)])
        self.assertEqual(meta["input_tokens"], 100)
        self.assertEqual(meta["model"], "jev-1.13.0")
        q = self.calls[0][1]["episode"]
        self.assertIn(self.syn.JEV_NONE, q["criteria"])      # abstention option
        self.assertEqual(q["criteria"]["S01E03 Ep3"], "plot 3")

    def test_none_is_an_abstention(self):
        self._mock([{"S01E01 Ep1": 0.2, "S01E02 Ep2": 0.1, "S01E03 Ep3": 0.0,
                     self.syn.JEV_NONE: 0.7}])
        ranked, ev, _ = self.syn.jev_rank("you you you", self.pool)
        self.assertEqual(ranked, [])
        self.assertIn("none", ev)

    def test_chunked_averages_slices(self):
        self._mock([{"S01E01 Ep1": 1.0, "S01E02 Ep2": 0.0, "S01E03 Ep3": 0.0,
                     self.syn.JEV_NONE: 0.0},
                    {"S01E01 Ep1": 0.2, "S01E02 Ep2": 0.8, "S01E03 Ep3": 0.0,
                     self.syn.JEV_NONE: 0.0}])
        ranked, _, meta = self.syn.jev_rank("aaaa bbbb", self.pool, chunk_chars=4)
        self.assertEqual(meta["requests"], 2)
        self.assertEqual([self.calls[0][0], self.calls[1][0]],
                         [{"transcript": "aaaa"}, {"transcript": "bbbb"}])
        self.assertEqual([(e.number, round(p, 2)) for e, p in ranked],
                         [(1, 0.6), (2, 0.4)])

    def test_chunks_never_split_words(self):
        parts = self.syn._chunks("alpha beta gamma delta", 11)
        self.assertEqual(parts, ["alpha beta", "gamma delta"])


class BenchScoreTests(unittest.TestCase):
    """bench_synopsis.score: top-1 vs post-bijection verdicts, distractor false
    claims, and cost from the price table."""

    def test_bijection_and_verdicts(self):
        import bench_synopsis as b
        pool = [Episode(1, n, f"Ep{n}", None, overview="x") for n in (1, 2)]
        cases = [b.Case(1, "D1", 800, 1, {"S01E01"}, "t", "subtitle-ocr", pool),
                 b.Case(2, "D1", 801, 1, {"S01E02"}, "t", "subtitle-ocr", pool),
                 b.Case(3, "D1", 900, 1, set(), "t", "audio", pool)]
        res = {  # both golden titles top-pick E01; the stronger one keeps it
            1: {"ranked": [["S01E01", 0.9], ["S01E02", 0.1]], "latency_s": 1.0,
                "input_tokens": 1_000_000, "output_tokens": 0},
            2: {"ranked": [["S01E01", 0.6], ["S01E02", 0.4]], "latency_s": 3.0,
                "input_tokens": 0, "output_tokens": 0},
            3: {"ranked": [], "latency_s": 2.0, "input_tokens": 0,
                "output_tokens": 0}}
        s = b.score("jev", cases, res)
        self.assertEqual(s["top1"], {"correct": 1, "wrong": 1, "abstain": 1})
        self.assertEqual(s["final"], {"correct": 2, "abstain": 1})   # bijection fixed #2
        self.assertEqual(s["n_golden"], 2)
        self.assertEqual(s["latency_serial_s"], 6.0)
        self.assertAlmostEqual(s["cost_usd"], 0.042)


class CueTimingTests(Base):
    """Timed subtitle cues: SRT parse, frame→cue timing, storage, and the
    --transcribe-only extraction path."""

    def test_srt_to_cues(self):
        from synopsis import srt_to_cues
        srt = ("1\n00:00:01,500 --> 00:00:03,000\n<i>Previously on</i>\n\n"
               "2\n00:01:02,250 --> 00:01:04,000\nHOLDEN: Hold on.\nNAOMI: No.\n")
        self.assertEqual(srt_to_cues(srt), [(1.5, 3.0, "Previously on"),
                                            (62.25, 64.0, "HOLDEN: Hold on. NAOMI: No.")])

    def test_frames_to_cues_ends_at_next_change_and_merges_repeats(self):
        from synopsis import frames_to_cues
        times = [0.0, 1.5, 3.0, 4.5, 6.0]
        texts = ["", "Hello there", "Hello  there", "", "Bye"]
        self.assertEqual(frames_to_cues(times, texts, 8.0),
                         [(1.5, 4.5, "Hello there"), (6.0, 8.0, "Bye")])

    def test_frames_to_cues_fails_soft_on_mismatch(self):
        from synopsis import frames_to_cues
        self.assertEqual(frames_to_cues([0.0], ["a", "b"], 5.0), [])

    def test_cues_round_trip_and_absent_for_audio(self):
        state.upsert_episodes(self.conn, [ep(1, 1, "E1", 1320.0)])
        did = self.add_disc([title(1, 1320, [1320]), title(2, 1320, [1320])])
        t1, t2 = state.title_id(self.conn, did, 1), state.title_id(self.conn, did, 2)
        state.put_transcript(self.conn, t1, "a b", 0, 0.0, "subtitle-ocr",
                             [(1.0, 2.5, "a"), (3.0, 4.0, "b")])
        state.put_transcript(self.conn, t2, "whisper", 0, 0.0, "audio")
        self.assertEqual(state.get_transcript_cues(self.conn, t1),
                         [(1.0, 2.5, "a"), (3.0, 4.0, "b")])
        self.assertIsNone(state.get_transcript_cues(self.conn, t2))

    def test_transcribe_only_stores_cues_and_writes_no_evidence(self):
        state.upsert_episodes(
            self.conn, [ep(1, k, f"E{k}", 1320.0) for k in range(1, 4)])
        ts = [Title(id=k, duration=1320.0, chapters=[1320.0], n_audio=2, n_sub=1,
                    kind="episode-candidate") for k in range(1, 4)]
        ts.append(Title(id=9, duration=2640.0, chapters=[2640.0], n_audio=2,
                        n_sub=1, kind="episode-candidate"))   # 2x: guard lifted
        did = self.add_disc(ts)
        orig_cc, orig_rc = compute.subtitle_cc, compute.rank_candidates
        compute.subtitle_cc = lambda d, t, w: (f"line {t.id}",
                                               [(1.0, 2.0, f"line {t.id}")])
        compute.rank_candidates = lambda *a: self.fail("judge must not be called")
        try:
            res = compute.run_synopsis(self.conn, auto_args(
                disc=did, title=None, all=False, judge_model=None,
                synopsis_windows=None, synopsis_length=None, retranscribe=True,
                synopsis_source="tmdb", transcript_source="subtitle",
                include_specials=False, transcribe_only=True))
        finally:
            compute.subtitle_cc, compute.rank_candidates = orig_cc, orig_rc
        self.assertTrue(res["ok"])
        self.assertEqual(len(res["transcribed"]), 4)
        tid9 = state.title_id(self.conn, did, 9)
        self.assertEqual(state.get_transcript_cues(self.conn, tid9),
                         [(1.0, 2.0, "line 9")])
        for k in (1, 2, 3, 9):
            tid = state.title_id(self.conn, did, k)
            self.assertFalse([e for e in state.evidence_for_title(self.conn, tid)
                              if e["category"] == "synopsis"])


class OrderCheckTests(Base):
    """review.order_check: do the discs, in play order, follow the project's
    numbering or a cached TMDB ordering? (Venture Bros: DVD order ≠ aired.)"""

    NAMES = ["A", "B", "C", "D", "E", "F"]
    # TMDB 'DVD Order' swaps aired E02 and E05 (disc positions 2 and 5)
    DVD = {(1, 1): (1, 1), (1, 2): (1, 5), (1, 3): (1, 3), (1, 4): (1, 4),
           (1, 5): (1, 2), (1, 6): (1, 6)}

    def setup_show(self, disc_content, fmt="dvd", extra=None):
        state.upsert_episodes(self.conn, [ep(1, n, self.NAMES[n - 1], 1320.0)
                                          for n in range(1, 7)])
        ts = [Title(id=k, duration=1320.0, chapters=[1320.0],
                    kind="episode-candidate", order_key=k)
              for k in range(1, len(disc_content) + 1)]
        if extra:
            ts.append(Title(id=99, duration=1000.0, chapters=[1000.0],
                            kind="episode-candidate", order_key=99))
        d = Disc(path=Path("/d/SHOW_S1D1.iso"), format=fmt, label="X",
                 season_hint=1, disc_hint=1)
        d.titles = ts
        did = state.add_disc(self.conn, d)
        for k, aired in enumerate(disc_content, 1):      # title card reads
            tid = state.title_id(self.conn, did, k)
            eid = state.episode_id(self.conn, 1, aired)
            state.put_evidence(self.conn, tid, "title-card-ocr", episode_id=eid,
                               verdict="read", confidence=1.0)
            # runtime-align assigned by POSITION (the Venture Bros failure)
            state.set_assignment(self.conn, tid, [state.episode_id(self.conn, 1, k)],
                                 status="proposed", decided_by="heuristic:runtime-align")
        if extra:                  # an unassigned extra flashing an episode card
            tid = state.title_id(self.conn, did, 99)
            state.put_evidence(self.conn, tid, "title-card-ocr",
                               episode_id=state.episode_id(self.conn, 1, extra),
                               verdict="read", confidence=1.0)
        state.put_order_maps(self.conn, [{
            "id": "g-dvd", "name": "DVD Order", "type": 3,
            "episodes": [(a[0], a[1], b[0], b[1]) for a, b in self.DVD.items()]}])
        return did

    def test_discs_in_dvd_order_flag_a_mismatch_naming_the_group(self):
        self.setup_show([1, 5, 3, 4, 2, 6])     # disc order = DVD order
        [c] = review.order_check(self.conn)
        self.assertTrue(c["mismatch"])
        self.assertEqual(c["best_group"]["name"], "DVD Order")
        self.assertEqual(c["best_group"]["fraction"], 1.0)
        self.assertLess(c["project_fraction"], 0.9)
        self.assertEqual(review.order_mismatches(self.conn), [c])

    def test_discs_in_project_order_are_fine(self):
        self.setup_show([1, 2, 3, 4, 5, 6])
        self.assertEqual(review.order_mismatches(self.conn), [])

    def test_unassigned_extra_with_a_card_is_ignored(self):
        self.setup_show([1, 2, 3, 4, 5, 6], extra=2)   # extra after E06 reads E02
        self.assertEqual(review.order_mismatches(self.conn), [])

    def test_bluray_without_play_all_is_not_checked(self):
        self.setup_show([1, 5, 3, 4, 2, 6], fmt="bluray")
        self.assertEqual(review.order_check(self.conn), [])

    def test_order_label_names_the_projects_numbering(self):
        self.setup_show([1, 2, 3, 4, 5, 6])
        self.assertEqual(state.order_label(self.conn), "TMDB aired order")
        state.set_project(self.conn, episode_order="g-dvd")
        self.assertEqual(state.order_label(self.conn), "TMDB 'DVD Order'")


class OrderStampTests(Base):
    """The numbering travels with every output: manifest, rip script, tags."""

    def test_manifest_rip_script_and_tags_carry_the_ordering(self):
        import identify
        import transcode
        state.upsert_episodes(self.conn, [
            Episode(1, 3, "Mid-Life Chrysalis", 1320.0, aired_season=1,
                    aired_number=8)])
        state.set_project(self.conn, episode_order="dvd")
        did = self.add_disc([title(4, 1320, [1320])])
        tid = state.title_id(self.conn, did, 4)
        state.set_assignment(self.conn, tid, [state.episode_id(self.conn, 1, 3)],
                             status="confirmed", decided_by="human")
        [rec] = export_mod.build_records(self.conn)
        self.assertEqual(rec["episode_order"], "TMDB episode group 'dvd'")
        self.assertEqual(rec["episode_order_id"], "dvd")
        self.assertEqual(rec["aired"], ["S01E08"])        # cross-reference
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            identify.emit_rip_commands([rec], "Fast 1080p30")
        self.assertIn("# Episode numbering: TMDB episode group 'dvd'", buf.getvalue())
        meta = transcode.meta_fields(rec, "Show", 2004, 2418)
        self.assertEqual(meta["VJ_ORDER"], "TMDB episode group 'dvd'")

    def test_aired_project_has_no_aired_cross_reference(self):
        state.upsert_episodes(self.conn, [ep(1, 1, "Pilot", 1320.0)])
        did = self.add_disc([title(1, 1320, [1320])])
        state.set_assignment(self.conn, state.title_id(self.conn, did, 1),
                             [state.episode_id(self.conn, 1, 1)], status="confirmed", decided_by="human")
        [rec] = export_mod.build_records(self.conn)
        self.assertEqual(rec["episode_order"], "TMDB aired order")
        self.assertNotIn("aired", rec)


if __name__ == "__main__":
    unittest.main()
