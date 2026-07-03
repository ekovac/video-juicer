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
        # a forced-but-unavailable backend must never prune (recall-first)
        self.assertTrue(tr.frame_has_text("/any.jpg", backend="east",
                                          model="/no/model.pb"))

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
        available = tr.paddle_available() or tr.east_available()
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


if __name__ == "__main__":
    unittest.main()
