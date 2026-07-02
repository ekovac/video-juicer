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
    def test_mrl(self):
        self.assertEqual(vj._play_mrl("dvd", "/x.iso", 4), "dvd:///x.iso#4")
        self.assertTrue(vj._play_mrl("bluray", "/bd", 1).startswith("bluray:///bd"))

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
        self.assertEqual(out["mrl"], "dvd:///d/x.iso#1")
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


if __name__ == "__main__":
    unittest.main()
