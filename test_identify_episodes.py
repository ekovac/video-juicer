"""Unit tests for identify_episodes (no disc images or network needed).

Run: python3 -m unittest test_identify_episodes -v
"""

import struct
import unittest
from pathlib import Path

import identify_episodes as ie


def title(id, dur, chapters=(), n_audio=2, n_sub=3, cells=1):
    return ie.Title(id=id, duration=float(dur), chapters=list(chapters),
                    n_audio=n_audio, n_sub=n_sub, cells=cells, order_key=id)


def disc(titles, fmt="dvd", name="TEST.iso"):
    return ie.Disc(path=Path(name), format=fmt, label="TEST", titles=titles)


def episodes(season, runtimes):
    return [ie.Episode(season=season, number=i + 1, name=f"Episode {i + 1}",
                       runtime=rt) for i, rt in enumerate(runtimes)]


# Real S1D2 layout: play-all (5 chapters), 5 episodes, then extras that
# include a 25-minute featurette and a 21-minute extra (both episode-length).
def s1d2_titles(with_play_all=True):
    eps = [title(2, 1356.5, [1356.5]), title(3, 1353.0, [1353.0]),
           title(4, 1353.7, [1353.7]), title(5, 1348.7, [1348.7]),
           title(6, 1416.9, [1416.9])]
    extras = [title(7, 24.0, n_sub=3),
              title(8, 1505.0, n_sub=0),    # 25 min featurette: the trap
              title(9, 686.8, n_sub=0),
              title(16, 1283.3, n_sub=0)]   # 21 min extra: also a trap
    ts = list(eps) + extras
    if with_play_all:
        pa = title(1, sum(t.duration for t in eps),
                   [t.duration for t in eps])
        ts = [pa] + ts
    return ts


S1D2_EPISODES = episodes(1, [1320, 1320, 1320, 1320, 1380])  # E09-E13


class PlayAllTest(unittest.TestCase):
    def test_one_chapter_per_episode(self):
        ts = s1d2_titles()
        pa, matched = ie.detect_play_all(ts)
        self.assertEqual(pa.id, 1)
        self.assertEqual([t.id for t in matched], [2, 3, 4, 5, 6])

    def test_multi_chapter_per_episode(self):
        # S7-style: play-all has several chapters per episode
        ep_durs = [1354.1, 1353.7, 1413.6, 1353.6, 1353.2]
        chapters = []
        for d in ep_durs:
            chapters += [d * 0.4, d * 0.35, d * 0.25]
        ts = [title(1, sum(ep_durs), chapters)] + [
            title(i + 2, d, [d]) for i, d in enumerate(ep_durs)]
        pa, matched = ie.detect_play_all(ts)
        self.assertEqual(pa.id, 1)
        self.assertEqual(len(matched), 5)

    def test_no_play_all(self):
        self.assertIsNone(ie.detect_play_all(s1d2_titles(with_play_all=False)))


class DvdPlayAllTest(unittest.TestCase):
    # detect_play_all's duration-sum FALLBACK (Broken Saints D3/D4 shape): a
    # play-all with no usable chapter marks, whose runtime == the sum of the
    # episode titles (sharing its audio layout), plus low-audio extras/dummies.
    def _disc(self, ep_durs, extra_durs=(), pa_audio=3):
        eps = [title(i + 3, d, [d], n_audio=pa_audio) for i, d in enumerate(ep_durs)]
        pa = title(1, sum(ep_durs), [], n_audio=pa_audio)   # no chapters
        dummy = title(2, 1.0, [], n_audio=0)
        extras = [title(50 + i, d, [d], n_audio=1) for i, d in enumerate(extra_durs)]
        return [pa, dummy] + eps + extras

    def test_sum_match_finds_playall_and_episodes(self):
        ts = self._disc([998, 606, 549, 660, 1758], extra_durs=(1283, 686))
        pa, matched = ie.detect_play_all(ts)
        self.assertEqual(pa.id, 1)
        self.assertEqual([t.id for t in matched], [3, 4, 5, 6, 7])  # not extras

    def test_episode_candidates_prefer_playall_over_length_band(self):
        # bogus uniform runtimes (Broken Saints: TMDB says 9 min for all) would
        # make the length band miss the real 16-29 min episodes; the play-all
        # set overrides it AND marks the titles so ocr_identify skips 2x-doubling.
        ts = self._disc([998, 606, 549, 660, 1758])
        d = disc(ts)                                   # fmt='dvd' (default)
        pool = [ie.Episode(1, i + 1, f"E{i+1}", 540) for i in range(24)]
        cands = ie.episode_candidates(d, pool)
        self.assertEqual([t.id for t in cands], [3, 4, 5, 6, 7])
        self.assertTrue(all(t.kind == "episode-candidate" for t in cands))

    def test_episode_with_extra_audio_track_still_counted(self):
        # a real episode may carry an extra commentary (n_audio 4 vs play-all 3)
        ts = self._disc([998, 606, 549, 660])
        ts[4].n_audio = 4                              # title id 5: 4 audio
        pa, matched = ie.detect_play_all(ts)
        self.assertIn(5, [t.id for t in matched])

    def test_chapter_match_precedes_sum_match(self):
        # when the play-all HAS chapter marks at episode boundaries, the precise
        # chapter-match wins (works even where sum-match would over-include)
        ep_durs = [1356.0, 1353.0, 1349.0, 1417.0]
        pa = title(1, sum(ep_durs), list(ep_durs), n_audio=1)   # 4 chapters
        eps = [title(i + 2, d, [d], n_audio=1) for i, d in enumerate(ep_durs)]
        found_pa, matched = ie.detect_play_all([pa] + eps)
        self.assertEqual(found_pa.id, 1)
        self.assertEqual([t.id for t in matched], [2, 3, 4, 5])

    def test_no_playall_when_nothing_concatenates(self):
        # ordinary disc: no chapters, episodes don't sum to a single title
        ts = [title(i + 1, 1320, [1320], n_audio=2) for i in range(5)]
        self.assertIsNone(ie.detect_play_all(ts))


class ClassifyTest(unittest.TestCase):
    def test_duplicate_titles_dropped(self):
        ts = [title(1, 1350, [1350]), title(2, 1350, [1350])]
        d = disc(ts)
        cands = ie.classify_disc(d, 1320.0)
        self.assertEqual(len(cands), 1)
        self.assertEqual(ts[1].kind, "junk")

    def test_signature_outlier_penalized(self):
        ts = s1d2_titles(with_play_all=False)
        cands = ie.classify_disc(disc(ts), 1320.0)
        by_id = {t.id: t for t in ts}
        self.assertGreater(by_id[2].evidence, by_id[8].evidence)


class AlignTest(unittest.TestCase):
    def _align_disc(self, ts, eps, expected_runtime=1320.0):
        d = disc(ts)
        cands = [(d, t) for t in ie.classify_disc(d, expected_runtime)]
        return ie.align(cands, eps)

    def test_s1d2_with_play_all(self):
        got, leftovers, missed = self._align_disc(s1d2_titles(), S1D2_EPISODES)
        self.assertEqual([a.title.id for a in got], [2, 3, 4, 5, 6])
        self.assertEqual([a.episodes[0].number for a in got], [1, 2, 3, 4, 5])
        self.assertEqual(missed, [])
        self.assertIn(8, [t.id for _, t in leftovers])

    def test_s1d2_trap_without_play_all(self):
        # The 25-min featurette and 21-min extra must lose to the real
        # episodes purely on order + duration + stream signature.
        got, leftovers, missed = self._align_disc(
            s1d2_titles(with_play_all=False), S1D2_EPISODES)
        self.assertEqual([a.title.id for a in got], [2, 3, 4, 5, 6])
        self.assertEqual(missed, [])

    def test_broadcast_slot_calibration(self):
        # TMDB says 30-minute slots; disc carries ~22.5-minute episodes.
        ts = [title(i + 1, d, [d]) for i, d in enumerate(
            [1413, 1367, 1358, 1368])]
        eps = episodes(4, [1800, 1800, 1800, 1800])
        got, leftovers, missed = self._align_disc(ts, eps, 1800.0)
        self.assertEqual(len(got), 4)
        self.assertEqual(missed, [])

    def test_two_parter_merge(self):
        # one 45-minute disc title covering two 22-minute TMDB episodes
        ts = [title(1, 1340, [1340]), title(2, 2700, [2700])]
        eps = episodes(1, [1320, 1350, 1350])
        got, leftovers, missed = self._align_disc(ts, eps)
        self.assertEqual(missed, [])
        merged = [a for a in got if len(a.episodes) == 2]
        self.assertEqual(len(merged), 1)
        self.assertEqual([e.number for e in merged[0].episodes], [2, 3])

    def test_missing_episode_reported(self):
        ts = [title(1, 1340, [1340])]
        eps = episodes(1, [1320, 1320])
        got, leftovers, missed = self._align_disc(ts, eps)
        self.assertEqual(len(got), 1)
        self.assertEqual(len(missed), 1)

    def test_null_runtimes_order_only(self):
        ts = [title(1, 1340, [1340]), title(2, 1500, [1500])]
        eps = [ie.Episode(1, 1, "A", None), ie.Episode(1, 2, "B", None)]
        got, _, missed = ie.align([(disc(ts), t) for t in ts], eps)
        self.assertEqual([a.episodes[0].number for a in got], [1, 2])
        self.assertEqual(missed, [])


class FuzzyTest(unittest.TestCase):
    EPS = [ie.Episode(1, 1, "Dia de los Dangerous!", 1320),
           ie.Episode(1, 2, "Careers in Science", 1320),
           ie.Episode(1, 5, "Eeney, Meeney, Miney… Magic!", 1320)]

    def test_substring_with_surrounding_text(self):
        ep, score = ie.fuzzy_best(
            "the\nVENTURE\nBROS.\n\"Día de los Dangerous\"\n"
            "PRESENTED IN GLORIOUS EXTRA COLOR", self.EPS)
        self.assertEqual(ep.number, 1)
        self.assertGreaterEqual(score, 0.8)

    def test_diacritics_and_case(self):
        ep, score = ie.fuzzy_best("EENEY MEENEY MINEY MAGIC", self.EPS)
        self.assertEqual(ep.number, 5)
        self.assertGreaterEqual(score, 0.8)

    def test_credits_do_not_match(self):
        _, score = ie.fuzzy_best("EXECUTIVE PRODUCER\nJackson Publick", self.EPS)
        self.assertLess(score, 0.8)

    def test_empty(self):
        ep, score = ie.fuzzy_best("", self.EPS)
        self.assertIsNone(ep)

    # Two-parters: TMDB writes "(1)/(2)"; cards say "PART ONE/I/1".
    PARTS = [ie.Episode(4, 1, "Storm Front (1)", 2640),
             ie.Episode(4, 2, "Storm Front (2)", 2640),
             ie.Episode(2, 1, "Shockwave (2)", 2640)]

    def test_part_word_matches_paren_number(self):
        ep, score = ie.fuzzy_best('"Storm Front" PART ONE', self.PARTS)
        self.assertEqual((ep.season, ep.number), (4, 1))
        self.assertGreaterEqual(score, 0.85)

    def test_part_roman_matches_and_discriminates(self):
        # "PART II" must pick (2), not (1)
        ep, score = ie.fuzzy_best('STORM FRONT, PART II', self.PARTS)
        self.assertEqual((ep.season, ep.number), (4, 2))
        self.assertGreaterEqual(score, 0.85)

    def test_part_digit_matches(self):
        ep, score = ie.fuzzy_best('"Shockwave" Part 2', self.PARTS)
        self.assertEqual((ep.season, ep.number), (2, 1))
        self.assertGreaterEqual(score, 0.85)

    def test_canon_parts_helper(self):
        self.assertEqual(ie.canon_parts("storm front part one"), "storm front 1")
        self.assertEqual(ie.canon_parts("a part ii b"), "a 2 b")
        # only number-words after "part" convert; ordinary words untouched
        self.assertEqual(ie.canon_parts("part of the crew"), "part of the crew")

    # Short single-word titles collide with credits/readouts/reasoning text in
    # the opening-credits window (observed on Star Trek: Enterprise S2).
    SHORT = [ie.Episode(2, 13, "Dawn", 2640),
             ie.Episode(2, 20, "Horizon", 2640),
             ie.Episode(2, 15, "Cease Fire", 2640)]

    def test_short_title_not_matched_in_credit_name(self):
        # "Dawn" must not match the producer's first name
        ep, score = ie.fuzzy_best("PRODUCER\nDAWN VELAZQUEZ", self.SHORT)
        self.assertLess(score, 0.8)

    def test_short_title_not_matched_as_subword(self):
        # "Horizon" must not match inside "horizontal"
        _, score = ie.fuzzy_best("three horizontal lines on the emblem", self.SHORT)
        self.assertLess(score, 0.8)

    def test_short_title_not_matched_in_reasoning(self):
        # the VLM's own reasoning enumerating words must not trip a match
        _, score = ie.fuzzy_best(
            "let's look at the image. words like tropics, equinoctials, "
            "horizon, etc. wait, let me check each part", self.SHORT)
        self.assertLess(score, 0.8)

    def test_short_title_on_structured_card_matches(self):
        # Avatar: "CHAPTER TEN: JET" — short title, no credit/reasoning noise
        eps = [ie.Episode(1, 10, "Jet", 1440),
               ie.Episode(1, 1, "The Boy in the Iceberg", 1440)]
        ep, score = ie.fuzzy_best("BOOK ONE: WATER  CHAPTER TEN: JET", eps)
        self.assertEqual(ep.number, 10)
        self.assertGreaterEqual(score, 0.85)

    def test_short_title_clean_card_matches(self):
        # a real title card (title dominates the frame) still matches
        ep, score = ie.fuzzy_best('"Horizon"', self.SHORT)
        self.assertEqual(ep.number, 20)
        self.assertGreaterEqual(score, 0.8)

    def test_multiword_title_matches_with_branding(self):
        # distinctive multi-word title verbatim amid branding stays conclusive
        ep, score = ie.fuzzy_best(
            'STAR TREK ENTERPRISE "Cease Fire" act one', self.SHORT)
        self.assertEqual(ep.number, 15)
        self.assertGreaterEqual(score, 0.8)

    # Regression (The Magicians): a thinking-model reasoning dump that *names*
    # a distinctive multi-word title once scored 1.0 — the `distinctive` flag
    # bypassed the incidental guard — and overrode correct metadata (E01->E04).
    MAGICIANS = [ie.Episode(1, 1, "Unauthorized Magic", 2520),
                 ie.Episode(1, 4, "The World in the Walls", 2520)]

    def test_distinctive_title_in_reasoning_dump_rejected(self):
        _, score = ie.fuzzy_best(
            "Got it, let's look at the image and transcribe all the text. "
            "The World in the Walls", self.MAGICIANS)
        self.assertLess(score, 0.8)

    def test_distinctive_title_with_credit_line_still_matches(self):
        # a genuine card carrying a credit line must stay conclusive (credits,
        # unlike reasoning, don't demote a distinctive verbatim hit)
        ep, score = ie.fuzzy_best(
            "The World in the Walls   directed by Chris Fisher", self.MAGICIANS)
        self.assertEqual(ep.number, 4)
        self.assertGreaterEqual(score, 0.9)


def build_mpls(playitems, marks):
    """Minimal valid MPLS: playitems=[(clip, in_t, out_t)], marks=[(type, ref, tick)]."""
    items = b""
    for clip, in_t, out_t in playitems:
        body = clip.encode() + b"M2TS" + struct.pack(">HB", 0, 0)
        body += struct.pack(">II", in_t, out_t)
        body += bytes(24)  # uo_mask etc + empty STN
        items += struct.pack(">H", len(body)) + body
    playlist = struct.pack(">IHHH", 0, 0, len(playitems), 0)[:10]
    playlist = struct.pack(">I", 6 + len(items)) + struct.pack(">HHH", 0, len(playitems), 0) + items
    mark_blob = struct.pack(">IH", 2 + 14 * len(marks), len(marks))
    for mtype, ref, tick in marks:
        mark_blob += struct.pack(">BBHIHI", 0, mtype, ref, tick, 0, 0)
    header_len = 40
    playlist_start = header_len
    mark_start = playlist_start + len(playlist)
    buf = b"MPLS0200" + struct.pack(">III", playlist_start, mark_start,
                                    mark_start + len(mark_blob))
    buf += bytes(header_len - len(buf)) + playlist + mark_blob
    return buf


class MplsTest(unittest.TestCase):
    def test_duration_and_chapters(self):
        forty_five_min = 45000 * 60 * 22  # 22 minutes in 45 kHz ticks
        buf = build_mpls(
            [("00001", 45000 * 100, 45000 * 100 + forty_five_min)],
            [(1, 0, 45000 * 100), (1, 0, 45000 * 100 + forty_five_min // 2)])
        info = ie.parse_mpls(buf)
        self.assertAlmostEqual(info["duration"], 22 * 60, places=1)
        self.assertEqual(len(info["chapters"]), 2)
        self.assertAlmostEqual(info["chapters"][0], 11 * 60, places=1)

    def test_garbage_rejected(self):
        self.assertIsNone(ie.parse_mpls(b"not an mpls file at all"))
        self.assertIsNone(ie.parse_mpls(b""))


class PlayAllOrderTest(unittest.TestCase):
    def _t(self, id, dur, clips):
        return ie.Title(id=id, duration=float(dur), chapters=[],
                        clips=tuple(clips), order_key=id)

    def test_clip_order_overrides_scrambled_mpls(self):
        # play-all clips give broadcast order a,b,c; .mpls ids are scrambled
        pa = self._t(100, 4500, ("intro", "a", "b", "c"))
        e_c = self._t(5, 1500, ("intro", "c"))   # mpls5 but plays last
        e_a = self._t(3, 1500, ("intro", "a"))   # mpls3 but plays first
        e_b = self._t(4, 1500, ("intro", "b"))
        found = ie.order_by_playall([pa, e_c, e_a, e_b])
        self.assertIs(found, pa)
        self.assertEqual(found.kind, "play-all")
        self.assertEqual((e_a.order_key, e_b.order_key, e_c.order_key), (0, 1, 2))

    def test_no_playall_returns_none(self):
        ts = [self._t(i, 1500, (c,)) for i, c in enumerate("abcd")]
        self.assertIsNone(ie.order_by_playall(ts))


class SubsetDedupTest(unittest.TestCase):
    def _t(self, id, dur, clips):
        return ie.Title(id=id, duration=float(dur), chapters=[], clips=clips,
                        order_key=id)

    def test_body_only_dropped_for_with_recap(self):
        # Avatar pattern: (body,) is a subset of (intro, recap, body)
        ts = [self._t(601, 1421, ("01100", "01062")),
              self._t(1601, 1420, ("01062",))]
        kept = ie.dedup_subset_playlists(ts)
        self.assertEqual([t.id for t in kept], [601])  # keep the fuller one

    def test_playall_not_swallow_episodes(self):
        # a long play-all is a superset of episode clips but ~Nx longer;
        # the 1.5x guard keeps the episodes
        ep1 = self._t(1, 1400, ("A",))
        ep2 = self._t(2, 1400, ("B",))
        playall = self._t(99, 2810, ("A", "B"))
        kept = ie.dedup_subset_playlists([ep1, ep2, playall])
        self.assertEqual(sorted(t.id for t in kept), [1, 2, 99])

    def test_distinct_episodes_kept(self):
        ts = [self._t(1, 1400, ("intro", "a")), self._t(2, 1400, ("intro", "b"))]
        self.assertEqual(len(ie.dedup_subset_playlists(ts)), 2)


class EpisodeGroupTest(unittest.TestCase):
    class StubTmdb:
        def episode_groups(self, tv_id):
            return [{"name": "DVD Order", "type": 3, "id": "abc123"},
                    {"name": "Digital Order", "type": 4, "id": "def456"}]

        def episode_group(self, group_id):
            assert group_id == "abc123"
            return {"groups": [
                {"order": 0, "name": "Specials", "episodes": [
                    {"order": 0, "name": "Pilot Special", "runtime": 22,
                     "season_number": 0, "episode_number": 1}]},
                {"order": 1, "name": "Season 1", "episodes": [
                    # DVD order swaps the aired order of these two
                    {"order": 0, "name": "B", "runtime": 22,
                     "season_number": 1, "episode_number": 2},
                    {"order": 1, "name": "A", "runtime": None,
                     "season_number": 1, "episode_number": 1}]},
            ]}

    def test_dvd_alias_resolves_type_3(self):
        pools, specials = ie.grouped_seasons(self.StubTmdb(), 99, "dvd", None)
        self.assertEqual(list(pools), [1])
        self.assertEqual([e.name for e in pools[1]], ["B", "A"])
        # group-relative numbering, aired numbering preserved as x-ref
        self.assertEqual([(e.number, e.aired_number) for e in pools[1]],
                         [(1, 2), (2, 1)])
        # null runtime falls back to the group median
        self.assertEqual(pools[1][1].runtime, 22 * 60.0)
        self.assertEqual([e.name for e in specials], ["Pilot Special"])

    def test_missing_group_type_errors_with_listing(self):
        with self.assertRaises(SystemExit) as ctx:
            ie.grouped_seasons(self.StubTmdb(), 99, "production", None)
        self.assertIn("DVD Order", str(ctx.exception))


class HandBrakeTitleTest(unittest.TestCase):
    # abbreviated HandBrakeCLI --json output (logs + the JSON Title Set block)
    HB_OUT = (
        'Version: {"Name":"HandBrake"}\n'
        '[12:00:00] scanning\n'
        'JSON Title Set: {\n'
        '  "MainFeature": 2,\n'
        '  "TitleList": [\n'
        '    {"Index": 1, "Playlist": "0", "Name": "preroll"},\n'
        '    {"Index": 2, "Playlist": "1", "Name": "ep"},\n'
        '    {"Index": 9, "Playlist": "10", "Name": "ep"}\n'
        '  ]\n'
        '}\n')

    def test_parse_playlist_to_title_index(self):
        m = ie._parse_hb_titles(self.HB_OUT)
        self.assertEqual(m, {0: 1, 1: 2, 10: 9})

    def test_parse_garbage(self):
        self.assertEqual(ie._parse_hb_titles("no json here"), {})
        self.assertEqual(ie._parse_hb_titles("JSON Title Set: {bad"), {})

    def test_rip_title_bluray_translates(self):
        d = disc([title(1, 2640)], fmt="bluray")
        d.hb_map = {1: 2, 10: 9}
        self.assertEqual(ie.rip_title_number(d, d.titles[0]), 2)  # mpls 1 -> t2

    def test_rip_title_bluray_missing_falls_back(self):
        d = disc([title(7, 2640)], fmt="bluray")
        d.hb_map = {1: 2}  # playlist 7 absent
        self.assertEqual(ie.rip_title_number(d, d.titles[0]), 7)

    def test_rip_title_dvd_is_identity(self):
        d = disc([title(3, 1320)], fmt="dvd")
        self.assertEqual(ie.rip_title_number(d, d.titles[0]), 3)


class OrderabilityTest(unittest.TestCase):
    def _asgs(self, d, specs, names=None):
        # specs: list of (runtime, delta); names optional
        names = names or [f"E{i+1}" for i in range(len(specs))]
        out = []
        for i, (rt, delta) in enumerate(specs):
            ep = ie.Episode(1, i + 1, names[i], rt)
            t = title(i + 1, rt)
            out.append(ie.Assignment(d, t, [ep], delta, "high"))
        return out

    def test_dvd_with_disc_hint_orderable(self):
        d = disc([], fmt="dvd")
        d.disc_hint = 1                                # S?D? in the filename
        ok, _ = ie.assess_ordering(d, self._asgs(d, [(1440, 5)] * 5))
        self.assertTrue(ok)

    def test_dvd_same_runtime_no_hint_unverifiable(self):
        # Sonic SatAM: ~all 22.7min, no S?D? hint -> aligner can drop/shift a
        # title and renumber, so cross-disc numbering isn't verifiable
        d = disc([], fmt="dvd")
        ok, why = ie.assess_ordering(d, self._asgs(d, [(1360, 5)] * 5))
        self.assertFalse(ok)
        self.assertIn("no disc hint", why)

    def test_dvd_runtime_separable_no_hint_orderable(self):
        # varied runtimes anchor the numbering even without a hint
        d = disc([], fmt="dvd")
        ok, _ = ie.assess_ordering(
            d, self._asgs(d, [(600, 5), (1000, 5), (1500, 5), (2900, 5)]))
        self.assertTrue(ok)

    def test_bluray_same_runtime_unverifiable(self):
        d = disc([], fmt="bluray")
        ok, why = ie.assess_ordering(d, self._asgs(d, [(1500, 8)] * 5))
        self.assertFalse(ok)
        self.assertIn("runtime-separable", why)

    def test_bluray_near_equal_runtimes_unverifiable(self):
        # MOTU D1 case: 25-27min episodes, small deltas, but two are equal
        d = disc([], fmt="bluray")
        ok, why = ie.assess_ordering(
            d, self._asgs(d, [(1500, 40), (1500, 70), (1560, 70), (1560, 20), (1620, 120)]))
        self.assertFalse(ok)
        self.assertIn("runtime-separable", why)

    def test_bluray_scrambled_large_deltas_unverifiable(self):
        # varied runtimes but big deltas => monotonic order conflicts => scramble
        d = disc([], fmt="bluray")
        ok, why = ie.assess_ordering(
            d, self._asgs(d, [(1400, 40), (1500, 300), (1900, 370), (1300, 130)]))
        self.assertFalse(ok)
        self.assertIn("scrambled", why)

    def test_bluray_runtimes_fit_orderable(self):
        # varied runtimes, small deltas => order fits the runtimes
        d = disc([], fmt="bluray")
        ok, _ = ie.assess_ordering(
            d, self._asgs(d, [(1200, 10), (1500, 12), (1800, 8), (2100, 15)]))
        self.assertTrue(ok)

    def test_bluray_multipart_corroborates(self):
        d = disc([], fmt="bluray")
        names = ["Storm Front (1)", "Storm Front (2)", "Home", "Borderland (1)"]
        ok, why = ie.assess_ordering(d, self._asgs(d, [(2640, 300)] * 4, names))
        self.assertTrue(ok)
        self.assertIn("multi-part", why)

    def test_bluray_play_all_corroborates(self):
        d = disc([title(1, 7200, [1440]*5)], fmt="bluray")
        d.titles[0].kind = "play-all"
        ok, why = ie.assess_ordering(d, self._asgs(d, [(1440, 200)] * 5))
        self.assertTrue(ok)
        self.assertIn("play-all", why)


class EliminationTest(unittest.TestCase):
    def test_premiere_recovered_by_elimination(self):
        # MOTU: D1 has E02-E05 + one unmatched (E01, no title card);
        # D2 has E07-E10 + one unmatched (E06). Constraint-propagate.
        d1 = ie.Disc(path=Path("D1"), format="bluray", label="")
        d2 = ie.Disc(path=Path("D2"), format="bluray", label="")
        ep = lambda n: ie.Episode(1, n, f"E{n}", 1500)
        asg = lambda d, n: ie.Assignment(d, title(n, 1500), [ep(n)], 5.0, "high")
        final = [asg(d1, n) for n in (2, 3, 4, 5)] + [asg(d2, n) for n in (7, 8, 9, 10)]
        leftovers = [(d1, title(91, 1500)), (d2, title(92, 1500))]
        f2, l2, m2 = ie.recover_by_elimination(final, leftovers, [ep(1), ep(6)])
        self.assertEqual(m2, [])
        self.assertEqual(l2, [])
        elim = {a.episodes[0].number: a.disc.path for a in f2 if a.method == "elimination"}
        self.assertEqual(elim, {1: Path("D1"), 6: Path("D2")})

    def test_ambiguous_left_alone(self):
        # two unmatched on one disc -> can't disambiguate -> leave them
        d = ie.Disc(path=Path("D"), format="bluray", label="")
        ep = lambda n: ie.Episode(1, n, f"E{n}", 1500)
        final = [ie.Assignment(d, title(n, 1500), [ep(n)], 5.0, "high")
                 for n in (2, 3)]
        f2, l2, m2 = ie.recover_by_elimination(
            final, [(d, title(91, 1500)), (d, title(92, 1500))], [ep(1), ep(4)])
        self.assertEqual(len(l2), 2)
        self.assertEqual(len(m2), 2)


class LengthFilterTest(unittest.TestCase):
    def _pool(self, runtimes):
        return [ie.Episode(1, i + 1, f"E{i+1}", rt)
                for i, rt in enumerate(runtimes)]

    def test_featurette_near_real_runtime_admitted(self):
        # Avatar: 24/25/26-min episodes -> 25.9-min featurette is a valid len
        lengths, tol = ie.valid_episode_lengths(self._pool([1440, 1500, 1560] * 5))
        self.assertLessEqual(min(abs(1554 - v) for v in lengths), tol)

    def test_gap_length_excluded(self):
        # a 35-min item on a 24-min show matches no single/double/multiple
        lengths, tol = ie.valid_episode_lengths(self._pool([1440] * 20))
        self.assertGreater(min(abs(2100 - v) for v in lengths), tol)

    def test_combined_double_admitted(self):
        lengths, tol = ie.valid_episode_lengths(self._pool([1440] * 20))
        self.assertLessEqual(min(abs(2880 - v) for v in lengths), tol)  # 2 eps

    def test_long_episode_robust_to_missing_runtime(self):
        # no TMDB entry near 47 min, but 2*median catches it
        lengths, tol = ie.valid_episode_lengths(self._pool([1440] * 20))
        self.assertLessEqual(min(abs(2820 - v) for v in lengths), tol)

    def test_no_runtimes_returns_none(self):
        self.assertIsNone(ie.valid_episode_lengths(
            [ie.Episode(1, 1, "x", None)]))


class CrossDiscTest(unittest.TestCase):
    def test_contiguous_run_wins(self):
        def ep(img, n):
            return {"image": img, "title": n, "kind": "episode", "season": 1,
                    "episodes": [n], "episode_name": f"E{n}", "title_seconds": 1440}
        # B1D2 owns the E12-16 run; B1D3 owns E17-20 but also claims E14
        recs = ([ep("B1D2", n) for n in (12, 13, 14, 15, 16)]
                + [ep("B1D3", n) for n in (14, 17, 18, 19, 20)])
        out = ie.resolve_cross_disc(recs)
        e14 = [r for r in out if r.get("episodes") == [14]]
        self.assertEqual(len(e14), 1)                       # one claim left
        self.assertEqual(e14[0]["image"], "B1D2")           # contiguous disc won
        demoted = [r for r in out if r["image"] == "B1D3" and r["kind"] == "extra"]
        self.assertEqual(len(demoted), 1)
        self.assertIn("cross-disc", demoted[0]["note"])

    def test_same_disc_double_not_demoted(self):
        # a single + its combined double on one disc is fine, not cross-disc
        recs = [{"image": "D", "title": 1, "kind": "episode", "season": 1,
                 "episodes": [12], "episode_name": "A", "title_seconds": 1440},
                {"image": "D", "title": 2, "kind": "episode", "season": 1,
                 "episodes": [12, 13], "episode_name": "A & B", "title_seconds": 2880}]
        out = ie.resolve_cross_disc(recs)
        self.assertEqual(sum(r["kind"] == "episode" for r in out), 2)


class CollisionTest(unittest.TestCase):
    def asg(self, d, tid, dur, eps):
        return ie.Assignment(d, title(tid, dur), eps, 0.0, "high", eps[0].name)

    def eps(self):
        return (ie.Episode(2, 11, "K", 1440), ie.Episode(2, 12, "Serpent", 1440),
                ie.Episode(2, 13, "Drill", 1440))

    def test_double_sole_source_demotes_redundant_single(self):
        d = disc([])
        e11, e12, e13 = self.eps()
        single11 = self.asg(d, 66, 1440, [e11])
        single12 = self.asg(d, 56, 1440, [e12])       # standalone E12
        double = self.asg(d, 67, 2880, [e12, e13])    # E13 lives only here
        leftovers = []
        final, claimed = ie.resolve_assignment_collisions(
            [single11, single12, double], leftovers)
        final_titles = {a.title.id for a in final}
        self.assertIn(67, final_titles)               # double kept (carries E13)
        self.assertIn(66, final_titles)               # untouched single kept
        self.assertNotIn(56, final_titles)            # redundant single demoted
        self.assertIn(56, {t.id for _, t in leftovers})
        self.assertEqual(claimed[(2, 12)].title.id, 67)
        self.assertEqual(claimed[(2, 13)].title.id, 67)

    def test_both_singles_present_double_dropped(self):
        d = disc([])
        _, e12, e13 = self.eps()
        single12 = self.asg(d, 56, 1440, [e12])
        single13 = self.asg(d, 58, 1440, [e13])
        double = self.asg(d, 67, 2880, [e12, e13])
        leftovers = []
        final, _ = ie.resolve_assignment_collisions(
            [single12, single13, double], leftovers)
        final_titles = {a.title.id for a in final}
        self.assertEqual(final_titles, {56, 58})      # both standalones win
        self.assertIn(67, {t.id for _, t in leftovers})  # double fully redundant

    def test_cross_disc_single_not_demoted_here(self):
        # different discs -> left for resolve_cross_disc, not demoted here
        da, db = disc([], name="A"), disc([], name="B")
        _, e12, e13 = self.eps()
        single12 = self.asg(da, 56, 1440, [e12])
        double = self.asg(db, 67, 2880, [e12, e13])
        leftovers = []
        final, _ = ie.resolve_assignment_collisions([single12, double], leftovers)
        self.assertEqual({a.title.id for a in final}, {56, 67})
        self.assertEqual(leftovers, [])


class FormatOutlierTest(unittest.TestCase):
    def asg(self, num, vf):
        t = title(num, 1440)
        t.video_format = vf
        return ie.Assignment(disc=disc([], fmt="bluray"), title=t,
                             episodes=[ie.Episode(3, num, f"E{num}", 1440)],
                             delta=0.0, confidence="high")

    def test_minority_flagged_against_majority(self):
        # Avatar shape: a 1080p season with a 480i finale block
        asgs = [self.asg(n, "1080p") for n in range(1, 18)] + \
               [self.asg(n, "480i") for n in range(18, 22)]
        maj, out = ie.format_outliers(asgs)
        self.assertEqual(maj, "1080p")
        self.assertEqual(sorted(a.episodes[0].number for a in out), [18, 19, 20, 21])

    def test_uniform_format_no_outliers(self):
        asgs = [self.asg(n, "1080p") for n in range(1, 13)]
        maj, out = ie.format_outliers(asgs)
        self.assertEqual(out, [])

    def test_unknown_formats_ignored(self):
        # DVD / unparsed STN -> video_format None -> no false warning
        asgs = [self.asg(n, None) for n in range(1, 13)]
        self.assertEqual(ie.format_outliers(asgs), (None, []))


class MergeTest(unittest.TestCase):
    def test_merge_replaces_only_reprocessed_discs(self):
        existing = [
            {"image": "discA", "kind": "episode", "season": 1, "episodes": [1]},
            {"image": "discB", "kind": "episode", "season": 1, "episodes": [2]},
            {"image": "discB", "kind": "extra", "season": 1, "episodes": []},
        ]
        # re-ran discB only; it now yields a corrected record + a new episode
        new = [
            {"image": "discB", "kind": "episode", "season": 1, "episodes": [2]},
            {"image": "discB", "kind": "episode", "season": 1, "episodes": [3]},
        ]
        merged = ie.merge_records(existing, new, [Path("discB")])
        imgs_eps = [(r["image"], r["episodes"]) for r in merged
                    if r["kind"] == "episode"]
        self.assertIn(("discA", [1]), imgs_eps)          # untouched disc kept
        self.assertIn(("discB", [3]), imgs_eps)          # new episode added
        self.assertEqual(sum(1 for r in merged if r["image"] == "discB"), 2)
        # the stale discB extra was dropped (discB replaced wholesale)
        self.assertFalse(any(r["kind"] == "extra" for r in merged))


class HintTest(unittest.TestCase):
    def test_filename_pattern(self):
        d = disc([], name="VENTURE_BROS_S3D2.iso")
        ie.parse_hints(d)
        self.assertEqual((d.season_hint, d.disc_hint), (3, 2))

    def test_volume_label_pattern(self):
        d = disc([], name="backup.iso")
        d.label = "VENTURE_BROS_VOL_1_DISC_2"
        ie.parse_hints(d)
        self.assertEqual((d.season_hint, d.disc_hint), (1, 2))

    def test_book_disc_pattern(self):
        d = disc([], name="Avatar_Book_2_Disc_3")
        ie.parse_hints(d)
        self.assertEqual((d.season_hint, d.disc_hint), (2, 3))

    def test_bare_book_label(self):
        d = disc([], name="Korra Book 4")
        ie.parse_hints(d)
        self.assertEqual((d.season_hint, d.disc_hint), (4, None))


class NamingTest(unittest.TestCase):
    def ep(self, season, number, name):
        return ie.Episode(season=season, number=number, name=name, runtime=1440)

    def test_plex_folder_layout(self):
        p = ie.suggested_filename("Enterprise", [self.ep(1, 1, "Broken Bow")],
                                  2001, 314)
        self.assertEqual(
            p, "Enterprise (2001) {tmdb-314}/Season 01/"
               "Enterprise (2001) - S01E01 - Broken Bow.mkv")

    def test_tmdb_id_only_on_show_folder(self):
        p = ie.suggested_filename("Show", [self.ep(1, 1, "X")], 2010, 99)
        top, season, fname = p.split("/")
        self.assertEqual(top, "Show (2010) {tmdb-99}")
        self.assertNotIn("tmdb", fname)

    def test_multi_episode_range(self):
        eps = [self.ep(1, 1, "Part One"), self.ep(1, 2, "Part Two")]
        p = ie.suggested_filename("Show", eps, 2010)
        self.assertIn("S01E01-E02 - Part One & Part Two", p)

    def test_specials_folder(self):
        p = ie.suggested_filename("Avatar", [self.ep(0, 3, "Bonus")], 2005)
        self.assertTrue(p.startswith("Avatar (2005)/Specials/"))

    def test_no_year(self):
        p = ie.suggested_filename("Show", [self.ep(2, 5, "X")])
        self.assertEqual(p, "Show/Season 02/Show - S02E05 - X.mkv")

    def test_illegal_chars_stripped_per_component(self):
        p = ie.suggested_filename("Star Trek: Enterprise",
                                  [self.ep(1, 1, "A/B?")], 2001)
        self.assertEqual(p.count("/"), 2)            # only the path separators
        self.assertNotIn(":", p)
        self.assertNotIn("?", p)


class RipCommandTest(unittest.TestCase):
    def rec(self):
        return [{"kind": "episode", "image": "/d/disc.iso", "title": 3,
                 "suggested_filename": "Show (2001) {tmdb-9}/Season 01/"
                                       "Show (2001) - S01E01 - X.mkv"}]

    def lines(self, *args):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            ie.emit_rip_commands(*args)
        return buf.getvalue().splitlines()

    def test_header_variables(self):
        out = self.lines(self.rec(), "Fast 1080p30", "/mnt/transcode")
        self.assertEqual(out[0], "#!/usr/bin/env bash")
        self.assertIn("PREFIX=/mnt/transcode", out)
        self.assertIn("PRESET='Fast 1080p30'", out)
        self.assertIn("HANDBRAKE_OPTS=()", out)

    def test_no_prefix_defaults_to_dot(self):
        out = self.lines(self.rec(), "Fast 1080p30")
        self.assertIn("PREFIX=.", out)

    def test_commands_reference_variables(self):
        out = self.lines(self.rec(), "Fast 1080p30", "/mnt/transcode")
        mkdir = next(l for l in out if l.startswith("mkdir"))
        rip = next(l for l in out if l.startswith("HandBrakeCLI"))
        self.assertEqual(
            mkdir, 'mkdir -p "$PREFIX/Show (2001) {tmdb-9}/Season 01"')
        self.assertIn('HandBrakeCLI "${HANDBRAKE_OPTS[@]}"', rip)
        self.assertIn("--preset \"$PRESET\"", rip)
        self.assertIn('-o "$PREFIX/Show (2001) {tmdb-9}/Season 01/'
                      'Show (2001) - S01E01 - X.mkv"', rip)
        self.assertIn("-i /d/disc.iso", rip)

    def test_image_with_spaces_is_shell_quoted(self):
        rec = self.rec()
        rec[0]["image"] = "/run/media/ST ENTERPRISE S1D1"
        rip = next(l for l in self.lines(rec, "Fast 1080p30")
                   if l.startswith("HandBrakeCLI"))
        self.assertIn("-i '/run/media/ST ENTERPRISE S1D1'", rip)

    def _ep(self, n, vf):
        return {"kind": "episode", "image": "/d/disc.iso", "title": n,
                "season": 1, "episodes": [n], "video_format": vf,
                "audio_format": "AC3" if vf == "480i" else "DTS-HDMA",
                "suggested_filename": f"Show/Season 01/Show - S01E{n:02d} - X.mkv"}

    def test_no_alt_block_when_uniform_format(self):
        recs = [self._ep(n, "1080p") for n in range(1, 6)]
        out = self.lines(recs, "P")
        self.assertNotIn("PRESET_ALT=\"$PRESET\"", out)
        self.assertFalse(any("NON-CONFORMING" in l for l in out))

    def test_outliers_split_into_alt_block(self):
        recs = [self._ep(n, "1080p") for n in range(1, 6)] + \
               [self._ep(n, "480i") for n in (6, 7)]
        out = self.lines(recs, "P")
        self.assertIn('PRESET_ALT="$PRESET"', out)
        self.assertTrue(any("NON-CONFORMING VIDEO FORMAT (480i)" in l for l in out))
        # the 480i episodes rip with $PRESET_ALT, the rest with $PRESET
        rips = [l for l in out if l.startswith("HandBrakeCLI")]
        alt = [l for l in rips if '"$PRESET_ALT"' in l]
        main = [l for l in rips if '"$PRESET"' in l]
        self.assertEqual(len(alt), 2)        # E06, E07
        self.assertEqual(len(main), 5)       # E01-E05
        # outlier lines are annotated with their format
        self.assertTrue(any(l.startswith("#") and "480i" in l and "AC3" in l
                            for l in out))

    def test_old_manifest_without_formats_unchanged(self):
        # records lacking video_format -> no majority -> single block
        recs = [{"kind": "episode", "image": "/d/d.iso", "title": n,
                 "suggested_filename": f"S/Season 01/S - S01E{n:02d} - X.mkv"}
                for n in range(1, 6)]
        out = self.lines(recs, "P")
        self.assertFalse(any("PRESET_ALT" in l for l in out))


class OllamaChatTest(unittest.TestCase):
    def _call(self, resp):
        import io, json as _json, os, tempfile
        from unittest import mock

        class FakeResp:
            def __enter__(self_): return self_
            def __exit__(self_, *a): return False
            def read(self_): return _json.dumps(resp).encode()

        fd, name = tempfile.mkstemp(suffix=".jpg")
        os.write(fd, b"x"); os.close(fd)
        try:
            with mock.patch("identify.urllib.request.urlopen",
                            return_value=FakeResp()):
                return ie.ollama_chat("m", "p", Path(name), "http://h")
        finally:
            os.unlink(name)

    def test_returns_clean_content(self):
        out = self._call({"message": {"content": '"Introitus"', "thinking": "x"},
                          "done_reason": "stop"})
        self.assertEqual(out, '"Introitus"')

    def test_truncated_midthink_returns_empty_not_thinking(self):
        # content empty + done_reason length: must NOT leak the thinking text
        out = self._call({"message": {"content": "",
                          "thinking": "Got it, let's look at the image..."},
                          "done_reason": "length"})
        self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main()
