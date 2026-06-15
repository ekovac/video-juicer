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


if __name__ == "__main__":
    unittest.main()
