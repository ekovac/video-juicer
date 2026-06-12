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


if __name__ == "__main__":
    unittest.main()
