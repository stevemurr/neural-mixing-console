import unittest

import numpy as np

from nmc.data.raw_folder import RATE
from nmc.desk_balance import active_level, anchored_faders, loudness_faders


class DeskBalanceTests(unittest.TestCase):
    def test_loudness_uses_active_passages_and_bounds_static_gains(self):
        time = np.arange(RATE * 2) / RATE
        tone = (0.1 * np.sin(2 * np.pi * 440 * time))[:, None]
        silence = np.zeros_like(tone)
        self.assertIsNone(active_level(silence))
        self.assertAlmostEqual(
            active_level(tone), active_level(np.vstack([silence, tone])), delta=0.1
        )
        self.assertEqual(
            loudness_faders(dict(a=-45, b=-20, c=-5, d=None)), dict(a=4, b=0, c=-4, d=0)
        )

    def test_anchor_protects_overlapping_source_bands(self):
        time = np.arange(RATE) / RATE
        voice = 0.1 * np.sin(2 * np.pi * 440 * time)[:, None]
        groups = dict(
            voice=voice, overlap=2 * voice, distant=0.2 * np.sin(2 * np.pi * 7000 * time)[:, None]
        )
        faders, _ = anchored_faders(groups, dict(voice=-1, overlap=0, distant=0), "voice")
        self.assertEqual(faders["voice"], 0)
        self.assertLess(faders["overlap"], -1)
        self.assertAlmostEqual(faders["distant"], 0, delta=0.05)
