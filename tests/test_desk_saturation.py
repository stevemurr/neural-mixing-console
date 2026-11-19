import unittest

import numpy as np

from nmc.data.raw_folder import RATE
from nmc.desk_saturation import render_colors, soft_knee, stage_spec


class ChannelSaturationTests(unittest.TestCase):
    def test_bias_has_zero_at_rest_and_unity_small_signal_slope(self):
        spec = stage_spec()
        self.assertEqual(soft_knee(0, spec["bias"]), 0)
        self.assertAlmostEqual(float(soft_knee(1e-7, spec["bias"]) / 1e-7), 1, places=6)
        _, renders = render_colors(np.zeros((RATE // 2, 2)), spec)
        self.assertTrue(all(np.count_nonzero(v) == 0 for v in renders.values()))

    def test_drive_progression_generates_even_and_odd_harmonics(self):
        time = np.arange(RATE) / RATE
        x = np.sin(2 * np.pi * 1000 * time)[:, None] * 0.1
        _, renders = render_colors(np.repeat(x, 2, axis=1), stage_spec())
        amounts = []
        for key in ("clean", "0", "6", "12", "24"):
            y = renders[key]
            np.testing.assert_array_equal(y[:, 0], y[:, 1])
            spectrum = abs(np.fft.rfft(y[:, 0]))
            harmonics = spectrum[2000:10001:1000] / spectrum[1000]
            amounts.append(np.linalg.norm(harmonics))
            if key != "clean":
                self.assertGreater(min(harmonics[:2]), 0.005)
                self.assertLess(abs(y.mean()), 1e-5)
        self.assertTrue(all(a < b for a, b in zip(amounts, amounts[1:])))
