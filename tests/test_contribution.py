import unittest

import torch
from nmc.eq.contribution_console import ContributionConsole
from nmc.eq.precompute import compute_target_delta_db


class ContributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.console = ContributionConsole()

    def test_neutral_render(self):
        stems = torch.randn(1, 3, 2, 16384, generator=torch.Generator().manual_seed(0))
        mix, _ = self.console(stems, torch.zeros(1, 26))
        torch.testing.assert_close(mix, stems.sum(1), rtol=1e-4, atol=1e-5)

    def test_target_measures_spectral_shape_after_matching_levels(self):
        t = torch.arange(32768) / self.console.sample_rate
        low = torch.sin(2 * torch.pi * 150 * t)
        high = torch.sin(2 * torch.pi * 4000 * t)
        stems = torch.stack([low, high]).unsqueeze(1).expand(-1, 2, -1)
        reference = stems[0] + 0.5 * stems[1]
        target = compute_target_delta_db(stems, reference, self.console)
        louder = compute_target_delta_db(stems, 4 * reference, self.console)
        torch.testing.assert_close(target, louder)
        dominant_bands = self.console.compute_C(stems.unsqueeze(0))[0].argmax(dim=1)
        self.assertGreater(target[dominant_bands[0]].item(), 0)
        self.assertLess(target[dominant_bands[1]].item(), 0)

    def test_dominant_track(self):
        gains = self.console.distribute_delta(
            torch.tensor([[6.0]]), torch.log1p(torch.tensor([[[10.0], [1.0]]]))
        )
        self.assertGreater(gains[0, 0, 0], gains[0, 1, 0])

    def test_equal_contributors(self):
        gains = self.console.distribute_delta(torch.full((1, 26), 6.0), torch.ones(1, 4, 26))
        torch.testing.assert_close(gains, torch.full_like(gains, 6.0))
