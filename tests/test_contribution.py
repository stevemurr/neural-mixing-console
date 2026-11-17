import unittest

import torch
from v13.models.contribution_console import ContributionConsole


class ContributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.console = ContributionConsole()

    def test_neutral_render(self):
        stems = torch.randn(1, 3, 2, 16384, generator=torch.Generator().manual_seed(0))
        mix, _ = self.console(stems, torch.zeros(1, 26))
        torch.testing.assert_close(mix, stems.sum(1), rtol=1e-4, atol=1e-5)

    def test_dominant_track(self):
        gains = self.console.distribute_delta(torch.tensor([[6.]]),
                                               torch.log1p(torch.tensor([[[10.], [1.]]])))
        self.assertGreater(gains[0, 0, 0], gains[0, 1, 0])

    def test_equal_contributors(self):
        gains = self.console.distribute_delta(torch.full((1, 26), 6.), torch.ones(1, 4, 26))
        torch.testing.assert_close(gains, torch.full_like(gains, 6.))
