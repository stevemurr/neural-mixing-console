"""Historical v13 console with optional mono panning."""
import torch
from nmc.eq.contribution_console import ContributionConsole as EQConsole
from .mono_pan import MonoPan


class ContributionConsole(EQConsole):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pan = MonoPan()

    def forward(self, stems, target_delta_db, pan_dir=None, is_stereo=None, track_mask=None):
        mix, aux = super().forward(stems, target_delta_db, track_mask=track_mask)
        if pan_dir is not None:
            if is_stereo is None:
                is_stereo = torch.zeros_like(pan_dir, dtype=torch.bool)
            panned = self.pan(aux["eq_stems"], pan_dir, is_stereo)
            aux.update(panned_stems=panned, pan_dir=pan_dir, is_stereo=is_stereo)
            mix = panned.sum(dim=1)
        return mix, aux


def _smoke() -> None:
    """Verify (1) shapes & gradients, (2) zero-delta is identity,
    (3) contribution math gives dominant track the lion's share."""
    import math as _m
    torch.manual_seed(0)
    sr, T = 48_000, 6 * 48_000
    B, N = 2, 8
    console = ContributionConsole(sample_rate=sr, n_bins=26)

    # Test 1: shapes + gradient flow
    stems = (torch.randn(B, N, 2, T) * 0.05).requires_grad_(True)
    target_delta_db = (torch.randn(B, 26) * 3.0).requires_grad_(True)
    mix, aux = console(stems, target_delta_db)
    assert mix.shape == (B, 2, T), f"bad mix shape {mix.shape}"
    assert aux["C"].shape == (B, N, 26), f"bad C shape {aux['C'].shape}"
    assert aux["per_track_delta_db"].shape == (B, N, 26)
    mix.sum().backward()
    assert stems.grad is not None and torch.isfinite(stems.grad).all(), \
        "no/NaN gradient into stems"
    assert target_delta_db.grad is not None and torch.isfinite(target_delta_db.grad).all(), \
        "no/NaN gradient into target_delta_db"
    print(f"shapes ok, gradients flow finite. mix={tuple(mix.shape)}")

    # Test 2: zero target delta → mix is close to dry sum
    with torch.no_grad():
        mix_zero, _ = console(stems.detach(), torch.zeros(B, 26))
        dry_sum = stems.detach().sum(dim=1)
        # Gain-invariant EQ has minor non-identity even at neutral due to RMS
        # rescaling of each stem from its individual normalization.
        # Compare with a tolerance relative to typical magnitudes.
        rel = (mix_zero - dry_sum).abs().mean() / dry_sum.abs().mean()
        print(f"zero-delta mix vs dry sum rel-err: {rel.item():.2e}  (want < 1e-3)")
        assert rel < 1e-3, "zero delta should preserve dry sum"

    # Test 3: dominance — synth a band where track 0 carries all the energy,
    # apply +6 dB delta there; check track 0 gets the lion's share.
    stems_d = torch.zeros(1, N, 2, T)
    t = torch.arange(T) / sr
    # 4 kHz sine on track 0; silence on others
    stems_d[0, 0] = torch.sin(2 * _m.pi * 4000 * t).unsqueeze(0).expand(2, -1) * 0.1
    delta = torch.zeros(1, 26)
    band_idx = int(torch.searchsorted(console.eq.hz_edges, torch.tensor(4000.0))) - 1
    delta[0, band_idx] = 6.0
    _, aux_d = console(stems_d, delta)
    deltas = aux_d["per_track_delta_db"][0, :, band_idx]   # (N,)
    share_t0 = deltas[0].abs() / deltas.abs().sum().clamp(min=1e-6)
    print(f"dominant-track share at boosted band: {share_t0.item():.3f}  (want > 0.95)")
    assert share_t0 > 0.95, "track with 100% contribution should absorb ≥95% of delta"

    # Test 4: LS distribution reaches the target — synth N equal-amplitude
    # noise tracks all contributing equally, ask for +6 dB on the mix.
    # Each track should get ~+6 dB (uniform case → full delta per track).
    torch.manual_seed(7)
    stems_u = torch.randn(1, N, 2, T) * 0.1
    delta_u = torch.zeros(1, 26)
    delta_u[0, band_idx] = 6.0
    _, aux_u = console(stems_u, delta_u)
    per_track_at_band = aux_u["per_track_delta_db"][0, :, band_idx]
    mean_db = per_track_at_band.mean().item()
    print(f"uniform contributions, target +6 dB → mean per-track delta: {mean_db:.2f} dB  (want ~6.0)")
    assert abs(mean_db - 6.0) < 0.5, f"LS distribution should give ~6 dB per track; got {mean_db:.2f}"

    print("\nall ContributionConsole smoke tests passed.")


if __name__ == "__main__":
    _smoke()
