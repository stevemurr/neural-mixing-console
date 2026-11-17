"""Historical v13 entry point."""
import math
from nmc.eq.mel_bin_eq import _next_pow2
import torch
from nmc.eq.mel_bin_eq import MelBinEQ


def _smoke() -> None:
    """Verify (1) neutral EQ is identity, (2) boosting a band raises that band,
    (3) gain invariance preserves RMS, (4) gain non-invariant lets RMS move."""
    sr = 48_000
    T = 4 * sr  # 4 s
    eq = MelBinEQ(sample_rate=sr, n_bins=26, gain_range_db=18.0,
                  gain_invariant=False)

    # Test 1: neutral EQ (gain_norm = 0.5 → 0 dB on every band) ≈ identity
    torch.manual_seed(0)
    x = torch.randn(2, 2, T) * 0.1
    g_neutral = torch.full((2, 26), 0.5)
    y = eq(x, g_neutral)
    err = (y - x).abs().mean() / x.abs().mean()
    print(f"neutral EQ rel-error vs identity:   {err.item():.2e}  (want < 1e-4)")
    assert err < 1e-4, "neutral EQ should be identity"

    # Test 2: boost the band containing 1 kHz, check that bin amplitude grows
    sine_freq = 1000.0
    t = torch.arange(T) / sr
    sine = torch.sin(2 * math.pi * sine_freq * t).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    sine = sine.expand(1, 2, T).contiguous()
    g_boost = torch.full((1, 26), 0.5)
    # Find the band index for 1 kHz
    band = int(torch.searchsorted(eq.hz_edges, torch.tensor(sine_freq))) - 1
    g_boost[0, band] = 1.0       # +18 dB at the 1 kHz band
    y = eq(sine, g_boost)
    # 1 kHz amplitude should increase by ~10^(18/20) ≈ 7.94×
    in_amp = sine.abs().max()
    out_amp = y.abs().max()
    ratio = out_amp.item() / in_amp.item()
    print(f"+18 dB boost at 1 kHz: amplitude ratio = {ratio:.2f}  (want ≈ 7.9)")
    assert 6.5 < ratio < 9.0, f"expected ~7.9× boost; got {ratio:.2f}"

    # Test 2b: bins above f_max get unity gain (in_range mask correctness).
    # Test the H matrix directly to avoid spectral-leakage confounds.
    n_fft = _next_pow2(T)
    band_idx, in_range = eq._build_bin_to_band(n_fft, x.device)
    bin_at_10k = int(10_000 * n_fft / sr)
    bin_at_18k = int(18_000 * n_fft / sr)
    bin_at_22k = int(22_000 * n_fft / sr)
    print(f"in_range at 10 kHz: {in_range[bin_at_10k].item()} (want True)")
    print(f"in_range at 18 kHz: {in_range[bin_at_18k].item()} (want False)")
    print(f"in_range at 22 kHz: {in_range[bin_at_22k].item()} (want False)")
    assert in_range[bin_at_10k].item(), "bin at 10 kHz should be in range"
    assert not in_range[bin_at_18k].item(), "bin at 18 kHz should be out of range"
    assert not in_range[bin_at_22k].item(), "bin at 22 kHz should be out of range"

    # And confirm masked bins get gain 1 in H even when band has +18 dB.
    g_max = torch.full((1, 26), 1.0)
    gain_lin = (10.0 ** ((g_max * 2 - 1) * eq.gain_range_db / 20.0))
    H = gain_lin[:, band_idx]
    H = torch.where(in_range.unsqueeze(0), H, torch.ones_like(H))
    print(f"H[0, bin_at_10k] = {H[0, bin_at_10k].item():.2f}  (want ≈ 7.94)")
    print(f"H[0, bin_at_18k] = {H[0, bin_at_18k].item():.2f}  (want = 1.00)")
    assert abs(H[0, bin_at_10k].item() - 7.94) < 0.1
    assert abs(H[0, bin_at_18k].item() - 1.00) < 1e-5
    assert abs(H[0, bin_at_22k].item() - 1.00) < 1e-5

    # Test 3: gain invariance preserves RMS
    eq_inv = MelBinEQ(sample_rate=sr, n_bins=26, gain_range_db=18.0,
                      gain_invariant=True)
    g_random = torch.rand(2, 26)
    y_inv = eq_inv(x, g_random)
    in_rms = x.pow(2).mean().sqrt()
    out_rms = y_inv.pow(2).mean().sqrt()
    rms_err = (in_rms - out_rms).abs() / in_rms
    print(f"gain-invariant RMS preservation:    {rms_err.item():.2e}  (want < 1e-4)")
    assert rms_err < 1e-4, "gain_invariant=True should preserve RMS"

    # Test 4: same EQ without invariance changes RMS
    y_var = eq(x, g_random)
    out_rms_var = y_var.pow(2).mean().sqrt()
    rms_change = (out_rms_var / in_rms).item()
    print(f"gain-variant RMS ratio (any value): {rms_change:.3f}")
    assert abs(rms_change - 1.0) > 0.01, "non-invariant EQ should change RMS"

    print("\nall MelBinEQ smoke tests passed.")


if __name__ == "__main__":
    _smoke()
