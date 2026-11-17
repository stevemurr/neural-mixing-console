"""26-band mel-spaced gain-invariant EQ — the v13 EQ processor.

Replaces the v12 RBJ-biquad cascade (FFTEQ) with disjoint mel-band gain
control. Design rationale (notes/v13_contribution_mixer_design.md):

  - Each FFT bin belongs to exactly one mel band → no cascade gauge.
    Every band's gain is independently identifiable from the output
    magnitude response, removing the dominant residual ambiguity in
    the v12 EQ.

  - Mel spacing matches our loss space (MR-STFT log-mel, AF log-mel).
    Direct linear mapping between EQ params and loss-space residual:
    the encoder's predicted EQ vector lives in the same coordinates as
    the loss's mel error.

  - Gain-invariant via post-EQ RMS normalization → decouples EQ shape
    from level. After this transform, "+6 dB at 1 kHz" doesn't make the
    track louder; it makes it brighter at the cost of darkness elsewhere.
    EQ and gain become orthogonal axes for the model to optimize.

Forward pass (FFT-based, no biquad coefficients):

    1. Project per-band gain (dB) onto every rfft bin via a precomputed
       bin→band assignment (one band per bin, disjoint).
    2. Multiply rfft of input by the frequency response.
    3. irfft back. If gain_invariant, rescale output to match input RMS.

Calling convention matches the v12 processor: forward(x, gain_norm) where
gain_norm ∈ [0,1] is the sigmoid-domain encoder output; the module maps
it to dB internally.
"""

from __future__ import annotations

import math
from typing import Final

import torch
import torch.nn as nn


def _hz_to_mel(f: float | torch.Tensor) -> float | torch.Tensor:
    """HTK mel scale: m = 2595 log10(1 + f/700)."""
    if isinstance(f, (int, float)):
        return 2595.0 * math.log10(1.0 + f / 700.0)
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def _mel_to_hz(m: float | torch.Tensor) -> float | torch.Tensor:
    if isinstance(m, (int, float)):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 1 else 1


class MelBinEQ(nn.Module):
    """N-band mel-spaced gain-invariant EQ.

    Args:
        sample_rate: audio sample rate (Hz)
        n_bins: number of mel bands (26 matches the v13 auditory filterbank
            spec; matches the per-track contribution vector dimensionality)
        gain_range_db: per-band gain range in dB; gain_norm ∈ [0,1] maps
            to [-gain_range_db, +gain_range_db]
        gain_invariant: if True, post-EQ output is rescaled to match input
            RMS — pure spectral shape, no level effect
        f_min, f_max: mel-band edge bounds (Hz)
    """

    def __init__(
        self,
        sample_rate: int = 48_000,
        n_bins: int = 26,
        gain_range_db: float = 18.0,
        gain_invariant: bool = True,
        f_min: float = 20.0,
        f_max: float = 16_000.0,
    ):
        super().__init__()
        self.sample_rate: Final = sample_rate
        self.n_bins: Final = n_bins
        self.gain_range_db: Final = gain_range_db
        self.gain_invariant: Final = gain_invariant

        # n_bins + 1 mel-spaced Hz edges. Bin i covers [hz_edges[i], hz_edges[i+1]).
        # Note: bands cover [f_min, f_max] exactly. FFT bins outside this range
        # get unity gain (no EQ effect), NOT the edge band's gain — this is
        # what lets us bound the EQ to a meaningful frequency range (e.g.,
        # 20–16 kHz) without applying the top band's gain to all the way to
        # Nyquist.
        mel_min = _hz_to_mel(f_min)
        mel_max = _hz_to_mel(f_max)
        mel_edges = torch.linspace(mel_min, mel_max, n_bins + 1)
        hz_edges = _mel_to_hz(mel_edges)
        self.register_buffer("hz_edges", hz_edges, persistent=False)

    def _build_bin_to_band(
        self, n_fft: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map each rfft bin to (band index, in_range mask).

        Returns:
          band_idx: (n_freq,) integer band index, clamped to [0, n_bins-1]
          in_range: (n_freq,) bool — True iff bin's center frequency falls
                    inside [f_min, f_max). Bins where in_range is False
                    should get unity gain (no EQ effect).
        """
        n_freq = n_fft // 2 + 1
        bin_freqs = torch.arange(n_freq, device=device, dtype=torch.float32) \
            * (self.sample_rate / n_fft)
        edges = self.hz_edges.to(device)
        in_range = (bin_freqs >= edges[0]) & (bin_freqs < edges[-1])
        idx = (torch.searchsorted(edges, bin_freqs, right=False) - 1).clamp(0, self.n_bins - 1)
        return idx, in_range

    def forward(self, x: torch.Tensor, gain_norm: torch.Tensor) -> torch.Tensor:
        """Apply mel-bin EQ to x.

        Args:
            x:          (B, C, T) audio
            gain_norm:  (B, n_bins) in [0,1]; mapped to dB in [-grange, +grange]
        Returns:
            (B, C, T) EQ'd (and optionally gain-compensated) audio.
        """
        if x.dim() != 3:
            raise ValueError(f"expected (B, C, T); got {tuple(x.shape)}")
        if gain_norm.shape[-1] != self.n_bins:
            raise ValueError(
                f"gain_norm last dim must be {self.n_bins}; "
                f"got {tuple(gain_norm.shape)}"
            )
        B, C, T = x.shape

        gain_db = (gain_norm * 2 - 1) * self.gain_range_db        # (B, n_bins)
        gain_lin = (10.0 ** (gain_db / 20.0)).to(x.dtype)          # (B, n_bins)

        n_fft = _next_pow2(T)
        band_idx, in_range = self._build_bin_to_band(n_fft, x.device)
        # H[b, k] = gain_lin[b, band_idx[k]] inside the band range, 1.0 outside.
        H = gain_lin[:, band_idx]                                  # (B, n_freq)
        H = torch.where(in_range.unsqueeze(0), H, torch.ones_like(H))
        H = H.unsqueeze(1)                                          # (B, 1, n_freq)

        X = torch.fft.rfft(x, n=n_fft)                             # (B, C, n_freq)
        Y = X * H
        y = torch.fft.irfft(Y, n=n_fft)[..., :T]                   # (B, C, T)

        if self.gain_invariant:
            in_rms = torch.sqrt((x.float() ** 2).mean(dim=(-2, -1), keepdim=True).clamp(min=1e-12))
            out_rms = torch.sqrt((y.float() ** 2).mean(dim=(-2, -1), keepdim=True).clamp(min=1e-12))
            y = y * (in_rms / out_rms).to(y.dtype)

        return y


__all__ = ["MelBinEQ"]


# ---------- Smoke test (run as: uv run python -m v13.models.mel_bin_eq) ----------

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
