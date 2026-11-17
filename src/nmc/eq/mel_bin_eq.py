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
