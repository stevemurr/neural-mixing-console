"""Differentiable anti-aliased variable-shape clipper.

Same architecture as `reference/clipper.py`:
  - 4× polyphase oversample (pinned 64-tap Kaiser FIR — shared design w/ DiffSat)
  - Variable-shape curve `(1 - (1 - tanh(|x|))^k)` with k ∈ [1, 8] from `shape ∈ [0, 1]`
  - Output ceiling
  - Parallel dry/wet mix

Smoothness w.r.t. all params (drive, shape, ceiling, mix) by construction:
the curve is C∞ in the input and analytic in `k`, and `k = 1 + 7 * shape` is
linear in `shape`. No discontinuities.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.signal as ss
import torch
import torch.nn as nn
import torch.nn.functional as F


OS_FACTOR = 4
NUMTAPS = 64


def _design_aa_taps(numtaps: int = NUMTAPS, os: int = OS_FACTOR) -> np.ndarray:
    return ss.firwin(
        numtaps=numtaps,
        cutoff=0.45 / os,
        window=("kaiser", 8.0),
        pass_zero=True,
    ).astype(np.float64) * os


_AA_TAPS_NP = _design_aa_taps()


class DiffClipper(nn.Module):
    """Anti-aliased variable-shape clipper.

    Forward args:
        x: (B, C, T) — operates per-channel.
        params: dict of (B,) tensors:
            clip_drive_db:   [0, +24]
            clip_shape:      [0, 1]  (0=tanh-soft, 1=~hard clip)
            clip_ceiling_db: [-6, 0]
            clip_mix:        [0, 1]
        bypass: optional (B,) bool — when True, returns x unchanged.
    """

    # v6.1: clip_shape and clip_ceiling_db dropped from the learned param set
    # (over-parameterized given drive + mix; the recon loss can't separate the
    # curve geometry from spectral content). Hardcoded to a gentle saturation
    # curve at 0 dBFS ceiling. Still accepted via `params` if supplied.
    PARAM_NAMES = ("clip_drive_db", "clip_mix")
    CLIP_SHAPE_DEFAULT = 0.3       # k = 1 + 7*0.3 = 3.1 — soft-ish
    CLIP_CEILING_DB_DEFAULT = 0.0  # ceiling = 1.0 (no extra reduction)

    def __init__(self):
        super().__init__()
        taps = torch.tensor(_AA_TAPS_NP, dtype=torch.float32)
        self.register_buffer("aa_taps", taps)
        self.os = OS_FACTOR

    def _upsample(self, x: torch.Tensor) -> torch.Tensor:
        """4× upsample, full convolution output, NO intermediate trim. Caller
        chains into `_downsample`, which handles end-to-end alignment in a
        single trim (matching scipy.signal.upfirdn's behavior in the reference)."""
        B_C, _, T = x.shape
        zs = torch.zeros((B_C, 1, T * self.os), device=x.device, dtype=x.dtype)
        zs[:, :, ::self.os] = x
        kernel = self.aa_taps.view(1, 1, -1)
        return F.conv1d(zs, kernel, padding=NUMTAPS - 1)

    def _downsample(self, x_up: torch.Tensor, T_target: int) -> torch.Tensor:
        """4× downsample. Upsample taps are scaled by OS for unity passband
        gain after zero-stuffing; downsample requires sum-to-1 taps for energy
        preservation, so divide output by OS. Trim offset = (NUMTAPS-1) // OS
        = 15 for 64-tap at 4×."""
        kernel = self.aa_taps.view(1, 1, -1)   # same OS-scaled taps
        y = F.conv1d(x_up, kernel, padding=NUMTAPS - 1) / self.os
        y_dec = y[..., ::self.os]
        offset = (NUMTAPS - 1) // self.os
        return y_dec[..., offset:offset + T_target]

    @staticmethod
    def _clip_curve(y_in: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        """y = sign(x) * (1 - (1 - tanh(|x|))^k), k = 1 + 7*shape."""
        # shape: (B, 1, 1) broadcasts; clamp to safe range.
        k = 1.0 + 7.0 * shape.clamp(0.0, 1.0)
        abs_y = y_in.abs()
        # Use a numerical floor to keep (1 - tanh) > 0 strictly (avoid exact 0
        # when tanh saturates, so the gradient through the power doesn't
        # explode).
        one_minus_tanh = (1.0 - torch.tanh(abs_y)).clamp(min=1e-7)
        # (1 - x^k) = 1 - exp(k * log(x))
        return torch.sign(y_in) * (1.0 - torch.exp(k * torch.log(one_minus_tanh)))

    def forward(self, x: torch.Tensor, params: dict, bypass: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, T = x.shape
        drive = torch.pow(10.0, params["clip_drive_db"] / 20.0).view(B, 1, 1)
        mix = params["clip_mix"].view(B, 1, 1)
        shape_param = params.get("clip_shape")
        if shape_param is not None:
            shape = shape_param.view(B, 1, 1)
        else:
            shape = torch.full((B, 1, 1), self.CLIP_SHAPE_DEFAULT, device=x.device, dtype=x.dtype)
        ceiling_param = params.get("clip_ceiling_db")
        if ceiling_param is not None:
            ceiling = torch.pow(10.0, ceiling_param / 20.0).view(B, 1, 1)
        else:
            ceiling = torch.pow(torch.tensor(10.0, device=x.device, dtype=x.dtype),
                                self.CLIP_CEILING_DB_DEFAULT / 20.0)  # scalar; 1.0 at 0 dB

        x_flat = (x * drive).reshape(B * C, 1, T)
        x_up = self._upsample(x_flat)   # full-conv, no intermediate trim

        # Broadcast shape across (B, C) and time
        shape_flat = shape.expand(B, C, 1).reshape(B * C, 1, 1)
        y_clip_up = self._clip_curve(x_up, shape_flat)

        y_clip = self._downsample(y_clip_up, T).reshape(B, C, T)

        y_out = ceiling * y_clip
        y = (1.0 - mix) * x + mix * y_out

        if bypass is not None:
            m = bypass.view(B, 1, 1).to(dtype=y.dtype)
            y = (1.0 - m) * y + m * x
        return y


__all__ = ["DiffClipper", "OS_FACTOR"]
