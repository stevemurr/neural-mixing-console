"""Anti-aliased clipper — Bitwig Over-style hard-soft variable clipper.

Single-band v1. Multiband variant (LR4 crossover into 3 bands, per-band
clip) is planned for v2 once this is shaken out.

Sound: pre-gain → variable-shape clipper (smooth → hard) → output ceiling →
parallel mix with dry. The shape parameter dials between pure tanh-soft
(`shape=0`) and hard-flat clipping (`shape=1`).

Math:

    y_in   = x * 10^(drive_db / 20)
    k      = 1 + 7 * shape                  # k=1: tanh-soft. k=8: ~hard clip.
    y_clip = sign(y_in) * (1 - (1 - tanh(|y_in|))^k)
    y_out  = 10^(ceiling_db / 20) * y_clip
    y      = (1 - mix) * x + mix * y_out

The `(1 - (1-tanh)^k)` shape converges to a flat-top brick-wall as `k→∞` while
preserving smooth gradients at finite k. We clamp k to [1, 8] so gradients
remain well-conditioned through the entire shape range.

Anti-aliasing: 4× polyphase oversampling using the same pinned 64-tap Kaiser
FIR as `reference/saturation.py`. For very hard settings (shape > 0.8) some
aliasing remains in the highest octave; this is the same trade-off Bitwig's
Over makes — perfect anti-aliasing requires PolyBLEP correction at the clip
edges, which we'll add in v2 if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.signal as ss

# Reuse the same anti-aliasing FIR design as saturation for consistency.
from .saturation import OS_FACTOR


_AA_TAPS = ss.firwin(
    numtaps=64,
    cutoff=0.45 / OS_FACTOR,
    window=("kaiser", 8.0),
    pass_zero=True,
).astype(np.float64) * OS_FACTOR


@dataclass
class ClipParams:
    drive_db: float = 0.0          # pre-gain into the clipper, [0, +24]
    shape: float = 0.5             # 0 = pure tanh, 1 = ~hard clip
    ceiling_db: float = 0.0        # output ceiling [-6, 0]
    mix: float = 1.0               # parallel mix with dry [0, 1]


def _clip_curve(y_in: np.ndarray, shape: float) -> np.ndarray:
    """Variable-shape symmetric clipper. shape ∈ [0, 1] -> k ∈ [1, 8]."""
    k = 1.0 + 7.0 * float(np.clip(shape, 0.0, 1.0))
    abs_y = np.abs(y_in)
    return np.sign(y_in) * (1.0 - (1.0 - np.tanh(abs_y)) ** k)


def _process_mono(x: np.ndarray, p: ClipParams, drive: float, ceiling_lin: float) -> np.ndarray:
    """4× oversample → clip → 4× downsample → output."""
    y_pre = (x * drive).astype(np.float64)

    x_up = ss.upfirdn(_AA_TAPS, y_pre, up=OS_FACTOR, down=1)
    y_clip_up = _clip_curve(x_up, p.shape)
    # Downsample taps need sum-to-1 (energy preservation); _AA_TAPS is scaled
    # by OS for upsample, so divide back here.
    y_clip = ss.upfirdn(_AA_TAPS, y_clip_up, up=1, down=OS_FACTOR) / OS_FACTOR

    delay_samples = (_AA_TAPS.size - 1) // OS_FACTOR
    y_clip = y_clip[delay_samples : delay_samples + x.size]
    if y_clip.size < x.size:
        y_clip = np.concatenate([y_clip, np.zeros(x.size - y_clip.size)])

    y_out = ceiling_lin * y_clip
    return ((1.0 - p.mix) * x + p.mix * y_out).astype(x.dtype, copy=False)


def apply_clipper(x: np.ndarray, p: ClipParams) -> np.ndarray:
    """Apply clipper. Operates per-channel for stereo input.

    Bypass: caller sets p.mix = 0 and p.drive_db = 0 (analytic identity).
    """
    if p.mix == 0.0 and p.drive_db == 0.0:
        return x.astype(x.dtype, copy=True)

    drive = 10.0 ** (p.drive_db / 20.0)
    ceiling_lin = 10.0 ** (p.ceiling_db / 20.0)

    if x.ndim == 1:
        return _process_mono(x, p, drive, ceiling_lin)
    if x.ndim == 2:
        return np.stack([_process_mono(c, p, drive, ceiling_lin) for c in x]).astype(x.dtype, copy=False)
    raise ValueError(f"unsupported input shape {x.shape}")


__all__ = ["ClipParams", "apply_clipper"]
