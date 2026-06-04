"""Per-track channel strip: gain -> sat -> EQ -> comp -> clip -> pan.

Mono input -> stereo output via constant-power pan.
Stereo input -> stereo output, pan acts as balance.

Order is strict. The clipper sits after the compressor (modern drum-track
convention: comp shapes dynamics, clip shapes peaks for loudness/punch);
on non-drum tracks the clipper is typically bypassed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .clipper import ClipParams, apply_clipper
from .compressor import CompParams, apply_compressor
from .rbj_biquads import EQParams, EQBypass, apply_eq
from .saturation import SatParams, apply_saturation


@dataclass
class StripParams:
    gain_db: float = 0.0
    sat: SatParams = field(default_factory=SatParams)
    eq: EQParams = field(default_factory=EQParams)
    comp: CompParams = field(default_factory=CompParams)
    clip: ClipParams = field(default_factory=ClipParams)
    pan: float = 0.0  # [-1, +1]


@dataclass
class StripBypass:
    sat: bool = False
    eq: EQBypass = field(default_factory=EQBypass)
    comp: bool = False    # caller sets ratio=1 to bypass; this flag is informational
    clip: bool = False    # caller sets clip.mix=0 to bypass; this flag is informational


def apply_pan_constant_power(x: np.ndarray, pan: float) -> np.ndarray:
    """Mono in -> stereo out with constant-power pan law."""
    angle = (pan + 1.0) * np.pi / 4.0
    L = x * np.cos(angle)
    R = x * np.sin(angle)
    return np.stack([L, R])


def apply_pan_balance(x_lr: np.ndarray, pan: float) -> np.ndarray:
    """Stereo in -> stereo out, pan = balance."""
    L = x_lr[0] * (1.0 - max(0.0, pan))
    R = x_lr[1] * (1.0 + min(0.0, pan))
    return np.stack([L, R])


def apply_strip(
    x: np.ndarray,
    fs: float,
    p: StripParams,
    bypass: Optional[StripBypass] = None,
) -> np.ndarray:
    """Apply full per-track channel strip. Returns stereo signal (2, N)."""
    bypass = bypass or StripBypass()

    # 1. Gain
    y = x * (10.0 ** (p.gain_db / 20.0))

    # 2. Saturation
    if not bypass.sat:
        y = apply_saturation(y, p.sat)

    # 3. EQ
    y = apply_eq(y, fs, p.eq, bypass.eq)

    # 4. Compressor
    y = apply_compressor(y, fs, p.comp)

    # 5. Clipper
    if not bypass.clip:
        y = apply_clipper(y, p.clip)

    # 6. Pan
    if y.ndim == 1:
        y = apply_pan_constant_power(y, p.pan)
    elif y.ndim == 2 and y.shape[0] == 2:
        y = apply_pan_balance(y, p.pan)
    else:
        raise ValueError(f"unsupported pre-pan shape {y.shape}")

    return y


__all__ = [
    "StripParams", "StripBypass", "apply_strip",
    "apply_pan_constant_power", "apply_pan_balance",
]
