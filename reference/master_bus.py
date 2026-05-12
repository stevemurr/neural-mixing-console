"""Master bus: bus EQ -> bus comp -> bus clip.

Stereo-only block. Order is strict. The bus clipper is the optional
final stage — used in modern productions to push perceived loudness
("bus glue + ceiling clip" pattern).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .clipper import ClipParams, apply_clipper
from .compressor import CompParams, apply_compressor
from .rbj_biquads import BusEQParams, BusEQBypass, apply_bus_eq


@dataclass
class BusParams:
    eq: BusEQParams = field(default_factory=BusEQParams)
    comp: CompParams = field(default_factory=lambda: CompParams(
        threshold_db=-6.0, ratio=1.5, attack_ms=30.0, release_ms=200.0,
        knee_db=6.0, makeup_db=0.0,
    ))
    clip: ClipParams = field(default_factory=ClipParams)


@dataclass
class BusBypass:
    eq: BusEQBypass = field(default_factory=BusEQBypass)
    comp: bool = False
    clip: bool = False


def apply_master_bus(
    x: np.ndarray,
    fs: float,
    p: BusParams,
    bypass: Optional[BusBypass] = None,
) -> np.ndarray:
    """Apply master bus: EQ -> comp -> clip on stereo input."""
    if x.ndim != 2 or x.shape[0] != 2:
        raise ValueError(f"master bus expects stereo (2, N); got shape {x.shape}")
    bypass = bypass or BusBypass()

    y = apply_bus_eq(x, fs, p.eq, bypass.eq)
    y = apply_compressor(y, fs, p.comp)
    if not bypass.clip:
        y = apply_clipper(y, p.clip)
    return y


__all__ = ["BusParams", "BusBypass", "apply_master_bus"]
