"""Peak-envelope feedforward compressor.

Single-stage, branched attack/release, soft-knee, makeup gain. Stereo linked
detector operating on max(|L|, |R|) — gain reduction applied identically to
both channels to preserve the stereo image.

Per spec §3.4: peak detection (matches plugin defaults across 1176, LA-2A
clones, dbx-160, Distressor, SSL bus comp). attack_ms floor is 0.5 ms to keep
the smoothing coefficient out of the degenerate regime at 48 kHz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):  # type: ignore[no-redef]
        # Decorator no-op fallback. Generation will be ~100x slower without numba.
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def wrap(fn):
            return fn
        return wrap


@dataclass
class CompParams:
    threshold_db: float = -18.0
    ratio: float = 4.0
    attack_ms: float = 10.0
    release_ms: float = 100.0
    knee_db: float = 6.0
    makeup_db: float = 0.0


_EPS = 1e-9  # small floor to avoid log(0) on silent input


def _coef(time_ms: float, fs: float) -> float:
    """Smoothing coefficient alpha for 1-pole IIR with time constant T63 = time_ms.

    state = alpha * state + (1 - alpha) * input. T63 corresponds to alpha = exp(-1 / N).
    """
    n_samples = max(1.0, time_ms * fs / 1000.0)
    return float(np.exp(-1.0 / n_samples))


@njit(cache=True)
def _compress_kernel(
    detector: np.ndarray,
    threshold_db: float,
    ratio: float,
    knee_db: float,
    a_attack: float,
    a_release: float,
) -> np.ndarray:
    """Compute per-sample linear gain (multiplier) given a detector signal.

    detector: magnitude envelope (e.g. abs(x) or max(|L|,|R|)) — already peak-rectified.
    Returns linear gain g in (0, 1] where g = 10^(gain_reduction_db / 20).
    """
    n = detector.size
    g = np.empty(n, dtype=np.float64)
    state_db = 0.0  # smoothed gain reduction in dB (negative or zero)

    inv_ratio_minus_1 = (1.0 / ratio) - 1.0  # negative
    half_knee = knee_db * 0.5

    for i in range(n):
        d = detector[i]
        if d < 1e-12:
            level_db = -240.0
        else:
            level_db = 20.0 * np.log10(d)

        over = level_db - threshold_db

        # Static curve gain reduction (dB, <= 0)
        if knee_db > 0.0 and 2.0 * abs(over) <= knee_db:
            # quadratic soft-knee region
            target_gr = inv_ratio_minus_1 * (over + half_knee) ** 2 / (2.0 * knee_db)
        elif over > half_knee:
            target_gr = inv_ratio_minus_1 * over
        else:
            target_gr = 0.0

        # Branched attack/release on the gain-reduction signal in dB
        # target_gr is more negative when more compression is needed.
        # "Attack" = increasing compression (state moves to more negative target),
        # "Release" = decreasing compression (state moves toward 0).
        if target_gr < state_db:
            # need more compression -> attack
            state_db = a_attack * state_db + (1.0 - a_attack) * target_gr
        else:
            # less compression -> release
            state_db = a_release * state_db + (1.0 - a_release) * target_gr

        g[i] = 10.0 ** (state_db / 20.0)

    return g


def apply_compressor(x: np.ndarray, fs: float, p: CompParams) -> np.ndarray:
    """Apply peak-envelope feedforward compressor.

    x: shape (samples,) for mono or (2, samples) for stereo. Stereo uses
    linked detection: detector = max(|L|, |R|), same gain applied to both.

    Bypass: caller sets ratio = 1.0 (analytic no-op; this function still
    runs the path but produces unity gain on every sample).
    """
    if p.ratio <= 1.0 + 1e-9:
        # No compression: just makeup gain
        return (x * (10.0 ** (p.makeup_db / 20.0))).astype(x.dtype, copy=False)

    a_attack = _coef(p.attack_ms, fs)
    a_release = _coef(p.release_ms, fs)
    makeup = 10.0 ** (p.makeup_db / 20.0)

    if x.ndim == 1:
        detector = np.abs(x.astype(np.float64))
        gain = _compress_kernel(detector, p.threshold_db, p.ratio, p.knee_db, a_attack, a_release)
        return (x * gain * makeup).astype(x.dtype, copy=False)

    if x.ndim == 2:
        # Linked stereo
        detector = np.maximum(np.abs(x[0].astype(np.float64)), np.abs(x[1].astype(np.float64)))
        gain = _compress_kernel(detector, p.threshold_db, p.ratio, p.knee_db, a_attack, a_release)
        out = x.astype(np.float64, copy=False) * gain[np.newaxis, :] * makeup
        return out.astype(x.dtype, copy=False)

    raise ValueError(f"unsupported input shape {x.shape}")


__all__ = ["CompParams", "apply_compressor"]
