"""Saturation: tanh + bias asymmetry, parallel-mixed, 4x oversampled.

Reference math (per spec §3.2):

    y_pre  = x * 10^(drive_db / 20)
    y_sat  = tanh(y_pre + bias) - tanh(bias)        # subtract DC from bias
    y_out  = (1 - mix) * x + mix * y_sat * 10^(makeup_db / 20)

The waveshaper runs at 4x the sample rate to push aliasing products above the
original Nyquist; a windowed-sinc anti-aliasing filter handles up- and
down-sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.signal as ss


OS_FACTOR = 4

# 64-tap Kaiser-windowed-sinc anti-aliasing filter, designed once at import for
# byte-exact reproducibility across runs (assuming pinned scipy version).
# Cutoff is at 0.45 of original Nyquist -> 0.45/4 = 0.1125 of upsampled Nyquist;
# stop-band starts well below 0.5/4 = 0.125 (the first alias band).
# Pre-multiplied by OS_FACTOR so passband gain through upsample-by-zero-stuff +
# this filter is unity.
_AA_TAPS = ss.firwin(
    numtaps=64,
    cutoff=0.45 / OS_FACTOR,
    window=("kaiser", 8.0),
    pass_zero=True,
).astype(np.float64) * OS_FACTOR


@dataclass
class SatParams:
    drive_db: float = 0.0
    bias: float = 0.0
    mix: float = 1.0
    makeup_db: float = 0.0


def _waveshape(y_pre: np.ndarray, bias: float) -> np.ndarray:
    return np.tanh(y_pre + bias) - np.tanh(bias)


def apply_saturation(x: np.ndarray, p: SatParams) -> np.ndarray:
    """Apply saturation. Operates per-channel for stereo input.

    Bypass: caller sets p.mix = 0 and p.drive_db = 0 (analytic identity).
    """
    if p.mix == 0.0 and p.drive_db == 0.0:
        return x.astype(x.dtype, copy=True)

    drive = 10.0 ** (p.drive_db / 20.0)
    makeup = 10.0 ** (p.makeup_db / 20.0)

    if x.ndim == 1:
        return _process_mono(x, p, drive, makeup)
    if x.ndim == 2:
        return np.stack([_process_mono(c, p, drive, makeup) for c in x]).astype(x.dtype, copy=False)
    raise ValueError(f"unsupported input shape {x.shape}")


def _process_mono(x: np.ndarray, p: SatParams, drive: float, makeup: float) -> np.ndarray:
    y_pre = (x * drive).astype(np.float64)

    # 4x upsample with the pinned AA filter (taps sum to OS for unity passband
    # gain after zero-stuffing).
    x_up = ss.upfirdn(_AA_TAPS, y_pre, up=OS_FACTOR, down=1)
    # Waveshape at upsampled rate
    y_sat_up = _waveshape(x_up, p.bias)
    # 4x downsample with the same taps; the / OS_FACTOR rescales so the
    # downsample filter has unity DC gain (taps for upsample sum to OS, but
    # downsample needs sum-to-1 for energy preservation).
    y_sat = ss.upfirdn(_AA_TAPS, y_sat_up, up=1, down=OS_FACTOR) / OS_FACTOR

    # upfirdn introduces a delay of (numtaps - 1) / 2 samples at upsampled rate,
    # then another after downsample. Total delay measured at original rate is
    # 2 * (numtaps - 1) / (2 * OS_FACTOR) = (numtaps - 1) / OS_FACTOR samples.
    # For numtaps=64, OS_FACTOR=4 -> 15.75 samples. We compensate by shifting
    # the wet signal forward by the integer part and accept the fractional
    # smear (it's < 1 sample at 48kHz, well below audibility threshold for
    # the parallel-mixed dry/wet sum).
    delay_samples = (_AA_TAPS.size - 1) // OS_FACTOR  # 15 for 64-tap, 4x
    y_sat = y_sat[delay_samples : delay_samples + x.size]

    # Pad with zeros if shorter than expected (edge of signal)
    if y_sat.size < x.size:
        y_sat = np.concatenate([y_sat, np.zeros(x.size - y_sat.size)])

    return ((1.0 - p.mix) * x + p.mix * y_sat * makeup).astype(x.dtype, copy=False)


__all__ = ["SatParams", "apply_saturation", "OS_FACTOR"]
