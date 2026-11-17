"""RBJ Audio EQ Cookbook biquad filter coefficients.

Reference: https://www.w3.org/TR/audio-eq-cookbook/

All filters return coefficients (b, a) in the standard biquad form
where a0 has been normalized to 1:

    y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]

scipy.signal.lfilter / sosfilt expects this convention.

HPF and LPF use scipy's Butterworth implementation directly (the RBJ HPF/LPF
formulas with Q = 1/sqrt(2) are mathematically identical to a 2nd-order
Butterworth response, and scipy's implementation is well-validated).

Shelves and peaks are explicit RBJ cookbook formulas — scipy doesn't ship them.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
import scipy.signal as ss


# ---------- Single-band coefficient functions ----------

def hpf_coeffs(fs: float, freq: float) -> tuple[np.ndarray, np.ndarray]:
    """2nd-order Butterworth high-pass (Q = 1/sqrt(2) = 0.7071)."""
    return ss.butter(2, freq, btype="highpass", fs=fs, output="ba")


def lpf_coeffs(fs: float, freq: float) -> tuple[np.ndarray, np.ndarray]:
    """2nd-order Butterworth low-pass (Q = 1/sqrt(2) = 0.7071)."""
    return ss.butter(2, freq, btype="lowpass", fs=fs, output="ba")


def low_shelf_coeffs(fs: float, freq: float, gain_db: float, q: float):
    """RBJ low-shelf. Q controls the slope steepness (higher Q = steeper)."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    sqrt_A = np.sqrt(A)

    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def high_shelf_coeffs(fs: float, freq: float, gain_db: float, q: float):
    """RBJ high-shelf."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    sqrt_A = np.sqrt(A)

    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def peak_coeffs(fs: float, freq: float, gain_db: float, q: float):
    """RBJ peaking EQ (bell)."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def apply_biquad(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Apply biquad to mono or per-channel-independent stereo signal.

    For 2D inputs of shape (channels, samples), each channel is filtered
    independently (no cross-channel state), as specified in spec §3.3.
    """
    if x.ndim == 1:
        return ss.lfilter(b, a, x).astype(x.dtype, copy=False)
    if x.ndim == 2:
        return np.stack([ss.lfilter(b, a, c) for c in x]).astype(x.dtype, copy=False)
    raise ValueError(f"unsupported input shape {x.shape}")


# ---------- 6-band EQ — used by per-track strip (§3.3) ----------

@dataclass
class EQParams:
    hpf_freq: float = 80.0
    ls_freq: float = 200.0
    ls_gain: float = 0.0
    ls_q: float = 0.7
    p1_freq: float = 800.0
    p1_gain: float = 0.0
    p1_q: float = 1.0
    p2_freq: float = 4000.0
    p2_gain: float = 0.0
    p2_q: float = 1.0
    hs_freq: float = 8000.0
    hs_gain: float = 0.0
    hs_q: float = 0.7
    lpf_freq: float = 18000.0


@dataclass
class EQBypass:
    hpf: bool = False
    ls: bool = False  # set ls_gain=0 for analytic identity
    p1: bool = False
    p2: bool = False
    hs: bool = False
    lpf: bool = False


def apply_eq(x: np.ndarray, fs: float, p: EQParams, bypass: Optional[EQBypass] = None) -> np.ndarray:
    """Apply 6-band EQ cascade: HPF -> LS -> P1 -> P2 -> HS -> LPF.

    Bypass for HPF/LPF short-circuits the biquad in code (true identity).
    Bypass for shelves/peaks is implemented by setting their gain_db to 0 in
    the params; this function honors a bypass flag if passed (for the explicit
    override path used at validation time).
    """
    bypass = bypass or EQBypass()
    y = x

    if not bypass.hpf:
        b, a = hpf_coeffs(fs, p.hpf_freq)
        y = apply_biquad(y, b, a)

    if not bypass.ls:
        b, a = low_shelf_coeffs(fs, p.ls_freq, p.ls_gain, p.ls_q)
        y = apply_biquad(y, b, a)

    if not bypass.p1:
        b, a = peak_coeffs(fs, p.p1_freq, p.p1_gain, p.p1_q)
        y = apply_biquad(y, b, a)

    if not bypass.p2:
        b, a = peak_coeffs(fs, p.p2_freq, p.p2_gain, p.p2_q)
        y = apply_biquad(y, b, a)

    if not bypass.hs:
        b, a = high_shelf_coeffs(fs, p.hs_freq, p.hs_gain, p.hs_q)
        y = apply_biquad(y, b, a)

    if not bypass.lpf:
        b, a = lpf_coeffs(fs, p.lpf_freq)
        y = apply_biquad(y, b, a)

    return y


# ---------- 4-band master-bus EQ (§5.1) ----------

@dataclass
class BusEQParams:
    """Pultec-style 2-low + 1-mid + 1-air."""
    low_boost_freq: float = 60.0
    low_boost_gain: float = 0.0
    low_attn_freq: float = 100.0
    low_attn_gain: float = 0.0
    mid_freq: float = 1000.0
    mid_gain: float = 0.0
    mid_q: float = 1.0
    air_freq: float = 12000.0
    air_gain: float = 0.0


@dataclass
class BusEQBypass:
    low_boost: bool = False
    low_attn: bool = False
    mid: bool = False
    air: bool = False


def apply_bus_eq(x: np.ndarray, fs: float, p: BusEQParams, bypass: Optional[BusEQBypass] = None) -> np.ndarray:
    bypass = bypass or BusEQBypass()
    y = x
    if not bypass.low_boost:
        b, a = low_shelf_coeffs(fs, p.low_boost_freq, p.low_boost_gain, 0.7)
        y = apply_biquad(y, b, a)
    if not bypass.low_attn:
        b, a = low_shelf_coeffs(fs, p.low_attn_freq, p.low_attn_gain, 0.7)
        y = apply_biquad(y, b, a)
    if not bypass.mid:
        b, a = peak_coeffs(fs, p.mid_freq, p.mid_gain, p.mid_q)
        y = apply_biquad(y, b, a)
    if not bypass.air:
        b, a = high_shelf_coeffs(fs, p.air_freq, p.air_gain, 0.7)
        y = apply_biquad(y, b, a)
    return y


# Re-export asdict for convenience
__all__ = [
    "EQParams", "EQBypass", "apply_eq",
    "BusEQParams", "BusEQBypass", "apply_bus_eq",
    "hpf_coeffs", "lpf_coeffs", "low_shelf_coeffs", "high_shelf_coeffs", "peak_coeffs",
    "apply_biquad", "asdict",
]
