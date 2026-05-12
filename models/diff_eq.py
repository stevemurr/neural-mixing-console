"""Differentiable 6-band EQ — RBJ cookbook biquads in PyTorch.

Mirrors `reference/rbj_biquads.py` but with all coefficient computations as
differentiable functions of the (freq, gain, Q) input tensors.

Filter chain (per-track strip): HPF -> low_shelf -> peak1 -> peak2 -> high_shelf -> LPF.
Bus EQ chain (4 bands): low_shelf_boost -> low_shelf_attn -> peak_mid -> high_shelf_air.

HPF/LPF use 2nd-order Butterworth (Q=1/sqrt(2)), expressed via the same RBJ
high-pass / low-pass cookbook formulas — those reduce exactly to Butterworth
at this Q value.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torchaudio.functional as taF


_SQRT2_INV = 1.0 / math.sqrt(2.0)  # Butterworth Q for HPF/LPF


# ---------- biquad coefficient functions ----------

def _hpf_coeffs(fs: float, freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """RBJ HPF formula at Q = 1/sqrt(2) (== 2nd-order Butterworth).

    freq: shape (B,) in Hz. Returns (b, a) each shape (B, 3) with a0=1.
    """
    w0 = 2.0 * math.pi * freq / fs
    cos_w0 = torch.cos(w0)
    sin_w0 = torch.sin(w0)
    alpha = sin_w0 * _SQRT2_INV  # = sin/(2Q) with Q=1/sqrt(2)
    b0 = (1.0 + cos_w0) * 0.5
    b1 = -(1.0 + cos_w0)
    b2 = (1.0 + cos_w0) * 0.5
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    b = torch.stack([b0, b1, b2], dim=-1) / a0.unsqueeze(-1)
    a = torch.stack([torch.ones_like(a0), a1 / a0, a2 / a0], dim=-1)
    return b, a


def _lpf_coeffs(fs: float, freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w0 = 2.0 * math.pi * freq / fs
    cos_w0 = torch.cos(w0)
    sin_w0 = torch.sin(w0)
    alpha = sin_w0 * _SQRT2_INV
    b0 = (1.0 - cos_w0) * 0.5
    b1 = 1.0 - cos_w0
    b2 = (1.0 - cos_w0) * 0.5
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    b = torch.stack([b0, b1, b2], dim=-1) / a0.unsqueeze(-1)
    a = torch.stack([torch.ones_like(a0), a1 / a0, a2 / a0], dim=-1)
    return b, a


def _low_shelf_coeffs(fs: float, freq: torch.Tensor, gain_db: torch.Tensor, q: torch.Tensor):
    A = torch.pow(10.0, gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / fs
    cos_w0 = torch.cos(w0)
    sin_w0 = torch.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    sqrt_A = torch.sqrt(A)

    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha

    b = torch.stack([b0, b1, b2], dim=-1) / a0.unsqueeze(-1)
    a = torch.stack([torch.ones_like(a0), a1 / a0, a2 / a0], dim=-1)
    return b, a


def _high_shelf_coeffs(fs: float, freq: torch.Tensor, gain_db: torch.Tensor, q: torch.Tensor):
    A = torch.pow(10.0, gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / fs
    cos_w0 = torch.cos(w0)
    sin_w0 = torch.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    sqrt_A = torch.sqrt(A)

    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha

    b = torch.stack([b0, b1, b2], dim=-1) / a0.unsqueeze(-1)
    a = torch.stack([torch.ones_like(a0), a1 / a0, a2 / a0], dim=-1)
    return b, a


def _peak_coeffs(fs: float, freq: torch.Tensor, gain_db: torch.Tensor, q: torch.Tensor):
    A = torch.pow(10.0, gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / fs
    cos_w0 = torch.cos(w0)
    sin_w0 = torch.sin(w0)
    alpha = sin_w0 / (2.0 * q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A

    b = torch.stack([b0, b1, b2], dim=-1) / a0.unsqueeze(-1)
    a = torch.stack([torch.ones_like(a0), a1 / a0, a2 / a0], dim=-1)
    return b, a


# ---------- helpers ----------

def _apply_biquad(x: torch.Tensor, b: torch.Tensor, a: torch.Tensor,
                  bypass: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply a per-batch-item biquad to (B, C, T) audio.

    b, a: shape (B, 3). Different filter for each batch item. torchaudio's
    `lfilter` with `batching=True` and `clamp=False` handles this directly when
    the filter coefficient leading dim matches the batch dim and is broadcast
    over channels.

    bypass: shape (B,) bool — when True, the output equals the input for that
    batch item.
    """
    # torchaudio.functional.lfilter expects (..., T) for waveform and
    # broadcastable (..., 3) for the filter taps. We have (B, C, T) and (B, 3)
    # — broadcasting requires the filter to be expanded across the channel dim.
    B, C, T = x.shape
    b_exp = b.unsqueeze(1).expand(B, C, 3).reshape(B * C, 3)
    a_exp = a.unsqueeze(1).expand(B, C, 3).reshape(B * C, 3)
    x_flat = x.reshape(B * C, T)
    y_flat = taF.lfilter(x_flat, a_exp, b_exp, clamp=False, batching=True)
    y = y_flat.reshape(B, C, T)

    if bypass is not None:
        # bypass: (B,) -> broadcast to (B, 1, 1)
        m = bypass.view(B, 1, 1).to(dtype=y.dtype)
        y = (1.0 - m) * y + m * x
    return y


# ---------- DiffEQ — 6-band per-track ----------

class DiffEQ(nn.Module):
    """6-band cascade: HPF -> low shelf -> peak1 -> peak2 -> high shelf -> LPF.

    Forward args:
        x:    (B, C, T) audio (C = 1 or 2; per-channel filtering, no cross-channel state).
        params: dict of Tensors, each shape (B,), with keys
            hpf_freq, ls_freq, ls_gain, ls_q,
            p1_freq, p1_gain, p1_q, p2_freq, p2_gain, p2_q,
            hs_freq, hs_gain, hs_q, lpf_freq.
        bypass: optional dict of (B,) bool tensors with keys
            hpf, ls, p1, p2, hs, lpf.
    """

    PARAM_NAMES = (
        "hpf_freq",
        "ls_freq", "ls_gain", "ls_q",
        "p1_freq", "p1_gain", "p1_q",
        "p2_freq", "p2_gain", "p2_q",
        "hs_freq", "hs_gain", "hs_q",
        "lpf_freq",
    )
    BYPASS_NAMES = ("hpf", "ls", "p1", "p2", "hs", "lpf")

    def __init__(self, sample_rate: int = 48_000):
        super().__init__()
        self.fs = sample_rate

    def forward(
        self,
        x: torch.Tensor,
        params: dict[str, torch.Tensor],
        bypass: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        bypass = bypass or {}
        y = x
        b, a = _hpf_coeffs(self.fs, params["hpf_freq"])
        y = _apply_biquad(y, b, a, bypass.get("hpf"))
        b, a = _low_shelf_coeffs(self.fs, params["ls_freq"], params["ls_gain"], params["ls_q"])
        y = _apply_biquad(y, b, a, bypass.get("ls"))
        b, a = _peak_coeffs(self.fs, params["p1_freq"], params["p1_gain"], params["p1_q"])
        y = _apply_biquad(y, b, a, bypass.get("p1"))
        b, a = _peak_coeffs(self.fs, params["p2_freq"], params["p2_gain"], params["p2_q"])
        y = _apply_biquad(y, b, a, bypass.get("p2"))
        b, a = _high_shelf_coeffs(self.fs, params["hs_freq"], params["hs_gain"], params["hs_q"])
        y = _apply_biquad(y, b, a, bypass.get("hs"))
        b, a = _lpf_coeffs(self.fs, params["lpf_freq"])
        y = _apply_biquad(y, b, a, bypass.get("lpf"))
        return y


# ---------- DiffBusEQ — 4-band master bus (Pultec-style) ----------

class DiffBusEQ(nn.Module):
    """4-band: low-shelf-boost -> low-shelf-attn -> peak-mid -> high-shelf-air."""

    PARAM_NAMES = (
        "low_boost_freq", "low_boost_gain",
        "low_attn_freq", "low_attn_gain",
        "mid_freq", "mid_gain", "mid_q",
        "air_freq", "air_gain",
    )
    BYPASS_NAMES = ("low_boost", "low_attn", "mid", "air")

    _SHELF_Q = 0.7

    def __init__(self, sample_rate: int = 48_000):
        super().__init__()
        self.fs = sample_rate

    def forward(self, x, params, bypass=None):
        bypass = bypass or {}
        B = x.shape[0]
        device = x.device
        shelf_q = torch.full((B,), self._SHELF_Q, device=device, dtype=x.dtype)
        y = x
        b, a = _low_shelf_coeffs(self.fs, params["low_boost_freq"], params["low_boost_gain"], shelf_q)
        y = _apply_biquad(y, b, a, bypass.get("low_boost"))
        b, a = _low_shelf_coeffs(self.fs, params["low_attn_freq"], params["low_attn_gain"], shelf_q)
        y = _apply_biquad(y, b, a, bypass.get("low_attn"))
        b, a = _peak_coeffs(self.fs, params["mid_freq"], params["mid_gain"], params["mid_q"])
        y = _apply_biquad(y, b, a, bypass.get("mid"))
        b, a = _high_shelf_coeffs(self.fs, params["air_freq"], params["air_gain"], shelf_q)
        y = _apply_biquad(y, b, a, bypass.get("air"))
        return y


__all__ = ["DiffEQ", "DiffBusEQ"]
