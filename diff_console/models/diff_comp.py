"""Differentiable peak-envelope compressor with parallel attack/release smoothers + min.

Decision per model_spec.md §3.2: option #2 — run two separate IIRs (attack and
release rate) in parallel, take element-wise min over the two states. Avoids
the non-differentiable `if target < state` branch in the reference's branched
formulation; matches a real plugin topology (parallel attack/release detectors
with min selection).

Stereo-linked detection: detector = max(|L|, |R|), gain applied to both
channels equally. Soft-knee quadratic between threshold ± knee/2.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .fsm_filter import fsm_one_pole


_EPS = 1e-12


def _level_db(detector: torch.Tensor) -> torch.Tensor:
    """Convert a magnitude envelope (>= 0) to dB. Floor very-small values."""
    return 20.0 * torch.log10(detector + _EPS)


def _static_curve_db(level_db: torch.Tensor,
                     threshold_db: torch.Tensor,
                     ratio: torch.Tensor,
                     knee_db: torch.Tensor) -> torch.Tensor:
    """Soft-knee static curve, returns target gain reduction in dB (<= 0).

    level_db, threshold_db, ratio, knee_db: shape broadcastable; level_db is
    typically (B, T), the others are (B, 1) or (B,).
    """
    over = level_db - threshold_db                       # how far above threshold
    inv_ratio_minus_1 = (1.0 / ratio) - 1.0              # negative
    half_knee = knee_db * 0.5

    # Three regions:
    # 1) over <= -knee/2:                            gr = 0  (below knee)
    # 2) -knee/2 < over < knee/2 (and knee > 0):     gr = (1/ratio - 1) * (over + knee/2)^2 / (2*knee)
    # 3) over >= knee/2:                             gr = (1/ratio - 1) * over

    # Build the three branches and select smoothly via masks. We can't use
    # `torch.where` with three regions and keep gradients clean, so use
    # arithmetic masks (clamp the linear portion above the knee, the quadratic
    # portion inside the knee, etc.). Math:
    above_knee = inv_ratio_minus_1 * over
    in_knee_safe_knee = torch.clamp(knee_db, min=_EPS)
    in_knee = inv_ratio_minus_1 * (over + half_knee).pow(2) / (2.0 * in_knee_safe_knee)

    in_knee_mask = ((2.0 * over.abs()) <= knee_db) & (knee_db > 0.0)
    above_mask = (over > half_knee)

    target_gr = torch.zeros_like(over)
    target_gr = torch.where(above_mask, above_knee, target_gr)
    target_gr = torch.where(in_knee_mask, in_knee, target_gr)
    return target_gr


def _smoothing_alpha(time_ms: torch.Tensor, fs: float) -> torch.Tensor:
    """alpha = exp(-1 / N) where N = time_ms * fs / 1000 (T63 samples)."""
    n_samples = torch.clamp(time_ms * fs / 1000.0, min=1.0)
    return torch.exp(-1.0 / n_samples)


def _comp_smoothers_fsm(target_db: torch.Tensor,
                        a_attack: torch.Tensor,
                        a_release: torch.Tensor) -> torch.Tensor:
    """Run parallel attack/release IIR smoothers via FSM, return per-sample min.

    Equivalent to a Python loop running two separate 1-pole IIRs and taking
    the per-sample minimum, but vectorized via FFT (~370× faster on GPU at
    6 s @ 48 kHz scale).

    target_db: (B, T) target gain reduction in dB (<= 0).
    a_attack, a_release: (B,) smoothing coefficients in [0, 1).
    """
    g_a = fsm_one_pole(target_db, a_attack)
    g_r = fsm_one_pole(target_db, a_release)
    return torch.minimum(g_a, g_r)


class DiffComp(nn.Module):
    """Peak-envelope feedforward compressor.

    Forward args:
        x: (B, C, T) audio. C = 1 (mono) or 2 (stereo).
        params: dict of (B,) tensors: threshold_db, ratio, attack_ms,
            release_ms, knee_db, makeup_db.
        bypass: optional (B,) bool — when True, return x * makeup only
            (matches reference's ratio==1 short-circuit). Note: encoder is
            free to set ratio=1 directly, in which case the static curve
            naturally produces zero gain reduction.
    """

    # v6.1: knee_db dropped from the learned param set (the reconstruction
    # loss can't supervise it). Hardcoded to a moderate soft knee. Still
    # accepted via `params` if a caller supplies it (e.g. reference parity).
    PARAM_NAMES = ("threshold_db", "ratio", "attack_ms", "release_ms", "makeup_db")
    KNEE_DB_DEFAULT = 6.0

    def __init__(self, sample_rate: int = 48_000):
        super().__init__()
        self.fs = sample_rate

    def forward(self, x: torch.Tensor, params: dict, bypass: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, T = x.shape

        # Detector: max(|L|, |R|) for stereo, |x| for mono.
        if C == 1:
            detector = x.abs().squeeze(1)         # (B, T)
        else:
            detector, _ = x.abs().max(dim=1)      # (B, T) — peak across channels

        level_db = _level_db(detector)

        threshold_db = params["threshold_db"].view(B, 1)
        ratio = params["ratio"].view(B, 1).clamp(min=1.0 + 1e-6)
        knee_param = params.get("knee_db")
        if knee_param is not None:
            knee_db = knee_param.view(B, 1).clamp(min=0.0)
        else:
            knee_db = torch.full((B, 1), self.KNEE_DB_DEFAULT, device=x.device, dtype=x.dtype)
        attack_ms = params["attack_ms"]
        release_ms = params["release_ms"]
        makeup_lin = torch.pow(10.0, params["makeup_db"] / 20.0).view(B, 1, 1)

        target_db = _static_curve_db(level_db, threshold_db, ratio, knee_db)   # (B, T), <= 0

        a_attack = _smoothing_alpha(attack_ms, self.fs)                         # (B,)
        a_release = _smoothing_alpha(release_ms, self.fs)                       # (B,)

        state_db = _comp_smoothers_fsm(target_db, a_attack, a_release)          # (B, T)

        gain_lin = torch.pow(10.0, state_db / 20.0).unsqueeze(1)                # (B, 1, T) -> broadcast to channels
        y = x * gain_lin * makeup_lin

        if bypass is not None:
            m = bypass.view(B, 1, 1).to(dtype=y.dtype)
            y = (1.0 - m) * y + m * (x * makeup_lin)
        return y


__all__ = ["DiffComp"]
