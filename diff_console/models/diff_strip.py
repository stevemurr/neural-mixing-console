"""Differentiable per-track channel strip (stage 3 v6).

Chain: gain → EQ → comp → clip → pan. Mono input → stereo output via
constant-power pan; stereo input → stereo output via balance.

The clipper sits *after* the compressor (modern drum-track convention).
The compressor has no makeup gain in v6 — loudness is handled globally by
the trim head outside this module, so the comp becomes attenuation-only.

Comp threshold can be specified two ways via the params dict:
  - `threshold_db`: absolute threshold in dB. Default behavior.
  - `threshold_offset_db` + `track_rms_db`: relative-to-track-RMS form. The
    effective threshold is `track_rms_db + offset`. This is the engineer
    prior described in spec §10 — "thresholds should sit near signal
    level". When both keys are present, offset takes precedence.

`track_rms_db` may be supplied externally (e.g. computed on the dry input
before any processing) or computed internally via `compute_track_rms_db`.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .diff_clipper import DiffClipper
from .diff_comp import DiffComp
from .diff_eq import DiffEQ


def compute_track_rms_db(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Per-batch-element RMS in dB. x: (B, C, T) -> (B,)."""
    rms = torch.sqrt((x ** 2).mean(dim=(-2, -1)) + eps)
    return 20.0 * torch.log10(rms + eps)


def apply_pan(x: torch.Tensor, pan: torch.Tensor) -> torch.Tensor:
    """Apply pan. x: (B, C, T) with C=1 (mono) or C=2 (stereo). pan: (B,) in [-1, 1]."""
    B, C, T = x.shape
    if C == 1:
        angle = (pan + 1.0) * (math.pi / 4.0)
        L = x[:, 0] * torch.cos(angle).unsqueeze(-1)
        R = x[:, 0] * torch.sin(angle).unsqueeze(-1)
        return torch.stack([L, R], dim=1)
    elif C == 2:
        gain_L = (1.0 - torch.clamp(pan, min=0.0)).unsqueeze(-1)
        gain_R = (1.0 + torch.clamp(pan, max=0.0)).unsqueeze(-1)
        L = x[:, 0] * gain_L
        R = x[:, 1] * gain_R
        return torch.stack([L, R], dim=1)
    else:
        raise ValueError(f"unsupported channel count {C}")


class DiffStrip(nn.Module):
    """gain → EQ → comp → clip → pan.

    Forward args:
        x: (B, C, T) — C in {1, 2}.
        params: dict with all per-block keyword args (see PARAM_NAMES tuples
            on each sub-module).
        bypass: optional dict with keys "eq" (band-keyed dict), "comp", "clip".
    """

    def __init__(self, sample_rate: int = 48_000):
        super().__init__()
        self.fs = sample_rate
        self.eq = DiffEQ(sample_rate=sample_rate)
        self.comp = DiffComp(sample_rate=sample_rate)
        self.clip = DiffClipper()

    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        bypass: Optional[dict] = None,
    ) -> torch.Tensor:
        bypass = bypass or {}
        B = x.shape[0]

        # Comp threshold: prefer offset-from-track-RMS if provided. Encoder
        # predicts threshold_offset_db ∈ [-30, +6]; effective threshold is
        # the track's RMS plus the offset (engineer prior — thresholds sit
        # near signal level).
        if "threshold_offset_db" in params:
            track_rms_db = params.get("track_rms_db")
            if track_rms_db is None:
                track_rms_db = compute_track_rms_db(x)
            threshold_db = track_rms_db + params["threshold_offset_db"]
        else:
            threshold_db = params["threshold_db"]

        # 1. Gain
        gain_lin = torch.pow(10.0, params["gain_db"] / 20.0).view(B, 1, 1)
        y = x * gain_lin

        # 2. EQ
        eq_params = {k: params[k] for k in DiffEQ.PARAM_NAMES}
        y = self.eq(y, eq_params, bypass=bypass.get("eq"))

        # 3. Compressor — makeup pinned to 0 dB (loudness handled by trim head).
        comp_params = {}
        for k in DiffComp.PARAM_NAMES:
            if k == "makeup_db":
                comp_params[k] = torch.zeros_like(params["ratio"])
            elif k == "threshold_db":
                comp_params[k] = threshold_db
            else:
                comp_params[k] = params[k]
        y = self.comp(y, comp_params, bypass=bypass.get("comp"))

        # 4. Clipper
        clip_params = {k: params[k] for k in DiffClipper.PARAM_NAMES}
        y = self.clip(y, clip_params, bypass=bypass.get("clip"))

        # 5. Pan
        y = apply_pan(y, params["pan"])
        return y


__all__ = ["DiffStrip", "apply_pan", "compute_track_rms_db"]
