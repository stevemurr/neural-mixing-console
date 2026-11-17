"""Differentiable master bus (stage 3 v6): bus EQ → bus comp.

Stereo-only. The bus comp's makeup is pinned to 0 dB so the comp is purely
attenuation-only — loudness is handled by the trim head, not by smearing
makeup gain across strip and bus comps. Final peak control is handled
outside the graph by an always-on safety limiter applied at inference time
after the trim.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .diff_comp import DiffComp
from .diff_eq import DiffBusEQ


class DiffMasterBus(nn.Module):
    """bus EQ → bus comp on stereo input.

    Forward args:
        x: (B, 2, T) stereo audio.
        params: dict with bus EQ params + bus comp params (no makeup).
        bypass: optional dict with keys "eq" (band-keyed dict) and "comp".
    """

    def __init__(self, sample_rate: int = 48_000):
        super().__init__()
        self.fs = sample_rate
        self.eq = DiffBusEQ(sample_rate=sample_rate)
        self.comp = DiffComp(sample_rate=sample_rate)

    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        bypass: Optional[dict] = None,
    ) -> torch.Tensor:
        bypass = bypass or {}
        if x.shape[1] != 2:
            raise ValueError(f"DiffMasterBus expects stereo, got channels={x.shape[1]}")

        eq_params = {k: params[k] for k in DiffBusEQ.PARAM_NAMES}
        y = self.eq(x, eq_params, bypass=bypass.get("eq"))

        # Bus comp threshold: same RMS-relative option as DiffStrip. When
        # `threshold_offset_db` is supplied, effective threshold is
        # bus_rms_db + offset. RMS is computed on the *post-EQ* bus signal.
        if "threshold_offset_db" in params:
            from .diff_strip import compute_track_rms_db
            bus_rms_db = params.get("bus_rms_db")
            if bus_rms_db is None:
                bus_rms_db = compute_track_rms_db(y)
            threshold_db = bus_rms_db + params["threshold_offset_db"]
        else:
            threshold_db = params["threshold_db"]

        comp_params = {}
        for k in DiffComp.PARAM_NAMES:
            if k == "threshold_db":
                comp_params[k] = threshold_db
            elif k == "makeup_db":
                comp_params[k] = torch.zeros_like(params["ratio"])
            else:
                comp_params[k] = params[k]
        return self.comp(y, comp_params, bypass=bypass.get("comp"))


__all__ = ["DiffMasterBus"]
