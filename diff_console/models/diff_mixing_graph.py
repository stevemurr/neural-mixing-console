"""Differentiable full multi-track mixing graph (stage 3 v6).

Topology:

    For each track k:
        post_strip_k = DiffStrip(track_k, strip_params_k)
    bus_input = sum_k post_strip_k
    mix       = DiffMasterBus(bus_input, bus_params)

Variable track count handled via mask. Tracks with mask=False contribute zero
to the sum and don't get their strip computed.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .diff_master_bus import DiffMasterBus
from .diff_strip import DiffStrip


class DiffMixingGraph(nn.Module):
    """Full multi-track mix render. All blocks share the same sample_rate.

    Forward args:
        tracks:        (B, N_max, C, T) — padded multi-track tensor.
        track_mask:    (B, N_max) — bool, True for active tracks.
        strip_params:  dict, each value shape (B, N_max) for per-track scalars.
                       Keys cover DiffStrip's full v6 param set.
        strip_bypass:  optional dict of (B, N_max) bool tensors per strip block.
        bus_params:    dict for DiffMasterBus (per-mix).
        global_bypass: optional dict with keys "bus_eq" (band-keyed dict of
                       (B,) bool) and "bus_comp" (B,).

    Returns:
        mix: (B, 2, T) stereo final mix. NB: no safety limiter applied here —
        that lives at inference time, after the trim, so the training loss
        sees the raw rendered signal.
    """

    def __init__(self, sample_rate: int = 48_000):
        super().__init__()
        self.fs = sample_rate
        self.strip = DiffStrip(sample_rate=sample_rate)
        self.bus = DiffMasterBus(sample_rate=sample_rate)

    def forward(
        self,
        tracks: torch.Tensor,
        track_mask: torch.Tensor,
        strip_params: dict,
        bus_params: Optional[dict] = None,
        strip_bypass: Optional[dict] = None,
        global_bypass: Optional[dict] = None,
    ) -> torch.Tensor:
        global_bypass = global_bypass or {}
        B, N_max, C, T = tracks.shape

        # Apply strips per active track. Vectorize by flattening (B, N_max) -> B*N_max.
        # Inactive tracks waste a bit of compute but produce zero-out via mask.
        flat_tracks = tracks.reshape(B * N_max, C, T)
        flat_strip_params = {k: v.reshape(B * N_max) for k, v in strip_params.items()}
        flat_strip_bypass = None
        if strip_bypass is not None:
            flat_strip_bypass = {}
            for k, v in strip_bypass.items():
                if isinstance(v, dict):
                    flat_strip_bypass[k] = {kk: vv.reshape(B * N_max) for kk, vv in v.items()}
                else:
                    flat_strip_bypass[k] = v.reshape(B * N_max)

        post_strip_flat = self.strip(flat_tracks, flat_strip_params, flat_strip_bypass)
        post_strip = post_strip_flat.reshape(B, N_max, 2, T)

        # Mask out padded tracks
        mask = track_mask.view(B, N_max, 1, 1).to(post_strip.dtype)
        post_strip = post_strip * mask

        # Bus input is just the summed dry strips (no sends in v6).
        bus_input = post_strip.sum(dim=1)

        bus_bypass = {
            "eq": global_bypass.get("bus_eq"),
            "comp": global_bypass.get("bus_comp"),
        }
        return self.bus(bus_input, bus_params, bypass=bus_bypass)


__all__ = ["DiffMixingGraph"]
