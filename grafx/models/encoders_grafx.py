"""Grafx-compatible MixEncoder — phase 2 of the grafx-prune refactor.

Replaces the v6.1 22-strip + 13-bus + 1-trim fixed-shape heads with
schema-driven dict outputs:

    strip_params: dict[processor_name -> dict[param_name -> (B, N, *shape)]]
    group_params: dict[processor_name -> dict[param_name -> (B, G, *shape)]]

Output shapes are auto-discovered from a `GrafxMixingConsole` instance,
so they always match what the renderer expects — and what grafx-prune's
per-song optimizer produces as labels.

Architecture (mostly unchanged from old `MixEncoder`):
    1. Per-track HybridBackbone features (mel-tx + waveform-CNN, fused).
    2. Optional MERT semantic embedding added to per-track features.
    3. Permutation-invariant transformer over track tokens.
    4. **NEW** schema-driven strip head (dict output).
    5. **NEW** masked mean-pool per instrument group → schema-driven group head.

No trim head — grafx-prune doesn't have one; gain lives in the per-track
`gain_panning` and `compressor` parameters.

`HybridBackbone` and `TrimHead` etc. are imported unchanged from
`encoders.py`; this module only redefines the heads + outer wrapper.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .encoders import HybridBackbone
from .grafx_console import GrafxMixingConsole


def _shape_numel(shape) -> int:
    """Number of elements in a shape tuple. `(1024,) -> 1024`, `(40, 2) -> 80`."""
    n = 1
    for d in shape:
        n *= int(d)
    return n


class ProcessorParamHead(nn.Module):
    """Predicts the parameter dict for ONE processor instance.

    Each processor (e.g. `Compressor`) has multiple named params with
    different shapes (e.g. `log_threshold: (1,)`, `log_ratio: (1,)`).
    This head shares a trunk MLP, then has one `Linear` per named param
    sized to that param's flattened length. Reshapes back to the named
    shape on output.

    Final-layer init: small-random weights (std=0.01), zero bias. This
    gives predictions ≈ 0 at init — effectively neutral for grafx's
    log-parameterized processors (log_magnitude ~ ±0.01 → EQ gain
    exp(±0.01) ≈ ±1 % from unity) — while keeping gradient alive
    through the head (`d/dW(Wx+b) = x ≠ 0`, unlike zero-weight init
    which kills gradient to all upstream layers).
    """

    HEAD_INIT_STD = 0.01

    def __init__(self, d_model: int, param_shapes: dict[str, tuple]):
        super().__init__()
        self.param_shapes = {k: tuple(v) for k, v in param_shapes.items()}
        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict({
            name: nn.Linear(d_model, _shape_numel(shape))
            for name, shape in self.param_shapes.items()
        })
        for h in self.heads.values():
            nn.init.normal_(h.weight, std=self.HEAD_INIT_STD)
            nn.init.zeros_(h.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """`x: (..., d_model)` → `dict[param_name -> (..., *param_shape)]`."""
        x = self.trunk(x)
        leading = x.shape[:-1]
        out: dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            flat = head(x)                          # (..., numel)
            out[name] = flat.view(*leading, *self.param_shapes[name])
        return out


class ConsoleParamHead(nn.Module):
    """Predicts the FULL param dict for one level of the console (strip OR group).

    `console_shapes` is `{processor_name: {param_name: shape}}`. One
    `ProcessorParamHead` is created per processor.
    """

    def __init__(self, d_model: int, console_shapes: dict[str, dict[str, tuple]]):
        super().__init__()
        self.processor_heads = nn.ModuleDict({
            proc_name: ProcessorParamHead(d_model, shapes)
            for proc_name, shapes in console_shapes.items()
        })

    def forward(self, x: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        return {name: head(x) for name, head in self.processor_heads.items()}


class MixEncoderGrafx(nn.Module):
    """Grafx-compatible mix encoder.

    Forward args:
        tracks:           (B, N_max, 2, T)
        track_mask:       (B, N_max) bool
        group_assignments:(B, N_max) long ∈ [0, n_groups)
        n_groups:         int — number of group buses to predict for this batch
                          (encoder produces params for every group, even
                          unused ones; renderer ignores those)
        ref_mix:          (B, 2, T) — only consumed when `use_ref_mix=True`
        mert_embeddings:  (B, N_max, mert_dim) — only when `mert_dim > 0`

    Returns dict:
        strip_params: dict[processor -> dict[param -> (B, N_max, *shape)]]
        group_params: dict[processor -> dict[param -> (B, n_groups, *shape)]]
        group_assignments: (B, N_max) — echo for the renderer's convenience
    """

    def __init__(
        self,
        console: GrafxMixingConsole,
        *,
        sample_rate: int = 48_000,
        d_model: int = 384,
        n_track_layers: int = 4,
        n_attn_heads: int | None = None,
        use_ref_mix: bool = False,
        mert_dim: int = 0,
    ):
        super().__init__()
        self.fs = sample_rate
        self.d_model = d_model
        self.use_ref_mix = use_ref_mix
        self.mert_dim = mert_dim

        # Resolve `n_attn_heads` first — needs to be valid for d_model and
        # is threaded into both HybridBackbone (which has internal attention)
        # and the per-track transformer below.
        # v6-v9 used hardcoded nhead=6 with d_model=384 (head dim 64). When
        # d_model grows (v10+) we need an nhead that still divides it.
        if n_attn_heads is None:
            # Prefer 64-dim heads when possible (matches v6-v9 head dim),
            # otherwise fall back to the largest divisor ≤ 12.
            for candidate in (d_model // 64, 8, 6, 4):
                if candidate > 0 and d_model % candidate == 0:
                    n_attn_heads = candidate
                    break
            if n_attn_heads is None:
                raise ValueError(
                    f"could not auto-pick n_attn_heads dividing d_model={d_model}; "
                    f"pass n_attn_heads explicitly."
                )
        elif d_model % n_attn_heads != 0:
            raise ValueError(
                f"n_attn_heads={n_attn_heads} must divide d_model={d_model}"
            )
        self.n_attn_heads = n_attn_heads

        # Per-track backbone: stereo dry track, plus stereo ref_mix if enabled.
        # n_heads is threaded so the internal MelTransformer + CrossAttn use
        # the same head count as the outer track transformer.
        in_channels = 4 if use_ref_mix else 2
        self.track_backbone = HybridBackbone(
            in_channels=in_channels, d_model=d_model, sample_rate=sample_rate,
            n_heads=n_attn_heads,
        )

        if mert_dim > 0:
            self.mert_proj = nn.Sequential(
                nn.Linear(mert_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
        else:
            self.mert_proj = None

        # Permutation-invariant transformer over track tokens.
        track_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_attn_heads, dim_feedforward=d_model * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.track_tx = nn.TransformerEncoder(track_layer, num_layers=n_track_layers)

        # Schema-driven heads. Per-track and per-group consume the same
        # processor inventory, so they're structurally symmetric — but they
        # have independent weights since their inputs differ.
        self.head_strip = ConsoleParamHead(d_model, console.strip_param_shapes)
        self.head_group = ConsoleParamHead(d_model, console.group_param_shapes)

    def _pool_per_group(
        self,
        track_feats: torch.Tensor,        # (B, N, d)
        track_mask: torch.Tensor,         # (B, N) bool
        group_assignments: torch.Tensor,  # (B, N) long
        n_groups: int,
    ) -> torch.Tensor:
        """Masked mean of per-track features grouped by `group_assignments`.

        Returns (B, n_groups, d). Groups with no active tracks get a zero
        feature vector (their predicted params are effectively wasted but
        harmless since no audio routes to those buses).
        """
        B, N, d = track_feats.shape
        mask_f = track_mask.to(track_feats.dtype).unsqueeze(-1)        # (B, N, 1)
        masked = track_feats * mask_f                                   # zero-out padded tracks
        idx = group_assignments.long().unsqueeze(-1).expand(B, N, d)    # (B, N, d)

        sums = torch.zeros(B, n_groups, d, dtype=track_feats.dtype, device=track_feats.device)
        sums.scatter_add_(dim=1, index=idx, src=masked)

        # Counts (active tracks per group) for the mean.
        idx_1d = group_assignments.long()
        counts = torch.zeros(B, n_groups, dtype=track_feats.dtype, device=track_feats.device)
        counts.scatter_add_(dim=1, index=idx_1d, src=mask_f.squeeze(-1))
        counts = counts.clamp(min=1.0).unsqueeze(-1)                    # (B, n_groups, 1)
        return sums / counts

    def forward(
        self,
        tracks: torch.Tensor,
        track_mask: torch.Tensor,
        group_assignments: torch.Tensor,
        n_groups: int,
        ref_mix: Optional[torch.Tensor] = None,
        mert_embeddings: Optional[torch.Tensor] = None,
    ) -> dict:
        B, N, C, T = tracks.shape
        device = tracks.device

        # ---- Per-track backbone ----
        if self.use_ref_mix:
            if ref_mix is None:
                ref_mix = torch.zeros(B, 2, T, device=device, dtype=tracks.dtype)
            ref_expanded = ref_mix.unsqueeze(1).expand(B, N, 2, T)
            track_in = torch.cat([tracks, ref_expanded], dim=2)
            flat = track_in.reshape(B * N, 4, T)
        else:
            flat = tracks.reshape(B * N, 2, T)

        pooled, _ = self.track_backbone(flat)                # (B*N, d_model)
        track_feats = pooled.reshape(B, N, self.d_model)

        if self.mert_proj is not None:
            if mert_embeddings is None:
                raise ValueError("mert_dim>0 but mert_embeddings not provided")
            track_feats = track_feats + self.mert_proj(mert_embeddings.to(track_feats.dtype))

        # ---- Permutation-invariant track transformer ----
        pad_mask = ~track_mask
        track_feats = self.track_tx(track_feats, src_key_padding_mask=pad_mask)

        # ---- Per-track strip head ----
        # ConsoleParamHead operates on the last d_model dim and preserves
        # all leading dims, so passing (B, N, d_model) gives outputs of
        # shape (B, N, *param_shape) per param.
        strip_params = self.head_strip(track_feats)

        # ---- Pool track features per instrument group → per-group head ----
        group_feats = self._pool_per_group(
            track_feats, track_mask, group_assignments, n_groups,
        )                                                     # (B, n_groups, d_model)
        group_params = self.head_group(group_feats)           # dict[proc -> dict[p -> (B, G, *shape)]]

        return {
            "strip_params": strip_params,
            "group_params": group_params,
            "group_assignments": group_assignments,
        }


__all__ = [
    "ProcessorParamHead", "ConsoleParamHead", "MixEncoderGrafx",
]
