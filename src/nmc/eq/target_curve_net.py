"""TargetCurveNet — Stage 2 of the v13 contribution-mixer.

Predicts the mix-bus delta curve (engineer_mix_log_mel - dry_sum_log_mel)
from the per-track contribution vectors C. Architecture: small transformer
over track tokens, permutation-invariant, with a learned [MIX] token whose
output is the prediction.

Why a transformer (vs MLP / CNN / DeepSets):

  - MLP on concatenated features assumes fixed track order/count; breaks
    permutation invariance.
  - CNN over time is wasteful — Stage 2's input is already time-averaged.
  - DeepSets is permutation-invariant but pools each track independently:
    no cross-track interactions. Misses masking-aware reasoning like
    "bass + kick both at 100 Hz → cut 100 Hz on both."
  - Transformer over track tokens: O(N²) attention (cheap; N ≤ 32),
    permutation-invariant by construction (no positional embed),
    full pairwise interactions.

Output is the dB DELTA from dry-sum to engineer-mix, not the absolute
target spectrum. The delta is a smaller, more invariant function to learn —
"this mix needs less 200 Hz" is the same prediction regardless of how
loud the input is.

Bounded by tanh × max_delta_db to keep gradients well-conditioned and to
prevent the model from blowing past engineering-realistic moves (±12 dB
is already heavy mix-bus EQ).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TargetCurveNet(nn.Module):
    def __init__(
        self,
        n_bins: int = 26,
        d_model: int = 192,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        max_delta_db: float = 12.0,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.d_model = d_model
        self.max_delta_db = max_delta_db

        # Per-track token projection (C[track] → token)
        self.in_proj = nn.Linear(n_bins, d_model)

        # Learnable [MIX] token: prepended to the sequence, attends to all
        # tracks, and its output is the mix-bus prediction.
        self.mix_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_bins),
        )

    def forward(
        self,
        C: torch.Tensor,                        # (B, N, n_bins)
        track_mask: torch.Tensor | None = None, # (B, N) bool, True = real track
    ) -> torch.Tensor:
        """Predict the mix-bus dB delta from per-track contribution vectors.

        Returns (B, n_bins) in dB, bounded by ±max_delta_db.
        """
        if C.dim() != 3 or C.shape[-1] != self.n_bins:
            raise ValueError(f"expected C (B, N, {self.n_bins}); got {C.shape}")
        B, N, _ = C.shape

        track_tokens = self.in_proj(C)                                      # (B, N, d_model)
        mix_tok = self.mix_token.expand(B, -1, -1)                          # (B, 1, d_model)
        tokens = torch.cat([mix_tok, track_tokens], dim=1)                  # (B, 1+N, d_model)

        # src_key_padding_mask: True = ignore in attention. [MIX] always real.
        if track_mask is not None:
            mix_keep = torch.zeros(B, 1, device=tokens.device, dtype=torch.bool)
            pad_mask = ~track_mask.to(torch.bool)
            src_key_padding_mask = torch.cat([mix_keep, pad_mask], dim=1)
        else:
            src_key_padding_mask = None

        out = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)
        mix_out = self.out_norm(out[:, 0])                                  # (B, d_model)
        raw_delta = self.out_head(mix_out)                                  # (B, n_bins)
        delta_db = torch.tanh(raw_delta) * self.max_delta_db
        return delta_db

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["TargetCurveNet"]


def load_predictor(path):
    """Restore a v13 target-curve checkpoint on CPU for inference."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    keys = ("n_bins", "d_model", "n_layers", "n_heads", "ffn_mult", "dropout", "max_delta_db")
    net = TargetCurveNet(**{k: checkpoint["args"][k] for k in keys}).eval()
    net.load_state_dict(checkpoint["model"])
    return net
