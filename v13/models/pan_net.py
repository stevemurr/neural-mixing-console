"""PanNet — Stage 3 of the v13 contribution-mixer.

Predicts per-track pan as a 3-way classification: {Left, Center, Right}.

Why classification: the engineering distribution is strongly trimodal at
{-1, 0, +1} (17% L, 65% C, 18% R for mono tracks in our corpus, with 28%
of those being fully hard-panned at |pan| ≥ 0.95). MSE on this distribution
collapses to "predict center" because wrong-direction predictions cost
4× squared-error vs centered predictions. Cross-entropy treats wrong-
direction as a class error regardless of distance — the model can commit
when confident without exponential penalty.

At inference we still emit a continuous pan_dir via softmax-weighted mean
of {-1, 0, +1}. When the model is sharp on a class, output approaches the
hard mode; when uncertain, output stays near center. This gives the model
smooth commitment under uncertainty.

Body is identical to TargetCurveNet (small transformer over track tokens),
but instead of pooling into a [MIX] token we read off the per-track outputs
— pan is a per-track decision. The `is_stereo` flag is an extra input
feature so the model knows which tracks should not be panned.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .group_classification import n_canonical_groups


class PanNet(nn.Module):
    def __init__(
        self,
        n_bins: int = 26,
        d_model: int = 192,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        n_groups: int | None = None,
        group_embed_dim: int = 16,
        use_group_embed: bool = True,
        continuous: bool = False,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.d_model = d_model
        self.use_group_embed = use_group_embed
        self.continuous = continuous
        # Optional group embedding. The supervised CE-on-LS-targets variant
        # uses this for "drums tend center" priors. The reconstruction-based
        # variant (Stage 3 v2) drops it so the model is fully audio-driven
        # and deployment-ready — no correspondence.yaml dependency at
        # inference.
        if use_group_embed:
            if n_groups is None:
                n_groups = n_canonical_groups()
            self.n_groups = n_groups
            self.group_embed = nn.Embedding(n_groups, group_embed_dim)
            in_dim = n_bins + 1 + group_embed_dim
        else:
            self.n_groups = 0
            self.group_embed = None
            in_dim = n_bins + 1
        self.in_proj = nn.Linear(in_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        # Output head. Continuous mode: a single scalar per track, squashed by
        # tanh to pan_dir ∈ (-1, +1). The reconstruction loss is differentiable
        # in pan_dir, so no classification / straight-through estimator is
        # needed — the gradient is exact end-to-end. Class mode: {L, C, R}
        # logits, for the CE-on-LS-targets trainer.
        out_dim = 1 if continuous else 3
        self.out_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, out_dim),
        )
        # Class-index → pan_dir mapping for soft inference
        self.register_buffer(
            "class_to_pan",
            torch.tensor([-1.0, 0.0, 1.0]),
            persistent=False,
        )

    def forward(
        self,
        C: torch.Tensor,                                # (B, N, n_bins)
        is_stereo: torch.Tensor,                        # (B, N) bool
        group_idx: torch.Tensor | None = None,          # (B, N) int64 (only if use_group_embed)
        track_mask: torch.Tensor | None = None,         # (B, N) bool
    ) -> torch.Tensor:
        """Per-track pan prediction.

        Continuous mode: returns pan_dir ∈ (-1, +1) of shape (B, N).
        Class mode:      returns logits over {L, C, R} of shape (B, N, 3).
        """
        if C.dim() != 3 or C.shape[-1] != self.n_bins:
            raise ValueError(f"expected C (B, N, {self.n_bins}); got {C.shape}")
        B, N, _ = C.shape

        if self.use_group_embed:
            if group_idx is None:
                raise ValueError("group_idx required when use_group_embed=True")
            g_emb = self.group_embed(group_idx)
            feat = torch.cat(
                [C, is_stereo.to(C.dtype).unsqueeze(-1), g_emb], dim=-1,
            )
        else:
            feat = torch.cat([C, is_stereo.to(C.dtype).unsqueeze(-1)], dim=-1)
        tokens = self.in_proj(feat)                     # (B, N, d_model)
        src_key_padding_mask = (~track_mask.to(torch.bool)) if track_mask is not None else None
        out = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)
        out = self.out_norm(out)                        # (B, N, d_model)
        head = self.out_head(out)                       # (B, N, out_dim)
        if self.continuous:
            return torch.tanh(head.squeeze(-1))         # (B, N) pan_dir ∈ (-1,+1)
        return head                                     # (B, N, 3) class logits

    @staticmethod
    def pan_target_to_class(pan_target: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
        """Map continuous pan_target ∈ [-1, +1] to class index {0=L, 1=C, 2=R}."""
        out = torch.full_like(pan_target, 1, dtype=torch.long)
        out[pan_target <= -thresh] = 0
        out[pan_target >= thresh] = 2
        return out

    def predict_pan_dir(
        self,
        C: torch.Tensor,
        is_stereo: torch.Tensor,
        group_idx: torch.Tensor | None = None,
        track_mask: torch.Tensor | None = None,
        hard: bool = False,
    ) -> torch.Tensor:
        """Inference helper. Returns per-track pan_dir in [-1, +1].

        hard=False (default): softmax-weighted mean of {-1, 0, +1}. Smooth
            commitment — confident L/R → near ±1, uncertain → near 0.
        hard=True:  argmax → discrete {-1, 0, +1}.
        """
        if self.continuous:
            # Continuous head already emits pan_dir; `hard` is ignored.
            return self.forward(C, is_stereo, group_idx, track_mask)
        logits = self.forward(C, is_stereo, group_idx, track_mask)
        if hard:
            cls = logits.argmax(dim=-1)
            return self.class_to_pan[cls]
        probs = torch.softmax(logits, dim=-1)
        return (probs * self.class_to_pan).sum(dim=-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["PanNet"]


# ---------- Smoke ----------

def _smoke() -> None:
    """Verify (1) shapes, (2) bounded output, (3) gradient flow,
    (4) permutation invariance, (5) padding mask respected."""
    torch.manual_seed(0)
    B, N, n_bins = 4, 12, 26
    net = PanNet(n_bins=n_bins)
    print(f"PanNet params: {net.n_params():,}")

    C = (torch.randn(B, N, n_bins) * 0.5 + 1.0).requires_grad_(True)
    is_stereo = torch.zeros(B, N, dtype=torch.bool)
    is_stereo[0, :3] = True                             # 3 stereo tracks in batch 0
    track_mask = torch.ones(B, N, dtype=torch.bool)
    track_mask[0, -3:] = False                          # 3 padded tracks in batch 0
    group_idx = torch.randint(0, net.n_groups, (B, N))

    logits = net(C, is_stereo, group_idx, track_mask)
    assert logits.shape == (B, N, 3), f"bad output shape {logits.shape}"
    print(f"shapes ok. logits shape: {logits.shape}")

    logits.sum().backward()
    assert C.grad is not None and torch.isfinite(C.grad).all(), "no/NaN grad into C"
    print("gradients flow.")

    # Soft + hard inference both bounded in [-1, +1]
    net.eval()
    with torch.no_grad():
        pan_soft = net.predict_pan_dir(C.detach(), is_stereo, group_idx, track_mask, hard=False)
        pan_hard = net.predict_pan_dir(C.detach(), is_stereo, group_idx, track_mask, hard=True)
    assert (pan_soft.abs() <= 1.0 + 1e-5).all()
    assert pan_hard.unique().tolist() == sorted(set([-1.0, 0.0, 1.0]).intersection(set(pan_hard.unique().tolist()))) or \
        set(pan_hard.unique().tolist()).issubset({-1.0, 0.0, 1.0})
    print(f"soft range: [{pan_soft.min().item():.3f}, {pan_soft.max().item():.3f}]")
    print(f"hard unique: {sorted(pan_hard.unique().tolist())}")

    # Permutation equivariance over real positions
    C_o = C.detach().clone()
    with torch.no_grad():
        logits_o = net(C_o, is_stereo, group_idx, track_mask)
        perm = torch.randperm(N)
        C_s = C_o.clone()
        is_s_s = is_stereo.clone()
        g_s = group_idx.clone()
        C_s[1] = C_o[1, perm]
        is_s_s[1] = is_stereo[1, perm]
        g_s[1] = group_idx[1, perm]
        logits_s = net(C_s, is_s_s, g_s, track_mask)
        diff = (logits_s[1] - logits_o[1, perm]).abs().max()
        print(f"permutation equivariance max diff: {diff.item():.2e}  (want < 1e-5)")
        assert diff < 1e-5, "permutation equivariance violated"

    # Padding mask: garbage at padded positions shouldn't affect real outputs
    with torch.no_grad():
        logits_clean = net(C_o, is_stereo, group_idx, track_mask)
        C_g = C_o.clone()
        C_g[0, -3:] = 100.0
        logits_g = net(C_g, is_stereo, group_idx, track_mask)
        mask_diff = (logits_g[0, :9] - logits_clean[0, :9]).abs().max()
        print(f"padded-token invariance (real positions): {mask_diff.item():.2e}  (want < 1e-4)")
        assert mask_diff < 1e-4, "padding mask not respected"

    # Class label mapping
    pan_targets = torch.tensor([-1.0, -0.7, -0.4, 0.0, 0.4, 0.7, 1.0])
    cls = PanNet.pan_target_to_class(pan_targets)
    expected = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    assert torch.equal(cls, expected), f"class mapping wrong: {cls.tolist()} vs {expected.tolist()}"
    print(f"class mapping OK: {pan_targets.tolist()} → {cls.tolist()}")

    # Continuous head: bounded scalar pan_dir, exact gradient (no STE)
    net_c = PanNet(n_bins=n_bins, continuous=True)
    C_c = C.detach().clone().requires_grad_(True)
    pan = net_c(C_c, is_stereo, group_idx, track_mask)
    assert pan.shape == (B, N), f"continuous output shape {pan.shape} != {(B, N)}"
    assert (pan.abs() <= 1.0).all(), "tanh head must keep pan_dir in (-1, +1)"
    pan.sum().backward()
    assert C_c.grad is not None and torch.isfinite(C_c.grad).all(), "no/NaN grad (continuous)"
    print(f"continuous head OK: shape={tuple(pan.shape)}  "
          f"range=[{pan.min().item():.3f}, {pan.max().item():.3f}]")

    print("\nall PanNet smoke tests passed.")


if __name__ == "__main__":
    _smoke()
