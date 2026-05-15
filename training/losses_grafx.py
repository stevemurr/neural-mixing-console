"""Loss functions for grafx-prune-style supervised training.

Two terms, weighted and combined upstream in the trainer:

  L_param  — Huber MSE on (encoder_predicted_params, grafx_prune_labels),
              masked by `label_track_mask` (per-track) and
              `label_example_mask` (per-example) so unlabeled examples
              and unmatched tracks contribute zero.

  L_recon — multi-resolution mid/side STFT loss between the rendered
              mix and the engineer's reference mix, matching the loss
              grafx-prune itself optimizes against per song (so encoder
              + per-song labels share the same audio-domain objective).
              Uses auraloss.freq.SumAndDifferenceSTFTLoss directly to
              match grafx-prune line-for-line.

`L_param` is the dense supervision signal — it directly aligns each
predicted parameter with the value the per-song optimizer converged to.
`L_recon` keeps the encoder calibrated to the engineer's audio aesthetic
on parts of the parameter space the labels don't perfectly cover (and
on un-labeled songs).

`grafx_param_huber_loss` walks the nested dict shape and reduces:

  sum over (proc, param) of  Huber(pred, target).mean_over_param_shape
                            * track_mask_or_group_mask
                            / sum_of_masks_for_normalization

Per-processor / per-param weights can be passed via `param_weights` to,
e.g., upweight EQ's dense 1024-bin output relative to the comp's 4
scalars — by default everything is weighted 1.0 and the loss is
proportional to the per-element MSE summed across the schema.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ---------- Reconstruction loss (matches grafx-prune line-for-line) ----------

def make_grafx_prune_recon_loss(
    sample_rate: int = 30_000,
    fft_sizes: tuple[int, ...] = (512, 1024, 4096),
    hop_sizes: tuple[int, ...] = (128, 256, 1024),
    win_lengths: tuple[int, ...] = (512, 1024, 4096),
    n_bins: int = 96,
    omit_sec: float = 1.0,
) -> torch.nn.Module:
    """Construct the multi-resolution mid/side STFT loss used by grafx-prune.

    Pulled directly from `code/loss.py:MRSTFTLoss` in the grafx-prune repo
    (DAFx 2024). Mel-scaled magnitude with A-weighting-style perceptual
    weighting; sum + diff (mid/side) decomposition; omits the first
    `omit_sec` of audio (filter warmup transients).

    Returns a `nn.Module` whose `forward(pred, target) → dict` produces:
        { "match/full": scalar, "match/lr": ..., "match/sum": ..., "match/diff": ... }
    """
    # Local import — auraloss is the only dep added for this loss family.
    from auraloss.freq import SumAndDifferenceSTFTLoss

    class GrafxPruneMRSTFTLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loss = SumAndDifferenceSTFTLoss(
                fft_sizes=list(fft_sizes),
                hop_sizes=list(hop_sizes),
                win_lengths=list(win_lengths),
                perceptual_weighting=True,
                sample_rate=sample_rate,
                scale="mel",
                n_bins=n_bins,
                eps=1e-4,
                output="full",  # returns (full, lr, sum, diff)
            )
            self.omit = int(sample_rate * omit_sec)

        def forward(self, pred: torch.Tensor, true: torch.Tensor) -> dict:
            T = pred.shape[-1]
            if pred.ndim > 3:
                pred = pred.view(-1, 2, T)
            if true.ndim > 3:
                true = true.view(-1, 2, T)
            if self.omit > 0 and T > self.omit:
                pred = pred[..., self.omit:]
                true = true[..., self.omit:]
            # auraloss 0.4 returns (full, sum, diff) for output="full".
            # (Older auraloss had per-channel L+R too; dropped in 0.4.)
            full, sum_loss, diff_loss = self.loss(pred, true)
            return {
                "match/full": full,
                "match/sum":  sum_loss,
                "match/diff": diff_loss,
            }

    return GrafxPruneMRSTFTLoss()


# ---------- Per-track parameter Huber loss ----------

def grafx_param_huber_loss(
    pred_strip: dict[str, dict[str, torch.Tensor]],   # encoder output
    pred_group: dict[str, dict[str, torch.Tensor]],
    label_strip: dict[str, dict[str, torch.Tensor]],  # collate output
    label_group: dict[str, dict[str, torch.Tensor]],
    label_track_mask: torch.Tensor,    # (B, N_max) bool — True where supervision exists
    n_groups_per_example: torch.Tensor, # (B,) long — # of valid groups per example
    label_example_mask: torch.Tensor,  # (B,) bool — True for labeled examples
    *,
    delta: float = 0.1,
    param_weights: Optional[dict[str, float]] = None,
) -> dict[str, torch.Tensor]:
    """Huber MSE between predicted and label params for strip + group.

    Masks:
      - Per-track loss: weighted by `label_track_mask` (filename matched)
        AND `label_example_mask` (example has labels at all).
      - Per-group loss: weighted by a `(B, max_groups)` mask derived from
        `n_groups_per_example` (only groups < n_groups[b] count) AND
        `label_example_mask`.

    `delta` is Huber's quadratic→linear threshold. 0.1 in normalized
    log-magnitude units corresponds roughly to ~0.4 dB / ±10 % gain;
    errors larger than that get linear gradient (robust to outliers,
    avoids the saturation collapse we hit with raw MSE in round 14).

    Returns dict for logging:
      { "L_param/strip": ..., "L_param/group": ..., "L_param/total": ... }
    """
    B, N_max = label_track_mask.shape
    device = label_track_mask.device

    weights = param_weights or {}

    # --- Per-track strip Huber ---
    # Combined mask: (B, N_max) — True where there's both an example label
    # and a per-track filename match.
    strip_mask = (label_track_mask & label_example_mask.unsqueeze(-1)).to(torch.float32)
    strip_denom = strip_mask.sum().clamp(min=1.0) * N_max     # rough normalization factor
    # ^ multiplying by N_max keeps strip + group losses on comparable scale
    #   regardless of how many tracks are masked in. (See per-param avg below.)

    l_strip_total = torch.zeros((), device=device)
    for proc, pp in pred_strip.items():
        proc_w = weights.get(proc, 1.0)
        for param_name, pred_t in pp.items():
            target_t = label_strip[proc][param_name].to(pred_t.device)
            # Huber per-element, then mean over the param's own shape dims
            # (everything past dim 1). Result has shape (B, N_max).
            per_elem = F.huber_loss(pred_t, target_t, reduction="none", delta=delta)
            # average across the param's shape dims (everything past N)
            per_track = per_elem.flatten(start_dim=2).mean(dim=-1)   # (B, N_max)
            l_strip_total = l_strip_total + proc_w * (per_track * strip_mask).sum() / strip_mask.sum().clamp(min=1.0)

    # --- Per-group bus Huber ---
    # Group mask: 1.0 where the group index < n_groups[b], 0 otherwise.
    max_groups = next(iter(next(iter(label_group.values())).values())).shape[1]
    group_idx_range = torch.arange(max_groups, device=device).unsqueeze(0)     # (1, max_g)
    group_mask = (group_idx_range < n_groups_per_example.unsqueeze(-1)).to(torch.float32)
    group_mask = group_mask * label_example_mask.unsqueeze(-1).to(torch.float32)
    # ^ (B, max_groups)

    l_group_total = torch.zeros((), device=device)
    for proc, pp in pred_group.items():
        proc_w = weights.get(proc, 1.0)
        for param_name, pred_t in pp.items():
            target_t = label_group[proc][param_name].to(pred_t.device)
            per_elem = F.huber_loss(pred_t, target_t, reduction="none", delta=delta)
            per_group = per_elem.flatten(start_dim=2).mean(dim=-1)            # (B, max_g)
            l_group_total = l_group_total + proc_w * (per_group * group_mask).sum() / group_mask.sum().clamp(min=1.0)

    return {
        "L_param/strip": l_strip_total,
        "L_param/group": l_group_total,
        "L_param/total": l_strip_total + l_group_total,
    }


__all__ = ["make_grafx_prune_recon_loss", "grafx_param_huber_loss"]
