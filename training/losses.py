"""Loss functions for stage 3 v6.1 multitrack training.

- `mss_loss`: multi-resolution STFT loss (Yamamoto+18 / Engel+19 style).
- `log_mel_l1`: L1 on log-mel spectrograms (auxiliary spectral loss).
- `stereo_side_mss_loss`: MSS on the (L − R)/2 side channel, for stereo imaging.
- `decoupled_recon_loss`: reconstruction loss that decouples loudness (trim head)
  from timbre (loudness-normalized waveform / spectrum / log-mel).
- `pan_mean_penalty`: |mean(pan)| penalty against catastrophic L/R bias. Note:
  trivially satisfied by all-tracks-to-center, so prefer `stereo_side_mss_loss`
  via `decoupled_recon_loss(..., w_stereo_side=...)` for the same job.

(v6.1 dropped `bypass_consistency_loss`: there are no bypass heads anymore —
"bypassed" lives in the param space, gently encouraged by the identity-prior
L2 in train_stage3.py rather than a discrete-target BCE.)
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
import torchaudio


def _stft_mag(x: torch.Tensor, n_fft: int, hop_length: int, win_length: int) -> torch.Tensor:
    """STFT magnitude. x shape (..., T) -> (..., F, frames)."""
    if x.dim() > 2:
        lead = x.shape[:-1]
        T = x.shape[-1]
        flat = x.reshape(-1, T)
    else:
        lead = ()
        flat = x

    window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        flat,
        n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=True, return_complex=True, normalized=False,
    )
    mag = spec.abs()
    if lead:
        mag = mag.reshape(*lead, *mag.shape[-2:])
    return mag


def mss_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    fft_sizes: Sequence[int] = (256, 1024, 4096),
    hop_ratio: float = 0.25,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Multi-resolution STFT loss = mean over scales of (spectral L1 + log-mag L1).

    pred, target: (B, C, T) or (B, T). Reduces to a scalar.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")

    total = 0.0
    n = 0
    for nfft in fft_sizes:
        hop = max(int(nfft * hop_ratio), 1)
        win = nfft
        mp = _stft_mag(pred, nfft, hop, win)
        mt = _stft_mag(target, nfft, hop, win)
        sc = (mp - mt).abs().mean()
        log = (torch.log(mp + eps) - torch.log(mt + eps)).abs().mean()
        total = total + sc + log
        n += 2
    return total / n


def log_mel_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_rate: int = 48_000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> torch.Tensor:
    """L1 on log-mel-spectrogram. pred/target: (B, C, T) or (B, T)."""
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=20.0,
        f_max=22_000.0,
        power=2.0,
    ).to(pred.device)

    if pred.dim() == 3:
        B, C, T = pred.shape
        pred = pred.reshape(B * C, T)
        target = target.reshape(B * C, T)

    log_p = torch.log(transform(pred) + 1e-6)
    log_t = torch.log(transform(target) + 1e-6)
    return (log_p - log_t).abs().mean()


def stereo_side_mss_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSS loss on the side channel only (S = (L − R) / 2).

    Stereo mixes carry most energy in the mid (L + R) channel; pan / width
    information lives in the side channel. Standard MSS on full stereo
    treats L and R independently; if a model predicts a mono-leaning mix
    (L == R) it can still match the per-channel target magnitude reasonably
    well since the L and R targets are also similar. This loss isolates
    the side channel — explicitly penalizing failures to reproduce the
    stereo image — so the model gets a strong gradient through pan / width.

    pred, target: (B, 2, T) stereo tensors. Returns scalar MSS on side.
    """
    if pred.shape != target.shape or pred.dim() != 3 or pred.shape[1] != 2:
        raise ValueError(f"expected (B, 2, T) stereo tensors; got pred {pred.shape}")
    side_p = 0.5 * (pred[:, 0:1, :] - pred[:, 1:2, :])
    side_t = 0.5 * (target[:, 0:1, :] - target[:, 1:2, :])
    return mss_loss(side_p, side_t)


def _combined_rms(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-batch combined-stereo RMS: (B, 2, T) → (B,).

    "Combined" means we treat L+R as one signal — RMS over channels and time
    jointly. This is the right loudness proxy for the trim head: the trim is a
    single global scalar, so the loss should also see a single global level.
    """
    return torch.sqrt((x ** 2).mean(dim=(-2, -1)) + eps)


def decoupled_recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    trim_db: torch.Tensor,
    *,
    w_time: float = 1.0,
    w_mss: float = 1.0,
    w_log_mel: float = 0.5,
    w_stereo_side: float = 0.0,
    w_loud: float = 0.1,
    loudness_target_dbfs: float | None = None,
    sample_rate: int = 48_000,
    rms_eps: float = 1e-4,
    renorm_clip_db: tuple[float, float] = (-12.0, 24.0),
) -> dict:
    """Reconstruction loss that decouples loudness from timbre.

    Two terms:

      L_timbre — MSS + L1-time (+ optional log-mel, stereo-side) computed on
        a *loudness-normalized* pred. Specifically, pred is rescaled so its
        combined-stereo RMS equals target's. The strip / bus params therefore
        only feel gradients from spectrum, transient shape, and stereo
        imaging — never from "be louder / quieter."

      L_loud — L1 between the predicted trim_db (from MixEncoder.head_trim)
        and a *detached* target trim in dB:
          - if `loudness_target_dbfs` is None: target = the dB-RMS difference
            between `target` and the un-trimmed `pred` (i.e. "match the
            reference mix's level"). This makes the trim head chase each
            reference's idiosyncratic mastering level.
          - if `loudness_target_dbfs` is set: target = `loudness_target_dbfs
            - 20·log10(pred_rms)`, i.e. "bring the un-trimmed pred to a fixed
            absolute level." A consistent target across the dataset → the
            trim head converges faster and the rendered output lands at a
            predictable level (≈ -14 LUFS for typical program at ~-15 dBFS
            RMS; combined-stereo RMS is a close proxy for integrated LUFS for
            continuous full-band material — tune the value to taste).
        The .detach() ensures this loss flows only into the trim head, not
        back into the strip / bus params via pred_rms.

    Inputs:
        pred, target: (B, 2, T) stereo, fp32.
        trim_db: (B,) the encoder's predicted trim in dB.
        rms_eps: floor below which pred_rms is treated as silent. Set so the
            renorm gain `target_rms / pred_rms` doesn't explode early in
            training (1e-4 ≈ −80 dBFS).
        renorm_clip_db: hard clamp on the renorm gain in dB. A degenerate
            near-silent pred can otherwise pull the renormalized signal far
            outside any sensible range and dominate the loss.

    Returns a dict for logging:
        L_timbre, L_time, L_mss, L_log_mel?, L_stereo_side?, L_loud, L_recon
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    if pred.dim() != 3 or pred.shape[1] != 2:
        raise ValueError(f"expected (B, 2, T) stereo; got pred {pred.shape}")
    if trim_db.dim() != 1 or trim_db.shape[0] != pred.shape[0]:
        raise ValueError(f"trim_db must be (B,); got {trim_db.shape} for B={pred.shape[0]}")

    pred_rms = _combined_rms(pred)
    target_rms = _combined_rms(target)

    pred_rms_safe = pred_rms.clamp(min=rms_eps)
    renorm_gain = (target_rms / pred_rms_safe).clamp(
        min=10.0 ** (renorm_clip_db[0] / 20.0),
        max=10.0 ** (renorm_clip_db[1] / 20.0),
    )
    pred_norm = pred * renorm_gain.view(-1, 1, 1)

    l_time = (pred_norm - target).abs().mean()
    l_mss = mss_loss(pred_norm, target)
    l_timbre = w_time * l_time + w_mss * l_mss
    out = {"L_time": l_time, "L_mss": l_mss}
    if w_log_mel > 0:
        l_logmel = log_mel_l1(pred_norm, target, sample_rate=sample_rate)
        out["L_log_mel"] = l_logmel
        l_timbre = l_timbre + w_log_mel * l_logmel
    if w_stereo_side > 0:
        l_side = stereo_side_mss_loss(pred_norm, target)
        out["L_stereo_side"] = l_side
        l_timbre = l_timbre + w_stereo_side * l_side
    out["L_timbre"] = l_timbre

    pred_level_db = 20.0 * torch.log10(pred_rms + rms_eps)
    if loudness_target_dbfs is None:
        target_trim_db = (20.0 * torch.log10(target_rms + rms_eps) - pred_level_db).detach()
    else:
        target_trim_db = (loudness_target_dbfs - pred_level_db).detach()
    l_loud = (trim_db - target_trim_db).abs().mean()
    out["L_loud"] = l_loud

    out["L_recon"] = l_timbre + w_loud * l_loud
    return out


def pan_mean_penalty(pred_pan: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Penalize systematic L/R bias across active tracks.

    Recon loss can have a local minimum where the encoder pans many tracks
    one direction, satisfying side-channel energy without breaking L/R
    symmetry. This loss penalizes |E[pan]|, pushing the optimizer back
    toward balanced (L/R-symmetric) per-track decisions.

    Caveat: trivially satisfied by `pan = 0 ∀ tracks`. Use
    `decoupled_recon_loss(..., w_stereo_side=...)` instead when you actually
    want to enforce stereo width — that loss penalizes failures to reproduce
    the side channel, which a center-collapsed prediction cannot do.

    Args:
        pred_pan: (B, N_max) — predicted pan ∈ [-1, 1] (already denormalized).
        mask:     (B, N_max) bool — True for active tracks.

    Returns scalar |E[pan over active tracks]|, averaged over the batch.
    """
    m = mask.to(pred_pan.dtype)
    n = m.sum(dim=1).clamp(min=1.0)
    mean_pan = (pred_pan * m).sum(dim=1) / n
    return mean_pan.abs().mean()


__all__ = [
    "mss_loss", "log_mel_l1", "stereo_side_mss_loss",
    "decoupled_recon_loss", "pan_mean_penalty",
]
