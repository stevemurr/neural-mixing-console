"""Loss functions for stage 3 v6.1+ multitrack training.

- `mss_loss`: multi-resolution STFT loss (Yamamoto+18 / Engel+19 style).
- `log_mel_l1`: L1 on log-mel spectrograms (auxiliary spectral loss).
- `stereo_side_mss_loss`: MSS on the (L − R)/2 side channel — *magnitude*,
  sign-invariant. Catches per-band side-energy mismatch but cannot
  distinguish a left-leaning mix from a right-leaning mix (same |STFT|).
- `stereo_imbalance` / `stereo_width`: scalar audio features from the
  Diff-MST (Steinmetz 2024) audio-feature loss. SI is sign-preserving and
  fixes the gap above; SW gives a scalar width target.
- `decoupled_recon_loss`: reconstruction loss that decouples loudness (trim
  head) from timbre (loudness-normalized waveform / spectrum / log-mel +
  optional stereo terms).
- `pan_mean_penalty`: |mean(pan)| penalty in *param space* against
  catastrophic L/R bias. Weak — trivially satisfied by all-tracks-to-center
  and not energy-weighted. Prefer `w_imbalance` (audio-domain SI) for the
  L/R-symmetry job; keep `pan_mean_penalty` only if you specifically want
  a param-space regularizer.

(v6.1 dropped `bypass_consistency_loss`: no bypass heads — "bypassed" lives
in the param space, gently encouraged by the identity-prior L2.)
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

    *Critical caveat:* MSS is `|STFT(·)|`-based — pure magnitude. At a mono
    pred (`L == R`, side == 0) the loss has nonzero VALUE (`|STFT(ref_side)|`)
    but **zero GRADIENT** w.r.t. pred (the derivative of `|0|` is zero), so
    the center is a stable fixed point the optimizer cannot escape through
    this loss alone. Pair with `stereo_side_time_l1` for a sign-preserving
    time-domain term that *does* have non-zero gradient at center.

    pred, target: (B, 2, T) stereo tensors. Returns scalar MSS on side.
    """
    if pred.shape != target.shape or pred.dim() != 3 or pred.shape[1] != 2:
        raise ValueError(f"expected (B, 2, T) stereo tensors; got pred {pred.shape}")
    side_p = 0.5 * (pred[:, 0:1, :] - pred[:, 1:2, :])
    side_t = 0.5 * (target[:, 0:1, :] - target[:, 1:2, :])
    return mss_loss(side_p, side_t)


def stereo_side_time_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 on the time-domain side signal `(L − R) / 2`.

    The companion to `stereo_side_mss_loss` that fixes its singular-point
    issue at the mono fixed point. Where the magnitude-MSS loss has zero
    gradient at pred-side == 0 (so a center-collapsed model cannot
    differentially learn to pan), this time-domain L1 has gradient
    `sign(side_pred − side_target)` per sample — sign-preserving, non-zero
    at center (gradient becomes `-sign(side_target)`, pointing toward the
    correct sign of the reference side signal). So the *sign* of the
    panning is supervised by this term, while the *magnitude / spectrum*
    of the panning is supervised by the MSS magnitude term — they're
    complementary and should be used together (round-12 onward).

    Phase-sensitive (unlike MSS magnitude). Fine for our setup because
    pred and ref are time-aligned (same source tracks, same start, trim
    head adjusts level not phase).

    pred, target: (B, 2, T) stereo. Returns scalar.
    """
    if pred.shape != target.shape or pred.dim() != 3 or pred.shape[1] != 2:
        raise ValueError(f"expected (B, 2, T) stereo tensors; got pred {pred.shape}")
    side_p = 0.5 * (pred[:, 0, :] - pred[:, 1, :])
    side_t = 0.5 * (target[:, 0, :] - target[:, 1, :])
    return (side_p - side_t).abs().mean()


def stereo_imbalance(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Sign-preserving stereo imbalance ∈ [-1, +1] per batch element.

    `SI = (P_R − P_L) / (P_R + P_L + eps)` where P_C is the per-channel mean
    power. −1 = full left, +1 = full right, 0 = balanced (audio-engineering
    "balance" convention). Diff-MST (Steinmetz 2024) "stereo imbalance"
    feature — the canonical sign-preserving complement to magnitude-MSS
    side losses, which alone cannot distinguish left- from right-leaning.

    Scale-invariant: SI(α·x) = SI(x) for any α > 0 — so computing it on a
    loudness-normalized prediction gives the same value as on the raw pred.

    x: (B, 2, T). Returns (B,) ∈ [-1, +1].
    """
    if x.dim() != 3 or x.shape[1] != 2:
        raise ValueError(f"expected (B, 2, T) stereo; got {x.shape}")
    p_l = (x[:, 0] ** 2).mean(dim=-1)
    p_r = (x[:, 1] ** 2).mean(dim=-1)
    return (p_r - p_l) / (p_r + p_l + eps)


def stereo_width(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Stereo width = side-to-mid power ratio per batch element.

    `SW = P_side / (P_mid + eps)` where mid = (L+R)/2, side = (L−R)/2. Low →
    narrow / mono-leaning; higher → wider. Diff-MST (Steinmetz 2024)
    "stereo width" feature; complements per-band side-channel MSS by
    targeting overall width *magnitude* independently of spectrum shape.

    Scale-invariant: SW(α·x) = SW(x) for any α > 0.

    x: (B, 2, T). Returns (B,) ≥ 0.
    """
    if x.dim() != 3 or x.shape[1] != 2:
        raise ValueError(f"expected (B, 2, T) stereo; got {x.shape}")
    mid = 0.5 * (x[:, 0] + x[:, 1])
    side = 0.5 * (x[:, 0] - x[:, 1])
    p_mid = (mid ** 2).mean(dim=-1)
    p_side = (side ** 2).mean(dim=-1)
    return p_side / (p_mid + eps)


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
    w_side_time: float = 0.0,
    w_imbalance: float = 0.0,
    w_width: float = 0.0,
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
    if w_side_time > 0:
        l_side_time = stereo_side_time_l1(pred_norm, target)
        out["L_side_time"] = l_side_time
        l_timbre = l_timbre + w_side_time * l_side_time
    if w_imbalance > 0:
        si_pred = stereo_imbalance(pred_norm)
        si_targ = stereo_imbalance(target)
        l_imbalance = (si_pred - si_targ).pow(2).mean()
        out["L_imbalance"] = l_imbalance
        l_timbre = l_timbre + w_imbalance * l_imbalance
    if w_width > 0:
        sw_pred = stereo_width(pred_norm)
        sw_targ = stereo_width(target)
        # log-ratio: scale-invariant on the ratio, and symmetric for SW < 1
        # vs SW > 1 deviations (a 2× too-narrow miss costs the same as a 2×
        # too-wide miss).
        l_width = (torch.log(sw_pred + 1e-8) - torch.log(sw_targ + 1e-8)).pow(2).mean()
        out["L_width"] = l_width
        l_timbre = l_timbre + w_width * l_width
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
    "mss_loss", "log_mel_l1", "stereo_side_mss_loss", "stereo_side_time_l1",
    "stereo_imbalance", "stereo_width",
    "decoupled_recon_loss", "pan_mean_penalty",
]
