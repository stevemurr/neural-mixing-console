"""Mix-level reconstruction loss for Stage 3 pan training.

Two complementary per-band terms, energy-weighted by the engineer's per-band
magnitude and both normalized to O(1):

  (1) DIRECTION — stereo balance (sign-aware):
        balance_b = (mel_L_b − mel_R_b) / (mel_L_b + mel_R_b) ∈ [−1, +1]
      Signed: +1 = hard left, −1 = hard right, 0 = centered. This carries the
      left-vs-right information.
  (2) COMMITMENT — width (sign-blind, and necessarily so):
        width_b = sqrt( mel_pow(L − R)_b / mel_pow(L + R)_b )  (side / mid)
      How much side energy exists at each band, regardless of which side.
      Magnitude-domain (sqrt of the power ratio) so it is linear in |pan| near
      center and keeps a non-vanishing gradient to escape the center plateau.

Both operate on the predicted vs engineer stereo mix; per-track attribution
happens automatically via backprop through the pan-sum-mix operation.

**Why two terms, and why one is sign-blind.** An earlier version paired ILD
with a *magnitude* MR-STFT of the side channel, |STFT(L − R)|, as the
"commit off-center" force. With that term dominant, the model collapsed to
driving *every* track to one side (the opposite channel went silent),
because magnitude side energy is **directionless** — |STFT(L − R)| is
identical for a hard-left and a hard-right pan, so it rewarded *having* side
energy without saying which side, and the weak ILD couldn't redistribute.

The fix is NOT to make the commitment term sign-aware — width genuinely has
no left/right sign (a wide mix split L/R has the same width as its mirror).
The fix is to pair a directionless commitment term with a co-equal,
*sign-aware* direction term (balance), so:

  - all-center          → width ≈ 0 vs engineer width > 0  → COMMITMENT high
  - committed-wrong-side → balance sign flipped             → DIRECTION  high
  - committed-right-side → both match                       → the minimum

A single per-band level ratio (ILD or balance alone) can't do this: when the
engineer mix is L/R-balanced in aggregate (opposite-panned sources of similar
level in a band), its balance reads ≈ 0, so "predict center" matches it — the
width term is what catches that case.

ILD is kept as a logged metric (interpretable dB) but is no longer in the
loss; balance subsumes its direction role without the ±clip saturation.
Per the 2024 ITD-loss paper (arXiv:2408.00344) we skip ITD (non-
differentiable; also moot since constant-power amplitude panning doesn't
manipulate phase or timing).

See `notes/v13_pan_recon_design.md` for the full design.
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn
import torchaudio


class PanReconLoss(nn.Module):
    """Mix-level pan reconstruction loss (sign-aware ILD + stereo balance).

    Args:
        sample_rate: audio sample rate (Hz)
        n_bins: number of mel bands (matches v13 EQ / contribution dictionary)
        n_fft, hop_length: STFT params for the mel spectrogram
        f_min, f_max: mel-band edge bounds (Hz). Matches MelBinEQ defaults.
        eps: numerical floor for the per-band ratios (balance, width, ILD).
        ild_clip_db: ILD clamp (±dB) for the logged ILD metric.
        w_balance: weight on the sign-aware DIRECTION term (which side).
        w_width: weight on the sign-blind COMMITMENT term (how wide). Both
             terms are O(1), so the two weights are directly comparable; a
             single-side collapse is high-loss under DIRECTION and an
             all-center collapse is high-loss under COMMITMENT.
    """

    def __init__(
        self,
        sample_rate: int = 48_000,
        n_bins: int = 26,
        n_fft: int = 8192,
        hop_length: int = 2048,
        f_min: float = 20.0,
        f_max: float = 16_000.0,
        eps: float = 1e-4,
        ild_clip_db: float = 30.0,
        w_balance: float = 1.0,
        w_width: float = 1.0,
        width_floor: float = 1e-4,
    ):
        super().__init__()
        self.eps: Final = eps
        self.n_bins: Final = n_bins
        self.ild_clip_db: Final = ild_clip_db
        self.w_balance: Final = w_balance
        self.w_width: Final = w_width
        self.width_floor: Final = width_floor
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_bins,
            f_min=f_min,
            f_max=f_max,
            power=1.0,                              # magnitude — for channel L/R balance
            norm=None,
            mel_scale="htk",
        )
        # Power-domain mel (energy, power=2.0) for the width term, which operates
        # on L − R. The side signal is *exactly* zero at all-center, and the
        # magnitude transform's sqrt has a NaN gradient at 0 — the power
        # transform (re²+im², no sqrt) has a clean zero gradient there, so the
        # model can pass through center during training without NaNs.
        self.mel_pow = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_bins,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            norm=None,
            mel_scale="htk",
        )

    # ---------- Per-band direction + width features ----------

    def per_band_features(
        self, mix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """mix: (B, 2, T) → (balance, width, ild, energy), each (B, n_bins).

        balance_b = (mel_L_b − mel_R_b) / (mel_L_b + mel_R_b + eps) ∈ [−1, +1]
            Signed per-band pan position (DIRECTION). +1 = hard left,
            −1 = hard right, 0 = centered. The sign is the left-vs-right signal
            the magnitude side spectrum threw away. Bounded and scale-invariant.
        width_b   = sqrt( mel_pow(L − R)_b / (mel_pow(L + R)_b + eps) + floor ) ∈ [0, ~1]
            Per-band side / mid MAGNITUDE ratio (COMMITMENT) — how much side
            energy at band b, regardless of side. ~0 = mono/centered, →1 = fully
            decorrelated/wide. Sign-blind by nature; this is what penalizes the
            all-center collapse that the direction term alone cannot (a band can
            be wide yet aggregate-balanced → balance ≈ 0). Magnitude domain
            (sqrt of the power ratio) so it is linear in |pan| near center — a
            constant escape gradient — instead of the power ratio's vanishing
            quadratic gradient that strands the model at center. The floor on
            the dimensionless ratio keeps the gradient finite at L − R = 0.
        ild_b     = clamp(20·log10(mel_L / mel_R), ±ild_clip_db)
            Logged metric only (interpretable dB), NOT in the loss.
        energy_b  = mel_L_b + mel_R_b
            Channel magnitude — weights the matches toward bands with content
            (silent bands get ≈ 0 weight, so their ill-defined ratios can't
            pull the loss).
        """
        if mix.dim() != 3 or mix.shape[1] != 2:
            raise ValueError(f"expected (B, 2, T); got {mix.shape}")
        L = mix[:, 0]
        R = mix[:, 1]
        mel_L = self.mel(L).mean(dim=-1)                                  # (B, n_bins)
        mel_R = self.mel(R).mean(dim=-1)
        balance = (mel_L - mel_R) / (mel_L + mel_R + self.eps)
        ild = 20.0 * torch.log10((mel_L + self.eps) / (mel_R + self.eps))
        ild = ild.clamp(-self.ild_clip_db, self.ild_clip_db)
        # Width = side/mid as a MAGNITUDE-domain ratio: sqrt(power_ratio +
        # width_floor). This is linear in |pan| near center → a non-vanishing
        # escape gradient. (The bare power ratio is quadratic near center, so
        # its gradient vanishes there and the model gets stuck in the center
        # plateau while the balance term's gradient pulls it toward center.)
        # The floor is on the dimensionless ratio (scale-independent) and keeps
        # the gradient finite at the L − R = 0 singularity → still NaN-safe at
        # exact all-center.
        side_pow = self.mel_pow(L - R).mean(dim=-1)
        mid_pow = self.mel_pow(L + R).mean(dim=-1)
        width = torch.sqrt(side_pow / (mid_pow + self.eps) + self.width_floor)
        energy = mel_L + mel_R
        return balance, width, ild, energy

    # ---------- Auxiliary metrics (logged, NOT in loss) ----------

    @staticmethod
    def stereo_width(mix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        L = mix[:, 0]; R = mix[:, 1]
        side_e = ((L - R) ** 2).sum(dim=-1)
        mid_e = ((L + R) ** 2).sum(dim=-1) + eps
        return side_e / mid_e                                              # (B,)

    @staticmethod
    def stereo_imbalance(mix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        L_e = (mix[:, 0] ** 2).sum(dim=-1)
        R_e = (mix[:, 1] ** 2).sum(dim=-1)
        return (R_e - L_e) / (R_e + L_e + eps)                             # (B,)

    # ---------- Forward ----------

    def forward(
        self,
        predicted_mix: torch.Tensor,                # (B, 2, T)
        target_mix: torch.Tensor,                   # (B, 2, T)
        stems: torch.Tensor | None = None,          # (B, N, 2, T) — for residual mode
        is_stereo: torch.Tensor | None = None,      # (B, N) bool
        track_mask: torch.Tensor | None = None,     # (B, N) bool
    ) -> tuple[torch.Tensor, dict]:
        """Returns (loss, metrics).

        Loss = w_balance · energy_weighted_L1(balance_pred − balance_eng)
             + w_width   · energy_weighted_L1(width_pred   − width_eng)

        all on the residual (stereo-subtracted) signals, weighted per band by
        the engineer's mid magnitude. DIRECTION (balance, signed) penalizes the
        wrong side; COMMITMENT (width, magnitude) penalizes collapsing to
        center. Together the only low-loss state is committed-to-the-right-side.

        When `stems` and `is_stereo` are provided, the loss operates on the
        **residual** signals after subtracting natively-stereo stem
        contributions:

            residual = mix  −  Σ_{t : is_stereo[t]} stems[t]

        This focuses the loss on the pan-controllable component — the
        difference between predicted and engineer mix that's *driven by
        mono-track pan choices*. Without the subtraction, stereo-source
        passthrough content dominates the side spectrum and makes
        "predict all-center" a near-optimal loss collapse, since stereo
        passthrough already accounts for most engineer side energy.

        The same `stems × is_stereo` mask is subtracted from BOTH predicted
        and target, so the subtraction is identity-preserving for the
        gradient (it doesn't change the gradient w.r.t. pan choices on mono
        tracks — only removes the loss component that pan can't affect).
        """
        # Residual subtraction: remove natively-stereo stem contributions
        # from both predicted and engineer mixes.
        if stems is not None and is_stereo is not None:
            mask = is_stereo.to(stems.dtype)
            if track_mask is not None:
                mask = mask * track_mask.to(stems.dtype)
            mask = mask.view(*is_stereo.shape, 1, 1)            # (B, N, 1, 1)
            stereo_contribution = (stems * mask).sum(dim=1)     # (B, 2, T)
            pred_res = predicted_mix - stereo_contribution
            eng_res = target_mix - stereo_contribution
        else:
            pred_res = predicted_mix
            eng_res = target_mix

        bal_pred, wid_pred, ild_pred, _ = self.per_band_features(pred_res)
        bal_eng, wid_eng, ild_eng, en_eng = self.per_band_features(eng_res)

        # Energy-weight both matches by the engineer's per-band magnitude:
        # bands with content drive the loss; silent bands (ill-defined ratios)
        # get ≈ 0 weight.
        w = en_eng.detach()                                     # (B, n_bins)
        wsum = w.sum(dim=-1) + self.eps
        # DIRECTION — signed balance: penalizes committing to the wrong side.
        balance_loss = ((w * (bal_pred - bal_eng).abs()).sum(dim=-1) / wsum).mean()
        # COMMITMENT — width magnitude: penalizes collapsing to center. Sign-
        # blind by nature (width has no left/right), so it CANNOT drive a
        # single-side collapse on its own — that's the direction term's job.
        width_loss = ((w * (wid_pred - wid_eng).abs()).sum(dim=-1) / wsum).mean()

        loss = self.w_balance * balance_loss + self.w_width * width_loss

        # ILD kept as an interpretable dB metric only (not in the loss).
        ild_l1_db = (ild_pred - ild_eng).abs().mean()

        with torch.no_grad():
            ild_diff = ild_pred - ild_eng
            ild_rmse_db = ild_diff.pow(2).mean().sqrt()
            # Note: width/imbalance metrics still computed on the FULL mix
            # (not residual) — those are sanity metrics for the rendered mix,
            # not the optimization target.
            width_pred = self.stereo_width(predicted_mix).mean()
            width_eng = self.stereo_width(target_mix).mean()
            imb_pred = self.stereo_imbalance(predicted_mix).mean()
            imb_eng = self.stereo_imbalance(target_mix).mean()
            metrics = {
                "total":              loss.item(),
                "ild/l1_db":          ild_l1_db.item(),
                "ild/rmse_db":        ild_rmse_db.item(),
                "ild/per_bin_l1_db":  ild_diff.abs().mean(dim=0).tolist(),
                "balance":            balance_loss.item(),
                "width":              width_loss.item(),
                "width/pred":         width_pred.item(),
                "width/eng":          width_eng.item(),
                "width/err":          (width_pred - width_eng).item(),
                "imbalance/pred":     imb_pred.item(),
                "imbalance/eng":      imb_eng.item(),
                "imbalance/err":      (imb_pred - imb_eng).item(),
            }
        return loss, metrics


__all__ = ["PanReconLoss"]


# ---------- Smoke ----------

def _smoke() -> None:
    """Verify the per-band features and the two-term loss behave as designed."""
    torch.manual_seed(0)
    sr, T = 48_000, 2 * 48_000
    loss_fn = PanReconLoss(sample_rate=sr)

    # White noise mono source, used in all tests
    mono = torch.randn(1, T) * 0.1

    centered = torch.stack([mono, mono], dim=1)                            # (1, 2, T)
    hard_L = torch.stack([mono, torch.zeros_like(mono)], dim=1)
    hard_R = torch.stack([torch.zeros_like(mono), mono], dim=1)

    bal_c, wid_c, ild_c, _ = loss_fn.per_band_features(centered)
    bal_L, wid_L, ild_L, _ = loss_fn.per_band_features(hard_L)
    bal_R, wid_R, ild_R, _ = loss_fn.per_band_features(hard_R)

    # Test 1: centered → ILD ≈ 0, balance ≈ 0, width ≈ 0
    print(f"centered: max|ILD|={ild_c.abs().max().item():.3f}dB  "
          f"max|bal|={bal_c.abs().max().item():.3f}  max|width|={wid_c.abs().max().item():.3f}")
    assert ild_c.abs().max() < 0.5, "centered should give ~0 ILD"
    assert bal_c.abs().max() < 0.05, "centered balance should be ~0"
    assert wid_c.abs().max() < 0.05, "centered width should be ~0"

    # Test 2/3: hard pans → ILD saturates ±clip, balance ±1, width 1 (both sides)
    print(f"hard-L: ILD min={ild_L.min().item():.1f}dB  bal={bal_L.mean().item():+.3f}  width={wid_L.mean().item():.3f}")
    print(f"hard-R: ILD max={ild_R.max().item():.1f}dB  bal={bal_R.mean().item():+.3f}  width={wid_R.mean().item():.3f}")
    assert ild_L.min() >= 20 and ild_L.max() <= loss_fn.ild_clip_db + 1e-3
    assert ild_R.max() <= -20 and ild_R.min() >= -loss_fn.ild_clip_db - 1e-3
    assert bal_L.mean() > 0.9 and bal_R.mean() < -0.9
    assert wid_L.mean() > 0.9 and wid_R.mean() > 0.9

    # Test 3b: the DESIGN INVARIANT (the whole point of the fix).
    #   DIRECTION (balance) MUST distinguish L from R (sign-aware).
    #   COMMITMENT (width) MUST NOT distinguish L from R (directionless) but
    #   MUST distinguish either from center (the all-center guard).
    assert (bal_L - bal_R).abs().min() > 1.5, "balance must be SIGN-AWARE (L≠R)"
    assert (wid_L - wid_R).abs().max() < 0.05, "width must be SIGN-BLIND (L==R)"
    assert (wid_L - wid_c).min() > 0.9, "width must separate committed from center"
    print("invariants OK: balance sign-aware, width sign-blind but commitment-sensitive")

    # Test 4: wrong-side mismatch is high-loss; perfect match ≈ 0
    loss_match, _ = loss_fn(hard_L, hard_L)
    loss_mismatch, _ = loss_fn(hard_L, hard_R)
    print(f"loss(L vs L) = {loss_match.item():.3f},  loss(L vs R) = {loss_mismatch.item():.3f}")
    assert loss_mismatch > loss_match * 10 + 0.1, "wrong-side mismatch should dominate"

    # Test 4b: all-center vs a committed target is ALSO high-loss — the width
    # term's job. (Balance alone would let this slide when aggregate-balanced.)
    loss_center_vs_L, _ = loss_fn(centered, hard_L)
    print(f"loss(center vs L) = {loss_center_vs_L.item():.3f}  (want large — width guard)")
    assert loss_center_vs_L > 0.5, "all-center must be penalized vs a committed target"

    # Test 5: gradient pushes a partially-correct pan toward the target. Start
    # already on the correct side (0.8/0.2) so direction and width agree, and
    # side ≠ 0 (defined width gradient).
    pan_L_factor = torch.tensor(0.8, requires_grad=True)
    pan_R_factor = torch.tensor(0.2, requires_grad=True)
    mono_d = mono.detach()
    pred = torch.stack([pan_L_factor * mono_d, pan_R_factor * mono_d], dim=1)
    loss, _ = loss_fn(pred, hard_L)                                        # want full L
    loss.backward()
    print(f"grad on pan_L = {pan_L_factor.grad.item():+.3f}  pan_R = {pan_R_factor.grad.item():+.3f}")
    assert torch.isfinite(pan_L_factor.grad) and torch.isfinite(pan_R_factor.grad), "NaN grad"
    assert pan_L_factor.grad < 0, "increasing pan_L should reduce loss → grad negative"
    assert pan_R_factor.grad > 0, "decreasing pan_R should reduce loss → grad positive"

    # Test 5b: NaN-safety at exact all-center (L − R = 0 → width sqrt would NaN).
    pc = torch.tensor(0.707, requires_grad=True)
    pred_c = torch.stack([pc * mono_d, pc * mono_d], dim=1)                # L == R exactly
    loss_c, _ = loss_fn(pred_c, hard_L)
    loss_c.backward()
    print(f"all-center grad finite: {torch.isfinite(pc.grad).item()}  (loss={loss_c.item():.3f})")
    assert torch.isfinite(pc.grad), "width term must not NaN at the all-center point"

    print("\nall PanReconLoss smoke tests passed.")


if __name__ == "__main__":
    _smoke()
