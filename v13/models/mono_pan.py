"""MonoPan — constant-power panning for mono stems; stereo stems pass through.

Pan parameter `pan_dir ∈ [-1, +1]`:
    -1 → full left   (pan_L=1, pan_R=0)
     0 → center      (pan_L=pan_R=√½ ≈ 0.707)
    +1 → full right  (pan_L=0, pan_R=1)

Constant-power: `pan_L² + pan_R² = 1` across all pan positions. This is the
DAW convention (e.g., a centered mono track plays back at −3 dB in each
channel, total power preserved).

`is_stereo[t] = True` bypasses panning for that track — L and R pass through
unchanged. This preserves the natural stereo image of stems that were
recorded or designed in stereo (organs, pads, stereo drum overheads).

See notes/v13_pan_design_2026-06.md for the design rationale.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _pan_LR(pan_dir: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map pan_dir ∈ [-1, +1] → (pan_L, pan_R) constant-power coefficients."""
    theta = (pan_dir.clamp(-1.0, 1.0) + 1.0) * (math.pi / 4)
    return torch.cos(theta), torch.sin(theta)


class MonoPan(nn.Module):
    """Apply per-track pan to mono stems; pass stereo stems through unchanged."""

    def forward(
        self,
        stems: torch.Tensor,                 # (B, N, 2, T)
        pan_dir: torch.Tensor,               # (B, N)  in [-1, +1]
        is_stereo: torch.Tensor,             # (B, N)  bool
    ) -> torch.Tensor:
        """Returns (B, N, 2, T) panned stems."""
        if stems.dim() != 4 or stems.shape[2] != 2:
            raise ValueError(f"expected (B, N, 2, T) stems; got {stems.shape}")
        if pan_dir.shape != stems.shape[:2]:
            raise ValueError(f"pan_dir shape {pan_dir.shape} != {stems.shape[:2]}")
        if is_stereo.shape != stems.shape[:2]:
            raise ValueError(f"is_stereo shape {is_stereo.shape} != {stems.shape[:2]}")

        # Mono signal (average of L+R; for duplicated mono stems L == R so this
        # is identity, for non-duplicated mono it averages the channels).
        mono = stems.mean(dim=-2)                            # (B, N, T)

        pan_L, pan_R = _pan_LR(pan_dir)
        pan_L = pan_L.unsqueeze(-1)                          # (B, N, 1)
        pan_R = pan_R.unsqueeze(-1)
        mono_L = pan_L * mono                                # (B, N, T)
        mono_R = pan_R * mono
        mono_panned = torch.stack([mono_L, mono_R], dim=-2)  # (B, N, 2, T)

        is_stereo_b = is_stereo.view(*is_stereo.shape, 1, 1).to(stems.dtype)
        return is_stereo_b * stems + (1.0 - is_stereo_b) * mono_panned


# ---------- Stereo-stem detection ----------

def detect_stereo(stems: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """Mark stems whose L and R differ substantially as stereo.

    Criterion: `RMS(L - R) / RMS(L + R) > threshold`. Mono stems with L
    duplicated to R yield ≈ 0; stems with real stereo content yield > 0.05.

    stems: (..., 2, T) → returns (...) bool of same leading dims.
    """
    eps = 1e-8
    diff = stems[..., 0, :] - stems[..., 1, :]
    summ = stems[..., 0, :] + stems[..., 1, :]
    diff_rms = (diff ** 2).mean(dim=-1).clamp(min=eps).sqrt()
    sum_rms = (summ ** 2).mean(dim=-1).clamp(min=eps).sqrt()
    return (diff_rms / sum_rms) > threshold


# ---------- LS pan-target recovery ----------
#
# For mono stems indexed by t with mel-magnitude m[t, b] at band b and
# pan-direction factor s[t] = pan_L[t] − pan_R[t] ∈ [−1, +1]:
#
#     mel_mag(L_mix)[b] − mel_mag(R_mix)[b] ≈ Σ_t m[t, b] · s[t] + stereo_bias[b]
#
# After subtracting stereo_bias, we solve a regularized LS for s[t]:
#
#     s = (M Mᵀ + λI)⁻¹ M · L_bias_to_pan
#
# Where M[t, b] = mel_mag for mono stems only. Then recover pan_dir from s:
#
#     pan_dir = 4 · arccos(clamp(s/√2)) / π − 2
#
# Limits:
#     s = +1  → pan_dir = −1  (full left)
#     s =  0  → pan_dir =  0  (center)
#     s = −1  → pan_dir = +1  (full right)

def recover_pan_targets(
    stem_mel_mag: torch.Tensor,    # (N, n_bins) — per-stem mel mag (mono mix-down)
    stem_mel_L: torch.Tensor,      # (N, n_bins) — left-channel mel mag per stem
    stem_mel_R: torch.Tensor,      # (N, n_bins) — right-channel mel mag per stem
    mix_mel_L: torch.Tensor,       # (n_bins,)
    mix_mel_R: torch.Tensor,       # (n_bins,)
    is_stereo: torch.Tensor,       # (N,) bool
    reg: float = 0.1,
) -> torch.Tensor:
    """Closed-form per-track pan_dir extraction. Returns (N,) in [-1, +1]."""
    N = stem_mel_mag.shape[0]
    device = stem_mel_mag.device
    dtype = stem_mel_mag.dtype

    L_bias_engineer = mix_mel_L - mix_mel_R                                  # (n_bins,)

    # Stereo stems' inherent L-R contribution (not from a pan choice)
    stereo_mask = is_stereo
    stereo_bias = (stem_mel_L - stem_mel_R) * stereo_mask.to(dtype).unsqueeze(-1)
    stereo_bias_sum = stereo_bias.sum(dim=0)                                 # (n_bins,)

    L_bias_to_pan = L_bias_engineer - stereo_bias_sum                        # (n_bins,)

    mono_mask = ~stereo_mask
    n_mono = int(mono_mask.sum().item())
    pan_dir = torch.zeros(N, device=device, dtype=dtype)
    if n_mono == 0:
        return pan_dir

    M = stem_mel_mag[mono_mask]                                              # (n_mono, n_bins)
    MMt = M @ M.t()                                                          # (n_mono, n_mono)
    reg_eye = reg * torch.eye(n_mono, device=device, dtype=dtype)
    rhs = M @ L_bias_to_pan                                                  # (n_mono,)
    s = torch.linalg.solve(MMt + reg_eye, rhs)                               # (n_mono,)

    # Recover pan_dir from s = pan_L − pan_R using
    #   pan_dir = 4 · arccos(s_physical/√2) / π − 2
    # Physical range of s is [-1, +1]; LS may overshoot when the engineer's
    # L-bias at some band exceeds what a single track can contribute.
    # Clamping to the physical range = "this track is hard-panned" in those
    # cases — engineering-correct, math-stable.
    s_physical = s.clamp(-1.0, 1.0)
    s_div = (s_physical / math.sqrt(2.0)).clamp(-1.0, 1.0)
    pan_dir_mono = 4.0 * torch.arccos(s_div) / math.pi - 2.0

    pan_dir[mono_mask] = pan_dir_mono
    return pan_dir


__all__ = ["MonoPan", "detect_stereo", "recover_pan_targets"]


# ---------- Smoke ----------

def _smoke() -> None:
    """Verify: (1) center pan preserves mid spectrum, (2) hard L silences R,
    (3) hard R silences L, (4) stereo stems pass through unchanged,
    (5) LS recovery matches a known pan choice."""
    torch.manual_seed(0)
    sr, T = 48_000, 2 * 48_000
    pan = MonoPan()

    # Construct (1, 4, 2, T): 4 mono tracks with L == R duplicated
    mono_signal = torch.randn(1, 4, T) * 0.1
    stems = torch.stack([mono_signal, mono_signal], dim=-2)  # L = R
    is_stereo = torch.zeros(1, 4, dtype=torch.bool)

    # Test 1: pan_dir = 0 (center) → output L == output R; total preserved up to gain
    pd = torch.zeros(1, 4)
    out = pan(stems, pd, is_stereo)
    diff = (out[..., 0, :] - out[..., 1, :]).abs().max()
    print(f"center pan: max |L − R| = {diff.item():.2e}  (want < 1e-6)")
    assert diff < 1e-6, "centered pan should keep L == R"

    # Test 2: pan_dir = -1 (full left) → R silent
    pd = torch.full((1, 4), -1.0)
    out = pan(stems, pd, is_stereo)
    R_amp = out[..., 1, :].abs().max()
    print(f"hard left:  R amp = {R_amp.item():.2e}  (want < 1e-6)")
    assert R_amp < 1e-6, "hard left should silence R"

    # Test 3: pan_dir = +1 (full right) → L silent
    pd = torch.full((1, 4), 1.0)
    out = pan(stems, pd, is_stereo)
    L_amp = out[..., 0, :].abs().max()
    print(f"hard right: L amp = {L_amp.item():.2e}  (want < 1e-6)")
    assert L_amp < 1e-6, "hard right should silence L"

    # Test 4: stereo stems pass through unchanged regardless of pan_dir
    stereo_L = torch.randn(1, 4, T) * 0.1
    stereo_R = torch.randn(1, 4, T) * 0.1
    stereo_stems = torch.stack([stereo_L, stereo_R], dim=-2)
    all_stereo = torch.ones(1, 4, dtype=torch.bool)
    pd = torch.full((1, 4), 0.5)                              # arbitrary pan — should be ignored
    out = pan(stereo_stems, pd, all_stereo)
    err = (out - stereo_stems).abs().max()
    print(f"stereo passthrough: max diff = {err.item():.2e}  (want < 1e-6)")
    assert err < 1e-6, "stereo stems must pass through unchanged"

    # Test 5: LS recovery on a known pan setup
    # 3 mono stems with distinct mel signatures + 1 stereo passthrough.
    n_bins = 8
    N = 4
    torch.manual_seed(1)
    # Per-stem mel magnitudes (mono mix-down): make them clearly distinct
    M = torch.zeros(N, n_bins)
    M[0, :3] = torch.tensor([3.0, 2.0, 1.0])       # bass-heavy
    M[1, 2:5] = torch.tensor([2.0, 3.0, 2.0])      # mid
    M[2, 5:8] = torch.tensor([1.0, 2.0, 3.0])      # bright
    M[3] = 0.5                                      # the stereo one
    is_stereo_t = torch.tensor([False, False, False, True])

    # Pretend the engineer panned: stem 0 hard left, stem 1 center, stem 2 hard right
    # Construct the resulting L_mix and R_mix mel mags:
    # L_mix = pan_L[t] · mono[t] (for mono); stereo stems contribute their L unchanged
    pan_truth = torch.tensor([-1.0, 0.0, +1.0, 0.0])
    pL, pR = _pan_LR(pan_truth)
    # For the stereo stem, just contribute equal L/R (since we pass them through)
    # Pretend stereo L = stereo R = M[3]
    stem_L_mag = torch.zeros(N, n_bins)
    stem_R_mag = torch.zeros(N, n_bins)
    # mono stems contribute pan_L/pan_R times mono mag
    for t in [0, 1, 2]:
        stem_L_mag[t] = pL[t] * M[t]
        stem_R_mag[t] = pR[t] * M[t]
    # stereo stem contributes equally (already in mid)
    stem_L_mag[3] = M[3]
    stem_R_mag[3] = M[3]
    mix_L = stem_L_mag.sum(dim=0)
    mix_R = stem_R_mag.sum(dim=0)

    recovered = recover_pan_targets(
        stem_mel_mag=M,
        stem_mel_L=stem_L_mag, stem_mel_R=stem_R_mag,
        mix_mel_L=mix_L, mix_mel_R=mix_R,
        is_stereo=is_stereo_t, reg=1e-4,
    )
    err = (recovered[:3] - pan_truth[:3]).abs()
    print(f"LS pan recovery error for [-1, 0, +1]: {err.tolist()}  (want < 0.1)")
    assert err.max() < 0.1, f"LS should recover pan_truth; got {recovered.tolist()}"

    print("\nall MonoPan smoke tests passed.")


if __name__ == "__main__":
    _smoke()
