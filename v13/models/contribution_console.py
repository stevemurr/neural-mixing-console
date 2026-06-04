"""ContributionConsole — v13's per-track EQ + sum stage with closed-form
delta distribution.

Architecture (notes/v13_contribution_mixer_design.md):

    Input:
        stems            (B, N, 2, T)
        target_delta_db  (B, n_bins)   — Stage 2's prediction in dB
        track_mask       (B, N) bool   — pad mask

    1. Compute per-stem contribution dictionary C[bin, track] via the top-k
       active-frame mel-magnitude aggregation used in the separability
       analysis (active_pct=0.20 by default).

    2. Distribute the mix-bus delta across tracks weighted by share at
       each bin:
            per_track_delta_db[b, t] = target_delta_db[b]
                                       * C[b, t] / Σ_t' C[b, t']

       Tracks with high contribution at the problem band absorb most of
       the change; near-silent tracks at that band get near-zero change.
       This is the closed-form distribution policy.

    3. Apply per-track 26-band mel EQ (gain-invariant — pure spectral
       shape, no level effect).

    4. Sum to stereo mix.

The only learned component fed into this module is `target_delta_db` from
the Stage 2 target-curve net. Everything else is deterministic.

This deliberately skips a per-instrument Stage 1 prior — the separability
analysis showed the bin contribution itself carries enough instrument-
class signal for the closed-form policy to make engineering-sensible
distribution decisions (Drums F1=0.85, Guitar 0.82, broad classes ≥0.54).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio

from .mel_bin_eq import MelBinEQ
from .mono_pan import MonoPan


class ContributionConsole(nn.Module):
    def __init__(
        self,
        sample_rate: int = 48_000,
        n_bins: int = 26,
        eq_gain_range_db: float = 18.0,
        eq_gain_invariant: bool = True,
        mel_n_fft: int = 8192,
        mel_hop: int = 2048,
        f_min: float = 20.0,
        f_max: float = 16_000.0,
        active_pct: float = 0.20,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_bins = n_bins
        self.eq_gain_range_db = eq_gain_range_db
        self.active_pct = active_pct
        self.eps = eps

        self.eq = MelBinEQ(
            sample_rate=sample_rate,
            n_bins=n_bins,
            gain_range_db=eq_gain_range_db,
            gain_invariant=eq_gain_invariant,
            f_min=f_min,
            f_max=f_max,
        )
        self.pan = MonoPan()

        # Mel-magnitude transform for the contribution dictionary.
        # We use torchaudio's standard HTK filterbank — slight overlap
        # vs MelBinEQ's disjoint bin-to-band mapping, but the distribution
        # math is robust to that ~boundary mismatch.
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=mel_n_fft,
            hop_length=mel_hop,
            n_mels=n_bins,
            f_min=f_min,
            f_max=f_max,
            power=1.0,
            norm=None,
            mel_scale="htk",
        )

    # ---------- Contribution dictionary ----------

    def _active_mel_from_audio(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Top-`active_pct` active-frame mean of raw mel magnitudes.

        x_flat: (M, T) flat mono audio
        returns: (M, n_bins) raw mel mag
        """
        mel = self.mel(x_flat.float())                         # (M, n_bins, n_frames)
        n_frames = mel.shape[-1]
        k = max(1, int(n_frames * self.active_pct))
        frame_energy = mel.sum(dim=1)                          # (M, n_frames)
        _, top_idx = torch.topk(frame_energy, k, dim=-1, largest=True)
        idx_exp = top_idx.unsqueeze(1).expand(-1, self.n_bins, -1)
        top_mel = torch.gather(mel, dim=-1, index=idx_exp)     # (M, n_bins, k)
        return top_mel.mean(dim=-1)                            # (M, n_bins)

    def compute_C(
        self, stems: torch.Tensor, track_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Top-`active_pct` active-frame mean *log1p* mel mag per stem (mono).

        stems:      (B, N, 2, T)
        returns:    C  (B, N, n_bins)  — log1p mag per stem per bin
        """
        B, N, _C, T = stems.shape
        mono = stems.mean(dim=2)                              # (B, N, T)
        raw = self._active_mel_from_audio(mono.reshape(B * N, T))
        C_mat = torch.log1p(raw).reshape(B, N, self.n_bins)
        if track_mask is not None:
            C_mat = C_mat * track_mask.to(C_mat.dtype).unsqueeze(-1)
        return C_mat

    def compute_mel_mag(
        self, stems: torch.Tensor, track_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Raw per-stem mel mag (mono mix-down). Used by the pan LS target
        recovery, where linear-magnitude space is the right place for the
        constraint Σ_t pan_factor[t] · m[t, b] = engineer L-bias[b]."""
        B, N, _C, T = stems.shape
        mono = stems.mean(dim=2)
        out = self._active_mel_from_audio(mono.reshape(B * N, T)).reshape(B, N, self.n_bins)
        if track_mask is not None:
            out = out * track_mask.to(out.dtype).unsqueeze(-1)
        return out

    def compute_mel_mag_LR(
        self, stems: torch.Tensor, track_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw per-stem mel mag for L and R channels separately. Stems'
        L-vs-R difference is the signal that distinguishes stereo content
        from mono content (and from intentional panning at the mix bus)."""
        B, N, C, T = stems.shape
        L = stems[..., 0, :].reshape(B * N, T)
        R = stems[..., 1, :].reshape(B * N, T)
        mel_L = self._active_mel_from_audio(L).reshape(B, N, self.n_bins)
        mel_R = self._active_mel_from_audio(R).reshape(B, N, self.n_bins)
        if track_mask is not None:
            m = track_mask.to(mel_L.dtype).unsqueeze(-1)
            mel_L = mel_L * m
            mel_R = mel_R * m
        return mel_L, mel_R

    # ---------- Distribution policy ----------

    def distribute_delta(
        self,
        target_delta_db: torch.Tensor,    # (B, n_bins)
        C_mat: torch.Tensor,              # (B, N, n_bins)
        track_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Minimum-disturbance distribution that actually reaches the target
        mix-bus delta.

        Working in linear magnitude, choose per-track factors f[t, b] that
        minimize ||f − 1||² subject to Σ_t c[t, b] · f[t, b] = Σ c · target_factor.
        Closed-form (regularized least-squares):

            f[t, b] = 1 + c[t, b] · Σ c[t', b] / Σ c[t', b]²  ·  (target_factor − 1)

        Limits:
          - uniform contributions  → every track gets the full target delta
          - one dominant track     → that track absorbs the full move; rest = 0
          - silent at the band     → factor stays at 1 (no change)

        Returns (B, N, n_bins) per-track dB deltas.

        (The earlier `target_delta * C/ΣC` formula under-shoots the target
        sum effect by ~1/N for N near-uniform contributors — confirmed in
        v13 audition: target +5 dB at the mix bus ⇒ +0.16 dB per track for
        32 tracks. The LS formula is the math that actually hits target.)
        """
        contrib_lin = torch.expm1(C_mat).clamp(min=self.eps)    # invert log1p
        if track_mask is not None:
            contrib_lin = contrib_lin * track_mask.to(contrib_lin.dtype).unsqueeze(-1)

        sum_c = contrib_lin.sum(dim=1)                                       # (B, n_bins)
        sum_c_sq = (contrib_lin ** 2).sum(dim=1).clamp(min=self.eps)        # (B, n_bins)
        target_factor = 10.0 ** (target_delta_db / 20.0)                     # (B, n_bins)

        # Closed-form factor per (track, bin)
        scale = (sum_c / sum_c_sq).unsqueeze(1)                              # (B, 1, n_bins)
        f = 1.0 + (contrib_lin * scale) * (target_factor - 1.0).unsqueeze(1) # (B, N, n_bins)
        f = f.clamp(min=self.eps)
        per_track_delta_db = 20.0 * torch.log10(f)
        return per_track_delta_db

    def _delta_db_to_gain_norm(self, delta_db: torch.Tensor) -> torch.Tensor:
        """Map per-track dB delta → [0,1] for MelBinEQ.
        gain_norm = 0.5 ↔ 0 dB, 0 ↔ -range, 1 ↔ +range. Clamped."""
        return ((delta_db / self.eq_gain_range_db) * 0.5 + 0.5).clamp(0.0, 1.0)

    # ---------- Forward ----------

    def forward(
        self,
        stems: torch.Tensor,                # (B, N, 2, T)
        target_delta_db: torch.Tensor,      # (B, n_bins)
        pan_dir: torch.Tensor | None = None,        # (B, N) in [-1, +1]
        is_stereo: torch.Tensor | None = None,      # (B, N) bool
        track_mask: torch.Tensor | None = None,     # (B, N)
    ) -> tuple[torch.Tensor, dict]:
        """Compose per-track EQ → per-track pan → sum.

        - `target_delta_db` is Stage 2's mix-bus EQ prediction.
        - `pan_dir` is Stage 3's per-track pan prediction; if None, no pan
          is applied (legacy / EQ-only behavior).
        - `is_stereo` marks tracks that should bypass pan (stereo sources).

        Returns (mix, aux). `aux` exposes intermediate state for logging.
        """
        if stems.dim() != 4 or stems.shape[2] != 2:
            raise ValueError(f"expected (B, N, 2, T) stems; got {stems.shape}")
        if target_delta_db.shape[-1] != self.n_bins:
            raise ValueError(
                f"target_delta_db last dim must be {self.n_bins}; "
                f"got {target_delta_db.shape}"
            )
        B, N, C, T = stems.shape

        C_mat = self.compute_C(stems, track_mask=track_mask)                # (B, N, n_bins)
        per_track_delta = self.distribute_delta(
            target_delta_db, C_mat, track_mask=track_mask,
        )                                                                    # (B, N, n_bins)
        gain_norm = self._delta_db_to_gain_norm(per_track_delta)            # (B, N, n_bins)

        x = stems.reshape(B * N, C, T)
        g = gain_norm.reshape(B * N, self.n_bins)
        eq_stems = self.eq(x, g).reshape(B, N, C, T)

        if track_mask is not None:
            eq_stems = eq_stems * track_mask.to(eq_stems.dtype).view(B, N, 1, 1)

        aux: dict = {
            "C": C_mat,
            "per_track_delta_db": per_track_delta,
            "gain_norm": gain_norm,
            "eq_stems": eq_stems,
        }

        if pan_dir is not None:
            if is_stereo is None:
                is_stereo = torch.zeros(
                    B, N, dtype=torch.bool, device=pan_dir.device,
                )
            panned = self.pan(eq_stems, pan_dir, is_stereo)                 # (B, N, 2, T)
            if track_mask is not None:
                panned = panned * track_mask.to(panned.dtype).view(B, N, 1, 1)
            aux["panned_stems"] = panned
            aux["pan_dir"] = pan_dir
            aux["is_stereo"] = is_stereo
            mix = panned.sum(dim=1)
        else:
            mix = eq_stems.sum(dim=1)
        return mix, aux


__all__ = ["ContributionConsole"]


# ---------- Smoke ----------

def _smoke() -> None:
    """Verify (1) shapes & gradients, (2) zero-delta is identity,
    (3) contribution math gives dominant track the lion's share."""
    import math as _m
    torch.manual_seed(0)
    sr, T = 48_000, 6 * 48_000
    B, N = 2, 8
    console = ContributionConsole(sample_rate=sr, n_bins=26)

    # Test 1: shapes + gradient flow
    stems = (torch.randn(B, N, 2, T) * 0.05).requires_grad_(True)
    target_delta_db = (torch.randn(B, 26) * 3.0).requires_grad_(True)
    mix, aux = console(stems, target_delta_db)
    assert mix.shape == (B, 2, T), f"bad mix shape {mix.shape}"
    assert aux["C"].shape == (B, N, 26), f"bad C shape {aux['C'].shape}"
    assert aux["per_track_delta_db"].shape == (B, N, 26)
    mix.sum().backward()
    assert stems.grad is not None and torch.isfinite(stems.grad).all(), \
        "no/NaN gradient into stems"
    assert target_delta_db.grad is not None and torch.isfinite(target_delta_db.grad).all(), \
        "no/NaN gradient into target_delta_db"
    print(f"shapes ok, gradients flow finite. mix={tuple(mix.shape)}")

    # Test 2: zero target delta → mix is close to dry sum
    with torch.no_grad():
        mix_zero, _ = console(stems.detach(), torch.zeros(B, 26))
        dry_sum = stems.detach().sum(dim=1)
        # Gain-invariant EQ has minor non-identity even at neutral due to RMS
        # rescaling of each stem from its individual normalization.
        # Compare with a tolerance relative to typical magnitudes.
        rel = (mix_zero - dry_sum).abs().mean() / dry_sum.abs().mean()
        print(f"zero-delta mix vs dry sum rel-err: {rel.item():.2e}  (want < 1e-3)")
        assert rel < 1e-3, "zero delta should preserve dry sum"

    # Test 3: dominance — synth a band where track 0 carries all the energy,
    # apply +6 dB delta there; check track 0 gets the lion's share.
    stems_d = torch.zeros(1, N, 2, T)
    t = torch.arange(T) / sr
    # 4 kHz sine on track 0; silence on others
    stems_d[0, 0] = torch.sin(2 * _m.pi * 4000 * t).unsqueeze(0).expand(2, -1) * 0.1
    delta = torch.zeros(1, 26)
    band_idx = int(torch.searchsorted(console.eq.hz_edges, torch.tensor(4000.0))) - 1
    delta[0, band_idx] = 6.0
    _, aux_d = console(stems_d, delta)
    deltas = aux_d["per_track_delta_db"][0, :, band_idx]   # (N,)
    share_t0 = deltas[0].abs() / deltas.abs().sum().clamp(min=1e-6)
    print(f"dominant-track share at boosted band: {share_t0.item():.3f}  (want > 0.95)")
    assert share_t0 > 0.95, "track with 100% contribution should absorb ≥95% of delta"

    # Test 4: LS distribution reaches the target — synth N equal-amplitude
    # noise tracks all contributing equally, ask for +6 dB on the mix.
    # Each track should get ~+6 dB (uniform case → full delta per track).
    torch.manual_seed(7)
    stems_u = torch.randn(1, N, 2, T) * 0.1
    delta_u = torch.zeros(1, 26)
    delta_u[0, band_idx] = 6.0
    _, aux_u = console(stems_u, delta_u)
    per_track_at_band = aux_u["per_track_delta_db"][0, :, band_idx]
    mean_db = per_track_at_band.mean().item()
    print(f"uniform contributions, target +6 dB → mean per-track delta: {mean_db:.2f} dB  (want ~6.0)")
    assert abs(mean_db - 6.0) < 0.5, f"LS distribution should give ~6 dB per track; got {mean_db:.2f}"

    print("\nall ContributionConsole smoke tests passed.")


if __name__ == "__main__":
    _smoke()
