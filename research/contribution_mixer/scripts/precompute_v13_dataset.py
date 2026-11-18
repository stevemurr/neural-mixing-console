"""Prepare the historical cache with per-track pan labels alongside EQ examples."""
import sys
from pathlib import Path

import torch

from nmc.data.groups import UNKNOWN_IDX
from nmc.eq.contribution_console import ContributionConsole
from nmc.eq.precompute import compute_target_delta_db, main as prepare_examples

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.mono_pan import detect_stereo, recover_pan_targets  # noqa: E402

__all__ = ["compute_target_delta_db", "compute_pan_targets_session", "main"]


def compute_pan_targets_session(
    stems: torch.Tensor,                # (N, 2, T)
    mix: torch.Tensor,                  # (2, T)
    console: ContributionConsole,
    stereo_threshold: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Session-level per-track pan targets via LS recovery.

    Procedure:
      1. Detect stereo stems (`is_stereo[t]`) by L-vs-R RMS divergence.
      2. Per-stem raw mel mag (mono and per-channel L/R), top-active aggregated.
      3. Per-mix raw mel mag L and R.
      4. Subtract stereo stems' inherent L-bias contribution from the engineer
         L-bias.
      5. LS-solve for the pan-direction factor s[t] on mono stems only.
      6. Recover pan_dir from s via constant-power inverse.

    Returns (pan_dir[N], is_stereo[N]).
    """
    with torch.no_grad():
        # (1) stereo detection
        is_stereo = detect_stereo(stems, threshold=stereo_threshold)        # (N,)

        # (2) per-stem mel mag mono + per-channel L/R
        stems_b = stems.unsqueeze(0)                                         # (1, N, 2, T)
        stem_mag_mono = console.compute_mel_mag(stems_b).squeeze(0)         # (N, n_bins)
        stem_mag_L, stem_mag_R = console.compute_mel_mag_LR(stems_b)
        stem_mag_L = stem_mag_L.squeeze(0)
        stem_mag_R = stem_mag_R.squeeze(0)

        # (3) per-mix mel mag L and R — treat mix as (1, 1, 2, T)
        mix_b = mix.unsqueeze(0).unsqueeze(0)
        mix_mag_L, mix_mag_R = console.compute_mel_mag_LR(mix_b)
        mix_mag_L = mix_mag_L.squeeze()                                      # (n_bins,)
        mix_mag_R = mix_mag_R.squeeze()

        # (4)+(5)+(6) LS
        pan_dir = recover_pan_targets(
            stem_mel_mag=stem_mag_mono,
            stem_mel_L=stem_mag_L,
            stem_mel_R=stem_mag_R,
            mix_mel_L=mix_mag_L,
            mix_mel_R=mix_mag_R,
            is_stereo=is_stereo,
        )
    return pan_dir, is_stereo


def pan_labels(stems, mix, console, stem_groups, n_max):
    pan, stereo = compute_pan_targets_session(stems, mix, console)
    n = len(stems)
    pan_target = torch.zeros(n_max)
    is_stereo = torch.zeros(n_max, dtype=torch.bool)
    group_idx = torch.full((n_max,), UNKNOWN_IDX, dtype=torch.long)
    pan_target[:n] = pan.cpu()
    is_stereo[:n] = stereo.cpu()
    group_idx[:n] = torch.tensor(stem_groups)
    return {"pan_target": pan_target, "is_stereo": is_stereo, "group_idx": group_idx}


def main():
    return prepare_examples(default_output="dmc-data/v13_precomputed.pt", extra_labels=pan_labels)


if __name__ == "__main__":
    sys.exit(main())
