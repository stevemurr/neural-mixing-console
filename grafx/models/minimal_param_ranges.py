"""Per-(proc, param) ranges for MinimalConsole, with normalize/denormalize.

The encoder outputs normalized values in [0, 1] (via sigmoid on its
raw Linear outputs). This module maps those to engineering units (Hz, dB,
ratios) and back, using either linear or log-uniform mappings depending
on the param's natural scale.

Used by:
  - `training/synth_data.ParamSampler`: samples uniform in [0,1], then
    denormalizes to engineering space for synth_mix rendering and as
    L_param targets in [0,1] space.
  - `training/train_minimal.py`: applies sigmoid to encoder logits,
    denormalizes before passing to console.

Ranges are tuned to be:
  - Wide enough to cover real engineering choices
  - Narrow enough to avoid numerically-degenerate biquad coefficients
    (e.g., frequencies above Nyquist, Q approaching 0)
"""

from __future__ import annotations
import math
import torch


# Per-(proc, param) range as (lo, hi, scale) where scale ∈ {"linear", "log"}.
# "linear": eng = lo + norm * (hi - lo)
# "log":    eng = lo * (hi / lo) ** norm  (== exp(log(lo) + norm * log(hi/lo)))
#
# norm = sigmoid(encoder_logit) ∈ [0, 1]
#
# Strip and group share the same ranges (same processor structure at both
# levels).
# v12.1: drywet removed from eq, stereo_imager, gain_panning to eliminate
# the proc-gain × drywet gauge orbit (drywet * gain(x) + (1-drywet) * x is
# identifiable only via the composite, not the split). Kept on compressor
# because parallel compression is a genuine engineering DoF (the output
# is a perceptually-distinct blend, not redundant with internal gain).
RANGES: dict[str, dict[str, tuple[float, float, str]]] = {
    "eq": {
        # Frequencies: separated bands to avoid overlap-fighting.
        # At sr=48 kHz, Nyquist is 24 kHz — upper bounds leave headroom.
        "hpf_freq":     (20.0,    150.0,   "log"),
        "ls_freq":      (80.0,    500.0,   "log"),
        "ls_gain":      (-8.0,    8.0,     "linear"),  # dB
        "ls_q":         (0.5,     2.0,     "log"),
        "p1_freq":      (200.0,   2000.0,  "log"),
        "p1_gain":      (-8.0,    8.0,     "linear"),
        "p1_q":         (0.5,     2.0,     "log"),
        "p2_freq":      (1000.0,  8000.0,  "log"),
        "p2_gain":      (-8.0,    8.0,     "linear"),
        "p2_q":         (0.5,     2.0,     "log"),
        "hs_freq":      (5000.0,  18000.0, "log"),  # air band
        "hs_gain":      (-8.0,    8.0,     "linear"),
        "hs_q":         (0.5,     2.0,     "log"),
        "lpf_freq":     (8000.0,  20000.0, "log"),
    },
    "compressor": {
        # Grafx's Compressor takes log-scale internal params. Ranges
        # roughly match what the relabeled teacher landed on.
        "log_threshold": (-2.0,   2.0,     "linear"),
        "log_ratio":     (-2.0,   2.0,     "linear"),
        "log_knee":      (-2.0,   2.0,     "linear"),
        "z_alpha_pre":   (-2.0,   2.0,     "linear"),
        # Drywet logit: maps norm ∈ [0,1] to engineering logit in [-3, 3]
        # → sigmoid(eng) ∈ [0.047, 0.953] — bypass to fully wet, enabling
        # parallel compression as a genuine engineering DoF.
        "drywet_logit":  (-3.0,   3.0,     "linear"),
    },
    "stereo_imager": {
        # Side-channel log_gain — ±1 → side amplitude scale exp(±1) = 0.37×–2.7×
        "log_gain":      (-1.0,   1.0,     "linear"),
    },
    "gain_panning": {
        # Per-channel log_gain ±1 → ±8.7 dB roughly. Shape (2,) for L+R.
        # Strip-level only: removed from group chain (see MinimalConsole)
        # to eliminate the strip×group gain product gauge.
        "log_gain":      (-1.0,   1.0,     "linear"),
    },
}


def denormalize_param(name: str, norm: torch.Tensor, proc: str) -> torch.Tensor:
    """Map norm ∈ [0,1] → engineering units per the range definition.

    `norm` may have any leading shape; the mapping is elementwise.
    """
    lo, hi, scale = RANGES[proc][name]
    if scale == "log":
        return lo * (hi / lo) ** norm
    return lo + norm * (hi - lo)


def normalize_param(name: str, eng: torch.Tensor, proc: str) -> torch.Tensor:
    """Inverse of `denormalize_param` — engineering units → [0,1]."""
    lo, hi, scale = RANGES[proc][name]
    if scale == "log":
        return torch.log(eng / lo) / math.log(hi / lo)
    return (eng - lo) / (hi - lo)


def denormalize_params_dict(
    params_norm: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Denormalize a full strip/group params dict element-wise."""
    return {
        proc: {
            name: denormalize_param(name, v, proc)
            for name, v in pp.items()
        }
        for proc, pp in params_norm.items()
    }


def normalize_params_dict(
    params_eng: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Normalize a full strip/group params dict element-wise."""
    return {
        proc: {
            name: normalize_param(name, v, proc)
            for name, v in pp.items()
        }
        for proc, pp in params_eng.items()
    }


__all__ = [
    "RANGES",
    "denormalize_param", "normalize_param",
    "denormalize_params_dict", "normalize_params_dict",
]
