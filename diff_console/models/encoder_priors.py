"""Engineer-default initial bias values for MixEncoder param heads.

Without an instrument-conditional prior, the encoder starts at the param
head's bias-initialized values which are typically near zero. After sigmoid,
that's 0.5 in normalized [0, 1] — meaning the model's initial guess is
"every param at the middle of its range." That's both unrealistic
(engineers don't compress at threshold = -30 dB by default) and slow to
recover from.

This module provides a *single global* default per param, set to a
plausible engineer baseline (medium-light strip processing, neutral bus).
MERT conditioning provides per-instrument shifts on top.

Usage:
    from models.encoder_priors import init_encoder_priors
    encoder = MixEncoder(...)
    init_encoder_priors(encoder)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from reference.param_norm import normalize


# Engineer baselines in PHYSICAL units. v6.1 schema: gain + EQ + comp
# (no makeup, no knee) + clip-drive+mix + pan. Hand-curated to match common
# modern-mix defaults. These also serve as the target of the identity-prior
# penalty in training (`--w-identity-prior`): params are gently pulled toward
# these "do something light, sensible" values unless the recon loss says
# otherwise — which is how a processor that's meant to be "off" lands at
# EQ gain 0 / comp ratio low / clip mix 0 without a separate bypass flag.
STRIP_DEFAULTS = {
    "gain_db":         0.0,    # don't change gain at init
    "hpf_freq":        40.0,   # gentle low-end clean
    "ls_freq":         100.0, "ls_gain": 0.0, "ls_q": 0.7,
    "p1_freq":         300.0, "p1_gain": 0.0, "p1_q": 1.0,
    "p2_freq":         3000.0, "p2_gain": 0.0, "p2_q": 1.0,
    "hs_freq":         10000.0, "hs_gain": 0.0, "hs_q": 0.7,
    "lpf_freq":        18000.0,
    "threshold_db":    -18.0,  # light comp
    "ratio":           2.5,
    "attack_ms":       10.0,
    "release_ms":      80.0,
    "clip_drive_db":   0.0,    # no clip at init
    "clip_mix":        0.0,    # ...and fully dry
    "pan":             0.0,    # centered
}

# Bus-level defaults — neutral by intent (bus processing should be subtle).
BUS_DEFAULTS = {
    "low_boost_freq":  60.0, "low_boost_gain": 0.0,
    "low_attn_freq":   100.0, "low_attn_gain":  0.0,
    "mid_freq":        1000.0, "mid_gain":      0.0, "mid_q": 1.0,
    "air_freq":        12000.0, "air_gain":     0.0,
    "threshold_db":    -10.0,
    "ratio":           2.0,
    "attack_ms":       30.0,
    "release_ms":      150.0,
}


def _logit(p: float) -> float:
    p = max(min(p, 1.0 - 1e-6), 1e-6)
    return math.log(p / (1.0 - p))


def _last_linear(module: nn.Module) -> nn.Linear | None:
    """Find the final nn.Linear in a sequential head."""
    last = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last = m
    return last


def init_strip_head(head: nn.Module, param_keys: tuple[str, ...]) -> None:
    """Initialize the final output layer of the strip param head so its sigmoid
    output is `STRIP_DEFAULTS` for each param."""
    bias_values = [_logit(normalize(k, STRIP_DEFAULTS.get(k, 0.5)))
                   for k in param_keys]

    final_linear = _last_linear(head)
    if final_linear is None or final_linear.bias.shape[0] != len(param_keys):
        return  # shape mismatch, skip silently
    with torch.no_grad():
        final_linear.bias.data = torch.tensor(bias_values,
                                              dtype=final_linear.bias.dtype,
                                              device=final_linear.bias.device)
        final_linear.weight.data.mul_(0.1)  # weaken initial weight so bias dominates


def init_bus_head(head: nn.Module, param_keys: tuple[str, ...]) -> None:
    """Same as init_strip_head but for bus params (renormalize keys to dataset
    naming with bus_ prefix)."""
    bias_values = []
    for k in param_keys:
        # bus param keys in dataset have "bus_" prefix; strip it for default lookup
        lookup_key = k[len("bus_"):] if k.startswith("bus_") else k
        default = BUS_DEFAULTS.get(lookup_key, 0.5)
        bias_values.append(_logit(normalize(k, default)))

    final_linear = _last_linear(head)
    if final_linear is None or final_linear.bias.shape[0] != len(param_keys):
        return
    with torch.no_grad():
        final_linear.bias.data = torch.tensor(bias_values,
                                              dtype=final_linear.bias.dtype,
                                              device=final_linear.bias.device)
        final_linear.weight.data.mul_(0.1)


def init_encoder_priors(encoder: nn.Module) -> None:
    """Apply engineer-default initial biases to MixEncoder param heads (v6.1).

    Looks up heads by attribute name; missing heads are skipped. Trim head is
    intentionally NOT touched — zero-init means it predicts 0 dB at step 0,
    which keeps fine-tunes stable. There are no bypass heads in v6.1.
    """
    from training.data import STRIP_PARAM_KEYS, BUS_PARAM_KEYS

    strip_head = getattr(encoder, "head_track", None)
    if strip_head is not None:
        params_head = getattr(strip_head, "params", None)
        if params_head is not None:
            init_strip_head(params_head, STRIP_PARAM_KEYS)

    if hasattr(encoder, "head_bus"):
        init_bus_head(encoder.head_bus, BUS_PARAM_KEYS)


def strip_default_norm() -> list[float]:
    """Normalized [0,1] engineer-default value for each STRIP_PARAM_KEYS entry.

    Used as the target of the identity-prior penalty in training. For keys not
    in STRIP_DEFAULTS, falls back to 0.5 (mid-range)."""
    from training.data import STRIP_PARAM_KEYS
    return [normalize(k, STRIP_DEFAULTS.get(k, 0.5)) for k in STRIP_PARAM_KEYS]


def bus_default_norm() -> list[float]:
    """Normalized [0,1] engineer-default value for each BUS_PARAM_KEYS entry."""
    from training.data import BUS_PARAM_KEYS
    out = []
    for k in BUS_PARAM_KEYS:
        lookup = k[len("bus_"):] if k.startswith("bus_") else k
        out.append(normalize(k, BUS_DEFAULTS.get(lookup, 0.5)))
    return out


__all__ = [
    "STRIP_DEFAULTS", "BUS_DEFAULTS",
    "init_strip_head", "init_bus_head", "init_encoder_priors",
    "strip_default_norm", "bus_default_norm",
]
