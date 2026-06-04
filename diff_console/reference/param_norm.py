"""Bidirectional normalization between raw param values and the [0, 1] space
the encoder predicts in.

Per spec §4.1 / §6.1:
  - Linear-range params: linear map [min, max] -> [0, 1].
  - Log-range params: log10 map [min, max] -> [0, 1].
  - Categorical params: one-hot encoded in the model output (no continuous norm).

Each param is registered in the PARAM_RANGES table below. `normalize(name, value)`
and `denormalize(name, value)` round-trip through float64.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


_LIN = "linear"
_LOG = "log"

# name -> (min, max, scale). Stage 3 v6 schema only.
PARAM_RANGES: dict[str, tuple[float, float, str]] = {
    # Gain
    "gain_db":         (-24.0, +12.0, _LIN),

    # Per-track clipper (in v6 schema even though the bypass head usually
    # learns to skip it; renderer needs the ranges to denormalize).
    "clip_drive_db":   (0.0,   +24.0, _LIN),
    "clip_shape":      (0.0,   1.0,   _LIN),
    "clip_ceiling_db": (-6.0,  0.0,   _LIN),
    "clip_mix":        (0.0,   1.0,   _LIN),

    # 6-band EQ.
    # v6.2: high-frequency band ranges separated so HPF / low-shelf / two bells
    # / high-shelf / LPF don't all overlap in the 5-14 kHz region. In v6/v6.1
    # the encoder exploited that overlap — pinning p2 at +12 dB @ 10 kHz (its
    # range corner) and hs at -10 dB @ 12 kHz simultaneously to synthesize a
    # band-pass shape, a degenerate solution that fed the training oscillation.
    # New layout: p1 = low-mid bell, p2 = high-mid bell (out of "air"),
    # hs = "air" shelf, lpf = top-octave roll-off only.
    "hpf_freq":        (20.0,    500.0,  _LOG),
    "ls_freq":         (60.0,    500.0,  _LOG),
    "ls_gain":         (-12.0,   +12.0,  _LIN),
    "ls_q":            (0.3,     1.5,    _LOG),
    "p1_freq":         (100.0,   2000.0, _LOG),
    "p1_gain":         (-15.0,   +15.0,  _LIN),
    "p1_q":            (0.3,     10.0,   _LOG),
    "p2_freq":         (1000.0,  5000.0, _LOG),
    "p2_gain":         (-15.0,   +15.0,  _LIN),
    "p2_q":            (0.3,     10.0,   _LOG),
    "hs_freq":         (5000.0,  16000.0, _LOG),
    "hs_gain":         (-12.0,   +12.0,  _LIN),
    "hs_q":            (0.3,     1.5,    _LOG),
    "lpf_freq":        (10000.0, 22000.0, _LOG),

    # Compressor (track) — no makeup in v6 (loudness handled by trim head).
    "threshold_db":    (-60.0,   0.0,    _LIN),
    "ratio":           (1.0,     20.0,   _LOG),
    "attack_ms":       (0.5,     100.0,  _LOG),
    "release_ms":      (10.0,    1000.0, _LOG),
    "knee_db":         (0.0,     12.0,   _LIN),
    # Optional alternate parameterization: comp threshold relative to track RMS.
    # Effective threshold = track_rms_db + threshold_offset_db. Range chosen
    # to bracket "below signal" through "well above signal" while keeping the
    # encoder's output bounded. DiffStrip handles either form.
    "threshold_offset_db": (-30.0, +6.0, _LIN),

    # Pan
    "pan":             (-1.0,    1.0,    _LIN),

    # Master bus EQ
    "bus_low_boost_freq": (30.0,  100.0,  _LOG),
    "bus_low_boost_gain": (0.0,   +4.0,   _LIN),
    "bus_low_attn_freq":  (40.0,  200.0,  _LOG),
    "bus_low_attn_gain":  (-4.0,  0.0,    _LIN),
    "bus_mid_freq":       (300.0, 3000.0, _LOG),
    "bus_mid_gain":       (-2.0,  +2.0,   _LIN),
    "bus_mid_q":          (0.5,   2.0,    _LOG),
    "bus_air_freq":       (8000.0, 16000.0, _LOG),
    "bus_air_gain":       (0.0,   +4.0,   _LIN),

    # Master bus compressor — no makeup in v6.
    "bus_threshold_db":   (-24.0, 0.0,    _LIN),
    "bus_ratio":          (1.0,   4.0,    _LOG),
    "bus_attack_ms":      (10.0,  100.0,  _LOG),
    "bus_release_ms":     (50.0,  1000.0, _LOG),
    "bus_knee_db":        (0.0,   12.0,   _LIN),
}


def normalize(name: str, value: float) -> float:
    lo, hi, scale = PARAM_RANGES[name]
    if scale == _LIN:
        x = (value - lo) / (hi - lo)
    elif scale == _LOG:
        x = (np.log10(value) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    else:
        raise ValueError(f"unknown scale {scale!r} for {name}")
    return float(np.clip(x, 0.0, 1.0))


def denormalize(name: str, value: float) -> float:
    lo, hi, scale = PARAM_RANGES[name]
    v = float(np.clip(value, 0.0, 1.0))
    if scale == _LIN:
        return lo + v * (hi - lo)
    if scale == _LOG:
        log_lo = np.log10(lo)
        log_hi = np.log10(hi)
        return float(10.0 ** (log_lo + v * (log_hi - log_lo)))
    raise ValueError(f"unknown scale {scale!r} for {name}")


def normalize_dict(raw: dict) -> dict:
    """Normalize every key in `raw` that exists in PARAM_RANGES; pass others through."""
    return {k: (normalize(k, v) if k in PARAM_RANGES else v) for k, v in raw.items()}


def denormalize_dict(norm: dict) -> dict:
    return {k: (denormalize(k, v) if k in PARAM_RANGES else v) for k, v in norm.items()}


# ---------- "Effective bypass" (params-in-param-space, v6.1+) ----------
#
# v6.1+ has no explicit bypass flags — a processor that's "off" lands at its
# identity parameters. These helpers report whether each processor is
# *effectively* bypassed, by checking the predicted params against identity.
# Purely for human-readable reporting (infer.py params.json, eval.py); the
# renderer ignores them.

def near_norm_edge(name: str, value: float, frac: float = 0.05, end: str = "min") -> bool:
    """True if `value` (physical units) lands within `frac` of the normalized
    [0, 1] range at `end` ("min" or "max"), per PARAM_RANGES."""
    n = normalize(name, value)
    return n <= frac if end == "min" else n >= 1.0 - frac


def strip_effective_bypass(strip: dict, gain_eps_db: float = 0.5) -> dict:
    """Per-processor "effectively bypassed?" booleans for a per-track strip
    param dict (physical units, keys per STRIP_PARAM_KEYS)."""
    g = lambda k, d=0.0: strip.get(k, d)  # noqa: E731
    return {
        "eq_hpf":  near_norm_edge("hpf_freq", g("hpf_freq", 20.0), end="min"),
        "eq_ls":   abs(g("ls_gain")) <= gain_eps_db,
        "eq_p1":   abs(g("p1_gain")) <= gain_eps_db,
        "eq_p2":   abs(g("p2_gain")) <= gain_eps_db,
        "eq_hs":   abs(g("hs_gain")) <= gain_eps_db,
        "eq_lpf":  near_norm_edge("lpf_freq", g("lpf_freq", 22000.0), end="max"),
        "comp":    g("ratio", 1.0) <= 1.1,
        "clip":    g("clip_mix") <= 0.05 or g("clip_drive_db") <= 0.5,
    }


def bus_effective_bypass(bus: dict, gain_eps_db: float = 0.1) -> dict:
    """Per-processor "effectively bypassed?" for a bus param dict. Accepts keys
    with or without the `bus_` prefix."""
    g = lambda k, d=0.0: bus.get(f"bus_{k}", bus.get(k, d))  # noqa: E731
    return {
        "bus_eq_low_boost": g("low_boost_gain") <= gain_eps_db,
        "bus_eq_low_attn":  g("low_attn_gain")  >= -gain_eps_db,
        "bus_eq_mid":       abs(g("mid_gain"))  <= gain_eps_db,
        "bus_eq_air":       g("air_gain")       <= gain_eps_db,
        "bus_comp":         g("ratio", 1.0)     <= 1.1,
    }


__all__ = [
    "PARAM_RANGES", "normalize", "denormalize", "normalize_dict", "denormalize_dict",
    "near_norm_edge", "strip_effective_bypass", "bus_effective_bypass",
]
