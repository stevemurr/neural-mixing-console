"""Two-tier parameter sampler with bypass logic.

Per spec §6:
  - With prob 0.8, draw from the "plausible" distribution.
  - With prob 0.2, draw from the "extreme" full-range distribution.
  - Bypass probabilities apply on top, independently per band/effect.

Each example's RNG is derived from `Hash(master_seed, example_id)`. This module
exposes per-block samplers — `sample_eq`, `sample_comp`, etc. — plus helpers
for the chain-level draws used by stages 2 / 2.5 / 2.7.

All samplers take a numpy.random.Generator instance (don't use the global RNG).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Optional

import numpy as np

from .clipper import ClipParams
from .compressor import CompParams
from .delay import DelayParams
from .rbj_biquads import EQParams, EQBypass, BusEQParams, BusEQBypass
from .reverb_fdn import ReverbParams
from .saturation import SatParams


# ---------- RNG helpers ----------

def make_rng(master_seed: int, example_id: str, sub_index: int = 0) -> np.random.Generator:
    """Derive a deterministic RNG from (master_seed, example_id[, sub_index]).

    Same inputs always produce the same RNG state. Used so stage 2.7's per-track
    sub-RNGs can be reproduced from the example_id alone.
    """
    h = sha256()
    h.update(master_seed.to_bytes(8, "little", signed=False))
    h.update(example_id.encode("utf-8"))
    h.update(sub_index.to_bytes(8, "little", signed=False))
    seed = int.from_bytes(h.digest()[:16], "little")
    # Cap to numpy's 128-bit SeedSequence input
    return np.random.default_rng(np.random.SeedSequence(seed))


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(10.0 ** rng.uniform(np.log10(lo), np.log10(hi)))


def _clipped_normal(rng: np.random.Generator, mu: float, sigma: float, lo: float, hi: float) -> float:
    return float(np.clip(rng.normal(mu, sigma), lo, hi))


# ---------- Tier selection ----------

PLAUSIBLE_PROB = 0.8


def pick_tier(rng: np.random.Generator) -> str:
    return "plausible" if rng.random() < PLAUSIBLE_PROB else "extreme"


# ---------- Track-aware bias (plausible-tier only) ----------

# Per-instrument overrides for high-impact params. Each entry is
# `(mean, sigma, lo, hi)` for clipped-normal sampling. Applied at the
# plausible-tier level only — extreme tier stays uniform across full ranges
# so the encoder still sees the long tail. Missing params/instruments fall
# back to uniform sampling at plausible-tier ranges.
#
# Why this exists: synthetic data alone doesn't have track-conditional
# structure (params are sampled independently of audio). MERT semantic
# embeddings need *something* to correlate with at training time. Biasing
# the synthetic distribution per instrument bakes in that structure so MERT
# learns "kick-like embedding → kick-like params."

_INSTRUMENT_BIASES: dict[str, dict[str, tuple[float, float, float, float]]] = {
    # Drums
    "kick": {
        "hpf_freq":      (25.0,  10.0,  20.0,  60.0),
        "comp_ratio":    (4.0,   1.0,   2.0,   8.0),
        "comp_attack":   (5.0,   2.0,   1.0,   15.0),
        "comp_release":  (80.0,  30.0,  30.0,  200.0),
        "pan":           (0.0,   0.05,  -0.2,  0.2),
        "drive_db":      (3.0,   2.0,   0.0,   8.0),
        "p1_freq":       (60.0,  15.0,  40.0,  120.0),    # boost low-end punch
        "p1_gain":       (2.0,   1.5,   -2.0,  6.0),
        "p2_freq":       (4000.0, 1500.0, 2000.0, 7000.0),  # click
        "p2_gain":       (2.0,   1.5,   -2.0,  6.0),
    },
    "snare": {
        "hpf_freq":      (60.0,  20.0,  30.0,  150.0),
        "comp_ratio":    (4.0,   1.0,   2.0,   8.0),
        "comp_attack":   (3.0,   2.0,   0.5,   10.0),
        "comp_release":  (100.0, 40.0,  40.0,  250.0),
        "pan":           (0.0,   0.1,   -0.3,  0.3),
        "drive_db":      (2.0,   2.0,   0.0,   8.0),
    },
    "hat": {
        "hpf_freq":      (200.0, 80.0,  100.0, 500.0),
        "pan":           (-0.3,  0.2,   -0.7,  0.7),
        "comp_ratio":    (2.0,   0.5,   1.0,   3.5),
    },
    "cymbal": {
        "hpf_freq":      (200.0, 80.0,  100.0, 500.0),
        "pan":           (0.0,   0.4,   -0.7,  0.7),
        "hs_freq":       (10000.0, 2000.0, 7000.0, 14000.0),
        "hs_gain":       (1.5,   1.5,   -2.0,  4.0),
    },
    "tom": {
        "hpf_freq":      (50.0,  20.0,  25.0,  100.0),
        "pan":           (0.0,   0.4,   -0.6,  0.6),
        "comp_ratio":    (3.0,   1.0,   1.5,   5.0),
    },
    "drum_overhead": {
        "hpf_freq":      (150.0, 50.0,  60.0,  300.0),
        "pan":           (0.0,   0.5,   -0.8,  0.8),
        "hs_freq":       (10000.0, 2000.0, 7000.0, 14000.0),
        "hs_gain":       (1.5,   1.5,   -2.0,  4.0),
    },
    "drum_other": {
        "hpf_freq":      (60.0,  30.0,  20.0,  200.0),
        "pan":           (0.0,   0.4,   -0.7,  0.7),
        "comp_ratio":    (3.0,   1.0,   1.5,   6.0),
    },

    # Bass
    "bass_electric": {
        "hpf_freq":      (35.0,  10.0,  20.0,  60.0),
        "comp_ratio":    (3.0,   1.0,   1.5,   6.0),
        "comp_attack":   (10.0,  4.0,   3.0,   30.0),
        "comp_release":  (100.0, 40.0,  50.0,  300.0),
        "pan":           (0.0,   0.05,  -0.2,  0.2),
        "drive_db":      (3.0,   2.0,   0.0,   8.0),
    },
    "bass_synth": {
        "hpf_freq":      (35.0,  10.0,  20.0,  60.0),
        "pan":           (0.0,   0.1,   -0.3,  0.3),
    },

    # Guitar
    "guitar_acoustic": {
        "hpf_freq":      (90.0,  30.0,  50.0,  200.0),
        "pan":           (-0.2,  0.4,   -0.7,  0.7),
        "comp_ratio":    (2.5,   0.5,   1.5,   4.0),
    },
    "guitar_electric": {
        "hpf_freq":      (80.0,  30.0,  40.0,  200.0),
        "pan":           (0.0,   0.5,   -0.8,  0.8),
        "comp_ratio":    (2.5,   0.5,   1.5,   4.0),
        "drive_db":      (2.0,   2.0,   0.0,   6.0),
    },

    # Vocals
    "vocal_lead": {
        "hpf_freq":      (100.0, 30.0,  60.0,  200.0),
        "comp_ratio":    (3.0,   1.0,   2.0,   5.0),
        "comp_attack":   (10.0,  5.0,   3.0,   30.0),
        "comp_release":  (100.0, 40.0,  40.0,  300.0),
        "pan":           (0.0,   0.05,  -0.15, 0.15),
        "p1_freq":       (250.0, 80.0,  150.0, 500.0),    # tame mud
        "p1_gain":       (-2.0,  1.5,   -6.0,  1.0),
        "p2_freq":       (5000.0, 1500.0, 2500.0, 8000.0),  # presence
        "p2_gain":       (1.5,   1.5,   -1.0,  4.0),
        "hs_freq":       (12000.0, 2000.0, 8000.0, 16000.0),  # air
        "hs_gain":       (1.5,   1.5,   -1.0,  4.0),
    },
    "vocal_bg": {
        "hpf_freq":      (120.0, 40.0,  60.0,  300.0),
        "pan":           (0.0,   0.5,   -0.8,  0.8),
        "comp_ratio":    (2.5,   0.5,   1.5,   4.0),
    },
    "vocal_other": {
        "hpf_freq":      (100.0, 40.0,  50.0,  250.0),
        "pan":           (0.0,   0.4,   -0.7,  0.7),
    },

    # Keys / piano
    "piano": {
        "hpf_freq":      (60.0,  30.0,  20.0,  150.0),
        "pan":           (0.0,   0.4,   -0.7,  0.7),
        "comp_ratio":    (2.5,   0.5,   1.5,   4.0),
    },
    "keys": {
        "hpf_freq":      (60.0,  30.0,  30.0,  150.0),
        "pan":           (0.0,   0.5,   -0.8,  0.8),
    },
    "synth": {
        "hpf_freq":      (40.0,  20.0,  20.0,  150.0),
        "pan":           (0.0,   0.5,   -0.9,  0.9),
    },

    # Wind / strings
    "brass":     {"pan": (0.0,  0.4,  -0.7,  0.7), "hpf_freq": (80.0, 30.0, 40.0, 200.0)},
    "strings":   {"pan": (0.0,  0.4,  -0.7,  0.7), "hpf_freq": (80.0, 30.0, 40.0, 200.0)},
    "woodwind":  {"pan": (0.0,  0.4,  -0.7,  0.7), "hpf_freq": (80.0, 30.0, 40.0, 200.0)},

    # FX / loops / fills (mostly leave default)
    "fx":        {"pan": (0.0,  0.5,  -0.9,  0.9)},
}


def _instrument_override(instrument_label: Optional[str], param: str
                         ) -> tuple[float, float, float, float] | None:
    if not instrument_label:
        return None
    return _INSTRUMENT_BIASES.get(instrument_label, {}).get(param)


# ---------- Per-block samplers ----------

def sample_gain_db(rng: np.random.Generator, tier: str) -> float:
    if tier == "plausible":
        return _clipped_normal(rng, 0.0, 4.0, -12.0, 6.0)
    return float(rng.uniform(-24.0, 12.0))


def sample_sat(rng: np.random.Generator, tier: str,
               instrument_label: Optional[str] = None) -> tuple[SatParams, bool]:
    """Returns (params, bypass_flag). Bypass prob 0.20 (spec §3.2).

    `instrument_label`, when provided, may override `drive_db` toward an
    instrument-typical mean for the plausible tier.
    """
    if rng.random() < 0.20:
        return SatParams(drive_db=0.0, bias=0.0, mix=0.0, makeup_db=0.0), True
    if tier == "plausible":
        ovr = _instrument_override(instrument_label, "drive_db")
        drive = _clipped_normal(rng, *ovr) if ovr else float(rng.uniform(0.0, 12.0))
        bias = _clipped_normal(rng, 0.0, 0.05, -0.1, 0.1)
        mix = float(rng.uniform(0.3, 1.0))
        makeup = float(rng.uniform(-6.0, 3.0))
    else:
        drive = float(rng.uniform(0.0, 24.0))
        bias = float(rng.uniform(-0.3, 0.3))
        mix = float(rng.uniform(0.0, 1.0))
        makeup = float(rng.uniform(-12.0, 6.0))
    return SatParams(drive_db=drive, bias=bias, mix=mix, makeup_db=makeup), False


def sample_eq(rng: np.random.Generator, tier: str,
              instrument_label: Optional[str] = None) -> tuple[EQParams, EQBypass]:
    """Returns (params, bypass_flags). Per-band bypass prob 0.15 (spec §3.3).

    Bypass for shelves/peaks zeros their gain; bypass for HPF/LPF sets the flag
    so the runtime can short-circuit (true identity, not just freq=20Hz).

    `instrument_label`, when provided, may bias HPF freq, mid/high band freqs
    and gains toward instrument-typical values at the plausible tier.
    """
    bp = EQBypass(
        hpf=rng.random() < 0.15,
        ls=rng.random() < 0.15,
        p1=rng.random() < 0.15,
        p2=rng.random() < 0.15,
        hs=rng.random() < 0.15,
        lpf=rng.random() < 0.15,
    )

    def _ov(name: str, default_fn):
        ovr = _instrument_override(instrument_label, name) if tier == "plausible" else None
        return _clipped_normal(rng, *ovr) if ovr else default_fn()

    if tier == "plausible":
        hpf_freq = _ov("hpf_freq", lambda: _log_uniform(rng, 30.0, 200.0))
        ls_freq = _log_uniform(rng, 60.0, 500.0)
        ls_gain = 0.0 if bp.ls else _clipped_normal(rng, 0.0, 2.0, -6.0, 6.0)
        ls_q = _log_uniform(rng, 0.3, 1.5)
        p1_freq = _ov("p1_freq", lambda: _log_uniform(rng, 100.0, 2000.0))
        p1_gain = 0.0 if bp.p1 else _ov("p1_gain", lambda: _clipped_normal(rng, 0.0, 2.0, -6.0, 6.0))
        p1_q = _log_uniform(rng, 0.5, 3.0)
        p2_freq = _ov("p2_freq", lambda: _log_uniform(rng, 1000.0, 10000.0))
        p2_gain = 0.0 if bp.p2 else _ov("p2_gain", lambda: _clipped_normal(rng, 0.0, 2.0, -6.0, 6.0))
        p2_q = _log_uniform(rng, 0.5, 3.0)
        hs_freq = _ov("hs_freq", lambda: _log_uniform(rng, 2000.0, 12000.0))
        hs_gain = 0.0 if bp.hs else _ov("hs_gain", lambda: _clipped_normal(rng, 0.0, 2.0, -6.0, 6.0))
        hs_q = _log_uniform(rng, 0.3, 1.5)
        lpf_freq = _log_uniform(rng, 8000.0, 18000.0)
    else:
        hpf_freq = _log_uniform(rng, 20.0, 500.0)
        ls_freq = _log_uniform(rng, 60.0, 500.0)
        ls_gain = 0.0 if bp.ls else float(rng.uniform(-12.0, 12.0))
        ls_q = _log_uniform(rng, 0.3, 1.5)
        p1_freq = _log_uniform(rng, 100.0, 2000.0)
        p1_gain = 0.0 if bp.p1 else float(rng.uniform(-15.0, 15.0))
        p1_q = _log_uniform(rng, 0.3, 10.0)
        p2_freq = _log_uniform(rng, 1000.0, 10000.0)
        p2_gain = 0.0 if bp.p2 else float(rng.uniform(-15.0, 15.0))
        p2_q = _log_uniform(rng, 0.3, 10.0)
        hs_freq = _log_uniform(rng, 2000.0, 12000.0)
        hs_gain = 0.0 if bp.hs else float(rng.uniform(-12.0, 12.0))
        hs_q = _log_uniform(rng, 0.3, 1.5)
        lpf_freq = _log_uniform(rng, 5000.0, 20000.0)

    return EQParams(
        hpf_freq=hpf_freq,
        ls_freq=ls_freq, ls_gain=ls_gain, ls_q=ls_q,
        p1_freq=p1_freq, p1_gain=p1_gain, p1_q=p1_q,
        p2_freq=p2_freq, p2_gain=p2_gain, p2_q=p2_q,
        hs_freq=hs_freq, hs_gain=hs_gain, hs_q=hs_q,
        lpf_freq=lpf_freq,
    ), bp


def sample_comp(rng: np.random.Generator, tier: str,
                instrument_label: Optional[str] = None) -> tuple[CompParams, bool]:
    """Returns (params, bypass). Bypass prob 0.15 -> ratio = 1 (spec §3.4).

    `instrument_label`, when provided, may bias `comp_ratio`, `comp_attack`,
    `comp_release` toward instrument-typical values at the plausible tier.
    """
    bypass = rng.random() < 0.15

    if tier == "plausible":
        threshold = float(rng.uniform(-40.0, -6.0))
        ratio_ov = _instrument_override(instrument_label, "comp_ratio")
        ratio = _clipped_normal(rng, *ratio_ov) if ratio_ov else _log_uniform(rng, 1.5, 8.0)
        attack_ov = _instrument_override(instrument_label, "comp_attack")
        attack = _clipped_normal(rng, *attack_ov) if attack_ov else _log_uniform(rng, 1.0, 30.0)
        release_ov = _instrument_override(instrument_label, "comp_release")
        release = _clipped_normal(rng, *release_ov) if release_ov else _log_uniform(rng, 50.0, 500.0)
        knee = float(rng.uniform(3.0, 9.0))
        makeup = float(rng.uniform(0.0, 12.0))
    else:
        threshold = float(rng.uniform(-60.0, 0.0))
        ratio = _log_uniform(rng, 1.0, 20.0)
        attack = _log_uniform(rng, 0.5, 100.0)
        release = _log_uniform(rng, 10.0, 1000.0)
        knee = float(rng.uniform(0.0, 12.0))
        makeup = float(rng.uniform(0.0, 24.0))

    if bypass:
        ratio = 1.0  # other params kept as sampled but no-op

    return CompParams(
        threshold_db=threshold, ratio=ratio,
        attack_ms=attack, release_ms=release,
        knee_db=knee, makeup_db=makeup,
    ), bypass


def sample_clip(rng: np.random.Generator, tier: str) -> tuple[ClipParams, bool]:
    """Returns (params, bypass_flag). Bypass prob 0.50 — clipping is a
    targeted tool used on maybe 30–50% of tracks in modern productions.
    """
    if rng.random() < 0.50:
        return ClipParams(drive_db=0.0, shape=0.5, ceiling_db=0.0, mix=0.0), True
    if tier == "plausible":
        # Most uses are 1–8 dB drive, mid-to-hard shape, near-zero ceiling.
        drive = float(rng.uniform(0.0, 8.0))
        shape = float(rng.uniform(0.3, 0.9))
        ceiling = _clipped_normal(rng, -1.0, 1.0, -3.0, 0.0)
        mix = float(rng.uniform(0.7, 1.0))
    else:
        drive = float(rng.uniform(0.0, 24.0))
        shape = float(rng.uniform(0.0, 1.0))
        ceiling = float(rng.uniform(-6.0, 0.0))
        mix = float(rng.uniform(0.0, 1.0))
    return ClipParams(drive_db=drive, shape=shape, ceiling_db=ceiling, mix=mix), False


def sample_bus_clip(rng: np.random.Generator, tier: str) -> tuple[ClipParams, bool]:
    """Master-bus clipper sampler. Narrower distributions (subtle bus clip;
    1–3 dB drive typical, ceiling near 0). Bypass prob 0.40.
    """
    if rng.random() < 0.40:
        return ClipParams(drive_db=0.0, shape=0.5, ceiling_db=0.0, mix=0.0), True
    if tier == "plausible":
        drive = float(rng.uniform(0.0, 4.0))
        shape = float(rng.uniform(0.5, 1.0))
        ceiling = _clipped_normal(rng, -0.5, 0.5, -2.0, 0.0)
        mix = float(rng.uniform(0.8, 1.0))
    else:
        drive = float(rng.uniform(0.0, 12.0))
        shape = float(rng.uniform(0.0, 1.0))
        ceiling = float(rng.uniform(-3.0, 0.0))
        mix = float(rng.uniform(0.0, 1.0))
    return ClipParams(drive_db=drive, shape=shape, ceiling_db=ceiling, mix=mix), False


def sample_pan(rng: np.random.Generator, tier: str,
               instrument_label: Optional[str] = None) -> float:
    """`instrument_label` biases pan distribution: kicks/snares/leads centered;
    others spread."""
    if tier == "plausible":
        ovr = _instrument_override(instrument_label, "pan")
        if ovr:
            return _clipped_normal(rng, *ovr)
        return _clipped_normal(rng, 0.0, 0.3, -0.7, 0.7)
    return float(rng.uniform(-1.0, 1.0))


def sample_delay(rng: np.random.Generator, tier: str) -> DelayParams:
    # Continuous delay time in quarter-note units; log-uniform over the same
    # span the prior 6-way categorical covered (1/16 = 0.0625 to 1/2 = 2.0).
    delay_time_qn = _log_uniform(rng, 0.0625, 2.0)
    if tier == "plausible":
        feedback = float(rng.uniform(0.0, 0.6))
        damping = float(rng.uniform(0.2, 0.7))
        wet_db = float(rng.uniform(-18.0, 0.0))
    else:
        feedback = float(rng.uniform(0.0, 0.85))
        damping = float(rng.uniform(0.0, 1.0))
        wet_db = float(rng.uniform(-24.0, 6.0))
    ping_pong = bool(rng.random() < 0.3)
    return DelayParams(
        delay_time_qn=delay_time_qn, feedback=feedback,
        hf_damping=damping, ping_pong=ping_pong, wet_db=wet_db,
    )


def sample_reverb(rng: np.random.Generator, tier: str) -> ReverbParams:
    if tier == "plausible":
        room_size = float(rng.uniform(0.2, 0.8))
        decay = _log_uniform(rng, 0.6, 3.5)
        damping = float(rng.uniform(0.3, 0.8))
        predelay = float(rng.uniform(0.0, 40.0))
        wet_db = float(rng.uniform(-18.0, 0.0))
    else:
        room_size = float(rng.uniform(0.0, 1.0))
        decay = _log_uniform(rng, 0.3, 8.0)
        damping = float(rng.uniform(0.0, 1.0))
        predelay = float(rng.uniform(0.0, 100.0))
        wet_db = float(rng.uniform(-24.0, 6.0))
    return ReverbParams(
        room_size=room_size, decay_time_s=decay,
        hf_damping=damping, predelay_ms=predelay,
        wet_db=wet_db,
    )


# ---------- Bus-block samplers (stage 2.5) ----------

def sample_bus_eq(rng: np.random.Generator, tier: str) -> tuple[BusEQParams, BusEQBypass]:
    """Bus EQ: tighter plausible distribution per spec §5.1."""
    bp = BusEQBypass(
        low_boost=rng.random() < 0.10,
        low_attn=rng.random() < 0.10,
        mid=rng.random() < 0.10,
        air=rng.random() < 0.10,
    )

    if tier == "plausible":
        low_boost_freq = _log_uniform(rng, 30.0, 100.0)
        low_boost_gain = 0.0 if bp.low_boost else float(rng.uniform(0.0, 1.5))
        low_attn_freq = _log_uniform(rng, 40.0, 200.0)
        low_attn_gain = 0.0 if bp.low_attn else float(rng.uniform(-1.5, 0.0))
        mid_freq = _log_uniform(rng, 300.0, 3000.0)
        mid_gain = 0.0 if bp.mid else float(rng.uniform(-1.0, 1.0))
        mid_q = _log_uniform(rng, 0.5, 2.0)
        air_freq = _log_uniform(rng, 8000.0, 16000.0)
        air_gain = 0.0 if bp.air else float(rng.uniform(0.0, 1.5))
    else:
        low_boost_freq = _log_uniform(rng, 30.0, 100.0)
        low_boost_gain = 0.0 if bp.low_boost else float(rng.uniform(0.0, 4.0))
        low_attn_freq = _log_uniform(rng, 40.0, 200.0)
        low_attn_gain = 0.0 if bp.low_attn else float(rng.uniform(-4.0, 0.0))
        mid_freq = _log_uniform(rng, 300.0, 3000.0)
        mid_gain = 0.0 if bp.mid else float(rng.uniform(-2.0, 2.0))
        mid_q = _log_uniform(rng, 0.5, 2.0)
        air_freq = _log_uniform(rng, 8000.0, 16000.0)
        air_gain = 0.0 if bp.air else float(rng.uniform(0.0, 4.0))

    return BusEQParams(
        low_boost_freq=low_boost_freq, low_boost_gain=low_boost_gain,
        low_attn_freq=low_attn_freq, low_attn_gain=low_attn_gain,
        mid_freq=mid_freq, mid_gain=mid_gain, mid_q=mid_q,
        air_freq=air_freq, air_gain=air_gain,
    ), bp


def sample_bus_comp(rng: np.random.Generator, tier: str) -> tuple[CompParams, bool]:
    """Bus compressor: slow, low-ratio glue (spec §5.2)."""
    bypass = rng.random() < 0.10
    if tier == "plausible":
        threshold = float(rng.uniform(-12.0, -2.0))
        ratio = float(rng.uniform(1.2, 2.5))
        attack = float(rng.uniform(20.0, 60.0))
        release = float(rng.uniform(100.0, 400.0))
        knee = float(rng.uniform(3.0, 9.0))
        makeup = float(rng.uniform(0.0, 6.0))
    else:
        threshold = float(rng.uniform(-24.0, 0.0))
        ratio = _log_uniform(rng, 1.0, 4.0)
        attack = _log_uniform(rng, 10.0, 100.0)
        release = _log_uniform(rng, 50.0, 1000.0)
        knee = float(rng.uniform(0.0, 12.0))
        makeup = float(rng.uniform(0.0, 12.0))
    if bypass:
        ratio = 1.0
    return CompParams(
        threshold_db=threshold, ratio=ratio,
        attack_ms=attack, release_ms=release,
        knee_db=knee, makeup_db=makeup,
    ), bypass


# ---------- Stage 2.7 send-level samplers ----------

def sample_send_db(
    rng: np.random.Generator,
    tier: str,
    *,
    plausible_mu: float,
    plausible_sigma: float,
    bypass_prob: float,
) -> tuple[float, bool]:
    """Returns (send_db, bypass). Per spec §10.1 table."""
    if rng.random() < bypass_prob:
        return -np.inf, True
    if tier == "plausible":
        v = _clipped_normal(rng, plausible_mu, plausible_sigma, -60.0, 0.0)
    else:
        v = float(rng.uniform(-60.0, 0.0))
    return v, False


def sample_delay_send(rng: np.random.Generator, tier: str) -> tuple[float, bool]:
    return sample_send_db(rng, tier, plausible_mu=-20.0, plausible_sigma=8.0, bypass_prob=0.70)


def sample_reverb_send(rng: np.random.Generator, tier: str) -> tuple[float, bool]:
    return sample_send_db(rng, tier, plausible_mu=-15.0, plausible_sigma=6.0, bypass_prob=0.40)


__all__ = [
    "make_rng", "pick_tier",
    "sample_gain_db", "sample_sat", "sample_eq", "sample_comp", "sample_pan",
    "sample_clip", "sample_bus_clip",
    "sample_delay", "sample_reverb",
    "sample_bus_eq", "sample_bus_comp",
    "sample_delay_send", "sample_reverb_send",
]
