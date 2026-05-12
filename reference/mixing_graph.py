"""Full multi-track mixing graph composer (stages 2.7 and 3 forward render).

Topology per spec §10:

    For each track k (mono or stereo):
        post_strip_k = strip_k(track_k)        # gain -> sat -> EQ -> comp -> pan

    delay_aux_in  = sum_k (post_strip_k * 10^(delay_send_db_k / 20))     # zero if bypassed
    delay_wet     = Delay(delay_aux_in)
    reverb_aux_in = sum_k (post_strip_k * 10^(reverb_send_db_k / 20)) + delay_wet
    reverb_wet    = Reverb(reverb_aux_in)

    bus_input = sum_k post_strip_k + delay_wet + reverb_wet
    mix       = MasterBus(bus_input)

The graph runs at a single sample rate and a single per-example BPM (stage 2.7
samples it; stage 3 reads it from MedleyDB metadata or beat-tracks it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .channel_strip import StripParams, StripBypass, apply_strip
from .delay import DelayParams, apply_delay
from .master_bus import BusParams, BusBypass, apply_master_bus
from .reverb_fdn import ReverbParams, apply_reverb


@dataclass
class TrackInputs:
    """Per-track input + send metadata. `audio` is mono (N,) or stereo (2, N)."""
    audio: np.ndarray
    strip: StripParams
    strip_bypass: StripBypass = field(default_factory=StripBypass)
    delay_send_db: float = -np.inf  # -inf == bypassed
    reverb_send_db: float = -np.inf


@dataclass
class GraphParams:
    bpm: float
    delay: DelayParams = field(default_factory=DelayParams)
    reverb: ReverbParams = field(default_factory=ReverbParams)
    bus: BusParams = field(default_factory=BusParams)
    delay_bypass: bool = False  # if True, delay returns silence regardless of sends
    reverb_bypass: bool = False
    bus_bypass: BusBypass = field(default_factory=BusBypass)


def _send_gain(send_db: float) -> float:
    if not np.isfinite(send_db) or send_db <= -120.0:
        return 0.0
    return float(10.0 ** (send_db / 20.0))


def _zeros_stereo(n: int) -> np.ndarray:
    return np.zeros((2, n), dtype=np.float64)


def render_graph(
    tracks: List[TrackInputs],
    fs: float,
    g: GraphParams,
) -> np.ndarray:
    """Render the full mix. Returns stereo (2, N).

    All tracks must have the same length (in samples) along their last axis.
    """
    if not tracks:
        raise ValueError("render_graph: at least one track required")

    n = tracks[0].audio.shape[-1]
    for t in tracks:
        if t.audio.shape[-1] != n:
            raise ValueError("all tracks must have identical sample length")

    # 1. Per-track strips
    post_strips: list[np.ndarray] = []
    for t in tracks:
        post = apply_strip(t.audio, fs, t.strip, t.strip_bypass)  # (2, N)
        post_strips.append(post.astype(np.float64, copy=False))

    # 2. Build send-bus inputs
    delay_in = _zeros_stereo(n)
    reverb_in = _zeros_stereo(n)
    for t, post in zip(tracks, post_strips):
        gd = _send_gain(t.delay_send_db)
        gr = _send_gain(t.reverb_send_db)
        if gd > 0.0:
            delay_in += post * gd
        if gr > 0.0:
            reverb_in += post * gr

    # 3. Delay
    if g.delay_bypass:
        delay_wet = _zeros_stereo(n)
    else:
        delay_wet = apply_delay(delay_in, fs, g.bpm, g.delay)
        # apply wet_db scaling (delay sub-mix gain on the bus return)
        delay_wet = delay_wet * _send_gain(g.delay.wet_db)

    # 4. Reverb (delay tail feeds reverb per spec §4)
    if g.reverb_bypass:
        reverb_wet = _zeros_stereo(n)
    else:
        rev_input = reverb_in + delay_wet
        reverb_wet = apply_reverb(rev_input, fs, g.reverb)
        reverb_wet = reverb_wet * _send_gain(g.reverb.wet_db)

    # 5. Sum to bus
    bus_in = _zeros_stereo(n)
    for post in post_strips:
        bus_in += post
    bus_in += delay_wet
    bus_in += reverb_wet

    # 6. Master bus
    mix = apply_master_bus(bus_in, fs, g.bus, g.bus_bypass)
    return mix


__all__ = ["TrackInputs", "GraphParams", "render_graph"]
