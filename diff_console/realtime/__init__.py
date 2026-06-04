"""Realtime mixing console prototype.

Loads a folder of stem audio files and plays them back through the
trained mixing model (eventually). V0 was passthrough; V1 adds the
model in the loop (ring buffers + scheduler) without yet applying any
DSP — predictions are printed each tick to verify cadence + stability.

Module growth:
  V0: session, engine, cli                 (shipped 7f4d688)
  V1: + ring_buffer, inference, scheduler  (this commit)
  V2: + dsp                                (per-track strip + master bus DSP applied)
  V3: + smoother                           (one-pole param ramping)
  V4: + minimal status / GUI hooks

Run:
    # V0 — passthrough only
    uv run python -m realtime.cli --stems-dir source_audio/cambridge-mt/<SESSION>/<SESSION>

    # V1 — model in the loop, predictions printed
    uv run python -m realtime.cli --stems-dir ... \\
        --checkpoint dmc-data/checkpoints/v6.2-round11_2/mix_encoder_best.pt
"""

from .engine import AudioEngine
from .ring_buffer import RingBuffer
from .session import Session

__all__ = ["AudioEngine", "RingBuffer", "Session"]
