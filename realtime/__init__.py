"""Realtime mixing console prototype.

Loads a folder of stem audio files and plays them back through the
trained mixing model (eventually). V0 is just a stems-summing playback
engine — no DSP, no model — to verify the audio-I/O path on this machine.

Modules grow in subsequent versions:
  V0: session, engine, cli                 (this commit)
  V1: + ring_buffer, inference, scheduler  (model in the loop, params logged only)
  V2: + dsp                                (per-track strip + master bus DSP applied)
  V3: + smoother                           (one-pole param ramping)
  V4: + minimal status / GUI hooks

Run:
    uv run python -m realtime.cli --stems-dir source_audio/cambridge-mt/<SESSION>/<SESSION>
"""

from .session import Session
from .engine import AudioEngine

__all__ = ["Session", "AudioEngine"]
