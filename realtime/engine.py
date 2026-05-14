"""Realtime audio engine.

V0: passthrough mix. The audio callback advances a playhead through the
session's stem tensor, slices the current block from every track, sums
them, applies a master gain, and writes the result to the output buffer.
No DSP, no model — just verifies that sounddevice can play the stems
without dropouts at the chosen block size.

In later versions:
  - the slice will also be pushed to per-track ring buffers (V1)
  - the slice will be processed through a per-track DSP strip + master
    bus before summing (V2)
  - the DSP params will be read from a smoother that's being driven by
    the inference thread (V3)

The audio callback must stay realtime-safe: no allocations on the hot
path, no Python-level blocking calls, no print, no logging beyond the
sounddevice-status warning. Everything user-facing happens on the main
thread (cli.py).
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

from .session import Session

# sounddevice is imported lazily inside start() — it loads the PortAudio
# shared library at import time, which fails on headless / no-audio
# machines. Deferring lets us still construct, unit-test, and dry-run the
# callback (e.g. on a compute-only box) without an audio backend.


class AudioEngine:
    def __init__(
        self,
        session: Session,
        block_size: int = 512,
        master_gain_db: float = -6.0,
        output_device: Optional[int | str] = None,
        loop: bool = True,
    ) -> None:
        self.session = session
        self.block_size = block_size
        self.master_gain = float(10.0 ** (master_gain_db / 20.0))
        self.output_device = output_device
        self.loop = loop
        self.sample_rate = session.sample_rate

        self._playhead = 0  # in samples
        self._stream = None  # sd.OutputStream once started
        self._lock = threading.Lock()  # protects _playhead during user seek

        # Preallocate the per-block scratch buffer so the callback doesn't
        # allocate on the hot path. (sum() into it; sounddevice copies into
        # outdata.)
        self._mix_buf = np.zeros((2, block_size), dtype=np.float32)

    @property
    def playhead_s(self) -> float:
        return self._playhead / self.sample_rate

    @property
    def is_running(self) -> bool:
        return self._stream is not None and self._stream.active  # type: ignore[union-attr]

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            # sounddevice flags XRUNs, late callbacks, etc. here. We just
            # log; the audio thread keeps going.
            logging.warning("audio status: %s", status)

        stems = self.session.stems          # (N, 2, T)
        T = stems.shape[-1]
        pos = self._playhead

        # Slice the next `frames` samples from every track. If we cross the
        # end of the session, either wrap (loop=True) or zero-fill the tail
        # and freeze the playhead.
        end = pos + frames
        if end <= T:
            block = stems[:, :, pos:end]
            self._playhead = end
        else:
            head_len = T - pos
            tail_len = frames - head_len
            if self.loop:
                block = np.concatenate(
                    [stems[:, :, pos:T], stems[:, :, :tail_len]], axis=-1
                )
                self._playhead = tail_len
            else:
                block = np.zeros((stems.shape[0], 2, frames), dtype=np.float32)
                block[:, :, :head_len] = stems[:, :, pos:T]
                self._playhead = T  # done

        # Sum stems → (2, frames), scale, write out. `outdata` is (frames, 2).
        # Use the preallocated scratch buffer when the block size matches
        # (the common case); otherwise (loop=False end-of-session edge)
        # allocate fresh — that path runs at most once per session.
        if block.shape[-1] == self.block_size:
            np.sum(block, axis=0, out=self._mix_buf)
            np.multiply(self._mix_buf, self.master_gain, out=self._mix_buf)
            mix = self._mix_buf
        else:
            mix = block.sum(axis=0) * self.master_gain
        outdata[:] = mix.T

    def start(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd  # lazy import — see top of module
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=2,
            dtype="float32",
            device=self.output_device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
        finally:
            self._stream.close()
            self._stream = None

    def seek(self, time_s: float) -> None:
        T = self.session.stems.shape[-1]
        with self._lock:
            self._playhead = max(0, min(T - 1, int(time_s * self.sample_rate)))
