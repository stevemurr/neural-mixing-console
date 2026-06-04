"""InferenceScheduler (V1): background thread that drives the encoder
at a fixed cadence and hands new parameters to a user callback.

Behavior:
  - Sleeps until the next tick (`cadence_ms` after the previous one).
  - Skips the tick if any ring buffer isn't warm yet (still filling).
  - Calls EncoderWrapper.predict() and forwards the result to
    `on_new_params(...)`. In V1 that callback prints; in V3 it'll be
    smoother.set_target.
  - Drop-if-busy: if predict() takes longer than the cadence, the next
    tick fires immediately (not queued) — we always want the most recent
    8-s snapshot, not a backlog of stale ones.

Stats exposed for status display: `last_predict_ms`, `last_predict_ema_ms`,
`total_ticks`, `cold_ticks`, `late_ticks`, `failed_ticks`.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np
import torch

from .inference import EncoderWrapper
from .ring_buffer import RingBuffer


class InferenceScheduler:
    def __init__(
        self,
        encoder: EncoderWrapper,
        ring_buffers: list[RingBuffer],
        mert: np.ndarray | None,
        on_new_params: Callable[[dict], None],
        *,
        cadence_ms: float = 250.0,
        name: str = "realtime-inference",
    ) -> None:
        if not ring_buffers:
            raise ValueError("ring_buffers is empty")
        self.encoder = encoder
        self.ring_buffers = list(ring_buffers)
        self.on_new_params = on_new_params
        self.cadence_s = float(cadence_ms) / 1000.0
        self._name = name

        n_tracks = len(self.ring_buffers)
        if n_tracks > encoder.n_max:
            raise ValueError(
                f"got {n_tracks} ring buffers but encoder n_max={encoder.n_max}"
            )

        # Per-tick stats.
        self.total_ticks = 0
        self.cold_ticks = 0       # skipped because buffers not warm
        self.late_ticks = 0       # predict took longer than the cadence
        self.failed_ticks = 0
        self.last_predict_ms = 0.0
        self.last_predict_ema_ms = 0.0

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Pre-allocate the padded scratch tensors used by every tick. Sized
        # to (1, n_max, 2, capacity). Capacity is taken from the first ring
        # buffer (all are sized identically by construction).
        self._n_active = n_tracks
        self._capacity = self.ring_buffers[0].capacity
        for rb in self.ring_buffers[1:]:
            if rb.capacity != self._capacity:
                raise ValueError("all ring buffers must have the same capacity")
        self._tracks_buf = torch.zeros(
            (1, encoder.n_max, 2, self._capacity), dtype=torch.float32
        )
        self._mask_buf = torch.zeros((1, encoder.n_max), dtype=torch.bool)
        self._mask_buf[0, :n_tracks] = True

        # Pad MERT to (1, n_max, mert_dim). mert may be None when the encoder
        # was built with mert_dim=0; pass None through in that case.
        if mert is not None and encoder.mert_dim > 0:
            if mert.shape != (n_tracks, encoder.mert_dim):
                raise ValueError(
                    f"mert has shape {mert.shape}, expected ({n_tracks}, {encoder.mert_dim})"
                )
            self._mert_buf = torch.zeros(
                (1, encoder.n_max, encoder.mert_dim), dtype=torch.float32
            )
            self._mert_buf[0, :n_tracks] = torch.from_numpy(mert.astype(np.float32, copy=False))
        else:
            self._mert_buf = None

    # ------- lifecycle -------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------- inner loop -------

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            sleep_for = next_tick - now
            if sleep_for > 0:
                if self._stop.wait(timeout=sleep_for):
                    return

            self.total_ticks += 1

            # Cold ring buffers: skip silently (warming up is normal at session start).
            if not all(rb.is_warm for rb in self.ring_buffers):
                self.cold_ticks += 1
            else:
                t0 = time.monotonic()
                try:
                    self._tick()
                except Exception:
                    self.failed_ticks += 1
                    logging.exception("inference tick failed")
                else:
                    elapsed_ms = (time.monotonic() - t0) * 1000.0
                    self.last_predict_ms = elapsed_ms
                    if self.last_predict_ema_ms == 0.0:
                        self.last_predict_ema_ms = elapsed_ms
                    else:
                        # 0.2 weight on the new sample — ~5-tick half-life.
                        self.last_predict_ema_ms = (
                            0.2 * elapsed_ms + 0.8 * self.last_predict_ema_ms
                        )
                    if elapsed_ms > self.cadence_s * 1000.0:
                        self.late_ticks += 1

            # Drop-if-busy: if we're already past the next tick (predict ran
            # over the cadence), fast-forward `next_tick` past `now` instead
            # of queueing every missed slot.
            next_tick += self.cadence_s
            now = time.monotonic()
            if next_tick < now:
                # Jump forward by an integer number of cadences so the phase
                # stays predictable (rather than drifting to "right now").
                missed = int((now - next_tick) // self.cadence_s) + 1
                next_tick += missed * self.cadence_s

    def _tick(self) -> None:
        for i, rb in enumerate(self.ring_buffers):
            snap = rb.snapshot()                                 # (2, capacity)
            self._tracks_buf[0, i].copy_(torch.from_numpy(snap))

        params = self.encoder.predict(
            self._tracks_buf, self._mask_buf, self._mert_buf,
        )
        self.on_new_params(params)
