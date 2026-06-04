"""Per-track circular audio buffer (V1).

Single producer (the audio callback) writes recent samples; single consumer
(the inference thread) snapshots the latest `capacity_samples` window in
chronological order. CPython's GIL makes single-int64 writes atomic, so we
piggyback on that for the write-pointer hand-off rather than pulling in a
mutex on the audio hot path.

Tearing tolerance: the consumer may miss the last few samples written
during its `snapshot()` copy. That's fine — the encoder window is 8 s and
the inference cadence is 250 ms, so a few ms of staleness at the trailing
edge doesn't matter.
"""

from __future__ import annotations

import numpy as np


class RingBuffer:
    """Per-track circular buffer of recent audio.

    Sized to hold one encoder window (default 8 s × 48 kHz = 384 000 samples
    per channel).

    Write side (audio thread): `write(block)` appends `block` samples,
    advances the write pointer. Wraps in-place. No allocations.

    Read side (inference thread): `snapshot()` returns a freshly allocated
    `(n_channels, capacity)` ndarray containing the most recent `capacity`
    samples in chronological order. Safe to call concurrently with `write`.
    """

    def __init__(self, n_channels: int, capacity_samples: int) -> None:
        if n_channels <= 0:
            raise ValueError(f"n_channels must be > 0, got {n_channels}")
        if capacity_samples <= 0:
            raise ValueError(f"capacity_samples must be > 0, got {capacity_samples}")
        self.n_channels = n_channels
        self.capacity = capacity_samples
        self._buf = np.zeros((n_channels, capacity_samples), dtype=np.float32)
        # Single-int64 cells — GIL-atomic single-element writes on CPython.
        self._write_idx = np.zeros(1, dtype=np.int64)        # next write position (0 .. capacity-1)
        self._total_written = np.zeros(1, dtype=np.int64)    # cumulative samples written

    def write(self, block: np.ndarray) -> None:
        """Append `block` (n_channels, n) to the ring. Audio thread."""
        n = block.shape[-1]
        if n == 0:
            return
        cap = self.capacity
        i = int(self._write_idx[0])
        if n >= cap:
            # Block is larger than the ring — keep only the tail.
            self._buf[:] = block[:, n - cap:]
            self._write_idx[0] = 0
        elif i + n <= cap:
            self._buf[:, i:i + n] = block
            self._write_idx[0] = (i + n) % cap
        else:
            head = cap - i
            self._buf[:, i:] = block[:, :head]
            self._buf[:, :n - head] = block[:, head:]
            self._write_idx[0] = n - head
        self._total_written[0] += n

    def snapshot(self) -> np.ndarray:
        """Return the most recent `capacity` samples in chronological order.

        Allocates a new ndarray on each call. Inference thread.
        """
        i = int(self._write_idx[0])
        # The slot at index `i` is the OLDEST sample (next to be overwritten),
        # so unwrap with `[i:] + [:i]`.
        return np.concatenate([self._buf[:, i:], self._buf[:, :i]], axis=1)

    @property
    def is_warm(self) -> bool:
        """True once `capacity` samples have been written (so `snapshot()` is
        fully real audio, not the initial zero-fill)."""
        return int(self._total_written[0]) >= self.capacity

    @property
    def fill_fraction(self) -> float:
        """[0, 1] — useful for diagnostics during warm-up."""
        return min(1.0, int(self._total_written[0]) / self.capacity)
