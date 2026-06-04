"""8-line Jot-style FDN reverb with Hadamard mixing matrix and per-line damping.

Topology:

    input -> [predelay] -> + ----> y_out (sum of taps)
                            |
              +-------------+--------------+
              |                            |
       [delay_1] ... [delay_8]      (8 parallel delay lines)
              |                            |
              +--->[Hadamard 8x8 mixer]<---+
                            |
                    [per-line damping LPF]
                            |
                    [per-line gain g_i for RT60]
                            |
                            +--> back to delay-line inputs

Output is summed from delay-line outputs with alternating L/R signs to
produce a stereo image. Returns wet only (caller mixes at wet_db on bus).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):  # type: ignore[no-redef]
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def wrap(fn):
            return fn
        return wrap


# Co-prime prime-number delay-line lengths in samples (pre-scaling) at 48 kHz.
# Single shared prime set chosen to span small-room to large-hall when
# combined with the widened room_size scale below. Mirrors Steinmetz / dasp:
# we drop the discrete "room"/"hall" categorical and let continuous
# room_size + decay_time + hf_damping describe the full reverb space.
_PRIMES = np.array([313, 421, 547, 661, 787, 911, 1031, 1163], dtype=np.int64)

# 8x8 Hadamard matrix (orthonormal after scaling by 1/sqrt(8)).
_HADAMARD_8 = np.array([
    [+1, +1, +1, +1, +1, +1, +1, +1],
    [+1, -1, +1, -1, +1, -1, +1, -1],
    [+1, +1, -1, -1, +1, +1, -1, -1],
    [+1, -1, -1, +1, +1, -1, -1, +1],
    [+1, +1, +1, +1, -1, -1, -1, -1],
    [+1, -1, +1, -1, -1, +1, -1, +1],
    [+1, +1, -1, -1, -1, -1, +1, +1],
    [+1, -1, -1, +1, -1, +1, +1, -1],
], dtype=np.float64) / np.sqrt(8.0)


@dataclass
class ReverbParams:
    room_size: float = 0.5     # [0, 1] -> scales delay-line lengths 0.2x..2.0x
    decay_time_s: float = 1.5  # RT60 in seconds
    hf_damping: float = 0.5    # [0, 1] -> per-line LPF cutoff
    predelay_ms: float = 10.0
    wet_db: float = -12.0      # used by caller when mixing wet into bus


def _per_line_decay_gains(lengths: np.ndarray, decay_time_s: float, fs: float) -> np.ndarray:
    """Per-line gain so total RT60 across the FDN matches decay_time_s."""
    if decay_time_s <= 0.0:
        return np.zeros_like(lengths, dtype=np.float64)
    rt60_samples = decay_time_s * fs
    # Standard Jot formula: g_i = 10^(-3 * delay_i / RT60_samples)
    return 10.0 ** (-3.0 * lengths.astype(np.float64) / rt60_samples)


def _damping_alpha(hf_damping: float, fs: float) -> float:
    """1-pole LPF coefficient. 0 -> no damping; 1 -> ~500 Hz cutoff."""
    if hf_damping <= 0.0:
        return 0.0
    cutoff_hz = (1.0 - hf_damping) * (fs * 0.5) + hf_damping * 500.0
    cutoff_hz = max(cutoff_hz, 100.0)
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    dt = 1.0 / fs
    return float(dt / (rc + dt))


@njit(cache=True)
def _fdn_kernel(
    x_l: np.ndarray, x_r: np.ndarray,
    lengths: np.ndarray, gains: np.ndarray,
    H: np.ndarray, lp_alpha: float, predelay_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = x_l.size
    n_lines = lengths.size
    max_len = 0
    for k in range(n_lines):
        if lengths[k] > max_len:
            max_len = lengths[k]
    buf = np.zeros((n_lines, max_len), dtype=np.float64)
    write = np.zeros(n_lines, dtype=np.int64)
    lp_state = np.zeros(n_lines, dtype=np.float64)

    # Predelay buffer (single channel — fed from sum of L+R input)
    pred_buf_size = predelay_samples + 4 if predelay_samples > 0 else 1
    pred_buf = np.zeros(pred_buf_size, dtype=np.float64)
    pred_write = 0

    out_l = np.zeros(n, dtype=np.float64)
    out_r = np.zeros(n, dtype=np.float64)

    # Output sign pattern: alternate +/- to spread spatially across L/R
    # Lines 0,2,4,6 -> L+ R-, Lines 1,3,5,7 -> L- R+
    inv_sqrt = 1.0 / np.sqrt(float(n_lines))

    for i in range(n):
        # Sum input to mono, push into predelay buffer
        in_mono = 0.5 * (x_l[i] + x_r[i])
        if predelay_samples > 0:
            pred_buf[pred_write] = in_mono
            read_idx = (pred_write - predelay_samples) % pred_buf_size
            in_delayed = pred_buf[read_idx]
            pred_write = (pred_write + 1) % pred_buf_size
        else:
            in_delayed = in_mono

        # Read from each delay line's tap (oldest sample = at write index)
        line_out = np.zeros(n_lines, dtype=np.float64)
        for k in range(n_lines):
            line_out[k] = buf[k, write[k]]

        # Sum line outputs into stereo output
        sum_l = 0.0
        sum_r = 0.0
        for k in range(n_lines):
            if (k % 2) == 0:
                sum_l += line_out[k]
                sum_r -= line_out[k]
            else:
                sum_l -= line_out[k]
                sum_r += line_out[k]
        out_l[i] = sum_l * inv_sqrt
        out_r[i] = sum_r * inv_sqrt

        # Apply per-line damping LPF + decay gain
        damped = np.empty(n_lines, dtype=np.float64)
        for k in range(n_lines):
            if lp_alpha > 0.0:
                lp_state[k] = (1.0 - lp_alpha) * lp_state[k] + lp_alpha * line_out[k]
                damped[k] = lp_state[k] * gains[k]
            else:
                damped[k] = line_out[k] * gains[k]

        # Hadamard mix the damped outputs to get feedback inputs
        feedback = np.zeros(n_lines, dtype=np.float64)
        for r in range(n_lines):
            s = 0.0
            for c in range(n_lines):
                s += H[r, c] * damped[c]
            feedback[r] = s

        # Write input + feedback into each delay line
        for k in range(n_lines):
            buf[k, write[k]] = in_delayed * inv_sqrt + feedback[k]
            write[k] = (write[k] + 1) % lengths[k]

    return out_l, out_r


def apply_reverb(x: np.ndarray, fs: float, p: ReverbParams) -> np.ndarray:
    """Apply FDN reverb. Returns wet-only stereo (shape (2, samples))."""
    # Widened room_size scale: 0 -> 0.2x (small room), 1 -> 2.0x (large hall).
    scale = 0.2 + p.room_size * 1.8
    lengths = np.maximum(np.round(_PRIMES * scale).astype(np.int64), np.int64(8))
    # Scale to actual sample rate (primes designed at 48 kHz)
    lengths = np.maximum((lengths.astype(np.float64) * fs / 48000.0).astype(np.int64), np.int64(8))

    gains = _per_line_decay_gains(lengths, p.decay_time_s, fs)
    lp_alpha = _damping_alpha(p.hf_damping, fs)
    predelay_samples = max(0, int(round(p.predelay_ms * fs / 1000.0)))

    if x.ndim == 1:
        x_l = x.astype(np.float64)
        x_r = x.astype(np.float64)
    elif x.ndim == 2 and x.shape[0] == 2:
        x_l = x[0].astype(np.float64)
        x_r = x[1].astype(np.float64)
    else:
        raise ValueError(f"unsupported input shape {x.shape}")

    out_l, out_r = _fdn_kernel(
        x_l, x_r, lengths, gains, _HADAMARD_8,
        float(lp_alpha), int(predelay_samples),
    )
    return np.stack([out_l, out_r])


__all__ = ["ReverbParams", "apply_reverb"]
