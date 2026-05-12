"""Tempo-synced stereo delay with feedback LPF and optional ping-pong.

- Continuous delay time in quarter-note units `delay_time_qn` (Steinmetz-style;
  log-uniform in [0.0625, 4.0] = 1/16-note to 1-note). The derived
  `nearest_subdivision` string is kept in dataset metadata for human reference
  but the model consumes `delay_time_qn` directly.
- Fractional read via 4-tap Lagrange interpolation.
- 1-pole damping LPF in the feedback path (hf_damping = 0 -> no damping;
  1 -> heavily damped, cutoff ~1 kHz).
- Ping-pong cross-couples L feedback into R input and vice versa.

apply_delay returns wet signal only (per spec §4.1: caller mixes with dry
at sampled wet_db on the bus side).
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


# Subdivision-name catalogue (kept for human-readable derived metadata only —
# the encoder consumes continuous `delay_time_qn`).
SUBDIV_NAME_TO_QN = {
    "1/16": 0.25,
    "1/8":  0.5,
    "1/8D": 0.75,
    "1/4":  1.0,
    "1/4D": 1.5,
    "1/2":  2.0,
    "1/1":  4.0,
}
VALID_SUBDIVISIONS = list(SUBDIV_NAME_TO_QN.keys())


def delay_qn_to_seconds(delay_time_qn: float, bpm: float) -> float:
    """Convert continuous delay time (in quarter-note units) to seconds."""
    return float(delay_time_qn) * (60.0 / float(bpm))


def nearest_subdivision_name(delay_time_qn: float) -> str:
    """Snap a continuous delay_time_qn to the nearest named subdivision (for
    human-readable metadata only)."""
    qn = float(delay_time_qn)
    return min(SUBDIV_NAME_TO_QN, key=lambda name: abs(SUBDIV_NAME_TO_QN[name] - qn))


def damping_to_alpha(hf_damping: float, fs: float) -> float:
    """Map hf_damping in [0, 1] to a 1-pole LPF coefficient alpha.

    0  -> alpha=0  (no LPF, full feedback bandwidth)
    1  -> alpha tuned so cutoff is ~1 kHz at fs (heavily damped)
    """
    if hf_damping <= 0.0:
        return 0.0
    cutoff_hz = (1.0 - hf_damping) * (fs * 0.5) + hf_damping * 1000.0
    cutoff_hz = max(cutoff_hz, 100.0)
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    dt = 1.0 / fs
    alpha = dt / (rc + dt)
    return float(alpha)


@dataclass
class DelayParams:
    delay_time_qn: float = 1.0   # in quarter-note units; log-uniform [0.0625, 4.0]
    feedback: float = 0.4
    hf_damping: float = 0.4
    ping_pong: bool = False
    wet_db: float = -12.0


@njit(cache=True)
def _delay_kernel(
    x_l: np.ndarray, x_r: np.ndarray,
    delay_int: int, delay_frac: float,
    feedback: float, lp_alpha: float,
    ping_pong: bool, buf_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = x_l.size
    out_l = np.zeros(n, dtype=np.float64)
    out_r = np.zeros(n, dtype=np.float64)
    buf_l = np.zeros(buf_size, dtype=np.float64)
    buf_r = np.zeros(buf_size, dtype=np.float64)
    write = 0
    state_l = 0.0
    state_r = 0.0

    d = delay_frac
    w0 = -d * (d - 1.0) * (d - 2.0) / 6.0
    w1 = (d + 1.0) * (d - 1.0) * (d - 2.0) / 2.0
    w2 = -(d + 1.0) * d * (d - 2.0) / 2.0
    w3 = (d + 1.0) * d * (d - 1.0) / 6.0

    for i in range(n):
        # Lagrange window: y[-1]..y[2] correspond to samples at
        # (delay_int - 1)..(delay_int + 2) ago. With d=delay_frac, the
        # interpolation point sits between y[0]=delay_int-ago and
        # y[1]=delay_int+1-ago, giving an effective delay of (delay_int +
        # delay_frac) = delay_samples. (Prior version had the window
        # reversed, producing an effective delay of delay_int+1-delay_frac.)
        i0 = (write - (delay_int - 1)) % buf_size
        i1 = (write - delay_int)       % buf_size
        i2 = (write - (delay_int + 1)) % buf_size
        i3 = (write - (delay_int + 2)) % buf_size

        wet_l = w0 * buf_l[i0] + w1 * buf_l[i1] + w2 * buf_l[i2] + w3 * buf_l[i3]
        wet_r = w0 * buf_r[i0] + w1 * buf_r[i1] + w2 * buf_r[i2] + w3 * buf_r[i3]

        if lp_alpha > 0.0:
            state_l = (1.0 - lp_alpha) * state_l + lp_alpha * wet_l
            state_r = (1.0 - lp_alpha) * state_r + lp_alpha * wet_r
            fb_l = state_l
            fb_r = state_r
        else:
            fb_l = wet_l
            fb_r = wet_r

        if ping_pong:
            buf_l[write] = x_r[i] + feedback * fb_r
            buf_r[write] = x_l[i] + feedback * fb_l
        else:
            buf_l[write] = x_l[i] + feedback * fb_l
            buf_r[write] = x_r[i] + feedback * fb_r

        out_l[i] = wet_l
        out_r[i] = wet_r
        write = (write + 1) % buf_size

    return out_l, out_r


def apply_delay(x: np.ndarray, fs: float, bpm: float, p: DelayParams) -> np.ndarray:
    """Apply delay. Returns wet-only stereo (shape (2, samples))."""
    delay_s = delay_qn_to_seconds(p.delay_time_qn, bpm)
    delay_samples = delay_s * fs
    delay_int = int(np.floor(delay_samples))
    delay_frac = float(delay_samples - delay_int)
    if delay_int < 1:
        delay_int = 1
        delay_frac = 0.0

    buf_size = max(int(2 * fs), delay_int + 4)

    if x.ndim == 1:
        x_l = x.astype(np.float64)
        x_r = x.astype(np.float64)
    elif x.ndim == 2 and x.shape[0] == 2:
        x_l = x[0].astype(np.float64)
        x_r = x[1].astype(np.float64)
    else:
        raise ValueError(f"unsupported input shape {x.shape}")

    lp_alpha = damping_to_alpha(p.hf_damping, fs)

    out_l, out_r = _delay_kernel(
        x_l, x_r, delay_int, delay_frac,
        float(p.feedback), float(lp_alpha),
        bool(p.ping_pong), int(buf_size),
    )

    return np.stack([out_l, out_r])


__all__ = [
    "DelayParams",
    "apply_delay",
    "delay_qn_to_seconds",
    "nearest_subdivision_name",
    "VALID_SUBDIVISIONS",
    "SUBDIV_NAME_TO_QN",
]
