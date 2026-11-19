"""Level compensation for realtime amplitude interpolation of aligned ERB filter outputs."""

import numpy as np
from scipy.signal import sosfilt

from nmc.data.raw_folder import RATE
from nmc.eq.erb_desk import AMOUNT, WIDTHS

DEFAULT_PERCENT = round(100 * AMOUNT)
# ITU-R BS.1770 Annex 1, 48 kHz (the desk's render rate).
K_FILTER = [
    [1.53512485958697, -2.69169618940638, 1.19839281085285, 1, -1.69065929318241, 0.73248077421585],
    [1, -2, 1, 1, -1.99004745483398, 0.99007225036621],
]


def anchors(width, percent):
    if percent <= DEFAULT_PERCENT:
        return "off", str(width), percent / DEFAULT_PERCENT
    return str(width), f"{width}-max", (percent - DEFAULT_PERCENT) / (100 - DEFAULT_PERCENT)


def block_product(a, b):
    """400 ms stereo K-weighted block cross-powers at 100 ms spacing."""
    step = RATE // 10
    samples = np.einsum("ij,ij->i", a, b)
    if len(samples) < step * 4:
        samples = np.pad(samples, (0, step * 4 - len(samples)))
    hops = samples[: len(samples) // step * step].reshape(-1, step).mean(1)
    return np.convolve(hops, np.ones(4) / 4, mode="valid")


def match_gain(power, target):
    """Solve the gated integrated loudness at playback level, including the absolute gate."""
    if power.max() <= 1e-20:
        return 1.0
    energy = power.mean()
    for _ in range(4):
        absolute = power >= energy * 10 ** ((-70 - target) / 10)
        relative = power >= 0.1 * power[absolute].mean()
        energy = power[absolute & relative].mean()
    return float(np.sqrt(10 ** ((target + 0.691) / 10) / energy))


def calibrate(versions):
    """Compute playback trims on phase-aligned FIR outputs, before browser monitoring."""
    peaks = {k: float(np.abs(v).max()) for k, v in versions.items()}
    top = max(peaks.values())
    scale = 0.85 / max(top, 1e-9)
    filtered = {}
    for key, wave in versions.items():
        scaled = wave * scale
        filtered[key] = sosfilt(K_FILTER, scaled, axis=0)
    powers = {k: block_product(v, v) for k, v in filtered.items()}
    energies, ceilings = {}, {}
    for width in WIDTHS:
        for percent in range(101):
            a, b, t = anchors(width, percent)
            if percent in (0, DEFAULT_PERCENT + 1):
                cross = block_product(filtered[a], filtered[b])
            energies[width, percent] = (
                (1 - t) ** 2 * powers[a] + t * t * powers[b] + 2 * t * (1 - t) * cross
            )
            ceilings[width, percent] = scale * ((1 - t) * peaks[a] + t * peaks[b])
    # One safe audition level for all widths and amounts on this bus, including bypass.
    target = -21.0
    for _ in range(3):
        trims = {k: match_gain(v, target) for k, v in energies.items()}
        max_peak = max(trims[k] * p for k, p in ceilings.items())
        if max_peak <= 0.85001:
            break
        target += 20 * np.log10(0.85 / max_peak)
    gains = {
        str(w): [float(match_gain(energies[w, p], target) * scale) for p in range(101)]
        for w in WIDTHS
    }
    return dict(lufs=round(float(target), 2) if top > 1e-9 else None, gains=gains)
