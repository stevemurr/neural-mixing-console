"""Static, linked-channel brown-noise voicing on an auditory-filter frequency scale."""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import oaconvolve

from nmc.data.raw_folder import RATE

WIDTHS = (1, 3, 6)  # Gaussian full width at half maximum, in ERB-rate units.
AMOUNT = 0.15
LIMIT_DB = 2.0
FFT = 8192


def erb_rate(hz):
    return 21.4 * np.log10(1 + 0.00437 * hz)


def frequency(erb):
    return np.expm1(erb * np.log(10) / 21.4) / 0.00437


class BrownDesk:
    def __init__(self):
        lo, hi = erb_rate(np.array([20, 22050]))
        self.grid = np.linspace(lo, hi, round(hi - lo) + 1)
        self.centers = frequency(self.grid)
        hz = np.fft.rfftfreq(FFT, 1 / RATE)
        delta = (self.grid[:, None] - erb_rate(hz)) / np.diff(self.grid)[0]
        self.bank = np.maximum(0, 1 - abs(delta)) * ((hz >= 20) & (hz <= 22050))
        # Expected power of band-limited brown noise through these exact filters.
        self.target_db = 10 * np.log10(self.bank @ (1 / np.maximum(hz, 20) ** 2))

    def measure(self, audio):
        """Average channel powers over active frames on a bounded whole-track grid."""
        count = min(24, max(1, int(np.ceil(len(audio) / RATE))))
        starts = np.linspace(0, max(0, len(audio) - RATE), count, dtype=int)
        windows = []
        for start in starts:
            samples = audio[start : start + RATE]
            if len(samples) < FFT:
                samples = np.pad(samples, ((0, FFT - len(samples)), (0, 0)))
            frames = np.lib.stride_tricks.sliding_window_view(samples, FFT, axis=0)[:: FFT // 2]
            power = abs(np.fft.rfft(frames * np.hanning(FFT))) ** 2
            windows.append(power.mean(axis=1))
        spectra = np.concatenate(windows)
        energy = spectra.sum(axis=1)
        active = (energy > 0) & (energy >= energy.max() * 1e-4)
        return self.bank @ spectra[active].mean(0) if active.any() else np.zeros(len(self.grid))

    def curve(self, power, width):
        if width <= 0:
            raise ValueError("smoothing width must be positive")
        if np.max(power) <= 0:
            return np.zeros_like(power)
        db = 10 * np.log10(np.maximum(power, 1e-20))
        support = np.clip((db - db.max() + 45) / 25, 0, 1)
        difference = self.target_db - db
        difference -= np.average(difference, weights=support)
        sigma = width / (np.sqrt(8 * np.log(2)) * np.diff(self.grid)[0])
        smooth = gaussian_filter1d(difference * support, sigma, mode="nearest")
        weight = gaussian_filter1d(support, sigma, mode="nearest")
        return support * LIMIT_DB * np.tanh(AMOUNT * smooth / np.maximum(weight, 1e-10) / LIMIT_DB)

    def kernel(self, curve):
        """Unit-delay-centered FIR coefficients shared with the Web Audio convolver."""
        size = 16384
        hz = np.fft.rfftfreq(size, 1 / RATE)
        response_db = PchipInterpolator(
            np.r_[0, self.grid, erb_rate(RATE / 2)], np.r_[0, curve, 0]
        )(erb_rate(hz))
        impulse = np.fft.fftshift(np.fft.irfft(10 ** (response_db / 20)))
        half = FFT // 2
        kernel = impulse[size // 2 - half : size // 2 + half + 1] * np.kaiser(2 * half + 1, 6)
        return kernel.astype(np.float32)

    def apply(self, audio, curve):
        if not np.any(curve):
            return audio.copy()
        return oaconvolve(audio, self.kernel(curve)[:, None], mode="same", axes=0)
