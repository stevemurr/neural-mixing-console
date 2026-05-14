"""Session loader: read N stems from a folder, resample to 48 kHz stereo.

A Session is the in-memory representation of "one multitrack project" —
all the stems pre-loaded, time-aligned, padded to the longest length,
ready for the audio engine to slice through. It's the realtime analog
of the training-time WebDataset bundle.
"""

from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 48_000


class Session:
    """A folder of stem WAV/FLAC files loaded into memory at the target rate.

    Attributes:
        stems:      (n_tracks, 2, T_max) float32 — all stems, padded to longest.
        stem_paths: list[Path] in the same order as `stems[i]`.
        sample_rate: int (default 48000 = the encoder's training rate).
        duration_s: float — playback length (longest stem).

    Mono stems are broadcast to stereo. Stems wider than 2 channels are
    truncated to L+R. Mix / preview / mixture files are filtered out by name.
    """

    def __init__(
        self,
        stems_dir: str | Path,
        sample_rate: int = SAMPLE_RATE,
        max_tracks: Optional[int] = None,
    ) -> None:
        d = Path(stems_dir).expanduser().resolve()
        if not d.is_dir():
            raise FileNotFoundError(f"stems dir not found: {d}")

        exts = (".wav", ".flac", ".aif", ".aiff")
        paths = sorted(
            p for p in d.iterdir()
            if p.suffix.lower() in exts
            and not p.name.startswith(".")
            and "mixture" not in p.name.lower()
            and "preview" not in p.name.lower()
            and not p.name.lower().endswith("_mix.wav")
        )
        if max_tracks is not None:
            paths = paths[:max_tracks]
        if not paths:
            raise ValueError(f"no stem audio files found in {d}")

        self.dir = d
        self.sample_rate = sample_rate
        self.stem_paths = paths

        stems = [self._load_stem(p, sample_rate) for p in paths]
        T_max = max(s.shape[1] for s in stems)
        padded = np.zeros((len(stems), 2, T_max), dtype=np.float32)
        for i, s in enumerate(stems):
            padded[i, :, : s.shape[1]] = s

        self.stems: np.ndarray = padded
        self.n_tracks: int = len(stems)
        self.duration_s: float = T_max / sample_rate

    @staticmethod
    def _load_stem(path: Path, target_sr: int) -> np.ndarray:
        """Load a single stem, return (2, T) float32 at target_sr."""
        data, src_sr = sf.read(str(path), dtype="float32", always_2d=True)
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        elif data.shape[1] > 2:
            data = data[:, :2]
        x = data.T.astype(np.float32, copy=False)  # (2, T)
        if src_sr != target_sr:
            g = gcd(int(src_sr), int(target_sr))
            up = target_sr // g
            down = int(src_sr) // g
            x = resample_poly(x, up, down, axis=-1).astype(np.float32, copy=False)
        return x

    def __repr__(self) -> str:
        return (
            f"Session(dir={self.dir.name!r}, n_tracks={self.n_tracks}, "
            f"duration={self.duration_s:.1f}s, sr={self.sample_rate})"
        )
