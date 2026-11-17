"""Named raw audio sources, aligned at sample zero for static mixing."""

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

RATE = 48_000
FORMATS = {".wav", ".wave", ".bwf", ".rf64", ".aif", ".aiff", ".aifc", ".flac", ".ogg", ".mp3"}


def discover(root):
    if not root.is_dir():
        raise ValueError(f"tracks folder does not exist: {root}")
    paths = sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in FORMATS
    )
    if not paths:
        raise ValueError(f"no supported audio files in {root}")
    return paths


def read_source(root: Path, name: str):
    path = root / name
    try:
        with sf.SoundFile(path) as source:
            if source.channels not in (1, 2) or source.frames < 1:
                raise ValueError("expected nonempty mono or stereo audio")
            rate = source.samplerate
            audio = source.read(dtype="float32", always_2d=True)
            if len(audio) != source.frames or not np.isfinite(audio).all():
                raise ValueError("incomplete or non-finite audio")
    except (ValueError, RuntimeError, OSError) as err:
        raise ValueError(f"{name}: {err}") from err
    if rate != RATE:
        divisor = gcd(rate, RATE)
        audio = resample_poly(audio, RATE // divisor, rate // divisor, axis=0)
    return torch.from_numpy(np.ascontiguousarray(audio.T))
