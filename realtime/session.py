"""Session loader: read N stems from a folder, resample to 48 kHz stereo.

A Session is the in-memory representation of "one multitrack project" —
all the stems pre-loaded, time-aligned, padded to the longest length,
ready for the audio engine to slice through. It's the realtime analog
of the training-time WebDataset bundle.

V1+ also resolves a precomputed MERT embedding per stem (from the
`dmc-data/mert_cache/<dataset>/<session>.npz` files produced by
`scripts/precompute_mert_cache.py`). Stems that aren't found in the
cache get zero embeddings — matches how training handles the same case.
"""

from __future__ import annotations

import logging
from math import gcd
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 48_000
DEFAULT_MERT_CACHE_ROOT = "dmc-data/mert_cache"
DEFAULT_MERT_DIM = 768


class Session:
    """A folder of stem WAV/FLAC files loaded into memory at the target rate.

    Attributes:
        stems:      (n_tracks, 2, T_max) float32 — all stems, padded to longest.
        stem_paths: list[Path] in the same order as `stems[i]`.
        sample_rate: int (default 48000 = the encoder's training rate).
        duration_s: float — playback length (longest stem).
        mert:       (n_tracks, mert_dim) float32 — per-stem MERT embedding,
                    or None when MERT loading was disabled. Stems that
                    weren't found in the cache get zero rows (see
                    `mert_missing` for the list).
        mert_missing: list[str] — stem filenames that had no embedding in
                      the cache (so their MERT row is zero).
        mert_source:  Path or None — the .npz file that satisfied the lookup.

    Mono stems are broadcast to stereo. Stems wider than 2 channels are
    truncated to L+R. Mix / preview / mixture files are filtered out by name.
    """

    def __init__(
        self,
        stems_dir: str | Path,
        sample_rate: int = SAMPLE_RATE,
        max_tracks: Optional[int] = None,
        *,
        mert_cache_root: Optional[str | Path] = DEFAULT_MERT_CACHE_ROOT,
        mert_cache_npz: Optional[str | Path] = None,
        mert_dim: int = DEFAULT_MERT_DIM,
        require_mert: bool = False,
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

        # ---- MERT lookup (V1+) ----
        # Either an explicit npz path or auto-detect by session-folder name
        # under cache_root/<dataset>/<session>.npz. Returns zeros for stems
        # that aren't in the cache file (training does the same).
        self.mert_dim: int = int(mert_dim)
        self.mert_source: Optional[Path] = None
        self.mert_missing: list[str] = []
        if mert_dim <= 0:
            self.mert: Optional[np.ndarray] = None
        else:
            self.mert = self._load_mert(
                mert_cache_root=mert_cache_root,
                mert_cache_npz=mert_cache_npz,
                mert_dim=mert_dim,
                require_mert=require_mert,
            )

    def _load_mert(
        self,
        *,
        mert_cache_root: Optional[str | Path],
        mert_cache_npz: Optional[str | Path],
        mert_dim: int,
        require_mert: bool,
    ) -> Optional[np.ndarray]:
        npz_path: Optional[Path] = None
        if mert_cache_npz is not None:
            npz_path = Path(mert_cache_npz).expanduser().resolve()
            if not npz_path.exists():
                raise FileNotFoundError(f"--mert-cache-npz not found: {npz_path}")
        elif mert_cache_root is not None:
            root = Path(mert_cache_root).expanduser().resolve()
            if root.is_dir():
                # Try each dataset subdir for a <session>.npz match. The
                # session name is the stems-dir basename (or, for the
                # cambridge-mt "<S>/<S>" doubled layout, the parent name —
                # try both).
                candidates: list[Path] = []
                for dataset_dir in sorted(root.iterdir()):
                    if not dataset_dir.is_dir():
                        continue
                    for name in (self.dir.name, self.dir.parent.name):
                        candidates.append(dataset_dir / f"{name}.npz")
                for c in candidates:
                    if c.exists():
                        npz_path = c
                        break

        if npz_path is None:
            msg = (
                "no MERT cache found for this session "
                f"(searched under {mert_cache_root!r}). Run "
                "scripts/precompute_mert_cache.py, pass --mert-cache-npz, "
                "or disable MERT (--mert-dim 0)."
            )
            if require_mert:
                raise FileNotFoundError(msg)
            logging.warning("Session: %s", msg)
            return np.zeros((self.n_tracks, mert_dim), dtype=np.float32)

        with np.load(npz_path, allow_pickle=True) as data:
            cached_names = list(data["filenames"])
            cached_embs = data["embeddings"]
        if cached_embs.size and cached_embs.shape[-1] != mert_dim:
            raise ValueError(
                f"MERT cache {npz_path} has dim {cached_embs.shape[-1]}, "
                f"expected {mert_dim}"
            )
        name_to_emb = {str(n): cached_embs[i] for i, n in enumerate(cached_names)}

        out = np.zeros((self.n_tracks, mert_dim), dtype=np.float32)
        for i, p in enumerate(self.stem_paths):
            emb = name_to_emb.get(p.name)
            if emb is None:
                self.mert_missing.append(p.name)
            else:
                out[i] = emb.astype(np.float32, copy=False)
        self.mert_source = npz_path
        if self.mert_missing:
            logging.warning(
                "Session: %d/%d stems missing from MERT cache %s (zero-filled): %s",
                len(self.mert_missing), self.n_tracks, npz_path.name,
                ", ".join(self.mert_missing[:5])
                + (" ..." if len(self.mert_missing) > 5 else ""),
            )
        return out

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
        mert_info = ""
        if self.mert is not None:
            src = self.mert_source.name if self.mert_source is not None else "<none>"
            miss = f", missing={len(self.mert_missing)}" if self.mert_missing else ""
            mert_info = f", mert={src}{miss}"
        return (
            f"Session(dir={self.dir.name!r}, n_tracks={self.n_tracks}, "
            f"duration={self.duration_s:.1f}s, sr={self.sample_rate}{mert_info})"
        )
