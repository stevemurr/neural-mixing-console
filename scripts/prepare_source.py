"""Source-audio preparation (spec §2.3).

Recursively scans `source_audio/` for raw multitrack WAV files, validates,
resamples to 48 kHz, chunks into 6-second segments, computes per-segment
features for filtering, writes segments as 24-bit FLAC, and emits a parquet
index.

Usage:
    python scripts/prepare_source.py \
        --source-root source_audio \
        [--n-workers 4] [--no-cache-segments]

Idempotent: re-running skips segments already on disk and rebuilds the index
parquet. New WAVs added to source_audio/ are picked up on next run.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import soxr
from tqdm import tqdm

# Make `scripts/` importable for _common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import write_flac, get_libflac_version  # noqa: E402


TARGET_FS = 48_000
SEGMENT_SECONDS = 6.0
SEGMENT_FRAMES = int(SEGMENT_SECONDS * TARGET_FS)  # 288_000

MIN_FILE_DURATION_S = 10.0
MIN_RMS_DBFS = -50.0   # drop segments quieter than this (silence/noise)
MAX_CREST_DB = 30.0    # drop segments with crazy crest factor (likely glitch)
MIN_INPUT_SAMPLERATE = 44100
MAX_INPUT_CHANNELS = 2

# Stable segment_id derivation: hash of (source_path, start_frame_at_native_sr).
def derive_segment_id(source_path: str, start_frame_native: int) -> str:
    h = hashlib.sha256()
    h.update(source_path.encode("utf-8"))
    h.update(b"|")
    h.update(start_frame_native.to_bytes(8, "little", signed=False))
    digest = h.digest()
    import uuid as _uuid
    return str(_uuid.UUID(bytes=digest[:16], version=4))


def stereo_is_truly_stereo(audio_lr: np.ndarray, threshold_db: float = 0.01) -> bool:
    """audio_lr shape (samples, 2). Stereo if L and R differ by >= threshold_db RMS."""
    l_rms = np.sqrt(np.mean(audio_lr[:, 0].astype(np.float64) ** 2)) + 1e-12
    r_rms = np.sqrt(np.mean(audio_lr[:, 1].astype(np.float64) ** 2)) + 1e-12
    rms_db_diff = abs(20 * np.log10(l_rms / r_rms))
    return rms_db_diff >= threshold_db


def compute_features(audio: np.ndarray) -> tuple[float, float, float]:
    """Compute (rms_db, crest_db, spectral_flatness) on mono representation.

    audio: (samples,) for mono or (samples, channels) for stereo (uses sum-to-mono).
    """
    if audio.ndim == 2:
        mono = audio.mean(axis=1)
    else:
        mono = audio
    mono = mono.astype(np.float64, copy=False)

    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono ** 2)))
    rms_db = 20.0 * np.log10(rms + 1e-12)
    crest_db = 20.0 * np.log10((peak + 1e-12) / (rms + 1e-12)) if rms > 0 else 60.0

    # Spectral flatness on a single windowed FFT (Wiener entropy of magnitudes)
    n_fft = min(8192, len(mono))
    if n_fft < 64:
        spec_flat = 0.0
    else:
        center = len(mono) // 2
        chunk = mono[center - n_fft // 2 : center + n_fft // 2]
        win = np.hanning(n_fft)
        mag = np.abs(np.fft.rfft(chunk * win)) + 1e-12
        # Skip DC bin
        mag = mag[1:]
        log_mag = np.log(mag)
        spec_flat = float(np.exp(log_mag.mean()) / mag.mean())

    return rms_db, crest_db, spec_flat


def _classify_dataset(rel_path: Path) -> str:
    parts = rel_path.parts
    if not parts:
        return "unknown"
    top = parts[0].lower()
    if "medleydb" in top:
        return "medleydb"
    if "cambridge" in top:
        return "cambridge-mt"
    return top


def _song_id(rel_path: Path) -> str:
    """Best-effort song id from path parts.

    Both MedleyDB and Cambridge-MT use a per-song top-level folder under their
    dataset root. We take the first folder under the dataset root.
    """
    parts = rel_path.parts
    # parts[0] = dataset folder, parts[1] = song folder
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return rel_path.stem


@dataclass
class FileTask:
    abs_path: Path
    rel_path: Path
    dataset: str
    song_id: str
    instrument_label: Optional[str]


def _discover_files(source_root: Path) -> list[FileTask]:
    """Find every .wav under source_root, skipping reference mixes (e.g. *_MIX.wav,
    files inside MIX/ folders) so we only ingest raw tracks."""
    out: list[FileTask] = []
    skip_substrings = ("_mix.wav", "_final.wav", "/mix/", "/mixture.wav", "_mixture.wav")
    for path in source_root.rglob("*.wav"):
        rel = path.relative_to(source_root)
        rel_lower = "/" + str(rel).lower()
        if any(s in rel_lower for s in skip_substrings):
            continue
        dataset = _classify_dataset(rel)
        out.append(FileTask(
            abs_path=path,
            rel_path=rel,
            dataset=dataset,
            song_id=_song_id(rel),
            instrument_label=None,  # mirdata enrichment can populate later for MedleyDB
        ))
    return sorted(out, key=lambda t: str(t.rel_path))


def _process_one_file(task: FileTask, segments_dir: Path, cache_segments: bool) -> list[dict]:
    """Validate + chunk one source WAV. Returns rows for the index parquet."""
    rows: list[dict] = []
    try:
        info = sf.info(str(task.abs_path))
    except Exception as e:
        print(f"  reject {task.rel_path}: cannot read header ({e})", file=sys.stderr)
        return rows

    if info.channels > MAX_INPUT_CHANNELS:
        print(f"  reject {task.rel_path}: {info.channels} channels (max {MAX_INPUT_CHANNELS})", file=sys.stderr)
        return rows
    if info.samplerate < MIN_INPUT_SAMPLERATE:
        print(f"  reject {task.rel_path}: {info.samplerate} Hz (min {MIN_INPUT_SAMPLERATE})", file=sys.stderr)
        return rows
    duration = info.frames / info.samplerate
    if duration < MIN_FILE_DURATION_S:
        return rows  # silently skip short files

    native_chunk = int(SEGMENT_SECONDS * info.samplerate)
    if native_chunk <= 0:
        return rows

    n_chunks = info.frames // native_chunk
    if n_chunks == 0:
        return rows

    for chunk_idx in range(n_chunks):
        start_native = chunk_idx * native_chunk
        try:
            audio_native, sr = sf.read(
                str(task.abs_path),
                start=start_native,
                frames=native_chunk,
                dtype="float32",
                always_2d=False,
            )
        except Exception as e:
            print(f"  read error {task.rel_path}@{start_native}: {e}", file=sys.stderr)
            continue

        # Determine effective channel count for stereo files
        if audio_native.ndim == 2:
            if audio_native.shape[1] == 1:
                audio_native = audio_native[:, 0]
                channels = 1
            elif stereo_is_truly_stereo(audio_native):
                channels = 2
            else:
                # near-mono stereo; collapse to mono
                audio_native = audio_native.mean(axis=1).astype(np.float32)
                channels = 1
        else:
            channels = 1

        # Resample to 48 kHz with soxr HQ
        if sr != TARGET_FS:
            if audio_native.ndim == 2:
                resampled = soxr.resample(audio_native, sr, TARGET_FS, quality="HQ")
            else:
                resampled = soxr.resample(audio_native, sr, TARGET_FS, quality="HQ")
        else:
            resampled = audio_native
        # Truncate / pad to exact SEGMENT_FRAMES
        if resampled.ndim == 2:
            n = resampled.shape[0]
            if n > SEGMENT_FRAMES:
                resampled = resampled[:SEGMENT_FRAMES]
            elif n < SEGMENT_FRAMES:
                pad = np.zeros((SEGMENT_FRAMES - n, resampled.shape[1]), dtype=resampled.dtype)
                resampled = np.vstack([resampled, pad])
        else:
            n = resampled.shape[0]
            if n > SEGMENT_FRAMES:
                resampled = resampled[:SEGMENT_FRAMES]
            elif n < SEGMENT_FRAMES:
                resampled = np.concatenate([resampled, np.zeros(SEGMENT_FRAMES - n, dtype=resampled.dtype)])

        # Features (use 1D mono representation)
        if resampled.ndim == 2:
            mono_for_feat = resampled.mean(axis=1)
        else:
            mono_for_feat = resampled
        rms_db, crest_db, spec_flat = compute_features(mono_for_feat)

        if rms_db < MIN_RMS_DBFS:
            continue
        if crest_db > MAX_CREST_DB:
            continue

        seg_id = derive_segment_id(str(task.rel_path), start_native)

        # Convert to channels-first for write_flac
        if resampled.ndim == 2:
            audio_cf = resampled.T  # (channels, samples)
        else:
            audio_cf = resampled

        if cache_segments:
            seg_path = segments_dir / f"{seg_id}.flac"
            if not seg_path.exists():
                seg_path.parent.mkdir(parents=True, exist_ok=True)
                write_flac(seg_path, audio_cf, fs=TARGET_FS)

        rows.append({
            "segment_id": seg_id,
            "song_id": task.song_id,
            "source_path": str(task.rel_path),
            "start_sample": int(start_native),
            "duration_samples": SEGMENT_FRAMES,
            "channels": int(channels),
            "rms_db": float(rms_db),
            "crest_db": float(crest_db),
            "spectral_flatness": float(spec_flat),
            "instrument_label": task.instrument_label,
            "dataset": task.dataset,
        })

    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", default="source_audio",
                    help="root directory containing medleydb/ and/or cambridge-mt/ subfolders")
    ap.add_argument("--n-workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--no-cache-segments", action="store_true",
                    help="don't write segment FLACs (saves disk; segments will be regenerated on demand)")
    args = ap.parse_args()

    source_root = Path(args.source_root).resolve()
    if not source_root.exists():
        print(f"source_root not found: {source_root}", file=sys.stderr)
        return 1

    segments_dir = source_root / "segments"
    cache_segments = not args.no_cache_segments
    if cache_segments:
        segments_dir.mkdir(parents=True, exist_ok=True)

    print(f"FLAC encoder: {get_libflac_version()}")
    print(f"Scanning {source_root} for raw WAVs...")
    tasks = _discover_files(source_root)
    print(f"  found {len(tasks)} WAV files")
    if not tasks:
        return 1

    print(f"Processing with {args.n_workers} workers...")
    all_rows: list[dict] = []
    t0 = time.time()

    if args.n_workers <= 1:
        for task in tqdm(tasks):
            all_rows.extend(_process_one_file(task, segments_dir, cache_segments))
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            futures = {ex.submit(_process_one_file, t, segments_dir, cache_segments): t for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures)):
                all_rows.extend(fut.result())

    elapsed = time.time() - t0
    print(f"Processed {len(tasks)} files in {elapsed:.1f}s -> {len(all_rows)} segments")

    if not all_rows:
        print("No segments produced; nothing to write.", file=sys.stderr)
        return 1

    df = pd.DataFrame(all_rows)
    out_path = source_root / "source_index.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(out_path))
    print(f"Wrote {out_path} ({len(df)} rows)")

    by_ds = df.groupby("dataset").size().to_dict()
    by_ch = df.groupby("channels").size().to_dict()
    print(f"  by dataset: {by_ds}")
    print(f"  by channels: {by_ch}")
    print(f"  rms_db quartiles: {df['rms_db'].quantile([0.25, 0.5, 0.75]).round(2).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
