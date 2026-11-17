"""Shared utilities for source prep and dataset generation.

- FLAC I/O wrappers (channels-first audio convention to match reference DSP)
- ShardWriter: WebDataset-compatible tar shard writer with resume support
- Split assignment via deterministic hash (spec §13)
- Deterministic example-id derivation
- JSON helpers (numpy-aware encoding, nested-to-normalized helpers)
- Manifest writing with FLAC encoder version pin (spec §14)
- run_parallel_generation: multiprocessing helper for the per-example DSP loop
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf



# ---------- Audio I/O (channels-first convention) ----------

def write_flac(path: str | os.PathLike, audio: np.ndarray, fs: int = 48000) -> None:
    """Write a FLAC at 24-bit. Audio is (samples,) mono or (channels, samples) stereo."""
    if audio.ndim == 1:
        data = audio.astype(np.float32, copy=False)
    elif audio.ndim == 2:
        # soundfile wants (samples, channels)
        data = audio.T.astype(np.float32, copy=False)
    else:
        raise ValueError(f"unsupported audio shape {audio.shape}")
    sf.write(str(path), data, fs, format="FLAC", subtype="PCM_24")


def encode_flac_bytes(audio: np.ndarray, fs: int = 48000) -> bytes:
    """Encode FLAC into an in-memory bytes object (for tar streaming)."""
    if audio.ndim == 1:
        data = audio.astype(np.float32, copy=False)
    elif audio.ndim == 2:
        data = audio.T.astype(np.float32, copy=False)
    else:
        raise ValueError(f"unsupported audio shape {audio.shape}")
    buf = io.BytesIO()
    sf.write(buf, data, fs, format="FLAC", subtype="PCM_24")
    return buf.getvalue()


def read_flac(path: str | os.PathLike) -> tuple[np.ndarray, int]:
    """Read FLAC. Returns (audio, fs) where audio is (N,) or (channels, N)."""
    data, fs = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.T  # to (channels, samples)
    return data, fs


# ---------- Splits (spec §13) ----------

def split_for(key: str) -> str:
    """Hash a segment_id or song_id into one of train/val/test (90/5/5)."""
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "val"
    return "test"


# ---------- Deterministic example-id derivation ----------

def derive_example_id(master_seed: int, dataset_name: str, example_index: int | str) -> str:
    """UUIDv4-shaped, deterministic from (master_seed, dataset, index).

    `example_index` may be int (numeric example position) or str (e.g. a
    session+segment composite key for stage 3 ingest)."""
    h = hashlib.sha256()
    h.update(master_seed.to_bytes(8, "little", signed=False))
    h.update(b"|")
    h.update(dataset_name.encode("utf-8"))
    h.update(b"|")
    if isinstance(example_index, int):
        h.update(example_index.to_bytes(8, "little", signed=False))
    else:
        h.update(str(example_index).encode("utf-8"))
    digest = h.digest()
    return str(uuid.UUID(bytes=digest[:16], version=4))


# ---------- JSON encoding (numpy-aware) ----------

class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return v
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, cls=_NpEncoder, indent=None, separators=(",", ":")).encode("utf-8")


def json_dumps_pretty(obj: Any) -> bytes:
    return json.dumps(obj, cls=_NpEncoder, indent=2).encode("utf-8")


# ---------- ShardWriter ----------

class ShardWriter:
    """Streams examples into WebDataset-compatible tar shards.

    Per spec §11.1: files within an example must be lex-sorted contiguously.
    We accumulate per-example file bytes in memory and write them in a single
    sorted batch to the tar; rollover happens at examples_per_shard.

    Resume support: on init we drop any .tmp.tar (incomplete from prior crash)
    and count completed examples in the index .jsonl. The next shard index is
    determined by counting existing .tar files in shards/.
    """

    def __init__(
        self,
        output_dir: str | os.PathLike,
        prefix: str,
        examples_per_shard: int = 1000,
    ):
        self.output_dir = Path(output_dir)
        self.shards_dir = self.output_dir / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.examples_per_shard = examples_per_shard
        self.index_path = self.output_dir / f"{prefix}_index.jsonl"

        # Drop crashed-mid-shard temp tars from prior runs
        for tmp in self.shards_dir.glob(f"{prefix}_*.tar.tmp"):
            tmp.unlink()

        existing = sorted(self.shards_dir.glob(f"{prefix}_*.tar"))
        self.current_shard_idx = len(existing)

        self._completed_count = 0
        if self.index_path.exists():
            with open(self.index_path) as f:
                self._completed_count = sum(1 for _ in f)

        self._tar: Optional[tarfile.TarFile] = None
        self._tar_path: Optional[Path] = None
        self._examples_in_shard = 0
        self._pending_rows: list[dict] = []

    @property
    def completed_count(self) -> int:
        return self._completed_count

    def already_done(self, example_index: int) -> bool:
        return example_index < self._completed_count

    def _open_shard(self) -> None:
        path = self.shards_dir / f"{self.prefix}_{self.current_shard_idx:05d}.tar"
        self._tar_path = path
        self._tar = tarfile.open(str(path) + ".tmp", "w")
        self._examples_in_shard = 0

    def add_example(self, example_id: str, files: dict[str, bytes], row: dict) -> None:
        """`files` keys are filename suffixes after the example_id (e.g. 'dry.flac', 'wet.flac', 'json').

        Files within an example are written in lex-sorted order per spec §11.1.
        """
        if self._tar is None:
            self._open_shard()
        # Ensure files are written contiguously and sorted by full filename
        for suffix in sorted(files):
            data = files[suffix]
            arc_name = f"{example_id}.{suffix}"
            info = tarfile.TarInfo(arc_name)
            info.size = len(data)
            info.mtime = 0  # reproducibility: zero out mtime
            self._tar.addfile(info, io.BytesIO(data))

        row["shard_path"] = f"shards/{self.prefix}_{self.current_shard_idx:05d}.tar"
        row["shard_index"] = self._examples_in_shard
        self._pending_rows.append(row)
        self._examples_in_shard += 1

        if self._examples_in_shard >= self.examples_per_shard:
            self._flush_shard()

    def _flush_shard(self) -> None:
        if self._tar is None:
            return
        self._tar.close()
        # Atomic rename .tmp -> final
        os.replace(str(self._tar_path) + ".tmp", str(self._tar_path))

        # Append rows to index .jsonl
        with open(self.index_path, "a") as f:
            for r in self._pending_rows:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")

        self._completed_count += len(self._pending_rows)
        self._pending_rows.clear()
        self.current_shard_idx += 1
        self._tar = None
        self._tar_path = None
        self._examples_in_shard = 0

    def close(self) -> None:
        if self._tar is not None and self._examples_in_shard > 0:
            self._flush_shard()


def jsonl_to_parquet(jsonl_path: str | os.PathLike, parquet_path: str | os.PathLike) -> int:
    """Convert an index .jsonl (one row per example) to a parquet table.

    Returns the row count.
    """
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            rows.append(json.loads(line))
    if not rows:
        return 0
    # pyarrow auto-infers schema; nested dicts become struct columns.
    # For consistency, ensure all rows have the same keys at top level.
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(parquet_path))
    return len(rows)


# ---------- Manifest ----------

def get_libflac_version() -> str:
    """Best-effort capture of the FLAC encoder version for reproducibility (spec §14)."""
    try:
        out = subprocess.check_output(["flac", "--version"], stderr=subprocess.STDOUT, timeout=5)
        return out.decode("utf-8", errors="replace").strip().splitlines()[0]
    except Exception as e:
        # Fall back to libsndfile-reported version
        try:
            return f"libsndfile via soundfile {sf.__version__}; libsndfile_version unknown ({e})"
        except Exception:
            return f"unknown ({e})"


def get_git_commit(cwd: str | os.PathLike = ".") -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd),
            stderr=subprocess.DEVNULL, timeout=5,
        )
        return out.decode("ascii").strip()
    except Exception:
        return "unknown (not a git repo or git unavailable)"


def write_manifest(
    output_dir: str | os.PathLike,
    *,
    dataset: str,
    master_seed: int,
    n_examples: int,
    extra: Optional[dict] = None,
) -> None:
    """Write {output_dir}/{dataset}_manifest.json (spec §14)."""
    output_dir = Path(output_dir)
    manifest = {
        "dataset": dataset,
        "n_examples": n_examples,
        "master_seed": master_seed,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": get_git_commit(output_dir),
        "flac_encoder_version": get_libflac_version(),
        "soundfile_version": sf.__version__,
        "numpy_version": np.__version__,
    }
    if extra:
        manifest.update(extra)
    with open(output_dir / f"{dataset}_manifest.json", "wb") as f:
        f.write(json_dumps_pretty(manifest))


# ---------- Multiprocessing-based parallel generation ----------

import multiprocessing as _mp
from typing import Callable as _Callable, Optional as _Optional


def run_parallel_generation(
    *,
    n_examples: int,
    writer: "ShardWriter",
    worker_init_fn: _Callable[..., None],
    worker_init_args: tuple,
    worker_fn: _Callable[[int], _Optional[dict]],
    n_workers: int = 16,
    chunksize: int = 4,
    desc: str = "generating",
    unpack_bundles: bool = False,
    skip_resume: bool = False,
) -> dict:
    """Run a parallel data-generation pipeline using multiprocessing.

    The worker function runs the heavy per-example DSP. Default mode: return
    either `None` (skip this index — e.g., missing source segment) or
    `{"example_id": str, "files": dict[str, bytes], "row": dict}` for a single
    example.

    With `unpack_bundles=True`, the worker may instead return
    `{"bundles": [bundle, bundle, ...], ...}` where each bundle has the same
    one-example shape. This is useful when one input (e.g. a multitrack song)
    produces many output segments. Bundles from each worker call are written
    in order; resume support is approximate at the input-index granularity.

    The main process owns the `ShardWriter` and consumes results in
    submission order (via `Pool.imap`), so resume support and shard ordering
    both still work.

    `worker_init_fn` is called once per worker process; `worker_init_args` is
    the tuple of arguments. Use it to load the source-index parquet and other
    per-process state into module globals; this avoids re-loading per task.

    Resume support: in default mode, indices already completed (per
    `writer.completed_count`) are filtered out of the submission list. With
    `skip_resume=True`, all indices are submitted (use this when the input
    granularity (e.g. song) doesn't map 1:1 to output examples and the
    writer's `already_done` semantics don't apply).
    """
    from tqdm import tqdm  # local import keeps import-time light

    if skip_resume:
        indices = list(range(n_examples))
    else:
        indices = [i for i in range(n_examples) if not writer.already_done(i)]
    if not indices:
        print(f"{desc}: nothing to do ({n_examples} already complete)")
        return {"completed": 0, "skipped": 0}

    print(f"{desc}: {len(indices)} inputs to process "
          f"(skipping {n_examples - len(indices)} already complete) "
          f"with {n_workers} workers")

    n_completed = 0
    n_skipped = 0
    pbar = tqdm(total=len(indices), desc=desc)

    # spawn context avoids fork-related issues with already-imported torch/numba state
    ctx = _mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=worker_init_fn, initargs=worker_init_args) as pool:
        for result in pool.imap(worker_fn, indices, chunksize=chunksize):
            if result is None:
                n_skipped += 1
            elif unpack_bundles:
                bundles = result.get("bundles", [])
                if not bundles:
                    n_skipped += 1
                else:
                    for b in bundles:
                        writer.add_example(b["example_id"], b["files"], b["row"])
                        n_completed += 1
            else:
                writer.add_example(result["example_id"], result["files"], result["row"])
                n_completed += 1
            pbar.update(1)
    pbar.close()
    return {"completed": n_completed, "skipped": n_skipped}


__all__ = [
    "write_flac", "encode_flac_bytes", "read_flac",
    "split_for", "derive_example_id",
    "json_dumps", "json_dumps_pretty",
    "ShardWriter", "jsonl_to_parquet",
    "write_manifest", "get_libflac_version", "get_git_commit",
    "run_parallel_generation",
]
