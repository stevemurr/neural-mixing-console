"""Compute MERT embeddings for every track in stage 3 shards.

Reads existing stage 3 webdataset shards (stage3_cambridge / stage3_slakh),
runs MERT on each per-track audio, stores the resulting (N_tracks, 768)
tensor as a `.mert.pt` companion file inside the shard tar.

This is a separate pass after stage 3 ingest so we can:
  1. Ingest data without GPU dependencies.
  2. Iterate on the embedder choice (95M vs 330M, layer weighting) without
     re-ingesting raw audio.
  3. Run the embedder on GPU at maximum throughput, batched across tracks.

Output format:
  Each example_id gets a new entry `{example_id}.mert.pt` in the shard,
  containing a torch tensor of shape (N_tracks, embed_dim) float16.
  Original .flac and .json files are preserved.

Output is written to a parallel directory (`<input_dir>_with_mert/`) by
default to avoid mutating shards in place.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_flac_to_tensor(buf: bytes, target_fs: int = 48000) -> torch.Tensor:
    """Decode FLAC bytes to (T,) float32 at target_fs (mono, mean-mixed)."""
    data, src_fs = sf.read(io.BytesIO(buf), dtype="float32", always_2d=True)
    assert src_fs == target_fs, f"expected {target_fs}, got {src_fs}"
    # mean-mix to mono
    return torch.from_numpy(data.mean(axis=1))


def _embed_shard(
    src_tar: Path, dst_tar: Path,
    embedder, batch_size: int, device: torch.device,
) -> tuple[int, int]:
    """Read src_tar, compute MERT for every track.flac, write enriched dst_tar.

    Returns (n_examples_processed, n_tracks_embedded)."""
    # Group files in tar by example_id
    examples: dict[str, dict[str, bytes]] = {}
    with tarfile.open(src_tar, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            if "." not in name:
                continue
            example_id, _, ext = name.partition(".")
            buf = tar.extractfile(member).read()
            examples.setdefault(example_id, {})[ext] = buf

    n_processed = 0
    n_tracks_total = 0
    out_members: list[tuple[str, bytes, int]] = []  # (name, bytes, mtime)

    # Process examples in alphabetical order (deterministic)
    for example_id in sorted(examples.keys()):
        files = examples[example_id]
        # Find track flac files
        track_keys = sorted(k for k in files if k.startswith("track_") and k.endswith(".flac"))
        if not track_keys:
            # Pass through any non-bundle examples unchanged
            for ext, buf in files.items():
                out_members.append((f"{example_id}.{ext}", buf, int(time.time())))
            continue

        # Decode all stems for this example to mono 48k tensors
        stems = []
        for k in track_keys:
            try:
                t = _load_flac_to_tensor(files[k])
            except Exception:
                t = None
            stems.append(t)

        # Filter Nones
        valid_idx = [i for i, t in enumerate(stems) if t is not None]
        if not valid_idx:
            continue
        valid_stems = [stems[i] for i in valid_idx]

        # Pad to common length, batch
        max_T = max(s.shape[0] for s in valid_stems)
        batch = torch.zeros(len(valid_stems), max_T, dtype=torch.float32)
        for i, s in enumerate(valid_stems):
            batch[i, :s.shape[0]] = s
        batch = batch.to(device)

        # Run in chunks to fit GPU memory
        emb_dim = embedder.embed_dim
        emb_full = torch.zeros(len(track_keys), emb_dim, dtype=torch.float16, device="cpu")
        for chunk_start in range(0, batch.shape[0], batch_size):
            chunk = batch[chunk_start : chunk_start + batch_size]
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
                emb = embedder(chunk)        # (chunk, embed_dim)
            emb = emb.detach().to(torch.float16).cpu()
            for i, idx in enumerate(valid_idx[chunk_start : chunk_start + chunk.shape[0]]):
                emb_full[idx] = emb[i]

        # Serialize embedding to bytes via torch.save
        buf = io.BytesIO()
        torch.save(emb_full, buf)
        files["mert.pt"] = buf.getvalue()
        n_tracks_total += len(track_keys)

        # Add all files for this example
        for ext in sorted(files.keys()):
            out_members.append((f"{example_id}.{ext}", files[ext], int(time.time())))
        n_processed += 1

    # Write the enriched tar atomically
    dst_tar.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_tar.with_suffix(dst_tar.suffix + ".tmp")
    with tarfile.open(tmp, "w") as tar:
        for name, data, mtime in out_members:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = mtime
            tar.addfile(info, io.BytesIO(data))
    tmp.rename(dst_tar)

    return n_processed, n_tracks_total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="dmc-data/stage3_<name>")
    ap.add_argument("--output-dir", default="",
                    help="default: <input-dir>_with_mert")
    ap.add_argument("--model", default="m-a-p/MERT-v1-95M")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="tracks per MERT forward pass")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--layer-weighting", default="uniform", choices=("uniform", "last"))
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else Path(str(in_dir) + "_with_mert")
    in_shards_dir = in_dir / "shards"
    out_shards_dir = out_dir / "shards"
    out_shards_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(in_shards_dir.glob("*.tar"))
    if not shards:
        print(f"no shards under {in_shards_dir}", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    print(f"loading MERT model: {args.model} ({args.layer_weighting} layer weighting)")
    from models.mert_embed import MertEmbedder
    embedder = MertEmbedder(model_name=args.model, layer_weighting=args.layer_weighting)
    embedder.to(device).eval()
    print(f"  embed_dim = {embedder.embed_dim}")

    # Copy non-shard files (manifest, index)
    for f in in_dir.iterdir():
        if f.is_file():
            (out_dir / f.name).write_bytes(f.read_bytes())

    n_examples = 0
    n_tracks = 0
    t0 = time.time()
    pbar = tqdm(total=len(shards), desc="embedding shards")
    for shard in shards:
        out_shard = out_shards_dir / shard.name
        if out_shard.exists():
            pbar.update(1)
            continue
        ne, nt = _embed_shard(shard, out_shard, embedder, args.batch_size, device)
        n_examples += ne
        n_tracks += nt
        pbar.update(1)
    pbar.close()
    elapsed = time.time() - t0
    print(f"done. {n_examples} examples, {n_tracks} tracks embedded in {elapsed/60.0:.1f}m")
    print(f"  ~{n_tracks/max(elapsed,1):.1f} tracks/sec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
