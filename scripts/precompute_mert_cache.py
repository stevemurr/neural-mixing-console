"""Pre-compute MERT embeddings per source track (full-length), cache to disk.

Strategy: a track's MERT embedding (instrument + timbre + style) is roughly
constant across its 6-second segments — a kick is a kick throughout a song.
So we embed once per source track (full length, mean-pooled over time) and
reuse the same vector across every segment that contains that track.

Embeds across both Cambridge-MT and Slakh source trees, writing one
.npz file per session under `dmc-data/mert_cache/<dataset>/<session>.npz`.
Each .npz contains:
  - filenames: array of stem filenames in this session
  - embeddings: (N, embed_dim) float16 array

The stage 3 dataset loader maps each segment's track filename to its
corresponding cached embedding at training time.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.signal as ss
import soundfile as sf
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.mert_embed import MertEmbedder


_MERT_NATIVE_SR = 24_000
_MAX_SECONDS_PER_TRACK = 60.0   # cap to keep memory reasonable; full track embedding is roughly time-pooled regardless


def _load_track_to_24k_mono(path: Path) -> np.ndarray | None:
    """Load any audio, resample to 24 kHz mono."""
    try:
        data, src_fs = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        return None
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    data = data[:, 0]
    if src_fs != _MERT_NATIVE_SR:
        from math import gcd
        g = gcd(int(src_fs), _MERT_NATIVE_SR)
        up, down = _MERT_NATIVE_SR // g, int(src_fs) // g
        data = ss.resample_poly(data, up, down).astype(np.float32, copy=False)
    # Cap length
    max_samples = int(_MAX_SECONDS_PER_TRACK * _MERT_NATIVE_SR)
    if data.shape[0] > max_samples:
        # Take a centered window
        start = (data.shape[0] - max_samples) // 2
        data = data[start : start + max_samples]
    return data


def _collect_cambridge_tracks(
    cambridge_root: Path,
    refmix_json: Path | None = None,
) -> dict[str, list[Path]]:
    """Walk cambridge-mt and gather (session -> [stem_paths]).

    Uses cambridge_reference_mixes.json's `content_root_rel` map when
    available, since some sessions place stems in a non-standard subfolder
    (neither at session root nor at session/session)."""
    import json as _json
    content_root_map: dict[str, str] = {}
    if refmix_json and refmix_json.exists():
        with open(refmix_json) as f:
            for r in _json.load(f):
                content_root_map[r["session"]] = r["content_root_rel"]

    out: dict[str, list[Path]] = {}
    for sess_dir in sorted(cambridge_root.iterdir()):
        if not sess_dir.is_dir():
            continue

        # First try the JSON-specified content_root, then fall back to common
        # layouts (top-level, doubled-folder), then any subfolder with .wav
        # files (handles case-mismatch and other irregular layouts).
        candidates: list[Path] = []
        cr = content_root_map.get(sess_dir.name)
        if cr is not None:
            if cr == ".":
                candidates.append(sess_dir)
            else:
                candidates.append(sess_dir / cr)
        candidates.extend([sess_dir, sess_dir / sess_dir.name])

        # Add any direct subdirectory of sess_dir as a fallback (case-
        # insensitive match for cr or just any subfolder containing wavs).
        for sub in sorted(sess_dir.iterdir()) if sess_dir.is_dir() else []:
            if sub.is_dir() and sub not in candidates:
                candidates.append(sub)

        seen: set[Path] = set()
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            if cand.exists() and cand.is_dir():
                stems = sorted(p for p in cand.iterdir()
                               if p.suffix.lower() == ".wav" and "preview" not in p.name.lower())
                if stems:
                    out[sess_dir.name] = stems
                    break
    return out


def _collect_slakh_tracks(slakh_root: Path, splits: list[str]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for s in splits:
        split_dir = slakh_root / s
        if not split_dir.exists():
            continue
        for track_dir in sorted(split_dir.iterdir()):
            stems_dir = track_dir / "stems"
            if not stems_dir.is_dir():
                continue
            stems = sorted(p for p in stems_dir.iterdir() if p.suffix.lower() == ".flac")
            if stems:
                out[track_dir.name] = stems
    return out


@torch.no_grad()
def _embed_session(
    embedder: MertEmbedder, stems: list[Path],
    device: torch.device, batch_size: int = 8,
) -> tuple[list[str], np.ndarray]:
    """Embed all stems of a session. Returns (filenames, (N, D) float16 array)."""
    valid_audio: list[np.ndarray] = []
    valid_names: list[str] = []
    for p in stems:
        audio = _load_track_to_24k_mono(p)
        if audio is None or audio.size == 0 or float(np.sqrt(np.mean(audio ** 2) + 1e-12)) < 1e-4:
            continue
        valid_audio.append(audio)
        valid_names.append(p.name)

    if not valid_audio:
        return [], np.zeros((0, embedder.embed_dim), dtype=np.float16)

    embed_dim = embedder.embed_dim
    out = np.zeros((len(valid_audio), embed_dim), dtype=np.float16)

    # MERT was trained at 24 kHz; the resampler in MertEmbedder expects 48 kHz
    # input. We bypass the resampler here by upsampling to 48 kHz ourselves
    # (since we stored at 24 kHz for cheap I/O) ... actually simpler: feed
    # the model directly at 24 kHz, skipping resampler. Construct a thin
    # wrapper that calls the underlying HF model directly.
    for i in range(0, len(valid_audio), batch_size):
        chunk = valid_audio[i : i + batch_size]
        max_T = max(a.shape[0] for a in chunk)
        batch = torch.zeros(len(chunk), max_T, dtype=torch.float32, device=device)
        for j, a in enumerate(chunk):
            batch[j, :a.shape[0]] = torch.from_numpy(a).to(device)
        # Bypass resampler by calling model directly (input is already 24 kHz)
        attn = torch.ones_like(batch, dtype=torch.long)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
            outputs = embedder.model(input_values=batch, attention_mask=attn,
                                     output_hidden_states=True)
            hs = outputs.hidden_states
            if embedder.layer_weighting == "last":
                pooled = hs[-1].mean(dim=1)
            else:
                pooled = torch.stack(hs, dim=0).mean(dim=0).mean(dim=1)
        emb = pooled.detach().to(torch.float16).cpu().numpy()
        out[i : i + len(chunk)] = emb

    return valid_names, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cambridge-root", default="source_audio/cambridge-mt")
    ap.add_argument("--cambridge-refmix-json",
                    default="source_audio/cambridge_reference_mixes.json")
    ap.add_argument("--slakh-root", default="source_audio/slakh2100")
    ap.add_argument("--slakh-splits", nargs="*", default=["train", "validation", "test"])
    ap.add_argument("--cache-dir", default="dmc-data/mert_cache")
    ap.add_argument("--model", default="m-a-p/MERT-v1-95M")
    ap.add_argument("--layer-weighting", default="uniform", choices=("uniform", "last"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--datasets", nargs="*", default=["cambridge", "slakh"],
                    choices=("cambridge", "slakh"))
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"loading MERT: {args.model} ({args.layer_weighting} pooling)")
    embedder = MertEmbedder(model_name=args.model, layer_weighting=args.layer_weighting).to(device).eval()
    print(f"  embed_dim = {embedder.embed_dim}")

    cache_root = Path(args.cache_dir)

    if "cambridge" in args.datasets:
        cambridge_sessions = _collect_cambridge_tracks(
            Path(args.cambridge_root),
            refmix_json=Path(args.cambridge_refmix_json),
        )
        print(f"\ncambridge: {len(cambridge_sessions)} sessions")
        out_dir = cache_root / "cambridge"
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        n_sessions_done = 0
        for sess, stems in tqdm(cambridge_sessions.items(), desc="cambridge"):
            cache_path = out_dir / f"{sess}.npz"
            if cache_path.exists():
                n_sessions_done += 1
                continue
            names, embs = _embed_session(embedder, stems, device, args.batch_size)
            np.savez_compressed(cache_path, filenames=np.array(names), embeddings=embs)
            n_sessions_done += 1
        print(f"  done in {(time.time() - t0) / 60:.1f}m")

    if "slakh" in args.datasets:
        slakh_sessions = _collect_slakh_tracks(Path(args.slakh_root), args.slakh_splits)
        print(f"\nslakh: {len(slakh_sessions)} tracks")
        out_dir = cache_root / "slakh"
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for sess, stems in tqdm(slakh_sessions.items(), desc="slakh"):
            cache_path = out_dir / f"{sess}.npz"
            if cache_path.exists():
                continue
            names, embs = _embed_session(embedder, stems, device, args.batch_size)
            np.savez_compressed(cache_path, filenames=np.array(names), embeddings=embs)
        print(f"  done in {(time.time() - t0) / 60:.1f}m")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
