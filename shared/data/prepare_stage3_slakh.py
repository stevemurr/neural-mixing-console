"""Ingest Slakh2100 multitracks + mix into stage 3 enrichment shards.

Slakh is MIDI-rendered, so:
  - No alignment check needed (mix and stems are bit-aligned by construction).
  - Per-stem instrument labels come from `metadata.yaml` (inst_class +
    midi_program_name), not from filename heuristics.

We cap at `--max-tracks` songs total (default 300) to keep the enrichment set
roughly balanced with the Cambridge-MT corpus and within disk budget.
Sampling is deterministic (seeded shuffle, then take first N).

Multiprocessing-enabled via `--workers N`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    ShardWriter, derive_example_id, encode_flac_bytes, json_dumps,
    jsonl_to_parquet, run_parallel_generation, split_for, write_manifest,
)


FS = 48_000
DEFAULT_SEGMENT_SECONDS = 6.0  # overridable via --segment-seconds


# Slakh's `inst_class` taxonomy → coarse labels matching Cambridge labels.
_SLAKH_CLASS_TO_LABEL = {
    "Acoustic Guitar":  "guitar_acoustic",
    "Bass":             "bass_electric",
    "Brass":            "brass",
    "Chromatic Percussion": "drum_other",
    "Clean Guitar":     "guitar_electric",
    "Distorted Guitar": "guitar_electric",
    "Drums":            "drum_other",      # Slakh drums are full-kit, not split
    "Ethnic":           "strings",
    "Guitar":           "guitar_electric",
    "Organ":            "keys",
    "Percussive":       "drum_other",
    "Piano":            "piano",
    "Pipe":             "woodwind",
    "Reed":             "woodwind",
    "Strings":          "strings",
    "Synth Effects":    "fx",
    "Synth Lead":       "synth",
    "Synth Pad":        "synth",
    "Synth Bass":       "bass_synth",
    "Vocal":            "vocal_other",
    "Sound Effects":    "fx",
    "World":            "strings",
}


def _read_audio_48k_stereo(path: Path) -> np.ndarray:
    """Load (FLAC or WAV), resample to 48 kHz, broadcast mono->stereo."""
    data, src_fs = sf.read(str(path), dtype="float32", always_2d=True)
    if src_fs != FS:
        from math import gcd
        g = gcd(int(src_fs), FS)
        up, down = FS // g, int(src_fs) // g
        data = ss.resample_poly(data, up, down, axis=0).astype(np.float32, copy=False)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    return data.T


def _slakh_label_for_stem(stem_meta: dict) -> str:
    """Map Slakh metadata's inst_class to our coarse label vocabulary."""
    inst_class = stem_meta.get("inst_class", "")
    label = _SLAKH_CLASS_TO_LABEL.get(inst_class, "unknown")
    # Refine using midi_program_name if available
    program = stem_meta.get("midi_program_name", "").lower()
    if "kick" in program or "bass drum" in program:
        return "kick"
    if "snare" in program:
        return "snare"
    if "hi-hat" in program or "hihat" in program:
        return "hat"
    if "tom" in program:
        return "tom"
    if "cymbal" in program:
        return "cymbal"
    if "lead vocal" in program:
        return "vocal_lead"
    if "background vocal" in program or "backing vocal" in program:
        return "vocal_bg"
    return label


_W: dict = {}


def _worker_init(track_paths_serialized: str, master_seed: int, max_stems: int,
                 segments_stride: float,
                 segment_seconds: float = DEFAULT_SEGMENT_SECONDS):
    track_paths = json.loads(track_paths_serialized)
    _W["track_paths"] = track_paths
    _W["master_seed"] = int(master_seed)
    _W["max_stems"] = int(max_stems)
    _W["segments_stride"] = float(segments_stride)
    _W["segment_seconds"] = float(segment_seconds)
    _W["segment_len"] = int(FS * float(segment_seconds))


def _worker_fn(track_index: int) -> dict | None:
    track_paths = _W["track_paths"]
    if track_index >= len(track_paths):
        return None
    track_dir = Path(track_paths[track_index])
    if not track_dir.exists():
        return None
    master_seed = _W["master_seed"]
    max_stems = _W["max_stems"]
    stride_seconds = _W["segments_stride"]

    metadata_path = track_dir / "metadata.yaml"
    mix_path = track_dir / "mix.flac"
    stems_dir = track_dir / "stems"
    if not (metadata_path.exists() and mix_path.exists() and stems_dir.exists()):
        return {"session": track_dir.name, "error": "missing files", "bundles": []}

    with open(metadata_path) as f:
        meta_yaml = yaml.safe_load(f)

    stems_meta = meta_yaml.get("stems", {})
    stem_files = sorted(p for p in stems_dir.iterdir() if p.suffix.lower() == ".flac")
    if not stem_files:
        return {"session": track_dir.name, "error": "no stems", "bundles": []}
    if len(stem_files) > max_stems:
        stem_files = stem_files[:max_stems]

    # Load mix
    try:
        mix = _read_audio_48k_stereo(mix_path)
    except Exception as e:
        return {"session": track_dir.name, "error": f"mix_read: {e}", "bundles": []}

    # Load stems with metadata
    stems = []
    stem_rec_meta = []
    for sf_path in stem_files:
        stem_id = sf_path.stem  # e.g. "S00"
        stem_meta = stems_meta.get(stem_id, {})
        if not stem_meta.get("audio_rendered", False):
            continue
        try:
            audio = _read_audio_48k_stereo(sf_path)
        except Exception:
            continue
        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
        if rms < 1e-4:
            continue
        stems.append(audio)
        stem_rec_meta.append({
            "filename": sf_path.name,
            "stem_id": stem_id,
            "instrument_label": _slakh_label_for_stem(stem_meta),
            "inst_class": stem_meta.get("inst_class"),
            "midi_program_name": stem_meta.get("midi_program_name"),
            "is_drum": bool(stem_meta.get("is_drum", False)),
            "rms_db": float(20.0 * np.log10(rms + 1e-12)),
        })

    if not stems:
        return {"session": track_dir.name, "error": "all stems rejected", "bundles": []}

    # Pad stems to common length (mix is the reference)
    common_T = min(min(s.shape[-1] for s in stems), mix.shape[-1])
    stems = [s[:, :common_T] for s in stems]
    mix = mix[:, :common_T]

    # Segment
    SEGMENT_LEN = _W["segment_len"]
    stride = int(FS * stride_seconds)
    bundles = []
    seg_idx = 0
    for start in range(0, common_T - SEGMENT_LEN + 1, stride):
        end = start + SEGMENT_LEN
        seg_mix = mix[:, start:end]
        seg_stems = [s[:, start:end] for s in stems]

        mix_rms = float(np.sqrt(np.mean(seg_mix ** 2) + 1e-12))
        if mix_rms < 1e-3:
            continue

        example_id = derive_example_id(master_seed, "stage3_slakh", f"{track_dir.name}_{seg_idx:04d}")

        files: dict[str, bytes] = {"mix.flac": encode_flac_bytes(seg_mix, FS)}
        for k_idx, s in enumerate(seg_stems):
            files[f"track_{k_idx:03d}.flac"] = encode_flac_bytes(s, FS)

        meta = {
            "example_id": example_id,
            "stage": "stage3_slakh",
            "session": track_dir.name,
            "segment_index": seg_idx,
            "segment_start_seconds": start / FS,
            "alignment_lag_samples": 0,
            "alignment_correlation": 1.0,  # MIDI-rendered, exact
            "track_count": len(seg_stems),
            "duration_samples": SEGMENT_LEN,
            "sample_rate": FS,
            "tracks": stem_rec_meta,
            "split": split_for(track_dir.name),
        }
        files["meta.json"] = json_dumps(meta)
        bundles.append({"example_id": example_id, "files": files, "row": meta})
        seg_idx += 1

    return {"session": track_dir.name, "bundles": bundles}


def _collect_track_paths(slakh_root: Path, splits: list[str]) -> list[Path]:
    out = []
    for s in splits:
        split_dir = slakh_root / s
        if not split_dir.exists():
            continue
        for d in sorted(split_dir.iterdir()):
            if d.is_dir() and (d / "mix.flac").exists():
                out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slakh-root", default="source_audio/slakh2100")
    ap.add_argument("--splits", nargs="*", default=["train", "validation", "test"])
    ap.add_argument("--output-dir", default="dmc-data/stage3_slakh")
    ap.add_argument("--master-seed", type=int, default=20260506)
    ap.add_argument("--examples-per-shard", type=int, default=200)
    ap.add_argument("--max-tracks", type=int, default=300,
                    help="Cap total Slakh tracks ingested to balance with Cambridge & disk budget.")
    ap.add_argument("--max-stems-per-song", type=int, default=80)
    ap.add_argument("--segment-stride-seconds", type=float, default=6.0)
    ap.add_argument("--segment-seconds", type=float, default=DEFAULT_SEGMENT_SECONDS,
                    help="duration of each output segment in seconds (default 6s)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunksize", type=int, default=1)
    ap.add_argument("--shuffle-seed", type=int, default=20260506,
                    help="Seed for deterministic track sampling when --max-tracks limits the set.")
    args = ap.parse_args()

    track_paths = _collect_track_paths(Path(args.slakh_root), args.splits)
    print(f"discovered {len(track_paths)} valid Slakh tracks across {args.splits}")

    if args.max_tracks > 0 and len(track_paths) > args.max_tracks:
        rng = random.Random(args.shuffle_seed)
        track_paths_sampled = list(track_paths)
        rng.shuffle(track_paths_sampled)
        track_paths = sorted(track_paths_sampled[:args.max_tracks])
        print(f"sampled to {len(track_paths)} tracks (seed={args.shuffle_seed})")

    if not track_paths:
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(out_dir, prefix="stage3_slakh", examples_per_shard=args.examples_per_shard)
    print(f"resuming from {writer.completed_count} segments already written")

    track_paths_serialized = json.dumps([str(p) for p in track_paths])
    stats = run_parallel_generation(
        n_examples=len(track_paths),
        writer=writer,
        worker_init_fn=_worker_init,
        worker_init_args=(track_paths_serialized, args.master_seed,
                          args.max_stems_per_song, args.segment_stride_seconds,
                          args.segment_seconds),
        worker_fn=_worker_fn,
        n_workers=args.workers,
        chunksize=args.chunksize,
        desc="stage3.slakh",
        unpack_bundles=True,
        skip_resume=True,
    )

    writer.close()
    n = jsonl_to_parquet(writer.index_path, out_dir / "stage3_slakh_index.parquet")
    print(f"finalized {n} segments  (written this run: {stats['completed']}, skipped: {stats['skipped']})")
    write_manifest(
        out_dir, dataset="stage3_slakh", master_seed=args.master_seed, n_examples=n,
        extra={
            "n_tracks": len(track_paths),
            "splits": args.splits, "max_tracks": args.max_tracks,
            "fs": FS, "segment_seconds": args.segment_seconds,
            "stride_seconds": args.segment_stride_seconds,
            "max_stems_per_song": args.max_stems_per_song,
            "workers": args.workers,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
