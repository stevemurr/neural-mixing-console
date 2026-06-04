"""Ingest Cambridge-MT real multitracks + reference mixes into stage 3 shards.

Joins each multitrack folder under `source_audio/cambridge-mt/<session>/` with
its matching reference mix wav under `source_audio/wavs/<song>_Full_Preview.wav`.

For each (mix, stems) pair this:

  1. Loads mix + all stems, resamples to 48 kHz, broadcasts mono → stereo.
  2. Verifies time alignment via envelope cross-correlation between sum(stems)
     and the mix; shifts stems by the best-correlation lag (typically a few
     hundred samples — the mix is engineer-trimmed but the stems within a
     session are tightly aligned to each other).
  3. Trims stems and mix to a common length post-shift.
  4. Tags each stem with a coarse instrument label inferred from the filename
     keywords (kick / snare / vox / guitar / bass / piano / synth / fx / etc.).
     Used only for synthesis-time priors; the model itself consumes MERT
     embeddings, not labels.
  5. Slices into 6-second segments aligned across stems and mix.
  6. Writes a webdataset tar shard with per-segment {tracks, mix, meta}.

The MERT embedding pass is a separate script (scripts/embed_stage3_mert.py)
that reads these shards and emits a parallel tensor file per segment.

Multiprocessing-enabled via `--workers N`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    ShardWriter, derive_example_id, encode_flac_bytes, json_dumps,
    jsonl_to_parquet, run_parallel_generation, split_for, write_manifest,
)


FS = 48_000
DEFAULT_SEGMENT_SECONDS = 6.0  # overridable via --segment-seconds
ENVELOPE_HOP_MS = 50.0  # for cross-corr alignment
ENVELOPE_HOP = int(FS * ENVELOPE_HOP_MS / 1000.0)
ALIGN_MAX_SHIFT_SECONDS = 5.0  # reject if best-shift abs > this
ALIGN_MIN_CORRELATION = 0.5    # reject if peak correlation below this


# Instrument-label keyword map. First match wins. Coarse but useful for
# synthesis-time priors (what kind of effect chain is typical for this stem).
# Use a permissive "leading boundary" that includes _ and digits, since
# Cambridge-MT filenames typically prefix with "NN_KEYWORD" patterns where
# Python's \b doesn't trigger between underscore and letter (both are \w).
_LB = r"(?:^|[^a-zA-Z])"   # match at start or after a non-letter char

_INSTR_PATTERNS = [
    ("kick",          _LB + r"(?:kick|kik|kk(?=[^a-z])|bd(?=[^a-z]))|bass[\s\-_]?drum"),
    ("snare",         _LB + r"(?:snare|snr|sd(?=[^a-z]))"),
    ("hat",           _LB + r"(?:hat|hh(?=[^a-z]))|hi[\s\-_]?hat"),
    ("cymbal",        _LB + r"(?:ride|crash)|cymbal"),
    ("tom",           _LB + r"tom|hi[\s\-_]?tom|low[\s\-_]?tom|floor[\s\-_]?tom|rack[\s\-_]?tom"),
    ("drum_overhead", _LB + r"(?:overhead|oh[lr]?(?=[^a-z]))|drum[\s\-_]?room"),
    ("drum_other",    _LB + r"(?:drum|drumkit|perc|bongo|conga|tabla|shaker|tambourine|cowbell|clap)"),
    ("bass_synth",    r"sub[\s\-_]?bass|synth[\s\-_]?bass"),
    ("bass_electric", _LB + r"(?:bass|bg(?=[^a-z]))|bass[\s\-_]?guitar"),
    ("guitar_acoustic", _LB + r"(?:acoustic|acgtr|acc?(?=[\s\-_]))"),
    ("guitar_electric", _LB + r"(?:eguitar|egtr|eg(?=[\s\-_]))|elec[\s\-_]?guit"),
    ("guitar_other",  _LB + r"(?:guitar|gtr)"),
    ("vocal_lead",    _LB + r"(?:ldvox|ldv|lv(?=[^a-z]))|lead[\s\-_]?(?:vox|vocal)|main[\s\-_]?vox"),
    ("vocal_bg",      _LB + r"(?:bgvox|bgv|bv(?=[^a-z])|harmon|backup)|backing[\s\-_]?vox|chorus[\s\-_]?vox"),
    ("vocal_other",   _LB + r"(?:vox|vocal|voc|sing)"),
    ("piano",         _LB + r"(?:piano|pno)|grand[\s\-_]?piano|upright"),
    ("keys",          _LB + r"(?:keys?|rhodes|wurli|organ|hammond)"),
    ("synth",         _LB + r"(?:synth|pad(?=[^a-z])|arp(?=[^a-z]))"),
    ("brass",         _LB + r"(?:trumpet|trombone|sax|saxophone|horn|brass)"),
    ("strings",       _LB + r"(?:violin|viola|cello|string|fiddle|harp)"),
    ("woodwind",      _LB + r"(?:flute|clarinet|oboe|bassoon)"),
    ("fx",            _LB + r"(?:fx(?=[^a-z])|effect|riser|sweep|whoosh|impact)"),
    ("loop",          _LB + r"loop"),
    ("fill",          _LB + r"fill"),
    ("room",          _LB + r"room"),
]
_INSTR_PATTERNS_COMPILED = [(label, re.compile(pat, re.IGNORECASE)) for label, pat in _INSTR_PATTERNS]


def infer_instrument_label(filename: str) -> str:
    """Map a stem filename to a coarse instrument label. Returns 'unknown' if
    no pattern matches."""
    for label, pat in _INSTR_PATTERNS_COMPILED:
        if pat.search(filename):
            return label
    return "unknown"


def _read_audio_48k_stereo(path: Path) -> np.ndarray:
    """Load WAV, resample to 48 kHz if needed, broadcast mono->stereo. Returns (2, T) float32."""
    data, src_fs = sf.read(str(path), dtype="float32", always_2d=True)
    # data: (T, C)
    if src_fs != FS:
        # scipy.signal.resample_poly is fast and high-quality
        from math import gcd
        g = gcd(int(src_fs), FS)
        up, down = FS // g, int(src_fs) // g
        # resample each channel; transpose for resample_poly which expects (T,) or (..., T)
        data = ss.resample_poly(data, up, down, axis=0).astype(np.float32, copy=False)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        # downmix to stereo by taking first two
        data = data[:, :2]
    return data.T  # (2, T)


def _envelope(x: np.ndarray, hop: int = ENVELOPE_HOP) -> np.ndarray:
    """RMS envelope, mono. x: (2, T) -> (T // hop,)."""
    mono = 0.5 * (x[0] + x[1]) if x.ndim == 2 else x
    n_frames = mono.shape[-1] // hop
    if n_frames == 0:
        return np.zeros(1, dtype=np.float32)
    trunc = mono[..., :n_frames * hop].reshape(-1, hop)
    return np.sqrt(np.mean(trunc ** 2, axis=-1) + 1e-12).astype(np.float32)


def estimate_alignment(stems_sum: np.ndarray, mix: np.ndarray) -> tuple[int, float]:
    """Cross-correlate envelopes to find sample-shift that best aligns
    sum(stems) to mix. Returns (lag_samples, peak_correlation_normalized).
    Positive lag means stems lead the mix (shift stems forward in time)."""
    env_s = _envelope(stems_sum)
    env_m = _envelope(mix)
    # Normalize for stable correlation
    env_s = env_s - env_s.mean()
    env_m = env_m - env_m.mean()
    norm = (np.linalg.norm(env_s) * np.linalg.norm(env_m)) + 1e-12
    corr = ss.correlate(env_s, env_m, mode="full", method="fft") / norm
    # corr index 0 corresponds to shift = -(len(env_m)-1); index N-1 corresponds to shift=0
    # for `correlate(s, m)`, lags are in [-(len(m)-1), len(s)-1].
    lags_envframes = np.arange(-(len(env_m) - 1), len(env_s))
    # Limit search range
    max_shift_envframes = int(ALIGN_MAX_SHIFT_SECONDS * 1000.0 / ENVELOPE_HOP_MS)
    valid = (np.abs(lags_envframes) <= max_shift_envframes)
    if not valid.any():
        return 0, 0.0
    corr_v = corr[valid]
    lags_v = lags_envframes[valid]
    best = int(np.argmax(corr_v))
    lag_envframes = int(lags_v[best])
    peak = float(corr_v[best])
    lag_samples = lag_envframes * ENVELOPE_HOP
    return lag_samples, peak


def _shift_and_trim(stems: list[np.ndarray], mix: np.ndarray, lag: int) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Apply lag to align stems with mix, then trim both to common length.

    lag > 0: stems lead the mix → drop `lag` samples from the front of stems
             (or equivalently pad mix with `lag` zeros at front).
    lag < 0: mix leads stems → drop `-lag` samples from the front of mix.

    Returns (aligned_stems, aligned_mix, common_length)."""
    if lag > 0:
        stems = [s[..., lag:] for s in stems]
    elif lag < 0:
        mix = mix[..., -lag:]
    common_T = min(min(s.shape[-1] for s in stems), mix.shape[-1])
    stems = [s[..., :common_T] for s in stems]
    mix = mix[..., :common_T]
    return stems, mix, common_T


_W: dict = {}


def _worker_init(pairs_serialized: str, master_seed: int, max_stems: int,
                 segments_stride: float, accept_min_corr: float,
                 segment_seconds: float = DEFAULT_SEGMENT_SECONDS):
    pairs = json.loads(pairs_serialized)
    _W["pairs"] = pairs
    _W["master_seed"] = int(master_seed)
    _W["max_stems"] = int(max_stems)
    _W["segments_stride"] = float(segments_stride)
    _W["accept_min_corr"] = float(accept_min_corr)
    _W["segment_len"] = int(FS * float(segment_seconds))


def _worker_fn(pair_index: int) -> dict | None:
    """Process one (mix, multitrack_dir) pair → dict with bundle list."""
    pairs = _W["pairs"]
    if pair_index >= len(pairs):
        return None
    pair = pairs[pair_index]
    session = pair["session"]
    mix_path = Path(pair["mix"])
    mt_dir = Path(pair["mt_dir"])
    master_seed = _W["master_seed"]
    max_stems = _W["max_stems"]
    stride_seconds = _W["segments_stride"]

    if not mix_path.exists() or not mt_dir.exists():
        return None

    # Load mix
    try:
        mix = _read_audio_48k_stereo(mix_path)
    except Exception as e:
        return {"session": session, "error": f"mix_read: {e}", "bundles": []}

    # Load stems
    stem_files = sorted(p for p in mt_dir.iterdir()
                        if p.suffix.lower() == ".wav" and "preview" not in p.name.lower())
    if not stem_files:
        return {"session": session, "error": "no_stems", "bundles": []}
    if len(stem_files) > max_stems:
        # Drop stems by deterministic-by-filename order; we keep the first
        # max_stems alphabetically to ensure reproducibility.
        stem_files = stem_files[:max_stems]

    stems = []
    stem_meta = []
    for sf_path in stem_files:
        try:
            audio = _read_audio_48k_stereo(sf_path)
        except Exception:
            continue
        # Skip mostly-silent stems (likely uninteresting)
        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
        if rms < 1e-4:
            continue
        stems.append(audio)
        stem_meta.append({
            "filename": sf_path.name,
            "instrument_label": infer_instrument_label(sf_path.name),
            "rms_db": float(20.0 * np.log10(rms + 1e-12)),
        })

    if not stems:
        return {"session": session, "error": "all_stems_silent_or_unreadable", "bundles": []}

    # Pad stems to common length so we can sum them
    max_T = max(s.shape[-1] for s in stems)
    for i in range(len(stems)):
        if stems[i].shape[-1] < max_T:
            pad = np.zeros((2, max_T - stems[i].shape[-1]), dtype=np.float32)
            stems[i] = np.concatenate([stems[i], pad], axis=-1)
    stems_sum = np.stack(stems).sum(axis=0)

    # Estimate alignment vs mix
    lag, corr = estimate_alignment(stems_sum, mix)
    if corr < _W["accept_min_corr"]:
        return {"session": session, "error": f"alignment_corr_too_low ({corr:.3f})", "bundles": []}

    # Apply lag and trim
    stems, mix, common_T = _shift_and_trim(stems, mix, lag)

    # Segment into stride-seconds chunks
    SEGMENT_LEN = _W["segment_len"]
    stride = int(FS * stride_seconds)
    bundles = []
    seg_idx = 0
    for start in range(0, common_T - SEGMENT_LEN + 1, stride):
        end = start + SEGMENT_LEN
        seg_mix = mix[:, start:end]
        seg_stems = [s[:, start:end] for s in stems]

        # Skip if mix is mostly silent in this segment
        mix_rms = float(np.sqrt(np.mean(seg_mix ** 2) + 1e-12))
        if mix_rms < 1e-3:
            continue

        example_id = derive_example_id(master_seed, "stage3_cambridge", f"{session}_{seg_idx:04d}")

        # Encode files
        files: dict[str, bytes] = {
            "mix.flac": encode_flac_bytes(seg_mix, FS),
        }
        for k_idx, (s, m) in enumerate(zip(seg_stems, stem_meta)):
            files[f"track_{k_idx:03d}.flac"] = encode_flac_bytes(s, FS)

        # Per-segment metadata
        meta = {
            "example_id": example_id,
            "stage": "stage3_cambridge",
            "session": session,
            "segment_index": seg_idx,
            "segment_start_seconds": start / FS,
            "alignment_lag_samples": int(lag),
            "alignment_correlation": float(corr),
            "track_count": len(seg_stems),
            "duration_samples": SEGMENT_LEN,
            "sample_rate": FS,
            "tracks": stem_meta,
            "split": split_for(session),
        }
        files["meta.json"] = json_dumps(meta)

        bundles.append({"example_id": example_id, "files": files, "row": meta})
        seg_idx += 1

    return {"session": session, "lag": int(lag), "corr": float(corr), "bundles": bundles}


def _build_pairs(map_tsv: Path, refmix_json: Path, mt_root: Path, wavs_root: Path) -> list[dict]:
    content_root_map = {}
    with open(refmix_json) as f:
        for r in json.load(f):
            content_root_map[r["session"]] = r["content_root_rel"]

    pairs = []
    seen_sessions = set()
    with open(map_tsv) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            session = r["session"]
            if session in seen_sessions:
                continue
            seen_sessions.add(session)
            mix_name = r["filename"].replace(".mp3", ".wav")
            mix_path = wavs_root / mix_name
            content_root = content_root_map.get(session, ".")
            mt_dir = mt_root / session if content_root == "." else mt_root / session / content_root
            if mix_path.exists() and mt_dir.exists():
                pairs.append({
                    "session": session,
                    "mix": str(mix_path),
                    "mt_dir": str(mt_dir),
                })
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cambridge-root", default="source_audio/cambridge-mt")
    ap.add_argument("--mix-wavs-root", default="source_audio/wavs")
    ap.add_argument("--destination-map", default="source_audio/cambridge_preview_destination_map.tsv")
    ap.add_argument("--reference-mixes", default="source_audio/cambridge_reference_mixes.json")
    ap.add_argument("--output-dir", default="dmc-data/stage3_cambridge")
    ap.add_argument("--master-seed", type=int, default=20260506)
    ap.add_argument("--examples-per-shard", type=int, default=200)
    ap.add_argument("--max-stems-per-song", type=int, default=80)
    ap.add_argument("--segment-stride-seconds", type=float, default=6.0,
                    help="6s = no overlap; 3s = 50%% overlap")
    ap.add_argument("--alignment-min-corr", type=float, default=0.5)
    ap.add_argument("--segment-seconds", type=float, default=DEFAULT_SEGMENT_SECONDS,
                    help="duration of each output segment in seconds (default 6s)")
    ap.add_argument("--workers", type=int, default=8,
                    help="Each worker peaks ~3-4 GB RAM during stem loading; tune to your box.")
    ap.add_argument("--chunksize", type=int, default=1)
    args = ap.parse_args()

    pairs = _build_pairs(
        Path(args.destination_map), Path(args.reference_mixes),
        Path(args.cambridge_root), Path(args.mix_wavs_root),
    )
    print(f"discovered {len(pairs)} valid mix↔multitrack pairs")
    if not pairs:
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(out_dir, prefix="stage3_cambridge", examples_per_shard=args.examples_per_shard)
    print(f"resuming from {writer.completed_count} segments already written")

    pairs_serialized = json.dumps(pairs)
    stats = run_parallel_generation(
        n_examples=len(pairs),
        writer=writer,
        worker_init_fn=_worker_init,
        worker_init_args=(pairs_serialized, args.master_seed, args.max_stems_per_song,
                          args.segment_stride_seconds, args.alignment_min_corr,
                          args.segment_seconds),
        worker_fn=_worker_fn,
        n_workers=args.workers,
        chunksize=args.chunksize,
        desc="stage3.cambridge",
        unpack_bundles=True,
        skip_resume=True,
    )

    writer.close()
    n = jsonl_to_parquet(writer.index_path, out_dir / "stage3_cambridge_index.parquet")
    print(f"finalized {n} segments  (written this run: {stats['completed']}, skipped: {stats['skipped']})")
    write_manifest(
        out_dir, dataset="stage3_cambridge", master_seed=args.master_seed, n_examples=n,
        extra={
            "n_sessions": len(pairs),
            "fs": FS, "segment_seconds": args.segment_seconds,
            "stride_seconds": args.segment_stride_seconds,
            "alignment_min_corr": args.alignment_min_corr,
            "max_stems_per_song": args.max_stems_per_song,
            "workers": args.workers,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
