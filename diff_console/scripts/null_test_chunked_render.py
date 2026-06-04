"""Null-test driver: chunked render vs single-shot render of infer.py.

The chunked render path (added to bound DiffReverb's `next_pow2(2T)` memory
on long songs) should produce output that's audibly identical to the
original full-song render. Differences are expected:

  * The DiffReverb / DiffDelay FSM evaluate the transfer function on a
    frequency grid set by `next_pow2(2T)`. A chunk's grid differs from the
    full-song grid, so even the steady-state body of a chunk will not be
    bit-equal to the full-render baseline. Body-region null is typically
    around -60 to -40 dB.
  * Non-first chunks lose the contribution of audio more than `overlap`
    seconds in the past. With overlap >= reverb decay_time, that contribution
    is below the decay floor and inaudible. With overlap < decay_time, you
    will see seam artefacts.
  * The crossfade at each seam is a brief mix of two slightly-different
    signals; expect a small dB bump localized to the crossfade region.

What this script reports:
  - shape match
  - peak / RMS of (chunked - baseline)
  - per-second RMS profile of the difference (so seam discrepancies show)
  - body-region null depth (samples away from any seam)
  - seam-region null depth (samples within ±50 ms of any seam)
  - verdict against `--threshold-db` (default -40 dB body, -25 dB seam)

Usage:
    python scripts/null_test_chunked_render.py \\
        --checkpoint dmc-data/checkpoints/stage3/mix_encoder_latest.pt \\
        --tracks-dir source_audio/Wolf \\
        --duration 25

Exit code 0 = both null thresholds pass, 1 = at least one fails.

Designed for short truncated sessions where the single-shot render fits in
memory. Don't point this at a 4-minute song with 21 stems — the baseline
run is exactly the configuration that crashed the box originally.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


SAMPLE_RATE = 48000


def _truncate_stems(src_dir: Path, dst_dir: Path, duration_s: float) -> int:
    """Copy each audio file in src_dir into dst_dir, trimmed to duration_s.

    Uses soundfile so we don't depend on an external sox/ffmpeg binary.
    Returns the number of stems written.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    exts = (".wav", ".flac", ".aif", ".aiff", ".mp3")
    n = 0
    for p in sorted(src_dir.iterdir()):
        if p.suffix.lower() not in exts:
            continue
        if "mixture" in p.name.lower() or "preview" in p.name.lower():
            continue
        if p.name.lower().endswith("_mix.wav"):
            continue
        data, fs = sf.read(str(p), always_2d=True)
        keep = int(round(duration_s * fs))
        data = data[:keep]
        sf.write(str(dst_dir / p.with_suffix(".wav").name), data, fs, subtype="FLOAT")
        n += 1
    return n


def _run_infer(
    py: Path, infer_py: Path, checkpoint: Path, tracks_dir: Path, out_dir: Path,
    chunk_seconds: float, overlap_seconds: float, crossfade_seconds: float,
    n_max: int, bpm: float, mert_dim: int, extra: list[str] | None = None,
) -> None:
    cmd = [
        str(py), str(infer_py),
        "--checkpoint", str(checkpoint),
        "--tracks-dir", str(tracks_dir),
        "--out-dir", str(out_dir),
        "--render-chunk-seconds", str(chunk_seconds),
        "--render-overlap-seconds", str(overlap_seconds),
        "--render-crossfade-seconds", str(crossfade_seconds),
        "--n-max", str(n_max),
        "--bpm", str(bpm),
        "--mert-dim", str(mert_dim),
    ]
    if extra:
        cmd.extend(extra)
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _db(x: np.ndarray, eps: float = 1e-12) -> float:
    return 20.0 * float(np.log10(np.sqrt((x ** 2).mean() + eps) + eps))


def _per_second_profile(diff: np.ndarray, fs: int) -> np.ndarray:
    """diff shape (T, C). Return (n_seconds,) RMS-dB per 1-s window."""
    n_sec = diff.shape[0] // fs
    if n_sec == 0:
        return np.array([_db(diff)])
    out = np.empty(n_sec, dtype=np.float64)
    for i in range(n_sec):
        block = diff[i * fs : (i + 1) * fs]
        out[i] = _db(block)
    return out


def _seam_mask(T: int, fs: int, chunk_seconds: float, half_window_ms: float) -> np.ndarray:
    """Boolean mask of length T marking samples within ±half_window of seams."""
    mask = np.zeros(T, dtype=bool)
    if chunk_seconds <= 0:
        return mask
    chunk_T = int(round(chunk_seconds * fs))
    half = int(round(half_window_ms * 1e-3 * fs))
    seam = chunk_T
    while seam < T:
        lo = max(0, seam - half)
        hi = min(T, seam + half)
        mask[lo:hi] = True
        seam += chunk_T
    return mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tracks-dir", required=True,
                    help="folder of stem files to truncate and run on")
    ap.add_argument("--duration", type=float, default=25.0,
                    help="seconds to truncate each stem to (default 25). Keep "
                         "this small enough that the single-shot baseline run "
                         "fits in memory; the chunked run will obviously fit.")
    ap.add_argument("--chunk-seconds", type=float, default=8.0,
                    help="chunk size for the chunked render (forces multiple "
                         "chunks within `duration`)")
    ap.add_argument("--overlap-seconds", type=float, default=4.0)
    ap.add_argument("--crossfade-seconds", type=float, default=0.025)
    ap.add_argument("--n-max", type=int, default=21)
    ap.add_argument("--bpm", type=float, default=120.0)
    ap.add_argument("--mert-dim", type=int, default=768)
    ap.add_argument("--keep-tmp", action="store_true",
                    help="don't delete the temp dir; useful for re-listening")
    ap.add_argument("--body-threshold-db", type=float, default=-40.0,
                    help="pass if body-region null is at or below this (dB). "
                         "Body = samples NOT within ±50 ms of any seam.")
    ap.add_argument("--seam-threshold-db", type=float, default=-25.0,
                    help="pass if seam-region null is at or below this (dB). "
                         "Seam = samples within ±50 ms of any chunk boundary.")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    py = Path(sys.executable)
    infer_py = repo / "infer.py"
    if not infer_py.exists():
        raise SystemExit(f"missing {infer_py}")

    src = Path(args.tracks_dir)
    if not src.is_dir():
        raise SystemExit(f"--tracks-dir not a directory: {src}")
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"missing checkpoint: {ckpt}")

    tmp_root = Path(tempfile.mkdtemp(prefix="null_test_chunked_"))
    print(f"working dir: {tmp_root}")
    try:
        stems = tmp_root / "stems"
        n_stems = _truncate_stems(src, stems, args.duration)
        print(f"truncated {n_stems} stems to {args.duration:.1f}s into {stems}")

        baseline_dir = tmp_root / "baseline"
        chunked_dir = tmp_root / "chunked"

        # Single-shot baseline: --render-chunk-seconds 0 disables chunking.
        _run_infer(
            py, infer_py, ckpt, stems, baseline_dir,
            chunk_seconds=0.0,
            overlap_seconds=args.overlap_seconds,
            crossfade_seconds=args.crossfade_seconds,
            n_max=args.n_max, bpm=args.bpm, mert_dim=args.mert_dim,
        )
        # Chunked render with the requested chunk/overlap/crossfade.
        _run_infer(
            py, infer_py, ckpt, stems, chunked_dir,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            crossfade_seconds=args.crossfade_seconds,
            n_max=args.n_max, bpm=args.bpm, mert_dim=args.mert_dim,
        )

        baseline, fs_b = sf.read(str(baseline_dir / "rendered_mix.wav"), always_2d=True)
        chunked, fs_c = sf.read(str(chunked_dir / "rendered_mix.wav"), always_2d=True)
        if fs_b != fs_c:
            raise SystemExit(f"sample-rate mismatch: {fs_b} vs {fs_c}")
        if baseline.shape != chunked.shape:
            raise SystemExit(f"shape mismatch: baseline={baseline.shape} chunked={chunked.shape}")

        diff = chunked - baseline
        T = diff.shape[0]

        peak = float(np.abs(diff).max())
        full_rms_db = _db(diff)
        baseline_rms_db = _db(baseline)
        # Express the residual relative to the baseline level too.
        rel_db = full_rms_db - baseline_rms_db

        seam = _seam_mask(T, fs_b, args.chunk_seconds, half_window_ms=50.0)
        body_db = _db(diff[~seam]) if (~seam).any() else float("nan")
        seam_db = _db(diff[seam]) if seam.any() else float("nan")

        per_sec = _per_second_profile(diff, fs_b)

        print()
        print("=" * 64)
        print("null-test report")
        print("=" * 64)
        print(f"  stems:                {n_stems}")
        print(f"  duration:             {T / fs_b:.2f} s ({T} samples)")
        print(f"  chunk size:           {args.chunk_seconds:.1f} s "
              f"(seams at {[float(i*args.chunk_seconds) for i in range(1, int(T/fs_b/args.chunk_seconds)+1)]})")
        print(f"  overlap / crossfade:  {args.overlap_seconds:.1f} s / {args.crossfade_seconds*1000:.0f} ms")
        print()
        print(f"  baseline level:       {baseline_rms_db:+.2f} dB RMS")
        print(f"  diff peak:            {peak:.4e}")
        print(f"  diff RMS (full):      {full_rms_db:+.2f} dB  "
              f"({rel_db:+.2f} dB rel. to baseline)")
        print(f"  diff RMS (body):      {body_db:+.2f} dB   "
              f"[threshold {args.body_threshold_db:+.1f} dB]")
        print(f"  diff RMS (seam ±50ms):{seam_db:+.2f} dB   "
              f"[threshold {args.seam_threshold_db:+.1f} dB]")
        print()
        print("  per-second diff RMS (dB):")
        for i, db in enumerate(per_sec):
            seam_marker = ""
            if args.chunk_seconds > 0:
                # Mark seconds that contain a seam.
                if int((i + 1) // args.chunk_seconds) != int(i // args.chunk_seconds):
                    seam_marker = "  ← seam"
            print(f"    [{i:>2d}-{i+1:>2d}s]  {db:+7.2f} dB{seam_marker}")

        body_pass = (not np.isnan(body_db)) and body_db <= args.body_threshold_db
        seam_pass = (np.isnan(seam_db)) or seam_db <= args.seam_threshold_db
        ok = body_pass and seam_pass
        print()
        print(f"  body verdict: {'PASS' if body_pass else 'FAIL'}")
        print(f"  seam verdict: {'PASS' if seam_pass else 'FAIL'}")
        print(f"  overall:      {'PASS' if ok else 'FAIL'}")

        if args.keep_tmp:
            print(f"\n  artefacts kept at {tmp_root}")
        return 0 if ok else 1
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
