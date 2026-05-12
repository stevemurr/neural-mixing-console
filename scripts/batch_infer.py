"""Run inference on N random multitrack sessions, save predicted mixes.

Picks sessions from --cambridge-root deterministically (via seed) and runs
infer.py on each, writing outputs to <out-root>/<session>/.

Outputs per session:
    rendered_mix.wav    — predicted full-song mix
    sum_baseline.wav    — sum-of-tracks baseline
    ref_mix.wav         — engineer's reference (if available)
    params.json         — per-track + global params
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path


def _build_pairs(map_tsv: Path, refmix_json: Path,
                 mt_root: Path, wavs_root: Path) -> list[dict]:
    """Mirror prepare_stage3_cambridge._build_pairs."""
    content_root_map: dict[str, str] = {}
    if refmix_json.exists():
        with open(refmix_json) as f:
            for r in json.load(f):
                content_root_map[r["session"]] = r["content_root_rel"]

    pairs: list[dict] = []
    seen = set()
    with open(map_tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            session = r["session"]
            if session in seen:
                continue
            seen.add(session)
            mix_name = r["filename"].replace(".mp3", ".wav")
            mix_path = wavs_root / mix_name
            cr = content_root_map.get(session, ".")
            mt_dir = mt_root / session if cr == "." else mt_root / session / cr

            # Fallback for case-mismatch
            if not mt_dir.exists():
                base = mt_root / session
                if base.exists():
                    for sub in base.iterdir():
                        if sub.is_dir() and any(p.suffix.lower() == ".wav" for p in sub.iterdir()):
                            mt_dir = sub
                            break

            if mt_dir.exists() and mix_path.exists():
                pairs.append({
                    "session": session,
                    "tracks_dir": str(mt_dir),
                    "ref_mix": str(mix_path),
                })
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="MixEncoder .pt checkpoint")
    ap.add_argument("--out-root", required=True,
                    help="parent dir; per-session subdir is created under here")
    ap.add_argument("--cambridge-root", default="source_audio/cambridge-mt")
    ap.add_argument("--mix-wavs-root", default="source_audio/wavs")
    ap.add_argument("--destination-map", default="source_audio/cambridge_preview_destination_map.tsv")
    ap.add_argument("--reference-mixes", default="source_audio/cambridge_reference_mixes.json")
    ap.add_argument("--n-sessions", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bpm", type=float, default=120.0)
    ap.add_argument("--n-max", type=int, default=80)
    ap.add_argument("--encoder-window-seconds", type=float, default=15.0)
    ap.add_argument("--mert-model", default="m-a-p/MERT-v1-95M")
    ap.add_argument("--mert-dim", type=int, default=768)
    ap.add_argument("--use-ref-mix", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip sessions whose out dir already has rendered_mix.wav")
    args = ap.parse_args()

    pairs = _build_pairs(
        Path(args.destination_map), Path(args.reference_mixes),
        Path(args.cambridge_root), Path(args.mix_wavs_root),
    )
    print(f"discovered {len(pairs)} valid mix↔multitrack pairs")
    if len(pairs) < args.n_sessions:
        print(f"WARNING: only {len(pairs)} pairs available; using all")
        sample = pairs
    else:
        rng = random.Random(args.seed)
        sample = rng.sample(pairs, args.n_sessions)
    print(f"selected {len(sample)} sessions (seed={args.seed})")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_skipped = 0
    n_failed = 0
    for i, pair in enumerate(sample, 1):
        sess = pair["session"]
        out_dir = out_root / sess
        rendered = out_dir / "rendered_mix.wav"
        if args.skip_existing and rendered.exists():
            print(f"[{i}/{len(sample)}] {sess}  SKIP (already done)")
            n_skipped += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            ".venv/bin/python", "infer.py",
            "--checkpoint", args.checkpoint,
            "--tracks-dir", pair["tracks_dir"],
            "--out-dir", str(out_dir),
            "--bpm", str(args.bpm),
            "--n-max", str(args.n_max),
            "--encoder-window-seconds", str(args.encoder_window_seconds),
            "--mert-model", args.mert_model,
            "--mert-dim", str(args.mert_dim),
            "--device", args.device,
        ]
        if pair.get("ref_mix"):
            cmd.extend(["--ref-mix", pair["ref_mix"]])
        if args.use_ref_mix:
            cmd.append("--use-ref-mix")

        print(f"\n[{i}/{len(sample)}] {sess}")
        print(f"   tracks: {pair['tracks_dir']}")
        print(f"   out:    {out_dir}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   FAILED (exit {result.returncode})")
            print(f"   stderr tail: {result.stderr[-500:]}")
            n_failed += 1
        else:
            n_done += 1
            print(f"   done")

    print(f"\nfinished: {n_done} rendered, {n_skipped} skipped, {n_failed} failed")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
