"""Render N random moisesdb sessions with the v6 checkpoint.

moisesdb's per-session layout is two levels deep:
    <session_uuid>/<category>/<stem_uuid>.wav   (+ data.json)
infer.py expects a flat dir of stems. This script symlinks each stem of a
session into a flat staging dir, renames symlinks by trackType for human
readability (`kick_drum.wav`, `lead_male_singer.wav`, …), then invokes
infer.py for each staged dir.

Usage:
    .venv/bin/python scripts/run_v6_moisesdb.py [--n 5] [--seed 0] [--max-stems 21]

Outputs land under:
    source_audio/predicted/moisesdb_v6/<song_artist_uuid8>/
        rendered_mix.wav, sum_baseline.wav, params.json, infer.log

Reports a one-line summary per session at the end (peak, RMS, trim_db).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


REPO = Path(__file__).resolve().parent.parent
MOISES_ROOT = REPO / "source_audio" / "moisesdb" / "moisesdb" / "moisesdb_v0.1"
CKPT = REPO / "dmc-data" / "checkpoints" / "stage3_v6" / "mix_encoder_latest.pt"
OUT_ROOT = REPO / "source_audio" / "predicted" / "moisesdb_v6"
PY = REPO / ".venv" / "bin" / "python"
INFER = REPO / "infer.py"


def _safe(name: str) -> str:
    """Filesystem-safe slug."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "untitled"


def _stage_session(session_dir: Path, staging_dir: Path) -> tuple[str, str, int]:
    """Symlink the session's stems into `staging_dir`, named by trackType.

    Returns (song_title, artist, n_stems).
    """
    data = json.loads((session_dir / "data.json").read_text())
    song = data.get("song") or "untitled"
    artist = data.get("artist") or "unknown"
    staging_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    used_names: set[str] = set()
    for stem in data.get("stems", []):
        for tr in stem.get("tracks", []):
            track_type = tr.get("trackType") or stem.get("stemName") or f"track_{n}"
            ext = tr.get("extension") or "wav"
            uid = tr["id"]
            src = session_dir / stem["stemName"] / f"{uid}.{ext}"
            if not src.exists():
                # tolerate occasional missing files
                continue
            base = _safe(track_type)
            name = f"{base}.{ext}"
            i = 2
            while name in used_names:
                name = f"{base}_{i}.{ext}"
                i += 1
            used_names.add(name)
            (staging_dir / name).symlink_to(src)
            n += 1
    return song, artist, n


def _summarize_render(out_dir: Path) -> dict:
    """Pull the headline numbers from a rendered mix + params.json."""
    info: dict = {}
    mix_path = out_dir / "rendered_mix.wav"
    params_path = out_dir / "params.json"
    if mix_path.exists():
        d, fs = sf.read(str(mix_path), always_2d=True)
        info["duration_s"] = round(d.shape[0] / fs, 2)
        info["peak"] = float(np.abs(d).max())
        info["rms_db"] = float(20 * np.log10(np.sqrt((d ** 2).mean()) + 1e-12))
    if params_path.exists():
        p = json.loads(params_path.read_text())
        info["trim_db"] = p.get("trim_db")
        info["n_tracks"] = p.get("n_tracks")
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5, help="number of sessions")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--max-stems", type=int, default=21,
                    help="cap on stems per session (passed to infer as --n-max)")
    ap.add_argument("--list", action="store_true",
                    help="just print which sessions would be picked, then exit")
    args = ap.parse_args()

    if not MOISES_ROOT.is_dir():
        raise SystemExit(f"missing {MOISES_ROOT}")
    if not CKPT.exists():
        raise SystemExit(f"missing checkpoint {CKPT}")

    sessions = sorted(p for p in MOISES_ROOT.iterdir() if p.is_dir())
    rng = random.Random(args.seed)
    picks = rng.sample(sessions, args.n)

    print(f"picked {args.n} sessions (seed={args.seed}):")
    for p in picks:
        meta = json.loads((p / "data.json").read_text())
        print(f"  {p.name[:8]}  '{meta.get('song', '?')}' by '{meta.get('artist', '?')}'  ({meta.get('genre', '?')})")
    if args.list:
        return 0

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for p in picks:
        meta = json.loads((p / "data.json").read_text())
        song = meta.get("song", "untitled")
        artist = meta.get("artist", "unknown")
        slug = f"{_safe(artist)}__{_safe(song)}__{p.name[:8]}"
        staging = OUT_ROOT / slug / "_stems"
        out_dir = OUT_ROOT / slug

        if staging.exists():
            shutil.rmtree(staging)
        song_, artist_, n_stems = _stage_session(p, staging)
        print(f"\n[{slug}]  {n_stems} stems staged at {staging}")

        log_path = out_dir / "infer.log"
        cmd = [
            str(PY), str(INFER),
            "--checkpoint", str(CKPT),
            "--tracks-dir", str(staging),
            "--out-dir", str(out_dir),
            "--n-max", str(args.max_stems),
            "--bpm", "120",
        ]
        print(f"  $ {' '.join(cmd)}")
        with open(log_path, "w") as logf:
            r = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"  FAILED (exit {r.returncode}); see {log_path}")
            summaries.append({"slug": slug, "song": song, "artist": artist,
                              "status": f"FAILED({r.returncode})"})
            continue
        info = _summarize_render(out_dir)
        info.update({"slug": slug, "song": song, "artist": artist, "status": "ok"})
        summaries.append(info)
        print(f"  ok: {info.get('duration_s', '?')}s  peak={info.get('peak', '?'):.3f}  "
              f"RMS={info.get('rms_db', '?'):+.2f} dB  trim={info.get('trim_db', '?'):+.2f} dB")

    print("\n" + "=" * 88)
    print(f"{'SUMMARY':^88}")
    print("=" * 88)
    print(f"  {'song':<35s} {'tracks':>6s} {'dur':>7s} {'peak':>7s} {'RMS':>9s} {'trim':>9s}")
    for s in summaries:
        if s.get("status") != "ok":
            print(f"  {s['song'][:34]:<35s} {s.get('status', '?'):>6s}")
            continue
        print(f"  {s['song'][:34]:<35s} {s.get('n_tracks', 0):>6d} "
              f"{s.get('duration_s', 0):>6.1f}s {s.get('peak', 0):>7.3f} "
              f"{s.get('rms_db', 0):>+8.2f} dB {s.get('trim_db', 0):>+7.2f} dB")
    print(f"\noutputs at {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
