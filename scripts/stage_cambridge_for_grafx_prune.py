"""Stage a Cambridge-MT session for `scripts/label_grafx_prune.py`.

Grafx-prune expects each song under one base dir with this layout (all 30 kHz):

    <staging>/<session>/
      ├── correspondence.yaml   (group → list of stem filenames; from Diff-MST)
      ├── alignment.pickle      (mix↔rough_mix offset; from Diff-MST)
      ├── mix.wav               (engineer reference mix, 30 kHz stereo)
      └── stems/                (per-track stems, 30 kHz stereo)
          ├── 01_Kick.wav
          ├── …

This script:
    1. Reads Diff-MST's pre-shipped metadata for the session.
    2. Resamples the source ref-mix wav from `source_audio/wavs/`
       (mapped via `cambridge_preview_destination_map.tsv`) to 30 kHz stereo.
    3. Resamples every stem listed in correspondence.yaml from
       `source_audio/cambridge-mt/<session>_Full/<session>_Full/` to 30 kHz.
    4. Copies correspondence.yaml + alignment.pickle into the staging dir.

Idempotent — skips files that already exist at the staged path.

Use `--list-ready` to dump the names of sessions for which all source
files are present locally (without staging anything). Use `--all` to
stage every ready session.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly


META_DIR = Path("/home/murr/Code/grafx-prune/code/data/mixing_secrets/metadata")
CAMBRIDGE = Path("/home/murr/Code/neural-mixing-console/source_audio/cambridge-mt")
WAVS = Path("/home/murr/Code/neural-mixing-console/source_audio/wavs")
TSV = Path("/home/murr/Code/neural-mixing-console/source_audio/cambridge_preview_destination_map.tsv")
TARGET_SR = 30_000


def session_to_refwav() -> dict[str, Path]:
    """`{session_basename_without_full: wav_path}` from the TSV mapping."""
    out: dict[str, Path] = {}
    with open(TSV) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sess_stripped = row["session"].replace("_Full", "")
            out[sess_stripped] = WAVS / row["filename"].replace(".mp3", ".wav")
    return out


def ready_sessions(refwav_map: dict[str, Path]) -> list[str]:
    """List Diff-MST sessions for which both stems folder and ref wav exist locally."""
    ready: list[str] = []
    for d in sorted(META_DIR.iterdir()):
        if not d.is_dir():
            continue
        s = d.name
        stems_dir = CAMBRIDGE / f"{s}_Full" / f"{s}_Full"
        ref = refwav_map.get(s)
        if (stems_dir.is_dir() and any(stems_dir.glob("*.wav"))
                and ref is not None and ref.is_file()):
            ready.append(s)
    return ready


def resample_to(src: Path, dst: Path, target_sr: int) -> None:
    """Resample one audio file to `target_sr`, save as stereo PCM."""
    if dst.exists():
        return
    data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    if sr != target_sr:
        g = gcd(int(sr), target_sr)
        y = resample_poly(data, target_sr // g, int(sr) // g, axis=0)
        data = y.astype(np.float32, copy=False)
    sf.write(str(dst), data, target_sr)


def stage_session(session: str, staging_dir: Path, refwav_map: dict[str, Path]) -> dict:
    """Stage one session; return a small status dict."""
    sess_dir = staging_dir / session
    stems_dir = sess_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)

    # Metadata
    meta_corr = META_DIR / session / "correspondence.yaml"
    meta_align = META_DIR / session / "alignment.pickle"
    shutil.copy2(meta_corr, sess_dir / "correspondence.yaml")
    shutil.copy2(meta_align, sess_dir / "alignment.pickle")

    # Ref mix
    src_ref = refwav_map[session]
    resample_to(src_ref, sess_dir / "mix.wav", TARGET_SR)

    # Stems (the filenames are listed in correspondence.yaml)
    corr = yaml.safe_load(open(sess_dir / "correspondence.yaml"))
    stem_filenames = [f for files in corr.values() for f in files]
    src_stems_root = CAMBRIDGE / f"{session}_Full" / f"{session}_Full"

    missing: list[str] = []
    n_resampled = 0
    for fname in stem_filenames:
        src = src_stems_root / fname
        if not src.is_file():
            missing.append(fname)
            continue
        dst = stems_dir / fname
        before = dst.exists()
        resample_to(src, dst, TARGET_SR)
        if not before:
            n_resampled += 1

    return {
        "session": session,
        "n_stems_expected": len(stem_filenames),
        "n_resampled_this_run": n_resampled,
        "missing": missing,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--staging-dir", default="dmc-data/grafx-prune-data",
                    help="Where to stage the prepared session dirs. Default "
                         "is under dmc-data/ (persistent, gitignored). Avoid "
                         "/tmp — systemd-tmpfiles will eventually delete it.")
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="Session names to stage. Use --all to stage every "
                         "ready session, --list-ready to just dump names.")
    ap.add_argument("--all", action="store_true",
                    help="Stage every fully-ready session (overrides --sessions).")
    ap.add_argument("--limit", type=int, default=None,
                    help="When using --all, stage at most this many sessions.")
    ap.add_argument("--list-ready", action="store_true",
                    help="Print the list of fully-ready sessions and exit, no staging.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Concurrent staging workers (each resamples its session's stems).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    refwav_map = session_to_refwav()
    ready = ready_sessions(refwav_map)

    if args.list_ready:
        for s in ready:
            print(s)
        print(f"\n{len(ready)} fully-ready sessions", file=sys.stderr)
        return 0

    if args.all:
        targets = ready
        if args.limit is not None:
            targets = targets[: args.limit]
    elif args.sessions:
        targets = list(args.sessions)
        unknown = [s for s in targets if s not in ready]
        if unknown:
            logging.warning("not in 'ready' set (may still partially stage): %s",
                            ", ".join(unknown))
    else:
        ap.error("specify --sessions, --all, or --list-ready")
        return 2

    staging = Path(args.staging_dir).expanduser()
    staging.mkdir(parents=True, exist_ok=True)
    logging.info(f"staging {len(targets)} session(s) to {staging} "
                 f"(workers={args.workers}, target_sr={TARGET_SR})")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(stage_session, s, staging, refwav_map): s for s in targets}
        for i, fut in enumerate(as_completed(futures), 1):
            s = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                miss = f"  MISSING {len(r['missing'])} stem(s)" if r["missing"] else ""
                logging.info(f"  [{i:3d}/{len(targets)}] {s:60s} "
                             f"resampled {r['n_resampled_this_run']:3d}/{r['n_stems_expected']:3d}{miss}")
            except Exception as e:
                logging.error(f"  [{i:3d}/{len(targets)}] {s} FAILED: {type(e).__name__}: {e}")

    n_ok = sum(1 for r in results if not r["missing"])
    n_partial = sum(1 for r in results if r["missing"])
    print(f"\nstaging complete: {n_ok} fully staged, {n_partial} partial (missing stems)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
