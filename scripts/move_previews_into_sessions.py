"""Move downloaded Cambridge-MT preview MP3s into their session folders.

Reads source_audio/cambridge_preview_destination_map.tsv (filename -> session
-> dest_path) and, for each row, finds the matching file in --downloads-dir
and moves it to dest_path. Idempotent: skips rows whose dest already exists.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads-dir", required=True,
                    help="directory containing the MP3s your browser/download manager saved")
    ap.add_argument("--manifest", default="/home/murr/Code/neural-mixing-console/source_audio/cambridge_preview_destination_map.tsv")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would happen without moving anything")
    args = ap.parse_args()

    downloads_dir = Path(args.downloads_dir).expanduser().resolve()
    if not downloads_dir.is_dir():
        print(f"downloads dir not found: {downloads_dir}", file=sys.stderr)
        return 2

    moved = skipped = missing = 0
    with open(args.manifest) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            src = downloads_dir / row["filename"]
            dest = Path(row["dest_path"])
            if dest.exists():
                skipped += 1
                continue
            if not src.exists():
                missing += 1
                print(f"MISS  {row['filename']}  (session={row['session']})")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if args.dry_run:
                print(f"would move  {src.name}  ->  {dest}")
            else:
                shutil.move(str(src), str(dest))
                print(f"moved  {src.name}  ->  {dest.relative_to(dest.parents[2])}")
            moved += 1

    print(f"\nmoved={moved}  skipped(already in place)={skipped}  missing={missing}")
    if missing:
        print(f"({missing} files weren't in {downloads_dir} — verify the download "
              f"manager finished and put them there)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
