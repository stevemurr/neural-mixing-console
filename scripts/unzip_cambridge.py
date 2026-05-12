"""Extract every cambridge-mt zip into a sibling folder, delete the zip on success.

- Skips macOS metadata (`._*`, `__MACOSX/*`, `.DS_Store`)
- Leaves `*.part` (incomplete downloads) untouched
- Idempotent: re-running picks up where it left off (zips that were already
  extracted are gone; partial folders get overwritten)
- Parallelism: thread pool — zlib.inflate releases the GIL so threads scale well
  and we avoid the cost of pickling/forking per-zip
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_ROOT = Path(
    "/home/murr/Code/neural-mixing-console/source_audio/cambridge-mt"
)


def is_metadata(name: str) -> bool:
    base = os.path.basename(name)
    return (
        base.startswith("._")
        or base == ".DS_Store"
        or name.startswith("__MACOSX/")
        or "/__MACOSX/" in name
    )


def extract_one(zip_path: Path) -> tuple[Path, str, str | None, int, float]:
    """Returns (zip_path, status, error_or_none, member_count, seconds)."""
    start = time.monotonic()
    target = zip_path.with_suffix("")  # strips trailing .zip
    target.mkdir(parents=True, exist_ok=True)
    members_extracted = 0
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir() or is_metadata(info.filename):
                    continue
                # Guard against zip-slip: refuse paths that escape the target
                outpath = (target / info.filename).resolve()
                if not str(outpath).startswith(str(target.resolve()) + os.sep) and outpath != target.resolve():
                    raise RuntimeError(f"unsafe path in zip: {info.filename!r}")
                outpath.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(outpath, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1 << 20)
                members_extracted += 1
    except Exception as e:
        return (zip_path, "fail", repr(e), members_extracted, time.monotonic() - start)

    try:
        zip_path.unlink()
    except OSError as e:
        return (zip_path, "extracted-but-zip-not-deleted", repr(e), members_extracted, time.monotonic() - start)
    return (zip_path, "ok", None, members_extracted, time.monotonic() - start)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    if not args.root.is_dir():
        print(f"root not found: {args.root}", file=sys.stderr)
        return 2

    zips = sorted(args.root.glob("*.zip"))
    if not zips:
        print(f"no zips in {args.root}")
        return 0

    total_bytes = sum(z.stat().st_size for z in zips)
    print(
        f"found {len(zips)} zips, {total_bytes / 1e9:.1f} GB total; "
        f"using {args.workers} workers",
        flush=True,
    )

    ok = fail = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_one, z): z for z in zips}
        for i, fut in enumerate(as_completed(futures), 1):
            zp, status, err, count, secs = fut.result()
            elapsed = time.monotonic() - started
            tag = "OK " if status == "ok" else "FAIL"
            if status == "ok":
                ok += 1
            else:
                fail += 1
            print(
                f"[{i:>3}/{len(zips)}] {tag} {zp.name}  "
                f"({count} files, {secs:.1f}s, total {elapsed/60:.1f}m)"
                + (f"  :: {err}" if err else ""),
                flush=True,
            )

    print(f"done: {ok} ok, {fail} failed, {time.monotonic()-started:.0f}s")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
