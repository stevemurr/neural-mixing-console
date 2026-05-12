"""Download Cambridge-MT multitracks from a browser-extracted index.

Two host patterns appear in the index:
  - mtkdata.cambridgemusictechnology.co.uk  -> open, direct download works
  - multitracks.cambridge-mt.com            -> Cloudflare-challenged, browser-only

This script downloads the open URLs unattended (rate-limited, resumable,
size-verified) and writes the challenged URLs to a separate file that the
user can feed to a browser extension like DownThemAll.

Concurrency: with --concurrency N (default 4), N downloads run in parallel
via a thread pool. Each origin connection on Cloudflare's CDN is rate-limited,
but aggregate per-IP throughput is not, so concurrent downloads typically
multiply total throughput linearly.

Usage:
    python scripts/download_cambridge_mt.py \
        --index source_audio/cambridge_index.json \
        --out   source_audio/cambridge-mt \
        --type  Full \
        --concurrency 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm


OPEN_HOST = "mtkdata.cambridgemusictechnology.co.uk"
CHALLENGED_HOST = "multitracks.cambridge-mt.com"

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:150.0) Gecko/20100101 Firefox/150.0"

# Headers matching a real browser file-download request. Confirmed accepted by
# the open host (mtkdata). Does NOT bypass the challenged host on its own.
BROWSER_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://cambridge-mt.com/",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "cross-site",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "Sec-GPC": "1",
}


def is_full_multitrack(entry: dict) -> bool:
    t = entry.get("type", "").replace("\n", " ").strip()
    return t.startswith("Full")


def filename_from_url(url: str) -> str:
    return os.path.basename(urlparse(url).path)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(BROWSER_HEADERS)
    return s


def head(url: str, session: requests.Session, timeout: float = 20.0):
    r = session.head(url, timeout=timeout, allow_redirects=True)
    return r


def download_one(
    url: str, dest: Path, session: requests.Session,
    *, chunk: int = 1 << 20,
    show_progress: bool = True,
    bytes_pbar=None,
    bytes_pbar_lock: threading.Lock | None = None,
) -> tuple[bool, str]:
    """Download with resume support. Returns (ok, message).

    If show_progress is True, draws a per-file tqdm bar (only sane when
    concurrency=1). If a global bytes_pbar is provided, every chunk's byte
    count is also added to it.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {}
    existing = tmp.stat().st_size if tmp.exists() else 0
    if existing:
        headers["Range"] = f"bytes={existing}-"

    try:
        r = session.get(url, headers=headers, stream=True, timeout=60)
    except requests.RequestException as e:
        return False, f"connection error: {e}"

    if r.status_code == 416:  # already complete
        tmp.rename(dest)
        return True, "already complete"
    if r.status_code not in (200, 206):
        return False, f"HTTP {r.status_code}"

    total = int(r.headers.get("Content-Length", 0)) + existing
    mode = "ab" if existing and r.status_code == 206 else "wb"
    if mode == "wb" and existing:
        existing = 0  # server ignored Range; restart

    pbar = None
    if show_progress:
        pbar = tqdm(
            total=total or None,
            initial=existing,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name[:40],
            leave=False,
        )

    try:
        with open(tmp, mode) as f:
            for buf in r.iter_content(chunk_size=chunk):
                if not buf:
                    continue
                f.write(buf)
                if pbar is not None:
                    pbar.update(len(buf))
                if bytes_pbar is not None:
                    if bytes_pbar_lock is not None:
                        with bytes_pbar_lock:
                            bytes_pbar.update(len(buf))
                    else:
                        bytes_pbar.update(len(buf))
    finally:
        if pbar is not None:
            pbar.close()

    if total and tmp.stat().st_size != total:
        return False, f"size mismatch: got {tmp.stat().st_size}, expected {total}"

    tmp.rename(dest)
    return True, "ok"


# ---------- per-task worker (used by ThreadPoolExecutor) ----------

class TaskState:
    """Shared mutable state across worker threads."""
    def __init__(self, log_writer, log_file, bytes_pbar, file_pbar):
        self.log_writer = log_writer
        self.log_file = log_file
        self.bytes_pbar = bytes_pbar
        self.file_pbar = file_pbar
        self.print_lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.bytes_lock = threading.Lock()
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.counts_lock = threading.Lock()


def _log_row(state: TaskState, fname: str, url: str, size: int, status: str, message: str) -> None:
    with state.log_lock:
        state.log_writer.writerow(
            [time.strftime("%Y-%m-%dT%H:%M:%S"), fname, url, size, status, message]
        )
        state.log_file.flush()


def _print(state: TaskState, msg: str, *, err: bool = False) -> None:
    with state.print_lock:
        # tqdm.write is the right way to interleave with active progress bars
        if err:
            tqdm.write(msg, file=sys.stderr)
        else:
            tqdm.write(msg)


def download_task(
    entry: dict,
    out_dir: Path,
    state: TaskState,
    *,
    rate_sleep: float,
    show_progress: bool,
) -> tuple[str, str]:
    url = entry["link"]
    fname = filename_from_url(url)
    dest = out_dir / fname

    if dest.exists() and dest.stat().st_size > 0:
        with state.counts_lock:
            state.skipped += 1
        _log_row(state, fname, url, dest.stat().st_size, "skipped", "exists")
        if state.file_pbar is not None:
            state.file_pbar.update(1)
        return ("skipped", fname)

    session = make_session()
    size_bytes = 0
    size_str = "?"
    try:
        r = head(url, session)
        if r.status_code == 200:
            size_bytes = int(r.headers.get("Content-Length", 0))
            size_str = f"{size_bytes/1e9:.2f} GB"
    except requests.RequestException:
        pass

    _print(state, f"START {fname} ({size_str})")

    ok, msg = download_one(
        url, dest, session,
        show_progress=show_progress,
        bytes_pbar=state.bytes_pbar,
        bytes_pbar_lock=state.bytes_lock,
    )

    actual_size = dest.stat().st_size if dest.exists() else 0
    _log_row(state, fname, url, actual_size, "ok" if ok else "fail", msg)

    if ok:
        with state.counts_lock:
            state.completed += 1
        _print(state, f"DONE  {fname} ({size_str})")
    else:
        with state.counts_lock:
            state.failed += 1
        _print(state, f"FAIL  {fname}: {msg}", err=True)

    if state.file_pbar is not None:
        state.file_pbar.update(1)
    if rate_sleep > 0:
        time.sleep(rate_sleep)
    return ("ok" if ok else "fail", fname)


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="path to cambridge_index.json (browser-extracted)")
    ap.add_argument("--out", required=True, help="output directory for downloaded zips")
    ap.add_argument("--type", default="Full", choices=("Full", "Any"),
                    help="filter by entry type (default: Full = Full Multitrack only)")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="number of parallel downloads (default 4; set to 1 for sequential)")
    ap.add_argument("--rate-sleep", type=float, default=0.5,
                    help="seconds for each worker to sleep after a download (politeness)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of downloads (0 = no cap; useful for testing)")
    ap.add_argument("--dry-run", action="store_true",
                    help="HEAD-probe each URL and report sizes without downloading")
    args = ap.parse_args()

    index_path = Path(args.index)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    if args.type == "Full":
        index = [e for e in index if is_full_multitrack(e)]

    # Split by host
    open_urls, challenged_urls, other = [], [], []
    for e in index:
        host = urlparse(e["link"]).netloc
        if host == OPEN_HOST:
            open_urls.append(e)
        elif host == CHALLENGED_HOST:
            challenged_urls.append(e)
        else:
            other.append(e)

    print(f"Index loaded: {len(index)} entries (filter type={args.type})")
    print(f"  open host       ({OPEN_HOST}):       {len(open_urls)}")
    print(f"  challenged host ({CHALLENGED_HOST}): {len(challenged_urls)}")
    if other:
        print(f"  other hosts: {len(other)}")
        for e in other:
            print(f"    {e['link']}")

    challenged_path = index_path.parent / "cambridge_challenged_urls.txt"
    with open(challenged_path, "w") as f:
        for e in challenged_urls:
            f.write(e["link"] + "\n")
    print(f"\nWrote {len(challenged_urls)} challenged URLs to: {challenged_path}")
    print("  Hand these to a browser extension (DownThemAll, Simple Mass Downloader)")
    print("  while logged in / past the Cloudflare challenge in your browser.")

    if args.limit:
        open_urls = open_urls[:args.limit]
        print(f"\nLimited to first {len(open_urls)} open-host downloads.")

    log_path = out_dir / "download_log.csv"
    log_exists = log_path.exists()
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if not log_exists:
        log.writerow(["timestamp", "filename", "url", "size_bytes", "status", "message"])

    # ---- Dry run path ----
    if args.dry_run:
        # Concurrent HEADs are also faster
        session_factory = make_session
        total_bytes = 0
        results: list[tuple[int, int, str]] = []
        lock = threading.Lock()

        def head_task(idx_entry):
            i, e = idx_entry
            local = session_factory()
            try:
                r = head(e["link"], local)
                size = int(r.headers.get("Content-Length", 0))
                return (i, size, filename_from_url(e["link"]))
            except requests.RequestException as exc:
                return (i, -1, f"{filename_from_url(e['link'])}: {exc}")

        max_head_workers = min(max(args.concurrency, 8), 16)
        with ThreadPoolExecutor(max_workers=max_head_workers) as ex:
            for res in tqdm(ex.map(head_task, list(enumerate(open_urls, 1))),
                            total=len(open_urls), desc="HEAD probe"):
                results.append(res)
        results.sort(key=lambda x: x[0])
        for i, size, label in results:
            if size < 0:
                print(f"[{i:4d}/{len(open_urls)}] HEAD FAILED: {label}")
            else:
                total_bytes += size
                print(f"[{i:4d}/{len(open_urls)}] {size/1e9:6.2f} GB  {label}")
        print(f"\nTotal open-host bytes: {total_bytes/1e9:.1f} GB across {len(open_urls)} files")
        log_f.close()
        return 0

    # ---- Real downloads ----
    show_per_file_progress = (args.concurrency == 1)
    file_pbar = tqdm(total=len(open_urls), desc="files", position=0)
    bytes_pbar = tqdm(total=None, desc="bytes", unit="B", unit_scale=True,
                      unit_divisor=1024, position=1)

    state = TaskState(
        log_writer=log,
        log_file=log_f,
        bytes_pbar=bytes_pbar,
        file_pbar=file_pbar,
    )

    if args.concurrency <= 1:
        for e in open_urls:
            download_task(e, out_dir, state,
                          rate_sleep=args.rate_sleep,
                          show_progress=show_per_file_progress)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [
                ex.submit(download_task, e, out_dir, state,
                          rate_sleep=args.rate_sleep,
                          show_progress=False)
                for e in open_urls
            ]
            for _ in as_completed(futures):
                pass  # results are tracked in state

    file_pbar.close()
    bytes_pbar.close()
    log_f.close()
    print(f"\nDone. completed={state.completed} failed={state.failed} skipped={state.skipped}")
    return 0 if state.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
