"""Generate (and optionally fetch) Cambridge-MT preview-mix URLs for the
local session folders under source_audio/cambridge-mt/.

URL pattern (confirmed by user from a working browser):
    https://previews.cambridge-mt.com/<Artist><Song>_<Type>_Preview.mp3
where the local session folder is <Artist>_<Song>_<Type>. The first
underscore — the one separating artist from song — is removed; everything
else is preserved. Example: 3DMARCo_ACapella_Full → 3DMARCoACapella_Full_Preview.mp3.

`previews.cambridge-mt.com` sits behind a Cloudflare managed challenge that
returns 403 to plain requests / curl / curl_cffi-Chrome120. Two fetch paths:

  1. Browser extension (DownThemAll, Simple Mass Downloader): point it at
     the URL list this script writes. The browser already holds the
     cf_clearance cookie from when you first opened cambridge-mt.com.
  2. Headless via cf_clearance: copy the `cf_clearance` cookie + matching
     `User-Agent` from your browser's DevTools (Application → Cookies on
     previews.cambridge-mt.com) and pass them via --cookie / --user-agent
     or the env vars CF_CLEARANCE and CF_USER_AGENT. The script then uses
     curl_cffi (Chrome impersonation) to download.

Output (in --out parent dir):
  - cambridge_preview_urls.txt        flat URL list, one per line
  - cambridge_previews_manifest.tsv   session<TAB>url<TAB>local_filename<TAB>note

Output (in --out itself, only with --fetch):
  - <Song>_<Type>_Preview.mp3         downloaded files
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PREVIEW_HOST = "https://previews.cambridge-mt.com"


def derive_remote_basename(session_name: str) -> tuple[str, str]:
    """Return (remote_basename_without_ext, note).

    Rule: remove the first underscore (the artist/song separator) and append
    "_Preview". Everything else in the folder name is preserved verbatim.
    note is empty when the convention is clean; otherwise a short string
    explaining what we did for an irregular session name.
    """
    parts = session_name.split("_", 1)  # split only on the first underscore
    if len(parts) < 2:
        # Single token — no artist/song separator at all. Best-effort guess.
        return (session_name + "_Preview",
                "no underscore in folder name — verify URL")
    artist, rest = parts
    stem = artist + rest  # concatenate, dropping the artist/song underscore
    note = ""
    if not (rest.endswith("_Full") or rest.endswith("_Excerpt") or "_Full_" in rest or "_Excerpt_" in rest):
        # Folder is missing a _Full/_Excerpt segment we'd expect.
        if "Full" not in stem and "Excerpt" not in stem:
            stem = stem + "_Full"
            note = "no _Full/_Excerpt in folder; guessing _Full"
        else:
            note = "non-standard segment layout — verify URL"
    elif session_name.count("_") >= 3:
        note = f"non-standard session name ({session_name.count('_')+1} segments) — verify URL"
    return (stem + "_Preview", note)


def build_records(session_root: Path) -> list[dict]:
    rows = []
    for p in sorted(session_root.iterdir()):
        if not p.is_dir():
            continue
        remote_stem, note = derive_remote_basename(p.name)
        fname = remote_stem + ".mp3"
        rows.append({
            "session": p.name,
            "url": f"{PREVIEW_HOST}/{fname}",
            "filename": fname,
            "note": note,
        })
    return rows


def write_outputs(rows: list[dict], parent: Path) -> tuple[Path, Path]:
    url_list = parent / "cambridge_preview_urls.txt"
    manifest = parent / "cambridge_previews_manifest.tsv"
    with open(url_list, "w") as f:
        f.write("\n".join(r["url"] for r in rows) + "\n")
    with open(manifest, "w") as f:
        f.write("session\turl\tfilename\tnote\n")
        for r in rows:
            f.write(f"{r['session']}\t{r['url']}\t{r['filename']}\t{r['note']}\n")
    return url_list, manifest


def fetch_one(row: dict, out_dir: Path, *, cookie: str, ua: str, timeout: float):
    from curl_cffi import requests as cr

    dest = out_dir / row["filename"]
    if dest.exists() and dest.stat().st_size > 0:
        return ("skipped", row, dest.stat().st_size, "exists")

    headers = {
        "User-Agent": ua,
        "Referer": "https://cambridge-mt.com/",
        "Accept": "audio/mpeg,audio/*;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    cookies = {}
    # cookie may be a single cf_clearance value or a full Cookie: header value
    if "=" in cookie:
        for pair in cookie.split(";"):
            pair = pair.strip()
            if not pair or "=" not in pair:
                continue
            k, v = pair.split("=", 1)
            cookies[k.strip()] = v.strip()
    else:
        cookies["cf_clearance"] = cookie

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        r = cr.get(row["url"], headers=headers, cookies=cookies,
                   impersonate="chrome120", timeout=timeout, stream=True)
    except Exception as e:
        return ("fail", row, 0, f"connection: {e}")

    if r.status_code != 200:
        body = r.content[:200] if hasattr(r, "content") else b""
        return ("fail", row, 0, f"HTTP {r.status_code} ({body!r:.120})")

    ctype = r.headers.get("content-type", "")
    if "html" in ctype.lower():
        return ("fail", row, 0, f"got HTML (challenge?) ct={ctype}")

    n = 0
    with open(tmp, "wb") as f:
        for buf in r.iter_content(chunk_size=1 << 20):
            if buf:
                f.write(buf)
                n += len(buf)
    tmp.rename(dest)
    return ("ok", row, n, "ok")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="/home/murr/Code/neural-mixing-console/source_audio/cambridge-mt",
                    help="root containing session folders (default: project source_audio/cambridge-mt)")
    ap.add_argument("--out", default="/home/murr/Code/neural-mixing-console/source_audio/cambridge-mt-previews",
                    help="output dir for preview MP3s (also where the URL list / manifest land in its parent)")
    ap.add_argument("--fetch", action="store_true",
                    help="actually download (requires --cookie or $CF_CLEARANCE)")
    ap.add_argument("--cookie", default=os.environ.get("CF_CLEARANCE", ""),
                    help="cf_clearance value, or full 'Cookie:' header string. Default: $CF_CLEARANCE")
    ap.add_argument("--user-agent", default=os.environ.get(
                        "CF_USER_AGENT",
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
                    help="User-Agent that the cf_clearance was issued for")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of fetches (useful to verify pattern on a few sessions first)")
    args = ap.parse_args()

    session_root = Path(args.sessions)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_records(session_root)
    url_list, manifest = write_outputs(rows, out_dir.parent)
    notes = [r for r in rows if r["note"]]

    print(f"sessions found:        {len(rows)}")
    print(f"URL list written to:   {url_list}")
    print(f"manifest written to:   {manifest}")
    if notes:
        print(f"\n{len(notes)} session(s) with non-standard names — verify these URLs by hand:")
        for r in notes:
            print(f"  {r['session']:50s} -> {r['filename']}   ({r['note']})")

    if not args.fetch:
        print("\n--list-only mode (default). Hand the URL list to a browser extension, "
              "or rerun with --fetch and a cf_clearance cookie.")
        return 0

    if not args.cookie:
        print("\nERROR: --fetch requires --cookie or $CF_CLEARANCE.", file=sys.stderr)
        print("Get it from DevTools > Application > Cookies on previews.cambridge-mt.com",
              file=sys.stderr)
        return 2

    todo = rows[: args.limit] if args.limit else rows
    print(f"\nfetching {len(todo)} previews into {out_dir}/  (concurrency={args.concurrency})")
    completed = failed = skipped = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(fetch_one, r, out_dir,
                           cookie=args.cookie, ua=args.user_agent, timeout=args.timeout)
                for r in todo]
        for fut in as_completed(futs):
            status, row, nbytes, msg = fut.result()
            if status == "ok":
                completed += 1
                print(f"  OK    {row['filename']:60s}  {nbytes/1e6:6.2f} MB")
            elif status == "skipped":
                skipped += 1
                print(f"  skip  {row['filename']:60s}  exists")
            else:
                failed += 1
                print(f"  FAIL  {row['filename']:60s}  {msg}", file=sys.stderr)

    dt = time.time() - t0
    print(f"\ndone in {dt:.1f}s  completed={completed}  failed={failed}  skipped={skipped}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
