"""Read-only local browser for manifest-backed experiment audio."""

from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
from pathlib import Path
import re
import socket
from urllib.parse import quote, unquote, urlsplit

import numpy as np
import soundfile as sf


def catalog(root):
    runs = []
    media = {}
    for manifest in sorted(root.glob("*/*/run.json")):
        folder = manifest.parent
        if not folder.resolve().is_relative_to(root) or manifest.is_symlink():
            continue
        try:
            record = json.loads(manifest.read_text())
            metrics = {}
            metric_path = folder / "metrics.json"
            if metric_path.is_file() and not metric_path.is_symlink():
                metrics = json.loads(metric_path.read_text())
        except (ValueError, OSError):
            continue

        def sources(paths):
            result = {}
            for label, path in paths.items():
                if (
                    path.suffix.lower() != ".wav"
                    or not path.resolve().is_relative_to(folder)
                    or not path.is_file()
                ):
                    continue
                key = path.relative_to(root).as_posix()
                media[key] = path
                result[label] = "/audio/" + quote(key)
            return result

        songs = []
        for song, detail in metrics.get("songs", {}).items():
            bundle = folder / detail["bundle"]
            clips = []
            for row in detail["windows"]:
                start = row["start_sample"]
                base = bundle / "audition" / str(start)
                variants = sources(
                    {key: base / f"{key}.wav" for key in ("predicted", "dry", "reference")}
                )
                if variants:
                    clips.append(
                        {
                            "start": start / metrics["sample_rate"],
                            "sources": variants,
                            "scores": {key: row[key] for key in ("dry", "predicted")},
                        }
                    )
            full = sources({"predicted": bundle / "preview.wav"})
            if full:
                clips.append({"start": None, "sources": full})
            if clips:
                songs.append({"name": song, "clips": clips})
        # Inspection runs store one native-rate listening pair per fold.
        if not songs:
            for row in metrics.get("windows", []):
                if "dry_wav" not in row:
                    continue
                variants = sources({k: folder / row[k + "_wav"] for k in ("dry", "reference")})
                if variants:
                    songs.append(
                        {
                            "name": row["song"],
                            "clips": [
                                {
                                    "start": row["start_sample"] / row["sample_rate"],
                                    "sources": variants,
                                }
                            ],
                        }
                    )
        runs.append(
            {k: record.get(k) for k in ("recipe", "id", "started", "status", "revision")}
            | {"songs": songs, "scores": metrics.get("rmse_db", {})}
        )
    return sorted(runs, key=lambda r: r["started"] or "", reverse=True), media


@lru_cache(maxsize=96)
def peaks(path, size, mtime):
    """Bound display resolution and memory for full-length multichannel renders."""
    with sf.SoundFile(path) as source:
        block = max(1, (source.frames + 159) // 160)
        values = [
            float(np.abs(samples).max())
            for samples in source.blocks(blocksize=block, dtype="float32")
        ]
    top = max(values, default=0)
    return [round(v / top, 3) if top else 0 for v in values]


def byte_range(header, size):
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
    if not match or not any(match.groups()) or not size:
        raise ValueError
    first, last = match.groups()
    start = int(first) if first else max(0, size - int(last))
    end = min(size - 1, int(last)) if first and last else size - 1
    if start > end:
        raise ValueError
    return start, end


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.respond(False)

    def do_HEAD(self):
        self.respond(True)

    def send_bytes(self, body, mime, head):
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def respond(self, head):
        route = unquote(urlsplit(self.path).path)
        static = {
            "/": ("index.html", "text/html"),
            "/player.js": ("player.js", "text/javascript"),
            "/style.css": ("style.css", "text/css"),
        }
        try:
            if route in static:
                asset, mime = static[route]
                return self.send_bytes(
                    files("nmc").joinpath("web", asset).read_bytes(), mime + "; charset=utf-8", head
                )
            if route == "/api/runs":
                runs, media = catalog(self.server.root)
                self.server.media = media
                return self.send_bytes(json.dumps(runs).encode(), "application/json", head)
            prefix, _, key = route.lstrip("/").partition("/")
            path = self.server.media.get(key)
            if (
                prefix not in ("audio", "peaks")
                or path is None
                or not path.resolve().is_relative_to(self.server.root)
            ):
                return self.send_error(404)
            stat = path.stat()
            if prefix == "peaks":
                data = peaks(path, stat.st_size, stat.st_mtime_ns)
                return self.send_bytes(json.dumps(data).encode(), "application/json", head)
            self.audio(path, stat.st_size, head)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except FileNotFoundError:
            self.send_error(404)

    def audio(self, path, size, head):
        start, end = 0, size - 1
        header = self.headers.get("Range")
        if header:
            try:
                start, end = byte_range(header, size)
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(206 if header else 200)
        if header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        if not head:
            with path.open("rb") as audio:
                audio.seek(start)
                remaining = end - start + 1
                while remaining:
                    block = audio.read(min(256 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)


def serve(root: Path, port, *, host="127.0.0.1"):
    root = root.resolve(strict=True)
    with ThreadingHTTPServer((host, port), Handler) as server:
        server.root = root
        _, server.media = catalog(root)
        address = socket.gethostname() if server.server_address[0] == "0.0.0.0" else host
        print(f"Audition at http://{address}:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
