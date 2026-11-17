"""Browser-based DAW-style audition server for a folder of multitrack sessions.

Run: `python3 scripts/audition_server.py [--root <path>] [--port 8765]`

Each subdirectory of `--root` is treated as a session; every audio file inside
becomes a track. The browser loads all tracks for a session and plays them back
in sample-accurate sync with per-track volume / mute / solo controls.

Stdlib only — no install required.
"""

from __future__ import annotations

import argparse
import html
import http.server
import json
import mimetypes
import os
import socket
import sys
import urllib.parse
from pathlib import Path
from typing import Optional

DEFAULT_ROOT = Path(
    "/home/murr/Code/neural-mixing-console/source_audio/cambridge-mt"
)

AUDIO_EXTS = {
    ".wav", ".flac", ".mp3", ".ogg", ".oga", ".opus",
    ".aif", ".aiff", ".m4a", ".aac",
}

# Map audio extensions to MIME types because the stdlib mimetypes table is
# incomplete on some systems (e.g. .flac, .opus).
AUDIO_MIME = {
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/opus",
    ".aif": "audio/aiff",
    ".aiff": "audio/aiff",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
}


# ---------- HTTP handler ----------

class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "AuditionServer/1.0"

    @property
    def root(self) -> Path:
        return self.server.root  # type: ignore[attr-defined]

    def do_GET(self):
        self._dispatch(head=False)

    def do_HEAD(self):
        self._dispatch(head=True)

    # Quieter logs
    def log_message(self, fmt, *args):  # noqa: D401
        sys.stderr.write(f"[{self.log_date_time_string()}] {fmt % args}\n")

    def _dispatch(self, head: bool):
        try:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            if path == "/":
                self._serve_index(head)
            elif path == "/api/sessions":
                self._serve_sessions_api(head)
            elif path.startswith("/session/"):
                name = urllib.parse.unquote(path[len("/session/"):])
                self._serve_session_html(name, head)
            elif path.startswith("/api/session/"):
                name = urllib.parse.unquote(path[len("/api/session/"):])
                self._serve_session_api(name, head)
            elif path.startswith("/audio/"):
                rel = urllib.parse.unquote(path[len("/audio/"):])
                self._serve_audio(rel, head)
            else:
                self.send_error(404, "not found")
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:  # last-resort safety net
            sys.stderr.write(f"handler error: {e!r}\n")
            try:
                self.send_error(500, str(e))
            except Exception:
                pass

    # ----- routes -----

    def _serve_index(self, head: bool):
        self._send_html(INDEX_HTML, head)

    def _serve_sessions_api(self, head: bool):
        self._send_json(self._list_sessions(), head)

    def _serve_session_html(self, name: str, head: bool):
        sd = self._resolve_session(name)
        if sd is None:
            return self.send_error(404, "session not found")
        body = SESSION_HTML.replace("__SESSION_NAME__", html.escape(name))
        self._send_html(body, head)

    def _serve_session_api(self, name: str, head: bool):
        sd = self._resolve_session(name)
        if sd is None:
            return self.send_error(404, "session not found")
        self._send_json(
            {"name": name, "tracks": self._list_tracks(sd)},
            head,
        )

    def _serve_audio(self, rel: str, head: bool):
        parts = rel.split("/", 1)
        if len(parts) != 2:
            return self.send_error(404)
        session_name, file_rel = parts
        sd = self._resolve_session(session_name)
        if sd is None:
            return self.send_error(404)

        target = (sd / file_rel).resolve()
        sd_resolved = sd.resolve()
        if not (target == sd_resolved or
                str(target).startswith(str(sd_resolved) + os.sep)):
            return self.send_error(403, "path escapes session")
        if not target.is_file():
            return self.send_error(404)

        ext = target.suffix.lower()
        ctype = AUDIO_MIME.get(ext) or mimetypes.guess_type(str(target))[0] \
            or "application/octet-stream"
        self._send_file(target, ctype, head)

    # ----- helpers -----

    def _list_sessions(self) -> list[dict]:
        out = []
        for entry in sorted(self.root.iterdir(), key=lambda p: p.name.lower()):
            if not entry.is_dir():
                continue
            tracks = self._list_tracks(entry)
            if not tracks:
                continue
            out.append({"name": entry.name, "track_count": len(tracks)})
        return out

    def _list_tracks(self, session_dir: Path) -> list[dict]:
        # Many Cambridge MT zips wrap their content in a redundant top-level
        # folder. Descend through any single-directory wrappers so the user
        # sees clean track names instead of "Foo_Full/Foo_Full/kick.wav".
        content_root = session_dir
        for _ in range(4):  # safety bound
            children = [c for c in content_root.iterdir()
                        if not c.name.startswith("._") and c.name != ".DS_Store"]
            if len(children) == 1 and children[0].is_dir():
                content_root = children[0]
            else:
                break

        tracks: list[dict] = []
        for path in content_root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in AUDIO_EXTS:
                continue
            base = path.name
            if base.startswith("._") or base == ".DS_Store":
                continue
            display = str(path.relative_to(content_root)).replace(os.sep, "/")
            url_rel = str(path.relative_to(session_dir)).replace(os.sep, "/")
            tracks.append({
                "name": display,
                "url": "/audio/"
                       + urllib.parse.quote(session_dir.name, safe="")
                       + "/"
                       + urllib.parse.quote(url_rel, safe="/"),
                "size": path.stat().st_size,
            })
        tracks.sort(key=lambda t: t["name"].lower())
        return tracks

    def _resolve_session(self, name: str) -> Optional[Path]:
        if not name or "/" in name or "\\" in name or name in (".", ".."):
            return None
        target = (self.root / name).resolve()
        root_resolved = self.root.resolve()
        if not (target == root_resolved or
                str(target).startswith(str(root_resolved) + os.sep)):
            return None
        if not target.is_dir():
            return None
        return target

    def _send_html(self, body: str, head: bool):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head:
            self.wfile.write(data)

    def _send_json(self, obj, head: bool):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head:
            self.wfile.write(data)

    def _send_file(self, target: Path, ctype: str, head: bool):
        size = target.stat().st_size
        rng = self.headers.get("Range")
        if rng:
            try:
                start, end = self._parse_range(rng, size)
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            if head:
                return
            with open(target, "rb") as f:
                f.seek(start)
                self._stream(f, length)
        else:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            if head:
                return
            with open(target, "rb") as f:
                self._stream(f, size)

    def _stream(self, fileobj, length: int):
        remaining = length
        chunk_size = 1 << 20
        while remaining > 0:
            chunk = fileobj.read(min(chunk_size, remaining))
            if not chunk:
                break
            try:
                self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return
            remaining -= len(chunk)

    @staticmethod
    def _parse_range(header: str, size: int) -> tuple[int, int]:
        v = header.strip()
        if not v.lower().startswith("bytes="):
            raise ValueError(f"unsupported range unit: {v}")
        spec = v[6:]
        if "," in spec:
            raise ValueError("multi-range not supported")
        a, _, b = spec.partition("-")
        if a == "":
            n = int(b)
            if n <= 0:
                raise ValueError("invalid suffix range")
            start = max(size - n, 0)
            end = size - 1
        else:
            start = int(a)
            end = int(b) if b else size - 1
        if start > end or start >= size:
            raise ValueError("range out of bounds")
        end = min(end, size - 1)
        return start, end


class ThreadingServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler_cls, root: Path):
        super().__init__(addr, handler_cls)
        self.root = root


# ---------- HTML / CSS / JS ----------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Audition · Sessions</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root {
    --bg: #0d1117;
    --surface: #161b22;
    --surface-2: #1c2330;
    --border: #30363d;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #58a6ff;
    --accent-2: #79c0ff;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'SF Pro Text', system-ui, sans-serif;
    font-size: 14px;
  }
  header {
    padding: 18px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: baseline;
    gap: 16px;
  }
  header h1 { margin: 0; font-size: 16px; font-weight: 600; letter-spacing: 0.02em; }
  header .root { color: var(--muted); font-family: ui-monospace, 'SF Mono', Menlo, monospace; font-size: 12px; }
  header .count { margin-left: auto; color: var(--muted); font-size: 12px; }
  .filter {
    padding: 16px 24px 0;
  }
  .filter input {
    width: 100%;
    padding: 10px 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-size: 13px;
    outline: none;
    transition: border-color 120ms;
  }
  .filter input:focus { border-color: var(--accent); }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 8px;
    padding: 16px 24px 32px;
  }
  a.session {
    display: block;
    padding: 14px 16px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    text-decoration: none;
    transition: background 120ms, border-color 120ms, transform 120ms;
  }
  a.session:hover {
    background: var(--surface-2);
    border-color: var(--accent);
    transform: translateY(-1px);
  }
  a.session .name {
    font-weight: 500;
    word-break: break-word;
  }
  a.session .meta {
    margin-top: 6px;
    color: var(--muted);
    font-size: 12px;
    font-family: ui-monospace, 'SF Mono', Menlo, monospace;
  }
  .empty {
    padding: 80px 24px;
    text-align: center;
    color: var(--muted);
  }
</style>
</head>
<body>
  <header>
    <h1>Audition</h1>
    <span class="root" id="root-path"></span>
    <span class="count" id="count"></span>
  </header>
  <div class="filter">
    <input id="filter" type="search" placeholder="Filter sessions…" autofocus>
  </div>
  <div class="grid" id="grid"></div>
<script>
const $grid = document.getElementById('grid');
const $count = document.getElementById('count');
const $filter = document.getElementById('filter');
let allSessions = [];

function render(filterText) {
  const ft = (filterText || '').toLowerCase().trim();
  const matches = ft
    ? allSessions.filter(s => s.name.toLowerCase().includes(ft))
    : allSessions;
  $grid.innerHTML = '';
  if (matches.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = ft ? 'No sessions match your filter.' : 'No sessions found yet — still unzipping?';
    $grid.appendChild(empty);
  } else {
    for (const s of matches) {
      const a = document.createElement('a');
      a.className = 'session';
      a.href = '/session/' + encodeURIComponent(s.name);
      const name = document.createElement('div');
      name.className = 'name';
      name.textContent = s.name;
      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.textContent = s.track_count + (s.track_count === 1 ? ' track' : ' tracks');
      a.append(name, meta);
      $grid.appendChild(a);
    }
  }
  $count.textContent = `${matches.length} of ${allSessions.length}`;
}

async function load() {
  const resp = await fetch('/api/sessions');
  allSessions = await resp.json();
  render('');
}

$filter.addEventListener('input', e => render(e.target.value));
load();
</script>
</body>
</html>
"""


SESSION_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__SESSION_NAME__ · Audition</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root {
    --bg: #0d1117;
    --surface: #161b22;
    --surface-2: #1c2330;
    --surface-3: #222b3a;
    --border: #30363d;
    --border-2: #444c56;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #58a6ff;
    --solo: #d29922;
    --mute: #f85149;
    --wave: #58a6ff;
    --wave-bg: #0a1018;
    --playhead: #f0883e;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; height: 100%; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'SF Pro Text', system-ui, sans-serif;
    font-size: 13px;
    overflow: hidden;
  }
  body { display: flex; flex-direction: column; }
  header {
    flex: 0 0 auto;
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 14px;
  }
  header a.back {
    color: var(--muted);
    text-decoration: none;
    font-size: 13px;
    transition: color 120ms;
  }
  header a.back:hover { color: var(--text); }
  header h1 {
    margin: 0;
    font-size: 14px;
    font-weight: 500;
    word-break: break-word;
  }
  header .spacer { flex: 1; }

  .transport {
    flex: 0 0 auto;
    display: grid;
    grid-template-columns: auto auto 1fr auto auto;
    align-items: center;
    gap: 16px;
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  .transport-buttons { display: flex; gap: 6px; }
  button.tx {
    width: 38px; height: 30px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 14px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 120ms, border-color 120ms;
  }
  button.tx:hover:not(:disabled) {
    background: var(--surface-3);
    border-color: var(--border-2);
  }
  button.tx:disabled { opacity: 0.4; cursor: not-allowed; }
  button.tx.play.playing { background: var(--accent); border-color: var(--accent); color: #0d1117; }

  .time {
    font-family: ui-monospace, 'SF Mono', Menlo, monospace;
    font-size: 13px;
    color: var(--muted);
    min-width: 130px;
    text-align: right;
  }
  .time .now { color: var(--text); }

  .seekbar {
    position: relative;
    height: 22px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    cursor: pointer;
  }
  .seekbar .fill {
    position: absolute; top: 0; left: 0; bottom: 0;
    background: linear-gradient(90deg, rgba(88,166,255,0.18), rgba(88,166,255,0.32));
    pointer-events: none;
    width: 0;
  }
  .seekbar .head {
    position: absolute; top: 0; bottom: 0;
    width: 2px;
    background: var(--playhead);
    pointer-events: none;
    left: 0;
  }

  .master {
    display: flex; align-items: center; gap: 8px;
    color: var(--muted);
    font-family: ui-monospace, monospace;
    font-size: 11px;
  }
  .master input[type=range] { width: 100px; }

  .status {
    color: var(--muted);
    font-size: 11px;
    font-family: ui-monospace, monospace;
    white-space: nowrap;
  }
  .status.warn { color: var(--solo); }
  .status.err { color: var(--mute); }

  .tracks {
    flex: 1 1 0;
    overflow: auto;
    padding: 8px 0 60px;
  }
  .track {
    display: flex;
    flex-direction: column;
    border-bottom: 1px solid var(--border);
    transition: opacity 120ms;
  }
  .track-main {
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 0;
    height: 64px;
  }
  .track.dimmed { opacity: 0.45; }
  .track.error { opacity: 0.6; }
  .track-controls {
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 8px 12px;
    border-right: 1px solid var(--border);
    background: var(--surface);
    gap: 4px;
    overflow: hidden;
  }
  .track-name {
    font-size: 12px;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .track-row {
    display: flex; align-items: center; gap: 6px;
  }
  .btn-ms {
    width: 22px; height: 22px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--surface-2);
    color: var(--muted);
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: background 120ms, color 120ms, border-color 120ms;
    display: flex; align-items: center; justify-content: center;
  }
  .btn-ms:hover { background: var(--surface-3); color: var(--text); }
  .btn-ms.mute.on { background: var(--mute); color: #0d1117; border-color: var(--mute); }
  .btn-ms.solo.on { background: var(--solo); color: #0d1117; border-color: var(--solo); }
  input.vol {
    --fill-pct: 90.9%;
    flex: 1 1 0;
    min-width: 0;
    -webkit-appearance: none;
    appearance: none;
    height: 10px;
    background: linear-gradient(90deg,
      var(--playhead) 0%, var(--playhead) var(--fill-pct),
      var(--border) var(--fill-pct), var(--border) 100%);
    border-radius: 2px;
    outline: none;
    cursor: pointer;
  }
  input.vol::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 4px;
    height: 20px;
    border-radius: 1px;
    background: var(--text);
    border: none;
    cursor: pointer;
  }
  input.vol::-moz-range-thumb {
    width: 4px;
    height: 20px;
    border-radius: 1px;
    background: var(--text);
    border: none;
    cursor: pointer;
  }
  .vol-readout {
    font-family: ui-monospace, monospace;
    color: var(--muted);
    font-size: 10px;
    min-width: 12px;
    text-align: right;
  }

  .waveform {
    position: relative;
    background: var(--wave-bg);
    cursor: pointer;
    overflow: hidden;
  }
  .waveform canvas { display: block; width: 100%; height: 100%; }
  .waveform .lane-head {
    position: absolute; top: 0; bottom: 0; width: 1px;
    background: var(--playhead);
    pointer-events: none;
    left: 0;
    box-shadow: 0 0 6px rgba(240, 136, 62, 0.6);
  }
  .waveform .progress {
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    display: flex; align-items: center; justify-content: center;
    color: var(--muted);
    font-family: ui-monospace, monospace;
    font-size: 11px;
    pointer-events: none;
  }
  .waveform .progress-bar {
    position: absolute; left: 0; bottom: 0; height: 2px;
    background: var(--accent);
    width: 0;
    transition: width 80ms linear;
  }
  .waveform.error .progress {
    color: var(--mute);
  }

  .empty {
    padding: 80px 24px;
    text-align: center;
    color: var(--muted);
  }

  .hint {
    position: fixed;
    bottom: 16px;
    left: 50%;
    transform: translateX(-50%);
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 14px;
    color: var(--muted);
    font-size: 11px;
    font-family: ui-monospace, monospace;
  }
  .hint kbd {
    display: inline-block;
    padding: 1px 6px;
    border: 1px solid var(--border-2);
    border-bottom-width: 2px;
    border-radius: 3px;
    background: var(--surface-3);
    color: var(--text);
    font-size: 10px;
    font-family: ui-monospace, monospace;
    margin: 0 1px;
  }

  input.pan {
    flex: none;
    width: 60px;
    -webkit-appearance: none;
    appearance: none;
    height: 10px;
    background: linear-gradient(90deg,
      var(--solo) 0%, var(--solo) 50%,
      var(--accent) 50%, var(--accent) 100%);
    border-radius: 2px;
    outline: none;
    cursor: pointer;
  }
  input.pan::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 4px;
    height: 20px;
    border-radius: 1px;
    background: var(--text);
    border: none;
    cursor: pointer;
  }
  input.pan::-moz-range-thumb {
    width: 4px;
    height: 20px;
    border-radius: 1px;
    background: var(--text);
    border: none;
    cursor: pointer;
  }
  .btn-ms.fx {
    font-size: 10px;
    letter-spacing: 0.04em;
  }
  .btn-ms.fx.on {
    background: var(--accent);
    color: #0d1117;
    border-color: var(--accent);
  }

  .fx-panel {
    display: none;
    padding: 14px 18px 16px;
    background: var(--surface);
    border-top: 1px solid var(--border);
    gap: 0;
    flex-wrap: wrap;
  }
  .track.fx-open .fx-panel {
    display: flex;
  }
  .fx-header {
    flex: 0 0 100%;
    margin: -4px 0 10px;
    font-family: ui-monospace, monospace;
    font-size: 11px;
    color: var(--accent-2);
    letter-spacing: 0.02em;
  }

  .fx-section {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 0 18px;
    border-right: 1px solid var(--border);
    min-width: 180px;
  }
  .fx-section:first-child { padding-left: 0; }
  .fx-section:last-child { border-right: none; padding-right: 0; }
  .fx-section.eq { min-width: 280px; }
  .fx-section.comp { min-width: 230px; }

  .fx-section h3 {
    margin: 0 0 4px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
  }

  .fx-knob {
    display: grid;
    grid-template-columns: 56px 1fr 60px;
    align-items: center;
    gap: 8px;
    font-family: ui-monospace, monospace;
    font-size: 11px;
  }
  .fx-knob > label {
    color: var(--muted);
  }
  .fx-knob input[type=range] {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    outline: none;
    cursor: pointer;
  }
  .fx-knob input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
  }
  .fx-knob input[type=range]::-moz-range-thumb {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--accent);
    border: none;
    cursor: pointer;
  }
  .fx-knob .val {
    color: var(--text);
    text-align: right;
  }

  .fx-pills {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
  }
  .fx-pill {
    padding: 3px 9px;
    font-size: 10px;
    font-family: ui-monospace, monospace;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--muted);
    cursor: pointer;
    transition: background 120ms, color 120ms, border-color 120ms;
  }
  .fx-pill:hover { color: var(--text); background: var(--surface-3); }
  .fx-pill.on { background: var(--accent); color: #0d1117; border-color: var(--accent); }
</style>
</head>
<body>
  <header>
    <a class="back" href="/">← Sessions</a>
    <h1 id="title">__SESSION_NAME__</h1>
    <div class="spacer"></div>
    <div class="status" id="status">loading session…</div>
  </header>
  <div class="transport">
    <div class="transport-buttons">
      <button class="tx play" id="btn-play" title="Play / Pause (space)" disabled>▶</button>
      <button class="tx" id="btn-stop" title="Stop (return to start)" disabled>■</button>
    </div>
    <div class="time"><span class="now" id="t-now">0:00.000</span> / <span id="t-total">0:00.000</span></div>
    <div class="seekbar" id="seekbar">
      <div class="fill" id="seek-fill"></div>
      <div class="head" id="seek-head"></div>
    </div>
    <div class="master">
      MASTER
      <input type="range" id="master-vol" min="-60" max="6" value="0" step="0.1">
      <span id="master-vol-val">0.0 dB</span>
    </div>
    <button class="tx" id="btn-soloclear" title="Clear all solos" disabled>S✕</button>
  </div>
  <div class="tracks" id="tracks"></div>
  <div class="hint"><kbd>Space</kbd> play/pause &nbsp; <kbd>Home</kbd> stop &nbsp; click waveform to seek</div>

<script>
const SESSION_NAME = decodeURIComponent(location.pathname.replace(/^\/session\//, ''));

// ---- DOM helpers ----
const $ = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));
function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'className') e.className = v;
    else if (k === 'dataset') Object.assign(e.dataset, v);
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null) continue;
    e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return e;
}

function fmtTime(sec) {
  if (!isFinite(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec % 1) * 1000);
  return `${m}:${String(s).padStart(2,'0')}.${String(ms).padStart(3,'0')}`;
}

function dbToGain(db) { return Math.pow(10, db / 20); }
function gainToDb(g)  { return 20 * Math.log10(Math.max(g, 1e-6)); }

// ---- Impulse responses (procedurally generated reverb tails) ----
function makeIR(ctx, type) {
  const sr = ctx.sampleRate;
  let duration, decayPow, gated = false;
  if (type === 'hall') {
    duration = 2.8; decayPow = 1.4;
  } else if (type === 'nonlin') {
    duration = 1.0; decayPow = 0; gated = true;
  } else {
    duration = 0.65; decayPow = 1.8;
  }
  const len = Math.floor(sr * duration);
  const buf = ctx.createBuffer(2, len, sr);
  for (let ch = 0; ch < 2; ch++) {
    const data = buf.getChannelData(ch);
    for (let i = 0; i < len; i++) {
      const t = i / len;
      let env;
      if (gated) {
        if (t < 0.78) env = 1 - t * 0.3;
        else if (t < 0.84) env = (1 - 0.78 * 0.3) * (1 - (t - 0.78) / 0.06);
        else env = 0;
      } else {
        env = Math.pow(1 - t, decayPow);
      }
      data[i] = (Math.random() * 2 - 1) * env;
    }
  }
  return buf;
}

// ---- Mixer ----
class Mixer {
  constructor() {
    this.ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: 'interactive' });
    this.master = this.ctx.createGain();
    this.master.gain.value = 1;
    this.master.connect(this.ctx.destination);
    this.tracks = [];
    this.duration = 0;
    this.isPlaying = false;
    this.startCtxTime = 0;
    this.startOffset = 0;
    this._listeners = new Set();
    this.irs = {
      room: makeIR(this.ctx, 'room'),
      hall: makeIR(this.ctx, 'hall'),
      nonlin: makeIR(this.ctx, 'nonlin'),
    };
  }

  on(fn) { this._listeners.add(fn); return () => this._listeners.delete(fn); }
  emit(kind, payload) { for (const fn of this._listeners) fn(kind, payload); }

  addTrack(meta) {
    const ctx = this.ctx;

    // Channel-strip FX chain (always present; defaults are neutral).
    const inGain = ctx.createGain(); inGain.gain.value = 1;
    const eqLow = ctx.createBiquadFilter();
    eqLow.type = 'lowshelf'; eqLow.frequency.value = 120; eqLow.gain.value = 0;
    const eqMid = ctx.createBiquadFilter();
    eqMid.type = 'peaking'; eqMid.frequency.value = 1000; eqMid.Q.value = 1; eqMid.gain.value = 0;
    const eqHigh = ctx.createBiquadFilter();
    eqHigh.type = 'highshelf'; eqHigh.frequency.value = 6000; eqHigh.gain.value = 0;
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -24; comp.knee.value = 6; comp.ratio.value = 1;
    comp.attack.value = 0.01; comp.release.value = 0.1;
    const compMakeup = ctx.createGain(); compMakeup.gain.value = 1;
    const fxBus = ctx.createGain();
    const dryGain = ctx.createGain(); dryGain.gain.value = 1;
    const delaySend = ctx.createGain(); delaySend.gain.value = 0;
    const delayNode = ctx.createDelay(2.0); delayNode.delayTime.value = 0.25;
    const delayFeedback = ctx.createGain(); delayFeedback.gain.value = 0.3;
    const reverbSend = ctx.createGain(); reverbSend.gain.value = 0;
    const reverb = ctx.createConvolver(); reverb.buffer = this.irs.room;
    const outGain = ctx.createGain(); outGain.gain.value = 1;

    // Track-level fader + pan (downstream of FX chain).
    const gain = ctx.createGain(); gain.gain.value = 1;
    const panner = ctx.createStereoPanner(); panner.pan.value = 0;

    inGain.connect(eqLow);
    eqLow.connect(eqMid);
    eqMid.connect(eqHigh);
    eqHigh.connect(comp);
    comp.connect(compMakeup);
    compMakeup.connect(fxBus);
    fxBus.connect(dryGain); dryGain.connect(outGain);
    fxBus.connect(delaySend); delaySend.connect(delayNode);
    delayNode.connect(delayFeedback); delayFeedback.connect(delayNode);
    delayNode.connect(outGain);
    fxBus.connect(reverbSend); reverbSend.connect(reverb);
    reverb.connect(outGain);
    outGain.connect(gain);
    gain.connect(panner);
    panner.connect(this.master);

    const t = {
      ...meta,
      idx: this.tracks.length,
      buffer: null,
      gain,
      panner,
      source: null,
      volume: 1,
      pan: 0,
      muted: false,
      solo: false,
      loadProgress: 0,
      loaded: false,
      error: null,
      fx: { inGain, eqLow, eqMid, eqHigh, comp, compMakeup, fxBus, dryGain,
            delaySend, delayNode, delayFeedback, reverbSend, reverb, outGain },
      fxState: {
        inGainDb: 0,
        eqLowGain: 0, eqMidFreq: 1000, eqMidGain: 0, eqMidQ: 1, eqHighGain: 0,
        compThreshold: -24, compRatio: 1, compAttack: 10, compRelease: 100, compMakeupDb: 0,
        delayTimeMs: 250, delayFeedback: 30, delayMix: 0,
        reverbType: 'room', reverbMix: 0,
        outGainDb: 0,
      },
    };
    this.tracks.push(t);
    return t;
  }

  async loadTrack(t) {
    try {
      const resp = await fetch(t.url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const total = +resp.headers.get('Content-Length') || 0;
      const reader = resp.body.getReader();
      const chunks = [];
      let received = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.byteLength;
        if (total) {
          t.loadProgress = (received / total) * 0.7;
          this.emit('progress', t);
        }
      }
      const merged = new Uint8Array(received);
      let off = 0;
      for (const c of chunks) { merged.set(c, off); off += c.byteLength; }
      t.loadProgress = 0.72;
      this.emit('progress', t);
      t.buffer = await this.ctx.decodeAudioData(merged.buffer);
      t.loaded = true;
      t.loadProgress = 1;
      this.duration = Math.max(this.duration, t.buffer.duration);
      this._applyGain(t);
      this.emit('loaded', t);
    } catch (e) {
      t.error = String(e.message || e);
      this.emit('error', t);
    }
  }

  async loadAll() {
    await Promise.allSettled(this.tracks.map(t => this.loadTrack(t)));
    this.emit('all-loaded', null);
  }

  _anySolo() { return this.tracks.some(t => t.solo); }

  _applyGain(t) {
    const anySolo = this._anySolo();
    const soloGate = anySolo ? (t.solo ? 1 : 0) : 1;
    const muteGate = t.muted ? 0 : 1;
    const target = t.volume * soloGate * muteGate;
    t.gain.gain.setTargetAtTime(target, this.ctx.currentTime, 0.01);
  }

  setVolume(idx, v) { this.tracks[idx].volume = v; this._applyGain(this.tracks[idx]); }
  setMute(idx, m)   { this.tracks[idx].muted  = m; this._applyGain(this.tracks[idx]); }
  setSolo(idx, s)   {
    this.tracks[idx].solo = s;
    for (const t of this.tracks) this._applyGain(t);
    this.emit('solo-changed', null);
  }
  clearSolos() {
    for (const t of this.tracks) t.solo = false;
    for (const t of this.tracks) this._applyGain(t);
    this.emit('solo-changed', null);
  }
  setMasterDb(db) {
    this.master.gain.setTargetAtTime(dbToGain(db), this.ctx.currentTime, 0.01);
  }

  setPan(idx, p) {
    const t = this.tracks[idx];
    t.pan = p;
    const now = this.ctx.currentTime;
    t.panner.pan.cancelScheduledValues(now);
    t.panner.pan.setValueAtTime(t.panner.pan.value, now);
    t.panner.pan.linearRampToValueAtTime(p, now + 0.015);
  }
  setInGain(idx, db) {
    const t = this.tracks[idx];
    console.log('[fx] setInGain idx=' + idx + ' name=' + t.name + ' db=' + db.toFixed(1));
    t.fxState.inGainDb = db;
    t.fx.inGain.gain.setTargetAtTime(dbToGain(db), this.ctx.currentTime, 0.01);
  }
  setEqLow(idx, db) {
    const t = this.tracks[idx];
    t.fxState.eqLowGain = db;
    t.fx.eqLow.gain.setTargetAtTime(db, this.ctx.currentTime, 0.01);
  }
  setEqMidFreq(idx, hz) {
    const t = this.tracks[idx];
    t.fxState.eqMidFreq = hz;
    t.fx.eqMid.frequency.setTargetAtTime(hz, this.ctx.currentTime, 0.02);
  }
  setEqMidGain(idx, db) {
    const t = this.tracks[idx];
    t.fxState.eqMidGain = db;
    t.fx.eqMid.gain.setTargetAtTime(db, this.ctx.currentTime, 0.01);
  }
  setEqMidQ(idx, q) {
    const t = this.tracks[idx];
    t.fxState.eqMidQ = q;
    t.fx.eqMid.Q.setTargetAtTime(q, this.ctx.currentTime, 0.02);
  }
  setEqHigh(idx, db) {
    const t = this.tracks[idx];
    t.fxState.eqHighGain = db;
    t.fx.eqHigh.gain.setTargetAtTime(db, this.ctx.currentTime, 0.01);
  }
  setCompThreshold(idx, db) {
    const t = this.tracks[idx];
    t.fxState.compThreshold = db;
    t.fx.comp.threshold.setTargetAtTime(db, this.ctx.currentTime, 0.01);
  }
  setCompRatio(idx, r) {
    const t = this.tracks[idx];
    t.fxState.compRatio = r;
    t.fx.comp.ratio.setTargetAtTime(r, this.ctx.currentTime, 0.01);
  }
  setCompAttack(idx, ms) {
    const t = this.tracks[idx];
    t.fxState.compAttack = ms;
    t.fx.comp.attack.setTargetAtTime(ms / 1000, this.ctx.currentTime, 0.01);
  }
  setCompRelease(idx, ms) {
    const t = this.tracks[idx];
    t.fxState.compRelease = ms;
    t.fx.comp.release.setTargetAtTime(ms / 1000, this.ctx.currentTime, 0.01);
  }
  setCompMakeup(idx, db) {
    const t = this.tracks[idx];
    t.fxState.compMakeupDb = db;
    t.fx.compMakeup.gain.setTargetAtTime(dbToGain(db), this.ctx.currentTime, 0.01);
  }
  setDelayTime(idx, ms) {
    const t = this.tracks[idx];
    t.fxState.delayTimeMs = ms;
    t.fx.delayNode.delayTime.setTargetAtTime(ms / 1000, this.ctx.currentTime, 0.05);
  }
  setDelayFeedback(idx, pct) {
    const t = this.tracks[idx];
    t.fxState.delayFeedback = pct;
    t.fx.delayFeedback.gain.setTargetAtTime(pct / 100, this.ctx.currentTime, 0.01);
  }
  setDelayMix(idx, pct) {
    const t = this.tracks[idx];
    t.fxState.delayMix = pct;
    t.fx.delaySend.gain.setTargetAtTime(pct / 100, this.ctx.currentTime, 0.01);
  }
  setReverbType(idx, type) {
    const t = this.tracks[idx];
    t.fxState.reverbType = type;
    t.fx.reverb.buffer = this.irs[type];
  }
  setReverbMix(idx, pct) {
    const t = this.tracks[idx];
    t.fxState.reverbMix = pct;
    t.fx.reverbSend.gain.setTargetAtTime(pct / 100, this.ctx.currentTime, 0.01);
  }
  setOutGain(idx, db) {
    const t = this.tracks[idx];
    t.fxState.outGainDb = db;
    t.fx.outGain.gain.setTargetAtTime(dbToGain(db), this.ctx.currentTime, 0.01);
  }

  position() {
    if (this.isPlaying) {
      return Math.min(this.startOffset + (this.ctx.currentTime - this.startCtxTime), this.duration);
    }
    return this.startOffset;
  }

  async play() {
    if (this.isPlaying) return;
    if (this.duration <= 0) return;
    if (this.ctx.state === 'suspended') await this.ctx.resume();
    if (this.startOffset >= this.duration) this.startOffset = 0;
    const startTime = this.ctx.currentTime + 0.06;
    for (const t of this.tracks) {
      if (!t.buffer) continue;
      const src = this.ctx.createBufferSource();
      src.buffer = t.buffer;
      src.connect(t.fx.inGain);
      const offset = Math.min(this.startOffset, t.buffer.duration);
      if (offset < t.buffer.duration - 1e-4) {
        try { src.start(startTime, offset); } catch (e) { /* already started */ }
      }
      t.source = src;
    }
    this.startCtxTime = startTime;
    this.isPlaying = true;
    this.emit('transport', null);
  }

  pause() {
    if (!this.isPlaying) return;
    this.startOffset = Math.min(
      this.startOffset + (this.ctx.currentTime - this.startCtxTime),
      this.duration
    );
    for (const t of this.tracks) {
      if (t.source) {
        try { t.source.stop(); } catch (e) {}
        try { t.source.disconnect(); } catch (e) {}
        t.source = null;
      }
    }
    this.isPlaying = false;
    this.emit('transport', null);
  }

  stop() { this.pause(); this.startOffset = 0; this.emit('transport', null); }

  seek(sec) {
    sec = Math.max(0, Math.min(sec, this.duration || 0));
    const wasPlaying = this.isPlaying;
    if (wasPlaying) this.pause();
    this.startOffset = sec;
    if (wasPlaying) this.play();
    else this.emit('transport', null);
  }
}

// ---- Waveform rendering ----
function renderWaveform(canvas, buffer) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth;
  const cssH = canvas.clientHeight;
  if (cssW <= 0 || cssH <= 0) return;
  canvas.width = Math.floor(cssW * dpr);
  canvas.height = Math.floor(cssH * dpr);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cssW, cssH);

  const ch0 = buffer.getChannelData(0);
  const ch1 = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : null;
  const w = cssW;
  const h = cssH;
  const mid = h / 2;
  const samplesPerPx = Math.max(1, Math.floor(ch0.length / w));

  ctx.fillStyle = 'rgba(88,166,255,0.18)';
  ctx.strokeStyle = 'rgba(88,166,255,0.95)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < w; x++) {
    const start = x * samplesPerPx;
    const end = Math.min(start + samplesPerPx, ch0.length);
    let mn = 0, mx = 0;
    for (let i = start; i < end; i++) {
      let v = ch0[i];
      if (ch1) v = (v + ch1[i]) * 0.5;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    const yMin = Math.max(0, mid - mn * mid);
    const yMax = Math.min(h, mid - mx * mid);
    ctx.moveTo(x + 0.5, yMin);
    ctx.lineTo(x + 0.5, yMax);
  }
  ctx.stroke();

  // Center line
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.beginPath();
  ctx.moveTo(0, mid + 0.5);
  ctx.lineTo(w, mid + 0.5);
  ctx.stroke();
}

// ---- UI wiring ----
const mixer = new Mixer();
const $tracks = $('#tracks');
const $status = $('#status');
const $btnPlay = $('#btn-play');
const $btnStop = $('#btn-stop');
const $btnSoloClear = $('#btn-soloclear');
const $tNow = $('#t-now');
const $tTotal = $('#t-total');
const $seekbar = $('#seekbar');
const $seekFill = $('#seek-fill');
const $seekHead = $('#seek-head');
const $masterVol = $('#master-vol');
const $masterVolVal = $('#master-vol-val');

const trackEls = []; // parallel to mixer.tracks

function setStatus(msg, kind) {
  $status.textContent = msg;
  $status.className = 'status' + (kind ? ' ' + kind : '');
}

function buildFxPanel(t) {
  const idx = t.idx;
  const state = t.fxState;

  function knob(label, min, max, step, initial, fmt, onChange) {
    const slider = el('input', { type: 'range', min, max, step, value: initial });
    const valEl = el('span', { className: 'val' }, fmt(initial));
    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      valEl.textContent = fmt(v);
      onChange(v);
    });
    slider.addEventListener('dblclick', () => {
      slider.value = String(initial);
      valEl.textContent = fmt(initial);
      onChange(initial);
    });
    return el('div', { className: 'fx-knob' },
      el('label', {}, label),
      slider,
      valEl
    );
  }

  const fmtDb = v => (v >= 0 ? '+' : '') + v.toFixed(1) + ' dB';
  const fmtHz = v => v >= 1000 ? (v / 1000).toFixed(1) + ' kHz' : v.toFixed(0) + ' Hz';
  const fmtMs = v => v.toFixed(0) + ' ms';
  const fmtPct = v => v.toFixed(0) + ' %';
  const fmtRatio = v => v.toFixed(1) + ':1';
  const fmtQ = v => v.toFixed(2);

  const inSection = el('section', { className: 'fx-section' },
    el('h3', {}, 'Input'),
    knob('Gain', -24, 24, 0.1, state.inGainDb, fmtDb, v => mixer.setInGain(idx, v)),
  );

  const eqSection = el('section', { className: 'fx-section eq' },
    el('h3', {}, 'EQ'),
    knob('Low', -18, 18, 0.1, state.eqLowGain, fmtDb, v => mixer.setEqLow(idx, v)),
    knob('Mid Hz', 200, 8000, 10, state.eqMidFreq, fmtHz, v => mixer.setEqMidFreq(idx, v)),
    knob('Mid', -18, 18, 0.1, state.eqMidGain, fmtDb, v => mixer.setEqMidGain(idx, v)),
    knob('Mid Q', 0.3, 5, 0.05, state.eqMidQ, fmtQ, v => mixer.setEqMidQ(idx, v)),
    knob('High', -18, 18, 0.1, state.eqHighGain, fmtDb, v => mixer.setEqHigh(idx, v)),
  );

  const compSection = el('section', { className: 'fx-section comp' },
    el('h3', {}, 'Comp'),
    knob('Thr', -60, 0, 0.5, state.compThreshold, fmtDb, v => mixer.setCompThreshold(idx, v)),
    knob('Ratio', 1, 20, 0.1, state.compRatio, fmtRatio, v => mixer.setCompRatio(idx, v)),
    knob('Atk', 1, 200, 1, state.compAttack, fmtMs, v => mixer.setCompAttack(idx, v)),
    knob('Rel', 10, 1000, 5, state.compRelease, fmtMs, v => mixer.setCompRelease(idx, v)),
    knob('Makeup', -12, 24, 0.1, state.compMakeupDb, fmtDb, v => mixer.setCompMakeup(idx, v)),
  );

  const delaySection = el('section', { className: 'fx-section' },
    el('h3', {}, 'Delay'),
    knob('Time', 0, 1500, 1, state.delayTimeMs, fmtMs, v => mixer.setDelayTime(idx, v)),
    knob('Fbk', 0, 95, 1, state.delayFeedback, fmtPct, v => mixer.setDelayFeedback(idx, v)),
    knob('Mix', 0, 100, 1, state.delayMix, fmtPct, v => mixer.setDelayMix(idx, v)),
  );

  const types = ['room', 'hall', 'nonlin'];
  const pillRow = el('div', { className: 'fx-pills' });
  for (const typ of types) {
    const btn = el('button', { className: 'fx-pill' + (state.reverbType === typ ? ' on' : '') }, typ);
    btn.addEventListener('click', () => {
      mixer.setReverbType(idx, typ);
      pillRow.querySelectorAll('.fx-pill').forEach(b => b.classList.remove('on'));
      btn.classList.add('on');
    });
    pillRow.appendChild(btn);
  }

  const reverbSection = el('section', { className: 'fx-section' },
    el('h3', {}, 'Reverb'),
    pillRow,
    knob('Mix', 0, 100, 1, state.reverbMix, fmtPct, v => mixer.setReverbMix(idx, v)),
  );

  const outSection = el('section', { className: 'fx-section' },
    el('h3', {}, 'Output'),
    knob('Gain', -24, 24, 0.1, state.outGainDb, fmtDb, v => mixer.setOutGain(idx, v)),
  );

  const header = el('div', { className: 'fx-header' }, 'Track ' + (t.idx + 1) + ' — ' + t.name);

  return el('div', { className: 'fx-panel' },
    header, inSection, eqSection, compSection, delaySection, reverbSection, outSection
  );
}

function buildTrackRow(t) {
  const cMute = el('button', { className: 'btn-ms mute', title: 'Mute' }, 'M');
  const cSolo = el('button', { className: 'btn-ms solo', title: 'Solo' }, 'S');
  const cFx = el('button', { className: 'btn-ms fx', title: 'Show/hide channel strip' }, 'FX');
  const cPan = el('input', { className: 'pan', type: 'range', min: '-100', max: '100', value: '0', step: '1', title: 'Pan: C (double-click to center)' });
  const cVol = el('input', { className: 'vol', type: 'range', min: '-60', max: '6', value: '0', step: '0.1' });
  const cVolVal = el('span', { className: 'vol-readout' }, '0.0');
  const cName = el('div', { className: 'track-name', title: t.name }, (t.idx + 1) + '. ' + t.name);
  const controlsRow = el('div', { className: 'track-row' }, cMute, cSolo, cFx, cPan, cVol, cVolVal);
  const controls = el('div', { className: 'track-controls' }, cName, controlsRow);

  const canvas = el('canvas');
  const head = el('div', { className: 'lane-head' });
  const progBar = el('div', { className: 'progress-bar' });
  const progLabel = el('div', { className: 'progress' }, 'queued');
  const wave = el('div', { className: 'waveform' }, canvas, head, progBar, progLabel);

  const main = el('div', { className: 'track-main' }, controls, wave);
  const fxPanel = buildFxPanel(t);
  const row = el('div', { className: 'track' }, main, fxPanel);

  cMute.addEventListener('click', () => {
    const newVal = !mixer.tracks[t.idx].muted;
    mixer.setMute(t.idx, newVal);
    cMute.classList.toggle('on', newVal);
    refreshDimming();
  });
  cSolo.addEventListener('click', () => {
    const newVal = !mixer.tracks[t.idx].solo;
    mixer.setSolo(t.idx, newVal);
    cSolo.classList.toggle('on', newVal);
    refreshDimming();
    refreshSoloClear();
  });
  cFx.addEventListener('click', () => {
    const open = row.classList.toggle('fx-open');
    cFx.classList.toggle('on', open);
  });
  cPan.addEventListener('input', () => {
    const p = parseInt(cPan.value, 10);
    const lbl = p === 0 ? 'C' : (p < 0 ? 'L' + (-p) : 'R' + p);
    cPan.title = 'Pan: ' + lbl;
    mixer.setPan(t.idx, p / 100);
  });
  cPan.addEventListener('dblclick', () => {
    cPan.value = '0';
    cPan.title = 'Pan: C';
    mixer.setPan(t.idx, 0);
  });
  const updateVolFill = () => {
    const db = parseFloat(cVol.value);
    cVol.style.setProperty('--fill-pct', ((db + 60) / 66 * 100).toFixed(2) + '%');
  };
  cVol.addEventListener('input', () => {
    const db = parseFloat(cVol.value);
    cVolVal.textContent = (db > 0 ? '+' : '') + db.toFixed(1);
    mixer.setVolume(t.idx, dbToGain(db));
    updateVolFill();
  });
  cVol.addEventListener('dblclick', () => {
    cVol.value = '0';
    cVolVal.textContent = '0.0';
    mixer.setVolume(t.idx, 1);
    updateVolFill();
  });
  wave.addEventListener('click', e => {
    if (!mixer.duration) return;
    const r = wave.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width;
    mixer.seek(x * mixer.duration);
  });

  $tracks.appendChild(row);
  return { row, canvas, head, progBar, progLabel, cMute, cSolo, cFx, cPan, cVol, cVolVal };
}

function refreshDimming() {
  const anySolo = mixer.tracks.some(t => t.solo);
  for (let i = 0; i < mixer.tracks.length; i++) {
    const t = mixer.tracks[i];
    const els = trackEls[i];
    const dim = (t.muted) || (anySolo && !t.solo);
    els.row.classList.toggle('dimmed', dim);
    if (t.error) els.row.classList.add('error');
  }
}

function refreshSoloClear() {
  $btnSoloClear.disabled = !mixer.tracks.some(t => t.solo);
}

function refreshTransport() {
  const canPlay = mixer.tracks.some(t => t.loaded);
  $btnPlay.disabled = !canPlay;
  $btnStop.disabled = !canPlay;
  $btnPlay.textContent = mixer.isPlaying ? '❚❚' : '▶';
  $btnPlay.classList.toggle('playing', mixer.isPlaying);
  $tTotal.textContent = fmtTime(mixer.duration);
}

let lastFrameTime = 0;
function tick() {
  const pos = mixer.position();
  $tNow.textContent = fmtTime(pos);
  const frac = mixer.duration ? Math.min(pos / mixer.duration, 1) : 0;
  const headPct = (frac * 100) + '%';
  $seekFill.style.width = headPct;
  $seekHead.style.left = headPct;
  // Per-track heads share the global session timeline so they always line up
  // vertically, regardless of individual track length or load state.
  for (let i = 0; i < mixer.tracks.length; i++) {
    trackEls[i].head.style.left = headPct;
  }
  if (mixer.isPlaying && pos >= mixer.duration && mixer.duration > 0) {
    mixer.stop();
    refreshTransport();
  }
  requestAnimationFrame(tick);
}

function updateProgressUi(t) {
  const els = trackEls[t.idx];
  if (!els) return;
  if (t.error) {
    els.row.classList.add('error');
    els.progLabel.textContent = 'failed: ' + t.error;
    els.progBar.style.width = '0';
    return;
  }
  if (t.loaded) {
    renderWaveform(els.canvas, t.buffer);
    els.progLabel.textContent = '';
    els.progBar.style.width = '0';
  } else {
    const pct = Math.round(t.loadProgress * 100);
    els.progBar.style.width = pct + '%';
    els.progLabel.textContent = (t.loadProgress < 0.7 ? 'loading ' : 'decoding ') + pct + '%';
  }
}

mixer.on((kind, t) => {
  if (kind === 'progress' || kind === 'loaded' || kind === 'error') {
    if (t) updateProgressUi(t);
    if (kind === 'loaded') refreshTransport();
  }
  if (kind === 'transport') refreshTransport();
  if (kind === 'all-loaded') {
    const okCount = mixer.tracks.filter(t => t.loaded).length;
    const errCount = mixer.tracks.filter(t => t.error).length;
    setStatus(
      `${okCount} loaded${errCount ? ` · ${errCount} failed` : ''} · ${fmtTime(mixer.duration)}`,
      errCount ? 'warn' : ''
    );
    refreshTransport();
  }
  if (kind === 'solo-changed') refreshSoloClear();
});

// Transport controls
$btnPlay.addEventListener('click', () => mixer.isPlaying ? mixer.pause() : mixer.play());
$btnStop.addEventListener('click', () => { mixer.stop(); refreshTransport(); });
$btnSoloClear.addEventListener('click', () => {
  mixer.clearSolos();
  for (let i = 0; i < mixer.tracks.length; i++) {
    trackEls[i].cSolo.classList.remove('on');
  }
  refreshDimming();
  refreshSoloClear();
});

$seekbar.addEventListener('click', e => {
  if (!mixer.duration) return;
  const r = $seekbar.getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width;
  mixer.seek(x * mixer.duration);
});

$masterVol.addEventListener('input', () => {
  const db = parseFloat($masterVol.value);
  $masterVolVal.textContent = (db > 0 ? '+' : '') + db.toFixed(1) + ' dB';
  mixer.setMasterDb(db);
});

window.addEventListener('keydown', e => {
  if (e.target.matches('input, textarea')) return;
  if (e.code === 'Space') {
    e.preventDefault();
    if (!$btnPlay.disabled) mixer.isPlaying ? mixer.pause() : mixer.play();
  } else if (e.code === 'Home' || e.code === 'Digit0' || e.code === 'Numpad0') {
    if (!$btnStop.disabled) { mixer.stop(); refreshTransport(); }
  }
});

// Re-render waveforms on resize (debounced)
let resizeT = null;
window.addEventListener('resize', () => {
  if (resizeT) clearTimeout(resizeT);
  resizeT = setTimeout(() => {
    for (let i = 0; i < mixer.tracks.length; i++) {
      const t = mixer.tracks[i];
      if (t.loaded) renderWaveform(trackEls[i].canvas, t.buffer);
    }
  }, 120);
});

// Boot
async function boot() {
  const resp = await fetch('/api/session/' + encodeURIComponent(SESSION_NAME));
  if (!resp.ok) {
    setStatus('failed to load session', 'err');
    return;
  }
  const data = await resp.json();
  if (!data.tracks || data.tracks.length === 0) {
    $tracks.appendChild(el('div', { className: 'empty' }, 'No audio files found in this session.'));
    setStatus('empty session', 'warn');
    return;
  }
  setStatus(`loading ${data.tracks.length} tracks…`);
  for (const meta of data.tracks) {
    const t = mixer.addTrack(meta);
    const els = buildTrackRow(t);
    trackEls.push(els);
  }
  requestAnimationFrame(tick);
  await mixer.loadAll();
}
boot();
</script>
</body>
</html>
"""


# ---------- main ----------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help="folder containing session subdirectories")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"--root not a directory: {root}", file=sys.stderr)
        return 2

    addr = (args.host, args.port)
    try:
        server = ThreadingServer(addr, Handler, root)
    except OSError as e:
        print(f"could not bind {args.host}:{args.port}: {e}", file=sys.stderr)
        return 1

    host_display = args.host if args.host != "0.0.0.0" else socket.gethostname()
    print(f"audition server listening on http://{host_display}:{args.port}")
    print(f"serving sessions from: {root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
