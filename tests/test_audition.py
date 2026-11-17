import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

import numpy as np
import soundfile as sf

from nmc.audition import Handler, catalog


class AuditionTests(unittest.TestCase):
    def test_catalog_and_seekable_audio(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            folder = root / "pilot" / "second"
            bundle = folder / "songs" / "Two tracks"
            excerpt = bundle / "audition" / "8000"
            excerpt.mkdir(parents=True)
            wav = np.linspace(-0.2, 0.2, 8000)
            for path in [
                bundle / "preview.wav",
                *[excerpt / f"{k}.wav" for k in ("dry", "reference", "predicted")],
            ]:
                sf.write(path, wav, 8000)
            (folder / "run.json").write_text(
                json.dumps({"recipe": "pilot", "id": "second", "started": "2026-11-17"})
            )
            (folder / "metrics.json").write_text(
                json.dumps(
                    {
                        "sample_rate": 8000,
                        "songs": {
                            "Two tracks": {
                                "bundle": "songs/Two tracks",
                                "windows": [{"start_sample": 8000, "dry": 2, "predicted": 1}],
                            }
                        },
                    }
                )
            )
            empty = root / "replay" / "first"
            empty.mkdir(parents=True)
            (empty / "run.json").write_text(
                '{"recipe":"replay","id":"first","started":"2026-11-16"}'
            )
            runs, media = catalog(root)
            self.assertEqual([r["id"] for r in runs], ["second", "first"])
            self.assertEqual(runs[1]["songs"], [])
            clips = runs[0]["songs"][0]["clips"]
            self.assertEqual(clips[0]["start"], 1)
            self.assertEqual(set(clips[0]["sources"]), {"dry", "reference", "predicted"})
            self.assertIsNone(clips[1]["start"])
            with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
                server.root, server.media = root, media
                thread = Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    conn = HTTPConnection(*server.server_address, timeout=5)
                    try:
                        url = clips[1]["sources"]["predicted"]
                        conn.request("GET", url, headers={"Range": "bytes=3-19"})
                        response = conn.getresponse()
                        self.assertEqual(response.status, 206)
                        self.assertEqual(
                            response.read(), (bundle / "preview.wav").read_bytes()[3:20]
                        )
                        conn.request("HEAD", url, headers={"Range": "bytes=-12"})
                        response = conn.getresponse()
                        self.assertEqual(response.getheader("Content-Length"), "12")
                        self.assertEqual(response.read(), b"")
                        conn.request("GET", url, headers={"Range": "bytes=9999999-"})
                        self.assertEqual(conn.getresponse().status, 416)
                        conn.request("GET", "/audio/pilot/second/run.json")
                        self.assertEqual(conn.getresponse().status, 404)
                    finally:
                        conn.close()
                finally:
                    server.shutdown()
                    thread.join()
