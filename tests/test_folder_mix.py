import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from nmc.cli import main
from nmc.eq.contribution_console import ContributionConsole
from nmc.eq.target_curve_net import TargetCurveNet
from nmc.mix import PASSAGE, RATE, contribution


class FolderMixTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name)
        self.raw = self.work / "raw"
        self.raw.mkdir()
        self.bundle = self.work / "raw-mix"
        self.document = self.bundle / "mix.json"
        self.checkpoint = self.work / "tiny.pt"
        args = dict(
            n_bins=26, d_model=8, n_layers=1, n_heads=2, ffn_mult=2, dropout=0.0, max_delta_db=12.0
        )
        with torch.random.fork_rng():
            torch.manual_seed(5)
            model = TargetCurveNet(**args)
        torch.save({"args": args, "model": model.state_dict()}, self.checkpoint)

    def cli(self, *args):
        with patch("sys.argv", ["nmc", *map(str, args)]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 0)

    def analyze(self):
        self.cli("mix", self.raw, "--checkpoint", self.checkpoint)
        return json.loads(self.document.read_text())

    def render(self, name):
        out = self.work / name
        self.cli("render", self.raw, self.document, "--output", out)
        audio, rate = sf.read(out, dtype="float32", always_2d=True)
        self.assertEqual(rate, RATE)
        return audio

    def test_source_identity_edits_and_checkpoint_free_replay(self):
        rng = np.random.default_rng(21)
        sf.write(self.raw / "a.wav", rng.normal(0, 0.1, 22050), 44100, subtype="FLOAT")
        sf.write(self.raw / "b.wav", np.zeros(32), RATE)
        stereo = rng.normal(0, 0.1, (90000, 2))
        sf.write(self.raw / "c.wav", stereo, RATE, subtype="FLOAT")
        mix = self.analyze()
        self.checkpoint.unlink()
        tracks = mix["tracks"]
        self.assertEqual([t["path"] for t in tracks], ["a.wav", "b.wav", "c.wav"])
        self.assertEqual(tracks[1]["eq_db"], [0.0] * 26)
        self.assertEqual([t["channels"] for t in tracks], [1, 1, 2])
        self.assertTrue(any(abs(x) > 0.01 for x in tracks[2]["eq_db"]))
        self.assertNotEqual(tracks[2]["eq_compensation_db"], 0.0)
        preview, rate = sf.read(self.bundle / "preview.wav", dtype="float32")
        self.assertEqual((len(preview), rate), (len(stereo), RATE))
        # Order of JSON records and the folder's absolute location are immaterial.
        tracks.reverse()
        self.raw = self.raw.rename(self.work / "moved")
        self.document.write_text(json.dumps(mix))
        np.testing.assert_array_equal(preview, self.render("replay.wav"))

        # Flatten EQ, set explicit compensation and put the mono track hard left.
        for track in tracks:
            track.update(eq_db=[0.0] * 26, eq_compensation_db=0.0)
        tracks[0]["level_db"] = -6.0
        tracks[2]["pan"] = -1.0
        mix["master_db"] = 0.0
        self.document.write_text(json.dumps(mix))
        edited = self.render("edited.wav")
        np.testing.assert_allclose(edited[:, 1], stereo[:, 1] * 10 ** (-6 / 20), atol=1e-6)
        np.testing.assert_allclose(
            edited[24000:, 0], stereo[24000:, 0] * 10 ** (-6 / 20), atol=1e-6
        )
        self.assertFalse(np.allclose(edited[:24000, 0], stereo[:24000, 0] * 10 ** (-6 / 20)))

    def test_inference_follows_audio_across_changed_file_order(self):
        rng = np.random.default_rng(17)
        first = self.raw / "a.wav"
        second = self.raw / "b.wav"
        sf.write(first, rng.normal(0, 0.08, 12000), RATE)
        sf.write(second, np.sin(np.arange(12000) / 12) * 0.1, RATE)
        original = self.analyze()
        first.rename(self.raw / "z.wav")
        other = self.work / "reordered"
        self.cli("mix", self.raw, "--checkpoint", self.checkpoint, "--out-dir", other)
        updated = json.loads((other / "mix.json").read_text())
        by_digest = {track["sha256"]: track for track in original["tracks"]}
        for track in updated["tracks"]:
            np.testing.assert_allclose(
                track["eq_db"], by_digest[track["sha256"]]["eq_db"], atol=2e-5
            )

    def test_large_session_and_nested_names(self):
        for idx in range(65):
            folder = self.raw / str(idx)
            folder.mkdir()
            sf.write(folder / "mic.wav", np.ones(80) * 0.02, RATE)
        mix = self.analyze()
        self.assertEqual(len(mix["tracks"]), 65)
        self.assertEqual(len({t["path"] for t in mix["tracks"]}), 65)
        self.assertTrue(all(t["analysis_starts"] == [0] for t in mix["tracks"]))
        self.assertLess(mix["master_db"], 0.0)

    def test_late_entrance_and_multiple_passages(self):
        audio = torch.zeros(1, PASSAGE * 5 + 100)
        audio[:, PASSAGE : PASSAGE + 100] = 0.1
        audio[:, -100:] = 0.2
        features, starts = contribution(audio, ContributionConsole())
        self.assertEqual(starts, [PASSAGE, 5 * PASSAGE])
        self.assertTrue(torch.any(features > 0))

    def test_unreadable_and_changed_sources_fail_by_name(self):
        wav = self.raw / "broken.wav"
        wav.write_bytes(b"bad")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()) as err:
            self.analyze()
        self.assertIn(wav.name, err.getvalue())
        self.assertFalse(self.bundle.exists())
        sf.write(wav, np.zeros(200), RATE)
        self.analyze()
        sf.write(wav, np.ones(200) * 0.1, RATE)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()) as err:
            self.render("changed.wav")
        self.assertIn(wav.name, err.getvalue())
        self.assertFalse((self.work / "changed.wav").exists())
