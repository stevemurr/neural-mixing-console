import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from nmc.data.medleydb import prepare
from nmc.data.splits import session_split
from nmc.runner import list_runs, run


class MedleyRunnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.audio = self.root / "Audio"
        names = [f"Artist_Song{i}" for i in range(100)]
        train, val = session_split(names, 0.2, 0)
        self.songs = [min(train), min(val)]
        for song in self.songs:
            folder = self.audio / song
            for kind, label in (("STEMS", "STEM_01"), ("RAW", "RAW_01_01")):
                inputs = folder / f"{song}_{kind}"
                inputs.mkdir(parents=True)
                sf.write(inputs / f"{song}_{label}.wav", np.full(12000, 0.1), 8000)
            sf.write(folder / f"{song}_MIX.wav", np.full((12000, 2), 0.1), 8000)
        self.inventory = self.root / "inventory.json"
        self.recipe = self.root / "inspect.toml"
        self.recipe.write_text(
            f'name="fixture"\ntask="inspect"\nmanifest="{self.inventory}"\n'
            "seed=0\nval_frac=0.2\nwindow_seconds=1\n"
        )
        self.runs = self.root / "runs"

    def test_repeat_preserves_both_runs_and_windows(self):
        prepare(self.audio, "stems", self.inventory)
        self.assertEqual(run(self.recipe, True, self.runs), 0)
        self.assertFalse(self.runs.exists())
        for _ in range(2):
            self.assertEqual(run(self.recipe, runs_dir=self.runs), 0)
        manifests = list(self.runs.glob("*/*/run.json"))
        self.assertEqual(len(manifests), 2)
        self.assertEqual(len({json.loads(p.read_text())["id"] for p in manifests}), 2)
        listing = io.StringIO()
        with contextlib.redirect_stdout(listing):
            list_runs(self.runs)
        for path in manifests:
            self.assertIn(path.parent.name, listing.getvalue())
            self.assertEqual(json.loads(path.read_text())["status"], "completed")
            result = json.loads((path.parent / "metrics.json").read_text())
            self.assertEqual(result["train"], self.songs[:1])
            self.assertEqual(result["validation"], self.songs[1:])
            for row in result["windows"]:
                data, rate = sf.read(path.parent / row["dry_wav"], always_2d=True)
                self.assertEqual((data.shape, rate), ((8000, 2), 8000))
                self.assertEqual(row["start_sample"], 2000)

    def test_failed_input_names_song_and_records_failure(self):
        prepare(self.audio, "raw", self.inventory)
        data = json.loads(self.inventory.read_text())
        song = self.songs[0]
        track = data["songs"][song]["inputs"][0]["path"]
        self.assertIn("_RAW/", track)
        (self.audio / track).unlink()
        with self.assertRaisesRegex(ValueError, song):
            run(self.recipe, True, self.runs)
        self.assertNotEqual(run(self.recipe, runs_dir=self.runs), 0)
        record = next(self.runs.glob("*/*/run.json"))
        self.assertEqual(json.loads(record.read_text())["status"], "failed")
        self.assertIn(song, (record.parent / "run.log").read_text())

    def test_missing_declared_stem_excludes_session(self):
        song = self.songs[0]
        metadata = self.audio / song / f"{song}_METADATA.yaml"
        metadata.write_text("stems:\n  S01:\n    filename: missing.wav\n")
        prepare(self.audio, "stems", self.inventory)
        result = json.loads(self.inventory.read_text())
        self.assertNotIn(song, result["songs"])
        self.assertEqual(result["excluded"][0]["song"], song)
        self.assertIn("missing.wav", result["excluded"][0]["reason"])
