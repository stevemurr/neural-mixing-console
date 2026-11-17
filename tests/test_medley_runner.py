import contextlib
from contextlib import chdir
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
from nmc.data.medleydb import load_split
from nmc.eq.target_curve_net import TargetCurveNet

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

    def test_raw_default_separate_inventories_and_shared_song_folds(self):
        with chdir(self.root):
            for options, kind in (([], "raw"), (["--inputs", "stems"], "stems")):
                with patch(
                    "sys.argv",
                    ["nmc", "data", "prepare", "medleydb", "--root", str(self.audio), *options],
                ):
                    self.assertEqual(main(), 0)
                inventory, folds = load_split(
                    {
                        "manifest": f"dmc-data/medleydb/{kind}.json",
                        "val_frac": 0.2,
                        "seed": 0,
                        "input_kind": kind,
                    }
                )
                self.assertEqual(folds, {"train": self.songs[:1], "validation": self.songs[1:]})
                self.assertEqual(inventory["input_kind"], kind)
            self.assertTrue(Path("dmc-data/medleydb/raw.json").is_file())
            with self.assertRaisesRegex(ValueError, "requires raw"):
                load_split({"manifest": "dmc-data/medleydb/stems.json", "input_kind": "raw"})

    def test_raw_prediction_is_independent_of_references_and_annotations(self):
        song = self.songs[1]
        folder = self.audio / song
        raw = folder / f"{song}_RAW"
        arbitrary = next(raw.iterdir()).rename(raw / "take 7.wav")
        wave = np.sin(np.arange(12000) * 0.42) * 0.1
        sf.write(arbitrary, wave, 8000)
        reference = folder / f"{song}_MIX.wav"
        sf.write(reference, wave, 8000)
        ckpt = self.root / "fixture.pt"
        args = dict(
            n_bins=26, d_model=8, n_layers=1, n_heads=2, ffn_mult=2, dropout=0.0, max_delta_db=12.0
        )
        torch.save({"args": args, "model": TargetCurveNet(**args).state_dict()}, ckpt)
        self.recipe.write_text(
            f'name="raw-audio"\ntask="medleydb-benchmark"\nmanifest="{self.inventory}"\n'
            f'checkpoint="{ckpt}"\ninput_kind="raw"\nseed=0\nval_frac=0.2\nwindows_per_song=3\nwindow_seconds=0.25\n'
        )

        def evaluate(path):
            prepare(self.audio, "raw", self.inventory)
            self.assertEqual(run(self.recipe, runs_dir=path), 0)
            directory = next(path.glob("*/*/metrics.json")).parent
            metrics = json.loads((directory / "metrics.json").read_text())
            self.assertEqual(metrics["validation"], [song])
            self.assertEqual(set(metrics["songs"]), {song})
            row = metrics["songs"][song]
            self.assertEqual(row["fold"], "validation")
            self.assertEqual(len(set(w["start_sample"] for w in row["windows"])), 3)
            bundle = directory / row["bundle"]
            settings = json.loads((bundle / "mix.json").read_text())
            self.assertEqual([t["path"] for t in settings["tracks"]], ["take 7.wav"])
            return settings, sf.read(bundle / "preview.wav")[0], metrics["rmse_db"]

        before, audio, first_score = evaluate(self.root / "first")
        # Prediction sees the same audio. Change both the reference and the optional metadata.
        (folder / f"{song}_METADATA.yaml").write_text(
            "stems:\n  S01:\n    raw:\n      R01:\n        filename: take 7.wav\n        instrument: violin\n"
        )
        sf.write(reference, np.sin(np.arange(12000) * 0.03) * 0.1, 8000)
        after, replay, second_score = evaluate(self.root / "second")
        self.assertEqual(before, after)
        np.testing.assert_array_equal(audio, replay)
        self.assertLess(first_score["dry"], 0.1)
        self.assertGreater(second_score["dry"], first_score["dry"] + 1)
