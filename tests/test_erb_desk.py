import base64
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import soundfile as sf
import torch

from nmc.data.raw_folder import RATE
from torchaudio.functional import loudness

from nmc.desk_session import matched_audio
from nmc.eq.amount import anchors, calibrate
from nmc.eq.erb_desk import BrownDesk, WIDTHS


class DeskTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "browser DSP parity test needs Node.js")
    def test_browser_import_matches_reference_eq_on_identical_audio(self):
        worker = Path(__file__).resolve().parents[1] / "src/nmc/web/desk-analysis.js"
        rng = np.random.default_rng(27)
        samples = rng.normal(0, 0.06, 8192).astype("float32")
        script = """
            global.self = {postMessage: ({curves, kernels}) => console.log(JSON.stringify({
              curves, fir: Buffer.from(kernels[1][1].buffer).toString('base64')
            }))};
            require(process.argv[1]);
            const input = Buffer.from(require('node:fs').readFileSync(0, 'utf8'), 'base64');
            const x = new Float32Array(Uint8Array.from(input).buffer);
            self.onmessage({data:[[x, x.map(v => -v)]]});
        """
        result = subprocess.run(
            ["node", "-e", script, str(worker)],
            input=base64.b64encode(samples.tobytes()),
            capture_output=True,
            check=True,
        )
        observed = json.loads(result.stdout)
        desk = BrownDesk()
        power = desk.measure(samples[:, None])
        for width in WIDTHS:
            np.testing.assert_allclose(
                observed["curves"][str(width)], desk.curve(power, width), atol=1e-8
            )
        fir = np.frombuffer(base64.b64decode(observed["fir"]), dtype="<f4")
        np.testing.assert_allclose(fir, desk.kernel(desk.curve(power, 1) / 0.15), atol=1e-7)

    def test_brown_input_and_silence_get_flat_response(self):
        desk = BrownDesk()
        brown = 10 ** (desk.target_db / 10)
        for width in WIDTHS:
            np.testing.assert_allclose(desk.curve(brown * 0.07, width), 0, atol=1e-10)
            np.testing.assert_array_equal(desk.curve(brown * 0, width), 0)

    def test_smoothing_support_and_stereo_link(self):
        desk = BrownDesk()
        band = 18
        power = 10 ** (desk.target_db / 10)
        power[band] *= 16
        power[-3:] = 0
        narrow, broad = (desk.curve(power, n) for n in (1, 6))
        self.assertLess(narrow[band], broad[band])
        self.assertLessEqual(np.abs(narrow).max(), 2)
        np.testing.assert_array_equal(narrow[-3:], 0)
        rng = np.random.default_rng(2)
        mono = rng.normal(0, 0.1, RATE).astype(np.float32)
        stereo = np.column_stack([mono, -mono])
        np.testing.assert_allclose(desk.measure(stereo), desk.measure(mono[:, None]), rtol=1e-7)
        out = desk.apply(stereo, narrow)
        self.assertEqual(out.shape, stereo.shape)
        np.testing.assert_array_equal(out[:, 0], -out[:, 1])

    def test_live_amount_compensates_interpolated_stereo_signal(self):
        torch.set_num_threads(2)
        time = np.arange(RATE) / RATE
        low, high = [np.sin(2 * np.pi * time * f) for f in (110, 1800)]
        versions = {"off": np.column_stack((low + high, low - high)).astype(np.float32) * 0.1}
        for w in WIDTHS:
            for key, strength in ((str(w), 1.1), (f"{w}-max", 3.0)):
                versions[key] = (
                    np.column_stack(
                        (low * strength + high / strength, low * strength - high / strength)
                    ).astype(np.float32)
                    * 0.1
                )
        meta = calibrate(versions)
        for w in WIDTHS:
            for p in (0, 7, 15, 53, 100):
                a, b, t = anchors(w, p)
                signal = ((1 - t) * versions[a] + t * versions[b]) * meta["gains"][str(w)][p]
                level = loudness(torch.from_numpy(signal.T.copy()), RATE).item()
                self.assertAlmostEqual(level, meta["lufs"], delta=0.12)
                self.assertLessEqual(abs(signal).max(), 0.8501)

    def test_matched_render_keeps_timing_and_headroom(self):
        torch.set_num_threads(2)
        time = np.arange(RATE) / RATE
        tone = np.sin(2 * np.pi * time * 480).astype(np.float32)[:, None]
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / "mix"
            level = matched_audio({"off": tone, "3": tone * 0.3}, folder)
            dry, rate = sf.read(folder / "off.flac")
            eq, _ = sf.read(folder / "3.flac")
            self.assertEqual(rate, RATE)
            self.assertLessEqual(np.abs(dry).max(), 0.850001)
            self.assertIsNotNone(level)
            np.testing.assert_allclose(eq, dry, atol=2e-6)
