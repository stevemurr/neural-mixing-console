"""Predict static controls from raw tracks and replay them without the predictor."""

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import soundfile as sf
import torch
import torch.nn.functional as F

from nmc.data.raw_folder import RATE, discover, read_source
from nmc.eq.contribution_console import ContributionConsole
from nmc.eq.mel_bin_eq import MelBinEQ
from nmc.eq.target_curve_net import load_predictor
from nmc.records import fingerprint, write_json

DEFAULT_CHECKPOINT = Path("dmc-data/checkpoints/v13-target-curve/target_curve_best.pt")
PASSAGE = 6 * RATE
FFT_SIZE = 65536
PAN_LAW = {"mono": "equal_power_center_unity", "stereo": "cosine_balance_center_unity"}


def contribution(audio, console):
    """Select the loudest six-second block in each third of the track's blocks."""
    blocks = audio.split(PASSAGE, dim=-1)
    energy = torch.stack([part.square().mean() for part in blocks])
    selected = []
    features = []
    for indices in torch.arange(len(blocks)).tensor_split(min(3, len(blocks))):
        idx = int(indices[energy[indices].argmax()])
        if energy[idx] == 0:
            continue
        part = blocks[idx]
        # Reflection padding inside MelSpectrogram needs > n_fft/2 samples.
        part = F.pad(part, (0, max(0, 8192 - part.size(-1))))
        features.append(console.compute_C(part.expand(2, -1)[None, None])[0, 0])
        selected.append(idx * PASSAGE)
    if not features:
        return torch.zeros(console.n_bins), []
    return torch.stack(features).mean(0), selected


def predict(root, checkpoint):
    model = load_predictor(checkpoint)
    console = ContributionConsole(n_bins=model.n_bins)
    tracks, features = [], []
    for name in discover(root):
        digest = fingerprint(root / name)
        audio = read_source(root, name)
        feature, starts = contribution(audio, console)
        tracks.append(
            {
                "path": name,
                "sha256": digest,
                "channels": audio.size(0),
                "analysis_starts": starts,
                "level_db": 0.0,
                "pan": 0.0,
            }
        )
        features.append(feature)
    matrix = torch.stack(features)[None]
    mask = torch.tensor([[bool(t["analysis_starts"]) for t in tracks]])
    curve = model(matrix, mask)
    gains = console.distribute_delta(curve, matrix, mask).clamp(
        -console.eq_gain_range_db, console.eq_gain_range_db
    )
    for track, db in zip(tracks, gains[0]):
        track["eq_db"] = db.tolist()
    return {
        "version": 1,
        "sample_rate": RATE,
        "band_edges_hz": console.eq.hz_edges.tolist(),
        "pan_law": PAN_LAW.copy(),
        "master_db": 0.0,
        "tracks": tracks,
        "checkpoint": Path(checkpoint).name,
        "checkpoint_sha256": fingerprint(checkpoint),
    }


def equalize(audio, eq, db):
    """Square-root Hann overlap-add keeps FFT working memory bounded per track."""
    if not torch.any(db):
        return audio
    hop = FFT_SIZE // 2
    end_pad = (-audio.size(-1)) % hop
    padded = F.pad(audio, (hop, end_pad + hop))
    windows = padded.unfold(-1, FFT_SIZE, hop).transpose(0, 1)
    result = torch.zeros_like(padded)
    taper = torch.hann_window(FFT_SIZE).sqrt()
    norm = (db / eq.gain_range_db + 1) / 2
    for start in range(0, len(windows), 8):
        batch = windows[start : start + 8] * taper
        filtered = eq(batch, norm.expand(len(batch), -1)) * taper
        for offset, block in enumerate(filtered):
            pos = (start + offset) * hop
            result[:, pos : pos + FFT_SIZE] += block
    return result[:, hop : hop + audio.size(-1)]


def number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return value


def apply_pan(audio, pan):
    if not -1 <= pan <= 1:
        raise ValueError("pan must be between -1 and 1")
    if pan == 0:
        return audio.expand(2, -1)
    if audio.size(0) == 1:
        angle = (1 + pan) * math.pi / 4
        gains = [math.sqrt(2) * math.cos(angle), math.sqrt(2) * math.sin(angle)]
    else:
        gains = [math.cos(max(pan, 0) * math.pi / 2), math.cos(min(pan, 0) * math.pi / 2)]
    return audio * audio.new_tensor(gains)[:, None]


def render_tracks(root, settings, *, measure):
    if (
        settings["version"] != 1
        or settings["sample_rate"] != RATE
        or settings["pan_law"] != PAN_LAW
    ):
        raise ValueError("unsupported mix format, sample rate or pan law")
    records = settings["tracks"]
    names = [t["path"] for t in records]
    if len(names) != len(set(names)):
        raise ValueError("duplicate track paths in mix settings")
    actual = set(discover(root))
    saved = set(names)
    if actual != saved:
        raise ValueError(
            f"track list changed: missing {sorted(saved - actual)}, added {sorted(actual - saved)}"
        )
    eq = MelBinEQ(n_bins=len(settings["band_edges_hz"]) - 1, gain_invariant=False)
    if settings["band_edges_hz"] != eq.hz_edges.tolist():
        raise ValueError("band edges do not match the v1 EQ")
    mix = torch.zeros(2, 0)
    for track in sorted(records, key=lambda t: t["path"]):
        name = track["path"]
        if fingerprint(root / name) != track["sha256"]:
            raise ValueError(f"{name}: source fingerprint changed")
        audio = read_source(root, name)
        if audio.size(0) != track["channels"]:
            raise ValueError(f"{name}: source channel count differs")
        db = torch.tensor([number(v, "eq_db") for v in track["eq_db"]], dtype=torch.float32)
        if db.shape != (eq.n_bins,) or torch.any(db.abs() > eq.gain_range_db):
            raise ValueError(f"{name}: expected {eq.n_bins} EQ gains in ±{eq.gain_range_db} dB")
        shaped = equalize(audio, eq, db)
        if measure:
            ratio = audio.square().mean().clamp_min(1e-12) / shaped.square().mean().clamp_min(1e-12)
            track["eq_compensation_db"] = float(10 * torch.log10(ratio))
        gain_db = number(track["eq_compensation_db"], "eq_compensation_db") + number(
            track["level_db"], "level_db"
        )
        placed = apply_pan(shaped, number(track["pan"], "pan")) * (10 ** (gain_db / 20))
        if mix.size(-1) < placed.size(-1):
            mix = F.pad(mix, (0, placed.size(-1) - mix.size(-1)))
        mix[:, : placed.size(-1)] += placed
    if measure:
        peak = float(mix.abs().max())
        settings["master_db"] = min(0.0, -1 - 20 * math.log10(peak)) if peak > 0 else 0.0
    mix *= 10 ** (number(settings["master_db"], "master_db") / 20)
    if not torch.isfinite(mix).all():
        raise ValueError("render contains non-finite samples")
    return mix


def outside_sources(root, output):
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("choose an output outside the raw tracks folder")


@torch.inference_mode()
def mix_folder(root, checkpoint, out_dir):
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    root = root.resolve()
    outside_sources(root, out_dir)
    if out_dir.exists():
        raise ValueError(f"choose a new mix output directory: {out_dir}")
    settings = predict(root, checkpoint)
    audio = render_tracks(root, settings, measure=True)
    out_dir.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=out_dir.parent) as temp:
        bundle = Path(temp) / "mix"
        bundle.mkdir()
        write_json(bundle / "mix.json", settings)
        sf.write(bundle / "preview.wav", audio.T.numpy(), RATE, subtype="FLOAT")
        bundle.rename(out_dir)
    print(f"{len(settings['tracks'])} tracks: {out_dir / 'mix.json'} and {out_dir / 'preview.wav'}")


@torch.inference_mode()
def replay(root, document, output):
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    outside_sources(root, output)
    if output.exists():
        raise ValueError(f"choose a new render output: {output}")
    if output.suffix.lower() != ".wav":
        raise ValueError("render output must be a .wav file")
    settings = json.loads(document.read_text())
    audio = render_tracks(root.resolve(), settings, measure=False)
    output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        wav = Path(temp) / "render.wav"
        sf.write(wav, audio.T.numpy(), RATE, subtype="FLOAT")
        # Publish only a complete file and preserve existing renders.
        output.hardlink_to(wav)
    print(f"Rendered {len(settings['tracks'])} tracks to {output}")
