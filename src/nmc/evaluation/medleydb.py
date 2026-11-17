"""Benchmark folder inference on held-out MedleyDB songs."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import soundfile as sf

import torch
import torch.nn.functional as F

from nmc.data.medleydb import check_source, load_split
from nmc.data.raw_folder import RATE, discover, read_source
from nmc.eq.contribution_console import ContributionConsole
from nmc.eq.target_curve_net import load_predictor
from nmc.mix import mix_folder
from nmc.records import write_json


def track_folder(root, song, entry, kind):
    reference = check_source(root, entry["mix"])
    folder = reference.parent / f"{song}_{'RAW' if kind == 'raw' else 'STEMS'}"
    expected = {check_source(root, track) for track in entry["inputs"]}
    if expected != {folder / name for name in discover(folder)}:
        raise ValueError(f"{song}: audio in {folder} differs from the prepared inventory")
    if any(reference.samefile(path) for path in expected):
        raise ValueError(f"{song}: reference mix is also an input")
    return folder


def spectrum_db(window, console):
    # Compare spectral shape at equal RMS. Short windows need STFT reflection padding.
    normalized = window / window.square().mean().clamp_min(1e-12).sqrt()
    normalized = F.pad(normalized, (0, max(0, 8192 - normalized.size(-1))))
    magnitude = console.compute_C(normalized[None, None]).expm1().clamp_min(1e-6)
    return 20 * magnitude.log10()


def save_listening_excerpt(folder, sources):
    """Match excerpt RMS and apply common peak headroom before PCM export."""
    matched = {
        name: audio / audio.square().mean().clamp_min(1e-12).sqrt()
        for name, audio in sources.items()
    }
    peak = max(float(audio.abs().max()) for audio in matched.values())
    scale = 0.89 / max(peak, 1.0)
    folder.mkdir(parents=True)
    for name, audio in matched.items():
        sf.write(folder / f"{name}.wav", (audio.T * scale).numpy(), RATE, subtype="PCM_16")


def score_song(root, entry, bundle, recipe, console):
    # The product command has already persisted the controls and rendered the entire song.
    reference = read_source(root, entry["mix"]["path"]).expand(2, -1)
    preview = read_source(bundle, "preview.wav")
    native_rate = entry["mix"]["rate"]
    length = min(reference.size(-1), preview.size(-1), entry["usable_frames"] * RATE // native_rate)
    frames = min(round(recipe["window_seconds"] * RATE), length)
    if frames < 1:
        raise ValueError(f"empty scoring window for {entry['mix']['path']}")
    # Midpoints of equal sections of the valid start range; deduplicate short songs.
    count = recipe["windows_per_song"]
    starts = sorted({(2 * i + 1) * (length - frames) // (2 * count) for i in range(count)})
    dry = torch.zeros(len(starts), 2, frames)
    for track in entry["inputs"]:
        audio = read_source(root, track["path"])
        for idx, start in enumerate(starts):
            dry[idx] += audio[:, start : start + frames]

    rows = []
    with TemporaryDirectory(dir=bundle) as stage:
        for idx, start in enumerate(starts):
            excerpt = {
                "reference": reference[:, start : start + frames],
                "dry": dry[idx],
                "predicted": preview[:, start : start + frames],
            }
            save_listening_excerpt(Path(stage) / "audition" / str(start), excerpt)
            target = spectrum_db(excerpt["reference"], console)
            dry_error = spectrum_db(dry[idx], console) - target
            predicted_error = spectrum_db(preview[:, start : start + frames], console) - target
            rows.append(
                {
                    "start_sample": start,
                    "frames": frames,
                    "dry": float(dry_error.square().mean().sqrt()),
                    "predicted": float(predicted_error.square().mean().sqrt()),
                }
            )
        (Path(stage) / "audition").rename(bundle / "audition")
    return rows


@torch.inference_mode()
def benchmark(recipe, output):
    inventory, folds = load_split(recipe)
    root = Path(inventory["root"])
    kind = inventory["input_kind"]
    folders = {
        song: track_folder(root, song, entry, kind) for song, entry in inventory["songs"].items()
    }
    if output is None:
        load_predictor(recipe["checkpoint"])
        print(
            f"Verified {kind} inventory, checkpoint and song split: {len(folds['validation'])} validation songs"
        )
        return
    console = ContributionConsole()
    results = {}
    for song in folds["validation"]:
        relative = Path("songs") / song
        bundle = output / relative
        # Only the audio folder and checkpoint cross the prediction boundary.
        mix_folder(folders[song], Path(recipe["checkpoint"]), bundle)
        windows = score_song(root, inventory["songs"][song], bundle, recipe, console)
        scores = {
            key: math.sqrt(sum(w[key] ** 2 for w in windows) / len(windows))
            for key in ("dry", "predicted")
        }
        results[song] = {
            "fold": "validation",
            "bundle": str(relative),
            "windows": windows,
            "rmse_db": scores,
        }
    summary = {
        k: math.sqrt(sum(r["rmse_db"][k] ** 2 for r in results.values()) / len(results))
        for k in ("dry", "predicted")
    }
    write_json(
        output / "metrics.json",
        {
            **folds,
            "input_kind": kind,
            "sample_rate": RATE,
            "metric": "rms_matched_active_mel_rmse_db",
            "songs": results,
            "rmse_db": summary,
        },
    )
    print(
        f"Held-out {kind} spectral RMSE over {len(results)} songs: dry {summary['dry']:.3f}, predicted {summary['predicted']:.3f} dB"
    )
