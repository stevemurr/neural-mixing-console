"""Inventory MedleyDB song folders and read aligned windows at native rate."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from nmc.data.raw_folder import discover
from nmc.data.splits import session_split
from nmc.records import write_json


def _inputs(folder, song, kind):
    directory = folder / f"{song}_{'STEMS' if kind == 'stems' else 'RAW'}"
    metadata = folder / f"{song}_METADATA.yaml"
    if metadata.is_file():
        stems = yaml.safe_load(metadata.read_text())["stems"].values()
        tracks = (
            stems if kind == "stems" else [raw for stem in stems for raw in stem["raw"].values()]
        )
        names = [t["filename"] for t in tracks]
        if any(Path(n).name != n for n in names) or len(set(names)) != len(names):
            raise ValueError("invalid or duplicate input filename in metadata")
        files = [directory / n for n in sorted(names)]
    else:
        files = [directory / name for name in discover(directory)]
    if not files:
        raise ValueError(f"no {kind} in {directory}")
    return files


def _inspect(path, root):
    info = sf.info(path)
    if info.channels not in (1, 2) or info.frames <= 0:
        raise ValueError(f"unsupported or empty audio: {path}")
    stat = path.stat()
    return {
        "path": str(path.relative_to(root)),
        "channels": info.channels,
        "rate": info.samplerate,
        "frames": info.frames,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def prepare(root, kind, output):
    root = root.resolve(strict=True)
    folders = set()
    for suffix in ("_MIX.wav", "_STEMS", "_RAW", "_METADATA.yaml"):
        folders.update((p.parent, p.name.removesuffix(suffix)) for p in root.rglob("*" + suffix))
    songs, excluded = {}, []
    for folder, song in sorted(folders):
        if folder.name != song:
            continue
        if song in songs:
            raise ValueError(f"duplicate song folder: {song}")
        try:
            mix = _inspect(folder / f"{song}_MIX.wav", root)
            tracks = [_inspect(p, root) for p in _inputs(folder, song, kind)]
            if any(t["rate"] != mix["rate"] for t in tracks):
                raise ValueError("input and mix sample rates differ")
            frames = min(t["frames"] for t in [mix, *tracks])
            songs[song] = {
                "mix": mix,
                "inputs": tracks,
                "usable_frames": frames,
                "usable_seconds": frames / mix["rate"],
            }
        except (
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            yaml.YAMLError,
        ) as err:
            excluded.append({"song": song, "reason": str(err)})
    if not folders:
        raise ValueError(f"no MedleyDB song folders under {root}")
    write_json(
        output,
        {"schema": 1, "root": str(root), "input_kind": kind, "songs": songs, "excluded": excluded},
    )
    print(f"{len(songs)} accepted, {len(excluded)} excluded: {output}")
    if not songs:
        raise ValueError("no usable songs; see inventory exclusions")


def check_source(root, track):
    path = root / track["path"]
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (track["size"], track["mtime_ns"]):
        raise ValueError(f"audio changed since preparation: {path}")
    return path


def _read(root, track, start, frames):
    path = check_source(root, track)
    data, sr = sf.read(path, start=start, frames=frames, always_2d=True, dtype="float32")
    if (
        sr != track["rate"]
        or data.shape != (frames, track["channels"])
        or not np.isfinite(data).all()
    ):
        raise ValueError(f"incomplete or invalid window in {path}")
    return np.broadcast_to(data, (frames, 2))


def load_split(recipe):
    """Assign folds by the canonical song folder name, before selecting any windows."""
    inventory = json.loads(Path(recipe["manifest"]).read_text())
    kind = inventory["input_kind"]
    if "input_kind" in recipe and kind != recipe["input_kind"]:
        raise ValueError(
            f"recipe requires {recipe['input_kind']} inputs; inventory contains {kind}"
        )
    songs = inventory["songs"]
    train, val = session_split(list(songs), recipe["val_frac"], recipe["seed"])
    if not train or not val:
        raise ValueError(f"split has {len(train)} training and {len(val)} validation songs")
    return inventory, {"train": sorted(train), "validation": sorted(val)}


def inspect_windows(recipe, output=None):
    """Read a centered window per song. Export one listening pair per fold."""
    inventory, folds = load_split(recipe)
    songs = inventory["songs"]
    report = {**folds, "input_kind": inventory["input_kind"], "windows": []}
    root = Path(inventory["root"])
    for fold, members in folds.items():
        for idx, song in enumerate(members):
            entry = songs[song]
            rate = entry["mix"]["rate"]
            frames = round(recipe["window_seconds"] * rate)
            if not 0 < frames <= entry["usable_frames"]:
                raise ValueError(f"{song}: requested window exceeds usable duration or is empty")
            start = (entry["usable_frames"] - frames) // 2
            try:
                reference = _read(root, entry["mix"], start, frames)
                dry = np.zeros((frames, 2), dtype=np.float32)
                for track in entry["inputs"]:
                    dry += _read(root, track, start, frames)
            except (OSError, RuntimeError, ValueError) as err:
                raise ValueError(f"{song}: {err}") from err
            row = {
                "song": song,
                "fold": fold,
                "start_sample": start,
                "frames": frames,
                "sample_rate": rate,
                "reference_rms": float(np.sqrt(np.mean(reference**2))),
                "dry_rms": float(np.sqrt(np.mean(dry**2))),
            }
            if output and idx == 0:
                folder = output / "audition" / fold
                folder.mkdir(parents=True)
                for name, waveform in (("reference", reference), ("dry", dry)):
                    path = folder / f"{name}.wav"
                    sf.write(path, waveform, rate, subtype="FLOAT")
                    row[name + "_wav"] = str(path.relative_to(output))
            report["windows"].append(row)
    if output:
        write_json(output / "metrics.json", report)
    print(
        f"Read {len(folds['train'])} training and {len(folds['validation'])} validation song windows"
    )
