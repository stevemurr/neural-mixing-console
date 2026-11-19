"""Prepare a level-matched ERB desk audition from a folder of audio tracks."""

from collections import Counter
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import soundfile as sf
import torch
from torchaudio.functional import loudness
from scipy.signal import oaconvolve

from nmc.data.raw_folder import RATE, discover, read_source
from nmc.eq.amount import calibrate
from nmc.eq.erb_desk import AMOUNT, FFT, LIMIT_DB, WIDTHS, BrownDesk
from nmc.records import write_json

CHOICES = ("off", *(str(w) for w in WIDTHS), *(f"{w}-max" for w in WIDTHS))


def choose_excerpt(paths, seconds, start):
    infos = [sf.info(p) for p in paths]
    duration = max(i.frames / i.samplerate for i in infos)
    if seconds <= 0 or not np.isfinite(seconds):
        raise ValueError("excerpt length must be positive")
    length = min(seconds, duration)
    if start is not None:
        if not 0 <= start < duration:
            raise ValueError("start is outside the song")
        return start, min(length, duration - start)
    candidates = np.linspace(0, duration - length, 12)
    scores = np.zeros(len(candidates))
    for path in paths:
        with sf.SoundFile(path) as source:
            levels = []
            for pos in candidates:
                offset = int((pos + length / 2) * source.samplerate)
                source.seek(min(offset, len(source)))
                block = source.read(
                    max(1, int(min(1, length / 2) * source.samplerate)), dtype="float32"
                )
                levels.append(np.mean(block**2) if block.size else 0)
            scores += np.sqrt(np.asarray(levels) / max(max(levels), 1e-12))
    return candidates[scores.argmax()], length


def matched_audio(versions, folder):
    """Match the four alternatives within this bus and apply common peak headroom."""
    folder.mkdir()
    gains, values = {}, {}
    for key, audio in versions.items():
        peak = np.abs(audio).max()
        if peak < 1e-9:
            gains[key] = 1.0
            continue
        # Measure at a safe level for the K-weighting filter; undo the calibration.
        calibrated = torch.from_numpy(np.ascontiguousarray((audio * (0.1 / peak)).T))
        if calibrated.shape[-1] < RATE // 2:
            calibrated = torch.nn.functional.pad(calibrated, (0, RATE // 2 - calibrated.shape[-1]))
        value = loudness(calibrated, RATE).item() + 20 * np.log10(peak / 0.1)
        if not np.isfinite(value):
            raise ValueError("could not measure audition loudness")
        gains[key] = 10 ** ((-21 - value) / 20)
        values[key] = value
    peak = max(np.abs(v).max() * gains[k] for k, v in versions.items())
    headroom = min(1, 0.85 / max(peak, 1e-12))
    for key, audio in versions.items():
        sf.write(folder / f"{key}.flac", audio * (headroom * gains[key]), RATE, subtype="PCM_24")
    return round(float(-21 + 20 * np.log10(headroom)), 2) if values else None


def envelope(audio, bins=320):
    signal = np.abs(audio).max(axis=1)
    stride = max(1, int(np.ceil(len(signal) / bins)))
    padded = np.pad(signal, (0, stride * bins - len(signal)))
    peaks = padded.reshape(bins, stride).max(axis=1)
    return np.round(peaks / max(peaks.max(), 1e-9), 3).tolist()


def prepare(
    root, runs, *, title=None, artist=None, seconds=20, start=None, metadata=None, upgrade=None
):
    root = root.resolve()
    names = discover(root)
    base = runs / "erb-desk"
    previous = None
    if upgrade:
        if not re.fullmatch(r"[0-9a-f]{12}", upgrade):
            raise ValueError("invalid desk session ID")
        previous = json.loads((base / upgrade / "session.json").read_text())
        if names != [t["name"] for t in previous["tracks"]] or previous.get("live_eq"):
            raise ValueError("expected matching raw tracks and a fixed-amount session")
        title, artist, start, seconds = (
            previous[k] for k in ("title", "artist", "start", "duration")
        )
    start, length = choose_excerpt([root / name for name in names], seconds, start)
    first, frames = round(start * RATE), max(1, round(length * RATE))
    desk = BrownDesk()
    labels = {}
    if metadata:
        import yaml

        doc = yaml.safe_load(Path(metadata).read_text())
        for stem in doc["stems"].values():
            for raw in stem["raw"].values():
                labels[raw["filename"]] = raw["instrument"]
        artist = artist or doc.get("artist")
        title = title or doc.get("title")
    labels = [str(labels.get(name, Path(name).stem)) for name in names]
    counts, used = Counter(labels), Counter()
    base.mkdir(parents=True, exist_ok=True)
    session_id = upgrade or uuid4().hex[:12]
    with TemporaryDirectory(dir=base) as tmp:
        stage = Path(tmp) / "session"
        stage.mkdir()
        live = stage / "live"
        live.mkdir()
        master = {k: np.zeros((frames, 2), dtype=np.float32) for k in CHOICES}
        channels = []
        for i, (name, label) in enumerate(zip(names, labels, strict=True), 1):
            audio = read_source(root, name).numpy().T
            power = desk.measure(audio)
            dry = audio[first : first + frames]
            dry = np.pad(dry, ((0, frames - len(dry)), (0, 0)))
            bus = f"ch{i:03}"
            folder = live / bus
            folder.mkdir()
            # Restore this encoding gain before summing the raw channels in the browser.
            raw_gain = max(float(np.abs(dry).max()), 1e-9) / 0.85
            sf.write(folder / "dry.flac", dry / raw_gain, RATE, subtype="PCM_24")
            versions = {"off": dry}
            curves, kernels = {}, []
            margin = FFT // 2
            padded = np.pad(dry, ((margin, margin), (0, 0)), mode="wrap")
            for width in WIDTHS:
                curve = desk.curve(power, width)
                curves[str(width)] = np.round(curve, 4).tolist()
                for key, gains in ((str(width), curve), (f"{width}-max", curve / AMOUNT)):
                    kernel = desk.kernel(gains)
                    kernels.append(kernel)
                    versions[key] = oaconvolve(padded, kernel[:, None], mode="valid", axes=0)
            np.asarray(kernels, dtype="<f4").tofile(folder / "kernels.f32")
            for key, waveform in versions.items():
                master[key] += waveform
            trim = calibrate({k: np.broadcast_to(v, (frames, 2)) for k, v in versions.items()})
            used[label] += 1
            channels.append(
                dict(
                    id=bus,
                    name=name,
                    label=label if counts[label] == 1 else f"{label} {used[label]}",
                    channels=audio.shape[1],
                    peaks=envelope(versions["off"]),
                    curves=curves,
                    **trim,
                    raw_gain=raw_gain,
                )
            )
        if previous:
            for track, old in zip(channels, previous["tracks"], strict=True):
                track["label"] = old["label"]
        trim = calibrate(master)
        session = dict(
            id=session_id,
            title=title or root.name,
            artist=artist or "",
            start=first / RATE,
            duration=frames / RATE,
            rate=RATE,
            centers=desk.centers.tolist(),
            widths=WIDTHS,
            amount=AMOUNT,
            limit_db=LIMIT_DB,
            tracks=channels,
            **trim,
            live_eq=True,
            kernel_frames=FFT + 1,
        )
        write_json(stage / "session.json", session)
        if previous:
            live.rename(base / session_id / "live")
            (stage / "session.json").replace(base / session_id / "session.json")
        else:
            stage.rename(base / session_id)
    return base / session_id


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tracks", type=Path)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--title")
    p.add_argument("--artist")
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--start", type=float)
    p.add_argument("--upgrade", help="upgrade a fixed-amount session in place, preserving its URL")
    p.add_argument("--metadata", type=Path, help="optional MedleyDB YAML for display labels")
    args = p.parse_args()
    torch.set_num_threads(2)
    print(
        prepare(
            args.tracks,
            args.runs,
            title=args.title,
            artist=args.artist,
            seconds=args.seconds,
            start=args.start,
            metadata=args.metadata,
            upgrade=args.upgrade,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
