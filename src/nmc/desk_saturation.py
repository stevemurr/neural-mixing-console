"""Prepare matched monitoring trims for a realtime, transformer-inspired channel stage."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import lfilter, resample_poly

from nmc.data.raw_folder import RATE
from nmc.desk_balance import master_trims, voiced_track
from nmc.records import write_json

DRIVES_DB = {"line": 0, "driven": 6, "hot": 12}
DRIVE_GRID_DB = list(range(-12, 25, 3))
BIAS = 0.14


def stage_spec():
    # Tilt LF into the soft knee; the reciprocal filter restores the small-signal response.
    pole = float(np.exp(-2 * np.pi * 180 / RATE))
    dc_pole = float(np.exp(-2 * np.pi * 8 / RATE))
    return dict(
        feed=[1 + (10 ** (5 / 20) - 1) * (1 - pole), -pole],
        feedback=[1, -pole],
        dc_pole=dc_pole,
        bias=BIAS,
        drives_db=DRIVES_DB,
        drive_grid_db=DRIVE_GRID_DB,
    )


def line_drive(x):
    hop = RATE // 20
    padded = np.pad(x, ((0, (-len(x)) % hop), (0, 0)))
    rms = np.sqrt(np.mean(padded.reshape(-1, hop, x.shape[1]) ** 2, axis=(1, 2)))
    active = rms[rms > max(rms.max() * 0.032, 1e-8)]
    return float(np.clip(0.30 / np.percentile(active, 90), 0.25, 8)) if active.size else 1.0


def soft_knee(x, bias):
    return (np.tanh(x + bias) - np.tanh(bias)) / (1 - np.tanh(bias) ** 2)


def render_colors(audio, spec):
    # Real loop history warms the filters at the start of the measured excerpt.
    margin = 8192
    loop = np.pad(audio, ((margin, margin), (0, 0)), mode="wrap")
    tilted = lfilter(spec["feed"], spec["feedback"], loop, axis=0)
    nominal = line_drive(tilted[margin:-margin])
    oversampled = resample_poly(tilted, 4, 1, axis=0)
    variants = {"clean": audio}
    pole = spec["dc_pole"]
    for extra in spec["drive_grid_db"]:
        drive = nominal * 10 ** (extra / 20)
        shaped = soft_knee(np.clip(oversampled * drive, -8, 8), spec["bias"]) / drive
        base = resample_poly(shaped, 1, 4, axis=0)
        restored = lfilter(spec["feedback"], spec["feed"], base, axis=0)
        colored = lfilter([(1 + pole) / 2, -(1 + pole) / 2], [1, -pole], restored, axis=0)
        variants[str(extra)] = colored[margin:-margin].astype(np.float32)
    return nominal, variants


def matched_trims(renders, spec):
    level, trims = master_trims(renders)
    # Older desk clients can still select the three original drive settings.
    trims.update({name: trims[str(db)] for name, db in spec["drives_db"].items()})
    return level, trims


def prepare(session_dir):
    path = session_dir / "session.json"
    session = json.loads(path.read_text())
    if not session.get("live_eq") or session["rate"] != RATE:
        raise ValueError("expected a 48 kHz live EQ session")
    spec = stage_spec()
    master = {}
    for track in session["tracks"]:
        clean = voiced_track(
            session_dir, track, session["widths"].index(1), session["kernel_frames"]
        )
        drive, variants = render_colors(clean, spec)
        stereo = {key: np.broadcast_to(x, (len(x), 2)) for key, x in variants.items()}
        for key, x in stereo.items():
            if key not in master:
                master[key] = np.zeros_like(x)
            master[key] += x
        level, trims = matched_trims(stereo, spec)
        track["saturation"] = dict(drive=drive, trims=trims, lufs=level if np.any(clean) else None)
    level, trims = matched_trims(master, spec)
    session["saturation"] = dict(
        **spec, trims=trims, lufs=level if np.any(master["clean"]) else None
    )
    write_json(path, session)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    prepare(parser.parse_args().session)
