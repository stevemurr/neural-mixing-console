"""Fit three static source balances to a prepared 1-ERB, full-amount desk excerpt."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import oaconvolve, sosfilt

from nmc.data.raw_folder import RATE
from nmc.eq.amount import K_FILTER, block_product, match_gain
from nmc.eq.erb_desk import BrownDesk, FFT
from nmc.records import write_json


def active_level(audio):
    weighted = sosfilt(K_FILTER, audio, axis=0)
    power = block_product(weighted, weighted)
    db = -0.691 + 10 * np.log10(np.maximum(power, 1e-20))
    active = db[db >= max(-70, db.max() - 20)]
    return float(np.median(active)) if active.size else None


def loudness_faders(levels):
    audible = [v for v in levels.values() if v is not None]
    middle = np.median(audible) if audible else 0
    return {
        g: float(np.clip((middle - v) / 3, -4, 4)) if v is not None else 0
        for g, v in levels.items()
    }


def spectrum(audio):
    padded = np.pad(audio, ((0, max(0, FFT - len(audio))), (0, 0)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, FFT, axis=0)[:: FFT // 2]
    return np.fft.rfft(frames * np.hanning(FFT)).astype(np.complex64)


def band_power(spec, bank):
    # Average L/R powers after the phase-coherent sum within each channel.
    return (abs(spec) ** 2).mean(axis=1) @ bank.T


def anchored_faders(groups, active, anchor):
    """Probe static faders by turning each group down 1 dB in the actual summed mix."""
    faders = {**active, anchor: 0.0}
    bank = BrownDesk().bank.astype(np.float32)
    mix = sum(audio * 10 ** (faders[g] / 20) for g, audio in groups.items())
    full = spectrum(mix)
    total = band_power(full, bank)
    voice = band_power(spectrum(groups[anchor]), bank)
    energy = voice.sum(axis=1)
    if active_level(groups[anchor]) is None:
        raise ValueError("selected vocal anchor is inaudible in the excerpt")
    singing = energy >= energy.max() * 0.01
    occupied = voice[singing] / np.maximum(energy[singing, None], 1e-15)
    evidence = {}
    for group, audio in groups.items():
        if group == anchor:
            continue
        source = spectrum(audio) * 10 ** (faders[group] / 20)
        reduced = band_power(full + (10 ** (-1 / 20) - 1) * source, bank)
        sensitivity = 10 * np.log10((total + 1e-12) / (reduced + 1e-12))
        power = band_power(source, bank)[singing]
        dominance = power / np.maximum(power + voice[singing], 1e-12)
        pressure = np.mean((occupied * dominance * np.clip(sensitivity[singing], 0, 1)).sum(1))
        faders[group] = float(np.clip(active[group] - 4 * pressure, -4, 4))
        gaps = sensitivity[~singing] @ occupied.mean(axis=0)
        evidence[group] = dict(
            singing_sensitivity_db=float(np.mean((occupied * sensitivity[singing]).sum(1))),
            gap_sensitivity_db=float(gaps.mean()) if gaps.size else None,
        )
    return faders, evidence


def master_trims(renders):
    """Calibrate a constant output gain per mode with common peak headroom."""
    powers = {}
    peaks = {}
    for mode, audio in renders.items():
        filtered = sosfilt(K_FILTER, audio, axis=0)
        powers[mode] = block_product(filtered, filtered)
        peaks[mode] = float(np.abs(audio).max())
    target = -21.0
    for _ in range(4):
        trims = {key: match_gain(power, target) for key, power in powers.items()}
        top = max(trims[k] * peaks[k] for k in trims)
        if top <= 0.85001:
            return target, trims
        target += 20 * np.log10(0.85 / top)
    raise ValueError("could not find a shared listening level")


def voiced_track(session_dir, track, width_index, taps):
    """Full-amount, circularly filtered channel as played on the desk loop."""
    folder = session_dir / "live" / track["id"]
    dry, rate = sf.read(folder / "dry.flac", always_2d=True, dtype="float32")
    if rate != RATE:
        raise ValueError("unexpected session rate")
    kernels = np.fromfile(folder / "kernels.f32", dtype="<f4").reshape(-1, 2, taps)
    padded = np.pad(dry * track["raw_gain"], ((taps // 2, taps // 2), (0, 0)), mode="wrap")
    wet = oaconvolve(padded, kernels[width_index, 1, :, None], mode="valid", axes=0)
    return wet


def prepare(session_dir, metadata, anchor):
    path = session_dir / "session.json"
    session = json.loads(path.read_text())
    labels = yaml.safe_load(metadata.read_text())
    if not session.get("live_eq") or session["rate"] != RATE or session["title"] != labels["title"]:
        raise ValueError("expected matching live desk session and stem metadata")
    stems = labels["stems"]
    stems_by_file = {r["filename"]: g for g, s in stems.items() for r in s["raw"].values()}
    if set(stems_by_file) != {t["name"] for t in session["tracks"]} or anchor not in stems:
        raise ValueError("metadata must cover the same raw tracks and the chosen anchor")
    groups, membership = {}, {}
    width_index = session["widths"].index(1)
    taps = session["kernel_frames"]
    for track in session["tracks"]:
        group = stems_by_file[track["name"]]
        membership[track["id"]] = group
        wet = voiced_track(session_dir, track, width_index, taps)
        if group not in groups:
            groups[group] = np.zeros((len(wet), 2), dtype=np.float32)
        groups[group] += wet
    levels = {g: active_level(v) for g, v in groups.items()}
    active = loudness_faders(levels)
    anchored, evidence = anchored_faders(groups, active, anchor)
    modes = dict(unity=dict.fromkeys(groups, 0), active=active, vocal=anchored)
    renders = {
        key: sum(v * 10 ** (db[g] / 20) for g, v in groups.items()) for key, db in modes.items()
    }
    target, trims = master_trims(renders)
    session["balance"] = dict(
        width=1,
        amount=100,
        anchor=anchor,
        membership=membership,
        lufs=target,
        labels={g: s["instrument"] for g, s in stems.items()},
        active_loudness=levels,
        sensitivity=evidence,
        modes={k: dict(db=v, trim=trims[k]) for k, v in modes.items()},
    )
    write_json(path, session)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--anchor", required=True, help="MedleyDB lead-vocal stem ID")
    args = parser.parse_args()
    prepare(args.session, args.metadata, args.anchor)
