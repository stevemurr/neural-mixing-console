"""Precompute (C, target_delta_db) examples for v13 Stage 2 training.

Stage 2 maps per-track contribution vectors C → mix-bus dB delta. Both
are derived from the staged 48 kHz audio:

  - Per-session target_delta_db (26,):
      compute the top-active-frame mel mag of the engineer mix vs the
      dry sum of stems; the dB ratio in 26 mel bins is the target. This
      is what the engineer's mix bus did to the dry sum, spectrally.
      One vector per session (static — engineer's EQ is set once).

  - Per-window C[stem, bin] (n_max, 26):
      same top-active-frame log1p-mel aggregation, but on a 6 s window
      of the dry stems. Captures the per-track spectral signature in
      that window. K windows per session → K examples per session, all
      sharing the same session-level target.

The precomputed dataset is small (~14 MB for 5800 examples at K=20).
Loads into memory at training time → no audio I/O during training.

Usage:
    uv run python v13/scripts/precompute_v13_dataset.py
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

# Repo root on sys.path so `models.contribution_console` resolves when
# this is run as `uv run python v13/scripts/precompute_v13_dataset.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.contribution_console import ContributionConsole
from models.group_classification import UNKNOWN_IDX, group_name_to_idx
from models.mono_pan import detect_stereo, recover_pan_targets


def load_session_audio(
    session_dir: Path, sample_rate: int,
) -> dict | None:
    """Load engineer mix + stems for a session, aligned + stereo + (N, 2, T).

    Also returns `stem_groups`: the canonical-group index for each stem,
    derived from the correspondence.yaml group it lives under. This is the
    input feature that lets Stage 3 learn engineering conventions like
    "drum-group tracks tend center, guitar-group tracks tend spread."
    """
    corr_path = session_dir / "correspondence.yaml"
    mix_path = session_dir / "mix.wav"
    if not (corr_path.is_file() and mix_path.is_file()):
        return None
    try:
        corr = yaml.safe_load(open(corr_path))
    except Exception:
        return None
    if not isinstance(corr, dict):
        return None
    # Flatten stems while preserving group identity per stem
    stem_filenames: list[str] = []
    stem_groups: list[int] = []
    for group_name, files in corr.items():
        if not isinstance(files, list):
            continue
        g_idx = group_name_to_idx(group_name)
        for f in files:
            stem_filenames.append(f)
            stem_groups.append(g_idx)
    if not stem_filenames:
        return None

    mix_audio, sr = sf.read(str(mix_path), dtype="float32", always_2d=True)
    if sr != sample_rate or mix_audio.shape[0] < 4096:
        return None
    if mix_audio.shape[1] == 1:
        mix_audio = np.repeat(mix_audio, 2, axis=1)
    mix_audio = mix_audio[:, :2]

    stems_list: list[np.ndarray] = []
    fnames: list[str] = []
    fgroups: list[int] = []
    for fname, g_idx in zip(stem_filenames, stem_groups):
        path = session_dir / "stems" / fname
        if not path.is_file():
            continue
        try:
            s, sr_s = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception:
            continue
        if sr_s != sample_rate or s.shape[0] < 4096:
            continue
        if s.shape[1] == 1:
            s = np.repeat(s, 2, axis=1)
        s = s[:, :2]
        stems_list.append(s)
        fnames.append(fname)
        fgroups.append(g_idx)

    if not stems_list:
        return None

    T = min([s.shape[0] for s in stems_list] + [mix_audio.shape[0]])
    mix_audio = mix_audio[:T]
    stems_arr = np.stack([s[:T] for s in stems_list], axis=0)   # (N, T, 2)
    stems = torch.from_numpy(np.ascontiguousarray(stems_arr.transpose(0, 2, 1)))
    mix = torch.from_numpy(np.ascontiguousarray(mix_audio.T))   # (2, T)
    return {
        "stems":          stems,
        "mix":            mix,
        "stem_filenames": fnames,
        "stem_groups":    fgroups,                # (N,) list of canonical group indices
        "T_total":        T,
    }


def compute_target_delta_db(
    stems: torch.Tensor, mix: torch.Tensor, console: ContributionConsole,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Engineer-mix - dry-sum delta in dB, after loudness-matching.

    BEFORE computing the delta we RMS-match the engineer mix to the dry
    sum. The engineer mix is loudness-normalized (typically -14 LUFS for
    streaming) and the dry sum is at raw recording level — without
    matching, the raw delta carries a ~+5 dB constant level offset that
    isn't really EQ. After matching, the delta is purely the spectral
    SHAPE difference: positive = engineer brought that band up *relative
    to overall level*; negative = brought it down.

    This is what v13's gain-invariant EQ can actually represent. The
    master-gain/loudness layer is a separate concern (future stage).

    Returns (n_bins,) dB delta.
    """
    dry_sum = stems.sum(dim=0)                                  # (2, T)
    dry_rms = (dry_sum ** 2).mean().clamp(min=eps).sqrt()
    eng_rms = (mix ** 2).mean().clamp(min=eps).sqrt()
    mix_norm = mix * (dry_rms / eng_rms)                        # RMS-match

    eng_C = console.compute_C(mix_norm.unsqueeze(0).unsqueeze(0))
    dry_C = console.compute_C(dry_sum.unsqueeze(0).unsqueeze(0))
    eng_mag = torch.expm1(eng_C).clamp(min=eps).squeeze()
    dry_mag = torch.expm1(dry_C).clamp(min=eps).squeeze()
    return 20.0 * torch.log10(eng_mag / dry_mag)


def compute_pan_targets_session(
    stems: torch.Tensor,                # (N, 2, T)
    mix: torch.Tensor,                  # (2, T)
    console: ContributionConsole,
    stereo_threshold: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Session-level per-track pan targets via LS recovery.

    Procedure:
      1. Detect stereo stems (`is_stereo[t]`) by L-vs-R RMS divergence.
      2. Per-stem raw mel mag (mono and per-channel L/R), top-active aggregated.
      3. Per-mix raw mel mag L and R.
      4. Subtract stereo stems' inherent L-bias contribution from the engineer
         L-bias.
      5. LS-solve for the pan-direction factor s[t] on mono stems only.
      6. Recover pan_dir from s via constant-power inverse.

    Returns (pan_dir[N], is_stereo[N]).
    """
    with torch.no_grad():
        # (1) stereo detection
        is_stereo = detect_stereo(stems, threshold=stereo_threshold)        # (N,)

        # (2) per-stem mel mag mono + per-channel L/R
        stems_b = stems.unsqueeze(0)                                         # (1, N, 2, T)
        stem_mag_mono = console.compute_mel_mag(stems_b).squeeze(0)         # (N, n_bins)
        stem_mag_L, stem_mag_R = console.compute_mel_mag_LR(stems_b)
        stem_mag_L = stem_mag_L.squeeze(0)
        stem_mag_R = stem_mag_R.squeeze(0)

        # (3) per-mix mel mag L and R — treat mix as (1, 1, 2, T)
        mix_b = mix.unsqueeze(0).unsqueeze(0)
        mix_mag_L, mix_mag_R = console.compute_mel_mag_LR(mix_b)
        mix_mag_L = mix_mag_L.squeeze()                                      # (n_bins,)
        mix_mag_R = mix_mag_R.squeeze()

        # (4)+(5)+(6) LS
        pan_dir = recover_pan_targets(
            stem_mel_mag=stem_mag_mono,
            stem_mel_L=stem_mag_L,
            stem_mel_R=stem_mag_R,
            mix_mel_L=mix_mag_L,
            mix_mel_R=mix_mag_R,
            is_stereo=is_stereo,
        )
    return pan_dir, is_stereo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--staging-dir", default="dmc-data/grafx-prune-data-48k")
    ap.add_argument("--out", default="dmc-data/v13_precomputed.pt")
    ap.add_argument("--sample-rate", type=int, default=48_000)
    ap.add_argument("--audio-len", type=int, default=288_000,
                    help="Samples per window (default 6 s @ 48 kHz)")
    ap.add_argument("--n-windows-per-session", type=int, default=20)
    ap.add_argument("--n-max", type=int, default=64,
                    help="Max stems per session — truncates longer (rare >5% of "
                         "corpus), pads shorter. Bumped from 24 because 46%% of "
                         "Cambridge sessions exceed 24 stems and truncating "
                         "silently drops everything past Keys/Synth in "
                         "correspondence.yaml order (vocals get nuked).")
    ap.add_argument("--n-bins", type=int, default=26)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    console = ContributionConsole(
        sample_rate=args.sample_rate, n_bins=args.n_bins,
    )

    staging = Path(args.staging_dir)
    if not staging.is_dir():
        ap.error(f"staging dir not found: {staging}")

    sessions = sorted(p for p in staging.iterdir()
                      if p.is_dir() and (p / "mix.wav").is_file())
    logging.info(f"found {len(sessions)} candidate sessions in {staging}")

    examples: list[dict] = []
    t0 = time.time()
    n_skipped = 0
    for i, session_dir in enumerate(sessions, 1):
        sess = load_session_audio(session_dir, args.sample_rate)
        if sess is None:
            n_skipped += 1
            continue
        stems, mix, T = sess["stems"], sess["mix"], sess["T_total"]
        stem_groups = sess["stem_groups"]
        N_real = stems.shape[0]
        if N_real > args.n_max:
            stems = stems[:args.n_max]
            stem_groups = stem_groups[:args.n_max]
            N_real = args.n_max

        try:
            with torch.no_grad():
                target_delta_db = compute_target_delta_db(stems, mix, console).cpu()
                pan_target_real, is_stereo_real = compute_pan_targets_session(
                    stems, mix, console,
                )
                pan_target_real = pan_target_real.cpu()
                is_stereo_real = is_stereo_real.cpu()
        except Exception as e:
            logging.warning(f"{session_dir.name}: target failed ({e}); skipped")
            n_skipped += 1
            continue

        # Pad pan targets to n_max (padded slots get pan=0, is_stereo=False)
        pan_target = torch.zeros(args.n_max, dtype=torch.float32)
        pan_target[:N_real] = pan_target_real.to(torch.float32)
        is_stereo_full = torch.zeros(args.n_max, dtype=torch.bool)
        is_stereo_full[:N_real] = is_stereo_real
        # Group-index per track (padded slots get UNKNOWN_IDX → embed as "no info")
        group_idx = torch.full((args.n_max,), UNKNOWN_IDX, dtype=torch.long)
        for i, g in enumerate(stem_groups[:N_real]):
            group_idx[i] = int(g)

        if T < args.audio_len + 1:
            n_skipped += 1
            continue

        for _ in range(args.n_windows_per_session):
            start = random.randint(0, T - args.audio_len - 1)
            win = stems[:, :, start:start + args.audio_len]    # (N_real, 2, audio_len)
            if N_real < args.n_max:
                pad = torch.zeros(args.n_max - N_real, 2, args.audio_len)
                padded = torch.cat([win, pad], dim=0)
            else:
                padded = win

            with torch.no_grad():
                C = console.compute_C(padded.unsqueeze(0)).squeeze(0).cpu()    # (n_max, n_bins)

            track_mask = torch.zeros(args.n_max, dtype=torch.bool)
            track_mask[:N_real] = True

            examples.append({
                "C":               C.to(torch.float32),
                "target_delta_db": target_delta_db.to(torch.float32),
                "pan_target":      pan_target,
                "is_stereo":       is_stereo_full,
                "group_idx":       group_idx,
                "track_mask":      track_mask,
                "session":         session_dir.name,
                "window_start":    int(start),
                "n_real_tracks":   int(N_real),
            })

        if i % 10 == 0 or i == len(sessions):
            dt = time.time() - t0
            logging.info(
                f"[{i:3d}/{len(sessions)}]  examples={len(examples):5d}  "
                f"elapsed={dt:6.1f}s  rate={i/max(dt, 1e-3):.2f} sess/s"
            )

    n_uniq_sessions = len(set(e["session"] for e in examples))
    logging.info(
        f"final: {len(examples)} examples from {n_uniq_sessions} sessions  "
        f"({n_skipped} sessions skipped). Saving to {args.out} ..."
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(examples, args.out)
    sz_mb = Path(args.out).stat().st_size / (1024 ** 2)
    logging.info(f"wrote {args.out}  ({sz_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
