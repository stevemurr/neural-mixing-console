"""Prepare EQ training examples from staged Cambridge sessions.

Each six-second input window supplies one contribution map. All windows from
a song share the target measured from its full dry sum and engineer mix.
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch

from nmc.data.staged import load_session_audio
from nmc.eq.contribution_console import ContributionConsole


def compute_target_delta_db(
    stems: torch.Tensor, mix: torch.Tensor, console: ContributionConsole,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Measure the bandwise engineer/dry magnitude ratio after stereo RMS matching."""
    dry_sum = stems.sum(dim=0)                                  # (2, T)
    dry_rms = (dry_sum ** 2).mean().clamp(min=eps).sqrt()
    eng_rms = (mix ** 2).mean().clamp(min=eps).sqrt()
    mix_norm = mix * (dry_rms / eng_rms)                        # RMS-match

    eng_C = console.compute_C(mix_norm.unsqueeze(0).unsqueeze(0))
    dry_C = console.compute_C(dry_sum.unsqueeze(0).unsqueeze(0))
    eng_mag = torch.expm1(eng_C).clamp(min=eps).squeeze()
    dry_mag = torch.expm1(dry_C).clamp(min=eps).squeeze()
    return 20.0 * torch.log10(eng_mag / dry_mag)


def main(
    *, default_output: str = "dmc-data/eq/target-curve-examples.pt",
    extra_labels: Callable[[torch.Tensor, torch.Tensor, ContributionConsole, list[int], int], dict]
    | None = None,
) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--staging-dir", default="dmc-data/grafx-prune-data-48k")
    ap.add_argument("--out", default=default_output)
    ap.add_argument("--sample-rate", type=int, default=48_000)
    ap.add_argument("--audio-len", type=int, default=288_000,
                    help="Samples per window (default 6 s @ 48 kHz)")
    ap.add_argument("--n-windows-per-session", type=int, default=20)
    ap.add_argument("--n-max", type=int, default=64,
                    help="Maximum number of tracks per song. Longer lists are truncated.")
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
        N_real = stems.shape[0]
        if N_real > args.n_max:
            stems = stems[:args.n_max]
            N_real = args.n_max

        try:
            with torch.no_grad():
                target_delta_db = compute_target_delta_db(stems, mix, console).cpu()
                labels = extra_labels(
                    stems, mix, console, sess["stem_groups"][:N_real], args.n_max,
                ) if extra_labels else {}
        except Exception as e:
            logging.warning(f"{session_dir.name}: target failed ({e}); skipped")
            n_skipped += 1
            continue

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
                **labels,
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
