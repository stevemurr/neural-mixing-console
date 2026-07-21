"""Diagnose pan behavior of a v13 pan-recon checkpoint across all val sessions.

Two questions this answers:

  1. VOCAL-LEFT BIAS — does the model pan vocals left? We bucket every mono
     track's predicted pan_dir by instrument category (from correspondence.yaml
     group names) and report mean / median / %L-C-R per category. Sign
     convention: pan_dir < 0 = LEFT, > 0 = RIGHT.

  2. MODEL vs DATA — for each track we also compute the LS-recovered engineer
     pan_dir (mono_pan.recover_pan_targets) and the engineer mix's per-band
     balance. If the model's vocal-left lean matches the LS target, the lean
     is in the data; if the model leans left where LS says center, it's a
     model/loss artifact.

  3. WIDTH USAGE — distribution of |pan_dir| (model vs LS engineer target):
     how much of the pan space each actually uses, and what fraction is
     "hard" (|pan| > 0.8).

Usage:
    uv run python v13/scripts/analyze_pan_bias.py \
        --pan-ckpt dmc-data/checkpoints/v13-pan-recon-w1.0/pan_recon_best.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.contribution_console import ContributionConsole
from models.group_classification import classify_group_name
from models.mono_pan import detect_stereo, recover_pan_targets
from models.pan_net import PanNet
from scripts.precompute_v13_dataset import load_session_audio
from training.data_v13 import session_split


def per_track_groups(session_dir: Path, n_real: int) -> list[str]:
    """Canonical category per flattened track index (same walk as audition)."""
    cats = ["Unknown"] * n_real
    try:
        corr = yaml.safe_load(open(session_dir / "correspondence.yaml"))
        if isinstance(corr, dict):
            flat = 0
            for group_name, files in corr.items():
                if not isinstance(files, list):
                    continue
                cat = classify_group_name(group_name)
                for _ in files:
                    if flat < n_real:
                        cats[flat] = cat
                    flat += 1
    except Exception as e:
        logging.warning(f"{session_dir.name}: group parse failed ({e})")
    return cats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pan-ckpt",
                    default="dmc-data/checkpoints/v13-pan-recon-w1.0/pan_recon_best.pt")
    ap.add_argument("--staging-dir", default="dmc-data/grafx-prune-data-48k")
    ap.add_argument("--precomputed", default="dmc-data/v13_precomputed.pt")
    ap.add_argument("--sample-rate", type=int, default=48_000)
    ap.add_argument("--audio-len", type=int, default=10 * 48_000)
    ap.add_argument("--window-start-frac", type=float, default=0.25)
    ap.add_argument("--n-max", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    pck = torch.load(args.pan_ckpt, weights_only=False, map_location="cpu")
    pa = pck["args"]
    pan_net = PanNet(
        n_bins=pa["n_bins"], d_model=pa["d_model"], n_layers=pa["n_layers"],
        n_heads=pa["n_heads"], ffn_mult=pa["ffn_mult"], dropout=pa["dropout"],
        use_group_embed=pa.get("use_group_embed", False),
        continuous=pa.get("continuous", True),
    ).to(device).eval()
    pan_net.load_state_dict(pck["model"])
    console = ContributionConsole(sample_rate=args.sample_rate, n_bins=pa["n_bins"]).to(device).eval()
    logging.info(f"pan ckpt step={pck['step']}  use_group_embed={pa.get('use_group_embed')}")

    # Mel for LS recovery + engineer balance (matches PanReconLoss defaults).
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate, n_fft=8192, hop_length=2048, n_mels=pa["n_bins"],
        f_min=20.0, f_max=16_000.0, power=1.0, norm=None, mel_scale="htk",
    ).to(device)
    band_centers = None  # set from console edges below for the balance-by-band print

    examples = torch.load(args.precomputed, weights_only=False, map_location="cpu")
    all_sessions = sorted({e["session"] for e in examples})
    _, val_sessions = session_split(all_sessions, args.val_frac, pa["seed"])

    # accumulators
    model_by_cat: dict[str, list[float]] = defaultdict(list)
    ls_by_cat: dict[str, list[float]] = defaultdict(list)
    # per-band engineer balance accumulated weighted by energy
    bal_num = np.zeros(pa["n_bins"]); bal_den = np.zeros(pa["n_bins"])
    n_sess = 0

    for sess in sorted(val_sessions):
        sdir = Path(args.staging_dir) / sess
        data = load_session_audio(sdir, args.sample_rate)
        if data is None:
            continue
        stems_full = data["stems"]; mix_full = data["mix"]
        T = stems_full.shape[-1]
        if T < args.audio_len + 1:
            continue
        start = int(min(args.window_start_frac, 0.9) * (T - args.audio_len))
        sl = slice(start, start + args.audio_len)
        stems_w = stems_full[:, :, sl]; mix_w = mix_full[:, sl]
        n_real = min(stems_w.shape[0], args.n_max)
        stems_w = stems_w[:n_real]
        cats = per_track_groups(sdir, n_real)

        # pad to n_max for the model
        pad = torch.zeros(args.n_max - n_real, 2, args.audio_len)
        stems_p = torch.cat([stems_w, pad], dim=0) if n_real < args.n_max else stems_w
        mask = torch.zeros(args.n_max, dtype=torch.bool); mask[:n_real] = True
        is_stereo = detect_stereo(stems_p) & mask

        with torch.no_grad():
            stems_b = stems_p.unsqueeze(0).to(device)
            mask_b = mask.unsqueeze(0).to(device)
            stereo_b = is_stereo.unsqueeze(0).to(device)
            C = console.compute_C(stems_b, track_mask=mask_b)
            gidx = None if not pa.get("use_group_embed", False) else torch.zeros(1, args.n_max, dtype=torch.long, device=device)
            pan = pan_net(C, stereo_b, gidx, mask_b).squeeze(0).cpu()  # (n_max,)
            pan = pan * (~is_stereo).float()                          # stereo bypass

            # LS-recovered engineer pan + engineer per-band balance
            sw = stems_w.to(device)
            mono = sw.mean(dim=1)                                     # (n_real, T)
            stem_mag = mel(mono).mean(-1)                             # (n_real, n_bins)
            stem_L = mel(sw[:, 0]).mean(-1); stem_R = mel(sw[:, 1]).mean(-1)
            mw = mix_w.to(device)
            mix_L = mel(mw[0]).mean(-1); mix_R = mel(mw[1]).mean(-1)  # (n_bins,)
            ls = recover_pan_targets(stem_mag, stem_L, stem_R, mix_L, mix_R,
                                     is_stereo[:n_real].to(device)).cpu()
            bal = ((mix_L - mix_R) / (mix_L + mix_R + 1e-4)).cpu().numpy()
            en = (mix_L + mix_R).cpu().numpy()

        bal_num += bal * en; bal_den += en
        if band_centers is None:
            edges = console.eq.hz_edges.cpu().numpy()
            band_centers = ((edges[:-1] + edges[1:]) / 2).astype(int)

        for i in range(n_real):
            if bool(is_stereo[i]):
                continue
            model_by_cat[cats[i]].append(float(pan[i]))
            ls_by_cat[cats[i]].append(float(ls[i]))
        n_sess += 1

    # ---- Report ----
    def summarize(vals):
        a = np.array(vals)
        L = (a <= -0.5).mean() * 100; R = (a >= 0.5).mean() * 100; Cc = 100 - L - R
        hard = (np.abs(a) > 0.8).mean() * 100
        return len(a), a.mean(), np.median(a), np.abs(a).mean(), L, Cc, R, hard

    print(f"\n=== {n_sess} val sessions | sign: pan_dir<0 = LEFT, >0 = RIGHT ===\n")
    print(f"{'category':<12}{'src':<7}{'n':>5}{'mean':>8}{'median':>8}{'mean|p|':>9}"
          f"{'%L':>6}{'%C':>6}{'%R':>6}{'%hard':>7}")
    order = ["LeadVox", "BackingVox", "Guitar", "Keys", "Synth", "Strings",
             "Brass", "Bass", "Drums", "Other", "Unknown"]
    for cat in order:
        for src, d in (("model", model_by_cat), ("LS-eng", ls_by_cat)):
            if cat not in d or not d[cat]:
                continue
            n, mean, med, absm, L, Cc, R, hard = summarize(d[cat])
            print(f"{cat:<12}{src:<7}{n:>5}{mean:>+8.3f}{med:>+8.3f}{absm:>9.3f}"
                  f"{L:>6.0f}{Cc:>6.0f}{R:>6.0f}{hard:>7.0f}")
        print()

    allm = [v for vs in model_by_cat.values() for v in vs]
    alll = [v for vs in ls_by_cat.values() for v in vs]
    for src, vals in (("model", allm), ("LS-eng", alll)):
        n, mean, med, absm, L, Cc, R, hard = summarize(vals)
        print(f"{'ALL':<12}{src:<7}{n:>5}{mean:>+8.3f}{med:>+8.3f}{absm:>9.3f}"
              f"{L:>6.0f}{Cc:>6.0f}{R:>6.0f}{hard:>7.0f}")

    # Engineer per-band balance (energy-weighted) — where does the mix lean?
    print("\n=== engineer mix per-band balance (energy-weighted; + = LEFT) ===")
    bal_avg = bal_num / np.clip(bal_den, 1e-9, None)
    for i in range(0, pa["n_bins"], 1):
        bar = "L" * int(max(0, bal_avg[i]) * 40) or ("R" * int(max(0, -bal_avg[i]) * 40))
        print(f"  {band_centers[i]:>6} Hz  {bal_avg[i]:+.3f}  {bar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
