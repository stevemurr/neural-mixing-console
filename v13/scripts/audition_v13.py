"""Audition the v13 contribution-mixer end-to-end on held-out val sessions.

For each chosen session:
  1. Re-load stems + engineer mix at full length from staging.
  2. Slice a ~30 s window (default starts 25 % into the song).
  3. Run the full pipeline:
        compute_C(stems) → Stage 2 (target_delta_db)
        → ContributionConsole.distribute → per-track EQ → sum → mix
  4. Save three WAVs side by side at the same peak headroom:
        engineer.wav    — reference
        dry_sum.wav     — baseline (just summing the stems, no EQ)
        predicted.wav   — what the v13 model produced
  5. Dump the predicted per-bin dB delta and a spectrum comparison plot.

Each output is loudness-matched to the engineer mix's RMS, then all
three are peak-normalized together so the WAVs are directly A/B-able
without level masking your impression.

Usage:
    uv run python v13/scripts/audition_v13.py \
        --ckpt dmc-data/checkpoints/v13-target-curve/target_curve_best.pt \
        --n 3                                # 3 random val sessions

    # or pick specific sessions:
    uv run python v13/scripts/audition_v13.py --sessions AMContra_HeartPeripheral
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import yaml

from models.contribution_console import ContributionConsole
from models.group_classification import UNKNOWN_IDX, group_name_to_idx
from models.mono_pan import detect_stereo
from models.pan_assignment import balanced_assignment
from models.pan_net import PanNet
from models.target_curve_net import TargetCurveNet
from scripts.precompute_v13_dataset import load_session_audio
from training.data_v13 import session_split


def normalize_for_compare(
    engineer: torch.Tensor, dry_sum: torch.Tensor, predicted: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMS-match dry_sum and predicted to engineer; then peak-normalize all
    three together so the loudest sample across the three sits at 0.9.

    This keeps relative levels honest (you hear the model's level decisions
    relative to the engineer's) while avoiding clipping."""
    eng_rms = (engineer ** 2).mean().clamp(min=eps).sqrt()
    dry_rms = (dry_sum ** 2).mean().clamp(min=eps).sqrt()
    pred_rms = (predicted ** 2).mean().clamp(min=eps).sqrt()
    dry_n = dry_sum * (eng_rms / dry_rms)
    pred_n = predicted * (eng_rms / pred_rms)

    peak = max(
        engineer.abs().max().item(),
        dry_n.abs().max().item(),
        pred_n.abs().max().item(),
        1e-6,
    )
    scale = 0.9 / peak
    return engineer * scale, dry_n * scale, pred_n * scale


def render_session(
    session_dir: Path,
    sample_rate: int,
    audio_len: int,
    n_max: int,
    window_start_frac: float,
    console: ContributionConsole,
    net: TargetCurveNet,
    pan_net: PanNet | None,
    device: torch.device,
    lambda_balance: float = 0.5,
    use_balance: bool = True,
) -> dict | None:
    """Run end-to-end inference on one session. Returns dict of tensors + meta."""
    data = load_session_audio(session_dir, sample_rate)
    if data is None:
        logging.warning(f"{session_dir.name}: failed to load")
        return None
    stems_full = data["stems"]                          # (N, 2, T_total)
    mix_full = data["mix"]                              # (2, T_total)
    T_total = stems_full.shape[-1]
    if T_total < audio_len + 1:
        logging.warning(f"{session_dir.name}: too short ({T_total} samples)")
        return None

    start = int(min(window_start_frac, 0.9) * (T_total - audio_len))
    end = start + audio_len
    stems_w = stems_full[:, :, start:end]               # (N_real, 2, audio_len)
    mix_w = mix_full[:, start:end]                      # (2, audio_len)

    N_real = stems_w.shape[0]
    if N_real > n_max:
        stems_w = stems_w[:n_max]
        N_real = n_max
    if N_real < n_max:
        pad = torch.zeros(n_max - N_real, 2, audio_len)
        stems_padded = torch.cat([stems_w, pad], dim=0)
    else:
        stems_padded = stems_w
    track_mask = torch.zeros(n_max, dtype=torch.bool)
    track_mask[:N_real] = True

    # Stereo detection runs on the actual window audio (not just metadata)
    is_stereo = detect_stereo(stems_padded)             # (n_max,) bool
    # padded slots must stay non-stereo (they're zero)
    is_stereo = is_stereo & track_mask

    # Group index per track — read from correspondence.yaml
    group_idx = torch.full((n_max,), UNKNOWN_IDX, dtype=torch.long)
    try:
        corr = yaml.safe_load(open(session_dir / "correspondence.yaml"))
        if isinstance(corr, dict):
            # Walk groups in the same order as load_session_audio flattens stems
            flat_idx = 0
            for group_name, files in corr.items():
                if not isinstance(files, list):
                    continue
                g = group_name_to_idx(group_name)
                for _ in files:
                    if flat_idx < N_real:
                        group_idx[flat_idx] = g
                    flat_idx += 1
    except Exception as e:
        logging.warning(f"{session_dir.name}: could not parse groups ({e})")

    with torch.no_grad():
        stems_b = stems_padded.unsqueeze(0).to(device)
        mask_b = track_mask.unsqueeze(0).to(device)
        stereo_b = is_stereo.unsqueeze(0).to(device)
        group_b = group_idx.unsqueeze(0).to(device)
        C = console.compute_C(stems_b, track_mask=mask_b)
        target_delta_db = net(C, mask_b)               # (1, n_bins)

        if pan_net is not None:
            if pan_net.continuous:
                # Continuous head emits per-track pan_dir ∈ (-1,+1) directly.
                pan_dir = pan_net(C, stereo_b, group_b, mask_b)  # (1, n_max)
            else:
                # Class head outputs logits over {L, C, R}. Default: soft
                # per-track prediction (softmax-weighted mean). Balance-aware
                # assignment is kept as an opt-in fallback; it produced
                # engineering-incorrect decisions for layered drum kits (which
                # should always stay center), so the per-track model with group
                # conditioning is the primary path for the class head.
                logits = pan_net(C, stereo_b, group_b, mask_b)   # (1, n_max, 3)
                if use_balance:
                    pan_dir = balanced_assignment(
                        logits.squeeze(0),                       # (n_max, 3)
                        C.squeeze(0),                            # (n_max, n_bins)
                        stereo_b.squeeze(0),                     # (n_max,)
                        track_mask=mask_b.squeeze(0),
                        lambda_balance=lambda_balance,
                    ).unsqueeze(0)                               # (1, n_max)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    pan_dir = (probs * pan_net.class_to_pan).sum(dim=-1)
            # Force pan_dir=0 on stereo tracks (they bypass pan anyway).
            pan_dir = pan_dir * (~stereo_b).to(pan_dir.dtype)
            predicted_mix, aux = console(
                stems_b, target_delta_db,
                pan_dir=pan_dir, is_stereo=stereo_b,
                track_mask=mask_b,
            )
            # Also compute EQ-only for A/B comparison
            predicted_eq_only, _ = console(
                stems_b, target_delta_db, track_mask=mask_b,
            )
            predicted_eq_only = predicted_eq_only.squeeze(0).cpu()
            pan_dir_cpu = pan_dir.squeeze(0).cpu()
        else:
            predicted_mix, aux = console(
                stems_b, target_delta_db, track_mask=mask_b,
            )
            predicted_eq_only = None
            pan_dir_cpu = None

        predicted_mix = predicted_mix.squeeze(0).cpu()
        C = C.squeeze(0).cpu()
        target_delta_db = target_delta_db.squeeze(0).cpu()
        per_track_delta_db = aux["per_track_delta_db"].squeeze(0).cpu()
        gain_norm = aux["gain_norm"].squeeze(0).cpu()
    dry_sum = stems_padded.sum(dim=0)                  # (2, audio_len)

    return {
        "session":             session_dir.name,
        "window":              (start, end),
        "N_real":              N_real,
        "engineer":            mix_w,
        "dry_sum":             dry_sum,
        "predicted":           predicted_mix,
        "predicted_eq_only":   predicted_eq_only,
        "C":                   C,
        "target_delta_db":     target_delta_db,
        "per_track_delta_db":  per_track_delta_db,
        "gain_norm":           gain_norm,
        "is_stereo":           is_stereo.cpu(),
        "pan_dir":             pan_dir_cpu,
        "stem_filenames":      data.get("stem_filenames", []),
    }


def measured_target_delta_db(
    engineer: torch.Tensor, dry_sum: torch.Tensor,
    console: ContributionConsole, eps: float = 1e-6,
) -> torch.Tensor:
    """Recompute the actual (loudness-matched) target dB delta for this
    window — what Stage 2 *should* have predicted."""
    dev = console.mel.mel_scale.fb.device
    engineer = engineer.to(dev)
    dry_sum = dry_sum.to(dev)
    dry_rms = (dry_sum ** 2).mean().clamp(min=eps).sqrt()
    eng_rms = (engineer ** 2).mean().clamp(min=eps).sqrt()
    eng_norm = engineer * (dry_rms / eng_rms)
    with torch.no_grad():
        eng_C = console.compute_C(eng_norm.unsqueeze(0).unsqueeze(0))
        dry_C = console.compute_C(dry_sum.unsqueeze(0).unsqueeze(0))
    eng_mag = torch.expm1(eng_C).clamp(min=eps).squeeze()
    dry_mag = torch.expm1(dry_C).clamp(min=eps).squeeze()
    return (20.0 * torch.log10(eng_mag / dry_mag)).cpu()


def plot_diagnostics(
    out_path: Path,
    result: dict,
    measured_delta: torch.Tensor,
    edges_hz: np.ndarray,
    title: str,
) -> None:
    """Two-panel plot:
       - Top: predicted vs measured target dB delta, per mel band
       - Bottom: log-mel spectra of dry_sum, predicted, engineer (mid-channel)"""
    n_bins = result["target_delta_db"].shape[0]
    band_centers = (edges_hz[:-1] + edges_hz[1:]) / 2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
    ax1.bar(
        np.arange(n_bins) - 0.18,
        result["target_delta_db"].numpy(),
        width=0.36, label="predicted (Stage 2)", color="C0",
    )
    ax1.bar(
        np.arange(n_bins) + 0.18,
        measured_delta.numpy(),
        width=0.36, label="measured (engineer - dry, RMS-matched)", color="C1",
    )
    ax1.axhline(0, color="0.5", lw=0.5)
    ax1.set_xticks(np.arange(n_bins))
    ax1.set_xticklabels(
        [f"{int(c)}" if c >= 1000 else f"{int(c)}" for c in band_centers],
        rotation=45, fontsize=7,
    )
    ax1.set_xlabel("mel band center (Hz)")
    ax1.set_ylabel("dB delta")
    ax1.set_title(f"{title} — per-bin dB delta")
    ax1.legend(loc="best")
    ax1.grid(alpha=0.3)

    # Mid-channel log-mel spectra (run on whichever device the console lives on)
    mel_device = console.mel.mel_scale.fb.device
    def log_mel(x: torch.Tensor) -> np.ndarray:
        m = console.mel((x[0] + x[1]).unsqueeze(0).to(mel_device))
        return torch.log1p(m.mean(dim=-1)).squeeze(0).cpu().numpy()

    ax2.plot(np.arange(n_bins), log_mel(result["dry_sum"]),
             label="dry_sum (raw)", color="0.5", lw=1.5)
    ax2.plot(np.arange(n_bins), log_mel(result["predicted"]),
             label="predicted (v13)", color="C0", lw=1.5)
    ax2.plot(np.arange(n_bins), log_mel(result["engineer"]),
             label="engineer", color="C1", lw=1.5)
    ax2.set_xticks(np.arange(n_bins))
    ax2.set_xticklabels(
        [f"{int(c)}" if c >= 1000 else f"{int(c)}" for c in band_centers],
        rotation=45, fontsize=7,
    )
    ax2.set_xlabel("mel band center (Hz)")
    ax2.set_ylabel("log1p mean mel mag")
    ax2.set_title("mid-channel mel spectra (time-averaged)")
    ax2.legend(loc="best")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# Captured at module scope for plot_diagnostics — set inside main.
console: ContributionConsole | None = None


def main() -> int:
    global console

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt",
                    default="dmc-data/checkpoints/v13-target-curve/target_curve_best.pt")
    ap.add_argument("--pan-ckpt",
                    default="dmc-data/checkpoints/v13-pan/pan_best.pt",
                    help="Stage 3 (pan) checkpoint. Set empty to disable.")
    ap.add_argument("--lambda-balance", type=float, default=0.5,
                    help="Weight on the L-vs-R balance penalty in the greedy "
                         "assignment. Only used when --use-balance is passed.")
    ap.add_argument("--use-balance", action="store_true",
                    help="Use the greedy balance-aware assignment instead of "
                         "the per-track soft prediction. Off by default — it "
                         "over-pans layered drum kit elements and other "
                         "engineering-should-be-center tracks.")
    ap.add_argument("--staging-dir", default="dmc-data/grafx-prune-data-48k")
    ap.add_argument("--precomputed",
                    default="dmc-data/v13_precomputed.pt",
                    help="Used only to recover the train/val split")
    ap.add_argument("--out-dir", default="dmc-data/audition/v13-target-curve")
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="Specific session names (otherwise --n random val sessions)")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--sample-rate", type=int, default=48_000)
    ap.add_argument("--audio-len", type=int, default=30 * 48_000,
                    help="Samples to render per session (default 30 s @ 48 kHz)")
    ap.add_argument("--window-start-frac", type=float, default=0.25,
                    help="Where to start the window in the song (fraction)")
    ap.add_argument("--n-max", type=int, default=64,
                    help="Must match the precomputed dataset's n_max")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # ---- Load checkpoint ----
    if not Path(args.ckpt).is_file():
        ap.error(f"checkpoint not found: {args.ckpt}")
    ck = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    ck_args = ck["args"]
    logging.info(
        f"ckpt step={ck['step']}  best_val_huber={ck.get('best_val', '?'):.4f}"
    )

    net = TargetCurveNet(
        n_bins=ck_args["n_bins"], d_model=ck_args["d_model"],
        n_layers=ck_args["n_layers"], n_heads=ck_args["n_heads"],
        ffn_mult=ck_args["ffn_mult"], dropout=ck_args["dropout"],
        max_delta_db=ck_args["max_delta_db"],
    ).to(device).eval()
    net.load_state_dict(ck["model"])

    console = ContributionConsole(
        sample_rate=args.sample_rate, n_bins=ck_args["n_bins"],
    ).to(device).eval()

    # ---- Optionally load Stage 3 (pan) ----
    pan_net = None
    if args.pan_ckpt and Path(args.pan_ckpt).is_file():
        pck = torch.load(args.pan_ckpt, weights_only=False, map_location="cpu")
        pck_args = pck["args"]
        pan_net = PanNet(
            n_bins=pck_args["n_bins"], d_model=pck_args["d_model"],
            n_layers=pck_args["n_layers"], n_heads=pck_args["n_heads"],
            ffn_mult=pck_args["ffn_mult"], dropout=pck_args["dropout"],
            use_group_embed=pck_args.get("use_group_embed", True),
            continuous=pck_args.get("continuous", False),
        ).to(device).eval()
        pan_net.load_state_dict(pck["model"])
        bv = pck.get("best_val")
        bv_str = f"{bv:.4f}" if isinstance(bv, (int, float)) else "?"
        logging.info(f"pan ckpt step={pck['step']}  best_val_mse={bv_str}")
    else:
        logging.info("no pan ckpt — EQ-only audition")

    # ---- Pick sessions ----
    examples = torch.load(args.precomputed, weights_only=False, map_location="cpu")
    all_sessions = sorted({e["session"] for e in examples})
    _, val_sessions = session_split(
        all_sessions, args.val_frac, ck_args["seed"],
    )
    if args.sessions:
        chosen = list(args.sessions)
        for s in chosen:
            if s not in val_sessions:
                logging.warning(f"{s} NOT in val set — auditioning it anyway "
                                f"(may have been a training session)")
    else:
        rng = random.Random(args.seed)
        chosen = rng.sample(sorted(val_sessions), min(args.n, len(val_sessions)))
    logging.info(f"auditioning {len(chosen)} sessions: {chosen}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edges_hz = console.eq.hz_edges.cpu().numpy()
    summary_lines: list[str] = []

    for sess_name in chosen:
        session_dir = Path(args.staging_dir) / sess_name
        result = render_session(
            session_dir, args.sample_rate, args.audio_len, args.n_max,
            args.window_start_frac, console, net, pan_net, device,
            lambda_balance=args.lambda_balance,
            use_balance=args.use_balance,
        )
        if result is None:
            continue

        # Loudness-match + peak-normalize for fair comparison
        eng, dry, pred = normalize_for_compare(
            result["engineer"], result["dry_sum"], result["predicted"],
        )

        sess_out = out_dir / sess_name
        sess_out.mkdir(parents=True, exist_ok=True)
        sr = args.sample_rate
        sf.write(sess_out / "engineer.wav", eng.numpy().T, sr)
        sf.write(sess_out / "dry_sum.wav", dry.numpy().T, sr)
        sf.write(sess_out / "predicted.wav", pred.numpy().T, sr)

        # When pan is composed, also save EQ-only for A/B
        if result.get("predicted_eq_only") is not None:
            # Re-normalize EQ-only to engineer's RMS + same peak headroom
            eo = result["predicted_eq_only"]
            eo_rms = (eo ** 2).mean().clamp(min=1e-8).sqrt()
            eng_rms = (result["engineer"] ** 2).mean().clamp(min=1e-8).sqrt()
            eo_n = eo * (eng_rms / eo_rms)
            peak = max(
                result["engineer"].abs().max().item(),
                eo_n.abs().max().item(), 1e-6,
            )
            eo_n = eo_n * (0.9 / peak)
            sf.write(sess_out / "predicted_eq_only.wav", eo_n.numpy().T, sr)

        # --- Verification: raw sum (no normalization) ----------------------
        # Literally stems.sum(dim=0), peak-normalized only (to avoid clipping
        # if the raw sum exceeds [-1, 1]). Listening to this confirms dry_sum
        # is just the additive sum with no processing.
        raw_sum = result["dry_sum"]
        raw_peak = raw_sum.abs().max().clamp(min=1e-8)
        raw_norm = raw_sum / raw_peak.item() * 0.9
        sf.write(sess_out / "raw_sum.wav", raw_norm.numpy().T, sr)

        # Quick spectral sanity check: raw_sum and (normalized) dry_sum
        # should have identical spectral SHAPE (they differ only by a
        # scalar gain). Their mel-spectrum ratio should be constant
        # across bins (= the gain difference). RMSE between log-mel
        # spectra after demeaning should be near-zero.
        with torch.no_grad():
            mel_dev = console.mel.mel_scale.fb.device
            m_raw = console.mel((raw_sum[0] + raw_sum[1]).unsqueeze(0).to(mel_dev))
            m_dry = console.mel((dry[0] + dry[1]).unsqueeze(0).to(mel_dev))
            lm_raw = torch.log1p(m_raw.mean(dim=-1)).squeeze(0).cpu()
            lm_dry = torch.log1p(m_dry.mean(dim=-1)).squeeze(0).cpu()
            shape_err = (
                (lm_raw - lm_raw.mean()) - (lm_dry - lm_dry.mean())
            ).pow(2).mean().sqrt().item()

        # --- Per-track EQ moves ------------------------------------------
        # Write CSV showing what dB delta each real track received per
        # mel band — so we can audit the model's distribution choices.
        per_track = result["per_track_delta_db"][: result["N_real"]].numpy()  # (N_real, 26)
        stem_names = result["stem_filenames"][: result["N_real"]]
        if len(stem_names) < result["N_real"]:
            stem_names += [f"track_{i:02d}" for i in
                           range(len(stem_names), result["N_real"])]
        csv_path = sess_out / "per_track_eq.csv"
        with open(csv_path, "w") as f:
            band_centers = ((edges_hz[:-1] + edges_hz[1:]) / 2).astype(int)
            f.write("stem," + ",".join(f"{c}Hz" for c in band_centers) + "\n")
            for name, deltas in zip(stem_names, per_track):
                f.write(name + "," + ",".join(f"{d:+.2f}" for d in deltas) + "\n")

        # ---- Per-track pan (if Stage 3 is active) -----------------------
        if result.get("pan_dir") is not None:
            pan_arr = result["pan_dir"][: result["N_real"]].numpy()
            is_stereo_arr = result["is_stereo"][: result["N_real"]].numpy()
            csv_pan = sess_out / "per_track_pan.csv"
            with open(csv_pan, "w") as f:
                f.write("stem,is_stereo,pan_dir,pan_mode\n")
                for name, pd, st in zip(stem_names, pan_arr, is_stereo_arr):
                    mode = "L" if pd <= -0.5 else ("R" if pd >= 0.5 else "C")
                    if st:
                        mode = "STEREO"
                    f.write(f"{name},{int(bool(st))},{pd:+.3f},{mode}\n")

        # ---- Move metrics: how aggressive is the model? -----------------
        abs_per_track = np.abs(per_track)
        move_metrics = {
            "max_abs_dB":        float(abs_per_track.max()),
            "mean_abs_dB":       float(abs_per_track.mean()),
            "p95_abs_dB":        float(np.percentile(abs_per_track, 95)),
            # Per-track summary: max abs move that track saw across any band
            "per_track_max_abs": abs_per_track.max(axis=1),  # (N_real,)
            # How many tracks got at least one >2 dB move
            "n_active_tracks":   int((abs_per_track.max(axis=1) > 2.0).sum()),
        }
        # Find the 8 single biggest moves (track, band, dB) for the report
        flat = abs_per_track.flatten()
        top_idx = np.argsort(flat)[-8:][::-1]
        top_moves = []
        for idx in top_idx:
            t, b = divmod(idx, abs_per_track.shape[1])
            top_moves.append((stem_names[t][:30], int(band_centers[b]),
                              float(per_track[t, b])))

        # Distribution sanity: does Σ_t (c_t · f_t) actually hit target?
        # (Verifies LS math composes correctly under gain-invariant EQ.)
        contrib_lin = np.expm1(result["C"][: result["N_real"]].numpy())
        gain_lin = 10 ** (per_track / 20)
        achieved = contrib_lin * gain_lin                                 # (N_real, n_bins)
        dry_per_bin = contrib_lin.sum(axis=0).clip(min=1e-8)               # (n_bins,)
        achieved_per_bin = achieved.sum(axis=0).clip(min=1e-8)             # (n_bins,)
        achieved_delta_db = 20 * np.log10(achieved_per_bin / dry_per_bin)  # (n_bins,)
        # vs target
        target_db_np = result["target_delta_db"].numpy()
        composition_err = float(np.sqrt(((achieved_delta_db - target_db_np) ** 2).mean()))

        # Per-track EQ visualization: each track's curve, color-coded by
        # the loudest band of its contribution (helps spot "this track
        # got a 10 dB cut at 1 kHz" outliers).
        fig, ax = plt.subplots(figsize=(11, 5))
        n_show = min(result["N_real"], 32)   # cap visual clutter
        cmap = plt.get_cmap("viridis")
        for i in range(n_show):
            label = stem_names[i][:24]
            ax.plot(np.arange(per_track.shape[1]),
                    per_track[i],
                    color=cmap(i / max(n_show - 1, 1)),
                    alpha=0.7, lw=1.2, label=label)
        ax.axhline(0, color="0.4", lw=0.5)
        ax.set_xticks(np.arange(per_track.shape[1]))
        ax.set_xticklabels(
            [f"{int(c)}" for c in band_centers], rotation=45, fontsize=7,
        )
        ax.set_xlabel("mel band center (Hz)")
        ax.set_ylabel("predicted per-track EQ delta (dB)")
        ax.set_title(f"{sess_name} — per-track EQ moves (top {n_show} tracks)")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
                  fontsize=7, ncol=1)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(sess_out / "per_track_eq.png", dpi=120,
                    bbox_inches="tight")
        plt.close()

        # Measured target (what Stage 2 should have predicted)
        measured_delta = measured_target_delta_db(
            result["engineer"], result["dry_sum"], console,
        )
        rmse = ((result["target_delta_db"] - measured_delta) ** 2).mean().sqrt().item()

        # Per-bin info dump
        info_path = sess_out / "info.txt"
        with open(info_path, "w") as f:
            f.write(f"session:           {sess_name}\n")
            f.write(f"window (samples):  {result['window'][0]}..{result['window'][1]}\n")
            f.write(f"window (seconds):  {result['window'][0]/sr:.1f}..{result['window'][1]/sr:.1f}\n")
            f.write(f"N_real_tracks:     {result['N_real']}\n")
            f.write(f"per-window RMSE:   {rmse:.2f} dB\n")
            f.write(f"\nFiles:\n")
            f.write(f"  engineer.wav      reference engineer mix\n")
            f.write(f"  dry_sum.wav       sum of stems, RMS-matched to engineer + peak-normalized\n")
            f.write(f"  raw_sum.wav       sum of stems, peak-normalized ONLY (verifies dry_sum has no processing)\n")
            f.write(f"  predicted.wav     v13 model output, RMS-matched to engineer + peak-normalized\n")
            f.write(f"  per_track_eq.csv  per-track dB delta the distribution policy chose\n")
            f.write(f"  per_track_eq.png  visualization of the per-track EQ moves\n")
            f.write(f"  diagnostic.png    predicted vs measured curve + mel spectra\n")
            if result.get("pan_dir") is not None:
                f.write(f"  predicted_eq_only.wav  EQ-only render (for A/B vs full composition)\n")
                f.write(f"  per_track_pan.csv  per-track pan_dir and stereo flag\n")
            f.write(f"\nVerification:\n")
            f.write(f"  raw_sum vs dry_sum mel-shape RMSE: {shape_err:.2e}  (near zero confirms only level differs)\n")
            f.write(f"\nMove aggressiveness:\n")
            f.write(f"  max  |per-track move|   = {move_metrics['max_abs_dB']:6.2f} dB\n")
            f.write(f"  mean |per-track move|   = {move_metrics['mean_abs_dB']:6.2f} dB\n")
            f.write(f"  95th pct |per-track|    = {move_metrics['p95_abs_dB']:6.2f} dB\n")
            f.write(f"  tracks with >2 dB move  = {move_metrics['n_active_tracks']} / {result['N_real']}\n")
            f.write(f"\nDistribution composition check (sum of EQ'd stems vs target):\n")
            f.write(f"  RMSE between achieved and predicted mix-bus delta: {composition_err:.2f} dB\n")
            f.write(f"  (low value confirms LS distribution actually reaches the predicted target)\n")
            f.write(f"\nTop 8 single biggest per-track moves:\n")
            for stem, freq, db in top_moves:
                f.write(f"  {stem:<32s}  {freq:>6d} Hz   {db:+6.2f} dB\n")
            f.write(f"\n{'bin':>3}  {'freq range':>17}   {'predicted':>10}   {'measured':>10}   {'err':>7}\n")
            for i in range(len(result["target_delta_db"])):
                f.write(
                    f"{i:>3}  "
                    f"[{int(edges_hz[i]):>5} .. {int(edges_hz[i+1]):>5} Hz]   "
                    f"{result['target_delta_db'][i].item():+10.2f}   "
                    f"{measured_delta[i].item():+10.2f}   "
                    f"{result['target_delta_db'][i].item() - measured_delta[i].item():+7.2f}\n"
                )

        # Diagnostic plot
        plot_diagnostics(
            sess_out / "diagnostic.png", result, measured_delta, edges_hz,
            title=sess_name,
        )

        logging.info(
            f"  [{sess_name}]  window RMSE={rmse:5.2f} dB  → {sess_out}"
        )
        summary_lines.append(
            f"{sess_name:<40s}  RMSE={rmse:5.2f} dB  N={result['N_real']}"
        )

    # Top-level summary
    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"v13 audition — {len(chosen)} sessions\n")
        f.write(f"ckpt: {args.ckpt}\n\n")
        for line in summary_lines:
            f.write(line + "\n")
    logging.info(f"summary: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
