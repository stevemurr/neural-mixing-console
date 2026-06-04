"""Train MinimalConsole encoder. Two modes:

    --mode synth   : Phase 2 (synthetic supervision).
        Sample random console params, render synth_mix on-the-fly, train
        encoder against (L_param + L_recon vs synth_mix + L_cons).
        Both supervision signals are clean by construction — the params
        caused the synth_mix, no gauge degeneracy.

    --mode finetune: Phase 3 (real engineer mix).
        Use real Cambridge engineer mixes as target. Train against
        L_recon vs engineer mix + L_cons only. No teacher labels.
        Warm-start from a Phase 2 ckpt via --init-from.

Both modes share: MinimalConsole DSP, MixEncoderGrafx schema-driven heads
(auto-adapts to MinimalConsole's 25-scalar-per-strip schema), bf16
encoder, cosine LR + warmup, HybridReconLoss for L_recon, 2-window val
averaging.

See `notes/minimal_console_design_2026-05.md` for design rationale.

Run:
    uv run python training/train_minimal.py --mode synth \\
        --staging-dir dmc-data/grafx-prune-data \\
        --max-steps 8000 --batch-size 4 --grad-accum 2
    uv run python training/train_minimal.py --mode finetune \\
        --init-from dmc-data/checkpoints/minimal-v12-synth/encoder_best.pt \\
        --max-steps 6000 ...
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.minimal_console import MinimalConsole
from models.minimal_param_ranges import denormalize_params_dict
from models.encoders_grafx import MixEncoderGrafx
from training.data_grafx import LabelStore, attach_grafx_labels
from training.losses_grafx import (
    HybridReconLoss, grafx_param_huber_loss, grafx_param_consistency_loss,
    make_grafx_prune_recon_loss,
)
from training.synth_data import ParamSampler, SyntheticDataset, collate_synth
from training.train_grafx_distill import (
    PairedMultiSongDataset, collate_pairs, _split_params, _to_device_params,
    _encoder_forward, w_cosine_ramp_at,
)
from training.train_grafx_multi import (
    _list_staged_sessions, _load_session, _example_from_session, collate_multi,
)


# ---------- Helpers ----------

def _params_to_device(
    params: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        proc: {p: t.to(device, non_blocking=True) for p, t in pp.items()}
        for proc, pp in params.items()
    }


def _sigmoid_params(
    raw: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Apply sigmoid to encoder logits → [0,1] per (proc, param).

    Keeps gradients well-behaved (sigmoid is differentiable everywhere
    and produces bounded outputs). The encoder's `ProcessorParamHead`
    outputs raw Linear values; this is the boundary where we constrain
    them to the normalized range MinimalConsole's denormalizer expects.
    """
    return {
        proc: {p: torch.sigmoid(t) for p, t in pp.items()}
        for proc, pp in raw.items()
    }


# ---------- Synthetic-mode val ----------

@torch.no_grad()
def run_val_synth(
    encoder, console, sampler, recon_loss_fn,
    val_sessions: list[str], staging_dir: Path, label_store: LabelStore,
    audio_len: int, n_max: int, max_groups: int,
    device: torch.device, use_bf16: bool,
) -> dict[str, float]:
    """Val in synth mode: render two windows per session with sampled
    params (synth_mix = target), encode student, score L_param + L_recon.
    """
    encoder.eval()
    L_p, L_r = [], []
    for session in val_sessions:
        try:
            data = _load_session(session, staging_dir, label_store)
        except Exception as e:
            logging.warning(f"val: failed to load {session}: {e}")
            continue
        labels = label_store.get(session)
        if labels is None: continue
        T_total = data["T_total"]
        if T_total < audio_len + 1: continue
        n_groups = len(labels["groups"])

        # Two fixed windows per session for variance reduction
        starts = [max(0, T_total // 4),
                  max(0, min(T_total - audio_len - 1, (3 * T_total) // 4))]
        l_p_halves, l_r_halves = [], []
        for start in starts:
            ex = _example_from_session(data, start, audio_len, n_max)
            tracks = ex["tracks"].unsqueeze(0).to(device)
            track_mask = ex["track_mask"].unsqueeze(0).to(device)

            # Group assignments from labels (engineer routing, params unused)
            stem_filenames = data["stem_filenames"][:int(ex["track_mask"].sum().item())]
            fname_to_group = {fn: int(g) for fn, g in zip(
                labels["stem_filenames"], labels["group_assignments"].tolist())}
            group_assignments = torch.zeros(1, n_max, dtype=torch.long, device=device)
            for ti, fn in enumerate(stem_filenames):
                group_assignments[0, ti] = fname_to_group.get(fn, 0)

            # Sample synthetic params for this val window (in [0,1] space)
            strip_norm = _params_to_device(sampler.sample("strip", 1, n_max), device)
            group_norm = _params_to_device(sampler.sample("group", 1, max_groups), device)
            strip_eng = denormalize_params_dict(strip_norm)
            group_eng = denormalize_params_dict(group_norm)
            synth_mix = console(tracks, strip_eng, group_eng, group_assignments,
                                 track_mask=track_mask, n_groups=max_groups)

            # Encoder prediction → sigmoid → [0,1] → denormalize
            out = _encoder_forward(encoder, tracks, track_mask, group_assignments,
                                   n_groups=max_groups, use_bf16=use_bf16)
            pred_strip_norm = _sigmoid_params(out["strip_params"])
            pred_group_norm = _sigmoid_params(out["group_params"])
            pred_strip_eng = denormalize_params_dict(pred_strip_norm)
            pred_group_eng = denormalize_params_dict(pred_group_norm)

            # L_param in [0,1] space
            label_mask = torch.ones(1, n_max, dtype=torch.bool, device=device)
            ex_mask = torch.tensor([True], dtype=torch.bool, device=device)
            n_groups_t = torch.tensor([max_groups], dtype=torch.long, device=device)
            L_param = grafx_param_huber_loss(
                pred_strip_norm, pred_group_norm,
                strip_norm, group_norm, label_mask, n_groups_t, ex_mask, delta=1.0,
            )
            l_p_halves.append(L_param["L_param/total"].item())

            # L_recon: student render (engineering params) vs synth_mix
            student_mix = console(tracks, pred_strip_eng, pred_group_eng,
                                  group_assignments, track_mask=track_mask,
                                  n_groups=max_groups)
            L_rec = recon_loss_fn(student_mix, synth_mix)
            l_r_halves.append(L_rec["total"].item())

        L_p.append(sum(l_p_halves) / len(l_p_halves))
        L_r.append(sum(l_r_halves) / len(l_r_halves))

    encoder.train()
    return {
        "val/L_param":   sum(L_p) / len(L_p) if L_p else float("nan"),
        "val/L_recon":   sum(L_r) / len(L_r) if L_r else float("nan"),
        "val/n_total":   float(len(L_r)),
    }


# ---------- Finetune-mode val ----------

@torch.no_grad()
def run_val_finetune(
    encoder, console, recon_loss_fn, gap_loss_fn,
    val_sessions: list[str], staging_dir: Path, label_store: LabelStore,
    audio_len: int, n_max: int, max_groups: int,
    device: torch.device, use_bf16: bool,
) -> dict[str, float]:
    """Val in finetune mode: render two windows vs engineer mix, plus
    amort gap (MR-STFT vs teacher recorded loss)."""
    encoder.eval()
    L_r, gaps = [], []
    for session in val_sessions:
        try:
            data = _load_session(session, staging_dir, label_store)
        except Exception as e:
            logging.warning(f"val: failed to load {session}: {e}")
            continue
        labels = label_store.get(session)
        if labels is None: continue
        T_total = data["T_total"]
        if T_total < audio_len + 1: continue
        n_groups = len(labels["groups"])
        tloss = (labels.get("training_meta") or {}).get("final_test_loss")

        starts = [max(0, T_total // 4),
                  max(0, min(T_total - audio_len - 1, (3 * T_total) // 4))]
        l_r_halves, gap_halves = [], []
        for start in starts:
            ex = _example_from_session(data, start, audio_len, n_max)
            tracks = ex["tracks"].unsqueeze(0).to(device)
            track_mask = ex["track_mask"].unsqueeze(0).to(device)
            mix_ref = ex["mix"].unsqueeze(0).to(device)

            stem_filenames = data["stem_filenames"][:int(ex["track_mask"].sum().item())]
            fname_to_group = {fn: int(g) for fn, g in zip(
                labels["stem_filenames"], labels["group_assignments"].tolist())}
            group_assignments = torch.zeros(1, n_max, dtype=torch.long, device=device)
            for ti, fn in enumerate(stem_filenames):
                group_assignments[0, ti] = fname_to_group.get(fn, 0)

            out = _encoder_forward(encoder, tracks, track_mask, group_assignments,
                                   n_groups=max_groups, use_bf16=use_bf16)
            pred_strip_eng = denormalize_params_dict(_sigmoid_params(out["strip_params"]))
            pred_group_eng = denormalize_params_dict(_sigmoid_params(out["group_params"]))
            student_mix = console(tracks, pred_strip_eng, pred_group_eng,
                                  group_assignments, track_mask=track_mask,
                                  n_groups=max_groups)
            L_rec = recon_loss_fn(student_mix, mix_ref)
            l_r_halves.append(L_rec["total"].item())
            if tloss is not None:
                student_mrstft = gap_loss_fn(student_mix, mix_ref)["match/full"].item()
                gap_halves.append(student_mrstft - float(tloss))

        L_r.append(sum(l_r_halves) / len(l_r_halves))
        if gap_halves:
            gaps.append(sum(gap_halves) / len(gap_halves))

    encoder.train()
    return {
        "val/L_recon":   sum(L_r) / len(L_r) if L_r else float("nan"),
        "val/amort_gap": sum(gaps) / len(gaps) if gaps else float("nan"),
        "val/n_total":   float(len(L_r)),
    }


# ---------- Main ----------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["synth", "finetune"], required=True)
    ap.add_argument("--staging-dir", default="dmc-data/grafx-prune-data")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tb-logdir", default=None)
    ap.add_argument("--device", default=None)

    # Data
    ap.add_argument("--sample-rate", type=int, default=48_000)
    ap.add_argument("--audio-len", type=int, default=288_000)   # 6 s @ 48 kHz
    ap.add_argument("--n-max", type=int, default=24)
    ap.add_argument("--max-groups", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--min-window-gap", type=int, default=48_000)  # 1 s @ 48 kHz
    ap.add_argument("--cache-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--pairs-per-session", type=int, default=8,
                    help="finetune mode only")

    # Model
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-track-layers", type=int, default=4)
    ap.add_argument("--init-from", default=None)

    # Optim
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-warmup-steps", type=int, default=300)
    ap.add_argument("--lr-cosine-min", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=20.0)
    ap.add_argument("--grad-accum", type=int, default=2)

    # Loss
    ap.add_argument("--w-param", type=float, default=1.0,
                    help="synth mode only — labels are clean")
    ap.add_argument("--w-cons", type=float, default=0.3)
    ap.add_argument("--w-recon", type=float, default=1.0)
    # AF feature weights — defaults match the Diff-MST paper exactly.
    ap.add_argument("--w-rms", type=float, default=0.1)
    ap.add_argument("--w-crest", type=float, default=0.001)
    ap.add_argument("--w-spec", type=float, default=0.1)
    ap.add_argument("--w-stereo-width", type=float, default=1.0)
    ap.add_argument("--w-stereo-imbalance", type=float, default=1.0)
    ap.add_argument("--w-af", type=float, default=1.0,
                    help="Multiplier on the whole AF sum (per-feature "
                         "weights are Diff-MST's; this scales them together).")
    ap.add_argument("--w-cosine-ramp", type=int, default=400,
                    help="cosine ramp of audio loss weights over this many steps")

    # Logging
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-bf16", action="store_true", default=False)
    ap.add_argument("--cuda-memory-fraction", type=float, default=0.0)

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    if args.run_name is None:
        args.run_name = f"minimal_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    if args.out_dir is None:
        args.out_dir = f"dmc-data/checkpoints/{args.run_name}"
    if args.tb_logdir is None:
        args.tb_logdir = f"runs/{args.run_name}"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tb_logdir).mkdir(parents=True, exist_ok=True)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logging.info(f"mode={args.mode}  run_name={args.run_name}  device={device}")

    if args.cuda_memory_fraction > 0.0 and device.type == "cuda":
        cuda_idx = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(
            float(args.cuda_memory_fraction), device=cuda_idx,
        )
        logging.info(f"cuda memory fraction capped at {args.cuda_memory_fraction:.2f}")

    # ---- Sessions / split ----
    staging_dir = Path(args.staging_dir).expanduser()
    label_store = LabelStore(staging_dir)
    sessions = _list_staged_sessions(staging_dir)
    if not sessions:
        raise SystemExit(f"no staged sessions found under {staging_dir}")
    # filter sessions with labels (we need labels for group_assignments only)
    labeled = [s for s in sessions if label_store.get(s) is not None]
    logging.info(f"sessions: {len(sessions)} staged, {len(labeled)} with labels")

    n_val = max(1, int(round(args.val_frac * len(labeled))))
    rng = random.Random(args.seed)
    shuffled = sorted(labeled); rng.shuffle(shuffled)
    val_sessions = sorted(shuffled[:n_val])
    train_sessions = sorted(shuffled[n_val:])
    logging.info(f"split: train={len(train_sessions)}  val={len(val_sessions)}")

    # ---- Model ----
    console = MinimalConsole(
        sample_rate=args.sample_rate, max_input_len=args.audio_len,
    ).to(device)
    encoder = MixEncoderGrafx(
        console, sample_rate=args.sample_rate, d_model=args.d_model,
        n_track_layers=args.n_track_layers, use_ref_mix=False, mert_dim=0,
    ).to(device)
    logging.info(f"encoder params: {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M")

    if args.init_from:
        ckpt_path = Path(args.init_from).expanduser()
        if not ckpt_path.is_file():
            raise SystemExit(f"--init-from: file not found: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), weights_only=False, map_location=device)
        state_dict = ckpt["encoder_state_dict"] if isinstance(ckpt, dict) and "encoder_state_dict" in ckpt else ckpt
        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        logging.info(f"warm-started encoder from {ckpt_path}  "
                     f"(missing={len(missing)}, unexpected={len(unexpected)})")

    # ---- Losses ----
    recon_loss_fn = HybridReconLoss(
        sample_rate=args.sample_rate,
        w_mrstft=1.0,
        w_af=args.w_af,
        w_rms=args.w_rms,
        w_crest=args.w_crest,
        w_spec=args.w_spec,
        w_width=args.w_stereo_width,
        w_imbalance=args.w_stereo_imbalance,
    ).to(device)
    gap_loss_fn = make_grafx_prune_recon_loss(sample_rate=args.sample_rate).to(device)

    # Sampled params now live in normalized [0,1] space and the encoder
    # predicts in the same space (via sigmoid on its logits), so L_param
    # is naturally unit-scale per element. No whitening needed.

    # ---- Dataset ----
    if args.mode == "synth":
        sampler = ParamSampler(
            console.strip_param_shapes, console.group_param_shapes, seed=args.seed,
        )
        train_ds = SyntheticDataset(
            train_sessions, staging_dir, label_store, sampler,
            audio_len=args.audio_len, n_max=args.n_max,
            max_groups=args.max_groups, seed=args.seed,
            cache_size=args.cache_size,
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, collate_fn=collate_synth,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            prefetch_factor=2 if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0,
        )
    else:  # finetune
        sampler = None
        train_ds = PairedMultiSongDataset(
            train_sessions, staging_dir, label_store,
            audio_len=args.audio_len, n_max=args.n_max,
            seed=args.seed, cache_size=args.cache_size,
            min_window_gap=args.min_window_gap,
            pairs_per_session=args.pairs_per_session,
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, collate_fn=collate_pairs,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            prefetch_factor=2 if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0,
        )

    # ---- Optimizer / schedule ----
    optimizer = AdamW(encoder.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, betas=(0.9, 0.95))

    def _lr_lambda(step: int) -> float:
        if args.lr_warmup_steps > 0 and step < args.lr_warmup_steps:
            return (step + 1) / args.lr_warmup_steps
        decay_total = max(1, args.max_steps - args.lr_warmup_steps)
        progress = min(1.0, (step - args.lr_warmup_steps) / decay_total)
        min_mult = args.lr_cosine_min / args.lr
        return min_mult + (1.0 - min_mult) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    tb = SummaryWriter(log_dir=args.tb_logdir)
    logging.info(f"tb: {args.tb_logdir}  ckpts: {args.out_dir}")

    # ---- Train loop ----
    t0 = time.time()
    encoder.train()
    best_val = float("inf")
    train_iter = iter(train_loader)

    def _next_batch():
        nonlocal train_iter
        try: return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    def _micro_step_synth(step: int) -> dict:
        batch = _next_batch()
        tracks = batch["tracks"].to(device, non_blocking=True)
        track_mask = batch["track_mask"].to(device, non_blocking=True)
        group_idx = batch["group_assignments"].to(device, non_blocking=True)
        # Sampled params are normalized [0,1]
        sampled_strip_norm = _params_to_device(batch["strip_norm"], device)
        sampled_group_norm = _params_to_device(batch["group_norm"], device)

        # Denormalize for synth_mix rendering (engineering units → console)
        sampled_strip_eng = denormalize_params_dict(sampled_strip_norm)
        sampled_group_eng = denormalize_params_dict(sampled_group_norm)

        # Render synth_mix (target) — no grad needed
        with torch.no_grad():
            synth_mix = console(
                tracks, sampled_strip_eng, sampled_group_eng, group_idx,
                track_mask=track_mask, n_groups=args.max_groups,
            )

        # Encoder forward → raw logits → sigmoid → [0,1] → denormalize
        out = _encoder_forward(encoder, tracks, track_mask, group_idx,
                               n_groups=args.max_groups, use_bf16=args.use_bf16)
        pred_strip_norm = _sigmoid_params(out["strip_params"])
        pred_group_norm = _sigmoid_params(out["group_params"])
        pred_strip_eng = denormalize_params_dict(pred_strip_norm)
        pred_group_eng = denormalize_params_dict(pred_group_norm)

        B = tracks.shape[0]
        label_mask = track_mask.bool()
        ex_mask = torch.ones(B, dtype=torch.bool, device=device)
        n_groups_t = torch.full((B,), args.max_groups, dtype=torch.long, device=device)

        # L_param: predicted [0,1] vs sampled [0,1] — clean, no whitening needed.
        # All per-param errors are unit-scale by construction.
        L_param = grafx_param_huber_loss(
            pred_strip_norm, pred_group_norm,
            sampled_strip_norm, sampled_group_norm,
            label_mask, n_groups_t, ex_mask, delta=1.0,
        )

        # L_recon: student render (via denormalized engineering params) vs
        # synth_mix (also from engineering params). Gradient flows back
        # through console → denormalize → sigmoid → encoder logits.
        w_now = w_cosine_ramp_at(step, args.w_cosine_ramp, args.w_recon)
        if w_now > 0:
            student_mix = console(
                tracks, pred_strip_eng, pred_group_eng,
                group_idx, track_mask=track_mask, n_groups=args.max_groups,
            )
            L_recon = recon_loss_fn(student_mix, synth_mix)
            L_recon_total = L_recon["total"]
        else:
            L_recon = None
            L_recon_total = torch.zeros((), device=device)

        L_total = (args.w_param * L_param["L_param/total"]
                   + w_now * L_recon_total)
        (L_total / args.grad_accum).backward()

        log = {
            "L_total":       L_total.item(),
            "L_param/total": L_param["L_param/total"].item(),
            "L_param/strip": L_param["L_param/strip"].item(),
            "L_param/group": L_param["L_param/group"].item(),
            "w_recon":       w_now,
        }
        if L_recon is not None:
            log["L_recon/total"]  = L_recon["total"].item()
            log["L_recon/mrstft"] = L_recon["mrstft/full"].item()
            log["L_recon/width"]  = L_recon["stereo/width"].item()
            log["L_recon/imbal"]  = L_recon["stereo/imbalance"].item()
        return log

    def _micro_step_finetune(step: int) -> dict:
        batch = _next_batch()
        B_pair = batch["B_pair"]
        batch = attach_grafx_labels(
            batch, label_store, n_max=args.n_max, max_groups=args.max_groups,
            strip_schema=console.strip_param_shapes,
            group_schema=console.group_param_shapes,
        )
        tracks = batch["tracks"].to(device, non_blocking=True)
        track_mask = batch["track_mask"].to(device, non_blocking=True)
        group_idx = batch["group_assignments"].to(device, non_blocking=True)
        mix_ref = batch["mix"].to(device, non_blocking=True)

        out = _encoder_forward(encoder, tracks, track_mask, group_idx,
                               n_groups=args.max_groups, use_bf16=args.use_bf16)
        # sigmoid + denormalize encoder logits → engineering units for console
        pred_strip_norm = _sigmoid_params(out["strip_params"])
        pred_group_norm = _sigmoid_params(out["group_params"])
        pred_strip_eng = denormalize_params_dict(pred_strip_norm)
        pred_group_eng = denormalize_params_dict(pred_group_norm)

        # L_cons computed in normalized [0,1] space (same scale as the
        # encoder's primary output; no whitening needed)
        strip_a, strip_b = _split_params(pred_strip_norm, B_pair)
        group_a, group_b = _split_params(pred_group_norm, B_pair)
        n_groups_pair = batch["n_groups_per_example"][:B_pair].to(device)
        L_cons = grafx_param_consistency_loss(
            strip_a, group_a, strip_b, group_b,
            track_mask[:B_pair], n_groups_pair,
        )

        w_now = w_cosine_ramp_at(step, args.w_cosine_ramp, args.w_recon)
        if w_now > 0:
            student_mix = console(
                tracks, pred_strip_eng, pred_group_eng,
                group_idx, track_mask=track_mask, n_groups=args.max_groups,
            )
            L_recon = recon_loss_fn(student_mix, mix_ref)
            L_recon_total = L_recon["total"]
        else:
            L_recon = None
            L_recon_total = torch.zeros((), device=device)

        L_total = (args.w_cons * L_cons["L_cons/total"]
                   + w_now * L_recon_total)
        (L_total / args.grad_accum).backward()

        log = {
            "L_total":      L_total.item(),
            "L_cons/total": L_cons["L_cons/total"].item(),
            "w_recon":      w_now,
        }
        if L_recon is not None:
            log["L_recon/total"]  = L_recon["total"].item()
            log["L_recon/mrstft"] = L_recon["mrstft/full"].item()
            log["L_recon/width"]  = L_recon["stereo/width"].item()
            log["L_recon/imbal"]  = L_recon["stereo/imbalance"].item()
        return log

    micro_step = _micro_step_synth if args.mode == "synth" else _micro_step_finetune

    for step in range(args.max_steps):
        optimizer.zero_grad(set_to_none=True)
        micro_logs: list[dict] = []
        for _ in range(args.grad_accum):
            micro_logs.append(micro_step(step))
        grad_norm = torch.nn.utils.clip_grad_norm_(
            encoder.parameters(), max_norm=args.grad_clip,
        )
        optimizer.step()
        scheduler.step()

        log_keys = set().union(*[m.keys() for m in micro_logs])
        agg = {
            k: sum(m[k] for m in micro_logs if k in m) /
               max(1, sum(1 for m in micro_logs if k in m))
            for k in log_keys
        }

        if step % args.log_every == 0:
            for k, v in agg.items():
                tb.add_scalar(f"train/{k}", v, step)
            tb.add_scalar("train/grad_norm", grad_norm.item(), step)
            tb.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
            parts = [f"step {step:6d}/{args.max_steps}",
                     f"L_tot={agg['L_total']:7.3f}"]
            if "L_param/total" in agg: parts.append(f"L_p={agg['L_param/total']:6.2f}")
            if "L_cons/total"  in agg: parts.append(f"L_c={agg['L_cons/total']:5.2f}")
            if "L_recon/total" in agg: parts.append(f"L_r={agg['L_recon/total']:5.2f}")
            parts.append(f"w_r={agg['w_recon']:.3f}")
            parts.append(f"grad={grad_norm.item():6.1f}")
            parts.append(f"({(time.time()-t0)/(step+1):.2f}s/step)")
            logging.info("  ".join(parts))

        if step > 0 and step % args.val_every == 0:
            if args.mode == "synth":
                val_metrics = run_val_synth(
                    encoder, console, sampler, recon_loss_fn,
                    val_sessions, staging_dir, label_store,
                    args.audio_len, args.n_max, args.max_groups,
                    device, args.use_bf16,
                )
                score = val_metrics["val/L_recon"]
            else:
                val_metrics = run_val_finetune(
                    encoder, console, recon_loss_fn, gap_loss_fn,
                    val_sessions, staging_dir, label_store,
                    args.audio_len, args.n_max, args.max_groups,
                    device, args.use_bf16,
                )
                score = val_metrics["val/L_recon"]
            for k, v in val_metrics.items():
                tb.add_scalar(k, v, step)
            parts = [f"  val[step {step}]:"]
            for k in ("val/L_param", "val/L_recon", "val/amort_gap"):
                v = val_metrics.get(k, float("nan"))
                if v == v: parts.append(f"{k.split('/')[-1]}={v:.3f}")
            logging.info(" ".join(parts))
            if score < best_val:
                best_val = score
                torch.save({
                    "step": step, "encoder_state_dict": encoder.state_dict(),
                    "val_metrics": val_metrics, "args": vars(args),
                }, Path(args.out_dir) / "encoder_best.pt")
                logging.info(f"  -> new best val score = {best_val:.4f}")

        if step > 0 and step % args.ckpt_every == 0:
            torch.save({
                "step": step, "encoder_state_dict": encoder.state_dict(),
                "args": vars(args),
            }, Path(args.out_dir) / f"encoder_step{step:06d}.pt")

    # Final
    final_step = args.max_steps
    torch.save({
        "step": final_step, "encoder_state_dict": encoder.state_dict(),
        "args": vars(args),
    }, Path(args.out_dir) / f"encoder_step{final_step:06d}.pt")
    tb.close()
    logging.info(f"\ntraining done. {final_step} steps in "
                 f"{(time.time()-t0)/60:.1f} min. best val = {best_val:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
