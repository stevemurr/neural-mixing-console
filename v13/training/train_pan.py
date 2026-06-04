"""Stage 3 trainer for the v13 contribution-mixer.

Supervised regression: per-track contribution C + is_stereo flag → per-track
pan_dir in [-1, +1]. Loss is MSE on the LS-recovered pan targets, masked to
real mono tracks (stereo tracks bypass pan and shouldn't pull the loss).

Reports both RMSE on pan_dir and "mode accuracy" — what fraction of tracks
land in the correct hard-pan mode (left if target ≤ -0.5, right if target
≥ +0.5, center otherwise). The mode accuracy is what matters audibly —
engineering pan moves cluster at {-1, 0, +1}.

Run via shared/run.py:
    uv run python shared/run.py v13/experiments/v13-pan.toml
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.pan_net import PanNet
from training.data_v13 import V13Dataset, collate_v13, session_split


def make_lr_schedule(warmup_steps: int, max_steps: int, lr_min_ratio: float):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        prog = min(1.0, max(0.0, prog))
        cos = 0.5 * (1 + math.cos(math.pi * prog))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cos
    return fn


def evaluate(
    net: PanNet, loader: DataLoader, device: torch.device,
) -> dict:
    net.eval()
    total_sq = 0.0
    total_ce = 0.0
    n_loss = 0
    total_mode_correct = 0
    total_modes = 0
    mode_counts = [0, 0, 0]
    mode_correct = [0, 0, 0]

    with torch.no_grad():
        for batch in loader:
            C = batch["C"].to(device)
            target = batch["pan_target"].to(device)
            is_stereo = batch["is_stereo"].to(device)
            group_idx = batch["group_idx"].to(device)
            track_mask = batch["track_mask"].to(device)
            mask = track_mask & ~is_stereo
            logits = net(C, is_stereo, group_idx, track_mask)        # (B, N, 3)
            target_cls = PanNet.pan_target_to_class(target)
            ce_per = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 3), target_cls.reshape(-1), reduction="none",
            ).reshape(target_cls.shape)
            ce_masked = ce_per * mask.to(ce_per.dtype)
            total_ce += ce_masked.sum().item()
            n_loss += int(mask.sum().item())

            # Soft inference for RMSE
            pan_pred = (torch.softmax(logits, dim=-1) * net.class_to_pan).sum(dim=-1)
            sq = (pan_pred - target) ** 2 * mask.to(target.dtype)
            total_sq += sq.sum().item()

            # Mode accuracy via argmax
            pred_cls = logits.argmax(dim=-1)              # 0=L, 1=C, 2=R
            correct = (pred_cls == target_cls) & mask
            total_mode_correct += int(correct.sum().item())
            total_modes += int(mask.sum().item())
            for c in (0, 1, 2):
                m_mask = (target_cls == c) & mask
                mode_counts[c] += int(m_mask.sum().item())
                mode_correct[c] += int(((pred_cls == c) & m_mask).sum().item())

    net.train()
    return {
        "ce":            total_ce / max(n_loss, 1),
        "mse":           total_sq / max(n_loss, 1),
        "rmse":          math.sqrt(total_sq / max(n_loss, 1)),
        "mode_acc":      total_mode_correct / max(total_modes, 1),
        "mode_acc_L":    mode_correct[0] / max(mode_counts[0], 1),
        "mode_acc_C":    mode_correct[1] / max(mode_counts[1], 1),
        "mode_acc_R":    mode_correct[2] / max(mode_counts[2], 1),
        "n_target_modes_L_C_R": (mode_counts[0], mode_counts[1], mode_counts[2]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--precomputed", default="dmc-data/v13_precomputed.pt")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tb-logdir", default=None)
    ap.add_argument("--device", default=None)

    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)

    ap.add_argument("--n-bins", type=int, default=26)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--ffn-mult", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-warmup-steps", type=int, default=200)
    ap.add_argument("--lr-cosine-min-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--weight-LR", type=float, default=2.0,
                    help="CE weight for L and R classes (corpus has 17% each "
                         "vs 65% C — use this to counter class imbalance "
                         "without over-correcting).")
    ap.add_argument("--weight-C", type=float, default=1.0,
                    help="CE weight for the Center class.")

    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir or f"dmc-data/checkpoints/v13-pan-{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(args.tb_logdir or f"runs/{out_dir.name}")
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))
    logging.info(f"device={device}  out={out_dir}  tb={tb_dir}")

    if not Path(args.precomputed).is_file():
        logging.error(f"precomputed dataset missing: {args.precomputed}")
        return 1
    all_examples = torch.load(args.precomputed, weights_only=False, map_location="cpu")
    if "pan_target" not in all_examples[0]:
        logging.error(
            "precomputed dataset has no pan_target — re-run "
            "v13/scripts/precompute_v13_dataset.py to add Stage 3 fields."
        )
        return 1
    all_sessions = sorted({e["session"] for e in all_examples})
    train_s, val_s = session_split(all_sessions, args.val_frac, args.seed)
    train_ds = V13Dataset(args.precomputed, train_s)
    val_ds = V13Dataset(args.precomputed, val_s)
    logging.info(
        f"sessions: total={len(all_sessions)}  "
        f"train={len(train_s)} ({len(train_ds)} ex)  "
        f"val={len(val_s)} ({len(val_ds)} ex)"
    )

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_v13, num_workers=args.num_workers, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_v13, num_workers=args.num_workers,
    )

    net = PanNet(
        n_bins=args.n_bins, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, ffn_mult=args.ffn_mult, dropout=args.dropout,
    ).to(device)
    logging.info(f"PanNet params: {net.n_params():,}")

    opt = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, make_lr_schedule(args.lr_warmup_steps, args.max_steps,
                              args.lr_cosine_min_ratio),
    )

    # ---- Class-balanced CE weights ----
    # Distribution in corpus is ~17% L, ~65% C, ~18% R for mono tracks.
    # Inverse-frequency weights would be ~3.8 / 1.0 / 3.7. We use a softer
    # ratio (2.0 / 1.0 / 2.0) — full inverse-frequency over-corrects and
    # makes the model trigger-happy on L/R for tracks that should be center.
    class_weights = torch.tensor(
        [args.weight_LR, args.weight_C, args.weight_LR], device=device,
    )

    train_iter = iter(train_dl)
    best_val = float("inf")
    last_log_time = time.time()
    last_log_step = 0

    for step in range(args.max_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch = next(train_iter)

        C = batch["C"].to(device, non_blocking=True)
        target = batch["pan_target"].to(device, non_blocking=True)
        is_stereo = batch["is_stereo"].to(device, non_blocking=True)
        group_idx = batch["group_idx"].to(device, non_blocking=True)
        track_mask = batch["track_mask"].to(device, non_blocking=True)
        loss_mask = track_mask & ~is_stereo                               # mono real tracks

        logits = net(C, is_stereo, group_idx, track_mask)                 # (B, N, 3)
        target_cls = PanNet.pan_target_to_class(target)
        # Masked CE: cross_entropy with reduction='none', then mask + mean
        ce_per = F.cross_entropy(
            logits.reshape(-1, 3), target_cls.reshape(-1),
            weight=class_weights, reduction="none",
        ).reshape(target_cls.shape)
        loss = (ce_per * loss_mask.to(ce_per.dtype)).sum() / loss_mask.sum().clamp(min=1)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        opt.step()
        sched.step()

        if step % args.log_every == 0:
            now = time.time()
            step_dt = (now - last_log_time) / max(1, step - last_log_step)
            cur_lr = sched.get_last_lr()[0]
            logging.info(
                f"step {step:5d}/{args.max_steps}  L_ce={loss.item():.4f}  "
                f"grad={grad_norm.item():.2f}  lr={cur_lr:.2e}  "
                f"({step_dt*1000:.0f} ms/step)"
            )
            writer.add_scalar("train/ce", loss.item(), step)
            writer.add_scalar("train/grad_norm", grad_norm.item(), step)
            writer.add_scalar("train/lr", cur_lr, step)
            last_log_time = now
            last_log_step = step

        if step % args.val_every == 0 and step > 0:
            val = evaluate(net, val_dl, device)
            logging.info(
                f"  val[step {step}]:  ce={val['ce']:.4f}  rmse={val['rmse']:.3f}  "
                f"mode_acc={val['mode_acc']*100:.1f}%  "
                f"(L={val['mode_acc_L']*100:.0f}%  C={val['mode_acc_C']*100:.0f}%  "
                f"R={val['mode_acc_R']*100:.0f}%)  "
                f"n_targets L/C/R={val['n_target_modes_L_C_R']}"
            )
            writer.add_scalar("val/ce", val["ce"], step)
            writer.add_scalar("val/rmse", val["rmse"], step)
            writer.add_scalar("val/mode_acc", val["mode_acc"], step)
            writer.add_scalar("val/mode_acc_L", val["mode_acc_L"], step)
            writer.add_scalar("val/mode_acc_C", val["mode_acc_C"], step)
            writer.add_scalar("val/mode_acc_R", val["mode_acc_R"], step)
            # Save best by mode accuracy (engineering-relevant) rather than
            # raw loss — CE prefers conservative even when wrong, mode_acc
            # tracks "did the model commit to the right L/C/R".
            if -val["mode_acc"] < best_val:
                best_val = -val["mode_acc"]
                ckpt_path = out_dir / "pan_best.pt"
                torch.save({
                    "step":     step,
                    "model":    net.state_dict(),
                    "args":     vars(args),
                    "val":      val,
                    "best_val": best_val,
                }, ckpt_path)
                logging.info(f"  -> new best val mode_acc = {-best_val*100:.1f}%; saved")

        if step % args.ckpt_every == 0 and step > 0:
            ckpt_path = out_dir / f"pan_step{step:06d}.pt"
            torch.save({"step": step, "model": net.state_dict(),
                        "args": vars(args)}, ckpt_path)

    writer.close()
    logging.info(f"training done. best val mode_acc = {-best_val*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
