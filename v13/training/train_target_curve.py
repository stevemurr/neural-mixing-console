"""Stage 2 trainer for the v13 contribution-mixer architecture.

Supervised regression: per-track contribution vectors C → mix-bus dB
delta. Tiny model (1.8 M params), tiny in-memory dataset (~14 MB
precomputed). Trains in minutes on a single GPU.

Loss is Huber on the dB delta (smooth L1, β=2 dB). Below 2 dB error
the gradient is quadratic (smooth); above 2 dB it is linear (robust to
weird sessions). Mix-bus EQ moves are mostly in [-6, +6] dB, so β=2
matches the engineering scale.

Per-bin MSE is also tracked to spot which mel bands the model gets
right (typically low-mid, where mix decisions concentrate) vs which
are noisy.

Run via shared/run.py:
    uv run python shared/run.py v13/experiments/v13-target-curve.toml
"""

from __future__ import annotations

import argparse
import json
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
from models.target_curve_net import TargetCurveNet
from training.data_v13 import V13Dataset, collate_v13, session_split


def make_lr_schedule(
    warmup_steps: int, max_steps: int, lr_min_ratio: float,
):
    """Linear warmup + cosine decay to lr_min_ratio * peak_lr."""
    def fn(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        # cosine 1 → lr_min_ratio
        prog = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        prog = min(1.0, max(0.0, prog))
        cos = 0.5 * (1 + math.cos(math.pi * prog))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cos
    return fn


def evaluate(
    net: TargetCurveNet, loader: DataLoader, device: torch.device,
) -> dict:
    net.eval()
    total_huber = 0.0
    total_mse = 0.0
    n = 0
    bin_sq_err = None
    with torch.no_grad():
        for batch in loader:
            C = batch["C"].to(device)
            target = batch["target_delta_db"].to(device)
            mask = batch["track_mask"].to(device)
            pred = net(C, mask)
            huber = F.smooth_l1_loss(pred, target, beta=2.0, reduction="sum").item()
            sq = (pred - target) ** 2
            total_mse += sq.sum().item()
            total_huber += huber
            if bin_sq_err is None:
                bin_sq_err = sq.sum(dim=0).cpu()
            else:
                bin_sq_err += sq.sum(dim=0).cpu()
            n += target.numel()
    net.train()
    return {
        "huber":   total_huber / max(n, 1),
        "mse":     total_mse / max(n, 1),
        "rmse_db": math.sqrt(total_mse / max(n, 1)),
        "per_bin_rmse": (bin_sq_err / max(len(loader.dataset), 1)).sqrt().tolist(),
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
    ap.add_argument("--max-delta-db", type=float, default=12.0)
    ap.add_argument("--huber-beta", type=float, default=2.0)

    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-warmup-steps", type=int, default=200)
    ap.add_argument("--lr-cosine-min-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir or f"dmc-data/checkpoints/v13-target-curve-{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(args.tb_logdir or f"runs/{out_dir.name}")
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))
    logging.info(f"device={device}  out={out_dir}  tb={tb_dir}")

    # ---- Data ----
    if not Path(args.precomputed).is_file():
        logging.error(f"precomputed dataset missing: {args.precomputed}")
        return 1
    all_examples = torch.load(args.precomputed, weights_only=False, map_location="cpu")
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

    # ---- Model ----
    net = TargetCurveNet(
        n_bins=args.n_bins, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, ffn_mult=args.ffn_mult, dropout=args.dropout,
        max_delta_db=args.max_delta_db,
    ).to(device)
    logging.info(f"TargetCurveNet params: {net.n_params():,}")

    opt = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    lr_fn = make_lr_schedule(
        args.lr_warmup_steps, args.max_steps, args.lr_cosine_min_ratio,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    # ---- Training loop ----
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
        target = batch["target_delta_db"].to(device, non_blocking=True)
        mask = batch["track_mask"].to(device, non_blocking=True)

        pred = net(C, mask)
        loss = F.smooth_l1_loss(pred, target, beta=args.huber_beta)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        opt.step()
        sched.step()

        if step % args.log_every == 0:
            now = time.time()
            dt = now - last_log_time
            step_dt = dt / max(1, step - last_log_step)
            cur_lr = sched.get_last_lr()[0]
            logging.info(
                f"step {step:5d}/{args.max_steps}  "
                f"L_huber={loss.item():.4f}  grad={grad_norm.item():.2f}  "
                f"lr={cur_lr:.2e}  ({step_dt*1000:.0f} ms/step)"
            )
            writer.add_scalar("train/huber", loss.item(), step)
            writer.add_scalar("train/grad_norm", grad_norm.item(), step)
            writer.add_scalar("train/lr", cur_lr, step)
            last_log_time = now
            last_log_step = step

        if step % args.val_every == 0 and step > 0:
            val = evaluate(net, val_dl, device)
            logging.info(
                f"  val[step {step}]:  huber={val['huber']:.4f}  "
                f"mse={val['mse']:.4f}  rmse_db={val['rmse_db']:.3f}"
            )
            writer.add_scalar("val/huber", val["huber"], step)
            writer.add_scalar("val/mse", val["mse"], step)
            writer.add_scalar("val/rmse_db", val["rmse_db"], step)
            for b, e in enumerate(val["per_bin_rmse"]):
                writer.add_scalar(f"val/bin_{b:02d}_rmse_db", e, step)
            if val["huber"] < best_val:
                best_val = val["huber"]
                ckpt_path = out_dir / "target_curve_best.pt"
                torch.save({
                    "step":      step,
                    "model":     net.state_dict(),
                    "args":      vars(args),
                    "val":       val,
                    "best_val":  best_val,
                }, ckpt_path)
                logging.info(f"  -> new best val huber = {best_val:.4f}; saved")

        if step % args.ckpt_every == 0 and step > 0:
            ckpt_path = out_dir / f"target_curve_step{step:06d}.pt"
            torch.save({
                "step":  step,
                "model": net.state_dict(),
                "args":  vars(args),
            }, ckpt_path)

    writer.close()
    logging.info(f"training done. best val huber = {best_val:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
