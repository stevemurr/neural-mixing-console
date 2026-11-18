"""Stage 3 trainer using mix-level reconstruction (ILD loss) + STE.

Self-supervised pan learning: no pan targets needed. Supervision is the
engineer's mix audio. PanNet predicts per-track pan logits over {L, C, R};
straight-through-estimator turns them into hard {-1, 0, +1} pans at forward,
while soft-softmax gradient flows on backward. MonoPan applies the pans to
the stems, sum gives a predicted mix, ILD loss compares it to the engineer
mix.

Each track's pan_dir receives backprop gradient automatically weighted by
its contribution at each band. Per-track attribution is "free" — no LS
targets, no group conditioning, no engineering-convention overrides.

Run via shared/run.py:
    uv run python shared/run.py v13/experiments/v13-pan-recon.toml
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
from models.contribution_console import ContributionConsole
from models.mono_pan import MonoPan
from models.pan_net import PanNet
from models.pan_recon_loss import PanReconLoss
from training.data_v13 import session_split
from training.data_v13_recon import V13ReconDataset, collate_recon


def make_lr_schedule(warmup_steps: int, max_steps: int, lr_min_ratio: float):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        prog = min(1.0, max(0.0, prog))
        cos = 0.5 * (1 + math.cos(math.pi * prog))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cos
    return fn


def ste_pan_dir(
    logits: torch.Tensor,                       # (B, N, 3)
    class_to_pan: torch.Tensor,                 # (3,) buffer [-1, 0, +1]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Straight-through estimator from logits to pan_dir in {-1, 0, +1}.

    Forward: argmax → one-hot → · [-1, 0, +1].
    Backward: gradient flows through the softmax probs (STE).

    Returns (pan_dir hard-valued, hard one-hot — for logging).
    """
    y_soft = F.softmax(logits, dim=-1)
    y_hard_idx = logits.argmax(dim=-1)                              # (B, N)
    y_hard = F.one_hot(y_hard_idx, num_classes=3).to(y_soft.dtype)  # (B, N, 3)
    # STE: forward uses y_hard, backward gradient flows via y_soft.
    y_ste = y_soft + (y_hard - y_soft).detach()
    pan_dir = (y_ste * class_to_pan.view(1, 1, 3)).sum(dim=-1)      # (B, N)
    return pan_dir, y_hard


def pan_spread_stats(pan_vals: torch.Tensor, n_hist_bins: int = 10) -> dict:
    """Distribution summary of predicted pan_dir over real mono tracks.

    Answers "how spread are the pan choices": mode split (thresholded at
    ±0.5), central tendency / dispersion, tail percentiles, and a 10-bin
    sparkline histogram across [-1, +1] so collapse vs commitment vs
    huddle-near-center is visible at a glance in the log.
    """
    n = int(pan_vals.numel())
    if n == 0:
        return {
            "mode_pct_L": 0.0, "mode_pct_C": 0.0, "mode_pct_R": 0.0,
            "pan_mean": 0.0, "pan_std": 0.0, "pan_abs_mean": 0.0,
            "pan_p05": 0.0, "pan_p50": 0.0, "pan_p95": 0.0,
            "pan_min": 0.0, "pan_max": 0.0, "n_mono_real": 0,
            "hist_counts": [0] * n_hist_bins, "hist_str": " " * n_hist_bins,
        }
    pct_L = (pan_vals <= -0.5).float().mean().item()
    pct_R = (pan_vals >= 0.5).float().mean().item()
    p05, p50, p95 = torch.quantile(
        pan_vals, torch.tensor([0.05, 0.5, 0.95])
    ).tolist()
    hist = torch.histc(pan_vals, bins=n_hist_bins, min=-1.0, max=1.0)
    counts = [int(x) for x in hist.tolist()]
    peak = max(counts) or 1
    blocks = " ▁▂▃▄▅▆▇█"
    bar = "".join(blocks[min(8, round(8 * c / peak))] for c in counts)
    return {
        "mode_pct_L": pct_L,
        "mode_pct_C": 1.0 - pct_L - pct_R,
        "mode_pct_R": pct_R,
        "pan_mean": pan_vals.mean().item(),
        "pan_std": pan_vals.std(unbiased=False).item() if n > 1 else 0.0,
        "pan_abs_mean": pan_vals.abs().mean().item(),
        "pan_p05": p05, "pan_p50": p50, "pan_p95": p95,
        "pan_min": pan_vals.min().item(), "pan_max": pan_vals.max().item(),
        "n_mono_real": n,
        "hist_counts": counts, "hist_str": bar,
    }


@torch.no_grad()
def evaluate(
    net: PanNet,
    pan_op: MonoPan,
    console: ContributionConsole,
    loss_fn: PanReconLoss,
    loader: DataLoader,
    class_to_pan: torch.Tensor,
    device: torch.device,
    continuous: bool,
) -> dict:
    net.eval()
    total_total = 0.0
    total_ild = 0.0
    total_balance = 0.0
    total_width = 0.0
    total_ild_rmse = 0.0
    total_width_err = 0.0
    total_imb_err = 0.0
    n_batches = 0
    pan_chunks = []

    for batch in loader:
        stems = batch["stems"].to(device, non_blocking=True)
        mix = batch["mix"].to(device, non_blocking=True)
        is_stereo = batch["is_stereo"].to(device, non_blocking=True)
        track_mask = batch["track_mask"].to(device, non_blocking=True)

        C = console.compute_C(stems, track_mask=track_mask)
        if continuous:
            pan_dir = net(C, is_stereo, track_mask=track_mask)
        else:
            logits = net(C, is_stereo, track_mask=track_mask)
            pan_dir, _ = ste_pan_dir(logits, class_to_pan)
        pan_dir = pan_dir * (~is_stereo).to(pan_dir.dtype)

        panned = pan_op(stems, pan_dir, is_stereo)
        panned = panned * track_mask.unsqueeze(-1).unsqueeze(-1).to(panned.dtype)
        predicted_mix = panned.sum(dim=1)

        loss, metrics = loss_fn(
            predicted_mix, mix,
            stems=stems, is_stereo=is_stereo, track_mask=track_mask,
        )
        total_total += metrics["total"]
        total_ild += metrics["ild/l1_db"]
        total_balance += metrics["balance"]
        total_width += metrics["width"]
        total_ild_rmse += metrics["ild/rmse_db"]
        total_width_err += abs(metrics["width/err"])
        total_imb_err += abs(metrics["imbalance/err"])
        n_batches += 1

        mono_real = track_mask & ~is_stereo
        if mono_real.any():
            pan_chunks.append(pan_dir[mono_real].detach().float().cpu())

    net.train()
    pan_vals = torch.cat(pan_chunks) if pan_chunks else torch.zeros(0)
    out = {
        "loss":          total_total / max(n_batches, 1),
        "ild_l1_db":     total_ild / max(n_batches, 1),
        "balance":       total_balance / max(n_batches, 1),
        "width":         total_width / max(n_batches, 1),
        "ild_rmse_db":   total_ild_rmse / max(n_batches, 1),
        "width_err":     total_width_err / max(n_batches, 1),
        "imbalance_err": total_imb_err / max(n_batches, 1),
        "pan_vals":      pan_vals,
    }
    out.update(pan_spread_stats(pan_vals))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--staging-dir", default="dmc-data/grafx-prune-data-48k")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tb-logdir", default=None)
    ap.add_argument("--device", default=None)

    ap.add_argument("--sample-rate", type=int, default=48_000)
    ap.add_argument("--audio-len", type=int, default=288_000)
    ap.add_argument("--n-max", type=int, default=64)
    ap.add_argument("--n-bins", type=int, default=26)
    ap.add_argument("--examples-per-session", type=int, default=20)
    ap.add_argument("--cache-size", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)

    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--ffn-mult", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--continuous", action=argparse.BooleanOptionalAction, default=True,
                    help="Continuous tanh pan head (default). The recon loss is "
                         "differentiable in pan_dir, so this trains with an exact "
                         "gradient and no STE. --no-continuous uses the legacy "
                         "3-class straight-through head.")

    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-warmup-steps", type=int, default=200)
    ap.add_argument("--lr-cosine-min-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--ild-eps", type=float, default=1e-4)
    ap.add_argument("--w-balance", type=float, default=1.0,
                    help="Weight on the sign-aware DIRECTION term (which side).")
    ap.add_argument("--w-width", type=float, default=1.0,
                    help="Weight on the sign-blind COMMITMENT term (how wide). "
                         "Both terms are O(1), so the two weights are comparable.")

    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    args = ap.parse_args()
    # Recon trainer is always group-free; record it so checkpoints are
    # self-describing for audition/inference reconstruction.
    args.use_group_embed = False

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir or f"dmc-data/checkpoints/v13-pan-recon-{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(args.tb_logdir or f"runs/{out_dir.name}")
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))
    logging.info(f"device={device}  out={out_dir}  tb={tb_dir}")

    # ---- Sessions + datasets ----
    staging = Path(args.staging_dir)
    all_sessions = sorted([
        p.name for p in staging.iterdir()
        if p.is_dir() and (p / "mix.wav").is_file()
    ])
    train_s, val_s = session_split(all_sessions, args.val_frac, args.seed)
    logging.info(f"sessions: total={len(all_sessions)}  train={len(train_s)}  val={len(val_s)}")

    train_ds = V13ReconDataset(
        sorted(train_s), staging, sample_rate=args.sample_rate,
        audio_len=args.audio_len, n_max=args.n_max,
        examples_per_session=args.examples_per_session,
        cache_size=args.cache_size, seed=args.seed,
    )
    val_ds = V13ReconDataset(
        sorted(val_s), staging, sample_rate=args.sample_rate,
        audio_len=args.audio_len, n_max=args.n_max,
        examples_per_session=4,                    # fewer windows in val
        cache_size=args.cache_size, seed=args.seed + 1,
    )
    logging.info(f"examples: train={len(train_ds)}  val={len(val_ds)}")

    # Sequential ordering (shuffle=False) + persistent workers keeps each
    # worker on one session at a time → 95%+ LRU cache hit rate → ~5× speedup
    # vs random shuffling. Within-session windows are random (selected at
    # __init__), so batches still see diverse audio content.
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_recon, num_workers=args.num_workers, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_recon, num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )

    # ---- Model + ops + loss ----
    net = PanNet(
        n_bins=args.n_bins, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, ffn_mult=args.ffn_mult, dropout=args.dropout,
        use_group_embed=False, continuous=args.continuous,
    ).to(device)
    logging.info(f"pan head: {'continuous tanh' if args.continuous else '3-class STE'}")
    pan_op = MonoPan().to(device)
    console = ContributionConsole(
        sample_rate=args.sample_rate, n_bins=args.n_bins,
    ).to(device).eval()
    loss_fn = PanReconLoss(
        sample_rate=args.sample_rate, n_bins=args.n_bins, eps=args.ild_eps,
        w_balance=args.w_balance, w_width=args.w_width,
    ).to(device)
    class_to_pan = torch.tensor([-1.0, 0.0, 1.0], device=device)

    logging.info(f"PanNet params: {net.n_params():,}")

    opt = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, make_lr_schedule(args.lr_warmup_steps, args.max_steps,
                              args.lr_cosine_min_ratio),
    )

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

        stems = batch["stems"].to(device, non_blocking=True)
        mix = batch["mix"].to(device, non_blocking=True)
        is_stereo = batch["is_stereo"].to(device, non_blocking=True)
        track_mask = batch["track_mask"].to(device, non_blocking=True)

        # Compute C from stems (no grad through C)
        with torch.no_grad():
            C = console.compute_C(stems, track_mask=track_mask)

        # PanNet → pan_dir. Continuous head emits pan_dir directly (loss is
        # differentiable in pan_dir → exact gradient); legacy class head routes
        # through the straight-through estimator.
        if args.continuous:
            pan_dir = net(C, is_stereo, track_mask=track_mask)            # (B, N)
        else:
            logits = net(C, is_stereo, track_mask=track_mask)
            pan_dir, _ = ste_pan_dir(logits, class_to_pan)
        # Force stereo tracks to 0 (they bypass pan in MonoPan anyway, but
        # we zero the prediction to keep training signal clean)
        pan_dir = pan_dir * (~is_stereo).to(pan_dir.dtype)

        # Apply pan, mask padded slots, sum to predicted mix
        panned = pan_op(stems, pan_dir, is_stereo)
        panned = panned * track_mask.unsqueeze(-1).unsqueeze(-1).to(panned.dtype)
        predicted_mix = panned.sum(dim=1)

        loss, metrics = loss_fn(
            predicted_mix, mix,
            stems=stems, is_stereo=is_stereo, track_mask=track_mask,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        opt.step()
        sched.step()

        if step % args.log_every == 0:
            now = time.time()
            step_dt = (now - last_log_time) / max(1, step - last_log_step)
            cur_lr = sched.get_last_lr()[0]
            mono_real = track_mask & ~is_stereo
            pv = pan_dir.detach()[mono_real].float()
            pan_abs = pv.abs().mean().item() if pv.numel() else 0.0
            pan_sd = pv.std(unbiased=False).item() if pv.numel() > 1 else 0.0
            logging.info(
                f"step {step:5d}/{args.max_steps}  "
                f"L={metrics['total']:.3f}  ild={metrics['ild/l1_db']:.2f}  "
                f"bal={metrics['balance']:.3f}  wid={metrics['width']:.3f}  "
                f"wErr={metrics['width/err']:+.3f}  "
                f"|pan|={pan_abs:.2f}  pan_sd={pan_sd:.2f}  "
                f"grad={grad_norm.item():.2f}  lr={cur_lr:.2e}  "
                f"({step_dt*1000:.0f} ms/step)"
            )
            writer.add_scalar("train/total", metrics["total"], step)
            writer.add_scalar("train/ild_l1_db", metrics["ild/l1_db"], step)
            writer.add_scalar("train/ild_rmse_db", metrics["ild/rmse_db"], step)
            writer.add_scalar("train/balance", metrics["balance"], step)
            writer.add_scalar("train/width", metrics["width"], step)
            writer.add_scalar("train/width_err", metrics["width/err"], step)
            writer.add_scalar("train/imbalance_err", metrics["imbalance/err"], step)
            writer.add_scalar("train/grad_norm", grad_norm.item(), step)
            writer.add_scalar("train/lr", cur_lr, step)
            writer.add_scalar("train/pan_abs_mean", pan_abs, step)
            writer.add_scalar("train/pan_std", pan_sd, step)
            if pv.numel() > 0:
                writer.add_histogram("train/pan_dir", pv.cpu(), step)
            last_log_time = now
            last_log_step = step

        if step % args.val_every == 0 and step > 0:
            val = evaluate(net, pan_op, console, loss_fn, val_dl,
                           class_to_pan, device, args.continuous)
            logging.info(
                f"  val[step {step}]:  L={val['loss']:.3f}  "
                f"ild={val['ild_l1_db']:.2f}  bal={val['balance']:.3f}  "
                f"wid={val['width']:.3f}  wErr={val['width_err']:.3f}  "
                f"mode L/C/R = {val['mode_pct_L']*100:.0f}/{val['mode_pct_C']*100:.0f}/"
                f"{val['mode_pct_R']*100:.0f}%  (n_mono={val['n_mono_real']})"
            )
            logging.info(
                f"     pan spread: mean={val['pan_mean']:+.3f}  "
                f"sd={val['pan_std']:.3f}  |pan|={val['pan_abs_mean']:.3f}  "
                f"p05/50/95={val['pan_p05']:+.2f}/{val['pan_p50']:+.2f}/{val['pan_p95']:+.2f}  "
                f"range=[{val['pan_min']:+.2f}, {val['pan_max']:+.2f}]"
            )
            logging.info(
                f"     hist[-1..+1]: |{val['hist_str']}|  {val['hist_counts']}"
            )
            writer.add_scalar("val/total", val["loss"], step)
            writer.add_scalar("val/ild_l1_db", val["ild_l1_db"], step)
            writer.add_scalar("val/balance", val["balance"], step)
            writer.add_scalar("val/width", val["width"], step)
            writer.add_scalar("val/ild_rmse_db", val["ild_rmse_db"], step)
            writer.add_scalar("val/width_err", val["width_err"], step)
            writer.add_scalar("val/imbalance_err", val["imbalance_err"], step)
            writer.add_scalar("val/mode_pct_L", val["mode_pct_L"], step)
            writer.add_scalar("val/mode_pct_C", val["mode_pct_C"], step)
            writer.add_scalar("val/mode_pct_R", val["mode_pct_R"], step)
            writer.add_scalar("val/pan_mean", val["pan_mean"], step)
            writer.add_scalar("val/pan_std", val["pan_std"], step)
            writer.add_scalar("val/pan_abs_mean", val["pan_abs_mean"], step)
            if val["n_mono_real"] > 0:
                writer.add_histogram("val/pan_dir", val["pan_vals"], step)
            if val["loss"] < best_val:
                best_val = val["loss"]
                ckpt_path = out_dir / "pan_recon_best.pt"
                torch.save({
                    "step":     step,
                    "model":    net.state_dict(),
                    "args":     vars(args),
                    "val":      val,
                    "best_val": best_val,
                }, ckpt_path)
                logging.info(f"  -> new best val total = {best_val:.4f}; saved")

        if step % args.ckpt_every == 0 and step > 0:
            torch.save({"step": step, "model": net.state_dict(),
                        "args": vars(args)},
                       out_dir / f"pan_recon_step{step:06d}.pt")

    writer.close()
    logging.info(f"training done. best val L_ild = {best_val:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
