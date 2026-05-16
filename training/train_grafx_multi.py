"""Multi-song trainer for the grafx encoder (phase 4b → production).

Generalizes `train_grafx_overfit.py` to a corpus of staged sessions
(typically Cambridge-MT under `dmc-data/grafx-prune-data/`):

  - `MultiSongDataset` reads from staged session dirs, yields one
    random 30s window from one randomly-chosen session per __next__.
  - Songs without `labels_grafx_prune.pt` still contribute via the
    audio-domain MR-STFT recon loss (their `label_example_mask` is
    False, so L_param contributes zero for them).
  - Loss balance: w_recon=0 during a warmup phase, then linearly
    ramps to its target over the next `recon_warmup` steps. This
    avoids the "recon descends, param climbs" pattern we saw in the
    overfit driver when the two losses fought each other from step 0.
  - Validation: every `--val-every` steps, forward each held-out
    session at a fixed 30s window and report mean L_param + L_recon
    + per-param L1 ratios (for labeled sessions).

Run:
    uv run python training/train_grafx_multi.py \\
        --staging-dir dmc-data/grafx-prune-data \\
        --max-steps 20000 --batch-size 2 --audio-len 90000 --n-max 32 \\
        --w-recon-target 0.5 --recon-warmup 5000

Memory tuning: `audio_len` and `n_max` together drive activation memory.
The grafx strip processors run for ALL n_max padded track slots (mask
zeros only the audio AFTER the chain, not the compute), so n_max=64 +
audio_len=300_000 OOMs the GB10 (unified host/GPU memory pool).
Defaults below (audio_len=90_000 / 3s, n_max=32) match the working
single-song overfit driver and peak ~38 GB on the GB10. Headroom from
there is ~3×: bumping any one of {audio_len → 150_000, n_max → 48,
batch → 4} should still fit; bumping two simultaneously needs a check.

Ckpts under `dmc-data/checkpoints/grafx_multi_<run_name>/`.
TB logs under `runs/grafx_multi_<run_name>/`.
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
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.grafx_console import GrafxMixingConsole
from models.encoders_grafx import MixEncoderGrafx
from training.data_grafx import LabelStore, attach_grafx_labels
from training.losses_grafx import (
    grafx_param_huber_loss, make_grafx_prune_recon_loss,
)


# ---------- Dataset ----------

def _list_staged_sessions(staging_dir: Path) -> list[str]:
    """Return sorted names of sessions with mix.wav + stems/ + correspondence.yaml."""
    out: list[str] = []
    for d in sorted(staging_dir.iterdir()):
        if not d.is_dir():
            continue
        if (d / "mix.wav").is_file() and (d / "stems").is_dir() and \
           (d / "correspondence.yaml").is_file():
            out.append(d.name)
    return out


def _load_session(
    session: str,
    staging_dir: Path,
    label_store: LabelStore,
) -> dict:
    """Load all stems + mix into RAM. Stem order follows labels (if present)
    else correspondence.yaml.
    """
    sess_dir = staging_dir / session
    labels = label_store.get(session)
    if labels is not None:
        stem_filenames = list(labels["stem_filenames"])
    else:
        corr = yaml.safe_load(open(sess_dir / "correspondence.yaml"))
        stem_filenames = [f for files in corr.values() for f in files]

    stems_data: list[torch.Tensor] = []
    for fname in stem_filenames:
        data, sr = sf.read(str(sess_dir / "stems" / fname),
                           dtype="float32", always_2d=True)
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        elif data.shape[1] > 2:
            data = data[:, :2]
        stems_data.append(torch.from_numpy(data.T.copy()))

    T_total_stems = min(s.shape[-1] for s in stems_data)
    mix_data, mix_sr = sf.read(str(sess_dir / "mix.wav"),
                               dtype="float32", always_2d=True)
    assert mix_sr == sr, f"sr mismatch in {session}: stems={sr}, mix={mix_sr}"
    if mix_data.shape[1] == 1:
        mix_data = np.repeat(mix_data, 2, axis=1)
    mix_tensor = torch.from_numpy(mix_data.T.copy())
    T_total = min(T_total_stems, mix_tensor.shape[-1])
    stems = torch.stack([s[:, :T_total] for s in stems_data])  # (N, 2, T_total)
    mix = mix_tensor[:, :T_total]                               # (2, T_total)
    return {
        "session": session,
        "stems": stems,
        "mix": mix,
        "T_total": T_total,
        "sr": sr,
        "stem_filenames": stem_filenames,
    }


def _example_from_session(
    data: dict,
    start: int,
    audio_len: int,
    n_max: int,
) -> dict:
    """Slice (stems, mix) at a given start, pad/truncate to n_max."""
    stems = data["stems"][:, :, start:start + audio_len]   # (N_real, 2, T)
    mix = data["mix"][:, start:start + audio_len]          # (2, T)
    stem_filenames = list(data["stem_filenames"])

    N_real = stems.shape[0]
    if N_real > n_max:
        stems = stems[:n_max]
        stem_filenames = stem_filenames[:n_max]
        N_real = n_max

    if N_real < n_max:
        pad = torch.zeros(n_max - N_real, 2, audio_len, dtype=stems.dtype)
        stems = torch.cat([stems, pad], dim=0)

    track_mask = torch.zeros(n_max, dtype=torch.bool)
    track_mask[:N_real] = True

    track_metas = [{"filename": fn} for fn in stem_filenames]
    track_metas += [{"filename": ""} for _ in range(n_max - N_real)]

    return {
        "tracks": stems,
        "mix": mix,
        "track_mask": track_mask,
        "meta": {"session": data["session"], "tracks": track_metas},
    }


class MultiSongDataset(IterableDataset):
    """Random-window over a list of staged sessions.

    Each __next__: pick a random session from the worker's shard, load
    it (LRU-cached), pick a random 30s window, return.
    """

    def __init__(
        self,
        sessions: list[str],
        staging_dir: Path,
        label_store: LabelStore,
        audio_len: int,
        n_max: int,
        seed: int = 0,
        cache_size: int = 4,
    ):
        super().__init__()
        if not sessions:
            raise ValueError("no sessions provided")
        self.sessions = list(sessions)
        self.staging_dir = Path(staging_dir)
        self.label_store = label_store
        self.audio_len = audio_len
        self.n_max = n_max
        self.seed = seed
        self.cache_size = cache_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # Each worker shards the session list deterministically
        if worker_info is not None:
            shard = self.sessions[worker_id::worker_info.num_workers]
        else:
            shard = self.sessions
        rng = random.Random(self.seed + worker_id)

        cache: dict[str, dict] = {}
        cache_order: list[str] = []  # MRU last

        def _evict_lru():
            if cache_order:
                victim = cache_order.pop(0)
                cache.pop(victim, None)

        while True:
            session = rng.choice(shard)
            if session not in cache:
                if len(cache) >= self.cache_size:
                    _evict_lru()
                try:
                    cache[session] = _load_session(
                        session, self.staging_dir, self.label_store,
                    )
                    cache_order.append(session)
                except Exception as e:
                    logging.warning(f"failed to load {session}: {e}")
                    continue
            else:
                cache_order.remove(session)
                cache_order.append(session)

            data = cache[session]
            T_total = data["T_total"]
            if T_total < self.audio_len + 1:
                continue
            start = rng.randint(0, T_total - self.audio_len - 1)
            yield _example_from_session(data, start, self.audio_len, self.n_max)


def collate_multi(batch: list[dict]) -> dict:
    """Stack per-example tensors into a batch."""
    tracks = torch.stack([ex["tracks"] for ex in batch])
    mix = torch.stack([ex["mix"] for ex in batch])
    track_mask = torch.stack([ex["track_mask"] for ex in batch])
    metas = [ex["meta"] for ex in batch]
    B, N = tracks.shape[:2]
    return {
        "tracks": tracks,
        "mix": mix,
        "track_mask": track_mask,
        "mert_embeddings": torch.zeros(B, N, 768),  # unused (mert_dim=0)
        "meta": metas,
    }


# ---------- Loss schedule ----------

def w_recon_at(step: int, warmup: int, target: float) -> float:
    """0 for steps < warmup, linear ramp to target over the next `warmup`
    steps, then constant.
    """
    if step < warmup:
        return 0.0
    if step < 2 * warmup:
        return target * (step - warmup) / max(1, warmup)
    return target


# ---------- Validation ----------

def run_val(
    encoder, console, recon_loss_fn,
    val_sessions: list[str],
    staging_dir: Path,
    label_store: LabelStore,
    audio_len: int,
    n_max: int,
    max_groups: int,
    device: torch.device,
    strip_schema: dict | None = None,
    group_schema: dict | None = None,
) -> dict[str, float]:
    """One forward per val session at a fixed 30s window starting 25% in.

    Returns:
      val/L_param: mean L_param across val sessions that have labels
      val/L_recon: mean L_recon across all val sessions
      val/n_labeled: # labeled val sessions contributing to L_param
    """
    encoder.eval()
    L_params, L_recons = [], []
    with torch.no_grad():
        for session in val_sessions:
            try:
                data = _load_session(session, staging_dir, label_store)
            except Exception as e:
                logging.warning(f"val: failed to load {session}: {e}")
                continue
            T_total = data["T_total"]
            if T_total < audio_len + 1:
                continue
            start = T_total // 4   # deterministic 25%-in window
            ex = _example_from_session(data, start, audio_len, n_max)
            batch = collate_multi([ex])
            batch = attach_grafx_labels(
                batch, label_store, n_max=n_max, max_groups=max_groups,
                strip_schema=strip_schema, group_schema=group_schema,
            )

            tracks = batch["tracks"].to(device, non_blocking=True)
            mix_ref = batch["mix"].to(device, non_blocking=True)
            track_mask = batch["track_mask"].to(device, non_blocking=True)
            group_idx = batch["group_assignments"].to(device, non_blocking=True)

            out = encoder(tracks, track_mask, group_idx, n_groups=max_groups)
            pred_mix = console(
                tracks, out["strip_params"], out["group_params"],
                out["group_assignments"], track_mask=track_mask,
                n_groups=max_groups,
            )
            L_recon = recon_loss_fn(pred_mix, mix_ref)
            L_recons.append(L_recon["match/full"].item())

            if batch["label_example_mask"][0].item():
                L_param = grafx_param_huber_loss(
                    out["strip_params"], out["group_params"],
                    batch["label_strip_params"], batch["label_group_params"],
                    batch["label_track_mask"].to(device),
                    batch["n_groups_per_example"].to(device),
                    batch["label_example_mask"].to(device),
                    delta=1.0,
                )
                L_params.append(L_param["L_param/total"].item())
    encoder.train()
    return {
        "val/L_param": sum(L_params) / len(L_params) if L_params else float("nan"),
        "val/L_recon": sum(L_recons) / len(L_recons) if L_recons else float("nan"),
        "val/n_labeled": float(len(L_params)),
        "val/n_total":   float(len(L_recons)),
    }


# ---------- Main ----------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--staging-dir", default="dmc-data/grafx-prune-data")
    ap.add_argument("--run-name", default=None,
                    help="Used in output paths. Defaults to a timestamp.")
    ap.add_argument("--out-dir", default=None,
                    help="Default dmc-data/checkpoints/grafx_multi_<run_name>")
    ap.add_argument("--tb-logdir", default=None,
                    help="Default runs/grafx_multi_<run_name>")
    ap.add_argument("--device", default=None)

    # Data
    ap.add_argument("--sample-rate", type=int, default=30_000)
    ap.add_argument("--audio-len", type=int, default=90_000,
                    help="Samples per training window (default 3s @ 30kHz). "
                         "Labels were fit at 30s but are window-agnostic; "
                         "3s matches the single-song overfit driver. See "
                         "module docstring for memory headroom notes.")
    ap.add_argument("--n-max", type=int, default=32,
                    help="Max stems per session (truncate beyond). Strips "
                         "run for all n_max padded slots — keep tight to "
                         "the corpus stem-count distribution (mean ~26).")
    ap.add_argument("--max-groups", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="Fraction of sessions held out for validation.")
    ap.add_argument("--cache-size", type=int, default=4,
                    help="Per-worker LRU cache of loaded sessions.")
    ap.add_argument("--num-workers", type=int, default=0)

    # Model
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-track-layers", type=int, default=4,
                    help="Transformer depth (deeper than overfit's 2).")

    # Optim / loss
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=20_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=5.0,
                    help="Re-enabled for multi-song (outlier protection).")
    ap.add_argument("--w-param", type=float, default=1.0)
    ap.add_argument("--w-recon-target", type=float, default=0.5)
    ap.add_argument("--recon-warmup", type=int, default=5000,
                    help="Steps of param-only training before recon ramps in.")

    # Logging / checkpointing
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.run_name is None:
        args.run_name = time.strftime("%Y%m%d_%H%M%S")
    if args.out_dir is None:
        args.out_dir = f"dmc-data/checkpoints/grafx_multi_{args.run_name}"
    if args.tb_logdir is None:
        args.tb_logdir = f"runs/grafx_multi_{args.run_name}"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tb_logdir).mkdir(parents=True, exist_ok=True)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logging.info(f"run_name = {args.run_name}")
    logging.info(f"device   = {device}")

    # ---- Sessions / split ----
    staging_dir = Path(args.staging_dir).expanduser()
    label_store = LabelStore(staging_dir)
    sessions = _list_staged_sessions(staging_dir)
    if not sessions:
        raise SystemExit(f"no staged sessions found under {staging_dir}")
    n_labeled, n_total = label_store.coverage(sessions)
    logging.info(f"sessions: {n_total} staged, {n_labeled} with labels")

    # Deterministic split: sort by name, take last val_frac as val.
    n_val = max(1, int(round(args.val_frac * n_total)))
    rng = random.Random(args.seed)
    shuffled = sorted(sessions)
    rng.shuffle(shuffled)
    val_sessions = sorted(shuffled[:n_val])
    train_sessions = sorted(shuffled[n_val:])
    logging.info(f"split: train={len(train_sessions)} val={len(val_sessions)}")
    logging.info(f"val sessions: {val_sessions}")

    # ---- Data ----
    train_ds = MultiSongDataset(
        train_sessions, staging_dir, label_store,
        audio_len=args.audio_len, n_max=args.n_max,
        seed=args.seed, cache_size=args.cache_size,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, collate_fn=collate_multi,
        num_workers=args.num_workers,
    )

    # ---- Model ----
    console = GrafxMixingConsole(
        sample_rate=args.sample_rate, max_input_len=args.audio_len,
    ).to(device)
    encoder = MixEncoderGrafx(
        console, sample_rate=args.sample_rate, d_model=args.d_model,
        n_track_layers=args.n_track_layers, use_ref_mix=False, mert_dim=0,
    ).to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    logging.info(f"encoder params: {n_params/1e6:.1f}M")

    # Console-derived label schemas. Passed to attach_grafx_labels so the
    # zero-padded supervision tensors always exist (even when a batch has
    # no labeled examples), keeping the loss path total and avoiding a
    # KeyError on dict-key iteration.
    strip_schema = console.strip_param_shapes
    group_schema = console.group_param_shapes

    # ---- Losses ----
    recon_loss_fn = make_grafx_prune_recon_loss(
        sample_rate=args.sample_rate,
    ).to(device)

    # ---- Optimizer ----
    optimizer = AdamW(
        encoder.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )

    # ---- TB ----
    tb = SummaryWriter(log_dir=args.tb_logdir)
    logging.info(f"tb logdir: {args.tb_logdir}")
    logging.info(f"ckpt dir:  {args.out_dir}")
    logging.info(
        f"loss: w_param={args.w_param}  w_recon target={args.w_recon_target} "
        f"warmup={args.recon_warmup}  huber_delta=1.0"
    )

    # ---- Train loop ----
    t0 = time.time()
    encoder.train()
    best_val = float("inf")
    train_iter = iter(train_loader)

    for step in range(args.max_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = attach_grafx_labels(
            batch, label_store, n_max=args.n_max, max_groups=args.max_groups,
            strip_schema=strip_schema, group_schema=group_schema,
        )

        tracks = batch["tracks"].to(device, non_blocking=True)
        mix_ref = batch["mix"].to(device, non_blocking=True)
        track_mask = batch["track_mask"].to(device, non_blocking=True)
        group_idx = batch["group_assignments"].to(device, non_blocking=True)

        out = encoder(tracks, track_mask, group_idx, n_groups=args.max_groups)
        pred_mix = console(
            tracks, out["strip_params"], out["group_params"],
            out["group_assignments"], track_mask=track_mask,
            n_groups=args.max_groups,
        )

        L_param = grafx_param_huber_loss(
            out["strip_params"], out["group_params"],
            batch["label_strip_params"], batch["label_group_params"],
            batch["label_track_mask"].to(device),
            batch["n_groups_per_example"].to(device),
            batch["label_example_mask"].to(device),
            delta=1.0,
        )
        L_recon = recon_loss_fn(pred_mix, mix_ref)

        w_recon = w_recon_at(step, args.recon_warmup, args.w_recon_target)
        L_total = (
            args.w_param * L_param["L_param/total"]
            + w_recon * L_recon["match/full"]
        )

        optimizer.zero_grad(set_to_none=True)
        L_total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            encoder.parameters(), max_norm=args.grad_clip,
        )
        optimizer.step()

        if step % args.log_every == 0:
            tb.add_scalar("train/L_total",       L_total.item(),                  step)
            tb.add_scalar("train/L_param/strip", L_param["L_param/strip"].item(), step)
            tb.add_scalar("train/L_param/group", L_param["L_param/group"].item(), step)
            tb.add_scalar("train/L_param/total", L_param["L_param/total"].item(), step)
            tb.add_scalar("train/L_recon/full",  L_recon["match/full"].item(),    step)
            tb.add_scalar("train/L_recon/sum",   L_recon["match/sum"].item(),     step)
            tb.add_scalar("train/L_recon/diff",  L_recon["match/diff"].item(),    step)
            tb.add_scalar("train/grad_norm",     grad_norm.item(),                step)
            tb.add_scalar("train/w_recon",       w_recon,                         step)
            elapsed = time.time() - t0
            logging.info(
                f"step {step:6d}/{args.max_steps}  "
                f"L_tot={L_total.item():7.3f}  "
                f"L_p={L_param['L_param/total'].item():6.2f}  "
                f"L_r={L_recon['match/full'].item():5.2f}  "
                f"w_r={w_recon:.3f}  "
                f"grad={grad_norm.item():6.1f}  "
                f"({elapsed/(step+1):.2f}s/step)"
            )

        # Validation
        if step > 0 and step % args.val_every == 0:
            val_metrics = run_val(
                encoder, console, recon_loss_fn,
                val_sessions, staging_dir, label_store,
                args.audio_len, args.n_max, args.max_groups, device,
                strip_schema=strip_schema, group_schema=group_schema,
            )
            for k, v in val_metrics.items():
                tb.add_scalar(k, v, step)
            logging.info(
                f"  val[step {step}]: "
                f"L_param={val_metrics['val/L_param']:.3f} "
                f"(n={int(val_metrics['val/n_labeled'])})  "
                f"L_recon={val_metrics['val/L_recon']:.3f} "
                f"(n={int(val_metrics['val/n_total'])})"
            )
            # Best ckpt: prefer val L_param when available, else val L_recon
            score = val_metrics["val/L_param"]
            if score != score or val_metrics["val/n_labeled"] == 0:
                # NaN or no labeled val sessions
                score = val_metrics["val/L_recon"]
            if score < best_val:
                best_val = score
                torch.save({
                    "step": step,
                    "encoder_state_dict": encoder.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                }, Path(args.out_dir) / "encoder_best.pt")
                logging.info(f"  → new best val score = {best_val:.4f}")

        # Checkpoint
        if step > 0 and step % args.ckpt_every == 0:
            torch.save({
                "step": step,
                "encoder_state_dict": encoder.state_dict(),
                "args": vars(args),
            }, Path(args.out_dir) / f"encoder_step{step:06d}.pt")

    # Final ckpt + val
    final_val = run_val(
        encoder, console, recon_loss_fn,
        val_sessions, staging_dir, label_store,
        args.audio_len, args.n_max, args.max_groups, device,
    )
    for k, v in final_val.items():
        tb.add_scalar(k, v, args.max_steps)
    torch.save({
        "step": args.max_steps,
        "encoder_state_dict": encoder.state_dict(),
        "val_metrics": final_val,
        "args": vars(args),
    }, Path(args.out_dir) / f"encoder_step{args.max_steps:06d}.pt")

    tb.close()
    elapsed = time.time() - t0
    logging.info(
        f"\ntraining done. {args.max_steps} steps in {elapsed/60:.1f} min. "
        f"best val score = {best_val:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
