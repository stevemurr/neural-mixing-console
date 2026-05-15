"""Phase 4b — Grafx training driver, single-song overfit test.

Smallest viable experiment to validate the full Phase 1-4a pipeline:
  - MixEncoderGrafx  (phase 2) — predicts dict params from stems
  - GrafxMixingConsole (phase 1) — renders that dict to audio
  - grafx_param_huber_loss (phase 4a) — direct supervision on a single
    song's grafx-prune-optimized labels (phase 3)
  - MR-STFT mid/side recon loss (phase 4a) — vs the engineer ref mix

If the encoder can memorize a single song's labels, the pipeline is
fundamentally correct and we can move on to the full multi-song
trainer. If it cannot, something is broken structurally.

Run:
    uv run python training/train_grafx_overfit.py \\
        --song AMContra_HeartPeripheral \\
        --steps 800 --batch-size 2 --lr 1e-4

Outputs to `runs/grafx_overfit_<song>/` and `dmc-data/checkpoints/
grafx_overfit_<song>/`.
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


class SingleSongOverfitDataset(IterableDataset):
    """In-memory random-window dataset over ONE song.

    Loads the song's stems + ref mix + labels into RAM once; yields
    one (tracks, mix, meta) example per `__next__`, picking a random
    `T`-sample window each time. Infinite stream — caller stops when
    enough steps have been taken.
    """

    def __init__(
        self,
        song_dir: Path,
        label_store: LabelStore,
        audio_len: int,
        seed: int = 0,
    ):
        super().__init__()
        self.song_dir = song_dir
        self.T = audio_len
        self.rng = random.Random(seed)

        labels = label_store.get(song_dir.name)
        if labels is None:
            raise FileNotFoundError(
                f"no labels found for {song_dir.name!r} under {label_store.root}"
            )
        self.session = song_dir.name
        self.stem_filenames = list(labels["stem_filenames"])
        self.n_stems = len(self.stem_filenames)

        # Load all stems and the mix into memory at the session's sample rate.
        stems_data: list[torch.Tensor] = []
        for fname in self.stem_filenames:
            data, sr = sf.read(str(song_dir / "stems" / fname),
                               dtype="float32", always_2d=True)
            if data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)
            elif data.shape[1] > 2:
                data = data[:, :2]
            stems_data.append(torch.from_numpy(data.T.copy()))     # (2, T_total)
        self.sample_rate = sr
        # All stems share the song's duration; truncate to the shortest.
        T_total_stems = min(s.shape[-1] for s in stems_data)

        mix_data, mix_sr = sf.read(str(song_dir / "mix.wav"),
                                   dtype="float32", always_2d=True)
        assert mix_sr == sr, f"sr mismatch: stems={sr}, mix={mix_sr}"
        if mix_data.shape[1] == 1:
            mix_data = np.repeat(mix_data, 2, axis=1)
        mix_tensor = torch.from_numpy(mix_data.T.copy())            # (2, T_total)
        T_total = min(T_total_stems, mix_tensor.shape[-1])
        self.T_total = T_total
        self.stems = torch.stack([s[:, :T_total] for s in stems_data])   # (N, 2, T_total)
        self.mix = mix_tensor[:, :T_total]                                # (2, T_total)

        if T_total < self.T + 1:
            raise ValueError(
                f"song too short: {T_total} samples < window {self.T}"
            )

    def __iter__(self):
        while True:
            start = self.rng.randint(0, self.T_total - self.T - 1)
            tracks = self.stems[:, :, start:start + self.T]              # (N, 2, T)
            mix = self.mix[:, start:start + self.T]                       # (2, T)
            yield {
                "tracks": tracks,
                "mix": mix,
                "meta": {
                    "session": self.session,
                    "tracks": [{"filename": fn} for fn in self.stem_filenames],
                },
            }


def collate_overfit(batch: list[dict]) -> dict:
    """Stack per-example tensors into a batch dict shaped for the trainer."""
    tracks = torch.stack([ex["tracks"] for ex in batch])     # (B, N, 2, T)
    mix = torch.stack([ex["mix"] for ex in batch])            # (B, 2, T)
    metas = [ex["meta"] for ex in batch]
    B, N = tracks.shape[:2]
    return {
        "tracks": tracks,
        "mix": mix,
        "track_mask": torch.ones(B, N, dtype=torch.bool),
        "mert_embeddings": torch.zeros(B, N, 768),
        "meta": metas,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--song", default="AMContra_HeartPeripheral",
                    help="Session name under --staging-dir.")
    ap.add_argument("--staging-dir", default="/tmp/grafx-prune-data")
    ap.add_argument("--out-dir", default=None,
                    help="Default dmc-data/checkpoints/grafx_overfit_<song>")
    ap.add_argument("--tb-logdir", default=None,
                    help="Default runs/grafx_overfit_<song>")
    ap.add_argument("--device", default=None,
                    help="'cuda' or 'cpu'; default = cuda if available.")
    ap.add_argument("--sample-rate", type=int, default=30_000)
    ap.add_argument("--audio-len", type=int, default=90_000,
                    help="Samples per training window (default 3 s at 30 kHz).")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-track-layers", type=int, default=2)
    ap.add_argument("--max-groups", type=int, default=16,
                    help="Capacity for per-group bus param tensors per example.")
    ap.add_argument("--w-param", type=float, default=1.0,
                    help="Weight on Huber MSE on grafx-prune labels.")
    ap.add_argument("--w-recon", type=float, default=0.5,
                    help="Weight on MR-STFT mid/side audio recon vs engineer mix.")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    staging_dir = Path(args.staging_dir).expanduser()
    song_dir = staging_dir / args.song
    if args.out_dir is None:
        args.out_dir = f"dmc-data/checkpoints/grafx_overfit_{args.song}"
    if args.tb_logdir is None:
        args.tb_logdir = f"runs/grafx_overfit_{args.song}"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tb_logdir).mkdir(parents=True, exist_ok=True)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logging.info(f"device = {device}")

    # ---- Dataset ----
    label_store = LabelStore(staging_dir)
    dataset = SingleSongOverfitDataset(
        song_dir, label_store, audio_len=args.audio_len, seed=args.seed,
    )
    logging.info(f"song={args.song} stems={dataset.n_stems} duration="
                 f"{dataset.T_total / dataset.sample_rate:.1f}s sr={dataset.sample_rate}")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate_overfit,
        num_workers=0,  # in-memory dataset; no need to subprocess
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

    # ---- Losses ----
    recon_loss_fn = make_grafx_prune_recon_loss(sample_rate=args.sample_rate).to(device)

    # ---- Optimizer ----
    optimizer = AdamW(encoder.parameters(), lr=args.lr, weight_decay=1e-4,
                      betas=(0.9, 0.95))

    # ---- TB ----
    tb = SummaryWriter(log_dir=args.tb_logdir)
    logging.info(f"tb logdir: {args.tb_logdir}")
    logging.info(f"ckpt dir:  {args.out_dir}")
    logging.info(f"loss weights: w_param={args.w_param} w_recon={args.w_recon}")

    # ---- Train loop ----
    t0 = time.time()
    encoder.train()
    best_L = float("inf")
    for step, batch in enumerate(loader):
        if step >= args.steps:
            break

        # Attach grafx labels for this batch (one song, so always the same labels)
        batch = attach_grafx_labels(
            batch, label_store, n_max=dataset.n_stems, max_groups=args.max_groups,
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

        # Losses
        L_param = grafx_param_huber_loss(
            out["strip_params"], out["group_params"],
            batch["label_strip_params"], batch["label_group_params"],
            batch["label_track_mask"].to(device),
            batch["n_groups_per_example"].to(device),
            batch["label_example_mask"].to(device),
            delta=1.0,
        )
        L_recon = recon_loss_fn(pred_mix, mix_ref)
        L_total = args.w_param * L_param["L_param/total"] + args.w_recon * L_recon["match/full"]

        optimizer.zero_grad(set_to_none=True)
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=10.0)
        optimizer.step()

        # Log
        if step % args.log_every == 0:
            tb.add_scalar("train/L_total",       L_total.item(),               step)
            tb.add_scalar("train/L_param/strip", L_param["L_param/strip"].item(), step)
            tb.add_scalar("train/L_param/group", L_param["L_param/group"].item(), step)
            tb.add_scalar("train/L_param/total", L_param["L_param/total"].item(), step)
            tb.add_scalar("train/L_recon/full",  L_recon["match/full"].item(),  step)
            tb.add_scalar("train/L_recon/sum",   L_recon["match/sum"].item(),   step)
            tb.add_scalar("train/L_recon/diff",  L_recon["match/diff"].item(),  step)
            elapsed = time.time() - t0
            logging.info(
                f"step {step:5d}/{args.steps}  "
                f"L_total={L_total.item():7.4f}  "
                f"L_param={L_param['L_param/total'].item():6.3f}  "
                f"L_recon={L_recon['match/full'].item():6.3f}  "
                f"({elapsed/(step+1):.2f}s/step)"
            )

        # Ckpt
        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
            ckpt_path = Path(args.out_dir) / f"encoder_step{step+1:06d}.pt"
            torch.save({
                "step": step + 1,
                "encoder_state_dict": encoder.state_dict(),
                "L_total": L_total.item(),
                "args": vars(args),
            }, ckpt_path)
            if L_total.item() < best_L:
                best_L = L_total.item()
                torch.save({
                    "step": step + 1,
                    "encoder_state_dict": encoder.state_dict(),
                    "L_total": best_L,
                    "args": vars(args),
                }, Path(args.out_dir) / "encoder_best.pt")

    tb.close()
    elapsed = time.time() - t0
    print(f"\noverfit done. {args.steps} steps in {elapsed:.1f}s. best L_total = {best_L:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
