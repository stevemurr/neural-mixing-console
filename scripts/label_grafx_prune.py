"""Run grafx-prune on one Cambridge session and save the optimized params
as supervision labels matching `models.grafx_console.GrafxMixingConsole`'s
schema.

Pipeline per session:
    1. Assume the session is already staged under
       `<staging_dir>/<song>/` with:
          - `stems/<stem_filenames>.wav` at 30 kHz
          - `mix.wav` at 30 kHz
          - `correspondence.yaml` mapping group → list of stem files
          - `alignment.pickle` with the mix/rough_mix offset
       (Use `scripts/stage_cambridge_for_grafx_prune.py` to prepare these.)
    2. Drive grafx-prune's training in-process (skip the CLI; we want
       direct access to the solver state).
    3. After training, walk grafx-prune's flat `graph_parameters` tensor
       per processor type and split it into the first N_stems rows
       (strip) and last N_groups rows (group bus).
    4. Save as a `.pt` next to the session under
       `<staging_dir>/<song>/labels_grafx_prune.pt` with the schema:
          {
            "session": str, "sample_rate": int,
            "stem_filenames": list[str],
            "groups": list[str],
            "group_assignments": (N_stems,) long,
            "strip_params": {proc: {param: (N_stems, *shape)}},
            "group_params": {proc: {param: (N_groups, *shape)}},
            "training_meta": {...},
          }

The output dicts plug directly into `GrafxMixingConsole.forward(...)` and
are the supervision target Phase 4's training loop will Huber-MSE against.

Default config: `config=mixing_console_full` (no pruning) for the smoke
test. For actual labeling runs use `config=prune_hybrid_1e_2` for the
full 24-epoch / 12k-step recipe.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import torch
import yaml

# Grafx-prune is a sibling repo (not pip-installed); fold its code into the path.
GRAFX_PRUNE_CODE_DIR = Path("/home/murr/Code/grafx-prune/code")
sys.path.insert(0, str(GRAFX_PRUNE_CODE_DIR))


def split_grafx_params(
    graph_parameters: dict,
    processors: Sequence[str],
    n_stems: int,
    n_groups: int,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]]]:
    """Split grafx-prune's flat per-processor param tensors into strip + group.

    `construct_mixing_console` adds nodes stem-first then group-first, so for
    every processor type the param tensor has shape `(N_stems + N_groups, *param_shape)`
    with the first `N_stems` rows being per-track strip params and the last
    `N_groups` being per-group-bus params.
    """
    strip = {}
    group = {}
    for proc_name in processors:
        proc_params = graph_parameters[proc_name]
        strip[proc_name] = {}
        group[proc_name] = {}
        for param_name, tensor in proc_params.items():
            assert tensor.shape[0] == n_stems + n_groups, (
                f"{proc_name}.{param_name}: expected first dim "
                f"{n_stems + n_groups} (stems+groups), got {tensor.shape[0]}"
            )
            strip[proc_name][param_name] = tensor[:n_stems].detach().cpu().clone()
            group[proc_name][param_name] = tensor[n_stems:].detach().cpu().clone()
    return strip, group


def load_correspondence_and_groups(song_dir: Path) -> tuple[list[str], list[str], list[int]]:
    """Read correspondence.yaml and return (stem_filenames, groups, group_assignments_per_stem).

    The order matches grafx-prune's `matched_dry_dirs` enumeration:
    flatten `correspondence.matched.values()` in dict insertion order.
    """
    corr = yaml.safe_load(open(song_dir / "correspondence.yaml"))
    groups = list(corr.keys())
    stem_filenames: list[str] = []
    group_assignments: list[int] = []
    for gi, group_name in enumerate(groups):
        for fname in corr[group_name]:
            stem_filenames.append(fname)
            group_assignments.append(gi)
    return stem_filenames, groups, group_assignments


def run_labeler(
    song: str,
    staging_dir: Path,
    config_name: str = "mixing_console_full",
    total_epochs: int = 6,
    steps_per_epoch: int = 100,
    wandb: bool = False,
    base_dir: Path = Path("/tmp/grafx-prune-logs"),
) -> dict:
    """Run grafx-prune training in-process and return the labels dict."""
    # Imports happen after sys.path is set up so they resolve to the sibling repo.
    import pytorch_lightning as pl
    from omegaconf import OmegaConf
    from solver import MusicMixingConsoleSolver
    from data.datamodule import SingleTrackOverfitDataModule
    from pytorch_lightning.loggers import CSVLogger

    song_dir = staging_dir / song
    if not (song_dir / "correspondence.yaml").is_file():
        raise FileNotFoundError(f"Session not staged at {song_dir}; run stage script first")

    # Build the OmegaConf args dict exactly as grafx-prune's train.py would,
    # but skipping the CLI plumbing.
    script_path = GRAFX_PRUNE_CODE_DIR / "train.py"
    base = OmegaConf.load(GRAFX_PRUNE_CODE_DIR / "configs/base.yaml")
    cfg = OmegaConf.load(GRAFX_PRUNE_CODE_DIR / f"configs/{config_name}.yaml")
    args = OmegaConf.merge(base, cfg)
    args.dataset = "mixing_secrets"
    args.song = song
    args.base_dir = str(base_dir)
    args.total_epochs = total_epochs
    args.steps_per_epoch = steps_per_epoch
    args.wandb = wandb
    args.debug = False
    args.name = f"label_{song}"
    args.save_dir = str(base_dir / args.name)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    args.id = config_name

    logging.info(f"running grafx-prune on '{song}' "
                 f"({total_epochs} epochs × {steps_per_epoch} steps)")

    # Override the staging base dir to ours
    import data.mixing_secrets.load as ms_load
    ms_load.BASE_DIR = str(staging_dir)

    # Read the staged session's structure (stems + groups) — we need this to
    # split grafx-prune's flat param tensors after training.
    stem_filenames, groups, group_assignments = load_correspondence_and_groups(song_dir)
    n_stems = len(stem_filenames)
    n_groups = len(groups)
    logging.info(f"  {n_stems} stems → {n_groups} groups: {groups}")

    # Build and train
    args_cont = OmegaConf.to_container(args)
    solver = MusicMixingConsoleSolver(args_cont)
    datamodule = SingleTrackOverfitDataModule(args_cont)

    max_steps = total_epochs * steps_per_epoch
    trainer = pl.Trainer(
        logger=CSVLogger(save_dir=args.save_dir),
        enable_checkpointing=False,
        default_root_dir=args.base_dir,
        max_steps=max_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        strategy="auto",
        fast_dev_run=False,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    t0 = time.time()
    trainer.fit(solver, datamodule)
    elapsed = time.time() - t0
    logging.info(f"  training done in {elapsed:.1f}s")

    # Extract optimized params + split strip vs group
    strip_params, group_params = split_grafx_params(
        solver.graph_parameters, args.processors, n_stems, n_groups,
    )

    # Final test loss for bookkeeping
    final_loss = None
    try:
        test_metrics = trainer.test(solver, datamodule, verbose=False)
        if test_metrics:
            final_loss = test_metrics[0].get("match/full", None)
    except Exception:
        pass

    labels = {
        "session": song,
        "sample_rate": args.sr,
        "stem_filenames": stem_filenames,
        "groups": groups,
        "group_assignments": torch.tensor(group_assignments, dtype=torch.long),
        "strip_params": strip_params,
        "group_params": group_params,
        "training_meta": {
            "config_name": config_name,
            "total_epochs": total_epochs,
            "steps_per_epoch": steps_per_epoch,
            "wall_seconds": elapsed,
            "final_test_loss": final_loss,
            "processors": list(args.processors),
        },
    }
    return labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--song", required=True,
                    help="Session name as it appears under --staging-dir")
    ap.add_argument("--staging-dir", default="dmc-data/grafx-prune-data",
                    help="Directory containing <song>/{stems/, mix.wav, "
                         "correspondence.yaml, alignment.pickle}, all 30 kHz")
    ap.add_argument("--config", default="mixing_console_full",
                    help="grafx-prune config name (configs/*.yaml). "
                         "'mixing_console_full' for no pruning, "
                         "'prune_hybrid_1e_2' for the full 24-epoch recipe.")
    ap.add_argument("--total-epochs", type=int, default=6,
                    help="Training epochs. Defaults to 6 for a smoke test; "
                         "use 24 for the full grafx-prune recipe.")
    ap.add_argument("--steps-per-epoch", type=int, default=100,
                    help="Defaults to 100; use 500 for the full recipe.")
    ap.add_argument("--out", default=None,
                    help="Output .pt path. Defaults to "
                         "<staging-dir>/<song>/labels_grafx_prune.pt")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    staging_dir = Path(args.staging_dir).expanduser()
    labels = run_labeler(
        song=args.song,
        staging_dir=staging_dir,
        config_name=args.config,
        total_epochs=args.total_epochs,
        steps_per_epoch=args.steps_per_epoch,
    )

    out_path = Path(args.out) if args.out else (staging_dir / args.song / "labels_grafx_prune.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(labels, out_path)

    # Summary
    print(f"\nsaved labels → {out_path}")
    print(f"  session:           {labels['session']}")
    print(f"  stems:             {len(labels['stem_filenames'])}")
    print(f"  groups:            {len(labels['groups'])} {labels['groups']}")
    print(f"  strip_params keys: {list(labels['strip_params'].keys())}")
    print(f"  training wall:     {labels['training_meta']['wall_seconds']:.1f}s")
    if labels["training_meta"]["final_test_loss"] is not None:
        print(f"  final test loss:   {labels['training_meta']['final_test_loss']:.4f}")
    total_numbers = sum(
        t.numel()
        for level in ("strip_params", "group_params")
        for proc_dict in labels[level].values()
        for t in proc_dict.values()
    )
    print(f"  total param count: {total_numbers:,}")
    print(f"  file size:         {out_path.stat().st_size/1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
