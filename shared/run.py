#!/usr/bin/env python
"""Run a training experiment from a TOML config.

    uv run scripts/run.py experiments/<name>.toml

Reads the config, assembles the `python training/train_stage3.py --flag value …`
command, sets PYTORCH_CUDA_ALLOC_CONF, starts a RAM watchdog (SIGTERMs the
trainer if available memory drops below a floor — on unified-memory hardware an
unconstrained CUDA OOM can wedge the GPU driver and reboot the box), and tees
the trainer's output to runs/<name>.log.

Config layout
-------------
    [run]
      name                       used to derive --out-dir / --tb-logdir / log path
      trainer                    path to the trainer script relative to repo root
                                 (default training/train_stage3.py). Set to
                                 training/train_grafx_distill.py etc. when running
                                 a non-stage3 trainer.

    [hardware]
      ram_floor_gb               watchdog floor in GiB (0 / absent disables it)
      pytorch_cuda_alloc_conf    value for the $PYTORCH_CUDA_ALLOC_CONF env var

    [train]
      <flag> = <value>           every key maps 1:1 to a CLI flag on the chosen
                                 trainer (key foo_bar -> --foo-bar);
                                 bool true -> bare flag; bool false -> omitted;
                                 list -> repeated values.

CLI flags
---------
    --print-cmd     print the assembled command and exit (don't run)
    --no-watchdog   disable the RAM watchdog
    --python PATH   python interpreter to use (default: this one)
"""
from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# This historical launcher also supports a system Python with --python pointing
# to a separate training environment. Its process helper is standard-library only.
sys.path.insert(0, str(REPO_ROOT / "src"))
from nmc.process import run_logged  # noqa: E402


def _build_cmd(cfg: dict, python: str) -> tuple[list[str], str, Path]:
    """Return (cmd, run_name, out_dir)."""
    run_cfg = cfg.get("run", {})
    name = run_cfg.get("name", "run")
    trainer_rel = run_cfg.get("trainer", "diff_console/training/train_stage3.py")
    train = dict(cfg.get("train", {}))
    train.setdefault("out_dir", f"dmc-data/checkpoints/{name}")
    train.setdefault("tb_logdir", f"runs/{name}")

    cmd = [python, str(REPO_ROOT / trainer_rel)]
    for key, val in train.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        elif isinstance(val, (list, tuple)):
            cmd.append(flag)
            cmd += [str(x) for x in val]
        else:
            cmd += [flag, str(val)]
    return cmd, name, Path(train["out_dir"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a training experiment from a TOML config.")
    ap.add_argument("config", type=Path, help="path to an experiments/*.toml config")
    ap.add_argument("--print-cmd", action="store_true", help="print the assembled command and exit")
    ap.add_argument("--no-watchdog", action="store_true", help="disable the RAM watchdog")
    ap.add_argument("--python", default=sys.executable, help="python interpreter (default: this one)")
    args = ap.parse_args()

    cfg = tomllib.loads(args.config.read_text())
    cmd, name, out_dir = _build_cmd(cfg, args.python)

    if args.print_cmd:
        print(" ".join(cmd))
        return 0

    env = dict(os.environ)
    hw = cfg.get("hardware", {})
    alloc_conf = hw.get("pytorch_cuda_alloc_conf")
    if alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = str(alloc_conf)
    ram_floor = float(hw.get("ram_floor_gb", 0) or 0)
    watchdog_on = ram_floor > 0 and not args.no_watchdog

    (REPO_ROOT / "runs").mkdir(exist_ok=True)
    out_dir_abs = (REPO_ROOT / out_dir) if not out_dir.is_absolute() else out_dir
    out_dir_abs.mkdir(parents=True, exist_ok=True)
    log_path = REPO_ROOT / "runs" / f"{name}.log"

    print(f"[run] experiment: {name}")
    print(f"[run] command:    {' '.join(cmd)}")
    print(f"[run] log:        {log_path}")
    if alloc_conf:
        print(f"[run] env:        PYTORCH_CUDA_ALLOC_CONF={alloc_conf}")
    if watchdog_on:
        print(f"[run] watchdog:   SIGTERM trainer if MemAvailable < {ram_floor:g} GiB")

    return run_logged(cmd, REPO_ROOT, env, log_path, ram_floor if watchdog_on else 0)


if __name__ == "__main__":
    raise SystemExit(main())
