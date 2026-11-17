"""Run EQ recipes from a checkout, retaining their inputs and outcomes."""

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tomllib
from uuid import uuid4

from nmc.process import run_logged
from nmc.records import fingerprint, write_json


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def resolve(recipe_path):
    recipe = tomllib.loads(recipe_path.read_text())
    task = recipe.get("task")
    optional = {"ram_floor_gib"}
    if task == "inspect":
        paths = ("manifest",)
        required = {"window_seconds"}
        optional.add("input_kind")
    elif task == "medleydb-benchmark":
        paths = ("manifest", "checkpoint")
        required = {"window_seconds", "windows_per_song", "input_kind"}
    elif task == "reproduce":
        paths = ("split", "cache", "checkpoint")
        required = {"cache_sha256", "checkpoint_sha256", "expected_rmse_db"}
    else:
        raise ValueError(f"unknown recipe task {task!r}")
    required.update({"task", "name", "val_frac", "seed", *paths})
    missing = required - recipe.keys()
    extra = recipe.keys() - required - optional
    if missing or extra:
        raise ValueError(f"missing recipe keys {sorted(missing)}, unknown keys {sorted(extra)}")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", recipe["name"]):
        raise ValueError("invalid recipe name")
    if not 0 < recipe["val_frac"] < 1 or type(recipe["seed"]) is not int:
        raise ValueError("val_frac must be between 0 and 1 and seed must be an integer")
    recipe.setdefault("ram_floor_gib", 0)
    if not math.isfinite(recipe["ram_floor_gib"]) or recipe["ram_floor_gib"] < 0:
        raise ValueError("invalid ram_floor_gib")
    if "window_seconds" in recipe and (
        not math.isfinite(recipe["window_seconds"]) or recipe["window_seconds"] <= 0
    ):
        raise ValueError("window_seconds must be positive and finite")
    if "input_kind" in recipe and recipe["input_kind"] not in ("raw", "stems"):
        raise ValueError("input_kind must be raw or stems")
    if "windows_per_song" in recipe and (
        type(recipe["windows_per_song"]) is not int or recipe["windows_per_song"] < 1
    ):
        raise ValueError("windows_per_song must be a positive integer")
    if task == "reproduce" and set(recipe["expected_rmse_db"]) != {
        "zero",
        "training_mean",
        "learned",
    }:
        raise ValueError("expected_rmse_db requires zero, training_mean and learned baselines")
    for key in paths:
        recipe[key] = str(Path(recipe[key]).resolve())
    return recipe, paths


def execute(recipe, output):
    if recipe["task"] == "inspect":
        from nmc.data.medleydb import inspect_windows

        inspect_windows(recipe, output)
    elif recipe["task"] == "medleydb-benchmark":
        from nmc.evaluation.medleydb import benchmark

        benchmark(recipe, output)
    else:
        from nmc.evaluation.reproduce import inputs, reproduce

        if output:
            reproduce(recipe, output / "metrics.json")
        else:
            examples, train, val, _ = inputs(recipe)
            print(
                f"Verified {len(examples)} cached windows, {len(train)} train and {len(val)} validation songs"
            )


def run(recipe_path, check=False, runs_dir=Path("runs")):
    recipe, paths = resolve(recipe_path)
    if check:
        execute(recipe, None)
        return 0
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:10]
    folder = (runs_dir / recipe["name"] / run_id).resolve()
    folder.mkdir(parents=True)
    record = {
        "id": run_id,
        "recipe": recipe["name"],
        "started": timestamp(),
        "status": "running",
        "source_recipe": str(recipe_path.resolve()),
    }
    manifest = folder / "run.json"
    write_json(manifest, record)
    write_json(folder / "recipe.json", recipe)
    print(f"Run directory: {folder}", flush=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")

    prior_handler = signal.signal(signal.SIGTERM, interrupted)
    try:

        def git(*args):
            return subprocess.check_output(["git", *args], text=True).strip()

        record.update(revision=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")))
        # Snapshot small inputs so a later data preparation cannot change this run.
        resolved = dict(recipe)
        for key in paths:
            if key in {"manifest", "split"}:
                target = folder / f"{key}.json"
                target.write_bytes(Path(recipe[key]).read_bytes())
                resolved[key] = str(target)
        write_json(folder / "recipe.json", resolved)
        record["inputs"] = {
            k: {"source": recipe[k], "sha256": fingerprint(resolved[k])} for k in paths
        }
        write_json(manifest, record)
        command = [sys.executable, "-u", "-m", "nmc.runner", str(folder)]
        code = run_logged(
            command, Path.cwd(), os.environ.copy(), folder / "run.log", recipe["ram_floor_gib"]
        )
        record.update(exit_code=code, status="completed" if code == 0 else "failed")
        if code == 0:
            metrics = json.loads((folder / "metrics.json").read_text())
            record["split_sha256"] = split_digest(metrics)
            record["rmse_db"] = metrics.get("rmse_db")
        return code if code >= 0 else 128 - code
    except BaseException as exc:
        record.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=str(exc) or type(exc).__name__,
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, prior_handler)
        record["finished"] = timestamp()
        write_json(manifest, record)


def split_digest(metrics):
    split = {k: metrics[k] for k in ("train", "validation")}
    return hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()


def list_runs(root):
    for path in sorted(root.glob("*/*/run.json")):
        run = json.loads(path.read_text())
        print(f"{run['recipe']}  {run['id']}  {run['status']}  {path.parent}")


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    execute(json.loads((directory / "recipe.json").read_text()), directory)
