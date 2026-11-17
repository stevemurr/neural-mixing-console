"""Replay the frozen v13 feature-cache experiment from the repository root."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tomllib

import torch

from v13.models.target_curve_net import TargetCurveNet
from v13.training.data_v13 import session_split


def fingerprint(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def inputs(recipe):
    for key in ("cache", "checkpoint"):
        if fingerprint(recipe[key]) != recipe[key + "_sha256"]:
            raise ValueError(f"{key} fingerprint mismatch: {recipe[key]}")
    examples = torch.load(recipe["cache"], map_location="cpu", weights_only=True)
    checkpoint = torch.load(recipe["checkpoint"], map_location="cpu", weights_only=True)
    train, val = session_split([e["session"] for e in examples], recipe["val_frac"], recipe["seed"])
    if not train or not val:
        raise ValueError("Both training and validation songs are required")
    expected = json.loads(Path(recipe["split"]).read_text())
    if expected != {"train": sorted(train), "validation": sorted(val)}:
        raise ValueError("Historical split membership mismatch")
    keys = ("n_bins", "d_model", "n_layers", "n_heads", "ffn_mult", "dropout", "max_delta_db")
    net = TargetCurveNet(**{k: checkpoint["args"][k] for k in keys}).eval()
    net.load_state_dict(checkpoint["model"])
    return examples, train, val, net


@torch.inference_mode()
def reproduce(recipe, output=None):
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    examples, train, val, net = inputs(recipe)
    mean = torch.stack([e["target_delta_db"] for e in examples if e["session"] in train]).mean(0)
    validation = [e for e in examples if e["session"] in val]
    rows = []
    for start in range(0, len(validation), 32):
        batch = validation[start:start + 32]
        predicted = net(torch.stack([e["C"] for e in batch]),
                        torch.stack([e["track_mask"] for e in batch]))
        for e, curve in zip(batch, predicted):
            target = e["target_delta_db"]
            errors = {"zero": target, "training_mean": mean - target, "learned": curve - target}
            rows.append({"song": e["session"], "start_sample": int(e["window_start"]),
                         "mse_db2": {k: float(x.square().mean()) for k, x in errors.items()}})
    def scores(records):
        return {k: math.sqrt(sum(r["mse_db2"][k] for r in records) / len(records))
                for k in ("zero", "training_mean", "learned")}
    result = {"rmse_db": scores(rows), "train": sorted(train), "validation": sorted(val),
              "per_song_rmse_db": {s: scores([r for r in rows if r["song"] == s]) for s in sorted(val)},
              "windows": rows}
    if output:
        Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(train)} training songs, {len(val)} validation songs, {len(rows)} validation windows")
    for k, value in result["rmse_db"].items():
        print(f"{k:>14}: {value:.6f} dB")
        if not math.isclose(value, recipe["expected_rmse_db"][k], abs_tol=0.0005):
            raise ValueError(f"Historical RMSE mismatch for {k}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reproduce(tomllib.loads(args.recipe.read_text()), args.output)


if __name__ == "__main__":
    main()
