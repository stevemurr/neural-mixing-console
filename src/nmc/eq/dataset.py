"""Read cached target curves and per-track contribution maps."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class TargetCurveDataset(Dataset):
    """Loads precomputed examples; optionally filters to a session subset."""

    def __init__(
        self,
        examples_path: str | Path,
        keep_sessions: set[str] | None = None,
    ):
        path = Path(examples_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"EQ dataset not found: {path}. "
                f"Run `uv run python -m nmc.eq.precompute` first."
            )
        raw = torch.load(path, weights_only=False, map_location="cpu")
        if keep_sessions is not None:
            self.examples = [e for e in raw if e["session"] in keep_sessions]
        else:
            self.examples = list(raw)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        e = self.examples[idx]
        out = {
            "C":               e["C"],                # (n_max, n_bins) float32
            "target_delta_db": e["target_delta_db"],  # (n_bins,)      float32
            "track_mask":      e["track_mask"],       # (n_max,)       bool
        }
        # Stage 3 fields — present only when the dataset was precomputed
        # with pan targets (newer .pt files); the EQ trainer ignores these
        # and the Pan trainer ignores `target_delta_db`.
        if "pan_target" in e:
            out["pan_target"] = e["pan_target"]       # (n_max,)       float32
            out["is_stereo"]  = e["is_stereo"]        # (n_max,)       bool
        if "group_idx" in e:
            out["group_idx"]  = e["group_idx"]        # (n_max,)       long
        return out

    def sessions_covered(self) -> set[str]:
        return {e["session"] for e in self.examples}


def collate_examples(batch: list[dict]) -> dict:
    out = {
        "C":               torch.stack([b["C"] for b in batch]),
        "target_delta_db": torch.stack([b["target_delta_db"] for b in batch]),
        "track_mask":      torch.stack([b["track_mask"] for b in batch]),
    }
    if "pan_target" in batch[0]:
        out["pan_target"] = torch.stack([b["pan_target"] for b in batch])
        out["is_stereo"]  = torch.stack([b["is_stereo"] for b in batch])
    if "group_idx" in batch[0]:
        out["group_idx"]  = torch.stack([b["group_idx"] for b in batch])
    return out
