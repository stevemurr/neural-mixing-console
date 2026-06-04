"""V13 Stage 2 in-memory dataset — loads precomputed (C, target) pairs.

Examples are produced by `v13/scripts/precompute_v13_dataset.py`; this module
just iterates them and splits by session (no within-session leakage).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch.utils.data import Dataset


def session_split(
    sessions: list[str], val_frac: float = 0.1, seed: int = 0,
) -> tuple[set[str], set[str]]:
    """Deterministic session-level split using stable hashing on (seed,name).

    The MD5-based bucketing means adding new sessions later doesn't reshuffle
    the existing split — each session keeps the same fold across runs.
    """
    train, val = set(), set()
    for name in sorted(set(sessions)):
        h = hashlib.md5(f"{seed}::{name}".encode()).hexdigest()
        # First 8 hex digits → 32-bit int / 2^32 → uniform in [0, 1)
        x = int(h[:8], 16) / (1 << 32)
        (val if x < val_frac else train).add(name)
    return train, val


class V13Dataset(Dataset):
    """Loads precomputed examples; optionally filters to a session subset."""

    def __init__(
        self,
        examples_path: str | Path,
        keep_sessions: set[str] | None = None,
    ):
        path = Path(examples_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"v13 precomputed dataset not found: {path}. "
                f"Run `uv run python v13/scripts/precompute_v13_dataset.py` first."
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


def collate_v13(batch: list[dict]) -> dict:
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


__all__ = ["V13Dataset", "collate_v13", "session_split"]
