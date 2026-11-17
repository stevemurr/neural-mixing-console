"""Grafx-prune label loading + per-batch supervision tensor assembly.

The training-side counterpart to `scripts/label_grafx_prune.py`. For each
training example, the existing `collate_stage3` produces (tracks, mix, mert,
meta) bundles with per-stem filenames in `meta["tracks"][i]["filename"]`.
This module:

  1. Loads + caches the `labels_grafx_prune.pt` file for that example's
     session (one per session, ~400 KB).
  2. Maps each batch-track-index to the corresponding row in the labels by
     filename.
  3. Stitches everything into batch tensors shaped for
     `models.grafx_console.GrafxMixingConsole.forward`:

         label_strip_params: dict[proc → dict[param → (B, N_max, *shape)]]
         label_group_params: dict[proc → dict[param → (B, max_groups, *shape)]]
         group_assignments:  (B, N_max) long — which group each track routes to
         n_groups_per_example: (B,) long — used by the loss to mask group-bus
                                supervision beyond each example's group count
         label_track_mask:   (B, N_max) bool — True where a per-track supervision
                              target exists (filename matched a row in labels)
         label_example_mask: (B,) bool — True for examples that have any labels

The trainer then computes Huber MSE on (predicted_params, label_params)
masked by label_track_mask and label_example_mask, in parallel with the
audio-domain reconstruction loss against the engineer reference mix.

Design notes:
  - The `LabelStore` is process-local and cached. Loaded labels live on CPU.
  - Sessions WITHOUT labels return None — supervision is masked off for them,
    only the recon loss applies. Lets us train on the full corpus while
    only a subset has been labeled by the per-song optimizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


class LabelStore:
    """Per-session label cache. Loads `labels_grafx_prune.pt` lazily."""

    def __init__(self, labels_root: str | Path):
        self.root = Path(labels_root)
        self._cache: dict[str, Optional[dict]] = {}

    def get(self, session: str) -> Optional[dict]:
        """Returns the labels dict for `session`, or `None` if not labeled yet."""
        if session not in self._cache:
            path = self.root / session / "labels_grafx_prune.pt"
            if not path.is_file():
                self._cache[session] = None
            else:
                self._cache[session] = torch.load(
                    path, weights_only=False, map_location="cpu",
                )
        return self._cache[session]

    def __len__(self) -> int:
        return sum(1 for v in self._cache.values() if v is not None)

    def coverage(self, sessions: list[str]) -> tuple[int, int]:
        """Returns `(n_labeled, n_total)` after probing every session."""
        n_total = len(sessions)
        n_labeled = sum(1 for s in sessions if self.get(s) is not None)
        return n_labeled, n_total


def _schema_from_labels(labels: dict) -> tuple[dict, dict]:
    """Pull `{proc: {param: shape}}` for strip + group from a labels dict.

    Used as a fallback when the trainer doesn't pass an explicit schema.
    Once the trainer is committed against a console-derived schema, this
    is just a sanity-check.
    """
    strip_shapes = {
        proc: {p: tuple(t.shape[1:]) for p, t in pp.items()}
        for proc, pp in labels["strip_params"].items()
    }
    group_shapes = {
        proc: {p: tuple(t.shape[1:]) for p, t in pp.items()}
        for proc, pp in labels["group_params"].items()
    }
    return strip_shapes, group_shapes


def attach_grafx_labels(
    batch: dict,
    label_store: LabelStore,
    n_max: int,
    *,
    strip_schema: Optional[dict[str, dict[str, tuple]]] = None,
    group_schema: Optional[dict[str, dict[str, tuple]]] = None,
    max_groups: int = 16,
) -> dict:
    """Augment `batch` (output of `collate_stage3`) with grafx-prune supervision.

    Args:
        batch: dict with at least 'meta' (list of per-example metas) and
               'tracks' (B, N_max, 2, T). Existing collate output.
        label_store: per-session label cache.
        n_max: same as the collate's n_max — labels are stitched into
               (B, n_max, ...) tensors padded with zeros for missing slots.
        strip_schema / group_schema: shape dicts to use for padding-zero
               targets. If None, inferred from the first labeled example
               in the batch. (For training loop you'd derive these from the
               GrafxMixingConsole and pass them in for stability.)
        max_groups: capacity of the group-bus supervision tensors per
               example. Songs with fewer groups get padded with zero
               targets in the unused slots (masked off downstream by
               `n_groups_per_example`).

    Returns the same `batch` dict, extended in-place with:
        label_strip_params, label_group_params,
        group_assignments (B, N_max),
        n_groups_per_example (B,),
        label_track_mask (B, N_max),
        label_example_mask (B,)
    """
    B = len(batch["meta"])

    # Pull the per-example label dicts (None for unlabeled sessions)
    per_example_labels: list[Optional[dict]] = []
    for meta in batch["meta"]:
        session = meta.get("session", None)
        per_example_labels.append(label_store.get(session) if session else None)

    # Derive schemas if not given. Use any labeled example.
    if strip_schema is None or group_schema is None:
        first_labeled = next((lb for lb in per_example_labels if lb is not None), None)
        if first_labeled is None:
            # No examples in the batch have labels — return batch with
            # empty supervision; the loss should fall back to recon-only.
            batch["label_strip_params"] = {}
            batch["label_group_params"] = {}
            batch["group_assignments"] = torch.zeros(B, n_max, dtype=torch.long)
            batch["n_groups_per_example"] = torch.zeros(B, dtype=torch.long)
            batch["label_track_mask"] = torch.zeros(B, n_max, dtype=torch.bool)
            batch["label_example_mask"] = torch.zeros(B, dtype=torch.bool)
            return batch
        s_schema, g_schema = _schema_from_labels(first_labeled)
        strip_schema = strip_schema or s_schema
        group_schema = group_schema or g_schema

    # Pre-allocate the batch label tensors. Zero-fill; we'll fill rows
    # where a filename match exists.
    label_strip = {
        proc: {p: torch.zeros(B, n_max, *shape) for p, shape in shapes.items()}
        for proc, shapes in strip_schema.items()
    }
    label_group = {
        proc: {p: torch.zeros(B, max_groups, *shape) for p, shape in shapes.items()}
        for proc, shapes in group_schema.items()
    }
    group_assignments = torch.zeros(B, n_max, dtype=torch.long)
    n_groups_per_example = torch.zeros(B, dtype=torch.long)
    label_track_mask = torch.zeros(B, n_max, dtype=torch.bool)
    label_example_mask = torch.zeros(B, dtype=torch.bool)

    for bi, (meta, labels) in enumerate(zip(batch["meta"], per_example_labels)):
        if labels is None:
            continue
        label_example_mask[bi] = True
        n_g = len(labels["groups"])
        n_groups_per_example[bi] = n_g

        # Filename → label row index
        fname_to_row = {fn: i for i, fn in enumerate(labels["stem_filenames"])}

        # Group params: copy first n_g rows; rest stay zero
        for proc, pp in labels["group_params"].items():
            for p, t in pp.items():
                label_group[proc][p][bi, :n_g] = t

        # Per-track strip params + group assignments, indexed by filename
        track_metas = meta.get("tracks", [])
        for ti, tm in enumerate(track_metas):
            if ti >= n_max:
                break
            fname = tm.get("filename", "")
            row = fname_to_row.get(fname)
            if row is None:
                continue  # unmatched track — stays masked off
            label_track_mask[bi, ti] = True
            group_assignments[bi, ti] = int(labels["group_assignments"][row])
            for proc, pp in labels["strip_params"].items():
                for p, t in pp.items():
                    label_strip[proc][p][bi, ti] = t[row]

    batch["label_strip_params"] = label_strip
    batch["label_group_params"] = label_group
    batch["group_assignments"] = group_assignments
    batch["n_groups_per_example"] = n_groups_per_example
    batch["label_track_mask"] = label_track_mask
    batch["label_example_mask"] = label_example_mask
    return batch


def compute_label_whiten_weights(
    label_store: LabelStore,
    sessions: list[str],
    *,
    eps: float = 1e-3,
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-(proc, param) reciprocal-std weights for whitened L_param.

    Walks `sessions`, pulls each session's labels, concatenates all values
    of each (proc, param) across tracks/groups/songs, computes a single
    std per (proc, param). Returns weights as `1 / max(std, eps)` shaped:

        {"strip": {proc: {param: w}}, "group": {proc: {param: w}}}

    The trainer feeds these into `grafx_param_huber_loss` so per-param
    contributions to the loss are unit-variance regardless of natural
    scale (EQ's 1024-dim log_magnitude no longer dominates the comp's 4
    scalars; strip and group can be balanced post-whitening).
    """
    strip_acc: dict[str, dict[str, list[torch.Tensor]]] = {}
    group_acc: dict[str, dict[str, list[torch.Tensor]]] = {}
    for s in sessions:
        labels = label_store.get(s)
        if labels is None:
            continue
        for proc, pp in labels["strip_params"].items():
            for p, t in pp.items():
                strip_acc.setdefault(proc, {}).setdefault(p, []).append(t.flatten())
        for proc, pp in labels["group_params"].items():
            for p, t in pp.items():
                group_acc.setdefault(proc, {}).setdefault(p, []).append(t.flatten())

    def _stds(acc):
        out: dict[str, dict[str, float]] = {}
        for proc, pp in acc.items():
            out[proc] = {}
            for p, chunks in pp.items():
                all_vals = torch.cat(chunks)
                std = float(all_vals.std().item())
                out[proc][p] = 1.0 / max(std, eps)
        return out

    return {"strip": _stds(strip_acc), "group": _stds(group_acc)}


__all__ = [
    "LabelStore",
    "attach_grafx_labels",
    "compute_label_whiten_weights",
]
