"""Synthetic-supervision data pipeline for MinimalConsole training (Phase 2).

Generates training examples by:
  1. Loading real stems from a staged Cambridge session (alignment-aware
     loader from `train_grafx_multi._load_session`).
  2. Sampling random params in normalized [0,1] space (uniform — aesthetic
     priors emerge from the per-(proc, param) RANGES definitions, not
     from the sampling distribution shape).
  3. Yielding `(stems, sampled_norm_params)`.

The trainer:
  - Denormalizes sampled params via `denormalize_params_dict` and renders
    the synth_mix on GPU via the same MinimalConsole the encoder is being
    trained against — giving self-consistent triplets.
  - The encoder predicts logits → sigmoid → [0,1] → denormalize before
    feeding to console (student render).
  - `L_param` compares encoder's [0,1] predictions to sampled norm params
    directly — no whitening needed, no engineering-unit scale issues.

See `notes/minimal_console_design_2026-05.md` and
`models/minimal_param_ranges.py` for the design.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset

from models.minimal_param_ranges import RANGES, denormalize_params_dict
from training.data_grafx import LabelStore
from training.train_grafx_multi import _load_session, _example_from_session


# ---------- Sampler in normalized [0,1] space ----------

class ParamSampler:
    """Samples MinimalConsole params uniformly in [0,1] normalized space.

    Per (proc, param) shape is inferred from the console's schema. The
    actual engineering-unit values come from `denormalize_params_dict`
    using RANGES — see `models/minimal_param_ranges.py`.

    Why uniform in [0,1] (instead of Beta or biased): the [0,1] space's
    "midpoint" 0.5 corresponds to:
      - log-uniform params: geometric mean of (lo, hi) — perceptually neutral
      - linear params: arithmetic midpoint — engineering-neutral
    So uniform [0,1] naturally explores the full valid range without
    pathological combinations. Aesthetic biases (centered on neutral)
    can be added later by sampling from non-uniform priors in [0,1].
    """

    def __init__(
        self,
        strip_schema: dict[str, dict[str, tuple[int, ...]]],
        group_schema: dict[str, dict[str, tuple[int, ...]]],
        seed: int = 0,
    ):
        self.strip_schema = strip_schema
        self.group_schema = group_schema
        self.rng = torch.Generator().manual_seed(seed)

    def sample(
        self, level: str, B: int, N_or_G: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Sample one level's params in [0,1] space.

        Returns `{proc: {name: (B, N_or_G, *param_shape) tensor in [0,1]}}`.
        Validates against RANGES — any (proc, param) in the schema must
        have a range defined.
        """
        schema = self.strip_schema if level == "strip" else self.group_schema
        out: dict[str, dict[str, torch.Tensor]] = {}
        for proc, pp in schema.items():
            out[proc] = {}
            for name, shape in pp.items():
                if proc not in RANGES or name not in RANGES[proc]:
                    raise KeyError(
                        f"no range defined for {proc}.{name}; "
                        f"add to RANGES in models/minimal_param_ranges.py"
                    )
                out[proc][name] = torch.rand(
                    (B, N_or_G, *shape), generator=self.rng,
                )
        return out


# ---------- Synthetic IterableDataset ----------

class SyntheticDataset(IterableDataset):
    """Yields `(stems, sampled_norm_params)` pairs for synth-mode training.

    Picks a random staged session per __next__, slices a random window
    of stems, and samples per-stem + per-group params in [0,1] space.
    The trainer is responsible for denormalizing and rendering the
    synth_mix on GPU.

    Group assignments come from the staged session's `correspondence.yaml`
    (via `LabelStore.get(session)['group_assignments']`). We use the
    teacher's group routing because it matches the engineer mix's
    instrument grouping; the teacher's param VALUES are not used.
    """

    def __init__(
        self,
        sessions: list[str],
        staging_dir: Path,
        label_store: LabelStore,
        sampler: ParamSampler,
        audio_len: int,
        n_max: int,
        max_groups: int,
        seed: int = 0,
        cache_size: int = 4,
    ):
        super().__init__()
        if not sessions:
            raise ValueError("no sessions provided")
        self.sessions = list(sessions)
        self.staging_dir = Path(staging_dir)
        self.label_store = label_store
        self.sampler = sampler
        self.audio_len = audio_len
        self.n_max = n_max
        self.max_groups = max_groups
        self.seed = seed
        self.cache_size = cache_size

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        if worker_info is not None:
            shard = self.sessions[worker_id::worker_info.num_workers]
        else:
            shard = self.sessions
        rng = random.Random(self.seed + worker_id)

        cache: dict[str, dict] = {}
        cache_order: list[str] = []

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
            ex = _example_from_session(data, start, self.audio_len, self.n_max)

            labels = self.label_store.get(session)
            if labels is None:
                continue
            n_real = int(ex["track_mask"].sum().item())
            stem_filenames = data["stem_filenames"][:n_real]
            fname_to_group = {
                fn: int(g) for fn, g in zip(
                    labels["stem_filenames"],
                    labels["group_assignments"].tolist(),
                )
            }
            group_assignments = torch.zeros(self.n_max, dtype=torch.long)
            for ti, fn in enumerate(stem_filenames):
                group_assignments[ti] = fname_to_group.get(fn, 0)
            n_groups = len(labels["groups"])

            # Sample normalized params for this single example (B=1).
            # Drop the B dim before yielding; the collator stacks.
            strip_norm = self.sampler.sample("strip", 1, self.n_max)
            group_norm = self.sampler.sample("group", 1, self.max_groups)
            strip_norm = {p: {k: v[0] for k, v in pp.items()}
                          for p, pp in strip_norm.items()}
            group_norm = {p: {k: v[0] for k, v in pp.items()}
                          for p, pp in group_norm.items()}

            yield {
                "tracks":            ex["tracks"],
                "mix":               ex["mix"],
                "track_mask":        ex["track_mask"],
                "meta":              ex["meta"],
                "group_assignments": group_assignments,
                "n_groups":          n_groups,
                "strip_norm":        strip_norm,
                "group_norm":        group_norm,
            }


def collate_synth(batch: list[dict]) -> dict:
    """Stack list of synth-dataset items into a batch dict."""
    tracks = torch.stack([ex["tracks"] for ex in batch])
    mix = torch.stack([ex["mix"] for ex in batch])
    track_mask = torch.stack([ex["track_mask"] for ex in batch])
    group_assignments = torch.stack([ex["group_assignments"] for ex in batch])
    n_groups_per_example = torch.tensor([ex["n_groups"] for ex in batch], dtype=torch.long)

    # Strip and group schemas can differ (Path B drops gain_panning from
    # the group chain), so iterate each independently.
    strip_norm: dict[str, dict[str, torch.Tensor]] = {
        proc: {
            k: torch.stack([ex["strip_norm"][proc][k] for ex in batch])
            for k in batch[0]["strip_norm"][proc]
        }
        for proc in batch[0]["strip_norm"]
    }
    group_norm: dict[str, dict[str, torch.Tensor]] = {
        proc: {
            k: torch.stack([ex["group_norm"][proc][k] for ex in batch])
            for k in batch[0]["group_norm"][proc]
        }
        for proc in batch[0]["group_norm"]
    }

    return {
        "tracks":               tracks,
        "mix":                  mix,
        "track_mask":           track_mask,
        "group_assignments":    group_assignments,
        "n_groups_per_example": n_groups_per_example,
        "strip_norm":           strip_norm,
        "group_norm":           group_norm,
        "meta":                 [ex["meta"] for ex in batch],
    }


__all__ = ["ParamSampler", "SyntheticDataset", "collate_synth"]
