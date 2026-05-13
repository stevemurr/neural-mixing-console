"""WebDataset-based shard streamer for stage 3 v6 multitrack training.

Each shard contains `{example_id}.mix.flac`, `{example_id}.track_NNN.flac`,
and `{example_id}.meta.json`. This module produces PyTorch IterableDatasets
that yield variable-track-count bundles, then collates them with masks.

No param targets — stage 3 trains via reconstruction loss only.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import webdataset as wds


SAMPLE_RATE = 48_000
SEGMENT_LEN = SAMPLE_RATE * 6  # 288000


# ---------- helpers ----------

def _decode_flac(buf: bytes) -> torch.Tensor:
    """Decode FLAC bytes to a (C, T) float32 tensor (channels-first)."""
    data, fs = sf.read(io.BytesIO(buf), dtype="float32", always_2d=True)
    assert fs == SAMPLE_RATE, f"unexpected sample rate {fs}"
    return torch.from_numpy(data.T.copy())


def _decode_json(buf: bytes) -> dict:
    return json.loads(buf.decode("utf-8"))


def _broadcast_to_stereo(x: torch.Tensor) -> torch.Tensor:
    """If mono (1, T), broadcast to (2, T). Stereo passes through."""
    if x.shape[0] == 1:
        return x.expand(2, -1).contiguous()
    return x


# ---------- v6 schema constants ----------
#
# v6.1 change: no explicit bypass heads. "Bypassed" is expressed in the
# parameter space itself — EQ gain → 0 dB, comp ratio → 1:1, clip mix → 0%.
# The comp knee and the clipper shape/ceiling were also dropped (the
# reconstruction loss can't supervise them; they're hardcoded to sensible
# defaults inside DiffComp / DiffClipper).

# Per-effect param key tuples used to construct the canonical strip / bus
# layouts below. EQ_PARAM_KEYS is also imported by encoder priors so its
# layout must stay frozen.
EQ_PARAM_KEYS = (
    "hpf_freq", "ls_freq", "ls_gain", "ls_q",
    "p1_freq", "p1_gain", "p1_q",
    "p2_freq", "p2_gain", "p2_q",
    "hs_freq", "hs_gain", "hs_q",
    "lpf_freq",
)

# Per-track strip layout: gain (1) + EQ (14) + comp-no-makeup-no-knee (4)
# + clip-drive+mix (2) + pan (1) = 22.
STRIP_PARAM_KEYS = (
    "gain_db",
    *EQ_PARAM_KEYS,
    "threshold_db", "ratio", "attack_ms", "release_ms",
    "clip_drive_db", "clip_mix",
    "pan",
)
assert len(STRIP_PARAM_KEYS) == 22, f"expected 22 strip keys, got {len(STRIP_PARAM_KEYS)}"

# Bus layout: 9 bus EQ + 4 bus comp (no makeup, no knee) = 13.
BUS_PARAM_KEYS = (
    "bus_low_boost_freq", "bus_low_boost_gain",
    "bus_low_attn_freq", "bus_low_attn_gain",
    "bus_mid_freq", "bus_mid_gain", "bus_mid_q",
    "bus_air_freq", "bus_air_gain",
    "bus_threshold_db", "bus_ratio", "bus_attack_ms", "bus_release_ms",
)
assert len(BUS_PARAM_KEYS) == 13, f"expected 13 bus keys, got {len(BUS_PARAM_KEYS)}"


# ---------- stage 3 (real-data multitrack + ref mix; no param targets) ----------

class MertCacheLookup:
    """In-memory MERT embedding lookup keyed by (dataset, session, filename).

    Caches are produced offline by `scripts/precompute_mert_cache.py` as one
    .npz per session under `dmc-data/mert_cache/<dataset>/<session>.npz`. Each
    .npz holds `filenames` (object array) and `embeddings` ((N, D) float16).

    Loaded lazily — only sessions that appear in the data are read into RAM.
    Total memory footprint is tiny (~15-30 MB for full corpus, fp16 768-dim).
    """

    def __init__(self, cache_root: str | Path):
        self.cache_root = Path(cache_root)
        self._sessions: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    def _load_session(self, dataset: str, session: str) -> dict[str, np.ndarray] | None:
        key = (dataset, session)
        if key in self._sessions:
            return self._sessions[key]
        npz_path = self.cache_root / dataset / f"{session}.npz"
        if not npz_path.exists():
            self._sessions[key] = {}
            return None
        with np.load(npz_path, allow_pickle=True) as data:
            names = list(data["filenames"])
            embs = data["embeddings"]
        mp = {n: embs[i] for i, n in enumerate(names)}
        self._sessions[key] = mp
        return mp

    def lookup(self, dataset: str, session: str, filename: str) -> np.ndarray | None:
        m = self._load_session(dataset, session)
        if not m:
            return None
        return m.get(filename)


class _RoundRobinIterableDataset(torch.utils.data.IterableDataset):
    """Round-robin between several IterableDatasets at the sample level.

    Used by `make_stage3_dataset(alternate=True)` to force a fixed dataset
    ratio across `--shard-dirs`: with two sources, pipeline A and pipeline B
    (each with its own shuffle reservoir) alternate every sample, so a
    batch_size=4 batch always contains 2 from each source. Defeats the
    periodic Cambridge↔Slakh sample-mix drift that the combined WebDataset
    shuffle introduces. Each child pipeline should have `repeat=True` for
    endless training; otherwise this iterator stops as soon as any child
    exhausts.
    """

    def __init__(self, datasets):
        super().__init__()
        self.datasets = list(datasets)

    def __iter__(self):
        iters = [iter(d) for d in self.datasets]
        while iters:
            for it in iters:
                try:
                    yield next(it)
                except StopIteration:
                    return


def make_stage3_dataset(
    shard_dirs: list[str | Path],
    *,
    shuffle: int = 200,
    split: Optional[str] = None,
    max_tracks: int = 80,
    repeat: bool = False,
    mert_cache_root: Optional[str | Path] = "dmc-data/mert_cache",
    drop_unmatched_mert: bool = False,
    alternate: bool = False,
):
    """Yield real-data stage-3 bundles from one or more `dmc-data/stage3_*` dirs.

    Each yielded dict contains:
      - tracks:           list of (2, T) tensors (variable count)
      - mix:              (2, T) tensor
      - mert_embeddings:  (N, D) tensor (one per track; zeros if missing)
      - meta:             dict from meta.json
      - dataset:          "cambridge" | "slakh" (derived from shard dir name)

    `alternate=True` with 2+ shard_dirs builds one WebDataset pipeline per
    dir and round-robins them at the sample level (instead of merging all
    shards into one shuffled pool). Use this when the per-dir distributions
    are visibly different and you want every batch to be balanced across
    sources.
    """
    if alternate and len(shard_dirs) >= 2:
        children = [
            make_stage3_dataset(
                [d], shuffle=shuffle, split=split, max_tracks=max_tracks,
                repeat=repeat, mert_cache_root=mert_cache_root,
                drop_unmatched_mert=drop_unmatched_mert, alternate=False,
            )
            for d in shard_dirs
        ]
        return _RoundRobinIterableDataset(children)

    all_shards = []
    for d in shard_dirs:
        d = Path(d)
        for shard in sorted((d / "shards").glob("*.tar")):
            all_shards.append(str(shard))

    if not all_shards:
        raise FileNotFoundError(f"no shards under {shard_dirs}")

    mert_cache = MertCacheLookup(mert_cache_root) if mert_cache_root else None

    def _yield(sample: dict) -> Optional[dict]:
        meta_buf = sample.get("meta.json")
        mix_buf = sample.get("mix.flac")
        if meta_buf is None or mix_buf is None:
            return None
        meta = _decode_json(meta_buf)
        if split is not None and meta.get("split") != split:
            return None
        track_keys = sorted(k for k in sample.keys() if k.startswith("track_") and k.endswith(".flac"))
        if not track_keys:
            return None
        if len(track_keys) > max_tracks:
            track_keys = track_keys[:max_tracks]
        tracks = []
        for k in track_keys:
            t = _decode_flac(sample[k])
            tracks.append(_broadcast_to_stereo(t))
        mix = _decode_flac(mix_buf)

        # Determine which dataset this shard came from. WebDataset doesn't
        # expose the source-shard filename in `sample`, so we infer from the
        # `meta["stage"]` field (set at ingest as e.g. "stage3_cambridge").
        stage_name = meta.get("stage", "")
        ds_name = stage_name.replace("stage3_", "") if stage_name.startswith("stage3_") else ""
        session = meta.get("session", "")

        # Look up MERT embeddings per track filename.
        n_tracks = len(track_keys)
        mert_embs: list[np.ndarray | None] = [None] * n_tracks
        if mert_cache and ds_name and session:
            track_metas = meta.get("tracks", [])
            for i in range(n_tracks):
                if i >= len(track_metas):
                    continue
                fn = track_metas[i].get("filename", "")
                emb = mert_cache.lookup(ds_name, session, fn)
                mert_embs[i] = emb
            if drop_unmatched_mert:
                keep = [i for i, e in enumerate(mert_embs) if e is not None]
                if not keep:
                    return None
                tracks = [tracks[i] for i in keep]
                mert_embs = [mert_embs[i] for i in keep]

        return {
            "tracks": tracks, "mix": mix, "mert_embeddings": mert_embs,
            "meta": meta, "dataset": ds_name,
        }

    pipeline = wds.WebDataset(all_shards, shardshuffle=True, empty_check=False, nodesplitter=wds.split_by_node)
    if shuffle > 0:
        pipeline = pipeline.shuffle(shuffle)
    pipeline = pipeline.map(_yield, handler=wds.warn_and_continue)
    pipeline = pipeline.select(lambda x: x is not None)
    if repeat:
        pipeline = pipeline.repeat()
    return pipeline


def collate_stage3(batch: list[dict], n_max: int = 12, mert_dim: int = 768,
                   min_track_rms_dbfs: float = float("-inf")) -> dict:
    """Stack stage 3 bundles. Returns:
        tracks:           (B, N_max, 2, T)
        track_mask:       (B, N_max)
        mix:              (B, 2, T)
        mert_embeddings:  (B, N_max, mert_dim)  — zeros where missing
        meta:             list[dict]

    `min_track_rms_dbfs`: if a track's segment-RMS is below this threshold,
    its `track_mask` slot is set to False so the encoder treats it as absent.
    Silent slots otherwise teach the encoder "for this MERT identity, predict
    identity-ish params" — which then poisons inference when the same track
    is active. Default `-inf` disables masking (back-compat). A value around
    -45 dBFS captures genuinely silent tracks while leaving quiet pads/BVs
    that have any signal alone. Safety: if ALL tracks in an example would be
    masked, the mask is NOT applied for that example (degenerate batch
    avoidance — the example is left intact and the trainer sees it normally).
    """
    B = len(batch)
    T = batch[0]["mix"].shape[-1]
    tracks_padded = torch.zeros((B, n_max, 2, T), dtype=torch.float32)
    masks = torch.zeros((B, n_max), dtype=torch.bool)
    mixes = torch.zeros((B, 2, T), dtype=torch.float32)
    mert = torch.zeros((B, n_max, mert_dim), dtype=torch.float32)
    metas = []
    apply_activity_mask = min_track_rms_dbfs > float("-inf")
    rms_floor_lin = 10.0 ** (min_track_rms_dbfs / 20.0) if apply_activity_mask else 0.0
    for bi, ex in enumerate(batch):
        K = min(len(ex["tracks"]), n_max)
        masks[bi, :K] = True
        track_rms_lin = torch.zeros(K, dtype=torch.float32) if apply_activity_mask else None
        for k_idx in range(K):
            t = ex["tracks"][k_idx]
            tt = t.shape[-1]
            if tt > T:
                t = t[:, :T]
            elif tt < T:
                t = torch.cat([t, torch.zeros(2, T - tt, dtype=t.dtype)], dim=-1)
            tracks_padded[bi, k_idx] = t
            if apply_activity_mask:
                track_rms_lin[k_idx] = torch.sqrt((t.float() ** 2).mean() + 1e-12)
            emb = ex["mert_embeddings"][k_idx] if k_idx < len(ex["mert_embeddings"]) else None
            if emb is not None:
                mert[bi, k_idx] = torch.from_numpy(np.asarray(emb)).to(torch.float32)
        if apply_activity_mask:
            silent = track_rms_lin < rms_floor_lin
            # Avoid degenerate "all silent" batch entries: only apply if at
            # least one track survives.
            if (~silent).any():
                masks[bi, :K] = ~silent
        # Pad/truncate the mix to T as well — examples can carry different
        # segment lengths if the batch mixes shards prepared with different
        # --segment-seconds (cambridge=15 s, slakh=6 s in the current prep).
        mx = ex["mix"]
        mt = mx.shape[-1]
        if mt > T:
            mx = mx[:, :T]
        elif mt < T:
            mx = torch.cat([mx, torch.zeros(2, T - mt, dtype=mx.dtype)], dim=-1)
        mixes[bi] = mx
        metas.append(ex["meta"])
    return {
        "tracks": tracks_padded, "track_mask": masks,
        "mix": mixes, "mert_embeddings": mert, "meta": metas,
    }


__all__ = [
    "SAMPLE_RATE", "SEGMENT_LEN",
    "EQ_PARAM_KEYS",
    "STRIP_PARAM_KEYS", "BUS_PARAM_KEYS",
    "MertCacheLookup", "make_stage3_dataset", "collate_stage3",
]
