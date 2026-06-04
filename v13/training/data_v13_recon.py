"""Audio-loading dataset for Stage 3 reconstruction-based pan training.

Loads stems + engineer mix audio per window on demand. Unlike the precomputed
v13 dataset (which stores C + targets), this provides raw audio for
reconstruction-based supervision (no LS pan targets needed — supervision is
the engineer mix audio itself, via ILD loss in the trainer).

Per-session loaded data is cached LRU-style to amortize I/O across the
multiple windows we sample per session.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from models.mono_pan import detect_stereo
from scripts.precompute_v13_dataset import load_session_audio


class V13ReconDataset(Dataset):
    """Per-window (stems, engineer_mix, is_stereo) examples from staged sessions."""

    def __init__(
        self,
        sessions: list[str],
        staging_dir: str | Path,
        sample_rate: int = 48_000,
        audio_len: int = 6 * 48_000,
        n_max: int = 64,
        examples_per_session: int = 20,
        cache_size: int = 4,
        seed: int = 0,
    ):
        super().__init__()
        self.staging_dir = Path(staging_dir)
        self.sample_rate = sample_rate
        self.audio_len = audio_len
        self.n_max = n_max
        self.cache_size = cache_size

        # Build a deterministic list of (session, window_start) tuples.
        rng = random.Random(seed)
        self.examples: list[tuple[str, int]] = []
        n_skipped = 0
        for sess in sessions:
            mix_path = self.staging_dir / sess / "mix.wav"
            if not mix_path.is_file():
                n_skipped += 1
                continue
            try:
                T_total = sf.info(str(mix_path)).frames
            except Exception:
                n_skipped += 1
                continue
            if T_total < audio_len + 1:
                n_skipped += 1
                continue
            for _ in range(examples_per_session):
                start = rng.randint(0, T_total - audio_len - 1)
                self.examples.append((sess, start))

        if n_skipped > 0:
            logging.info(f"V13ReconDataset: skipped {n_skipped} session(s) (missing or too short)")

        # LRU cache of session data: {session_name: data_dict}
        self._cache: dict[str, dict] = {}
        self._cache_order: list[str] = []

    def __len__(self) -> int:
        return len(self.examples)

    def _get_session(self, sess: str) -> dict | None:
        if sess in self._cache:
            # Touch LRU
            self._cache_order.remove(sess)
            self._cache_order.append(sess)
            return self._cache[sess]

        # Evict LRU if at capacity
        while len(self._cache) >= self.cache_size and self._cache_order:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)

        try:
            data = load_session_audio(self.staging_dir / sess, self.sample_rate)
        except Exception as e:
            logging.warning(f"V13ReconDataset: failed to load {sess}: {e}")
            return None
        if data is None:
            return None
        self._cache[sess] = data
        self._cache_order.append(sess)
        return data

    def __getitem__(self, idx: int) -> dict:
        sess, start = self.examples[idx]
        data = self._get_session(sess)
        # If a session fails to load mid-training, fall back to first example.
        # (Should be rare; happens only if disk failure or file corruption.)
        if data is None:
            sess, start = self.examples[0]
            data = self._get_session(sess)
            if data is None:
                raise RuntimeError(f"failed to load fallback session {sess}")

        stems_full = data["stems"]                          # (N_total, 2, T_actual)
        mix_full = data["mix"]                              # (2, T_actual)
        # T_actual may be shorter than sf.info(mix).frames because
        # load_session_audio truncates to the shortest stem. Clamp start to
        # fit; if the session itself is shorter than audio_len, pad the time
        # dim with zeros so the batch is uniform.
        T_actual = stems_full.shape[-1]
        if start + self.audio_len > T_actual:
            start = max(0, T_actual - self.audio_len)
        end = start + self.audio_len
        stems_w = stems_full[:, :, start:end].clone()        # (N_real, 2, ≤audio_len)
        mix_w = mix_full[:, start:end].clone()
        T_clip = stems_w.shape[-1]
        if T_clip < self.audio_len:
            pad_t = self.audio_len - T_clip
            stems_w = F.pad(stems_w, (0, pad_t))
            mix_w = F.pad(mix_w, (0, pad_t))

        N_real = stems_w.shape[0]
        if N_real > self.n_max:
            stems_w = stems_w[:self.n_max]
            N_real = self.n_max
        if N_real < self.n_max:
            pad = torch.zeros(self.n_max - N_real, 2, self.audio_len)
            stems_padded = torch.cat([stems_w, pad], dim=0)
        else:
            stems_padded = stems_w

        track_mask = torch.zeros(self.n_max, dtype=torch.bool)
        track_mask[:N_real] = True
        is_stereo = detect_stereo(stems_padded) & track_mask

        return {
            "stems":      stems_padded,                      # (n_max, 2, audio_len)
            "mix":        mix_w,                             # (2, audio_len)
            "is_stereo":  is_stereo,                         # (n_max,) bool
            "track_mask": track_mask,                        # (n_max,) bool
            "session":    sess,
            "start":      start,
        }


def collate_recon(batch: list[dict]) -> dict:
    return {
        "stems":      torch.stack([b["stems"] for b in batch]),
        "mix":        torch.stack([b["mix"] for b in batch]),
        "is_stereo":  torch.stack([b["is_stereo"] for b in batch]),
        "track_mask": torch.stack([b["track_mask"] for b in batch]),
        "sessions":   [b["session"] for b in batch],
        "starts":     [b["start"] for b in batch],
    }


__all__ = ["V13ReconDataset", "collate_recon"]
