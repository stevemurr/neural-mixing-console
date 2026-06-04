"""EncoderWrapper (V1): load a MixEncoder checkpoint, run no-grad forward,
denormalize the output to physical DSP parameters.

Sits between the inference thread (scheduler) and the trained encoder. The
predict() method is the only thing the scheduler calls — it takes a padded
batch-of-1 tracks tensor + mask + MERT and returns a dict structured for
downstream consumption (printing in V1, the smoother in V3).

Threshold-mode note: training defaults to `--use-rms-relative-threshold=1`,
which reinterprets the encoder's `threshold_db` head output as
`threshold_offset_db ∈ [-30, +6]` added to the per-track RMS. We expose
both interpretations on the returned dict so V2's DSP can use the
training-correct value while V1's CLI can still print human-readable
numbers without depending on the audio thread's RMS measurement.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

# The realtime package lives next to the project root; import the existing
# model + denorm machinery directly so we share one source of truth.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.encoders import MixEncoder
from training.data import BUS_PARAM_KEYS, STRIP_PARAM_KEYS
from training.train_stage3 import (
    MERT_DIM as DEFAULT_MERT_DIM,
    SAMPLE_RATE,
    _build_denorm_table,
    _denorm,
)


# The threshold_db -> offset reinterpretation used at training time when
# `--use-rms-relative-threshold=1`. Mirrors _build_static_render_args in
# infer.py (which is the authority for inference-time threshold handling).
_THRESH_OFFSET_MIN_DB = -30.0
_THRESH_OFFSET_SPAN_DB = 36.0


class EncoderWrapper:
    """Trained MixEncoder + parameter denormalization, wrapped for realtime use.

    Construction loads the checkpoint to `device` and prepares the
    denormalization tables. `predict()` is reentrant-safe within a single
    thread (the scheduler's worker thread); it is NOT designed to be
    called from multiple threads concurrently.
    """

    def __init__(
        self,
        ckpt_path: str | Path,
        *,
        device: str = "cpu",
        n_max: int = 42,
        mert_dim: int = DEFAULT_MERT_DIM,
        trim_max_db: float = 18.0,
        sample_rate: int = SAMPLE_RATE,
        use_rms_relative_threshold: bool = True,
        use_bf16_on_cuda: bool = True,
    ) -> None:
        self.ckpt_path = Path(ckpt_path)
        self.device = torch.device(device)
        self.n_max = int(n_max)
        self.mert_dim = int(mert_dim)
        self.trim_max_db = float(trim_max_db)
        self.sample_rate = int(sample_rate)
        self.use_rms_relative_threshold = bool(use_rms_relative_threshold)
        self._use_bf16 = bool(use_bf16_on_cuda) and self.device.type == "cuda"

        self.encoder = MixEncoder(
            sample_rate=self.sample_rate,
            use_ref_mix=False,
            mert_dim=self.mert_dim,
            trim_max_db=self.trim_max_db,
        ).to(self.device).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.ckpt_path}")
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "encoder_state_dict" in ckpt:
            ckpt = ckpt["encoder_state_dict"]
        missing, unexpected = self.encoder.load_state_dict(ckpt, strict=False)
        logging.info(
            "EncoderWrapper: loaded %s (%d missing, %d unexpected) on %s",
            self.ckpt_path.name, len(missing), len(unexpected), self.device,
        )

        self.strip_table = _build_denorm_table(self.device, STRIP_PARAM_KEYS)
        self.bus_table = _build_denorm_table(self.device, BUS_PARAM_KEYS)

        self._threshold_idx = STRIP_PARAM_KEYS.index("threshold_db")

    @torch.no_grad()
    def predict(
        self,
        tracks: torch.Tensor,
        track_mask: torch.Tensor,
        mert: torch.Tensor | None,
    ) -> dict:
        """Run one forward pass and return a denormalized parameter dict.

        Args:
            tracks:     (1, n_max, 2, T) float32 — already padded.
            track_mask: (1, n_max) bool       — True = real track.
            mert:       (1, n_max, mert_dim) float32, or None when mert_dim==0.

        Returns a dict with:
            "strip_phys":             ndarray (n_active, 22) — physical units per STRIP_PARAM_KEYS
            "strip_norm":             ndarray (n_active, 22) — raw sigmoid [0, 1] (lossless)
            "bus_phys":               ndarray (13,)          — physical units per BUS_PARAM_KEYS
            "bus_norm":               ndarray (13,)          — raw sigmoid [0, 1]
            "trim_db":                float
            "n_active":               int
            "rms_relative_threshold": bool — copy of the wrapper setting
            (when rms_relative_threshold=True only:)
            "threshold_offset_db":    ndarray (n_active,) — offset added to per-track RMS
        """
        tracks = tracks.to(self.device, non_blocking=True)
        track_mask = track_mask.to(self.device, non_blocking=True)
        if mert is not None and self.mert_dim > 0:
            mert = mert.to(self.device, non_blocking=True)
        else:
            mert = None

        autocast_kw = dict(device_type="cuda", dtype=torch.bfloat16, enabled=self._use_bf16)
        with torch.amp.autocast(**autocast_kw):
            out = self.encoder(
                tracks, track_mask,
                ref_mix=None,
                mert_embeddings=mert if self.mert_dim > 0 else None,
            )

        n_active = int(track_mask[0].sum().item())
        track_params_norm = out["track_params"][0].float()                # (n_max, 22)
        bus_params_norm = out["bus_params"][0].float()                    # (13,)
        trim_db = float(out["trim_db"][0].item())

        strip_phys = _denorm(track_params_norm, self.strip_table)         # (n_max, 22)
        bus_phys = _denorm(bus_params_norm, self.bus_table)               # (13,)

        result = {
            "strip_phys": strip_phys[:n_active].cpu().numpy(),
            "strip_norm": track_params_norm[:n_active].cpu().numpy(),
            "bus_phys": bus_phys.cpu().numpy(),
            "bus_norm": bus_params_norm.cpu().numpy(),
            "trim_db": trim_db,
            "n_active": n_active,
            "rms_relative_threshold": self.use_rms_relative_threshold,
        }

        if self.use_rms_relative_threshold and n_active > 0:
            thr_norm = track_params_norm[:n_active, self._threshold_idx].clamp(0.0, 1.0)
            offset = _THRESH_OFFSET_MIN_DB + thr_norm * _THRESH_OFFSET_SPAN_DB
            result["threshold_offset_db"] = offset.cpu().numpy()

        return result

    # ------- helpers for the scheduler -------

    @property
    def strip_keys(self) -> tuple[str, ...]:
        return STRIP_PARAM_KEYS

    @property
    def bus_keys(self) -> tuple[str, ...]:
        return BUS_PARAM_KEYS
