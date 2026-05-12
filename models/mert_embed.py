"""Frozen MERT-v1 audio embedder for per-track semantic conditioning.

MERT (Music undERstanding model with large-scale self-supervised Training,
Li et al., 2023) is pretrained on 160k hours of music. Strong inductive
bias for instrument identity, playing style, and timbre — exactly the
information our MixEncoder needs to condition param predictions on.

Operating points:
  - Native sample rate: 24 kHz. We resample 48 kHz audio down before encoding.
  - Output: hidden states at every transformer layer; we take a uniform
    weighted average per the MERT paper's recommendation for unsupervised
    feature use.
  - Pool over time (mean) to a single 768-d (95M variant) or 1024-d (330M)
    vector per track.

Usage:
    embedder = MertEmbedder(model_name="m-a-p/MERT-v1-95M")
    # x: (B, T) float32 at 48 kHz, mono
    emb = embedder(x)   # (B, 768)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_MERT_NATIVE_SR = 24_000


class MertEmbedder(nn.Module):
    """Frozen MERT-v1 wrapper.

    Args:
        model_name: HuggingFace model id (`m-a-p/MERT-v1-95M` or `-330M`).
        device: where to place the model.
        layer_weighting: "uniform" (mean over all layers) or "last" (last
            layer only). Paper recommends a learned weighted sum during
            fine-tuning; for frozen feature extraction, uniform mean is the
            standard practical choice.
    """

    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-95M",
        device: Optional[torch.device] = None,
        layer_weighting: str = "uniform",
    ):
        super().__init__()
        from transformers import AutoModel, AutoFeatureExtractor

        self.model_name = model_name
        self.layer_weighting = layer_weighting
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Resampler is a fixed sinc kernel; no trainable params either way.
        # Use torchaudio's polyphase resample (fast, GPU-capable).
        import torchaudio.transforms as T
        self.resampler = T.Resample(orig_freq=48000, new_freq=_MERT_NATIVE_SR,
                                    resampling_method="sinc_interp_kaiser",
                                    lowpass_filter_width=16, beta=8.555504641634386)

        if device is not None:
            self.to(device)

        # Cache the embedding dim for callers
        self._embed_dim: Optional[int] = None

    @property
    def embed_dim(self) -> int:
        if self._embed_dim is None:
            self._embed_dim = self.model.config.hidden_size
        return self._embed_dim

    @torch.no_grad()
    def forward(self, audio_48k: torch.Tensor) -> torch.Tensor:
        """Embed audio at 48 kHz to a per-clip semantic vector.

        Args:
            audio_48k: (B, T) or (B, C, T) float32 at 48 kHz. If stereo, mix
                to mono before encoding (MERT is mono-trained).

        Returns:
            (B, embed_dim) — single vector per clip, mean-pooled over time
            and uniform-averaged across MERT's transformer layers.
        """
        if audio_48k.dim() == 3:
            # (B, C, T) → mono via mean across channels
            audio = audio_48k.mean(dim=1)
        elif audio_48k.dim() == 2:
            audio = audio_48k
        else:
            raise ValueError(f"audio_48k must be (B, T) or (B, C, T); got {audio_48k.shape}")

        # Resample to 24 kHz
        audio_24k = self.resampler(audio)   # (B, T_24)

        # MERT's feature extractor expects (B, T) raw audio at 24 kHz.
        # We bypass the HF tokenizer (which expects numpy) and feed tensors
        # directly; the model accepts `input_values` of shape (B, T).
        # Normalize to zero-mean unit-variance if extractor would.
        # The MERT-v1 extractor is `do_normalize=True, return_attention_mask=True`.
        attention_mask = torch.ones_like(audio_24k, dtype=torch.long)
        outputs = self.model(
            input_values=audio_24k,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states  # tuple of (B, T_h, D)

        if self.layer_weighting == "last":
            stacked = hidden_states[-1]                                # (B, T_h, D)
        else:
            # uniform mean across layers
            stacked = torch.stack(hidden_states, dim=0).mean(dim=0)    # (B, T_h, D)

        # Time-mean pool
        pooled = stacked.mean(dim=1)                                   # (B, D)
        return pooled


__all__ = ["MertEmbedder"]
