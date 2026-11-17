"""MixEncoder for stage 3 v6.1 multitrack training.

Hybrid mel + waveform backbone (Transformer-on-mel + 1D-CNN-on-waveform fused
via cross-attention), permutation-invariant transformer over track tokens,
and heads for per-track strip params, master bus params, and a global trim
scalar. No bypass heads (v6.1) — "bypassed" is expressed in the parameter
space (EQ gain → 0 dB, comp ratio → 1, clip mix → 0).

Outputs:
- `params` heads emit sigmoid → [0, 1]; downstream denormalizer converts to
  physical units via `reference.param_norm.PARAM_RANGES`.
- `trim_db` is the only loudness lever, predicting a scalar in
  `[-trim_max_db, +trim_max_db]`.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ---------- Mel feature extractor ----------

class MelFrontend(nn.Module):
    """Log-mel spectrogram. Default: 128 mels, 1024 FFT, 512 hop @ 48kHz (~94 frames/s)."""

    def __init__(
        self,
        sample_rate: int = 48_000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: float = 20.0,
        f_max: float = 22000.0,
    ):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )
        self.n_mels = n_mels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> mel: (B, C, n_mels, F)
        B, C, T = x.shape
        flat = x.reshape(B * C, T)
        mel = self.mel(flat)                                  # (B*C, n_mels, F)
        log_mel = torch.log(mel + 1e-6)
        return log_mel.reshape(B, C, self.n_mels, -1)


# ---------- 1D CNN waveform backbone ----------

class WaveformCNN(nn.Module):
    """Strided dilated 1D CNN downsampler. Input (B, C, T) -> output (B, D, T').

    With default config: 4 strided blocks of stride 4 each → ~256× temporal
    downsample. At 48 kHz this gives ~187 Hz feature rate (~5 ms hop).
    """

    def __init__(self, in_channels: int, hidden: int = 128, out_dim: int = 256, n_blocks: int = 4):
        super().__init__()
        layers = []
        c_in = in_channels
        c_out = hidden
        for k in range(n_blocks):
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=7, stride=4, padding=3, dilation=1),
                nn.GELU(),
                nn.Conv1d(c_out, c_out, kernel_size=3, padding=2, dilation=2),
                nn.GELU(),
            ]
            c_in = c_out
            c_out = min(out_dim, c_out * 2)
        self.body = nn.Sequential(*layers)
        self.proj = nn.Conv1d(c_in, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.body(x))


# ---------- AST-style mel transformer ----------

class MelTransformer(nn.Module):
    """Patch-free time-frame Transformer on log-mel.

    Each time frame's mel vector becomes a token; positional encoding on
    time axis only.
    """

    def __init__(
        self,
        in_channels: int,
        n_mels: int = 128,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 6,
        max_frames: int = 4096,   # supports up to ~43s @ hop=512 / 48 kHz
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels * n_mels, d_model, kernel_size=1)
        self.time_pe = nn.Parameter(torch.zeros(1, max_frames, d_model))
        nn.init.trunc_normal_(self.time_pe, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d_model = d_model

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        # log_mel: (B, C, n_mels, F)
        B, C, n_mels, F = log_mel.shape
        x = log_mel.reshape(B, C * n_mels, F)            # (B, C*n_mels, F)
        x = self.in_proj(x)                              # (B, d_model, F)
        x = x.transpose(1, 2)                            # (B, F, d_model)
        x = x + self.time_pe[:, :F, :]
        x = self.encoder(x)                              # (B, F, d_model)
        return x


# ---------- Hybrid backbone (mel transformer + waveform CNN, cross-attended) ----------

class HybridBackbone(nn.Module):
    """Mel-transformer trunk + waveform-CNN branch, fused via cross-attention.

    Output: pooled global feature `(B, d_model)` plus the joint sequence.
    """

    def __init__(
        self,
        in_channels: int,
        n_mels: int = 128,
        d_model: int = 384,
        n_heads: int = 6,
        n_mel_layers: int = 6,
        n_fuse_layers: int = 2,
        wave_cnn_dim: int = 256,
        sample_rate: int = 48_000,
    ):
        super().__init__()
        self.mel_front = MelFrontend(sample_rate=sample_rate, n_mels=n_mels)
        self.mel_tx = MelTransformer(
            in_channels=in_channels, n_mels=n_mels,
            d_model=d_model, n_heads=n_heads, n_layers=n_mel_layers,
        )
        self.wave_cnn = WaveformCNN(in_channels=in_channels, out_dim=wave_cnn_dim)
        self.wave_proj = nn.Conv1d(wave_cnn_dim, d_model, kernel_size=1)
        self.cross_layers = nn.ModuleList([
            CrossAttnBlock(d_model, n_heads) for _ in range(n_fuse_layers)
        ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, C, T). Returns (pooled (B, d_model), tokens (B, F+1, d_model))."""
        log_mel = self.mel_front(x)                              # (B, C, n_mels, F)
        mel_tokens = self.mel_tx(log_mel)                        # (B, F, d_model)

        wave_feat = self.wave_cnn(x)                             # (B, wave_cnn_dim, F')
        wave_feat = self.wave_proj(wave_feat).transpose(1, 2)    # (B, F', d_model)

        # Prepend CLS token to mel sequence
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, mel_tokens], dim=1)             # (B, 1+F, d_model)

        # Cross-attend mel tokens to waveform features (bring transient info into mel)
        for layer in self.cross_layers:
            tokens = layer(tokens, wave_feat)

        # Pool: take CLS
        pooled = tokens[:, 0]
        return pooled, tokens


class CrossAttnBlock(nn.Module):
    """One block of self-attn on q + cross-attn (q -> kv) + FFN."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.cross_norm_q = nn.LayerNorm(d_model)
        self.cross_norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model), nn.Dropout(0.1),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_n = self.self_norm(q)
        a, _ = self.self_attn(q_n, q_n, q_n, need_weights=False)
        q = q + a

        q_n = self.cross_norm_q(q)
        kv_n = self.cross_norm_kv(kv)
        a, _ = self.cross_attn(q_n, kv_n, kv_n, need_weights=False)
        q = q + a

        q = q + self.ffn(self.ffn_norm(q))
        return q


# ---------- Heads ----------

class StripParamHead(nn.Module):
    """Per-track strip head: sigmoid params in [0, 1].

    v6.1 layout: 22 params (gain + EQ + comp-no-makeup-no-knee + clip-drive+mix
    + pan). No bypass head — "bypassed" is expressed in param space (EQ gain 0,
    comp ratio 1, clip mix 0). See training/data.py:STRIP_PARAM_KEYS.
    """

    N_PARAMS = 22

    def __init__(self, d_model: int, n_params: int = N_PARAMS):
        super().__init__()
        self.n_params = n_params
        self.params = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_params))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.params(x))


class TrimHead(nn.Module):
    """Single-scalar output trim in dB, range ±max_db via tanh.

    Initialized to predict exactly 0 dB regardless of input — final-layer
    weights and bias are zero. This keeps a fine-tune from a pre-v6
    checkpoint stable: the model starts behaving identically to its
    pre-trim self, and the trim only departs from 0 when the loudness
    loss has accumulated enough gradient pressure to do so.
    """

    def __init__(self, in_dim: int, max_db: float = 12.0):
        super().__init__()
        self.max_db = max_db
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.GELU(),
            nn.Linear(in_dim, 1),
        )
        # Zero-init the final layer so trim_db starts at 0 dB.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim) → trim_db: (B,) in [-max_db, +max_db]
        return self.max_db * torch.tanh(self.mlp(x).squeeze(-1))


# ---------- MixEncoder (full multi-track v6) ----------

class MixEncoder(nn.Module):
    """Stage 3 v6 full-mix inversion: N raw tracks → all v6 params.

    Architecture:
        1. Per-track HybridBackbone features (mel-tx + waveform-CNN, fused).
        2. Optional MERT semantic embedding added to per-track features.
        3. Permutation-invariant transformer over track tokens (no track PE).
        4. Heads: per-track strip params, master bus params, global trim_db
           scalar. No bypass heads (v6.1) — "bypassed" lives in param space.

    Forward args:
        tracks:           (B, N_max, 2, T)
        track_mask:       (B, N_max) bool
        ref_mix:          (B, 2, T) — only consumed when use_ref_mix=True
                          at __init__; v6 stage 3 trains with use_ref_mix=False.
        mert_embeddings:  (B, N_max, mert_dim) — required iff mert_dim > 0.

    Returns:
        track_params:  (B, N_max, 22)  sigmoid in [0, 1]
        bus_params:    (B, 13)         sigmoid in [0, 1]
        trim_db:       (B,)            dB in [-trim_max_db, +trim_max_db]
    """

    def __init__(
        self,
        sample_rate: int = 48_000,
        d_model: int = 384,
        n_track_layers: int = 4,
        use_ref_mix: bool = False,
        mert_dim: int = 0,
        *,
        trim_max_db: float = 12.0,
    ):
        super().__init__()
        self.fs = sample_rate
        self.d_model = d_model
        self.use_ref_mix = use_ref_mix
        self.mert_dim = mert_dim

        # Per-track backbone in_channels:
        #   - 2 (mono-broadcast-to-stereo dry track) when use_ref_mix=False (v6 default)
        #   - 4 (dry stereo + ref_mix stereo) when use_ref_mix=True
        in_channels = 4 if use_ref_mix else 2
        self.track_backbone = HybridBackbone(in_channels=in_channels, d_model=d_model, sample_rate=sample_rate)

        # Optional MERT conditioning: project per-track embedding to d_model
        # and add to track tokens before the track transformer.
        if mert_dim > 0:
            self.mert_proj = nn.Sequential(
                nn.Linear(mert_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
        else:
            self.mert_proj = None

        # Permutation-invariant transformer over track tokens (no track-axis PE).
        track_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=6, dim_feedforward=d_model * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.track_tx = nn.TransformerEncoder(track_layer, num_layers=n_track_layers)

        # v6.1 strip head: gain (1) + EQ (14) + comp-no-makeup-no-knee (4)
        # + clip-drive+mix (2) + pan (1) = 22. No bypass head.
        self.head_track = StripParamHead(d_model, n_params=22)
        self.n_strip_params = 22

        # Global heads operate on mean+max pooled track tokens.
        global_dim = d_model * 2

        # v6.1 bus head: 9 bus EQ + 4 bus comp (no makeup, no knee) = 13.
        self.head_bus = self._mlp_head(global_dim, 13)
        self.n_bus_params = 13

        self.head_trim = TrimHead(global_dim, max_db=trim_max_db)

    @staticmethod
    def _mlp_head(in_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, out_dim),
        )

    def forward(
        self,
        tracks: torch.Tensor,
        track_mask: torch.Tensor,
        ref_mix: Optional[torch.Tensor] = None,
        mert_embeddings: Optional[torch.Tensor] = None,
    ) -> dict:
        B, N_max, C, T = tracks.shape
        device = tracks.device

        if self.use_ref_mix:
            if ref_mix is None:
                ref_mix = torch.zeros(B, 2, T, device=device, dtype=tracks.dtype)
            ref_expanded = ref_mix.unsqueeze(1).expand(B, N_max, 2, T)
            track_in = torch.cat([tracks, ref_expanded], dim=2)
            flat = track_in.reshape(B * N_max, 4, T)
        else:
            flat = tracks.reshape(B * N_max, 2, T)

        pooled, _ = self.track_backbone(flat)
        track_feats = pooled.reshape(B, N_max, self.d_model)

        if self.mert_proj is not None:
            if mert_embeddings is None:
                raise ValueError("mert_dim>0 but mert_embeddings not provided")
            track_feats = track_feats + self.mert_proj(mert_embeddings.to(track_feats.dtype))

        pad_mask = ~track_mask
        track_feats = self.track_tx(track_feats, src_key_padding_mask=pad_mask)

        track_params = self.head_track(track_feats.reshape(B * N_max, self.d_model)).reshape(B, N_max, -1)

        # Zero padded slots' params (they're masked out of the render anyway;
        # downstream consumers should still use track_mask).
        m = track_mask.unsqueeze(-1).to(track_params.dtype)
        track_params = track_params * m

        # Global pool: masked mean + max over track tokens.
        big_neg = -1e4
        masked_for_max = track_feats.masked_fill(pad_mask.unsqueeze(-1), big_neg)
        pooled_max, _ = masked_for_max.max(dim=1)
        m_sum = track_mask.sum(dim=1, keepdim=True).clamp(min=1).to(track_feats.dtype)
        masked_for_mean = track_feats * m
        pooled_mean = masked_for_mean.sum(dim=1) / m_sum
        global_feat = torch.cat([pooled_mean, pooled_max], dim=-1)

        return {
            "track_params": track_params,
            "bus_params": torch.sigmoid(self.head_bus(global_feat)),
            "trim_db": self.head_trim(global_feat),
        }


__all__ = [
    "MelFrontend", "WaveformCNN", "MelTransformer", "HybridBackbone",
    "StripParamHead", "TrimHead", "MixEncoder",
]
