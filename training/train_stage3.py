"""Stage 3 v6.1 — train MixEncoder on real multitrack data with reconstruction loss.

Real-data training. No engineer params available (the dataset only has
`(stems, engineer_mix)` pairs, not the actual chain parameters used by the
human mixer). All learning flows through DiffMixingGraph backward via the
reconstruction loss.

v6.1 vs v6: no bypass heads. "Bypassed" is expressed in the parameter space
itself (EQ gain → 0 dB, comp ratio → 1:1, clip mix → 0%). The bypass-
consistency BCE is gone — it created a discrete-threshold loss landscape and
a self-reinforcing "everything bypassed" basin (the rounds-3-6 oscillation).
Its job — gently preferring "do nothing unless it helps" — is now done by a
smooth `--w-identity-prior` L2 toward the engineer-default param values. The
comp knee and clipper shape/ceiling were also dropped (the recon loss can't
supervise them; hardcoded inside DiffComp / DiffClipper).

Architecture:
  - MixEncoder with `use_ref_mix=False` — predicts params from tracks alone.
    The `ref_mix` is held out as the supervision target, never an input.
  - MERT semantic conditioning (per-track 768-d embedding) drives all the
    per-instrument prior knowledge.
  - Engineer-default initial biases on heads.
  - RMS-relative comp threshold (encoder predicts threshold_offset_db; the
    effective threshold is track_rms_db + offset).

Loss:
  decoupled_recon_loss(pred_mix, engineer_mix, trim_db) — L_timbre on
    loudness-normalized pred + L_loud on the trim head; w_log_mel and
    w_stereo_side fold into L_timbre.
  + w_identity_prior * ||params_norm − engineer_default_norm||² — smooth
    regularizer toward "light, sensible" processing; recon loss overrides.
  + w_pan_mean * pan_mean_penalty(...) — optional |E[pan]| penalty.
    Trivially satisfied by all-tracks-to-center; prefer w_stereo_side.

Usage:
    python training/train_stage3.py \
        --shard-dirs dmc-data/stage3_cambridge_8s dmc-data/stage3_slakh_8s \
        --init-from dmc-data/checkpoints/stage3_v6_round4/mix_encoder_latest.pt \
        --steps 6000 --batch-size 4 --grad-accum-steps 2 --n-max 42 --lr 2e-5 \
        --w-stereo-side 1.5 --w-identity-prior 5e-3 --min-track-rms-dbfs -45
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torchaudio.transforms as T_audio
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.diff_mixing_graph import DiffMixingGraph
from models.diff_strip import compute_track_rms_db
from models.encoders import MixEncoder
from models.encoder_priors import (
    init_encoder_priors, strip_default_norm, bus_default_norm,
)
from reference.param_norm import PARAM_RANGES
from training.data import (
    BUS_PARAM_KEYS, STRIP_PARAM_KEYS,
    collate_stage3, make_stage3_dataset,
)
from training.losses import decoupled_recon_loss, pan_mean_penalty


SAMPLE_RATE = 48_000
N_MAX = 12
MERT_DIM = 768


# ---------- Param denorm tables for the recon path ----------

def _build_denorm_table(device: torch.device, keys: tuple) -> dict:
    los, his, is_log = [], [], []
    for k in keys:
        lo, hi, scale = PARAM_RANGES[k]
        los.append(lo); his.append(hi); is_log.append(scale == "log")
    return {
        "lo": torch.tensor(los, device=device, dtype=torch.float32),
        "hi": torch.tensor(his, device=device, dtype=torch.float32),
        "is_log": torch.tensor(is_log, device=device, dtype=torch.bool),
    }


def _denorm(norm: torch.Tensor, table: dict) -> torch.Tensor:
    lo, hi, is_log = table["lo"], table["hi"], table["is_log"]
    norm = norm.clamp(0.0, 1.0)
    log_lo = torch.log10(lo.clamp(min=1e-9))
    log_hi = torch.log10(hi.clamp(min=1e-9))
    lin = lo + norm * (hi - lo)
    log = torch.pow(10.0, log_lo + norm * (log_hi - log_lo))
    return torch.where(is_log, log, lin)


# ---------- Render path: encoder outputs → DiffMixingGraph kwargs ----------

def _render_full_mix(
    graph: DiffMixingGraph,
    tracks: torch.Tensor, track_mask: torch.Tensor,
    out: dict, tables: dict,
    use_rms_relative_threshold: bool = True,
) -> torch.Tensor:
    """Render predicted full mix from MixEncoder outputs.

    `out` carries `track_params` (B, N_max, 22), `bus_params` (B, 13). No
    bypass — "bypassed" is expressed in param space (EQ gain 0, comp ratio 1,
    clip mix 0). The DiffMixingGraph's bypass args are left at their None
    defaults (no-op).
    """
    B, N_max = tracks.shape[:2]

    # ---- Strip params (22) ----
    strip_phys = _denorm(out["track_params"].float(), tables["strip"])
    strip_params = {k: strip_phys[..., i] for i, k in enumerate(STRIP_PARAM_KEYS)}

    if use_rms_relative_threshold:
        # Reinterpret the encoder's threshold sigmoid as offset in [-30, +6].
        flat = tracks.reshape(B * N_max, 2, tracks.shape[-1]).float()
        track_rms = compute_track_rms_db(flat).view(B, N_max)
        threshold_norm_idx = STRIP_PARAM_KEYS.index("threshold_db")
        threshold_offset_norm = out["track_params"][..., threshold_norm_idx].float().clamp(0, 1)
        offset = -30.0 + threshold_offset_norm * 36.0
        strip_params.pop("threshold_db", None)
        strip_params["threshold_offset_db"] = offset
        strip_params["track_rms_db"] = track_rms

    # ---- Bus params (13) ----
    bus_phys = _denorm(out["bus_params"].float(), tables["bus"])
    bus_params: dict = {}
    for i, k in enumerate(BUS_PARAM_KEYS):
        v = bus_phys[..., i]
        # DiffBusEQ / DiffComp expect un-prefixed keys (low_boost_freq, threshold_db, …).
        bus_params[k[len("bus_"):] if k.startswith("bus_") else k] = v

    return graph(tracks.float(), track_mask, strip_params, bus_params=bus_params)


# ---------- TensorBoard helpers ----------

def _safe_audio_for_tb(x: torch.Tensor, peak_target: float = 0.95) -> torch.Tensor:
    """Normalize a (..., T) audio tensor for TensorBoard add_audio.

    TB clips outside [-1, 1] without ceremony. Predicted mixes from a fresh
    encoder can have arbitrary peaks, so peak-normalize to `peak_target` to
    keep the audio listenable while preserving inter-channel ratio.
    """
    x = x.detach().float().cpu()
    peak = x.abs().max().clamp(min=1e-9)
    return (x / peak * peak_target).clamp(-1.0, 1.0)


def _log_stereo_audio(writer: SummaryWriter, tag: str, audio_2T: torch.Tensor,
                     step: int, sample_rate: int = 48000) -> None:
    """Write a stereo audio summary to TensorBoard via a single audio panel.

    `torch.utils.tensorboard.SummaryWriter.add_audio` hardcodes
    `num_channels=1` (mono), splitting stereo across two players. This
    writer encodes a true stereo WAV using soundfile and constructs the
    audio summary protobuf with `num_channels=2`, giving you a single
    panel per tag with proper L+R playback.
    """
    import io
    import soundfile as sf
    from tensorboard.compat.proto.summary_pb2 import Summary

    if audio_2T.dim() != 2 or audio_2T.shape[0] != 2:
        raise ValueError(f"expected (2, T) stereo, got {audio_2T.shape}")

    audio_np = audio_2T.detach().float().cpu().numpy().T   # (T, 2)
    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()
    audio_proto = Summary.Audio(
        sample_rate=float(sample_rate),
        num_channels=2,
        length_frames=audio_np.shape[0],
        encoded_audio_string=wav_bytes,
        content_type="audio/wav",
    )
    summary = Summary(value=[Summary.Value(tag=tag, audio=audio_proto)])
    writer._get_file_writer().add_summary(summary, global_step=step)


def _log_mel_image(audio: torch.Tensor, sample_rate: int = 48000) -> torch.Tensor:
    """Compute a log-mel spectrogram image (3, H, W) for TB add_image.

    audio: (T,) or (C, T). Returns a normalized 3-channel tensor in [0, 1].
    """
    if audio.dim() == 2:
        audio = audio.mean(dim=0)
    audio = audio.detach().float().cpu()
    mel = T_audio.MelSpectrogram(
        sample_rate=sample_rate, n_fft=2048, hop_length=512, n_mels=128,
        f_min=20.0, f_max=20000.0, power=2.0,
    )(audio.unsqueeze(0)).squeeze(0)
    log_mel = torch.log10(mel + 1e-6)
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-9)
    img = log_mel.flip(0).unsqueeze(0)
    img = img.expand(3, -1, -1).contiguous()
    return img


# ---------- Validation passes ----------

def _val_loss_pass(
    encoder, mix_graph, val_loader_iter, val_batches: int,
    tables: dict, device: torch.device,
    *,
    use_rms_relative_threshold: bool,
    w_stereo_side: float, w_loud: float, w_log_mel: float,
    w_imbalance: float = 0.0, w_width: float = 0.0,
    loudness_target_dbfs: float | None = None,
) -> dict[str, float]:
    """Average losses over `val_batches` random validation batches."""
    encoder.eval()
    accum: dict[str, list[float]] = {
        "L_recon": [], "L_timbre": [], "L_time": [], "L_mss": [], "L_loud": [],
    }
    if w_log_mel > 0:
        accum["L_log_mel"] = []
    if w_stereo_side > 0:
        accum["L_stereo_side"] = []
    if w_imbalance > 0:
        accum["L_imbalance"] = []
    if w_width > 0:
        accum["L_width"] = []
    with torch.no_grad():
        for _ in range(val_batches):
            try:
                batch = next(val_loader_iter)
            except StopIteration:
                break
            tracks = batch["tracks"].to(device, non_blocking=True)
            track_mask = batch["track_mask"].to(device, non_blocking=True)
            ref_mix = batch["mix"].to(device, non_blocking=True)
            mert = batch["mert_embeddings"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = encoder(tracks, track_mask, ref_mix=None, mert_embeddings=mert)
            with torch.amp.autocast("cuda", enabled=False):
                pred_mix = _render_full_mix(
                    mix_graph, tracks, track_mask, out, tables,
                    use_rms_relative_threshold=use_rms_relative_threshold,
                )
                rec = decoupled_recon_loss(
                    pred_mix, ref_mix.float(), out["trim_db"].float(),
                    w_log_mel=w_log_mel, w_loud=w_loud, w_stereo_side=w_stereo_side,
                    w_imbalance=w_imbalance, w_width=w_width,
                    loudness_target_dbfs=loudness_target_dbfs,
                )
            for k in accum:
                if k in rec:
                    accum[k].append(rec[k].item())
    encoder.train()
    return {k: (sum(v) / max(len(v), 1)) for k, v in accum.items()}


def _val_audio_pass(
    encoder, mix_graph, preview_batch: dict,
    tables: dict, device: torch.device,
    *,
    use_rms_relative_threshold: bool, max_examples: int = 4,
) -> list[dict]:
    """Render the fixed preview batch; return per-example audio samples for TB.

    The preview_batch is loaded once at startup and reused across every
    validation pass — so the audio in TB shows the SAME songs improving over
    training steps.
    """
    encoder.eval()
    samples: list[dict] = []
    with torch.no_grad():
        tracks = preview_batch["tracks"].to(device, non_blocking=True)
        track_mask = preview_batch["track_mask"].to(device, non_blocking=True)
        ref_mix = preview_batch["mix"].to(device, non_blocking=True)
        mert = preview_batch["mert_embeddings"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = encoder(tracks, track_mask, ref_mix=None, mert_embeddings=mert)
        with torch.amp.autocast("cuda", enabled=False):
            pred_mix = _render_full_mix(
                mix_graph, tracks, track_mask, out, tables,
                use_rms_relative_threshold=use_rms_relative_threshold,
            )
        # Sum baseline: mask-aware sum of tracks (stereo). Sanity-check
        # reference for "what would the mix be if we did nothing?"
        mask_4d = track_mask.float().view(*track_mask.shape, 1, 1)
        sum_mix = (tracks * mask_4d).sum(dim=1)

        n = min(max_examples, pred_mix.shape[0])
        for i in range(n):
            samples.append({
                "pred_mix": pred_mix[i].detach().cpu(),
                "ref_mix":  ref_mix[i].detach().cpu(),
                "sum_mix":  sum_mix[i].detach().cpu(),
                "session":  preview_batch["meta"][i].get("session", "?"),
                "stage":    preview_batch["meta"][i].get("stage", "?"),
            })
    encoder.train()
    return samples


# ---------- Warm-start ----------

class _EMA:
    """Exponential moving average of model parameters.

    Updated each optimizer step with `decay * ema + (1 - decay) * current`.
    For a 250-step bounce period like round 9's, decay 0.999 (half-life ~693
    optimizer steps) smooths several bounce cycles together — the resulting
    state captures the "central" weight value the SGD is oscillating around,
    not whichever bounce phase the most recent step happens to land on.
    Saved as `mix_encoder_ema.pt` alongside every numbered checkpoint.
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.state = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.state:
                self.state[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    def state_dict(self):
        return self.state


def _load_compatible_state(
    encoder: MixEncoder, ckpt_path: Path,
    *, exclude_prefixes: tuple[str, ...] = (),
) -> int:
    """Best-effort load: copy params where shapes match, skip prefixes.

    `exclude_prefixes`: keys starting with any of these strings are skipped.
    Use to preserve engineer-default biases (init_encoder_priors output) on
    the final linear of each head — without this, the warm-start can carry
    forward a saturating bias (e.g. p2_freq stuck at 10 kHz max) that the
    new training never escapes.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "encoder_state_dict" in ckpt:
        ckpt = ckpt["encoder_state_dict"]
    own = encoder.state_dict()
    loaded = 0
    skipped_by_prefix = 0
    for k, v in ckpt.items():
        if any(k.startswith(p) for p in exclude_prefixes):
            skipped_by_prefix += 1
            continue
        if k in own and own[k].shape == v.shape:
            own[k] = v
            loaded += 1
    encoder.load_state_dict(own, strict=False)
    if skipped_by_prefix:
        print(f"  warm-start: skipped {skipped_by_prefix} tensors via exclude_prefixes "
              f"(preserves engineer-default head biases)")
    return loaded


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dirs", nargs="+", required=True,
                    help="One or more `dmc-data/stage3_*` directories.")
    ap.add_argument("--mert-cache-root", default="dmc-data/mert_cache")
    ap.add_argument("--out-dir", default="dmc-data/checkpoints/stage3_v6")
    ap.add_argument("--init-from", default="",
                    help="Optional MixEncoder checkpoint to warm-start from.")
    ap.add_argument("--preserve-head-priors", action="store_true",
                    help="When warm-starting, skip the FINAL linear layer of "
                         "head_track.params and head_bus — and re-apply "
                         "init_encoder_priors after the load. Counters the case "
                         "where a saturating prior bias "
                         "(e.g. p2_freq pinned at the prior max) carries forward "
                         "and the new training never escapes it.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--n-max", type=int, default=N_MAX)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--prefetch-factor", type=int, default=4,
                    help="DataLoader prefetch_factor for the TRAIN loader: how many "
                         "batches each worker preloads ahead of consumption. PyTorch's "
                         "default (2) is too low for our setup — each step consumes 8 "
                         "samples (B=4 × accum=2) of ~96 MB raw FLAC, and the workers "
                         "can't always have the next batch ready when the GPU finishes "
                         "the previous; the queue runs dry and the GPU idles ~750 ms / "
                         "step (~4 %% wall-time). 4 keeps the queue topped up. Cost: "
                         "RAM scales linearly with prefetch_factor × num_workers. "
                         "Ignored when num_workers=0.")
    ap.add_argument("--grad-accum-steps", type=int, default=1,
                    help="Gradient accumulation: run this many micro-batches "
                         "(each of --batch-size) and accumulate their gradients "
                         "before one optimizer step. Effective batch = "
                         "batch_size * grad_accum_steps. Memory cost is set by "
                         "the MICRO-batch only (the per-micro-batch graph is freed "
                         "after each backward), so this is a free way to lower "
                         "gradient noise / smooth the loss curve when you can't "
                         "afford a bigger --batch-size. NOTE: --steps, --warmup-steps, "
                         "--ckpt-every, --val-every, --log-every all count OPTIMIZER "
                         "steps, not micro-batches.")
    ap.add_argument("--val-num-workers", type=int, default=0,
                    help="DataLoader workers for the val loader. Default 0 — val "
                         "runs every --val-every steps; a persistent worker process "
                         "would idle holding a large WebDataset shuffle reservoir "
                         "for nothing. On unified-memory hardware (e.g. DGX Spark) "
                         "that idle buffer can be the difference between a clean run "
                         "and an NVRM OOM that takes the box down.")
    ap.add_argument("--shuffle-buffer", type=int, default=200,
                    help="WebDataset shuffle reservoir size, applied to BOTH train "
                         "and val datasets. Each DataLoader worker maintains its OWN "
                         "reservoir of raw .tar samples (FLAC bytes), so the real RAM "
                         "bill is `(num_workers + val_num_workers) * shuffle_buffer * "
                         "bytes_per_sample`. 200 is overkill given `shardshuffle=True`; "
                         "32 typically gives indistinguishable decorrelation at ~6× "
                         "less RAM.")
    ap.add_argument("--cuda-memory-fraction", type=float, default=0.0,
                    help="If > 0, cap the fraction of total CUDA memory PyTorch may "
                         "allocate via `torch.cuda.set_per_process_memory_fraction`. "
                         "On unified-memory hardware (DGX Spark / Jetson), the GPU "
                         "and system share one physical pool and an unconstrained "
                         "training run can drive the kernel NVRM into OOM, which "
                         "tends to wedge the driver and reboot the system. A cap "
                         "of 0.7-0.8 leaves headroom for the OS + display + CUDA "
                         "driver bookkeeping.")
    ap.add_argument("--min-track-rms-dbfs", type=float, default=float("-inf"),
                    help="Per-segment activity mask: tracks whose segment-RMS is "
                         "below this dBFS threshold are masked out (track_mask=False) "
                         "in the batch. Silent slots otherwise teach the encoder "
                         "'for this MERT identity, predict identity-ish params' — "
                         "which then poisons inference when the same track (e.g. a "
                         "background vocal) is active. Typical value: -45.0. "
                         "Default -inf disables masking. Inference is unaffected.")
    ap.add_argument("--trim-max-db", type=float, default=12.0,
                    help="±range of the global trim head (TrimHead max_db). The trim "
                         "head absorbs the global output-level adjustment so the "
                         "per-track gains stay free for relative balance. Widen this "
                         "(e.g. 18) if the trim head saturates at its cap (seen when "
                         "the un-trimmed chain runs very quiet and needs a big lift).")
    ap.add_argument("--loudness-target-dbfs", type=float, default=None,
                    help="If set, the trim head is trained to bring the un-trimmed "
                         "predicted mix to this fixed combined-stereo RMS level (dBFS) "
                         "instead of matching each reference mix's idiosyncratic level. "
                         "A consistent target → faster trim-head convergence and a "
                         "predictable rendered-output level. ~-15 dBFS ≈ -14 LUFS for "
                         "typical continuous full-band program; tune to taste. Default "
                         "(None) keeps the match-the-reference behavior.")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)

    # Loss weights
    ap.add_argument("--w-recon", type=float, default=1.0)
    ap.add_argument("--w-loud", type=float, default=0.1,
                    help="weight on the loudness loss (trim_db vs target trim).")
    ap.add_argument("--w-log-mel", type=float, default=0.5,
                    help="weight on log-mel L1 in the timbre loss.")
    ap.add_argument("--w-stereo-side", type=float, default=0.0,
                    help="extra MSS loss on the side channel (L-R)/2 to encourage pan / "
                         "stereo imaging. 0 disables; 1.0-3.0 recommended. Note: "
                         "MAGNITUDE only — catches per-band side mismatch but is "
                         "sign-invariant (can't distinguish left-leaning from right-"
                         "leaning mixes). Pair with --w-imbalance for sign supervision.")
    ap.add_argument("--w-imbalance", type=float, default=0.0,
                    help="weight on stereo-imbalance MSE: (P_L-P_R)/(P_L+P_R) for pred "
                         "vs ref. Scalar, sign-preserving — penalizes left-vs-right "
                         "energy asymmetry directly in the audio domain. Diff-MST "
                         "(Steinmetz 2024) 'stereo imbalance' audio feature. The "
                         "principled replacement for --w-pan-mean: SI puts gradient "
                         "on actual rendered energy distribution, not the pan-knob "
                         "values, and is energy-weighted by construction. 0.5-1.0 "
                         "recommended; 0 disables.")
    ap.add_argument("--w-width", type=float, default=0.0,
                    help="weight on stereo-width log-ratio MSE: P_side/P_mid for pred "
                         "vs ref, log-domain so a 2× too-narrow miss costs the same "
                         "as a 2× too-wide miss. Scalar width target — complements "
                         "the per-band side MSS by constraining overall stereo "
                         "spread independently of spectrum shape. Diff-MST 'stereo "
                         "width' audio feature. 0.1-0.3 recommended; 0 disables.")
    ap.add_argument("--w-pan-mean", type=float, default=0.0,
                    help="penalty on |mean(pan) over active tracks| in PARAM SPACE "
                         "(not audio). Weak — unweighted by track energy, trivially "
                         "satisfied by all-tracks-to-center. Default off. Prefer "
                         "--w-imbalance for L/R-symmetry supervision.")
    ap.add_argument("--w-identity-prior", type=float, default=0.0,
                    help="weight on an L2 penalty pulling predicted params toward the "
                         "engineer-default values (encoder_priors.STRIP_DEFAULTS / "
                         "BUS_DEFAULTS, in normalized [0,1] space). Replaces the old "
                         "bypass-consistency BCE: it's a smooth, well-behaved regularizer "
                         "with no discrete threshold and no self-reinforcing basin — it "
                         "gently prefers 'do something light/sensible' (so a processor "
                         "meant to be off lands at EQ gain 0 / clip mix 0 etc.) but yields "
                         "to the recon loss wherever real processing helps. Recommended: "
                         "1e-3 to 1e-2. 0 disables.")

    ap.add_argument("--no-rms-relative-threshold", action="store_true")

    ap.add_argument("--alternate-datasets", action="store_true",
                    help="With 2+ --shard-dirs, build one WebDataset pipeline per "
                         "dir and round-robin them at the sample level so every "
                         "batch has a fixed ratio of samples per source. Defeats "
                         "the periodic train-loss bouncing seen when the combined "
                         "shuffle reservoir's Cambridge↔Slakh mix drifts (round 9). "
                         "Default off: original combined-shuffle behavior.")
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="If > 0, maintain an exponential moving average of the "
                         "encoder's float parameters (`ema = d * ema + (1-d) * "
                         "current` after each optimizer step) and save it to "
                         "mix_encoder_ema.pt at every --ckpt-every. Smooths over "
                         "training-loop oscillation: the EMA state is the central "
                         "value the SGD is bouncing around, not whichever phase the "
                         "latest weights happen to land on. Recommended: 0.999 "
                         "(~693-step half-life) for our 6000-step / 250-step-bounce "
                         "regime; lower (0.99) for fast-moving runs, higher (0.9999) "
                         "for very long runs. 0 disables.")

    # Bookkeeping
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=500,
                    help="run validation pass every N steps (audio logged to TB)")
    ap.add_argument("--val-batches", type=int, default=4)
    ap.add_argument("--preview-size", type=int, default=4,
                    help="number of FIXED val examples used for audio/spectrogram logging "
                         "across the entire run.")
    ap.add_argument("--tb-logdir", default="",
                    help="default: <out-dir>/tb. SummaryWriter is always created.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if args.cuda_memory_fraction > 0.0 and device.type == "cuda":
        # set_per_process_memory_fraction requires an explicit device index;
        # `torch.device("cuda")` (no index) is rejected.
        cuda_idx = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(
            float(args.cuda_memory_fraction), device=cuda_idx,
        )
        print(f"  cuda memory fraction capped at {args.cuda_memory_fraction:.2f} (cuda:{cuda_idx})")

    print(f"[stage3] training MixEncoder on real multitracks ({len(args.shard_dirs)} sources)")
    print(f"  device:    {device}")
    print(f"  shard_dirs: {args.shard_dirs}")
    print(f"  batch:     {args.batch_size}, n_max: {args.n_max}, "
          f"steps: {args.steps}, lr: {args.lr}")
    print(f"  rms_relative_threshold: {not args.no_rms_relative_threshold}")
    print(f"  trim_max_db: {args.trim_max_db}  loudness_target_dbfs: {args.loudness_target_dbfs}")
    print(f"  loss weights: w_recon={args.w_recon} w_log_mel={args.w_log_mel} "
          f"w_loud={args.w_loud} w_stereo_side={args.w_stereo_side} "
          f"w_imbalance={args.w_imbalance} w_width={args.w_width} "
          f"w_identity_prior={args.w_identity_prior} w_pan_mean={args.w_pan_mean}")
    print(f"  alternate_datasets: {args.alternate_datasets}  ema_decay: {args.ema_decay}")

    encoder = MixEncoder(
        sample_rate=SAMPLE_RATE,
        use_ref_mix=False,
        mert_dim=MERT_DIM,
        trim_max_db=args.trim_max_db,
    ).to(device)
    print(f"  MixEncoder params: {sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M")

    init_encoder_priors(encoder)
    print("  applied engineer-default initial biases")

    if args.init_from:
        head_final_prefixes = (
            "head_track.params.2.",
            "head_bus.2.",
        ) if args.preserve_head_priors else ()
        loaded = _load_compatible_state(
            encoder, Path(args.init_from),
            exclude_prefixes=head_final_prefixes,
        )
        print(f"  warm-started: {loaded} tensors loaded from {args.init_from}")
        if args.preserve_head_priors:
            init_encoder_priors(encoder)
            print("  re-applied engineer-default initial biases (--preserve-head-priors)")

    encoder.train()

    mix_graph = DiffMixingGraph(sample_rate=SAMPLE_RATE).to(device).eval()
    for p in mix_graph.parameters():
        p.requires_grad_(False)

    tables = {
        "strip": _build_denorm_table(device, STRIP_PARAM_KEYS),
        "bus":   _build_denorm_table(device, BUS_PARAM_KEYS),
    }

    # Identity-prior targets: engineer-default value (normalized [0,1]) per
    # param. Pulled toward by --w-identity-prior; the recon loss overrides.
    strip_prior = torch.tensor(strip_default_norm(), device=device, dtype=torch.float32)  # (n_strip,)
    bus_prior = torch.tensor(bus_default_norm(), device=device, dtype=torch.float32)        # (n_bus,)

    train_ds = make_stage3_dataset(
        args.shard_dirs, shuffle=args.shuffle_buffer, split="train",
        max_tracks=args.n_max, repeat=True,
        mert_cache_root=args.mert_cache_root,
        alternate=args.alternate_datasets,
    )
    # prefetch_factor + persistent_workers are only valid when num_workers > 0.
    train_loader_kwargs = {}
    if args.num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = args.prefetch_factor
        train_loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=lambda b: collate_stage3(b, n_max=args.n_max, mert_dim=MERT_DIM,
                                            min_track_rms_dbfs=args.min_track_rms_dbfs),
        drop_last=True, **train_loader_kwargs,
    )

    val_ds = make_stage3_dataset(
        args.shard_dirs, shuffle=args.shuffle_buffer, split="val",
        max_tracks=args.n_max, repeat=True,
        mert_cache_root=args.mert_cache_root,
        alternate=args.alternate_datasets,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=args.val_num_workers,
        collate_fn=lambda b: collate_stage3(b, n_max=args.n_max, mert_dim=MERT_DIM,
                                            min_track_rms_dbfs=args.min_track_rms_dbfs),
        drop_last=True,
    )
    val_iter = iter(val_loader)

    # Build a FIXED preview batch with --preview-size DISTINCT sessions, with
    # at least one example from each `--shard-dirs` source if possible.
    #
    # The val pipeline shuffles raw .tar samples BEFORE filtering by split, so
    # val emissions tend to come in bursts of ~10-20 contiguous segments from
    # the same session (split_for hashes by session, so all of a session's
    # segments land in the same split). A small search budget can land
    # entirely inside one or two such bursts and miss whole datasets — see
    # the round-3 launch where the preview was 2× Slakh and 0× Cambridge.
    n_expected_stages = len(args.shard_dirs)
    preview_examples: list[dict] = []
    seen_sessions: set[str] = set()
    seen_stages: set[str] = set()
    # Generous budget — val emissions are sparse and clumpy; spending a few
    # extra seconds at startup is far cheaper than a useless preview set.
    max_search_batches = max(64, args.preview_size * 32)
    for _ in range(max_search_batches):
        have_size = len(preview_examples) >= args.preview_size
        have_all_stages = len(seen_stages) >= n_expected_stages
        if have_size and have_all_stages:
            break
        try:
            b = next(val_iter)
        except StopIteration:
            break
        for i in range(b["tracks"].shape[0]):
            sess = b["meta"][i].get("session", f"_unknown_{len(preview_examples)}")
            stage = b["meta"][i].get("stage", "") or ""
            if sess in seen_sessions:
                continue
            # If we already have preview_size examples but are missing a stage,
            # only accept further examples from a not-yet-seen stage.
            if (len(preview_examples) >= args.preview_size
                    and stage in seen_stages):
                continue
            seen_sessions.add(sess)
            if stage:
                seen_stages.add(stage)
            preview_examples.append({
                "tracks":          b["tracks"][i],
                "track_mask":      b["track_mask"][i],
                "mix":             b["mix"][i],
                "mert_embeddings": b["mert_embeddings"][i],
                "meta":            b["meta"][i],
            })
            if (len(preview_examples) >= args.preview_size
                    and len(seen_stages) >= n_expected_stages):
                break

    if len(seen_stages) < n_expected_stages:
        print(f"  WARNING: preview set covers {len(seen_stages)}/{n_expected_stages} "
              f"shard sources after {max_search_batches} batches "
              f"(saw stages: {sorted(seen_stages)}). Bump --preview-size or check splits.")

    if not preview_examples:
        raise RuntimeError("Could not build preview set from val_loader; check splits.")

    n_preview = len(preview_examples)
    # When the dataset mixes shards prepared with different segment lengths
    # (e.g. cambridge=15 s, slakh=6 s), each batch can have a different T —
    # so the per-example tensors collected here may not match in time. Pad
    # everything to the longest T in the preview set before stacking.
    preview_T = max(e["tracks"].shape[-1] for e in preview_examples)

    def _pad_T(x: torch.Tensor, T: int) -> torch.Tensor:
        if x.shape[-1] == T:
            return x
        pad = torch.zeros(*x.shape[:-1], T - x.shape[-1], dtype=x.dtype)
        return torch.cat([x, pad], dim=-1)

    preview_batch = {
        "tracks":          torch.stack([_pad_T(e["tracks"], preview_T) for e in preview_examples]),
        "track_mask":      torch.stack([e["track_mask"] for e in preview_examples]),
        "mix":             torch.stack([_pad_T(e["mix"], preview_T) for e in preview_examples]),
        "mert_embeddings": torch.stack([e["mert_embeddings"] for e in preview_examples]),
        "meta":            [e["meta"] for e in preview_examples],
    }
    preview_sessions = [m.get("session", "?") for m in preview_batch["meta"]]
    preview_stages = [m.get("stage", "?") for m in preview_batch["meta"]]
    print(f"  preview set ({n_preview} unique-session examples):")
    for sess, stg in zip(preview_sessions, preview_stages):
        print(f"    {stg}: {sess}")

    tb_dir = Path(args.tb_logdir) if args.tb_logdir else out_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(log_dir=str(tb_dir))
    print(f"  tensorboard logdir: {tb_dir}")

    # Log preview-set ground-truth audio at step 0 so the user has a baseline.
    n_preview_examples = preview_batch["tracks"].shape[0]
    for i in range(n_preview_examples):
        ref_i = _safe_audio_for_tb(preview_batch["mix"][i])
        mask = preview_batch["track_mask"][i].float().view(-1, 1, 1)
        sum_i = (preview_batch["tracks"][i] * mask).sum(dim=0)
        sum_i = _safe_audio_for_tb(sum_i)
        _log_stereo_audio(tb, f"audio/preview_{i}/ref", ref_i, 0, SAMPLE_RATE)
        _log_stereo_audio(tb, f"audio/preview_{i}/sum", sum_i, 0, SAMPLE_RATE)
        tb.add_image(f"specgram/preview_{i}/ref",
                     _log_mel_image(ref_i, SAMPLE_RATE), 0)
        tb.add_image(f"specgram/preview_{i}/sum",
                     _log_mel_image(sum_i, SAMPLE_RATE), 0)
        tb.add_text(f"preview/{i}/session",
                    f"{preview_batch['meta'][i].get('stage', '?')}: "
                    f"{preview_batch['meta'][i].get('session', '?')}", 0)
    tb.flush()

    ema = _EMA(encoder, decay=args.ema_decay) if args.ema_decay > 0.0 else None
    if ema is not None:
        print(f"  EMA enabled (decay={args.ema_decay}) -> mix_encoder_ema.pt saved at every ckpt")

    optimizer = AdamW(encoder.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95))

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    accum = max(1, args.grad_accum_steps)
    pbar = tqdm(total=args.steps, desc="stage3")
    losses_acc: dict[str, list[float]] = {}
    step = 0
    micro_in_window = 0
    window_losses: dict[str, list[float]] = {}
    # Track the best validation L_recon so a usable artifact survives even if
    # the run ends mid-oscillation (bypass-mask regime switches make the
    # last-step weights a coin flip — see runs 3-6). L_recon (timbre + loud)
    # is the reconstruction-quality objective; the bypass-consistency BCE is
    # a regularizer, not a quality metric, so it's excluded from the criterion.
    best_val_recon = float("inf")
    best_val_step = -1
    t0 = time.time()

    for batch in train_loader:
        if step >= args.steps:
            break

        # Start of an accumulation window: set LR, zero grads, reset accumulators.
        if micro_in_window == 0:
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            optimizer.zero_grad(set_to_none=True)
            window_losses = {}

        tracks = batch["tracks"].to(device, non_blocking=True)
        track_mask = batch["track_mask"].to(device, non_blocking=True)
        ref_mix = batch["mix"].to(device, non_blocking=True)
        mert = batch["mert_embeddings"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = encoder(tracks, track_mask, ref_mix=None, mert_embeddings=mert)

        with torch.amp.autocast("cuda", enabled=False):
            pred_mix = _render_full_mix(
                mix_graph, tracks, track_mask, out, tables,
                use_rms_relative_threshold=not args.no_rms_relative_threshold,
            )
            rec = decoupled_recon_loss(
                pred_mix, ref_mix.float(), out["trim_db"].float(),
                w_log_mel=args.w_log_mel, w_loud=args.w_loud,
                w_stereo_side=args.w_stereo_side,
                w_imbalance=args.w_imbalance, w_width=args.w_width,
                loudness_target_dbfs=args.loudness_target_dbfs,
            )

        l_total = args.w_recon * rec["L_recon"]

        # Optional pan-mean penalty (use sparingly — trivially satisfied by
        # all-tracks-to-center; w_stereo_side is the principled lever).
        if args.w_pan_mean > 0.0:
            pan_idx = STRIP_PARAM_KEYS.index("pan")
            pan_phys = 2.0 * out["track_params"][..., pan_idx].float() - 1.0
            l_pan = pan_mean_penalty(pan_phys, track_mask)
            l_total = l_total + args.w_pan_mean * l_pan
            window_losses.setdefault("L_pan_mean", []).append(l_pan.item())

        # Identity prior: smooth L2 pulling params toward the engineer-default
        # values. Replaces the old bypass-consistency BCE — no discrete
        # threshold, no self-reinforcing "everything bypassed" basin. Strip
        # term is averaged over active tracks only.
        if args.w_identity_prior > 0.0:
            with torch.amp.autocast("cuda", enabled=False):
                tp = out["track_params"].float()                       # (B, N, n_strip)
                m = track_mask.unsqueeze(-1).float()                   # (B, N, 1)
                diff_strip = (tp - strip_prior.view(1, 1, -1)) * m
                denom = m.sum().clamp(min=1.0) * tp.shape[-1]
                l_id_strip = diff_strip.pow(2).sum() / denom
                bp_ = out["bus_params"].float()                        # (B, n_bus)
                l_id_bus = (bp_ - bus_prior.view(1, -1)).pow(2).mean()
                l_id = l_id_strip + l_id_bus
            l_total = l_total + args.w_identity_prior * l_id
            window_losses.setdefault("L_identity", []).append(l_id.item())

        # Scale by 1/accum so the accumulated gradient equals the mean over
        # the window — equivalent to a single backward on the effective batch.
        scaler.scale(l_total / accum).backward()

        # Record this micro-batch's loss values for the window mean.
        window_losses.setdefault("L_total", []).append(l_total.item())
        window_losses.setdefault("L_recon", []).append(rec["L_recon"].item())
        window_losses.setdefault("L_time", []).append(rec["L_time"].item())
        window_losses.setdefault("L_mss", []).append(rec["L_mss"].item())
        window_losses.setdefault("L_timbre", []).append(rec["L_timbre"].item())
        window_losses.setdefault("L_loud", []).append(rec["L_loud"].item())
        if "L_log_mel" in rec:
            window_losses.setdefault("L_log_mel", []).append(rec["L_log_mel"].item())
        if "L_stereo_side" in rec:
            window_losses.setdefault("L_stereo_side", []).append(rec["L_stereo_side"].item())
        if "L_imbalance" in rec:
            window_losses.setdefault("L_imbalance", []).append(rec["L_imbalance"].item())
        if "L_width" in rec:
            window_losses.setdefault("L_width", []).append(rec["L_width"].item())
        with torch.no_grad():
            trim = out["trim_db"].float()
            window_losses.setdefault("trim_db_mean", []).append(trim.mean().item())
            window_losses.setdefault("trim_db_absmean", []).append(trim.abs().mean().item())

        micro_in_window += 1
        if micro_in_window < accum:
            continue  # accumulate more micro-batches before stepping

        # --- end of accumulation window: one optimizer step ---
        micro_in_window = 0
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(encoder)

        # Per-(optimizer-)step log = mean of the window's micro-batch values.
        per_step_log = {k: (sum(v) / len(v)) for k, v in window_losses.items()}
        for k, v in per_step_log.items():
            losses_acc.setdefault(k, []).append(v)

        if step % args.log_every == 0:
            recent = {k: (sum(v[-args.log_every:]) / max(1, len(v[-args.log_every:])))
                      for k, v in losses_acc.items() if v}
            pbar.set_postfix({k: f"{v:.4f}" for k, v in recent.items()})
            for k, v in recent.items():
                tb.add_scalar(f"train/{k}", v, step)
            tb.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)

        # Periodic validation pass.
        if args.val_every > 0 and (step + 1) % args.val_every == 0:
            val_means = _val_loss_pass(
                encoder, mix_graph, val_iter, args.val_batches,
                tables, device,
                use_rms_relative_threshold=not args.no_rms_relative_threshold,
                w_stereo_side=args.w_stereo_side,
                w_imbalance=args.w_imbalance, w_width=args.w_width,
                w_loud=args.w_loud, w_log_mel=args.w_log_mel,
                loudness_target_dbfs=args.loudness_target_dbfs,
            )
            for k, v in val_means.items():
                tb.add_scalar(f"val/{k}", v, step)

            val_recon = val_means.get("L_recon", float("inf"))
            if val_recon < best_val_recon:
                best_val_recon = val_recon
                best_val_step = step + 1
                torch.save({"step": step + 1, "val_recon": val_recon,
                            "encoder_state_dict": encoder.state_dict(),
                            "args": vars(args)},
                           out_dir / "mix_encoder_best.pt")
                tb.add_scalar("val/best_L_recon", best_val_recon, step)
                print(f"  [val] new best L_recon={val_recon:.4f} @ step {step+1} "
                      f"-> mix_encoder_best.pt")

            preview_samples = _val_audio_pass(
                encoder, mix_graph, preview_batch,
                tables, device,
                use_rms_relative_threshold=not args.no_rms_relative_threshold,
                max_examples=args.preview_size,
            )
            for i, sample in enumerate(preview_samples):
                pred = _safe_audio_for_tb(sample["pred_mix"])
                ref = _safe_audio_for_tb(sample["ref_mix"])
                sum_ = _safe_audio_for_tb(sample["sum_mix"])
                _log_stereo_audio(tb, f"audio/preview_{i}/pred", pred, step, SAMPLE_RATE)
                _log_stereo_audio(tb, f"audio/preview_{i}/ref",  ref,  step, SAMPLE_RATE)
                _log_stereo_audio(tb, f"audio/preview_{i}/sum",  sum_, step, SAMPLE_RATE)
                tb.add_image(f"specgram/preview_{i}/pred", _log_mel_image(pred, SAMPLE_RATE), step)
                tb.add_image(f"specgram/preview_{i}/ref",  _log_mel_image(ref,  SAMPLE_RATE), step)
                tb.add_image(f"specgram/preview_{i}/sum",  _log_mel_image(sum_, SAMPLE_RATE), step)
            tb.flush()

        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
            torch.save({"step": step + 1, "encoder_state_dict": encoder.state_dict(),
                        "args": vars(args)},
                       out_dir / f"mix_encoder_step{step+1:08d}.pt")
            torch.save(encoder.state_dict(), out_dir / "mix_encoder_latest.pt")
            if ema is not None:
                torch.save({"step": step + 1, "encoder_state_dict": ema.state_dict(),
                            "ema_decay": args.ema_decay, "args": vars(args)},
                           out_dir / "mix_encoder_ema.pt")

        pbar.update(1)
        step += 1
    pbar.close()

    elapsed = time.time() - t0
    summary = {
        "steps": step,
        "elapsed_s": round(elapsed, 1),
        "final_losses": {k: (sum(v[-args.log_every:]) / max(1, len(v[-args.log_every:])))
                         for k, v in losses_acc.items() if v},
        "best_val_recon": (best_val_recon if best_val_step >= 0 else None),
        "best_val_step": (best_val_step if best_val_step >= 0 else None),
        "args": vars(args),
    }
    with open(out_dir / "stage3_train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    tb.close()
    if best_val_step >= 0:
        print(f"\n[stage3] done. {step} steps in {elapsed:.1f}s. "
              f"best val L_recon={best_val_recon:.4f} @ step {best_val_step} "
              f"(mix_encoder_best.pt)")
    else:
        print(f"\n[stage3] done. {step} steps in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
