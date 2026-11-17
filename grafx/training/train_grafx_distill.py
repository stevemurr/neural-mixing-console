"""Distillation trainer: encoder ≈ grafx-prune's per-song optimizer.

Goal: a feedforward encoder that, given a 6–12s audio window of a song,
predicts the song-global label that grafx-prune would have produced for
that song. Amortizes the slow per-song teacher into a single forward pass
suitable for realtime DAW use.

Key differences from `train_grafx_multi.py`:
  - Two windows from the same song per training slot. Encoder sees both
    in one forward (stacked into 2B). Enables a consistency loss that
    pushes windows of the same song to predict the same label — directly
    trains for the realtime DAW stability requirement (stable knobs as
    the window slides).
  - No L_recon-vs-engineer. The teacher is the spec; the engineer mix is
    only relevant insofar as the teacher tried to match it (and it has
    already, when it produced the labels we train against).
  - Distillation in audio output space: `L_distill_audio` is the MR-STFT
    between the student's rendered mix and the TEACHER's rendered mix
    (graph(teacher_params, stems)), not the engineer mix. Invariant to
    which local minimum the teacher happened to land in for a given
    song — relevant because grafx-prune is non-deterministic.
  - L_param is whitened: per-(proc, param) std is computed once at
    startup over all labels and used as the loss weight, so the EQ's
    1024-dim log_magnitude no longer dominates the comp's 4 scalars.
  - Stems-only encoder (use_ref_mix=False). At inference in a DAW there
    is no engineer reference mix, so the student is trained without one.
    (The teacher saw the mix when it produced the labels — that gap is
    fixed; we just don't make it worse by training with privileged info.)

Defaults trade memory for two-windows-per-song: audio_len=180_000 (6s),
n_max=20, batch_size=2 — yields 2*2=4 forward examples per step, each
~20 stems @ 6s. Expected peak ~80GB on the GB10. Bump audio_len to 270k
(9s) only if you drop batch_size to 1 or n_max to 16.

Run:
    uv run python training/train_grafx_distill.py \\
        --staging-dir dmc-data/grafx-prune-data \\
        --run-name distill_v1 \\
        --max-steps 20000 --batch-size 2 --audio-len 180000 --n-max 20 \\
        --w-param 1.0 --w-cons 0.5 --w-distill 0.5

Ckpts under `dmc-data/checkpoints/grafx_distill_<run_name>/`.
TB logs under `runs/grafx_distill_<run_name>/`.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.grafx_console import GrafxMixingConsole
from models.encoders_grafx import MixEncoderGrafx
from training.data_grafx import (
    LabelStore, attach_grafx_labels, compute_label_whiten_weights,
)
from training.losses_grafx import (
    AudioFeatureLoss,
    HybridReconLoss,
    grafx_intermediate_audio_loss,
    grafx_param_huber_loss,
    grafx_param_consistency_loss,
    make_grafx_prune_recon_loss,
)
from training.train_grafx_multi import (
    _list_staged_sessions, _load_session, _example_from_session,
    collate_multi,
)


# ---------- Paired dataset ----------

class PairedMultiSongDataset(IterableDataset):
    """Yields two random windows from the same session per __next__.

    Output: {"a": example, "b": example} where each example is the same
    shape as `MultiSongDataset` produces. Both halves share the same
    `meta["session"]` and (by construction of the same session) the same
    set of stems / group assignments — only the temporal slice differs.

    Pair collate stacks B such pairs into a single batch of 2B examples,
    indices [0:B] holding the "a" halves and [B:2B] holding the "b"
    halves. This lets the encoder forward once for both halves and the
    trainer split predictions for the consistency loss without a second
    forward pass.
    """

    def __init__(
        self,
        sessions: list[str],
        staging_dir: Path,
        label_store: LabelStore,
        audio_len: int,
        n_max: int,
        seed: int = 0,
        cache_size: int = 4,
        min_window_gap: int = 0,
        pairs_per_session: int = 1,
    ):
        super().__init__()
        if not sessions:
            raise ValueError("no sessions provided")
        self.sessions = list(sessions)
        self.staging_dir = Path(staging_dir)
        self.label_store = label_store
        self.audio_len = audio_len
        self.n_max = n_max
        self.seed = seed
        self.cache_size = cache_size
        self.min_window_gap = min_window_gap  # samples between the two window starts
        self.pairs_per_session = max(1, int(pairs_per_session))

    def __iter__(self):
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

            # Yield `pairs_per_session` pair-windows from this cached session
            # before picking another. Amortizes the per-session disk load
            # (~1.8 GB decoded WAV for a 30-stem session) over many training
            # iterations — without it, one load supports a single yield and
            # the workers can't keep up with compute. K=1 reproduces the
            # original "one pair per load" behavior. Adjacent batches within
            # the K-block are correlated (same song, different windows); the
            # encoder's permutation invariance and AdamW's momentum smoothing
            # make that benign for K in the small-tens range.
            for _ in range(self.pairs_per_session):
                # Pick two start positions for the same song. If
                # min_window_gap > 0, enforce a minimum separation between
                # window starts so the two halves see meaningfully
                # different audio (the consistency loss signal degrades to
                # zero if both halves are identical).
                for _attempt in range(8):
                    a = rng.randint(0, T_total - self.audio_len - 1)
                    b = rng.randint(0, T_total - self.audio_len - 1)
                    if abs(a - b) >= self.min_window_gap:
                        break

                ex_a = _example_from_session(data, a, self.audio_len, self.n_max)
                ex_b = _example_from_session(data, b, self.audio_len, self.n_max)
                yield {"a": ex_a, "b": ex_b}


def w_distill_at(step: int, warmup: int, target: float) -> float:
    """v6/v7 step-piecewise ramp: 0 for steps<warmup, linear ramp to
    `target` over the next `warmup` steps, then constant. Kept for
    backward compat with the `--distill-warmup` flag.
    """
    if warmup <= 0:
        return target
    if step < warmup:
        return 0.0
    if step < 2 * warmup:
        return target * (step - warmup) / max(1, warmup)
    return target


def w_cosine_ramp_at(step: int, ramp_steps: int, target: float) -> float:
    """Cosine half-wave ramp from 0 to `target` over `ramp_steps` steps.

    `0.5 * (1 - cos(pi * progress))` — same shape used for LR cosine
    warmup. Smooth derivative everywhere (no kinks), so Adam's second-
    moment estimate adapts gradually rather than seeing an abrupt
    activation spike at warmup-end (which is what v7's step-piecewise
    ramp produced at its `distill_warmup` boundary, visible as the val
    `L_distill` spike at step 2000). Applies to any single loss weight.
    """
    if ramp_steps <= 0:
        return target
    if step >= ramp_steps:
        return target
    progress = step / ramp_steps
    return target * 0.5 * (1.0 - math.cos(math.pi * progress))


def collate_pairs(batch: list[dict]) -> dict:
    """Stack B pairs into a 2B batch, [0:B] = a halves, [B:2B] = b halves."""
    a = collate_multi([item["a"] for item in batch])
    b = collate_multi([item["b"] for item in batch])
    return {
        "tracks":          torch.cat([a["tracks"],          b["tracks"]],          dim=0),
        "mix":             torch.cat([a["mix"],             b["mix"]],             dim=0),
        "track_mask":      torch.cat([a["track_mask"],      b["track_mask"]],      dim=0),
        "mert_embeddings": torch.cat([a["mert_embeddings"], b["mert_embeddings"]], dim=0),
        "meta": a["meta"] + b["meta"],
        "B_pair": len(batch),
    }


# ---------- Helpers ----------

def _split_params(
    params: dict[str, dict[str, torch.Tensor]],
    B_pair: int,
) -> tuple[dict, dict]:
    """Split a (2B, ...) param dict into the two (B, ...) halves."""
    a = {proc: {p: t[:B_pair] for p, t in pp.items()} for proc, pp in params.items()}
    b = {proc: {p: t[B_pair:] for p, t in pp.items()} for proc, pp in params.items()}
    return a, b


def _to_device_params(
    params: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        proc: {p: t.to(device, non_blocking=True) for p, t in pp.items()}
        for proc, pp in params.items()
    }


def _params_to_fp32(
    params: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Cast every leaf tensor in a strip/group param dict to float32.

    Used when `--use-bf16` is on: the encoder runs under autocast(bfloat16),
    its output dict tensors come out in bf16, and grafx's FFT-based DSP
    processors (`view_as_complex` in MultitapDelay, the STFT in reverb)
    reject bf16. We cast back to fp32 at the encoder/console boundary so
    the console + losses stay numerically identical to the fp32 run.
    """
    return {
        proc: {p: t.float() for p, t in pp.items()}
        for proc, pp in params.items()
    }


def _encoder_forward(
    encoder, tracks, track_mask, group_idx, n_groups, use_bf16: bool,
) -> dict:
    """Run encoder; cast its dict outputs to fp32 if running under autocast."""
    if use_bf16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_raw = encoder(tracks, track_mask, group_idx, n_groups=n_groups)
        return {
            "strip_params": _params_to_fp32(out_raw["strip_params"]),
            "group_params": _params_to_fp32(out_raw["group_params"]),
            "group_assignments": out_raw["group_assignments"],
        }
    return encoder(tracks, track_mask, group_idx, n_groups=n_groups)


# ---------- Validation ----------

@torch.no_grad()
def run_val(
    encoder, console, recon_loss_fn,
    val_sessions: list[str],
    staging_dir: Path,
    label_store: LabelStore,
    audio_len: int,
    n_max: int,
    max_groups: int,
    device: torch.device,
    whiten: dict | None = None,
    strip_schema: dict | None = None,
    group_schema: dict | None = None,
    use_bf16: bool = False,
    audio_total_key: str = "match/full",
    compute_recon: bool = False,
    gap_loss_fn: torch.nn.Module | None = None,
) -> dict[str, float]:
    """Validation: for each val session, predict at two fixed windows
    (25% and 75% in) and report L_param, L_consistency, L_distill_audio,
    L_recon, and the amortization gap.

    `audio_total_key` selects which key from the loss dict to record as
    the scalar L_distill / L_recon total — "match/full" for MR-STFT,
    "af/total" for AudioFeatureLoss. `compute_recon` enables the
    student-vs-engineer-mix loss when w_recon > 0 in training.

    Amortization-gap metric (per session, mean across sessions):
        gap_i = MR-STFT(student_render, engineer_mix) - teacher_final_test_loss_i
    The teacher's `final_test_loss` is in MR-STFT units, so we always
    compute the student side via `gap_loss_fn` (MR-STFT) regardless of
    which loss family the run trains under. Sign: positive = student is
    worse than the teacher's recorded test loss (room to grow); zero =
    student matches teacher's ceiling; negative = student exceeds the
    teacher (rare; bounded by the teacher being an imperfect amortizer
    in the first place).
    """
    encoder.eval()
    L_p, L_c, L_d, L_r, gaps = [], [], [], [], []
    for session in val_sessions:
        try:
            data = _load_session(session, staging_dir, label_store)
        except Exception as e:
            logging.warning(f"val: failed to load {session}: {e}")
            continue
        T_total = data["T_total"]
        if T_total < audio_len + 1:
            continue
        start_a = max(0, T_total // 4)
        start_b = max(0, min(T_total - audio_len - 1, (3 * T_total) // 4))
        ex_a = _example_from_session(data, start_a, audio_len, n_max)
        ex_b = _example_from_session(data, start_b, audio_len, n_max)
        batch = collate_pairs([{"a": ex_a, "b": ex_b}])
        batch = attach_grafx_labels(
            batch, label_store, n_max=n_max, max_groups=max_groups,
            strip_schema=strip_schema, group_schema=group_schema,
        )

        tracks = batch["tracks"].to(device, non_blocking=True)
        track_mask = batch["track_mask"].to(device, non_blocking=True)
        group_idx = batch["group_assignments"].to(device, non_blocking=True)
        # group_assignments is (2B, N) but came from B_pair=1 a/b halves —
        # both halves have the same assignments (same session), so this
        # is redundant rather than wrong.

        out = _encoder_forward(
            encoder, tracks, track_mask, group_idx,
            n_groups=max_groups, use_bf16=use_bf16,
        )
        strip_a, strip_b = _split_params(out["strip_params"], 1)
        group_a, group_b = _split_params(out["group_params"], 1)

        # L_param vs labels (collate_pairs already produced 2 entries —
        # one per window — so labels are already 2-wide; no tile.)
        label_strip = _to_device_params(batch["label_strip_params"], device)
        label_group = _to_device_params(batch["label_group_params"], device)
        if batch["label_example_mask"][0].item():
            L_param = grafx_param_huber_loss(
                out["strip_params"], out["group_params"],
                label_strip, label_group,
                batch["label_track_mask"].to(device),
                batch["n_groups_per_example"].to(device),
                batch["label_example_mask"].to(device),
                delta=1.0,
                param_weights=whiten,
            )
            L_p.append(L_param["L_param/total"].item())

        # L_consistency between the two halves
        L_cons = grafx_param_consistency_loss(
            strip_a, group_a, strip_b, group_b,
            batch["track_mask"][:1].to(device),
            batch["n_groups_per_example"][:1].to(device),
        )
        L_c.append(L_cons["L_cons/total"].item())

        # Render BOTH 'a' and 'b' halves, average per-session — cuts
        # within-session window-to-window noise by ~sqrt(2) at ~2x val
        # cost. The encoder forward already produced predictions for
        # both windows (it sees the full pair); previously we discarded
        # the 'b' half's audio render to save compute. Re-enabled
        # 2026-05-27 after v11's step-2000 val showed a noisy 1-window
        # gap of 1.97 against step-1500's 1.01 — the variance was
        # masking the actual training trajectory.
        labels_full = label_store.get(session) if compute_recon else None
        tloss = None
        if labels_full is not None:
            tloss = (labels_full.get("training_meta") or {}).get("final_test_loss")
        ld_halves, lr_halves, gap_halves = [], [], []

        for hi, (sp_h, gp_h) in enumerate([(strip_a, group_a),
                                            (strip_b, group_b)]):
            tracks_h = tracks[hi:hi+1]
            track_mask_h = track_mask[hi:hi+1]
            group_idx_h = group_idx[hi:hi+1]
            mix_ref_h = batch["mix"][hi:hi+1].to(device, non_blocking=True)
            student_mix = console(
                tracks_h, sp_h, gp_h, group_idx_h,
                track_mask=track_mask_h, n_groups=max_groups,
            )
            if batch["label_example_mask"][hi].item():
                label_strip_h = {proc: {p: t[hi:hi+1] for p, t in pp.items()}
                                 for proc, pp in label_strip.items()}
                label_group_h = {proc: {p: t[hi:hi+1] for p, t in pp.items()}
                                 for proc, pp in label_group.items()}
                teacher_mix = console(
                    tracks_h, label_strip_h, label_group_h, group_idx_h,
                    track_mask=track_mask_h, n_groups=max_groups,
                )
                ld_halves.append(
                    recon_loss_fn(student_mix, teacher_mix)[audio_total_key].item()
                )
            if compute_recon:
                lr_halves.append(
                    recon_loss_fn(student_mix, mix_ref_h)[audio_total_key].item()
                )
                # Amort gap is always MR-STFT regardless of training loss
                # family, so it stays comparable across run designs.
                if gap_loss_fn is not None and tloss is not None:
                    student_mrstft = gap_loss_fn(student_mix, mix_ref_h)["match/full"].item()
                    gap_halves.append(student_mrstft - float(tloss))

        if ld_halves:
            L_d.append(sum(ld_halves) / len(ld_halves))
        if lr_halves:
            L_r.append(sum(lr_halves) / len(lr_halves))
        if gap_halves:
            gaps.append(sum(gap_halves) / len(gap_halves))

    encoder.train()
    return {
        "val/L_param":   sum(L_p) / len(L_p) if L_p else float("nan"),
        "val/L_cons":    sum(L_c) / len(L_c) if L_c else float("nan"),
        "val/L_distill": sum(L_d) / len(L_d) if L_d else float("nan"),
        "val/L_recon":   sum(L_r) / len(L_r) if L_r else float("nan"),
        "val/amort_gap": sum(gaps) / len(gaps) if gaps else float("nan"),
        "val/n_labeled": float(len(L_p)),
        "val/n_total":   float(len(L_c)),
    }


# ---------- Main ----------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--staging-dir", default="dmc-data/grafx-prune-data")
    ap.add_argument("--run-name", default=None,
                    help="Used in output paths. Defaults to a timestamp.")
    ap.add_argument("--out-dir", default=None,
                    help="Default dmc-data/checkpoints/grafx_distill_<run_name>")
    ap.add_argument("--tb-logdir", default=None,
                    help="Default runs/grafx_distill_<run_name>")
    ap.add_argument("--device", default=None)

    # Data
    ap.add_argument("--sample-rate", type=int, default=48_000)
    ap.add_argument("--audio-len", type=int, default=288_000,
                    help="Samples per window (default 6s @ 48kHz). The DAW "
                         "deployment buffer is 6–12s; matching the training "
                         "window length closes the train/test gap.")
    ap.add_argument("--n-max", type=int, default=20,
                    help="Max stems per session. Lower than the multi "
                         "trainer's 32 because pair-windows double the "
                         "effective batch at the same memory cost.")
    ap.add_argument("--max-groups", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--min-window-gap", type=int, default=48_000,
                    help="Minimum separation (in samples) between the two "
                         "window starts of a pair. 48k = 1s @ 48kHz so the "
                         "two halves see meaningfully different audio.")
    ap.add_argument("--max-label-loss", type=float, default=None,
                    help="Drop labeled train sessions whose recorded "
                         "grafx-prune final_test_loss exceeds this. See "
                         "train_grafx_multi.py for behavior.")
    ap.add_argument("--cache-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--pairs-per-session", type=int, default=8,
                    help="How many pair-windows to yield from a cached "
                         "session before picking another. Amortizes the "
                         "per-session disk load (~1.8 GB decoded WAV) "
                         "over training iterations. K=1 reproduces the "
                         "original one-pair-per-load behavior — produces "
                         "a stepwise GPU-util pattern because workers "
                         "can't keep up with compute. K=8 was the empirical "
                         "sweet spot before this comment was written; "
                         "raise if data loading still bottlenecks, lower "
                         "if you want per-batch session diversity.")

    # Model
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-track-layers", type=int, default=4)
    ap.add_argument("--init-from", default=None,
                    help="Path to a previous encoder_*.pt to warm-start from.")

    # Optim
    ap.add_argument("--batch-size", type=int, default=2,
                    help="Number of song pairs per step. Effective forward "
                         "batch is 2 × this (one entry per window).")
    ap.add_argument("--max-steps", type=int, default=20_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-warmup-steps", type=int, default=500)
    ap.add_argument("--lr-cosine-min", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=20.0,
                    help="Pre-step grad-norm clip. v2 used 5.0; this run "
                         "raised to 20 because the typical post-accum grad "
                         "norm sits around 200–300 in whitened-loss units "
                         "and clip=5 forced unit-direction updates that "
                         "alternated basins.")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="Number of micro-batches accumulated per optimizer "
                         "step. Each micro-batch is a full forward + backward "
                         "with `loss / grad_accum`. Effective batch size is "
                         "batch_size * grad_accum. Reduces per-step gradient "
                         "variance without raising peak memory. All other "
                         "step-count CLI args (max-steps, lr-warmup-steps, "
                         "distill-warmup, val-every, ckpt-every, log-every) "
                         "are in OPTIMIZER steps, not micro-batches.")

    # Loss weights
    ap.add_argument("--w-param",   type=float, default=1.0)
    ap.add_argument("--w-cons",    type=float, default=0.5,
                    help="Window-consistency weight. Higher → stabler knobs "
                         "across DAW rolling windows; too high → encoder "
                         "regresses toward a single song-independent label.")
    ap.add_argument("--w-distill", type=float, default=0.5,
                    help="Audio distillation weight (student mix vs teacher "
                         "mix on same stems). Invariant to teacher non-"
                         "determinism. Costs one extra console forward per "
                         "step (no-grad on the teacher side).")
    ap.add_argument("--distill-warmup", type=int, default=5000,
                    help="Steps of param+cons-only training before "
                         "L_distill_audio ramps in. 0 means start "
                         "immediately. Default 5000 mirrors the recon "
                         "warmup in train_grafx_multi.py — the diff-graph "
                         "backward produces gradients ~1000x the param-loss "
                         "scale before the model has settled, so phasing "
                         "the loss in avoids the encoder being dominated "
                         "by clipped distill gradients in early training. "
                         "v8+ (use-af-loss=true, w-param=0) sets this to 0 "
                         "since there's no L_param to warm up over.")
    ap.add_argument("--whiten-labels", action="store_true", default=True,
                    help="Compute per-(proc, param) std over labels and "
                         "use 1/std as L_param weights so EQ doesn't drown "
                         "out comp/gate/etc. Enabled by default. Only "
                         "consulted when w_param > 0.")
    ap.add_argument("--no-whiten-labels", dest="whiten_labels",
                    action="store_false")
    ap.add_argument("--w-recon", type=float, default=0.0,
                    help="Audio-recon weight against the ENGINEER MIX "
                         "(not the teacher render). 0 disables — v6/v7 had "
                         "this off because distillation alone should "
                         "suffice. v8+ enables it: with non-unique solutions "
                         "the amortized-optimization framework says to "
                         "use the original objective when available, and "
                         "Diff-MST (ISMIR 2024) shows audio-only training "
                         "works without parameter supervision. See "
                         "notes/distillation_research_2026-05.md. Same "
                         "console render as the student side of L_distill "
                         "so no extra forward cost; just one extra loss "
                         "evaluation against batch['mix'].")
    ap.add_argument("--use-af-loss", action="store_true", default=False,
                    help="Use the 5-feature Audio-Feature loss "
                         "(`AudioFeatureLoss`, Diff-MST style) for "
                         "L_distill_audio and L_recon instead of "
                         "MR-STFT. Weights stereo width + imbalance at "
                         "10x spectral — gives sign-preserving gradient "
                         "at the mono fixed point that MR-STFT magnitude "
                         "lacks. v8/v9/v10 used this. Mutually exclusive "
                         "with --use-hybrid-loss.")
    ap.add_argument("--use-hybrid-loss", action="store_true", default=False,
                    help="Use HybridReconLoss = MR-STFT + AF stereo "
                         "(width + imbalance only) for L_distill and "
                         "L_recon. Motivated by the v9/v10 audition "
                         "(notes/capacity_research_2026-05.md and chat "
                         "log 2026-05-26): pure AF satisfied stereo + "
                         "dynamics but left student WORSE than sum "
                         "baseline on MR-STFT. Hybrid adds back full-"
                         "spectrum supervision while keeping the sign-"
                         "preserving stereo terms that MR-STFT alone "
                         "lacks. v11+ default. Mutually exclusive with "
                         "--use-af-loss.")
    ap.add_argument("--w-stereo-width", type=float, default=1.0,
                    help="Weight on the stereo-width MSE inside "
                         "HybridReconLoss (only consulted when "
                         "--use-hybrid-loss is set). Default 1.0 — "
                         "comparable magnitude to MR-STFT, sign-"
                         "preserving stereo gradient.")
    ap.add_argument("--w-stereo-imbalance", type=float, default=1.0,
                    help="Weight on the stereo-imbalance MSE inside "
                         "HybridReconLoss (only consulted when "
                         "--use-hybrid-loss is set). Default 1.0.")
    ap.add_argument("--w-inter", type=float, default=0.0,
                    help="Weight for grafx_intermediate_audio_loss "
                         "(FitNets-style hint matching, Romero ICLR 2015). "
                         "Matches per-processor audio after EQ / comp / "
                         "noisegate / stereo_imager / gain_panning / "
                         "delay / reverb against the teacher's renders "
                         "of the same stages. Denser supervision than "
                         "matching just the final mix — student learns "
                         "the teacher's process. 0 disables. Costs one "
                         "console.forward_with_intermediates call on the "
                         "teacher side (no grad) on top of the student "
                         "one, plus ~3 GB extra peak memory at b_pair=4 "
                         "for the held intermediate tensors. Internal "
                         "INTERNAL_SCALE=1e4 in the loss puts values in "
                         "the same order of magnitude as L_param, so "
                         "w_inter ~= w_param scale-wise.")
    ap.add_argument("--w-cosine-ramp", type=int, default=0,
                    help="Steps over which audio loss weights "
                         "(w_distill, w_recon, w_inter) ramp from 0 to "
                         "their target via a cosine half-wave. 0 "
                         "disables (then --distill-warmup's step-piecewise "
                         "ramp applies to w_distill only — the v6/v7 "
                         "behavior). v8+ uses cosine for smoother regime "
                         "change: avoids the L_distill activation spike "
                         "v7 showed at step 2000 when its step-piecewise "
                         "ramp engaged. Set to ~10%% of max_steps.")

    # Logging / checkpointing
    ap.add_argument("--log-every",  type=int, default=50)
    ap.add_argument("--val-every",  type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--use-bf16", action="store_true", default=False,
                    help="Run the encoder forward under "
                         "`torch.autocast(device_type='cuda', dtype=bfloat16)`, "
                         "then cast its outputs back to fp32 before the console "
                         "and losses. Halves activation memory in the encoder "
                         "and roughly halves encoder wallclock. The DSP "
                         "console stays in fp32 — grafx's FFT-based processors "
                         "(`MultitapDelay` calls `view_as_complex` which "
                         "rejects bfloat16) are not autocast-safe. Smoke "
                         "tested 2026-05-25: identical losses within 0.6%%, no "
                         "NaN/Inf, peak memory drops ~30%% at batch_size=2.")
    ap.add_argument("--cuda-memory-fraction", type=float, default=0.0,
                    help="If > 0, cap the fraction of CUDA memory PyTorch may "
                         "allocate via torch.cuda.set_per_process_memory_fraction. "
                         "On unified-memory hardware (DGX Spark / GB10) host and "
                         "GPU share one physical pool and an unconstrained run "
                         "can drive the NVRM driver into OOM, which on this box "
                         "reboots the system (see notes/RUNS.md and our 2026-05 "
                         "v6 distillation crash). The RAM watchdog catches slow "
                         "growth but a sub-poll-interval spike (autograd graph "
                         "construction on the first backward) can outpace it. "
                         "0.65 = ~78 GB cap on the GB10's 121 GB unified pool — "
                         "headroom for OS + page cache + worker subprocesses.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.run_name is None:
        args.run_name = time.strftime("%Y%m%d_%H%M%S")
    if args.out_dir is None:
        args.out_dir = f"dmc-data/checkpoints/grafx_distill_{args.run_name}"
    if args.tb_logdir is None:
        args.tb_logdir = f"runs/grafx_distill_{args.run_name}"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tb_logdir).mkdir(parents=True, exist_ok=True)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logging.info(f"run_name = {args.run_name}")
    logging.info(f"device   = {device}")

    if args.cuda_memory_fraction > 0.0 and device.type == "cuda":
        # `set_per_process_memory_fraction` needs an explicit device index;
        # `torch.device("cuda")` (no index) is rejected.
        cuda_idx = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(
            float(args.cuda_memory_fraction), device=cuda_idx,
        )
        logging.info(
            f"cuda memory fraction capped at {args.cuda_memory_fraction:.2f} "
            f"(cuda:{cuda_idx})"
        )

    # ---- Sessions / split ----
    staging_dir = Path(args.staging_dir).expanduser()
    label_store = LabelStore(staging_dir)
    sessions = _list_staged_sessions(staging_dir)
    if not sessions:
        raise SystemExit(f"no staged sessions found under {staging_dir}")
    n_labeled, n_total = label_store.coverage(sessions)
    logging.info(f"sessions: {n_total} staged, {n_labeled} with labels")

    n_val = max(1, int(round(args.val_frac * n_total)))
    rng = random.Random(args.seed)
    shuffled = sorted(sessions)
    rng.shuffle(shuffled)
    val_sessions = sorted(shuffled[:n_val])
    train_sessions = sorted(shuffled[n_val:])
    logging.info(f"split: train={len(train_sessions)} val={len(val_sessions)}")

    if args.max_label_loss is not None:
        n_before = len(train_sessions)
        kept = []
        for s in train_sessions:
            labels = label_store.get(s)
            if labels is None:
                kept.append(s)
                continue
            loss = (labels.get("training_meta") or {}).get("final_test_loss")
            if loss is None or float(loss) > args.max_label_loss:
                continue
            kept.append(s)
        train_sessions = kept
        logging.info(
            f"label-quality filter: kept {len(train_sessions)}/{n_before} train"
        )

    # ---- Whitening stats ----
    # Skip when w_param == 0 (v8+); whitening is only consulted by
    # `grafx_param_huber_loss`, and computing 1/std across 289 labels is
    # ~1 s of startup we can avoid when L_param isn't in the budget.
    whiten_weights = None
    if args.whiten_labels and args.w_param > 0:
        logging.info("computing per-(proc, param) label std for whitening...")
        whiten = compute_label_whiten_weights(label_store, train_sessions + val_sessions)
        # Keep separate dicts for strip/group; merge by-procname for the
        # huber loss (it indexes by proc, not by level). Predicted-side
        # strip and group don't share param names with conflicting scales
        # for the procs we currently use, but to be safe we pass the strip
        # weights and let group reuse the same per-(proc, param) values.
        whiten_weights = whiten["strip"]
        for proc, pp in whiten.get("group", {}).items():
            whiten_weights.setdefault(proc, {})
            for p, w in pp.items():
                # If strip and group disagree, prefer strip (higher cardinality).
                whiten_weights[proc].setdefault(p, w)
        for proc, pp in sorted(whiten_weights.items()):
            for p, w in sorted(pp.items()):
                logging.info(f"  whiten[{proc}.{p}] = {w:.4g}  (1/std)")

    # ---- Data ----
    train_ds = PairedMultiSongDataset(
        train_sessions, staging_dir, label_store,
        audio_len=args.audio_len, n_max=args.n_max,
        seed=args.seed, cache_size=args.cache_size,
        min_window_gap=args.min_window_gap,
        pairs_per_session=args.pairs_per_session,
    )
    # DataLoader tuning. PairedMultiSongDataset loads ~1.8 GB of decoded
    # WAVs per cache miss (30 stems × ~60 MB each); with `num_workers=0`
    # those loads block the trainer's forward pass and produce a stepwise
    # GPU-utilization pattern. Workers prefetch in parallel with compute,
    # `pin_memory` accelerates host→device transfers, `persistent_workers`
    # keeps worker session caches warm across IterableDataset restarts.
    loader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=collate_pairs,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_ds, **loader_kwargs)

    # ---- Model ----
    console = GrafxMixingConsole(
        sample_rate=args.sample_rate, max_input_len=args.audio_len,
    ).to(device)
    encoder = MixEncoderGrafx(
        console, sample_rate=args.sample_rate, d_model=args.d_model,
        n_track_layers=args.n_track_layers, use_ref_mix=False, mert_dim=0,
    ).to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    logging.info(f"encoder params: {n_params/1e6:.1f}M")

    if args.init_from:
        ckpt_path = Path(args.init_from).expanduser()
        if not ckpt_path.is_file():
            raise SystemExit(f"--init-from: file not found: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), weights_only=False, map_location=device)
        state_dict = (
            ckpt["encoder_state_dict"]
            if isinstance(ckpt, dict) and "encoder_state_dict" in ckpt
            else ckpt
        )
        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        logging.info(
            f"warm-started encoder from {ckpt_path}  "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    strip_schema = console.strip_param_shapes
    group_schema = console.group_param_shapes

    # ---- Losses ----
    # `distill_loss_fn` is used both for L_distill (vs teacher render) and
    # L_recon (vs engineer mix). Three families available:
    #   v6/v7: MR-STFT (auraloss, perceptually-weighted; matches teacher's
    #          training loss exactly)
    #   v8/v9/v10: AF (Diff-MST 5-feature; sign-preserving stereo, but
    #              insufficient full-spectrum supervision — found in v9/v10
    #              audition)
    #   v11+:  Hybrid (MR-STFT + AF stereo width/imbalance only; gets full-
    #          spectrum from MR-STFT + sign-preserving stereo from AF)
    if args.use_af_loss and args.use_hybrid_loss:
        raise SystemExit("--use-af-loss and --use-hybrid-loss are mutually exclusive")
    if args.use_hybrid_loss:
        distill_loss_fn = HybridReconLoss(
            sample_rate=args.sample_rate,
            w_mrstft=1.0,
            w_stereo_width=args.w_stereo_width,
            w_stereo_imbalance=args.w_stereo_imbalance,
        ).to(device)
        af_total_key = "total"
        af_log_keys = ("total", "mrstft/full", "mrstft/sum", "mrstft/diff",
                       "stereo/width", "stereo/imbalance")
    elif args.use_af_loss:
        distill_loss_fn = AudioFeatureLoss(sample_rate=args.sample_rate).to(device)
        af_total_key = "af/total"
        af_log_keys = ("af/total", "af/rms", "af/crest", "af/spec",
                       "af/width", "af/imbalance")
    else:
        distill_loss_fn = make_grafx_prune_recon_loss(sample_rate=args.sample_rate).to(device)
        af_total_key = "match/full"
        af_log_keys = ("match/full", "match/sum", "match/diff")

    # MR-STFT loss for the val-time amortization-gap metric. The teacher's
    # recorded `final_test_loss` is in MR-STFT units (auraloss
    # SumAndDifferenceSTFTLoss), so the gap = student_recon_mrstft − teacher_loss
    # is only meaningful when both terms use the SAME loss family. We
    # construct MR-STFT separately and always use it for the gap metric,
    # regardless of whether training uses MR-STFT or AF — so the gap stays
    # comparable across loss-design experiments. Cost: ~50 MB of auraloss
    # filterbanks, no extra forward.
    gap_loss_fn = make_grafx_prune_recon_loss(sample_rate=args.sample_rate).to(device)

    # ---- Optimizer / schedule ----
    optimizer = AdamW(
        encoder.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )

    def _lr_lambda(step: int) -> float:
        if args.lr_warmup_steps > 0 and step < args.lr_warmup_steps:
            return (step + 1) / args.lr_warmup_steps
        decay_total = max(1, args.max_steps - args.lr_warmup_steps)
        progress = min(1.0, (step - args.lr_warmup_steps) / decay_total)
        min_mult = args.lr_cosine_min / args.lr
        return min_mult + (1.0 - min_mult) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    # ---- TB ----
    tb = SummaryWriter(log_dir=args.tb_logdir)
    logging.info(f"tb logdir: {args.tb_logdir}")
    logging.info(f"ckpt dir:  {args.out_dir}")
    logging.info(
        f"loss weights: w_param={args.w_param} w_cons={args.w_cons} "
        f"w_distill={args.w_distill}  whiten={args.whiten_labels}"
    )

    # ---- Train loop (with gradient accumulation) ----
    t0 = time.time()
    encoder.train()
    best_val = float("inf")
    train_iter = iter(train_loader)

    def _next_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    def _micro_step(step: int) -> dict:
        """One forward + scaled backward on a fresh batch.
        Returns log-scalar dict for accumulation across micro-steps.

        v6/v7: w_param > 0, w_recon = 0 → param + cons + (after warmup) distill.
        v8+:  w_param = 0, w_recon > 0 → cons + distill + recon (audio-only).
        Both regimes route through the same code path; zeros disable terms.
        """
        batch = _next_batch()
        B_pair = batch["B_pair"]
        batch = attach_grafx_labels(
            batch, label_store, n_max=args.n_max, max_groups=args.max_groups,
            strip_schema=strip_schema, group_schema=group_schema,
        )

        tracks = batch["tracks"].to(device, non_blocking=True)
        track_mask = batch["track_mask"].to(device, non_blocking=True)
        group_idx = batch["group_assignments"].to(device, non_blocking=True)
        mix_ref = batch["mix"].to(device, non_blocking=True)

        out = _encoder_forward(
            encoder, tracks, track_mask, group_idx,
            n_groups=args.max_groups, use_bf16=args.use_bf16,
        )

        label_strip = _to_device_params(batch["label_strip_params"], device)
        label_group = _to_device_params(batch["label_group_params"], device)

        # ---- L_param (regression-based) ----
        compute_lparam = args.w_param > 0
        if compute_lparam:
            L_param = grafx_param_huber_loss(
                out["strip_params"], out["group_params"],
                label_strip, label_group,
                batch["label_track_mask"].to(device),
                batch["n_groups_per_example"].to(device),
                batch["label_example_mask"].to(device),
                delta=1.0, param_weights=whiten_weights,
            )

        # ---- L_cons (window consistency) ----
        strip_a, strip_b = _split_params(out["strip_params"], B_pair)
        group_a, group_b = _split_params(out["group_params"], B_pair)
        n_groups_pair = batch["n_groups_per_example"][:B_pair].to(device)
        L_cons = grafx_param_consistency_loss(
            strip_a, group_a, strip_b, group_b,
            track_mask[:B_pair], n_groups_pair,
            param_weights=whiten_weights,
        )

        # ---- Audio losses ----
        # Resolve per-step loss weights. v8+ default = cosine ramp on all
        # audio weights (smoother regime change than v6/v7's step-piecewise
        # `distill_warmup`, which produced a visible val spike at the
        # ramp boundary). v6/v7 path: cosine_ramp=0, distill_warmup>0,
        # only w_distill ramps.
        if args.w_cosine_ramp > 0:
            w_distill_now = w_cosine_ramp_at(step, args.w_cosine_ramp, args.w_distill)
            w_recon_now   = w_cosine_ramp_at(step, args.w_cosine_ramp, args.w_recon)
            w_inter_now   = w_cosine_ramp_at(step, args.w_cosine_ramp, args.w_inter)
        else:
            w_distill_now = w_distill_at(step, args.distill_warmup, args.w_distill)
            w_recon_now   = args.w_recon
            w_inter_now   = args.w_inter

        compute_audio = (w_distill_now > 0) or (w_recon_now > 0) or (w_inter_now > 0)
        need_intermediates = w_inter_now > 0
        need_teacher = (w_distill_now > 0) or (w_inter_now > 0)
        L_distill = None
        L_recon = None
        L_inter = None
        pred_strip_inter = pred_group_inter = None
        gt_strip_inter = gt_group_inter = None

        if compute_audio:
            if need_intermediates:
                student_mix, pred_strip_inter, pred_group_inter = (
                    console.forward_with_intermediates(
                        tracks, out["strip_params"], out["group_params"],
                        group_idx, track_mask=track_mask,
                        n_groups=args.max_groups,
                    )
                )
            else:
                student_mix = console(
                    tracks, out["strip_params"], out["group_params"],
                    group_idx, track_mask=track_mask,
                    n_groups=args.max_groups,
                )

            if need_teacher:
                with torch.no_grad():
                    if need_intermediates:
                        teacher_mix, gt_strip_inter, gt_group_inter = (
                            console.forward_with_intermediates(
                                tracks, label_strip, label_group, group_idx,
                                track_mask=track_mask,
                                n_groups=args.max_groups,
                            )
                        )
                    else:
                        teacher_mix = console(
                            tracks, label_strip, label_group, group_idx,
                            track_mask=track_mask,
                            n_groups=args.max_groups,
                        )
                if w_distill_now > 0:
                    L_distill = distill_loss_fn(student_mix, teacher_mix)
                if w_inter_now > 0:
                    L_inter = grafx_intermediate_audio_loss(
                        pred_strip_inter, gt_strip_inter,
                        pred_group_inter, gt_group_inter,
                        batch["label_track_mask"].to(device),
                        batch["n_groups_per_example"].to(device),
                        batch["label_example_mask"].to(device),
                    )

            if w_recon_now > 0:
                L_recon = distill_loss_fn(student_mix, mix_ref)

        # ---- Aggregate ----
        L_total = args.w_cons * L_cons["L_cons/total"]
        if compute_lparam:
            L_total = L_total + args.w_param * L_param["L_param/total"]
        if L_distill is not None:
            L_total = L_total + w_distill_now * L_distill[af_total_key]
        if L_recon is not None:
            L_total = L_total + w_recon_now * L_recon[af_total_key]
        if L_inter is not None:
            L_total = L_total + w_inter_now * L_inter["L_inter/total"]

        # Scale by 1/grad_accum so the accumulated gradient is the MEAN
        # over micro-batches (consistent with batch-size scaling).
        (L_total / args.grad_accum).backward()

        log = {
            "L_total":       L_total.item(),
            "L_cons/total":  L_cons["L_cons/total"].item(),
            "L_cons/strip":  L_cons["L_cons/strip"].item(),
            "L_cons/group":  L_cons["L_cons/group"].item(),
            "w_distill":     w_distill_now,
            "w_recon":       w_recon_now,
            "w_inter":       w_inter_now,
        }
        if compute_lparam:
            log["L_param/total"] = L_param["L_param/total"].item()
            log["L_param/strip"] = L_param["L_param/strip"].item()
            log["L_param/group"] = L_param["L_param/group"].item()
        if L_distill is not None:
            for k in af_log_keys:
                log[f"L_distill/{k.split('/')[-1]}"] = L_distill[k].item()
        if L_recon is not None:
            for k in af_log_keys:
                log[f"L_recon/{k.split('/')[-1]}"] = L_recon[k].item()
        if L_inter is not None:
            log["L_inter/total"] = L_inter["L_inter/total"].item()
            log["L_inter/strip"] = L_inter["L_inter/strip"].item()
            log["L_inter/group"] = L_inter["L_inter/group"].item()
        return log

    for step in range(args.max_steps):
        optimizer.zero_grad(set_to_none=True)
        micro_logs: list[dict] = []
        for _ in range(args.grad_accum):
            micro_logs.append(_micro_step(step))
        grad_norm = torch.nn.utils.clip_grad_norm_(
            encoder.parameters(), max_norm=args.grad_clip,
        )
        optimizer.step()
        scheduler.step()

        # Mean log values across micro-steps. Any key only some micros emit
        # (e.g. L_distill/* during warmup boundary) is averaged over only
        # the micros that emitted it.
        log_keys = set().union(*[m.keys() for m in micro_logs])
        agg = {
            k: sum(m[k] for m in micro_logs if k in m) /
               max(1, sum(1 for m in micro_logs if k in m))
            for k in log_keys
        }

        if step % args.log_every == 0:
            # Log whichever keys this step's micros actually produced;
            # absent keys (e.g. L_param when w_param=0) are skipped.
            tb.add_scalar("train/L_total",       agg["L_total"],       step)
            tb.add_scalar("train/L_cons/strip",  agg["L_cons/strip"],  step)
            tb.add_scalar("train/L_cons/group",  agg["L_cons/group"],  step)
            tb.add_scalar("train/L_cons/total",  agg["L_cons/total"],  step)
            for k in ("L_param/strip", "L_param/group", "L_param/total"):
                if k in agg:
                    tb.add_scalar(f"train/{k}", agg[k], step)
            for prefix in ("L_distill", "L_recon", "L_inter"):
                for k in agg:
                    if k.startswith(f"{prefix}/"):
                        tb.add_scalar(f"train/{k}", agg[k], step)
            tb.add_scalar("train/w_distill", agg["w_distill"], step)
            tb.add_scalar("train/w_recon",   agg.get("w_recon", 0.0), step)
            tb.add_scalar("train/w_inter",   agg.get("w_inter", 0.0), step)
            tb.add_scalar("train/grad_norm", grad_norm.item(), step)
            tb.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
            elapsed = time.time() - t0
            # Compact one-line summary; works for both v6/v7 (L_p+L_d) and
            # v8 (L_d+L_r) loss regimes.
            dist_total_log_key = (
                "L_distill/total" if "L_distill/total" in agg
                else "L_distill/full" if "L_distill/full" in agg
                else None
            )
            recon_total_log_key = (
                "L_recon/total" if "L_recon/total" in agg
                else "L_recon/full" if "L_recon/full" in agg
                else None
            )
            parts = [f"step {step:6d}/{args.max_steps}",
                     f"L_tot={agg['L_total']:7.3f}"]
            if "L_param/total" in agg:
                parts.append(f"L_p={agg['L_param/total']:6.2f}")
            parts.append(f"L_c={agg['L_cons/total']:6.2f}")
            if dist_total_log_key is not None:
                parts.append(f"L_d={agg[dist_total_log_key]:5.2f}")
            else:
                parts.append("L_d=  off")
            if recon_total_log_key is not None:
                parts.append(f"L_r={agg[recon_total_log_key]:5.2f}")
            if "L_inter/total" in agg:
                parts.append(f"L_i={agg['L_inter/total']:5.2f}")
            parts.append(f"w_d={agg['w_distill']:.3f}")
            parts.append(f"grad={grad_norm.item():6.1f}")
            parts.append(f"({elapsed/(step+1):.2f}s/step)")
            logging.info("  ".join(parts))

        if step > 0 and step % args.val_every == 0:
            val_metrics = run_val(
                encoder, console, distill_loss_fn,
                val_sessions, staging_dir, label_store,
                args.audio_len, args.n_max, args.max_groups, device,
                whiten=whiten_weights,
                strip_schema=strip_schema, group_schema=group_schema,
                use_bf16=args.use_bf16,
                audio_total_key=af_total_key,
                compute_recon=(args.w_recon > 0),
                gap_loss_fn=gap_loss_fn,
            )
            for k, v in val_metrics.items():
                tb.add_scalar(k, v, step)
            # Log line tolerates NaN/missing components (v6 vs v8 regime).
            parts = [f"  val[step {step}]:"]
            for key, short in (("val/L_param", "L_p"),
                                ("val/L_cons",  "L_c"),
                                ("val/L_distill", "L_d"),
                                ("val/L_recon",  "L_r"),
                                ("val/amort_gap", "gap")):
                v = val_metrics.get(key, float("nan"))
                if v == v:  # not NaN
                    parts.append(f"{short}={v:.3f}")
            parts.append(f"(n_lab={int(val_metrics['val/n_labeled'])} "
                          f"n_tot={int(val_metrics['val/n_total'])})")
            logging.info(" ".join(parts))
            # Best-ckpt score: prefer L_recon (the actual training-target
            # against engineer mix) when available; else fall back to
            # L_distill (audio-space vs teacher); else L_param.
            if val_metrics.get("val/L_recon", float("nan")) == val_metrics.get("val/L_recon"):
                score = val_metrics["val/L_recon"]
            elif val_metrics["val/L_distill"] == val_metrics["val/L_distill"]:
                score = val_metrics["val/L_distill"]
            else:
                score = val_metrics["val/L_param"]
            if score < best_val:
                best_val = score
                torch.save({
                    "step": step,
                    "encoder_state_dict": encoder.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                    "whiten_weights": whiten_weights,
                }, Path(args.out_dir) / "encoder_best.pt")
                logging.info(f"  → new best val score = {best_val:.4f}")

        if step > 0 and step % args.ckpt_every == 0:
            torch.save({
                "step": step,
                "encoder_state_dict": encoder.state_dict(),
                "args": vars(args),
                "whiten_weights": whiten_weights,
            }, Path(args.out_dir) / f"encoder_step{step:06d}.pt")

    final_val = run_val(
        encoder, console, distill_loss_fn,
        val_sessions, staging_dir, label_store,
        args.audio_len, args.n_max, args.max_groups, device,
        whiten=whiten_weights,
        strip_schema=strip_schema, group_schema=group_schema,
        use_bf16=args.use_bf16,
        audio_total_key=af_total_key,
        compute_recon=(args.w_recon > 0),
        gap_loss_fn=gap_loss_fn,
    )
    for k, v in final_val.items():
        tb.add_scalar(k, v, args.max_steps)
    torch.save({
        "step": args.max_steps,
        "encoder_state_dict": encoder.state_dict(),
        "val_metrics": final_val,
        "args": vars(args),
        "whiten_weights": whiten_weights,
    }, Path(args.out_dir) / f"encoder_step{args.max_steps:06d}.pt")

    tb.close()
    elapsed = time.time() - t0
    logging.info(
        f"\ntraining done. {args.max_steps} steps in {elapsed/60:.1f} min. "
        f"best val score = {best_val:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
