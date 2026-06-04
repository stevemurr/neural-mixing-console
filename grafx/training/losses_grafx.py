"""Loss functions for grafx-prune-style supervised training.

Two terms, weighted and combined upstream in the trainer:

  L_param  — Huber MSE on (encoder_predicted_params, grafx_prune_labels),
              masked by `label_track_mask` (per-track) and
              `label_example_mask` (per-example) so unlabeled examples
              and unmatched tracks contribute zero.

  L_recon — multi-resolution mid/side STFT loss between the rendered
              mix and the engineer's reference mix, matching the loss
              grafx-prune itself optimizes against per song (so encoder
              + per-song labels share the same audio-domain objective).
              Uses auraloss.freq.SumAndDifferenceSTFTLoss directly to
              match grafx-prune line-for-line.

`L_param` is the dense supervision signal — it directly aligns each
predicted parameter with the value the per-song optimizer converged to.
`L_recon` keeps the encoder calibrated to the engineer's audio aesthetic
on parts of the parameter space the labels don't perfectly cover (and
on un-labeled songs).

`grafx_param_huber_loss` walks the nested dict shape and reduces:

  sum over (proc, param) of  Huber(pred, target).mean_over_param_shape
                            * track_mask_or_group_mask
                            / sum_of_masks_for_normalization

Per-processor / per-param weights can be passed via `param_weights` to,
e.g., upweight EQ's dense 1024-bin output relative to the comp's 4
scalars — by default everything is weighted 1.0 and the loss is
proportional to the per-element MSE summed across the schema.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Audio-feature loss (Diff-MST style, Vanka et al. ISMIR 2024) ----------

class AudioFeatureLoss(nn.Module):
    """Five-feature audio-production loss for mix-level supervision.

    Diff-MST (ISMIR 2024) introduced this in place of full-spectrum MR-STFT
    for mixing-network training. The design rationale (see
    `notes/distillation_research_2026-05.md` for the full discussion):

      - MR-STFT magnitude is sign-invariant on the side channel
        (`STFT(L−R)` has the same magnitude for left-leaning and
        right-leaning mixes), so it gives zero gradient at the mono
        fixed point — the encoder can't escape "everything centered".
      - Replacing it with stereo-aware features (width + imbalance) at
        heavy weight (10× spectral) provides the sign-preserving
        gradient mix training actually needs.
      - The five features together (RMS + crest + spectrum + width +
        imbalance) cover dynamics, perceptual spectrum, and stereo
        without redundancy.

    Implementation notes vs Diff-MST:

      - Bark spectrum → log-mel: Diff-MST uses Bark filterbank; we use
        mel (similar perceptual scale, available in torchaudio). 32 mel
        bands by default — matches Diff-MST's Bark band count roughly.
      - We omit the first `omit_seconds` of audio to skip filter-warmup
        transients (matches `make_grafx_prune_recon_loss` convention).
      - Default weights from the Diff-MST paper:
            w_rms       = 0.1
            w_crest     = 0.001
            w_spec      = 0.1
            w_width     = 1.0      # stereo gets 10x spectral
            w_imbalance = 1.0
        Configurable via __init__.

    Returns a dict for TB logging:
        {"af/total", "af/rms", "af/crest", "af/spec",
         "af/width", "af/imbalance"}
    """

    def __init__(
        self,
        sample_rate: int = 48_000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 32,
        eps: float = 1e-8,
        w_rms: float = 0.1,
        w_crest: float = 0.001,
        w_spec: float = 0.1,
        w_width: float = 1.0,
        w_imbalance: float = 1.0,
        omit_seconds: float = 1.0,
    ):
        super().__init__()
        import torchaudio  # local import to avoid top-level dep when unused
        self.sample_rate = sample_rate
        self.eps = eps
        self.w_rms = w_rms
        self.w_crest = w_crest
        self.w_spec = w_spec
        self.w_width = w_width
        self.w_imbalance = w_imbalance
        self.omit_samples = int(sample_rate * omit_seconds)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=1.0,  # magnitude (Diff-MST uses |STFT(x)| not power)
        )

    def _trim(self, x: torch.Tensor) -> torch.Tensor:
        if self.omit_samples > 0 and x.shape[-1] > self.omit_samples:
            return x[..., self.omit_samples:]
        return x

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, T) -> (B, 2)
        return torch.sqrt((x ** 2).mean(dim=-1) + self.eps)

    def _crest_db(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, T) -> (B, 2), dB-scaled (clamp avoids -inf at silence)
        peak = x.abs().amax(dim=-1).clamp(min=self.eps)
        rms = self._rms(x).clamp(min=self.eps)
        return 20.0 * torch.log10(peak / rms)

    def _log_mel(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, T) -> (B, 2, n_mels, n_frames)
        B = x.shape[0]
        x_flat = x.reshape(B * 2, x.shape[-1])
        mag = self.mel(x_flat)
        return torch.log(mag + self.eps).reshape(
            B, 2, mag.shape[-2], mag.shape[-1]
        )

    def _stereo_width(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, T) -> (B,). Side / Mid energy ratio.
        L = x[:, 0]; R = x[:, 1]
        side_e = ((L - R) ** 2).sum(dim=-1)
        mid_e = ((L + R) ** 2).sum(dim=-1) + self.eps
        return side_e / mid_e

    def _stereo_imbalance(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, T) -> (B,). (R-L)/(R+L) energy. Sign-preserving.
        L_e = (x[:, 0] ** 2).sum(dim=-1)
        R_e = (x[:, 1] ** 2).sum(dim=-1)
        return (R_e - L_e) / (R_e + L_e + self.eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """`pred`, `target`: `(B, 2, T)` stereo audio."""
        pred = self._trim(pred)
        target = self._trim(target)

        l_rms = F.mse_loss(self._rms(pred), self._rms(target))
        l_crest = F.mse_loss(self._crest_db(pred), self._crest_db(target))
        l_spec = F.mse_loss(self._log_mel(pred), self._log_mel(target))
        l_width = F.mse_loss(self._stereo_width(pred),
                             self._stereo_width(target))
        l_imb = F.mse_loss(self._stereo_imbalance(pred),
                           self._stereo_imbalance(target))

        total = (self.w_rms * l_rms
                 + self.w_crest * l_crest
                 + self.w_spec * l_spec
                 + self.w_width * l_width
                 + self.w_imbalance * l_imb)

        return {
            "af/total":     total,
            "af/rms":       l_rms,
            "af/crest":     l_crest,
            "af/spec":      l_spec,
            "af/width":     l_width,
            "af/imbalance": l_imb,
        }


# ---------- Hybrid: MR-STFT + full Diff-MST AF feature set ----------

class HybridReconLoss(nn.Module):
    """MR-STFT (fine spectral) + the full Diff-MST AF feature set.

    The AF half uses Diff-MST's exact paper feature set and weights:

        RMS         w=0.1     → strip/group gain levels
        Crest       w=0.001   → compressor threshold/ratio (peak÷RMS in dB)
        Log-mel     w=0.1     → EQ band gains (perceptual spectrum)
        Width       w=1.0     → stereo imager
        Imbalance   w=1.0     → pan / L-R balance

    Each feature correlates closely with a specific console DoF — the
    intent is to give the encoder per-knob gradient signal that pure
    MR-STFT averages over. The big stereo weights (10× spectral) are
    Diff-MST's prescription and counter the mono-fixed-point blind
    spot in magnitude STFT loss (|STFT(L-R)| is sign-invariant on the
    side channel).

    MR-STFT is kept on top as the fine-grained spectral target —
    necessary because v8/v9 showed pure AF leaves the student worse
    than sum-of-stems on MR-STFT. It is `match/full` from the
    `make_grafx_prune_recon_loss` factory.

    Returns a dict with `"total"` as the weighted sum and individual
    components (MR-STFT + all 5 AF features) for TB logging.
    """

    def __init__(
        self,
        sample_rate: int = 48_000,
        w_mrstft: float = 1.0,
        w_af: float = 1.0,
        # Per-feature AF weights — defaults match Diff-MST paper exactly.
        w_rms: float = 0.1,
        w_crest: float = 0.001,
        w_spec: float = 0.1,
        w_width: float = 1.0,
        w_imbalance: float = 1.0,
        eps: float = 1e-8,
        omit_seconds: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.w_mrstft = w_mrstft
        self.w_af = w_af
        self.eps = eps
        self.omit_samples = int(sample_rate * omit_seconds)
        self.mrstft = make_grafx_prune_recon_loss(
            sample_rate=sample_rate, omit_sec=omit_seconds,
        )
        self.af = AudioFeatureLoss(
            sample_rate=sample_rate,
            w_rms=w_rms, w_crest=w_crest, w_spec=w_spec,
            w_width=w_width, w_imbalance=w_imbalance,
            eps=eps, omit_seconds=omit_seconds,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """`pred`, `target`: `(B, 2, T)` stereo audio."""
        mr = self.mrstft(pred, target)
        af = self.af(pred, target)

        total = self.w_mrstft * mr["match/full"] + self.w_af * af["af/total"]

        return {
            "total":            total,
            "mrstft/full":      mr["match/full"],
            "mrstft/sum":       mr["match/sum"],
            "mrstft/diff":      mr["match/diff"],
            "af/total":         af["af/total"],
            "af/rms":           af["af/rms"],
            "af/crest":         af["af/crest"],
            "af/spec":          af["af/spec"],
            "af/width":         af["af/width"],
            "af/imbalance":     af["af/imbalance"],
            # Back-compat aliases for existing TB / log readers.
            "stereo/width":     af["af/width"],
            "stereo/imbalance": af["af/imbalance"],
        }


# ---------- Reconstruction loss (matches grafx-prune line-for-line) ----------

def make_grafx_prune_recon_loss(
    sample_rate: int = 48_000,
    fft_sizes: tuple[int, ...] = (1024, 2048, 8192),
    hop_sizes: tuple[int, ...] = (256, 512, 2048),
    win_lengths: tuple[int, ...] = (1024, 2048, 8192),
    n_bins: int = 96,
    omit_sec: float = 1.0,
) -> torch.nn.Module:
    """Construct the multi-resolution mid/side STFT loss used by grafx-prune.

    Pulled directly from `code/loss.py:MRSTFTLoss` in the grafx-prune repo
    (DAFx 2024). Mel-scaled magnitude with A-weighting-style perceptual
    weighting; sum + diff (mid/side) decomposition; omits the first
    `omit_sec` of audio (filter warmup transients).

    Returns a `nn.Module` whose `forward(pred, target) → dict` produces:
        { "match/full": scalar, "match/lr": ..., "match/sum": ..., "match/diff": ... }
    """
    # Local import — auraloss is the only dep added for this loss family.
    from auraloss.freq import SumAndDifferenceSTFTLoss

    class GrafxPruneMRSTFTLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loss = SumAndDifferenceSTFTLoss(
                fft_sizes=list(fft_sizes),
                hop_sizes=list(hop_sizes),
                win_lengths=list(win_lengths),
                perceptual_weighting=True,
                sample_rate=sample_rate,
                scale="mel",
                n_bins=n_bins,
                eps=1e-4,
                output="full",  # returns (full, lr, sum, diff)
            )
            self.omit = int(sample_rate * omit_sec)

        def forward(self, pred: torch.Tensor, true: torch.Tensor) -> dict:
            T = pred.shape[-1]
            if pred.ndim > 3:
                pred = pred.view(-1, 2, T)
            if true.ndim > 3:
                true = true.view(-1, 2, T)
            if self.omit > 0 and T > self.omit:
                pred = pred[..., self.omit:]
                true = true[..., self.omit:]
            # auraloss 0.4 returns (full, sum, diff) for output="full".
            # (Older auraloss had per-channel L+R too; dropped in 0.4.)
            full, sum_loss, diff_loss = self.loss(pred, true)
            return {
                "match/full": full,
                "match/sum":  sum_loss,
                "match/diff": diff_loss,
            }

    return GrafxPruneMRSTFTLoss()


# ---------- Per-track parameter Huber loss ----------

def grafx_param_huber_loss(
    pred_strip: dict[str, dict[str, torch.Tensor]],   # encoder output
    pred_group: dict[str, dict[str, torch.Tensor]],
    label_strip: dict[str, dict[str, torch.Tensor]],  # collate output
    label_group: dict[str, dict[str, torch.Tensor]],
    label_track_mask: torch.Tensor,    # (B, N_max) bool — True where supervision exists
    n_groups_per_example: torch.Tensor, # (B,) long — # of valid groups per example
    label_example_mask: torch.Tensor,  # (B,) bool — True for labeled examples
    *,
    delta: float = 1.0,
    param_weights: Optional[dict[str, float | dict[str, float]]] = None,
) -> dict[str, torch.Tensor]:
    """Huber MSE between predicted and label params for strip + group.

    Masks:
      - Per-track loss: weighted by `label_track_mask` (filename matched)
        AND `label_example_mask` (example has labels at all).
      - Per-group loss: weighted by a `(B, max_groups)` mask derived from
        `n_groups_per_example` (only groups < n_groups[b] count) AND
        `label_example_mask`.

    `delta` is Huber's quadratic→linear threshold. At delta=1.0 the loss
    is effectively MSE for typical label magnitudes (most params live in
    [-2, +3] log-units) — the linear branch only kicks in for pathological
    outliers. Smaller deltas clip the gradient at ±delta in the linear
    region, which strangles convergence during overfit; round-14's MSE
    collapse came from sigmoid saturation in the old heads, and the new
    heads are pure Linear so that failure mode doesn't apply.

    Returns dict for logging:
      { "L_param/strip": ..., "L_param/group": ..., "L_param/total": ... }
    """
    B, N_max = label_track_mask.shape
    device = label_track_mask.device

    weights = param_weights or {}

    # --- Per-track strip Huber ---
    # Combined mask: (B, N_max) — True where there's both an example label
    # and a per-track filename match.
    strip_mask = (label_track_mask & label_example_mask.unsqueeze(-1)).to(torch.float32)

    # Normalization: SUM over tracks, MEAN over batch. Each active track
    # contributes its full per-shape-averaged loss to the total — i.e.
    # gradient per per-track output element is 1/B regardless of how many
    # sibling tracks are active. Previous "/ strip_mask.sum()" divided by
    # the active-track count (~32 for AMContra), which made per-track
    # gradient ~6× weaker than per-group (active-groups ~5), and let the
    # group bus memorize ~6× faster than the strip during overfit. The
    # loss VALUE now scales linearly with active-track count; that's a
    # logging concern, not a training one (we also log /strip and /group
    # separately so monitoring stays interpretable).
    strip_denom = float(B)

    def _w(proc_entry, param_name: str) -> float:
        # `param_weights[proc]` may be a float (per-proc, legacy) or a dict (per-param).
        if isinstance(proc_entry, dict):
            return float(proc_entry.get(param_name, 1.0))
        return float(proc_entry)

    l_strip_total = torch.zeros((), device=device)
    for proc, pp in pred_strip.items():
        proc_w_entry = weights.get(proc, 1.0)
        for param_name, pred_t in pp.items():
            target_t = label_strip[proc][param_name].to(pred_t.device)
            # Huber per-element, then mean over the param's own shape dims
            # (everything past dim 1). Result has shape (B, N_max).
            per_elem = F.huber_loss(pred_t, target_t, reduction="none", delta=delta)
            # average across the param's shape dims (everything past N)
            per_track = per_elem.flatten(start_dim=2).mean(dim=-1)   # (B, N_max)
            l_strip_total = l_strip_total + _w(proc_w_entry, param_name) * (per_track * strip_mask).sum() / strip_denom

    # --- Per-group bus Huber ---
    # Group mask: 1.0 where the group index < n_groups[b], 0 otherwise.
    max_groups = next(iter(next(iter(label_group.values())).values())).shape[1]
    group_idx_range = torch.arange(max_groups, device=device).unsqueeze(0)     # (1, max_g)
    group_mask = (group_idx_range < n_groups_per_example.unsqueeze(-1)).to(torch.float32)
    group_mask = group_mask * label_example_mask.unsqueeze(-1).to(torch.float32)
    # ^ (B, max_groups)
    group_denom = float(B)   # symmetric with strip; sum over groups, mean over batch

    l_group_total = torch.zeros((), device=device)
    for proc, pp in pred_group.items():
        proc_w_entry = weights.get(proc, 1.0)
        for param_name, pred_t in pp.items():
            target_t = label_group[proc][param_name].to(pred_t.device)
            per_elem = F.huber_loss(pred_t, target_t, reduction="none", delta=delta)
            per_group = per_elem.flatten(start_dim=2).mean(dim=-1)            # (B, max_g)
            l_group_total = l_group_total + _w(proc_w_entry, param_name) * (per_group * group_mask).sum() / group_denom

    return {
        "L_param/strip": l_strip_total,
        "L_param/group": l_group_total,
        "L_param/total": l_strip_total + l_group_total,
    }


def grafx_intermediate_audio_loss(
    pred_strip_inter: dict[str, torch.Tensor],   # {proc: (B, N, 2, T)}
    gt_strip_inter: dict[str, torch.Tensor],     # same shape
    pred_group_inter: dict[str, torch.Tensor],   # {proc: (B, G, 2, T)}
    gt_group_inter: dict[str, torch.Tensor],     # same shape
    label_track_mask: torch.Tensor,              # (B, N) bool
    n_groups_per_example: torch.Tensor,          # (B,) long
    label_example_mask: torch.Tensor,            # (B,) bool
    proc_weights: Optional[dict[str, float]] = None,
) -> dict[str, torch.Tensor]:
    """Per-(track,processor) audio supervision against label-rendered intermediates.

    For each of the 7 strip stages and 7 group stages, compute time-domain
    MSE between the encoder-rendered intermediate audio and the label-rendered
    intermediate audio. Masked slots (padding tracks, unused groups, unlabeled
    examples) are zeroed in both pred and gt so they contribute 0 to the loss.

    Why MSE and not MR-STFT: at intermediate stages we want the encoder's
    output to *exactly* reproduce the label-rendered audio (sample-by-sample
    equality where possible), not just be perceptually close to it. MSE is the
    right objective for that; MR-STFT throws away phase + mel-scales magnitude
    + A-weights, all of which are appropriate for "final mix sounds like
    engineer's mix" (L_recon) but wrong for "intermediate signals exactly
    match the labels' intermediate signals." MSE also has gradient magnitude
    comparable to the Huber on params used by L_param, so w_inter ~= 1 means
    "L_inter is as influential as L_param" (rather than ~10x dominant as
    MR-STFT was). And MSE is ~order-of-magnitude cheaper than 14 MR-STFTs.

    Reductions:
      For each stage, MSE is computed as `mean((pred - gt)^2)` over
      (B, N|G, 2, T), then scaled by INTERNAL_SCALE (see below). Per-stage
      losses are summed weighted, then divided by n_stages.

      L_inter/strip = mean_over_stages(INTERNAL_SCALE * MSE)
      L_inter/group = mean_over_stages(INTERNAL_SCALE * MSE)
      L_inter/total = L_inter/strip + L_inter/group

    INTERNAL_SCALE: 1e4. Audio amplitudes live in ~[-1, 1] so typical
    per-sample squared errors are ~1e-2; without this scale the MSE values
    (and gradients) come out ~4 orders of magnitude smaller than the Huber
    on log-space params used by L_param, so w_inter=1 would mean
    "L_inter is inert relative to L_param." With 1e4 scale, L_inter loss
    values land in the same order of magnitude as L_param (and gradient
    contribution is comparable), so `w_inter` reads as "relative influence
    vs L_param" (1.0 = same, 0.5 = half, 2.0 = double). Tunable per
    workload but should not need to change for our sample_rate / n_max
    configurations.

    Per-stage losses are returned for TB logging so we can see which stages
    converge first / which are stuck.
    """
    INTERNAL_SCALE = 1e4

    weights = proc_weights or {}
    device = label_example_mask.device

    B, N = label_track_mask.shape
    G = next(iter(pred_group_inter.values())).shape[1]

    strip_mask = (
        label_track_mask & label_example_mask.unsqueeze(-1)
    ).to(torch.float32).view(B, N, 1, 1)

    g_idx = torch.arange(G, device=device).unsqueeze(0)
    group_mask = (g_idx < n_groups_per_example.unsqueeze(-1)).to(torch.float32)
    group_mask = group_mask * label_example_mask.to(torch.float32).unsqueeze(-1)
    group_mask = group_mask.view(B, G, 1, 1)

    out: dict[str, torch.Tensor] = {}
    strip_total = torch.zeros((), device=device)
    n_strip_stages = 0
    for proc in pred_strip_inter:
        w = weights.get(proc, 1.0)
        pred = pred_strip_inter[proc] * strip_mask
        gt   = gt_strip_inter[proc]   * strip_mask
        l = INTERNAL_SCALE * F.mse_loss(pred, gt)
        out[f"L_inter/strip/{proc}"] = l
        strip_total = strip_total + w * l
        n_strip_stages += 1
    out["L_inter/strip"] = strip_total / max(1, n_strip_stages)

    group_total = torch.zeros((), device=device)
    n_group_stages = 0
    for proc in pred_group_inter:
        w = weights.get(proc, 1.0)
        pred = pred_group_inter[proc] * group_mask
        gt   = gt_group_inter[proc]   * group_mask
        l = INTERNAL_SCALE * F.mse_loss(pred, gt)
        out[f"L_inter/group/{proc}"] = l
        group_total = group_total + w * l
        n_group_stages += 1
    out["L_inter/group"] = group_total / max(1, n_group_stages)

    out["L_inter/total"] = out["L_inter/strip"] + out["L_inter/group"]
    return out


def grafx_param_consistency_loss(
    pred_strip_a: dict[str, dict[str, torch.Tensor]],
    pred_group_a: dict[str, dict[str, torch.Tensor]],
    pred_strip_b: dict[str, dict[str, torch.Tensor]],
    pred_group_b: dict[str, dict[str, torch.Tensor]],
    track_mask: torch.Tensor,            # (B, N_max) bool — active tracks
    n_groups_per_example: torch.Tensor,  # (B,) long
    *,
    param_weights: Optional[dict[str, dict[str, float]]] = None,
) -> dict[str, torch.Tensor]:
    """MSE between two predictions of the same song (different windows).

    Trains the encoder toward window-invariance: regardless of which window
    of the song it sees, it should produce the same song-global label.
    No supervision signal needed — applicable to labeled AND unlabeled songs.

    Args mirror the predicted-side of `grafx_param_huber_loss`. Both halves
    of the pair share the same `track_mask` and `n_groups_per_example`
    (same session → same active tracks/groups).

    Returns:
      { "L_cons/strip": ..., "L_cons/group": ..., "L_cons/total": ... }
    """
    B, N_max = track_mask.shape
    device = track_mask.device
    weights = param_weights or {}

    strip_mask = track_mask.to(torch.float32)            # (B, N_max)
    denom = float(B)

    l_strip = torch.zeros((), device=device)
    for proc, pp_a in pred_strip_a.items():
        pp_b = pred_strip_b[proc]
        proc_w = weights.get(proc, {}) if isinstance(weights.get(proc), dict) else {}
        for param_name, a in pp_a.items():
            b = pp_b[param_name]
            w = float(proc_w.get(param_name, 1.0)) if isinstance(proc_w, dict) else 1.0
            diff = (a - b).pow(2)
            per_track = diff.flatten(start_dim=2).mean(dim=-1)     # (B, N_max)
            l_strip = l_strip + w * (per_track * strip_mask).sum() / denom

    # Group mask
    max_groups = next(iter(next(iter(pred_group_a.values())).values())).shape[1]
    idx_range = torch.arange(max_groups, device=device).unsqueeze(0)
    group_mask = (idx_range < n_groups_per_example.unsqueeze(-1)).to(torch.float32)

    l_group = torch.zeros((), device=device)
    for proc, pp_a in pred_group_a.items():
        pp_b = pred_group_b[proc]
        proc_w = weights.get(proc, {}) if isinstance(weights.get(proc), dict) else {}
        for param_name, a in pp_a.items():
            b = pp_b[param_name]
            w = float(proc_w.get(param_name, 1.0)) if isinstance(proc_w, dict) else 1.0
            diff = (a - b).pow(2)
            per_group = diff.flatten(start_dim=2).mean(dim=-1)     # (B, max_g)
            l_group = l_group + w * (per_group * group_mask).sum() / denom

    return {
        "L_cons/strip": l_strip,
        "L_cons/group": l_group,
        "L_cons/total": l_strip + l_group,
    }


__all__ = [
    "AudioFeatureLoss",
    "HybridReconLoss",
    "make_grafx_prune_recon_loss",
    "grafx_param_huber_loss",
    "grafx_param_consistency_loss",
    "grafx_intermediate_audio_loss",
]
