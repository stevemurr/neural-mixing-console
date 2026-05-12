# DMC Model Spec — Stage 3 v6

**Project:** Differentiable Mixing Console — predict mixing-console parameters
from raw multitrack stems and a target reference mix.
**Scope:** PyTorch implementations of the differentiable DSP graph, the v6
MixEncoder, loss functions, training loop, and evaluation. Consumes the
stage-3 multitrack dataset described in `spec.md`.

---

## 1. Goal — single trainable artifact

`MixEncoder` (v6 schema): consumes N raw tracks (+ MERT semantic embeddings)
and emits the parameters of a differentiable mixing graph. Trained end-to-end
on `(stems, engineer_mix)` pairs via reconstruction loss against the engineer
mix; rendered through the differentiable graph backward.

No synthetic supervision, no per-effect pretraining, no separate channel /
bus encoders. The whole pipeline is one model + one graph + one loss.

---

## 2. Common conventions

- All audio float32, 48 kHz.
- Tensor convention: `(B, C, T)` — batch, channels, time. Mono is `C=1`,
  stereo is `C=2`. Per-track tensors carry an extra leading `N_max` axis with
  a parallel `track_mask` of shape `(B, N_max)`.
- Param convention: encoder output is sigmoid → **normalized [0, 1]**. A
  denorm layer (built from `reference/param_norm.py:PARAM_RANGES`) converts
  to physical units before feeding the differentiable DSP.
- Bypass flags: per-effect sigmoid → bool at threshold 0.5. The DSP modules
  short-circuit to identity when bypass=True for the corresponding bit.

---

## 3. Differentiable DSP blocks

The v6 graph uses five Diff* modules:

### 3.1 `DiffEQ` — 6-band cascade (HPF · LS · P1 · P2 · HS · LPF)
- HPF / LPF: 2nd-order Butterworth on `freq`.
- LS / HS: RBJ shelf formula on `(freq, gain, Q)`.
- P1 / P2: RBJ peaking formula on `(freq, gain, Q)`.
- All coefficients are differentiable functions of params (smooth on the
  supported ranges). Biquad chain via `torchaudio.functional.lfilter`,
  per-channel for stereo, no cross-channel state.
- Per-band bypass flag short-circuits to identity.

### 3.2 `DiffComp` — peak-envelope compressor
Decision locked: parallel attack/release smoothers + min trick (preserves
gradients without temperature dials). Soft-knee, stereo-linked detector.
v6 has **no makeup gain** — compressor is attenuation-only; loudness is
handled globally by the trim head outside the strip.

### 3.3 `DiffClipper` — anti-aliased variable-shape clipper
Bitwig Over-style: single `shape` param interpolates from soft tanh to hard
threshold clip; `drive_db` and `mix` control input gain and parallel blend;
`ceiling_db` sets the output ceiling. Bypass flag short-circuits to identity.

### 3.4 `DiffStrip` — per-track chain `gain → EQ → comp → clip → pan`
Mono-to-stereo via constant-power pan; stereo-to-stereo via balance.
Comp threshold can be specified absolute (`threshold_db`) or relative to the
track's RMS via `threshold_offset_db + track_rms_db` (engineer prior:
thresholds sit near signal level — see §6).

### 3.5 `DiffMasterBus` — `bus EQ → bus comp` on stereo input
v6 has **no bus clipper** — peak control is handled at inference by the
always-on safety limiter applied after the trim head. v6 bus comp also has
no makeup gain.

### 3.6 `DiffMixingGraph` — full multi-track graph
For each active track `k`: `post_strip_k = DiffStrip(track_k, params_k)`.
Bus input is the masked sum: `bus_input = sum_k post_strip_k`.
Final mix: `mix = DiffMasterBus(bus_input, bus_params)`.

The trim_db and the always-on safety limiter live **outside** this graph —
applied at inference time by `infer.py` after the graph render. Training
loss sees the un-trimmed, un-limited render so the loudness signal flows
cleanly into the trim head via `decoupled_recon_loss`.

---

## 4. `MixEncoder` v6

Input:
- `tracks`: `(B, N_max, 2, T)` padded multi-track tensor.
- `track_mask`: `(B, N_max)` bool — True for active tracks.
- `mert_embeddings`: `(B, N_max, 768)` MERT-v1-95M per-track semantic
  embeddings (precomputed offline via `scripts/precompute_mert_cache.py`).

Architecture:
1. **Per-track HybridBackbone** — log-mel transformer + waveform 1-D CNN,
   fused via cross-attention. Per-track features pooled to a CLS token.
2. **MERT injection** — per-track 768-d embedding projected to `d_model=384`
   and added to the track tokens.
3. **Track-axis transformer** (4 layers, no positional encoding —
   permutation invariant).
4. Heads:
   - `head_track`: per-track strip params (sigmoid, 25 dims) + bypass logits (8 dims).
   - `head_bus`: bus params (sigmoid, 14 dims).
   - `head_bus_bypass`: bus bypass logits (5 dims).
   - `head_trim`: scalar `trim_db` ∈ `[-trim_max_db, +trim_max_db]` via tanh.
     Zero-init so the model starts predicting 0 dB; loudness departs from
     zero only as `L_loud` accumulates gradient.

Output dict:
```
{
    "track_params":  (B, N_max, 25)  sigmoid in [0, 1]
    "track_bypass":  (B, N_max, 8)   raw logits (BCE-friendly)
    "bus_params":    (B, 14)         sigmoid in [0, 1]
    "bus_bypass":    (B, 5)          raw logits
    "trim_db":       (B,)            dB
}
```

Engineer-default initial biases on each head's final linear (see
`models/encoder_priors.py`) so the model starts at plausible baseline
settings (e.g. HPF at 40 Hz, comp ratio 2.5:1 with -18 dB threshold) rather
than at sigmoid mid-range.

---

## 5. Loss functions

Total loss (per `training/losses.py` and `training/train_stage3.py`):

```
L_total = w_recon * L_recon                                # decoupled_recon_loss
        + w_bypass_consistency * L_bypass_consist          # bypass_consistency_loss
        + w_pan_mean * L_pan_mean                          # optional, off by default
```

### 5.1 `decoupled_recon_loss` — timbre vs loudness decoupling

`L_recon = L_timbre + w_loud * L_loud` where:

- `L_timbre` is `w_time * L_time + w_mss * L_mss + w_log_mel * L_log_mel
  (+ w_stereo_side * L_stereo_side)`, all computed on a *loudness-normalized*
  prediction (rescaled so its combined-stereo RMS equals the target's). This
  ensures the strip / bus params only feel gradients from spectrum, transient
  shape, and stereo imaging — never from "be louder / quieter."
- `L_loud` is L1 between predicted `trim_db` and the *detached* dB-RMS
  difference between target and un-trimmed pred. The detach ensures gradient
  flows only into the trim head.

`L_stereo_side` is MSS on the side channel `(L − R) / 2` only — strong
gradient for pan / stereo-imaging params that the standard L+R MSS
doesn't push hard on.

### 5.2 `bypass_consistency_loss` — self-supervised bypass supervision

Stage 3 has no ground-truth bypass labels. Without direct supervision the
bypass head collapses (observed empirically: strip heads flat at p≈0.05,
bus heads flat at p≈0.5).

Pseudo-targets are derived from whether the predicted params sit at the
effect's identity value:
- EQ HPF: `hpf_freq_norm ≈ 0` (HPF disabled at ~20 Hz).
- EQ LS / P1 / P2 / HS: `gain_norm ≈ 0.5` (gain ≈ 0 dB).
- EQ LPF: `lpf_freq_norm ≈ 1` (LPF disabled at ~20 kHz).
- Comp: `ratio_norm ≈ 0` (ratio ≈ 1 = pass-through).
- Clip: `drive_norm ≈ 0` OR `mix_norm ≈ 0` (either disables it).
- Bus EQ bands: `gain_norm` at the band's identity value.
- Bus comp: `bus_ratio_norm ≈ 0`.

BCE between the bypass logit and these targets, with targets detached so
the loss flows only into the bypass head (decouples it from the param
head's degeneracies). `deadband=0.05` ≈ ±1.2 dB on a ±12 dB EQ band.

### 5.3 `pan_mean_penalty` — optional, prefer `--w-stereo-side`

Penalizes `|E[pan]|` across active tracks. Off by default
(`w_pan_mean=0.0`) because it's trivially satisfied by `pan = 0 ∀ tracks`,
which is the failure mode it's supposed to prevent. Use `w_stereo_side > 0`
in `decoupled_recon_loss` instead — that loss penalizes failures to
reproduce the side channel, which a center-collapsed prediction cannot do.

---

## 6. Param ranges (v6 schema)

See `reference/param_norm.py` for the canonical table. Highlights:

- `gain_db ∈ [-24, +12]` linear.
- EQ: HPF `[20, 500]` log; LPF `[5000, 20000]` log; LS/HS `[60, 500]` /
  `[2000, 12000]` log freq; gains `[-12, +12]` linear; Qs `[0.3, 1.5]` log.
- Peaks P1 / P2: gains `[-15, +15]` linear, Qs `[0.3, 10]` log; freqs
  `[100, 2000]` / `[1000, 10000]` log.
- Comp: `threshold_db ∈ [-60, 0]` linear; `ratio ∈ [1, 20]` log;
  `attack_ms ∈ [0.5, 100]` log; `release_ms ∈ [10, 1000]` log;
  `knee_db ∈ [0, 12]` linear.
- Comp **RMS-relative** (engineer prior): `threshold_offset_db ∈ [-30, +6]`
  linear; effective threshold = `track_rms_db + offset`. The encoder's
  normalized `threshold_db` head is reinterpreted as offset when
  `--use-rms-relative-threshold 1` (the training default).
- Bus EQ: 4 bands (low_boost / low_attn / mid / air) with band-specific
  gain ranges (`[0, +4]` for boost-only, `[-4, 0]` for cut-only,
  `[-2, +2]` for symmetric mid).
- Bus comp: `bus_threshold_db ∈ [-24, 0]`, `bus_ratio ∈ [1, 4]`.
- `trim_db ∈ [-12, +12]` (set by `MixEncoder.trim_max_db`).

---

## 7. Training loop

See `training/train_stage3.py`. Recommended invocation:

```
python training/train_stage3.py \
  --shard-dirs dmc-data/stage3_cambridge dmc-data/stage3_slakh \
  --init-from dmc-data/checkpoints/stage3_v6/mix_encoder_latest.pt \
  --preserve-head-priors \
  --steps 20000 --batch-size 4 --n-max 21 --lr 1e-4 \
  --w-recon 1.0 --w-log-mel 0.5 --w-loud 0.1 \
  --w-stereo-side 1.5 --w-bypass-consistency 0.5 \
  --val-every 500
```

Key flags:
- `--preserve-head-priors`: skip the final-linear bias of each predictive
  head on warm-start; re-apply `init_encoder_priors` after. Counters the
  case where a saturating prior bias (e.g. `p2_freq` pinned at the prior
  max) carries forward and the new training never escapes it.
- `--w-stereo-side`: enable the side-channel MSS loss for stereo width.
- `--w-bypass-consistency`: enable the self-supervised bypass loss.
- `--no-rms-relative-threshold`: switch back to absolute comp threshold.

TensorBoard logs go to `<out-dir>/tb/` by default.

---

## 8. Evaluation

Two complementary scripts, both producing JSON reports under `dmc-data/eval/`
that share the same param/bypass distribution shape so they can be diffed
side by side:

- **`eval/run_eval.py`** — in-distribution. Loads the v6 checkpoint, runs
  the held-out val split, computes the same losses the trainer optimizes
  (`L_recon`, `L_timbre`, `L_time`, `L_mss`, `L_log_mel`, `L_loud`, optional
  `L_stereo_side`) plus per-param distribution stats (mean, std, percentiles,
  boundary pinning) and bypass-head distribution stats.
- **`scripts/aggregate_predictions.py`** — out-of-distribution. Walks every
  `source_audio/predicted/**/params.json`, re-normalizes the predicted
  values via `PARAM_RANGES`, and computes the same distribution stats. Use
  this to detect collapses on the songs you actually inferred against.

Together they tell training-collapse from generalization issues: if a
collapse appears in both, training is the fix; if only out-of-dist, it's a
distribution mismatch.

---

## 9. Inference

See `infer.py`. Loads stems from a folder, picks the loudest
`--encoder-window-seconds` window for the encoder forward pass, then renders
the full song length via chunked rendering (`--render-chunk-seconds`,
default 30 s, with ~2 s of pre-roll for the comp envelope to settle).

Outputs (under `--out-dir`):
- `rendered_mix.wav` — predicted mix (after trim + safety limiter).
- `sum_baseline.wav` — mask-aware sum of stems (no-op baseline).
- `params.json` — denormalized per-track and bus params + bypass probabilities.
- `ref_mix.wav` — copy of `--ref-mix` if provided.

The trim_db scalar is applied after the graph render; the always-on safety
limiter (tanh soft-clip at `--safety-limiter-db`, default -0.3 dBFS) runs
after the trim to guarantee output peak ≤ ceiling.
