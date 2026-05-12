# neural-mixing-console

A **differentiable audio mixing console** and an **encoder that inverts engineer mixes**.

The console is a fully differentiable signal chain — per-track channel strip
(`gain → 6-band EQ → compressor → soft-clipper → pan`) summed into a master
bus (`bus EQ → bus compressor`) plus a global trim. The encoder is a
permutation-invariant neural network that takes the raw stems of a song and
predicts the console parameters that best reconstruct a human engineer's
mix of those stems. Everything trains end-to-end: the reconstruction loss
backpropagates through the DSP graph into the encoder's parameter predictions.

> **Status:** research / work-in-progress. The model schema is evolving (current:
> v6.2 — no explicit bypass flags, "bypassed" expressed in the parameter space;
> compressor knee and clipper shape/ceiling fixed; separated EQ band ranges;
> fixed-loudness trim target). Not production-ready. See `notes/` (untracked) for
> the running experiment log.

---

## How it works

**The differentiable graph** (`models/`):
- `diff_eq.py` — RBJ biquad EQ (HPF, low-shelf, two parametric bells, high-shelf, LPF), differentiable via `torchaudio.functional.lfilter`.
- `diff_comp.py` — peak-envelope compressor with parallel attack/release smoothers + min-selection (differentiable substitute for the branched reference), soft knee fixed at 6 dB.
- `diff_clipper.py` — 4× oversampled anti-aliased soft-clipper; learned drive + dry/wet mix, curve shape and ceiling fixed.
- `diff_strip.py` / `diff_master_bus.py` / `diff_mixing_graph.py` — the per-track strip, the master bus, and the full multitrack render (variable track count via mask).
- `fsm_filter.py` — FFT-based one-pole IIR (used by the compressor smoothers; ~370× faster than a Python loop at training scale).

**The encoder** (`models/encoders.py`):
- Per-track hybrid backbone: a Transformer over log-mel frames fused (cross-attention) with a 1-D CNN over the raw waveform.
- Optional per-track MERT semantic embedding (`m-a-p/MERT-v1-95M`) added to the track tokens — carries the per-instrument prior knowledge.
- A permutation-invariant Transformer over the track tokens (no track-axis positional encoding).
- Heads: per-track strip params (22), master-bus params (13), global trim (dB). No bypass heads — a processor that should be "off" lands at its identity parameters (EQ gain 0, comp ratio 1, clip mix 0), gently encouraged by an L2 "identity prior."

**The training signal** (`training/`, "stage 3"): the dataset is `(stems, engineer_mix)` pairs — there are *no* ground-truth console parameters. All learning flows through the differentiable graph via a reconstruction loss (`training/losses.py`):
- `decoupled_recon_loss` — multi-resolution STFT + L1-time + log-mel + a side-channel (`(L−R)/2`) MSS term, all computed on a *loudness-normalized* prediction so the per-track params only feel "balance / spectrum / stereo" gradients, never "be louder/quieter"; plus an `L_loud` term that trains the trim head to a fixed target loudness.
- An identity-prior L2 (toward sensible engineer-default params).
- An optional `|mean(pan)|` penalty (against degenerate "shove the whole mix left" solutions).

**Reference DSP** (`reference/`): plain (non-differentiable) implementations of the same effects, plus parameter normalization (`param_norm.py`) and the parameter-sampling code. The differentiable versions are validated against these.

---

## Setup

Requires Python ≥ 3.11. Uses [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

For a CUDA-specific PyTorch build, configure the PyTorch index in `pyproject.toml`
(`[tool.uv.sources]` / `[[tool.uv.index]]` — there's a commented template at the
bottom of the file) or `uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cuXXX`
after `uv sync`.

---

## Data

The repo does **not** include audio (the raw datasets are hundreds of GB and
externally licensed). You need:

- **Cambridge-MT Multitrack Library** — real multitracks + reference full mixes.
  Non-commercial license; obtain from <https://cambridge-mt.com/ms/mtk/>.
- **Slakh2100** — MIDI-rendered multitracks + mixes (CC-licensed). <http://www.slakh.com/>

Place them under `source_audio/`, then run the prep pipeline (writes WebDataset
shards + metadata under `dmc-data/` — also not tracked):

```bash
# 1. Segment into 8-second shards.
uv run scripts/prepare_stage3_cambridge.py --output-dir dmc-data/stage3_cambridge_8s --segment-seconds 8.0 --segment-stride-seconds 8.0
uv run scripts/prepare_stage3_slakh.py     --output-dir dmc-data/stage3_slakh_8s     --segment-seconds 8.0 --segment-stride-seconds 8.0

# 2. Precompute per-stem MERT embeddings (keyed by source filename — independent of segment length).
uv run scripts/precompute_mert_cache.py --cache-dir dmc-data/mert_cache
```

(`scripts/regen_stage3_8s.sh` wraps step 1 for both datasets.)

---

## Training

Runs are driven by a small TOML config + a thin launcher (`scripts/run.py`),
which assembles the `train_stage3.py` command, sets `PYTORCH_CUDA_ALLOC_CONF`,
and runs a RAM watchdog that SIGTERMs the trainer if available memory drops
below a floor (this repo is developed on a unified-memory NVIDIA GB10, where an
unconstrained CUDA OOM can wedge the driver and reboot the box):

```bash
uv run scripts/run.py experiments/v6.2-round8.toml
```

A config (see `experiments/v6.2-round8.toml`) is `[run]` (the experiment name)
+ `[hardware]` (watchdog floor, env var) + `[train]` (every key maps 1:1 to a
`train_stage3.py` CLI flag — `key foo_bar → --foo-bar`; the `experiments/*.toml`
files are the version-controlled record of "what we tried"). `--out-dir` and
`--tb-logdir` default to `dmc-data/checkpoints/<name>` and `runs/<name>`.

Or run the trainer directly (`uv run training/train_stage3.py --help` for the
full flag list):

```bash
uv run training/train_stage3.py \
    --shard-dirs dmc-data/stage3_cambridge_8s dmc-data/stage3_slakh_8s \
    --out-dir dmc-data/checkpoints/run1 --tb-logdir runs/run1 \
    --steps 6000 --batch-size 4 --grad-accum-steps 2 --n-max 42 --lr 2e-5 \
    --min-track-rms-dbfs -45.0 \
    --w-stereo-side 2.5 --w-pan-mean 0.2 --w-identity-prior 5e-3 \
    --w-loud 0.3 --trim-max-db 18.0 --loudness-target-dbfs -15.0
```

Checkpoints land in `--out-dir` (incl. `mix_encoder_best.pt` — the lowest-val-loss
state, so a usable artifact survives even if the run stops mid-flight);
TensorBoard logs (scalars + preview audio + spectrograms) in `--tb-logdir`.

---

## Inference

```bash
uv run infer.py \
    --checkpoint dmc-data/checkpoints/run1/mix_encoder_best.pt \
    --tracks-dir /path/to/a/folder/of/stem/wavs \
    --out-dir /path/to/output \
    --n-max 42 --encoder-window-seconds 8.0 \
    [--ref-mix /path/to/engineer_mix.wav]
```

Outputs: `rendered_mix.wav` (the model's mix), `sum_baseline.wav` (a mask-aware
sum of the stems — sanity reference), `ref_mix.wav` (copy of `--ref-mix`, if
given, for A/B), and `params.json` (every predicted per-track and bus parameter,
plus "effective bypass" flags derived by snapping near-identity params).

---

## Evaluation

```bash
# Run a checkpoint on a held-out split: recon losses, loudness match,
# per-parameter distributions (incl. an at-range-edge % = the corner-pinning
# detector), effective-bypass rates, pan distribution, health flags.
uv run eval.py recon --checkpoint dmc-data/checkpoints/run1/mix_encoder_best.pt \
    --shard-dirs dmc-data/stage3_cambridge_8s dmc-data/stage3_slakh_8s \
    --split test --n-batches 50 --loudness-target-dbfs -15.0

# Aggregate already-rendered params.json files (no model needed) — same
# per-parameter / effective-bypass / pan / trim report.
uv run eval.py params --params-glob 'dmc-data/inference/*/params.json'
```

Add `--out report.json` in either mode to also dump the report as JSON.

---

## Repo layout

```
models/        differentiable DSP (EQ, comp, clipper, strip, bus, mixing graph) + the MixEncoder
reference/     non-differentiable reference DSP + param normalization + "effective bypass" helpers + sampling
training/      data loading (WebDataset shards + MERT cache), losses, the stage-3 train loop
scripts/       data prep, MERT cache build, run.py (experiment launcher), misc helpers
experiments/   one TOML config per training run (the version-controlled experiment record)
infer.py       single-session inference: stems → predicted mix + params
eval.py        checkpoint evaluation / params.json aggregation
*.md           design notes (spec.md, model_spec.md)
```

Not tracked (see `.gitignore`): `source_audio/` (raw datasets), `dmc-data/`
(shards, MERT cache, checkpoints, renders), `runs/` (TensorBoard + per-run logs),
`logs/`, `notes/` (maintainer's experiment log + retired launchers), `.venv/`.

---

## Known caveats

- Research code in active development — the parameter schema and the loss are
  still evolving (current: v6.2). Don't expect API stability.
- Trained checkpoints are not in the repo (too large for git). A reference
  checkpoint may be published separately (GitHub Releases / HuggingFace).
- The data pipeline assumes you've separately obtained the raw datasets; only the
  prep/training/eval code is here, not the audio.

## License

TODO — pick one. Note the source datasets carry their own licenses (Cambridge-MT
is non-commercial); anything derived from them inherits those terms.
