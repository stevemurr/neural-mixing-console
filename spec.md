# DMC Dataset Spec — Stage 3 v6

**Project:** Differentiable Mixing Console (real-time, parameter-estimation)
**Scope of this spec:** Stage 3 multitrack dataset preparation for the v6
MixEncoder. Real-data only — no synthetic supervision.
**Format conventions:** all audio 48 kHz / 24-bit FLAC; encoder outputs in
float32 normalized form (see `model_spec.md` §6).

---

## 1. Goal

Produce per-window stage-3 bundles that pair raw multitrack stems with the
published reference mix, sharded for `webdataset` streaming. The trainer
(`training/train_stage3.py`) consumes these via `make_stage3_dataset` /
`collate_stage3` in `training/data.py`.

Two source datasets are supported and can be mixed at training time:
- **Cambridge-MT** — full multitrack sessions with reference mixes, prepared
  by `scripts/prepare_stage3_cambridge.py`.
- **Slakh2100** — synthesized multitrack with reference mixes, prepared by
  `scripts/prepare_stage3_slakh.py`.

The `MixEncoder` v6 is trained on `(stems, engineer_mix)` only — no
ground-truth params. Supervision is reconstruction of the engineer mix
plus a self-supervised bypass-consistency loss (see `model_spec.md` §5).

---

## 2. Source audio (user provides)

- `source_audio/cambridge-mt/<session>/...` — Cambridge-MT downloads,
  unzipped. Each session contains the per-track stems (mono or stereo) and
  one or more reference-mix WAV files (recognized by name patterns like
  `* reference mix*.wav` or `*Reference Mix*.wav`).
- `source_audio/moisesdb/moisesdb/moisesdb_v0.1/<uuid>/...` — moisesdb-style
  layout (per-instrument-class subfolders). Used at inference time, not
  currently in the stage-3 prep pipeline.
- `source_audio/slakh2100/...` — Slakh2100 layout (per-track FLACs +
  `mixture.wav` reference + `metadata.yaml`).

The prep scripts emit aligned, segmented bundles to:
- `dmc-data/stage3_cambridge/shards/*.tar`
- `dmc-data/stage3_slakh/shards/*.tar`

Plus a per-shard index (`stage3_<dataset>_index.jsonl`) and a manifest with
sample counts and split histograms.

---

## 3. Window construction

Per source song:
1. Resample all raw tracks and the reference mix to 48 kHz.
2. **Cambridge** only: verify sample alignment between the mix and
   sum-of-tracks via cross-correlation (peak at lag 0 ± 5 samples,
   correlation > 0.5). Reject the song if it can't be aligned.
3. Per `--segment-seconds` window (default 6 s), keep tracks with `rms_db
   > −50` in that window. Drop the window if fewer than 2 tracks active.
4. Drop windows where mix RMS < −40 dBFS (silence / pre-roll).
5. Emit non-overlapping windows.

---

## 4. Per-window bundle

Each example in the tar shard:

```
{example_id}.mix.flac              # stereo, 48k, 24-bit
{example_id}.track_000.flac        # mono or stereo per track
{example_id}.track_001.flac
...
{example_id}.track_NNN.flac        # NNN up to track_count - 1
{example_id}.meta.json
```

`meta.json`:
```json
{
  "example_id": "uuid4",
  "stage": "stage3_cambridge",
  "session": "BigHeadToddAndTheMonsters_HeyDelilah_Full",
  "split": "train",
  "window_start_sample": 1440000,
  "duration_samples": 288000,
  "sample_rate": 48000,
  "track_count": 12,
  "tracks": [
    {
      "index": 0,
      "filename": "Drums_OH_L.wav",
      "channels": 2,
      "instrument_label": "drums",
      "rms_db": -22.4
    },
    ...
  ]
}
```

`filename` is the stem's original filename (without path); used by the
MERT cache lookup at training time to attach per-track semantic embeddings.

---

## 5. MERT semantic embeddings (precomputed cache)

Per-track 768-d MERT-v1-95M embeddings are precomputed offline and cached
under `dmc-data/mert_cache/<dataset>/<session>.npz`. Each .npz holds
`filenames` (object array) and `embeddings` ((N, D) float16). Loaded
lazily by `MertCacheLookup` in `training/data.py` — only sessions that
appear in the consumed shards are read into RAM.

Run `python scripts/precompute_mert_cache.py` after the shards are
prepared. The trainer auto-attaches embeddings per `(dataset, session,
filename)` tuple; missing tracks get a zero embedding.

---

## 6. Train / val / test splits

Splits hash on `session` so all windows of one song stay together:

```python
bucket = int(hashlib.sha256(session.encode()).hexdigest(), 16) % 100
split = "train" if bucket < 90 else ("val" if bucket < 95 else "test")
```

The prep scripts write the chosen split into each meta.json's `split`
field, and `make_stage3_dataset(split=...)` filters by that.

---

## 7. On-disk layout (target)

```
dmc-data/
├── stage3_cambridge/
│   ├── shards/stage3_cambridge_*.tar
│   ├── stage3_cambridge_index.jsonl
│   └── stage3_cambridge_manifest.json
├── stage3_slakh/
│   ├── shards/stage3_slakh_*.tar
│   ├── stage3_slakh_index.jsonl
│   └── stage3_slakh_manifest.json
├── mert_cache/
│   ├── cambridge/<session>.npz
│   └── slakh/<session>.npz
├── checkpoints/
│   └── stage3_v6/
│       ├── mix_encoder_latest.pt
│       └── mix_encoder_step*.pt
└── eval/
    ├── eval_stage3_*.json            # in-distribution audit
    └── v6_prediction_audit.json      # out-of-distribution audit

source_audio/
├── cambridge-mt/...                  # user-provided
├── moisesdb/...                      # user-provided
├── slakh2100/...                     # user-provided
└── predicted/                        # infer.py outputs
    └── <song>/
        ├── rendered_mix.wav
        ├── sum_baseline.wav
        ├── ref_mix.wav               # if --ref-mix provided
        └── params.json
```

---

## 8. Generation pipeline

Order of operations:

1. **Acquire source data** (manual): download Cambridge-MT zips and unzip
   into `source_audio/cambridge-mt/`. Place Slakh under `source_audio/slakh2100/`.
2. **Prepare shards**:
   ```
   python scripts/prepare_stage3_cambridge.py --out-dir dmc-data/stage3_cambridge ...
   python scripts/prepare_stage3_slakh.py     --out-dir dmc-data/stage3_slakh ...
   ```
3. **Precompute MERT cache**:
   ```
   python scripts/precompute_mert_cache.py --shard-dirs dmc-data/stage3_cambridge dmc-data/stage3_slakh
   ```
4. **Train**: see `model_spec.md` §7 for the recommended command.
5. **Inference**: `python infer.py --checkpoint ... --tracks-dir ...`.
6. **Eval**: `python eval/run_eval.py --checkpoint ...` and
   `python scripts/aggregate_predictions.py`.

---

## 9. Reproducibility

- All audio decoded as float32 via soundfile, resampled with torchaudio's
  Resample (Kaiser window). Sample-rate verification in `_decode_flac`.
- WebDataset shards are tar files; per-example file ordering inside a tar
  is lex-sorted.
- Dataset splits are deterministic (sha256 hash of session name) so adding
  a new song doesn't reshuffle existing assignments.
- Checkpoint files include the trainer's full `args` dict for round-trip
  reproducibility.
