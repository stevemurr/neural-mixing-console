# realtime/ — implementation plan

Roadmap for the realtime Python prototype that runs the trained
multitrack mixing model on live audio. V0 (stems passthrough player)
is shipped (`7f4d688`); this file is the design + V1+ plan.

## Goal

A standalone Python app that loads a multitrack project, plays it
through `sounddevice`, and applies the trained `MixEncoder`'s
predicted mixing parameters to each track in (near-)realtime via a
forward DSP graph. Python first for iteration speed; port the hot
path to C++ + JUCE/CLAP once the architecture has stopped moving.

Two-thread design:

- **Audio thread** (sounddevice callback, ~5–25 ms per call). DSP only.
  Must not allocate, must not block, must not call into Python-level
  blocking APIs. Reads parameters from an atomic snapshot the
  inference thread updates.
- **Inference thread** (background, ~250 ms cadence). Reads the last
  8 s of audio from per-track ring buffers, runs the encoder on
  CPU/GPU, writes new parameter targets. Inference latency (10–80 ms
  on CPU) is fine — it doesn't block audio.

## Versions

| Version | Scope | New code |
|---|---|---|
| **V0** ✓ | Stems passthrough sum, sounddevice playback, CLI loop. | `session.py`, `engine.py`, `cli.py` |
| **V1** | Ring buffers; encoder wrapper; scheduler thread; print predicted params each tick. NO DSP applied yet — just verify the model-in-the-loop architecture, cadence, and that predictions are stable. | `ring_buffer.py`, `inference.py`, `scheduler.py` |
| **V2** | Realtime-safe DSP chain (per-track strip + master bus + trim). Apply latest predicted params with no smoothing (will click on transitions). Hear the model's mix. | `dsp.py` |
| **V3** | Parameter smoother (one-pole, per-param-type ramps). Clicks disappear. | `smoother.py` |
| **V4** | Minimal status / GUI hooks: live param display, ring-buffer fullness, scheduler-late warnings, manual param overrides. | `status.py` (and optional `gui.py`) |
| **V5+** | Live MERT computation (new stems not in cache); session save/load; A/B between encoder checkpoints; record output to disk; eventually port to C++ + Anira. |  |

## Architecture (V3 = the target shape)

```
       ┌──────────────────────────────────────────────────────────┐
       │  Session: stems on disk (one .wav per track)             │
       │  MERT cache lookup at load time                          │
       └────────────────────┬─────────────────────────────────────┘
                            │ loaded into memory once
                            ↓
                   stems[N][2, T_total]   mert[N, 768]
                            │
                            ↓
       ┌─────────────── AudioEngine.audio_callback ──────────────┐
       │   playhead pos → slice each stem: (N, 2, block_size)    │
       │   write each slice → RingBuffer[i] (per-track 8 s)      │
       │   smoother.step(block_size) → current params            │
       │   per-track: RealtimeTrackStrip.process(slice, params)  │
       │   sum + RealtimeMasterBus.process(...)                  │
       │   → outdata (to soundcard)                              │
       └─────────────────────────────────────────────────────────┘
                            │  smoothed params (atomic snapshot)
                            ↑
       ┌─────────────── InferenceScheduler (thread) ─────────────┐
       │   every cadence_ms (default 250):                       │
       │     snapshot = [rb.snapshot() for rb in ring_buffers]   │
       │     tracks_t = stack(snapshot) → (1, N, 2, 384k)        │
       │     params = encoder.predict(tracks_t, mert, mask)      │
       │     smoother.set_target(params)                         │
       │   drops queued ticks if previous predict is still running│
       └─────────────────────────────────────────────────────────┘
```

## Module sketches (V1–V3)

### `ring_buffer.py` (V1)

```python
class RingBuffer:
    """Per-track circular buffer of recent audio.

    Sized to hold `--encoder-window-seconds` * sample_rate samples
    (default 8 s × 48 kHz = 384 000 samples per channel).

    Write side (audio thread): `write(block)` appends `block` samples,
    advances the write pointer. Wraps in-place. No allocs.

    Read side (inference thread): `snapshot()` returns a (n_channels,
    capacity) np.ndarray containing the *most recent capacity samples*
    in chronological order. Allocates each call (inference thread is
    not realtime-critical). Safe to call concurrently with `write`;
    reads a consistent ~8 s window using an atomic write-pointer load
    and unwrapping the ring after the copy.
    """
    def __init__(self, n_channels: int, capacity_samples: int): ...
    def write(self, block: np.ndarray) -> None: ...        # audio thread
    def snapshot(self) -> np.ndarray: ...                   # inference thread
    @property
    def is_warm(self) -> bool: ...   # True once `capacity_samples` have been written
                                     # (before this, snapshot would return mostly zeros)
```

Lock-free design: single producer (audio thread), single consumer
(inference thread). Write pointer is `threading.atomic`-style — in
practice a `numpy.int64[0]` view since CPython's GIL makes single-int
writes atomic. The consumer reads the pointer, then copies the ring
into a contiguous output buffer. Slight tearing possible (the
consumer might miss the last few samples written during the copy) —
that's fine, the encoder window is 8 s and a few ms of staleness on
either end of the window doesn't matter.

### `inference.py` (V1)

```python
class EncoderWrapper:
    """Loads a MixEncoder checkpoint, runs no-grad forward, denormalizes
    the output to physical DSP parameters.

    `predict()` is called from the inference thread. The returned dict
    is structured for direct consumption by the DSP modules / smoother.
    """
    def __init__(
        self,
        ckpt_path: Path,
        device: str = "cpu",          # 'cpu' or 'cuda'
        trim_max_db: float = 18.0,    # must match training config
    ): ...

    @torch.no_grad()
    def predict(
        self,
        tracks: torch.Tensor,         # (1, N_max, 2, T) — fp32, padded with zeros
        track_mask: torch.Tensor,     # (1, N_max) bool — True = real, False = pad
        mert: torch.Tensor,           # (1, N_max, 768)
    ) -> dict:
        # returns {
        #   "strip": np.ndarray (N_active, 22)  — denormalized per-track params
        #                                         (gain_db, EQ × 14, comp × 4, clip × 2, pan)
        #   "bus":   np.ndarray (13,)           — denormalized bus params
        #   "trim_db": float                    — global trim
        # }
```

Loads via the same `_load_compatible_state` machinery as
`training/train_stage3.py`, so it accepts both bare state_dicts
(`mix_encoder_latest.pt`) and dict-wrapped checkpoints (`best.pt`,
`ema.pt`).

The denormalization uses `reference/param_norm.PARAM_RANGES` so the
output is in the same physical units as the DSP expects (freq in Hz,
gain in dB, ratio dimensionless, etc.).

### `scheduler.py` (V1)

```python
class InferenceScheduler:
    """Background thread that runs the encoder at fixed cadence and writes
    the result into a target-param store. Drop-if-busy: if the previous
    predict hasn't finished by the next tick, the next tick is skipped
    (not queued — we always want the *most recent* 8-s snapshot, not a
    backlog of stale ones)."""

    def __init__(
        self,
        encoder: EncoderWrapper,
        session: Session,
        ring_buffers: list[RingBuffer],
        on_new_params: Callable[[dict], None],   # smoother.set_target in V3, print in V1
        cadence_ms: float = 250.0,
    ): ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    # exposes: last_predict_ms (rolling EWMA), missed_ticks, queue_depth
```

The thread loop is straightforward: `while running: wait_for_next_tick();
snapshot_all_rings(); predict(); on_new_params(...)`. The interesting
bit is the cadence + drop logic, which we'll need for diagnostic
visibility (a 60 ms inference on CPU ≈ no drops; a 400 ms inference
means we're behind and predictions are stale).

### `dsp.py` (V2)

Realtime-safe forward implementations of the strip + bus DSP. NOT the
differentiable versions in `models/diff_*` — those use FFT-based
filters and block-sized envelope smoothers, neither of which is
realtime-friendly. Port from `reference/` (which are sample-by-sample
forward impls), with two changes:

1. **Carry state across blocks.** Biquad filters need `(z1, z2)` per
   biquad per channel per track. Compressor needs envelope follower
   state. Each `RealtimeTrackStrip` instance owns its state arrays;
   they're sized at construction and never re-allocated.
2. **Performance.** scipy.signal.sosfilt with state works for biquads
   (~vectorized internally). Compressor envelope follower is the slow
   path in pure Python; numba `@njit` lifts it to ~native speed
   (10–100× speedup). Soft clipper + pan are stateless and vectorize
   trivially with numpy.

```python
class RealtimeTrackStrip:
    """gain → EQ (HPF, LS, P1, P2, HS, LPF) → comp → soft-clipper → pan.

    Sample-aware: process(block, params) returns a (2, block_size)
    output. Filter state is carried inside the strip instance.
    """
    def __init__(self, sample_rate: int = 48_000): ...
    def process(self, audio: np.ndarray, params: dict) -> np.ndarray: ...
    # params keys (denormalized, from EncoderWrapper.predict):
    #   gain_db (float), hpf_freq, ls_freq, ls_gain, ls_q,
    #   p1_freq, p1_gain, p1_q, p2_freq, p2_gain, p2_q,
    #   hs_freq, hs_gain, hs_q, lpf_freq,
    #   threshold_db, ratio, attack_ms, release_ms,
    #   clip_drive_db, clip_mix, pan

class RealtimeMasterBus:
    """bus EQ (low_boost, low_attn, mid, air) → bus comp → trim."""
    def __init__(self, sample_rate: int = 48_000): ...
    def process(self, audio: np.ndarray, bus_params: dict,
                trim_db: float) -> np.ndarray: ...
```

Numerical validation: write a script that takes the same input + the
same params, runs it through (a) the existing PyTorch differentiable
forward and (b) our realtime impl, and asserts they match to ~1e-3
peak error. Quick way to catch porting bugs.

### `smoother.py` (V3)

```python
class ParamSmoother:
    """One-pole interpolation between target params and current params.

    Per-param-type curve:
      - LINEAR for gain_db, pan, trim_db, ls_gain, p1_gain, etc.
      - LOG    for all *_freq params (so 100→200 Hz sweeps musically)
      - GEOM   for ratio (the comp ratio 1:1→4:1 doesn't pass through
               2.5:1 linearly — that's "harder" mid-sweep than the
               endpoints suggest)

    Time constant: 100 ms by default. `step_to_current(n_samples)`
    advances the one-pole state by `n_samples` and returns the new
    current params. Cheap (closed-form exponential per param).
    """
    TYPE_LINEAR = "lin"; TYPE_LOG = "log"; TYPE_GEOM = "geom"
    PARAM_TYPES: dict[str, str] = {
        "gain_db": TYPE_LINEAR, "pan": TYPE_LINEAR,
        "ls_gain": TYPE_LINEAR, "p1_gain": TYPE_LINEAR,
        "p2_gain": TYPE_LINEAR, "hs_gain": TYPE_LINEAR,
        "ls_q": TYPE_LINEAR, "p1_q": TYPE_LINEAR, "p2_q": TYPE_LINEAR,
        "hs_q": TYPE_LINEAR, "threshold_db": TYPE_LINEAR,
        "attack_ms": TYPE_LOG, "release_ms": TYPE_LOG,
        "clip_drive_db": TYPE_LINEAR, "clip_mix": TYPE_LINEAR,
        "hpf_freq": TYPE_LOG, "ls_freq": TYPE_LOG,
        "p1_freq": TYPE_LOG, "p2_freq": TYPE_LOG,
        "hs_freq": TYPE_LOG, "lpf_freq": TYPE_LOG,
        "ratio": TYPE_GEOM,
        # bus params analogous
        "trim_db": TYPE_LINEAR,
    }

    def __init__(self, time_constant_ms: float = 100.0,
                 sample_rate: int = 48_000): ...
    def set_target(self, params: dict) -> None: ...     # inference thread
    def step(self, n_samples: int) -> dict: ...          # audio thread
```

Threading: `set_target` writes to a "pending target" slot. `step`
reads + clears the pending slot atomically (or just on every call —
since the inference thread only writes new pending targets every 250
ms, and the audio thread reads every ~10 ms, occasional double-reads
are harmless).

## Performance considerations (recap from design doc)

Ranked by leverage:

1. **MERT bottleneck.** ~95 M params, transformer-based, slow.
   **Decision for prototype: load from precomputed cache.** Use the
   existing `dmc-data/mert_cache/<dataset>/<session>.npz` files that
   `scripts/precompute_mert_cache.py` generates. Means the prototype
   only works on sessions that have MERT cached — fine for V1–V4. Live
   MERT computation is V5 (run `MertEmbedder` once per new stem at
   session load — adds ~3 s of latency on session open but no runtime
   cost during playback).

2. **Inference cadence > buffer size.** 250 ms cadence (default).
   Sliding-window snapshot of the last 8 s of audio. NOT every audio
   buffer.

3. **Pre-allocation in the audio thread.** All scratch buffers
   (per-track strip output, mix output, master bus output) sized at
   `block_size` and allocated once. Never resize.

4. **Numba `@njit` for the compressor envelope follower.** Pure-Python
   sample-by-sample loops are ~50× too slow; numba lifts to native.
   EQ uses `scipy.signal.sosfilt` (already C-backed).

5. **CPU inference is fine for prototype.** The encoder is 27.7 M
   params; ONNX-free PyTorch on CPU should land at 30–80 ms per
   predict. GPU optional via `--device cuda`.

6. **Parameter smoothing.** Without it, clicks every 250 ms. With it,
   smooth transitions. ~100 ms time constant. Cheap.

7. **Quantization, ONNX, TensorRT, etc.** Deferred to the C++ port.
   Not needed for the Python prototype to feel realtime.

## Open decisions before V1

1. **Which checkpoint?** Plan: CLI flag `--checkpoint` pointing at any
   `mix_encoder_*.pt`. Wire the flag through; default to the freshest
   `best.pt` available. (Round 11_2 best.pt is current best per
   audition feedback; round 12 may produce a new best soon.)
2. **Stems that aren't in MERT cache?** V1–V4: error out at session
   load with a clear message ("session not in MERT cache; rerun
   `scripts/precompute_mert_cache.py` first"). V5: live MERT.
3. **Number of tracks.** Encoder trained at `n_max=42`. Plan: pad the
   stems tensor to 42, mask out the unused slots. Same as training.
4. **Mono → stereo broadcasting.** Already handled in `session.py`.
   Stays the same.
5. **Block size in the audio thread.** 512 samples by default
   (10.7 ms at 48 kHz). Configurable via CLI. The model doesn't care
   — only DSP latency does.
6. **Loop behavior.** V0 loops. Plan: keep looping default; provide
   `--no-loop`. When looping, the ring buffer carries audio across
   the loop boundary (i.e., the model sees a continuous "8 s preceding
   the playhead" even when the playhead wraps).

## Out of scope (for now)

- **VST3/AU/CLAP plugin packaging.** Once the Python prototype is
  proven, port hot paths to C++ and wrap with [Anira](https://github.com/anira-project/anira)
  + [CLAP](https://github.com/free-audio/clap). See the design doc
  in chat history.
- **Live recording / overdubbing.** The prototype plays back
  pre-loaded stems only.
- **MIDI control / automation.** No MIDI; params come from the model
  only (with optional manual override sliders in V4).
- **DAW integration.** Standalone first; plugin later.
- **Multi-output routing.** Stereo master out only.

## Status

- V0: shipped (`7f4d688`).
- V1–V4: not started. Plan above is the blueprint.

When resuming, the next concrete step is V1 — `ring_buffer.py`,
`inference.py`, `scheduler.py`, and wiring them into `engine.py`'s
audio callback (write side) + the CLI (start the scheduler thread,
print predictions). Estimate: ~2 hours of focused work + smoke
testing.
