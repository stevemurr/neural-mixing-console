"""Stage 3 v6 MixEncoder inference: tracks → predicted mix + params.

Usage:
    python infer.py \
        --checkpoint dmc-data/checkpoints/stage3_v6/mix_encoder_latest.pt \
        --tracks-dir /path/to/multitrack/folder \
        --out-dir /path/to/output \
        [--mert-model m-a-p/MERT-v1-95M] [--ref-mix /path/to/engineer_mix.wav]

The folder is expected to contain one or more WAV/FLAC files, one per stem.
Mono stems are broadcast to stereo. Tracks longer than --encoder-window-seconds
are processed via a representative window for parameter prediction; the rendered
output covers the full duration via chunked rendering.

Outputs (under --out-dir):
    rendered_mix.wav   — predicted final mix (stereo, 48 kHz, fp32 WAV)
    sum_baseline.wav   — simple mask-aware sum of tracks (sanity check)
    ref_mix.wav        — copy of --ref-mix if provided, for A/B
    params.json        — per-track and global predicted params (denormalized)
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T_audio

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.diff_mixing_graph import DiffMixingGraph
from models.diff_strip import compute_track_rms_db
from models.encoders import MixEncoder
from models.mert_embed import MertEmbedder
from reference.param_norm import bus_effective_bypass, strip_effective_bypass
from training.data import BUS_PARAM_KEYS, STRIP_PARAM_KEYS
from training.train_stage3 import (
    MERT_DIM, N_MAX, SAMPLE_RATE, _build_denorm_table, _denorm, _render_full_mix,
)


def _reclaim(device: torch.device) -> None:
    """gc.collect() + torch.cuda.empty_cache() (no-op on CPU)."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _load_audio_48k_stereo(
    path: Path,
    device: torch.device,
    resampler_cache: dict[int, T_audio.Resample] | None = None,
) -> np.ndarray:
    """Load any audio, resample to 48 kHz on `device`, return (2, T) float32."""
    data, src_fs = sf.read(str(path), dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    src_fs = int(src_fs)
    if src_fs != SAMPLE_RATE:
        if resampler_cache is None:
            resampler_cache = {}
        rs = resampler_cache.get(src_fs)
        if rs is None:
            rs = T_audio.Resample(
                orig_freq=src_fs, new_freq=SAMPLE_RATE, dtype=torch.float32
            ).to(device)
            resampler_cache[src_fs] = rs
        x = torch.from_numpy(data.T).to(device)
        with torch.no_grad():
            x = rs(x)
        return x.cpu().numpy().astype(np.float32, copy=False)
    return data.T.astype(np.float32, copy=False)


def _gather_stems(tracks_dir: Path) -> list[Path]:
    """Find audio files in tracks_dir (excluding obvious mix files)."""
    exts = (".wav", ".flac", ".aif", ".aiff", ".mp3")
    return sorted(p for p in tracks_dir.iterdir()
                  if p.suffix.lower() in exts
                  and "mixture" not in p.name.lower()
                  and "preview" not in p.name.lower()
                  and not p.name.lower().endswith("_mix.wav"))


@torch.no_grad()
def _compute_mert_for_tracks(audio_per_track: list[np.ndarray], device: torch.device,
                             model_name: str = "m-a-p/MERT-v1-95M") -> torch.Tensor:
    """audio_per_track: list of (2, T) at 48 kHz. Returns (N, 768) fp32.

    Releases the MertEmbedder (~95M params, ~380 MB fp32) before returning so
    the encoder + render path don't compete with it for unified memory.
    """
    print(f"loading MERT model {model_name}...")
    embedder = MertEmbedder(model_name=model_name).to(device).eval()
    print(f"  embed_dim = {embedder.embed_dim}")

    embeddings = []
    for audio in audio_per_track:
        mono = audio.mean(axis=0)
        x = torch.from_numpy(mono).unsqueeze(0).to(device)
        emb = embedder(x).squeeze(0)
        embeddings.append(emb.cpu().float())
    out = torch.stack(embeddings, dim=0)
    del embedder
    _reclaim(device)
    return out


def _safety_limit(x: torch.Tensor, ceiling_db: float = -0.3) -> torch.Tensor:
    """Differentiable soft-clip at ceiling_db. tanh-based: unit gain at small x,
    saturates smoothly at ±ceiling. Applied once at the very end of inference
    (after trim) to guarantee output peak ≤ ceiling.

    Set ceiling_db to a high positive value (e.g. +60) to effectively disable.
    """
    if ceiling_db >= 60.0:
        return x
    ceiling = 10.0 ** (ceiling_db / 20.0)
    return ceiling * torch.tanh(x / ceiling)


def _build_static_render_args(
    out: dict,
    full_tracks: torch.Tensor,
    tables: dict,
    use_rms_relative_threshold: bool,
) -> tuple[dict, dict]:
    """Build the (strip_params, bus_params) tuple that DiffMixingGraph.forward
    needs, with one correctness-critical twist: when
    `use_rms_relative_threshold` is on, the per-track RMS used for the comp
    threshold offset is computed once over the FULL song length.
    `_render_full_mix` would recompute it from whatever slice of `tracks` it
    was given, which would make the comp threshold jump at every chunk
    boundary if the renderer is called per-chunk.

    v6.1: no bypass — "bypassed" is expressed in the params themselves.
    """
    B, N_max = full_tracks.shape[:2]

    strip_phys = _denorm(out["track_params"].float(), tables["strip"])
    strip_params = {k: strip_phys[..., i] for i, k in enumerate(STRIP_PARAM_KEYS)}

    if use_rms_relative_threshold:
        flat = full_tracks.reshape(B * N_max, 2, full_tracks.shape[-1]).float()
        track_rms = compute_track_rms_db(flat).view(B, N_max)
        threshold_norm_idx = STRIP_PARAM_KEYS.index("threshold_db")
        threshold_offset_norm = out["track_params"][..., threshold_norm_idx].float().clamp(0, 1)
        offset = -30.0 + threshold_offset_norm * 36.0
        strip_params.pop("threshold_db", None)
        strip_params["threshold_offset_db"] = offset
        strip_params["track_rms_db"] = track_rms

    bus_phys = _denorm(out["bus_params"].float(), tables["bus"])
    bus_params: dict = {}
    for i, k in enumerate(BUS_PARAM_KEYS):
        v = bus_phys[..., i]
        bus_params[k[len("bus_"):] if k.startswith("bus_") else k] = v

    return (strip_params, bus_params)


def _render_chunked(
    mix_graph: DiffMixingGraph,
    full_tracks: torch.Tensor,            # (1, N_max, 2, T_full)
    full_mask: torch.Tensor,              # (1, N_max)
    static_args: tuple,                   # _build_static_render_args output
    chunk_seconds: float,
    overlap_seconds: float,
    crossfade_seconds: float,
    sample_rate: int,
) -> torch.Tensor:
    """Render the full mix in time chunks, bounding peak memory.

    v6 has no reverb (and no delay), so the memory pressure is much lower
    than the v5 days — chunking is now about predictability rather than
    necessity, and the overlap budget can drop dramatically (compressor
    envelope only, ~1 s).

    Each chunk is rendered with `overlap` seconds of pre-roll, the leading
    `overlap - crossfade` samples are discarded (their compressor envelope
    hasn't settled), and the seam is crossfaded against the previous chunk's
    tail. The first chunk has no pre-roll.

    Returns the predicted mix as (1, 2, T_full) on `full_tracks.device`.
    """
    device = full_tracks.device
    T_full = full_tracks.shape[-1]
    chunk_T = max(1, int(round(chunk_seconds * sample_rate)))
    overlap_T = max(0, int(round(overlap_seconds * sample_rate)))
    cf_T = max(0, int(round(crossfade_seconds * sample_rate)))
    cf_T = min(cf_T, overlap_T)

    out_mix = torch.zeros((1, 2, T_full), dtype=torch.float32, device=device)
    fade_in = (
        torch.linspace(0.0, 1.0, cf_T, device=device, dtype=torch.float32).view(1, 1, cf_T)
        if cf_T > 0 else None
    )

    cursor = 0
    while cursor < T_full:
        is_first = cursor == 0
        pre = 0 if is_first else overlap_T
        start = cursor - pre
        end = min(cursor + chunk_T, T_full)
        slice_tracks = full_tracks[..., start:end]

        chunk = mix_graph(
            slice_tracks.float(), full_mask,
            static_args[0],            # strip_params
            bus_params=static_args[1],
        )

        if is_first:
            out_mix[..., 0:end] = chunk[..., :end]
        else:
            # Body of this chunk in song coordinates: [cursor - cf_T, end].
            # The first cf_T samples of the body crossfade against `out_mix`'s
            # already-written prior content (which was the previous chunk's tail).
            body = chunk[..., pre - cf_T:]
            place = cursor - cf_T
            if cf_T > 0:
                prior = out_mix[..., place:place + cf_T]
                out_mix[..., place:place + cf_T] = prior * (1.0 - fade_in) + body[..., :cf_T] * fade_in
                tail = body[..., cf_T:]
                out_mix[..., cursor:cursor + tail.shape[-1]] = tail
            else:
                out_mix[..., cursor:cursor + body.shape[-1]] = body

        del chunk
        _reclaim(device)
        cursor = end

    return out_mix


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="MixEncoder .pt checkpoint")
    ap.add_argument("--tracks-dir", required=True, help="folder containing per-track audio files")
    ap.add_argument("--out-dir", default="dmc-data/inference",
                    help="where to write rendered_mix.wav, params.json, etc.")
    ap.add_argument("--ref-mix", default="",
                    help="optional reference mix path for A/B comparison (copied to out-dir)")
    ap.add_argument("--mert-model", default="m-a-p/MERT-v1-95M")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-max", type=int, default=N_MAX,
                    help="max tracks the model was trained on (extra are ignored)")
    ap.add_argument("--use-rms-relative-threshold", type=int, default=1,
                    help="must match training (1 = on, 0 = off)")
    ap.add_argument("--mert-dim", type=int, default=MERT_DIM,
                    help="MERT embedding dim; 0 disables MERT (model w/o MERT conditioning)")
    ap.add_argument("--encoder-window-seconds", type=float, default=15.0,
                    help="length of the audio window passed to the encoder for param "
                         "prediction. Must match the segment length the model was "
                         "trained on. Render path uses full audio length regardless.")
    ap.add_argument("--render-chunk-seconds", type=float, default=30.0,
                    help="time-axis chunk size for the render. Set <=0 to disable chunking.")
    ap.add_argument("--render-overlap-seconds", type=float, default=2.0,
                    help="pre-roll fed into each non-first chunk so the compressor envelope "
                         "has settled before the kept output starts. v6 has no reverb/delay "
                         "so 1-2 s is sufficient.")
    ap.add_argument("--render-crossfade-seconds", type=float, default=0.025,
                    help="cosmetic crossfade across the chunk seam to absorb any tiny "
                         "numerical discontinuity. Set to 0 to disable.")
    ap.add_argument("--safety-limiter-db", type=float, default=-0.3,
                    help="ceiling for the always-on safety limiter applied at the very "
                         "end of inference (after trim). Pass +60 to disable. The limiter "
                         "is a tanh soft-clip — peaks above the ceiling round off smoothly.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ---- 1. Load audio ----
    tracks_dir = Path(args.tracks_dir)
    if not tracks_dir.is_dir():
        raise SystemExit(f"--tracks-dir does not exist: {tracks_dir}")
    stem_paths = _gather_stems(tracks_dir)
    if not stem_paths:
        # Allow nested layout (e.g. cambridge-mt has session/<stems>)
        for sub in tracks_dir.iterdir():
            if sub.is_dir():
                stem_paths = _gather_stems(sub)
                if stem_paths:
                    print(f"using nested folder: {sub.name}")
                    break
    if not stem_paths:
        raise SystemExit(f"no audio files in {tracks_dir}")
    print(f"found {len(stem_paths)} stems in {tracks_dir}")

    if len(stem_paths) > args.n_max:
        print(f"WARNING: {len(stem_paths)} stems > n_max={args.n_max}; "
              f"truncating to first {args.n_max}")
        stem_paths = stem_paths[:args.n_max]

    print("loading audio...")
    resampler_cache: dict[int, T_audio.Resample] = {}
    track_audio: list[np.ndarray] = []
    for p in stem_paths:
        try:
            audio = _load_audio_48k_stereo(p, device, resampler_cache)
        except Exception as e:
            print(f"  {p.name}: load failed ({e}); skipping")
            continue
        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
        if rms < 1e-5:
            print(f"  {p.name}: silent; skipping")
            continue
        track_audio.append(audio)
        print(f"  {p.name}: {audio.shape[-1]/SAMPLE_RATE:.1f}s, rms={20*np.log10(rms+1e-9):+.1f} dB")
    if not track_audio:
        raise SystemExit("no valid (non-silent) tracks loaded")

    # Pad/trim all tracks to common length (= max).
    max_T = max(a.shape[-1] for a in track_audio)
    for i in range(len(track_audio)):
        if track_audio[i].shape[-1] < max_T:
            pad = np.zeros((2, max_T - track_audio[i].shape[-1]), dtype=np.float32)
            track_audio[i] = np.concatenate([track_audio[i], pad], axis=-1)

    # Pick a representative window for encoder input. The MixEncoder's mel
    # transformer has a fixed positional encoding sized to the training
    # segment length; longer audio overflows. Predicted params are
    # supposed to be roughly constant per track over time, so picking the
    # loudest window is a reasonable proxy. We render the full length
    # downstream (DiffMixingGraph supports arbitrary T).
    SEG_LEN = int(SAMPLE_RATE * args.encoder_window_seconds)
    if max_T > SEG_LEN:
        sum_energy = np.zeros(max_T, dtype=np.float32)
        for a in track_audio:
            sum_energy += np.mean(a ** 2, axis=0)
        window_rms = np.sqrt(np.convolve(sum_energy, np.ones(SEG_LEN) / SEG_LEN, mode="valid") + 1e-12)
        peak_start = int(np.argmax(window_rms))
        encoder_audio = [a[:, peak_start : peak_start + SEG_LEN] for a in track_audio]
        print(f"  encoder window: [{peak_start/SAMPLE_RATE:.1f}s, {(peak_start+SEG_LEN)/SAMPLE_RATE:.1f}s]")
    elif max_T < SEG_LEN:
        encoder_audio = []
        for a in track_audio:
            pad = np.zeros((2, SEG_LEN - a.shape[-1]), dtype=np.float32)
            encoder_audio.append(np.concatenate([a, pad], axis=-1))
    else:
        encoder_audio = list(track_audio)

    # ---- 2. Build encoder + load checkpoint ----
    print(f"\nbuilding MixEncoder (mert_dim={args.mert_dim})")
    encoder = MixEncoder(
        sample_rate=SAMPLE_RATE,
        use_ref_mix=False,
        mert_dim=args.mert_dim,
    ).to(device).eval()
    print(f"  loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "encoder_state_dict" in ckpt:
        ckpt = ckpt["encoder_state_dict"]
    missing, unexpected = encoder.load_state_dict(ckpt, strict=False)
    print(f"  loaded ({len(missing)} missing, {len(unexpected)} unexpected)")

    mix_graph = DiffMixingGraph(sample_rate=SAMPLE_RATE).to(device).eval()
    for p in mix_graph.parameters():
        p.requires_grad_(False)

    tables = {
        "strip": _build_denorm_table(device, STRIP_PARAM_KEYS),
        "bus":   _build_denorm_table(device, BUS_PARAM_KEYS),
    }

    # ---- 3. MERT embeddings ----
    if args.mert_dim > 0:
        mert = _compute_mert_for_tracks(track_audio, device, args.mert_model)
        print(f"  MERT shape: {mert.shape}")
    else:
        mert = torch.zeros(len(track_audio), 1, dtype=torch.float32)
        print("  MERT disabled")

    # ---- 4. Build batches + run encoder on the chosen window ----
    n = len(track_audio)
    n_max = args.n_max

    enc_T = encoder_audio[0].shape[-1]
    enc_tracks = torch.zeros((1, n_max, 2, enc_T), dtype=torch.float32)
    enc_mask = torch.zeros((1, n_max), dtype=torch.bool)
    mert_padded = torch.zeros((1, n_max, max(args.mert_dim, 1)), dtype=torch.float32)
    for i in range(n):
        enc_tracks[0, i] = torch.from_numpy(encoder_audio[i])
        enc_mask[0, i] = True
        if args.mert_dim > 0:
            mert_padded[0, i] = mert[i]

    # Render batch (full-length tracks).
    full_tracks = torch.zeros((1, n_max, 2, max_T), dtype=torch.float32)
    full_mask = torch.zeros((1, n_max), dtype=torch.bool)
    for i in range(n):
        full_tracks[0, i] = torch.from_numpy(track_audio[i])
        full_mask[0, i] = True

    enc_tracks = enc_tracks.to(device)
    enc_mask = enc_mask.to(device)
    mert_padded = mert_padded.to(device)
    full_tracks = full_tracks.to(device)
    full_mask = full_mask.to(device)

    print(f"\nrunning MixEncoder (n_tracks={n}, encoder window={enc_T/SAMPLE_RATE:.1f}s)...")
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = encoder(
                enc_tracks, enc_mask,
                ref_mix=None,
                mert_embeddings=mert_padded if args.mert_dim > 0 else None,
            )

    # Encoder + MERT + numpy stems are all unused from here on. On unified-
    # memory boxes every byte of cache competes with the render.
    encoder = None
    enc_tracks = None
    mert_padded = None
    track_audio = None
    encoder_audio = None
    _reclaim(device)

    # ---- 5. Render full-length mix (chunked along time) ----
    chunk_s = args.render_chunk_seconds
    overlap_s = args.render_overlap_seconds
    cf_s = args.render_crossfade_seconds
    chunk_T = int(round(chunk_s * SAMPLE_RATE)) if chunk_s > 0 else max_T
    needs_chunking = chunk_s > 0 and max_T > chunk_T

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=False):
            if needs_chunking:
                static_args = _build_static_render_args(
                    out, full_tracks, tables,
                    use_rms_relative_threshold=bool(args.use_rms_relative_threshold),
                )
                n_chunks = (max_T + chunk_T - 1) // chunk_T
                print(f"rendering predicted mix on full {max_T/SAMPLE_RATE:.1f}s audio "
                      f"in {n_chunks} chunks of {chunk_s:.1f}s "
                      f"(pre-roll {overlap_s:.1f}s, crossfade {cf_s*1000:.0f}ms)...")
                pred_mix = _render_chunked(
                    mix_graph, full_tracks, full_mask, static_args,
                    chunk_seconds=chunk_s,
                    overlap_seconds=overlap_s,
                    crossfade_seconds=cf_s,
                    sample_rate=SAMPLE_RATE,
                )
            else:
                print(f"rendering predicted mix on full {max_T/SAMPLE_RATE:.1f}s audio "
                      f"in one shot...")
                pred_mix = _render_full_mix(
                    mix_graph, full_tracks, full_mask, out, tables,
                    use_rms_relative_threshold=bool(args.use_rms_relative_threshold),
                )

            # Apply predicted trim_db, then the always-on safety limiter.
            trim_db_value = float(out["trim_db"][0].item())
            gain = 10.0 ** (out["trim_db"][0:1] / 20.0)
            pred_mix = pred_mix * gain.view(1, 1, 1)
            pred_mix = _safety_limit(pred_mix, ceiling_db=args.safety_limiter_db)

    pred_np = pred_mix[0].detach().cpu().numpy().T
    pred_mix = None
    _reclaim(device)

    # ---- 6. Sum baseline ----
    # Stream the sum over real tracks instead of (1, N_max, 2, T) * mask
    # broadcast that allocates a duplicate of the full track buffer.
    sum_t = torch.zeros((2, max_T), dtype=torch.float32, device=device)
    for i in range(n):
        sum_t.add_(full_tracks[0, i].float())
    sum_mix = sum_t.cpu().numpy().T
    sum_t = None

    # ---- 7. Save outputs ----
    out_pred = out_dir / "rendered_mix.wav"
    out_sum = out_dir / "sum_baseline.wav"
    sf.write(str(out_pred), pred_np, SAMPLE_RATE, subtype="FLOAT")
    sf.write(str(out_sum), sum_mix, SAMPLE_RATE, subtype="FLOAT")
    print(f"\nwrote {out_pred} ({pred_np.shape[0]/SAMPLE_RATE:.1f}s, "
          f"peak={np.abs(pred_np).max():.3f}, trim={trim_db_value:+.2f} dB, "
          f"safety_limiter={args.safety_limiter_db:+.1f} dBFS)")
    print(f"wrote {out_sum} (sum baseline)")

    if args.ref_mix:
        ref_path = Path(args.ref_mix)
        if ref_path.exists():
            ref_audio = _load_audio_48k_stereo(ref_path, device, resampler_cache).T
            sf.write(str(out_dir / "ref_mix.wav"), ref_audio, SAMPLE_RATE, subtype="FLOAT")
            print(f"wrote {out_dir/'ref_mix.wav'} (ref copy)")

    # ---- 8. Denormalize + save params ----
    strip_phys = _denorm(out["track_params"][0, :n].float().cpu(),
                         {k: v.cpu() for k, v in tables["strip"].items()})
    bus_phys = _denorm(out["bus_params"][0].float().cpu(),
                       {k: v.cpu() for k, v in tables["bus"].items()})

    track_strip_dicts = [
        {k: float(strip_phys[t, i]) for i, k in enumerate(STRIP_PARAM_KEYS)}
        for t in range(strip_phys.shape[0])
    ]
    bus_dict = {k: float(bus_phys[i]) for i, k in enumerate(BUS_PARAM_KEYS)}

    params_doc: dict = {
        "checkpoint": str(args.checkpoint),
        "n_tracks": n,
        "trim_db": trim_db_value,
        "safety_limiter_db": float(args.safety_limiter_db),
        "tracks": [],
        "bus_params": bus_dict,
        "bus_effective_bypass": bus_effective_bypass(bus_dict),
    }
    for i in range(n):
        params_doc["tracks"].append({
            "filename": stem_paths[i].name,
            "strip": track_strip_dicts[i],
            "strip_effective_bypass": strip_effective_bypass(track_strip_dicts[i]),
        })

    params_path = out_dir / "params.json"
    with open(params_path, "w") as f:
        json.dump(params_doc, f, indent=2)
    print(f"wrote {params_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
