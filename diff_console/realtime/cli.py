"""V0/V1 entry point.

V0 (no --checkpoint): load a stems folder, play it back, show playhead.

V1 (with --checkpoint): same as V0, plus
  - per-track ring buffers fed by the audio callback;
  - a background InferenceScheduler that snapshots the buffers every
    --cadence-ms and runs the encoder;
  - one-line param summary printed each tick (trim_db, mean gain/pan,
    extreme bus moves) so we can confirm the model-in-the-loop architecture
    works end-to-end before V2 wires the predictions into a DSP graph.

Usage:
    # V0 — passthrough sum, no model:
    uv run python -m realtime.cli --stems-dir source_audio/cambridge-mt/<S>/<S>

    # V1 — model in the loop, predictions printed (no DSP yet):
    uv run python -m realtime.cli \\
        --stems-dir source_audio/cambridge-mt/<S>/<S> \\
        --checkpoint dmc-data/checkpoints/v6.2-round11_2/mix_encoder_best.pt

Press Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np

from .engine import AudioEngine
from .ring_buffer import RingBuffer
from .session import Session

# sounddevice and torch imported lazily inside main(): top-level imports
# fail on headless machines without PortAudio (sounddevice) and pull in
# a lot of state we don't need on a dry-run (torch).


def _format_predict_summary(params: dict, strip_keys: tuple, bus_keys: tuple) -> str:
    """Compact one-liner for a predict() result. The interesting V1 signal is
    "did the model produce stable, plausible numbers?", so we summarize: trim,
    mean/range of gain_db, mean of pan, mean of bus_threshold, and a sentinel
    for any per-track at-range-edge corner pinning.
    """
    strip = params["strip_phys"]                   # (n_active, 22)
    bus = params["bus_phys"]                       # (13,)
    n = params["n_active"]
    trim_db = params["trim_db"]

    gi = strip_keys.index("gain_db")
    pi = strip_keys.index("pan")
    ri = strip_keys.index("ratio")
    bt = bus_keys.index("bus_threshold_db")
    br = bus_keys.index("bus_ratio")

    gains = strip[:, gi]
    pans = strip[:, pi]
    ratios = strip[:, ri]

    return (
        f"trim={trim_db:+5.2f}dB | "
        f"gain μ={gains.mean():+5.2f} [{gains.min():+5.2f}, {gains.max():+5.2f}] | "
        f"pan μ={pans.mean():+5.2f} | "
        f"ratio μ={ratios.mean():4.2f} | "
        f"bus thr={bus[bt]:+5.2f}dB ratio={bus[br]:4.2f} | "
        f"n={n}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Realtime mixing console (V0/V1): stems player + optional model-in-the-loop"
    )
    ap.add_argument("--stems-dir", help="Folder containing per-track WAV/FLAC stems")
    ap.add_argument("--block-size", type=int, default=512,
                    help="Audio callback block size in samples (default 512). "
                         "Smaller = lower latency, more callback overhead.")
    ap.add_argument("--master-gain-db", type=float, default=-6.0,
                    help="Master gain in dB on the summed stems (default -6, "
                         "lowers the risk of clipping with many tracks).")
    ap.add_argument("--max-tracks", type=int, default=None,
                    help="Truncate session to first N stems (alphabetical).")
    ap.add_argument("--device", default=None,
                    help="Output device: name substring or integer index "
                         "(see --list-devices).")
    ap.add_argument("--no-loop", action="store_true",
                    help="Stop at end of session instead of looping.")
    ap.add_argument("--seek", type=float, default=0.0,
                    help="Start playback at this position in seconds.")
    ap.add_argument("--list-devices", action="store_true",
                    help="List available audio devices and exit.")

    # ---- V1: model-in-the-loop options ----
    v1 = ap.add_argument_group("V1 (model in the loop)")
    v1.add_argument("--checkpoint", default=None,
                    help="MixEncoder .pt checkpoint. Enables V1 mode: ring buffers "
                         "+ inference scheduler. Predictions are printed only — "
                         "no DSP is applied yet (V2 will).")
    v1.add_argument("--encoder-window-seconds", type=float, default=8.0,
                    help="Length of the audio window passed to the encoder for "
                         "each predict() call. Must match the encoder's training "
                         "segment length (v6.2 round 11_2 trained on 8 s).")
    v1.add_argument("--cadence-ms", type=float, default=250.0,
                    help="Inference scheduler cadence in milliseconds.")
    v1.add_argument("--n-max", type=int, default=42,
                    help="Max tracks the encoder was trained on (extras get masked).")
    v1.add_argument("--mert-dim", type=int, default=768,
                    help="MERT embedding dim; 0 disables MERT.")
    v1.add_argument("--mert-cache-root", default="dmc-data/mert_cache",
                    help="Where to auto-detect <dataset>/<session>.npz MERT cache files.")
    v1.add_argument("--mert-cache-npz", default=None,
                    help="Explicit path to a single MERT .npz (overrides auto-detect).")
    v1.add_argument("--trim-max-db", type=float, default=18.0,
                    help="Must match the encoder's training trim_max_db.")
    v1.add_argument("--inference-device", default=None,
                    help="'cpu' or 'cuda'. Default: cuda if available, else cpu.")
    v1.add_argument("--print-cadence-ticks", type=int, default=1,
                    help="Print every Nth prediction (1 = every tick).")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    if args.stems_dir is None:
        ap.error("--stems-dir is required (unless --list-devices)")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    print(f"loading session from {args.stems_dir}")
    sess = Session(
        args.stems_dir,
        max_tracks=args.max_tracks,
        mert_cache_root=args.mert_cache_root if args.checkpoint else None,
        mert_cache_npz=args.mert_cache_npz,
        mert_dim=args.mert_dim if args.checkpoint else 0,
        require_mert=False,
    )
    print(f"  {sess}")
    print(f"  stems:")
    for p in sess.stem_paths:
        print(f"    {p.name}")

    ring_buffers: Optional[list[RingBuffer]] = None
    scheduler = None
    if args.checkpoint:
        # Heavy imports deferred to V1 path.
        import torch
        from .inference import EncoderWrapper
        from .scheduler import InferenceScheduler

        if args.inference_device is None:
            inference_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            inference_device = args.inference_device

        capacity = int(round(args.encoder_window_seconds * sess.sample_rate))
        ring_buffers = [
            RingBuffer(n_channels=2, capacity_samples=capacity)
            for _ in range(sess.n_tracks)
        ]
        print(
            f"\nbuilding ring buffers: {sess.n_tracks} x (2, {capacity}) = "
            f"{args.encoder_window_seconds:.1f} s window per track"
        )

        print(f"loading encoder from {args.checkpoint}")
        encoder = EncoderWrapper(
            args.checkpoint,
            device=inference_device,
            n_max=args.n_max,
            mert_dim=args.mert_dim,
            trim_max_db=args.trim_max_db,
            sample_rate=sess.sample_rate,
        )

        strip_keys = encoder.strip_keys
        bus_keys = encoder.bus_keys
        tick_counter = {"n": 0}
        print_lock = Lock()
        every_n = max(1, args.print_cadence_ticks)

        def on_new_params(params: dict) -> None:
            tick_counter["n"] += 1
            if tick_counter["n"] % every_n != 0:
                return
            line = _format_predict_summary(params, strip_keys, bus_keys)
            sched = scheduler  # captured via closure; set below
            timing = (
                f"  {sched.last_predict_ms:5.1f}ms (ema {sched.last_predict_ema_ms:5.1f})"
                if sched is not None else ""
            )
            with print_lock:
                # Newline so we don't interfere with the playhead line.
                sys.stdout.write("\n[predict] " + line + timing + "\n")
                sys.stdout.flush()

        scheduler = InferenceScheduler(
            encoder, ring_buffers,
            mert=sess.mert if args.mert_dim > 0 else None,
            on_new_params=on_new_params,
            cadence_ms=args.cadence_ms,
        )

    engine = AudioEngine(
        sess,
        block_size=args.block_size,
        master_gain_db=args.master_gain_db,
        output_device=args.device,
        loop=not args.no_loop,
        ring_buffers=ring_buffers,
    )
    if args.seek > 0:
        engine.seek(args.seek)

    print("\nstarting audio. Ctrl-C to stop.")
    engine.start()
    if scheduler is not None:
        scheduler.start()
        print(f"inference scheduler started (cadence={args.cadence_ms:.0f} ms, "
              f"device={scheduler.encoder.device})")
    try:
        while engine.is_running:
            if scheduler is not None:
                rb_min = min(rb.fill_fraction for rb in ring_buffers)  # type: ignore[arg-type]
                status = (
                    f"\r  playhead: {engine.playhead_s:6.2f} s / {sess.duration_s:6.2f} s  "
                    f"| rb: {rb_min*100:5.1f}%  "
                    f"| ticks: {scheduler.total_ticks} "
                    f"(cold {scheduler.cold_ticks}, late {scheduler.late_ticks}, "
                    f"fail {scheduler.failed_ticks})  "
                )
            else:
                status = (
                    f"\r  playhead: {engine.playhead_s:6.2f} s / {sess.duration_s:6.2f} s  "
                )
            print(status, end="", flush=True)
            time.sleep(0.1)
        print()  # newline after the carriage-return updates
    except KeyboardInterrupt:
        print()
    finally:
        if scheduler is not None:
            scheduler.stop()
        engine.stop()
    print("stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
