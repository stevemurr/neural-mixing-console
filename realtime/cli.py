"""V0 entry point: load a stems folder, play it back, show playhead.

Usage:
    uv run python -m realtime.cli --stems-dir source_audio/cambridge-mt/<S>/<S>

Press Ctrl-C to stop. The audio callback runs on its own thread; this
process just opens the stream, polls a live playhead display, and stops
the engine cleanly on exit.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .engine import AudioEngine
from .session import Session

# sounddevice imported lazily inside main(): top-level import fails on
# headless machines without PortAudio. The CLI is also useful as a
# dry-run on such boxes (load + report a session, validate paths)
# without needing audio.


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Realtime mixing console (V0): stems passthrough player"
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
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    if args.stems_dir is None:
        ap.error("--stems-dir is required (unless --list-devices)")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    print(f"loading session from {args.stems_dir}")
    sess = Session(args.stems_dir, max_tracks=args.max_tracks)
    print(f"  {sess}")
    print(f"  stems:")
    for p in sess.stem_paths:
        print(f"    {p.name}")

    engine = AudioEngine(
        sess,
        block_size=args.block_size,
        master_gain_db=args.master_gain_db,
        output_device=args.device,
        loop=not args.no_loop,
    )
    if args.seek > 0:
        engine.seek(args.seek)

    print("\nstarting audio. Ctrl-C to stop.")
    engine.start()
    try:
        while engine.is_running:
            print(
                f"\r  playhead: {engine.playhead_s:6.2f} s / {sess.duration_s:6.2f} s  ",
                end="",
                flush=True,
            )
            time.sleep(0.1)
        print()  # newline after the carriage-return updates
    except KeyboardInterrupt:
        print()
    finally:
        engine.stop()
    print("stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
