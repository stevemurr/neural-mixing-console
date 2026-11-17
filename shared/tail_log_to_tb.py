"""Tail a training log file and emit TensorBoard events for the scalars
embedded in the tqdm postfix. Non-invasive — does not touch the running
training process; just reads the log file as it grows.

Supported log formats:
  stage2 / stage2.5: `[..., L_total=0.123, L_param=0.04, L_bypass=0.5, L_recon=0.7]`
  stage2.7:          `[..., tot=3.5, trk=0.05, snd=0.04, glb=3.1]`

Step number is extracted from the leading `<step>/<total>` token.

Usage:
    python scripts/tail_log_to_tb.py \
        --log /tmp/train_stage2_7.log --tag stage2_7 --logdir runs/

    tensorboard --logdir runs/ --bind_all --port 6006
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


# tqdm uses \r to overwrite within a line; the file as a whole has many
# \r-separated frames concatenated. Split on \r and parse each frame.
_STEP_RE = re.compile(r"(\d+)/\d+")
_KV_RE = re.compile(r"([A-Za-z_]+)=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def parse_frame(frame: str) -> tuple[int, dict[str, float]] | None:
    if "it/s" not in frame and "s/it" not in frame:
        return None
    m = _STEP_RE.search(frame)
    if not m:
        return None
    step = int(m.group(1))
    pairs = _KV_RE.findall(frame)
    metrics = {k: float(v) for k, v in pairs}
    metrics.pop("it", None)
    if not metrics:
        return None
    return step, metrics


def tail_to_tb(log_path: Path, writer: SummaryWriter, poll_seconds: float = 2.0) -> None:
    """Tail a log file, handling truncation (e.g. when a chained `>` redirect
    rewrites the file)."""
    import os
    last_step = -1
    f = open(log_path, "rb")
    pos = 0
    buf = b""
    while True:
        try:
            size = os.path.getsize(log_path)
        except OSError:
            time.sleep(poll_seconds)
            continue

        if size < pos:
            # File shrank → truncated. Reopen and reset state.
            f.close()
            f = open(log_path, "rb")
            pos = 0
            buf = b""
            last_step = -1

        f.seek(pos)
        chunk = f.read()
        if chunk:
            pos = f.tell()
            buf += chunk
            text = buf.decode("utf-8", errors="ignore")
            frames = text.split("\r")
            buf = frames[-1].encode("utf-8")
            for frame in frames[:-1]:
                parsed = parse_frame(frame)
                if not parsed:
                    continue
                step, metrics = parsed
                if step <= last_step:
                    continue
                for k, v in metrics.items():
                    writer.add_scalar(k, v, step)
                last_step = step
            writer.flush()
        time.sleep(poll_seconds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path, help="path to training log file")
    ap.add_argument("--tag", required=True, help="run name under logdir")
    ap.add_argument("--logdir", default="runs", type=Path)
    ap.add_argument("--poll", type=float, default=2.0)
    args = ap.parse_args()

    args.logdir.mkdir(parents=True, exist_ok=True)
    run_dir = args.logdir / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"tailing {args.log} -> tensorboard run '{args.tag}' at {run_dir}")
    writer = SummaryWriter(log_dir=str(run_dir))

    while not args.log.exists():
        print(f"waiting for {args.log} to appear...")
        time.sleep(args.poll)

    try:
        tail_to_tb(args.log, writer, poll_seconds=args.poll)
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
