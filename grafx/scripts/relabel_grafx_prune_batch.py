"""Batch driver for re-labeling all staged sessions with the new recipe + schema.

Walks `--staging-dir`, and for each session with `correspondence.yaml`:
  - If `labels_grafx_prune.pt` exists AND already has the new schema
    (per-processor `drywet_logit`), skip — already done.
  - Else, back up the old label (rename to `labels_grafx_prune.pt.old_recipe`)
    if present, then invoke `scripts/label_grafx_prune.py` in a subprocess
    with the new recipe defaults (12 × 500). Subprocessing isolates each
    session's GPU state — one OOM or NaN doesn't kill the whole batch.

Run:
    uv run python scripts/relabel_grafx_prune_batch.py \\
        --staging-dir dmc-data/grafx-prune-data \\
        --manifest dmc-data/grafx-prune-data/relabel_manifest.jsonl

Manifest line per session (jsonl):
    {"session": str, "started": iso, "finished": iso, "elapsed_seconds": float,
     "final_test_loss": float | None, "skipped_reason": str | None,
     "error": str | None}

Restartable: re-running picks up where it left off (skips sessions already
labeled at the new schema). Manifest is APPENDED to (not rewritten), so the
record of "what happened, when" survives interruption.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent


def _has_new_schema(label_path: Path) -> bool:
    """True if the saved label has per-processor `drywet_logit`.

    Cheap structural check — opens the file once and looks at one nested
    key. Fast enough to do on the full corpus at startup (<1s/session).
    """
    if not label_path.is_file():
        return False
    try:
        L = torch.load(label_path, weights_only=False, map_location="cpu")
    except Exception as e:
        logging.warning(f"  could not read {label_path}: {e}")
        return False
    strip = L.get("strip_params", {})
    # Pick any one processor and look for drywet_logit. If even one is
    # missing we consider the file old-schema and queue for relabeling.
    for proc, pp in strip.items():
        if "drywet_logit" not in pp:
            return False
    # Empty strip_params means corrupt; treat as needs-relabel.
    return len(strip) > 0


def _list_staged_sessions(staging_dir: Path) -> list[str]:
    out: list[str] = []
    for d in sorted(staging_dir.iterdir()):
        if not d.is_dir():
            continue
        if (d / "correspondence.yaml").is_file() and (d / "mix.wav").is_file() \
           and (d / "stems").is_dir():
            out.append(d.name)
    return out


def _label_one_subprocess(
    session: str,
    staging_dir: Path,
    total_epochs: int,
    steps_per_epoch: int,
    timeout_seconds: Optional[float] = None,
) -> tuple[bool, str | None, float | None]:
    """Run the per-session labeler as a subprocess.

    Returns (ok, error_message, final_test_loss_if_known). Stdout/stderr
    of the subprocess streams to our log directory; we don't capture it
    into memory (it can run hundreds of MB across 6 days).
    """
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/label_grafx_prune.py"),
        "--song", session,
        "--staging-dir", str(staging_dir),
        "--total-epochs", str(total_epochs),
        "--steps-per-epoch", str(steps_per_epoch),
    ]
    try:
        res = subprocess.run(
            cmd, cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout_seconds, text=True,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout", None
    if res.returncode != 0:
        tail = (res.stdout or "")[-2000:]
        return False, f"subprocess rc={res.returncode}; tail:\n{tail}", None
    # Extract final_test_loss from stdout if printed by the labeler.
    loss = None
    for line in (res.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("final test loss:"):
            try:
                loss = float(line.split(":", 1)[1].strip())
            except Exception:
                pass
    return True, None, loss


def _append_manifest(manifest_path: Path, entry: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--staging-dir", default="dmc-data/grafx-prune-data")
    ap.add_argument("--manifest", default=None,
                    help="Default <staging-dir>/relabel_manifest.jsonl")
    ap.add_argument("--total-epochs", type=int, default=12)
    ap.add_argument("--steps-per-epoch", type=int, default=500)
    ap.add_argument("--per-session-timeout", type=float, default=3600.0,
                    help="Wallclock cap per session (seconds). 1 hr is "
                         "~2x the projected median for 12x500 — anything "
                         "above is almost certainly hung. Default 3600.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N sessions this run. Useful "
                         "for short overnight chunks or for trying a "
                         "few sessions before committing the whole corpus.")
    ap.add_argument("--shuffle", action="store_true",
                    help="Process sessions in random order (so interrupted "
                         "runs cover a representative sample first). "
                         "Defaults to alphabetical for reproducibility.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for --shuffle. Ignored if --shuffle is off.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    staging_dir = Path(args.staging_dir).expanduser()
    manifest_path = Path(args.manifest) if args.manifest else \
                    (staging_dir / "relabel_manifest.jsonl")

    sessions = _list_staged_sessions(staging_dir)
    if not sessions:
        raise SystemExit(f"no staged sessions found under {staging_dir}")

    if args.shuffle:
        import random
        random.Random(args.seed).shuffle(sessions)

    logging.info(f"found {len(sessions)} staged sessions")
    logging.info(f"recipe: total_epochs={args.total_epochs} "
                 f"steps_per_epoch={args.steps_per_epoch}")
    logging.info(f"manifest: {manifest_path}")

    # First pass: count what needs doing.
    todo = []
    skip_already_new = 0
    for s in sessions:
        label = staging_dir / s / "labels_grafx_prune.pt"
        if _has_new_schema(label):
            skip_already_new += 1
            continue
        todo.append(s)
    logging.info(f"skip (already new-schema): {skip_already_new}")
    logging.info(f"to process: {len(todo)}")

    if args.limit is not None:
        todo = todo[: args.limit]
        logging.info(f"--limit {args.limit} → trimmed todo to {len(todo)}")

    t_start = time.time()
    n_ok = 0
    n_err = 0
    for i, session in enumerate(todo, start=1):
        label = staging_dir / session / "labels_grafx_prune.pt"
        backup = staging_dir / session / "labels_grafx_prune.pt.old_recipe"

        # Back up old-schema label before overwriting.
        if label.is_file() and not backup.is_file():
            shutil.move(str(label), str(backup))
            logging.info(f"  backed up old label → {backup.name}")

        elapsed_total = time.time() - t_start
        rate = elapsed_total / max(1, i - 1) if i > 1 else float("nan")
        eta = rate * (len(todo) - i + 1) if i > 1 else float("nan")
        logging.info(
            f"[{i}/{len(todo)}] {session}  "
            f"(elapsed {elapsed_total/3600:.2f}h  ETA {eta/3600:.2f}h)"
        )

        t0 = time.time()
        ok, err, loss = _label_one_subprocess(
            session, staging_dir, args.total_epochs, args.steps_per_epoch,
            timeout_seconds=args.per_session_timeout,
        )
        dt = time.time() - t0
        entry = {
            "session": session,
            "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S",
                                          time.localtime(t0)),
            "elapsed_seconds": dt,
            "final_test_loss": loss,
            "ok": ok,
            "error": err,
        }
        _append_manifest(manifest_path, entry)

        if ok:
            n_ok += 1
            logging.info(
                f"  ok  ({dt:.0f}s, final_test_loss="
                f"{loss if loss is not None else 'unknown'})"
            )
        else:
            n_err += 1
            logging.error(f"  FAIL  ({dt:.0f}s): {err[:200] if err else ''}")

    logging.info(f"\ndone. ok={n_ok} err={n_err}  total_elapsed="
                 f"{(time.time() - t_start)/3600:.2f}h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
