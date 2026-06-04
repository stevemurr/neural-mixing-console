"""Write a minimal labels_grafx_prune.pt per staged session for v12 training.

v12's train_minimal.py only reads `stem_filenames`, `groups`, and
`group_assignments` from the LabelStore — it does NOT need the teacher's
DSP params, soft_masks, or training_meta. So instead of running the
~6-day grafx-prune teacher, we derive those three fields directly from
correspondence.yaml (which is already copied into each staged session
by scripts/stage_cambridge_for_grafx_prune.py).

This unblocks v12.1 training on the 48 kHz re-staged corpus without
ever running the teacher.

Usage:
    uv run python scripts/write_minimal_labels.py --staging-dir dmc-data/grafx-prune-data-48k

Idempotent — skips sessions that already have a labels file unless
--force is passed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

# Reuse the teacher's correspondence-parsing helper so we stay consistent
# with how stems are enumerated and grouped.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.label_grafx_prune import load_correspondence_and_groups


def write_stub(session_dir: Path, force: bool) -> str:
    """Write a minimal labels file. Returns one of: 'wrote', 'skipped', 'failed'."""
    out = session_dir / "labels_grafx_prune.pt"
    if out.exists() and not force:
        return "skipped"
    try:
        stem_filenames, groups, group_assignments = load_correspondence_and_groups(session_dir)
    except Exception as e:
        logging.error(f"{session_dir.name}: {type(e).__name__}: {e}")
        return "failed"
    labels = {
        "stem_filenames": stem_filenames,
        "groups": groups,
        "group_assignments": torch.tensor(group_assignments, dtype=torch.long),
    }
    torch.save(labels, out)
    return "wrote"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--staging-dir", default="dmc-data/grafx-prune-data-48k",
                    help="Root dir containing <session>/correspondence.yaml")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing labels_grafx_prune.pt files.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    staging = Path(args.staging_dir)
    if not staging.is_dir():
        ap.error(f"staging dir not found: {staging}")

    sessions = sorted(p for p in staging.iterdir()
                      if p.is_dir() and (p / "correspondence.yaml").is_file())
    logging.info(f"found {len(sessions)} sessions with correspondence.yaml")

    counts = {"wrote": 0, "skipped": 0, "failed": 0}
    for s in sessions:
        counts[write_stub(s, args.force)] += 1

    logging.info(f"done: wrote={counts['wrote']}  "
                 f"skipped={counts['skipped']}  failed={counts['failed']}")
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
