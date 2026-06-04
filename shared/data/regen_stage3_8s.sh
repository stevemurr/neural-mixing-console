#!/usr/bin/env bash
# Regenerate Cambridge + Slakh stage-3 shards at 8-second segments.
#
# Drops segment length from 15s/6s -> 8s for both datasets so we can fit a
# larger N_max without blowing past the unified-memory ceiling on DGX Spark.
# MERT cache (`dmc-data/mert_cache/`) is keyed by source stem filename, NOT
# by segment, so it does NOT need to be regenerated.
#
# Outputs to `*_8s` directories so the original 15s/6s shards remain intact
# and can be diffed against if anything goes sideways.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-.venv/bin/python}"
SEG=8.0

CAMBRIDGE_OUT="dmc-data/stage3_cambridge_8s"
SLAKH_OUT="dmc-data/stage3_slakh_8s"

echo "[regen] Cambridge -> ${CAMBRIDGE_OUT}  (segment=${SEG}s, stride=${SEG}s)"
"$PY" shared/data/prepare_stage3_cambridge.py \
    --output-dir "$CAMBRIDGE_OUT" \
    --segment-seconds "$SEG" \
    --segment-stride-seconds "$SEG"

echo "[regen] Slakh -> ${SLAKH_OUT}  (segment=${SEG}s, stride=${SEG}s)"
"$PY" shared/data/prepare_stage3_slakh.py \
    --output-dir "$SLAKH_OUT" \
    --segment-seconds "$SEG" \
    --segment-stride-seconds "$SEG"

echo "[regen] done. New shards under:"
echo "  $CAMBRIDGE_OUT/shards/"
echo "  $SLAKH_OUT/shards/"
echo
echo "Train with:"
echo "  scripts/train_stage3_v6_round3.sh"
