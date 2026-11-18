# Target-curve EQ

## Prepare and train

From the repository root, use the staged Cambridge sessions to build the training cache and fit the curve predictor with the training recipe.

    uv run python -m nmc.eq.precompute --staging-dir /path/to/staged-sessions
    uv run python shared/run.py experiments/eq/train-target-curve.toml

Each session directory contains `stems/`, `mix.wav` and `correspondence.yaml`. Preparation writes `dmc-data/eq/target-curve-examples.pt`. The trainer keeps all windows from a session together and saves the best checkpoint to `dmc-data/checkpoints/target-curve/target_curve_best.pt`. Use `--checkpoint` with `nmc mix` to audition it. For raw MedleyDB evaluation, use the raw benchmark in the [project guide](../../README.md#benchmark-on-raw-medleydb-tracks).

## Replay the frozen comparison

    uv run nmc run experiments/eq/reproduce-target-curve.toml --check
    uv run nmc run experiments/eq/reproduce-target-curve.toml

The recipe pins the original cache and checkpoint by SHA-256 and checks the historical song split. The run's `metrics.json` records per-window squared errors and per-song and overall RMSE in dB for the zero, training-mean and checkpoint curves.
