# Neural mixing console

The active experiment studies how to distribute frequency corrections across stems using their measured contributions in 26 mel bands. The target-curve predictor and EQ renderer can be evaluated separately.

## Reproduce the historical comparison

From the repository root, install the locked environment with `uv sync --locked`. The original cache and checkpoint paths and SHA-256 fingerprints are in the recipe. Audio and model artifacts are stored locally.

    uv run python -m nmc.evaluation.reproduce experiments/eq/reproduce-v13.toml --output /tmp/v13-result.json
    uv run python -m unittest discover -s tests

On the frozen split of 247 training songs and 32 validation songs, the zero, training-mean and learned curves score 4.309, 3.855 and 3.846 dB RMSE respectively.

The supported code lives in `src/nmc/data`, `src/nmc/eq`, and `src/nmc/evaluation`. The [research index](research/README.md) describes the historical approaches.
