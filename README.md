# Neural mixing console

The active research workflow studies how to distribute a requested frequency correction across stems using each stem’s measured contribution in 26 mel bands.

## Reproduce and track an experiment

From the repository root, install the project with `uv sync --locked`. For an existing research environment with locally installed audio tools, `uv pip install --no-deps --no-build-isolation -e .` installs the command using the environment’s existing dependencies and setuptools.

    uv run nmc run experiments/eq/reproduce-v13.toml --check
    uv run nmc run experiments/eq/reproduce-v13.toml
    uv run nmc runs list

The original local cache and checkpoint are required. Their paths and hashes are in the recipe. The check verifies both inputs, model compatibility, and the frozen split. On 32 validation songs and 640 windows, zero EQ, the training-mean curve and the learned curve score 4.309, 3.855 and 3.846 dB RMSE.

Every launch creates `runs/<recipe>/<run-id>/`. Open `run.json` for the command’s outcome, revision, working-tree state, input fingerprints and split fingerprint. The resolved settings are in `recipe.json`, the worker output in `run.log`, and the song IDs, window offsets and measurements in `metrics.json`. The reproduction recipe records the original checkpoint by path and hash. `--check` reads the inputs without allocating a run directory. Relative input paths are resolved from the current working directory.

## Prepare and inspect MedleyDB audio

    uv run nmc data prepare medleydb --root /path/to/MedleyDB
    uv run nmc run experiments/eq/medleydb-inspect.toml --check
    uv run nmc run experiments/eq/medleydb-inspect.toml
    uv run nmc runs list

Preparation inventories grouped stems by default. Pass `--inputs raw` to inventory raw tracks. Each inventory records the reference mix, selected files, rates, channel counts, common usable length and named exclusions. Adjacent song metadata defines the expected inputs when available. With audio-only distributions the inventory covers the files present on disk. Review exclusions and regenerate the inventory after the download finishes. File sizes and modification times are recorded to detect changed audio. The tracked run snapshots the inventory.

The inspection recipe probes one centered window per accepted song at its native rate, duplicates mono inputs to stereo, and exports one float-WAV dry sum and reference pair per fold. The metrics give the exact source offsets. This recipe checks ingestion and audition export. The five-way EQ research comparison is a separate experiment.

## Development and history

    uv run python -m unittest discover -s tests

The supported code lives under `src/nmc`. Historical trainers and checkpoints are indexed in the [research overview](research/README.md). The adapter follows the [MedleyDB song layout](https://github.com/marl/medleydb/blob/master/medleydb/multitrack.py).
