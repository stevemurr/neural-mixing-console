# Neural mixing console

The active research workflow studies how to distribute a requested frequency correction across stems using each stem’s measured contribution in 26 mel bands.

## Mix a raw track folder

From the checkout, run the following with the local v13 checkpoint. Installing the project is described below.

    uv run nmc mix /audio/song/raw --out-dir /audio/song/first-mix
    uv run nmc render /audio/song/raw /audio/song/first-mix/mix.json --output /audio/song/replay.wav

`mix` writes an editable `mix.json` and full-song stereo `preview.wav`. The default output is a new sibling directory named `<tracks-folder>-mix`. Use `--checkpoint FILE` to select another compatible v13 target-curve checkpoint. Replay needs only the source folder and JSON, so both can be moved together or independently.

Point the command at aligned, individual tracks starting at time zero. It includes audio in nested directories and supports mono or stereo WAV (including BWF/RF64), AIFF, FLAC, Ogg and MP3 files readable by libsndfile. Keep reference mixes and bounces outside this tree. Files are resampled to 48 kHz, summed through the end of the longest track, and identified by relative path and content hash. Silent tracks retain their entries. Unreadable audio stops the run with its filename.

**Controls.** Each track receives one static EQ curve. Analysis takes the loudest six-second block in each third of that track's blocks and averages the contribution features of the active selections. The model predicts a song-wide curve and the allocator distributes it across the tracks. The JSON records the selected starts in 48 kHz samples.

- `eq_db` gives the applied gain for each interval in `band_edges_hz`, bounded to ±18 dB. Replay uses the v1 mel EQ with 65,536-sample FFT windows, square-root Hann tapers, and 50% overlap.
- `eq_compensation_db` is the measured full-track RMS compensation after EQ. It is saved as a fixed gain so editing the EQ has predictable effects.
- `level_db` starts at 0 dB. `pan` starts at 0 and ranges from −1 (left) to +1 (right). Mono uses equal-power pan with unity gain per channel at center. Stereo preserves its image at center and attenuates the opposite channel when balanced to either side.
- `master_db` attenuates the initial stereo sum if needed for −1 dBFS sample-peak headroom. Replay applies the saved value, including after edits. WAV output uses 32-bit float.

The model currently predicts EQ. Pan and fader values are editable starting settings. Use a fresh output path for each mix or replay.

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
