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

## Audition runs in the browser

    uv run nmc audition

Open **http://127.0.0.1:8766** on the machine running the checkout. Select a run, song and listening point. The scored excerpts offer **Prediction**, **Dry sum**, and **Engineer mix** at matched RMS and common peak headroom. Switching source keeps playback at the same position in the excerpt. Click the waveform to seek, or enable Loop to repeat it. **Full preview** plays the complete model output with its original master gain.

The library reads local run records and WAVs directly. Refresh the run library after a new run finishes. Inspection runs expose their saved dry/reference pairs; feature-cache replays appear with a “No audio” status. Use `--runs-dir PATH` for another run collection or `--port PORT` to change the listening port. For direct access from your LAN or tailnet, start `uv run nmc audition --host 0.0.0.0` and open `http://<project-machine-ip>:8766/` in your browser. You can also bind a specific interface by passing its IPv4 address to `--host`.

Current attempts live under `runs/<recipe>/<run-id>/`. The legacy logs, PID files and TensorBoard event directories from this checkout were moved together into `dmc-data/archive/runs-pre-manifests/` with their original names.

## Reproduce and track an experiment

From the repository root, install the project with `uv sync --locked`. For an existing research environment with locally installed audio tools, `uv pip install --no-deps --no-build-isolation -e .` installs the command using the environment’s existing dependencies and setuptools.

    uv run nmc run experiments/eq/reproduce-v13.toml --check
    uv run nmc run experiments/eq/reproduce-v13.toml
    uv run nmc runs list

The original local cache and checkpoint are required. Their paths and hashes are in the recipe. The check verifies both inputs, model compatibility, and the frozen split. On 32 validation songs and 640 windows, zero EQ, the training-mean curve and the learned curve score 4.309, 3.855 and 3.846 dB RMSE.

Every launch creates `runs/<recipe>/<run-id>/`. Open `run.json` for the command’s outcome, revision, working-tree state, input fingerprints and split fingerprint. The resolved settings are in `recipe.json`, the worker output in `run.log`, and the song IDs, window offsets and measurements in `metrics.json`. The reproduction recipe records the original checkpoint by path and hash. `--check` reads the inputs without allocating a run directory. Relative input paths are resolved from the current working directory.

## Benchmark on raw MedleyDB tracks

Run from the repository root on the extracted dataset:

    uv run nmc data prepare medleydb --inputs raw --root /path/to/MedleyDB
    uv run nmc run experiments/eq/medleydb-raw.toml --check
    uv run nmc run experiments/eq/medleydb-raw.toml
    uv run nmc runs list

Raw tracks are the default and produce `dmc-data/medleydb/raw.json`. The inventory records each song's selected audio, reference, common usable length and exclusions. Review exclusions and regenerate after the download finishes. Adjacent metadata can check expected files. Audio-only folders use the audio files present. For a grouped-stem diagnostic, prepare with `--inputs stems` to write `stems.json` and set both `manifest` and `input_kind` in a copy of the recipe. Existing inventories can be used by pointing a recipe at their path.

The recipe assigns each canonical MedleyDB song ID to a fold using the shared split seed and validation fraction. Every window and input variant inherits the song's assignment. Keep those split settings fixed across comparisons. The run records the actual membership. When variants have different exclusions, compare results on their shared held-out songs.

For each validation song the benchmark runs the same folder inference as `nmc mix`. Only the raw audio directory and checkpoint go to the predictor. It writes the controls and full preview under `songs/<song>/`. The evaluator then reads the engineer mix and compares the preview and dry sum on the same three six-second windows. Short songs use the available duration. The metric is RMS-matched spectral RMSE in dB over 26 active-frame mel bands. Smaller values indicate closer tonal shape. The report records the window offsets at 48 kHz, per-song errors and equal-song-weighted aggregate errors in `metrics.json`.

For a faster ingestion check with native-rate listening exports, run `experiments/eq/medleydb-inspect.toml` on the raw inventory.

## Development and history

    uv run python -m unittest discover -s tests

The supported code lives under `src/nmc`. Historical trainers and checkpoints are indexed in the [research overview](research/README.md). The adapter follows the [MedleyDB song layout](https://github.com/marl/medleydb/blob/master/medleydb/multitrack.py).
