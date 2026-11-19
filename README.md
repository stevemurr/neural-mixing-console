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

### Brown Noise Desk

For fresh-checkout startup, DSP formulas, signal routing and port verification, see the [EQ and saturation handoff](docs/brown-noise-desk.md).

Choose **Brown Noise Desk** in the run library or open `/desk/` on the same server. Press Space to play. Keys 1, 2 and 3 switch between 1, 3 and 6 ERB widths during playback. Key 0 selects bypass. Click a channel name to inspect its response or its S button to hear the isolated track.

Click **Open folder** to play aligned mono or stereo files on your computer, or **Open files** to select individual tracks. The browser decodes the full songs at 48 kHz. Tracks start together and shorter files end in silence. Imported audio stays in browser memory until you open another set or close the page. Large sessions need enough browser memory for the decoded audio.

**Desk** is the default sound of every live session. Each channel runs through its own brown-noise EQ, 4× oversampled saturation, and mixer strip. Start at 1 ERB, 100% EQ and nominal drive. Turn Amount and Drive independently while playing. **Bypass desk** switches both color stages together and keeps the faders and pan. Desk matches the processed stereo bus to bypass using K-weighted loudness measured after the current faders, pans and mutes. The browser measures seven drive settings from −12 to +24 dB, then interpolates monitor gain in decibels as you turn Drive. Each setting has a constant gain, so a held setting preserves dynamics. Up to 12 seconds are measured in full. For longer songs, the browser measures synchronized passages distributed across the song. **Measuring level match…** appears when changing EQ, mixer settings or solo requires a fresh measurement. Both choices share peak headroom based on the measured audio. **Audition presets** opens the existing calibrated comparisons on prepared sessions.

On import, a browser worker analyzes up to 48 active windows sampled across each file. It derives a stereo-linked EQ curve against the same ERB brown-noise target and sets nominal drive from active input RMS. The EQ profile stays fixed for playback. The saturator responds to the instantaneous post-EQ signal. The prototype plays files from memory and loops the session.

For a reusable, calibrated audition excerpt, prepare a session from aligned tracks:

    uv run python -m nmc.desk_session /audio/song/raw --title "Song title" --seconds 20

The command writes a new session under `runs/erb-desk/`. Reload the desk to see it. Use `--start` for a specific excerpt in seconds. The default selects an excerpt with activity across the tracks. Optional MedleyDB YAML supplies display labels.

The excerpt preparation command samples up to 24 one-second passages across each track, averages active-frame channel powers and measures a band-limited brown-noise spectrum through the same 43 overlapping ERB filters (20 Hz to 22.05 kHz). It aligns the target to the track's supported frequencies and smooths the difference at the chosen Gaussian full width at half maximum. The browser plays the dry tracks through stereo-linked FIR filters. Turn **Amount** by dragging up or right, or use the arrow keys while the knob has focus. Moving the knob engages the selected width. Double-click to return to 15%.

At 0% the response is flat. At 15% the original soft limit is ±2 dB. At 100% the per-track response can reach ±13.3 dB. Intermediate settings blend aligned filter responses in linear amplitude. The inspector shows the selected track’s applied curve and gain range. The curve is derived once per song. Filtering and gain changes run in Web Audio while the loop plays.

In the prepared audition presets, each bus has an integrated-loudness trim for every width and integer Amount. Preparation calculates these from K-weighted block powers, including correlation between filter outputs, using a common peak headroom for the bus. A fixed-amount session can be upgraded at its existing URL by supplying its original track folder and `--upgrade SESSION_ID`.

#### Channel mixer

Live sessions open in Mixer view. Each raw track has a post-insert fader, pan or stereo balance, mute, solo and two post-fader sample-peak meters (dBFS). The fader runs from silence to +12 dB. Double-click a fader for unity or a pan slider for center. Mono uses equal-power panning referenced to the original center gain. Stereo balance attenuates the opposite side independently. In Audition presets, the original EQ and saturation loudness calibration applies at unity with centered, unmuted strips. Manual edits persist per song while the page is open. Switch to Arrangement to see the waveforms.

#### Channel saturation

On *Miss Thing* and *Alive*, **Audition presets** opens on **Clean EQ**. Drag **Drive** up or right to sweep from −12 to +24 dB while the loop plays. Double-click for nominal drive (0 dB). Switch between **Clean EQ** and **Saturation** to compare at the same knob setting, or solo a raw channel. The shared knob drives each channel’s nonlinear stage after its 1 ERB / 100% EQ. The mix uses unity faders. Each channel has a fixed operating level, chosen from its active 50 ms post-EQ levels. Stereo channels share that gain. A low shelf drives low frequencies into a mildly asymmetric soft knee, followed by reciprocal EQ and DC removal. The browser applies 4× oversampling at the nonlinear stage. This is an original transformer-inspired voicing; an individual hardware console has its own measured response.

Preparation measures the excerpt loudness every 3 dB on the mix bus and separately for each solo. The browser interpolates the monitoring trim in decibels while you turn the knob. Use the clean setting for a level-matched EQ-only reference. Prepare the operating levels and trims on an existing live session with:

    uv run python -m nmc.desk_saturation runs/erb-desk/<session-id>

#### Fader comparison

Expand **Fader comparison / Explore EQ** to audition the three static balances. All three use the same 20-second loop, 1 ERB at 100%, and static source faders. **Unity faders** keeps the raw recording levels. **Active loudness** sums the post-EQ microphones in each annotated stem, measures median K-weighted loudness over its active 400 ms blocks, and closes one third of the gap to the median source, up to ±4 dB.

**Vocal-anchored ERB** holds the designated vocal group at unity and starts other groups at their active-loudness faders. A 1 dB trial cut measures each group's effect on the actual mix spectrum. Bands with vocal energy receive more weight, and competing sources receive a further static cut when they dominate those bands during singing. All faders remain within ±4 dB. Settings are fitted to the audition excerpt. The mix monitor applies one constant loudness trim per mode with shared headroom. The inspector shows the selected track's group fader, and solo monitors the level-matched pre-fader channel. Choose **Explore EQ** to return to width and Amount controls.

To prepare the comparison on an existing live session, pass the session folder, its MedleyDB YAML and the anchor stem ID:

    uv run python -m nmc.desk_balance runs/erb-desk/<session-id> <song>_METADATA.yaml --anchor S05

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
