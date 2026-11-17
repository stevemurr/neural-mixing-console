# realtime/

Realtime prototype: load a folder of stem audio files and play them
back through the trained mixing model. Iteration target is "fast to
change," not "fast at runtime" — Python first; we port hot paths to
C++ once the architecture stops moving (see the design discussion in
the project notes).

## Versions

- **V0** (`session.py` + `engine.py` + `cli.py`): stems passthrough
  player. No DSP, no model — just sum the stems through a sounddevice
  output stream. Confirms the audio-I/O path works on your machine.
- *V1+*: ring buffers + inference scheduler + DSP chain + smoother.
  Sketched in the design doc; not yet implemented.

## Install

`sounddevice` is in the `realtime` optional extra. PortAudio (the
native lib sounddevice wraps) must also be installed at the OS level
for actual playback; not required for unit-testing the callback.

```bash
uv sync --extra realtime              # python deps
sudo apt install libportaudio2        # Debian/Ubuntu — PortAudio
# macOS:  brew install portaudio
# Fedora: sudo dnf install portaudio
```

## Run V0

```bash
uv run python -m realtime.cli \
    --stems-dir source_audio/cambridge-mt/AnnaBlanton_Rachel_Full/AnnaBlanton_Rachel_Full \
    --block-size 512 \
    --master-gain-db -6
```

Press Ctrl-C to stop. Loops by default; pass `--no-loop` to stop at
end of session. `--list-devices` shows available output devices;
`--device <name>` (substring or index) selects one.

## Layout

```
realtime/
├── __init__.py    re-exports Session, AudioEngine
├── session.py     Session — load N stems → (N, 2, T_max) at 48 kHz
├── engine.py     AudioEngine — sounddevice callback, playhead, summing
└── cli.py        entry point: argparse + live playhead display
```

## Constraints

The audio callback (`AudioEngine._callback`) is the realtime hot path.
It must not allocate, must not block, must not log beyond
sounddevice-flagged statuses. All user-facing work happens on the main
thread (`cli.py`). All inference + GUI threads will be separate; they
hand parameters to the callback via lock-free atomic snapshots — see
V1+ design notes.
