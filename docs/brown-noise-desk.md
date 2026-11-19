# Brown Noise Desk — DSP handoff

This prototype gives the console a sound of its own. Open a folder of aligned mono or stereo tracks. Every channel automatically gets brown-noise EQ followed by level-dependent saturation. Width, Amount and Drive are shared desk controls. Each source gets its own EQ curve and nominal input drive. The listening default is **1 ERB, 100% Amount, 0 dB Drive**. The only input is audio.

## Run from a fresh checkout

From the repository root, with Python 3.11 or later and uv installed, run

```sh
uv sync --locked
mkdir -p runs
uv run nmc audition
```

Open [the desk on localhost](http://127.0.0.1:8766/desk/) and choose **Open folder** or **Open files**. An empty run library is sufficient. Use files that start on the same timeline. The browser decodes them at 48 kHz and holds the full tracks in memory. Shorter tracks end in silence. Once the desk reports **Loudness matched**, play the loop, sweep Drive, and switch between **Desk** and **Bypass desk**. The prepared MedleyDB comparisons are optional local run artifacts. The [desk guide](../README.md#brown-noise-desk) includes commands to prepare a listening excerpt.

## Signal path

```text
At import, per track
    Active windows → stereo-linked ERB power → EQ curve → FIR endpoints
    Active window levels → nominal saturation drive

During playback, per track
    Raw audio → interpolated EQ → saturator → matched wet return ┐
    Raw audio → alignment delay → matched bypass return         ├→ fader / mute / pan
                                                               ┘
    Stereo sum → monitor output
```

The EQ profile stays fixed throughout a song. Saturation responds sample by sample. A bass boost therefore changes what reaches the saturator, a louder snare hit drives the knee harder, and moving a fader changes the balance after both processors. Stereo pairs share a curve and drive calibration while their audio channels are processed separately. [Live graph](../src/nmc/web/desk-live.js)

## Per-track EQ

1. **Measure active audio.** For browser imports, divide the track into up to 48 regions and take the highest-energy 8,192-sample window from each. Apply a Hann window and compute FFT power. Average left and right powers for stereo so opposite-polarity signals still register. Retain windows within 40 dB of the strongest window in energy.
2. **Measure the target through the same bands.** Sum FFT power through 43 overlapping triangular bands, evenly spaced on the ERB-rate scale from 20 Hz to 22.05 kHz. Pass the theoretical brown-noise power spectrum, proportional to `1/f²`, through that same bank. Its band powers include the bandwidth of each filter.
3. **Derive a shape correction.** Subtract track power in dB from target power in dB. Bands within 20 dB of the strongest track band have full weight. The weight tapers linearly to zero at 45 dB below it. Subtract the weighted mean delta to remove the overall level offset. Smooth the remaining delta across the ERB axis with a support-weighted Gaussian. The widths **1, 3 and 6 ERB** are full widths at half maximum. A silent track receives a flat curve.
4. **Build the Amount endpoints.** The 15% curve in dB is `2 × support × tanh(0.15 × smoothed_delta_dB / 2)`, bounded to ±2 dB. Divide this curve by 0.15 for the 100% endpoint, bounded to about ±13.3 dB. Interpolate the band gains in dB using shape-preserving cubic interpolation on the ERB axis, convert to linear magnitude, mirror the spectrum, inverse-transform, center, and Kaiser-window the impulse to **8,193 taps**.
5. **Apply Amount in playback.** Between 0% and 15%, blend delay-aligned dry and the 15% FIR output. Above 15%, blend the 15% and 100% FIR outputs. These blends are linear in amplitude. The FIR center delay is 4,096 samples, about **85.3 ms at 48 kHz**. Preserve that alignment for bypass, seeks, loop boundaries and DSP parity comparisons.

The browser worker is the source for local-folder behavior. The Python reference supplies the same curve and kernel design. Prepared excerpts use a different passage sampler, so compare the implementations on identical audio windows. [Import and sampling](../src/nmc/web/desk-import.js) · [Browser EQ design](../src/nmc/web/desk-analysis.js) · [Python EQ reference](../src/nmc/eq/erb_desk.py)

## Per-track saturation

For imported tracks, set `nominal = clamp(0.30 / P90(active window RMS), 0.25, 8)`. Silence uses 1. The Drive knob adds −12 to +24 dB relative to nominal. At 0 dB the nonlinear stage is already active.

Feed a **+5 dB low-frequency emphasis around 180 Hz** into the **4× oversampled asymmetric soft knee**. For emphasized input sample `x`, the transfer with reciprocal drive compensation is

```text
g = nominal × 10^(Drive_dB / 20)
u = clamp(g × x, −8, 8)
b = 0.14
y = (tanh(u + b) − tanh(b)) / ((1 − tanh(b)²) × g)
```

Downsample, apply the inverse emphasis filter and remove DC with an 8 Hz blocker. The bias gives even as well as odd harmonics. Subtracting the offset gives zero at rest, and the denominator gives unity small-signal slope. The emphasis makes low frequencies reach the knee earlier. This is an original transformer-inspired curve. The prepared Python audition path uses active 50 ms post-emphasis, post-EQ levels to set nominal drive. [Browser stage coefficients](../src/nmc/web/desk-import.js) · [ChannelSaturation implementation](../src/nmc/web/desk-live.js) · [Python saturation reference](../src/nmc/desk_saturation.py)

## Level-matched monitoring

**Bypass desk** switches both color stages while retaining faders, pan, mute and solo. Measure gated K-weighted energy on the summed stereo bus under those controls for bypass and seven drive settings from −12 to +24 dB. The wet trim is the square root of dry-to-wet energy ratio. Apply shared peak headroom, interpolate trim in dB between measured settings, and ramp trim in sync with Drive. A held control setting gets constant gain, preserving the dynamics of the processed signal.

The browser measures sessions up to 12 seconds in full. For longer material it measures ten synchronized 0.8-second passages with filter history, so the full-song match is approximate. In the original 32-track *Miss Thing* check, the +24 dB render was 0.53 LU above bypass measured across the full song. Changes to EQ or the mix balance trigger recalibration. [Loudness calibration](../src/nmc/web/desk-match.js) · [Monitor switching and drive ramps](../src/nmc/web/desk-live.js)

## Verify a port or change

From the checkout, run the Python DSP regressions and the Node monitor tests

```sh
uv run python -m unittest discover -s tests -p 'test_erb_desk.py'
uv run python -m unittest discover -s tests -p 'test_desk*.py'
node --test tests/desk-match.test.cjs
```

- A brown-noise-shaped input should yield negligible correction. Silence should stay silent. Dual-mono and opposite-polarity stereo should produce the same EQ profile on identical windows. [EQ parity tests](../tests/test_erb_desk.py)
- The saturator should have zero output at rest and unity small-signal slope. More drive should increase harmonics. [Saturation tests](../tests/test_desk_saturation.py)
- Audition a moving Drive control and a large jump back to nominal while the loop plays. Check both the settled loudness difference and the transition. A matched endpoint can still hide a volume spike during the ramp. [Monitor tests](../tests/desk-match.test.cjs)
