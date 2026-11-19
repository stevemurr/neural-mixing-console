"use strict";

// Coefficients of parallel, phase-aligned FIR filters. Their sum is the live EQ response.
function amountWeights(percent, normal) {
  if (percent <= normal) return [1 - percent / normal, percent / normal, 0];
  const t = (percent - normal) / (100 - normal);
  return [0, 1 - t, t];
}

async function loadLiveTrack(ctx, session, track, signal) {
  const base = "/desk/live/" + session.id + "/" + track.id + "/";
  async function bytes(name) {
    const response = await fetch(base + name, { signal });
    if (!response.ok)
      throw Error(name + " unavailable (" + response.status + ")");
    return response.arrayBuffer();
  }
  const [audio, bytesIR] = await Promise.all([
    bytes("dry.flac").then((b) => ctx.decodeAudioData(b)),
    bytes("kernels.f32"),
  ]);
  const taps = session.kernel_frames;
  if (
    ctx.sampleRate !== session.rate ||
    bytesIR.byteLength !== session.widths.length * 2 * taps * 4
  )
    throw Error("Incompatible live filter bank");
  const raw = new Float32Array(bytesIR);
  const kernels = {};
  session.widths.forEach((width, i) => {
    kernels[width] = [0, 1].map((endpoint) => {
      const fir = ctx.createBuffer(1, taps, ctx.sampleRate);
      const start = (i * 2 + endpoint) * taps;
      fir.copyToChannel(raw.subarray(start, start + taps), 0);
      return fir;
    });
  });
  // Circular padding supplies FIR history at every seek, including time zero.
  const pad = taps - 1,
    n = audio.length;
  const padded = ctx.createBuffer(
    audio.numberOfChannels,
    n + 2 * pad,
    ctx.sampleRate,
  );
  for (let ch = 0; ch < audio.numberOfChannels; ch++) {
    const input = audio.getChannelData(ch),
      out = padded.getChannelData(ch);
    for (let j = 0; j < n; j++) out[pad + j] = input[j] * track.raw_gain;
    for (let j = 0; j < pad; j++) {
      out[j] = out[pad + ((((j - pad) % n) + n) % n)];
      out[pad + n + j] = out[pad + (j % n)];
    }
  }
  return {
    id: track.id,
    audio: padded,
    kernels,
    lineDrive: track.saturation?.drive,
  };
}

// Match measured excerpt loudness in log gain between adjacent drive settings.
function saturationTrim(spec, trims, key) {
  if (Object.hasOwn(trims, key)) return trims[key];
  const db = Number(key);
  const grid = spec.drive_grid_db;
  const upper = grid.findIndex((v) => v >= db);
  if (upper === -1) return trims[grid.at(-1)];
  if (upper === 0) return trims[grid[0]];
  const lo = grid[upper - 1],
    hi = grid[upper];
  return trims[lo] * (trims[hi] / trims[lo]) ** ((db - lo) / (hi - lo));
}

// Establish an explicit ramp start even after a long idle interval. A ramp
// without an anchor can begin at an old event and reach its target immediately.
function holdParameter(param, now) {
  const current = param.value;
  if (param.cancelAndHoldAtTime) param.cancelAndHoldAtTime(now);
  else param.cancelScheduledValues(now);
  param.setValueAtTime(current, now);
}

// A single nonlinear stage per raw channel. Stereo pairs share the same drive.
class ChannelSaturation {
  constructor(ctx, input, output, spec, nominal, curve, key) {
    this.ctx = ctx;
    this.spec = spec;
    this.nominal = nominal;
    const tilt = ctx.createIIRFilter(spec.feed, spec.feedback);
    const restore = ctx.createIIRFilter(spec.feedback, spec.feed);
    const hp = (1 + spec.dc_pole) / 2;
    const dcBlock = ctx.createIIRFilter([hp, -hp], [1, -spec.dc_pole]);
    const shape = ctx.createWaveShaper();
    shape.curve = curve;
    shape.oversample = "4x";
    this.inputDrive = ctx.createGain();
    this.returnGain = ctx.createGain();
    this.clean = ctx.createGain();
    this.wet = ctx.createGain();
    input.connect(this.clean).connect(output);
    input
      .connect(tilt)
      .connect(this.inputDrive)
      .connect(shape)
      .connect(this.returnGain)
      .connect(restore)
      .connect(dcBlock)
      .connect(this.wet)
      .connect(output);
    this.nodes = [
      tilt,
      restore,
      dcBlock,
      shape,
      this.inputDrive,
      this.returnGain,
      this.clean,
      this.wet,
    ];
    this.set(key, true);
  }

  static curve(bias) {
    const curve = new Float32Array(16385);
    const offset = Math.tanh(bias);
    for (let i = 0; i < curve.length; i++) {
      const x = 8 * ((2 * i) / (curve.length - 1) - 1);
      curve[i] = (Math.tanh(x + bias) - offset) / (1 - offset * offset);
    }
    return curve;
  }

  set(key, initial = false, seconds = 0.025) {
    const colored = key != null && key !== "clean";
    const db = colored ? (this.spec.drives_db[key] ?? Number(key)) : 0;
    const drive = this.nominal * 10 ** (db / 20);
    const now = this.ctx.currentTime;
    for (const [param, value, exponential] of [
      [this.inputDrive.gain, drive / 8, true],
      [this.returnGain.gain, 1 / drive, true],
      [this.clean.gain, colored ? 0 : 1, false],
      [this.wet.gain, colored ? 1 : 0, false],
    ]) {
      if (!initial && exponential && !colored) continue;
      if (initial) {
        param.value = value;
        continue;
      }
      holdParameter(param, now);
      if (exponential) param.exponentialRampToValueAtTime(value, now + seconds);
      else param.linearRampToValueAtTime(value, now + seconds);
    }
  }

  disconnect() {
    this.nodes.forEach((node) => node.disconnect());
  }
}

// The center position reproduces the original desk's unity L/R routing.
class ChannelStrip {
  constructor(ctx, channels, output, state) {
    this.ctx = ctx;
    this.input = ctx.createGain();
    this.input.channelCountMode = "explicit";
    this.input.channelCount = channels;
    this.fader = ctx.createGain();
    this.input.connect(this.fader);
    this.split = ctx.createChannelSplitter(2);
    if (channels === 1) {
      this.panner = ctx.createStereoPanner();
      // The dry desk duplicates mono to both speakers at unity. Equal-power
      // panning puts each side at -3 dB in the center, so restore that reference.
      this.input.gain.value = Math.SQRT2;
      this.fader.connect(this.panner);
      this.tail = this.panner;
    } else {
      const sides = ctx.createChannelSplitter(2);
      this.fader.connect(sides);
      this.tail = ctx.createChannelMerger(2);
      this.balance = [0, 1].map((i) => {
        const gain = ctx.createGain();
        sides.connect(gain, i);
        gain.connect(this.tail, 0, i);
        return gain;
      });
      this.sides = sides;
    }
    this.tail.connect(output);
    this.tail.connect(this.split);
    this.meters = [0, 1].map((i) => {
      const meter = ctx.createAnalyser();
      meter.fftSize = 2048;
      this.split.connect(meter, i);
      return meter;
    });
    this.samples = new Float32Array(this.meters[0].fftSize);
    this.set(state, true);
  }

  set({ db = 0, pan = 0, muted = false } = {}, initial = false) {
    const update = (param, value) => {
      if (initial) param.value = value;
      else param.setTargetAtTime(value, this.ctx.currentTime, 0.006);
    };
    update(this.fader.gain, muted || db <= -60 ? 0 : 10 ** (db / 20));
    if (this.panner) update(this.panner.pan, pan);
    else {
      // Stereo balance preserves separation and the level of the near side.
      update(this.balance[0].gain, Math.cos((Math.max(0, pan) * Math.PI) / 2));
      update(this.balance[1].gain, Math.cos((Math.min(0, pan) * Math.PI) / 2));
    }
  }

  peak(side) {
    this.meters[side].getFloatTimeDomainData(this.samples);
    let peak = 0;
    for (const sample of this.samples) peak = Math.max(peak, Math.abs(sample));
    return peak;
  }

  disconnect() {
    this.input.disconnect();
    this.fader.disconnect();
    this.tail.disconnect();
    this.sides?.disconnect();
    this.balance?.forEach((gain) => gain.disconnect());
    this.split.disconnect();
    this.meters.forEach((meter) => meter.disconnect());
  }
}

class LiveDesk {
  constructor(
    ctx,
    output,
    session,
    tracks,
    calibration,
    offset,
    looping,
    mode,
    amount,
    lastWidth,
    balanceKey = null,
    colorKey = null,
    mixer = new Map(),
    tone = null,
  ) {
    this.ctx = ctx;
    this.tone = tone;
    this.toneInputs = new Map();
    this.toneStages = [];
    this.toneReturns = [];
    this.session = session;
    this.calibration = calibration;
    this.balanceKey = balanceKey;
    this.colorKey = colorKey;
    const spec = session.saturation || tone?.spec;
    this.shapeCurve = spec ? ChannelSaturation.curve(spec.bias) : null;
    this.tracks = tracks;
    this.banks = new Map();
    this.output = ctx.createGain();
    this.output.connect(output);
    const history = (session.kernel_frames - 1) / ctx.sampleRate;
    this.history = history;
    this.strips = new Map();
    this.dry = [];
    this.delays = [];
    this.sources = tracks.map((track) => {
      const strip = new ChannelStrip(
        ctx,
        track.audio.numberOfChannels,
        this.output,
        mixer.get(track.id),
      );
      this.strips.set(track.id, strip);
      let input = strip.input;
      let bypassReturn;
      if (tone) {
        input = ctx.createGain();
        const wetReturn = ctx.createGain();
        bypassReturn = ctx.createGain();
        wetReturn.connect(strip.input);
        bypassReturn.connect(strip.input);
        this.toneReturns.push({ wetReturn, bypassReturn });
        this.toneStages.push(
          new ChannelSaturation(
            ctx,
            input,
            wetReturn,
            spec,
            track.lineDrive ?? 1,
            this.shapeCurve,
            String(tone.drive),
          ),
        );
      }
      this.toneInputs.set(track.id, input);
      const source = ctx.createBufferSource();
      source.buffer = track.audio;
      source.loop = looping;
      source.loopStart = history;
      source.loopEnd = history + session.duration;
      const delay = ctx.createDelay(1);
      delay.delayTime.value = history / 2;
      const dry = ctx.createGain();
      source.connect(delay).connect(dry).connect(input);
      if (bypassReturn) delay.connect(bypassReturn);
      this.dry.push(dry);
      this.delays.push(delay);
      return source;
    });
    this.set(mode, amount, lastWidth, true);
    if (tone) this.setMonitorMatch(true);
    const start = ctx.currentTime + 0.02;
    this.origin = start + history;
    this.output.gain.setValueAtTime(0, ctx.currentTime);
    this.output.gain.setValueAtTime(
      tone ? (session.monitorTrim ?? session.gains["1"][0]) : 1,
      this.origin,
    );
    if (!looping)
      this.output.gain.setValueAtTime(
        0,
        this.origin + session.duration - offset,
      );
    this.sources.forEach((s) => s.start(start, offset + history / 2));
  }

  makeBank(width) {
    const gains = [[], []];
    const filters = [];
    const faders = [];
    const colors = [];
    gains.forEach((endpoint, i) => {
      this.tracks.forEach((track, j) => {
        const g = this.ctx.createGain();
        g.gain.value = 0;
        endpoint.push(g);
        g.connect(this.toneInputs.get(track.id));
        const filter = this.ctx.createConvolver();
        filter.normalize = false;
        filter.buffer = track.kernels[width][i];
        this.sources[j].connect(filter);
        let destination = g;
        if (
          !this.tone &&
          this.session.balance &&
          this.calibration === this.session &&
          width === "1" &&
          i === 1
        ) {
          const fader = this.ctx.createGain();
          fader.gain.value = this.faderGain(track.id);
          destination = fader;
          fader.connect(g);
          faders.push({ id: track.id, fader });
        }
        if (!this.tone && this.session.saturation && width === "1" && i === 1) {
          colors.push(
            new ChannelSaturation(
              this.ctx,
              filter,
              destination,
              this.session.saturation,
              track.lineDrive,
              this.shapeCurve,
              this.colorKey,
            ),
          );
        } else {
          filter.connect(destination);
        }
        filters.push([this.sources[j], filter]);
      });
    });
    const bank = {
      gains,
      filters,
      faders,
      colors,
      ready: this.ctx.currentTime + this.history,
    };
    this.banks.set(width, bank);
    return bank;
  }

  faderGain(id) {
    if (!this.balanceKey) return 1;
    const experiment = this.session.balance;
    const variant = experiment.modes[this.balanceKey];
    // The existing wet bus already applies the original mix calibration.
    return (
      (10 ** (variant.db[experiment.membership[id]] / 20) * variant.trim) /
      this.session.gains["1"][100]
    );
  }

  updateTone() {
    const stage = this.toneStages[0];
    const from =
      20 * Math.log10((stage.inputDrive.gain.value * 8) / stage.nominal);
    const seconds = Math.max(0.025, Math.abs(this.tone.drive - from) * 0.006);
    this.toneStages.forEach((s) =>
      s.set(String(this.tone.drive), false, seconds),
    );
    this.setMonitorMatch(false, from, seconds);
  }

  setMonitorMatch(initial = false, from = this.tone.drive, seconds = 0.025) {
    const { match, drive, bypassed } = this.tone;
    const common = match?.headroom ?? 1;
    const wet =
      (match ? saturationTrim(match, match.trims, drive) : 1) * common;
    const now = this.ctx.currentTime;
    const switching = this.lastBypassed !== bypassed;
    this.lastBypassed = bypassed;
    const knots = (match?.drive_grid_db ?? []).filter(
      (db) => db > Math.min(from, drive) && db < Math.max(from, drive),
    );
    if (from > drive) knots.reverse();
    for (const { wetReturn, bypassReturn } of this.toneReturns) {
      for (const [param, value] of [
        [wetReturn.gain, bypassed ? 0 : wet],
        [bypassReturn.gain, bypassed ? common : 0],
      ]) {
        if (initial) {
          param.value = value;
          continue;
        }
        holdParameter(param, now);
        // Visit intermediate calibration points on large moves, in sync
        // with the drive's dB ramp, and crossfade the two returns for bypass.
        if (!switching && value > 0 && param.value > 0) {
          if (param === wetReturn.gain)
            for (const db of knots) {
              const time = now + (seconds * (db - from)) / (drive - from);
              param.exponentialRampToValueAtTime(
                common * match.trims[db],
                time,
              );
            }
          param.exponentialRampToValueAtTime(value, now + seconds);
        } else param.linearRampToValueAtTime(value, now + 0.025);
      }
    }
  }

  setSaturation(key) {
    if (this.colorKey === key) return;
    this.colorKey = key;
    this.banks.get("1")?.colors.forEach((stage) => stage.set(key));
  }

  setBalance(key) {
    this.balanceKey = key;
    for (const { id, fader } of this.banks.get("1")?.faders || []) {
      fader.gain.setTargetAtTime(
        this.faderGain(id),
        this.ctx.currentTime,
        0.008,
      );
    }
  }

  set(mode, amount, lastWidth, initial = false) {
    clearTimeout(this.cleanup);
    const width = mode === "off" ? lastWidth : mode;
    const bank = this.banks.get(width) || this.makeBank(width);
    const p = mode === "off" ? 0 : amount;
    const weights = amountWeights(p, Math.round(this.session.amount * 100));
    const colorActive = this.colorKey != null;
    const trim = this.tone
      ? 1
      : colorActive
        ? saturationTrim(
            this.session.saturation,
            this.calibration.saturation.trims,
            this.colorKey,
          )
        : this.calibration.gains[width][p];
    const now = this.ctx.currentTime;
    const when = initial || p === 0 ? now : Math.max(now, bank.ready);
    function move(gain, value) {
      const param = gain.gain;
      if (initial) {
        param.value = value;
        return;
      }
      if (param.cancelAndHoldAtTime) param.cancelAndHoldAtTime(now);
      else {
        const valueNow = param.value;
        param.cancelScheduledValues(now);
        param.setValueAtTime(valueNow, now);
      }
      if (colorActive && when === now && param.value > 0 && value > 0)
        param.exponentialRampToValueAtTime(value, now + 0.025);
      else param.setTargetAtTime(value, when, 0.006);
    }
    this.dry.forEach((g) => move(g, trim * weights[0]));
    for (const [key, b] of this.banks) {
      b.gains.forEach((endpoint, i) =>
        endpoint.forEach((g) =>
          move(g, key === width ? weights[i + 1] * trim : 0),
        ),
      );
    }
    this.cleanup = setTimeout(
      () => {
        for (const key of this.banks.keys()) if (key !== width) this.drop(key);
      },
      Math.max(0, when - now) * 1000 + 180,
    );
  }

  drop(key) {
    const bank = this.banks.get(key);
    for (const [source, filter] of bank.filters) {
      source.disconnect(filter);
      filter.disconnect();
    }
    bank.colors.forEach((stage) => stage.disconnect());
    bank.faders.forEach(({ fader }) => fader.disconnect());
    bank.gains.flat().forEach((g) => g.disconnect());
    this.banks.delete(key);
  }

  stop() {
    clearTimeout(this.cleanup);
    for (const key of this.banks.keys()) this.drop(key);
    this.sources.forEach((s) => {
      s.stop();
      s.disconnect();
    });
    this.delays.forEach((d) => d.disconnect());
    this.dry.forEach((g) => g.disconnect());
    this.toneStages.forEach((stage) => stage.disconnect());
    this.toneReturns.forEach(({ wetReturn, bypassReturn }) => {
      wetReturn.disconnect();
      bypassReturn.disconnect();
    });
    if (this.tone) this.toneInputs.forEach((input) => input.disconnect());
    this.strips.forEach((strip) => strip.disconnect());
    this.output.disconnect();
  }
}
