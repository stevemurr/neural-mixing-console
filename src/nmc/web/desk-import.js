"use strict";
function consoleSpec(rate) {
  const pole = Math.exp((-2 * Math.PI * 180) / rate);
  return {
    feed: [1 + (10 ** (5 / 20) - 1) * (1 - pole), -pole],
    feedback: [1, -pole],
    dc_pole: Math.exp((-2 * Math.PI * 8) / rate),
    bias: 0.14,
    drives_db: {},
    drive_grid_db: [-12, 24],
  };
}

// Use active windows across the whole file, including isolated late entries.
function sampleTrack(audio) {
  const data = Array.from({ length: audio.numberOfChannels }, (_, i) =>
    audio.getChannelData(i),
  );
  const peaks = Array(320).fill(0),
    frames = [];
  const block = 8192,
    blocks = Math.ceil(audio.length / block),
    groups = Math.min(48, blocks);
  const matchPower = new Float32Array(blocks);
  for (let group = 0; group < groups; group++) {
    let best = -1,
      start = 0;
    for (
      let b = Math.floor((group * blocks) / groups);
      b < Math.floor(((group + 1) * blocks) / groups);
      b++
    ) {
      let energy = 0;
      const end = Math.min((b + 1) * block, audio.length);
      for (const ch of data) {
        for (let i = b * block; i < end; i++) {
          const v = ch[i];
          energy += v * v;
          const bin = Math.floor((i * peaks.length) / audio.length);
          peaks[bin] = Math.max(peaks[bin], Math.abs(v));
        }
      }
      matchPower[b] = energy / ((end - b * block) * data.length);
      if (energy > best) {
        best = energy;
        start = b * block;
      }
    }
    frames.push(data.map((ch) => ch.slice(start, start + block)));
  }
  return { frames, peaks, matchPower };
}

function measureTrack(worker, frames, signal) {
  return new Promise((resolve, reject) => {
    const abort = () =>
      reject(new DOMException("Import cancelled", "AbortError"));
    signal.addEventListener("abort", abort, { once: true });
    worker.onmessage = (e) => {
      signal.removeEventListener("abort", abort);
      resolve(e.data);
    };
    worker.onerror = (e) => {
      signal.removeEventListener("abort", abort);
      reject(Error(e.message || "Channel analysis failed"));
    };
    worker.postMessage(
      frames,
      frames.flat().map((a) => a.buffer),
    );
  });
}

function summedPeak(tracks, length) {
  let peak = 0;
  const block = new Float64Array(16384);
  const channels = tracks.map((t) => [
    t.audio.getChannelData(0),
    t.audio.getChannelData(t.audio.numberOfChannels - 1),
  ]);
  for (let start = 0; start < length; start += block.length)
    for (let side = 0; side < 2; side++) {
      block.fill(0);
      for (const pair of channels) {
        const data = pair[side],
          n = Math.min(block.length, data.length - start);
        for (let j = 0; j < n; j++) block[j] += data[start + j];
      }
      for (const v of block) peak = Math.max(peak, Math.abs(v));
    }
  return peak;
}

function alignedLoop(ctx, audio, length, pad) {
  const padded = ctx.createBuffer(
    audio.numberOfChannels,
    length + 2 * pad,
    ctx.sampleRate,
  );
  for (let i = 0; i < audio.numberOfChannels; i++) {
    const out = padded.getChannelData(i);
    out.set(audio.getChannelData(i), pad);
    for (let j = 0; j < pad; j++) {
      out[j] = out[pad + ((((j - pad) % length) + length) % length)];
      out[pad + length + j] = out[pad + (j % length)];
    }
  }
  return padded;
}

async function importMultitracks(ctx, files, signal, progress) {
  const audioFiles = Array.from(files)
    .filter(
      (f) =>
        (!f.webkitRelativePath ||
          f.webkitRelativePath.split("/").length === 2) &&
        !f.name.startsWith(".") &&
        (/\.(wav|wave|aif|aiff|flac|mp3|ogg|opus|m4a|aac)$/i.test(f.name) ||
          f.type.startsWith("audio/")),
    )
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  if (!audioFiles.length)
    throw Error(
      "Choose a folder with aligned audio tracks at its top level, or select the files.",
    );
  const title =
    audioFiles[0].webkitRelativePath?.split("/")[0] ||
    audioFiles[0].name.replace(/\.[^.]+$/, "") +
      (audioFiles.length > 1 ? " + " + (audioFiles.length - 1) : "");
  const spec = consoleSpec(ctx.sampleRate),
    tracks = [],
    cache = new Map();
  let centers,
    length = 0;
  const worker = new Worker("/desk/desk-analysis.js");
  try {
    for (const [i, file] of audioFiles.entries()) {
      signal.throwIfAborted();
      progress(
        "Opening " + (i + 1) + "/" + audioFiles.length + " · " + file.name,
      );
      let audio;
      try {
        audio = await ctx.decodeAudioData(await file.arrayBuffer());
      } catch {
        throw Error(
          "Could not decode " + file.name + ". Try a PCM WAV or FLAC.",
        );
      }
      signal.throwIfAborted();
      if (audio.numberOfChannels > 2)
        throw Error(
          file.name + " has more than two channels. Use mono or stereo tracks.",
        );
      const { frames, peaks, matchPower } = sampleTrack(audio);
      const analysis = await measureTrack(worker, frames, signal);
      signal.throwIfAborted();
      centers = analysis.centers;
      const id = "ch" + String(i + 1).padStart(3, "0"),
        kernels = {};
      for (const [width, pair] of Object.entries(analysis.kernels)) {
        kernels[width] = pair.map((taps) => {
          const ir = ctx.createBuffer(1, taps.length, ctx.sampleRate);
          ir.copyToChannel(taps, 0);
          return ir;
        });
      }
      length = Math.max(length, audio.length);
      cache.set(id, { id, audio, kernels, lineDrive: analysis.nominal });
      tracks.push({
        id,
        name: file.name,
        label: file.name.replace(/\.[^.]+$/, ""),
        channels: audio.numberOfChannels,
        frames: audio.length,
        matchPower,
        peaks,
        curves: analysis.curves,
      });
    }
  } finally {
    worker.terminate();
  }
  progress("Aligning tracks…");
  const monitorTrim = Math.min(
    1,
    0.5 / Math.max(1e-9, summedPeak([...cache.values()], length)),
  );
  for (const t of tracks) {
    const data = cache.get(t.id);
    data.audio = alignedLoop(ctx, data.audio, length, 8192);
    // Show silence after the end of a shorter file on the shared ruler.
    const fraction = t.frames / length;
    t.peaks = t.peaks.map((_, i, a) =>
      i / a.length < fraction ? a[Math.floor(i / fraction)] : 0,
    );
  }
  return {
    id: "local",
    local: true,
    live_eq: true,
    cache,
    title,
    artist: "Local multitracks",
    rate: ctx.sampleRate,
    start: 0,
    duration: length / ctx.sampleRate,
    kernel_frames: 8193,
    widths: [1, 3, 6],
    amount: 0.15,
    limit_db: 2,
    centers,
    tracks,
    saturation: spec,
    monitorTrim,
  };
}
