"use strict";

// Compare the same synchronized passages, post-fader, through the playback DSP.
const matchDrives = [-12, -6, 0, 6, 12, 18, 24];
function matchProbes(session, tracks, mixer) {
  const rate = session.rate,
    pad = session.kernel_frames - 1,
    n = Math.round(session.duration * rate);
  if (n <= rate * 12) return { session, tracks, regions: [[0, n]] };
  const count = 10,
    span = Math.round(rate * 0.8),
    stride = span + 2 * pad;
  const block = 8192,
    bins = Math.ceil(n / block);
  const activity = new Float64Array(bins);
  const meta = new Map(session.tracks.map((t) => [t.id, t]));
  for (const t of tracks) {
    const state = mixer.get(t.id),
      info = meta.get(t.id);
    if (state?.muted || state?.db <= -60) continue;
    const gainPower = 10 ** ((state?.db ?? 0) / 10);
    if (info.matchPower) {
      for (let i = 0; i < info.matchPower.length; i++)
        activity[i] += gainPower * info.matchPower[i];
    } else {
      for (let i = 0; i < bins; i++) {
        const peak = info.peaks[Math.floor((i * info.peaks.length) / bins)];
        activity[i] += gainPower * (peak * (info.raw_gain ?? 1)) ** 2;
      }
    }
  }
  const starts = [];
  for (let g = 0; g < count; g++) {
    const first = Math.floor((g * bins) / count),
      end = Math.floor(((g + 1) * bins) / count);
    let largest = 0;
    for (let i = first; i < end; i++) largest = Math.max(largest, activity[i]);
    const active = [];
    for (let i = first; i < end; i++)
      if (activity[i] > largest * 0.001) active.push(i);
    active.sort((a, b) => activity[a] - activity[b]);
    const pick =
      active[Math.floor(active.length * 0.6)] ?? Math.floor((first + end) / 2);
    starts.push(
      Math.max(
        0,
        Math.min(n - span, Math.round((pick + 0.5) * block - span / 2)),
      ),
    );
  }
  const frames = count * stride;

  const sampled = tracks.map((track) => {
    const audio = new AudioBuffer({
      numberOfChannels: track.audio.numberOfChannels,
      length: frames + 2 * pad,
      sampleRate: rate,
    });
    for (let ch = 0; ch < audio.numberOfChannels; ch++) {
      const source = track.audio.getChannelData(ch),
        out = audio.getChannelData(ch);
      starts.forEach((start, i) =>
        out.set(source.subarray(start, start + stride), pad + i * stride),
      );
      out.set(out.subarray(frames, frames + pad), 0);
      out.set(out.subarray(pad, 2 * pad), pad + frames);
    }
    return { ...track, audio };
  });
  return {
    session: { ...session, duration: frames / rate },
    tracks: sampled,
    regions: starts.map((_, i) => [i * stride + pad, span]),
  };
}

function gatedEnergy(blocks) {
  let absolute = blocks.filter((v) => v > 10 ** ((-70 + 0.691) / 10));
  // Very quiet files can still provide a relative match below the LUFS meter floor.
  if (!absolute.length) absolute = blocks.filter((v) => v > 1e-20);
  if (!absolute.length) return 0;
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const threshold = mean(absolute) * 0.1;
  return mean(absolute.filter((v) => v >= threshold));
}

async function measureReturn(probes, mixer, mode, amount, width, drive) {
  const { session, tracks, regions } = probes,
    rate = session.rate;
  const ctx = new OfflineAudioContext(
    4,
    Math.ceil((session.duration + 0.25) * rate),
    rate,
  );
  const bus = ctx.createGain(),
    packed = ctx.createChannelMerger(4);
  const raw = ctx.createChannelSplitter(2),
    weighted = ctx.createChannelSplitter(2);
  // The same 48 kHz BS.1770 K-weighting sections used by eq/amount.py.
  const shelf = ctx.createIIRFilter(
    [1.53512485958697, -2.69169618940638, 1.19839281085285],
    [1, -1.69065929318241, 0.73248077421585],
  );
  const highpass = ctx.createIIRFilter(
    [1, -2, 1],
    [1, -1.99004745483398, 0.99007225036621],
  );
  bus.connect(raw);
  bus.connect(shelf).connect(highpass).connect(weighted);
  for (let i = 0; i < 2; i++) {
    raw.connect(packed, i, i);
    weighted.connect(packed, i, i + 2);
  }
  packed.connect(ctx.destination);
  const tone = {
    spec: consoleSpec(rate),
    bypassed: drive === null,
    drive: drive ?? 0,
  };
  const player = new LiveDesk(
    ctx,
    bus,
    session,
    tracks,
    session,
    0,
    true,
    mode,
    amount,
    width,
    null,
    null,
    mixer,
    tone,
  );
  let rendered;
  try {
    rendered = await ctx.startRendering();
  } finally {
    player.stop();
  }
  const [l, r, kl, kr] = [0, 1, 2, 3].map((i) => rendered.getChannelData(i));
  const origin = Math.round(player.origin * rate),
    hop = rate / 10,
    blocks = [];
  let peak = 0;
  for (const [offset, length] of regions) {
    const powers = [];
    for (let start = 0; start < length; start += hop) {
      const a = origin + offset + start,
        end = Math.min(a + hop, origin + offset + length);
      let energy = 0;
      for (let i = a; i < end; i++) {
        energy += kl[i] ** 2 + kr[i] ** 2;
        peak = Math.max(peak, Math.abs(l[i]), Math.abs(r[i]));
      }
      powers.push(energy / hop);
    }
    while (powers.length < 4) powers.push(0);
    for (let i = 3; i < powers.length; i++)
      blocks.push(
        (powers[i] + powers[i - 1] + powers[i - 2] + powers[i - 3]) / 4,
      );
  }
  return { power: gatedEnergy(blocks), peak };
}

async function calibrateDesk(
  session,
  tracks,
  mixer,
  mode,
  amount,
  width,
  signal,
  progress,
) {
  const probes = matchProbes(session, tracks, mixer);
  progress?.(1, matchDrives.length + 1);
  const dry = await measureReturn(probes, mixer, mode, amount, width, null);
  const trims = {},
    result = {
      drive_grid_db: matchDrives,
      trims,
      headroom: 1,
      silent: dry.power === 0,
    };
  let top = dry.peak;
  for (const [i, drive] of matchDrives.entries()) {
    signal.throwIfAborted();
    progress?.(i + 2, matchDrives.length + 1);
    const wet = await measureReturn(probes, mixer, mode, amount, width, drive);
    // Silence or a phase-null has no meaningful loudness ratio.
    trims[drive] =
      wet.power && dry.power ? Math.sqrt(dry.power / wet.power) : 1;
    if (!wet.power) result.silent = true;
    top = Math.max(top, wet.peak * trims[drive]);
  }
  result.headroom = Math.min(1, 0.85 / Math.max(top, 1e-10));
  return result;
}
