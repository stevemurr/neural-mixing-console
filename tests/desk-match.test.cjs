const { readFileSync } = require("node:fs");
const { runInNewContext } = require("node:vm");
const { test } = require("node:test");
const assert = require("node:assert/strict");

class TestBuffer {
  constructor({ length, numberOfChannels }) {
    this.numberOfChannels = numberOfChannels;
    this.data = Array.from(
      { length: numberOfChannels },
      () => new Float32Array(length),
    );
  }
  getChannelData(i) {
    return this.data[i];
  }
}
const source = readFileSync(
  require.resolve("../src/nmc/web/desk-match.js"),
  "utf8",
);
const { matchProbes, gatedEnergy } = runInNewContext(
  source + "\n;({matchProbes,gatedEnergy})",
  { AudioBuffer: TestBuffer },
);

test("loudness matching excludes silence and works for quiet audible signals", () => {
  assert.equal(gatedEnergy([0, 0, 0]), 0);
  assert.equal(gatedEnergy([1, 0.00001, 0, 1]), 1);
  assert.equal(gatedEnergy([1e-12, 0, 1e-12]), 1e-12);
});

test("calibration includes sparse hits and keeps stereo tracks aligned including filter history", () => {
  const rate = 48000,
    length = rate * 40,
    pad = 8192,
    hit = rate * 37 + 4000;
  const power = new Float32Array(Math.ceil(length / 8192));
  power[Math.floor(hit / 8192)] = 0.2;
  const metadata = [
    { id: "a", matchPower: power },
    { id: "b", matchPower: power },
  ];
  const tracks = metadata.map(({ id }, i) => {
    const audio = new TestBuffer({
      length: length + 2 * pad,
      numberOfChannels: 2,
    });
    for (let ch = 0; ch < 2; ch++)
      audio.data[ch][pad + hit] = (i ? 1 : -1) * (ch + 1);
    return { id, audio };
  });
  const probes = matchProbes(
    { rate, duration: length / rate, kernel_frames: pad + 1, tracks: metadata },
    tracks,
    new Map(),
  );
  const data = probes.tracks[0].audio.data;
  const index = data[0].indexOf(-1, pad) - pad;
  assert(
    probes.regions.some(
      ([start, len]) => index >= start && index < start + len,
    ),
  );
  assert.equal(data[1][pad + index], -2);
  assert.equal(probes.tracks[1].audio.data[1][pad + index], 2);
});

test("a large drive jump follows the calibrated dB knots and bypass crossfades to dry", () => {
  const liveSource = readFileSync(
    require.resolve("../src/nmc/web/desk-live.js"),
    "utf8",
  );
  const LiveDesk = runInNewContext(liveSource + "\n; LiveDesk");
  function param(value) {
    return {
      value,
      events: [],
      cancelAndHoldAtTime() {},
      setValueAtTime(v, t) {
        this.events.push(["start", v, t]);
      },
      exponentialRampToValueAtTime(v, t) {
        this.events.push(["ramp", v, t]);
      },
      linearRampToValueAtTime(v, t) {
        this.events.push(["fade", v, t]);
      },
    };
  }
  const w = param(4),
    d = param(0);
  const desk = Object.create(LiveDesk.prototype);
  desk.ctx = { currentTime: 5 };
  desk.toneReturns = [{ wetReturn: { gain: w }, bypassReturn: { gain: d } }];
  desk.lastBypassed = false;
  desk.tone = {
    drive: 0,
    bypassed: false,
    match: {
      headroom: 0.5,
      drive_grid_db: [0, 12, 24],
      trims: { 0: 1, 12: 2, 24: 8 },
    },
  };
  desk.setMonitorMatch(false, 24, 0.2);
  assert.deepEqual(w.events, [
    ["start", 4, 5],
    ["ramp", 1, 5.1],
    ["ramp", 0.5, 5.2],
  ]);
  desk.tone.bypassed = true;
  desk.setMonitorMatch();
  assert.deepEqual(w.events.at(-1), ["fade", 0, 5.025]);
  assert.deepEqual(d.events.at(-1), ["fade", 0.5, 5.025]);
});
