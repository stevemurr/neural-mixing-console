"use strict";
// Import worker. Frequency, reference power and support follow BrownDesk in eq/erb_desk.py.
const rate = 48000,
  size = 8192,
  amount = 0.15;
const erb = (hz) => 21.4 * Math.log10(1 + 0.00437 * hz);
const lo = erb(20),
  hi = erb(22050),
  count = Math.round(hi - lo) + 1;
const step = (hi - lo) / (count - 1);
const grid = Array.from({ length: count }, (_, i) => lo + i * step);
const centers = grid.map((v) => (10 ** (v / 21.4) - 1) / 0.00437);
const hann = Float64Array.from(
  { length: size },
  (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (size - 1)),
);

function fft(re, im, inverse = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let span = 2; span <= n; span *= 2) {
    const angle = ((inverse ? 2 : -2) * Math.PI) / span,
      c = Math.cos(angle),
      s = Math.sin(angle);
    for (let i = 0; i < n; i += span) {
      let wr = 1,
        wi = 0;
      for (let j = i; j < i + span / 2; j++) {
        const k = j + span / 2,
          a = wr * re[k] - wi * im[k],
          b = wr * im[k] + wi * re[k];
        re[k] = re[j] - a;
        im[k] = im[j] - b;
        re[j] += a;
        im[j] += b;
        const next = wr * c - wi * s;
        wi = wr * s + wi * c;
        wr = next;
      }
    }
  }
  if (inverse)
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
}

function bands(bins) {
  const powers = new Float64Array(count);
  for (let k = 1; k < bins.length; k++) {
    const hz = (k * rate) / size;
    if (hz < 20 || hz > 22050) continue;
    const pos = (erb(hz) - lo) / step,
      j = Math.floor(pos),
      t = pos - j;
    powers[j] += (1 - t) * bins[k];
    if (j + 1 < count) powers[j + 1] += t * bins[k];
  }
  return powers;
}
const target = bands(
  Float64Array.from(
    { length: size / 2 + 1 },
    (_, k) => 1 / Math.max(20, (k * rate) / size) ** 2,
  ),
);

function curve(power, width) {
  if (!power.some((v) => v > 0)) return Array(count).fill(0);
  const db = Array.from(power, (v) => 10 * Math.log10(Math.max(1e-20, v))),
    max = Math.max(...db);
  const support = db.map((v) => Math.max(0, Math.min(1, (v - max + 45) / 25)));
  const delta = db.map((v, i) => 10 * Math.log10(target[i]) - v);
  const mean =
    delta.reduce((acc, v, i) => acc + v * support[i], 0) /
    support.reduce((a, b) => a + b);
  const sigma = width / (Math.sqrt(8 * Math.log(2)) * step),
    radius = Math.round(4 * sigma);
  return grid.map((_, i) => {
    let sum = 0,
      weight = 0;
    for (let j = -radius; j <= radius; j++) {
      const k = Math.min(count - 1, Math.max(0, i + j)),
        w = support[k] * Math.exp(-0.5 * (j / sigma) ** 2);
      sum += w * (delta[k] - mean);
      weight += w;
    }
    return (
      2 * support[i] * Math.tanh((amount * sum) / (2 * Math.max(weight, 1e-10)))
    );
  });
}

// Shape-preserving cubic interpolation between the ERB control points.
function slopes(x, y) {
  const h = x.slice(1).map((v, i) => v - x[i]),
    m = h.map((v, i) => (y[i + 1] - y[i]) / v);
  function end(a, b, u, v) {
    const d = ((2 * a + b) * u - a * v) / (a + b);
    if (Math.sign(d) !== Math.sign(u)) return 0;
    return Math.sign(u) !== Math.sign(v) && Math.abs(d) > 3 * Math.abs(u)
      ? 3 * u
      : d;
  }
  return x.map((_, i) => {
    if (!i) return end(h[0], h[1], m[0], m[1]);
    if (i === x.length - 1) return end(h.at(-1), h.at(-2), m.at(-1), m.at(-2));
    if (m[i - 1] * m[i] <= 0) return 0;
    const a = 2 * h[i] + h[i - 1],
      b = h[i] + 2 * h[i - 1];
    return (a + b) / (a / m[i - 1] + b / m[i]);
  });
}
function bessel0(x) {
  let sum = 1,
    term = 1;
  for (let k = 1; k < 20; k++) {
    term *= (x / (2 * k)) ** 2;
    sum += term;
  }
  return sum;
}
function kernel(db) {
  const n = 2 * size,
    re = new Float64Array(n),
    im = new Float64Array(n);
  const x = [0, ...grid, erb(rate / 2)],
    y = [0, ...db, 0],
    d = slopes(x, y);
  let j = 0;
  for (let k = 0; k <= n / 2; k++) {
    const v = erb((k * rate) / n);
    while (j + 2 < x.length && v > x[j + 1]) j++;
    const h = x[j + 1] - x[j],
      t = (v - x[j]) / h,
      u = 1 - t;
    const gain =
      u * u * (1 + 2 * t) * y[j] +
      t * t * (3 - 2 * t) * y[j + 1] +
      h * t * u * (u * d[j] - t * d[j + 1]);
    re[k] = 10 ** (gain / 20);
    if (k > 0 && k < n / 2) re[n - k] = re[k];
  }
  fft(re, im, true);
  const normal = bessel0(6);
  return Float32Array.from(
    { length: size + 1 },
    (_, i) =>
      (re[(n + i - size / 2) % n] *
        bessel0(6 * Math.sqrt(1 - ((2 * i) / size - 1) ** 2))) /
      normal,
  );
}

function analyze(frames) {
  const rows = [],
    levels = [];
  for (const channels of frames) {
    const bins = new Float64Array(size / 2 + 1);
    let square = 0;
    for (const channel of channels) {
      const re = new Float64Array(size),
        im = new Float64Array(size);
      for (let j = 0; j < channel.length; j++) {
        re[j] = channel[j] * hann[j];
        square += channel[j] ** 2;
      }
      fft(re, im);
      for (let k = 0; k < bins.length; k++)
        bins[k] += (re[k] ** 2 + im[k] ** 2) / channels.length;
    }
    const power = bands(bins);
    rows.push({ power, energy: bins.reduce((a, b) => a + b) });
    levels.push(Math.sqrt(square / (channels[0].length * channels.length)));
  }
  const max = Math.max(...rows.map((r) => r.energy));
  const active = rows.filter((r) => r.energy > 0 && r.energy >= max * 1e-4);
  const power = grid.map(
    (_, i) =>
      active.reduce((sum, r) => sum + r.power[i], 0) /
      Math.max(1, active.length),
  );
  const curves = {},
    kernels = {};
  for (const w of [1, 3, 6]) {
    const c = curve(power, w);
    curves[w] = c;
    kernels[w] = [kernel(c), kernel(c.map((v) => v / amount))];
  }
  const loud = Math.max(...levels);
  const ordered = levels
    .filter((v) => v > Math.max(loud * 0.032, 1e-8))
    .sort((a, b) => a - b);
  const nominal = ordered.length
    ? Math.min(
        8,
        Math.max(0.25, 0.3 / ordered[Math.floor(0.9 * (ordered.length - 1))]),
      )
    : 1;
  return { centers, curves, kernels, nominal };
}
self.onmessage = ({ data }) => {
  const result = analyze(data);
  self.postMessage(
    result,
    Object.values(result.kernels)
      .flat()
      .map((k) => k.buffer),
  );
};
