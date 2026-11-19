"use strict";
const el = (id) => document.getElementById(id);
const colors = [
  "#88d4b3",
  "#87bbd8",
  "#bba3e0",
  "#e1ac83",
  "#c7cc81",
  "#dc97ad",
  "#7fc8c8",
];
let sessions = [],
  session,
  selected,
  bus = "mix",
  mode = "off",
  previousWidth = "3";
let ctx,
  monitor,
  meters,
  buffers = {},
  nodes = [],
  offset = 0,
  origin = 0;
let tone = null,
  importRequest,
  localSong;
let balanceKey = null,
  colorKey = null;
let amount = 15,
  livePlayer,
  liveTracks = [],
  liveCache = new Map();
let playing = false,
  looping = true,
  loading,
  error = false;

// Manual mix settings belong to a song and survive graph rebuilds and session switches.
const songMixers = new Map();
let mixer = new Map(),
  stripViews = new Map(),
  mixerView = true,
  meterTime = 0;

function setView(showMixer) {
  mixerView = showMixer;
  el("mixer").hidden = !showMixer;
  el("arrangement").hidden = showMixer;
  el("mixerView").setAttribute("aria-pressed", String(showMixer));
  el("arrangeView").setAttribute("aria-pressed", String(!showMixer));
  if (showMixer) updateStripNavigation();
  else redraw();
}

function updateStripNavigation() {
  const bank = el("strips");
  el("stripsBack").disabled = bank.scrollLeft < 1;
  el("stripsNext").disabled =
    bank.scrollLeft + bank.clientWidth >= bank.scrollWidth - 1;
}
function scrollStrips(direction) {
  const bank = el("strips");
  const first = bank.children[0],
    next = bank.children[1];
  if (!next) return;
  const step = next.offsetLeft - first.offsetLeft;
  bank.scrollBy({
    left: direction * step * Math.max(1, Math.floor(bank.clientWidth / step)),
    behavior: "smooth",
  });
}

function buildMixer() {
  stripViews = new Map();
  el("deskViews").hidden = !session.live_eq;
  const strips = session.live_eq
    ? session.tracks.map((track, i) => {
        const strip =
          el("stripTemplate").content.firstElementChild.cloneNode(true);
        strip.style.setProperty("--color", colors[i % colors.length]);
        const index = String(i + 1).padStart(2, "0");
        const label = index + " · " + track.label;
        strip.setAttribute("aria-label", label);
        const name = strip.querySelector(".strip-name");
        name.querySelector("small").textContent = "CH " + index;
        name.querySelector("strong").textContent = track.label;
        name.title = track.name;
        name.onclick = () => inspect(track);
        const pan = strip.querySelector(".strip-pan input"),
          panValue = strip.querySelector(".strip-pan output");
        const fader = strip.querySelector(".strip-fader"),
          readout = strip.querySelector(".strip-db");
        const mute = strip.querySelector(".mute"),
          soloButton = strip.querySelector(".strip-solo");
        pan.setAttribute("aria-label", "Pan " + label);
        pan.title =
          track.channels === 1
            ? "Equal-power pan · double-click to center"
            : "Stereo balance · double-click to center";
        fader.setAttribute("aria-label", "Fader " + label);
        fader.title = "Channel gain · double-click for unity";
        mute.setAttribute("aria-label", "Mute " + label);
        soloButton.setAttribute("aria-label", "Solo " + label);
        const state = mixer.get(track.id);
        function display() {
          pan.value = Math.round(state.pan * 100);
          panValue.textContent =
            state.pan === 0
              ? "C"
              : (state.pan < 0 ? "L" : "R") + Math.abs(Number(pan.value));
          pan.setAttribute("aria-valuetext", panValue.textContent);
          fader.value = state.db;
          readout.textContent =
            state.db <= -60
              ? "−∞ dB"
              : (state.db > 0 ? "+" : "") + state.db.toFixed(1) + " dB";
          fader.setAttribute("aria-valuetext", readout.textContent);
          mute.setAttribute("aria-pressed", String(state.muted));
          strip.classList.toggle("is-muted", state.muted);
        }
        function change(edit) {
          Object.assign(state, edit);
          livePlayer?.strips.get(track.id)?.set(state);
          display();
          requestLevelMatch();
          listenLabel();
        }
        fader.oninput = () => change({ db: Number(fader.value) });
        fader.ondblclick = () => change({ db: 0 });
        pan.oninput = () => change({ pan: Number(pan.value) / 100 });
        pan.ondblclick = () => change({ pan: 0 });
        mute.onclick = () => change({ muted: !state.muted });
        soloButton.onclick = () => solo(track);
        display();
        stripViews.set(track.id, {
          strip,
          soloButton,
          bars: strip.querySelectorAll(".strip-meters i"),
          peak: strip.querySelector(".strip-peak"),
        });
        return strip;
      })
    : [];
  el("strips").replaceChildren(...strips);
  el("strips").scrollLeft = 0;
  setView(!!session.live_eq);
}

function drawStripMeters() {
  for (const [id, view] of stripViews) {
    const strip = livePlayer?.strips.get(id);
    let top = -Infinity;
    view.bars.forEach((bar, side) => {
      const db = 20 * Math.log10(strip ? strip.peak(side) : 0);
      top = Math.max(top, db);
      bar.style.setProperty(
        "--signal",
        Math.min(100, Math.max(0, ((db + 60) * 100) / 60)) + "%",
      );
    });
    view.peak.textContent = (top < -60 ? "−∞" : top.toFixed(1)) + " dBFS";
    view.peak.classList.toggle("over", top >= 0);
  }
}
function clock(t) {
  const tenths = Math.floor(Math.max(0, t) * 10 + 1e-6);
  return (
    String(Math.floor(tenths / 600)).padStart(2, "0") +
    ":" +
    String(Math.floor(tenths / 10) % 60).padStart(2, "0") +
    "." +
    (tenths % 10)
  );
}
function status(text) {
  el("status").textContent = text;
}
function button(label, action, cls) {
  const node = document.createElement("button");
  node.type = "button";
  node.textContent = label;
  node.onclick = action;
  node.className = cls;
  return node;
}
function graph(canvas, draw) {
  const { width: w, height: h } = canvas.getBoundingClientRect();
  if (!w || !h) return;
  const dpi = devicePixelRatio || 1;
  canvas.width = Math.round(w * dpi);
  canvas.height = Math.round(h * dpi);
  const g = canvas.getContext("2d");
  g.scale(dpi, dpi);
  draw(g, w, h);
}
function waveform(canvas, peaks, color) {
  graph(canvas, (g, w, h) => {
    const center = h / 2;
    g.fillStyle = color;
    g.globalAlpha = 0.82;
    peaks.forEach((p, i) => {
      const size = Math.max(0.65, p ** 0.68 * (center - 3));
      g.fillRect(
        (i * w) / peaks.length,
        center - size,
        Math.max(1, w / peaks.length - 0.6),
        size * 2,
      );
    });
  });
}
function response() {
  if (!selected) return;
  graph(el("curve"), (g, w, h) => {
    const left = 25,
      right = w - 5,
      top = 10,
      base = h - 18;
    const x = (f) =>
      left + (Math.log(f / 20) / Math.log(22050 / 20)) * (right - left);
    let gains =
      (!tone?.bypassed && selected.curves[mode]) ||
      session.centers.map(() => 0);
    if (session.live_eq && mode !== "off") {
      const [a, b, c] = amountWeights(amount, Math.round(100 * session.amount));
      gains = gains.map(
        (db) =>
          20 *
          Math.log10(
            a + b * 10 ** (db / 20) + c * 10 ** (db / session.amount / 20),
          ),
      );
    }
    const low = Math.min(...gains),
      high = Math.max(...gains);
    const signed = (db) => (db > 0 ? "+" : "") + db.toFixed(1) + " dB";
    el("trackRange").textContent = signed(low) + " to " + signed(high);
    const scale = Math.max(2, 2 * Math.ceil(Math.max(-low, high) / 2));
    const y = (db) =>
      top + ((1.125 * scale - db) / (2.25 * scale)) * (base - top);
    g.font = "10px monospace";
    g.lineWidth = 1;
    for (const db of [scale, 0, -scale]) {
      g.strokeStyle = db === 0 ? "#626d68" : "#343c3c";
      g.beginPath();
      g.moveTo(left, y(db));
      g.lineTo(right, y(db));
      g.stroke();
      g.fillStyle = "#b1bdb8";
      g.fillText((db > 0 ? "+" : "") + db, 2, y(db) + 3);
    }
    for (const [freq, label] of [
      [100, "100"],
      [1000, "1k"],
      [10000, "10k"],
    ]) {
      g.strokeStyle = "#323a39";
      g.beginPath();
      g.moveTo(x(freq), top);
      g.lineTo(x(freq), base);
      g.stroke();
      g.fillStyle = "#a8b3af";
      g.fillText(label, x(freq) - 6, h - 3);
    }
    g.beginPath();
    session.centers.forEach((f, i) =>
      i ? g.lineTo(x(f), y(gains[i])) : g.moveTo(x(f), y(gains[i])),
    );
    g.strokeStyle = "#9be4c0";
    g.lineWidth = 2;
    g.stroke();
  });
}
function redraw() {
  if (!session) return;
  document
    .querySelectorAll(".wave canvas")
    .forEach((c, i) =>
      waveform(c, session.tracks[i].peaks, colors[i % colors.length]),
    );
  response();
}
function ensureContext() {
  if (ctx) return;
  ctx = new AudioContext({ sampleRate: 48000 });
  monitor = ctx.createGain();
  monitor.gain.value = 10 ** (Number(el("volume").value) / 20);
  monitor.connect(ctx.destination);
  const split = ctx.createChannelSplitter(2);
  monitor.connect(split);
  meters = [0, 1].map((i) => {
    const analyzer = ctx.createAnalyser();
    analyzer.fftSize = 1024;
    split.connect(analyzer, i);
    return { analyzer, data: new Float32Array(analyzer.fftSize) };
  });
}
function position() {
  if (!playing) return offset;
  const pos = offset + Math.max(0, ctx.currentTime - origin);
  return looping ? pos % session.duration : Math.min(pos, session.duration);
}
function stop() {
  offset = position();
  playing = false;
  for (const { source, gain } of nodes) {
    source.stop();
    source.disconnect();
    gain.disconnect();
  }
  livePlayer?.stop();
  livePlayer = null;
  nodes = [];
  el("playtext").textContent = "Play";
  el("play").setAttribute("aria-label", "Play");
  el("play").firstElementChild.textContent = "▶";
}
function start() {
  if ((!buffers.off && !liveTracks.length) || playing) return;
  if (offset >= session.duration - 0.001) offset = 0;
  if (session.live_eq) {
    const calibration =
      bus === "mix" ? session : session.tracks.find((t) => t.id === bus);
    livePlayer = new LiveDesk(
      ctx,
      monitor,
      session,
      liveTracks,
      calibration,
      offset,
      looping,
      mode,
      amount,
      previousWidth,
      bus === "mix" ? balanceKey : null,
      colorKey,
      mixer,
      tone,
    );
    origin = livePlayer.origin;
  } else {
    origin = ctx.currentTime + 0.015;
    for (const [key, buffer] of Object.entries(buffers)) {
      const source = ctx.createBufferSource(),
        gain = ctx.createGain();
      source.buffer = buffer;
      source.loop = looping;
      source.loopEnd = session.duration;
      gain.gain.value = Number(key === mode);
      source.connect(gain);
      gain.connect(monitor);
      source.start(origin, offset);
      nodes.push({ key, source, gain });
    }
  }
  playing = true;
  el("playtext").textContent = "Pause";
  el("play").setAttribute("aria-label", "Pause");
  el("play").firstElementChild.textContent = "Ⅱ";
  status("Playing");
}
async function playPause() {
  if (el("play").disabled) return;
  if (error) return loadBus();
  if (playing) {
    stop();
    status("Paused");
    return;
  }
  try {
    await ctx.resume();
    start();
  } catch (e) {
    status("Playback error · " + e.message);
  }
}
function seek(t) {
  if (!session) return;
  const resume = playing;
  stop();
  offset = Math.max(0, Math.min(t, session.duration));
  if (resume) start();
}
function seekPointer(event) {
  const rect = event.currentTarget.getBoundingClientRect();
  seek(((event.clientX - rect.left) / rect.width) * session.duration);
}
function setMode(choice) {
  if (balanceKey || colorKey) choice = "1";
  mode = choice;
  if (choice !== "off") previousWidth = choice;
  for (const { key, gain } of nodes)
    gain.gain.setTargetAtTime(Number(key === mode), ctx.currentTime, 0.006);
  el("bypass").setAttribute("aria-pressed", String(mode === "off"));
  document
    .querySelectorAll("#widths button")
    .forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.width === mode)),
    );
  el("curveLabel").textContent =
    mode === "off" || tone?.bypassed
      ? "Unity gain · bypass"
      : mode + " ERB · applied EQ (dB)";
  livePlayer?.set(mode, amount, previousWidth);
  requestLevelMatch();
  updateAmountDisplay();
  response();
}
function updateAmountDisplay() {
  const normal = session.amount * 100;
  el("amountKnob").disabled = !session.live_eq || !!balanceKey || !!colorKey;
  el("amountKnob").value = session.live_eq ? amount : normal;
  const p = Number(el("amountKnob").value);
  el("amount").textContent = p + "%";
  el("amountKnob").setAttribute("aria-valuetext", p + " percent");
  el("amountKnob").parentElement.style.setProperty("--turn", p * 2.7 + "deg");
  // Show the maximum possible correction under this interpolation law.
  const [a, b, c] = amountWeights(p, Math.round(normal));
  const limit =
    20 *
    Math.log10(
      a +
        b * 10 ** (session.limit_db / 20) +
        c * 10 ** (session.limit_db / session.amount / 20),
    );
  el("limit").textContent =
    mode === "off" || tone?.bypassed
      ? "Bypassed"
      : limit.toFixed(1) + " dB max";
}
function inspect(track) {
  selected = track;
  el("channelName").textContent = track.label;
  el("channelInfo").textContent =
    (track.channels === 1 ? "Mono" : "Stereo") + " input";
  el("channelInfo").title = track.name;
  const comparison = session.balance;
  if (balanceKey) {
    const group = comparison.membership[track.id];
    const db = comparison.modes[balanceKey].db[group];
    el("faderInfo").textContent =
      group + " fader · " + (db > 0 ? "+" : "") + db.toFixed(1) + " dB";
  } else {
    el("faderInfo").textContent = "";
  }
  [...el("lanes").children].forEach((lane, i) => {
    lane.classList.toggle("selected", session.tracks[i] === track);
    lane
      .querySelector(".solo")
      .setAttribute("aria-pressed", String(bus === session.tracks[i].id));
  });
  for (const [id, view] of stripViews) {
    view.strip.classList.toggle("selected", id === track.id);
    view.soloButton.setAttribute("aria-pressed", String(id === bus));
  }
  el("inspectSolo").setAttribute("aria-pressed", String(track.id === bus));
  response();
}
function solo(track) {
  bus = bus === track.id ? "mix" : track.id;
  inspect(track);
  loadBus();
}
const matchCache = new WeakMap();
let matchTask;
let matchWork = Promise.resolve();
function cancelMatch() {
  if (matchTask) {
    clearTimeout(matchTask.timer);
    matchTask.controller.abort();
    matchTask = null;
  }
}
function requestLevelMatch(delay = 140) {
  if (!tone || !liveTracks.length || !session) return;
  const settings = liveTracks.map((t) => [t.id, mixer.get(t.id)]);
  const key = JSON.stringify([mode, mode === "off" ? 0 : amount, settings]);
  if (matchTask?.key === key && matchTask.tone === tone) return;
  cancelMatch();
  if (!matchCache.has(session)) matchCache.set(session, new Map());
  const cache = matchCache.get(session);
  const apply = (match) => {
    tone.match = match;
    tone.matchKey = key;
    tone.matchStatus = match.silent
      ? "No measurable level match"
      : "Loudness matched";
    livePlayer?.setMonitorMatch();
    el("play").disabled = false;
    listenLabel();
    if (!playing) status("Ready");
    if (tone.resumeAfterMatch && ctx.state === "running") {
      tone.resumeAfterMatch = false;
      start();
    }
  };
  if (tone.matchKey === key && tone.match) {
    tone.matchStatus = tone.match.silent
      ? "No measurable level match"
      : "Loudness matched";
    listenLabel();
    return;
  }
  if (cache.has(key)) {
    apply(cache.get(key));
    return;
  }
  const task = { key, tone, controller: new AbortController() };
  matchTask = task;
  tone.matchStatus = "Measuring level match…";
  el("play").disabled = !playing && !tone.match;
  if (el("play").disabled) status("Matching Desk to bypass…");
  listenLabel();
  const state = new Map(settings.map(([id, s]) => [id, { ...s }]));
  task.timer = setTimeout(() => {
    // An offline render cannot be interrupted mid-block of work. Serialize
    // replacements so dragging a fader never spawns a pile of render graphs.
    matchWork = matchWork.then(async () => {
      if (task.controller.signal.aborted) return;
      try {
        const match = await calibrateDesk(
          session,
          liveTracks,
          state,
          mode,
          amount,
          previousWidth,
          task.controller.signal,
          (index, total) => {
            if (task.controller.signal.aborted) return;
            tone.matchStatus = "Measuring level match · " + index + "/" + total;
            el("matchStatus").textContent = tone.matchStatus;
          },
        );
        if (task.controller.signal.aborted) return;
        if (cache.size >= 12) cache.delete(cache.keys().next().value);
        cache.set(key, match);
        apply(match);
      } catch (e) {
        if (task.controller.signal.aborted) return;
        tone.matchStatus = "Level match unavailable";
        listenLabel();
        status("Level matching failed · " + e.message);
        el("play").disabled = false;
      } finally {
        if (matchTask === task) matchTask = null;
      }
    });
  }, delay);
}

function listenLabel() {
  const target =
    bus === "mix"
      ? balanceKey
        ? session.balance
        : session
      : session.tracks.find((t) => t.id === bus);
  const lufs = colorKey ? target.saturation.lufs : target.lufs;
  const edited = [...mixer].some(
    ([id, state]) =>
      (bus === "mix" || id === bus) &&
      (state.db !== 0 || state.pan !== 0 || state.muted),
  );
  el("matchStatus").textContent = tone
    ? tone.matchStatus || "Measuring level match…"
    : edited
      ? "Manual mix"
      : "Loudness matched";
  el("listenBus").textContent =
    (bus === "mix"
      ? "Mix bus"
      : (balanceKey ? "Pre-fader solo · " : "Solo · ") + target.label) +
    (tone
      ? tone.bypassed
        ? " · desk bypassed"
        : " · console sound"
      : edited
        ? " · manual mix"
        : lufs == null
          ? " · silent"
          : " · " + lufs.toFixed(1) + " LUFS");
}
async function loadBus(resume = playing || !!tone?.resumeAfterMatch) {
  cancelMatch();
  if (tone) {
    delete tone.match;
    delete tone.matchKey;
    delete tone.matchStatus;
    tone.resumeAfterMatch = resume;
  }
  loading?.abort();
  const request = new AbortController();
  loading = request;
  stop();
  buffers = {};
  liveTracks = [];
  error = false;
  el("play").disabled = true;
  const target =
    bus === "mix" ? session : session.tracks.find((t) => t.id === bus);
  listenLabel();
  status("Loading matched audio…");
  try {
    ensureContext();
    if (session.live_eq) {
      const tracks = bus === "mix" ? session.tracks : [target];
      const loaded = [];
      for (let i = 0; i < tracks.length; i += 4) {
        const chunk = await Promise.all(
          tracks.slice(i, i + 4).map(async (t) => {
            if (!liveCache.has(t.id)) {
              const data = await loadLiveTrack(ctx, session, t, request.signal);
              if (request.signal.aborted)
                throw new DOMException("Aborted", "AbortError");
              liveCache.set(t.id, data);
            }
            return liveCache.get(t.id);
          }),
        );
        loaded.push(...chunk);
      }
      if (request.signal.aborted) return;
      liveTracks = loaded;
    } else {
      const entries = await Promise.all(
        ["off", ...session.widths.map(String)].map(async (key) => {
          const url =
            "/desk/audio/" + session.id + "/" + bus + "/" + key + ".flac";
          const r = await fetch(url, { signal: request.signal });
          if (!r.ok) throw Error("Audio unavailable (" + r.status + ")");
          return [key, await ctx.decodeAudioData(await r.arrayBuffer())];
        }),
      );
      if (request.signal.aborted) return;
      buffers = Object.fromEntries(entries);
    }
    el("play").disabled = false;
    status("Ready");
    if (!tone && resume && ctx.state === "running") start();
    requestLevelMatch(0);
  } catch (e) {
    if (request.signal.aborted) return;
    error = true;
    el("play").disabled = false;
    el("playtext").textContent = "Retry";
    status("Audio could not load · " + e.message);
  }
}
const balanceModes = [
  ["unity", "Unity faders"],
  ["active", "Active loudness"],
  ["vocal", "Vocal-anchored ERB"],
  [null, "Explore EQ"],
];

const colorModes = [
  ["clean", "Clean EQ"],
  ["line", "Nominal"],
  ["driven", "+6 dB"],
  ["hot", "+12 dB"],
];
let driveDb = 0;
function showDrive() {
  const knob = el("driveKnob");
  knob.value = driveDb;
  const label = (driveDb > 0 ? "+" : "") + driveDb.toFixed(1) + " dB";
  el("driveReadout").textContent = label;
  knob.setAttribute("aria-valuetext", label);
  knob.parentElement.style.setProperty(
    "--turn",
    (270 * (driveDb - Number(knob.min))) /
      (Number(knob.max) - Number(knob.min)) +
      "deg",
  );
}
function changeDrive(value) {
  const grid = (tone?.spec || session?.saturation)?.drive_grid_db;
  if (!grid) return;
  driveDb =
    Math.round(10 * Math.max(grid[0], Math.min(grid.at(-1), value))) / 10;
  showDrive();
  if (tone) {
    tone.drive = driveDb;
    livePlayer?.updateTone();
    return;
  }
  const key = String(driveDb);
  if (colorKey === null || colorKey === "clean") selectAudition(null, key);
  else {
    colorKey = key;
    livePlayer?.setSaturation(key);
    livePlayer?.set(mode, amount, previousWidth);
  }
}
function selectAudition(key, color = null) {
  colorKey = color;
  balanceKey = key;
  const locked = key !== null || color !== null;
  if (locked) amount = 100;
  el("bypass").disabled = locked;
  document
    .querySelectorAll("#widths button")
    .forEach((b) => (b.disabled = locked));
  document
    .querySelectorAll("#balances button")
    .forEach((b) =>
      b.setAttribute(
        "aria-pressed",
        String(!color && b.dataset.key === (key || "")),
      ),
    );
  document
    .querySelectorAll("#colors button")
    .forEach((b) =>
      b.setAttribute(
        "aria-pressed",
        String(
          b.dataset.key === color ||
            (b.dataset.key === "drive" && color != null && color !== "clean"),
        ),
      ),
    );
  el("colorInfo").textContent = tone
    ? "Per channel · after EQ"
    : color
      ? "Post-EQ · 1 ERB / 100%"
      : "";
  el("balanceInfo").textContent = key
    ? "1 ERB · 100% · Compression off" +
      (key === "vocal"
        ? " · Anchor: " +
          session.balance.labels[session.balance.anchor] +
          " (" +
          session.balance.anchor +
          ")"
        : "")
    : "";
  livePlayer?.setSaturation(color);
  livePlayer?.setBalance(bus === "mix" ? key : null);
  setMode(locked ? "1" : mode);
  inspect(selected);
  listenLabel();
}

function showToneControls() {
  el("tonePanel").hidden = !session.live_eq;
  el("comparison").hidden = !!session.local;
  el("toneOn").setAttribute("aria-pressed", String(!!tone && !tone.bypassed));
  el("toneOff").setAttribute("aria-pressed", String(!!tone?.bypassed));
  el("comparison").setAttribute("aria-pressed", String(!tone));
  el("colorPanel").hidden = !tone && !session.saturation;
  el("colors").hidden = !!tone;
  el("balancePanel").hidden =
    !!tone || (!session.balance && !session.saturation);
  el("driveControl").hidden = !tone && !session.saturation?.drive_grid_db;
}
function switchDesk(choice) {
  if (!session?.live_eq) return;
  if (tone && choice !== "comparison") {
    tone.bypassed = choice === "off";
    livePlayer?.updateTone();
    setMode(mode);
    showToneControls();
    listenLabel();
    return;
  }
  const resume = playing || !!tone?.resumeAfterMatch;
  cancelMatch();
  stop();
  tone =
    choice === "comparison"
      ? null
      : {
          spec: consoleSpec(session.rate),
          resumeAfterMatch: resume,
          drive: driveDb,
          bypassed: choice === "off",
        };
  if (tone) {
    mode = previousWidth = "1";
    amount = 100;
  }
  if (!tone && liveTracks.length) el("play").disabled = false;
  showToneControls();
  selectAudition(
    !tone && session.balance && !session.saturation ? "unity" : null,
    !tone && session.saturation ? "clean" : null,
  );
  if (resume && !tone) start();
}

function openSession(s) {
  importRequest?.abort();
  cancelMatch();
  stop();
  liveTracks = [];
  offset = 0;
  if ((balanceKey || colorKey) && !s.balance && !s.saturation) {
    mode = "off";
    previousWidth = "3";
    amount = 15;
  }
  balanceKey = null;
  colorKey = null;
  session = s;
  tone = s.live_eq
    ? { spec: consoleSpec(s.rate), drive: 0, bypassed: false }
    : null;
  if (tone) {
    mode = previousWidth = "1";
    amount = 100;
  } else {
    mode = "off";
    previousWidth = String(s.widths[0]);
    amount = Math.round(100 * s.amount);
  }
  liveCache = s.cache || new Map();
  selected = s.tracks[0];
  bus = "mix";
  history.replaceState(null, "", "#" + s.id);
  document
    .querySelectorAll(".session")
    .forEach((b) =>
      b.setAttribute("aria-current", String(b.dataset.id === s.id)),
    );
  el("song").textContent = s.title;
  el("artist").textContent = s.artist;
  el("duration").textContent = "/ " + clock(s.duration);
  el("trackCount").textContent = s.tracks.length + " tracks";
  el("seek").max = s.duration;
  el("amount").textContent = Math.round(100 * s.amount) + "%";
  el("limit").textContent = "±" + s.limit_db + " dB";
  el("format").textContent =
    s.rate / 1000 + (s.local ? " kHz · Stereo" : " kHz · 24-bit · Stereo");
  el("excerpt").textContent =
    clock(s.start) + " – " + clock(s.start + s.duration);
  el("widths").replaceChildren(
    ...s.widths.map((n) => {
      const b = button(n + " ERB", () => setMode(String(n)), "");
      b.dataset.width = n;
      return b;
    }),
  );
  el("ruler").replaceChildren(
    ...[0, 0.25, 0.5, 0.75, 1].map((f) => {
      const tick = document.createElement("span");
      tick.textContent = Math.round(s.duration * f * 10) / 10 + "s";
      return tick;
    }),
  );
  el("ruler").onclick = seekPointer;
  el("lanes").replaceChildren(
    ...s.tracks.map((t, i) => {
      const lane = document.createElement("div");
      lane.className = "lane";
      lane.style.setProperty("--color", colors[i % colors.length]);
      const track = document.createElement("div");
      track.className = "track";
      const num = document.createElement("span");
      num.className = "chNo";
      num.textContent = String(i + 1).padStart(2, "0");
      const name = button(t.label, () => inspect(t), "chName");
      name.title = t.name;
      const sb = button("S", () => solo(t), "solo");
      sb.setAttribute("aria-label", "Solo " + t.label);
      track.append(num, name, sb);
      const wave = document.createElement("div");
      wave.className = "wave";
      wave.title = "Seek and inspect " + t.label;
      wave.onclick = (e) => {
        inspect(t);
        seekPointer(e);
      };
      wave.append(document.createElement("canvas"));
      lane.append(track, wave);
      return lane;
    }),
  );
  el("lanes").scrollTop = 0;
  el("colorPanel").hidden = !s.saturation;
  const grid = tone?.spec.drive_grid_db || s.saturation?.drive_grid_db;
  el("driveControl").hidden = !grid;
  if (grid) {
    driveDb = 0;
    el("driveKnob").min = grid[0];
    el("driveKnob").max = grid.at(-1);
    showDrive();
  }
  const modes = grid
    ? [
        ["clean", "Clean EQ"],
        ["drive", "Saturation"],
      ]
    : colorModes;
  el("colors").replaceChildren(
    ...(s.saturation ? modes : []).map(([key, label]) => {
      const b = button(
        label,
        () => selectAudition(null, key === "drive" ? String(driveDb) : key),
        "balanceChoice",
      );
      b.dataset.key = key;
      return b;
    }),
  );
  el("balancePanel").hidden = !s.balance && !s.saturation;
  el("balancePanel").open = !!s.balance && !s.saturation;
  el("balances").replaceChildren(
    ...(s.balance
      ? balanceModes
      : balanceModes.filter(([key]) => key === null)
    ).map(([key, label]) => {
      const b = button(label, () => selectAudition(key), "balanceChoice");
      b.dataset.key = key || "";
      return b;
    }),
  );
  if (!songMixers.has(s.id))
    songMixers.set(
      s.id,
      new Map(s.tracks.map((t) => [t.id, { db: 0, pan: 0, muted: false }])),
    );
  mixer = songMixers.get(s.id);
  buildMixer();
  showToneControls();
  selectAudition(
    !tone && s.balance && !s.saturation ? "unity" : null,
    !tone && s.saturation ? "clean" : null,
  );
  redraw();
  loadBus(false);
}
function tick(timestamp = 0) {
  if (mixerView && timestamp - meterTime >= 45) {
    drawStripMeters();
    meterTime = timestamp;
  }
  if (session) {
    const t = position();
    el("time").textContent = clock(t);
    el("seek").value = t;
    document.documentElement.style.setProperty(
      "--position",
      (100 * t) / session.duration + "%",
    );
    if (playing && !looping && t >= session.duration) {
      stop();
      status("End of excerpt");
    }
  }
  meters?.forEach(({ analyzer, data }, i) => {
    let peak = 0;
    if (playing) {
      analyzer.getFloatTimeDomainData(data);
      for (const v of data) peak = Math.max(peak, Math.abs(v));
    }
    el(i ? "meterR" : "meterL").style.width =
      Math.max(
        0,
        Math.min(100, ((60 + 20 * Math.log10(peak || 1e-9)) / 60) * 100),
      ) + "%";
  });
  requestAnimationFrame(tick);
}
function changeAmount(value) {
  if (!session?.live_eq || balanceKey || colorKey) return;
  amount = Math.round(Math.max(0, Math.min(100, value)));
  setMode(mode === "off" ? previousWidth : mode);
}
function bindDial(knob, change, reset, sensitivity) {
  knob.oninput = () => change(Number(knob.value));
  knob.ondblclick = () => change(reset);
  knob.onpointerdown = (e) => {
    if (knob.disabled || e.button !== 0) return;
    e.preventDefault();
    knob.focus();
    knob.setPointerCapture(e.pointerId);
    const x = e.clientX,
      y = e.clientY,
      initial = Number(knob.value);
    knob.onpointermove = (move) =>
      change(initial + (move.clientX - x + y - move.clientY) * sensitivity);
  };
  knob.onlostpointercapture = () => {
    knob.onpointermove = null;
  };
}
bindDial(el("amountKnob"), changeAmount, 15, 0.7);
bindDial(el("driveKnob"), changeDrive, 0, 0.25);
el("toneOn").onclick = () => switchDesk("on");
el("toneOff").onclick = () => switchDesk("off");
el("comparison").onclick = () => switchDesk("comparison");
el("openFolder").onclick = () => el("folderPicker").click();
el("openFiles").onclick = () => el("filePicker").click();
el("localSong").onclick = () => localSong && openSession(localSong);
async function openLocalFiles(e) {
  const files = Array.from(e.target.files);
  e.target.value = "";
  if (!files.length) return;
  importRequest?.abort();
  const request = new AbortController();
  importRequest = request;
  try {
    ensureContext();
    const song = await importMultitracks(ctx, files, request.signal, status);
    if (request.signal.aborted) return;
    localSong = song;
    songMixers.delete(song.id);
    el("localSong").hidden = false;
    el("localSong").dataset.id = song.id;
    el("localSong").textContent = song.title;
    openSession(song);
  } catch (err) {
    if (!request.signal.aborted)
      status("Could not open tracks · " + err.message);
  }
}
el("folderPicker").onchange = openLocalFiles;
el("filePicker").onchange = openLocalFiles;
el("stripsBack").onclick = () => scrollStrips(-1);
el("stripsNext").onclick = () => scrollStrips(1);
el("strips").onscroll = updateStripNavigation;
window.addEventListener("resize", updateStripNavigation);
el("mixerView").onclick = () => setView(true);
el("arrangeView").onclick = () => setView(false);
el("play").onclick = playPause;
el("rewind").onclick = () => seek(0);
el("loop").onclick = () => {
  const pos = position();
  looping = !looping;
  if (playing) {
    stop();
    offset = pos;
    start();
  }
  el("loop").setAttribute("aria-pressed", String(looping));
};
el("volume").oninput = (e) => {
  const db = Number(e.target.value);
  el("volread").textContent = db + " dB";
  if (monitor)
    monitor.gain.setTargetAtTime(10 ** (db / 20), ctx.currentTime, 0.01);
};
el("bypass").onclick = () => setMode(mode === "off" ? previousWidth : "off");
el("inspectSolo").onclick = () => selected && solo(selected);
el("seek").oninput = (e) => seek(Number(e.target.value));
document.addEventListener("keydown", (e) => {
  if (
    !session ||
    e.ctrlKey ||
    e.altKey ||
    e.metaKey ||
    e.target.matches("input,select") ||
    e.repeat
  )
    return;
  if (e.code === "Space") {
    if (e.target.closest("#mixer button, #deskViews button")) return;
    e.preventDefault();
    playPause();
  } else if (e.key === "Home") {
    e.preventDefault();
    seek(0);
  } else if (["0", "1", "2", "3"].includes(e.key))
    setMode(["off", ...session.widths.map(String)][Number(e.key)]);
  else if (e.key.toLowerCase() === "l") el("loop").click();
  else if (e.key.toLowerCase() === "s") solo(selected);
});
window.addEventListener("resize", redraw);
tick();
(async () => {
  try {
    const r = await fetch("/api/desk");
    if (!r.ok) throw Error(r.status);
    sessions = await r.json();
    el("count").textContent = sessions.length;
    if (!sessions.length) {
      status("No prepared desk sessions");
      return;
    }
    el("sessions").replaceChildren(
      ...sessions.map((s) => {
        const b = button("", () => openSession(s), "session");
        b.dataset.id = s.id;
        const icon = document.createElement("span");
        icon.className = "sessIcon";
        icon.textContent = "≋";
        icon.setAttribute("aria-hidden", "true");
        const group = document.createElement("span"),
          title = document.createElement("strong"),
          artist = document.createElement("small");
        title.textContent = s.title;
        artist.textContent = s.artist;
        group.append(title, artist);
        b.append(icon, group);
        return b;
      }),
    );
    openSession(
      sessions.find((s) => s.id === location.hash.slice(1)) ||
        sessions.find((s) => s.title === "Struttin") ||
        sessions[0],
    );
  } catch (e) {
    status("Could not open sessions · " + e.message);
  }
})();
