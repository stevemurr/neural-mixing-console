"use strict";
const $ = (id) => document.getElementById(id);
const labels = {
  predicted: "Prediction",
  dry: "Dry sum",
  reference: "Engineer mix",
};
let runs = [],
  run,
  song,
  clip,
  source,
  audio,
  wantsPlay = false;
const fmt = (sec) => {
  sec = Number.isFinite(sec) ? Math.max(0, Math.floor(sec)) : 0;
  return Math.floor(sec / 60) + ":" + String(sec % 60).padStart(2, "0");
};
function node(tag, text) {
  const el = document.createElement(tag);
  el.textContent = text;
  return el;
}
function notice(text) {
  $("notice").textContent = text;
  $("notice").hidden = !text;
}
function stop() {
  wantsPlay = false;
  if (audio) {
    const old = audio;
    audio = null;
    old.pause();
    old.removeAttribute("src");
    old.load();
  }
}
function time() {
  if (!audio) return;
  const fraction = audio.duration ? audio.currentTime / audio.duration : 0;
  $("time").textContent = fmt(audio.currentTime) + " / " + fmt(audio.duration);
  $("seek").value = fraction * 1000;
  $("seek").setAttribute("aria-valuetext", $("time").textContent);
  $("cursor").style.left = 100 * fraction + "%";
  $("play").textContent = wantsPlay ? "Ⅱ Pause" : "▶ Play";
}
function play(a) {
  wantsPlay = true;
  a.play().catch((err) => {
    if (a !== audio || err.name === "AbortError") return;
    wantsPlay = false;
    notice("Playback could not start. Use Play to retry.");
    time();
  });
  time();
}
function chooseSource(key, carry = false) {
  const pos = carry && audio ? audio.currentTime : 0;
  const resume = carry && wantsPlay;
  stop();
  source = key;
  notice("");
  const a = new Audio();
  audio = a;
  const url = clip.sources[key];
  $("wav").href = url;
  $("play").disabled = false;
  $("bars").replaceChildren();
  for (const b of $("variants").children)
    b.setAttribute("aria-pressed", b.dataset.key === key);
  a.loop = $("loop").checked;
  a.preload = "auto";
  a.onloadedmetadata = () => {
    if (audio !== a) return;
    a.currentTime = Math.min(pos, Math.max(0, a.duration - 0.001));
    time();
  };
  a.ontimeupdate = a.ondurationchange = () => {
    if (audio === a) time();
  };
  a.onended = () => {
    if (audio === a) {
      wantsPlay = false;
      time();
    }
  };
  a.onerror = () => {
    if (audio === a) {
      wantsPlay = false;
      notice(
        "This audio could not be loaded. Refresh the run library or open the WAV.",
      );
      time();
    }
  };
  a.src = url;
  if (resume) play(a);
  time();
  fetch(url.replace("/audio/", "/peaks/"))
    .then((r) => r.json())
    .then((peaks) => {
      if (audio !== a) return;
      $("bars").replaceChildren(
        ...peaks.map((p) => {
          const bar = node("i", "");
          bar.style.height = Math.max(2, p * 100) + "%";
          return bar;
        }),
      );
    })
    .catch(() => {});
}
function clipName(c, i) {
  return c.start === null
    ? "Full preview"
    : "Excerpt " + (i + 1) + " · " + fmt(c.start);
}
function chooseClip(index) {
  clip = song.clips[index];
  $("passage").textContent = clipName(clip, index);
  for (const [i, b] of Array.from($("clips").children).entries())
    b.setAttribute("aria-current", i === index);
  const buttons = Object.keys(clip.sources).map((k) => {
    const b = node("button", labels[k]);
    b.dataset.key = k;
    b.onclick = () => {
      if (source !== k) chooseSource(k, true);
    };
    return b;
  });
  $("variants").replaceChildren(...buttons);
  chooseSource(source in clip.sources ? source : Object.keys(clip.sources)[0]);
}
function chooseSong(index) {
  song = run.songs[index];
  $("song").value = index;
  $("count").textContent =
    song.clips.length + (song.clips.length === 1 ? " point" : " points");
  $("clips").replaceChildren(
    ...song.clips.map((c, i) => {
      const b = node("button", "");
      b.className = "clip";
      b.append(node("strong", clipName(c, i)));
      if (c.scores)
        b.append(
          node(
            "small",
            "RMSE · Dry " +
              c.scores.dry.toFixed(2) +
              " / Prediction " +
              c.scores.predicted.toFixed(2) +
              " dB",
          ),
        );
      else
        b.append(
          node(
            "small",
            Object.keys(c.sources)
              .map((k) => labels[k])
              .join(" / "),
          ),
        );
      b.onclick = () => chooseClip(i);
      return b;
    }),
  );
  chooseClip(0);
}
function stamp(r) {
  const d = new Date(r.started);
  return Number.isNaN(+d)
    ? r.id
    : d.toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
}
function chooseRun(r) {
  stop();
  run = r;
  for (const [i, b] of Array.from($("runs").children).entries())
    b.setAttribute("aria-current", runs[i] === r);
  $("workspace").hidden = !r.songs.length;
  notice(
    r.songs.length
      ? ""
      : r.recipe +
          " · " +
          r.id +
          " · " +
          r.status +
          ". This run has no saved audio.",
  );
  if (!r.songs.length) return;
  $("recipe").textContent = r.recipe;
  $("stamp").textContent = r.id + " · " + r.status;
  $("song").replaceChildren(
    ...r.songs.map((s, i) => {
      const o = node("option", s.name);
      o.value = i;
      return o;
    }),
  );
  chooseSong(0);
}
async function refresh() {
  try {
    const response = await fetch("/api/runs", { cache: "no-store" });
    if (!response.ok) throw Error();
    runs = await response.json();
    $("runs").replaceChildren(
      ...runs.map((r) => {
        const b = node("button", "");
        b.className = "run";
        b.title = r.id;
        b.append(
          node("strong", r.recipe),
          node("small", stamp(r)),
          node(
            "em",
            r.songs.length
              ? r.songs.length +
                  (r.songs.length === 1 ? " song" : " songs") +
                  " · " +
                  r.status
              : r.status + " · No audio",
          ),
        );
        b.onclick = () => chooseRun(r);
        return b;
      }),
    );
    if (runs.length)
      chooseRun(
        runs.find((r) => r.id === run?.id && r.recipe === run?.recipe) ||
          runs.find((r) => r.songs.length) ||
          runs[0],
      );
    else {
      stop();
      $("workspace").hidden = true;
      notice("No experiment runs found in this library.");
    }
  } catch {
    notice("The run library could not be loaded. Refresh to retry.");
  }
}
$("refresh").onclick = refresh;
$("song").onchange = (e) => chooseSong(Number(e.target.value));
$("play").onclick = () => {
  if (!audio) return;
  if (wantsPlay) {
    wantsPlay = false;
    audio.pause();
    time();
  } else play(audio);
};
$("seek").oninput = (e) => {
  if (audio && Number.isFinite(audio.duration))
    audio.currentTime = (audio.duration * Number(e.target.value)) / 1000;
};
$("loop").onchange = (e) => {
  if (audio) audio.loop = e.target.checked;
};
refresh();
