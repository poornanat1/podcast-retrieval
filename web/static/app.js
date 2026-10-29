// PodFind web app: typed and spoken search with filters, the understood
// query shown back as editable chips, an editable transcript, results with
// server-side explanations, one shared player, and impression/playback
// events the API attributes from the stored search. Data controls export or
// delete everything stored for this session. No framework, no build step.
"use strict";

const $ = (id) => document.getElementById(id);

// --- session, http -------------------------------------------------------------

function sessionId() {
  const make = () => "s_" + crypto.getRandomValues(new Uint32Array(4)).join("");
  try {
    let id = sessionStorage.getItem("podfind.session");
    if (!id) { id = make(); sessionStorage.setItem("podfind.session", id); }
    return id;
  } catch { return window.__podfindSession ||= make(); }
}

async function post(path, body) {
  const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

// --- small helpers --------------------------------------------------------------

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== "" && v !== false) node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children) if (c !== null && c !== undefined && c !== false) node.append(c);
  return node;
}
function minutes(seconds) { return seconds ? `${Math.max(1, Math.round(seconds / 60))} min` : ""; }
function clock(s) { if (!isFinite(s)) return "–:––"; s = Math.floor(s); const m = Math.floor(s / 60), r = s % 60; return `${m}:${String(r).padStart(2, "0")}`; }
function date(iso) { return iso ? new Date(iso).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : ""; }
function publishedAfter(value) {
  if (!value) return "";
  const days = { week: 7, month: 30, year: 365 }[value];
  return new Date(Date.now() - days * 86400000).toISOString().slice(0, 10);
}
const LANGUAGES = { en: "English", es: "Spanish", fr: "French", de: "German", pt: "Portuguese", it: "Italian", nl: "Dutch", ja: "Japanese", ko: "Korean", zh: "Chinese", ru: "Russian", ar: "Arabic", hi: "Hindi" };
function languageName(code) { return LANGUAGES[code] || (code || "").toUpperCase(); }

// --- theme -----------------------------------------------------------------------

function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme; else delete document.documentElement.dataset.theme;
  $("theme").textContent = theme === "light" ? "Dark" : "Light";
  document.querySelector('meta[name="theme-color"]').content = theme === "light" ? "#f1ece6" : "#262626";
}
$("theme").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "light" ? "" : "light";
  try { localStorage.setItem("podfind.theme", next); } catch {}
  applyTheme(next);
});
try { applyTheme(localStorage.getItem("podfind.theme") || ""); } catch { applyTheme(""); }

// --- state ---------------------------------------------------------------------------

const state = { queryId: "", results: [], voice: null, understood: null, playing: null };

function filters() {
  return {
    language: $("language").value || undefined,
    max_duration_seconds: Number($("duration").value) || undefined,
    published_after: publishedAfter($("published").value) || undefined,
    no_explicit: $("noexplicit").checked || undefined,
  };
}

// --- rendering -------------------------------------------------------------------------

function setStatus(text, warn) {
  const s = $("status");
  s.replaceChildren();
  if (!text) return;
  s.append(warn ? el("span", { class: "warn" }, text) : text);
}

function skeletons(n = 5) {
  const list = $("results");
  list.replaceChildren();
  for (let i = 0; i < n; i++) {
    list.append(el("li", { class: "skeleton", "aria-hidden": "true" }, el("i"), el("i", { class: "s-art" }),
      el("div", {}, el("i", { class: "s-line" }), el("i", { class: "s-line short" }), el("i", { class: "s-line" }))));
  }
  $("empty").hidden = true;
}

// The query as the server understood it: the search text plus every filter
// it parsed or we sent, each removable. Removing one re-runs the search.
function renderUnderstood(resp) {
  const box = $("understood");
  box.replaceChildren(el("span", { class: "label" }, "Understood as"));
  const f = resp.query.filters || {};
  const chips = [];
  if (resp.query.text) chips.push([null, resp.query.text, null]);
  if (f.language) chips.push(["language", languageName(f.language), () => { $("language").value = ""; }]);
  if (f.max_duration_seconds) chips.push(["duration", `under ${minutes(f.max_duration_seconds)}`, () => { $("duration").value = "0"; }]);
  if (f.published_after) chips.push(["published", `after ${date(f.published_after)}`, () => { $("published").value = ""; }]);
  if (f.no_explicit) chips.push(["explicit", "no explicit", () => { $("noexplicit").checked = false; }]);
  for (const [kind, text, clear] of chips) {
    const chip = el("span", { class: "chip" }, kind ? text : el("b", {}, text));
    if (clear) {
      chip.append(el("button", { type: "button", "aria-label": `Remove ${text}`, onclick: () => {
        clear();
        // The filter may have come from the typed text; strip it there too by
        // searching the understood text instead of the raw query.
        $("q").value = resp.query.text || $("q").value;
        state.voice = null;
        search();
      } }, "×"));
    }
    box.append(chip);
  }
  box.hidden = chips.length === 0;
}

const PLAY_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path class="ic-play" d="M8 5v14l11-7z"/><path class="ic-pause" d="M7 5h4v14H7zm6 0h4v14h-4z"/></svg>';
function playButton(r) {
  const b = el("button", { class: "play", type: "button", "aria-label": `Play ${r.title}`, onclick: () => play(r) });
  b.innerHTML = PLAY_ICON; // SVG needs its own namespace; markup is static
  return b;
}

function renderResults(resp) {
  state.queryId = resp.query_id;
  state.results = resp.results;
  const list = $("results");
  list.replaceChildren();
  $("empty").hidden = resp.results.length > 0;
  $("examples").hidden = true;
  $("hero").classList.add("compact");

  resp.results.forEach((r, i) => {
    const item = el("li", { class: "result", id: `r-${r.episode_id}`, style: `animation-delay:${Math.min(i, 8) * 40}ms` },
      el("span", { class: "rank", "aria-hidden": "true" }, String(i + 1).padStart(2, "0")),
      el("img", { class: "art", src: r.artwork_url, alt: "", loading: "lazy", width: "84", height: "84" }),
      el("div", {},
        el("h3", { class: "result-title" }, el("button", { type: "button", onclick: () => play(r) }, r.title)),
        el("div", { class: "result-show" }, el("em", {}, r.show), r.publisher && r.publisher !== r.show ? ` · ${r.publisher}` : ""),
        el("div", { class: "meta" },
          r.duration_seconds ? el("span", {}, minutes(r.duration_seconds)) : null,
          r.published_at ? el("span", {}, date(r.published_at)) : null,
          r.language ? el("span", {}, languageName(r.language)) : null,
          r.explicit ? el("span", {}, "explicit") : null),
        r.explanation?.length ? el("p", { class: "why" }, r.explanation.join(" · ")) : null),
      playButton(r));
    list.append(item);
  });
  markPlaying();

  const degraded = Object.entries(resp.legs || {}).filter(([, v]) => v.status !== "ok" && v.status !== "skipped");
  const count = resp.results.length;
  setStatus(`${count} ${count === 1 ? "result" : "results"}${degraded.length ? ` · ${degraded.map(([k, v]) => `${k} ${v.status}`).join(", ")}` : ""}`, degraded.length > 0);
  $("versions").textContent = `${Math.round(resp.timings_ms.total_ms)} ms · bundle ${resp.versions.bundle_id} · ${resp.versions.ordering}`;

  if (count) {
    post("/v1/events/impression", {
      session_id: sessionId(), query_id: resp.query_id, client_time: new Date().toISOString(),
      items: resp.results.map((r, i) => ({ episode_id: r.episode_id, displayed_rank: i + 1 })),
    }).catch((e) => console.warn("impression event", e));
  }
}

function render(resp) {
  resp.results = resp.results || []; // the API sends null for no results
  renderUnderstood(resp);
  renderResults(resp);
}

// --- search --------------------------------------------------------------------------------

async function search() {
  const query = $("q").value.trim();
  if (!query) { if (!recorder.active) setStatus("Type something, or tap the microphone."); return; }
  skeletons();
  setStatus("Searching");
  $("transcript").hidden = true;
  const body = { query, session_id: sessionId(), limit: 20, filters: filters() };
  // A transcript corrected before searching stays linked to its voice search.
  if (state.voice && query !== state.voice.text) body.input = { mode: "voice_edited", parent_query_id: state.voice.queryId };
  state.voice = null;
  try {
    render(await post("/v1/search/text", body));
    const url = new URL(location.href); url.searchParams.set("q", query);
    if (url.href !== location.href) history.pushState({}, "", url);
  } catch (e) {
    $("results").replaceChildren();
    setStatus(`Search failed: ${e.message}`, true);
  }
}

$("search").addEventListener("submit", (event) => { event.preventDefault(); search(); });
for (const id of ["language", "duration", "published", "noexplicit"]) {
  $(id).addEventListener("change", () => { if (state.queryId && $("q").value.trim()) search(); });
}
for (const b of document.querySelectorAll(".example")) {
  b.addEventListener("click", () => { $("q").value = b.textContent; search(); });
}
document.addEventListener("keydown", (e) => {
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
  if (e.key === "/" && !typing) { e.preventDefault(); $("q").focus(); $("q").select(); }
  if (e.key === "Escape" && document.activeElement === $("q")) { $("q").value = ""; }
});

// --- player: one <audio>, the result list reflects what is playing --------------------------------

const audio = $("audio");

function playbackEvent(action, result) {
  post(`/v1/events/playback/${action}`, {
    session_id: sessionId(), query_id: state.queryId, episode_id: result.episode_id,
    position_seconds: audio.currentTime || 0, duration_seconds: isFinite(audio.duration) ? audio.duration : 0,
    client_time: new Date().toISOString(),
  }).catch((e) => console.warn("playback event", e));
}

function markPlaying() {
  for (const li of document.querySelectorAll(".result")) li.classList.toggle("playing", !!state.playing && !audio.paused && li.id === `r-${state.playing.episode_id}`);
}

function play(result) {
  const same = state.playing && state.playing.episode_id === result.episode_id;
  if (same) { if (audio.paused) audio.play(); else audio.pause(); return; }
  if (state.playing && !audio.ended && audio.currentTime < 0.9 * (audio.duration || Infinity) && state.playing.started) {
    playbackEvent("skip", state.playing); // moving on before the end is a skip
  }
  state.playing = { ...result, started: false, lastProgress: 0 };
  audio.src = result.enclosure_url;
  $("player-art").src = result.artwork_url || "";
  $("player-title").textContent = result.title;
  $("player-show").textContent = result.show;
  $("player").hidden = false;
  document.body.classList.add("has-player");
  audio.play().catch((e) => setStatus(`Cannot play: ${e.message}`, true));
}

audio.addEventListener("play", () => {
  const p = state.playing; if (!p) return;
  if (!p.started) { p.started = true; playbackEvent("start", p); }
  $("player").classList.remove("paused"); $("player-toggle").setAttribute("aria-label", "Pause"); markPlaying();
});
audio.addEventListener("pause", () => { $("player").classList.add("paused"); $("player-toggle").setAttribute("aria-label", "Play"); markPlaying(); });
audio.addEventListener("timeupdate", () => {
  const p = state.playing; if (!p) return;
  if (audio.currentTime - p.lastProgress >= 30) { p.lastProgress = audio.currentTime; playbackEvent("progress", p); }
  $("player-time").textContent = clock(audio.currentTime);
  if (isFinite(audio.duration) && audio.duration > 0) $("player-seek").value = Math.round(1000 * audio.currentTime / audio.duration);
});
audio.addEventListener("durationchange", () => { $("player-duration").textContent = clock(audio.duration); });
audio.addEventListener("ended", () => { if (state.playing) playbackEvent("complete", state.playing); });
$("player-toggle").addEventListener("click", () => { if (audio.paused) audio.play(); else audio.pause(); });
$("player-back").addEventListener("click", () => { audio.currentTime = Math.max(0, audio.currentTime - 15); });
$("player-fwd").addEventListener("click", () => { audio.currentTime = Math.min(audio.duration || Infinity, audio.currentTime + 30); });
$("player-seek").addEventListener("input", (e) => { if (isFinite(audio.duration)) audio.currentTime = audio.duration * e.target.value / 1000; });
$("player-close").addEventListener("click", () => {
  audio.pause(); audio.removeAttribute("src"); audio.load();
  state.playing = null; $("player").hidden = true; document.body.classList.remove("has-player"); markPlaying();
});
if ("mediaSession" in navigator) {
  audio.addEventListener("play", () => {
    const p = state.playing; if (!p) return;
    navigator.mediaSession.metadata = new MediaMetadata({ title: p.title, artist: p.show, artwork: p.artwork_url ? [{ src: p.artwork_url }] : [] });
  });
  navigator.mediaSession.setActionHandler("play", () => audio.play());
  navigator.mediaSession.setActionHandler("pause", () => audio.pause());
}

// --- voice: 16 kHz 16-bit WAV captured in the page, a live level meter, the ----
// transcript in the search box where it can be corrected and resent.

const RECORD_RATE = 16000, MAX_RECORD_SECONDS = 29;
const recorder = { active: false, stream: null, context: null, node: null, analyser: null, chunks: [], timer: 0, raf: 0 };

function downsample(chunks, fromRate) {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const input = new Float32Array(total);
  let offset = 0;
  for (const c of chunks) { input.set(c, offset); offset += c.length; }
  if (fromRate === RECORD_RATE) return input;
  const ratio = fromRate / RECORD_RATE, out = new Float32Array(Math.floor(total / ratio));
  for (let i = 0; i < out.length; i++) {
    const start = Math.floor(i * ratio), end = Math.min(total, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j++) sum += input[j];
    out[i] = end > start ? sum / (end - start) : 0;
  }
  return out;
}

function wav(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2), v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE"); str(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); str(36, "data");
  v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, samples[i])) * 32767, true);
  return new Blob([buf], { type: "audio/wav" });
}

function meterFrame() {
  if (!recorder.active) return;
  const bars = $("meter").children, data = new Uint8Array(recorder.analyser.frequencyBinCount);
  recorder.analyser.getByteFrequencyData(data);
  const step = Math.floor(data.length / bars.length);
  for (let i = 0; i < bars.length; i++) {
    let sum = 0;
    for (let j = 0; j < step; j++) sum += data[i * step + j];
    bars[i].style.height = `${4 + Math.round((sum / step / 255) * 28)}px`;
  }
  recorder.raf = requestAnimationFrame(meterFrame);
}

async function startRecording() {
  recorder.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false } });
  recorder.context = new AudioContext();
  const source = recorder.context.createMediaStreamSource(recorder.stream);
  recorder.analyser = recorder.context.createAnalyser(); recorder.analyser.fftSize = 256;
  recorder.node = recorder.context.createScriptProcessor(4096, 1, 1);
  recorder.chunks = [];
  recorder.node.onaudioprocess = (e) => recorder.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  source.connect(recorder.analyser); source.connect(recorder.node); recorder.node.connect(recorder.context.destination);
  recorder.active = true;
  $("mic").setAttribute("aria-pressed", "true");
  $("search-box").classList.add("listening");
  setStatus("Listening · tap the microphone again to search");
  recorder.timer = setTimeout(stopRecording, MAX_RECORD_SECONDS * 1000);
  meterFrame();
}

async function stopRecording() {
  clearTimeout(recorder.timer); cancelAnimationFrame(recorder.raf);
  recorder.active = false;
  $("mic").setAttribute("aria-pressed", "false");
  $("search-box").classList.remove("listening");
  const rate = recorder.context.sampleRate;
  recorder.node.disconnect(); recorder.stream.getTracks().forEach((t) => t.stop()); await recorder.context.close();
  const samples = downsample(recorder.chunks, rate);
  recorder.chunks = [];
  if (samples.length < RECORD_RATE / 4) { setStatus("Too short · keep the microphone open while you speak", true); return; }
  await voiceSearch(wav(samples, RECORD_RATE), samples.length / RECORD_RATE);
}

async function voiceSearch(blob, seconds) {
  skeletons();
  setStatus(`Transcribing ${seconds.toFixed(1)} s`);
  const form = new FormData();
  form.append("audio", blob, "query.wav");
  form.append("session_id", sessionId());
  form.append("limit", "20");
  form.append("filters", JSON.stringify(filters()));
  form.append("retain", $("retain").checked ? "true" : "false");
  if ($("language").value) form.append("language", $("language").value);
  try {
    const res = await fetch("/v1/search/voice", { method: "POST", body: form });
    const resp = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(resp.error || res.statusText);
    const t = resp.transcript;
    $("q").value = t.text;
    state.voice = { queryId: resp.query_id, text: t.text };
    render(resp);
    const box = $("transcript");
    const kept = resp.retained ? "Recording kept." : resp.retention ? `Recording not kept: ${resp.retention}.` : "Recording deleted.";
    box.replaceChildren("Heard ", el("q", {}, t.text),
      el("span", { class: "confidence" }, [languageName(t.language), `${t.audio_seconds.toFixed(1)} s`, t.confidence != null ? `confidence ${t.confidence.toFixed(2)}` : ""].filter(Boolean).join(" · ")),
      el("br"), `Edit the text above and search again if that is not what you said. ${kept}`);
    box.hidden = false;
  } catch (e) {
    $("results").replaceChildren();
    setStatus(`Voice search failed: ${e.message}`, true);
  }
}

$("mic").addEventListener("click", async () => {
  if (!navigator.mediaDevices?.getUserMedia) { setStatus("This browser cannot record audio", true); return; }
  try {
    if (recorder.active) await stopRecording(); else await startRecording();
  } catch (e) {
    recorder.active = false;
    $("mic").setAttribute("aria-pressed", "false");
    $("search-box").classList.remove("listening");
    setStatus(`Microphone unavailable: ${e.message}`, true);
  }
});

// --- data controls ---------------------------------------------------------------------------------

$("privacy-open").addEventListener("click", () => { $("session-label").textContent = sessionId(); $("privacy-status").textContent = ""; $("privacy").showModal(); });
$("export").addEventListener("click", () => {
  window.open(`/v1/me/data?session_id=${encodeURIComponent(sessionId())}`, "_blank");
});
$("delete").addEventListener("click", async () => {
  if (!confirm("Delete every search, event and kept recording of this session?")) return;
  try {
    const res = await fetch(`/v1/me/data?session_id=${encodeURIComponent(sessionId())}`, { method: "DELETE" });
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || res.statusText);
    state.queryId = ""; state.results = []; state.voice = null;
    $("results").replaceChildren(); $("transcript").hidden = true; $("understood").hidden = true; $("versions").textContent = "";
    $("privacy-status").textContent = `Deleted ${d.searches} searches, ${d.events} events and ${d.recordings} recordings.`;
  } catch (e) {
    $("privacy-status").textContent = `Delete failed: ${e.message}`;
  }
});

// A shareable URL: /?q=… runs the search on load and the address bar follows
// each search.
window.addEventListener("popstate", () => { const q = new URLSearchParams(location.search).get("q"); if (q) { $("q").value = q; search(); } });
{
  const q = new URLSearchParams(location.search).get("q");
  if (q) { $("q").value = q; search(); }
}

// --- catalog size in the masthead, from the served indexes -------------------------------------------

fetch("/v1/model").then((r) => r.json()).then((m) => {
  // The API's index metrics carry the row count; the manifest does not, so
  // keep the static figure unless the metrics endpoint is reachable.
  return fetch("/metrics").then((r) => r.ok ? r.text() : "");
}).then((text) => {
  const m = /podfind_index_rows\{model="intfloat\/multilingual-e5-small"\} (\d+)/.exec(text || "");
  if (m) $("catalog-size").textContent = Number(m[1]).toLocaleString();
}).catch(() => {});
