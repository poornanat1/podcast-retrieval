// PodFind web app: typed and spoken search with filters, an editable
// transcript, results with server-side explanations, playback, and
// impression/playback events attributed by the API from the stored search.
// Data controls export or delete everything stored for this session.
"use strict";

const $ = (id) => document.getElementById(id);

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

function publishedAfter(value) {
  if (!value) return "";
  const days = { week: 7, month: 30, year: 365 }[value];
  const d = new Date(Date.now() - days * 86400000);
  return d.toISOString().slice(0, 10);
}

function minutes(seconds) { return seconds ? `${Math.max(1, Math.round(seconds / 60))} min` : ""; }
function date(iso) { return iso ? new Date(iso).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : ""; }
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v; else if (v !== undefined && v !== "") node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

let current = { queryId: "", playing: null, voice: null };

function filters() {
  return {
    language: $("language").value || undefined,
    max_duration_seconds: Number($("duration").value) || undefined,
    published_after: publishedAfter($("published").value) || undefined,
    no_explicit: $("noexplicit").checked || undefined,
  };
}

function playbackEvent(action, result, audio) {
  post(`/v1/events/playback/${action}`, {
    session_id: sessionId(), query_id: current.queryId, episode_id: result.episode_id,
    position_seconds: audio.currentTime || 0, duration_seconds: isFinite(audio.duration) ? audio.duration : 0,
    client_time: new Date().toISOString(),
  }).catch((e) => console.warn("playback event", e));
}

function attachPlayback(audio, result) {
  let started = false, lastProgress = 0;
  audio.addEventListener("play", () => {
    const prev = current.playing;
    if (prev && prev.audio !== audio) {
      // Starting another episode before finishing this one is a skip.
      if (!prev.audio.ended && prev.audio.currentTime < 0.9 * (prev.audio.duration || Infinity)) {
        playbackEvent("skip", prev.result, prev.audio);
      }
      prev.audio.pause();
    }
    current.playing = { audio, result };
    if (!started) { started = true; playbackEvent("start", result, audio); }
  });
  audio.addEventListener("timeupdate", () => {
    if (audio.currentTime - lastProgress >= 30) { lastProgress = audio.currentTime; playbackEvent("progress", result, audio); }
  });
  audio.addEventListener("ended", () => playbackEvent("complete", result, audio));
}

function render(resp) {
  current.queryId = resp.query_id;
  const status = $("status");
  status.replaceChildren();
  const legs = Object.entries(resp.legs || {}).filter(([, v]) => v.status !== "ok" && v.status !== "skipped");
  status.append(`${resp.results.length} results for “${resp.query.text || resp.query.raw}”`);
  if (resp.query.parsed?.length) {
    const p = el("span", { class: "parsed" }, " · understood:");
    for (const item of resp.query.parsed) p.append(el("span", {}, item));
    status.append(p);
  }
  if (legs.length) status.append(` · degraded: ${legs.map(([k, v]) => `${k} ${v.status}`).join(", ")}`);

  const list = $("results");
  list.replaceChildren();
  for (const r of resp.results) {
    const why = el("ul", { class: "why" });
    for (const reason of r.explanation) why.append(el("li", {}, reason));
    const audio = el("audio", { controls: "", preload: "none", src: r.enclosure_url });
    attachPlayback(audio, r);
    list.append(el("li", { class: "result" },
      el("img", { src: r.artwork_url, alt: "", loading: "lazy" }),
      el("div", {},
        el("div", { class: "title" }, r.title),
        el("div", { class: "meta" }, [r.show, minutes(r.duration_seconds), date(r.published_at), r.explicit ? "explicit" : ""].filter(Boolean).join(" · ")),
        why, audio)));
  }
  $("versions").textContent = `query ${resp.query_id} · model bundle ${resp.versions.bundle_id} · features v${resp.versions.feature_version} · ordering ${resp.versions.ordering} · ${Math.round(resp.timings_ms.total_ms)} ms`;

  if (resp.results.length) {
    post("/v1/events/impression", {
      session_id: sessionId(), query_id: resp.query_id, client_time: new Date().toISOString(),
      items: resp.results.map((r, i) => ({ episode_id: r.episode_id, displayed_rank: i + 1 })),
    }).catch((e) => console.warn("impression event", e));
  }
}

$("search").addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = $("q").value.trim();
  if (!query) { if (!recorder.active) $("status").textContent = "Type a query or tap the microphone."; return; }
  $("status").textContent = "Searching…";
  const body = { query, session_id: sessionId(), limit: 20, filters: filters() };
  // A transcript the user corrected before searching is linked to the voice
  // search it came from, so transcription errors can be studied later.
  if (current.voice && query !== current.voice.text) body.input = { mode: "voice_edited", parent_query_id: current.voice.queryId };
  current.voice = null;
  try {
    render(await post("/v1/search/text", body));
    $("transcript").hidden = true;
  } catch (e) {
    $("status").textContent = `Search failed: ${e.message}`;
    $("results").replaceChildren();
  }
});

// Voice: capture from the microphone as 16 kHz 16-bit WAV in the page (no
// compressed container the server would have to decode), post it, and show
// the transcript in the search box where it can be corrected and resent.
const RECORD_RATE = 16000, MAX_RECORD_SECONDS = 30;
const recorder = { active: false, stream: null, context: null, node: null, chunks: [], timer: 0 };

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

async function startRecording() {
  recorder.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false } });
  recorder.context = new AudioContext();
  const source = recorder.context.createMediaStreamSource(recorder.stream);
  recorder.node = recorder.context.createScriptProcessor(4096, 1, 1);
  recorder.chunks = [];
  recorder.node.onaudioprocess = (e) => recorder.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  source.connect(recorder.node); recorder.node.connect(recorder.context.destination);
  recorder.active = true;
  $("mic").setAttribute("aria-pressed", "true");
  $("status").textContent = "Listening… tap the microphone again to search.";
  recorder.timer = setTimeout(stopRecording, MAX_RECORD_SECONDS * 1000);
}

async function stopRecording() {
  clearTimeout(recorder.timer);
  recorder.active = false;
  $("mic").setAttribute("aria-pressed", "false");
  const rate = recorder.context.sampleRate;
  recorder.node.disconnect(); recorder.stream.getTracks().forEach((t) => t.stop()); await recorder.context.close();
  const samples = downsample(recorder.chunks, rate);
  recorder.chunks = [];
  if (samples.length < RECORD_RATE / 4) { $("status").textContent = "Too short; hold the microphone open while you speak."; return; }
  await voiceSearch(wav(samples, RECORD_RATE), samples.length / RECORD_RATE);
}

async function voiceSearch(blob, seconds) {
  $("status").textContent = `Transcribing ${seconds.toFixed(1)} s of audio…`;
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
    current.voice = { queryId: resp.query_id, text: t.text };
    render(resp);
    const box = $("transcript");
    box.replaceChildren(`Heard “`, el("b", {}, t.text), `” (${t.language || "?"}, ${t.audio_seconds.toFixed(1)} s, transcribed in ${Math.round(resp.timings_ms.transcribe_ms)} ms` +
      (t.confidence != null ? `, confidence ${t.confidence.toFixed(2)}` : "") + `). Edit the text above and search again if it is wrong.` +
      (resp.retained ? " Recording kept." : resp.retention ? ` Recording not kept: ${resp.retention}.` : " Recording deleted."));
    box.hidden = false;
  } catch (e) {
    $("status").textContent = `Voice search failed: ${e.message}`;
  }
}

$("mic").addEventListener("click", async () => {
  if (!navigator.mediaDevices?.getUserMedia) { $("status").textContent = "This browser cannot record audio."; return; }
  try {
    if (recorder.active) await stopRecording(); else await startRecording();
  } catch (e) {
    recorder.active = false;
    $("mic").setAttribute("aria-pressed", "false");
    $("status").textContent = `Microphone unavailable: ${e.message}`;
  }
});

// Data controls: everything the API stores for this session.
$("export").addEventListener("click", (event) => {
  event.preventDefault();
  window.open(`/v1/me/data?session_id=${encodeURIComponent(sessionId())}`, "_blank");
});
$("delete").addEventListener("click", async (event) => {
  event.preventDefault();
  if (!confirm("Delete every search, event and kept recording of this session?")) return;
  try {
    const res = await fetch(`/v1/me/data?session_id=${encodeURIComponent(sessionId())}`, { method: "DELETE" });
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || res.statusText);
    current = { queryId: "", playing: null, voice: null };
    $("results").replaceChildren(); $("transcript").hidden = true; $("versions").textContent = "";
    $("status").textContent = `Deleted ${d.searches} searches, ${d.events} events and ${d.recordings} recordings.`;
  } catch (e) {
    $("status").textContent = `Delete failed: ${e.message}`;
  }
});
