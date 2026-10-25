// PodFind web app: typed search with filters, results with server-side
// explanations, playback, and impression/playback events attributed by the
// API from the stored search.
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

let current = { queryId: "", playing: null };

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
  $("status").textContent = "Searching…";
  try {
    const resp = await post("/v1/search/text", {
      query: $("q").value, session_id: sessionId(), limit: 20,
      filters: {
        language: $("language").value || undefined,
        max_duration_seconds: Number($("duration").value) || undefined,
        published_after: publishedAfter($("published").value) || undefined,
        no_explicit: $("noexplicit").checked || undefined,
      },
    });
    render(resp);
  } catch (e) {
    $("status").textContent = `Search failed: ${e.message}`;
    $("results").replaceChildren();
  }
});
