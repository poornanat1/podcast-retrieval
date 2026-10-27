"""Voice-robustness evaluation: how much retrieval loses to transcription.

    uv run python -m ml.speech.evaluate --testset data/speech/testset-v1 --api http://localhost:8090

Every clip in the test set goes through ``POST /v1/search/voice`` on the
running API (debug mode, so the query's filters are used verbatim), with
query understanding on, since spoken queries carry a carrier phrase or a
conversational framing it must strip. Two references are typed through
``POST /v1/search/text``: the same spoken text (same understanding), whose
difference from the transcript's result is the degradation attributable
to transcription; and the labeled terse query with understanding skipped,
whose difference from the first is what the phrasing itself costs. Per
clip, the report has the word error rate against what was spoken, whether
each named entity survived, both references' metrics, the transcript's,
the degradation, and the stage latencies.

Aggregates: by tier, condition, speed, voice, query type, phrasing, word
error rate bin, entity outcome, and audio duration; the degradation among
clips with zero word errors (what casing and punctuation alone cost); a
paired bootstrap interval on the base tier's degradation; and language
detection on the non-English clips.

The served ranker was trained on these queries, so absolute metrics are
in-sample; the differences between reference and transcript are what the
report is about. Quality should be measured with the API's leg time
limits raised (a timed-out leg changes results for reasons unrelated to
speech) and latency with the production limits; ``--tier`` selects the
clips for a run. Searches made here use session ``speech-eval`` and are
deleted at the end through the data controls unless ``--keep`` is given.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.request
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np

from ml.evaluation import metrics
from ml.evaluation.run import load_qrels
from ml.speech.wer import entity_error_rate, entity_matches, word_errors

SESSION = "speech-eval"
METRICS = ("ndcg_at_10", "mrr", "recall_at_100")
WER_BINS = (("0", 0.0, 0.0), ("(0, 0.25]", 0.0, 0.25), ("(0.25, 0.5]", 0.25, 0.5),
            ("> 0.5", 0.5, float("inf")))
DURATION_BINS = (("< 2 s", 0, 2), ("2–4 s", 2, 4), ("4–8 s", 4, 8), ("≥ 8 s", 8, float("inf")))


class API:
    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def _call(self, req: urllib.request.Request) -> dict:
        # A cold lexical query can outlast the API's write timeout, which
        # closes the connection; the retry finds it warm.
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    return json.load(resp)
            except (http.client.RemoteDisconnected, ConnectionResetError, TimeoutError) as err:
                if attempt == 2:
                    raise
                print(f"retrying after {err}", flush=True)
                time.sleep(2)
        raise AssertionError("unreachable")

    def text(self, query: str, filters: dict, skip_understanding: bool, limit: int = 100) -> dict:
        body = json.dumps({"query": query, "filters": filters, "session_id": SESSION,
                           "limit": limit, "debug": True,
                           "skip_understanding": skip_understanding}).encode()
        return self._call(urllib.request.Request(f"{self.base}/v1/search/text", body,
                                                 {"content-type": "application/json"}))

    def voice(self, wav: bytes, filters: dict, skip_understanding: bool, limit: int = 100) -> dict:
        boundary = uuid.uuid4().hex
        fields = {"session_id": SESSION, "limit": str(limit), "filters": json.dumps(filters),
                  "debug": "true", "skip_understanding": "true" if skip_understanding else "false"}
        parts = b""
        for name, value in fields.items():
            parts += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                      f"{value}\r\n").encode()
        parts += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; "
                  f"filename=\"clip.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
        parts += wav + f"\r\n--{boundary}--\r\n".encode()
        return self._call(urllib.request.Request(
            f"{self.base}/v1/search/voice", parts,
            {"content-type": f"multipart/form-data; boundary={boundary}"}))

    def delete_session(self) -> dict:
        return self._call(urllib.request.Request(f"{self.base}/v1/me/data?session_id={SESSION}",
                                                 method="DELETE"))


def score(ranked: list[int], grades: dict, relevant: set) -> dict:
    return {"ndcg_at_10": metrics.ndcg_at_k(ranked, grades, 10),
            "mrr": metrics.reciprocal_rank(ranked, relevant),
            "recall_at_100": metrics.recall_at_k(ranked, relevant, 100)}


def bootstrap_mean(values: list[float], rng: np.random.Generator, n: int = 2000) -> dict:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"mean": None, "ci95": None}
    means = [float(arr[rng.integers(len(arr), size=len(arr))].mean()) for _ in range(n)]
    return {"mean": round(float(arr.mean()), 4),
            "ci95": [round(float(np.percentile(means, 2.5)), 4),
                     round(float(np.percentile(means, 97.5)), 4)]}


def bucket(value: float, bins) -> str:
    for name, lo, hi in bins:
        if value == lo == hi or lo < value <= hi or (value == 0 and lo == 0 < hi):
            return name
    return bins[-1][0]


def subset_with_base(rows: list[dict], tier: str) -> list[dict]:
    """The clips of a derived tier plus the base clips of the same queries,
    so every by-tier table has the clean, base-voice, terse row to compare
    against."""
    queries = {r["query_id"] for r in rows if r["tier"] == tier}
    return [r for r in rows if r["tier"] == tier
            or (r["tier"] == "base" and r["query_id"] in queries)]


def summarize(rows: list[dict], key) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out = {}
    for name, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        entry = {"clips": len(rs),
                 "wer": round(float(np.mean([r["wer"] for r in rs])), 4),
                 "wer_zero_share": round(float(np.mean([r["wer"] == 0 for r in rs])), 4)}
        with_entities = [r for r in rs if r["entity_error_rate"] is not None]
        if with_entities:
            entry["entity_error_rate"] = round(float(np.mean(
                [r["entity_error_rate"] for r in with_entities])), 4)
        for m in METRICS:
            ref = float(np.mean([r["reference"][m] for r in rs]))
            got = float(np.mean([r["from_transcript"][m] for r in rs]))
            entry[f"reference_{m}"] = round(ref, 4)
            entry[f"transcript_{m}"] = round(got, 4)
            entry[f"degradation_{m}"] = round(ref - got, 4)
        out[str(name)] = entry
    return out


def latency(rows: list[dict], key) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out = {}
    for name, rs in groups.items():
        entry = {"clips": len(rs), "audio_seconds_mean": round(float(np.mean(
            [r["audio_seconds"] for r in rs])), 2)}
        for stage in ("transcribe_ms", "total_ms"):
            values = [r["timings_ms"].get(stage, 0.0) for r in rs]
            entry[stage] = {"p50": round(metrics.percentile(values, 50), 1),
                            "p95": round(metrics.percentile(values, 95), 1)}
        out[str(name)] = entry
    order = [b[0] for b in DURATION_BINS]
    return dict(sorted(out.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99))


def entity_outcome(r: dict) -> str:
    if all(m["found"] for m in r["entities"]):
        return "found"
    if all(m["found"] or m["cer"] <= 0.34 for m in r["entities"]):
        return "near miss"
    return "missed"


def leg_counts(rows: list[dict]) -> dict:
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        for leg, status in r["legs"].items():
            counts[f"{leg}:{status}"] += 1
    return dict(sorted(counts.items()))


def evaluate(testset: Path, api: API, relevance: Path, limit_clips: int | None = None,
             tiers: list[str] | None = None) -> tuple[dict, list[dict]]:
    manifest = [json.loads(line)
                for line in (testset / "manifest.jsonl").read_text().splitlines()]
    if tiers:
        manifest = [r for r in manifest if r["tier"] in tiers]
    if limit_clips:
        manifest = manifest[:limit_clips]
    qrels = load_qrels(relevance / "qrels.txt")
    references: dict[tuple, dict] = {}

    def typed(text: str, r: dict, grades: dict, relevant: set, skip: bool) -> dict:
        key = (r["query_id"], text, skip)
        if key not in references:
            resp = api.text(text, r["filters"], skip)
            ranked = [x["episode_id"] for x in resp.get("results") or []]
            references[key] = {"understood": resp["query"]["text"],
                               **score(ranked, grades, relevant)}
        return references[key]

    rows = []
    started = time.time()
    for n, r in enumerate(manifest, 1):
        grades = qrels.get(r["query_id"], {})
        relevant = {e for e, g in grades.items() if g >= 2}
        ref = typed(r["text"], r, grades, relevant, skip=False)
        terse = typed(r["terse_text"], r, grades, relevant, skip=True)
        resp = api.voice((testset / r["path"]).read_bytes(), r["filters"], False)
        ranked = [x["episode_id"] for x in resp.get("results") or []]
        t = resp["transcript"]
        # Candidate mitigation, measured: the transcript with the recognizer's
        # capitalization removed, typed.
        lowered = typed(t["text"].lower(), r, grades, relevant, skip=False)
        w = word_errors(r["text"], t["text"])
        ents = entity_matches(r["entities"], t["text"])
        row = {
            "clip_id": r["clip_id"], "query_id": r["query_id"], "query_type": r["query_type"],
            "tier": r["tier"], "phrasing": r["phrasing"], "voice": r["voice"],
            "locale": r["locale"], "condition": r["condition"], "speed": r["speed"],
            "language": r["language"], "audio_seconds": r["seconds"],
            "spoken": r["text"], "heard": t["text"], "backend": t.get("backend"),
            "detected_language": t.get("language"), "confidence": t.get("confidence"),
            "wer": round(w.wer, 4), "substitutions": w.substitutions, "deletions": w.deletions,
            "insertions": w.insertions, "words": w.words,
            "entities": [m.__dict__ for m in ents], "entity_error_rate": entity_error_rate(ents),
            "understood": resp["query"]["text"], "reference_understood": ref["understood"],
            "reference": {m: ref[m] for m in METRICS},
            "labeled_query": {m: terse[m] for m in METRICS},
            "from_transcript": score(ranked, grades, relevant),
            "from_transcript_lowercased": {m: lowered[m] for m in METRICS},
            "timings_ms": resp["timings_ms"],
            "legs": {k: v["status"] for k, v in resp["legs"].items()},
        }
        row["degradation"] = {m: round(row["reference"][m] - row["from_transcript"][m], 4)
                              for m in METRICS}
        row["phrasing_cost"] = {m: round(row["labeled_query"][m] - row["reference"][m], 4)
                                for m in METRICS}
        rows.append(row)
        if n % 50 == 0 or n == len(manifest):
            print(f"{n}/{len(manifest)} clips, {(time.time() - started) / n:.2f} s each",
                  flush=True)

    rng = np.random.default_rng(0)
    base = [r for r in rows if r["tier"] == "base"]
    zero = [r for r in base if r["wer"] == 0]
    nonzero = [r for r in base if r["wer"] > 0]
    non_english = [r for r in rows if r["language"] != "en"]

    def mean_degradation(rs: list[dict]) -> float | None:
        return round(float(np.mean([r["degradation"]["ndcg_at_10"] for r in rs])), 4) \
            if rs else None

    def lowercase_effect(rs: list[dict]) -> dict:
        return {"clips": len(rs), **{m: {
            "transcript": round(float(np.mean([r["from_transcript"][m] for r in rs])), 4),
            "transcript_lowercased": round(float(np.mean(
                [r["from_transcript_lowercased"][m] for r in rs])), 4)} for m in METRICS}} \
            if rs else {"clips": 0}

    def phrasing_cost(rs: list[dict]) -> dict:
        return {m: {"labeled_query": round(float(np.mean([r["labeled_query"][m] for r in rs])), 4),
                    "spoken_text_typed": round(float(np.mean([r["reference"][m] for r in rs])), 4),
                    "cost": round(float(np.mean([r["phrasing_cost"][m] for r in rs])), 4)}
                for m in METRICS} if rs else {}

    report = {
        "testset": str(testset), "clips": len(rows),
        "queries": len({r["query_id"] for r in rows}),
        "recognizers": sorted({r["backend"] for r in rows if r["backend"]}),
        "base": {
            **summarize(base, lambda r: "all")["all"],
            "degradation_ci": {m: bootstrap_mean([r["degradation"][m] for r in base], rng)
                               for m in METRICS},
            "wer_zero": {"clips": len(zero), "degradation_ndcg_at_10": mean_degradation(zero),
                         "lowercased": lowercase_effect(zero)},
            "wer_nonzero": {"clips": len(nonzero),
                            "degradation_ndcg_at_10": mean_degradation(nonzero)},
        },
        "lowercasing_the_transcript": {"base": lowercase_effect(base),
                                       "all": lowercase_effect(rows)},
        "phrasing_cost": {
            "terse_with_carrier": phrasing_cost(base),
            "conversational": phrasing_cost([r for r in rows if r["tier"] == "conversational"]),
        },
        "by_tier": summarize(rows, lambda r: r["tier"]),
        "by_query_type": summarize(base, lambda r: r["query_type"]),
        "by_voice": summarize(subset_with_base(rows, "accent"), lambda r: r["locale"]),
        "by_condition": summarize(subset_with_base(rows, "condition"), lambda r: r["condition"]),
        "by_speed": summarize(subset_with_base(rows, "speed"), lambda r: f"{r['speed']}x"),
        "by_phrasing": summarize(subset_with_base(rows, "conversational"),
                                 lambda r: r["phrasing"]),
        "by_wer_bin": summarize(rows, lambda r: bucket(r["wer"], WER_BINS)),
        "by_entity_outcome": summarize([r for r in rows if r["entities"]], entity_outcome),
        "latency_by_duration": latency(rows, lambda r: bucket(r["audio_seconds"], DURATION_BINS)),
        "language_detection": {
            "non_english_clips": len(non_english),
            "detected_correctly": sum(r["detected_language"] == r["language"]
                                      for r in non_english),
            "english_detected_as_other": sum(r["detected_language"] not in ("en", None)
                                             for r in rows if r["language"] == "en"),
        },
        "legs": leg_counts(rows),
    }
    return report, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--testset", default="data/speech/testset-v1")
    parser.add_argument("--api", default="http://localhost:8090")
    parser.add_argument("--relevance-dir", default="data/relevance")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--clips-out", default="",
                        help="per-clip rows (default: <testset>/clips.jsonl)")
    parser.add_argument("--limit", type=int, default=None, help="first N clips only")
    parser.add_argument("--tier", action="append", default=None,
                        help="only these tiers (repeatable): base, accent, conversational, "
                             "condition, speed")
    parser.add_argument("--keep", action="store_true", help="keep the searches this run stored")
    args = parser.parse_args(argv)
    api = API(args.api)
    report, rows = evaluate(Path(args.testset), api, Path(args.relevance_dir), args.limit,
                            args.tier)
    clips_out = Path(args.clips_out or Path(args.testset) / "clips.jsonl")
    clips_out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    print(text)
    if not args.keep:
        print("deleted", api.delete_session())
    return 0


if __name__ == "__main__":
    sys.exit(main())
