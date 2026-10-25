"""Score the live search API on the relevance set.

    uv run python -m ml.serving.online_eval --api http://localhost:8090 [--json-out out.json]

Sends every relevance query through ``POST /v1/search/text`` (debug mode,
query text and filters used verbatim, so query understanding does not
change the labeled query) and scores the returned top 100 with the offline
harness metrics against the same qrels. Reports the metrics, per-leg
outcomes (how often each leg timed out), and stage latency percentiles.

The served ranker is the final model, trained on all relevance queries, so
these numbers are in-sample: use them to compare serving configurations
(for example lexical time limits) and to confirm the online path, not to
estimate quality on new queries. Out-of-fold numbers are in
experiments/ranker-report.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ml.evaluation import metrics
from ml.evaluation.run import load_qrels, load_queries

KEYS = ("ndcg_at_5", "ndcg_at_10", "map_at_10", "precision_at_10", "mrr", "recall_at_100")


def search(api: str, q: dict, limit: int) -> dict:
    filters = {k: q[k] for k in ("language", "max_duration_seconds", "published_after",
                                 "no_explicit") if q.get(k)}
    body = json.dumps({"query": q["query_text"], "filters": filters, "session_id": "online-eval",
                       "limit": limit, "debug": True, "skip_understanding": True}).encode()
    req = urllib.request.Request(f"{api}/v1/search/text", body,
                                 {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def evaluate(api: str, relevance: Path, limit: int = 100, min_grade: int = 2) -> dict:
    queries = load_queries(relevance / "queries.jsonl")
    qrels = load_qrels(relevance / "qrels.txt")
    rows, legs, stages, orderings = [], Counter(), defaultdict(list), Counter()
    for q in queries:
        grades = qrels.get(q["query_id"], {})
        relevant = {e for e, g in grades.items() if g >= min_grade}
        if not relevant:
            continue
        r = search(api, q, limit)
        ranked = [x["episode_id"] for x in r["results"]]
        rows.append({
            "query_type": q["query_type"],
            "ndcg_at_5": metrics.ndcg_at_k(ranked, grades, 5),
            "ndcg_at_10": metrics.ndcg_at_k(ranked, grades, 10),
            "map_at_10": metrics.average_precision_at_k(ranked, relevant, 10),
            "precision_at_10": metrics.precision_at_k(ranked, relevant, 10),
            "mrr": metrics.reciprocal_rank(ranked, relevant),
            "recall_at_100": metrics.recall_at_k(ranked, relevant, 100),
        })
        for leg, v in r["legs"].items():
            legs[f"{leg}:{v['status']}"] += 1
        for stage, ms in r["timings_ms"].items():
            stages[stage].append(ms)
        orderings[r["versions"]["ordering"]] += 1

    def mean(rs):
        return {k: round(float(np.mean([x[k] for x in rs])), 4) for k in KEYS}

    return {
        "queries": len(rows),
        "global": mean(rows),
        "by_query_type": {t: mean([x for x in rows if x["query_type"] == t])
                          for t in sorted({x["query_type"] for x in rows})},
        "legs": dict(legs),
        "orderings": dict(orderings),
        "latency_ms": {s: {"p50": round(metrics.percentile(v, 50), 1),
                           "p95": round(metrics.percentile(v, 95), 1)}
                       for s, v in sorted(stages.items())},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://localhost:8090")
    parser.add_argument("--relevance-dir", default="data/relevance")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)
    report = evaluate(args.api.rstrip("/"), Path(args.relevance_dir))
    text = json.dumps(report, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
