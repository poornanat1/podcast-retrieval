"""Retrieval evaluation harness.

A retrieval system is anything with a ``name`` and a
``search(query: dict, k: int) -> list[int]`` method taking a relevance-set
query row. The harness runs every judged query, times each call, and
aggregates metrics globally and per query type.

Metric conventions:
- Binary metrics (recall, hit rate, MRR) treat grade >= ``min_relevant_grade``
  as relevant; NDCG uses the full graded scale.
- Queries whose qrels contain no relevant document are excluded from
  averaged quality metrics (reported as ``skipped_queries``).
- Coverage is measured over all retrieved results: ``catalog_coverage`` is
  distinct retrieved episodes / catalog size, and ``tail_coverage`` is the
  share of results from podcasts outside the head set (a supplied set of
  head podcast ids — the project uses the top decile by episode count as a
  popularity proxy until real play data exists).
"""

from __future__ import annotations

import dataclasses
import time
from collections import defaultdict
from typing import Protocol

from ml.evaluation import metrics


class RetrievalSystem(Protocol):
    name: str

    def search(self, query: dict, k: int) -> list[int]: ...


@dataclasses.dataclass
class EvalReport:
    system: str
    queries: int
    skipped_queries: int
    global_metrics: dict[str, float]
    by_query_type: dict[str, dict[str, float]]
    by_query_language: dict[str, dict[str, float]]
    per_query: list[dict]


def _aggregate(rows: list[dict], ks: list[int]) -> dict[str, float]:
    if not rows:
        return {}
    out: dict[str, float] = {}
    keys = [f"recall_at_{k}" for k in ks] + [f"hit_rate_at_{k}" for k in ks]
    keys += ["mrr", "ndcg_at_10"]
    for key in keys:
        out[key] = round(sum(r[key] for r in rows) / len(rows), 4)
    return out


#: Ks reported for item-side cohorts (transcript / head-tail partitions).
COHORT_KS = (10, 100)


def _cohort_recalls(
    ranked: list[int], partitions: dict[str, set[int]]
) -> dict[str, float]:
    """Recall restricted to each non-empty partition of the relevant set:
    how well does the system retrieve *this kind* of relevant item?"""
    out: dict[str, float] = {}
    for label, members in partitions.items():
        if not members:
            continue
        for k in COHORT_KS:
            out[f"recall_at_{k}__{label}"] = metrics.recall_at_k(ranked, members, k)
    return out


def evaluate(
    system: RetrievalSystem,
    queries: list[dict],
    qrels: dict[str, dict[int, int]],
    ks: list[int] | None = None,
    min_relevant_grade: int = 2,
    catalog_size: int = 0,
    episode_podcast: dict[int, int] | None = None,
    head_podcasts: set[int] | None = None,
    episodes_with_transcript: set[int] | None = None,
) -> EvalReport:
    ks = ks or [10, 50, 100]
    max_k = max(ks)
    episode_podcast = episode_podcast or {}
    head_podcasts = head_podcasts or set()
    episodes_with_transcript = episodes_with_transcript or set()

    per_query: list[dict] = []
    latencies: list[float] = []
    retrieved_all: set[int] = set()
    cohort_values: dict[str, list[float]] = defaultdict(list)
    result_count = 0
    tail_count = 0
    skipped = 0

    for query in queries:
        grades = qrels.get(query["query_id"], {})
        relevant = {e for e, g in grades.items() if g >= min_relevant_grade}
        if not relevant:
            skipped += 1
            continue

        start = time.perf_counter()
        ranked = system.search(query, max_k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        retrieved_all.update(ranked)
        result_count += len(ranked)
        tail_count += sum(
            1 for e in ranked
            if e in episode_podcast and episode_podcast[e] not in head_podcasts
        )

        row = {
            "query_id": query["query_id"],
            "query_type": query["query_type"],
            "query_language": query.get("language") or "en",
            "results": len(ranked),
            "relevant": len(relevant),
            "latency_ms": round(elapsed_ms, 2),
            "mrr": metrics.reciprocal_rank(ranked, relevant),
            "ndcg_at_10": round(metrics.ndcg_at_k(ranked, grades, 10), 4),
        }
        for k in ks:
            row[f"recall_at_{k}"] = round(metrics.recall_at_k(ranked, relevant, k), 4)
            row[f"hit_rate_at_{k}"] = metrics.hit_rate_at_k(ranked, relevant, k)
        per_query.append(row)

        with_transcript = {e for e in relevant if e in episodes_with_transcript}
        head_relevant = {
            e for e in relevant if episode_podcast.get(e) in head_podcasts
        }
        for key, value in _cohort_recalls(ranked, {
            "transcript": with_transcript,
            "no_transcript": relevant - with_transcript,
            "head": head_relevant,
            "tail": relevant - head_relevant,
        }).items():
            cohort_values[key].append(value)

    by_type: dict[str, list[dict]] = defaultdict(list)
    by_language: dict[str, list[dict]] = defaultdict(list)
    for row in per_query:
        by_type[row["query_type"]].append(row)
        by_language[row["query_language"]].append(row)

    global_metrics = _aggregate(per_query, ks)
    for key, values in sorted(cohort_values.items()):
        global_metrics[key] = round(sum(values) / len(values), 4)
        global_metrics[f"{key}__queries"] = len(values)
    global_metrics["latency_ms_p50"] = round(metrics.percentile(latencies, 50), 2)
    global_metrics["latency_ms_p95"] = round(metrics.percentile(latencies, 95), 2)
    if catalog_size:
        global_metrics["catalog_coverage"] = round(len(retrieved_all) / catalog_size, 6)
    if result_count and episode_podcast:
        global_metrics["tail_coverage"] = round(tail_count / result_count, 4)

    return EvalReport(
        system=system.name,
        queries=len(per_query),
        skipped_queries=skipped,
        global_metrics=global_metrics,
        by_query_type={t: _aggregate(rows, ks) for t, rows in sorted(by_type.items())},
        by_query_language={
            lang: {**_aggregate(rows, ks), "queries": len(rows)}
            for lang, rows in sorted(by_language.items())
        },
        per_query=per_query,
    )
