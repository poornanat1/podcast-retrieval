"""Evaluation tests: metric math against hand-computed values, and the
harness end to end with a fake retrieval system."""

import math

import pytest

from ml.evaluation import metrics
from ml.evaluation.harness import evaluate


def test_recall_and_hit_rate() -> None:
    ranked = [1, 2, 3, 4, 5]
    relevant = {2, 5, 9}
    assert metrics.recall_at_k(ranked, relevant, 3) == pytest.approx(1 / 3)
    assert metrics.recall_at_k(ranked, relevant, 5) == pytest.approx(2 / 3)
    assert metrics.hit_rate_at_k(ranked, relevant, 1) == 0.0
    assert metrics.hit_rate_at_k(ranked, relevant, 2) == 1.0
    with pytest.raises(ValueError):
        metrics.recall_at_k(ranked, set(), 5)


def test_reciprocal_rank() -> None:
    assert metrics.reciprocal_rank([7, 8, 9], {9}) == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank([7, 8, 9], {7, 9}) == 1.0
    assert metrics.reciprocal_rank([7, 8], {1}) == 0.0


def test_ndcg_hand_computed() -> None:
    grades = {1: 3, 2: 2, 3: 0}
    # Ranked [3, 1, 2]: DCG = 0 + 7/log2(3) + 3/log2(4); ideal = 7 + 3/log2(3)
    dcg = 7 / math.log2(3) + 3 / math.log2(4)
    ideal = 7 + 3 / math.log2(3)
    assert metrics.ndcg_at_k([3, 1, 2], grades, 10) == pytest.approx(dcg / ideal)
    # Perfect ordering scores 1.
    assert metrics.ndcg_at_k([1, 2, 3], grades, 10) == pytest.approx(1.0)
    # No graded documents at all: 0.
    assert metrics.ndcg_at_k([1, 2], {}, 10) == 0.0


def test_percentile_nearest_rank() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert metrics.percentile(values, 50) == 20.0
    assert metrics.percentile(values, 95) == 40.0
    assert metrics.percentile([], 50) == 0.0


class FakeSystem:
    name = "fake"

    def __init__(self, results: dict[str, list[int]]):
        self.results = results

    def search(self, query: dict, k: int) -> list[int]:
        return self.results.get(query["query_id"], [])[:k]


def test_harness_end_to_end() -> None:
    queries = [
        {"query_id": "q1", "query_type": "navigational"},
        {"query_id": "q2", "query_type": "exploratory"},
        {"query_id": "q3", "query_type": "exploratory"},  # no relevant → skipped
    ]
    qrels = {
        "q1": {10: 3, 11: 2, 12: 0},
        "q2": {20: 2, 21: 1},
        "q3": {30: 0, 31: 1},
    }
    system = FakeSystem({
        "q1": [10, 11, 99],  # both relevant found, ranks 1-2
        "q2": [99, 98, 20],  # single relevant at rank 3
        "q3": [1, 2, 3],
    })

    report = evaluate(
        system, queries, qrels, ks=[2, 10],
        catalog_size=100,
        episode_podcast={10: 1, 11: 1, 99: 2, 98: 2, 20: 3},
        head_podcasts={1},
    )

    assert report.queries == 2 and report.skipped_queries == 1
    g = report.global_metrics
    assert g["recall_at_2"] == pytest.approx((1.0 + 0.0) / 2)
    assert g["recall_at_10"] == pytest.approx(1.0)
    assert g["hit_rate_at_2"] == pytest.approx(0.5)
    assert g["mrr"] == pytest.approx((1.0 + 1 / 3) / 2, abs=1e-4)
    # 5 distinct retrieved episodes over catalog of 100.
    assert g["catalog_coverage"] == pytest.approx(0.05)
    # q1: [10, 11] head, 99 tail; q2: 99, 98, 20 tail → 4 tail of 6 results.
    assert g["tail_coverage"] == pytest.approx(4 / 6, abs=1e-4)
    assert g["latency_ms_p95"] >= g["latency_ms_p50"] >= 0

    assert set(report.by_query_type) == {"navigational", "exploratory"}
    assert report.by_query_type["navigational"]["mrr"] == 1.0
    assert report.by_query_type["exploratory"]["mrr"] == pytest.approx(1 / 3, abs=1e-4)
