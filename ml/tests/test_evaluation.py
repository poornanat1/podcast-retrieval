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
        episodes_with_transcript={10, 20},
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

    # Item-side cohorts: transcript-bearing relevant items {10, 20} are both
    # retrieved by rank 10 (recall 1.0 for both queries); the no-transcript
    # partition is {11} (q1 only), also retrieved.
    assert g["recall_at_10__transcript"] == pytest.approx(1.0)
    assert g["recall_at_10__transcript__queries"] == 2
    assert g["recall_at_10__no_transcript"] == pytest.approx(1.0)
    # Head/tail: q1's relevant {10, 11} are head (podcast 1); q2's {20} is
    # tail (podcast 3, not in head set).
    assert g["recall_at_10__head__queries"] == 1
    assert g["recall_at_10__tail"] == pytest.approx(1.0)

    # Language grouping defaults to "en" when queries carry no language.
    assert set(report.by_query_language) == {"en"}
    assert report.by_query_language["en"]["queries"] == 2


def test_judged_share_counts_any_judgment_in_top_10() -> None:
    from ml.evaluation.harness import evaluate

    class Fixed:
        name = "fixed"

        def search(self, query, k):
            return [1, 2, 3, 4]

    queries = [{"query_id": "q", "query_type": "entity"}]
    # 1 relevant, 2 judged irrelevant, 3 and 4 unjudged -> 2 of 4 judged.
    report = evaluate(Fixed(), queries, {"q": {1: 3, 2: 0}}, ks=[10])
    assert report.per_query[0]["judged_share_at_10"] == 0.5
    assert report.global_metrics["judged_share_at_10"] == 0.5


def test_paired_bootstrap_detects_consistent_gain_and_null() -> None:
    from ml.evaluation.significance import compare, paired_bootstrap

    base = {f"q{i}": 0.5 for i in range(50)}
    better = {f"q{i}": 0.6 if i % 5 else 0.5 for i in range(50)}
    gain = paired_bootstrap(better, base, resamples=2000)
    assert gain["queries"] == 50 and gain["wins"] == 40 and gain["ties"] == 10
    assert gain["ci95"][0] > 0 and gain["p_not_better"] == 0.0
    null = paired_bootstrap(base, base, resamples=500)
    assert null["difference"] == 0 and null["p_not_better"] == 1.0

    rows = [{"query_id": "a", "query_type": "entity", "ndcg_at_10": 1.0},
            {"query_id": "b", "query_type": "navigational", "ndcg_at_10": 0.0}]
    flipped = [{**r, "ndcg_at_10": 1.0 - r["ndcg_at_10"]} for r in rows]
    report = compare(rows, flipped, "ndcg_at_10", by_type=True, resamples=200)
    assert report["entity"]["wins"] == 1 and report["navigational"]["losses"] == 1


def test_cold_start_cohorts_partition_by_training_evidence() -> None:
    from ml.evaluation.harness import cold_start_cohorts

    episode_podcast = {1: 10, 2: 10, 3: 20, 4: 30, 5: 30}
    cohorts = cold_start_cohorts({1, 4}, snapshot_episodes={1, 2, 3, 4},
                                 episode_podcast=episode_podcast)
    assert cohorts["seen"] == {1, 4}
    assert cohorts["known_podcast"] == {2, 5}
    assert cohorts["unseen_podcast"] == {3}
    assert cohorts["post_snapshot"] == {5}


def test_item_cohorts_add_recall_per_cohort() -> None:
    queries = [{"query_id": "q1", "query_type": "exploratory"},
               {"query_id": "q2", "query_type": "exploratory"}]
    qrels = {"q1": {10: 2, 11: 2}, "q2": {20: 2}}
    system = FakeSystem({"q1": [10, 99], "q2": [20]})
    report = evaluate(system, queries, qrels, ks=[10],
                      item_cohorts={"seen": {10, 20}, "unseen_podcast": {11}})
    g = report.global_metrics
    assert g["recall_at_10__seen"] == 1.0 and g["recall_at_10__seen__queries"] == 2
    assert g["recall_at_10__unseen_podcast"] == 0.0
    assert g["recall_at_10__unseen_podcast__queries"] == 1


def test_cohort_report_flags_regressions_beyond_tolerance() -> None:
    from ml.evaluation.cohort_report import render

    def summary(name, nav, ent, unseen):
        return {"system": name,
                "by_query_type": {"navigational": {"ndcg_at_10": nav, "queries": 40},
                                  "entity": {"ndcg_at_10": ent, "queries": 20}},
                "by_query_language": {"en": {"ndcg_at_10": 0.7, "queries": 172}},
                "global": {"recall_at_100__unseen_podcast": unseen,
                           "recall_at_100__unseen_podcast__queries": 127}}

    text = render([summary("base", 0.94, 0.58, 0.40), summary("model", 0.91, 0.40, 0.30)],
                  baselines=["base"], labels={"model": "champion"}, tolerance=0.05)
    assert "| champion | 0.910 | 0.400 ▼ |" in text      # navigational within tolerance
    assert "unseen_podcast (127)" in text and "0.300 ▼" in text
    assert "champion vs base: entity (query type) 0.400 vs 0.580 (-0.180)" in text
    assert "navigational (query type)" not in text.split("Regressions")[1]
    assert render([summary("base", 0.9, 0.5, 0.4)], ["base"], {}).rstrip().endswith("- none")


def test_cohort_report_never_flags_tiny_cohorts() -> None:
    from ml.evaluation.cohort_report import render

    def summary(name, pt):
        return {"system": name, "by_query_type": {},
                "by_query_language": {"pt": {"ndcg_at_10": pt, "queries": 1}}, "global": {}}

    text = render([summary("base", 0.9), summary("model", 0.3)], ["base"], {})
    assert "▼" not in text and text.rstrip().endswith("- none")
