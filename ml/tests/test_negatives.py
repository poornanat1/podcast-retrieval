"""Hard-negative audit, shared-pool test-split scoring, and acceptance gates."""

import json

import pandas as pd
import torch

from ml.evaluation.test_split import build_pool, score
from ml.features import FeatureContract, Featurizer
from ml.hard_negatives.audit import audit, sample_pairs, summarize
from ml.models.two_tower import TwoTowerConfig, TwoTowerModel
from ml.registry.register import evaluate_gates
from ml.relevance.llm_judge import EpisodeGrade
from ml.tests.helpers import tiny_backbone, tiny_tokenizer
from ml.tests.test_build import write_snapshot
from ml.training.data import load_episode_rows


def test_audit_samples_grades_and_summarizes() -> None:
    queries = [{"episode_id": i, "query_text": f"query {i}"} for i in range(1, 5)]
    negatives = {
        1: [{"episode_id": 11, "sources": [{"system": "vector-e5-small", "rank": 3}]},
            {"episode_id": 12, "sources": [{"system": "lexical-fts", "rank": 15},
                                           {"system": "vector-e5-small", "rank": 8}]}],
        2: [{"episode_id": 21, "sources": [{"system": "lexical-fts", "rank": 1}]}],
        3: [],
    }
    pairs = sample_pairs(queries, negatives, sample_queries=10, per_query=5, hard_pool=20, seed=1)
    assert {p["query_episode_id"] for p in pairs} == {1, 2}  # 3 has no negatives, 4 none at all
    twelve = next(p for p in pairs if p["episode_id"] == 12)
    assert twelve["rank"] == 8 and twelve["sources"] == ["lexical-fts", "vector-e5-small"]

    rows = {e: {"podcast_title": "Show", "episode_title": f"Ep {e}", "description": "d"}
            for e in (11, 12)}  # 21 missing -> skipped
    calls = []

    def grader(tasks):
        calls.append([t["episode_id"] for t in tasks])
        return [EpisodeGrade(episode_id=t["episode_id"], grade=2 if t["episode_id"] == 11 else 0,
                             rationale="r") for t in tasks]

    graded = audit(pairs, rows, grader)
    assert sorted(c for call in calls for c in call) == [11, 12]
    summary = summarize(graded)
    assert summary["overall"] == {"pairs": 2, "relevant_share": 0.5}
    assert summary["by_rank"]["1-5"]["relevant_share"] == 1.0
    assert summary["by_source"]["lexical-fts+vector-e5-small"]["relevant_share"] == 0.0


def test_shared_pool_test_split_scoring(tmp_path) -> None:
    root = write_snapshot(tmp_path, n_episodes=30)
    test_df = pd.DataFrame({"query_text": ["topic number 1", "topic number 2"],
                            "positive_episode_id": [1, 2]})
    negatives = {1: [{"episode_id": 5, "sources": []}, {"episode_id": 6, "sources": []}],
                 2: [{"episode_id": 7, "sources": []}]}
    queries, pool = build_pool(test_df, negatives, hard_pool=20)
    assert pool == [1, 2, 5, 6, 7]
    rows = load_episode_rows(root, set(pool))
    contract = FeatureContract(languages=("en", "fr"), categories=("history", "technology"),
                               max_metadata_tokens=32, max_query_tokens=16)
    featurizer = Featurizer(contract, tiny_tokenizer(tmp_path))
    torch.manual_seed(0)
    model = TwoTowerModel(TwoTowerConfig(embedding_dim=32, hidden_dim=16, dropout=0.0),
                          contract, tiny_backbone())
    result = score(model, featurizer, queries, rows, pool, "cpu")
    assert result["queries"] == 2 and result["pool"] == 5
    assert 0 < result["mrr"] <= 1 and result["recall_at_100"] == 1.0
    assert result["recall_at_1"] <= result["recall_at_10"] <= result["recall_at_100"]


def test_acceptance_gates() -> None:
    baseline = {
        "system": "b",
        "global": {"ndcg_at_10": 0.70, "recall_at_100": 0.40, "recall_at_100__tail": 0.40,
                   "recall_at_100__tail__queries": 9},
        "by_query_type": {"entity": {"ndcg_at_10": 0.60}},
    }
    winner = {"system": "c", "global": {"ndcg_at_10": 0.72, "recall_at_100": 0.55,
                                         "recall_at_100__tail": 0.38},
              "by_query_type": {"entity": {"ndcg_at_10": 0.57}}}
    result = evaluate_gates(winner, baseline)
    assert result.passed and len(result.checks) == 4  # 2 primary + 1 type + 1 cohort
    loser = json.loads(json.dumps(winner))
    loser["global"]["ndcg_at_10"] = 0.63
    loser["by_query_type"]["entity"]["ndcg_at_10"] = 0.40
    result = evaluate_gates(loser, baseline)
    assert not result.passed
    failed = [c["metric"] for c in result.checks if not c["passed"]]
    assert failed == ["ndcg_at_10", "entity__ndcg_at_10"]
    assert evaluate_gates(loser, baseline, cohort_tolerance=0.3).checks[2]["passed"]
