from ml.evaluation import reproduce


def test_check_reports_deviations_and_missing():
    summaries = [{"system": "a", "global": {"ndcg_at_10": 0.70, "mrr": 0.9, "recall_at_100": 0.5}},
                 {"system": "b", "global": {"ndcg_at_10": 0.60, "mrr": 0.8, "recall_at_100": 0.4}}]
    expected = {"a": {"ndcg_at_10": 0.705, "mrr": 0.9, "recall_at_100": 0.5}}
    problems = reproduce.check(summaries, expected, 0.01)
    assert problems == ["b: no recorded expectation"]
    expected["a"]["mrr"] = 0.95
    assert any("a mrr" in p for p in reproduce.check(summaries, expected, 0.01))
