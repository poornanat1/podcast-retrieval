import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ml.ranking.candidates import CandidateSet
from ml.ranking.features import FEATURES, featurize, lexical_score, rrf_score
from ml.ranking.train import RankerConfig, assign_folds, order_by, train

REF = datetime(2026, 10, 21, tzinfo=UTC)


def row(episode_id, *, show="The Rest Is History", publisher="Goalhanger",
        title="Napoleon at Waterloo", tt=0.8, e5=0.7, lex=(0.1, 0.0, 0.2),
        ranks=(1, 2, 3), transcript=0.0, days_old=10, has_transcript=False):
    return {
        "episode_id": episode_id, "podcast_id": 1, "podcast_title": show,
        "publisher": publisher, "title": title, "description": "the battle of waterloo",
        "language": "en", "duration_seconds": 3600,
        "published_at": datetime(2026, 10, 21, tzinfo=UTC) - timedelta(days=days_old),
        "popularity": 0.5, "show_episodes": 400, "has_transcript": has_transcript,
        "lex_episode": lex[0], "lex_transcript": lex[1], "lex_podcast": lex[2],
        "two_tower_sim": tt, "e5_sim": e5, "transcript_sim": transcript,
        "lexical_rank": ranks[0], "e5_rank": ranks[1], "two_tower_rank": ranks[2],
    }


def test_featurize_matches_sources_and_relative_scores() -> None:
    cands = CandidateSet(
        query={"query_id": "q", "query_text": "the rest is history"},
        rows=[row(1), row(2, show="History Hit", publisher="Hit Network", tt=0.6, e5=0.75,
                          lex=(0.0, 0.0, 0.0), ranks=(None, 5, None), title="rest day")],
        timings_ms={})
    x = featurize(cands, REF)
    f = {name: x[:, i] for i, name in enumerate(FEATURES)}
    assert x.shape == (2, len(FEATURES)) and x.dtype == np.float32
    assert list(f["show_exact"]) == [1.0, 0.0]
    assert list(f["query_in_show"]) == [1.0, 0.0]
    assert f["show_coverage"][1] == pytest.approx(0.25)    # "history" of four tokens
    assert f["two_tower_sim_gap"][0] == 0 and f["two_tower_sim_gap"][1] == pytest.approx(-0.2)
    assert f["e5_sim_gap"][1] == 0
    assert f["lex_score"][0] == pytest.approx(0.2) and f["lex_score_rel"][1] == 0
    assert list(f["legs"]) == [3.0, 1.0]
    assert f["lexical_rr"][1] == 0 and f["e5_rr"][1] == pytest.approx(0.2)
    assert f["rrf_score"][0] == pytest.approx(1 / 61 + 1 / 62 + 1 / 63)
    assert f["log_age_days"][0] == pytest.approx(np.log1p(10), rel=1e-5)
    assert featurize(CandidateSet({"query_text": "x"}, [], {}), REF).shape == (0, len(FEATURES))


def test_scores_and_order_are_deterministic() -> None:
    r = row(1, lex=(0.2, 0.1, 0.4), ranks=(None, 1, 2))
    assert lexical_score(r) == pytest.approx(0.5)
    assert rrf_score(r) == pytest.approx(1 / 61 + 1 / 62)
    assert order_by(np.array([0.5, 0.9, 0.5]), [30, 10, 20]) == [10, 20, 30]


def test_folds_are_stratified_seeded_and_balanced() -> None:
    queries = [{"query_id": f"n{i}", "query_type": "navigational"} for i in range(10)] + \
              [{"query_id": f"e{i}", "query_type": "exploratory"} for i in range(15)]
    folds = assign_folds(queries, 5, seed=1)
    assert folds == assign_folds(queries, 5, seed=1) != assign_folds(queries, 5, seed=2)
    for prefix, per_fold in (("n", 2), ("e", 3)):
        counts = np.bincount([f for q, f in folds.items() if q.startswith(prefix)], minlength=5)
        assert list(counts) == [per_fold] * 5


def test_train_cross_validates_on_cached_candidates(tmp_path) -> None:
    rng = np.random.default_rng(0)
    rel = tmp_path / "relevance"
    rel.mkdir()
    types = ["navigational", "exploratory"]
    queries = [{"query_id": f"q{i}", "query_text": f"topic {i}", "query_type": types[i % 2]}
               for i in range(20)]
    (rel / "queries.jsonl").write_text("".join(json.dumps(q) + "\n" for q in queries))
    records, qrels = [], []
    for q in queries:
        for e in range(30):
            episode = int(q["query_id"][1:]) * 100 + e
            good = e < 5
            r = row(episode, tt=float(rng.uniform(0.6, 0.7) + (0.2 if good else 0)),
                    e5=float(rng.uniform(0.6, 0.9)), ranks=(None, e + 1, None),
                    title=f"episode {e}")
            records.append({"query_id": q["query_id"], **r, "timing_total_ms": 5.0})
            if e < 20:
                qrels.append(f"{q['query_id']} 0 {episode} {3 if good else 0}\n")
    (rel / "qrels.txt").write_text("".join(qrels))
    config = RankerConfig(name="t", two_tower_artifact="unused", reference_time=REF.isoformat(),
                          folds=4, seed=3, relevance_dir=str(rel),
                          output_dir=str(tmp_path / "models"),
                          params={"n_estimators": 20, "num_leaves": 4, "min_child_samples": 5})
    out = tmp_path / "models" / f"t-{config.sha256()[:8]}"
    out.mkdir(parents=True)
    pd.DataFrame.from_records(records).to_parquet(out / "candidates.parquet", index=False)

    out_dir, report = train(config, use_mlflow=False)
    assert out_dir == out
    for name in ("model.txt", "fold_0.txt", "fold_3.txt", "folds.json", "ranker.json",
                 "metrics.json", "per_query.jsonl"):
        assert (out_dir / name).exists(), name
    meta = json.loads((out_dir / "ranker.json").read_text())
    assert meta["features"] == list(FEATURES)
    assert report["queries"] == 20 and report["judged_share"] == pytest.approx(20 / 30, abs=1e-4)
    ranker = report["orderings"]["ranker"]["global"]
    # The two-tower similarity separates relevant items here; the out-of-fold
    # ranker must learn that and beat the e5 ordering, which is noise.
    assert ranker["ndcg_at_10"] > report["orderings"]["e5"]["global"]["ndcg_at_10"]
    assert ranker["candidate_recall"] == 1.0
    assert set(report["ranker_vs_rrf"]) == {"ndcg_at_5", "ndcg_at_10", "map_at_10", "mrr"}
    lines = (out_dir / "per_query.jsonl").read_text().splitlines()
    assert len(lines) == 20 and {"ranker", "rrf", "fold"} <= set(json.loads(lines[0]))


def test_torch_and_lightgbm_share_a_process_safely() -> None:
    """The sequence the ranker runs (torch CPU work, LightGBM training,
    prediction, and loading a saved model, more torch CPU work) in a fresh
    process that imports through ``ml.ranking``. Without the package's
    OpenMP handling this segfaults on macOS."""
    import subprocess
    import sys

    script = """
from ml.ranking.lightgbm_runtime import LIGHTGBM_THREADS, load_lightgbm
lgb = load_lightgbm()
import numpy as np
import torch
x = torch.randn(32, 256, 384)
torch.nn.functional.layer_norm(x, (384,)).softmax(-1).sum()
X = np.random.rand(2000, 30).astype(np.float32)
m = lgb.LGBMRanker(n_estimators=10, verbose=-1, num_threads=LIGHTGBM_THREADS)
m.fit(X, np.random.randint(0, 4, 2000), group=[50] * 40)
m.predict(X[:10], num_threads=LIGHTGBM_THREADS)
torch.nn.functional.layer_norm(x, (384,)).softmax(-1).sum()
import os, tempfile
path = os.path.join(tempfile.mkdtemp(), "model.txt")
m.booster_.save_model(path)
b = lgb.Booster(model_file=path)
b.predict(X[:10], num_threads=LIGHTGBM_THREADS)
torch.nn.functional.layer_norm(x, (384,)).softmax(-1).sum()
print("ok")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            timeout=300)
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr[-2000:]


def test_feature_definitions_import_without_torch_or_lightgbm() -> None:
    """The model server imports the shared features and must not pull in
    PyTorch or LightGBM."""
    import subprocess
    import sys

    script = ("import sys, ml.ranking.features, ml.ranking.types; "
              "print(sorted(m for m in ('torch', 'lightgbm') if m in sys.modules))")
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0 and result.stdout.strip() == "[]", result.stderr[-2000:]


def test_missing_numbers_featurize_the_same_from_json_and_pandas() -> None:
    """Serving sends nulls; training reads Parquet, where nulls become NaN.
    Both must give the same feature values."""
    base = row(1)
    as_null = {**base, "duration_seconds": None, "popularity": None, "show_episodes": None}
    as_nan = {**base, "duration_seconds": float("nan"), "popularity": float("nan"),
              "show_episodes": float("nan")}
    query = {"query_text": "the rest is history"}
    a = featurize(CandidateSet(query, [as_null], {}), REF)
    b = featurize(CandidateSet(query, [as_nan], {}), REF)
    assert np.array_equal(a, b) and not np.isnan(b).any()
