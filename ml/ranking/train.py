"""Train and evaluate the second-stage ranker with query-grouped cross-validation.

    uv run python -m ml.ranking.train --config experiments/ranking/ranker-v1.json

The only graded labels are the relevance set's 180 queries, so every number
this reports is out of fold: queries are split into ``folds`` groups,
stratified by query type, and each query is scored by a model that never
saw it. Hyperparameters are fixed in the config before training and never
tuned against these folds.

Steps:

1. Candidate sets for every relevance query (``ml.ranking.candidates``),
   cached in ``<output>/candidates.parquet`` (the slow, database-bound
   part; delete the file to rebuild).
2. Features (``ml.ranking.features``) and labels: the judged grade, or 0
   for an unjudged candidate. The report states the judged share.
3. One LightGBM LambdaRank model per fold, plus a final model on all
   queries for serving.
4. Every ordering of the same candidates is scored per query: the ranker
   (out of fold), RRF over the three legs, and each leg alone. Metrics are
   NDCG@5/@10, MAP@10, P@5/@10, MRR, and pairwise accuracy, globally and by
   query type, with a paired bootstrap of the ranker against RRF.

Writes ``<output_dir>/<name>-<config hash>/``: ``model.txt`` (all
queries), ``fold_<k>.txt``, ``folds.json`` (query id to fold),
``ranker.json`` (features, config, data hashes, code revision),
``metrics.json``, and ``per_query.jsonl``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.evaluation import metrics
from ml.evaluation.significance import paired_bootstrap
from ml.ranking.lightgbm_runtime import LIGHTGBM_THREADS, load_lightgbm

ORDERINGS = ("ranker", "rrf", "two_tower", "e5", "lexical")
RANKING_METRICS = ("ndcg_at_5", "ndcg_at_10", "map_at_10", "precision_at_5",
                   "precision_at_10", "mrr", "pairwise_accuracy", "candidate_recall")


@dataclasses.dataclass(frozen=True)
class RankerConfig:
    name: str
    two_tower_artifact: str
    reference_time: str
    depth: int = 50
    folds: int = 5
    seed: int = 0
    relevance_dir: str = "data/relevance"
    output_dir: str = "data/models"
    min_relevant_grade: int = 2
    params: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> RankerConfig:
        return cls(**json.loads(Path(path).read_text()))

    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(dataclasses.asdict(self), sort_keys=True).encode()).hexdigest()


def assign_folds(queries: list[dict], folds: int, seed: int) -> dict[str, int]:
    """Stratified by query type: within each type, queries in a seeded order
    are dealt round-robin to the folds."""
    out: dict[str, int] = {}
    by_type: dict[str, list[str]] = defaultdict(list)
    for q in queries:
        by_type[q["query_type"]].append(q["query_id"])
    offset = 0
    for qtype in sorted(by_type):
        ordered = sorted(by_type[qtype], key=lambda i: hashlib.sha256(
            f"{seed}:{i}".encode()).hexdigest())
        for i, qid in enumerate(ordered):
            out[qid] = (i + offset) % folds
        offset += len(ordered)
    return out


def order_by(scores: np.ndarray, episode_ids: list[int]) -> list[int]:
    """Descending score, ties broken by episode id for determinism."""
    return [episode_ids[i] for i in sorted(range(len(episode_ids)),
                                           key=lambda i: (-scores[i], episode_ids[i]))]


def query_metrics(ranked: list[int], grades: dict[int, int], min_grade: int) -> dict:
    relevant = {e for e, g in grades.items() if g >= min_grade}
    return {
        "ndcg_at_5": metrics.ndcg_at_k(ranked, grades, 5),
        "ndcg_at_10": metrics.ndcg_at_k(ranked, grades, 10),
        "map_at_10": metrics.average_precision_at_k(ranked, relevant, 10),
        "precision_at_5": metrics.precision_at_k(ranked, relevant, 5),
        "precision_at_10": metrics.precision_at_k(ranked, relevant, 10),
        "mrr": metrics.reciprocal_rank(ranked, relevant),
        "pairwise_accuracy": metrics.pairwise_accuracy(ranked, grades),
        "candidate_recall": len(set(ranked) & relevant) / len(relevant),
    }


def summarize(rows: list[dict]) -> dict[str, float]:
    out = {}
    for key in RANKING_METRICS:
        values = [r[key] for r in rows if r[key] is not None]
        out[key] = round(float(np.mean(values)), 4) if values else None
    return out


def build_candidates(config: RankerConfig, queries: list[dict], cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_parquet(cache)
    import psycopg

    from ml.datasets.snapshot import DEFAULT_DATABASE_URL
    from ml.ranking.candidates import CandidateGenerator

    records = []
    started = time.time()
    with psycopg.connect(os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)) as conn:
        generator = CandidateGenerator(conn, config.two_tower_artifact, config.depth)
        for n, query in enumerate(queries, 1):
            cands = generator.generate(query)
            for row in cands.rows:
                records.append({"query_id": query["query_id"], **row,
                                **{f"timing_{k}": v for k, v in cands.timings_ms.items()}})
            if n % 20 == 0:
                print(f"candidates {n}/{len(queries)} ({time.time() - started:.0f}s)",
                      flush=True)
    df = pd.DataFrame.from_records(records)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def features_for(df: pd.DataFrame, queries: dict[str, dict],
                 reference_time: datetime) -> dict[str, tuple[list[int], np.ndarray]]:
    from ml.ranking.candidates import CandidateSet
    from ml.ranking.features import featurize

    out = {}
    for qid, part in df.groupby("query_id", sort=True):
        rows = []
        for r in part.to_dict("records"):
            for leg in ("lexical", "e5", "two_tower"):
                v = r.get(f"{leg}_rank")
                r[f"{leg}_rank"] = None if v is None or pd.isna(v) else int(v)
            published = r.get("published_at")
            r["published_at"] = None if published is None or pd.isna(published) \
                else pd.Timestamp(published).to_pydatetime()
            rows.append(r)
        cands = CandidateSet(query=queries[qid], rows=rows, timings_ms={})
        out[qid] = (cands.episode_ids, featurize(cands, reference_time))
    return out


def train(config: RankerConfig, use_mlflow: bool = True) -> tuple[Path, dict]:
    lgb = load_lightgbm()
    from ml.datasets.build import code_revision
    from ml.evaluation.run import load_qrels, load_queries
    from ml.ranking.features import FEATURE_VERSION, FEATURES, lexical_score, rrf_score

    relevance = Path(config.relevance_dir)
    queries = {q["query_id"]: q for q in load_queries(relevance / "queries.jsonl")}
    qrels = load_qrels(relevance / "qrels.txt")
    usable = [q for qid, q in sorted(queries.items())
              if any(g >= config.min_relevant_grade for g in qrels.get(qid, {}).values())]
    out_dir = Path(config.output_dir) / f"{config.name}-{config.sha256()[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_candidates(config, usable, out_dir / "candidates.parquet")
    reference_time = datetime.fromisoformat(config.reference_time)
    feats = features_for(df, queries, reference_time)
    folds = assign_folds(usable, config.folds, config.seed)

    labels = {qid: np.array([qrels.get(qid, {}).get(e, 0) for e in feats[qid][0]],
                            dtype=np.int32) for qid in feats}
    judged = sum(sum(1 for e in feats[qid][0] if e in qrels.get(qid, {})) for qid in feats)
    total = sum(len(feats[qid][0]) for qid in feats)

    def stack(qids: list[str]) -> tuple[np.ndarray, np.ndarray, list[int]]:
        return (np.vstack([feats[q][1] for q in qids]),
                np.concatenate([labels[q] for q in qids]),
                [len(feats[q][0]) for q in qids])

    params = {"objective": "lambdarank", "random_state": config.seed, "verbose": -1,
              **config.params, "num_threads": LIGHTGBM_THREADS}
    oof: dict[str, np.ndarray] = {}
    for k in range(config.folds):
        train_q = [q for q in feats if folds[q] != k]
        test_q = [q for q in feats if folds[q] == k]
        x, y, groups = stack(train_q)
        model = lgb.LGBMRanker(**params).fit(x, y, group=groups)
        model.booster_.save_model(str(out_dir / f"fold_{k}.txt"))
        for q in test_q:
            oof[q] = model.predict(feats[q][1], num_threads=LIGHTGBM_THREADS)
    x, y, groups = stack(list(feats))
    final = lgb.LGBMRanker(**params).fit(x, y, group=groups)
    final.booster_.save_model(str(out_dir / "model.txt"))

    by_ordering: dict[str, list[dict]] = defaultdict(list)
    per_query = []
    for qid in sorted(feats):
        ids, _ = feats[qid]
        part = df[df.query_id == qid].set_index("episode_id").loc[ids]
        rows = part.reset_index().to_dict("records")
        for r in rows:
            for leg in ("lexical", "e5", "two_tower"):
                v = r.get(f"{leg}_rank")
                r[f"{leg}_rank"] = None if v is None or pd.isna(v) else int(v)
        legs = np.array([sum(1 for leg in ("lexical", "e5", "two_tower") if r[f"{leg}_rank"])
                         for r in rows], dtype=float)
        scores = {
            "ranker": oof[qid],
            "rrf": np.array([rrf_score(r) for r in rows]) + 1e-9 * legs,
            "two_tower": part["two_tower_sim"].to_numpy(),
            "e5": part["e5_sim"].to_numpy(),
            "lexical": np.array([lexical_score(r) for r in rows]),
        }
        record = {"query_id": qid, "query_type": queries[qid]["query_type"],
                  "fold": folds[qid]}
        for name in ORDERINGS:
            m = query_metrics(order_by(scores[name], ids), qrels[qid], config.min_relevant_grade)
            by_ordering[name].append({"query_type": record["query_type"], **m})
            record[name] = m
        per_query.append(record)

    report: dict[str, Any] = {"queries": len(feats), "candidates": total,
                              "judged_share": round(judged / total, 4),
                              "mean_candidates": round(total / len(feats), 1),
                              "orderings": {}}
    for name, rows in by_ordering.items():
        types = sorted({r["query_type"] for r in rows})
        report["orderings"][name] = {
            "global": summarize(rows),
            "by_query_type": {t: summarize([r for r in rows if r["query_type"] == t])
                              for t in types},
        }
    report["ranker_vs_rrf"] = {}
    for key in ("ndcg_at_5", "ndcg_at_10", "map_at_10", "mrr"):
        report["ranker_vs_rrf"][key] = paired_bootstrap(
            {r["query_id"]: r["ranker"][key] for r in per_query},
            {r["query_id"]: r["rrf"][key] for r in per_query})
    importance = final.booster_.feature_importance(importance_type="gain")
    report["feature_importance_gain"] = {
        name: round(float(v), 1) for name, v in
        sorted(zip(FEATURES, importance, strict=True), key=lambda kv: -kv[1])}
    timing_cols = [c for c in df.columns if c.startswith("timing_")]
    per_query_timing = df.groupby("query_id")[timing_cols].first()
    report["candidate_timings_ms"] = {
        c.removeprefix("timing_"): {"p50": round(float(per_query_timing[c].quantile(0.5)), 1),
                                    "p95": round(float(per_query_timing[c].quantile(0.95)), 1)}
        for c in timing_cols}
    started = time.perf_counter()
    for qid in feats:
        final.predict(feats[qid][1], num_threads=LIGHTGBM_THREADS)
    report["score_ms_mean"] = round((time.perf_counter() - started) * 1000 / len(feats), 2)

    (out_dir / "folds.json").write_text(json.dumps(folds, indent=1, sort_keys=True) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out_dir / "per_query.jsonl").open("w") as f:
        for record in per_query:
            f.write(json.dumps(record) + "\n")
    meta = {
        "name": config.name, "config": dataclasses.asdict(config),
        "config_sha256": config.sha256(), "feature_version": FEATURE_VERSION,
        "features": list(FEATURES),
        "qrels_sha256": hashlib.sha256((relevance / "qrels.txt").read_bytes()).hexdigest(),
        "created_at": datetime.now().astimezone().isoformat(), **code_revision(),
    }
    (out_dir / "ranker.json").write_text(json.dumps(meta, indent=2) + "\n")

    if use_mlflow:
        import mlflow

        from ml.evaluation.run import DEFAULT_MLFLOW_URI

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI))
        mlflow.set_experiment("ranker-training")
        with mlflow.start_run(run_name=config.name):
            mlflow.log_params({"config_sha256": config.sha256()[:12], "depth": config.depth,
                               "folds": config.folds, "two_tower_artifact":
                               Path(config.two_tower_artifact).name,
                               "feature_version": FEATURE_VERSION, **config.params})
            for name, values in report["orderings"].items():
                mlflow.log_metrics({f"{name}__{k}": v for k, v in values["global"].items()
                                    if v is not None})
            mlflow.log_dict(report, "metrics.json")
    return out_dir, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)
    out_dir, report = train(RankerConfig.load(args.config), use_mlflow=not args.no_mlflow)
    summary = {name: values["global"] for name, values in report["orderings"].items()}
    print(json.dumps({"artifact": str(out_dir), "judged_share": report["judged_share"],
                      "mean_candidates": report["mean_candidates"], "global": summary,
                      "ranker_vs_rrf": report["ranker_vs_rrf"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
