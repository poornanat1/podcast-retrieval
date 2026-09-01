"""Run a retrieval evaluation from a checked-in experiment config.

    uv run python -m ml.evaluation.run --config experiments/eval/lexical-v1.json

Loads the relevance queries and qrels, runs the configured system against
the catalog database, computes retrieval metrics globally and per query
type, and logs parameters, metrics, and per-query results to MLflow
(``MLFLOW_TRACKING_URI``, default local compose server). ``--no-mlflow``
skips tracking for ad-hoc runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import psycopg

from ml.datasets.build import code_revision
from ml.datasets.snapshot import DEFAULT_DATABASE_URL
from ml.evaluation.harness import evaluate
from ml.retrieval import build_system

DEFAULT_MLFLOW_URI = "http://localhost:5001"


def load_queries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_qrels(path: Path) -> dict[str, dict[int, int]]:
    qrels: dict[str, dict[int, int]] = {}
    for line in path.read_text().splitlines():
        query_id, _, episode_id, grade = line.split()
        qrels.setdefault(query_id, {})[int(episode_id)] = int(grade)
    return qrels


def catalog_stats(conn: psycopg.Connection, head_share: float) -> tuple[int, dict, set]:
    """Catalog size, episode→podcast for all episodes, and the head-podcast
    set (top ``head_share`` of podcasts by episode count — a popularity
    proxy until real play data exists)."""
    with conn.cursor() as cur:
        cur.execute("SELECT podcast_id, count(*) FROM episodes GROUP BY podcast_id")
        counts = dict(cur.fetchall())
        cur.execute("SELECT id, podcast_id FROM episodes")
        episode_podcast = dict(cur.fetchall())
    catalog_size = sum(counts.values())
    head_n = max(1, int(len(counts) * head_share))
    head = set(sorted(counts, key=lambda p: (-counts[p], p))[:head_n])
    return catalog_size, episode_podcast, head


def run_config(config_path: str | Path, database_url: str, use_mlflow: bool) -> dict:
    """Evaluate one config; returns the summary dict (and logs to MLflow)."""
    config = json.loads(Path(config_path).read_text())
    relevance_dir = Path(config.get("relevance_dir", "data/relevance"))
    queries = load_queries(relevance_dir / "queries.jsonl")
    qrels = load_qrels(relevance_dir / "qrels.txt")

    with psycopg.connect(database_url) as conn:
        catalog_size, episode_podcast, head = catalog_stats(
            conn, config.get("head_podcast_share", 0.1)
        )
        system = build_system(config["system"], conn)
        report = evaluate(
            system,
            queries,
            qrels,
            ks=config.get("ks", [10, 50, 100]),
            min_relevant_grade=config.get("min_relevant_grade", 2),
            catalog_size=catalog_size,
            episode_podcast=episode_podcast,
            head_podcasts=head,
        )

    summary = {
        "system": report.system,
        "queries": report.queries,
        "skipped_queries": report.skipped_queries,
        "global": report.global_metrics,
        "by_query_type": report.by_query_type,
    }

    if use_mlflow:
        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI))
        mlflow.set_experiment(config.get("mlflow_experiment", "retrieval-eval"))
        qrels_hash = hashlib.sha256(
            (relevance_dir / "qrels.txt").read_bytes()
        ).hexdigest()[:12]
        with mlflow.start_run(run_name=f"{report.system}"):
            mlflow.log_params({
                "system": report.system,
                "config": Path(config_path).name,
                "config_sha256": hashlib.sha256(
                    json.dumps(config, sort_keys=True).encode()
                ).hexdigest()[:12],
                "qrels_sha256": qrels_hash,
                "catalog_size": catalog_size,
                "min_relevant_grade": config.get("min_relevant_grade", 2),
                **code_revision(),
            })
            mlflow.log_metrics(report.global_metrics)
            mlflow.log_metrics({"queries": report.queries,
                                "skipped_queries": report.skipped_queries})
            for query_type, values in report.by_query_type.items():
                mlflow.log_metrics({f"{query_type}__{k}": v for k, v in values.items()})
            mlflow.log_dict(summary, "summary.json")
            mlflow.log_dict({"per_query": report.per_query}, "per_query.json")
            summary["mlflow_experiment"] = config.get("mlflow_experiment", "retrieval-eval")

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    )
    args = parser.parse_args(argv)

    summary = run_config(args.config, args.database_url, use_mlflow=not args.no_mlflow)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
