"""Paired bootstrap over per-query metrics of two evaluation runs.

    uv run python -m ml.evaluation.significance --candidate <run id> --baseline <run id> \
        [--metric ndcg_at_10] [--by-type] [--resamples 10000]

Each evaluation run logs ``per_query.json`` to MLflow. Both runs must cover
the same queries on the same judgment pool. Queries are resampled with
replacement, and the mean per-query difference (candidate minus baseline)
is recomputed each time. The report gives the observed difference, a 95%
percentile interval, and the share of resamples at or below zero (a
one-sided p-value). Win, tie, and loss counts are per query.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def paired_bootstrap(candidate: dict[str, float], baseline: dict[str, float],
                     resamples: int = 10_000, seed: int = 0) -> dict:
    """Bootstrap the mean paired difference over the queries both runs scored."""
    ids = sorted(set(candidate) & set(baseline))
    if not ids:
        raise ValueError("no shared queries")
    diff = np.array([candidate[i] - baseline[i] for i in ids])
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(resamples, len(diff)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "queries": len(ids),
        "difference": round(float(diff.mean()), 4),
        "ci95": [round(float(low), 4), round(float(high), 4)],
        "p_not_better": round(float((means <= 0).mean()), 4),
        "wins": int((diff > 0).sum()),
        "ties": int((diff == 0).sum()),
        "losses": int((diff < 0).sum()),
    }


def load_per_query(run_id: str, tracking_uri: str | None = None) -> list[dict]:
    path = Path(run_id)
    if path.exists():
        return json.loads(path.read_text())["per_query"]
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return mlflow.artifacts.load_dict(f"runs:/{run_id}/per_query.json")["per_query"]


def compare(candidate_rows: list[dict], baseline_rows: list[dict], metric: str,
            by_type: bool = False, resamples: int = 10_000) -> dict:
    def scores(rows, query_type=None):
        return {r["query_id"]: float(r[metric]) for r in rows
                if query_type is None or r["query_type"] == query_type}

    report = {"metric": metric,
              "all": paired_bootstrap(scores(candidate_rows), scores(baseline_rows), resamples)}
    if by_type:
        for t in sorted({r["query_type"] for r in candidate_rows}):
            report[t] = paired_bootstrap(scores(candidate_rows, t), scores(baseline_rows, t),
                                         resamples)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", required=True, help="MLflow run id or per_query.json")
    parser.add_argument("--baseline", required=True, help="MLflow run id or per_query.json")
    parser.add_argument("--metric", default="ndcg_at_10")
    parser.add_argument("--by-type", action="store_true")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args(argv)
    report = compare(load_per_query(args.candidate, args.tracking_uri),
                     load_per_query(args.baseline, args.tracking_uri),
                     args.metric, args.by_type, args.resamples)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
