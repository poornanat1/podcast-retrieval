"""Evaluate several configs and print a side-by-side comparison.

    uv run python -m ml.evaluation.compare experiments/eval/*.json

Each config is run (and logged to MLflow unless ``--no-mlflow``); the table
shows global metrics per system plus NDCG@10 by query type.
"""

from __future__ import annotations

import argparse
import os
import sys

from ml.datasets.snapshot import DEFAULT_DATABASE_URL
from ml.evaluation.run import run_config

GLOBAL_COLUMNS = [
    "recall_at_10", "hit_rate_at_10", "mrr", "ndcg_at_10",
    "tail_coverage", "latency_ms_p50", "latency_ms_p95",
]


def markdown_table(summaries: list[dict]) -> str:
    types = sorted({t for s in summaries for t in s["by_query_type"]})
    header = ["system", "queries"] + GLOBAL_COLUMNS + [f"ndcg@10 {t}" for t in types]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for s in summaries:
        cells = [s["system"], str(s["queries"])]
        cells += [f"{s['global'].get(c, float('nan')):.3f}" for c in GLOBAL_COLUMNS]
        cells += [
            f"{s['by_query_type'].get(t, {}).get('ndcg_at_10', float('nan')):.3f}"
            for t in types
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    )
    args = parser.parse_args(argv)

    summaries = [
        run_config(path, args.database_url, use_mlflow=not args.no_mlflow)
        for path in args.configs
    ]
    print(markdown_table(summaries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
