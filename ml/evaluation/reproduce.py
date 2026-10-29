"""One command that reproduces the headline experiment and checks it.

    uv run python -m ml.evaluation.reproduce [--record] [--tolerance 0.01]

Evaluates the systems the reports compare (lexical, pretrained E5, RRF
over three legs, the two-tower champion, and the learned ranker) on the
relevance set against the frozen catalog, with MLflow logging off, and
compares NDCG@10, MRR and Recall@100 with the values recorded in
``experiments/eval/reproduce-expected.json``. Exit 0 when every number is
within the tolerance, 1 when one is not, and 2 when a system fails to run.
``--record`` writes the current numbers as the expectation instead.

Reproducibility here means the same catalog snapshot, indexes, artifacts
and code give the same evaluation; the systems read the live database, so
the catalog must be the frozen corpus the reports describe
(docs/data-card.md).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ml.datasets.snapshot import DEFAULT_DATABASE_URL
from ml.evaluation.compare import markdown_table
from ml.evaluation.run import run_config

CONFIGS = [
    "experiments/eval/lexical-v1.json",
    "experiments/eval/vector-e5small-v1.json",
    "experiments/eval/hybrid-rrf3-v1.json",
    "experiments/eval/two-tower-names-v1.json",
    "experiments/eval/reranker-v1.json",
]
METRICS = ("ndcg_at_10", "mrr", "recall_at_100")
EXPECTED = Path("experiments/eval/reproduce-expected.json")


def check(summaries: list[dict], expected: dict, tolerance: float) -> list[str]:
    problems = []
    for s in summaries:
        want = expected.get(s["system"])
        if want is None:
            problems.append(f"{s['system']}: no recorded expectation")
            continue
        for metric in METRICS:
            got, exp = s["global"].get(metric), want.get(metric)
            if got is None or exp is None:
                problems.append(f"{s['system']} {metric}: missing")
            elif abs(got - exp) > tolerance:
                problems.append(f"{s['system']} {metric}: {got:.3f}, expected {exp:.3f} "
                                f"(tolerance {tolerance})")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--database-url",
                        default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL))
    args = parser.parse_args(argv)

    summaries = []
    for path in CONFIGS:
        try:
            summaries.append(run_config(path, args.database_url, use_mlflow=False))
        except Exception as err:  # noqa: BLE001 - report which system failed
            print(f"{path}: {err}", file=sys.stderr)
            return 2
    print(markdown_table(summaries))

    if args.record:
        EXPECTED.write_text(json.dumps(
            {s["system"]: {m: s["global"].get(m) for m in METRICS} for s in summaries},
            indent=2) + "\n")
        print(f"expectations recorded in {EXPECTED}")
        return 0
    if not EXPECTED.exists():
        print(f"no expectations recorded ({EXPECTED}); run with --record", file=sys.stderr)
        return 1
    problems = check(summaries, json.loads(EXPECTED.read_text()), args.tolerance)
    for p in problems:
        print("MISMATCH " + p)
    if not problems:
        print(f"reproduced: every metric within {args.tolerance} of the record")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
