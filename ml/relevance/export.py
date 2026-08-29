"""Validate judgments and export them as TREC qrels for the eval harness.

    uv run python -m ml.relevance.export

Emits ``qrels.txt`` ("query_id 0 episode_id grade") plus a coverage summary
by query type. Fails when the judgment file violates its contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ml.relevance.contracts import JudgmentModel, QueryModel, is_human_judge


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.read_text().strip():
        return pd.DataFrame()
    # convert_dates=False: judged_at/published_after are strings in the
    # contract; auto-parsing them into timestamps would violate it.
    return pd.read_json(path, lines=True, convert_dates=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queries", default="data/relevance/queries.jsonl")
    parser.add_argument("--judgments", default="data/relevance/judgments.jsonl")
    parser.add_argument("--out", default="data/relevance/qrels.txt")
    args = parser.parse_args(argv)

    queries = QueryModel.validate(pd.read_json(args.queries, lines=True, convert_dates=False))
    judgments = load_jsonl(Path(args.judgments))
    if judgments.empty:
        print("no judgments yet; run `make relevance-review`")
        return 1
    judgments = JudgmentModel.validate(judgments)

    unknown = set(judgments["query_id"]) - set(queries["query_id"])
    if unknown:
        raise SystemExit(f"judgments reference unknown queries: {sorted(unknown)[:5]}")

    # One grade per (query, episode): a human judgment overrides an LLM one.
    judgments = judgments.assign(_human=is_human_judge(judgments["judge"]))
    effective = (
        judgments.sort_values("_human")
        .drop_duplicates(subset=["query_id", "episode_id"], keep="last")
        .drop(columns=["_human"])
    )

    ordered = effective.sort_values(["query_id", "episode_id"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in ordered.itertuples():
            f.write(f"{row.query_id} 0 {row.episode_id} {row.grade}\n")

    merged = effective.merge(queries[["query_id", "query_type"]], on="query_id")
    human = is_human_judge(effective["judge"])
    summary = {
        "judgments": len(effective),
        "human_judgments": int(human.sum()),
        "llm_judgments": int((~human).sum()),
        "human_overrides": len(judgments) - len(effective),
        "judged_queries": int(effective["query_id"].nunique()),
        "total_queries": len(queries),
        "relevant_share": round(float((effective["grade"] >= 2).mean()), 3),
        "by_query_type": {
            t: {"queries": int(g["query_id"].nunique()), "judgments": len(g)}
            for t, g in merged.groupby("query_type")
        },
        "qrels": str(out),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
