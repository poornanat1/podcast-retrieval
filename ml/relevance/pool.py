"""Pool candidate episodes for every relevance query.

For each query, the current retrieval system (lexical full-text search with
the query's structured filters applied) contributes its top K results. The
pooled (query, episode) pairs become the judgment tasks; as new retrieval
systems land, their results are pooled into the same file so judgments stay
comparable across systems.

    uv run python -m ml.relevance.pool --queries data/relevance/queries.jsonl \
        --out data/relevance/tasks.jsonl --top-k 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg

from ml.datasets.snapshot import DEFAULT_DATABASE_URL
from ml.retrieval.lexical import LexicalSearch


def pool_query(conn: psycopg.Connection, query: dict, top_k: int) -> list[dict]:
    out = []
    for rank, row in enumerate(LexicalSearch(conn).search_rows(query, top_k), start=1):
        row["published_at"] = str(row["published_at"]) if row["published_at"] else ""
        row.update(query_id=query["query_id"], query_text=query["query_text"],
                   query_type=query["query_type"], rank=rank)
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queries", default="data/relevance/queries.jsonl")
    parser.add_argument("--out", default="data/relevance/tasks.jsonl")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    )
    args = parser.parse_args(argv)

    queries = [json.loads(line) for line in Path(args.queries).read_text().splitlines()]
    tasks: list[dict] = []
    empty = 0
    with psycopg.connect(args.database_url) as conn:
        for query in queries:
            rows = pool_query(conn, query, args.top_k)
            if not rows:
                empty += 1
            tasks.extend(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for task in tasks:
            f.write(json.dumps(task, sort_keys=True, default=str) + "\n")

    pairs = pd.DataFrame(tasks)
    print(json.dumps({
        "queries": len(queries),
        "queries_with_no_candidates": empty,
        "judgment_pairs": len(pairs),
        "distinct_episodes": int(pairs["episode_id"].nunique()) if len(pairs) else 0,
        "out": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
