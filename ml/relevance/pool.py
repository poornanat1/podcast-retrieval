"""Pool candidate episodes for every relevance query.

For each query, every listed retrieval system contributes its top K results
(with the query's structured filters applied). The union of (query, episode)
pairs becomes the judgment tasks, each recording which systems surfaced it
and at what rank; pooling from all systems under comparison keeps their
judgments comparable.

    uv run python -m ml.relevance.pool --systems lexical-fts,popularity-global \
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
from ml.retrieval import build_system, fetch_episode_rows


def pool_query(conn: psycopg.Connection, systems: list, query: dict, top_k: int) -> list[dict]:
    """Union of each system's top-k for one query, with per-system ranks."""
    sources: dict[int, list[dict]] = {}
    order: list[int] = []
    for system in systems:
        for rank, episode_id in enumerate(system.search(query, top_k), start=1):
            if episode_id not in sources:
                sources[episode_id] = []
                order.append(episode_id)
            sources[episode_id].append({"system": system.name, "rank": rank})

    out = []
    for row in fetch_episode_rows(conn, order):
        episode_id = row["episode_id"]
        row["published_at"] = str(row["published_at"]) if row["published_at"] else ""
        row.update(
            query_id=query["query_id"], query_text=query["query_text"],
            query_type=query["query_type"],
            rank=min(s["rank"] for s in sources[episode_id]),
            sources=sources[episode_id],
        )
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queries", default="data/relevance/queries.jsonl")
    parser.add_argument("--out", default="data/relevance/tasks.jsonl")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--systems", default="lexical-fts",
                        help="comma-separated retrieval systems to pool from")
    parser.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    )
    args = parser.parse_args(argv)

    queries = [json.loads(line) for line in Path(args.queries).read_text().splitlines()]
    tasks: list[dict] = []
    empty = 0
    with psycopg.connect(args.database_url) as conn:
        systems = [build_system(name.strip(), conn) for name in args.systems.split(",")]
        for query in queries:
            rows = pool_query(conn, systems, query, args.top_k)
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
