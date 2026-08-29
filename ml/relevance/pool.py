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

POOL_SQL = """
WITH q AS (
    SELECT websearch_to_tsquery(podfind_ts_config(%(parse_lang)s), %(query)s) AS tsq
),
cand AS (
    SELECT e.id FROM episodes e, q WHERE e.search_tsv @@ q.tsq
    UNION
    SELECT t.episode_id FROM transcripts t, q WHERE t.search_tsv @@ q.tsq
    UNION
    SELECT e.id FROM episodes e, q
    WHERE e.podcast_id IN (SELECT p.id FROM podcasts p WHERE p.search_tsv @@ q.tsq)
)
SELECT e.id AS episode_id,
       p.title AS podcast_title,
       e.title AS episode_title,
       left(e.description, 300) AS description,
       e.language,
       e.duration_seconds,
       e.published_at,
       e.explicit,
       (t.episode_id IS NOT NULL) AS has_transcript
FROM episodes e
JOIN cand ON cand.id = e.id
JOIN podcasts p ON p.id = e.podcast_id
LEFT JOIN transcripts t ON t.episode_id = e.id
CROSS JOIN q
WHERE (%(lang)s = '' OR e.language LIKE %(lang)s || '%%')
  AND (%(max_duration)s = 0 OR e.duration_seconds <= %(max_duration)s)
  AND (%(published_after)s = '' OR e.published_at >= %(published_after)s::timestamptz)
  AND (NOT %(no_explicit)s OR NOT e.explicit)
ORDER BY coalesce(ts_rank(e.search_tsv, q.tsq), 0)
       + coalesce(ts_rank(t.search_tsv, q.tsq), 0)
       + 0.5 * coalesce(ts_rank(p.search_tsv, q.tsq), 0) DESC,
       e.published_at DESC NULLS LAST
LIMIT %(top_k)s
"""


def pool_query(conn: psycopg.Connection, query: dict, top_k: int) -> list[dict]:
    language = query.get("language") or ""
    params = {
        "query": query["query_text"],
        "lang": language,
        # The parse config must stem the way the indexed text was stemmed;
        # the query set is English unless the query says otherwise. The
        # language *filter* stays driven by the query's explicit constraint.
        "parse_lang": language or "en",
        "max_duration": int(query.get("max_duration_seconds") or 0),
        "published_after": query.get("published_after") or "",
        "no_explicit": bool(query.get("no_explicit")),
        "top_k": top_k,
    }
    with conn.cursor() as cur:
        cur.execute(POOL_SQL, params)
        columns = [d.name for d in cur.description]
        out = []
        for rank, values in enumerate(cur.fetchall(), start=1):
            row = dict(zip(columns, values, strict=True))
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
