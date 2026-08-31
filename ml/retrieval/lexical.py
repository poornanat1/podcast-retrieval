"""Lexical full-text retrieval over the catalog database.

The same weighted UNION query shipped in ``data/samples/search.sql``:
candidates from independently indexed matches on episode text, transcript
text, and podcast metadata, ranked by summed ts_rank, with the query's
structured filters applied as hard predicates.
"""

from __future__ import annotations

import psycopg

SEARCH_SQL = """
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


def query_params(query: dict, k: int) -> dict:
    language = query.get("language") or ""
    return {
        "query": query["query_text"],
        "lang": language,
        # Stemming must match how the indexed text was stemmed; the query
        # set is English unless the query says otherwise.
        "parse_lang": language or "en",
        "max_duration": int(query.get("max_duration_seconds") or 0),
        "published_after": query.get("published_after") or "",
        "no_explicit": bool(query.get("no_explicit")),
        "top_k": k,
    }


class LexicalSearch:
    """Full-text search system for the evaluation harness."""

    name = "lexical-fts"

    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def search(self, query: dict, k: int) -> list[int]:
        with self.conn.cursor() as cur:
            cur.execute(SEARCH_SQL, query_params(query, k))
            return [row[0] for row in cur.fetchall()]

    def search_rows(self, query: dict, k: int) -> list[dict]:
        """Full result rows (used by relevance-set pooling)."""
        with self.conn.cursor() as cur:
            cur.execute(SEARCH_SQL, query_params(query, k))
            columns = [d.name for d in cur.description]
            return [dict(zip(columns, values, strict=True)) for values in cur.fetchall()]
