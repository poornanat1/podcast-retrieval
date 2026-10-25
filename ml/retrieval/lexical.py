"""Lexical full-text retrieval over the catalog database.

The same weighted UNION query shipped in ``data/samples/search.sql``:
candidates from independently indexed matches on episode text, transcript
text, and podcast metadata, ranked by summed ts_rank, with the query's
structured filters applied as hard predicates.
"""

from __future__ import annotations

import psycopg

from ml import sql

SEARCH_SQL = sql.load("lexical_search")


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
