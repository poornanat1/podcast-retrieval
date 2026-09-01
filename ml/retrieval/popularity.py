"""Popularity baselines.

Query-agnostic (global) and lightly query-aware (category) rankings by
provider-reported show popularity, honoring the query's structured filters.
These are the floor every learned system must clear: if semantic retrieval
cannot beat "show the most popular episodes", it is not earning its cost.

Popularity is the discovery provider's global percentile per show (0 when
unreported); ties break by recency.
"""

from __future__ import annotations

import re

import psycopg

_WORD_RE = re.compile(r"[^0-9a-z]+")

FILTER_SQL = """
  (%(lang)s = '' OR e.language LIKE %(lang)s || '%%')
  AND (%(max_duration)s = 0 OR e.duration_seconds <= %(max_duration)s)
  AND (%(published_after)s = '' OR e.published_at >= %(published_after)s::timestamptz)
  AND (NOT %(no_explicit)s OR NOT e.explicit)
"""

GLOBAL_SQL = f"""
SELECT e.id
FROM episodes e
JOIN podcasts p ON p.id = e.podcast_id
WHERE {FILTER_SQL}
ORDER BY p.popularity DESC, e.published_at DESC NULLS LAST, e.id
LIMIT %(top_k)s
"""

CATEGORY_SQL = f"""
SELECT e.id
FROM episodes e
JOIN podcasts p ON p.id = e.podcast_id
WHERE p.categories && %(categories)s::text[]
  AND {FILTER_SQL}
ORDER BY p.popularity DESC, e.published_at DESC NULLS LAST, e.id
LIMIT %(top_k)s
"""


def filter_params(query: dict, k: int) -> dict:
    return {
        "lang": query.get("language") or "",
        "max_duration": int(query.get("max_duration_seconds") or 0),
        "published_after": query.get("published_after") or "",
        "no_explicit": bool(query.get("no_explicit")),
        "top_k": k,
    }


def tokens(text: str) -> list[str]:
    return [t for t in _WORD_RE.sub(" ", text.lower()).split() if t]


def match_categories(query_text: str, categories: list[str]) -> list[str]:
    """Categories whose every word appears in the query, e.g. "true crime"
    matches "clean true crime for road trips". Returns original names."""
    query_tokens = set(tokens(query_text))
    matched = []
    for category in categories:
        category_tokens = tokens(category)
        if category_tokens and set(category_tokens) <= query_tokens:
            matched.append(category)
    return matched


class GlobalPopularity:
    """Most popular episodes regardless of the query text."""

    name = "popularity-global"

    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def search(self, query: dict, k: int) -> list[int]:
        with self.conn.cursor() as cur:
            cur.execute(GLOBAL_SQL, filter_params(query, k))
            return [row[0] for row in cur.fetchall()]


class CategoryPopularity:
    """Most popular episodes within categories named in the query, falling
    back to global popularity when no category matches."""

    name = "popularity-category"

    def __init__(self, conn: psycopg.Connection):
        self.conn = conn
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT c FROM podcasts, unnest(categories) AS c")
            self.categories = sorted(row[0] for row in cur.fetchall())
        self._global = GlobalPopularity(conn)

    def search(self, query: dict, k: int) -> list[int]:
        matched = match_categories(query["query_text"], self.categories)
        if not matched:
            return self._global.search(query, k)
        with self.conn.cursor() as cur:
            cur.execute(CATEGORY_SQL, {**filter_params(query, k), "categories": matched})
            return [row[0] for row in cur.fetchall()]
