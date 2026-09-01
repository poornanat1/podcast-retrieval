"""Retrieval systems evaluated by the offline harness.

Every system exposes ``name`` and ``search(query, k) -> list[int]``; the
registry maps config names to constructors over a database connection.
"""

from __future__ import annotations

import psycopg


def build_system(name: str, conn: psycopg.Connection):
    from ml.retrieval.lexical import LexicalSearch
    from ml.retrieval.popularity import CategoryPopularity, GlobalPopularity

    registry = {
        LexicalSearch.name: LexicalSearch,
        GlobalPopularity.name: GlobalPopularity,
        CategoryPopularity.name: CategoryPopularity,
    }
    if name not in registry:
        raise ValueError(f"unknown retrieval system {name!r}; known: {sorted(registry)}")
    return registry[name](conn)


def fetch_episode_rows(conn: psycopg.Connection, episode_ids: list[int]) -> list[dict]:
    """Episode metadata for a ranked id list, in the same order (for
    pooling and judgment prompts)."""
    if not episode_ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.id AS episode_id, p.title AS podcast_title, e.title AS episode_title,
                   left(e.description, 300) AS description, e.language,
                   e.duration_seconds, e.published_at, e.explicit,
                   (t.episode_id IS NOT NULL) AS has_transcript
            FROM episodes e
            JOIN podcasts p ON p.id = e.podcast_id
            LEFT JOIN transcripts t ON t.episode_id = e.id
            WHERE e.id = ANY(%(ids)s)
            """,
            {"ids": episode_ids},
        )
        columns = [d.name for d in cur.description]
        by_id = {row[0]: dict(zip(columns, row, strict=True)) for row in cur.fetchall()}
    return [by_id[i] for i in episode_ids if i in by_id]
