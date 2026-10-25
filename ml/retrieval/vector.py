"""ANN retrieval over pretrained sentence embeddings in pgvector.

Embeds the query in-process, then scans the model's HNSW index with the
query's structured filters as hard predicates. Iterative scanning keeps
filtered searches from coming back short. Its ``relaxed_order`` mode can
return rows slightly out of distance order, so the scan is materialized and
re-sorted by exact distance, as pgvector recommends.
"""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from ml import sql
from ml.embeddings.text import MODEL_ID, build_query

SEARCH_SQL = sql.load("vector_search")


def session_statements() -> list[str]:
    """The shared per-connection search settings (``ml/sql/session_settings.sql``)."""
    return [line.rstrip(";") for line in sql.load("session_settings").splitlines()
            if line.strip() and not line.startswith("--")]


def enable_iterative_scan(conn: psycopg.Connection) -> None:
    """Apply the shared vector search settings to this connection: iterative
    scans (pgvector >= 0.8) keep filtered ANN queries scanning until LIMIT is
    satisfied, and a wide ``ef_search`` keeps recall up."""
    try:
        with conn.cursor() as cur:
            for statement in session_statements():
                cur.execute(statement)
        conn.commit()
    except psycopg.Error:
        conn.rollback()


def filter_params(query: dict) -> dict:
    """Structured-filter bind parameters for ``SEARCH_SQL``."""
    return {
        "lang": query.get("language") or "",
        "max_duration": int(query.get("max_duration_seconds") or 0),
        "published_after": query.get("published_after") or "",
        "no_explicit": bool(query.get("no_explicit")),
    }


class VectorSearch:
    """Pretrained-embedding ANN retrieval (multilingual E5 small)."""

    name = "vector-e5-small"

    def __init__(self, conn: psycopg.Connection):
        from sentence_transformers import SentenceTransformer

        self.conn = conn
        register_vector(conn)
        self.model = SentenceTransformer(MODEL_ID)
        enable_iterative_scan(conn)

    def search(self, query: dict, k: int) -> list[int]:
        vector = self.model.encode(
            build_query(query["query_text"]), normalize_embeddings=True
        )
        params = {"model": MODEL_ID, "qvec": vector, **filter_params(query), "top_k": k}
        with self.conn.cursor() as cur:
            cur.execute(SEARCH_SQL, params)
            return [row[0] for row in cur.fetchall()]
