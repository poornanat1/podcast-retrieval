"""ANN retrieval over pretrained sentence embeddings in pgvector.

Embeds the query in-process, then scans the model's HNSW index with the
query's structured filters as hard predicates. Iterative scanning keeps
filtered searches from coming back short.
"""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from ml.embeddings.text import MODEL_ID, build_query

SEARCH_SQL = """
SELECT e.id
FROM episode_embeddings emb
JOIN episodes e ON e.id = emb.episode_id
WHERE emb.model = %(model)s
  AND (%(lang)s = '' OR e.language LIKE %(lang)s || '%%')
  AND (%(max_duration)s = 0 OR e.duration_seconds <= %(max_duration)s)
  AND (%(published_after)s = '' OR e.published_at >= %(published_after)s::timestamptz)
  AND (NOT %(no_explicit)s OR NOT e.explicit)
ORDER BY emb.embedding <=> %(qvec)s
LIMIT %(top_k)s
"""


class VectorSearch:
    """Pretrained-embedding ANN retrieval (multilingual E5 small)."""

    name = "vector-e5-small"

    def __init__(self, conn: psycopg.Connection):
        from sentence_transformers import SentenceTransformer

        self.conn = conn
        register_vector(conn)
        self.model = SentenceTransformer(MODEL_ID)
        # Iterative scanning (pgvector >= 0.8) keeps filtered ANN queries
        # scanning until LIMIT is satisfied instead of returning short.
        try:
            with conn.cursor() as cur:
                cur.execute("SET hnsw.iterative_scan = relaxed_order")
                cur.execute("SET hnsw.ef_search = 200")
            conn.commit()
        except psycopg.Error:
            conn.rollback()

    def search(self, query: dict, k: int) -> list[int]:
        vector = self.model.encode(
            build_query(query["query_text"]), normalize_embeddings=True
        )
        params = {
            "model": MODEL_ID,
            "qvec": vector,
            "lang": query.get("language") or "",
            "max_duration": int(query.get("max_duration_seconds") or 0),
            "published_after": query.get("published_after") or "",
            "no_explicit": bool(query.get("no_explicit")),
            "top_k": k,
        }
        with self.conn.cursor() as cur:
            cur.execute(SEARCH_SQL, params)
            return [row[0] for row in cur.fetchall()]
