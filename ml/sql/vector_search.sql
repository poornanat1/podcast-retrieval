-- Vector leg: HNSW scan for one model's partial index with the query's hard
-- filters, re-sorted by exact distance (relaxed_order iterative scans can
-- return rows slightly out of order). Shared by Python and Go.
WITH nearest AS MATERIALIZED (
    SELECT e.id, emb.embedding <=> %(qvec)s AS distance
    FROM episode_embeddings emb
    JOIN episodes e ON e.id = emb.episode_id
    WHERE emb.model = %(model)s
      AND (%(lang)s = '' OR e.language LIKE %(lang)s || '%%')
      AND (%(max_duration)s = 0 OR e.duration_seconds <= %(max_duration)s)
      AND (%(published_after)s = '' OR e.published_at >= %(published_after)s::timestamptz)
      AND (NOT %(no_explicit)s OR NOT e.explicit)
    ORDER BY distance
    LIMIT %(top_k)s
)
SELECT id FROM nearest ORDER BY distance, id
