-- "More like this": the nearest episodes to one episode's own vector in a
-- model's HNSW index, the source excluded, with the caller's hard filters.
-- Diversification across shows happens in the caller over this candidate
-- list. Shared by Python and Go.
WITH src AS (
    SELECT embedding FROM episode_embeddings
    WHERE episode_id = %(episode_id)s AND model = %(model)s
),
nearest AS MATERIALIZED (
    SELECT e.id, e.podcast_id, emb.embedding <=> src.embedding AS distance
    FROM episode_embeddings emb
    JOIN episodes e ON e.id = emb.episode_id
    CROSS JOIN src
    WHERE emb.model = %(model)s
      AND emb.episode_id <> %(episode_id)s
      AND (%(lang)s = '' OR e.language LIKE %(lang)s || '%%')
      AND (%(max_duration)s = 0 OR e.duration_seconds <= %(max_duration)s)
      AND (%(published_after)s = '' OR e.published_at >= %(published_after)s::timestamptz)
      AND (NOT %(no_explicit)s OR NOT e.explicit)
    ORDER BY distance
    LIMIT %(top_k)s
)
SELECT id, podcast_id, distance FROM nearest ORDER BY distance, id
