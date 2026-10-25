-- Lexical leg: weighted full-text search over episode, transcript, and show
-- fields with the query's hard filters. Shared by Python and Go (Go rewrites
-- the psycopg-style named parameters to pgx's @name form).
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
