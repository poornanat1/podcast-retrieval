-- Lexical leg: weighted full-text search over episode, transcript, and show
-- fields with the query's hard filters. Shared by Python and Go (Go rewrites
-- the psycopg-style named parameters to pgx's @name form).
--
-- Each source is ranked on its own and cut to top_k before the union, so
-- the final ordering sees at most 3 x top_k rows. Within a source, at most
-- 50 x top_k matches are ranked: a query that matches a large share of the
-- catalog ("news") is ranked over a bounded sample of its matches instead
-- of all of them, which keeps the leg inside its time limit; such queries
-- are the vector legs' job anyway. Transcripts rank on rank_tsv, the small
-- vector over their opening; they still match on the full vector.
WITH q AS (
    SELECT websearch_to_tsquery(podfind_ts_config(%(parse_lang)s), %(query)s) AS tsq
),
ep_match AS (
    SELECT e.id, e.search_tsv, e.published_at
    FROM episodes e, q
    WHERE e.search_tsv @@ q.tsq
      AND (%(lang)s = '' OR e.language LIKE %(lang)s || '%%')
      AND (%(max_duration)s = 0 OR e.duration_seconds <= %(max_duration)s)
      AND (%(published_after)s = '' OR e.published_at >= %(published_after)s::timestamptz)
      AND (NOT %(no_explicit)s OR NOT e.explicit)
    LIMIT %(top_k)s * 50
),
ep AS (
    SELECT m.id, ts_rank(m.search_tsv, q.tsq) AS r, m.published_at
    FROM ep_match m, q
    ORDER BY r DESC, m.published_at DESC NULLS LAST
    LIMIT %(top_k)s
),
tr_match AS (
    SELECT e.id, t.rank_tsv, e.published_at
    FROM transcripts t
    JOIN episodes e ON e.id = t.episode_id, q
    WHERE t.search_tsv @@ q.tsq
      AND (%(lang)s = '' OR e.language LIKE %(lang)s || '%%')
      AND (%(max_duration)s = 0 OR e.duration_seconds <= %(max_duration)s)
      AND (%(published_after)s = '' OR e.published_at >= %(published_after)s::timestamptz)
      AND (NOT %(no_explicit)s OR NOT e.explicit)
    LIMIT %(top_k)s * 50
),
tr AS (
    SELECT m.id, ts_rank(m.rank_tsv, q.tsq) AS r, m.published_at
    FROM tr_match m, q
    ORDER BY r DESC, m.published_at DESC NULLS LAST
    LIMIT %(top_k)s
),
pc_match AS (
    SELECT e.id, p.search_tsv, e.published_at
    FROM podcasts p
    JOIN episodes e ON e.podcast_id = p.id, q
    WHERE p.search_tsv @@ q.tsq
      AND (%(lang)s = '' OR e.language LIKE %(lang)s || '%%')
      AND (%(max_duration)s = 0 OR e.duration_seconds <= %(max_duration)s)
      AND (%(published_after)s = '' OR e.published_at >= %(published_after)s::timestamptz)
      AND (NOT %(no_explicit)s OR NOT e.explicit)
    LIMIT %(top_k)s * 50
),
pc AS (
    SELECT m.id, 0.5 * ts_rank(m.search_tsv, q.tsq) AS r, m.published_at
    FROM pc_match m, q
    ORDER BY r DESC, m.published_at DESC NULLS LAST
    LIMIT %(top_k)s
),
cand AS (
    SELECT id FROM ep UNION SELECT id FROM tr UNION SELECT id FROM pc
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
ORDER BY coalesce(ts_rank(e.search_tsv, q.tsq), 0)
       + coalesce(ts_rank(t.rank_tsv, q.tsq), 0)
       + 0.5 * coalesce(ts_rank(p.search_tsv, q.tsq), 0) DESC,
       e.published_at DESC NULLS LAST
LIMIT %(top_k)s
