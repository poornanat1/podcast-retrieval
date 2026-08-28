-- Lexical episode search with structured filters.
--
-- Run against the compose database, e.g.:
--   docker compose exec -T postgres psql -U podfind -d podfind \
--     -v query="practical uses of artificial intelligence" \
--     -v lang=en -v max_duration=2700 -v published_after='2026-01-01' \
--     -v no_explicit=true -f - < data/samples/search.sql
--
-- Filters are hard predicates, not ranking signals: language, duration,
-- publication date, and explicit content are enforced in SQL exactly as the
-- query-understanding layer will emit them.
--
-- Candidates come from a UNION of three independently indexed matches
-- (episode text, transcript text, podcast metadata). OR-ing the conditions
-- across joined tables instead defeats the GIN indexes: ~2.5s vs ~25ms over
-- a ~540k-episode catalog.
\if :{?query} \else \set query 'artificial intelligence' \endif
\if :{?lang} \else \set lang '' \endif
\if :{?max_duration} \else \set max_duration 0 \endif
\if :{?published_after} \else \set published_after '' \endif
\if :{?no_explicit} \else \set no_explicit false \endif

WITH q AS (
    SELECT websearch_to_tsquery(podfind_ts_config(:'lang'), :'query') AS tsq
),
cand AS (
    SELECT e.id FROM episodes e, q WHERE e.search_tsv @@ q.tsq
    UNION
    SELECT t.episode_id FROM transcripts t, q WHERE t.search_tsv @@ q.tsq
    UNION
    SELECT e.id FROM episodes e, q
    WHERE e.podcast_id IN (SELECT p.id FROM podcasts p WHERE p.search_tsv @@ q.tsq)
)
SELECT e.id,
       p.title AS podcast,
       e.title,
       e.language,
       e.duration_seconds,
       e.published_at::date,
       t.episode_id IS NOT NULL AS has_transcript,
       round((coalesce(ts_rank(e.search_tsv, q.tsq), 0)
            + coalesce(ts_rank(t.search_tsv, q.tsq), 0)
            + 0.5 * coalesce(ts_rank(p.search_tsv, q.tsq), 0))::numeric, 4) AS rank
FROM episodes e
JOIN cand ON cand.id = e.id
JOIN podcasts p ON p.id = e.podcast_id
LEFT JOIN transcripts t ON t.episode_id = e.id
CROSS JOIN q
WHERE (:'lang' = '' OR e.language LIKE :'lang' || '%')
  AND (:max_duration = 0 OR e.duration_seconds <= :max_duration)
  AND (:'published_after' = '' OR e.published_at >= :'published_after'::timestamptz)
  AND (NOT :no_explicit OR NOT e.explicit)
ORDER BY rank DESC, e.published_at DESC NULLS LAST
LIMIT 20;
