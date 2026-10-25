-- Ranker evidence for a candidate set: catalog fields and lexical score
-- components for every candidate, whichever leg retrieved it. Shared by
-- Python and Go.
WITH q AS (
    SELECT websearch_to_tsquery(podfind_ts_config(%(parse_lang)s), %(query)s) AS tsq
)
SELECT e.id, e.podcast_id, e.title, coalesce(left(e.description, 1000), '') AS description,
       e.language, e.duration_seconds, e.published_at,
       p.title AS podcast_title, p.publisher, p.popularity,
       (t.episode_id IS NOT NULL) AS has_transcript,
       coalesce(ts_rank(e.search_tsv, q.tsq), 0) AS lex_episode,
       coalesce(ts_rank(t.search_tsv, q.tsq), 0) AS lex_transcript,
       coalesce(ts_rank(p.search_tsv, q.tsq), 0) AS lex_podcast
FROM episodes e
JOIN podcasts p ON p.id = e.podcast_id
LEFT JOIN transcripts t ON t.episode_id = e.id
CROSS JOIN q
WHERE e.id = ANY(%(ids)s)
