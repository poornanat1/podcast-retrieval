-- A small tsvector for ranking transcript matches. The full search_tsv (up
-- to 500k characters of spoken text) is what a query matches against, but
-- ranking every match with it means decompressing hundreds of kilobytes per
-- row: the lexical leg spent its whole time budget on broad queries. The
-- opening of a transcript ranks nearly as well and costs a few kilobytes.
-- The ranker's lex_transcript feature still uses the full vector
-- (candidate_evidence.sql), so training-serving parity is untouched.
ALTER TABLE transcripts ADD COLUMN rank_tsv tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector(podfind_ts_config(language), left(content, 8000)), 'D')
    ) STORED;
