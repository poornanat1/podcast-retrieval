-- Weighted, language-aware full-text search over episodes, podcasts, and
-- transcripts. Episodes without transcripts stay searchable through their
-- own metadata and their podcast's.

-- Map a feed language tag to a text-search configuration. IMMUTABLE so it
-- can drive generated columns.
CREATE FUNCTION podfind_ts_config(lang text) RETURNS regconfig
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT CASE lower(split_part(coalesce(lang, ''), '-', 1))
        WHEN 'en' THEN 'english'::regconfig
        WHEN 'es' THEN 'spanish'::regconfig
        WHEN 'fr' THEN 'french'::regconfig
        WHEN 'de' THEN 'german'::regconfig
        WHEN 'pt' THEN 'portuguese'::regconfig
        WHEN 'it' THEN 'italian'::regconfig
        WHEN 'nl' THEN 'dutch'::regconfig
        ELSE 'simple'::regconfig
    END
$$;

-- array_to_string is not IMMUTABLE; this wrapper is safe for text[] input.
CREATE FUNCTION podfind_join(arr text[]) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT coalesce(array_to_string(arr, ' '), '')
$$;

ALTER TABLE episodes ADD COLUMN search_tsv tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector(podfind_ts_config(language), title), 'A') ||
        setweight(to_tsvector(podfind_ts_config(language), left(description, 100000)), 'B')
    ) STORED;
CREATE INDEX episodes_search_idx ON episodes USING gin (search_tsv);

ALTER TABLE podcasts ADD COLUMN search_tsv tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector(podfind_ts_config(language), title), 'A') ||
        setweight(to_tsvector(podfind_ts_config(language), publisher), 'B') ||
        setweight(to_tsvector(podfind_ts_config(language), podfind_join(categories)), 'B') ||
        setweight(to_tsvector(podfind_ts_config(language), left(description, 50000)), 'C')
    ) STORED;
CREATE INDEX podcasts_search_idx ON podcasts USING gin (search_tsv);

-- Transcript text is weighted lowest: long spoken text should support, not
-- dominate, title and description matches. Capped well under the tsvector
-- size limit.
ALTER TABLE transcripts ADD COLUMN search_tsv tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector(podfind_ts_config(language), left(content, 500000)), 'D')
    ) STORED;
CREATE INDEX transcripts_search_idx ON transcripts USING gin (search_tsv);
