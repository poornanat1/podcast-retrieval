-- Every served search, with what produced it: the understood query and
-- filters, the model bundle and feature version, the vector indexes read,
-- per-leg outcomes, and the ranked results with their scores and candidate
-- sources. Impression and playback events are attributed from this row, so
-- clients never have to (and cannot) assert scores or versions.
CREATE TABLE searches (
    query_id        TEXT        PRIMARY KEY,
    session_id      TEXT        NOT NULL DEFAULT '',
    raw_query       TEXT        NOT NULL,
    query_text      TEXT        NOT NULL,
    filters         JSONB       NOT NULL DEFAULT '{}',
    bundle_id       TEXT        NOT NULL,
    feature_version TEXT        NOT NULL,
    index_version   TEXT        NOT NULL,
    ordering        TEXT        NOT NULL CHECK (ordering IN ('ranker', 'rrf')),
    legs            JSONB       NOT NULL DEFAULT '{}',
    timings_ms      JSONB       NOT NULL DEFAULT '{}',
    results         JSONB       NOT NULL DEFAULT '[]',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX searches_session_idx ON searches (session_id, created_at);
