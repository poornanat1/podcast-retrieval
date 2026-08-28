-- Interaction events carry the attribution needed to reconstruct training
-- examples: displayed rank, candidate sources, scores, and the feature,
-- model, and index versions that produced each result.
CREATE TABLE interaction_events (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_type        TEXT        NOT NULL CHECK (event_type IN (
                          'search', 'impression', 'select',
                          'playback_start', 'playback_progress',
                          'playback_skip', 'playback_complete',
                          'like', 'save')),
    occurred_at       TIMESTAMPTZ NOT NULL,
    session_id        TEXT        NOT NULL DEFAULT '',
    query_id          TEXT        NOT NULL DEFAULT '',
    user_id           TEXT        NOT NULL DEFAULT '',
    episode_id        BIGINT      REFERENCES episodes (id) ON DELETE SET NULL,
    displayed_rank    INTEGER,
    candidate_sources TEXT[]      NOT NULL DEFAULT '{}',
    score             DOUBLE PRECISION,
    feature_version   TEXT        NOT NULL DEFAULT '',
    model_version     TEXT        NOT NULL DEFAULT '',
    index_version     TEXT        NOT NULL DEFAULT '',
    payload           JSONB       NOT NULL DEFAULT '{}',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX interaction_events_occurred_at_idx ON interaction_events (occurred_at);
CREATE INDEX interaction_events_episode_id_idx ON interaction_events (episode_id);
CREATE INDEX interaction_events_query_id_idx ON interaction_events (query_id);
