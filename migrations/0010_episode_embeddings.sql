-- Versioned episode embeddings: one row per (episode, model). content_hash
-- records the episode content the vector was computed from, so the embed
-- job can re-embed exactly the episodes whose searchable text changed.
-- ANN indexes are partial per model and created by the embedding job.
CREATE TABLE episode_embeddings (
    episode_id   BIGINT      NOT NULL REFERENCES episodes (id) ON DELETE CASCADE,
    model        TEXT        NOT NULL,
    embedding    vector(384) NOT NULL,
    content_hash TEXT        NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (episode_id, model)
);
