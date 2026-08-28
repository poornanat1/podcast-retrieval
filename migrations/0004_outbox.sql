CREATE TABLE outbox (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_type    TEXT        NOT NULL,
    event_version INTEGER     NOT NULL DEFAULT 1,
    payload       JSONB       NOT NULL DEFAULT '{}',
    attempts      INTEGER     NOT NULL DEFAULT 0,
    last_error    TEXT        NOT NULL DEFAULT '',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    processed_at  TIMESTAMPTZ
);

-- Outbox workers drain unprocessed events in insertion order.
CREATE INDEX outbox_unprocessed_idx ON outbox (id)
    WHERE processed_at IS NULL;
