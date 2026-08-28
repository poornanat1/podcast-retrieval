CREATE TABLE podcasts (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    feed_url         TEXT        NOT NULL UNIQUE,
    -- Which discovery provider surfaced this feed, and its provider-native id.
    discovery_source TEXT        NOT NULL DEFAULT '',
    discovery_id     TEXT        NOT NULL DEFAULT '',
    title            TEXT        NOT NULL DEFAULT '',
    description      TEXT        NOT NULL DEFAULT '',
    publisher        TEXT        NOT NULL DEFAULT '',
    link_url         TEXT        NOT NULL DEFAULT '',
    artwork_url      TEXT        NOT NULL DEFAULT '',
    language         TEXT        NOT NULL DEFAULT '',
    categories       TEXT[]      NOT NULL DEFAULT '{}',
    explicit         BOOLEAN     NOT NULL DEFAULT FALSE,
    -- Conditional-fetch state for the RSS poller.
    etag             TEXT        NOT NULL DEFAULT '',
    last_modified    TEXT        NOT NULL DEFAULT '',
    last_fetched_at  TIMESTAMPTZ,
    next_fetch_at    TIMESTAMPTZ,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX podcasts_next_fetch_at_idx ON podcasts (next_fetch_at);
CREATE UNIQUE INDEX podcasts_discovery_key ON podcasts (discovery_source, discovery_id)
    WHERE discovery_id <> '';

CREATE TABLE episodes (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    podcast_id       BIGINT      NOT NULL REFERENCES podcasts (id) ON DELETE CASCADE,
    rss_guid         TEXT,
    enclosure_url    TEXT,
    -- Hash of the searchable fields; unchanged hash means no re-embedding needed.
    content_hash     TEXT        NOT NULL DEFAULT '',
    title            TEXT        NOT NULL DEFAULT '',
    description      TEXT        NOT NULL DEFAULT '',
    link_url         TEXT        NOT NULL DEFAULT '',
    artwork_url      TEXT        NOT NULL DEFAULT '',
    language         TEXT        NOT NULL DEFAULT '',
    duration_seconds INTEGER,
    published_at     TIMESTAMPTZ,
    explicit         BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Dedup keys in order of trust: RSS GUID, then enclosure URL, then content hash.
CREATE UNIQUE INDEX episodes_guid_key ON episodes (podcast_id, rss_guid)
    WHERE rss_guid IS NOT NULL;
CREATE UNIQUE INDEX episodes_enclosure_key ON episodes (podcast_id, enclosure_url)
    WHERE enclosure_url IS NOT NULL;
CREATE INDEX episodes_content_hash_idx ON episodes (content_hash);
CREATE INDEX episodes_podcast_id_idx ON episodes (podcast_id);
-- Structured-filter columns.
CREATE INDEX episodes_published_at_idx ON episodes (published_at);
CREATE INDEX episodes_language_idx ON episodes (language);

CREATE TABLE transcripts (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    episode_id  BIGINT      NOT NULL UNIQUE REFERENCES episodes (id) ON DELETE CASCADE,
    source_url  TEXT        NOT NULL DEFAULT '',
    format      TEXT        NOT NULL CHECK (format IN ('text', 'srt', 'vtt', 'json')),
    content     TEXT        NOT NULL,
    -- Raw upstream file retained in object storage.
    object_key  TEXT        NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
