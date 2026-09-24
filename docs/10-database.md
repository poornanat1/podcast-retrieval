# Database

PostgreSQL 17 with `pgvector` is the single store. Schema lives in
`migrations/` and is applied by `make migrate` (`cmd/migrate`,
`internal/migrate`), which records applied versions.

| migration | contents |
|---|---|
| `0001_extensions.sql` | the `vector` extension |
| `0002_catalog.sql` | `podcasts`, `episodes`, `transcripts` |
| `0003_jobs.sql` | job queue |
| `0004_outbox.sql` | transactional outbox |
| `0005_interaction_events.sql` | interaction events (schema only; nothing writes to it yet) |
| `0007_transcript_language.sql` | transcript language for text-search configs |
| `0008_full_text_search.sql` | generated `tsvector` columns and GIN indexes |
| `0009_podcast_popularity.sql` | provider-reported popularity on podcasts |
| `0010_episode_embeddings.sql` | `episode_embeddings` |

## Catalog tables

**`podcasts`** — feed URL (unique), discovery source and id, title,
description, publisher, link, artwork, language, categories (`text[]`),
explicit flag, provider popularity, and the poller's conditional-fetch
state (`etag`, `last_modified`, `last_fetched_at`, `next_fetch_at`).

**`episodes`** — `podcast_id`, RSS GUID, enclosure URL, `content_hash`
(hash of the searchable fields), title, description, link, artwork,
language, `duration_seconds`, `published_at`, explicit flag. Unique partial
indexes on `(podcast_id, rss_guid)` and `(podcast_id, enclosure_url)`;
indexes on `podcast_id`, `published_at`, `language`, `content_hash`.

**`transcripts`** — one row per episode: source URL, format (`text`,
`srt`, `vtt`, `json`), normalized plain-text content, the object-storage
key of the raw file, and language.

## Full-text search

Generated, stored `tsvector` columns with GIN indexes: `episodes.search_tsv`
(title A, description B), `podcasts.search_tsv` (title A, publisher and
categories B, description C), `transcripts.search_tsv` (content D, capped
at 500k characters). `podfind_ts_config(lang)` maps a feed language to a
PostgreSQL text-search configuration (English, Spanish, French, German,
Portuguese, Italian, Dutch, else `simple`) and is used both when indexing
and when parsing queries.

## Embeddings

`episode_embeddings(episode_id, model, embedding vector(384), content_hash, created_at)`,
primary key `(episode_id, model)`, one partial HNSW index per model created
by the embedding job. Details in [05-embeddings.md](05-embeddings.md).

## Jobs, outbox, events

`jobs` holds typed JSON payloads with attempts, lease, backoff, and
dead-letter state; `outbox` holds versioned domain events with processing
attempts; `interaction_events` is the schema for impressions and playback
events (rank, sources, scores, model and index versions) that the serving
path will write once it exists.

## Local access

```
DATABASE_URL=postgres://podfind:podfind@localhost:5432/podfind?sslmode=disable
```

The compose Postgres runs `deploy/compose/postgres-init/01-extensions.sql`
on first start. Go tests that need a database use `internal/pgtest` to
create isolated databases; the CI job runs a `pgvector/pgvector:pg17`
service container.
