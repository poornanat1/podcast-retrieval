# Catalog

The catalog worker (`cmd/catalog-worker`, packages under `internal/`)
keeps a self-maintaining podcast catalog in PostgreSQL: it discovers
shows, ingests their RSS feeds, deduplicates episodes, parses publisher
transcripts, and reschedules each feed by how often it publishes.

```sh
docker compose up -d catalog-worker                       # run continuously
docker compose run --rm catalog-worker -oneshot -seed     # enqueue the standard seeding set and exit
docker compose run --rm catalog-worker -oneshot -trending 100 -language en
docker compose run --rm catalog-worker -oneshot -discover "machine learning"
```

Environment: `DATABASE_URL`; `PARTICLE_API_KEY` (discovery jobs dead-letter
without it); `PODFIND_DAILY_TRENDING` (podcasts discovered from the
trending list per day, default 25, 0 disables).

## Corpus freeze

> **Since 2026-09-13** the worker is intentionally stopped
> (`docker compose stop catalog-worker`) to hold the evaluation corpus
> fixed while retrieval systems are compared. Resume with
> `docker compose start catalog-worker`; re-freeze the same way and record
> the new date and episode count here and in the data card.
>
> **Deviation (2026-09-18):** a `docker compose up -d` restarted the worker
> for about a minute before it was stopped again. It added 1,683 episodes
> and 2 podcasts and refreshed metadata (and therefore the pretrained
> vectors) of 8,976 existing episodes, leaving 797,216 episodes / 1,790
> podcasts. Evaluations from that date re-run every baseline on this
> corpus rather than reusing earlier numbers. Restarting the stack starts
> the worker: start only the services you need
> (`docker compose up -d postgres mlflow minio`).

## Discovery

`internal/discovery` defines a provider-agnostic contract (search by
query, list trending by language); `internal/particle` implements it for
the [Particle data platform](https://docs.particle.pro). Providers surface
shows and their canonical RSS feed URLs; RSS stays the authoritative source
for episodes. Discovered shows are keyed by `(discovery_source,
discovery_id)` and by feed URL, and carry the provider's popularity
percentile (`migrations/0009_podcast_popularity.sql`).

## Polling

Each feed's next fetch is derived from the age of its newest episode
(`internal/catalog/schedule.go`): hourly if it published in the last day,
every 3 hours within 3 days, 6 hours within a week, 12 hours within a
month, daily otherwise, never more often than every 15 minutes. Fetches
are conditional (`ETag` / `Last-Modified`), so unchanged feeds cost one
request.

## Deduplication

Episodes are matched in order of trust: RSS GUID, then enclosure URL, then
a content hash of the searchable fields; unique partial indexes enforce
the first two per show. `content_hash` also drives incremental
re-embedding: a vector is recomputed only when the hash changes.

## Transcripts

Publisher-provided transcripts linked from the feed (plain text, SRT, VTT,
podcast-namespace JSON) are normalized to plain text
(`internal/transcript`), stored one row per episode with the raw file
retained in object storage, and indexed for full-text search at the lowest
weight. About 11% of episodes have one. Episodes without transcripts stay
searchable through their own and their show's metadata.

## Jobs and events

All work runs through a PostgreSQL job queue (`internal/jobs`): workers
claim jobs with `FOR UPDATE SKIP LOCKED` under a lease, retries back off
exponentially, and jobs that exhaust their attempts are parked in a
dead-letter state with their last error. Catalog state changes and the
versioned domain events describing them commit in one transaction through
the outbox (`internal/outbox`); a worker drains the events asynchronously.

## Size

1,790 shows and 797,216 episodes at the last count, ~87k with
transcripts, 77% English with real Spanish, French, German, and Portuguese
segments. The data card ([data-card.md](data-card.md)) records the sources
and the label taxonomy.
