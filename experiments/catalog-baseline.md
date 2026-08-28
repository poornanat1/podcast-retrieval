# Catalog baseline

Measurements taken 2026-08-26 against the local Compose stack (PostgreSQL 17
+ pgvector, single machine) after seeding the catalog with the standard seed
set (`catalog-worker -seed -oneshot`: global trending 500, trending for
es/fr/de/pt at 100 each, ten topical searches at 50 each, discovered through
the Particle API and ingested from publisher RSS feeds).

These are engineering baselines for later comparison, not tuned results.

## Catalog composition

| Metric | Value |
| --- | --- |
| Podcasts | 1,314 |
| Episodes | 539,626 |
| Transcripts (parsed, publisher-provided) | 48,464 |
| Raw transcript files retained in object storage | 48,464 |
| Episode transcript coverage | 9.0% |
| Podcasts with ≥1 transcript | 223 (17%) |
| Distinct languages (primary subtag) | 9 — en (905), fr (101), es (101), de (101), pt (99) |
| Distinct categories | 152 |
| Episodes missing duration | 0.1% |
| Episodes missing publication date | 0.0% |
| Database size | 5.4 GB |

## Ingestion throughput

Full seed — discovery, 1,300+ feed fetches, 540k episode upserts, and 123k
transcript fetches — drained in about 2.5 hours end to end on one worker
process.

| Metric | Value |
| --- | --- |
| Feed refreshes completed | 1,736 (177 dead-lettered: unreachable/malformed feeds) |
| Transcript fetches completed | 120,918 (2,494 dead-lettered: unreachable/unparseable files) |
| Peak transcript throughput (12 workers) | 5,199 jobs/min |
| Sustained transcript throughput (4 workers) | ~612 jobs/min |

Worker concurrency (`PODFIND_WORKER_CONCURRENCY`) scales transcript
ingestion roughly linearly in this range; fetches spread across many hosts,
so per-host politeness is preserved.

Observation fixed during measurement: permanently failing feeds were
re-enqueued by the scheduler every cycle, growing the dead-letter count from
22 to 177 across cycles. Failing feeds are now parked for the maximum poll
interval before dead-lettering.

## Full-text search latency

Query: "practical uses of artificial intelligence" with hard filters
(language en, duration ≤ 45 min, published 2026+, no explicit), ranked,
LIMIT 20, over 540k episodes. Warm cache, `EXPLAIN ANALYZE`, single run.

| Query shape | Execution time |
| --- | --- |
| OR across episode/transcript/podcast tsvectors in one joined scan | 2,525 ms |
| UNION of three independently GIN-indexed candidate scans (shipped in `data/samples/search.sql`) | 23 ms |

Finding: OR-ing match conditions across joined tables prevents GIN index
use and sequential-scans the catalog. The UNION-of-indexed-scans shape is
~100x faster and is the formulation later retrieval stages should build on.
An unranked full match count for a two-term query runs ~240 ms, dominated by
match volume rather than index lookup.

## Functional checks (verified by automated tests and live run)

- Re-ingesting a feed produces zero duplicate episodes (GUID → enclosure →
  content-hash dedup; verified twice end to end, plus at seed scale).
- A malformed feed dead-letters its job with the error retained; the worker
  continues processing other feeds.
- Transcript arrival and feed-content changes each emit a versioned
  `episode.content_changed` outbox event, committed atomically with the data
  change and drained by the outbox worker.
- Episodes without transcripts (91% of the catalog) are fully searchable
  through episode and podcast metadata.
- A crashed worker's lease expires and its job is reclaimed exactly once.

## Reproduction

```sh
docker compose up -d
make migrate
docker compose run --rm catalog-worker -seed -oneshot   # requires PARTICLE_API_KEY in .env
# wait for the job queue to drain, then:
docker compose exec -T postgres psql -U podfind -d podfind \
  -v query="practical uses of artificial intelligence" \
  -v lang=en -v max_duration=2700 -v published_after='2026-01-01' \
  -v no_explicit=true -f - < data/samples/search.sql
```

Discovery results and feed contents change over time, so re-runs will not
reproduce these exact counts; the seed set and measurement queries are fixed
so the methodology reproduces.
