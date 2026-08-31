# PodFind

Voice-aware podcast search and recommendation.

Podcasts hold an enormous amount of spoken knowledge, but discovery is stuck
at the show level: charts, categories, and title matching. The moment worth
finding is usually buried inside an episode nobody can name. PodFind's goal
is to make that findable — say or type what you actually want:

> *"Find English episodes under 45 minutes about practical uses of AI,
> published this year, with no explicit content."*

and get the right episodes back, ranked, with the constraints treated as
guarantees rather than suggestions.

## Vision

- **Search by meaning, not just words.** Queries, listeners, and episodes
  live in a shared embedding space; exploratory questions land on relevant
  episodes even when no title contains the query. Exact names and phrases
  still win through lexical search — retrieval is hybrid on purpose.
- **Voice as a first-class input.** Spoken queries are transcribed, parsed
  into intent plus structured filters (language, duration, recency,
  explicit content), and evaluated as part of retrieval quality — not
  treated as a UI garnish.
- **Recommendations that respect intent.** Personalized retrieval from
  listening history, and "more like this episode" from any starting point,
  with diversity across shows rather than more-of-the-same.
- **Evidence before complexity.** Every learned component must beat
  reproducible baselines on a human-auditable relevance set before it
  ships; datasets, models, and indexes are versioned and rebuildable
  bit-for-bit. Claims come with checked-in experiment configs.
- **Respect for the ecosystem.** Publisher RSS remains the source of truth,
  attribution is preserved, audio is never copied, and removal requests are
  honored (see [docs/licensing.md](docs/licensing.md)).

## Scope

A self-hosted stack: podcast discovery and RSS ingestion, publisher
transcript parsing, hybrid lexical + vector retrieval with a learned
reranker, interaction-event capture feeding back into training data, and
versioned model serving with offline evaluation, monitoring, and rollback —
Python owns the ML lifecycle, Go owns ingestion and serving, PostgreSQL
(with `pgvector`) owns the data.

**Working today:** a self-maintaining catalog (~1,300 shows, ~540k
episodes, ~48k transcripts) with adaptive feed polling and daily discovery;
weighted multilingual full-text search with hard structured filters; a
graded relevance set with an evaluation harness (Recall@K, MRR, NDCG,
coverage, latency) logging to MLflow; and a deterministic dataset pipeline
publishing versioned, honestly-labeled training snapshots.

**Ahead:** pretrained-embedding and two-tower retrieval, hard-negative
mining, the learned reranker, voice capture with speech-robustness
evaluation, the public search API, and production deployment with drift
monitoring.

## Development

Go 1.26+, [uv](https://docs.astral.sh/uv/), and Docker. `docker compose up
-d && make migrate && make ci` brings up the stack (Postgres + pgvector,
MinIO, MLflow, catalog worker) and runs all checks; see the
[Makefile](Makefile) for dataset, relevance-set, and evaluation targets,
and [.env.example](.env.example) for optional credentials.

- `ml/` — datasets, features, models, training, evaluation (Python)
- `cmd/`, `internal/` — API, catalog worker, ML worker (Go)
- `migrations/` — PostgreSQL schema
- `deploy/` — Compose, Helm, and OpenShift manifests
- `experiments/` — checked-in experiment configs and reports
- `docs/` — [data card](docs/data-card.md), [licensing notes](docs/licensing.md)

## Licensing and data responsibilities

Podcast metadata, artwork, and transcripts delivered over public RSS remain
the property of their publishers. This project stores only required
metadata and artifacts, preserves attribution, and honors feed removal
requests — see [docs/licensing.md](docs/licensing.md) before ingesting or
retaining any content.
