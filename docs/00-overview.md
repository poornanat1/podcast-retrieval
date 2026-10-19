# Overview

PodFind is a podcast search system built end to end: catalog ingestion,
lexical and vector retrieval, a fine-tuned retrieval model, and offline
evaluation against a graded relevance set. Python owns the ML lifecycle,
Go owns ingestion (and, later, serving), PostgreSQL with `pgvector` holds
all the data.

## Status

Built and measured:

- **Catalog** — 1,790 shows, ~797k episodes, ~87k publisher transcripts,
  maintained by a Go worker with adaptive feed polling and daily
  discovery. The evaluation corpus is held fixed; see
  [09-catalog.md](09-catalog.md).
- **Retrieval** — weighted multilingual full-text search, pretrained
  embedding search (`multilingual-e5-small` in `pgvector`), hybrid
  reciprocal-rank fusion, and a fine-tuned two-tower model with its own
  versioned index. [04-retrieval.md](04-retrieval.md)
- **Relevance set and evaluation** — 180 queries, 50,856 graded judgments
  pooled from every system, LLM-graded with human override; a harness
  reporting recall, hit rate, MRR, NDCG, coverage, cohorts, latency, and
  pool coverage, logging to MLflow. [06-relevance.md](06-relevance.md),
  [07-evaluation.md](07-evaluation.md)
- **Datasets and training** — deterministic versioned datasets, a
  training loop with judged-pool validation, hard-negative mining and
  auditing, LLM-written training queries, and acceptance gates with MLflow
  model registration. [03-data-pipeline.md](03-data-pipeline.md),
  [08-models.md](08-models.md)

Not built yet: the public search API and web app, voice input, a learned
reranker, deployment manifests, and monitoring. Their directories and Go
entry points are scaffolds.

## Documents

| Document | Contents |
|---|---|
| [01-architecture.md](01-architecture.md) | Components and how data moves between them |
| [02-tech-stack.md](02-tech-stack.md) | Languages, libraries, and services, and why |
| [03-data-pipeline.md](03-data-pipeline.md) | Snapshots, dataset builder, raw inputs, versioning |
| [04-retrieval.md](04-retrieval.md) | Lexical, vector, two-tower, and hybrid retrieval |
| [05-embeddings.md](05-embeddings.md) | Embedding model choice, batch jobs, vector storage |
| [06-relevance.md](06-relevance.md) | Queries, pooling, grading, judges |
| [07-evaluation.md](07-evaluation.md) | Metrics, configs, harness, reproducibility |
| [08-models.md](08-models.md) | Two-tower model, training loop, artifacts, indexes |
| [09-catalog.md](09-catalog.md) | Discovery, RSS polling, dedup, transcripts, jobs |
| [10-database.md](10-database.md) | Schema and migrations |
| [11-deployment.md](11-deployment.md) | What runs where today |
| [12-observability.md](12-observability.md) | What is tracked today |
| [13-development.md](13-development.md) | Setup, Make targets, tests, conventions |
| [data-card.md](data-card.md) | Data sources, label taxonomy, freeze state, training-data coverage |
| [model-card.md](model-card.md) | The registered retriever: data, evaluation, cold start, failure modes |
| [licensing.md](licensing.md) | Rights and responsibilities for podcast content |

## Layout

```
cmd/            catalog-worker, migrate; api and ml-worker are scaffolds
internal/       catalog, rss, transcript, discovery, particle, jobs, outbox, objstore, migrate, pgtest, version
migrations/     PostgreSQL schema, applied by `make migrate`
ml/             datasets, features, models, training, embeddings, hard_negatives, retrieval, relevance, evaluation, tests
experiments/    datasets/, training/, eval/ configs and the reports
data/           relevance/ (tracked); snapshots/, datasets/, models/ (local, gitignored)
deploy/         compose init script; helm/ and openshift/ are placeholders
```
