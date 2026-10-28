# Overview

PodFind is a podcast search system built end to end: catalog ingestion,
lexical and vector retrieval, a fine-tuned retrieval model, and offline
evaluation against a graded relevance set. Python owns the ML lifecycle,
Go owns ingestion (and, later, serving), PostgreSQL with `pgvector` holds
all the data.

## Status

Built and measured:

- **Catalog** — 1,788 shows, ~796k episodes, ~87k publisher transcripts,
  maintained by a Go worker with adaptive feed polling and daily
  discovery. The evaluation corpus is held fixed; see
  [09-catalog.md](09-catalog.md).
- **Retrieval** — weighted multilingual full-text search, pretrained
  embedding search (`multilingual-e5-small` in `pgvector`), hybrid
  reciprocal-rank fusion, a fine-tuned two-tower model with its own
  versioned index, and a learned LightGBM ranker over the union of all
  three legs' candidates. [04-retrieval.md](04-retrieval.md)
- **Relevance set and evaluation** — 180 queries, 71,518 graded judgments
  pooled from every system, LLM-graded with human override; a harness
  reporting recall, hit rate, MRR, NDCG, coverage, cohorts, latency, and
  pool coverage, logging to MLflow. [06-relevance.md](06-relevance.md),
  [07-evaluation.md](07-evaluation.md)
- **Datasets and training** — deterministic versioned datasets, a
  training loop with judged-pool validation, hard-negative mining and
  auditing, LLM-written training queries, and acceptance gates with MLflow
  model registration. [03-data-pipeline.md](03-data-pipeline.md),
  [08-models.md](08-models.md)

- **Learned ranking** — a LightGBM ranker over the union of the three
  retrieval legs' candidates, evaluated out of fold.
  [04-retrieval.md](04-retrieval.md), [08-models.md](08-models.md)
- **Serving** — a Go API with query understanding, concurrent hybrid
  retrieval, reranking through an ONNX model server, stored searches,
  attributed interaction events, and a minimal web app; promotion of a
  model bundle is gated on training-serving parity checks.
  [14-serving.md](14-serving.md)

- **Voice** — spoken queries through a local speech server, stored
  transcripts, consent-gated recording retention, session data export and
  deletion, and a voice-robustness evaluation on a synthetic speech test
  set. [15-voice.md](15-voice.md)

- **Deployment and monitoring** — one container image per process,
  declared for Railway as infrastructure in code with Postgres and object
  storage; Prometheus metrics from every service, a provisioned Grafana
  dashboard with alert rules, a scheduled drift check against a recorded
  baseline (exercised with a real serving skew), and a load tester for
  the SLOs. [11-deployment.md](11-deployment.md),
  [16-monitoring.md](16-monitoring.md)

The ML worker entry point is a scaffold.

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
| [11-deployment.md](11-deployment.md) | Railway: services, images, first deployment, catalog load, rollback; the local stack |
| [12-observability.md](12-observability.md) | What is tracked today |
| [13-development.md](13-development.md) | Setup, Make targets, tests, conventions |
| [14-serving.md](14-serving.md) | Model bundle, model server, Go API, events, parity and promotion |
| [15-voice.md](15-voice.md) | Voice search, the speech server, data controls, the speech test set and its evaluation |
| [16-monitoring.md](16-monitoring.md) | Metrics, alerts, the drift check and its skew demonstration, the load test |
| [data-card.md](data-card.md) | Data sources, label taxonomy, freeze state, training-data coverage |
| [model-card.md](model-card.md) | The registered retriever: data, evaluation, cold start, failure modes |
| [licensing.md](licensing.md) | Rights and responsibilities for podcast content |

## Layout

```
cmd/            api, catalog-worker, migrate, loadtest; ml-worker is a scaffold
internal/       api, search, catalog, rss, transcript, discovery, particle, jobs, outbox, objstore, metrics, logging, migrate, pgtest, version
migrations/     PostgreSQL schema, applied by `make migrate`
ml/             datasets, features, models, training, embeddings, hard_negatives, retrieval, ranking, serving, speech, monitoring, sql, relevance, evaluation, tests
web/            the web app, embedded into the API
api/openapi/    the API's OpenAPI description
experiments/    datasets/, training/, eval/, ranking/, speech/, monitoring/, load/ configs, baselines and the reports
data/           relevance/ (tracked); snapshots/, datasets/, models/ (local, gitignored)
deploy/         compose init script, Railway helpers, the Prometheus and Grafana images
.railway/       the Railway project as infrastructure in code
```
