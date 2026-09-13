# PodFind: Overview

**PodFind** is a self-hosted podcast discovery system that makes spoken knowledge findable through voice-aware, semantically-aware search and recommendations.

## Vision & Mission

Podcasts hold enormous amounts of knowledge, but discovery is stuck at the show level. The moments worth finding are usually buried inside episodes nobody can name. PodFind's goal is to make that findable—say or type what you want and get the right episodes back, ranked, with constraints treated as guarantees rather than suggestions.

### Core Principles

1. **Search by meaning, not just words.** Hybrid lexical + vector retrieval puts queries, listeners, and episodes in a shared embedding space. Exploratory questions land on relevant episodes even when no title contains the query.

2. **Voice as a first-class input.** Spoken queries are transcribed, parsed into intent + structured filters (language, duration, recency, explicit content), and evaluated as part of retrieval quality—not a UI garnish.

3. **Recommendations that respect intent.** Personalized retrieval from listening history and "more like this episode" diversity across shows, not just more-of-the-same.

4. **Evidence before complexity.** Every learned component must beat reproducible baselines on a human-auditable relevance set before it ships. Datasets, models, and indexes are versioned and rebuildable bit-for-bit.

5. **Respect for the ecosystem.** Publisher RSS remains the source of truth. Attribution is preserved, audio is never copied, and removal requests are honored.

## Scope

A self-hosted stack:
- **Data ingestion:** Podcast discovery and RSS ingestion, publisher transcript parsing
- **Retrieval:** Hybrid lexical + vector retrieval with potential learned reranking
- **Training:** Interaction-event capture feeding back into training data
- **Operations:** Versioned model serving with offline evaluation, monitoring, and rollback

Technology split: Python owns the ML lifecycle (datasets, training, evaluation, embeddings); Go owns ingestion and serving; PostgreSQL (with `pgvector`) owns the data.

### What's Working Today

- **Catalog:** ~1,800 shows, ~800k episodes, ~87k transcripts with adaptive feed polling (evaluation corpus frozen at 795,533 episodes on 2026-09-13)
- **Search:** Weighted multilingual full-text search with hard structured filters
- **Relevance:** Graded relevance set (180 queries, ~8.9k LLM-graded judgments with human override) with evaluation harness (Recall@K, MRR, NDCG, coverage, latency)
- **Data:** Deterministic dataset pipeline publishing versioned, honestly-labeled training snapshots

### What's Ahead

- Pretrained embeddings and two-tower retrieval models
- Hard-negative mining for training
- Learned reranking
- Voice capture with speech-robustness evaluation
- Public search API and production deployment with drift monitoring

## Reading Guide

This documentation is structured in layers:

| Document | Audience | Purpose |
|----------|----------|---------|
| **[01-architecture.md](01-architecture.md)** | All | How the system is organized and what talks to what |
| **[02-tech-stack.md](02-tech-stack.md)** | All | Technology choices and why |
| **[03-data-pipeline.md](03-data-pipeline.md)** | ML, Data | Catalog ingestion, snapshots, datasets, versioning |
| **[04-retrieval.md](04-retrieval.md)** | ML, Backend | Lexical, vector, and hybrid search implementations |
| **[05-embeddings.md](05-embeddings.md)** | ML | Embedding models, indexing, vector storage |
| **[06-relevance.md](06-relevance.md)** | ML, Eval | Relevance judgments, human review, LLM annotation |
| **[07-evaluation.md](07-evaluation.md)** | ML, All | Metrics, experiment configs, eval harness |
| **[08-models.md](08-models.md)** | ML | ML model definitions, training, export |
| **[09-catalog.md](09-catalog.md)** | Backend, Data | Podcast catalog, RSS polling, feed management |
| **[10-database.md](10-database.md)** | Backend, Data | PostgreSQL schema, migrations, queries |
| **[11-deployment.md](11-deployment.md)** | Backend, DevOps | Docker Compose, Kubernetes, cloud deployment |
| **[12-observability.md](12-observability.md)** | All | Monitoring, MLflow, dashboards, logging |
| **[13-development.md](13-development.md)** | All | Development workflow, local setup, running targets |

## Key Directories

```
ml/              # Python: datasets, embeddings, models, training, evaluation
├── datasets/    # Dataset building and snapshots
├── embeddings/  # Embedding models and inference
├── evaluation/  # Metrics and eval harness
├── models/      # ML model definitions
├── relevance/   # Human relevance judgments and review
├── retrieval/   # Lexical, vector, and hybrid search
├── training/    # Model training scripts
└── tests/       # Python tests

cmd/             # Go: CLI entry points (migrate, server, workers)
internal/        # Go: core business logic
├── catalog/     # Podcast catalog and episode management
├── discovery/   # Feed discovery
├── jobs/        # Job queue and background workers
├── rss/         # RSS parsing and polling
├── transcript/  # Transcript parsing
└── ...

migrations/      # PostgreSQL schema migrations
deploy/          # Docker Compose, Helm, OpenShift manifests
experiments/     # Checked-in experiment configs and reports
data/            # Snapshots, datasets, relevance judgments
docs/            # This documentation
```

## Development Quick Start

```bash
# Start services (Postgres, MinIO, MLflow, catalog worker)
docker compose up -d

# Run migrations
make migrate

# Run all CI checks (lint, test, build)
make ci

# Build a dataset
make dataset CONFIG=experiments/datasets/search-bootstrap-v1.json

# Run evaluation
make eval-all
```

See [13-development.md](13-development.md) for the full development guide.

## Data Responsibilities

Podcast metadata, artwork, and transcripts delivered over public RSS remain the property of their publishers. PodFind stores only required metadata and artifacts, preserves attribution, and honors feed removal requests. See [../docs/licensing.md](../docs/licensing.md) for details.
