# PodFind

Voice-aware podcast search and recommendation, built as an end-to-end ML engineering project. PodFind learns query, user, and episode representations; retrieves candidates with a two-tower model; reranks them with behavioral and content features; and serves versioned models through a low-latency Go API backed by PostgreSQL and `pgvector`.

## Prerequisites

- Go 1.26+
- [uv](https://docs.astral.sh/uv/) (manages the Python 3.11+ environment)
- Docker with Compose

## Quickstart

```sh
cp .env.example .env   # optional: Particle API key for discovery
docker compose up -d   # PostgreSQL + pgvector, MinIO, MLflow, catalog worker
make migrate           # apply database migrations
make ci                # lint, test, and build both toolchains
```

The catalog worker runs as a Compose service and keeps the catalog hands-off:
it polls every feed on an adaptive schedule (at least daily) and, when a
[Particle](https://docs.particle.pro) API key is set in `.env`, discovers
trending podcasts once a day (`PODFIND_DAILY_TRENDING`, default 25).

`make ci` bootstraps the Python environment via `uv sync` on first run.

Local services: MLflow at http://localhost:5001 (host port 5001 because macOS AirPlay occupies 5000), MinIO console at http://localhost:9001, PostgreSQL on 5432 (`podfind`/`podfind`).

## Layout

- `ml/` — datasets, features, models, training, evaluation (Python)
- `cmd/`, `internal/` — API, catalog worker, ML worker (Go)
- `migrations/` — PostgreSQL schema
- `deploy/` — Compose, Helm, and OpenShift manifests
- `experiments/` — checked-in experiment configs and reports
- `docs/` — data card, model card, [licensing notes](docs/licensing.md)

## Licensing and data responsibilities

Podcast metadata, artwork, and transcripts delivered over public RSS remain the property of their publishers. This project stores only required metadata and artifacts, preserves attribution, and honors feed removal requests — see [docs/licensing.md](docs/licensing.md) before ingesting or retaining any content.
