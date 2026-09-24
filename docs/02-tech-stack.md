# Tech stack

| Layer | Choice | Why |
|---|---|---|
| Database | PostgreSQL 17 + `pgvector` | Relational catalog, multilingual full-text search, and ANN vector search in one store; transactional job queue and outbox on the same connection |
| Ingestion | Go 1.26, `pgx` | Long-running, I/O-bound worker with a small footprint and one binary |
| ML | Python 3.11+, PyTorch, `transformers`, `sentence-transformers` | The models and tooling live here |
| Data | pandas, PyArrow (Parquet), Pandera | Columnar, typed, validated dataset files |
| Tracking | MLflow 3 | Runs, parameters, metric histories, artifacts |
| Object store | MinIO (S3 API), `minio-go` | Published dataset versions and raw transcripts |
| LLM | OpenAI API (`gpt-5.1`) | Relevance judging and training-query generation |
| Tooling | `uv`, `ruff`, `pytest`, `go vet`, Docker Compose, GitHub Actions | |

## Go

Dependencies are `github.com/jackc/pgx/v5` and `github.com/minio/minio-go/v7`;
everything else is the standard library. The job queue uses
`FOR UPDATE SKIP LOCKED` with leases, exponential backoff, and a
dead-letter state; the outbox commits domain events with the state change
that caused them.

## Python

`pyproject.toml` pins ranges; `uv.lock` pins exact versions. Runtime
dependencies: `torch`, `mlflow`, `pandera`, `pyarrow`, `psycopg`, `minio`,
`openai`, `sentence-transformers`, `pgvector`. Dev: `pytest`, `ruff`.

Models run on Apple Metal (`mps`), CUDA, or CPU; the training loop and
embedding jobs pick whichever is available.

## Services (Docker Compose)

| Service | Port | Notes |
|---|---|---|
| `postgres` (`pgvector/pgvector:pg17`) | 5432 | user/db `podfind` |
| `minio` | 9000 / 9001 | buckets created by `minio-init` |
| `mlflow` | 5001 | port 5000 is taken by macOS AirPlay |
| `catalog-worker` | | reads `PARTICLE_API_KEY` from `.env` |

## Why not …

- **A separate vector database** — at ~800k vectors of 384 dimensions
  the HNSW index fits in memory next to the catalog, and one store is
  simpler to keep consistent.
- **Hosted embeddings** — self-hosted encoders are deterministic,
  versioned, and keep catalog text in the stack; the fine-tuning work
  needs the weights anyway.
- **An English-only encoder** — 23% of the catalog is not English, and
  the language filter is a hard predicate; `multilingual-e5-small` was
  chosen for that (see [05-embeddings.md](05-embeddings.md)).
