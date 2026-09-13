# Technology Stack & Choices

## Overview

| Layer | Technology | Why |
|-------|-----------|-----|
| **Language (ML)** | Python 3.11+ | ML/data libraries; experiment velocity |
| **Language (Backend)** | Go 1.26+ | Low-latency serving; minimal dependencies |
| **Database** | PostgreSQL 15+ with pgvector | Single source of truth; vector search built-in |
| **Object Storage** | MinIO or S3 | Vectors, datasets, experiment outputs |
| **ML Framework** | PyTorch 2.6+ | Industry standard; stable API |
| **Embeddings** | sentence-transformers 3.x | Fast, lightweight, community models |
| **Experiment Tracking** | MLflow 3.1+ | Reproducible runs; parameter + metric logging |
| **Data Validation** | Pandera 0.22+ | Schema validation for datasets |
| **Data Format** | Parquet (PyArrow 17+) | Compression; type safety; Spark compatible |
| **Local Dev** | Docker Compose 2.x | Postgres + MinIO + MLflow + workers |

## Backend (Go)

### Language & Runtime
- **Go 1.26+**: Low-latency serving, static compilation, standard library covers most needs
- **Deployment**: Single binary; no runtime dependencies beyond PostgreSQL

### Core Libraries
- **pgx**: PostgreSQL driver with prepared statements and connection pooling
- **chi**: HTTP router; lightweight and composable
- **pgvector-go**: Native pgvector support in Go
- **etcd/bbolt**: Embedded KV store for distributed locking (future: Postgres advisory locks)

### Worker Frameworks
- **Background jobs**: Custom job queue in PostgreSQL (outbox pattern)
- **Observability**: gRPC for inter-service communication; structured JSON logging

### Why Go for Ingestion & Serving

1. **Performance**: Single-threaded Go goroutine scheduling outperforms Python threads for I/O-heavy work (RSS polling, DB queries)
2. **Deployability**: Compiles to single binary; no runtime to manage
3. **Concurrency**: Goroutines + channels elegantly handle background workers and fan-out operations
4. **Library maturity**: Excellent database and web frameworks; stable APIs

### Tradeoffs
- No ML inference in Go; calls out to Python workers or uses precomputed embeddings
- Larger binary size than minimal languages (but still <50MB)

## ML & Data (Python)

### Python 3.11+
- **Version lock**: Specified in pyproject.toml; `uv` locks exact dependency tree in uv.lock
- **Environment**: Managed via `uv sync`; no system Python pollution

### PyTorch 2.6+
- **Use cases**: Embedding inference, future reranker training
- **Device handling**: CPU for small models; GPU optional for training
- **Serialization**: torch.save() for checkpoints; ONNX for cross-platform serving

### sentence-transformers 3.x
- **Models**: multilingual-e5-small (384-dim); evaluate larger models (e5-base-v2, BGE) for quality
- **Pretrained weights**: Cached locally or via HuggingFace Hub
- **Inference**: Batch processing via workers; GPU optional
- **Licensing**: Apache 2.0 / MIT; compatible with ecosystem

### MLflow 3.1+
- **Experiment tracking**: Log hyperparameters, metrics, model artifacts
- **UI**: Local (sqlite) or remote (server); accessible at http://localhost:5000
- **Model registry**: Version models; promote to "production"
- **Integration**: Automatic logging in training scripts; no boilerplate

### Pandera 0.22+
- **Schema validation**: Ensure datasets conform to expected columns + types
- **Error reporting**: Clear errors on validation failure
- **Performance**: Compiled checks; minimal overhead

### PyArrow 17+ (Parquet)
- **Format**: Columnar, compressed; efficient for ML pipelines
- **Cross-language**: Parquet is language-agnostic; Go can read it
- **Type system**: Preserves column types (bool, int32, float64, string, list, struct)
- **Integration**: pandas.read_parquet() / write_parquet()

### scikit-learn Metrics
- **Recall, MRR, NDCG**: Industry-standard ranking metrics
- **Lightweight**: No neural networks; pure numpy

## Data Storage

### PostgreSQL 15+

**Why PostgreSQL over MongoDB, DynamoDB, or specialized stores?**

1. **Relational schema**: Shows, episodes, transcripts have clear relationships
2. **Full-text search**: Native multilingual FTS via `tsvector` and `GIN` indices
3. **Transactions**: Atomic updates; ACID guarantees for catalog consistency
4. **Ecosystem**: Excellent tooling, monitoring, backups (pg_dump, WAL archiving)
5. **Cost**: Self-hosted; no per-query billing

**Extensions:**
- **pgvector**: Adds vector type and similarity operators (L2, cosine, IP)
  - Indexing: HNSW or IVFFlat for approximate nearest neighbor search
  - Queries: `<->` (L2 distance), `<#>` (negative IP), `<=>` (cosine)
  - Scaling: Indexes on vectors > 384 dims for catalog size

**Schema design:**
- Episodes, shows, transcripts normalized
- Vectors stored as `vector(384)` columns (one per embedding model version)
- Full-text indices on title, description, transcript
- Foreign keys enforce consistency

### MinIO (S3-compatible Object Storage)

**Why MinIO?**

1. **Self-hosted**: No AWS account needed for local development
2. **S3-compatible**: Swap MinIO for AWS S3 in production (no code changes)
3. **Cost**: Free and open source
4. **Storage**: Vectors, datasets, experiment outputs

**Use cases:**
- Store embedding vectors as numpy arrays or feather files (backup)
- Large training datasets too big for PostgreSQL
- Experiment reports (metrics, plots, model weights)

**Lifecycle:**
- Versioned by experiment name; never overwrite
- Cleanup: Explicit deletion after analysis

### Hybrid Data Location

| Data | Location | Why |
|------|----------|-----|
| Episodes, shows, transcripts | PostgreSQL | Relational, queryable |
| Episode vectors | pgvector (PostgreSQL) | Fast NN search, no extra service |
| Large datasets (Parquet) | MinIO | Compression, version control |
| Experiment outputs | MinIO | Audit trail, reproducibility |
| Relevance judgments | PostgreSQL | Queryable, referenced by eval |

## Development Environment

### Docker Compose
- **Services**: PostgreSQL, MinIO, MLflow, Postgres admin UI
- **Networking**: All services on same bridge; accessible via container names
- **Volumes**: Persistent data between restarts
- **Configuration**: .env file for credentials; .env.example for defaults

### uv (Python Package Manager)
- **Installation**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Workflow**: `uv sync` (installs + locks), `uv run` (runs commands in venv)
- **Benefits**: Faster than pip/Poetry; deterministic locks; Python 3.11+ support
- **Lock file**: uv.lock committed to git; ensures reproducible installs

### Makefile
- **Targets**: Common tasks (migrate, test, lint, dataset building, eval)
- **Variables**: CONFIG, SNAPSHOT, POOL_SYSTEMS for parameterization
- **Conventions**: `make ci` runs all pre-commit checks; `make build` compiles

## Testing

### Python Tests
- **Framework**: pytest 8.x
- **Location**: ml/tests/
- **Scope**: Unit tests for datasets, metrics, retrieval logic
- **Mocking**: Mocks PostgreSQL via fixtures; no live DB in unit tests
- **CI**: Run via `make test`

### Go Tests
- **Framework**: testing package (stdlib)
- **Location**: Tests alongside code in same package
- **Scope**: Unit + integration tests for retrieval, catalog, migrations
- **Database**: pgtest helper for isolated test databases
- **CI**: Run via `make test`

### Linting

**Python:**
- **ruff 0.9+**: Fast, unified linter (flake8 + isort + pyupgrade)
- **Config**: pyproject.toml; rules: E, F, I, UP, B
- **CI**: `uv run ruff check .`

**Go:**
- **vet**: Built-in Go analyzer
- **goimports**: Format imports
- **CI**: `go vet ./...`

## Dependency Management

### Python (uv.lock)
```
[project]
dependencies = [
    "torch>=2.6,<3",
    "mlflow>=3.1,<4",
    ...
]

[dependency-groups]
dev = ["pytest>=8,<9", "ruff>=0.9,<1"]
```
- Ranges allow patch updates; minor/major locked
- Dev dependencies separate; not included in production installs

### Go (go.mod / go.sum)
- Pinned to exact semver
- Prefer stable, battle-tested libraries
- No monorepo vendoring; rely on go.sum for reproducibility

## Versioning Strategy

### Datasets & Snapshots
- Named by date + experiment name: `snapshot-2026-01-15-v1`, `dataset-search-bootstrap-v1`
- Versioned in experiments/configs: explicit reference in eval configs
- Never overwrite; create new version for changes

### Models
- Exported with git commit hash + timestamp
- Registered in MLflow; promoted to "production" via UI
- Rollback via pointing API to prior model version

### Code
- Git commit hash is source of truth
- Tags for releases; branches for features
- Experiment configs commit hash of code; ensures reproducible re-runs

## Production Readiness

### What's Required
- [ ] PostgreSQL 15+ with pgvector extension (tested on Linux + macOS)
- [ ] Object storage (S3 or MinIO)
- [ ] Python 3.11+ and uv
- [ ] Go 1.26+
- [ ] Docker + Compose (optional, for full stack deployment)

### What's Recommended
- Postgres replication for HA
- S3 versioning for object storage
- MLflow remote server (vs. local sqlite)
- Monitoring (Prometheus + Grafana) on API latency
- Structured logging (JSON) with ELK aggregation

### What's Not Included
- Authentication/authorization (application layer)
- Rate limiting (should add to API server)
- Content delivery (no CDN strategy for transcripts)
- Multi-region failover
