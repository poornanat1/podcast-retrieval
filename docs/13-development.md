# Development: Workflow, Testing & Contributing

Guide for local development, testing, and contribution workflows.

## Development Environment Setup

### Prerequisites

- **Go 1.26+**: https://golang.org/dl/
- **Python 3.11+**: https://www.python.org/downloads/
- **PostgreSQL 15+**: https://www.postgresql.org/download/
- **Docker & Docker Compose**: https://www.docker.com/products/docker-desktop/
- **Git**: https://git-scm.com/

### Quick Start

```bash
# Clone repository
git clone https://github.com/your/podcast-retrieval.git
cd podcast-retrieval

# Setup Python environment
uv sync

# Setup Go dependencies
go mod download

# Copy environment file
cp .env.example .env

# Start Docker services
docker compose up -d

# Run migrations
make migrate

# Run all checks (lint + test + build)
make ci

# Start API server
go run ./cmd/serve

# In another terminal: run catalog worker
go run ./cmd/catalog-worker

# In another terminal: run embedding worker
go run ./cmd/embed-worker
```

**Access points:**
- API: http://localhost:8080
- MLflow: http://localhost:5000
- PostgreSQL: localhost:5432 (user: podfind, pass: podfind)

## Make Targets

Common development tasks:

```bash
make python-env      # Install Python dependencies
make build           # Build Go binaries + compile Python
make test            # Run Go + Python tests
make lint            # Run linters (ruff for Python, vet for Go)
make ci              # Run full CI pipeline (lint + test + build)

make migrate         # Run database migrations
make snapshot        # Create snapshot of current catalog
make dataset         # Build dataset from snapshot
make relevance-pool  # Pool results from multiple retrieval systems
make relevance-judge # LLM-assisted relevance labeling
make eval            # Evaluate a single config
make eval-all        # Evaluate all configs in experiments/eval/

make embed           # Embed all episodes with vectors
```

## Project Structure

```
ml/                      # Python: ML pipeline
├── __init__.py
├── datasets/            # Dataset building
│   ├── snapshot.py      # Create snapshots
│   ├── build.py         # Build datasets from snapshots
│   └── __init__.py
├── embeddings/          # Embedding models
│   ├── embed.py         # Batch embed episodes
│   └── __init__.py
├── evaluation/          # Evaluation harness
│   ├── run.py           # Run evaluation
│   ├── metrics.py       # Metric computation
│   └── __init__.py
├── models/              # ML model definitions
│   ├── reranker.py      # Two-tower reranker
│   └── __init__.py
├── relevance/           # Relevance judgments
│   ├── pool.py          # Create relevance pool
│   ├── queries.py       # Manage queries
│   ├── review.py        # UI for human review
│   ├── llm_judge.py     # LLM-assisted labeling
│   └── __init__.py
├── retrieval/           # Search implementations
│   ├── lexical.py       # BM25 search
│   ├── vector.py        # Vector search
│   ├── hybrid.py        # RRF hybrid search
│   └── __init__.py
├── training/            # Model training
│   ├── prepare_data.py  # Create training triplets
│   ├── train.py         # Training loop
│   └── __init__.py
└── tests/               # Python unit tests
    ├── test_retrieval.py
    ├── test_datasets.py
    └── __init__.py

cmd/                     # Go entry points
├── serve/               # API server
├── catalog-worker/      # RSS polling worker
├── embed-worker/        # Embedding inference worker
├── migrate/             # Database migrations
└── ...

internal/                # Go internal libraries
├── catalog/             # Podcast catalog management
├── discovery/           # Feed discovery
├── jobs/                # Job queue
├── migrate/             # Migration runner
├── objstore/            # Object storage (MinIO)
├── rss/                 # RSS parsing
├── transcript/          # Transcript handling
├── version/             # Version info
└── ...

migrations/              # PostgreSQL migrations
├── 0001_init_schema.sql
├── 0002_add_pgvector.sql
└── ...

experiments/             # Checked-in configs
├── datasets/
│   └── search-bootstrap-v1.json
├── eval/
│   ├── lexical-v1.json
│   ├── vector-v1.json
│   └── hybrid-rrf-v1.json
└── ...

data/                    # Data artifacts
├── snapshots/           # Database dumps
├── datasets/            # Training/eval datasets
├── relevance/           # Judgment sets
└── ...

deploy/                  # Deployment configs
├── docker-compose.yml
├── helm/                # Kubernetes
├── ...

docs/                    # This documentation
├── 00-overview.md
├── 01-architecture.md
└── ...
```

## Code Style & Conventions

### Go

- **Formatting**: `gofmt` (enforced by `go fmt`)
- **Linting**: `go vet`
- **Naming**: CamelCase for exported, camelCase for internal
- **Comments**: Explain WHY, not WHAT; comment exported functions

```go
// Good: Explains decision
if err := db.Ping(ctx); err != nil {
    // Connection pool might be exhausted; retry with exponential backoff
    return retryWithBackoff(func() error { return db.Ping(ctx) })
}

// Avoid: Obvious
if err != nil {
    return err
}
```

### Python

- **Formatting**: Enforced by `ruff` (line length 100)
- **Type hints**: Use type annotations
- **Docstrings**: Brief (one-liner); avoid multi-paragraph blocks

```python
# Good: Type hints + brief docstring
def recall_at_k(predictions: List[int], targets: Set[int], k: int = 10) -> float:
    """Compute recall@k."""
    top_k = set(predictions[:k])
    return len(top_k & targets) / len(targets) if targets else 0.0

# Avoid: No types, verbose comments
def recall_at_k(predictions, targets, k):
    # This function computes the recall@k metric, which is the
    # percentage of relevant items found in the top-k predictions.
    # The function takes three parameters: predictions (a list of
    # predicted item IDs), targets (a set of relevant item IDs),
    # and k (the cutoff for top-k).
    ...
```

## Testing

### Python Tests

```bash
# Run all tests
uv run pytest -q

# Run specific test file
uv run pytest ml/tests/test_retrieval.py

# Run with coverage
uv run pytest --cov=ml --cov-report=html

# Run with verbose output
uv run pytest -v
```

**Test structure:**
```python
# ml/tests/test_retrieval.py
import pytest
from ml.retrieval import lexical_search, vector_search

def test_lexical_search_finds_exact_match(mock_db):
    """Lexical search returns exact matches first."""
    results = lexical_search("machine learning", mock_db)
    assert any("machine learning" in r.title for r in results)

def test_vector_search_handles_missing_embedding():
    """Vector search returns empty results if embedding is None."""
    results = vector_search(None)
    assert len(results) == 0

@pytest.fixture
def mock_db():
    """Mock database for testing."""
    # Setup mock
    yield mock_connection
    # Teardown
```

### Go Tests

```bash
# Run all tests
go test ./...

# Run with coverage
go test -cover ./...

# Run specific package
go test ./internal/catalog

# Verbose output
go test -v ./...
```

**Test structure:**
```go
// internal/retrieval/lexical_test.go
package retrieval

import (
    "testing"
)

func TestLexicalSearchFindsExactMatch(t *testing.T) {
    db := setupTestDB(t)
    defer db.Close()
    
    results, err := LexicalSearch(db, "machine learning", 10)
    if err != nil {
        t.Fatalf("LexicalSearch failed: %v", err)
    }
    
    if len(results) == 0 {
        t.Errorf("Expected results, got none")
    }
}
```

### Integration Tests

Run full stack tests (with real database):

```bash
# Start Docker services if not already running
docker compose up -d

# Run integration tests
go test -tags=integration ./...

# Run with specific database
DATABASE_URL=postgres://... go test -tags=integration ./...
```

## Debugging

### Logging

Enable debug logging:

```bash
LOG_LEVEL=debug go run ./cmd/serve
```

In code:
```go
log.WithField("query", query).Debug("Executing search")
log.WithError(err).Error("Database query failed")
```

### Debugger (Delve for Go)

```bash
# Install delve
go install github.com/go-delve/delve/cmd/dlv@latest

# Debug Go program
dlv debug ./cmd/serve

# Set breakpoint and run
(dlv) break main.main
(dlv) continue
(dlv) print variable_name
(dlv) next
```

### Python Debugger (pdb)

```python
import pdb

def my_function():
    x = 42
    pdb.set_trace()  # Debugger will stop here
    y = x + 1
```

Or with IPython:
```bash
uv run ipython
In [1]: from ml.retrieval import lexical_search
In [2]: %debug lexical_search("query")
```

## Database Debugging

### Query Logs

Enable query logging in PostgreSQL:

```sql
-- In postgresql.conf
log_statement = 'all'
log_duration = on
log_min_duration_statement = 100  -- Log queries > 100ms
```

### Explain Plan

Understand query performance:

```sql
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM episodes 
WHERE tsv @@ websearch_to_tsquery('english', 'AI podcast')
LIMIT 10;
```

### Monitor Active Queries

```sql
SELECT pid, usename, query, state, query_start
FROM pg_stat_activity
WHERE state = 'active'
ORDER BY query_start;
```

## Documentation

### Adding Docs

1. Create file in `docs/` (e.g., `docs/14-new-topic.md`)
2. Link from `00-overview.md` reading guide
3. Update `README.md` if top-level change

### Docs Format

- Use Markdown
- Link related docs with `[doc-name](path-to-doc.md)`
- Include code examples
- Explain trade-offs & decisions

## Git Workflow

### Branch Naming

```
feature/implement-xyz          # New feature
bugfix/fix-crash-in-search     # Bug fix
docs/add-retrieval-guide       # Documentation
refactor/simplify-xyz          # Refactoring
perf/optimize-query-latency    # Performance improvement
```

### Commit Messages

```
Conventional Commits format:
feat: add new retrieval method
fix: handle null embedding vectors
docs: document deployment process
refactor: simplify lexical search
perf: cache embedding vectors
test: add tests for RRF merging
```

### Pull Requests

1. Create feature branch
2. Implement + test
3. Run `make ci` locally (ensure pass)
4. Create PR with description
5. Code review (at least 1 approval)
6. Merge to main

**PR template:**
```markdown
## Description
What this PR does...

## Changes
- Change 1
- Change 2

## Testing
How to test...

## Checklist
- [ ] Tests pass (`make ci`)
- [ ] Docs updated
- [ ] No breaking changes
- [ ] Database migration if needed
```

## Release Process

### Versioning

Semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Release Checklist

1. Update `VERSION` file
2. Update `CHANGELOG.md`
3. Tag commit: `git tag v1.2.3`
4. Push tag: `git push origin v1.2.3`
5. Build release artifacts (Docker image, binaries)
6. Create GitHub release
7. Deploy to production

## Performance Profiling

### Go CPU Profiling

```bash
# Run with CPU profiling
go run -cpuprofile=cpu.prof ./cmd/serve

# Analyze
go tool pprof cpu.prof
(pprof) top  # Top functions by CPU time
(pprof) list LexicalSearch  # Show code with profile
```

### Go Memory Profiling

```bash
# Heap snapshot
curl http://localhost:6060/debug/pprof/heap > heap.prof
go tool pprof heap.prof

# Memory allocations
curl http://localhost:6060/debug/pprof/allocs > allocs.prof
```

### Python Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
results = hybrid_search(query)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
```

## Continuous Integration

### GitHub Actions (.github/workflows/ci.yml)

```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: ankane/pgvector:latest
        env:
          POSTGRES_DB: podfind
          POSTGRES_USER: podfind
          POSTGRES_PASSWORD: podfind
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-go@v4
        with:
          go-version: '1.26'
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Run migrations
        run: DATABASE_URL=postgres://podfind:podfind@localhost/podfind make migrate

      - name: Run CI
        run: make ci
```

## Troubleshooting

### Common Issues

**PostgreSQL connection error:**
```bash
# Check if server is running
psql postgres -c "SELECT 1"

# Verify credentials in .env
grep DATABASE_URL .env

# Restart services
docker compose restart postgres
```

**Python import errors:**
```bash
# Reinstall dependencies
uv sync

# Check Python path
python -c "import sys; print(sys.path)"
```

**Go build errors:**
```bash
# Clear cache
go clean -cache

# Tidy dependencies
go mod tidy
```

## Next Steps for Contributors

1. Pick an issue labeled `good-first-issue`
2. Comment "I'll work on this"
3. Create feature branch
4. Implement + test
5. Open PR for review
6. Address feedback
7. Merge! 🎉

See [CONTRIBUTING.md](../CONTRIBUTING.md) for more details.
