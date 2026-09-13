# Database: PostgreSQL Schema

PostgreSQL with pgvector extension is the single source of truth for all data.

## Schema Overview

```mermaid
erDiagram
    SHOWS ||--o{ EPISODES : contains
    EPISODES ||--o{ RELEVANCE_JUDGMENTS : has
    EPISODES ||--o{ USER_INTERACTIONS : records
    
    SHOWS {
        bigint id PK
        string title
        string rss_url UK
        string description
        string language
        string category
        string artwork_url
        timestamp last_polled_at
        int polling_interval_hours
        int poll_error_count
        boolean is_active
        timestamp created_at
        timestamp updated_at
    }
    
    EPISODES {
        bigint id PK
        bigint show_id FK
        string guid UK
        string title
        string description
        string transcript
        int duration_seconds
        boolean explicit
        string enclosure_url
        timestamp published_at
        timestamp created_at
        timestamp updated_at
        tsvector tsv "FTS index"
        vector embedding "384-dim e5"
        string embedding_model
        timestamp embedding_updated_at
        int listen_count
        float popularity_score
        string transcript_source
        timestamp transcript_updated_at
    }
    
    RELEVANCE_JUDGMENTS {
        bigint id PK
        string query
        bigint episode_id FK
        string relevance_grade
        string annotator_id
        timestamp judged_at
        string notes
    }
    
    USER_INTERACTIONS {
        bigint id PK
        string user_id
        string query
        bigint episode_id FK
        string event_type
        int rank
        timestamp click_timestamp
        int play_duration_seconds
        timestamp created_at
    }
```

## Table Details

### Shows
Podcast show metadata with RSS feed tracking.

**Key columns:**
- `rss_url`: Source feed URL (unique)
- `is_active`: Whether feed is currently being polled
- `last_polled_at`: Most recent update timestamp
- `polling_interval_hours`: Adaptive schedule (4h for active, 48h for inactive)

**Indices:**
- `(is_active, last_polled_at)` for finding feeds needing poll
- `(created_at DESC)` for recent shows

### Episodes
Individual podcast episodes with searchable content.

**Key columns:**
- `guid`: RSS unique identifier (for deduplication)
- `transcript`: Full-text transcript (if available)
- `tsv`: Tsvector for PostgreSQL full-text search (auto-maintained trigger)
- `embedding`: 384-dim vector from e5-small-v2 model
- `published_at`: Publication timestamp (not creation time)

**Indices:**
- `GIN(tsv)` for full-text search
- `HNSW(embedding)` for vector similarity search
- `(show_id, published_at DESC)` for feed chronology
- `(published_at DESC)` for recent episodes

### Relevance Judgments
Human-labeled ground truth for evaluation.

**Key columns:**
- `relevance_grade`: {relevant, somewhat_relevant, not_relevant}
- `annotator_id`: Who made the judgment
- `query`: Search query text (not just ID)

**Unique constraint:** `(query, episode_id, annotator_id)` to prevent duplicate judgments

### User Interactions (Future)
Implicit feedback from search behavior.

**Key columns:**
- `event_type`: search, click, play, share
- `rank`: Position in search results (useful for learning to rank)
- `play_duration_seconds`: How long user listened

## Initialization

### Quick Start

```bash
# Start PostgreSQL (via Docker or local install)
docker compose up postgres

# Create database & user
createdb podfind
psql podfind -c "CREATE USER podfind WITH PASSWORD 'podfind';"

# Install pgvector extension
psql podfind -c "CREATE EXTENSION vector;"

# Run migrations
make migrate
```

### Environment

Set in `.env`:
```bash
DATABASE_URL=postgres://podfind:podfind@localhost:5432/podfind?sslmode=disable
```

## Migrations

Versioned schema changes in `migrations/`:

```
0001_init_schema.sql        # Create shows, episodes tables
0002_add_pgvector.sql        # Install pgvector extension
0003_add_relevance.sql       # Create relevance_judgments
0010_episode_embeddings.sql  # Add embedding columns
...
```

Run migrations:
```bash
make migrate
```

## Connection

### Go

```go
config, _ := pgxpool.ParseConfig(os.Getenv("DATABASE_URL"))
config.MaxConns = 25
pool, _ := pgxpool.NewWithConfig(context.Background(), config)
```

### Python

```python
import psycopg_pool
pool = psycopg_pool.ConnectionPool(os.getenv("DATABASE_URL"), max_size=20)
```

## Key Search Operations

| Operation | Index | Latency |
|-----------|-------|---------|
| Lexical (BM25) | GIN(tsv) | ~20ms |
| Vector (semantic) | HNSW(embedding) | ~50ms |
| By show_id + date | (show_id, published_at) | ~5ms |
| By relevance | (episode_id, relevance_grade) | ~10ms |

## Connection Pooling

Maintain 5-25 persistent connections; reuse across requests.

**Configuration:**
- Max connections: 25
- Min idle: 5
- Timeout: 5 minutes
- Prepared statements: enabled

## Backup & Recovery

**Backup strategy:**
- Daily full backup to S3 (gzip)
- Hourly incremental (WAL archiving)
- Retention: 30 days

**RTO/RPO:**
- Recovery Time: < 1 hour
- Recovery Point: < 15 minutes

## Maintenance Tasks

**Weekly:**
- `VACUUM ANALYZE` on large tables (episodes, shows)
- Monitor index bloat

**Monthly:**
- Review slow query logs
- Reindex if fragmentation > 10%
- Test backup restoration

**Key queries:**
- `pg_stat_statements` for slow queries
- `pg_stat_user_indexes` for unused indices
- `pg_stat_activity` for connection leaks

## Production Deployment

### Hardware

| Scenario | RAM | CPU | Storage |
|----------|-----|-----|---------|
| Development | 2GB | 2 | 20GB |
| Staging | 8GB | 4 | 100GB |
| Production | 32GB+ | 8+ | 500GB+ |

### High Availability

1. **Replication**: Standby replica with streaming WAL
2. **Monitoring**: Alert on lag, connection leaks, disk space
3. **Failover**: Automatic via pg_auto_failover or Patroni

### Scaling Strategy

- **Step 1**: Vertical (more RAM/CPU on single machine)
- **Step 2**: Read replicas for query load
- **Step 3**: Sharding by show_id if catalog > 10M episodes (future)

## Quick Reference

### Common Administrative Tasks

**Create backup:**
```bash
pg_dump podfind | gzip > backup-$(date +%Y%m%d).sql.gz
```

**Restore backup:**
```bash
gunzip < backup-20260115.sql.gz | psql podfind
```

**List table sizes:**
```bash
\dt+ episodes shows  # in psql
```

**Find slow queries (once enabled):**
```bash
SELECT query, mean_exec_time FROM pg_stat_statements 
ORDER BY mean_exec_time DESC LIMIT 10;
```

**Check connection count:**
```bash
SELECT datname, count(*) FROM pg_stat_activity GROUP BY datname;
```

## Architecture Notes

### Why PostgreSQL + pgvector?

- **Single source of truth**: All data in one system
- **ACID guarantees**: Catalog updates are consistent
- **Full-text search**: Native multilingual FTS
- **Vector indexing**: HNSW/IVFFlat for semantic search
- **Operational simplicity**: No separate vector DB

### Trade-offs

- **Pro**: Simpler operations, lower cost, single backup
- **Con**: Vector search slower than specialized DBs at massive scale (>10M vectors)
- **Decision**: For ~540k episodes, PostgreSQL is sufficient

## Future Enhancements

1. **Partitioning**: Partition episodes by publication date for faster queries
2. **Materialized views**: Pre-compute popular queries
3. **Read replicas**: Scale read-heavy workloads
4. **Sharding**: Partition by show_id if catalog grows to 10M+ episodes
