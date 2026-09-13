# Data Pipeline: Snapshots, Datasets & Versioning

The data pipeline ensures reproducible, auditable ML workflows by versioning snapshots and datasets.

## Data Pipeline Flow

```mermaid
graph LR
    A["PostgreSQL<br/>Live Catalog<br/>~800k episodes"] -->|"make snapshot"| B["Snapshot<br/>Parquet<br/>Point-in-time"]
    B -->|"make dataset"| C["Dataset<br/>Filtered + Enriched<br/>Training-ready"]
    C -->|"make relevance-pool"| D["Relevance Pool<br/>180 queries<br/>~8.9k graded judgments"]
    D -->|"make eval"| E["Evaluation<br/>Metrics<br/>MLflow"]
    
    style A fill:#e1f5ff
    style B fill:#f3e5f5
    style C fill:#e8f5e9
    style D fill:#fff3e0
    style E fill:#fce4ec
```

## Snapshots: Point-in-Time Captures

**Purpose**: Immutable database snapshot for reproducible dataset building.

### Key Properties

| Property | Benefit |
|----------|---------|
| **Immutable** | Never modified; new data = new version |
| **Self-contained** | No live DB dependency for dataset building |
| **Auditable** | Links experiment → dataset → catalog state |
| **Versionable** | Checked in; fully reproducible from git |

**Create snapshot:**
```bash
make snapshot
```

**Output**: Parquet files + metadata.json (timestamp, row counts, git commit)

### Snapshot Retention
- Keep: Snapshots used in published experiments (indefinitely)
- Archive: Older snapshots to S3 (after 6 months)
- Cleanup: Development snapshots (rolling 2-week window)

## Datasets: Versioned Training & Eval Data

**Purpose**: Apply filters, enrich features, and join with relevance judgments.

### Dataset Construction Steps

```mermaid
sequenceDiagram
    participant Snapshot
    participant Filter
    participant Enrich
    participant Merge
    participant Split
    participant Validate

    Snapshot->>Filter: Load snapshot (~800k episodes)
    Filter->>Filter: Apply filters (language, dates, transcripts)
    Filter->>Enrich: eligible episodes pass filter
    Enrich->>Enrich: Add features (embeddings, BM25 scores, popularity)
    Enrich->>Merge: Enrich episodes
    Merge->>Merge: Generate labeled pairs (weak + synthetic)
    Merge->>Split: ~55k labeled examples
    Split->>Split: Train/Val/Test split (stratified)
    Split->>Validate: Validate schema
    Validate->>Validate: ✓ Parquet export
```

### Configuration & Build

Specify dataset via config file (checked in):
```bash
experiments/datasets/search-bootstrap-v1.json
```

Build:
```bash
make dataset CONFIG=experiments/datasets/search-bootstrap-v1.json
```

**Output**: train.parquet, val.parquet, test.parquet (+ metadata.json)

### Features Added

| Feature Type | Examples |
|--------------|----------|
| **Query** | Intent, embedding, difficulty |
| **Episode** | Title, description, transcript, embedding |
| **Signals** | BM25 score, vector similarity, popularity |
| **Labels** | Relevance grade (human-judged) |

### Validation

Schema validation via Pandera:
- Column presence & types
- Value constraints (duration > 0, language in {en, es, fr})
- Null handling
- Fails loudly on violations

## Experiment-Config Lineage

Every experiment references exact versions:

```mermaid
graph TB
    A["Experiment Config<br/>hybrid-rrf-v1.json"] -->|specifies| B["Dataset<br/>search-bootstrap-v1"]
    B -->|built from| C["Snapshot<br/>2026-01-15"]
    C -->|captured from| D["PostgreSQL Catalog<br/>git commit abc123"]
    A -->|logs to| E["MLflow<br/>Run ID: xyz"]
    E -->|records| F["Metrics<br/>Recall@10: 0.70"]

    style A fill:#fff9c4
    style B fill:#c8e6c9
    style C fill:#b3e5fc
    style D fill:#f8bbd0
    style E fill:#ffe0b2
    style F fill:#f0f4c3
```

**Reproducibility**: Can re-run any historical experiment by specifying snapshot + config + git commit.

## Dataset Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Creation
    Creation --> Validation: Build from snapshot
    Validation --> Published: Schema OK
    Published --> InUse: Referenced by experiments
    InUse --> Archived: Experiment complete
    Archived --> [*]: Keep forever (reproducibility)
```

**Strategy**:
- Meaningful versioning: `search-bootstrap-v1` (not by date)
- Never overwrite versions
- Archive: Keep indefinitely (disk is cheap)
- Track row counts: Before/after at each filter step

## Best Practices

✓ **Version by content** (search-bootstrap-v1), not date  
✓ **Document filters** in config (explain exclusions)  
✓ **Validate schema** with Pandera (fail loud on mismatches)  
✓ **Commit configs** to git (part of reproducible artifact)  
✓ **Track lineage** (snapshot → dataset → experiment → metrics)  

✗ **Don't overwrite** dataset versions  
✗ **Don't delete** published datasets (breaks reproducibility)  

## Future Enhancements

- Incremental snapshots (faster delta updates)
- MLflow dataset versioning (automatic lineage UI)
- Feature caching (avoid recomputing embeddings)
