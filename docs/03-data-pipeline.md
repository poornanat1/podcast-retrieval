# Data pipeline

```mermaid
flowchart LR
    PG[("PostgreSQL")] -->|"make snapshot"| SNAP["Snapshot<br/>data/snapshots/&lt;id&gt;/"]
    SNAP -->|"make dataset"| DS["Dataset<br/>data/datasets/&lt;name&gt;-&lt;version&gt;/"]
    SNAP -->|"make llm-queries"| LQ["llm-queries-*.jsonl"]
    SNAP -->|"make name-queries"| NQ["name-queries.jsonl"]
    LQ -->|"make mine-negatives"| NEG["*-negatives.jsonl"]
    NQ -->|"make mine-negatives"| NEG
    NQ --> DS
    LQ --> DS
    NEG --> DS
    DS -->|"--publish"| S3[("Object store")]
    DS --> TRAIN["make train"]
```

## Snapshots

`make snapshot` (`ml/datasets/snapshot.py`) exports `episodes.parquet`
(catalog columns plus a 4,000-character transcript excerpt),
`podcasts.parquet`, and a `manifest.json` with file hashes into
`data/snapshots/<id>/`, where the id is derived from the file hashes.
`data/snapshots/latest` points at the newest one. Capturing a snapshot is
the only non-deterministic step; everything downstream is a pure function
of a snapshot, a config, and the code.

`python -m ml.datasets.snapshot --with-transcripts-for data/snapshots/<id>`
derives a snapshot from an existing one: identical episode and podcast
rows, with `has_transcript` and `transcript_excerpt` filled from the
database's current transcripts, and `derived_from` in its manifest.
Training reads it in place of the parent for datasets built from the
parent, so an input ablation changes nothing else. It does not move
`latest`.

## Datasets

`make dataset CONFIG=experiments/datasets/<config>.json` (`ml/datasets/build.py`)
builds one dataset from a snapshot: generators produce rows in the shared
training-example contract (`ml/datasets/contracts.py`,
[data/contracts/training-example.md](../data/contracts/training-example.md)),
rows are validated (schema and label-composition checks), split by
`event_time` into train / validation / test, and written as Parquet with
a manifest. The version is `<base_version>-<content hash>`, so rebuilding
the same inputs yields the same version byte for byte, and any change to
the rows changes the version.

Generators:

| type | query | positive | label class |
|---|---|---|---|
| `synthetic_query` | normalized episode title | that episode | synthetic |
| `topic_similarity` | a category name | an episode of a show in that category | weak |
| `llm_query` | an LLM-written search for the episode | that episode | synthetic |

Negatives: random catalog episodes by default; `llm_query` mixes mined
hard negatives, popularity-weighted episodes, and random ones. Its options
include `hard_sources` (keep only candidates a named system mined, e.g.
`["lexical-fts"]`), `id_prefix` (so two query files can share one
dataset), and a per-query `exclude_podcasts` list in the queries file (no
episode of those shows becomes a negative). The `llm_query` generator
also reads the name-queries file, which has the same shape.

Checked-in configs: `search-bootstrap-v1.json` (titles + topics, English),
`search-bootstrap-v2.json` (same, all languages),
`search-llm-v1.json` (LLM queries with mined negatives),
`search-llm-{random,hard}-negatives-v1.json` (negative-strategy arms),
`search-llm-names-v1.json` (LLM queries plus name queries, lexical-mined
negatives only).

## Raw inputs beyond the snapshot

Optional inputs live next to the snapshot and are hashed into the
dataset manifest (`inputs`), so a dataset stays reproducible even though
producing them is not deterministic:

- `llm-queries-<model>.jsonl` — one natural-language query per sampled
  episode, written by an LLM (`make llm-queries`,
  `ml/datasets/llm_queries.py`), resumable and shardable.
- `llm-queries-<model>-negatives.jsonl` — hard negatives mined for those
  queries with the baseline retrievers (`make mine-negatives`,
  `ml/hard_negatives/mine.py`): top-50 candidates from vector and lexical
  search minus the positive and its show. Repeated query texts are
  searched once, and `--shard k/n` partitions by query text.
- `name-queries.jsonl` — show-name and publisher-name queries paired with
  episodes of that show (`make name-queries`,
  `ml/datasets/name_queries.py`), deterministic. Shows whose title or
  publisher is a relevance-set query are left out, so the evaluation's
  navigational and entity queries are never trained on.

## Validation

`ml/datasets/validate.py` runs the schema (types, uniqueness, id lists,
positive not among negatives or in history, no future timestamps) and a
distribution report (label class shares, distinct positives, negatives per
example) with per-dataset thresholds from the config. A build that fails
validation writes nothing.

## Publishing

`--publish` uploads a built dataset to the object store's `datasets` bucket under
its version; an existing version is never overwritten.

## Relevance data

The relevance set is separate from training data and tracked in git:
[06-relevance.md](06-relevance.md).
