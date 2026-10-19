# PodFind

Podcast search over ~800k episodes: hybrid lexical + vector retrieval, a
fine-tuned two-tower model, and an evaluation setup honest enough to say
when the model loses.

Type or (eventually) say what you want — *"English episodes under 45
minutes about practical uses of AI, published this year, no explicit
content"* — and get episodes back, ranked, with the constraints enforced
as hard filters.

## What exists

| Part | State |
|---|---|
| Catalog | Go worker: podcast discovery (Particle), RSS polling on an adaptive schedule, deduplication, publisher transcript parsing. 1,788 shows, ~796k episodes, ~87k transcripts. |
| Lexical search | PostgreSQL full-text search, weighted across episode, transcript, and show fields, per-language configs, hard filters for language, duration, date, explicit. |
| Vector search | `multilingual-e5-small` embeddings in `pgvector` with HNSW, plus a versioned index per fine-tuned model. |
| Hybrid | Reciprocal-rank fusion of the lexical and vector rankings. |
| Relevance set | 180 queries (navigational, entity, exploratory, filtered), 50,591 graded judgments pooled from every system under comparison, LLM-graded with human override. |
| Evaluation | Recall@K, hit rate, MRR, NDCG@10, coverage, cohort recall, latency, judged share; one config per system, runs logged to MLflow. |
| Datasets | Deterministic builder: snapshot + config + code revision → identical versioned Parquet splits. Generators for title queries, weak topic labels, and LLM-written queries with mined hard negatives. |
| Two-tower model | PyTorch, shared E5 encoder, structured episode features, shared training/serving feature contract, resumable training with judged-pool validation, distillation, hard-negative mining and auditing, acceptance gates with MLflow registration. |
| Serving API, web app, voice, deployment, monitoring | Not built. Entry points and directories exist as scaffolds. |

## Results so far

On the 180-query relevance set, latest judgment pool:

| system | NDCG@10 | Recall@100 | entity NDCG@10 |
|---|---|---|---|
| lexical full-text search | 0.535 | 0.197 | 0.796 |
| pretrained E5 vector search | 0.694 | 0.367 | 0.584 |
| fine-tuned two-tower | 0.730 | 0.519 | 0.750 |

The fine-tuned model is trained on LLM-written queries plus show-name and
publisher-name queries, with lexical-mined and random negatives. It
passes the acceptance gates against pretrained E5 and is the registered
champion. Its Recall@100 gain is well outside noise; its NDCG@10 gain over
E5 is probable but its 95% interval still includes zero. Lexical search
keeps the edge on relevant episodes found through their transcripts,
which the model does not read. Mined hard negatives from vector search
turned out to be one-fifth false negatives and hurt. Ablations show the
model's quality comes entirely from content features, and its gains over
E5 hold for shows it never trained on. Its known failure modes are listed
in the [model card](docs/model-card.md). Write-ups:
[baseline report](experiments/baseline-report.md),
[two-tower runs](experiments/two-tower-first-run.md),
[negative sampling](experiments/negative-sampling-report.md),
[name queries](experiments/name-queries-report.md),
[ablations and cold start](experiments/ablation-cold-start-report.md).

## Layout

```
cmd/            Go entry points: catalog-worker, migrate (api and ml-worker are scaffolds)
internal/       Go packages: catalog, rss, transcript, discovery, jobs, outbox, objstore
migrations/     PostgreSQL schema
ml/             Python: datasets, features, models, training, embeddings, retrieval, relevance, evaluation
experiments/    Checked-in dataset, training, and eval configs, and reports
data/           Relevance set (tracked); snapshots, datasets, models (local)
docs/           Documentation, data card, licensing notes
```

## Running it

Requires Go 1.26+, [uv](https://docs.astral.sh/uv/), and Docker.

```sh
cp .env.example .env            # optional: Particle API key for discovery
docker compose up -d postgres minio mlflow
make migrate
make ci                         # lint, tests, build

docker compose up -d catalog-worker            # ingestion (see docs/09-catalog.md)
make embed                                     # E5 vectors for the catalog
make eval-all                                  # every system against the relevance set
make snapshot && make dataset                  # versioned training data
make train && make index                       # two-tower model and its index
```

Every target is in the [Makefile](Makefile); the docs start at
[docs/00-overview.md](docs/00-overview.md).

## Data responsibilities

Podcast metadata, artwork, and transcripts delivered over public RSS remain
the property of their publishers. This project stores only required
metadata, never downloads audio, preserves attribution, and honors removal
requests. See [docs/licensing.md](docs/licensing.md).
