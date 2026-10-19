# Development

## Setup

Go 1.26+, Python 3.11+ with [uv](https://docs.astral.sh/uv/), Docker.

```sh
cp .env.example .env                      # PARTICLE_API_KEY for discovery; OPENAI_API_KEY for judging and query generation
docker compose up -d postgres minio mlflow
make migrate
make ci                                   # lint + tests + build
```

## Make targets

| target | what it does |
|---|---|
| `make ci` / `lint` / `test` / `build` | ruff + go vet; pytest + go test; go build + Python compile |
| `make migrate` | apply `migrations/` |
| `make snapshot` | export the catalog to `data/snapshots/<id>/` |
| `make dataset CONFIG=…` | build a versioned dataset from a snapshot |
| `make llm-queries` | LLM-written queries for sampled episodes (raw dataset input) |
| `make name-queries` | show-name and publisher-name queries, relevance-set names held out |
| `make mine-negatives` | hard negatives for a queries file from the baseline retrievers (`QUERIES_FILE`, `MINE_SYSTEMS`) |
| `make embed` | E5 vectors for every episode into `pgvector` |
| `make train TRAIN_CONFIG=…` | train the two-tower model |
| `make index ARTIFACT=…` | build a two-tower artifact's vector index |
| `make relevance-queries` / `relevance-pool` / `relevance-judge` / `relevance-review` / `relevance-export` | relevance-set workflow |
| `make pool-export` | judged episode rows for pool validation during training |
| `make audit-negatives` | grade a sample of mined negatives for false negatives |
| `make test-split DATASET_DIR=… ARTIFACTS=…` | rank held-out positives against one shared pool |
| `make gates COMPARE_JSON=… ARTIFACT=…` | acceptance gates against the baseline (add `--register` to record the version) |
| `make significance CANDIDATE_RUN=… BASELINE_RUN=…` | paired bootstrap between two evaluation runs |
| `make cohort-report COMPARE_JSON=…` | per-cohort tables with regressions against the baselines flagged |
| `make eval EVAL_CONFIG=…` / `eval-all` | evaluate one system / every config in `experiments/eval/` |

Variables (`CONFIG`, `SNAPSHOT`, `POOL_SYSTEMS`, `TRAIN_CONFIG`, `ARTIFACT`,
`EVAL_CONFIG`) have defaults in the Makefile.

## Layout

```
ml/
├── datasets/       snapshot.py, build.py, contracts.py, validate.py, publish.py, llm_queries.py
├── features/       contract.py, featurize.py, text.py
├── models/two_tower/model.py
├── training/       train.py, data.py, loss.py, sync_run.py
├── embeddings/     embed.py (E5), two_tower_index.py, text.py
├── hard_negatives/ mine.py, audit.py
├── registry/       register.py
├── retrieval/      lexical.py, vector.py, two_tower.py, hybrid.py, popularity.py
├── relevance/      queries.py, pool.py, llm_judge.py, review.py, export.py, contracts.py
├── evaluation/     harness.py, metrics.py, run.py, compare.py, pool_rerank.py, test_split.py
└── tests/
cmd/                catalog-worker, migrate (api, ml-worker: scaffolds)
internal/           catalog, rss, transcript, discovery, particle, jobs, outbox, objstore, migrate, pgtest, version
migrations/         SQL, embedded into the migrate binary
experiments/        datasets/, training/, eval/ configs; reports
data/               relevance/ tracked; snapshots/, datasets/, models/ local
```

## Tests

```sh
uv run pytest -q                 # Python: offline, no database; tiny stand-in encoder and tokenizer
go test ./...                    # Go: unit tests; database-backed tests use internal/pgtest
```

Python tests never hit the network or the database; model tests use a
tiny randomly initialized BERT and a hand-written vocabulary
(`ml/tests/helpers.py`). CI (`.github/workflows/ci.yml`) runs `make ci`
against a `pgvector/pgvector:pg17` service container.

## Conventions

- Python: `ruff` with rules E, F, I, UP, B, line length 100; type hints;
  short docstrings that say why.
- Go: `gofmt`, `go vet`; comments explain decisions, not syntax.
- Commits follow the conventional prefixes (`feat:`, `fix:`, `docs:`,
  `data:`, …); data commits change `data/relevance/`.
- Reproducibility over convenience: new numbers come with a checked-in
  config, and datasets, artifacts, and judgments are versioned rather than
  overwritten.
- Anything not built is labeled as such in the docs rather than described
  aspirationally.
