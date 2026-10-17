GO ?= go
UV ?= uv
DATABASE_URL ?= postgres://podfind:podfind@localhost:5432/podfind?sslmode=disable

.PHONY: all build test lint ci python-env migrate snapshot dataset relevance-queries relevance-pool relevance-review relevance-export embed llm-queries name-queries mine-negatives pool-export audit-negatives test-split gates significance eval eval-all train index reproduce

all: build

migrate:
	DATABASE_URL=$(DATABASE_URL) $(GO) run ./cmd/migrate

python-env:
	$(UV) sync

build:
	$(GO) build ./...
	$(UV) run python -m compileall -q ml pipelines

test:
	$(GO) test ./...
	$(UV) run pytest -q

lint:
	$(GO) vet ./...
	$(UV) run ruff check .

ci: python-env lint test build

CONFIG ?= experiments/datasets/search-bootstrap-v1.json
SNAPSHOT ?= data/snapshots

snapshot:
	$(UV) run python -m ml.datasets.snapshot --out data/snapshots

dataset:
	$(UV) run python -m ml.datasets.build --config $(CONFIG) --snapshot $(SNAPSHOT) --out data/datasets

relevance-queries:
	$(UV) run python -m ml.relevance.queries --snapshot $(SNAPSHOT)

POOL_SYSTEMS ?= lexical-fts,popularity-global,popularity-category,vector-e5-small,hybrid-rrf

embed:
	$(UV) run python -m ml.embeddings.embed

LLM_QUERY_EPISODES ?= 20000

llm-queries:
	$(UV) run python -m ml.datasets.llm_queries --snapshot $(SNAPSHOT) --episodes $(LLM_QUERY_EPISODES)

QUERIES_FILE ?= data/snapshots/$(shell cat data/snapshots/latest)/llm-queries-gpt-5.1.jsonl

name-queries:
	$(UV) run python -m ml.datasets.name_queries --snapshot data/snapshots/$(shell cat data/snapshots/latest)

MINE_SYSTEMS ?= vector-e5-small,lexical-fts

mine-negatives:
	$(UV) run python -m ml.hard_negatives.mine --queries $(QUERIES_FILE) --systems $(MINE_SYSTEMS)

pool-export:
	$(UV) run python -m ml.evaluation.pool_rerank --export

audit-negatives:
	$(UV) run python -m ml.hard_negatives.audit --queries $(QUERIES_FILE)

DATASET_DIR ?= data/datasets/search-retrieval-llm-1.0.0-d52b7c3b104f
ARTIFACTS ?= $(ARTIFACT)

test-split:
	$(UV) run python -m ml.evaluation.test_split --dataset $(DATASET_DIR) --artifacts $(ARTIFACTS) --pretrained

COMPARE_JSON ?= experiments/eval/latest-compare.json

gates:
	$(UV) run python -m ml.registry.register --summaries $(COMPARE_JSON) --candidate two-tower:$(notdir $(ARTIFACT)) --baseline vector-e5-small --artifact $(ARTIFACT)

CANDIDATE_RUN ?=
BASELINE_RUN ?=

significance:
	$(UV) run python -m ml.evaluation.significance --candidate $(CANDIDATE_RUN) --baseline $(BASELINE_RUN) --by-type

relevance-pool:
	$(UV) run python -m ml.relevance.pool --systems $(POOL_SYSTEMS)

relevance-review:
	$(UV) run python -m ml.relevance.review

relevance-judge:
	$(UV) run python -m ml.relevance.llm_judge

relevance-export:
	$(UV) run python -m ml.relevance.export

EVAL_CONFIG ?= experiments/eval/lexical-v1.json

eval:
	$(UV) run python -m ml.evaluation.run --config $(EVAL_CONFIG)

eval-all:
	$(UV) run python -m ml.evaluation.compare experiments/eval/*.json

TRAIN_CONFIG ?= experiments/training/two-tower-names-v1.json

train:
	$(UV) run python -m ml.training.train --config $(TRAIN_CONFIG)

ARTIFACT ?= data/models/two-tower-llm-names-1.0.0-56e2c0708f7e-3b1c13c6

index:
	$(UV) run python -m ml.embeddings.two_tower_index --artifact $(ARTIFACT)

reproduce:
	@echo "make reproduce: not implemented yet" && exit 1
