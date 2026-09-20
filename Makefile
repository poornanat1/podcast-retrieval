GO ?= go
UV ?= uv
DATABASE_URL ?= postgres://podfind:podfind@localhost:5432/podfind?sslmode=disable

.PHONY: all build test lint ci python-env migrate snapshot dataset relevance-queries relevance-pool relevance-review relevance-export embed eval eval-all train index reproduce

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

TRAIN_CONFIG ?= experiments/training/two-tower-v2.json

train:
	$(UV) run python -m ml.training.train --config $(TRAIN_CONFIG)

ARTIFACT ?= data/models/two-tower-bootstrap-v2-1.2.0-e2e0bf0793e6-67868919

index:
	$(UV) run python -m ml.embeddings.two_tower_index --artifact $(ARTIFACT)

reproduce:
	@echo "make reproduce: not implemented yet" && exit 1
