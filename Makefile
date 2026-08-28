GO ?= go
UV ?= uv
DATABASE_URL ?= postgres://podfind:podfind@localhost:5432/podfind?sslmode=disable

.PHONY: all build test lint ci python-env migrate snapshot dataset reproduce

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

reproduce:
	@echo "make reproduce: not implemented yet" && exit 1
