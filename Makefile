GO ?= go
UV ?= uv
DATABASE_URL ?= postgres://podfind:podfind@localhost:5432/podfind?sslmode=disable

.PHONY: all build test lint ci python-env migrate images publish-bundle drift drift-baseline loadtest monitoring snapshot dataset relevance-queries relevance-pool relevance-review relevance-export embed embed-transcripts train-ranker llm-queries name-queries mine-negatives pool-export audit-negatives test-split gates significance cohort-report eval eval-all train index bundle golden parity promote serve-models prewarm api online-eval serve-speech speech-testset speech-eval reproduce

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

embed-transcripts:
	$(UV) run python -m ml.embeddings.transcripts

RANKER_CONFIG ?= experiments/ranking/ranker-v1.json

train-ranker:
	$(UV) run python -m ml.ranking.train --config $(RANKER_CONFIG)

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

cohort-report:
	$(UV) run python -m ml.evaluation.cohort_report --summaries $(COMPARE_JSON)

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

# The headline experiment, checked against the recorded numbers
# (experiments/eval/reproduce-expected.json); RECORD=1 re-records them.
reproduce:
	$(UV) run python -m ml.evaluation.reproduce $(if $(RECORD),--record,)

# --- serving -------------------------------------------------------------

TWO_TOWER ?= data/models/two-tower-llm-names-1.0.0-56e2c0708f7e-3b1c13c6
RANKER ?= data/models/ranker-lambdamart-7bb99f61
# The promoted bundle by default; pass BUNDLE=data/serving/bundles/<id> before the first promotion.
BUNDLE ?= data/serving/bundles/$(shell cat data/serving/ACTIVE 2>/dev/null)
API ?= http://localhost:8080

bundle:
	$(UV) run python -m ml.serving.bundle --two-tower $(TWO_TOWER) --ranker $(RANKER)

golden:
	$(UV) run python -m ml.serving.golden --bundle $(BUNDLE)

# Parity needs the API running in debug mode with leg limits above the
# cold-cache worst case:
#   PODFIND_DEBUG=1 PODFIND_LEXICAL_TIMEOUT_MS=60000 PODFIND_VECTOR_TIMEOUT_MS=60000 make api
parity:
	$(UV) run python -m ml.serving.parity --bundle $(BUNDLE) --api $(API)

promote:
	$(UV) run python -m ml.serving.promote --bundle $(BUNDLE)

serve-models:
	$(UV) run python -m ml.serving.server --bundle $(BUNDLE)

serve-speech:
	$(UV) run python -m ml.speech.server

SPEECH_CONFIG ?= experiments/speech/testset-v2.json
SPEECH_SET ?= data/speech/testset-v2

speech-testset:
	$(UV) run python -m ml.speech.testset --config $(SPEECH_CONFIG)

speech-eval:
	$(UV) run python -m ml.speech.evaluate --testset $(SPEECH_SET) --api $(API) --json-out experiments/speech/results.json

prewarm:
	$(UV) run python -m ml.serving.prewarm --bundle $(BUNDLE)

api:
	DATABASE_URL=$(DATABASE_URL) $(GO) run ./cmd/api

online-eval:
	$(UV) run python -m ml.serving.online_eval --api $(API)

# --- deployment and operations ------------------------------------------------

# The three service images, as Railway builds them (docs/11-deployment.md).
images:
	docker build -f Dockerfile --build-arg PODFIND_CMD=api -t podfind-api .
	docker build -f ml/serving/Dockerfile -t podfind-model-server .
	docker build -f ml/speech/Dockerfile -t podfind-speech-server .

# Upload the promoted bundle to the object store (bucket OBJECT_STORE_BUCKET_MODELS).
publish-bundle:
	$(UV) run python -m ml.serving.bundlestore publish --bundle $(BUNDLE)

DRIFT_HOURS ?= 6
PROMETHEUS ?= http://localhost:9090

drift:
	$(UV) run python -m ml.monitoring.drift --database-url "$(DATABASE_URL)" --hours $(DRIFT_HOURS) --prometheus $(PROMETHEUS)

drift-baseline:
	$(UV) run python -m ml.monitoring.drift --database-url "$(DATABASE_URL)" --hours $(DRIFT_HOURS) --prometheus $(PROMETHEUS) --write-baseline

LOAD_CONCURRENCY ?= 8
LOAD_DURATION ?= 2m
LOAD_OUT ?= experiments/load/latest.json

loadtest:
	DATABASE_URL="$(DATABASE_URL)" $(GO) run ./cmd/loadtest -api $(API) -concurrency $(LOAD_CONCURRENCY) -duration $(LOAD_DURATION) -out $(LOAD_OUT)

# Prometheus and Grafana against the local servers (http://localhost:3000, admin/podfind).
monitoring:
	docker compose -f deploy/monitoring/docker-compose.yml up -d --build
