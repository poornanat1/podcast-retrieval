# Deployment

Nothing is deployed. What runs today runs locally, the infrastructure
through Docker Compose:

| service | image / build | role |
|---|---|---|
| `postgres` | `pgvector/pgvector:pg17` | catalog, indexes, jobs, events |
| `objstore` + `objstore-init` | `rustfs/rustfs` | S3-compatible store: published datasets, raw transcripts, kept recordings |
| `mlflow` | `ghcr.io/mlflow/mlflow` | tracking server on host port 5001 |
| `catalog-worker` | built from the repo `Dockerfile` | ingestion (stopped during the corpus freeze) |

The Python jobs (embedding, training, evaluation) run on the host with
`uv`, against the compose services. The serving processes also run on the
host: the model server (`make serve-models`, port 8081), the speech
server (`make serve-speech`, port 8082), and the Go API (`make api`,
port 8080). Their configuration, start order, and memory needs are in
[14-serving.md](14-serving.md#operations) and [15-voice.md](15-voice.md).
Voice recordings a user asks to keep go to the object store when the API has the
`OBJECT_STORE_*` variables; without them retention is refused.

`deploy/helm/` and `deploy/openshift/` are empty placeholders, and
`cmd/ml-worker` prints its version and exits. Kubernetes manifests, probes,
autoscaling, and canary rollout are planned work and will be documented
when they exist.

Two practical notes from running the stack on a laptop:

- Training and the Docker VM compete for memory. The training pipeline
  used for the reports stops Docker during training, logs to a local
  SQLite MLflow store, and syncs the run to the server afterwards
  (`ml/training/sync_run.py`).
- Restarting the whole stack starts the catalog worker; while the corpus
  is frozen, start only the services you need.
- Postgres runs with `shared_buffers=2GB` (`docker-compose.yml`). Vector
  search is memory-bound: serving needs about 6 GB of indexes and vector
  rows hot, more than an 8 GB Docker VM leaves for the cache.
