# Deployment

Nothing is deployed. What runs today runs locally through Docker Compose:

| service | image / build | role |
|---|---|---|
| `postgres` | `pgvector/pgvector:pg17` | catalog, indexes, jobs, events |
| `minio` + `minio-init` | `minio/minio` | published datasets, raw transcripts |
| `mlflow` | `ghcr.io/mlflow/mlflow` | tracking server on host port 5001 |
| `catalog-worker` | built from the repo `Dockerfile` | ingestion (stopped during the corpus freeze) |

The Python jobs (embedding, training, evaluation) run on the host with
`uv`, against the compose services.

`deploy/helm/` and `deploy/openshift/` are empty placeholders; `cmd/api`
and `cmd/ml-worker` print their version and exit. Kubernetes manifests,
probes, autoscaling, canary rollout, and the model server are planned
work and will be documented when they exist.

Two practical notes from running the stack on a laptop:

- Training and the Docker VM compete for memory. The training pipeline
  used for the reports stops Docker during training, logs to a local
  SQLite MLflow store, and syncs the run to the server afterwards
  (`ml/training/sync_run.py`).
- Restarting the whole stack starts the catalog worker; while the corpus
  is frozen, start only the services you need.
