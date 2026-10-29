# Deployment

PodFind runs on [Railway](https://railway.com): one service per process,
each built from this repository by its Dockerfile, with Postgres on a
volume and two object-storage buckets. Everything is declared in
[`.railway/railway.ts`](../.railway/railway.ts) and applied with the
Railway CLI, so a deployment is a `git push` to `main` and a change of
infrastructure is a pull request. Kubernetes and Helm were the original
target; a single-engineer project does not need a cluster to operate, and
Railway gives private networking, volumes, health checks and rollbacks
without one.

## What runs

| service | image | role | state |
|---|---|---|---|
| `api` | `Dockerfile` (`PODFIND_CMD=api`) | the public API and web app on `:8080` | stateless |
| `models` | `ml/serving/Dockerfile` | ONNX model server on `:8081`; fetches the bundle named by `PODFIND_BUNDLE_ID` from the `models` bucket into its volume at start | volume `/data` (bundle cache) |
| `speech` | `ml/speech/Dockerfile` | Whisper on CPU on `:8082`; downloads the weights into its volume at first start | volume `/data` (model cache) |
| `postgres` | `pgvector/pgvector:pg17` | catalog, vectors, searches, events | volume, 60 GB |
| `prometheus`, `grafana` | `deploy/monitoring/*/Dockerfile` | metrics, dashboards, alerts ([16-monitoring.md](16-monitoring.md)) | volumes |
| `drift` | `ml/serving/Dockerfile`, cron | the drift check every 30 minutes | none |
| bucket `models` | | serving bundles, published with `make publish-bundle` | |
| bucket `recordings` | | voice recordings users asked to keep | |

Services talk over Railway's private network (`<service>.railway.internal`,
IPv6, so every server listens on `::`). Only `api` has a public domain;
Grafana gets one when you want to look at it. The catalog worker is not
deployed while the corpus is frozen: starting it would resume ingestion.

Sizing: the served hot set (two HNSW indexes and their rows) is about
6 GB, so Postgres runs with `shared_buffers=6GB` and needs a plan whose
per-service memory ceiling is above that with room for the API's
queries. The Pro plan's 24 GB per replica is enough; the Hobby plan's
8 GB is not. If memory is short, the fallbacks are half-precision vectors
for the served models, or a managed Postgres (Neon, Supabase) with the
stateless services on Railway.

## First deployment

1. **Create the project.** `railway login`, then from the repository root
   `railway link` (choose or create the `podfind` project and the
   `production` environment), `cd .railway && npm install && cd ..`, then
   `railway config plan` and `railway config apply`. The plan lists every
   service, volume and bucket the file creates.
2. **Set the secrets** once; every later apply preserves them:

   ```sh
   railway variable set -s postgres POSTGRES_PASSWORD=…
   railway variable set -s grafana GF_SECURITY_ADMIN_PASSWORD=…
   railway variable set -s speech OPENAI_API_KEY=…    # only for the OpenAI STT backend
   ```

   Alerts are visible in
   Grafana's alert list; to have them posted somewhere, set
   `ALERT_WEBHOOK_URL` (a Slack incoming webhook or anything that accepts
   `{"text": ...}`) on the `grafana` and `drift` services.
3. **Publish the bundle.** With the `models` bucket's credentials from its
   Credentials tab:

   ```sh
   OBJECT_STORE_ENDPOINT=https://t3.storageapi.dev OBJECT_STORE_REGION=auto \
   OBJECT_STORE_ACCESS_KEY=… OBJECT_STORE_SECRET_KEY=… OBJECT_STORE_BUCKET_MODELS=<bucket name> \
   make publish-bundle
   ```

   The model server verifies every file against the manifest before it
   serves; a new bundle is a change of `BUNDLE_ID` in `railway.ts`.
4. **Load the catalog.** Add a TCP proxy to `postgres` (Settings >
   Networking) for the duration of the copy, then from a machine with the
   source database:

   ```sh
   deploy/railway/copy-catalog.sh "postgres://podfind:<password>@<proxy host>:<port>/podfind"
   DATABASE_URL="postgres://…" make prewarm
   ```

   The script applies the migrations, copies podcasts, episodes,
   transcripts and the served models' vectors only (about 20 GB of the
   39 GB local database), builds the two HNSW indexes and analyzes.
   Remove the TCP proxy afterwards; the API reaches Postgres privately.
5. **Open the API.** Generate a domain for `api` (Settings > Networking)
   or attach your own. `/healthz` reports the database, the model server
   and the bundle; the web app is at `/`.

## Day to day

- **Deploy:** push to `main`. Railway builds each service whose files
  changed and swaps it in once `/healthz` passes; the previous deployment
  stays available.
- **Roll back:** in the service's Deployments list, redeploy the previous
  deployment. Nothing is rebuilt. Configuration changes are rolled back
  by reverting the commit to `railway.ts` and applying.
- **Change the bundle:** `make bundle && make golden && make parity &&
  make promote` locally as before, `make publish-bundle`, then set
  `BUNDLE_ID` in `railway.ts` and apply. The model server restarts on the
  new bundle; the API's health check fails if the served bundle differs
  from the one it started with, so the API is redeployed after.
- **Logs:** every service logs one JSON object per line
  (`PODFIND_LOG_FORMAT=json`), searchable in Railway's log view.
- **Secrets** never enter the repository: `.env` is gitignored and shared
  variables live in Railway.

## Local stack

Development still runs the infrastructure in Docker Compose and the
servers on the host:

| service | image | role |
|---|---|---|
| `postgres` | `pgvector/pgvector:pg17` | catalog, indexes, jobs, events |
| `objstore` + `objstore-init` | `rustfs/rustfs` | S3-compatible store: published datasets, raw transcripts, kept recordings, bundles |
| `mlflow` | `ghcr.io/mlflow/mlflow` | tracking server on host port 5001 |
| `catalog-worker` | built from the repo `Dockerfile` | ingestion (stopped during the corpus freeze) |

`make images` builds the three service images as Railway does, and the
API image runs against the host's services with
`-e DATABASE_URL=postgres://podfind:podfind@host.docker.internal:5432/podfind?sslmode=disable
-e MODEL_SERVER_URL=http://host.docker.internal:8081`. `make monitoring`
starts Prometheus and Grafana against the host servers. Voice recordings
a user asks to keep go to the object store when the API has the
`OBJECT_STORE_*` variables; without them retention is refused.

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
