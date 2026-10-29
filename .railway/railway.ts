// PodFind on Railway: one service per process, all built from this
// repository's Dockerfiles, plus Postgres with pgvector on a volume and two
// buckets. Apply with the Railway CLI from the repository root:
//
//   railway link                 # once: pick the project and environment
//   railway config plan          # preview
//   railway config apply         # create or update
//
// Secrets are set once with the CLI and preserved by every apply:
//   railway variable set -s postgres POSTGRES_PASSWORD=…
//   railway variable set -s grafana GF_SECURITY_ADMIN_PASSWORD=…
//   railway variable set -s speech OPENAI_API_KEY=…   (optional)
// Alerts show in Grafana's own alert list;
// to also post them somewhere, add ALERT_WEBHOOK_URL (Slack-compatible)
// to the grafana and drift services.
// Each service names its Dockerfile; the build context is the repository
// root. Loading the catalog needs a TCP proxy on postgres for the duration
// of the copy; add it in the dashboard and remove it after
// (docs/11-deployment.md).
import {
  bucket,
  defineRailway,
  github,
  group,
  image,
  preserve,
  project,
  ref,
  service,
  volume,
} from "railway/iac";

const REPO = "poornanat1/podcast-retrieval";
const BUNDLE_ID = "42df3779d7cf"; // the promoted serving bundle (data/serving/ACTIVE)
const REGION = "us-east4-eqdc4a"; // where the volumes live; Railway assigned it on first apply

export default defineRailway((ctx) => {
  const source = github(REPO, { branch: "main" });
  const dockerfile = (path: string) => ({ builder: "DOCKERFILE" as const, dockerfilePath: path });
  // A bucket's credentials, in the OBJECT_STORE_* variables the services read.
  const store = (b: ReturnType<typeof bucket>, bucketVar: string) => ({
    OBJECT_STORE_ENDPOINT: ref(b, "ENDPOINT"),
    OBJECT_STORE_REGION: ref(b, "REGION"),
    OBJECT_STORE_ACCESS_KEY: ref(b, "ACCESS_KEY_ID"),
    OBJECT_STORE_SECRET_KEY: ref(b, "SECRET_ACCESS_KEY"),
    [bucketVar]: ref(b, "BUCKET"),
  });

  // --- storage ------------------------------------------------------------

  // Buckets keep the names Railway generated when they were created with
  // `railway bucket create` (the file cannot rename them).
  const models = bucket("neat-vase-vkFf", { region: "iad" }); // serving bundles
  const recordings = bucket("stackable-toolchest-Aq6L", { region: "iad" }); // consented voice recordings

  const pgData = volume("postgres-data", { sizeMB: 61440, region: REGION }); // served catalog ≈ 20 GB + headroom

  // pgvector on our own image: the served hot set (two HNSW indexes and
  // their rows) is about 6 GB, so the buffer cache must hold it.
  const postgres = service("postgres", {
    source: image("pgvector/pgvector:pg17"),
    // Through the image's entrypoint, which drops to the postgres user.
    start:
      "docker-entrypoint.sh postgres -c shared_buffers=6GB -c effective_cache_size=16GB -c work_mem=16MB " +
      "-c maintenance_work_mem=2GB -c max_connections=100 -c max_parallel_workers_per_gather=2 " +
      // Containers get a 64 MB /dev/shm; parallel queries and index builds
      // need dynamic shared memory, so back it with mmap'd files instead.
      "-c dynamic_shared_memory_type=mmap",
    env: {
      POSTGRES_USER: "podfind",
      POSTGRES_DB: "podfind",
      POSTGRES_PASSWORD: preserve(),
      PGDATA: "/var/lib/postgresql/data/pgdata",
    },
    volumeMounts: { "/var/lib/postgresql/data": pgData },
    // Public TCP proxy for loading the catalog from outside; remove (and
    // apply) once the copy is done, the services reach Postgres privately.
    tcp: [5432],
  });

  const databaseUrl =
    "postgres://podfind:${{postgres.POSTGRES_PASSWORD}}@${{postgres.RAILWAY_PRIVATE_DOMAIN}}:5432/podfind?sslmode=disable";

  // --- inference ------------------------------------------------------------

  const bundleCache = volume("bundle-cache", { sizeMB: 5120, region: REGION });
  const modelServer = service("models", {
    source,
    build: dockerfile("ml/serving/Dockerfile"),
    healthcheck: "/healthz",
    healthcheckTimeout: 1200, // first start downloads the bundle (≈ 1 GB)
    env: {
      PORT: "8081",
      PODFIND_BUNDLE_ID: BUNDLE_ID,
      PODFIND_BUNDLE_DIR: "/data/bundles",
      ...store(models, "OBJECT_STORE_BUCKET_MODELS"),
    },
    volumeMounts: { "/data": bundleCache },
  });

  const speechCache = volume("speech-models", { sizeMB: 5120, region: REGION });
  const speechServer = service("speech", {
    source,
    build: dockerfile("ml/speech/Dockerfile"),
    healthcheck: "/healthz",
    healthcheckTimeout: 1200, // first start downloads Whisper small (≈ 1 GB)
    env: {
      PORT: "8082",
      PODFIND_STT: "whisper:openai/whisper-small", // or openai:gpt-4o-mini-transcribe
      PODFIND_STT_DEVICE: "cpu",
      HF_HOME: "/data/hf",
      OMP_NUM_THREADS: "4",
      OPENAI_API_KEY: preserve(),
    },
    volumeMounts: { "/data": speechCache },
  });

  // --- the public API and web app ------------------------------------------

  const api = service("api", {
    source,
    build: dockerfile("Dockerfile"),
    healthcheck: "/healthz",
    healthcheckTimeout: 300, // waits for the model server on its own
    replicas: { [REGION]: ctx.environment === "production" ? 2 : 1 },
    env: {
      PODFIND_CMD: "api",
      PORT: "8080",
      DATABASE_URL: databaseUrl,
      MODEL_SERVER_URL: "http://${{models.RAILWAY_PRIVATE_DOMAIN}}:8081",
      SPEECH_SERVER_URL: "http://${{speech.RAILWAY_PRIVATE_DOMAIN}}:8082",
      PODFIND_LEXICAL_TIMEOUT_MS: "300",
      PODFIND_VECTOR_TIMEOUT_MS: "1500",
      PODFIND_LOG_FORMAT: "json",
      ...store(recordings, "OBJECT_STORE_BUCKET"),
    },
  });

  // --- monitoring -----------------------------------------------------------

  const promData = volume("prometheus-data", { sizeMB: 5120, region: REGION });
  const prometheus = service("prometheus", {
    source,
    build: dockerfile("deploy/monitoring/prometheus/Dockerfile"),
    healthcheck: "/-/ready",
    env: {
      PORT: "9090",
    },
    volumeMounts: { "/prometheus": promData },
  });

  const grafanaData = volume("grafana-data", { sizeMB: 1024, region: REGION });
  const grafana = service("grafana", {
    source,
    build: dockerfile("deploy/monitoring/grafana/Dockerfile"),
    healthcheck: "/api/health",
    env: {
      PORT: "3000",
      GF_SERVER_HTTP_PORT: "3000",
      GF_SECURITY_ADMIN_USER: "admin",
      GF_SECURITY_ADMIN_PASSWORD: preserve(),
      PROMETHEUS_URL: "http://${{prometheus.RAILWAY_PRIVATE_DOMAIN}}:9090",
    },
    volumeMounts: { "/var/lib/grafana": grafanaData },
  });

  // The drift check: the model-server image with a different command, run
  // every 30 minutes. It exits 2 (a failed run in Railway's view) when the
  // served distributions drift from the checked-in baseline.
  const drift = service("drift", {
    source,
    build: dockerfile("ml/serving/Dockerfile"),
    start: "python -m ml.monitoring.drift --hours 6",
    deploy: { cronSchedule: "*/30 * * * *", restartPolicyType: "NEVER" },
    env: {
      DATABASE_URL: databaseUrl,
      PROMETHEUS_URL: "http://${{prometheus.RAILWAY_PRIVATE_DOMAIN}}:9090",
    },
  });

  return project("podfind", {
    resources: [
      group("Data", [postgres, pgData, models, recordings]),
      group("Serving", [api, modelServer, speechServer, bundleCache, speechCache]),
      group("Monitoring", [prometheus, grafana, drift, promData, grafanaData]),
    ],
  });
});
