# Observability

What is tracked today:

- **MLflow** (`http://localhost:5001`). Experiment `retrieval-eval` holds
  one run per evaluation with the config hash, qrels hash, catalog size,
  git revision, every global, per-type, per-language, and cohort metric,
  and the per-query results as an artifact. Experiment
  `two-tower-training` holds one run per training job with the dataset
  version, snapshot id, config and contract fingerprints, model and
  training parameters, per-step loss and candidate-set diagnostics,
  validation and judged-pool metrics, and the artifact metadata.
- **Job queue state** in PostgreSQL: attempts, leases, backoff, and
  dead-lettered jobs with their last error are queryable directly.
- **Artifacts and indexes** carry their own records: `artifact.json`
  (what was trained on what, with which code) and `index.json` (when the
  index was built, against which contract, how many vectors).

- **Online metrics, dashboards, alerts, and the drift check** are in
  [16-monitoring.md](16-monitoring.md): Prometheus metrics from the API,
  model server and speech server, a provisioned Grafana dashboard with
  alert rules, and a scheduled comparison of served behaviour (leg
  outcomes, latency, score and source distributions, feature means)
  against a recorded baseline. The feature contract fingerprint and
  `judged_share_at_10` remain the offline checks for training-serving
  skew and evaluation-pool drift.

Not built: OpenTelemetry traces. Request timing is in the per-request
JSON logs and the stage histograms instead.
