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

Not built: OpenTelemetry traces, Prometheus metrics, Grafana dashboards
(`dashboards/` is a placeholder), drift detection, and alerting. The
feature contract fingerprint and `judged_share_at_10` exist partly so that
training-serving skew and evaluation-pool drift are detectable offline
now, before any online monitoring exists.
