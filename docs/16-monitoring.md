# Monitoring, drift, and load

What the running system reports about itself, how drift between training
and serving is caught, and how the SLOs are checked. The deployment it
describes is in [11-deployment.md](11-deployment.md).

## Metrics

Every service exposes Prometheus metrics at `/metrics`; Prometheus scrapes
them every 15 s and Grafana shows the `PodFind` dashboard
(`deploy/monitoring/grafana/dashboards/podfind.json`).

**API** (`internal/metrics`):

| metric | what it watches |
|---|---|
| `podfind_http_requests_total`, `podfind_http_request_seconds` | request rate, status and latency by route |
| `podfind_searches_total{input, ordering, bundle}` | searches by input mode, by whether the ranker or the RRF fallback ordered them, and by bundle |
| `podfind_search_stage_seconds{stage}` | encode, retrieve, evidence, rerank and total time |
| `podfind_search_legs_total{leg, status}`, `podfind_search_leg_candidates` | each retrieval leg's outcome (ok, timeout, error, skipped) and candidate count |
| `podfind_search_results`, `podfind_search_zero_results_total` | results per search, searches with none |
| `podfind_search_top_score` | the ranker score of the first result |
| `podfind_search_result_sources_total{source}` | which legs the served results came from |
| `podfind_voice_*` | transcription time, audio length, confidence, low-confidence count, language, retention |
| `podfind_events_total{type}` | impressions and playback events recorded |
| `podfind_index_age_seconds{model}`, `podfind_index_rows{model}` | age of the newest vector and row count per served model, refreshed every 10 minutes |
| `podfind_build_info` | version, bundle and feature version as labels |

**Model server** (`ml/serving/server.py`): `podfind_model_encode_seconds`,
`podfind_model_rerank_seconds`, `podfind_model_query_norm{tower}` (the
towers produce unit vectors; anything else means the served tower or
tokenizer differs from the one that built the index),
`podfind_model_rerank_candidates`, `podfind_model_score`, and
`podfind_model_feature_positive_sum_total{feature}`,
`podfind_model_feature_negative_sum_total{feature}` and
`podfind_model_feature_values_total{feature}`, the signed parts of each ranker
feature's running sum and its count, so the mean of every feature at
serving time can be compared with its training distribution (both parts
are monotonic, which a plain sum of a negative-valued feature is not, and
Prometheus would read as counter resets), with
`podfind_model_feature_nan_total{feature}` for non-finite values.

**Speech server** (`ml/speech/server.py`): `podfind_speech_transcribe_seconds`,
`podfind_speech_audio_seconds`, `podfind_speech_confidence`,
`podfind_speech_language_total`, `podfind_speech_errors_total`.

Logs are one JSON object per line on Railway (`PODFIND_LOG_FORMAT=json`);
each request logs its method, path, status and milliseconds.

## Alerts

Grafana evaluates these rules every minute
(`deploy/monitoring/grafana/provisioning/alerting/podfind.yml`) and posts
to `ALERT_WEBHOOK_URL` (Slack-compatible):

| rule | fires when | for |
|---|---|---|
| Zero-result rate | more than 20% of searches return nothing (30 min) | 10 min |
| Vector leg failures | e5 or two_tower fails or times out on more than 10% of searches (15 min) | 5 min |
| Reranker fallback to RRF | more than 5% of searches are ordered by RRF | 5 min |
| Search p95 latency | p95 total above 1 s (15 min); the budget is 500 ms | 10 min |
| Non-finite feature values | any NaN or infinite feature reached the ranker | 1 min |
| Query vector norm drift | more than 1% of query vectors are not unit-norm | 5 min |
| Vector index age | the newest vector of a served model is older than 45 days | 1 h |
| API server errors | more than 1% of requests are 5xx (10 min); availability target 99.9% | 5 min |
| Service not reachable | Prometheus cannot scrape api, models or speech | 3 min |

The lexical leg times out on about half of the queries at its 300 ms
limit by design ([serving report](../experiments/serving-report.md)), so
it has no alert; the drift check watches its rate against the baseline.

## The drift check

`ml/monitoring/drift.py` is the scheduled, model-aware check. It reads the
stored searches of the last window (the `searches` table keeps every
search's legs, timings, ordering, result scores and sources) and, from
Prometheus, the per-feature means and NaN counts the model server saw,
and compares them with `experiments/monitoring/baseline.json`:

- zero-result rate (absolute increase over the baseline), vector-leg
  failure rate, lexical timeout rate (change over the baseline), RRF
  fallback rate;
- p95 per stage against twice the baseline (above a 1 s floor);
- the distribution of top scores and of result sources, as a population
  stability index (PSI above 0.25 is drift);
- each feature's serving mean against its baseline mean (relative shift
  above 0.5, with at least 200 observations) and any NaN;
- a change of served bundle, reported for information.

It exits 2 and posts the findings to the webhook when anything drifted,
0 otherwise, and needs at least 100 searches in the window to judge: a
handful of real searches against a baseline recorded from the relevance
queries flags the difference between people and a test set, not drift.
The baseline should be re-recorded from production traffic once there is
enough of it. On
Railway it runs every 30 minutes as the `drift` cron service over the last
6 hours; locally, `make drift` (and `make drift-baseline` to record a new
baseline after a model change). The baseline is checked in and re-recorded
when a new bundle is promoted.

## The skew demonstration

The check and the alerts were exercised against a real fault, not a
synthetic signal: the API was restarted with the vector legs' time limit
set to 1 ms, so every e5 and two-tower leg timed out and only the lexical
leg fed the ranker. That is a serving skew the ranker was never trained
on (its `*_rr` and similarity features all go missing). The drift check
flagged it on the first window, the `Vector leg failures` alert fired in
Grafana, and both reached the webhook. The run is written up in
[experiments/deployment-report.md](../experiments/deployment-report.md).

## Load test

`cmd/loadtest` drives `/v1/search/text` with concurrent clients over the
relevance queries and, after each search, sends the impression and
playback events the web app would send. It reports throughput, latency
percentiles, availability (share of 200s), and event persistence (events
stored, from the database when `DATABASE_URL` is given, against events
accepted). `make loadtest API=https://… LOAD_CONCURRENCY=8 LOAD_DURATION=2m`
writes `experiments/load/latest.json`.

SLOs from the design: search p95 under 500 ms excluding transcription,
availability 99.9%, event persistence 99.99%. The measurements are in the
deployment report.
