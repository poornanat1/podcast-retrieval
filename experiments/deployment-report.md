# Deployment, monitoring, and load

The system packaged for Railway, with monitoring that was shown to catch a
real serving skew, and the search SLOs measured on the local stack ahead of
the deployed measurement. Measured 2026-10-28 on a laptop: Postgres, the
object store, Prometheus and Grafana in Docker (15 CPUs, 8 GB VM); the
API, model server and speech server native. The design is in
[docs/11-deployment.md](../docs/11-deployment.md) and
[docs/16-monitoring.md](../docs/16-monitoring.md).

## What was built

- **Images.** One per process from the repository: the API (distroless,
  26 MB, embeds the web app), the model server (ONNX Runtime and the
  feature code only, 742 MB), the speech server (CPU torch and Whisper,
  2 GB). Each was built and run against the local stack: the API served
  the web app and searches, the model server fetched and verified the
  promoted bundle from the object store at start, the speech server
  transcribed a test clip (6.1 s for a 2.3 s clip on CPU; the OpenAI
  backend is the low-latency alternative).
- **Railway project** as infrastructure in code (`.railway/railway.ts`,
  type-checked against the Railway SDK): Postgres with pgvector on a
  60 GB volume, two buckets, the three services, Prometheus, Grafana and
  the drift check as a cron service. `deploy/railway/copy-catalog.sh`
  loads the served catalog (about 20 GB of the 39 GB local database).
- **Object storage** through the S3 API only (AWS SDK for Go, boto3),
  after MinIO's community edition stopped being maintained; RustFS
  locally, Railway buckets in production.
- **Metrics, dashboard, alerts, drift check, load tester**, below.
- **Before exposure:** three audit findings fixed. Stored searches no
  longer return the session id that authorizes export and deletion; the
  crawler refuses non-public addresses on every fetch, redirect and DNS
  answer; transiently failing feeds back off instead of being re-enqueued
  every cycle.

Not yet done: the deployment itself. Creating the project needs a Railway
login and plan choice (the hot set needs more than the Hobby plan's 8 GB
per service), and loading the catalog is a multi-hour copy. Everything is
ready to apply; the steps are in the deployment document.

## The skew demonstration

The drift check and the alerts were exercised against a real fault
rather than a synthetic signal. With the normal configuration, 913
searches over the relevance queries (two minutes, six clients) were
recorded as the baseline: 2.5% zero results, vector legs ok on 97%, the
lexical leg timing out on 84% at its 300 ms limit under this load,
result sources 52% two-tower, 45% E5, 3% lexical, and the serving mean of
all 31 ranker features. Checking that same window found nothing.

The API was then restarted with the vector legs' time limit at 1 ms, so
every E5 and two-tower leg timed out and the ranker saw only lexical
candidates: a skew it was never trained on. Seven minutes of traffic
(6,635 searches) followed. The outcome:

| detector | result |
|---|---|
| drift check, first window | 14 findings, exit 2, posted to the webhook |
| zero-result rate | 61% against a 3% baseline |
| e5 and two_tower legs | timed out on 100% of searches (critical) |
| top-score distribution | PSI 0.41, mean 2.28 against 2.69 (critical) |
| result-source mix | PSI 11.65, lexical 100% (critical) |
| feature means | `e5_rr` and `two_tower_rr` at 0, `two_tower_sim_gap` −11.5 against −1.07, `has_transcript` 0.50 against 0.12, and five more |
| Grafana `Vector leg failures` | active at 23:26:20Z, six minutes after the fault at 23:20:26Z (15-minute window above 10%, held 5 minutes), posted to the webhook |

The full findings are in `monitoring/drift-demo.json`, the baseline in
`monitoring/baseline.json`. One correction came out of the run: rules
whose query returns no series when healthy (5xx share, NaN features,
unreachable services) paged as `DatasourceNoData`; they now treat an
empty result as OK.

## Load

`cmd/loadtest` over the 180 relevance queries, each search followed by an
impression event for its results and a playback-start event, with the
stored event count read back from the database:

| clients | duration | searches | per s | p50 ms | p95 ms | p99 ms | availability | events accepted / stored |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 min | 93 | 1.5 | 545 | 1,213 | 1,715 | 100% | 1,953 / 1,953 |
| 6 | 2 min | 913 | 7.6 | 697 | 1,524 | 1,692 | 100% | 18,690 / 18,690 (acceptance only) |
| 8 | 2 min | 1,104 | 9.2 | 781 | 1,527 | 1,722 | 100% | 22,706 / 22,706 |

Against the SLOs:

- **Availability 99.9%:** met; every search returned 200 across all
  three runs (2,110 searches).
- **Event persistence 99.99%:** met; every accepted event was stored
  (24,659 verified in the database).
- **Search p95 under 500 ms:** not met on the laptop, as it was not in
  the serving report (593 ms with one client). Under eight clients p95 is
  1.5 s, and it is 1.2 s even with one client with the monitoring
  containers and the speech server sharing the machine. The cause is the
  one the serving report established: the vector hot set (about 6 GB)
  does not fit the Docker VM's cache beside the lexical scans, so the
  legs contend for memory. The deployment gives Postgres its own host
  with `shared_buffers=6GB` and prewarmed indexes; the p95 is to be
  re-measured there with this same tool before the budget is called met
  or missed.

Event handling is cheap (p50 under 10 ms, p99 75 ms at eight clients)
and did not affect search latency.

## Exit criterion

- Public URL serving text and voice search: **pending the Railway
  apply and catalog load**; the images, configuration and load script
  are complete and tested locally.
- Drift demo fires an alert: **met**, through both the drift check and
  Grafana, to the webhook.
- Rollback: documented as a redeploy of the previous Railway deployment;
  not yet exercised because nothing is deployed.
- Load-test report: this document, local numbers; the deployed
  measurement is the follow-up.
