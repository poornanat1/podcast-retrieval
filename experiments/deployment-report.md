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

## The deployment

Applied the same evening on the Pro plan, at
[api-production-0ff59.up.railway.app](https://api-production-0ff59.up.railway.app):
seven services (postgres, models, speech, api with two replicas,
prometheus, grafana, drift), six volumes and two buckets, from the
infrastructure file in four applies. The catalog copy over a temporary
TCP proxy took 46 minutes for 1,788 podcasts, 795,533 episodes, 87,300
transcripts and 1,678,366 vectors (three models' sets, 10 GB on the
volume), and the two HNSW indexes 14 minutes serially. What the first
deployment taught, each fixed in the repository:

- Railway's builder rejects Docker cache mounts without an id (dropped).
- A start command bypasses the image entrypoint, so Postgres ran as root
  and refused (`docker-entrypoint.sh postgres …`).
- Volumes mount root-owned, so the images that write to them run as
  root.
- Health checks arrive over IPv4 and the private network is IPv6, so the
  Python servers bind a dual-stack socket themselves.
- The container's 64 MB `/dev/shm` breaks parallel index builds and
  parallel queries; indexes build serially and Postgres uses mmap-backed
  dynamic shared memory.
- The bucket helper in the infrastructure file did not create buckets;
  they were created with the CLI and adopted by name.
- The drift cron job needs the checked-in baseline in its image.

Rollback is Railway's redeploy of a previous deployment in the
dashboard; the CLI only redeploys the latest, so it was not exercised
from here.

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
- **Search p95 under 500 ms:** not met on the laptop, nor on the first
  deployment (681 ms); met after the lexical fix (179 ms at one client,
  359 ms at eight). See below.

Event handling is cheap (p50 under 10 ms, p99 75 ms at eight clients)
and did not affect search latency.

### On Railway

The same tool from the laptop against the public URL, after the catalog
load, index build and prewarm, with the stored event count read through
the proxy:

| clients | duration | searches | per s | p50 ms | p95 ms | p99 ms | availability | events accepted / stored |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 min | 100 | 1.7 | 476 | 681 | 1,189 | 100% | 2,079 / 2,079 |
| 8 | 2 min | 1,869 | 15.6 | 376 | 915 | 1,671 | 100% | 38,834 / 38,834 |

These times include the round trip from the laptop, which the event
requests put at about 50 ms (their p50). Availability and event
persistence are met again. Search p95 was 681 ms with one client and
915 ms with eight: better than the laptop, still above 500 ms. Within a
search, the lexical leg hit its 300 ms limit on most queries, as it had
on the laptop.

### After the lexical fix

The lexical leg ranked every match with the full transcript text vector,
which cost the whole budget on broad queries. With a small ranking
vector over each transcript's opening, per-source ranking cut to the
candidate depth, and a cap on matches ranked per source (commit
`e59f1d6`, [final-report.md](final-report.md)), the same load test
against the public URL, caches warm:

| clients | duration | searches | per s | p50 ms | p95 ms | p99 ms | availability | lexical timeouts |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 min | 249 | 4.1 | 140 | 179 | 214 | 100% | |
| 8 | 2 min | 2,787 | 23.2 | 166 | 359 | 1,576 | 100% | 4% (65 of 1,463 in the window) |

**The 500 ms p95 budget is met**, with the internet round trip included:
179 ms at one client and 359 ms at eight, against 681 and 915 before.
Throughput at eight clients went from 15.6 to 23.2 searches per second
on the same two API replicas. The first minute after the deployment
still shows the cold-cache tail (p95 2.4 s, p99 10 s) until the API's
start-time prewarm and the operating system's cache catch up; the
one-client row above was measured after that.

## Exit criterion

- Public URL serving text and voice search: **met**; searches answer in
  under half a second at the median over the internet.
- Drift demo fires an alert: **met**, through both the drift check and
  Grafana, to the webhook.
- Rollback: documented as a redeploy of the previous Railway deployment
  in the dashboard; not exercised from the CLI.
- Load-test report: this document, local and deployed numbers, before
  and after the lexical fix.
