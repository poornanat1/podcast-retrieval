# Serving: online path, parity, and events

The learned search pipeline served end to end, from typed text to ranked
results and events, with the training-side models proved equal to the
production ones before promotion. Measured 2026-10-24 on a laptop: the
model server and the Go API run natively, and PostgreSQL runs in Docker
(15 CPUs, 8 GB VM). The catalog is the 795,533-episode frozen corpus. The
design is in [docs/14-serving.md](../docs/14-serving.md).

## What runs

| component | serves |
|---|---|
| bundle `42df3779d7cf` | two-tower `two-tower-llm-names-…-3b1c13c6` query tower, E5 query encoder, and ranker `ranker-lambdamart-7bb99f61`, all ONNX; tokenizers; feature contract |
| model server (`ml/serving/server.py`) | `/v1/encode`, `/v1/rerank` on ONNX Runtime |
| Go API (`cmd/api`) | `/v1/search/text`, `/v1/search/{id}`, `/v1/episodes/{id}`, `/v1/events/impression`, `/v1/events/playback/{action}`, the web app at `/` |

The example query from the design document,

    English episodes under 45 minutes about practical uses of AI, published this year, no explicit content

is understood as the text *practical uses of AI* with four hard filters
(English, at most 2,700 s, published after 2026-01-01, not explicit). Every
returned result satisfies them, with explanations such as *Episode title
mentions "ai"*, *33 min (under 45)*, and *Published Jan 20, 2026*.

## Parity: what the gate caught

The first bundle (`f23d0c248d56`) failed parity, and promotion refused it
with seven failed checks. The failures came from three causes, all fixed
before the bundle that was promoted:

1. **ONNX ranker thresholds.** The LightGBM exporter stores split
   thresholds as float32. On 32 of the 3,628 golden candidate rows, a
   feature sat between a threshold and its float32 rounding, and the
   ONNX score differed from LightGBM's by up to 0.052. That changed the
   top 10 of two golden queries. The export check had passed: on random
   inputs these collisions almost never happen. The fix is to round every
   threshold down to the largest float32 at or below it, which gives the
   same decisions on every float32 input. The export check now probes
   values sitting exactly on every threshold.
2. **A null-versus-NaN skew in the features.** The ranker trains from
   cached Parquet, where a missing duration is NaN, and serving receives
   JSON null. In the feature code `NaN or 0` stays NaN, because NaN is
   truthy, while `None or 0` is 0. So 42 training rows had NaN durations,
   which LightGBM routes as missing, while serving would send 0. The
   distribution check surfaced it as a NaN statistic. Both paths now map
   any missing number to the same default. The ranker was retrained: out
   of fold, NDCG@10 went from 0.792 to 0.794 and MAP@10 from 0.813 to
   0.817. A test now pins the behavior.
3. **Cold-cache lexical queries.** Right after Postgres restarted, two
   navigational queries took 12.9 s and 15.4 s in the lexical leg (1.4 s
   and 2.6 s warm). That exceeded the parity run's 10 s leg limit and
   removed their lexical candidates. Parity now runs with leg limits above
   the cold-cache worst case, because it tests the pipeline logic, not
   latency.

The promoted bundle passes all 17 checks:

| check | result | threshold |
|---|---|---|
| feature list and version | identical | identical |
| tokenizer: serving vs training inputs | 30 / 30 queries identical | all |
| query vectors: ONNX vs PyTorch | 3.9e-7 | 1e-4 |
| query vectors: unit norm | within 3.6e-7 | 1e-3 |
| ranker: ONNX vs LightGBM | 3.9e-6 | 1e-4 |
| features: serving path vs training | 0 | 1e-4 |
| missing values: nulled fields | 0 | 0 |
| training rows vs serving, same queries | 2.7e-5 SD | 0.05 SD |
| no NaN in training features | 0 | 0 |
| online: served bundle, no leg failure | yes, 30 / 30 | all |
| online: query vectors | 3.9e-7 | 1e-4 |
| online: candidate sets (Jaccard) | 1.000 mean, 1.000 worst | 0.98, 0.90 |
| online: features of shared candidates | 1.3e-6 | 1e-4 |
| online: scores within 1e-3 | 99.9% | 99% |
| online: identical top 10 | 30 / 30 | 95% |

The Go API and the training pipeline produce the same candidates, the same
features to 1e-6, and the same top 10 on every golden query, including the
edge cases.

## Latency

All 180 relevance queries, sequentially, in the production configuration
(lexical leg limited to 300 ms):

| stage | p50 ms | p95 ms |
|---|---|---|
| query encoding (both towers, model server) | 14 | 16 |
| retrieval (three legs, concurrent) | 301 | 495 |
| candidate evidence | 43 | 76 |
| reranking (features + ONNX, model server) | 11 | 16 |
| **total** | **349** | **593** |

The 500 ms p95 budget is missed by about 90 ms, and the excess is in
retrieval. What the measurements show:

- **Postgres memory.** The default 128 MB buffer cache made every vector
  query read hundreds of pages of the 15 GB embedding table. A cold query
  took 242 ms (668 blocks read) against 22 ms warm. Raising
  `shared_buffers` to 2 GB and prewarming the served indexes and rows
  (`make prewarm`) is now part of the setup.
- **Cached prepared statements.** pgx's statement cache let Postgres
  switch to generic plans, which cannot use the per-model partial HNSW
  indexes. Running statements in exec mode cut the median from 442 to 358
  ms.
- **Contention between legs.** The two-tower leg alone takes 13 ms p50 and
  24 ms p95. Alongside the E5 leg it takes 19 and 34 ms; alongside lexical,
  26 and 61 ms; alongside both, 87 and 291 ms. Disabling parallel query
  workers changed nothing. The broad lexical scans evict vector pages from
  a cache too small for the whole hot set, which is about 6 GB in an 8 GB
  VM. The fixes are sizing, not code: a database host with memory for the
  hot set, a table holding only the served models' vectors (or
  half-precision vectors), and lexical search on its own replica.

### The lexical time limit

The lexical leg times out on 94 of 180 queries at 300 ms. Raising the
limit to 20 s, the most the API's 30 s write timeout allows, gives:

| lexical limit | lexical timeouts | NDCG@10 (in-sample) | Recall@100 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| 300 ms | 94 | 0.932 | 0.478 | 349 | 593 |
| 20 s | 2 | 0.937 | 0.520 | 592 | 5,739 |

The limit costs 0.005 NDCG@10 and 0.04 Recall@100, and saves 5 s at p95.
When the lexical leg times out, the ranker still receives lexical scores
for every candidate the vector legs found, because the evidence query
computes them by id. Only lexical-only candidates are lost. The 300 ms
limit stays. These quality numbers are in-sample, because the served
ranker was trained on all 180 queries (out of fold it scores 0.794). They
compare configurations; they are not a quality estimate.

## Interaction events

One live search followed by the events the web app sends (`s_live`
session):

| event | episode | displayed rank | candidate sources | score | model | features | index |
|---|---|---|---|---|---|---|---|
| search | | | | | 42df3779d7cf | 1 | two_tower=two-tower:two-tower-llm-names-… |
| impression | 3552 | 1 | lexical, e5, two_tower | 3.756 | 42df3779d7cf | 1 | same |
| impression | 3282 | 2 | lexical, e5, two_tower | 3.439 | 42df3779d7cf | 1 | same |
| playback_start | 3552 | 1 | lexical, e5, two_tower | 3.756 | 42df3779d7cf | 1 | same |
| playback_progress | 3552 | 1 | lexical, e5, two_tower | 3.756 | 42df3779d7cf | 1 | same |
| playback_complete | 3552 | 1 | lexical, e5, two_tower | 3.756 | 42df3779d7cf | 1 | same |

The client sent only the query id, episode ids, displayed ranks, and
playback positions. Everything else was copied from the stored search. An
impression for an episode that search did not return was rejected.

## Exit criterion

- **A text search runs the full learned pipeline end to end.** Query
  understanding, three retrieval legs, ONNX reranking, explanations, and
  storage all run, with a stored record of the bundle, features, and
  indexes used.
- **Parity tests pass and gate promotion.** All 17 checks pass for
  `42df3779d7cf`. Promotion refused `f23d0c248d56`, and the reasons it
  failed were real defects.
- **Events land with full attribution,** server-side from the stored
  search.

Not met: the 500 ms p95 search budget (593 ms), for the sizing reasons
above. It moves to the load and deployment work.

## Not covered

- **No browser automation.** The web app was checked through the API it
  calls and by serving its files; it has not been driven in a browser.
- **Query understanding is unlabeled.** Its rules are unit-tested, but the
  relevance set has no labels for parsed filters, so its accuracy on real
  queries is unmeasured.
- **One sequential client.** All latency is for one client issuing queries
  in sequence. Concurrency and load are separate work.
