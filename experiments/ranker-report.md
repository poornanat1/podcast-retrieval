# Learned ranker vs reciprocal-rank fusion

A second-stage ranker over the hybrid candidate set, compared with RRF
ordering of exactly the same candidates, with every reported number
measured out of fold. Measured 2026-10-21 against the 795,533-episode
catalog and the 180-query relevance set, on one judgment pool of 71,518
judgments in which every candidate of every query is judged.

> **Update (2026-10-24).** The numbers below are for the ranker retrained
> after the serving parity checks found a training-serving skew: training
> rows read from Parquet carried NaN for missing durations, where serving
> sends null and the feature code produced 0
> ([serving-report.md](serving-report.md)). Out of fold, NDCG@10 moved from
> 0.792 to 0.794 and MAP@10 from 0.813 to 0.817. The harness comparison
> was re-run with the retrained model; the other systems are unchanged.

## Setup

**Candidates.** The top 50 of each of three legs, each applying the
query's hard filters: lexical full-text search, pretrained E5, and the
champion two-tower model. Their union averages 123.5 episodes per query
(22,231 in total) and holds on average 56% of a query's judged-relevant
episodes. For
every candidate the generator reads the evidence of all three legs, not
only the legs that retrieved it: exact cosine similarity under both vector
models, lexical score components, and a transcript-to-query similarity
from E5 vectors of the first 2,000 transcript characters (87,300
transcripts embedded for this). `ml/ranking/candidates.py`.

**Features.** 31, defined once in `ml/ranking/features.py` for training,
evaluation, and a future online path:

- leg evidence: both vector similarities, each also as the gap to the best
  candidate for the query, transcript similarity, lexical score, its
  components, and its share of the query's best;
- candidate source: reciprocal rank in each leg (0 when absent), number of
  legs, RRF score;
- matches: query against show title, publisher, and episode title, as
  exact match, whole-token containment either way, and token coverage,
  plus query length;
- item: publication age, duration, show size, provider popularity,
  transcript availability.

The design also lists user–podcast affinity, language and duration
preferences, and completion and skip rates. No interaction data exists,
so they are not built. Query type is never a feature: it is an
evaluation label, unknown at serving time.

**Model.** LightGBM LambdaRank, hyperparameters fixed in
`experiments/ranking/ranker-v1.json` before training and never tuned on
these queries: 300 trees, learning rate 0.05, 15 leaves, 20 minimum rows
per leaf, row and feature subsampling 0.8, truncation level 20. Labels are
the graded judgments 0–3.

**Evaluation.** The 180 queries are split into 5 folds stratified by
query type, and each query is scored by the fold model trained without
it. The harness runs the ranker that way too (`reranker` system,
`out_of_fold`), so its numbers are the same out-of-fold numbers measured
through the standard evaluation path, with latency. A final model trained
on all queries is saved for serving.

**A retrieval fix first.** Measuring the candidates exposed two problems
in the vector legs, fixed before the comparison below
(`ml/retrieval/vector.py`, [05-embeddings.md](../docs/05-embeddings.md)):
`relaxed_order` iterative scans returned rows slightly out of distance
order (31 of 180 queries), and `hnsw.ef_search = 200` found only 83–85%
of the true top 10. Results are now re-sorted by exact distance and
`ef_search` is 1000 (top-10 recall 0.89 two-tower, 0.92 E5, median latency
unchanged). This moved the baselines up: the champion's NDCG@10 rose from
0.729 to 0.758 on the current pool. Every number below uses the fixed
search.

## Results

Harness, all five systems on one pool (every top-10 result judged):

| system | NDCG@5 | NDCG@10 | MAP@10 | P@5 | P@10 | MRR | Hit@10 | R@100 |
|---|---|---|---|---|---|---|---|---|
| lexical | 0.563 | 0.538 | 0.507 | 0.608 | 0.567 | 0.713 | 0.861 | 0.261 |
| pretrained E5 | 0.749 | 0.720 | 0.730 | 0.821 | 0.783 | 0.905 | 0.972 | 0.369 |
| two-tower (champion) | 0.778 | 0.758 | 0.770 | 0.846 | 0.820 | 0.913 | 0.967 | 0.421 |
| RRF over three legs | 0.791 | 0.766 | 0.782 | 0.853 | 0.830 | **0.923** | 0.972 | 0.475 |
| **ranker** | **0.812** | **0.794** | **0.817** | **0.878** | **0.859** | 0.915 | **0.978** | **0.514** |

Pairwise accuracy (share of judged candidate pairs with different grades
in the right order), over the candidate set: ranker 0.761, two-tower
similarity 0.733, E5 0.681, RRF 0.592, lexical 0.517.

Paired bootstrap, ranker minus each system (180 queries, 10,000
resamples):

| vs | NDCG@10 | MAP@10 | NDCG@5 | Recall@100 |
|---|---|---|---|---|
| RRF over three legs | +0.028 (+0.007 to +0.049) | +0.035 (+0.006 to +0.064) | +0.021 (−0.004 to +0.047) | +0.039 (+0.032 to +0.047) |
| two-tower (champion) | +0.036 (+0.013 to +0.061) | +0.047 (+0.019 to +0.076) | +0.034 (+0.005 to +0.064) | +0.094 (+0.077 to +0.112) |
| pretrained E5 | +0.074 (+0.045 to +0.105) | +0.087 (+0.049 to +0.126) | +0.063 (+0.031 to +0.096) | +0.146 (+0.130 to +0.162) |
| lexical | +0.256 (+0.215 to +0.298) | +0.310 (+0.258 to +0.363) | +0.249 (+0.204 to +0.294) | +0.253 (+0.219 to +0.289) |

Against RRF the ranker wins 75 queries and loses 47 on NDCG@10. MRR is
the one metric where it does not lead (0.915 vs 0.923, within noise): the
first relevant result was already near the top in most queries.

Ordering only, with the candidates held fixed (the training script scores
every ordering of the same candidate set): ranker 0.794 NDCG@10, two-tower
similarity 0.773, RRF 0.766, E5 similarity 0.736, lexical score 0.628. The
two-tower similarity over the union beats the two-tower system itself
(0.758) because the other legs bring in episodes the ANN index missed.

### By cohort

NDCG@10 by query type (query counts in parentheses); ranker minus RRF with
95% interval below:

| system | entity (20) | exploratory (85) | filtered (35) | navigational (40) |
|---|---|---|---|---|
| lexical | 0.830 | 0.436 | 0.310 | 0.808 |
| pretrained E5 | 0.668 | 0.654 | 0.605 | 0.987 |
| two-tower | 0.805 | 0.684 | 0.681 | 0.958 |
| RRF over three legs | 0.813 | 0.698 | 0.660 | 0.979 |
| **ranker** | **0.879** | **0.722** | **0.692** | **0.995** |
| ranker − RRF | +0.066 (−0.021 to +0.171) | +0.023 (−0.001 to +0.048) | +0.031 (−0.033 to +0.096) | +0.017 (+0.000 to +0.039) |

Recall@100 by relevant-item cohort:

| system | transcript (122) | no transcript (176) | head (177) | tail (134) | seen (152) | known show (180) | unseen show (128) | after snapshot (169) |
|---|---|---|---|---|---|---|---|---|
| lexical | **0.607** | 0.191 | 0.272 | 0.223 | 0.254 | 0.254 | 0.231 | 0.253 |
| pretrained E5 | 0.205 | 0.405 | 0.369 | 0.385 | 0.399 | 0.356 | 0.395 | 0.299 |
| two-tower | 0.192 | 0.471 | 0.393 | 0.449 | 0.430 | 0.411 | 0.433 | 0.457 |
| RRF over three legs | 0.532 | 0.480 | 0.483 | 0.489 | 0.468 | 0.461 | 0.504 | 0.495 |
| **ranker** | 0.473 | **0.531** | **0.509** | **0.525** | **0.500** | **0.499** | **0.552** | **0.519** |

The ranker leads every cohort except relevant episodes with transcripts,
where it gives up some of RRF's recall (0.473 vs 0.532) and stays below
lexical search (0.607). It places lexical-only transcript matches lower
than RRF does; the transcript similarity feature carries almost no weight
(below). Cold-start slices (relative to the champion's training data) show
no penalty: the ranker's best slice is shows never seen in training.
Query languages other than English have one to three queries each and are
not reportable.

### What the ranker uses

Share of total split gain in the final model:

| feature | gain share |
|---|---|
| two-tower similarity gap to the query's best | 34.9% |
| RRF score | 9.6% |
| lexical score, absolute and relative | 9.2% |
| E5 similarity and its gap | 7.7% |
| show size (episodes) | 3.8% |
| exact show-title match | 3.6% |
| duration | 3.0% |
| provider popularity | 2.9% |
| publication age | 2.9% |
| transcript similarity | 0.2% |

One feature is never used (`query_in_title`). The
transcript similarity is nearly unused. Only about 11% of episodes have a
transcript, so the feature is zero for most candidates; whether a
different excerpt than the first 2,000 characters would carry more signal
is untested.

### Failure queries from the ablation report

NDCG@10 on the queries the champion failed in
[ablation-cold-start-report.md](ablation-cold-start-report.md), with the
fixed vector search:

| query | two-tower | RRF | ranker | E5 |
|---|---|---|---|---|
| *ted talks daily* | 0.00 | 0.69 | **1.00** | 1.00 |
| *today explained* | 1.00 | 1.00 | 1.00 | 1.00 |
| *football ramble* | 1.00 | 1.00 | 1.00 | 1.00 |
| *short daily news briefing* (under 20 min) | 0.10 | 0.57 | 0.11 | **1.00** |
| *is there life on europa* | 0.77 | **0.93** | 0.89 | 0.88 |
| *the new statesman* | 0.22 | **0.93** | 0.56 | 0.86 |
| *vince ferrara* | 0.00 | 0.00 | **0.79** | 0.00 |
| *jim metzner* | 0.51 | 0.60 | **0.77** | 0.00 |

Two of the three show-collapse failures, and most of *is there life on
europa*, came from the ANN search rather than the model: with exact
re-sorting and the wider search, the two-tower system gets them right. The
ranker fixes the remaining collapse (*ted talks daily*) and the
person-name misses through the exact-match and lexical features. It does
not fix *short daily news briefing*, where the two-tower similarity (its
strongest feature) is confidently wrong, or *the new statesman*, where RRF
is better.

## Latency

Per-query stage latency through the Python harness on 2026-10-21, with the
legs run one after another, on a laptop with PostgreSQL in Docker. The
served pipeline, with the legs run concurrently, is measured in
[serving-report.md](serving-report.md):

| stage | p50 ms | p95 ms |
|---|---|---|
| query encoding (E5 + two-tower) | 22 | 40 |
| lexical leg | 59 | 1,046 |
| E5 leg | 94 | 262 |
| two-tower leg | 208 | 446 |
| candidate evidence (one query for rows, one for vectors) | 41 | 57 |
| **ranker: features** | **6** | **8** |
| **ranker: scoring (LightGBM, 1 thread)** | **1.4** | **1.8** |
| total, ranker | 481 | 1,387 |
| total, RRF over three legs | 491 | 1,692 |

The ranker stage adds about 10 ms at p95 to the RRF pipeline it replaces.
The whole hybrid pipeline is over the design's search budget of 500 ms at
p95 with or without it, because the three legs run one after another and
the lexical leg has a long tail on broad queries. The legs are
independent, so running them concurrently would bound the pipeline by the
slowest leg; the lexical tail needs its own fix (a statement timeout or a
cheaper ranking for very broad matches).

## Decision

The exit criterion is met for the ordering stage: the learned ranker beats
RRF ordering of the same candidates on NDCG@10 (+0.028) and MAP@10
(+0.035), both outside noise, at a cost of about 10 ms. It also beats every
single system, including the champion two-tower model (+0.036 NDCG@10).
The ranker replaces RRF as the ordering stage. NDCG@5 and MRR gains over
RRF are within noise.

The 500 ms end-to-end budget is not met by any hybrid configuration, and
that is retrieval latency, not ranking: it is the first item for the
serving work (concurrent legs, a bounded lexical leg).

## Limits

- **180 queries, LLM judgments.** Every label is an LLM grade with no
  human audit; the ranker is trained on the same judge's grades it is
  scored with, so judge bias is learned as well as measured.
- **Candidate ceiling.** The candidate set holds on average 56% of a
  query's judged-relevant episodes; no ordering can exceed that recall.
- **Fixed hyperparameters, one seed.** No tuning was done; a nested
  cross-validation would be needed to tune without leaking.
- **Not built:** user features (no interaction data), and the
  cross-encoder challenger, which is out of scope here. A cross-encoder
  must beat this ranker by enough to justify scoring ~125 pairs per query
  on every search.

## Reproduce

```sh
make embed-transcripts
make relevance-pool POOL_SYSTEMS=hybrid-rrf3,lexical-fts ...  # judge the candidate set
make relevance-judge && make relevance-export && make pool-export
make train-ranker RANKER_CONFIG=experiments/ranking/ranker-v1.json
uv run python -m ml.evaluation.compare experiments/eval/lexical-v1.json \
    experiments/eval/vector-e5small-v1.json experiments/eval/two-tower-names-v1.json \
    experiments/eval/hybrid-rrf3-v1.json experiments/eval/reranker-v1.json \
    --cold-start-dataset data/datasets/search-retrieval-llm-names-1.0.0-56e2c0708f7e \
    --json-out compare.json
make cohort-report COMPARE_JSON=compare.json
make significance CANDIDATE_RUN=<reranker run> BASELINE_RUN=<hybrid-rrf3 run>
```
