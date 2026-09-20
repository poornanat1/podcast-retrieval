# Two-tower retrieval: first fine-tuned runs

Measured 2026-09-20 against the 797,216-episode catalog (the frozen
corpus plus the 2026-09-18 deviation recorded in
[docs/09-catalog.md](../docs/09-catalog.md)), on the 180-query graded
relevance set used by the [baseline report](baseline-report.md). Every
number below was produced on this corpus in the same session; baseline
rows that were not re-run are marked and quoted from the baseline report.

**Reproduce:** `make train TRAIN_CONFIG=experiments/training/<run>.json`,
`make index ARTIFACT=data/models/<artifact>`, then
`make eval EVAL_CONFIG=experiments/eval/<config>.json`. Training runs log
to MLflow experiment `two-tower-training`, evaluations to
`retrieval-eval`.

## Setup shared by both runs

| | |
|---|---|
| Text encoder | `intfloat/multilingual-e5-small`, shared by both towers, top 6 of 12 layers and heads trainable (11.6M of 118.5M parameters); token embeddings and lower layers frozen |
| Episode tower inputs | titles, categories, publisher, description (256 tokens); language, category, duration-bucket and publication-age embeddings; **no transcript** (the training snapshot predates transcript excerpts) |
| Objective | contrastive softmax over in-batch positives + 4 random catalog negatives per query, temperature 0.05, sibling-positive masking |
| Optimization | AdamW (encoder 2e-5, heads 1e-3), 100 warmup steps then linear decay, one epoch, gradient checkpointing; run 1 batch 16 queries (80 candidates, 3,205 steps), run 2 batch 8 (40 candidates, 6,358 steps) |
| Hardware | Apple laptop GPU; run 1 ~2.6 s/step (2.4 h), run 2 ~0.8 s/step (84 min) once Docker was stopped during training; index build ~55 min including the HNSW build |

The candidate-set diagnostics saturate within a few hundred steps on both
runs (validation accuracy ≥ 0.997 by step 200): synthetic title queries
against random negatives are trivially separable, so those numbers say
nothing about retrieval quality. Only the relevance set does.

## Run 1: `two-tower-v1` — invalidated by a feature skew

Config [experiments/training/two-tower-v1.json](training/two-tower-v1.json),
dataset `search-retrieval-bootstrap-1.1.0` (English only, 51,273 training
examples), artifact `two-tower-bootstrap-1.1.0-30f0542e74a5-95d621bd`,
3,205 steps.

| system | R@10 | R@50 | R@100 | Hit@10 | MRR | NDCG@10 | tail cov. | p50 / p95 ms |
|---|---|---|---|---|---|---|---|---|
| vector-e5-small (re-run) | **0.292** | **0.598** | **0.637** | **0.939** | **0.896** | **0.731** | 0.486 | 83 / 328 |
| two-tower-v1 | 0.007 | 0.023 | 0.038 | 0.106 | 0.046 | 0.014 | 0.707 | 12 / 192 |
| hybrid-rrf (baseline report, frozen corpus) | 0.285 | 0.900 | 0.946 | 0.978 | 0.884 | 0.731 | 0.486 | 461 / 6,475 |
| hybrid-rrf-two-tower-v1 | 0.119 | 0.383 | 0.447 | 0.844 | 0.648 | 0.355 | 0.606 | 282 / 8,830 |

The fine-tuned index is effectively random (NDCG@10 0.014 across every
query type), and as the vector leg of the hybrid it drags the fusion down
to what lexical search achieves alone (R@100 by cohort: transcript 0.85 /
no-transcript 0.34, the lexical baseline's signature).

**Cause: a training-serving skew in the publication-age feature.** Every
bootstrap example's event time *is* the episode's publication time, so
during training the age feature was always bucket 1 ("published within a
week"). The index computes age against the build time, where the catalog
falls into buckets 4–7, whose embeddings were never trained — random
vectors that the (trained) head then amplified. Measured on the trained
model: episode vectors featurized with training-time ages versus
index-time ages agree at cosine 0.60 on average (minimum 0.20), while the
query side is unaffected (0.993 with and without filter context). A second
skew is likely: the dataset is English-only, so the language embeddings of
the ~30% non-English catalog were never trained either.

The model itself learned what it was shown: with training-time
featurization it ranks an episode's own title query first 96% of the time
among 48 catalog episodes. The failure is in the contract between training
features and index features, not in the towers or the loss.

## Run 2: `two-tower-v2` — skew removed

Config [experiments/training/two-tower-v2.json](training/two-tower-v2.json),
dataset `search-retrieval-bootstrap-1.2.0` (all catalog languages, 50,858
training examples, 6,358 steps), artifact
`two-tower-bootstrap-v2-1.2.0-e2e0bf0793e6-67868919`.

Changes from run 1, all on the training side (the dataset's event times
and the index job are untouched):

- **Publication-age jitter.** Each example's age reference time is moved
  forward by a seeded, log-uniform delay of 1 day to 10 years, so every
  age bucket is trained (`age_jitter_days`).
- **Query-language context.** Half of the queries carry their positive's
  language as context, so the query tower's language embeddings are
  trained (`query_language_prob`).
- **Multilingual examples**, so episode-side language embeddings are
  trained for every catalog language.

Evaluated against the judgment pool as it stood before this run
(candidates pooled from the five baseline systems only):

| system | R@10 | R@50 | R@100 | Hit@10 | MRR | NDCG@10 | judged share of top-10 |
|---|---|---|---|---|---|---|---|
| vector-e5-small (re-run) | 0.292 | 0.598 | 0.637 | 0.939 | 0.896 | 0.731 | 0.97 |
| two-tower-v2 | 0.052 | 0.111 | 0.147 | 0.480 | 0.306 | 0.144 | **0.15** |
| hybrid-rrf-two-tower-v2 | 0.139 | 0.428 | 0.503 | 0.888 | 0.649 | 0.405 | — |

Removing the skew moved the standalone model from random (0.014) to 0.144
NDCG@10 and the hybrid from 0.355 to 0.405, still far below the pretrained
baseline. The last column explains most of the remaining gap: only 15% of
the fine-tuned model's top-10 results had ever been judged, against 97% for
E5, whose own top-20 was one of the pooled sources. Under pooled
evaluation every unjudged result counts as irrelevant, so a system that
retrieves *different* episodes is penalized for novelty before relevance is
even measured. Two checks confirmed that the model itself is close to the
baseline:

- **Re-ranking the judged pool only** (each query's judged candidates,
  scored by dot product, no ANN): two-tower-v2 NDCG@10 0.790 / MRR 0.878,
  the same model with its residual heads disabled 0.805 / 0.888, pretrained
  E5 0.850 / 0.918. The fine-tuned model ranks judged episodes 6 points
  below E5, not 60. (On the extended pool below, which includes the
  fine-tuned model's own low-graded candidates, the same re-ranking gives
  0.587 / 0.745 against 0.832 / 0.911 for E5 — a harder pool that tracks
  the full-catalog gap and is now the validation signal training uses.)
- **No hubness**: across all 180 queries the fine-tuned index returned
  1,769 distinct episodes in 1,800 top-10 slots (E5: 1,771), and no episode
  appeared more than four times.

So the fine-tuned model's candidates were pooled (top 50 per query, 8,363
new (query, episode) pairs) and graded by the same LLM judge with the same
rubric, then the affected systems were re-evaluated on the extended pool:

| system | R@10 | R@50 | R@100 | Hit@10 | MRR | NDCG@10 | judged share | tail cov. | p50 / p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| vector-e5-small | **0.207** | 0.444 | 0.498 | 0.933 | **0.891** | **0.707** | 0.97 | 0.484 | 125 / 456 |
| two-tower-v2 | 0.095 | 0.351 | 0.374 | 0.733 | 0.594 | 0.413 | 1.00 | 0.646 | 81 / 250 |
| hybrid-rrf-two-tower-v2 | 0.141 | 0.486 | 0.646 | 0.944 | 0.748 | 0.546 | 0.98 | 0.568 | 329 / 3,587 |
| hybrid-rrf (E5 leg) | 0.202 | **0.642** | **0.712** | **0.967** | 0.875 | 0.698 | 0.98 | 0.485 | 355 / 3,647 |

180 queries are scored now (the extended pool gave the last query a
relevant document). Every recall figure fell for E5 relative to the earlier
table because the pool grew: 8,363 new judgments found more relevant
episodes, so the denominators are larger. That is the pooling effect
working as intended, and it is why all three systems are compared on the
same, extended pool.

NDCG@10 by query type on the extended pool:

| system | navigational | entity | exploratory | filtered |
|---|---|---|---|---|
| vector-e5-small | **0.942** | **0.589** | **0.669** | **0.600** |
| two-tower-v2 | 0.444 | 0.162 | 0.385 | 0.586 |
| hybrid-rrf-two-tower-v2 | 0.716 | 0.513 | 0.488 | 0.512 |
| hybrid-rrf (E5 leg) | 0.915 | 0.741 | 0.652 | 0.534 |

Recall@100 by relevant-item cohort:

| system | transcript | no transcript | head | tail |
|---|---|---|---|---|
| vector-e5-small | 0.268 | 0.550 | 0.514 | 0.507 |
| two-tower-v2 | 0.204 | 0.398 | 0.296 | 0.409 |
| hybrid-rrf-two-tower-v2 | **0.886** | 0.590 | 0.605 | 0.654 |
| hybrid-rrf (E5 leg) | 0.861 | **0.687** | **0.760** | **0.688** |

## Run 3: `two-tower-v3` — natural queries, hard negatives, pool-selected checkpoint

Config [experiments/training/two-tower-v3.json](training/two-tower-v3.json),
dataset `search-retrieval-llm-1.0.0` (18,451 training examples), artifact
`two-tower-llm-v3-1.0.0-d52b7c3b104f-62d87868`. Everything run 2 diagnosed
was changed at once, so this run measures the package, not the parts:

- **Queries written by an LLM, not copied titles.** 19,991 sampled episodes
  (all catalog languages) each got one search a listener wanting that
  episode might type — a question, a topic, a name plus subject — written
  by `gpt-5.1` from the show, title, and description
  (`ml/datasets/llm_queries.py`). The category-name examples are gone.
- **Hard negatives.** For every query the top-50 from pretrained vector and
  lexical search, minus the positive and its own show
  (`ml/hard_negatives/mine.py`); each example trains against 4 of them, 2
  popularity-weighted episodes, and 2 random ones.
- **Checkpoint selection on the judged pool.** Every 200 steps the model
  re-ranks each relevance-set query's judged candidates; the best
  checkpoint by pool NDCG@10 is the one saved (step 200 of 576: 0.798,
  against 0.789 at the end and 0.832 for pretrained E5).
- **Distillation and a smaller head learning rate.** `1 - cos` against the
  frozen pretrained encoder on both towers (weight 0.5); heads at 1e-4.
- **Effective batch 32** (8 queries × 4 accumulation steps, 72 candidates
  per micro-batch).

Candidate-set accuracy no longer saturates (0.90–0.95 through the epoch):
the hard negatives made the training task informative. Its candidates were
pooled (top 50; 6,828 new pairs judged, 31,330 judgments in total) and the
systems below were scored on that pool:

| system | R@10 | R@50 | R@100 | Hit@10 | MRR | NDCG@10 | judged share | tail cov. | p50 / p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| vector-e5-small | **0.139** | 0.348 | 0.420 | 0.933 | **0.891** | **0.693** | 0.97 | 0.484 | 162 / 440 |
| two-tower-v3 | 0.131 | **0.486** | 0.544 | 0.906 | 0.819 | 0.632 | 1.00 | **0.574** | **17 / 39** |
| hybrid-rrf-two-tower-v3 | 0.127 | 0.461 | **0.670** | 0.956 | 0.877 | 0.652 | 0.98 | 0.531 | 257 / 6,135 |
| hybrid-rrf (E5 leg) | 0.135 | 0.460 | 0.547 | **0.967** | 0.875 | 0.684 | 0.98 | 0.485 | 510 / 6,272 |

NDCG@10 by query type:

| system | navigational | entity | exploratory | filtered |
|---|---|---|---|---|
| vector-e5-small | **0.942** | 0.589 | 0.641 | **0.595** |
| two-tower-v3 | 0.705 | 0.391 | **0.689** | 0.550 |
| hybrid-rrf-two-tower-v3 | 0.798 | 0.618 | 0.652 | 0.500 |
| hybrid-rrf (E5 leg) | 0.914 | **0.741** | 0.625 | 0.532 |

Recall@100 by relevant-item cohort:

| system | transcript | no transcript | head | tail |
|---|---|---|---|---|
| vector-e5-small | 0.260 | 0.441 | 0.419 | 0.399 |
| two-tower-v3 | 0.413 | 0.573 | 0.552 | 0.627 |
| hybrid-rrf-two-tower-v3 | **0.862** | **0.650** | **0.693** | **0.714** |
| hybrid-rrf (E5 leg) | 0.756 | 0.515 | 0.561 | 0.498 |

## What this establishes

- **Bootstrap labels alone lose to the pretrained baseline (run 2).**
  Trained on episode titles and category names against random negatives,
  two-tower-v2 reached NDCG@10 0.413 against 0.707 for E5. Candidate-set
  accuracy was 100% by step 200 of 6,358; the rest of the epoch could only
  sharpen title matching, and the model lost most on navigational and
  entity queries, where exact names decide relevance.
- **Real-shaped queries and hard negatives recover most of it (run 3).**
  two-tower-v3 reaches NDCG@10 0.632 against 0.693 for E5 on the same
  pool — still behind on top-10 ordering and MRR, and behind by a wide
  margin on navigational (0.705 vs 0.942) and entity (0.391 vs 0.589)
  queries, where the fine-tuned encoder has traded exact-name matching for
  topical generalization.
- **Where it already wins.** Deep recall, which is a first-stage
  retriever's job: Recall@50 0.486 vs 0.348 and Recall@100 0.544 vs
  0.420, with higher tail coverage (0.574 vs 0.484) and better recall on
  every relevant-item cohort. On exploratory queries it is the best
  standalone system (NDCG@10 0.689 vs 0.641), and its query tower answers
  in 17 ms at p50 against 162 ms for the sentence-transformers baseline.
- **Fusion favors it.** As the vector leg of hybrid RRF it delivers
  Recall@100 0.670 and the best recall on every cohort, including 0.862 on
  the transcript cohort; the hybrid's ordering (0.652) sits between its
  legs. The E5 hybrid re-run on this pool keeps the best top-10 ordering of any system (NDCG@10 0.684, hit rate 0.967) but trails the v3 hybrid on Recall@100 (0.547 vs 0.670) and on every cohort.
- **Two evaluation lessons paid for in full.** A training-serving skew in a
  single structured feature turned a working model into noise (run 1), and
  pooled judgments silently penalize any system that retrieves new
  episodes (run 2 before pooling). Both are now guarded: the featurizer is
  shared and fingerprinted, `judged_share_at_10` is reported by the
  harness, pooling a new system's candidates before scoring it is part of
  the evaluation procedure, and training validates on the judged pool
  rather than on its own candidate set.
- **Decision.** No fine-tuned model is promoted: the acceptance gate is
  beating the pretrained baseline on the primary top-10 metrics, and
  two-tower-v3 does not. The E5 index stays the vector leg of hybrid
  retrieval. two-tower-v3 is registered as the candidate retriever: it is
  the better candidate generator (deep recall, tail, exploratory intent),
  and the next experiments are (1) fusing it as a *third* leg alongside E5
  and lexical, so its recall adds to E5's name matching instead of
  replacing it, and (2) a second mining round using v3 itself as the
  miner, with name-bearing queries over-sampled to recover navigational
  and entity performance.

## Operational notes

- The runs above needed six relaunches before one completed, all for the
  same reason: on a 24 GB laptop shared with other applications, the
  training process was swapped whenever its footprint passed ~7 GB, and
  step time went from ~2.6 s to minutes. What made it finish, in order of
  impact: reading only the referenced snapshot rows through Arrow instead
  of materializing the whole episodes table in pandas (3.2 GB → 2.0 GB
  resident); freezing the token embeddings and lower encoder (steady-state
  GPU memory 2.9 GB → 1.5 GB); capping the MPS allocator; fixed-length
  padding so the allocator reuses buffers; checkpointing with optimizer
  state every 200 steps plus `--resume`, supervised by a script that kills
  and resumes a run that stops logging; and running the whole thing under
  `caffeinate` and outside macOS's background task class, which throttles
  timers and disk I/O for processes started from a terminal.
- Both embedding jobs now advance by episode id instead of re-scanning
  every already-indexed row per batch; the previous pattern was quadratic
  and had slowed the index build to 7 rows/s.
