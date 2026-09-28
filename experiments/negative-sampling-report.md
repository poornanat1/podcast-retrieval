# Negative sampling for two-tower training

A controlled comparison of negative strategies, everything else held at
the settings of the third fine-tuned run
([two-tower-first-run.md](two-tower-first-run.md)): LLM-written queries,
one epoch, effective batch 32, top 6 encoder layers trainable,
distillation toward pretrained E5, age jitter, and checkpoint selection
on the judged pool. Measured 2026-09-27 against the 797,216-episode
catalog and the 180-query relevance set.

## Strategies

Each training example pairs one LLM-written query with its episode and
eight negatives. The strategies differ only in where the eight come from:

| arm | negatives | dataset | artifact |
|---|---|---|---|
| `random` | 8 random catalog episodes | `search-retrieval-llm-random-1.0.0-d161eb39b7c0` | `two-tower-llm-random-negatives-…-10cbbe81` |
| `hard` | 8 mined (seeded sample from the top-20 candidates of pretrained vector + lexical search, minus the positive's show) | `search-retrieval-llm-hard-1.0.0-4c4ae8ff8489` | `two-tower-llm-hard-negatives-…-4d257f67` |
| `mixed` | 4 mined + 2 popularity-weighted + 2 random | `search-retrieval-llm-1.0.0-d52b7c3b104f` | `two-tower-llm-v3-…-62d87868` (run 3) |

In-batch negatives (the other queries' positives in a micro-batch) apply
to every arm. Configs: `experiments/training/two-tower-{random,hard}-negatives-v1.json`,
`two-tower-v3.json`; datasets `experiments/datasets/search-llm-*.json`.

## How many "hard negatives" are false negatives?

The label is one positive per query, so a mined candidate can be a
relevant episode that the label simply never mentions. 292 sampled
(query, mined negative) pairs from 60 training queries were graded with
the relevance judge's rubric (`make audit-negatives`, `llm:gpt-5.1`):

| slice | pairs | graded relevant (≥ 2) |
|---|---|---|
| all | 292 | **20.9%** |
| mining rank 1–5 | 64 | 25.0% |
| rank 6–10 | 84 | 19.0% |
| rank 11–20 | 101 | 20.8% |
| rank 21+ | 43 | 18.6% |
| surfaced by vector search only | 262 | 22.5% |
| surfaced by lexical search only | 29 | 3.4% |

Grade distribution: 141 irrelevant, 90 marginal, 39 relevant, 22 highly
relevant. One in five mined negatives is an episode the query is about,
and the share is highest exactly where the negatives are hardest. Nearly
all of them come from the vector baseline; lexical near-misses are
genuinely about something else. The `hard` arm therefore trains against
~1.7 relevant episodes per query on average; `mixed` against ~0.8.

## Held-out test split (in-distribution)

Every test query's positive ranked against one shared pool of 10,149
episodes (all 842 test positives plus the top-20 mined candidates of each
test query), so no model is scored against negatives chosen by its own
recipe (`make test-split`):

| model | MRR | Recall@1 | Recall@10 | Recall@100 |
|---|---|---|---|---|
| pretrained E5 | 0.715 | 0.645 | 0.859 | 0.976 |
| `random` | 0.855 | 0.789 | 0.972 | 0.999 |
| `hard` | **0.905** | **0.853** | **0.988** | 0.999 |
| `mixed` (run 3) | 0.888 | 0.825 | **0.988** | 0.999 |

Every arm beats the pretrained model by a wide margin on queries shaped
like its training data, and the ordering is `hard` > `mixed` > `random`.
Read this with the audit in mind: the shared pool is built from the same
mined candidates, so the one in five that are actually relevant count as
wrong answers here too. A model that learns to push them down looks best
on this split precisely because the split shares the label's blind spot.
The relevance set below, judged independently, is the check on that.

## Relevance set (the real target)

All arms' candidates were pooled (top 50) and graded before scoring, on
one judgment pool:

| system | R@10 | R@50 | R@100 | Hit@10 | MRR | NDCG@10 | judged share | tail cov. | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| pretrained E5 | **0.116** | 0.313 | 0.393 | **0.933** | **0.891** | **0.693** | 0.97 | 0.484 | 142 |
| `random` | 0.115 | **0.446** | **0.556** | **0.933** | 0.833 | 0.679 | 1.00 | 0.542 | 137 |
| `hard` | 0.101 | 0.381 | 0.486 | 0.878 | 0.768 | 0.583 | 1.00 | **0.576** | 18 |
| `mixed` (run 3) | 0.109 | 0.408 | 0.540 | 0.906 | 0.819 | 0.628 | 1.00 | 0.574 | 184 |

36,564 judgments after pooling both new arms (5,234 new pairs). Recall
denominators grew again, so E5's numbers differ slightly from earlier
reports; only rows in this table are comparable to each other. (Latency
differences between the two-tower rows are noise from the host's load;
all three run the same query tower.)

By query type (NDCG@10):

| system | navigational | entity | exploratory | filtered |
|---|---|---|---|---|
| pretrained E5 | **0.942** | **0.581** | 0.637 | **0.607** |
| `random` | 0.791 | 0.474 | **0.724** | 0.558 |
| `hard` | 0.660 | 0.333 | 0.643 | 0.491 |
| `mixed` (run 3) | 0.705 | 0.389 | 0.681 | 0.551 |

Recall@100 by relevant-item cohort:

| system | transcript | no transcript | head | tail |
|---|---|---|---|---|
| pretrained E5 | 0.243 | 0.405 | 0.390 | 0.360 |
| `random` | 0.415 | **0.580** | **0.571** | 0.580 |
| `hard` | 0.398 | 0.510 | 0.492 | 0.547 |
| `mixed` (run 3) | **0.429** | 0.560 | 0.546 | **0.608** |

## Training behavior

Judged-pool NDCG@10 at each validation pass (the checkpoint-selection
signal; pretrained E5 scores 0.832 on it) and the candidate-set accuracy
that each arm's own negatives produce:

| arm | step 200 | step 400 | final (576) | best | candidate-set accuracy at steps 10 / 100 / 300 / 500 |
|---|---|---|---|---|---|
| `random` | 0.773 | 0.766 | 0.764 | **0.773** (step 200) | 0.97 / 0.99 / 1.00 / 1.00 |
| `hard` | 0.681 | 0.685 | 0.678 | **0.685** (step 400) | 0.78 / 0.91 / 0.90 / 0.96 |
| `mixed` | 0.798 | 0.791 | 0.789 | **0.798** (step 200) | 0.80 / 0.91 / 0.94 / 0.97 |

The candidate-set numbers say how hard each arm's training task is
(random negatives are separable within a few steps; mined ones never
are), not how good the model is. On the pool every arm peaks early and
drifts for the rest of the epoch, and the best checkpoint is the one
saved.

## Findings

- **On the real target the ordering is the reverse of the in-distribution
  one.** Test split: `hard` > `mixed` > `random`. Relevance set: `random`
  (NDCG@10 0.679) > `mixed` (0.628) > `hard` (0.583). The more mined
  negatives an arm trained against, the better it scored on a pool that
  shares the label's blind spot and the worse it scored on independently
  judged ground truth. The audit explains the mechanism: one in five mined
  negatives is a relevant episode, so hard-negative training teaches the
  model that relevant episodes are wrong, and the damage lands exactly on
  the query types where the false negatives concentrate — names and
  navigation, where vector search surfaces sibling episodes of the right
  show.
- **Random negatives are not the weak baseline they looked like.** With
  LLM-written queries the `random` arm is within 0.014 NDCG@10 of the
  pretrained model, equals its hit rate (0.933), beats it on Recall@50
  (+0.13), Recall@100 (+0.16), tail coverage, every relevant-item cohort,
  and exploratory queries (0.724 vs 0.637, the best of any system). The
  earlier conclusion that random negatives were the limiting factor
  (run 2) was confounded with the title-copying queries of that run; the
  query distribution mattered far more than the negative distribution.
- **Mined negatives need cleaning before they help.** The `hard` arm is
  the only arm that loses on hit rate and every query type but
  exploratory. Two fixes follow directly from the audit: drop mined
  candidates from the positive's own show more aggressively (already
  done) *and* from shows the LLM query names; and grade mined negatives
  with the judge before use, keeping only grade 0–1 (a few thousand LLM
  calls per dataset, the same cost as pooling). Distillation toward E5
  (weight 0.5 in every arm) did not prevent the damage.
- **Early stopping on the judged pool was decisive.** Every arm peaked at
  step 200 or 400 and drifted afterwards; without pool-based checkpoint
  selection the final checkpoints would have been 0.009–0.010 worse.
- **What each arm is for.** `random` is the strongest candidate generator
  measured so far: E5-level top-10 quality on topical queries with far
  deeper recall. It is still behind on names (navigational 0.791 vs
  0.942, entity 0.474 vs 0.581), the same gap as every fine-tuned run, and
  the same fusion remedy applies.

## Acceptance gates

`make gates` (`ml/registry/register.py`) checks a candidate against the
baseline on the same comparison: primary metrics (`ndcg_at_10`,
`recall_at_100`) must be at least the baseline's, and no query type
(NDCG@10) or relevant-item cohort (Recall@100) may fall more than 0.05
below it.

| arm | primary NDCG@10 | primary Recall@100 | cohort failures | verdict |
|---|---|---|---|---|
| `random` | 0.679 vs 0.693 — fail | 0.556 vs 0.393 — pass | entity (−0.11), navigational (−0.15) | **not promoted** |
| `mixed` | 0.628 vs 0.693 — fail | 0.540 vs 0.393 — pass | entity, filtered, navigational | not promoted |
| `hard` | 0.583 vs 0.693 — fail | 0.486 vs 0.393 — pass | entity, filtered, navigational | not promoted |

The best arm, `random`, is registered in MLflow as `podfind-two-tower`
version 1 with the alias `candidate` and every gate check recorded on the
version; the pretrained E5 index remains the vector leg of hybrid
retrieval. Promotion to `champion` requires passing all gates on one
comparison run.

## Reproduce

```sh
make dataset CONFIG=experiments/datasets/search-llm-random-negatives-v1.json
make train TRAIN_CONFIG=experiments/training/two-tower-random-negatives-v1.json
make index ARTIFACT=data/models/<artifact>
make relevance-pool POOL_SYSTEMS=two-tower ...   # then relevance-judge, relevance-export
make eval EVAL_CONFIG=experiments/eval/two-tower-random-negatives-v1.json
make test-split DATASET_DIR=... ARTIFACTS=...
make gates COMPARE_JSON=... ARTIFACT=...
```
