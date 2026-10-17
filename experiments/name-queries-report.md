# Name queries for two-tower training

A single, time-boxed retrain aimed at the one gap every earlier
fine-tuned model shared: queries that name a show or a person. Measured
2026-10-17 against the 797,216-episode catalog and the 180-query relevance
set. The negative-strategy comparison that motivated it is in
[negative-sampling-report.md](negative-sampling-report.md).

## Why name queries

The best earlier model (`random`: LLM-written queries, random negatives)
had no mined false negatives at all and still lost to pretrained E5 on
navigational (0.791 vs 0.942 NDCG@10) and entity (0.471 vs 0.584)
queries. The LLM-written training queries are almost all topical, so
fine-tuning wore down the exact-name matching the pretrained encoder
already had. Cleaning negatives could not fix that; the training data had
to contain the query shape.

## What changed

Relative to the `random` arm, everything else held fixed (one epoch,
effective batch 32, top 6 encoder layers trainable, age jitter, judged-pool
checkpoint selection):

| | `random` arm | this run |
|---|---|---|
| queries | 18.5k LLM-written | the same 18.5k + 5.9k show/publisher names |
| mined negatives | none | 2 per example, lexical-mined only |
| other negatives | 8 random | 6 random |
| distillation weight toward E5 | 0.5 | 1.0 |

- **Name queries** (`make name-queries`, `ml/datasets/name_queries.py`):
  each show's normalized title paired with up to four of its episodes, and
  its publisher's normalized name with up to two, 6,852 in total (4,699
  show, 2,153 publisher) over 2,106 distinct names. Every show whose title
  or publisher matches a relevance-set query text is excluded, so none of
  the 60 navigational and entity evaluation queries is trained on. A
  publisher query lists every show by that publisher, and none of their
  episodes is used as a negative.
- **Lexical-only hard negatives**: the audit found 3.4% of lexical-mined
  candidates relevant against 22.5% of vector-mined ones, so the builder's
  new `hard_sources` filter keeps only the former. Name queries were mined
  with lexical search alone; 3,508 of them had no candidate outside the
  named show and fall back to random negatives.
- Dataset `search-retrieval-llm-names-1.0.0-56e2c0708f7e` (24,348 train /
  1,087 validation / 1,408 test), config
  `experiments/datasets/search-llm-names-v1.json`; model
  `two-tower-llm-names-1.0.0-56e2c0708f7e-3b1c13c6`, config
  `experiments/training/two-tower-names-v1.json`.

Four things changed at once, so this run does not say how much each one
contributed. That was the point of a single time-boxed attempt; an
ablation is the follow-up if the attribution matters.

## Training

Judged-pool NDCG@10 at each validation pass (pretrained E5: 0.832):

| step | 200 | 400 | 600 | 761 (final) |
|---|---|---|---|---|
| pool NDCG@10 | 0.834 | **0.839** | 0.836 | 0.837 |

The first fine-tuned run to reach the pretrained model on this signal
(the `random` arm's best was 0.773, `mixed` 0.798). Step 400 was restored
as the final artifact.

## Relevance set

The new model's top-50 candidates were pooled and graded before scoring
(3,507 new pairs, 40,071 judgments in total). All rows below are on that
one pool; E5's and `random`'s numbers moved slightly from the previous
report because the recall denominators grew.

| system | NDCG@10 | MRR | Hit@10 | R@50 | R@100 | tail cov. |
|---|---|---|---|---|---|---|
| lexical | 0.535 | 0.708 | 0.861 | 0.178 | 0.211 | 0.458 |
| pretrained E5 | 0.694 | **0.891** | 0.933 | 0.289 | 0.378 | 0.484 |
| `random` arm | 0.678 | 0.833 | 0.933 | 0.400 | **0.533** | **0.542** |
| **name queries** | **0.731** | 0.878 | **0.944** | **0.420** | 0.519 | 0.507 |

By query type (NDCG@10):

| system | navigational | entity | exploratory | filtered |
|---|---|---|---|---|
| lexical | 0.808 | **0.796** | 0.439 | 0.306 |
| pretrained E5 | **0.942** | 0.584 | 0.636 | 0.611 |
| `random` arm | 0.791 | 0.471 | **0.724** | 0.558 |
| **name queries** | 0.908 | 0.750 | 0.656 | **0.697** |

Recall@100 by relevant-item cohort:

| system | transcript | no transcript | head | tail |
|---|---|---|---|---|
| lexical | **0.588** | 0.159 | 0.232 | 0.179 |
| pretrained E5 | 0.248 | 0.387 | 0.374 | 0.357 |
| **name queries** | 0.267 | **0.541** | **0.473** | **0.503** |

## Is the difference real?

Paired bootstrap over the 180 queries (`ml/evaluation/significance.py`,
10,000 resamples), name-query model minus baseline:

| comparison | metric | difference | 95% interval | P(not better) | wins / ties / losses |
|---|---|---|---|---|---|
| vs E5 | NDCG@10 | +0.037 | −0.006 to +0.080 | 0.042 | 79 / 47 / 54 |
| vs E5 | Recall@100 | +0.140 | +0.112 to +0.168 | 0.000 | 150 / 8 / 22 |
| vs E5, entity only | NDCG@10 | +0.166 | +0.001 to +0.343 | 0.024 | 7 / 10 / 3 |
| vs E5, navigational only | NDCG@10 | −0.034 | −0.143 to +0.076 | 0.727 | 6 / 29 / 5 |
| vs lexical | NDCG@10 | +0.196 | +0.141 to +0.251 | 0.000 | 117 / 31 / 32 |

The recall gain over E5 and every gain over lexical are well outside
noise. The top-10 gain over E5 is probable but not established: its
interval includes zero. The navigational gap that remains is within noise
on 40 queries.

## In-distribution test split

The `random` arm's test split (842 LLM-written queries, one shared pool of
10,149 episodes; name queries would be scored against their own show's
other episodes there, so they are not included):

| model | MRR | Recall@1 | Recall@10 | Recall@100 |
|---|---|---|---|---|
| pretrained E5 | 0.715 | 0.645 | 0.859 | 0.976 |
| `random` arm | **0.854** | **0.789** | **0.972** | 0.999 |
| name queries | 0.847 | 0.779 | 0.966 | 0.999 |

Adding name queries and lexical negatives cost almost nothing on topical
queries shaped like the training data.

## Acceptance gates and registration

Against the gate baseline (pretrained E5), every check passes: both
primary metrics (NDCG@10 0.731 vs 0.694, Recall@100 0.519 vs 0.378),
every query type within the 0.05 tolerance (navigational is the closest,
−0.034), and every relevant-item cohort. Registered in MLflow as
`podfind-two-tower` version 2 with the alias `champion`; version 1 (the
`random` arm) keeps `candidate`.

Against lexical, both primary metrics pass by wide margins; the one cohort
that fails is recall on transcript-bearing relevant items (0.267 vs
0.588). This model, like pretrained E5 (0.248), does not read
transcripts; lexical search does. That gap is what hybrid fusion with the
lexical leg is for, and it is the input a transcript-aware episode tower
would address.

## Caveats

- **Checkpoint selection used the judged pool**, which is built from the
  same 180 queries. The four checkpoints scored within 0.005 of each
  other, so the selection effect is small, but it is not zero; every
  fine-tuned arm shares it.
- **The top-10 gain over E5 is not significant at 95%.** Promotion follows
  the gates as defined, and the gates do not require significance.
- **The judge is an LLM** with no human audit yet
  ([docs/06-relevance.md](../docs/06-relevance.md)); differences smaller
  than its disagreement rate with a human grader are not trustworthy.
- **Latency is not compared.** All two-tower rows run the same query tower;
  their p50 differences are host load.

## Operational note

The machine slept for about six hours mid-run. The training supervisor
then killed only the `uv run` wrapper, so the original trainer survived,
woke up, and ran alongside a resumed copy. The resumed copy was stopped
before it saved anything; the original ran to completion without
interruption and wrote the final artifact, so the numbers above come from
one continuous run.

## Reproduce

```sh
make name-queries
make mine-negatives QUERIES_FILE=data/snapshots/<id>/name-queries.jsonl MINE_SYSTEMS=lexical-fts
make dataset CONFIG=experiments/datasets/search-llm-names-v1.json
make train TRAIN_CONFIG=experiments/training/two-tower-names-v1.json
make index ARTIFACT=data/models/<artifact>
make relevance-pool ...   # then relevance-judge, relevance-export, pool-export
make eval EVAL_CONFIG=experiments/eval/two-tower-names-v1.json
make significance CANDIDATE_RUN=<eval run id> BASELINE_RUN=<eval run id>
make gates COMPARE_JSON=... ARTIFACT=data/models/<artifact>
```
