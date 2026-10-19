# Ablations, cold-start slices, and failure modes

What the current champion's quality comes from, whether it holds for
episodes and shows the model never trained on, and where it fails.
Measured 2026-10-18 against the 797,216-episode catalog and the 180-query
relevance set, one comparison on one judgment pool (50,856 judgments,
after pooling and grading the top 50 of each new arm). The champion and
its training data are described in
[name-queries-report.md](name-queries-report.md).

## Arms

Four episode towers, everything else identical: the same dataset
(`search-retrieval-llm-names-1.0.0-56e2c0708f7e`), negatives, one epoch,
effective batch 32, distillation toward pretrained E5 at weight 1.0, and
checkpoint selection on the judged pool.

| arm | episode tower reads | artifact | config |
|---|---|---|---|
| ids only | a learned podcast-id embedding; no text, no structured features | `two-tower-ablation-ids-…-b69c5da4` | `two-tower-ablation-ids-v1.json` |
| metadata (champion) | title, description, show, publisher, categories text + language, duration, age, categories | `two-tower-llm-names-…-3b1c13c6` | `two-tower-names-v1.json` |
| metadata + ids | the champion's inputs + the podcast-id embedding | `two-tower-ablation-ids-metadata-…-f94e0d70` | `two-tower-ablation-ids-metadata-v1.json` |
| metadata + transcript | the champion's inputs + a 2,000-character transcript excerpt | `two-tower-ablation-transcript-…-115d3866` | `two-tower-ablation-transcript-v1.json` |

The query tower is the same in every arm. The transcript arm reads a
snapshot derived from the training snapshot (`990134867a2f`: identical
rows, with current transcript excerpts added), so its dataset is
byte-for-byte the champion's. Only 2,078 of the 24,348 training positives
have a transcript, so that arm has little to learn the transcript input
from.

"Learned ids" here means ids learned from synthetic training pairs. There
is no interaction data yet, so this measures whether a show identity
learned from the training set helps, not whether collaborative signal
from real users would.

The distillation teacher is pretrained E5 in every arm. Before these runs
it was built from the student's own config, which would have made the
ids-only teacher a random id embedding. That was fixed first.

## Results

| system | NDCG@10 | MRR | Hit@10 | R@50 | R@100 | tail cov. |
|---|---|---|---|---|---|---|
| lexical | 0.535 | 0.708 | 0.861 | 0.166 | 0.197 | 0.458 |
| pretrained E5 | 0.694 | 0.891 | 0.933 | 0.277 | 0.367 | 0.484 |
| ids only | 0.054 | 0.069 | 0.106 | 0.017 | 0.018 | 0.648 |
| **metadata (champion)** | 0.730 | 0.878 | 0.944 | 0.389 | 0.519 | 0.507 |
| metadata + ids | 0.723 | 0.861 | 0.911 | 0.384 | 0.511 | 0.525 |
| metadata + transcript | **0.744** | **0.892** | **0.944** | **0.398** | **0.529** | 0.496 |

Judged-pool NDCG@10 during training (E5 ≈ 0.83):

| arm | step 200 | 400 | 600 | 761 | selected |
|---|---|---|---|---|---|
| ids only | 0.426 | 0.446 | 0.454 | 0.464 | 761 |
| metadata (champion) | 0.834 | 0.839 | 0.836 | 0.837 | 400 |
| metadata + ids | 0.833 | 0.835 | 0.829 | 0.831 | 400 |
| metadata + transcript | 0.833 | 0.833 | 0.831 | 0.833 | 200 |

### Paired bootstrap against the champion

180 queries, 10,000 resamples (`make significance`):

| arm | metric | difference | 95% interval | wins / ties / losses |
|---|---|---|---|---|
| metadata + transcript | NDCG@10 | +0.014 | −0.009 to +0.040 | 61 / 63 / 56 |
| metadata + transcript | Recall@100 | +0.010 | −0.006 to +0.027 | 83 / 33 / 64 |
| metadata + ids | NDCG@10 | −0.008 | −0.041 to +0.025 | 61 / 66 / 53 |
| metadata + ids, entity queries | NDCG@10 | −0.098 | −0.233 to −0.004 | 1 / 13 / 6 |
| metadata + ids, entity queries | Recall@100 | −0.064 | −0.132 to −0.008 | 5 / 7 / 8 |

And against pretrained E5, the transcript arm is the first fine-tuned
model whose top-10 gain is outside noise: NDCG@10 +0.051 (95% interval
+0.012 to +0.090; 85 wins, 46 losses). The champion's own gain is +0.037
(−0.006 to +0.080) on this pool, as before.

### What the ablations say

- **Content carries all of the weight.** With only a learned show id, the
  model scores 0.054 NDCG@10 and 0.018 Recall@100. It barely works even
  for shows it trained on (Recall@100 0.031 on relevant episodes that were
  training positives), because one vector per show cannot tell its
  episodes apart, and unseen shows get untrained vectors (0.004).
- **Learned ids add nothing on top of content and hurt entity queries.**
  Overall the difference is within noise, but publisher-name queries lose
  0.10 NDCG@10 and 0.06 Recall@100, both outside noise. The id embedding
  gives the model a cheaper way to separate shows than reading the name,
  and entity queries are exactly the ones that need the name.
- **Transcripts help a little, and not where expected.** The transcript
  arm is the best single model, but its lead over the champion is within
  noise. Its gain is on navigational queries (0.967 vs 0.908), not on
  transcript-bearing relevant items, whose Recall@100 did not move (0.249
  vs 0.271; lexical search reaches 0.583). With 2,078 transcript-bearing
  positives, the transcript pass had little to learn from; a dataset that
  samples positives from transcript-bearing episodes is the way to test
  the input properly.
- **Single runs.** Each arm was trained once. The failure analysis below
  shows that individual name queries flip between perfect and zero from
  run to run, so differences of one or two hundredths between arms are
  inside run-to-run variance. A multi-seed repeat is needed before the
  transcript arm replaces the champion.

## Cold-start slices

Relevant items partitioned by the training evidence behind them
(`--cold-start-dataset`, `ml/evaluation/harness.py` `cold_start_cohorts`):

- **seen**: the episode was a training positive (24,122 episodes);
- **known show**: a new episode of a show that had training positives
  (518,044);
- **unseen show**: its show had no training positive (255,050);
- **after snapshot**: the episode entered the catalog after the training
  snapshot (256,341; overlaps the other slices).

Recall@100 by slice (queries with at least one relevant item in the slice
in parentheses):

| system | seen (143) | known show (180) | unseen show (127) | after snapshot (166) |
|---|---|---|---|---|
| lexical | 0.193 | 0.198 | 0.179 | 0.188 |
| pretrained E5 | 0.410 | 0.354 | 0.361 | 0.286 |
| ids only | 0.031 | 0.021 | 0.004 | 0.005 |
| **metadata (champion)** | **0.547** | 0.501 | **0.483** | **0.522** |
| metadata + ids | 0.524 | 0.488 | 0.475 | 0.486 |
| metadata + transcript | 0.554 | **0.510** | 0.472 | 0.478 |

Gain over pretrained E5, by slice:

| system | seen | known show | unseen show | after snapshot |
|---|---|---|---|---|
| metadata (champion) | +0.137 | +0.147 | +0.122 | +0.236 |
| metadata + transcript | +0.144 | +0.156 | +0.111 | +0.192 |

The champion's advantage holds where training evidence is absent: shows
it never trained on get nearly the same gain as episodes it trained on
(+0.12 vs +0.14), and the largest gain is on episodes that did not exist
when the training snapshot was taken. The model learned how queries
relate to episode text, not which episodes answer which queries.

## Full cohort report

`ml/evaluation/cohort_report.py`; ▼ marks a value more than 0.05 below
pretrained E5 or lexical on a cohort of at least 10 queries. Query
language is not reportable yet: 172 of 180 queries are English, and the
other four languages have one to three queries each.

NDCG@10 by query type:

| system | entity (20) | exploratory (85) | filtered (35) | navigational (40) |
|---|---|---|---|---|
| E5 | 0.584 | 0.636 | 0.613 | 0.942 |
| lexical | 0.796 | 0.439 | 0.306 | 0.808 |
| metadata (champion) | 0.750 | 0.655 | 0.697 | 0.908 |
| ids only | 0.100 ▼ | 0.008 ▼ | 0.141 ▼ | 0.054 ▼ |
| metadata + ids | 0.652 ▼ | 0.665 | 0.692 | 0.907 |
| metadata + transcript | 0.736 ▼ | 0.665 | 0.686 | 0.967 |

Recall@100 by relevant-item cohort:

| system | transcript (116) | no transcript (179) | head show (176) | tail show (134) |
|---|---|---|---|---|
| E5 | 0.247 | 0.372 | 0.360 | 0.355 |
| lexical | 0.583 | 0.148 | 0.214 | 0.173 |
| metadata (champion) | 0.271 ▼ | 0.542 | 0.472 | 0.508 |
| ids only | 0.011 ▼ | 0.016 ▼ | 0.028 ▼ | 0.004 ▼ |
| metadata + ids | 0.278 ▼ | 0.531 | 0.471 | 0.491 |
| metadata + transcript | 0.249 ▼ | 0.562 | 0.501 | 0.494 |

Regressions flagged for the metadata arms (ids only regresses everywhere):

- every metadata arm vs lexical on transcript-bearing relevant items
  (0.25–0.28 vs 0.58);
- metadata + ids vs lexical on entity queries (0.652 vs 0.796);
- metadata + transcript vs lexical on entity queries (0.736 vs 0.796).

No metadata arm regresses beyond tolerance against pretrained E5 on any
cohort, and all three pass the acceptance gates against it on this
comparison.

## Known failure modes

From the queries where the champion trails pretrained E5 or lexical by
the most (17 queries trail E5 by more than 0.2 NDCG@10; 28 lead it by
more than 0.2):

1. **Topical queries read as show names.** Training on show-name queries
   taught the model that a short, name-like query means "episodes of one
   show", and it over-applies it. *short daily news briefing* (under 20
   minutes) returns *Short Wave* (0.10 vs E5's 1.00); *is there life on
   europa* returns *Living The Red Life* (0.03 vs 0.88). One salient word
   decides the show. Every fine-tuned arm shares these two.
2. **Show collapse on unseen names.** The relevance set's navigational
   shows were held out of training, and the model sometimes maps the name
   wholesale to a different show sharing a word: *today explained* returns
   only *Sportsday*, *ted talks daily* only *DK's Daily Shot*, *football
   ramble* only *First Things First* (each 0.00 vs E5's 1.00). Which
   queries collapse varies by run: the transcript and metadata + ids arms,
   trained on the same data, get those three right and collapse on others
   (*the ramsey show*, *the dan bongino show*, *kara berry* for metadata +
   ids). A model that can collapse a top-40 show name to zero is a serving
   risk; an exact show-title match in a later ranking stage would catch
   it.
3. **Weak ranking inside the right show.** Name queries label every
   episode of the show as relevant, so the model learns show membership
   and little else. *the new statesman* finds the right show (*The
   Politics Show*) but not the episodes about the magazine itself (0.22 vs
   E5's 0.86).
4. **Transcript-only relevance.** Relevant episodes that lexical search
   finds through their transcript text are out of reach (Recall@100 0.27
   vs 0.58), including for the transcript arm, which reads only a 2,000-
   character excerpt.
5. **Entity queries still trail lexical** (0.750 vs 0.796), mostly
   person-name publishers lexical matches exactly (*vince ferrara*, *jim
   metzner*, *michael rand* score 0 for the champion, 0.8–1.0 for
   lexical).
6. **Non-English queries are unmeasured**: eight queries in total.

Items 1, 2, 4 and 5 are what hybrid fusion with lexical search and a
learned ranker with exact title and show-match features are for; item 3
needs episode-level labels within a show rather than show membership.

## Decision

The champion stays registered. The transcript arm is the strongest single
model and the first with a top-10 gain over E5 outside noise, but its lead
over the champion is not, and per-query run-to-run variance is large. It
is the first candidate for a multi-seed comparison, with a dataset that
samples more transcript-bearing positives. Learned show ids are dropped
until real interaction data exists.

## Reproduce

```sh
uv run python -m ml.datasets.snapshot --with-transcripts-for data/snapshots/baf91376ae98
make pool-export
make train TRAIN_CONFIG=experiments/training/two-tower-ablation-{ids,ids-metadata,transcript}-v1.json
make index ARTIFACT=data/models/<artifact>      # each arm
make relevance-pool ...; make relevance-judge; make relevance-export; make pool-export
uv run python -m ml.evaluation.compare experiments/eval/vector-e5small-v1.json \
    experiments/eval/lexical-v1.json experiments/eval/two-tower-names-v1.json \
    experiments/eval/two-tower-ablation-*-v1.json \
    --cold-start-dataset data/datasets/search-retrieval-llm-names-1.0.0-56e2c0708f7e \
    --json-out compare.json
uv run python -m ml.evaluation.cohort_report --summaries compare.json
make significance CANDIDATE_RUN=<run> BASELINE_RUN=<run>
```
