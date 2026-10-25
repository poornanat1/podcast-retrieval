# Model card — PodFind two-tower retriever

**Status: draft.** Describes the registered champion,
`podfind-two-tower` version 2 in the MLflow model registry. Numbers come
from the reports linked under each section; update this card when the
champion changes.

## Model

| | |
|---|---|
| artifact | `data/models/two-tower-llm-names-1.0.0-56e2c0708f7e-3b1c13c6` |
| architecture | two towers sharing one `intfloat/multilingual-e5-small` encoder (12 layers, 384-dim, mean pooling), with residual heads over structured features; cosine similarity, temperature 0.05 |
| query tower | query text + query language and intent context |
| episode tower | metadata text (episode title and description, show title, publisher, categories) + language, duration bucket, publication-age bucket, categories; no transcript, no learned ids |
| parameters | 118.5M total, 11.6M trained (top 6 encoder layers and the heads) |
| embedding | 384-dim, L2-normalized, indexed with pgvector HNSW under `two-tower:<artifact name>` |
| feature contract | version 1, `069be232…`, stored with the artifact; training and indexing use the same featurizer |

## Intended use

First-stage candidate retrieval for podcast episode search: given a typed
query, return the top few hundred episodes for a later fusion or ranking
stage, with the query's hard filters (language, duration, date, explicit)
applied in the index scan.

Not intended for: final ranking on its own (see failure modes), personal
recommendations (it has no user input), audio or spoken queries (it has
never seen speech-recognition output), or exact-name lookup without a
lexical or exact-match leg alongside it.

## Training data

Dataset `search-retrieval-llm-names-1.0.0-56e2c0708f7e`, built from
snapshot `baf91376ae98` (540,875 episodes, 1,314 shows, captured
2026-08-28). All labels are synthetic; see [data-card.md](data-card.md).

| | rows (train) |
|---|---|
| LLM-written natural-language queries, one per sampled episode (`gpt-5.1`) | 18,451 |
| show-name and publisher-name queries paired with episodes of that show | 5,897 |

Negatives per example: 2 lexical-mined candidates (a false-negative audit
found vector-mined ones relevant 22% of the time, lexical-mined ones 3%)
and 6 random catalog episodes, plus in-batch negatives. No show whose
title or publisher is a relevance-set query appears in the name queries.

One epoch, effective batch 32, encoder learning rate 2e-5 and head 1e-4,
distillation toward pretrained E5 at weight 1.0, age jitter, checkpoint
selected by NDCG@10 re-ranking the judged pool (step 400 of 761).
[name-queries-report.md](../experiments/name-queries-report.md).

The artifact records git commit `af3aeb0` with a dirty tree: it was
trained from the code committed in `1a2ff9b`.

## Evaluation

180-query relevance set, LLM-graded with human override (no human grades
yet), pool of 71,518 judgments in which every result in every compared
system's top 10 is judged, on the 795,533-episode frozen catalog, with the
vector search fixed on 2026-10-21 (exact re-sorting, `ef_search = 1000`;
[05-embeddings.md](05-embeddings.md#search-settings-and-index-quality)).

| | pretrained E5 | lexical | this model |
|---|---|---|---|
| NDCG@10 | 0.720 | 0.538 | 0.758 |
| MAP@10 | 0.730 | 0.507 | 0.770 |
| Recall@100 | 0.369 | 0.261 | 0.421 |
| MRR | 0.905 | 0.713 | 0.913 |

Against E5 (paired bootstrap): NDCG@10 +0.038 (95% interval +0.005 to
+0.071), Recall@100 +0.052 (+0.030 to +0.072). Against lexical: both
primary metrics far ahead; transcript-bearing relevant items far behind
(Recall@100 0.19 vs 0.61).

By query type (NDCG@10): navigational 0.958 (E5 0.987), entity 0.805
(lexical 0.830), exploratory 0.684 (E5 0.654), filtered 0.681 (E5 0.605).

Earlier numbers in this card's history (0.730 NDCG@10 on a 50,856-judgment
pool) were measured with the old vector search, which missed about 17% of
the true top 10; the model is unchanged.

**Cold start.** Recall@100 gains over E5 hold for shows with no training
positives (+0.04 on the current pool) and are largest for episodes added
after the training snapshot (+0.16); see
[ablation-cold-start-report.md](../experiments/ablation-cold-start-report.md).

**As a candidate source.** The learned ranker
([ranker-report.md](../experiments/ranker-report.md)) reorders this
model's top 50 together with lexical and E5 candidates and reaches 0.794
NDCG@10.

**Ablations.** Content features carry the model: a learned-show-id-only
tower scores 0.054 NDCG@10. Adding learned ids to metadata hurts
publisher-name queries. Adding a transcript excerpt is the best single
model (0.744) but not distinguishable from this one on 180 queries.

## Serving

Served as the query tower and candidate index of model bundle
`42df3779d7cf` (with the E5 query encoder and the learned ranker), exported
to ONNX. On the 30 golden parity queries the ONNX query tower matches
PyTorch to 4e-7, and the production pipeline returns the same candidates,
features (to 1e-6), and top 10 as the training-side code; see
[experiments/serving-report.md](../experiments/serving-report.md).

## Known failure modes

1. Topical queries that look like names are mapped to one show sharing a
   word (*short daily news briefing* → *Short Wave*).
2. Some show names collapse onto a different show (*ted talks daily* →
   *DK's Daily Shot*). Two of the three examples first reported (*today
   explained*, *football ramble*) were lost by the old ANN search, not
   the model, and are right with the fixed search.
3. Within the right show, episode ranking is weak (*the new statesman*).
4. Relevance found only in transcript text is out of reach (Recall@100
   0.27 vs lexical's 0.58).
5. Person-name publisher queries trail exact lexical matching.
6. Non-English quality is unmeasured (8 of 180 queries).

Mitigation today: serve it as a candidate source with lexical and E5
candidates, reordered by the learned ranker, which fixes the remaining
show-collapse example and most person-name misses
([ranker-report.md](../experiments/ranker-report.md)).

## Ethical and data considerations

- Trained on publisher metadata and LLM-written queries only; no user
  data, no audio.
- Head shows are over-represented in the catalog and the relevance set;
  tail-show recall is reported separately and is not worse than head-show
  recall for this model.
- All evaluation labels are LLM grades. Differences smaller than the
  judge's disagreement with human graders, once measured, should not be
  trusted.
