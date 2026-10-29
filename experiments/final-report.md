# PodFind: final experiment report

What was built, what was measured, and what the measurements say, in the
order the work happened. Each section links the report that holds the
full method and tables; the numbers here are the headline ones. The
system runs at
[api-production-0ff59.up.railway.app](https://api-production-0ff59.up.railway.app).

## The problem and the data

Podcast episode search over a frozen catalog of 1,788 shows, 795,533
episodes and 87,300 publisher transcripts, judged against a 180-query
relevance set (navigational, entity, exploratory and filtered queries)
with 71,518 graded judgments pooled from every system compared, graded
by an LLM with human override ([data-card.md](../docs/data-card.md)).
Every training label is synthetic: LLM-written queries for sampled
episodes, show- and publisher-name queries, weak topic labels. The
relevance set is held out from all of it.

## Baselines first

[baseline-report.md](baseline-report.md). Pretrained `multilingual-e5-small`
vectors in pgvector were the strongest single baseline; lexical search was
best on entity queries and worst everywhere else; hybrid reciprocal-rank
fusion had by far the best deep recall.

| system | NDCG@10 | MRR | Recall@100 |
|---|---|---|---|
| lexical (full-text) | 0.567 | 0.717 | 0.469 |
| pretrained E5 | 0.750 | 0.908 | 0.653 |
| hybrid RRF | 0.731 | 0.884 | 0.946 |
| popularity (category) | 0.067 | 0.087 | 0.038 |

## Learned retrieval

[two-tower-first-run.md](two-tower-first-run.md),
[negative-sampling-report.md](negative-sampling-report.md),
[name-queries-report.md](name-queries-report.md),
[ablation-cold-start-report.md](ablation-cold-start-report.md). The first
fine-tuned two-tower run was invalidated by a training-serving feature
skew the parity tooling later made impossible; the second beat nothing;
the third, with natural-language training queries, lexical-mined hard
negatives (vector-mined ones were relevant 22% of the time in audit) and
pool-selected checkpoints, matched E5 on precision and beat it on recall
and tail coverage. Name queries fixed the entity-query weakness.

The champion, `two-tower-llm-names-…-3b1c13c6`
([model-card.md](../docs/model-card.md)), on the final evaluation:

| system | NDCG@10 | MRR | Recall@100 | tail coverage |
|---|---|---|---|---|
| lexical | 0.535 | 0.708 | 0.197 | 0.458 |
| pretrained E5 | 0.694 | 0.891 | 0.367 | 0.484 |
| **two-tower champion** | **0.730** | 0.878 | **0.519** | **0.507** |

Ablations: the model needs metadata text (ids alone score 0.054 NDCG@10);
a learned podcast-id embedding hurts entity queries (−0.098, interval
excludes zero); a transcript excerpt helps a little (+0.014, interval
includes zero) and is not worth the index cost. Cold-start slices show the
champion ahead of E5 on unseen shows (+0.122 Recall@100) and on episodes
published after the training snapshot (+0.236).

## Learned ranking

[ranker-report.md](ranker-report.md). A LightGBM LambdaRank ranker over
the union of the three legs' candidates, 31 features under one contract
shared by training and serving, query-grouped cross-validation:

| system | NDCG@10 | MAP@10 | Recall@100 |
|---|---|---|---|
| RRF over three legs | 0.766 | 0.782 | 0.475 |
| **ranker** | **0.794** | **0.817** | **0.514** |
| ranker − RRF (paired bootstrap) | +0.028 (+0.007 to +0.049) | +0.035 (+0.006 to +0.064) | +0.039 (+0.032 to +0.047) |

The ranker's gain comes mostly from the two-tower similarity gap to the
query's best candidate (35% of split gain) and the fusion score.

## Serving and parity

[serving-report.md](serving-report.md). The Go API runs query
understanding, three concurrent retrieval legs, candidate evidence and
ONNX reranking through a model server; promotion of a bundle is gated on
17 parity checks between the training-side and serving-side code. The
gate refused the first bundle for three real defects (float32 threshold
rounding in the ONNX ranker, a null-versus-NaN skew in a feature, cold
lexical timeouts), all fixed. The promoted bundle returns the same
candidates, features to 1e-6, and top 10 on every golden query.

## Voice

[voice-robustness-report.md](voice-robustness-report.md). On a synthetic
speech test set (ElevenLabs voices in six English accents and four other
languages, with noise, telephone, laptop-microphone, reverb and speed
degradations), Whisper small transcribes at 5.3% WER and 22% named-entity
error rate, and spoken queries lose 0.087 NDCG@10 (95% CI 0.064–0.113)
against the same queries typed. Half the loss is transcription errors on
names; the other half is casing, which lowercasing recovers.

## Deployment, monitoring and load

[deployment-report.md](deployment-report.md). One container image per
process on Railway from an infrastructure file; Postgres with pgvector on
a volume; Prometheus, Grafana with nine alert rules, and a scheduled
drift check that compares served leg outcomes, latencies, score and
source distributions and feature means with a recorded baseline. The
check and the alert were shown to catch a real fault (vector legs forced
to time out: 14 findings, alert in six minutes). Load tests: availability
100% and event persistence 100% met; search p95 681 ms at one client over
the internet against a 500 ms budget, which led to the lexical work
below.

## The lexical leg, revisited

The lexical leg timed out on most queries at its 300 ms limit because it
ranked every match with the full transcript text vector. A small ranking
vector over each transcript's opening (`rank_tsv`), per-source ranking cut
to the candidate depth, and a cap on matches ranked per source made the
leg finish in under 240 ms on the broadest queries warm. Measured online on
the relevance set against the local stack:

| lexical query | lexical timeouts | NDCG@10 | Recall@100 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| before | 112 of 180 | 0.897 | 0.458 | 403 | 1,007 |
| after | 72 of 180 | 0.898 | 0.476 | 419 | 868 |

Same precision (the ranker orders the same top ten), 0.019 more deep
recall from the lexical-only candidates that now arrive in time, and a
lower p95 on a laptop sharing its memory with the monitoring stack (the
absolute numbers here are below the serving report's 0.932 for the same
reason: the whole stack, Docker included, ran on one machine during this
measurement). The broadest queries ("news", "true crime") went from 19 s
and 0.6 s warm to 0.23 s and 0.16 s; cold, from 22 s and 8 s to 1.2 s and
1.8 s. Deployed, the search p95 went from 681 ms to 179 ms at one client
and from 915 ms to 359 ms at eight, over the internet: the 500 ms budget
the design set is met ([deployment-report.md](deployment-report.md)).

## More like this

`GET /v1/episodes/{id}/similar` recommends the nearest episodes to an
episode's own vector, diversified across shows. There are no labels for
episode-to-episode similarity, so the evaluation
(`ml.evaluation.similar`, {SIM_SEEDS} seed episodes sampled from the
catalog, ten results each) measures the list properties the design
asked for:

| index | intra-list diversity | novelty (other show) | shows per list | repetition | top-decile-show share (catalog {SIM_CATALOG_SHARE}) | similarity mean / min | p50 / p95 ms |
|---|---|---|---|---|---|---|---|
| two-tower (learned) | {TT_ILD} | {TT_NOV} | {TT_SHOWS} | {TT_REP} | {TT_TOP} | {TT_SIM} | {TT_LAT} |
| E5 (pretrained) | {E5_ILD} | {E5_NOV} | {E5_SHOWS} | {E5_REP} | {E5_TOP} | {E5_SIM} | {E5_LAT} |

{SIM_VERDICT}

## Reproducibility

`make reproduce` evaluates the five systems above on the relevance set
against the frozen catalog and checks NDCG@10, MRR and Recall@100 against
`experiments/eval/reproduce-expected.json` to within 0.01. Datasets are
versioned by content hash, model artifacts record their data, config and
code revision, the serving bundle is hash-named and parity-gated, and the
catalog is frozen; the data card lists what a fresh machine needs.

## What did not work, honestly

- The first two two-tower runs lost to the pretrained baseline; the third
  won on recall, not precision. The ranker is what beats E5 on every
  metric.
- Transcripts help lexical entity lookup and little else; the
  transcript-aware two-tower arm was not worth its index.
- The 500 ms search p95 was missed on the laptop and on the first
  deployment, and only met once the lexical leg's query plan was
  changed; the budget was set before anything was measured and held by
  luck of the fix rather than by design.
- Interaction events are collected but no model uses them yet; the
  personalized-retrieval evaluation stays future work.
- Relevance grades are LLM-made with human override, not human-made; the
  human agreement rate is not measured.

## Artifacts

| required | where |
|---|---|
| reproducible dataset pipeline and data card | `ml/datasets`, [data-card.md](../docs/data-card.md) |
| popularity, lexical and pretrained baselines | [baseline-report.md](baseline-report.md) |
| two-tower training | `ml/training`, [two-tower-first-run.md](two-tower-first-run.md) |
| hard-negative-mining experiment | [negative-sampling-report.md](negative-sampling-report.md) |
| learned-ranker comparison | [ranker-report.md](ranker-report.md) |
| offline evaluation with cohort metrics | `ml/evaluation`, every report's cohort tables |
| ablation study, cold-start analysis | [ablation-cold-start-report.md](ablation-cold-start-report.md) |
| voice-robustness analysis | [voice-robustness-report.md](voice-robustness-report.md) |
| model card | [model-card.md](../docs/model-card.md) |
| training-serving parity test | [serving-report.md](serving-report.md), `ml/serving/parity.py` |
| load-test report, drift demonstration | [deployment-report.md](deployment-report.md) |
| one-command reproducible experiment | `make reproduce` |
