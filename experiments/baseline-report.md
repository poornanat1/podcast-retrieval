# Retrieval baseline report

Measured 2026-09-14 against the frozen evaluation corpus (795,533 episodes,
1,788 podcasts, polling stopped 2026-09-13). This report is the bar every
learned retrieval model must clear before it ships.

**Reproduce every number with one command:** `make eval-all` (configs in
`experiments/eval/`, one per system; runs log to MLflow experiment
`retrieval-eval`).

## Ground truth and methodology

- 180 queries (40 navigational, 20 entity, 85 exploratory, 35 filtered);
  179 have at least one relevant document and are scored.
- 16,139 graded judgments (0–3) over candidates pooled top-20 from all five
  systems; LLM-judged (`llm:gpt-5.1`) with human override at export.
  Grade ≥ 2 counts as relevant for binary metrics; NDCG uses full grades.
- Structured filters (language, duration, recency, explicit) are hard
  predicates applied inside every system before ranking.
- **Recall ceilings:** the median query has 30 relevant documents, so a
  perfect retriever scores Recall@10 = 0.440, Recall@50 = 0.996,
  Recall@100 = 1.000. Read Recall@10 against its ceiling, not against 1.0.

## Global results

| system | R@10 (max .44) | R@50 | R@100 | Hit@10 | MRR | NDCG@10 | tail cov. | p50 / p95 ms |
|---|---|---|---|---|---|---|---|---|
| vector-e5-small | **0.304** | 0.614 | 0.653 | 0.955 | **0.908** | **0.750** | 0.488 | 143 / 574 |
| hybrid-rrf | 0.285 | **0.900** | **0.946** | **0.978** | 0.884 | 0.731 | 0.486 | 461 / 6,475 |
| lexical-fts | 0.190 | 0.432 | 0.469 | 0.866 | 0.717 | 0.567 | 0.457 | 136 / 4,543 |
| popularity-category | 0.017 | 0.037 | 0.038 | 0.140 | 0.087 | 0.067 | 0.379 | 1 / 11 |
| popularity-global | 0.004 | 0.015 | 0.018 | 0.061 | 0.022 | 0.019 | 0.180 | 1 / 34 |

Latency is measured on a laptop under concurrent load and varies across
runs (standalone runs measured vector p95 at 364 ms, hybrid at 2.8 s);
treat relative differences as robust, absolute values as indicative.

## By query type (NDCG@10)

| system | entity | exploratory | filtered | navigational |
|---|---|---|---|---|
| vector-e5-small | 0.609 | **0.708** | **0.666** | **0.987** |
| hybrid-rrf | 0.767 | 0.682 | 0.583 | 0.951 |
| lexical-fts | **0.885** | 0.473 | 0.323 | 0.828 |
| popularity-category | 0.021 | 0.047 | 0.179 | 0.032 |
| popularity-global | 0.007 | 0.008 | 0.047 | 0.024 |

Semantic retrieval wins wherever meaning matters (exploratory +0.24,
filtered +0.34 over lexical); lexical keeps a clear edge on entity queries,
where exact name matching is the task. Hybrid sits between its parents on
each type — fusion hedges rather than dominates.

## By relevant-item cohort (recall over each partition of the relevant set)

| system | R@100 transcript | R@100 no-transcript | R@100 head | R@100 tail |
|---|---|---|---|---|
| vector-e5-small | 0.299 | 0.742 | 0.639 | 0.670 |
| hybrid-rrf | **0.984** | **0.935** | **0.953** | **0.940** |
| lexical-fts | 0.873 | 0.356 | 0.485 | 0.440 |

(108 queries have transcript-bearing relevant items; 175 have
transcript-less ones; head = top decile of podcasts by episode count.)

**This is the sharpest finding in the report.** The two retrievers have
complementary blind spots: lexical search finds transcript-bearing items it
can keyword-match into (0.873) but misses more than half of everything
else, while vector search — which embeds only titles and descriptions —
is nearly blind to items whose relevance lives in the transcript (0.299).
Hybrid covers both above 0.93. Consequences:

1. Hybrid RRF is the only viable candidate generator today.
2. Transcript-aware embeddings (bounded transcript representations in the
   episode tower) are the highest-leverage modeling improvement available.
3. Neither retriever shows a head/tail popularity bias worth correcting.

## By query language (NDCG@10)

171 of 179 scored queries are English; the non-English cohorts (de 2, es 3,
fr 2, pt 1) are far too small for conclusions — reported only to show the
multilingual path works at all: vector/hybrid retrieve es and pt queries
(0.54 / 0.91 NDCG) where lexical scores exactly 0, its stemming and content
coverage failing entirely. Growing the non-English query set is the cheap
fix for real language cohorts.

## The bar for learned retrieval

A fine-tuned two-tower model must, on this corpus and ground truth:

1. **Candidate generation:** Recall@100 ≥ 0.90 (hybrid's floor; its
   measured 0.946 is the target to beat toward the 1.0 ceiling).
2. **Ordering:** NDCG@10 ≥ 0.750 and MRR ≥ 0.908 (vector's marks).
3. **No cohort collapse:** transcript and no-transcript R@100 both ≥ 0.90
   (only hybrid manages this today); entity NDCG@10 ≥ 0.767 (hybrid's; the
   0.885 lexical mark is the stretch goal); tail recall within 5 points of
   head.
4. **Serving:** retrieval-stage p95 within the 500 ms budget on this
   hardware (vector meets it; hybrid's lexical tail currently does not).

## Caveats

- Judgments are pooled from these five systems' top-20: absolute recall
  flatters pool members, and a genuinely novel future system should have
  its candidates pooled and judged before comparison (the pooling tooling
  supports this).
- Ground truth is LLM-graded with rationales; human spot-audit agreement
  has not yet been measured and should be before trusting deltas smaller
  than a few points.
- One query ("actualites politiques francaises", filtered fr) has no
  relevant documents in the pool and is excluded from averages.
