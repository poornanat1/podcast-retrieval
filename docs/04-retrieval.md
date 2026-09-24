# Retrieval

Every retrieval system implements `search(query, k) -> [episode ids]` over
a relevance-set query row (`query_text` plus optional `language`,
`max_duration_seconds`, `published_after`, `no_explicit`) and applies the
structured filters as hard SQL predicates before ranking. The registry is
`ml/retrieval/__init__.py`; eval configs select a system by name.

## Systems

### `lexical-fts` — PostgreSQL full-text search

The query is parsed with `websearch_to_tsquery` in the text-search
configuration for its language (English, Spanish, French, German,
Portuguese, Italian, Dutch, else `simple`). Candidates come from three
independently indexed generated `tsvector` columns — episode title and
description (weights A/B), transcript text (weight D), and show title,
publisher, categories, and description (A/B/C) — and are ranked by the sum
of their `ts_rank` scores, with the show's score at half weight. Hard
filters apply in the same query. Source: `ml/retrieval/lexical.py`, schema
in `migrations/0008_full_text_search.sql`.

### `vector-e5-small` — pretrained embedding search

`intfloat/multilingual-e5-small` encodes `query: <text>` in-process; the
`episode_embeddings` HNSW index for that model is scanned by cosine
distance with the filters as predicates, using pgvector's iterative scan so
filtered searches still fill `k`. Episodes were encoded from
`passage: <show>. <title>. <description[:1000]>`. Source:
`ml/retrieval/vector.py`, [05-embeddings.md](05-embeddings.md).

### `two-tower` — fine-tuned embedding search

Same scan over the index built for a two-tower artifact (`make index`),
with the query encoded by the artifact's query tower. Selected in an eval
config with `system_options.artifact`. Each artifact's vectors sit under
their own `model` key, so trained models and the pretrained baseline are
compared on the same catalog. [08-models.md](08-models.md)

### `hybrid-rrf` — reciprocal-rank fusion

Lexical and vector rankings (depth 200 each) are merged by
`score = Σ 1 / (60 + rank)`; ties break toward items present in both lists.
The vector leg is E5 unless `system_options.vector_artifact` names a
two-tower artifact (reported as `hybrid-rrf-two-tower`). Source:
`ml/retrieval/hybrid.py`.

### `popularity-global`, `popularity-category`

Baselines that ignore the query text: most popular shows' episodes, or
those of shows in categories the query names. Source:
`ml/retrieval/popularity.py`.

## Measured behavior

Numbers, cohorts, and query-type breakdowns are in
[experiments/baseline-report.md](../experiments/baseline-report.md) and
[experiments/two-tower-first-run.md](../experiments/two-tower-first-run.md).
The shape of the results: vector search wins on exploratory and filtered
queries, lexical search wins on exact names, hybrid fusion has by far the
best deep recall, and the fine-tuned model beats E5 on Recall@100 and
exploratory queries but not on top-10 ordering.

Latency per query on a laptop, as measured by the harness (which includes
the database round trip): lexical p50 ~140 ms with a long tail of several
seconds on broad queries, E5 vector ~80–160 ms, two-tower ~20 ms (a plain
encoder forward pass on CPU), hybrid the sum of its legs.

## Not built

Query understanding (turning a free-text or spoken request into filters),
a learned reranker over the fused candidates, and any online serving path.
