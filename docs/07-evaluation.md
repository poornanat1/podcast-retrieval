# Evaluation

## Running it

```sh
make eval EVAL_CONFIG=experiments/eval/two-tower-v3.json   # one system
make eval-all                                              # every config in experiments/eval/
```

`ml/evaluation/run.py` loads the config, the relevance queries and qrels,
runs the system against the live catalog, and logs parameters, metrics,
and per-query results to MLflow experiment `retrieval-eval` (pass
`--no-mlflow` for ad-hoc runs). `ml/evaluation/compare.py` runs several
configs and prints a side-by-side table.

One config per system:

```json
{
  "system": "two-tower",
  "system_options": {
    "artifact": "data/models/two-tower-llm-v3-1.0.0-d52b7c3b104f-62d87868"
  },
  "ks": [10, 50, 100],
  "min_relevant_grade": 2,
  "relevance_dir": "data/relevance",
  "head_podcast_share": 0.1,
  "mlflow_experiment": "retrieval-eval"
}
```

`system` names a registered retrieval system (`lexical-fts`,
`popularity-global`, `popularity-category`, `vector-e5-small`, `two-tower`,
`hybrid-rrf`); `system_options` go to its constructor — the artifact for
`two-tower`, or `vector_artifact` to give `hybrid-rrf` a two-tower leg.

## Metrics

Computed per query by `ml/evaluation/harness.py` and averaged over queries
that have at least one relevant document (grade ≥ `min_relevant_grade`):

| metric | meaning |
|---|---|
| `recall_at_{k}` | share of a query's relevant episodes in the top k |
| `hit_rate_at_{k}` | 1 if any relevant episode is in the top k |
| `mrr` | 1 / rank of the first relevant episode |
| `ndcg_at_10` | graded ranking quality of the top 10, normalized by the ideal ordering |
| `judged_share_at_10` | share of the top 10 that has any judgment (pool coverage; see below) |
| `recall_at_{10,100}__{transcript,no_transcript,head,tail}` | recall restricted to each partition of the relevant set |
| `catalog_coverage` | distinct episodes retrieved across all queries / catalog size |
| `tail_coverage` | share of results from shows outside the top decile by episode count |
| `latency_ms_p50`, `latency_ms_p95` | wall time per `search` call, database included |

Results are also broken down by query type and by query language. The
head/tail split uses episode count per show as a popularity proxy until
real play data exists.

**Recall ceilings.** The median query has dozens of relevant documents, so
Recall@10 is bounded well below 1; read it against the ceiling reported in
the baseline report, not against 1.0.

**Judged share.** Judgments are pooled from named systems, so a system
whose results were never pooled scores near zero on `judged_share_at_10`
and its other numbers are meaningless until its candidates are pooled and
judged ([06-relevance.md](06-relevance.md)). Extending the pool also
raises the recall denominators for every system, so numbers are only
compared within one pool state.

## Reports

- [experiments/baseline-report.md](../experiments/baseline-report.md) —
  popularity, lexical, pretrained vector, and hybrid baselines on the
  frozen corpus, globally, by query type, and by cohort.
- [experiments/two-tower-first-run.md](../experiments/two-tower-first-run.md) —
  three fine-tuned runs against the baselines on the extended pool, with
  the diagnoses that shaped them.

## Reproducibility

Every number comes from a checked-in config, a named snapshot or catalog
state, a dataset version with a content hash, and an artifact whose name
embeds its training-config hash; runs record the git commit and whether
the tree was dirty. The catalog is held fixed while systems are compared
([09-catalog.md](09-catalog.md)).

## Judged-pool re-ranking

`ml/evaluation/pool_rerank.py` scores a model by ordering each query's
judged candidates with its own dot product — no index, a few minutes — and
reports `pool_ndcg_at_10` / `pool_mrr`. It is the validation metric the
training loop uses to pick checkpoints; on the current pool the pretrained
baseline scores 0.832 / 0.911.
