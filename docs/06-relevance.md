# Relevance set

The graded ground truth every retrieval system is scored against. Files
and the grading rubric are documented with the data in
[data/relevance/README.md](../data/relevance/README.md); this page is the
process.

## Queries

180 queries in four types, generated deterministically from a catalog
snapshot (`make relevance-queries`, `ml/relevance/queries.py`):

| type | count | example | what it tests |
|---|---|---|---|
| navigational | 40 | `history hit` | finding a specific show |
| entity | 20 | `joe rogan` | a host or publisher name |
| exploratory | 85 | `practical uses of artificial intelligence` | topical intent |
| filtered | 35 | topic + language / duration / date / explicit constraints | hard predicates plus intent |

Query ids are stable hashes of the query, so judgments survive rebuilds.

## Pooling

`make relevance-pool POOL_SYSTEMS=<a,b,...>` (`ml/relevance/pool.py`)
runs each named system on every query and writes the union of their top-K
results as judgment tasks, recording which systems surfaced each pair and
at what rank. Learned systems take their artifact through
`--system-options`. The pool has been extended in stages: lexical,
popularity baselines, pretrained vector and hybrid (top 20 each), and each
fine-tuned two-tower model (top 50).

Pooling is what makes numbers comparable: the harness treats an unjudged
result as irrelevant, so a system whose candidates were never pooled is
penalized for retrieving anything new. The harness reports
`judged_share_at_10` for exactly this reason, and no system's numbers are
quoted until its candidates have been pooled and judged.

## Grading

Grades are 0–3 (irrelevant, marginal, relevant, highly relevant) against
the query's *intent*; structured constraints were already enforced during
pooling. Binary metrics count grade ≥ 2 as relevant; NDCG uses the full
scale.

Two kinds of judge, always distinguishable in `judgments.jsonl`:

- **LLM** (`judge = llm:<model>`, currently `gpt-5.1`) via
  `make relevance-judge` (`ml/relevance/llm_judge.py`): one call per
  query grading all of its pending candidates with the rubric, low
  reasoning effort, structured output. Append-only and resumable; the
  tasks file can be sharded across processes.
- **Human** (`judge = <name>`) via `make relevance-review`
  (`ml/relevance/review.py`), an interactive grader.

`make relevance-export` (`ml/relevance/export.py`) validates the judgment
file against its contract and writes `qrels.txt` in TREC format; where
both judges graded a pair, the human grade wins. The export summary
reports the human/LLM composition.

## Current state

36,564 judgments over 180 queries, all LLM-graded so far (no human
overrides yet). Spot-auditing a random sample against a human grader, and
recording the agreement rate, is the outstanding piece of hygiene before
trusting metric differences smaller than that disagreement.

## Judged pool as a validation signal

`data/relevance/judged-episodes.parquet` (`make pool-export`) holds the
catalog rows of every judged episode so that a model can be scored by
re-ranking each query's judged candidates without an index. Training uses
this every validation pass; see [08-models.md](08-models.md).
