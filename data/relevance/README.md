# Relevance set

The human-reviewed ground truth every retrieval model is evaluated against.

## Files

| File | Tracked | Produced by |
| --- | --- | --- |
| `queries.jsonl` | yes | `make relevance-queries` (deterministic from a catalog snapshot) |
| `tasks.jsonl` | no (machine-local) | `make relevance-pool` (top-K lexical candidates per query) |
| `judgments.jsonl` | yes | `make relevance-review` (interactive human grading) |
| `qrels.txt` | yes | `make relevance-export` (TREC format for the eval harness) |

## Query set

~200 queries in four types: **navigational** (find a specific show, generated
from the largest catalog shows), **entity** (host/publisher names, generated),
**exploratory** (curated open-ended topical queries), and **filtered**
(curated queries carrying structured constraints — language, max duration,
publish date, explicit content — that retrieval enforces as hard predicates,
so pooling applies them too).

## Pooling

Judgments are collected over pooled candidates: each retrieval system
contributes its top K results per query. The pool has been extended in
stages — lexical full-text search, the popularity baselines, pretrained
vector and hybrid retrieval (top 20 each), and each fine-tuned two-tower
model (top 50, so its hybrid's candidates are covered too) — into
the same tasks file so systems are compared on shared judgments. Unjudged
documents are treated as irrelevant by the eval harness, as is standard for
pooled evaluation, which is exactly why every new system must be pooled
before its numbers mean anything: the harness reports `judged_share_at_10`,
the share of a system's top-10 that has any judgment, and a system scoring
far below the pooled systems on it is being penalized for novelty, not
relevance. Learned systems are pooled with their artifact via
`--system-options` (see `ml/relevance/pool.py`).

## Grading scale

| Grade | Meaning |
| --- | --- |
| 0 | irrelevant |
| 1 | marginal — topic-adjacent, would not satisfy the query |
| 2 | relevant — satisfies the query |
| 3 | highly relevant — near-perfect answer |

Grade the episode against the query's *intent*. Structured constraints were
already applied during pooling, so grade topical intent only. Judgments are
append-only and carry the judge's name and timestamp.

## Judges: human and LLM

Judgments come from two kinds of judge, always distinguishable:

- **Human** (`judge` = a person's name) — collected via `make relevance-review`.
- **LLM** (`judge` = `llm:<model>`) — collected via `make relevance-judge`,
  which grades the pooled candidates with the rubric above. Machine
  judgments are never recorded as human.

At export, one grade per (query, episode) pair applies, and a human judgment
always overrides an LLM one — so auditing is cheap: re-grade any pair in
`make relevance-review` and your grade wins. The export summary reports the
human/LLM composition so downstream reports can state exactly how much of
the ground truth is machine-labeled. Spot-auditing a random sample of LLM
judgments (and recording the agreement rate) is strongly recommended before
trusting evaluation deltas smaller than the observed disagreement.
