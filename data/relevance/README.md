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
contributes its top K (default 20) results per query. Today that is lexical
full-text search; as embedding and two-tower retrieval land, their candidates
are pooled into the same tasks file so systems are compared on shared
judgments. Unjudged documents are treated as irrelevant by the eval harness,
as is standard for pooled evaluation.

## Grading scale

| Grade | Meaning |
| --- | --- |
| 0 | irrelevant |
| 1 | marginal — topic-adjacent, would not satisfy the query |
| 2 | relevant — satisfies the query |
| 3 | highly relevant — near-perfect answer |

Grade the episode against the query's *intent*, including its structured
constraints. Judgments are append-only, one per (query, episode) pair, and
carry the judge's name and timestamp. Only human judgments enter
`qrels.txt`; nothing here is machine-labeled.
