"""Audit mined hard negatives for false negatives.

    uv run python -m ml.hard_negatives.audit --queries <snapshot>/llm-queries-<model>.jsonl \
        [--sample-queries 60] [--per-query 5] [--model gpt-5.1]

A training label is one positive episode per query, so a "hard negative"
retrieved by the baselines may in fact satisfy the query — training then
pushes a relevant episode away. This samples (query, mined negative)
pairs, grades them with the relevance judge's rubric, and reports the
share graded relevant (grade >= 2) overall, by mining rank, and by which
baseline surfaced them. Results are appended next to the queries file as
``<stem>-audit.jsonl`` and the summary printed as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path


def sample_pairs(
    queries: list[dict], negatives: dict[int, list[dict]], sample_queries: int,
    per_query: int, hard_pool: int, seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    with_hard = [q for q in queries if negatives.get(int(q["episode_id"]))]
    picked = rng.sample(with_hard, min(sample_queries, len(with_hard)))
    pairs = []
    for q in picked:
        candidates = negatives[int(q["episode_id"])][:hard_pool]
        for cand in rng.sample(candidates, min(per_query, len(candidates))):
            pairs.append({
                "query_episode_id": int(q["episode_id"]),
                "query_text": q["query_text"],
                "episode_id": int(cand["episode_id"]),
                "rank": min(s["rank"] for s in cand["sources"]),
                "sources": sorted({s["system"] for s in cand["sources"]}),
            })
    return pairs


def audit(pairs: list[dict], rows: dict[int, dict], grader) -> list[dict]:
    """Grade every pair (one grader call per query); pairs whose episode
    row is missing are skipped."""
    by_query: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        if pair["episode_id"] in rows:
            by_query[pair["query_text"]].append(pair)
    graded = []
    for query_text, group in by_query.items():
        tasks = []
        for pair in group:
            row = rows[pair["episode_id"]]
            tasks.append({
                "query_id": f"audit:{pair['query_episode_id']}", "query_text": query_text,
                "query_type": "exploratory", "episode_id": pair["episode_id"],
                "podcast_title": row.get("podcast_title", ""),
                "episode_title": row.get("episode_title", row.get("title", "")),
                "description": row.get("description", ""),
                "duration_seconds": row.get("duration_seconds"),
                "published_at": row.get("published_at", ""),
            })
        try:
            grades = {g.episode_id: g for g in grader(tasks)}
        except Exception as exc:  # noqa: BLE001 - one bad query must not kill the audit
            print(f"  query {query_text[:40]!r} failed: {exc}", file=sys.stderr)
            continue
        for pair in group:
            grade = grades.get(pair["episode_id"])
            if grade is None:
                continue
            graded.append({**pair, "grade": grade.grade, "rationale": grade.rationale[:300]})
    return graded


def summarize(graded: list[dict], min_relevant_grade: int = 2) -> dict:
    def share(items: list[dict]) -> dict:
        n = len(items)
        return {"pairs": n, "relevant_share": round(
            sum(1 for g in items if g["grade"] >= min_relevant_grade) / n, 3) if n else None}

    def bucket(rank: int) -> str:
        return "1-5" if rank <= 5 else "6-10" if rank <= 10 else "11-20" if rank <= 20 else "21+"

    by_rank: dict[str, list[dict]] = defaultdict(list)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for g in graded:
        by_rank[bucket(g["rank"])].append(g)
        by_source["+".join(g["sources"])].append(g)
    return {
        "overall": share(graded),
        "by_rank": {k: share(v) for k, v in sorted(by_rank.items())},
        "by_source": {k: share(v) for k, v in sorted(by_source.items())},
        "grade_counts": {str(k): sum(1 for g in graded if g["grade"] == k) for k in range(4)},
    }


def main(argv: list[str] | None = None) -> int:
    from ml.datasets.snapshot import DEFAULT_DATABASE_URL

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queries", required=True)
    parser.add_argument("--negatives", default="", help="default: <queries stem>-negatives.jsonl")
    parser.add_argument("--sample-queries", type=int, default=60)
    parser.add_argument("--per-query", type=int, default=5)
    parser.add_argument("--hard-pool", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    )
    args = parser.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()

    queries_path = Path(args.queries)
    negatives_path = Path(args.negatives) if args.negatives else queries_path.with_name(
        queries_path.stem + "-negatives.jsonl"
    )
    queries = [json.loads(line) for line in queries_path.read_text().splitlines() if line]
    negatives = {
        int(rec["episode_id"]): rec["hard"]
        for rec in (json.loads(line) for line in negatives_path.read_text().splitlines() if line)
    }
    pairs = sample_pairs(queries, negatives, args.sample_queries, args.per_query,
                         args.hard_pool, args.seed)

    import psycopg
    from openai import OpenAI

    from ml.relevance.llm_judge import grade_batch
    from ml.retrieval import fetch_episode_rows

    with psycopg.connect(args.database_url) as conn:
        rows = {r["episode_id"]: r for r in fetch_episode_rows(
            conn, sorted({p["episode_id"] for p in pairs})
        )}
    client = OpenAI()
    graded = audit(pairs, rows, lambda tasks: grade_batch(client, args.model, tasks))

    out = queries_path.with_name(queries_path.stem + "-audit.jsonl")
    with out.open("a") as f:
        for g in graded:
            f.write(json.dumps(g, sort_keys=True, default=str) + "\n")
    summary = {**summarize(graded), "sampled_pairs": len(pairs), "graded": len(graded),
               "judge": f"llm:{args.model}", "out": str(out)}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
