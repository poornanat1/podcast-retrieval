"""Judged-pool re-ranking: a minutes-long proxy for the relevance-set metrics.

Scoring a model on the full catalog needs a fresh vector index (an hour
or more), so training cannot validate on it. Re-ranking only each query's
*judged* candidates with the model's own dot product needs no index and
tracks the same NDCG@10 / MRR the harness reports (on the extended pool
the pretrained E5 baseline scores 0.832 / 0.911 and the first fine-tuned
model 0.587 / 0.745, mirroring their full-catalog gap), so training
validates on it every evaluation pass and keeps its best checkpoint.

    uv run python -m ml.evaluation.pool_rerank --export   # judged rows -> parquet
    uv run python -m ml.evaluation.pool_rerank --artifact data/models/<run>

``--export`` writes ``data/relevance/judged-episodes.parquet`` (the catalog
rows of every judged episode) from the database, so training can run with
the database stopped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch

from ml.datasets.snapshot import DEFAULT_DATABASE_URL, TRANSCRIPT_EXCERPT_CHARS
from ml.evaluation.run import load_qrels, load_queries

JUDGED_EPISODES = "judged-episodes.parquet"

ROWS_SQL = """
    SELECT e.id, e.podcast_id, e.title, e.description, e.language,
           e.duration_seconds, e.published_at, e.content_hash,
           p.title AS podcast_title, p.publisher, p.categories,
           coalesce(left(t.content, %(excerpt_chars)s), '') AS transcript_excerpt
    FROM episodes e JOIN podcasts p ON p.id = e.podcast_id
    LEFT JOIN transcripts t ON t.episode_id = e.id
    WHERE e.id = ANY(%(ids)s)
    ORDER BY e.id
"""


def export_judged_episodes(relevance_dir: Path, database_url: str) -> Path:
    import psycopg

    qrels = load_qrels(relevance_dir / "qrels.txt")
    ids = sorted({e for grades in qrels.values() for e in grades})
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(ROWS_SQL, {"ids": ids, "excerpt_chars": TRANSCRIPT_EXCERPT_CHARS})
        columns = [d.name for d in cur.description]
        frame = pd.DataFrame(cur.fetchall(), columns=columns)
    frame["categories"] = frame["categories"].map(lambda c: list(c) if c is not None else None)
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    out = relevance_dir / JUDGED_EPISODES
    frame.to_parquet(out, index=False)
    return out


def load_pool(
    relevance_dir: str | Path,
) -> tuple[list[dict], dict[str, dict[int, int]], list[dict]]:
    """Queries, qrels, and judged episode rows; rows come from the exported
    parquet (``--export``)."""
    relevance_dir = Path(relevance_dir)
    queries = load_queries(relevance_dir / "queries.jsonl")
    qrels = load_qrels(relevance_dir / "qrels.txt")
    frame = pd.read_parquet(relevance_dir / JUDGED_EPISODES)
    rows = []
    for row in frame.to_dict("records"):
        cats = row.get("categories")
        row["categories"] = None if cats is None or isinstance(cats, float) else list(cats)
        rows.append(row)
    return queries, qrels, rows


def _ndcg(order: list[int], grades: dict[int, int], k: int = 10) -> float:
    dcg = sum(grades[e] / math.log2(r + 2) for r, e in enumerate(order[:k]))
    top = sorted(grades.values(), reverse=True)[:k]
    ideal = sum(g / math.log2(r + 2) for r, g in enumerate(top))
    return dcg / ideal if ideal else 0.0


@torch.no_grad()
def evaluate_pool(
    model, featurizer, queries: list[dict], qrels: dict[str, dict[int, int]],
    rows: list[dict], device: str = "cpu", min_relevant_grade: int = 2,
    reference_time: datetime | None = None, batch_size: int = 128,
) -> dict[str, float]:
    """NDCG@10 and MRR of each query's judged pool ordered by the model."""
    was_training = model.training
    model.eval()
    reference_time = reference_time or datetime.now(UTC)
    position = {int(r["id"]): i for i, r in enumerate(rows)}
    episode_vectors = torch.cat([
        model.encode_episode({
            k: v.to(device)
            for k, v in featurizer.episode_batch(rows[i:i + batch_size], reference_time).items()
        }).cpu()
        for i in range(0, len(rows), batch_size)
    ])
    query_vectors = model.encode_query(
        {k: v.to(device) for k, v in featurizer.query_batch(queries).items()}
    ).cpu()

    ndcg = mrr = 0.0
    n = 0
    for qi, query in enumerate(queries):
        grades = qrels.get(query["query_id"], {})
        candidates = [e for e in grades if e in position]
        if not any(grades[e] >= min_relevant_grade for e in candidates):
            continue
        scores = query_vectors[qi] @ episode_vectors[[position[e] for e in candidates]].T
        order = [candidates[i] for i in scores.argsort(descending=True).tolist()]
        ndcg += _ndcg(order, grades)
        mrr += next(
            (1 / (r + 1) for r, e in enumerate(order) if grades[e] >= min_relevant_grade), 0.0
        )
        n += 1
    if was_training:
        model.train()
    if n == 0:
        return {}
    return {"pool_ndcg_at_10": round(ndcg / n, 4), "pool_mrr": round(mrr / n, 4), "pool_queries": n}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--relevance-dir", default="data/relevance")
    parser.add_argument("--export", action="store_true", help="export judged episode rows")
    parser.add_argument("--artifact", default="", help="model artifact to score")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    )
    args = parser.parse_args(argv)

    relevance_dir = Path(args.relevance_dir)
    if args.export:
        out = export_judged_episodes(relevance_dir, args.database_url)
        print(f"judged episode rows written to {out}")
    if args.artifact:
        from ml.features import Featurizer
        from ml.models.two_tower import TwoTowerModel

        queries, qrels, rows = load_pool(relevance_dir)
        model = TwoTowerModel.load(args.artifact).to(args.device)
        featurizer = Featurizer.load(args.artifact)
        print(json.dumps(evaluate_pool(model, featurizer, queries, qrels, rows, args.device)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
