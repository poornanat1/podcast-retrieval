"""Golden examples for training-serving parity.

    uv run python -m ml.serving.golden --bundle data/serving/bundles/<id>

Runs the training-side reference pipeline (PyTorch query towers,
``ml.ranking.candidates``, ``ml.ranking.features``, the LightGBM ranker) on
a fixed set of queries and writes ``golden.jsonl`` into the bundle
directory: for each query its model inputs (token ids, language id,
intents), both query vectors, the candidate set with every leg's rank, the
feature matrix, and the LightGBM scores. ``ml.serving.parity`` replays the
same queries through the production path and compares.

Queries: six per relevance-set query type (deterministic choice) plus edge
cases for tokenization and missing values: accented and non-English text,
a query longer than the tokenizer budget, punctuation only, and filters.
The freshness reference time is fixed and stored, so features are
reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

GOLDEN = "golden.jsonl"
PER_TYPE = 6
EDGE_CASES = [
    {"query_id": "edge-accents", "query_type": "edge",
     "query_text": "noticias de economía en español", "language": "es"},
    {"query_id": "edge-long", "query_type": "edge", "query_text": " ".join(
        ["how do central banks set interest rates and what happens to inflation"] * 8)},
    {"query_id": "edge-punctuation", "query_type": "edge", "query_text": "!!! ???"},
    {"query_id": "edge-filters", "query_type": "edge", "query_text": "startup advice",
     "max_duration_seconds": 1800, "published_after": "2025-06-01", "no_explicit": True},
    {"query_id": "edge-single-word", "query_type": "edge", "query_text": "stoicism"},
    {"query_id": "edge-mixed-case", "query_type": "edge", "query_text": "TED Talks DAILY"},
]
QUERY_FIELDS = ("query_text", "language", "max_duration_seconds", "published_after", "no_explicit")


def golden_queries(relevance: Path) -> list[dict]:
    queries = [json.loads(line) for line in (relevance / "queries.jsonl").read_text().splitlines()
               if line]
    chosen = []
    for qtype in sorted({q["query_type"] for q in queries}):
        of_type = sorted((q for q in queries if q["query_type"] == qtype),
                         key=lambda q: hashlib.sha256(q["query_id"].encode()).hexdigest())
        chosen.extend(of_type[:PER_TYPE])
    return chosen + EDGE_CASES


def model_query(q: dict) -> dict:
    return {k: q.get(k) for k in QUERY_FIELDS} | {"no_explicit": bool(q.get("no_explicit"))}


def _json_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, datetime):
            v = v.isoformat()
        elif isinstance(v, np.floating | np.integer):
            v = v.item()
        out[k] = v
    return out


def build(bundle: Path, relevance: Path, database_url: str, reference_time: datetime) -> Path:
    import psycopg
    import torch
    from sentence_transformers import SentenceTransformer

    from ml.embeddings.text import MODEL_ID
    from ml.features import Featurizer
    from ml.ranking.candidates import CandidateGenerator
    from ml.ranking.features import featurize
    from ml.ranking.lightgbm_runtime import LIGHTGBM_THREADS, load_lightgbm

    manifest = json.loads((bundle / "manifest.json").read_text())
    two_tower = Path("data/models") / manifest["sources"]["two_tower"]["artifact"]
    ranker_dir = Path("data/models") / manifest["sources"]["ranker"]["artifact"]
    booster = load_lightgbm().Booster(model_file=str(ranker_dir / "model.txt"))
    featurizer = Featurizer.load(two_tower)
    st = SentenceTransformer(MODEL_ID, device="cpu")

    out = bundle / GOLDEN
    with psycopg.connect(database_url) as conn, out.open("w") as f:
        generator = CandidateGenerator(conn, str(two_tower), manifest["candidate_depth"])
        for q in golden_queries(relevance):
            query = model_query(q)
            inputs = featurizer.query_batch([query])
            with torch.no_grad():
                tt_vec = generator.two_tower.model.encode_query(inputs)[0].numpy()
            e5_vec = st.encode(f"query: {query['query_text'].strip()}", normalize_embeddings=True)
            cands = generator.generate(query)
            x = featurize(cands, reference_time)
            scores = booster.predict(x, num_threads=LIGHTGBM_THREADS) if len(x) else np.zeros(0)
            ids = cands.episode_ids
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], ids[i]))
            f.write(json.dumps({
                "query_id": q["query_id"], "query_type": q["query_type"], "query": query,
                "reference_time": reference_time.isoformat(),
                "inputs": {k: v.tolist() for k, v in inputs.items()},
                "two_tower_vector": tt_vec.tolist(), "e5_vector": e5_vec.tolist(),
                "rows": [_json_row(r) for r in cands.rows],
                "features": x.tolist(), "scores": [float(s) for s in scores],
                "top10": [cands.rows[i]["episode_id"] for i in order[:10]],
            }) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    from ml.datasets.snapshot import DEFAULT_DATABASE_URL

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--relevance-dir", default="data/relevance")
    parser.add_argument("--reference-time", default="2026-10-21T00:00:00+00:00")
    parser.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL))
    args = parser.parse_args(argv)
    out = build(Path(args.bundle), Path(args.relevance_dir), args.database_url,
                datetime.fromisoformat(args.reference_time).astimezone(UTC))
    n = sum(1 for _ in out.open())
    print(json.dumps({"golden": str(out), "queries": n}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
