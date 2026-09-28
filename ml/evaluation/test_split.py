"""Score models on a dataset's untouched test split with one shared pool.

    uv run python -m ml.evaluation.test_split --dataset data/datasets/<name>-<version> \
        --artifacts data/models/<a>,data/models/<b> [--pretrained] [--device mps]

Each strategy's dataset carries its own negatives, so the candidate-set
diagnostics of one run are not comparable to another's. This ranks every
test query's positive against the *same* pool for every model: all test
positives plus the top mined candidates of every test query (from the
dataset's negatives input), so no model is scored against negatives
chosen by its own recipe. Reports MRR and recall at 1, 10, and 100 per
model. ``--pretrained`` adds the untrained (pretrained E5) model built
from the first artifact's contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from ml.datasets import snapshot as snapshot_mod
from ml.datasets.validate import load_examples
from ml.features import Featurizer
from ml.models.two_tower import TwoTowerModel
from ml.training.data import load_episode_rows


def build_pool(
    test_df, negatives: dict[int, list[dict]], hard_pool: int
) -> tuple[list[dict], list[int]]:
    """Queries (text + positive) and the shared candidate id list."""
    queries = [
        {"query_text": str(r.query_text), "positive": int(r.positive_episode_id)}
        for r in test_df.itertuples(index=False)
    ]
    pool = {q["positive"] for q in queries}
    for q in queries:
        pool.update(int(h["episode_id"]) for h in negatives.get(q["positive"], [])[:hard_pool])
    return queries, sorted(pool)


@torch.no_grad()
def score(model, featurizer, queries: list[dict], rows: dict[int, dict], pool: list[int],
          device: str, ks=(1, 10, 100), batch_size: int = 128) -> dict[str, float]:
    model = model.to(device).eval()
    present = [e for e in pool if e in rows]
    position = {e: i for i, e in enumerate(present)}
    episodes = [rows[e] for e in present]
    now = datetime.now(UTC)
    chunks = []
    for i in range(0, len(episodes), batch_size):
        batch = featurizer.episode_batch(episodes[i:i + batch_size], now)
        chunks.append(model.encode_episode({k: v.to(device) for k, v in batch.items()}).cpu())
    vectors = torch.cat(chunks)
    ranks = []
    for start in range(0, len(queries), batch_size):
        chunk = queries[start:start + batch_size]
        q = model.encode_query({k: v.to(device) for k, v in
                                featurizer.query_batch(chunk).items()}).cpu()
        sims = q @ vectors.T
        for i, query in enumerate(chunk):
            if query["positive"] not in position:
                continue
            target = sims[i, position[query["positive"]]]
            ranks.append(int((sims[i] > target).sum().item()) + 1)
    n = len(ranks)
    out = {"queries": n, "pool": len(present),
           "mrr": round(sum(1 / r for r in ranks) / n, 4)}
    for k in ks:
        out[f"recall_at_{k}"] = round(sum(1 for r in ranks if r <= k) / n, 4)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--snapshot", default="data/snapshots")
    parser.add_argument("--artifacts", required=True, help="comma-separated artifact dirs")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--hard-pool", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset)
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    snapshot_dir = snapshot_mod.resolve(args.snapshot)
    negatives: dict[int, list[dict]] = {}
    for name in manifest.get("inputs", {}):
        if name.endswith("-negatives.jsonl"):
            lines = (snapshot_dir / name).read_text().splitlines()
            negatives = {
                int(rec["episode_id"]): rec["hard"]
                for rec in (json.loads(line) for line in lines if line)
            }
    test_df = load_examples(dataset_dir / "test.parquet")
    queries, pool = build_pool(test_df, negatives, args.hard_pool)
    rows = load_episode_rows(snapshot_dir, set(pool))

    results = {}
    artifacts = [Path(a) for a in args.artifacts.split(",") if a]
    for artifact in artifacts:
        results[artifact.name] = score(TwoTowerModel.load(artifact), Featurizer.load(artifact),
                                       queries, rows, pool, args.device)
    if args.pretrained:
        featurizer = Featurizer.load(artifacts[0])
        base = TwoTowerModel.from_pretrained_text(
            featurizer.contract, TwoTowerModel.load(artifacts[0]).config
        )
        results["pretrained-e5"] = score(base, featurizer, queries, rows, pool, args.device)
    summary = {"dataset_version": manifest["dataset_version"], "results": results}
    print(json.dumps(summary, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
