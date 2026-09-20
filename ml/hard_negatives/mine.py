"""Mine hard negatives for training queries with the baseline retrievers.

    uv run python -m ml.hard_negatives.mine --queries <snapshot>/llm-queries-<model>.jsonl \
        [--systems vector-e5-small,lexical-fts] [--depth 50] [--shard 0/4]

For every (query, positive episode) pair, retrieve the top ``depth``
candidates from each baseline system, drop the positive and anything from
the positive's own show (the label is one episode, not the show, and
sibling episodes are more often relevant than not), and keep the rest with
their source and rank. The output (``<queries file stem>-negatives.jsonl``,
one line per query with ``hard`` candidates) is a raw dataset input: the
builder's ``llm_query`` generator mixes these with popularity-weighted and
random negatives and records the file's hash in the manifest.

Random negatives are almost never topically close, so a model trained on
them only learns to separate unrelated episodes; these are the candidates
the baselines actually confuse with the positive.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psycopg

from ml.datasets.snapshot import DEFAULT_DATABASE_URL
from ml.retrieval import build_system


def mine_query(systems: list, query_text: str, positive: int, positive_podcast: int,
               episode_podcast: dict[int, int], depth: int) -> list[dict]:
    """Ranked candidates from every system, minus the positive and its show."""
    seen: dict[int, dict] = {}
    for system in systems:
        for rank, episode_id in enumerate(system.search({"query_text": query_text}, depth), 1):
            if episode_id == positive or episode_podcast.get(episode_id) == positive_podcast:
                continue
            entry = seen.setdefault(episode_id, {"episode_id": episode_id, "sources": []})
            entry["sources"].append({"system": system.name, "rank": rank})
    return sorted(seen.values(), key=lambda e: min(s["rank"] for s in e["sources"]))


def mine(queries: list[dict], out: Path, systems: list, episode_podcast: dict[int, int],
         depth: int, log_every: int = 500) -> dict:
    """Append one line per query, resuming past episode ids already in ``out``."""
    done: set[int] = set()
    if out.exists():
        done = {json.loads(line)["episode_id"] for line in out.read_text().splitlines() if line}
    pending = [q for q in queries if int(q["episode_id"]) not in done]
    stats = {"queries": len(queries), "already_done": len(done), "written": 0, "empty": 0}
    started = time.time()
    with out.open("a") as f:
        for i, q in enumerate(pending, 1):
            positive = int(q["episode_id"])
            hard = mine_query(systems, q["query_text"], positive,
                              episode_podcast.get(positive, -1), episode_podcast, depth)
            if not hard:
                stats["empty"] += 1
            f.write(json.dumps({"episode_id": positive, "hard": hard}, sort_keys=True) + "\n")
            f.flush()
            stats["written"] += 1
            if i % log_every == 0:
                rate = i / max(time.time() - started, 1)
                print(f"mined {i}/{len(pending)} ({rate:.1f}/s)", flush=True)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queries", required=True,
                        help="llm-queries jsonl (episode_id, query_text)")
    parser.add_argument("--systems", default="vector-e5-small,lexical-fts")
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--shard", default="0/1", help="k/n: process the k-th of n slices")
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    )
    args = parser.parse_args(argv)

    queries = [json.loads(line) for line in Path(args.queries).read_text().splitlines() if line]
    k, n = (int(x) for x in args.shard.split("/"))
    queries = queries[k::n]
    suffix = f"-shard{k}" if n > 1 else ""
    out = Path(args.out) if args.out else Path(args.queries).with_name(
        Path(args.queries).stem + f"-negatives{suffix}.jsonl"
    )

    with psycopg.connect(args.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, podcast_id FROM episodes")
            episode_podcast = dict(cur.fetchall())
        systems = [build_system(name.strip(), conn) for name in args.systems.split(",")]
        stats = mine(queries, out, systems, episode_podcast, args.depth)
    stats["out"] = str(out)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
