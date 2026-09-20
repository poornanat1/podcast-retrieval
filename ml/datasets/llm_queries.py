"""Generate natural-language search queries for catalog episodes with an LLM.

    uv run python -m ml.datasets.llm_queries --snapshot data/snapshots \
        [--episodes 20000] [--shard 0/4] [--model gpt-5.1]

Title-as-query bootstrap examples teach a retriever to copy titles; real
searches are questions and topic phrases. This writes, for a seeded sample
of episodes, one query a listener who wanted *that* episode might type or
say — in the episode's language, without copying the title. The output
(``<snapshot>/llm-queries-<model>[-shardK].jsonl``) is a raw dataset input
like the snapshot itself: the deterministic builder consumes it through
the ``llm_query`` generator and records its hash in the manifest, so the
non-deterministic step happens once, here, and is versioned by content.

Append-only and resumable: episodes already present in the output are
skipped. Requires ``OPENAI_API_KEY`` (environment or ``.env``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from ml.datasets import snapshot as snapshot_mod

DEFAULT_MODEL = "gpt-5.1"
BATCH_SIZE = 20

INSTRUCTIONS = """You write realistic podcast search queries for a search engine's training data.

For EACH episode below, write ONE query that a listener who wanted this \
particular episode might type or say into podcast search, before knowing \
it exists.

Rules:
- Write in the episode's language (given per episode).
- Do NOT copy or lightly paraphrase the title. Describe what the listener \
wants: the topic, question, person, or situation the episode addresses.
- Vary the style across episodes: a question ("how do ..."), a topic phrase \
("episodes about ..."), a need ("something on ... for beginners"), a name \
plus topic. 4 to 14 words.
- If the description is too thin to know what the episode is about, write \
a query for the show's general topic instead.
- Return a query for every episode id given."""


class EpisodeQuery(BaseModel):
    episode_id: int
    query: str


class BatchQueries(BaseModel):
    queries: list[EpisodeQuery]


def sample_episodes(snapshot_dir: Path, n: int, seed: int) -> pd.DataFrame:
    """Seeded sample of episodes with a usable description, joined to their
    show; the same snapshot, n, and seed always pick the same episodes."""
    episodes = pd.read_parquet(
        snapshot_dir / "episodes.parquet",
        columns=["id", "podcast_id", "title", "description", "language", "published_at"],
    )
    podcasts = pd.read_parquet(snapshot_dir / "podcasts.parquet", columns=["id", "title"])
    episodes = episodes[
        episodes["published_at"].notna()
        & (episodes["title"].str.strip() != "")
        & (episodes["description"].str.len() >= 80)
    ]
    picked = episodes.sample(n=min(n, len(episodes)), random_state=seed).sort_values("id")
    return picked.merge(
        podcasts.rename(columns={"id": "podcast_id", "title": "podcast_title"}), on="podcast_id"
    )


def format_batch(rows: list[dict]) -> str:
    lines = ["EPISODES:", ""]
    for row in rows:
        lines.append(
            f"- episode_id={row['id']} | language: {row['language'] or 'en'} | "
            f"show: {row['podcast_title']} | title: {row['title']}"
        )
        lines.append(f"  description: {str(row['description'])[:600]}")
    lines.append("")
    lines.append("Write one query per episode.")
    return "\n".join(lines)


def generate_batch(client, model: str, rows: list[dict]) -> list[EpisodeQuery]:
    response = client.responses.parse(
        model=model,
        reasoning={"effort": "low"},
        instructions=INSTRUCTIONS,
        input=format_batch(rows),
        text_format=BatchQueries,
    )
    return response.output_parsed.queries


def generate(
    rows: list[dict], out: Path, generator, model: str, batch_size: int = BATCH_SIZE
) -> dict:
    """Write one query per episode, resuming past ids already in ``out``.
    The generator is injected so tests run without the network."""
    done: set[int] = set()
    if out.exists():
        done = {json.loads(line)["episode_id"] for line in out.read_text().splitlines() if line}
    pending = [r for r in rows if int(r["id"]) not in done]
    stats = {"episodes": len(rows), "already_done": len(done), "written": 0, "failed_batches": 0}
    with out.open("a") as f:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            wanted = {int(r["id"]): r for r in batch}
            try:
                queries = generator(batch)
            except Exception as exc:  # noqa: BLE001 - one bad batch must not kill the run
                print(f"  batch at {batch[0]['id']} failed: {exc}", file=sys.stderr)
                stats["failed_batches"] += 1
                continue
            seen: set[int] = set()
            now = datetime.now(UTC).isoformat()
            for q in queries:
                text = q.query.strip()
                if q.episode_id not in wanted or q.episode_id in seen or len(text) < 3:
                    continue
                seen.add(q.episode_id)
                f.write(json.dumps({
                    "episode_id": q.episode_id,
                    "query_text": text,
                    "language": wanted[q.episode_id]["language"] or "",
                    "model": model,
                    "generated_at": now,
                }, sort_keys=True) + "\n")
                stats["written"] += 1
            f.flush()
    return stats


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", default="data/snapshots")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--shard", default="0/1", help="k/n: process the k-th of n slices")
    parser.add_argument("--out", default="", help="output path (default: in the snapshot dir)")
    args = parser.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set (env or .env)", file=sys.stderr)
        return 1

    snapshot_dir = snapshot_mod.resolve(args.snapshot)
    k, n = (int(x) for x in args.shard.split("/"))
    rows = sample_episodes(snapshot_dir, args.episodes, args.seed).to_dict("records")[k::n]
    suffix = f"-shard{k}" if n > 1 else ""
    out = Path(args.out) if args.out else (
        snapshot_dir / f"llm-queries-{args.model.replace('/', '_')}{suffix}.jsonl"
    )

    from openai import OpenAI

    client = OpenAI()
    stats = generate(rows, out, lambda batch: generate_batch(client, args.model, batch), args.model)
    stats["out"] = str(out)
    print(json.dumps(stats, indent=2))
    return 0 if stats["failed_batches"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
