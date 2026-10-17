"""Show-name and publisher-name training queries.

    uv run python -m ml.datasets.name_queries --snapshot data/snapshots/<id> \
        [--relevance data/relevance/queries.jsonl] [--per-show 4] [--per-publisher 2]

LLM-written queries are almost all topical, so a model fine-tuned on them
drifts away from matching names. This writes the missing query shape: a
show's normalized title, or its publisher's normalized name, paired with
episodes of that show (``--per-show`` / ``--per-publisher`` seeded picks
per show). Each line carries ``exclude_podcasts``, every show the name
legitimately refers to, so no episode of those shows is used as a negative.

Any show whose title or publisher matches a relevance-set query text is
left out entirely: the relevance set's navigational and entity queries are
the biggest shows and publishers by name, and training on them would score
memorization, not the skill.

Output ``<snapshot>/name-queries.jsonl`` (episode_id, query_text,
query_kind, exclude_podcasts) is a raw dataset input in the same shape as
the LLM queries file, so ``ml.hard_negatives.mine`` and the builder's
``llm_query`` generator consume it unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from ml.relevance.queries import normalize

MIN_NAME_CHARS = 3


def _pick(seed: int, kind: str, podcast_id: int, episode_ids: list[int], k: int) -> list[int]:
    def key(e: int) -> str:
        return hashlib.sha256(f"{seed}:{kind}:{podcast_id}:{e}".encode()).hexdigest()
    return sorted(episode_ids, key=key)[:k]


def build(episodes: pd.DataFrame, podcasts: pd.DataFrame, held_out: set[str],
          per_show: int = 4, per_publisher: int = 2, seed: int = 20261017) -> list[dict]:
    """Name queries for every show not referenced by the relevance set."""
    podcasts = podcasts.assign(
        title_norm=podcasts["title"].fillna("").map(normalize),
        publisher_norm=podcasts["publisher"].fillna("").map(normalize),
    )
    blocked = podcasts["title_norm"].isin(held_out) | podcasts["publisher_norm"].isin(held_out)
    by_title = podcasts.groupby("title_norm")["id"].apply(lambda s: sorted(int(i) for i in s))
    by_publisher = podcasts.groupby("publisher_norm")["id"].apply(
        lambda s: sorted(int(i) for i in s))
    eligible = episodes[episodes["published_at"].notna()]
    show_episodes = eligible.groupby("podcast_id")["id"].apply(lambda s: [int(i) for i in s])

    used: set[int] = set()
    rows: list[dict] = []
    for show in podcasts[~blocked].sort_values("id").itertuples():
        pool = [e for e in show_episodes.get(show.id, []) if e not in used]
        if not pool:
            continue
        kinds = [("show", show.title_norm, by_title[show.title_norm], per_show)]
        if show.publisher_norm and show.publisher_norm != show.title_norm:
            kinds.append(("publisher", show.publisher_norm,
                          by_publisher[show.publisher_norm], per_publisher))
        for kind, text, shows, k in kinds:
            if len(text) < MIN_NAME_CHARS:
                continue
            for episode_id in _pick(seed, kind, int(show.id), pool, k):
                used.add(episode_id)
                pool.remove(episode_id)
                rows.append({"episode_id": episode_id, "query_text": text,
                             "query_kind": kind, "exclude_podcasts": shows})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--relevance", default="data/relevance/queries.jsonl")
    parser.add_argument("--per-show", type=int, default=4)
    parser.add_argument("--per-publisher", type=int, default=2)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    snapshot = Path(args.snapshot)
    episodes = pd.read_parquet(snapshot / "episodes.parquet",
                               columns=["id", "podcast_id", "published_at"])
    podcasts = pd.read_parquet(snapshot / "podcasts.parquet")
    held_out = {normalize(json.loads(line)["query_text"])
                for line in Path(args.relevance).read_text().splitlines() if line}
    rows = build(episodes, podcasts, held_out, args.per_show, args.per_publisher)

    out = Path(args.out) if args.out else snapshot / "name-queries.jsonl"
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    kinds = pd.Series([r["query_kind"] for r in rows]).value_counts().to_dict()
    print(json.dumps({"out": str(out), "queries": len(rows), "by_kind": kinds,
                      "held_out_texts": len(held_out),
                      "distinct_texts": len({r["query_text"] for r in rows})}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
