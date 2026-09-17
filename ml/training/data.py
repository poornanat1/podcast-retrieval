"""Batches of training examples joined to catalog rows.

A dataset split (the normalized training-example rows) references
episodes by id; the snapshot it was built from supplies their text and
metadata. The batcher loads only the episodes a split references, then
yields batches whose candidate block is ``[positives | random negatives]``
so the loss can use every encoded episode as an in-batch negative.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

from ml.features.featurize import Featurizer


@dataclasses.dataclass
class Batch:
    query: dict[str, Tensor]
    episodes: dict[str, Tensor]  # candidates: positives first, then negatives
    positive_index: Tensor  # [n_queries] index of each query's positive
    positive_ids: Tensor  # [n_queries] episode ids
    candidate_ids: Tensor  # [n_candidates] episode ids
    query_keys: Tensor  # [n_queries] id of the query text within the batch
    candidate_keys: Tensor  # [n_candidates] query key for positives, -1 for negatives
    weights: Tensor  # [n_queries] label strength

    def to(self, device: torch.device | str) -> Batch:
        moved = {
            f.name: getattr(self, f.name) for f in dataclasses.fields(self)
        }
        for name, value in moved.items():
            if isinstance(value, dict):
                moved[name] = {k: v.to(device) for k, v in value.items()}
            else:
                moved[name] = value.to(device)
        return Batch(**moved)

    @property
    def size(self) -> int:
        return int(self.positive_index.shape[0])


def load_episode_rows(snapshot_dir: str | Path, episode_ids: set[int]) -> dict[int, dict]:
    """Catalog rows (episode + podcast fields) for the referenced ids only."""
    snapshot_dir = Path(snapshot_dir)
    episodes = pd.read_parquet(snapshot_dir / "episodes.parquet")
    episodes = episodes[episodes["id"].isin(episode_ids)]
    podcasts = pd.read_parquet(snapshot_dir / "podcasts.parquet").rename(
        columns={"id": "podcast_id", "title": "podcast_title", "language": "podcast_language",
                 "explicit": "podcast_explicit"}
    )[["podcast_id", "podcast_title", "publisher", "categories"]]
    joined = episodes.merge(podcasts, on="podcast_id", how="left")
    rows = {}
    for row in joined.to_dict("records"):
        cats = row.get("categories")
        row["categories"] = None if cats is None or isinstance(cats, float) else list(cats)
        rows[int(row["id"])] = row
    return rows


class ExampleBatcher:
    def __init__(
        self,
        examples: pd.DataFrame,
        episode_rows: dict[int, dict],
        featurizer: Featurizer,
        negatives_per_example: int,
    ) -> None:
        missing = set(int(i) for i in examples["positive_episode_id"]) - set(episode_rows)
        if missing:
            raise ValueError(f"{len(missing)} positives are missing from the snapshot")
        self.examples = examples.reset_index(drop=True)
        self.rows = episode_rows
        self.featurizer = featurizer
        self.negatives_per_example = negatives_per_example

    def __len__(self) -> int:
        return len(self.examples)

    def collate(self, examples: pd.DataFrame) -> Batch:
        queries, positives, negatives = [], [], []
        pos_ids, cand_ids, cand_times, weights = [], [], [], []
        query_keys, cand_keys = [], []
        key_of: dict[str, int] = {}
        for row in examples.itertuples(index=False):
            key = key_of.setdefault(row.query_text, len(key_of))
            queries.append({"query_text": row.query_text})
            positive = int(row.positive_episode_id)
            positives.append(self.rows[positive])
            pos_ids.append(positive)
            cand_ids.append(positive)
            cand_times.append(row.event_time)
            query_keys.append(key)
            cand_keys.append(key)
            weights.append(float(row.label_strength))
            chosen = [int(n) for n in row.negative_episode_ids][: self.negatives_per_example]
            for negative in chosen:
                if negative in self.rows:
                    negatives.append((self.rows[negative], row.event_time, negative))
        for row, event_time, negative in negatives:
            positives.append(row)
            cand_ids.append(negative)
            cand_times.append(event_time)
            cand_keys.append(-1)

        return Batch(
            query=self.featurizer.query_batch(queries),
            episodes=self.featurizer.episode_batch(positives, reference_time=cand_times),
            positive_index=torch.arange(len(queries)),
            positive_ids=torch.tensor(pos_ids, dtype=torch.long),
            candidate_ids=torch.tensor(cand_ids, dtype=torch.long),
            query_keys=torch.tensor(query_keys, dtype=torch.long),
            candidate_keys=torch.tensor(cand_keys, dtype=torch.long),
            weights=torch.tensor(weights, dtype=torch.float32),
        )

    def batches(self, batch_size: int, shuffle: bool, seed: int = 0) -> Iterator[Batch]:
        order = list(range(len(self.examples)))
        if shuffle:
            random.Random(seed).shuffle(order)
        for start in range(0, len(order), batch_size):
            chunk = self.examples.iloc[order[start:start + batch_size]]
            if len(chunk) < 2:
                continue  # a single query has no in-batch negatives
            yield self.collate(chunk)
