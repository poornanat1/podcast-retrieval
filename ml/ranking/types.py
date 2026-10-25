"""Types shared by candidate generation, features, and the model server."""

from __future__ import annotations

import dataclasses

LEGS = ("lexical", "e5", "two_tower")


@dataclasses.dataclass
class CandidateSet:
    """One query's candidates: ``rows`` (one dict per episode, in a stable
    order) plus the per-stage wall time in milliseconds."""

    query: dict
    rows: list[dict]
    timings_ms: dict[str, float]

    @property
    def episode_ids(self) -> list[int]:
        return [r["episode_id"] for r in self.rows]
