"""Hybrid retrieval: lexical and vector candidates fused with
reciprocal-rank fusion.

Both systems retrieve independently (each applying the query's structured
filters), and their rankings merge by

    score(episode) = Σ_systems  weight / (constant + rank)

which needs no score calibration between systems. Exact names and phrases
arrive via the lexical ranking; paraphrased and exploratory intent via the
vector ranking.
"""

from __future__ import annotations

import psycopg

from ml.retrieval.lexical import LexicalSearch
from ml.retrieval.vector import VectorSearch

RRF_CONSTANT = 60
CANDIDATE_DEPTH = 200


def rrf_merge(
    rankings: list[list[int]],
    weights: list[float] | None = None,
    constant: int = RRF_CONSTANT,
) -> list[int]:
    """Fuse rankings into one list ordered by summed reciprocal-rank score.
    Ties break toward the item ranked in more lists, then by id for
    determinism."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    appearances: dict[int, int] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (constant + rank)
            appearances[item] = appearances.get(item, 0) + 1
    return sorted(scores, key=lambda i: (-scores[i], -appearances[i], i))


class HybridRRF:
    """Lexical + vector candidates, RRF-merged."""

    name = "hybrid-rrf"

    def __init__(self, conn: psycopg.Connection):
        self.lexical = LexicalSearch(conn)
        self.vector = VectorSearch(conn)

    def search(self, query: dict, k: int) -> list[int]:
        depth = max(CANDIDATE_DEPTH, k)
        merged = rrf_merge([
            self.lexical.search(query, depth),
            self.vector.search(query, depth),
        ])
        return merged[:k]
