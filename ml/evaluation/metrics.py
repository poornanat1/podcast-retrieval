"""Pure per-query retrieval metrics.

Conventions: ``ranked`` is the system's ranked list of episode ids;
``relevant`` is the set of episode ids with a qualifying grade; ``grades``
maps episode id → graded relevance (0-3). Unjudged documents count as
irrelevant, as is standard for pooled evaluation.
"""

from __future__ import annotations

import math


def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    """Share of the relevant set retrieved in the top k."""
    if not relevant:
        raise ValueError("recall undefined with no relevant documents")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def hit_rate_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    """1.0 when any relevant document appears in the top k."""
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def reciprocal_rank(ranked: list[int], relevant: set[int]) -> float:
    for position, episode in enumerate(ranked, start=1):
        if episode in relevant:
            return 1.0 / position
    return 0.0


def dcg_at_k(ranked: list[int], grades: dict[int, int], k: int) -> float:
    return sum(
        (2 ** grades.get(episode, 0) - 1) / math.log2(position + 1)
        for position, episode in enumerate(ranked[:k], start=1)
    )


def ndcg_at_k(ranked: list[int], grades: dict[int, int], k: int) -> float:
    ideal = sorted(grades.values(), reverse=True)[:k]
    ideal_dcg = sum(
        (2**grade - 1) / math.log2(position + 1)
        for position, grade in enumerate(ideal, start=1)
    )
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(ranked, grades, k) / ideal_dcg


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile; 0 for empty input."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]
