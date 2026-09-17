"""Contrastive softmax loss over in-batch and sampled negatives.

For each query the candidate set is every episode encoded in the step: all
examples' positives (in-batch negatives) plus each example's own random
catalog negatives. The loss is cross-entropy of the query's positive over
that set, with two masks that keep label noise from becoming a training
signal:

- a candidate that *is* the query's positive episode (the same episode
  appearing as another example's positive or negative) is never a
  negative;
- a candidate that is the positive of an example with the *same query
  text* is never a negative either — weak topic labels use the category
  name as the query, so sibling positives are relevant, not wrong.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def candidate_mask(
    positive_index: Tensor,
    positive_ids: Tensor,
    candidate_ids: Tensor,
    query_keys: Tensor,
    candidate_keys: Tensor,
) -> Tensor:
    """``[n_queries, n_candidates]`` boolean mask: True where the candidate
    may be scored against the query (its positive, or a valid negative).

    ``query_keys`` identifies each query's text; ``candidate_keys`` carries
    the same key for in-batch positives and ``-1`` for sampled negatives.
    """
    own = positive_index.unsqueeze(1) == torch.arange(
        candidate_ids.shape[0], device=candidate_ids.device
    ).unsqueeze(0)
    same_item = positive_ids.unsqueeze(1) == candidate_ids.unsqueeze(0)
    same_query = (query_keys.unsqueeze(1) == candidate_keys.unsqueeze(0)) & (
        candidate_keys.unsqueeze(0) >= 0
    )
    return own | ~(same_item | same_query)


def contrastive_softmax_loss(
    logits: Tensor,
    positive_index: Tensor,
    mask: Tensor | None = None,
    weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Weighted mean cross-entropy of each query's positive over its allowed
    candidates. Returns the loss and batch diagnostics (accuracy = positive
    ranked first, mean reciprocal rank of the positive)."""
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    per_query = F.cross_entropy(logits, positive_index, reduction="none")
    if weights is None:
        loss = per_query.mean()
    else:
        loss = (per_query * weights).sum() / weights.sum().clamp(min=1e-9)

    with torch.no_grad():
        positive_logits = logits.gather(1, positive_index.unsqueeze(1))
        rank = (logits > positive_logits).sum(dim=1) + 1
        diagnostics = {
            "accuracy": (rank == 1).float().mean().item(),
            "mrr": (1.0 / rank.float()).mean().item(),
            "candidates": float(logits.shape[1]),
        }
    return loss, diagnostics
