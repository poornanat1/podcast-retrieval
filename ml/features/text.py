"""Text composition for the two towers.

Episode metadata is flattened into one passage: show and episode titles
first, then the short always-present fields (categories, publisher), then
the bounded description — so truncation at the token budget eats the tail
of the description, never a title. Host and guest names are not structured
catalog fields yet; they reach the model through the description text.
"""

from __future__ import annotations

from collections.abc import Iterable

from ml.features.contract import FeatureContract


def _clean(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def build_metadata_text(
    contract: FeatureContract,
    podcast_title: object,
    episode_title: object,
    description: object,
    publisher: object = "",
    categories: Iterable[object] | None = None,
) -> str:
    parts = [p for p in (_clean(podcast_title), _clean(episode_title)) if p]
    text = ". ".join(parts)
    if categories is not None and not isinstance(categories, float | str):
        names = [c for c in (_clean(c) for c in categories) if c]
        if names:
            text = f"{text}. Categories: {', '.join(names)}" if text else (
                f"Categories: {', '.join(names)}"
            )
    publisher = _clean(publisher)
    if publisher:
        text = f"{text}. Publisher: {publisher}" if text else f"Publisher: {publisher}"
    description = _clean(description)[: contract.max_description_chars]
    if description:
        text = f"{text}. {description}" if text else description
    return f"{contract.passage_prefix}{text}" if text else ""


def build_transcript_text(contract: FeatureContract, transcript: object) -> str:
    text = _clean(transcript)[: contract.max_transcript_chars]
    return f"{contract.passage_prefix}{text}" if text else ""


def build_query_text(contract: FeatureContract, query_text: object) -> str:
    return f"{contract.query_prefix}{_clean(query_text)}"
