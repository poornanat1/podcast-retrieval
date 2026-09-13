"""Text preparation for embedding models.

The E5 family requires role prefixes: documents embed as ``passage: ...``
and queries as ``query: ...``; mixing them up costs significant retrieval
quality. Episode passages combine show title, episode title, and a bounded
slice of the description — the same fields lexical search indexes.
"""

from __future__ import annotations

MODEL_ID = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
MAX_DESCRIPTION_CHARS = 1000


def build_passage(podcast_title: str, episode_title: str, description: str) -> str:
    parts = [p.strip() for p in (podcast_title, episode_title) if p and p.strip()]
    text = ". ".join(parts)
    description = (description or "").strip()[:MAX_DESCRIPTION_CHARS]
    if description:
        text = f"{text}. {description}" if text else description
    return f"passage: {text}" if text else ""


def build_query(query_text: str) -> str:
    return f"query: {query_text.strip()}"
