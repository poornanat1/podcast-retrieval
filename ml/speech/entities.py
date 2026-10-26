"""Named entities in the relevance queries, for the named-entity error rate.

Entity and navigational queries are names by construction (a show, a host,
a publisher). The entity is the longest span of the query that appears,
as whole words, in a catalog show title or publisher name; when only a
single word matches a multi-word query (for example "today" in "today
explained", "sports" in "sports 180"), the whole query is the entity. The
annotations are written into the test-set manifest so the evaluation does
not need the database.
"""

from __future__ import annotations

import re
import unicodedata

NAME_TYPES = ("entity", "navigational")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def catalog_names(conn) -> set[str]:
    rows = conn.execute("SELECT title, publisher FROM podcasts").fetchall()
    names = set()
    for title, publisher in rows:
        for value in (title, publisher):
            if value:
                names.add(" " + normalize(value) + " ")
    return names


def entity_span(query_text: str, names: set[str]) -> str:
    tokens = normalize(query_text).split()
    for n in range(len(tokens), 0, -1):
        for i in range(len(tokens) - n + 1):
            span = " ".join(tokens[i:i + n])
            if any(f" {span} " in name for name in names):
                if n == 1 and len(tokens) > 1:
                    return " ".join(tokens)
                return span
    return " ".join(tokens)


def annotate(queries: list[dict], names: set[str]) -> dict[str, list[str]]:
    """query_id to its entities (one per name query, none otherwise)."""
    out = {}
    for q in queries:
        if q["query_type"] in NAME_TYPES:
            out[q["query_id"]] = [entity_span(q["query_text"], names)]
        else:
            out[q["query_id"]] = []
    return out
