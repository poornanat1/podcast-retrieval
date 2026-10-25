"""Ranker feature definitions, shared by training, evaluation, and serving.

``featurize(candidates, reference_time)`` turns one query's candidate set
(``ml.ranking.candidates``) into a ``[n_candidates, len(FEATURES)]`` float
matrix. Every feature is a pure function of the query, the candidate's
evidence, and the other candidates for the same query, so an online path
that assembles the same evidence gets the same numbers. ``FEATURES`` is the
contract: a trained ranker records the list and its version, and refuses a
different one.

Groups:

- **Leg evidence**: both vector similarities, the transcript similarity,
  the lexical score and its components, each also relative to the best
  candidate for the query (so scores are comparable across queries).
- **Candidate source**: reciprocal rank in each leg (0 when the leg did not
  retrieve it), the number of legs, and the RRF score.
- **Exact and partial matches**: query against show title, publisher, and
  episode title, as whole-string and token-coverage matches.
- **Item**: publication age, duration, show size, provider popularity,
  transcript availability.

Not included: query type (an evaluation label, unknown at serving time),
and the user features in the design (affinity, preferences, completion and
skip rates), because no interaction data exists yet.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from ml.ranking.types import LEGS, CandidateSet
from ml.relevance.queries import normalize

FEATURE_VERSION = "1"
RRF_CONSTANT = 60

FEATURES: tuple[str, ...] = (
    # leg evidence
    "two_tower_sim", "two_tower_sim_gap", "e5_sim", "e5_sim_gap",
    "transcript_sim", "lex_score", "lex_score_rel", "lex_episode", "lex_transcript",
    "lex_podcast",
    # candidate source
    "lexical_rr", "e5_rr", "two_tower_rr", "legs", "rrf_score",
    # matches
    "show_exact", "show_in_query", "query_in_show", "show_coverage",
    "publisher_exact", "publisher_in_query", "publisher_coverage",
    "query_in_title", "title_coverage", "description_coverage", "query_tokens",
    # item
    "log_age_days", "log_duration", "log_show_episodes", "popularity", "has_transcript",
)


def lexical_score(row: dict) -> float:
    """The lexical leg's own ranking score (``ml.retrieval.lexical``)."""
    return float(row["lex_episode"]) + float(row["lex_transcript"]) \
        + 0.5 * float(row["lex_podcast"])


def rrf_score(row: dict, constant: int = RRF_CONSTANT) -> float:
    return sum(1.0 / (constant + row[f"{leg}_rank"]) for leg in LEGS
               if row.get(f"{leg}_rank"))


def _tokens(text: str) -> list[str]:
    return [t for t in normalize(text or "").split() if t]


def _coverage(query_tokens: list[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    have = set(_tokens(text))
    return sum(1 for t in query_tokens if t in have) / len(query_tokens)


def _contains(haystack: str, needle: str) -> float:
    """Whole-token containment of ``needle`` in ``haystack`` (both
    normalized), ignoring needles shorter than three characters."""
    if len(needle) < 3:
        return 0.0
    return 1.0 if f" {needle} " in f" {haystack} " else 0.0


def featurize(candidates: CandidateSet, reference_time: datetime) -> np.ndarray:
    q = normalize(candidates.query["query_text"])
    q_tokens = q.split()
    rows = candidates.rows
    if not rows:
        return np.zeros((0, len(FEATURES)), dtype=np.float32)
    best_tt = max(r["two_tower_sim"] for r in rows)
    best_e5 = max(r["e5_sim"] for r in rows)
    best_lex = max(lexical_score(r) for r in rows)

    out = np.zeros((len(rows), len(FEATURES)), dtype=np.float32)
    for i, r in enumerate(rows):
        show = normalize(r.get("podcast_title") or "")
        publisher = normalize(r.get("publisher") or "")
        title = normalize(r.get("title") or "")
        lex = lexical_score(r)
        published = r.get("published_at")
        age_days = max((reference_time - published).total_seconds() / 86400, 0.0) \
            if published is not None else 3650.0
        values = {
            "two_tower_sim": r["two_tower_sim"],
            "two_tower_sim_gap": r["two_tower_sim"] - best_tt,
            "e5_sim": r["e5_sim"],
            "e5_sim_gap": r["e5_sim"] - best_e5,
            "transcript_sim": r["transcript_sim"],
            "lex_score": lex,
            "lex_score_rel": lex / best_lex if best_lex > 0 else 0.0,
            "lex_episode": float(r["lex_episode"]),
            "lex_transcript": float(r["lex_transcript"]),
            "lex_podcast": float(r["lex_podcast"]),
            "lexical_rr": 1.0 / r["lexical_rank"] if r.get("lexical_rank") else 0.0,
            "e5_rr": 1.0 / r["e5_rank"] if r.get("e5_rank") else 0.0,
            "two_tower_rr": 1.0 / r["two_tower_rank"] if r.get("two_tower_rank") else 0.0,
            "legs": sum(1 for leg in LEGS if r.get(f"{leg}_rank")),
            "rrf_score": rrf_score(r),
            "show_exact": 1.0 if show and q == show else 0.0,
            "show_in_query": _contains(q, show),
            "query_in_show": _contains(show, q),
            "show_coverage": _coverage(q_tokens, show),
            "publisher_exact": 1.0 if publisher and q == publisher else 0.0,
            "publisher_in_query": _contains(q, publisher),
            "publisher_coverage": _coverage(q_tokens, publisher),
            "query_in_title": _contains(title, q),
            "title_coverage": _coverage(q_tokens, title),
            "description_coverage": _coverage(q_tokens, r.get("description") or ""),
            "query_tokens": len(q_tokens),
            "log_age_days": math.log1p(age_days),
            "log_duration": math.log1p(float(r.get("duration_seconds") or 0)),
            "log_show_episodes": math.log1p(float(r.get("show_episodes") or 0)),
            "popularity": float(r.get("popularity") or 0.0),
            "has_transcript": 1.0 if r.get("has_transcript") else 0.0,
        }
        out[i] = [values[name] for name in FEATURES]
    return out
