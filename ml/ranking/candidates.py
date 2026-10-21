"""Hybrid candidate generation for second-stage ranking.

Three legs each retrieve their top ``depth`` episodes with the query's hard
filters: lexical full-text search, pretrained E5 vector search, and a
fine-tuned two-tower model. The candidate set is their union. For every
candidate the generator then reads the evidence all three legs have about
it, not only the legs that retrieved it: its exact cosine similarity under
both vector models (from the stored vectors), its lexical score
components, and its transcript similarity. Leg ranks are kept for the
candidate-source features and for RRF.

The same object serves offline feature generation and the harness systems
in ``ml.ranking.system``, so training and evaluation see identical
candidates and evidence.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from ml.embeddings.text import MODEL_ID, build_query
from ml.embeddings.transcripts import TRANSCRIPT_MODEL_ID
from ml.embeddings.two_tower_index import model_id_for_artifact
from ml.retrieval.lexical import SEARCH_SQL as LEXICAL_SQL
from ml.retrieval.lexical import query_params
from ml.retrieval.vector import SEARCH_SQL as VECTOR_SQL
from ml.retrieval.vector import enable_iterative_scan, filter_params

LEGS = ("lexical", "e5", "two_tower")
DEFAULT_DEPTH = 50

EVIDENCE_SQL = """
WITH q AS (
    SELECT websearch_to_tsquery(podfind_ts_config(%(parse_lang)s), %(query)s) AS tsq
)
SELECT e.id, e.podcast_id, e.title, coalesce(left(e.description, 1000), '') AS description,
       e.language, e.duration_seconds, e.published_at,
       p.title AS podcast_title, p.publisher, p.popularity,
       (t.episode_id IS NOT NULL) AS has_transcript,
       coalesce(ts_rank(e.search_tsv, q.tsq), 0) AS lex_episode,
       coalesce(ts_rank(t.search_tsv, q.tsq), 0) AS lex_transcript,
       coalesce(ts_rank(p.search_tsv, q.tsq), 0) AS lex_podcast
FROM episodes e
JOIN podcasts p ON p.id = e.podcast_id
LEFT JOIN transcripts t ON t.episode_id = e.id
CROSS JOIN q
WHERE e.id = ANY(%(ids)s)
"""

VECTORS_SQL = """
SELECT episode_id, model, embedding FROM episode_embeddings
WHERE episode_id = ANY(%(ids)s) AND model = ANY(%(models)s)
"""


def _as_array(value) -> np.ndarray:
    """pgvector returns ``Vector`` objects or arrays depending on version."""
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value, dtype=np.float32)


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


class CandidateGenerator:
    """Retrieves and annotates the hybrid candidate set for a query."""

    def __init__(self, conn: psycopg.Connection, two_tower_artifact: str,
                 depth: int = DEFAULT_DEPTH):
        from sentence_transformers import SentenceTransformer

        from ml.retrieval.two_tower import TwoTowerSearch

        self.conn = conn
        self.depth = depth
        register_vector(conn)
        enable_iterative_scan(conn)
        self.e5 = SentenceTransformer(MODEL_ID)
        self.two_tower = TwoTowerSearch(conn, two_tower_artifact)
        self.two_tower_model = model_id_for_artifact(Path(two_tower_artifact))
        with conn.cursor() as cur:
            cur.execute("SELECT podcast_id, count(*) FROM episodes GROUP BY podcast_id")
            self.show_episodes = dict(cur.fetchall())

    def generate(self, query: dict) -> CandidateSet:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        e5_vec = self.e5.encode(build_query(query["query_text"]), normalize_embeddings=True)
        tt_vec = self.two_tower.encode(query).numpy()
        timings["encode_ms"] = (time.perf_counter() - start) * 1000

        ranks: dict[str, dict[int, int]] = {}
        with self.conn.cursor() as cur:
            for leg, sql, params in (
                ("lexical", LEXICAL_SQL, query_params(query, self.depth)),
                ("e5", VECTOR_SQL, {"model": MODEL_ID, "qvec": e5_vec,
                                    **filter_params(query), "top_k": self.depth}),
                ("two_tower", VECTOR_SQL, {"model": self.two_tower_model, "qvec": tt_vec,
                                           **filter_params(query), "top_k": self.depth}),
            ):
                t0 = time.perf_counter()
                cur.execute(sql, params)
                ranks[leg] = {row[0]: i for i, row in enumerate(cur.fetchall(), start=1)}
                timings[f"{leg}_ms"] = (time.perf_counter() - t0) * 1000

            ids = sorted(set().union(*(r.keys() for r in ranks.values())))
            t0 = time.perf_counter()
            lex = query_params(query, 0)
            cur.execute(EVIDENCE_SQL, {"query": lex["query"], "parse_lang": lex["parse_lang"],
                                       "ids": ids})
            columns = [d.name for d in cur.description]
            evidence = {row[0]: dict(zip(columns, row, strict=True)) for row in cur.fetchall()}
            cur.execute(VECTORS_SQL, {"ids": ids, "models": [
                MODEL_ID, self.two_tower_model, TRANSCRIPT_MODEL_ID]})
            vectors: dict[tuple[int, str], np.ndarray] = {
                (row[0], row[1]): _as_array(row[2]) for row in cur.fetchall()}
            timings["evidence_ms"] = (time.perf_counter() - t0) * 1000

        def cosine(episode: int, model: str, qvec: np.ndarray) -> float:
            v = vectors.get((episode, model))
            return float(v @ qvec) if v is not None else 0.0

        rows = []
        for episode in ids:
            row = evidence[episode]
            row["episode_id"] = row.pop("id")
            row["show_episodes"] = self.show_episodes.get(row["podcast_id"], 0)
            row["e5_sim"] = cosine(episode, MODEL_ID, e5_vec)
            row["two_tower_sim"] = cosine(episode, self.two_tower_model, tt_vec)
            row["transcript_sim"] = cosine(episode, TRANSCRIPT_MODEL_ID, e5_vec)
            row["has_transcript_vector"] = (episode, TRANSCRIPT_MODEL_ID) in vectors
            for leg in LEGS:
                row[f"{leg}_rank"] = ranks[leg].get(episode)
            rows.append(row)
        timings["total_ms"] = (time.perf_counter() - start) * 1000
        return CandidateSet(query=query, rows=rows, timings_ms=timings)
