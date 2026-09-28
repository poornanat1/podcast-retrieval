"""ANN retrieval with a fine-tuned two-tower model.

The query tower runs in-process (CPU is fine for one short query); the
episode side is the versioned index built by ``ml.embeddings.two_tower_index``
for the same artifact, scanned with the query's structured filters as hard
predicates.
"""

from __future__ import annotations

from pathlib import Path

import psycopg
import torch
from pgvector.psycopg import register_vector

from ml.embeddings.two_tower_index import model_id_for_artifact
from ml.features import Featurizer
from ml.models.two_tower import TwoTowerModel
from ml.retrieval.vector import SEARCH_SQL, enable_iterative_scan, filter_params


class TwoTowerSearch:
    """Fine-tuned two-tower ANN retrieval over the artifact's index."""

    name = "two-tower"

    def __init__(self, conn: psycopg.Connection, artifact: str):
        self.conn = conn
        self.artifact = Path(artifact)
        self.model_id = model_id_for_artifact(self.artifact)
        # Reports and MLflow runs carry the artifact, so two trained models
        # never share a system name.
        self.name = f"two-tower:{self.artifact.name}"
        self.model = TwoTowerModel.load(self.artifact).eval()
        self.featurizer = Featurizer.load(self.artifact)
        register_vector(conn)
        enable_iterative_scan(conn)

    @torch.no_grad()
    def encode(self, query: dict) -> torch.Tensor:
        return self.model.encode_query(self.featurizer.query_batch([query]))[0]

    def search(self, query: dict, k: int) -> list[int]:
        params = {
            "model": self.model_id,
            "qvec": self.encode(query).numpy(),
            **filter_params(query),
            "top_k": k,
        }
        with self.conn.cursor() as cur:
            cur.execute(SEARCH_SQL, params)
            return [row[0] for row in cur.fetchall()]
