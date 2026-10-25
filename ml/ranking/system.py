"""Harness systems over the shared hybrid candidate set.

- ``hybrid-rrf3``: the three legs' candidates (``ml.ranking.candidates``)
  ordered by reciprocal-rank fusion. The ordering the ranker must beat.
- ``reranker``: the same candidates ordered by a trained LightGBM ranker
  (``ml.ranking.train``). With ``out_of_fold`` (the default), each
  relevance-set query is scored by the fold model that never saw it, so the
  harness measures the ranker honestly on the queries it was trained on;
  without it, the final all-query model scores every query (for serving
  and for queries outside the relevance set).

Both names carry the artifacts they read, so runs never collide.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psycopg

from ml.ranking.candidates import DEFAULT_DEPTH, CandidateGenerator
from ml.ranking.features import FEATURE_VERSION, FEATURES, featurize, rrf_score
from ml.ranking.lightgbm_runtime import LIGHTGBM_THREADS, load_lightgbm


class HybridRRF3:
    name = "hybrid-rrf3"

    def __init__(self, conn: psycopg.Connection, two_tower_artifact: str,
                 depth: int = DEFAULT_DEPTH):
        self.generator = CandidateGenerator(conn, two_tower_artifact, depth)
        self.name = f"hybrid-rrf3:{Path(two_tower_artifact).name}"

    def search(self, query: dict, k: int) -> list[int]:
        rows = self.generator.generate(query).rows
        ordered = sorted(rows, key=lambda r: (
            -rrf_score(r), -sum(1 for leg in ("lexical", "e5", "two_tower")
                                if r.get(f"{leg}_rank")), r["episode_id"]))
        return [r["episode_id"] for r in ordered[:k]]


class Reranker:
    name = "reranker"

    def __init__(self, conn: psycopg.Connection, ranker: str, out_of_fold: bool = True,
                 reference_time: str | None = None):
        lgb = load_lightgbm()

        self.artifact = Path(ranker)
        meta = json.loads((self.artifact / "ranker.json").read_text())
        if meta["feature_version"] != FEATURE_VERSION or meta["features"] != list(FEATURES):
            raise ValueError(f"{ranker} was trained on a different feature set")
        config = meta["config"]
        self.generator = CandidateGenerator(conn, config["two_tower_artifact"], config["depth"])
        # Out-of-fold evaluation reproduces training's freshness features, so
        # it uses the training reference time; serving uses the clock.
        if reference_time:
            self.reference_time: datetime | None = datetime.fromisoformat(reference_time)
        elif out_of_fold:
            self.reference_time = datetime.fromisoformat(config["reference_time"])
        else:
            self.reference_time = None
        self.final = lgb.Booster(model_file=str(self.artifact / "model.txt"))
        self.out_of_fold = out_of_fold
        self.folds: dict[str, int] = {}
        self.fold_models: dict[int, lgb.Booster] = {}
        if out_of_fold:
            self.folds = json.loads((self.artifact / "folds.json").read_text())
            self.fold_models = {k: lgb.Booster(model_file=str(self.artifact / f"fold_{k}.txt"))
                                for k in sorted(set(self.folds.values()))}
        self.name = f"reranker:{self.artifact.name}" + ("" if out_of_fold else ":final")
        self.stage_ms: list[dict[str, float]] = []

    def model_for(self, query: dict):
        if self.out_of_fold and query.get("query_id") in self.folds:
            return self.fold_models[self.folds[query["query_id"]]]
        return self.final

    def search(self, query: dict, k: int) -> list[int]:
        cands = self.generator.generate(query)
        t0 = time.perf_counter()
        x = featurize(cands, self.reference_time or datetime.now(UTC))
        t1 = time.perf_counter()
        scores = self.model_for(query).predict(x, num_threads=LIGHTGBM_THREADS) \
            if len(x) else np.zeros(0)
        t2 = time.perf_counter()
        self.stage_ms.append({**cands.timings_ms, "features_ms": (t1 - t0) * 1000,
                              "score_ms": (t2 - t1) * 1000})
        ids = cands.episode_ids
        order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
        return [ids[i] for i in order[:k]]
