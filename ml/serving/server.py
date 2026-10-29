"""PodFind model server: versioned query-tower and ranker inference.

    uv run python -m ml.serving.server [--bundle data/serving/bundles/<id>] [--port 8081]

Loads exactly one immutable bundle at startup (``--bundle``, or the
promoted one in ``data/serving/ACTIVE``) and refuses to start if any file
differs from the bundle manifest. Changing models means starting a server
on another bundle; nothing is reloaded in place.

Endpoints (JSON):

- ``GET  /healthz``: liveness.
- ``GET  /v1/model``: the bundle manifest (versions, feature list, the
  vector-index model keys and candidate depth retrieval must use).
- ``POST /v1/encode``: ``{"query": {...}}`` to the two-tower and E5 query
  vectors.
- ``POST /v1/rerank``: ``{"query": {...}, "candidates": [evidence rows],
  "reference_time": optional ISO time, "return_features": bool}`` to one
  score per candidate, in input order.

``query`` carries ``query_text`` and the structured filters (``language``,
``max_duration_seconds``, ``published_after``, ``no_explicit``). Evidence
rows carry the fields ``ml.ranking.candidates`` produces.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from ml.ranking.features import FEATURES
from ml.serving.runtime import ServingBundle, resolve_bundle

# Prometheus metrics at /metrics: what the drift check and dashboards read.
# Feature values are sums and counts per feature, enough to watch each
# feature's mean drift against the training distribution.
_SECONDS = (.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1, 2.5)
ENCODE_SECONDS = Histogram("podfind_model_encode_seconds", "Query encoding time.",
                           buckets=_SECONDS)
RERANK_SECONDS = Histogram("podfind_model_rerank_seconds", "Reranking time.",
                           buckets=_SECONDS)
QUERY_NORM = Histogram("podfind_model_query_norm", "L2 norm of query vectors.",
                       ["tower"], buckets=(0.5, 0.9, 0.99, 0.999, 1.001, 1.01, 1.1, 2))
RERANK_CANDIDATES = Histogram("podfind_model_rerank_candidates", "Candidates per rerank call.",
                              buckets=(0, 10, 50, 100, 200, 400, 800))
SCORE = Histogram("podfind_model_score", "Ranker scores.",
                  buckets=[round(-4 + 0.5 * i, 1) for i in range(29)])
# A feature's running sum must be monotonic for Prometheus to read it as a
# counter, and several features are negative (the similarity gaps), so the
# positive and negative parts are summed separately: mean = (pos - neg) / count.
FEATURE_POS = Counter("podfind_model_feature_positive_sum_total",
                      "Sum of the positive feature values.", ["feature"])
FEATURE_NEG = Counter("podfind_model_feature_negative_sum_total",
                      "Sum of the absolute negative feature values.", ["feature"])
FEATURE_COUNT = Counter("podfind_model_feature_values_total", "Feature values observed.",
                        ["feature"])
FEATURE_NAN = Counter("podfind_model_feature_nan_total", "Non-finite feature values.",
                      ["feature"])
REQUEST_ERRORS = Counter("podfind_model_request_errors_total", "Rejected requests.",
                         ["endpoint"])


class Query(BaseModel):
    query_text: str = ""
    language: str | None = None
    max_duration_seconds: int | None = None
    published_after: str | None = None
    no_explicit: bool = False


class EncodeRequest(BaseModel):
    query: Query


class RerankRequest(BaseModel):
    query: Query
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    reference_time: datetime | None = None
    return_features: bool = False


def create_app(bundle: ServingBundle) -> FastAPI:
    app = FastAPI(title="podfind-model-server", version=bundle.bundle_id)

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "bundle_id": bundle.bundle_id}

    @app.get("/v1/model")
    def model() -> dict:
        return bundle.manifest

    @app.post("/v1/encode")
    def encode(req: EncodeRequest) -> dict:
        start = time.perf_counter()
        vectors = bundle.encode(req.query.model_dump())
        ENCODE_SECONDS.observe(time.perf_counter() - start)
        for tower, vec in vectors.items():
            QUERY_NORM.labels(tower).observe(float(np.linalg.norm(vec)))
        return {"bundle_id": bundle.bundle_id,
                "two_tower": vectors["two_tower"].tolist(), "e5": vectors["e5"].tolist(),
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 3)}

    @app.post("/v1/rerank")
    def rerank(req: RerankRequest) -> dict:
        start = time.perf_counter()
        try:
            scores, features = bundle.rerank(req.query.model_dump(), req.candidates,
                                             req.reference_time)
        except (KeyError, TypeError, ValueError) as err:
            REQUEST_ERRORS.labels("rerank").inc()
            raise HTTPException(status_code=422, detail=f"bad candidate evidence: {err}") \
                from err
        RERANK_SECONDS.observe(time.perf_counter() - start)
        observe_rerank(scores, features)
        return {"bundle_id": bundle.bundle_id,
                "feature_version": bundle.manifest["feature_version"],
                "scores": scores.tolist(),
                "features": features.tolist() if req.return_features else None,
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 3)}

    return app


def observe_rerank(scores: np.ndarray, features: np.ndarray) -> None:
    RERANK_CANDIDATES.observe(len(scores))
    for score in scores.tolist():
        SCORE.observe(score)
    if not len(features):
        return
    finite = np.isfinite(features)
    for j in range(min(len(FEATURES), features.shape[1])):
        name = FEATURES[j]
        column = features[:, j]
        bad = int((~finite[:, j]).sum())
        if bad:
            FEATURE_NAN.labels(name).inc(bad)
        good = column[finite[:, j]]
        if len(good):
            FEATURE_POS.labels(name).inc(float(good[good > 0].sum()))
            FEATURE_NEG.labels(name).inc(float(-good[good < 0].sum()))
            FEATURE_COUNT.labels(name).inc(len(good))


def main(argv: list[str] | None = None) -> int:
    from ml.serving.listen import serve

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", default=os.environ.get("PODFIND_BUNDLE"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8081")))
    args = parser.parse_args(argv)
    bundle = ServingBundle.load(resolve_bundle(args.bundle))
    print(f"serving bundle {bundle.bundle_id} from {bundle.directory}", flush=True)
    serve(create_app(bundle), args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
