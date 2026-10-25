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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml.serving.runtime import ServingBundle, resolve_bundle


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
            raise HTTPException(status_code=422, detail=f"bad candidate evidence: {err}") \
                from err
        return {"bundle_id": bundle.bundle_id,
                "feature_version": bundle.manifest["feature_version"],
                "scores": scores.tolist(),
                "features": features.tolist() if req.return_features else None,
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 3)}

    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", default=os.environ.get("PODFIND_BUNDLE"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8081")))
    args = parser.parse_args(argv)
    bundle = ServingBundle.load(resolve_bundle(args.bundle))
    print(f"serving bundle {bundle.bundle_id} from {bundle.directory}", flush=True)
    uvicorn.run(create_app(bundle), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
