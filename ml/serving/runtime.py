"""Model-server runtime: one immutable bundle, ONNX inference only.

``ServingBundle.load(dir)`` verifies every file against the bundle
manifest, then serves:

- ``encode(query)``: the two-tower and E5 query vectors;
- ``rerank(query, rows, reference_time)``: ranker scores (and, on request,
  the feature matrix) for a candidate set whose evidence rows the caller
  assembled, using the same feature definitions as training
  (``ml.ranking.features``).

Neither PyTorch nor LightGBM is imported: production inference is ONNX
Runtime, and the parity checks compare it with the training-side models.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ml.features.contract import FeatureContract
from ml.ranking.features import FEATURE_VERSION, FEATURES, featurize
from ml.ranking.types import CandidateSet
from ml.serving.bundle import verify
from ml.serving.query_features import E5QueryInputs, TwoTowerQueryInputs

ACTIVE_POINTER = Path("data/serving/ACTIVE")


def resolve_bundle(path: str | None) -> Path:
    """An explicit bundle directory, or the promoted one named by
    ``data/serving/ACTIVE``."""
    if path:
        return Path(path)
    if not ACTIVE_POINTER.exists():
        raise FileNotFoundError("no bundle given and no promoted bundle (data/serving/ACTIVE)")
    return ACTIVE_POINTER.parent / "bundles" / ACTIVE_POINTER.read_text().strip()


def _parse_time(value) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


class ServingBundle:
    def __init__(self, directory: Path, manifest: dict):
        self.directory = directory
        self.manifest = manifest
        self.bundle_id = manifest["bundle_id"]
        if manifest["feature_version"] != FEATURE_VERSION or \
                manifest["features"] != list(FEATURES):
            raise ValueError(f"bundle {self.bundle_id} was built for a different feature set "
                             f"({manifest['feature_version']}) than this code "
                             f"({FEATURE_VERSION})")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]

        def session(name: str) -> ort.InferenceSession:
            return ort.InferenceSession(str(directory / name), options, providers=providers)

        self.query_tower = session("query_tower.onnx")
        self.e5 = session("e5_query.onnx")
        self.ranker = session("ranker.onnx")
        self.contract = FeatureContract.from_dict(
            json.loads((directory / "feature_contract.json").read_text()))
        self.two_tower_inputs = TwoTowerQueryInputs(self.contract,
                                                    directory / "two_tower_tokenizer.json")
        self.e5_inputs = E5QueryInputs(directory / "e5_tokenizer.json")

    @classmethod
    def load(cls, directory: str | Path) -> ServingBundle:
        directory = Path(directory)
        return cls(directory, verify(directory))

    def encode(self, query: dict) -> dict[str, np.ndarray]:
        two_tower = self.query_tower.run(None, self.two_tower_inputs([query]))[0][0]
        e5 = self.e5.run(None, self.e5_inputs(query.get("query_text", "")))[0][0]
        return {"two_tower": two_tower, "e5": e5}

    def features(self, query: dict, rows: list[dict],
                 reference_time: datetime | None = None) -> np.ndarray:
        rows = [{**r, "published_at": _parse_time(r.get("published_at"))} for r in rows]
        return featurize(CandidateSet(query=query, rows=rows, timings_ms={}),
                         reference_time or datetime.now(UTC))

    def rerank(self, query: dict, rows: list[dict],
               reference_time: datetime | None = None) -> tuple[np.ndarray, np.ndarray]:
        x = self.features(query, rows, reference_time)
        if not len(x):
            return np.zeros(0, dtype=np.float32), x
        return self.ranker.run(None, {"features": x})[0].ravel(), x
