import hashlib
import json

import numpy as np
import pytest

from ml.features import FeatureContract, Featurizer
from ml.serving.bundle import MODEL_FILES, bundle_id, sha256, verify
from ml.serving.promote import PromotionRefused, gate, promote
from ml.serving.query_features import TwoTowerQueryInputs
from ml.tests.helpers import tiny_tokenizer


def test_serving_query_inputs_match_training_featurizer(tmp_path) -> None:
    contract = FeatureContract(languages=("en", "es"), categories=("news",), max_query_tokens=16)
    tok = tiny_tokenizer(tmp_path)
    tok.save_pretrained(tmp_path / "tok")
    queries = [{"query_text": "space rockets launch"},
               {"query_text": "the history of bread", "language": "es",
                "max_duration_seconds": 600, "no_explicit": True},
               {"query_text": " ".join(["news"] * 40)},  # truncated
               {"query_text": ""}]
    want = Featurizer(contract, tok).query_batch(queries)
    got = TwoTowerQueryInputs(contract, tmp_path / "tok" / "tokenizer.json")(queries)
    for key in want:
        assert np.array_equal(got[key], want[key].numpy()), key


def _fake_bundle(root):
    bundle = root / "b"
    bundle.mkdir(parents=True)
    for name in MODEL_FILES:
        (bundle / name).write_text(f"contents of {name}")
    files = {name: sha256(bundle / name) for name in MODEL_FILES}
    manifest = {"bundle_id": bundle_id(files), "files": files}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    (bundle / "golden.jsonl").write_text('{"query_id": "q"}\n')
    golden_sha = hashlib.sha256((bundle / "golden.jsonl").read_bytes()).hexdigest()
    report = {"bundle_id": manifest["bundle_id"], "golden_sha256": golden_sha, "passed": True,
              "checked_at": "2026-10-24T00:00:00+00:00",
              "checks": [{"name": "a", "passed": True}]}
    (bundle / "parity.json").write_text(json.dumps(report))
    return bundle, report


def test_promotion_requires_passing_current_parity(tmp_path) -> None:
    bundle, report = _fake_bundle(tmp_path)
    assert verify(bundle)["bundle_id"] == report["bundle_id"]
    pointer = tmp_path / "serving" / "ACTIVE"
    entry = promote(bundle, pointer)
    assert pointer.read_text().strip() == report["bundle_id"] and entry["previous"] is None
    assert len((pointer.parent / "promotions.jsonl").read_text().splitlines()) == 1

    def refused(mutate, message):
        b, r = _fake_bundle(tmp_path / message.replace(" ", "_"))
        mutate(b, r)
        with pytest.raises(PromotionRefused, match=message):
            gate(b)

    refused(lambda b, r: (b / "parity.json").unlink(), "no parity report")
    refused(lambda b, r: (b / "parity.json").write_text(json.dumps(
        {**r, "passed": False, "checks": [{"name": "online: identical top 10",
                                            "passed": False}]})), "parity failed")
    refused(lambda b, r: (b / "golden.jsonl").write_text("changed\n"), "different golden")
    refused(lambda b, r: (b / "parity.json").write_text(json.dumps({**r, "bundle_id": "x"})),
            "is for bundle")
    refused(lambda b, r: (b / "ranker.onnx").write_text("tampered"), "does not verify")


def test_float32_safe_ranker_export_is_exact(tmp_path) -> None:
    from ml.ranking.lightgbm_runtime import LIGHTGBM_THREADS, load_lightgbm
    from ml.serving.export import export_ranker, float32_safe_booster, threshold_probe

    lgb = load_lightgbm()
    rng = np.random.default_rng(0)
    x = rng.random((3000, 6)).astype(np.float32)
    x[:, 0] = (x[:, 0] > 0.5).astype(np.float32)   # binary feature (splits at +-1e-35)
    x[:, 1] = np.round(x[:, 1] * 7) / 7             # few distinct values
    y = ((x[:, 0] + x[:, 2] * 3 + rng.random(3000)) * 1.2).astype(int).clip(0, 3)
    model = lgb.LGBMRanker(n_estimators=40, num_leaves=7, min_child_samples=5, verbose=-1,
                           num_threads=LIGHTGBM_THREADS).fit(x, y, group=[30] * 100)
    (tmp_path / "ranker").mkdir()
    model.booster_.save_model(str(tmp_path / "ranker" / "model.txt"))
    booster = lgb.Booster(model_file=str(tmp_path / "ranker" / "model.txt"))

    probe = threshold_probe(booster)
    safe = float32_safe_booster(booster, lgb)
    assert np.array_equal(safe.predict(probe, num_threads=1), booster.predict(probe, num_threads=1))
    assert export_ranker(tmp_path / "ranker", tmp_path / "ranker.onnx") < 1e-4


def test_model_server_endpoints_with_a_stub_bundle() -> None:
    from fastapi.testclient import TestClient

    from ml.serving.server import create_app

    class Stub:
        bundle_id = "stub"
        manifest = {"bundle_id": "stub", "feature_version": "1"}

        def encode(self, query):
            return {"two_tower": np.ones(3, dtype=np.float32), "e5": np.zeros(3, dtype=np.float32)}

        def rerank(self, query, rows, reference_time):
            if any("episode_id" not in r for r in rows):
                raise KeyError("episode_id")
            return np.arange(len(rows), dtype=np.float32), np.ones((len(rows), 2))

    client = TestClient(create_app(Stub()))
    assert client.get("/healthz").json()["bundle_id"] == "stub"
    assert client.post("/v1/encode", json={"query": {"query_text": "x"}}).json()["e5"] == [0, 0, 0]
    r = client.post("/v1/rerank", json={"query": {"query_text": "x"}, "return_features": True,
                                        "candidates": [{"episode_id": 1}, {"episode_id": 2}]})
    assert r.json()["scores"] == [0, 1] and r.json()["features"] == [[1, 1], [1, 1]]
    bad = client.post("/v1/rerank", json={"query": {"query_text": "x"}, "candidates": [{}]})
    assert bad.status_code == 422
