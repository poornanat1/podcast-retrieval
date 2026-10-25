"""ONNX export of the serving models.

- ``export_query_tower``: the two-tower model's query tower (encoder, mean
  pooling, language and intent context, head, L2 normalization). Inputs:
  ``input_ids``, ``attention_mask`` [batch, seq] int64, ``language_id``
  [batch] int64, ``intent`` [batch, n_intents] float32; output
  ``embedding`` [batch, dim].
- ``export_e5_query``: the pretrained E5 encoder with mean pooling and L2
  normalization, as ``sentence-transformers`` encodes queries. Inputs
  ``input_ids``, ``attention_mask``; output ``embedding``.
- ``export_ranker``: the LightGBM ranker's final model. Input ``features``
  [n, n_features] float32; output ``variable`` [n, 1].

Each export is checked against the source model on sample inputs before it
is written, and fails rather than writing a model that disagrees.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ONNX_OPSET = 17
RANKER_OPSET = 15  # highest the LightGBM converter supports
EXPORT_TOLERANCE = 1e-4
LIGHTGBM_ZERO = 1e-35  # LightGBM's kZeroThreshold


def _check(name: str, got: np.ndarray, want: np.ndarray, tol: float = EXPORT_TOLERANCE) -> float:
    diff = float(np.abs(np.asarray(got) - np.asarray(want)).max())
    if diff > tol:
        raise ValueError(f"{name}: ONNX output differs from source by {diff:.2e} (> {tol})")
    return diff


def export_query_tower(artifact: Path, out: Path) -> float:
    import onnxruntime as ort
    import torch

    from ml.features import Featurizer
    from ml.models.two_tower import TwoTowerModel

    model = TwoTowerModel.load(artifact).eval()
    featurizer = Featurizer.load(artifact)

    class QueryTower(torch.nn.Module):
        def __init__(self, tower):
            super().__init__()
            self.tower = tower

        def forward(self, input_ids, attention_mask, language_id, intent):
            return self.tower({"input_ids": input_ids, "attention_mask": attention_mask,
                               "language_id": language_id, "intent": intent})

    sample = featurizer.query_batch([
        {"query_text": "history of the roman empire"},
        {"query_text": "noticias de economia", "language": "es", "max_duration_seconds": 1200},
    ])
    names = ["input_ids", "attention_mask", "language_id", "intent"]
    args = tuple(sample[n] for n in names)
    torch.onnx.export(
        QueryTower(model.query_tower).eval(), args, out, input_names=names,
        output_names=["embedding"], opset_version=ONNX_OPSET, dynamo=False,
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "language_id": {0: "batch"}, "intent": {0: "batch"},
                      "embedding": {0: "batch"}})
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = session.run(None, {n: a.numpy() for n, a in zip(names, args, strict=True)})[0]
    with torch.no_grad():
        want = model.encode_query(sample).numpy()
    return _check("query tower", got, want)


def export_e5_query(out: Path, tokenizer_dir: Path) -> float:
    import onnxruntime as ort
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer

    from ml.embeddings.text import MODEL_ID, build_query

    st = SentenceTransformer(MODEL_ID, device="cpu")
    st.tokenizer.save_pretrained(tokenizer_dir)

    class E5(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, input_ids, attention_mask):
            hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask) \
                .last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return F.normalize(pooled, dim=-1)

    text = build_query("practical uses of artificial intelligence")
    enc = st.tokenizer([text], return_tensors="pt")
    torch.onnx.export(
        E5(st[0].auto_model).eval(), (enc["input_ids"], enc["attention_mask"]), out,
        input_names=["input_ids", "attention_mask"], output_names=["embedding"],
        opset_version=ONNX_OPSET, dynamo=False,
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"}, "embedding": {0: "batch"}})
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = session.run(None, {"input_ids": enc["input_ids"].numpy(),
                             "attention_mask": enc["attention_mask"].numpy()})[0]
    return _check("e5 query encoder", got, st.encode([text], normalize_embeddings=True))


def float32_safe_booster(booster, lgb):
    """The booster with every split threshold rounded down to the largest
    float32 at or below it.

    ONNX tree ensembles store thresholds as float32 while LightGBM keeps
    doubles, so a float32 feature equal to a rounded-up threshold goes the
    other way in ONNX. Serving features are float32, and for any float32
    ``x``, ``x <= t`` exactly when ``x <= floor32(t)``: the rewritten model
    makes the same decisions as the original on every serving input, and its
    thresholds convert to float32 losslessly. Numerical splits only.
    """
    import re

    text = booster.model_to_string()
    if any(int(n) for n in re.findall(r"^num_cat=(\d+)", text, re.M)):
        raise ValueError("categorical splits are not supported by the float32 rewrite")

    def floor32(value: str) -> str:
        t = float(value)
        f = np.float32(t)
        if float(f) > t:
            f = np.nextafter(f, np.float32(-np.inf))
        return repr(float(f))

    text = re.sub(r"^threshold=(.*)$",
                  lambda m: "threshold=" + " ".join(floor32(v) for v in m.group(1).split()),
                  text, flags=re.M)
    # Tree byte sizes change with the rewrite; LightGBM parses without them.
    text = re.sub(r"^tree_sizes=.*\n", "", text, flags=re.M)
    return lgb.Booster(model_str=text)


def threshold_probe(booster, rows: int = 2000, seed: int = 0) -> np.ndarray:
    """Random inputs plus inputs sitting exactly on split thresholds (and one
    float32 step above), where float32/float64 disagreements show up."""
    rng = np.random.default_rng(seed)
    n = booster.num_feature()
    tree = booster.dump_model()["tree_info"]
    thresholds: dict[int, list[float]] = {}

    def walk(node):
        if "split_feature" in node:
            thresholds.setdefault(node["split_feature"], []).append(node["threshold"])
            walk(node["left_child"])
            walk(node["right_child"])

    for t in tree:
        walk(t["tree_structure"])
    x = rng.normal(size=(2 * rows, n)).astype(np.float32)
    for j, ts in thresholds.items():
        at = np.float32(rng.choice(ts, size=rows))
        x[rows:, j] = np.where(rng.random(rows) < 0.5, at,
                               np.nextafter(at, np.float32(np.inf)))
    # LightGBM reads any |value| <= 1e-35 as exactly zero (its splits between
    # 0 and 1 sit at +-1e-35); real features are 0 or far larger, so probe
    # that band with 0, matching LightGBM's own semantics.
    x[np.abs(x) <= LIGHTGBM_ZERO] = 0.0
    return x


def export_ranker(ranker: Path, out: Path) -> float:
    import onnxruntime as ort
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    from ml.ranking.lightgbm_runtime import LIGHTGBM_THREADS, load_lightgbm

    lgb = load_lightgbm()
    booster = lgb.Booster(model_file=str(ranker / "model.txt"))
    safe = float32_safe_booster(booster, lgb)
    n = booster.num_feature()
    onx = convert_lightgbm(safe, initial_types=[("features", FloatTensorType([None, n]))],
                           target_opset=RANKER_OPSET, zipmap=False)
    out.write_bytes(onx.SerializeToString())
    x = threshold_probe(booster)
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = session.run(None, {"features": x})[0].ravel()
    return _check("ranker", got, booster.predict(x, num_threads=LIGHTGBM_THREADS))
