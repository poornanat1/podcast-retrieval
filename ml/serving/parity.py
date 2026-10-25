"""Training-serving parity checks for a bundle; the gate for promotion.

    uv run python -m ml.serving.parity --bundle data/serving/bundles/<id> \
        --api http://localhost:8090

Compares the training-side reference (``golden.jsonl``, from
``ml.serving.golden``) with production on the same queries and writes
``parity.json`` into the bundle. ``ml.serving.promote`` refuses a bundle
whose report is missing, failing, or stale.

Offline (bundle files only):

- feature schema: the bundle's feature list and version match this code
  and the ranker it was exported from;
- tokenizer: the serving tokenizers produce exactly the training model
  inputs (ids, masks, language ids, intents) for every golden query;
- query embeddings: ONNX vectors match the PyTorch vectors, and are unit
  length;
- ranker: ONNX scores match LightGBM scores on the golden features;
- missing values: features computed the serving way (ISO strings, nulls)
  equal the training computation, including on rows with nulled fields;
- numeric distributions: on the same relevance queries, each feature's mean
  over the serving path's candidates is within 0.05 training standard
  deviations of its mean over the ranker's training rows, and the training
  rows hold no NaN (the training path reads Parquet, where nulls are NaN).

Online (the Go API in debug mode, with a lexical time limit long enough
that no leg times out): for every golden query the API's query vectors,
candidate set, per-candidate features, scores, and top 10 are compared
with the reference.

Thresholds are module constants and recorded in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ml.ranking.features import FEATURE_VERSION, FEATURES, featurize
from ml.ranking.types import CandidateSet
from ml.serving.bundle import verify
from ml.serving.golden import GOLDEN
from ml.serving.runtime import ServingBundle

PARITY = "parity.json"
EMBEDDING_TOL = 1e-4
NORM_TOL = 1e-3
RANKER_TOL = 1e-4
FEATURE_TOL = 1e-4
SCORE_TOL = 1e-3
MIN_SCORE_AGREEMENT = 0.99   # share of shared candidates within SCORE_TOL
MIN_MEAN_JACCARD = 0.98      # candidate sets, averaged over queries
MIN_JACCARD = 0.90           # candidate sets, worst query
MIN_TOP10_IDENTICAL = 0.95   # share of queries with an identical top 10
MAX_STD_MEAN_DIFF = 0.05     # |serving mean - training mean| / training std, same queries


def check(name: str, passed: bool, value, threshold, detail: str = "") -> dict:
    return {"name": name, "passed": bool(passed), "value": value, "threshold": threshold,
            **({"detail": detail} if detail else {})}


def _parse_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        r = dict(r)
        if r.get("published_at"):
            r["published_at"] = datetime.fromisoformat(r["published_at"])
        out.append(r)
    return out


def offline_checks(bundle: ServingBundle, golden: list[dict], training_features: np.ndarray
                   ) -> list[dict]:
    manifest = bundle.manifest
    ranker_meta = json.loads((bundle.directory / "ranker.json").read_text())
    checks = [check("feature schema",
                    manifest["features"] == list(FEATURES) == ranker_meta["features"]
                    and manifest["feature_version"] == FEATURE_VERSION
                    == ranker_meta["feature_version"],
                    manifest["feature_version"], FEATURE_VERSION)]

    mismatched = []
    for g in golden:
        got = bundle.two_tower_inputs([g["query"]])
        if any(got[k].tolist() != g["inputs"][k] for k in g["inputs"]):
            mismatched.append(g["query_id"])
    checks.append(check("tokenizer: two-tower model inputs identical", not mismatched,
                        len(golden) - len(mismatched), len(golden), ", ".join(mismatched)))

    diffs, norms = [], []
    for g in golden:
        vecs = bundle.encode(g["query"])
        diffs.append(float(np.abs(vecs["two_tower"] - np.array(g["two_tower_vector"])).max()))
        diffs.append(float(np.abs(vecs["e5"] - np.array(g["e5_vector"])).max()))
        norms += [abs(float(np.linalg.norm(vecs["two_tower"])) - 1),
                  abs(float(np.linalg.norm(vecs["e5"])) - 1)]
    checks.append(check("query embeddings: ONNX vs PyTorch, max abs diff",
                        max(diffs) <= EMBEDDING_TOL, max(diffs), EMBEDDING_TOL))
    checks.append(check("query embeddings: unit norm, max deviation",
                        max(norms) <= NORM_TOL, max(norms), NORM_TOL))

    ranker_diff, feature_diff, missing_diff = 0.0, 0.0, 0.0
    for g in golden:
        if not g["rows"]:
            continue
        x = np.array(g["features"], dtype=np.float32)
        onnx_scores = bundle.ranker.run(None, {"features": x})[0].ravel()
        ranker_diff = max(ranker_diff, float(np.abs(onnx_scores - np.array(g["scores"])).max()))
        ref_time = datetime.fromisoformat(g["reference_time"])
        served = bundle.features(g["query"], g["rows"], ref_time)  # ISO strings, as over HTTP
        feature_diff = max(feature_diff, float(np.abs(served - x).max()))
        nulled = [{**r, "published_at": None, "duration_seconds": None, "popularity": None,
                   "lexical_rank": None, "e5_rank": None, "two_tower_rank": None,
                   "description": None, "publisher": None} for r in g["rows"][:20]]
        a = bundle.features(g["query"], nulled, ref_time)
        b = featurize(CandidateSet(query=g["query"], rows=_parse_rows(nulled), timings_ms={}),
                      ref_time)
        missing_diff = max(missing_diff, float(np.abs(a - b).max()))
    checks.append(check("ranker: ONNX vs LightGBM, max abs diff", ranker_diff <= RANKER_TOL,
                        ranker_diff, RANKER_TOL))
    checks.append(check("features: serving path vs training, max abs diff",
                        feature_diff <= FEATURE_TOL, feature_diff, FEATURE_TOL))
    checks.append(check("missing values: nulled fields, serving vs training", missing_diff == 0,
                        missing_diff, 0))

    # Same queries both ways: the ranker's training rows for the golden
    # relevance queries against what the serving path computes for them now.
    # A skew in data handling (NaN vs null, a changed default) shifts these;
    # sampling differences cannot.
    served = np.vstack([np.array(g["features"]) for g in golden
                        if g["rows"] and g["query_type"] != "edge"])
    nan_cols = [FEATURES[j] for j in np.nonzero(np.isnan(training_features).any(0))[0]]
    mean, std = np.nanmean(training_features, 0), np.nanstd(training_features, 0)
    shift = np.abs(np.nanmean(served, 0) - mean) / np.where(std > 0, std, 1)
    worst = [f"{FEATURES[i]} {shift[i]:.3f}" for i in np.argsort(-shift)[:3]]
    checks.append(check("numeric distributions: training rows vs serving, same queries",
                        float(shift.max()) <= MAX_STD_MEAN_DIFF, float(shift.max()),
                        MAX_STD_MEAN_DIFF, "largest: " + ", ".join(worst)))
    checks.append(check("numeric distributions: no NaN in training features", not nan_cols,
                        len(nan_cols), 0, ", ".join(nan_cols)))
    return checks


def _search(api: str, g: dict) -> dict:
    q = g["query"]
    filters = {k: q[k] for k in ("language", "max_duration_seconds", "published_after",
                                 "no_explicit") if q.get(k)}
    body = json.dumps({"query": q["query_text"], "filters": filters, "session_id": "parity",
                       "limit": 10, "debug": True, "skip_understanding": True,
                       "reference_time": g["reference_time"]}).encode()
    req = urllib.request.Request(f"{api}/v1/search/text", body,
                                 {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def online_checks(api: str, golden: list[dict], bundle_id: str) -> tuple[list[dict], list[dict]]:
    per_query = []
    vec_diff, feat_diff, jaccards, agree, total, top10_same = 0.0, 0.0, [], 0, 0, 0
    bundles = set()
    for g in golden:
        r = _search(api, g)
        bundles.add(r["versions"]["bundle_id"])
        d = r["debug"]
        vec_diff = max(vec_diff,
                       float(np.abs(np.array(d["two_tower_vector"]) - g["two_tower_vector"]).max()),
                       float(np.abs(np.array(d["e5_vector"]) - g["e5_vector"]).max()))
        ref = {row["episode_id"]: i for i, row in enumerate(g["rows"])}
        got = {c["episode_id"]: i for i, c in enumerate(d["candidates"])}
        shared = set(ref) & set(got)
        union = set(ref) | set(got)
        jac = len(shared) / len(union) if union else 1.0
        jaccards.append(jac)
        legs_ok = all(v["status"] in ("ok", "skipped") for v in r["legs"].values())
        for eid in shared:
            fi, fj = ref[eid], got[eid]
            if d["features"]:
                feat_diff = max(feat_diff, float(np.abs(
                    np.array(d["features"][fj]) - np.array(g["features"][fi])).max()))
            total += 1
            agree += abs(d["scores"][fj] - g["scores"][fi]) <= SCORE_TOL
        same = [x["episode_id"] for x in r["results"]] == g["top10"]
        top10_same += same
        per_query.append({"query_id": g["query_id"], "jaccard": round(jac, 4),
                          "candidates": [len(ref), len(got)], "top10_identical": same,
                          "legs_ok": legs_ok})
    n = len(golden)
    checks = [
        check("online: served bundle", bundles == {bundle_id}, sorted(bundles), bundle_id),
        check("online: no leg timed out or failed", all(p["legs_ok"] for p in per_query),
              sum(p["legs_ok"] for p in per_query), n),
        check("online: query vectors vs PyTorch, max abs diff", vec_diff <= EMBEDDING_TOL,
              vec_diff, EMBEDDING_TOL),
        check("online: candidate sets, mean Jaccard", float(np.mean(jaccards)) >= MIN_MEAN_JACCARD,
              float(np.mean(jaccards)), MIN_MEAN_JACCARD),
        check("online: candidate sets, worst Jaccard", min(jaccards) >= MIN_JACCARD,
              min(jaccards), MIN_JACCARD),
        check("online: features of shared candidates, max abs diff", feat_diff <= FEATURE_TOL,
              feat_diff, FEATURE_TOL),
        check(f"online: scores within {SCORE_TOL}", agree / max(total, 1) >= MIN_SCORE_AGREEMENT,
              agree / max(total, 1), MIN_SCORE_AGREEMENT),
        check("online: identical top 10", top10_same / n >= MIN_TOP10_IDENTICAL,
              top10_same / n, MIN_TOP10_IDENTICAL),
    ]
    return checks, per_query


def training_feature_matrix(bundle: ServingBundle, query_ids: set[str] | None = None
                            ) -> np.ndarray:
    """The ranker's training features (its cached candidates, featurized at
    the training reference time), optionally for some queries only."""
    import pandas as pd

    from ml.ranking.train import features_for

    ranker_meta = json.loads((bundle.directory / "ranker.json").read_text())
    ranker_dir = Path("data/models") / bundle.manifest["sources"]["ranker"]["artifact"]
    df = pd.read_parquet(ranker_dir / "candidates.parquet")
    if query_ids is not None:
        df = df[df.query_id.isin(query_ids)]
    queries = {json.loads(line)["query_id"]: json.loads(line) for line in
               Path(ranker_meta["config"]["relevance_dir"], "queries.jsonl").read_text()
               .splitlines() if line}
    ref = datetime.fromisoformat(ranker_meta["config"]["reference_time"])
    return np.vstack([x for _, x in features_for(df, queries, ref).values()])


def run(bundle_dir: Path, api: str) -> dict:
    manifest = verify(bundle_dir)
    bundle = ServingBundle(bundle_dir, manifest)
    golden_path = bundle_dir / GOLDEN
    golden = [json.loads(line) for line in golden_path.read_text().splitlines() if line]
    relevance_ids = {g["query_id"] for g in golden if g["query_type"] != "edge"}
    checks = offline_checks(bundle, golden, training_feature_matrix(bundle, relevance_ids))
    online, per_query = online_checks(api, golden, manifest["bundle_id"])
    checks += online
    report = {
        "bundle_id": manifest["bundle_id"],
        "golden_sha256": hashlib.sha256(golden_path.read_bytes()).hexdigest(),
        "golden_queries": len(golden),
        "checked_at": datetime.now(UTC).isoformat(),
        "api": api,
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "per_query": per_query,
    }
    (bundle_dir / PARITY).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--api", default="http://localhost:8090")
    args = parser.parse_args(argv)
    report = run(Path(args.bundle), args.api.rstrip("/"))
    for c in report["checks"]:
        value = c["value"] if not isinstance(c["value"], float) else f"{c['value']:.3g}"
        print(f"{'PASS' if c['passed'] else 'FAIL'}  {c['name']}: {value} "
              f"(threshold {c['threshold']}){'  ' + c['detail'] if c.get('detail') else ''}")
    print("parity", "PASSED" if report["passed"] else "FAILED")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
