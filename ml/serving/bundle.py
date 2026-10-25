"""Build an immutable serving bundle.

    uv run python -m ml.serving.bundle \
        --two-tower data/models/<two-tower artifact> --ranker data/models/<ranker artifact>

A bundle is everything the model server needs, in one directory named by
its content: the three ONNX models (``ml.serving.export``), both
tokenizers, the two-tower feature contract, and the ranker's metadata.
``manifest.json`` records the sha256 of every file, the source artifacts,
the feature version and feature list, the vector-index model keys the
online retrieval must query, and the candidate depth. The bundle id is a
hash of the file hashes, so a bundle cannot change without changing its
id, and the server refuses a bundle whose files no longer match its
manifest.

Written to ``data/serving/bundles/<bundle id>/``. Golden examples and the
parity report are added next to it later (``ml.serving.golden``,
``ml.serving.parity``); they are derived, not part of the bundle's
identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from ml.serving.export import export_e5_query, export_query_tower, export_ranker

BUNDLE_ROOT = Path("data/serving/bundles")
MANIFEST = "manifest.json"
MODEL_FILES = ("query_tower.onnx", "e5_query.onnx", "ranker.onnx",
               "two_tower_tokenizer.json", "e5_tokenizer.json",
               "feature_contract.json", "ranker.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_id(file_hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(file_hashes, sort_keys=True).encode()).hexdigest()[:12]


def build(two_tower: Path, ranker: Path, root: Path = BUNDLE_ROOT) -> Path:
    from ml.datasets.build import code_revision
    from ml.embeddings.text import MODEL_ID
    from ml.embeddings.transcripts import TRANSCRIPT_MODEL_ID
    from ml.embeddings.two_tower_index import model_id_for_artifact

    ranker_meta = json.loads((ranker / "ranker.json").read_text())
    if Path(ranker_meta["config"]["two_tower_artifact"]).name != two_tower.name:
        raise SystemExit(f"{ranker} was trained on candidates from "
                         f"{ranker_meta['config']['two_tower_artifact']}, not {two_tower}")
    if not (two_tower / "index.json").exists():
        raise SystemExit(f"{two_tower} has no built index (make index ARTIFACT=...)")

    root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".bundle-", dir=root))
    tmp.chmod(0o755)  # mkdtemp is owner-only; servers may run as another user
    checks = {
        "query_tower_max_abs_diff": export_query_tower(two_tower, tmp / "query_tower.onnx"),
        "e5_query_max_abs_diff": export_e5_query(tmp / "e5_query.onnx", tmp / "e5_tokenizer"),
        "ranker_max_abs_diff": export_ranker(ranker, tmp / "ranker.onnx"),
    }
    shutil.copy(tmp / "e5_tokenizer" / "tokenizer.json", tmp / "e5_tokenizer.json")
    shutil.rmtree(tmp / "e5_tokenizer")
    shutil.copy(two_tower / "tokenizer" / "tokenizer.json", tmp / "two_tower_tokenizer.json")
    shutil.copy(two_tower / "feature_contract.json", tmp / "feature_contract.json")
    shutil.copy(ranker / "ranker.json", tmp / "ranker.json")

    files = {name: sha256(tmp / name) for name in MODEL_FILES}
    bid = bundle_id(files)
    two_tower_meta = json.loads((two_tower / "artifact.json").read_text())
    manifest = {
        "bundle_id": bid,
        "created_at": datetime.now(UTC).isoformat(),
        "files": files,
        "sources": {
            "two_tower": {"artifact": two_tower.name,
                          "training_config_sha256": two_tower_meta.get("training_config_sha256"),
                          "feature_contract_sha256": two_tower_meta.get(
                              "feature_contract_sha256")},
            "ranker": {"artifact": ranker.name, "config_sha256": ranker_meta["config_sha256"],
                       "qrels_sha256": ranker_meta["qrels_sha256"]},
            "e5": {"model": MODEL_ID},
        },
        "feature_version": ranker_meta["feature_version"],
        "features": ranker_meta["features"],
        "candidate_depth": ranker_meta["config"]["depth"],
        "index_models": {"two_tower": model_id_for_artifact(two_tower), "e5": MODEL_ID,
                         "transcript": TRANSCRIPT_MODEL_ID},
        "export_checks": checks,
        **code_revision(),
    }
    (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    dest = root / bid
    if dest.exists():
        shutil.rmtree(tmp)
    else:
        tmp.rename(dest)
    return dest


def verify(bundle: Path) -> dict:
    """The manifest, after checking every file against its recorded hash."""
    manifest = json.loads((bundle / MANIFEST).read_text())
    for name, digest in manifest["files"].items():
        if sha256(bundle / name) != digest:
            raise ValueError(f"bundle {bundle}: {name} does not match its manifest hash")
    if bundle_id(manifest["files"]) != manifest["bundle_id"]:
        raise ValueError(f"bundle {bundle}: id does not match its file hashes")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--two-tower", required=True)
    parser.add_argument("--ranker", required=True)
    parser.add_argument("--out", default=str(BUNDLE_ROOT))
    args = parser.parse_args(argv)
    dest = build(Path(args.two_tower), Path(args.ranker), Path(args.out))
    manifest = verify(dest)
    print(json.dumps({"bundle": str(dest), "bundle_id": manifest["bundle_id"],
                      "export_checks": manifest["export_checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
