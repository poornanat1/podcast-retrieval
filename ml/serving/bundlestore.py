"""Publish serving bundles to the object store and fetch them back.

    uv run python -m ml.serving.bundlestore publish [--bundle data/serving/bundles/<id>]
    uv run python -m ml.serving.bundlestore fetch --bundle-id <id> [--dest /data/bundles]

A bundle is immutable and named by the hash of its files, so publishing
the same id twice is a no-op and the fetch verifies every file against the
manifest before the model server will load it. The bucket is
``OBJECT_STORE_BUCKET_MODELS`` (default ``models``); objects live under
``bundles/<id>/<file>``.

The model server container calls ``fetch`` at start when
``PODFIND_BUNDLE_ID`` is set, into a volume, so a restart with the same
bundle costs nothing and a new bundle is a variable change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ml.objstore import client_from_env
from ml.serving.bundle import MANIFEST, verify
from ml.serving.runtime import resolve_bundle

PREFIX = "bundles"


def publish(bundle: Path) -> str:
    manifest = verify(bundle)
    client, bucket = client_from_env("OBJECT_STORE_BUCKET_MODELS", "models")
    prefix = f"{PREFIX}/{manifest['bundle_id']}"
    try:
        client.head_object(Bucket=bucket, Key=f"{prefix}/{MANIFEST}")
        return f"{bucket}/{prefix} (already published)"
    except client.exceptions.ClientError as err:
        if err.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey", "NotFound"):
            raise
    # The manifest goes last: its presence marks the bundle complete.
    for name in manifest["files"]:
        client.upload_file(str(bundle / name), bucket, f"{prefix}/{name}")
    client.upload_file(str(bundle / MANIFEST), bucket, f"{prefix}/{MANIFEST}",
                       ExtraArgs={"ContentType": "application/json"})
    return f"{bucket}/{prefix}"


def fetch(bundle_id: str, dest: Path) -> Path:
    """Download the bundle into ``dest/<id>`` unless it is already there
    and verifies; returns the bundle directory."""
    target = dest / bundle_id
    if (target / MANIFEST).exists():
        try:
            verify(target)
            return target
        except (ValueError, OSError, KeyError):
            pass  # partial or corrupt: fetch again
    client, bucket = client_from_env("OBJECT_STORE_BUCKET_MODELS", "models")
    prefix = f"{PREFIX}/{bundle_id}"
    tmp = dest / f".{bundle_id}.partial"
    tmp.mkdir(parents=True, exist_ok=True)
    manifest_body = client.get_object(Bucket=bucket, Key=f"{prefix}/{MANIFEST}")["Body"].read()
    manifest = json.loads(manifest_body)
    if manifest["bundle_id"] != bundle_id:
        raise ValueError(f"object {prefix} holds bundle {manifest['bundle_id']}, not {bundle_id}")
    for name in manifest["files"]:
        client.download_file(bucket, f"{prefix}/{name}", str(tmp / name))
    (tmp / MANIFEST).write_bytes(manifest_body)
    verify(tmp)
    if target.exists():
        import shutil
        shutil.rmtree(target)
    tmp.rename(target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    pub = sub.add_parser("publish", help="upload a bundle")
    pub.add_argument("--bundle", default=None, help="bundle directory (default: the promoted one)")
    get = sub.add_parser("fetch", help="download a bundle")
    get.add_argument("--bundle-id", required=True)
    get.add_argument("--dest", default="data/serving/bundles")
    args = parser.parse_args(argv)
    if args.command == "publish":
        print(publish(resolve_bundle(args.bundle)))
    else:
        print(fetch(args.bundle_id, Path(args.dest)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
