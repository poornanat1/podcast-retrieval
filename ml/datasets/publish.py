"""Publish datasets to the object store as immutable versions.

A dataset version, once published, never changes: re-publishing the same
version with identical content is a no-op, and attempting to overwrite it
with different content is an error. Configuration comes from the same
``OBJECT_STORE_*`` environment variables the Go services use, defaulting to
the local compose stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.objstore import client_from_env

_FILES = ("train.parquet", "validation.parquet", "test.parquet", "manifest.json")


def _remote_manifest(client, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except client.exceptions.NoSuchKey:
        return None
    return json.load(response["Body"])


def publish_dataset(dataset_dir: Path, manifest: dict[str, Any]) -> str:
    """Upload a built dataset. Returns its object-store location."""
    client, bucket = client_from_env("OBJECT_STORE_BUCKET_DATASETS", "datasets")
    prefix = f"{manifest['name']}/{manifest['dataset_version']}"
    manifest_key = f"{prefix}/manifest.json"

    existing = _remote_manifest(client, bucket, manifest_key)
    if existing is not None:
        if existing.get("content_hash") == manifest["content_hash"]:
            return f"{bucket}/{prefix} (already published)"
        raise RuntimeError(
            f"immutable dataset version {prefix} already exists with different "
            f"content ({existing.get('content_hash')!r} != {manifest['content_hash']!r})"
        )

    for name in _FILES:
        path = dataset_dir / name
        if name == "manifest.json":
            continue  # uploaded last, marking the version complete
        client.upload_file(str(path), bucket, f"{prefix}/{name}",
                           ExtraArgs={"ContentType": "application/octet-stream"})

    body = json.dumps(manifest, indent=2).encode()
    client.put_object(Bucket=bucket, Key=manifest_key, Body=body,
                      ContentType="application/json")
    return f"{bucket}/{prefix}"
