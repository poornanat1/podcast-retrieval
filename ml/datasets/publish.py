"""Publish datasets to the object store as immutable versions.

A dataset version, once published, never changes: re-publishing the same
version with identical content is a no-op, and attempting to overwrite it
with different content is an error. Configuration comes from the same
``OBJECT_STORE_*`` environment variables the Go services use, defaulting to
the local compose stack.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

from minio import Minio
from minio.error import S3Error

_FILES = ("train.parquet", "validation.parquet", "test.parquet", "manifest.json")


def _client() -> tuple[Minio, str]:
    endpoint = os.environ.get("OBJECT_STORE_ENDPOINT", "localhost:9000")
    bucket = os.environ.get("OBJECT_STORE_BUCKET_DATASETS", "datasets")
    client = Minio(
        endpoint,
        access_key=os.environ.get("OBJECT_STORE_ACCESS_KEY", "podfind"),
        secret_key=os.environ.get("OBJECT_STORE_SECRET_KEY", "podfind-dev-secret"),
        secure=os.environ.get("OBJECT_STORE_USE_SSL", "") == "true",
    )
    return client, bucket


def _remote_manifest(client: Minio, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        response = client.get_object(bucket, key)
        try:
            return json.load(response)
        finally:
            response.close()
            response.release_conn()
    except S3Error as exc:
        if exc.code == "NoSuchKey":
            return None
        raise


def publish_dataset(dataset_dir: Path, manifest: dict[str, Any]) -> str:
    """Upload a built dataset. Returns its object-store location."""
    client, bucket = _client()
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
        client.fput_object(bucket, f"{prefix}/{name}", str(path),
                           content_type="application/octet-stream")

    body = json.dumps(manifest, indent=2).encode()
    client.put_object(bucket, manifest_key, io.BytesIO(body), len(body),
                      content_type="application/json")
    return f"{bucket}/{prefix}"
