"""S3-compatible object storage, configured like the Go services.

``OBJECT_STORE_ENDPOINT`` is a plain ``host:port`` for a local store
(addressed by path) or a URL for a managed bucket (addressed by virtual
host); ``OBJECT_STORE_USE_SSL`` upgrades a plain endpoint to https and
``OBJECT_STORE_PATH_STYLE`` overrides the addressing. Credentials come
from ``OBJECT_STORE_ACCESS_KEY`` / ``OBJECT_STORE_SECRET_KEY`` and the
signing region from ``OBJECT_STORE_REGION`` (default ``us-east-1``).
Defaults point at the compose stack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    endpoint: str
    access_key: str
    secret_key: str
    region: str
    path_style: bool


def endpoint_url(raw: str, ssl: bool) -> tuple[str, bool]:
    """The endpoint as a URL, and whether to address buckets by path."""
    if "://" in raw:
        return raw, False
    return ("https://" if ssl else "http://") + raw, True


def config_from_env() -> Config:
    raw = os.environ.get("OBJECT_STORE_ENDPOINT", "localhost:9000")
    endpoint, path_style = endpoint_url(raw, os.environ.get("OBJECT_STORE_USE_SSL", "") == "true")
    if (override := os.environ.get("OBJECT_STORE_PATH_STYLE", "")) != "":
        path_style = override == "true"
    return Config(
        endpoint=endpoint,
        access_key=os.environ.get("OBJECT_STORE_ACCESS_KEY", "podfind"),
        secret_key=os.environ.get("OBJECT_STORE_SECRET_KEY", "podfind-dev-secret"),
        region=os.environ.get("OBJECT_STORE_REGION", "us-east-1"),
        path_style=path_style,
    )


def client(config: Config | None = None):
    """A boto3 S3 client for the configured store."""
    import boto3
    from botocore.config import Config as BotoConfig

    config = config or config_from_env()
    return boto3.client(
        "s3",
        endpoint_url=config.endpoint,
        aws_access_key_id=config.access_key,
        aws_secret_access_key=config.secret_key,
        region_name=config.region,
        config=BotoConfig(s3={"addressing_style": "path" if config.path_style else "virtual"},
                          retries={"max_attempts": 5, "mode": "standard"}),
    )


def client_from_env(bucket_var: str, default_bucket: str):
    """``(client, bucket)`` with the bucket named by ``bucket_var``."""
    return client(), os.environ.get(bucket_var, default_bucket)
