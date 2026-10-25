"""Load a serving bundle's vector indexes and rows into memory.

    uv run python -m ml.serving.prewarm [--bundle data/serving/bundles/<id>]

Every vector search visits about a thousand HNSW nodes and fetches each
node's row from ``episode_embeddings`` (HNSW has no index-only scans). That
table holds every model's vectors (about 15 GB), so on a cold cache a search
reads hundreds of random pages and takes ~250 ms instead of ~20 ms. This job
reads, with ``pg_prewarm``, the HNSW indexes of the bundle's two vector
models and the block ranges their rows (and the transcript vectors) occupy,
into the operating system's page cache. Run it after Postgres starts and
after promoting a bundle; it is idempotent.

It helps only if the hot set fits in memory: about 1.5 GB of index plus
1.3-1.6 GB of rows per model. A smaller footprint needs a table holding only
the served models' vectors (or half-precision vectors).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import psycopg

from ml.datasets.snapshot import DEFAULT_DATABASE_URL
from ml.embeddings.embed import index_name
from ml.serving.runtime import resolve_bundle

RANGES_SQL = """
SELECT model, (min(ctid)::text::point)[0]::bigint, (max(ctid)::text::point)[0]::bigint, count(*)
FROM episode_embeddings WHERE model = ANY(%(models)s) GROUP BY model
"""


def prewarm(conn: psycopg.Connection, models: dict[str, str]) -> dict:
    report: dict = {}
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
        for role, model in models.items():
            if role == "transcript":
                continue  # read by id only; no ANN index
            start = time.perf_counter()
            cur.execute("SELECT pg_prewarm(%s::regclass, 'read')", (index_name(model),))
            report[f"{role}_index_blocks"] = cur.fetchone()[0]
            report[f"{role}_index_s"] = round(time.perf_counter() - start, 1)
        cur.execute(RANGES_SQL, {"models": list(models.values())})
        for model, first, last, rows in cur.fetchall():
            role = next(r for r, m in models.items() if m == model)
            start = time.perf_counter()
            cur.execute("SELECT pg_prewarm('episode_embeddings', 'read', 'main', %s, %s)",
                        (first, last))
            report[f"{role}_rows"] = rows
            report[f"{role}_heap_blocks"] = cur.fetchone()[0]
            report[f"{role}_heap_s"] = round(time.perf_counter() - start, 1)
    conn.commit()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", default=os.environ.get("PODFIND_BUNDLE"))
    parser.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL))
    args = parser.parse_args(argv)
    bundle = resolve_bundle(args.bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    with psycopg.connect(args.database_url) as conn:
        report = prewarm(conn, manifest["index_models"])
    print(json.dumps({"bundle_id": manifest["bundle_id"], **report}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
