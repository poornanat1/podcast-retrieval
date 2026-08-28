"""Export an immutable raw-data snapshot from the catalog database.

Dataset builds never read the live database: they read a snapshot, so the
same (config, code revision, snapshot) triple always produces an identical
dataset. Capturing a snapshot is the one non-deterministic step — it records
the catalog as of a moment — and everything downstream is pure.

    uv run python -m ml.datasets.snapshot --out data/snapshots

Writes ``data/snapshots/<snapshot_id>/`` containing ``episodes.parquet``,
``podcasts.parquet``, and ``manifest.json``; ``data/snapshots/latest``
points at the newest snapshot id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import psycopg

DEFAULT_DATABASE_URL = "postgres://podfind:podfind@localhost:5432/podfind"

EPISODES_QUERY = """
    SELECT e.id, e.podcast_id, e.title, e.description, e.language,
           e.duration_seconds, e.published_at, e.explicit, e.content_hash,
           (t.episode_id IS NOT NULL) AS has_transcript
    FROM episodes e
    LEFT JOIN transcripts t ON t.episode_id = e.id
    ORDER BY e.id
"""

PODCASTS_QUERY = """
    SELECT id, title, publisher, language, categories, explicit
    FROM podcasts
    ORDER BY id
"""


def _frame(conn: psycopg.Connection, query: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(query)
        columns = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columns)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture(database_url: str, out_root: Path) -> Path:
    """Export the catalog and return the snapshot directory."""
    with psycopg.connect(database_url) as conn:
        episodes = _frame(conn, EPISODES_QUERY)
        podcasts = _frame(conn, PODCASTS_QUERY)

    episodes["published_at"] = pd.to_datetime(episodes["published_at"], utc=True)

    tmp = out_root / ".snapshot-tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(tmp / "episodes.parquet", index=False)
    podcasts.to_parquet(tmp / "podcasts.parquet", index=False)

    file_hashes = {
        name: _sha256(tmp / name) for name in ("episodes.parquet", "podcasts.parquet")
    }
    snapshot_id = hashlib.sha256(
        json.dumps(file_hashes, sort_keys=True).encode()
    ).hexdigest()[:12]

    manifest = {
        "snapshot_id": snapshot_id,
        "created_at": datetime.now(UTC).isoformat(),
        "rows": {"episodes": len(episodes), "podcasts": len(podcasts)},
        "files": file_hashes,
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    dest = out_root / snapshot_id
    if dest.exists():
        # Identical content already captured; keep the immutable original.
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()
    else:
        tmp.rename(dest)
    (out_root / "latest").write_text(snapshot_id + "\n")
    return dest


def resolve(path: str | Path) -> Path:
    """Resolve a snapshot argument: a snapshot dir, or a root containing a
    ``latest`` pointer."""
    path = Path(path)
    if (path / "manifest.json").exists():
        return path
    pointer = path / "latest"
    if pointer.exists():
        return path / pointer.read_text().strip()
    raise FileNotFoundError(f"no snapshot at {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/snapshots")
    parser.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    )
    args = parser.parse_args(argv)

    dest = capture(args.database_url, Path(args.out))
    manifest = json.loads((dest / "manifest.json").read_text())
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
