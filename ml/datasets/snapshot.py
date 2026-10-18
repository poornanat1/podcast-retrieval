"""Export an immutable raw-data snapshot from the catalog database.

Dataset builds never read the live database: they read a snapshot, so the
same (config, code revision, snapshot) triple always produces an identical
dataset. Capturing a snapshot is the one non-deterministic step — it records
the catalog as of a moment — and everything downstream is pure.

    uv run python -m ml.datasets.snapshot --out data/snapshots

Writes ``data/snapshots/<snapshot_id>/`` containing ``episodes.parquet`` (with
a bounded transcript excerpt per episode), ``podcasts.parquet``, and
``manifest.json``; ``data/snapshots/latest``
points at the newest snapshot id.

    uv run python -m ml.datasets.snapshot --with-transcripts-for data/snapshots/<id>

derives a snapshot from an existing one: the same episode and podcast rows,
with ``has_transcript`` and ``transcript_excerpt`` filled from the
database's current transcripts. Its manifest names the parent in
``derived_from``; a model trained on a dataset built from the parent can
read it, so a transcript ablation changes nothing but the input. It does
not move ``latest``.
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

# Transcripts are bounded to a head excerpt: the episode tower reads a few
# hundred tokens of transcript, and full transcripts would dominate the
# snapshot's size.
TRANSCRIPT_EXCERPT_CHARS = 4000

EPISODES_QUERY = f"""
    SELECT e.id, e.podcast_id, e.title, e.description, e.language,
           e.duration_seconds, e.published_at, e.explicit, e.content_hash,
           (t.episode_id IS NOT NULL) AS has_transcript,
           coalesce(left(t.content, {TRANSCRIPT_EXCERPT_CHARS}), '') AS transcript_excerpt
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
    dest = _write_snapshot(episodes, podcasts, out_root, {})
    (out_root / "latest").write_text(dest.name + "\n")
    return dest


def _write_snapshot(episodes: pd.DataFrame, podcasts: pd.DataFrame, out_root: Path,
                    extra: dict) -> Path:
    tmp = out_root / ".snapshot-tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(tmp / "episodes.parquet", index=False)
    podcasts.to_parquet(tmp / "podcasts.parquet", index=False)
    file_hashes = {name: _sha256(tmp / name) for name in ("episodes.parquet", "podcasts.parquet")}
    snapshot_id = hashlib.sha256(json.dumps(file_hashes, sort_keys=True).encode()).hexdigest()[:12]
    manifest = {
        "snapshot_id": snapshot_id,
        "created_at": datetime.now(UTC).isoformat(),
        "rows": {"episodes": len(episodes), "podcasts": len(podcasts)},
        "files": file_hashes,
        **extra,
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    dest = out_root / snapshot_id
    if dest.exists():
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()
    else:
        tmp.rename(dest)
    return dest


def with_transcripts(episodes: pd.DataFrame, transcripts: pd.DataFrame) -> pd.DataFrame:
    """The parent's rows with transcript columns from ``transcripts``
    (``episode_id``, ``transcript_excerpt``); row order and every other
    column unchanged."""
    excerpts = dict(zip(transcripts["episode_id"], transcripts["transcript_excerpt"],
                        strict=True))
    out = episodes.copy()
    out["transcript_excerpt"] = [excerpts.get(int(i), "") or "" for i in out["id"]]
    out["has_transcript"] = out["id"].map(lambda i: int(i) in excerpts)
    return out


def derive_with_transcripts(parent: Path, database_url: str, out_root: Path) -> Path:
    """Write a snapshot with ``parent``'s rows plus current transcript excerpts."""
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    episodes = pd.read_parquet(parent / "episodes.parquet")
    podcasts = pd.read_parquet(parent / "podcasts.parquet")
    with psycopg.connect(database_url) as conn:
        transcripts = _frame(conn, f"""
            SELECT episode_id, left(content, {TRANSCRIPT_EXCERPT_CHARS}) AS transcript_excerpt
            FROM transcripts ORDER BY episode_id""")
    ids = set(int(i) for i in episodes["id"])
    transcripts = transcripts[transcripts["episode_id"].isin(ids)]
    derived = with_transcripts(episodes, transcripts)
    return _write_snapshot(derived, podcasts, out_root, {
        "derived_from": parent_manifest["snapshot_id"],
        "derivation": "parent rows + current transcript excerpts",
        "transcripts": int(derived["has_transcript"].sum()),
    })


def compatible(snapshot_manifest: dict, dataset_snapshot_id: str) -> bool:
    """A dataset can be read against its own snapshot or one derived from it."""
    return dataset_snapshot_id in (snapshot_manifest["snapshot_id"],
                                   snapshot_manifest.get("derived_from"))


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
    parser.add_argument("--with-transcripts-for", default="",
                        help="derive a transcript snapshot from this snapshot dir")
    parser.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    )
    args = parser.parse_args(argv)

    if args.with_transcripts_for:
        dest = derive_with_transcripts(Path(args.with_transcripts_for), args.database_url,
                                       Path(args.out))
    else:
        dest = capture(args.database_url, Path(args.out))
    manifest = json.loads((dest / "manifest.json").read_text())
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
