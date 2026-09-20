"""Build a versioned episode vector index from a two-tower model artifact.

    uv run python -m ml.embeddings.two_tower_index --artifact data/models/<artifact>

Encodes every episode through the artifact's episode tower — featurized by
the featurizer saved with it, so index-time features match training — and
upserts the vectors into ``episode_embeddings`` under the model id
``two-tower:<artifact name>``. The artifact name embeds the dataset version
and training-config hash, so each trained model gets its own index and
its own partial HNSW index. Like the pretrained job, this is resumable:
only episodes without a vector, or whose content changed, are encoded, and
the scan advances by episode id so already-indexed rows are skipped once
rather than re-scanned on every batch.

Publication age is computed against the index build time, recorded in
``index.json`` next to the artifact; an index that is not rebuilt ages
away from the features the model was trained on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import torch
from pgvector.psycopg import register_vector

from ml.datasets.snapshot import DEFAULT_DATABASE_URL, TRANSCRIPT_EXCERPT_CHARS
from ml.embeddings.embed import ensure_index, pick_device
from ml.embeddings.text import EMBEDDING_DIM
from ml.features import Featurizer
from ml.features.featurize import to_device
from ml.models.two_tower import TwoTowerModel

INDEX_FILENAME = "index.json"

PENDING_SQL = f"""
    SELECT e.id, e.podcast_id, e.title, e.description, e.language,
           e.duration_seconds, e.published_at, e.content_hash,
           p.title AS podcast_title, p.publisher, p.categories,
           coalesce(left(t.content, {TRANSCRIPT_EXCERPT_CHARS}), '') AS transcript_excerpt
    FROM episodes e
    JOIN podcasts p ON p.id = e.podcast_id
    LEFT JOIN transcripts t ON t.episode_id = e.id
    LEFT JOIN episode_embeddings emb
           ON emb.episode_id = e.id AND emb.model = %(model)s
    WHERE e.id > %(after)s
      AND (emb.episode_id IS NULL OR emb.content_hash <> e.content_hash)
    ORDER BY e.id
    LIMIT %(batch)s
"""

UPSERT_SQL = """
    INSERT INTO episode_embeddings (episode_id, model, embedding, content_hash)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (episode_id, model) DO UPDATE
    SET embedding = EXCLUDED.embedding,
        content_hash = EXCLUDED.content_hash,
        created_at = now()
"""

COLUMNS = (
    "id", "podcast_id", "title", "description", "language", "duration_seconds",
    "published_at", "content_hash", "podcast_title", "publisher", "categories",
    "transcript_excerpt",
)


def model_id_for_artifact(artifact: str | Path) -> str:
    """The ``episode_embeddings.model`` key for an artifact's index."""
    return f"two-tower:{Path(artifact).name}"


def rows_to_episodes(rows: list[tuple]) -> list[dict]:
    return [dict(zip(COLUMNS, row, strict=True)) for row in rows]


@torch.no_grad()
def encode_episodes(
    model: TwoTowerModel, featurizer: Featurizer, episodes: list[dict],
    reference_time: datetime, device: str, encode_batch: int = 64,
) -> torch.Tensor:
    vectors = []
    for start in range(0, len(episodes), encode_batch):
        batch = featurizer.episode_batch(episodes[start:start + encode_batch], reference_time)
        vectors.append(model.encode_episode(to_device(batch, device)).cpu())
    return torch.cat(vectors)


def build_index(
    artifact: Path, database_url: str, batch_size: int = 256, limit: int = 0,
    device: str | None = None,
) -> dict:
    device = device or pick_device()
    model = TwoTowerModel.load(artifact).to(device).eval()
    featurizer = Featurizer.load(artifact)
    if model.config.embedding_dim != EMBEDDING_DIM:
        raise SystemExit(
            f"model dimension {model.config.embedding_dim} != schema dimension {EMBEDDING_DIM}"
        )
    model_id = model_id_for_artifact(artifact)
    built_at = datetime.now(UTC)

    embedded = 0
    after = 0  # keyset cursor: each batch scans forward from the last id
    started = time.time()
    with psycopg.connect(database_url) as conn:
        register_vector(conn)
        while not limit or embedded < limit:
            with conn.cursor() as cur:
                cur.execute(PENDING_SQL, {"model": model_id, "batch": batch_size, "after": after})
                rows = cur.fetchall()
            if not rows:
                break
            after = rows[-1][0]
            episodes = rows_to_episodes(rows)
            vectors = encode_episodes(model, featurizer, episodes, built_at, device).numpy()
            with conn.cursor() as cur:
                cur.executemany(
                    UPSERT_SQL,
                    [(ep["id"], model_id, vector, ep["content_hash"])
                     for ep, vector in zip(episodes, vectors, strict=True)],
                )
            conn.commit()
            embedded += len(rows)
            if embedded % (batch_size * 40) < batch_size:
                rate = embedded / max(time.time() - started, 1)
                print(f"embedded {embedded} ({rate:.0f}/s)", flush=True)

        ensure_index(conn, model_id)
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM episode_embeddings WHERE model = %s", (model_id,))
            total = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM episodes")
            catalog_size = cur.fetchone()[0]

    artifact_meta = json.loads((artifact / "artifact.json").read_text())
    record = {
        "model": model_id,
        "artifact": str(artifact),
        "feature_contract_sha256": artifact_meta.get("feature_contract_sha256"),
        "dataset_version": artifact_meta.get("dataset_version"),
        "built_at": built_at.isoformat(),
        "reference_time": built_at.isoformat(),
        "device": device,
        "embedded_now": embedded,
        "total_vectors": total,
        "catalog_size": catalog_size,
        "seconds": round(time.time() - started, 1),
    }
    (artifact / INDEX_FILENAME).write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="max episodes (0 = all)")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    )
    args = parser.parse_args(argv)
    record = build_index(
        Path(args.artifact), args.database_url, args.batch_size, args.limit, args.device
    )
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
