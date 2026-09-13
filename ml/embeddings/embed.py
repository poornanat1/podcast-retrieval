"""Batch-embed the episode catalog into pgvector.

    uv run python -m ml.embeddings.embed [--batch-size 512] [--limit 0]

Embeds every episode that has no vector for the model yet — or whose
content hash changed since it was embedded — so the job is resumable and
re-runs converge to a no-op. Finishes by ensuring the model's partial HNSW
index exists.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import psycopg
from pgvector.psycopg import register_vector

from ml.datasets.snapshot import DEFAULT_DATABASE_URL
from ml.embeddings.text import EMBEDDING_DIM, MODEL_ID, build_passage

PENDING_SQL = """
    SELECT e.id, e.title, e.description, e.content_hash, p.title AS podcast_title
    FROM episodes e
    JOIN podcasts p ON p.id = e.podcast_id
    LEFT JOIN episode_embeddings emb
           ON emb.episode_id = e.id AND emb.model = %(model)s
    WHERE emb.episode_id IS NULL OR emb.content_hash <> e.content_hash
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


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def index_name(model_id: str) -> str:
    return "episode_embeddings_hnsw_" + re.sub(r"[^a-z0-9]+", "_", model_id.lower())


def ensure_index(conn: psycopg.Connection, model_id: str) -> None:
    # DDL cannot take bound parameters; compose identifiers and the model
    # literal safely instead.
    statement = psycopg.sql.SQL(
        "CREATE INDEX IF NOT EXISTS {index} ON episode_embeddings "
        "USING hnsw (embedding vector_cosine_ops) WHERE model = {model}"
    ).format(
        index=psycopg.sql.Identifier(index_name(model_id)),
        model=psycopg.sql.Literal(model_id),
    )
    with conn.cursor() as cur:
        cur.execute("SET maintenance_work_mem = '512MB'")
        cur.execute(statement)
    conn.commit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="max episodes (0 = all)")
    parser.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    )
    args = parser.parse_args(argv)

    from sentence_transformers import SentenceTransformer

    device = pick_device()
    model = SentenceTransformer(args.model, device=device)
    if model.get_sentence_embedding_dimension() != EMBEDDING_DIM:
        raise SystemExit(
            f"model dimension {model.get_sentence_embedding_dimension()} != "
            f"schema dimension {EMBEDDING_DIM}"
        )

    embedded = 0
    started = time.time()
    with psycopg.connect(args.database_url) as conn:
        register_vector(conn)
        while True:
            if args.limit and embedded >= args.limit:
                break
            with conn.cursor() as cur:
                cur.execute(PENDING_SQL, {"model": args.model, "batch": args.batch_size})
                rows = cur.fetchall()
            if not rows:
                break

            passages = [build_passage(r[4], r[1], r[2]) or "passage: " for r in rows]
            vectors = model.encode(
                passages, batch_size=64, normalize_embeddings=True,
                show_progress_bar=False,
            )
            with conn.cursor() as cur:
                cur.executemany(
                    UPSERT_SQL,
                    [(row[0], args.model, vector, row[3])
                     for row, vector in zip(rows, vectors, strict=True)],
                )
            conn.commit()
            embedded += len(rows)
            if embedded % 10240 < args.batch_size:
                rate = embedded / max(time.time() - started, 1)
                print(f"embedded {embedded} ({rate:.0f}/s)", flush=True)

        ensure_index(conn, args.model)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM episode_embeddings WHERE model = %s",
                (args.model,),
            )
            total = cur.fetchone()[0]

    print(json.dumps({
        "model": args.model, "device": device, "embedded_now": embedded,
        "total_vectors": total, "seconds": round(time.time() - started, 1),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
