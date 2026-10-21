"""Embed transcript excerpts with the pretrained E5 model.

    uv run python -m ml.embeddings.transcripts [--batch-size 256] [--limit 0]

One vector per episode with a transcript: the first
``TRANSCRIPT_PASSAGE_CHARS`` characters of the transcript as an E5
passage, stored in ``episode_embeddings`` under ``TRANSCRIPT_MODEL_ID``.
The ranker reads it by episode id for a transcript-to-query similarity
feature; nothing searches it, so no ANN index is built. Resumable like
``ml.embeddings.embed``: an episode is re-embedded only when its excerpt
changes (its hash is the row's ``content_hash``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import psycopg
from pgvector.psycopg import register_vector

from ml.datasets.snapshot import DEFAULT_DATABASE_URL
from ml.embeddings.embed import UPSERT_SQL, pick_device
from ml.embeddings.text import MODEL_ID

TRANSCRIPT_MODEL_ID = f"{MODEL_ID}#transcript"
# About the 512-token limit of the encoder for English speech.
TRANSCRIPT_PASSAGE_CHARS = 2000

PENDING_SQL = """
    SELECT t.episode_id, left(t.content, %(chars)s) AS excerpt, emb.content_hash
    FROM transcripts t
    LEFT JOIN episode_embeddings emb
           ON emb.episode_id = t.episode_id AND emb.model = %(model)s
    WHERE t.episode_id > %(after)s
    ORDER BY t.episode_id
    LIMIT %(batch)s
"""


def transcript_passage(excerpt: str) -> str:
    return "passage: " + " ".join((excerpt or "").split())


def excerpt_hash(excerpt: str) -> str:
    return hashlib.sha256((excerpt or "").encode()).hexdigest()[:32]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="max transcripts (0 = all)")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    )
    args = parser.parse_args(argv)

    from sentence_transformers import SentenceTransformer

    device = args.device or pick_device()
    model = SentenceTransformer(MODEL_ID, device=device)

    embedded = skipped = 0
    after = 0
    started = time.time()
    with psycopg.connect(args.database_url) as conn:
        register_vector(conn)
        while not (args.limit and embedded >= args.limit):
            with conn.cursor() as cur:
                cur.execute(PENDING_SQL, {"model": TRANSCRIPT_MODEL_ID, "chars":
                                          TRANSCRIPT_PASSAGE_CHARS, "after": after,
                                          "batch": args.batch_size})
                rows = cur.fetchall()
            if not rows:
                break
            after = rows[-1][0]
            todo = [(eid, excerpt) for eid, excerpt, stored in rows
                    if stored != excerpt_hash(excerpt)]
            skipped += len(rows) - len(todo)
            if not todo:
                continue
            vectors = model.encode([transcript_passage(x) for _, x in todo], batch_size=32,
                                   normalize_embeddings=True, show_progress_bar=False)
            with conn.cursor() as cur:
                cur.executemany(UPSERT_SQL, [
                    (eid, TRANSCRIPT_MODEL_ID, vector, excerpt_hash(excerpt))
                    for (eid, excerpt), vector in zip(todo, vectors, strict=True)
                ])
            conn.commit()
            embedded += len(todo)
            if embedded % 5120 < len(todo):
                rate = embedded / max(time.time() - started, 1)
                print(f"embedded {embedded} ({rate:.0f}/s)", flush=True)

    print(json.dumps({"model": TRANSCRIPT_MODEL_ID, "device": device,
                      "embedded_now": embedded, "unchanged": skipped}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
