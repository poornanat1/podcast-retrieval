# Embeddings

## Model

`intfloat/multilingual-e5-small`: 12 layers, 384 dimensions, ~118M
parameters (96M of them the token embedding table), mean pooling, trained
for asymmetric retrieval with `query:` / `passage:` prefixes.

Why this one:

1. **The catalog is multilingual** — about 77% English with real Spanish,
   French, German, and Portuguese segments — and language is a hard
   filter, so an English-only encoder would degrade a fifth of the catalog
   and every filtered query.
2. **Retrieval-trained**, not a paraphrase encoder: the E5 family is
   contrastively trained for query→passage retrieval.
3. **384 dimensions** keep ~800k vectors around 1.2 GB plus an HNSW index
   that fits in memory next to the database.
4. **Small enough for both towers**: queries encode in milliseconds on CPU
   and the whole catalog in about an hour on a laptop GPU, so full
   re-indexing is routine.
5. **Self-hosted**: deterministic, versioned, no catalog text leaves the
   stack.

Larger E5 variants would likely rank better; the project's rule is that a
fine-tuned model must beat the *small* pretrained model first.

## Text

`ml/embeddings/text.py` builds `passage: <show title>. <episode title>.
<description[:1000]>` for episodes and `query: <text>` for queries. The
two-tower model uses a richer passage (categories and publisher before the
description) defined by its feature contract; see
[08-models.md](08-models.md).

## Jobs

`make embed` (`ml/embeddings/embed.py`) encodes every episode without a
vector for the model, or whose `content_hash` changed since it was
encoded, in batches of 512, and finishes by creating the model's partial
HNSW index. It advances by episode id, so re-runs skip indexed rows once
rather than re-scanning them per batch, and converge to a no-op.

`make index ARTIFACT=<dir>` (`ml/embeddings/two_tower_index.py`) does the
same through a two-tower artifact's episode tower, under the model key
`two-tower:<artifact name>`, and writes `index.json` next to the artifact
(build time, contract fingerprint, vector count). Publication age is
computed against the build time.

Measured on an Apple laptop GPU: ~300 episodes/s for E5 (full catalog
under an hour) and ~200/s through the two-tower episode tower; the HNSW
build for 797k vectors takes about 40 minutes.

## Storage

`episode_embeddings(episode_id, model, embedding vector(384), content_hash, created_at)`
with primary key `(episode_id, model)` and one partial HNSW index per
model (`vector_cosine_ops`, `WHERE model = '<model>'`). Vectors are unit
length; search uses cosine distance (`<=>`). `content_hash` is the hash of
the episode's searchable fields at encode time, which is what makes
re-encoding incremental.
