#!/usr/bin/env bash
# Copies the served catalog from a source database into a target database:
# podcasts, episodes, transcripts, and the vectors of the served models
# only (the full embedding table holds every experiment's vectors). Then
# builds the served HNSW indexes and analyzes.
#
#   deploy/railway/copy-catalog.sh "$TARGET_DATABASE_URL"
#
# Environment: SOURCE_DATABASE_URL (default: the compose stack),
# BUNDLE (default: the promoted bundle; its manifest names the models).
# Needs psql, go, and uv on the path. Idempotent for the schema (migrations)
# and the indexes; the table copies append, so run it once into an empty
# target. The embeddings go over the wire in binary (about 3 GB for two
# models), the rest as text.
set -euo pipefail

TARGET=${1:?target DATABASE_URL}
SOURCE=${SOURCE_DATABASE_URL:-postgres://podfind:podfind@localhost:5432/podfind}
BUNDLE=${BUNDLE:-data/serving/bundles/$(cat data/serving/ACTIVE)}

models=$(uv run python - "$BUNDLE" <<'PY'
import json, sys
m = json.load(open(f"{sys.argv[1]}/manifest.json"))["index_models"]
print(",".join(sorted({v for v in m.values() if v})))
PY
)
echo "served models: $models"
IFS=',' read -r -a MODELS <<< "$models"
in_list=$(printf "'%s'," "${MODELS[@]}"); in_list=${in_list%,}

echo "== schema (migrations)"
DATABASE_URL="$TARGET" go run ./cmd/migrate

columns() { # non-generated columns of a table, comma-separated
  psql "$SOURCE" -Atc "SELECT string_agg(quote_ident(column_name), ',' ORDER BY ordinal_position)
    FROM information_schema.columns WHERE table_schema='public' AND table_name='$1' AND is_generated='NEVER'"
}
copy() { # table, where-clause
  local table=$1 where=${2:-} cols
  cols=$(columns "$table")
  echo "== $table ($where)"
  psql "$SOURCE" -Atc "COPY (SELECT $cols FROM $table $where) TO STDOUT (FORMAT binary)" \
    | psql "$TARGET" -c "COPY $table ($cols) FROM STDIN (FORMAT binary)"
}

copy podcasts
copy episodes
copy transcripts
copy episode_embeddings "WHERE model IN ($in_list)"

echo "== HNSW indexes"
for m in "${MODELS[@]}"; do
  name="episode_embeddings_hnsw_$(echo "$m" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g')"
  psql "$TARGET" -c "SET maintenance_work_mem = '2GB'; SET max_parallel_maintenance_workers = 4;
    CREATE INDEX IF NOT EXISTS $name ON episode_embeddings USING hnsw (embedding vector_cosine_ops) WHERE model = '$m';"
done

echo "== analyze"
psql "$TARGET" -c "ANALYZE podcasts; ANALYZE episodes; ANALYZE transcripts; ANALYZE episode_embeddings;"
psql "$TARGET" -Atc "SELECT 'podcasts', count(*) FROM podcasts UNION ALL SELECT 'episodes', count(*) FROM episodes
  UNION ALL SELECT 'transcripts', count(*) FROM transcripts UNION ALL SELECT 'vectors '||model, count(*) FROM episode_embeddings GROUP BY model"
echo "done; run 'make prewarm DATABASE_URL=<target>' before measuring latency"
