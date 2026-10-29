package search

import (
	"context"
	"log/slog"
	"regexp"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

var nonAlnum = regexp.MustCompile(`[^a-z0-9]+`)

// IndexName is the HNSW index the embedding job creates for a model
// (ml.embeddings.embed.index_name), truncated as Postgres truncates
// identifiers.
func IndexName(model string) string {
	name := "episode_embeddings_hnsw_" + nonAlnum.ReplaceAllString(toLower(model), "_")
	if len(name) > 63 {
		name = name[:63]
	}
	return name
}

func toLower(s string) string {
	b := []byte(s)
	for i, c := range b {
		if 'A' <= c && c <= 'Z' {
			b[i] = c + 32
		}
	}
	return string(b)
}

// Prewarm reads the served vector indexes into the buffer cache and the
// rows and lexical indexes they visit into the operating system's cache,
// with pg_prewarm. Vector search is memory-bound: a cold HNSW search reads
// hundreds of random pages and takes ten times longer than a warm one, so
// a database restart otherwise costs minutes of timeouts. It is idempotent
// and safe to run from every API replica.
func Prewarm(ctx context.Context, pool *pgxpool.Pool, indexModels map[string]string) {
	start := time.Now()
	if _, err := pool.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS pg_prewarm"); err != nil {
		slog.Warn("prewarm: extension", "error", err)
		return
	}
	for _, leg := range []string{"two_tower", "e5"} {
		model := indexModels[leg]
		if model == "" {
			continue
		}
		var blocks int64
		if err := pool.QueryRow(ctx, "SELECT pg_prewarm($1::regclass)", IndexName(model)).Scan(&blocks); err != nil {
			slog.Warn("prewarm: index", "leg", leg, "index", IndexName(model), "error", err)
			continue
		}
		slog.Info("prewarm: index", "leg", leg, "blocks", blocks)
	}
	for _, rel := range []string{"episode_embeddings", "episodes", "episodes_search_idx", "transcripts_search_idx", "podcasts"} {
		var blocks int64
		if err := pool.QueryRow(ctx, "SELECT pg_prewarm($1::regclass, 'read')", rel).Scan(&blocks); err != nil {
			slog.Warn("prewarm: relation", "relation", rel, "error", err)
			continue
		}
		slog.Info("prewarm: relation", "relation", rel, "blocks", blocks)
	}
	slog.Info("prewarm done", "seconds", time.Since(start).Seconds())
}
