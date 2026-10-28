package metrics

import (
	"context"
	"log/slog"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// WatchIndexes refreshes the index age and row gauges for the served vector
// models every interval until ctx ends. indexModels maps a leg (two_tower,
// e5, transcript) to the model id whose vectors it reads.
func WatchIndexes(ctx context.Context, pool *pgxpool.Pool, indexModels map[string]string, interval time.Duration) {
	ids := make([]string, 0, len(indexModels))
	for _, id := range indexModels {
		if id != "" {
			ids = append(ids, id)
		}
	}
	for {
		refreshIndexes(ctx, pool, ids)
		select {
		case <-ctx.Done():
			return
		case <-time.After(interval):
		}
	}
}

func refreshIndexes(ctx context.Context, pool *pgxpool.Pool, ids []string) {
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	rows, err := pool.Query(ctx, `
		SELECT model, count(*), extract(epoch FROM now() - max(created_at))
		FROM episode_embeddings WHERE model = ANY($1) GROUP BY model`, ids)
	if err != nil {
		slog.Warn("index metrics", "error", err)
		return
	}
	defer rows.Close()
	seen := map[string]bool{}
	for rows.Next() {
		var model string
		var n int64
		var age float64
		if err := rows.Scan(&model, &n, &age); err != nil {
			slog.Warn("index metrics", "error", err)
			return
		}
		IndexRows.WithLabelValues(model).Set(float64(n))
		IndexAge.WithLabelValues(model).Set(age)
		seen[model] = true
	}
	for _, id := range ids {
		if !seen[id] {
			IndexRows.WithLabelValues(id).Set(0)
		}
	}
}
