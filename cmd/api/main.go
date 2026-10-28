// Command api serves the public PodFind HTTP API and the web app.
//
// Configuration (environment):
//
//	DATABASE_URL                 Postgres DSN (default: local compose stack)
//	MODEL_SERVER_URL             model server base URL (default http://127.0.0.1:8081)
//	SPEECH_SERVER_URL            speech server base URL (default http://127.0.0.1:8082);
//	                             voice search is unavailable while it is down
//	OBJECT_STORE_*               object storage for voice recordings a user asked
//	                             to keep; unset refuses retention
//	PODFIND_API_ADDR             listen address (default :$PORT, or :8080)
//	PODFIND_LEXICAL_TIMEOUT_MS   lexical leg time limit (default 300)
//	PODFIND_VECTOR_TIMEOUT_MS    vector leg time limit (default 1500)
//	PODFIND_DEBUG                1 enables debug search responses and fixed
//	                             reference times (parity checks only)
//	PODFIND_LOG_FORMAT           json for one JSON object per line (the default
//	                             on Railway), text otherwise
//
// Prometheus metrics are served at /metrics (internal/metrics).
//
// The model bundle is whatever the model server serves; the API reads its
// manifest at startup for the index keys and candidate depth, and health
// fails if the server later serves a different bundle.
package main

import (
	"context"
	"errors"
	"log"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"podfind/internal/api"
	"podfind/internal/logging"
	"podfind/internal/metrics"
	"podfind/internal/migrate"
	"podfind/internal/objstore"
	"podfind/internal/search"
	"podfind/internal/version"
	"podfind/migrations"
	"podfind/web"
)

const defaultDatabaseURL = "postgres://podfind:podfind@localhost:5432/podfind?sslmode=disable"

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envMillis(key string, def int) time.Duration {
	n, err := strconv.Atoi(env(key, strconv.Itoa(def)))
	if err != nil || n <= 0 {
		n = def
	}
	return time.Duration(n) * time.Millisecond
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	logging.Setup()

	cfg, err := pgxpool.ParseConfig(env("DATABASE_URL", defaultDatabaseURL))
	if err != nil {
		log.Fatalf("database url: %v", err)
	}
	cfg.MaxConns = 16
	search.ConfigurePool(cfg)
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		log.Fatalf("database: %v", err)
	}
	defer pool.Close()
	if applied, err := migrate.Apply(ctx, pool, migrations.FS); err != nil {
		log.Fatalf("migrations: %v", err)
	} else if len(applied) > 0 {
		slog.Info("applied migrations", "versions", applied)
	}

	model := search.NewModelClient(env("MODEL_SERVER_URL", "http://127.0.0.1:8081"), 5*time.Second)
	var info search.ModelInfo
	for attempt := 1; ; attempt++ {
		info, err = model.Model(ctx)
		if err == nil || attempt == 30 || ctx.Err() != nil {
			break
		}
		time.Sleep(time.Second)
	}
	if err != nil {
		log.Fatalf("model server: %v", err)
	}

	store, err := objstore.NewFromEnv()
	if err != nil {
		log.Fatalf("object store: %v", err)
	}
	var recordings search.Recordings
	if store != nil {
		recordings = store
	}
	svc := &search.Service{
		Pool:       pool,
		Model:      model,
		Speech:     search.NewSpeechClient(env("SPEECH_SERVER_URL", "http://127.0.0.1:8082"), 30*time.Second),
		Recordings: recordings,
		Debug:      os.Getenv("PODFIND_DEBUG") == "1",
		Retriever: &search.Retriever{
			Pool: pool, Model: model, Info: info,
			LexicalTimeout: envMillis("PODFIND_LEXICAL_TIMEOUT_MS", 300),
			VectorTimeout:  envMillis("PODFIND_VECTOR_TIMEOUT_MS", 1500),
		},
	}
	metrics.BuildInfo.WithLabelValues(version.String(), info.BundleID, info.FeatureVersion).Set(1)
	go metrics.WatchIndexes(ctx, pool, info.IndexModels, 10*time.Minute)
	srv := &http.Server{
		Addr:              env("PODFIND_API_ADDR", ":"+env("PORT", "8080")),
		Handler:           api.New(svc, web.Static()),
		ReadHeaderTimeout: 5 * time.Second,
		WriteTimeout:      30 * time.Second,
	}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdown)
	}()
	slog.Info("podfind api", "version", version.String(), "addr", srv.Addr, "bundle", info.BundleID,
		"feature_version", info.FeatureVersion, "debug", svc.Debug, "recordings", store != nil)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("serve: %v", err)
	}
}
