// Command catalog-worker discovers podcasts through the Particle data
// platform and keeps their RSS feeds ingested, deduplicated, and on a
// polling schedule.
//
// Configuration (environment):
//
//	DATABASE_URL              Postgres DSN (default: local compose stack)
//	PARTICLE_API_KEY          Particle API key; discovery jobs dead-letter
//	                          when unset
//	PODFIND_DAILY_TRENDING    size of the recurring daily trending discovery
//	                          (default 25; 0 disables; needs the API key)
//
// Flags enqueue one-off discovery work on startup, e.g.:
//
//	catalog-worker -discover "machine learning" -trending 50
package main

import (
	"context"
	"flag"
	"log"
	"log/slog"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"podfind/internal/catalog"
	"podfind/internal/jobs"
	"podfind/internal/migrate"
	"podfind/internal/objstore"
	"podfind/internal/outbox"
	"podfind/internal/particle"
	"podfind/internal/rss"
	"podfind/internal/version"
	"podfind/migrations"
)

const defaultDatabaseURL = "postgres://podfind:podfind@localhost:5432/podfind?sslmode=disable"

func main() {
	discoverQuery := flag.String("discover", "", "enqueue a discovery search for this query on startup")
	trendingMax := flag.Int("trending", 0, "enqueue a trending-podcasts discovery of this size on startup")
	language := flag.String("language", "", "language filter for -trending")
	seed := flag.Bool("seed", false, "enqueue the standard catalog-seeding discovery set")
	oneshot := flag.Bool("oneshot", false, "enqueue requested jobs and exit without processing")
	flag.Parse()

	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	logger.Info("catalog-worker starting", "version", version.String())

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	url := os.Getenv("DATABASE_URL")
	if url == "" {
		url = defaultDatabaseURL
	}
	pool, err := pgxpool.New(ctx, url)
	if err != nil {
		log.Fatalf("connect: %v", err)
	}
	defer pool.Close()

	if applied, err := migrate.Apply(ctx, pool, migrations.FS); err != nil {
		log.Fatalf("migrate: %v", err)
	} else if len(applied) > 0 {
		logger.Info("applied migrations", "count", len(applied))
	}

	var provider *particle.Client
	if key := os.Getenv("PARTICLE_API_KEY"); key != "" {
		provider = &particle.Client{APIKey: key}
	} else {
		logger.Warn("PARTICLE_API_KEY not set; discovery jobs will dead-letter")
	}

	objects, err := objstore.NewFromEnv()
	if err != nil {
		log.Fatalf("object store: %v", err)
	}
	if objects == nil {
		logger.Warn("OBJECT_STORE_ENDPOINT not set; raw transcripts will not be retained")
	}

	queue := jobs.New(pool)
	worker := &catalog.Worker{
		Pool:    pool,
		Queue:   queue,
		Fetcher: &rss.Fetcher{},
		Objects: objects,
		Log:     logger,
	}
	if provider != nil {
		worker.Discovery = provider
	}

	if *discoverQuery != "" {
		if _, err := queue.Enqueue(ctx, catalog.JobDiscover,
			catalog.DiscoverPayload{Query: *discoverQuery}); err != nil {
			log.Fatalf("enqueue discovery: %v", err)
		}
	}
	if *trendingMax > 0 {
		if _, err := queue.Enqueue(ctx, catalog.JobDiscover,
			catalog.DiscoverPayload{Trending: true, Max: *trendingMax, Language: *language}); err != nil {
			log.Fatalf("enqueue trending discovery: %v", err)
		}
	}
	if *seed {
		n, err := worker.EnqueueSeed(ctx)
		if err != nil {
			log.Fatalf("enqueue seed: %v", err)
		}
		logger.Info("seed discovery enqueued", "jobs", n)
	}
	if *oneshot {
		logger.Info("oneshot: jobs enqueued, exiting")
		return
	}
	if err := worker.EnqueueSchedulerTick(ctx, time.Now()); err != nil {
		log.Fatalf("enqueue scheduler: %v", err)
	}

	dailyTrending := 25
	if v := os.Getenv("PODFIND_DAILY_TRENDING"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 0 {
			log.Fatalf("invalid PODFIND_DAILY_TRENDING %q", v)
		}
		dailyTrending = n
	}
	if provider != nil && dailyTrending > 0 {
		if err := worker.EnqueueDailyDiscovery(ctx, time.Now(), dailyTrending, ""); err != nil {
			log.Fatalf("enqueue daily discovery: %v", err)
		}
		logger.Info("daily trending discovery enabled", "max", dailyTrending)
	}

	concurrency := 4
	if v := os.Getenv("PODFIND_WORKER_CONCURRENCY"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 {
			log.Fatalf("invalid PODFIND_WORKER_CONCURRENCY %q", v)
		}
		concurrency = n
	}

	// Drain content-change events; the logging consumer is replaced by the
	// embedding pipeline once search artifacts exist.
	outboxWorker := &outbox.Worker{
		Pool: pool,
		Log:  logger,
		Handlers: map[string]outbox.Handler{
			catalog.EventContentChanged: func(ctx context.Context, e outbox.Event) error {
				logger.Debug("content changed", "event", e.ID, "payload", string(e.Payload))
				return nil
			},
		},
	}
	go outboxWorker.Run(ctx)

	runner := &jobs.Runner{
		Queue:       queue,
		Handlers:    worker.Handlers(),
		Concurrency: concurrency,
		Log:         logger,
	}
	if err := runner.Run(ctx); err != nil && ctx.Err() == nil {
		log.Fatalf("runner: %v", err)
	}
	logger.Info("catalog-worker stopped")
}
