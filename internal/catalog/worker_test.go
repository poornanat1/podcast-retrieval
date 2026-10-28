package catalog_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"podfind/internal/catalog"
	"podfind/internal/jobs"
	"podfind/internal/particle"
	"podfind/internal/pgtest"
	"podfind/internal/rss"
)

func startWorker(t *testing.T, pool *pgxpool.Pool) (*catalog.Worker, *jobs.Queue, context.Context) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)

	queue := jobs.New(pool, jobs.WithBackoff(0, 0))
	worker := &catalog.Worker{Pool: pool, Queue: queue, Fetcher: &rss.Fetcher{AllowPrivate: true}}
	runner := &jobs.Runner{
		Queue:        queue,
		Handlers:     worker.Handlers(),
		PollInterval: 20 * time.Millisecond,
	}
	go runner.Run(ctx)
	return worker, queue, ctx
}

func waitFor(t *testing.T, ctx context.Context, what string, check func() bool) {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for !check() {
		if time.Now().After(deadline) || ctx.Err() != nil {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// End-to-end done check: refreshing the same feed twice through the runner
// produces zero duplicate episodes, and the second pass rides a 304.
func TestRefreshTwiceEndToEndNoDuplicates(t *testing.T) {
	pool := pgtest.Pool(t, "catalog_worker")
	ctx := context.Background()

	fixture, err := os.ReadFile("../rss/testdata/feed.xml")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.Header.Get("If-None-Match") == `"v1"` {
			w.WriteHeader(http.StatusNotModified)
			return
		}
		w.Header().Set("ETag", `"v1"`)
		w.Write(fixture)
	}))
	defer server.Close()

	podcastID := insertPodcast(t, pool, server.URL)
	worker, queue, runCtx := startWorker(t, pool)
	_ = worker

	enqueueRefresh := func(key string) {
		t.Helper()
		if _, err := queue.Enqueue(ctx, catalog.JobFeedRefresh,
			catalog.FeedRefreshPayload{PodcastID: podcastID},
			jobs.WithIdempotencyKey(key)); err != nil {
			t.Fatalf("enqueue: %v", err)
		}
	}
	completedRefreshes := func() int {
		var n int
		if err := pool.QueryRow(ctx,
			"SELECT count(*) FROM jobs WHERE job_type = $1 AND status = 'completed'",
			catalog.JobFeedRefresh).Scan(&n); err != nil {
			t.Fatalf("count jobs: %v", err)
		}
		return n
	}

	enqueueRefresh("refresh-1")
	waitFor(t, runCtx, "first refresh", func() bool { return completedRefreshes() == 1 })

	enqueueRefresh("refresh-2")
	waitFor(t, runCtx, "second refresh", func() bool { return completedRefreshes() == 2 })

	if n := countEpisodes(t, pool, podcastID); n != 3 {
		t.Fatalf("episode count = %d after two refreshes, want 3", n)
	}

	// Fetch state advanced: validators stored, next poll scheduled.
	var etag string
	var nextFetch *time.Time
	if err := pool.QueryRow(ctx,
		"SELECT etag, next_fetch_at FROM podcasts WHERE id = $1", podcastID,
	).Scan(&etag, &nextFetch); err != nil {
		t.Fatalf("read podcast: %v", err)
	}
	if etag != `"v1"` {
		t.Errorf("etag = %q, want stored validator", etag)
	}
	if nextFetch == nil || !nextFetch.After(time.Now()) {
		t.Errorf("next_fetch_at = %v, want in the future", nextFetch)
	}

	// Channel metadata landed on the podcast row.
	var title string
	if err := pool.QueryRow(ctx,
		"SELECT title FROM podcasts WHERE id = $1", podcastID).Scan(&title); err != nil {
		t.Fatalf("read title: %v", err)
	}
	if title != "Practical AI Weekly" {
		t.Errorf("podcast title = %q", title)
	}
}

// End-to-end: a recurring trending discovery registers the feed, the chain
// ingests its episodes hands-off, and the next daily tick is scheduled.
func TestRecurringDiscoveryIngestsAndReenqueues(t *testing.T) {
	pool := pgtest.Pool(t, "catalog_worker")
	ctx := context.Background()

	fixture, err := os.ReadFile("../rss/testdata/feed.xml")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	feedServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(fixture)
	}))
	defer feedServer.Close()
	providerServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/podcasts" || r.URL.Query().Get("sort") != "popularity" {
			t.Errorf("unexpected request: %s %s", r.URL.Path, r.URL.RawQuery)
		}
		fmt.Fprintf(w, `{"data":[{"id":"pt7","title":"Practical AI Weekly","url":%q,"language":"en","popularity":0.91,"topics":[{"name":"Technology"}]}],"has_more":false}`,
			feedServer.URL)
	}))
	defer providerServer.Close()

	runCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	queue := jobs.New(pool, jobs.WithBackoff(0, 0))
	worker := &catalog.Worker{
		Pool:    pool,
		Queue:   queue,
		Fetcher: &rss.Fetcher{AllowPrivate: true},
		Discovery: &particle.Client{
			APIKey: "test-key", BaseURL: providerServer.URL,
			MinInterval: time.Millisecond,
		},
	}
	runner := &jobs.Runner{
		Queue:        queue,
		Handlers:     worker.Handlers(),
		PollInterval: 20 * time.Millisecond,
	}
	go runner.Run(runCtx)

	if err := worker.EnqueueDailyDiscovery(ctx, time.Now(), 5, ""); err != nil {
		t.Fatalf("enqueue daily discovery: %v", err)
	}

	// The discovered feed's episodes arrive without any further action.
	var podcastID int64
	waitFor(t, runCtx, "feed to be registered and ingested", func() bool {
		if err := pool.QueryRow(ctx,
			"SELECT id FROM podcasts WHERE feed_url = $1", feedServer.URL,
		).Scan(&podcastID); err != nil {
			return false
		}
		return countEpisodes(t, pool, podcastID) == 3
	})

	// Provider attribution and popularity are recorded.
	var source, discoveryID string
	var popularity float64
	if err := pool.QueryRow(ctx,
		"SELECT discovery_source, discovery_id, popularity FROM podcasts WHERE id = $1", podcastID,
	).Scan(&source, &discoveryID, &popularity); err != nil {
		t.Fatalf("read discovery attribution: %v", err)
	}
	if source != "particle" || discoveryID != "pt7" || popularity != 0.91 {
		t.Fatalf("attribution = (%q, %q, %v), want (particle, pt7, 0.91)", source, discoveryID, popularity)
	}

	// Tomorrow's tick is pending roughly a day out.
	var runAt time.Time
	waitFor(t, runCtx, "next daily tick to be scheduled", func() bool {
		err := pool.QueryRow(ctx, `
			SELECT run_at FROM jobs
			WHERE job_type = $1 AND status = 'pending'
			  AND payload->>'recurring' = 'true'`,
			catalog.JobDiscover).Scan(&runAt)
		return err == nil
	})
	if until := time.Until(runAt); until < 20*time.Hour || until > 25*time.Hour {
		t.Fatalf("next tick scheduled %v out, want ~24h", until)
	}
}

// End-to-end done check: a malformed feed dead-letters its job with the
// error retained; the worker keeps running and processes other feeds.
func TestMalformedFeedDeadLettersWithoutCrashing(t *testing.T) {
	pool := pgtest.Pool(t, "catalog_worker")
	ctx := context.Background()

	badServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("definitely not <valid xml"))
	}))
	defer badServer.Close()
	fixture, err := os.ReadFile("../rss/testdata/feed.xml")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	goodServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(fixture)
	}))
	defer goodServer.Close()

	badID := insertPodcast(t, pool, badServer.URL)
	goodID := insertPodcast(t, pool, goodServer.URL)
	_, queue, runCtx := startWorker(t, pool)

	for key, id := range map[string]int64{"bad": badID, "good": goodID} {
		if _, err := queue.Enqueue(ctx, catalog.JobFeedRefresh,
			catalog.FeedRefreshPayload{PodcastID: id},
			jobs.WithIdempotencyKey(key)); err != nil {
			t.Fatalf("enqueue %s: %v", key, err)
		}
	}

	var status, lastError string
	waitFor(t, runCtx, "bad feed to dead-letter", func() bool {
		err := pool.QueryRow(ctx, `
			SELECT j.status, j.last_error FROM jobs j
			WHERE j.job_type = $1 AND j.payload->>'podcast_id' = $2`,
			catalog.JobFeedRefresh, strconv.FormatInt(badID, 10)).Scan(&status, &lastError)
		return err == nil && status == "dead"
	})
	if lastError == "" {
		t.Error("dead-lettered job has no error retained")
	}

	// The worker survived and ingested the healthy feed.
	waitFor(t, runCtx, "good feed to ingest", func() bool {
		return countEpisodes(t, pool, goodID) == 3
	})
}

// A feed that fails transiently (here, 503) is not polled again until its
// backoff has elapsed, so the scheduler cannot re-enqueue it every cycle.
func TestTransientFeedFailureBacksOff(t *testing.T) {
	pool := pgtest.Pool(t, "catalog_worker")
	ctx := context.Background()

	flaky := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer flaky.Close()
	id := insertPodcast(t, pool, flaky.URL)
	_, queue, runCtx := startWorker(t, pool)

	if _, err := queue.Enqueue(ctx, catalog.JobFeedRefresh,
		catalog.FeedRefreshPayload{PodcastID: id}, jobs.WithMaxAttempts(2)); err != nil {
		t.Fatalf("enqueue: %v", err)
	}
	var status string
	waitFor(t, runCtx, "flaky feed to exhaust its attempts", func() bool {
		err := pool.QueryRow(ctx, `SELECT status FROM jobs WHERE job_type = $1 AND payload->>'podcast_id' = $2`,
			catalog.JobFeedRefresh, strconv.FormatInt(id, 10)).Scan(&status)
		return err == nil && status == "dead"
	})

	var wait time.Duration
	if err := pool.QueryRow(ctx, "SELECT next_fetch_at - now() FROM podcasts WHERE id = $1", id).Scan(&wait); err != nil {
		t.Fatalf("next_fetch_at: %v", err)
	}
	// Two failed attempts: the second pushes the poll out by twice the minimum.
	if want := catalog.TransientBackoff(2); wait < want-time.Minute || wait > want {
		t.Errorf("next poll in %v, want about %v", wait, want)
	}
	// Not due: the scheduler would not pick it up.
	var due bool
	if err := pool.QueryRow(ctx, "SELECT next_fetch_at <= now() FROM podcasts WHERE id = $1", id).Scan(&due); err != nil {
		t.Fatalf("due: %v", err)
	}
	if due {
		t.Error("feed is still due after failing")
	}
}

func TestTransientBackoff(t *testing.T) {
	for attempts, want := range map[int]time.Duration{
		0: catalog.MinFetchInterval, 1: catalog.MinFetchInterval, 2: 2 * catalog.MinFetchInterval,
		5: 16 * catalog.MinFetchInterval, 20: catalog.MaxFetchInterval,
	} {
		if got := catalog.TransientBackoff(attempts); got != want {
			t.Errorf("TransientBackoff(%d) = %v, want %v", attempts, got, want)
		}
	}
}
