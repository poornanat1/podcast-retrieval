package catalog_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"podfind/internal/catalog"
	"podfind/internal/jobs"
	"podfind/internal/outbox"
	"podfind/internal/pgtest"
	"podfind/internal/rss"
)

const feedWithTranscriptTemplate = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
     xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"
     xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <channel>
    <title>Transcribed Show</title>
    <language>en</language>
    <item>
      <title>Episode with transcript</title>
      <description>An episode that ships an SRT transcript.</description>
      <guid isPermaLink="false">tr-ep-1</guid>
      <enclosure url="https://cdn.example.com/tr1.mp3" length="1" type="audio/mpeg"/>
      <pubDate>Mon, 17 Aug 2026 09:30:00 +0000</pubDate>
      <podcast:transcript url="%s/tr1.srt" type="application/srt"/>
    </item>
    <item>
      <title>Episode without transcript</title>
      <description>Still searchable through its metadata.</description>
      <guid isPermaLink="false">tr-ep-2</guid>
      <enclosure url="https://cdn.example.com/tr2.mp3" length="1" type="audio/mpeg"/>
      <pubDate>Mon, 10 Aug 2026 09:30:00 +0000</pubDate>
    </item>
  </channel>
</rss>`

const srtBody = "1\n00:00:01,000 --> 00:00:04,000\nHello from the transcript pipeline.\n\n2\n00:00:04,100 --> 00:00:06,000\nSearchable speech.\n"

// Full chain: feed refresh discovers the transcript tag, a transcript job
// fetches and parses it, both steps emit content-change events, and the
// outbox worker consumes them in order.
func TestTranscriptPipelineEndToEnd(t *testing.T) {
	pool := pgtest.Pool(t, "catalog_transcripts")
	ctx := context.Background()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/feed.xml":
			fmt.Fprintf(w, feedWithTranscriptTemplate, "http://"+r.Host)
		case "/tr1.srt":
			w.Header().Set("Content-Type", "application/srt")
			w.Write([]byte(srtBody))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	podcastID := insertPodcast(t, pool, server.URL+"/feed.xml")

	runCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	queue := jobs.New(pool, jobs.WithBackoff(0, 0))
	worker := &catalog.Worker{Pool: pool, Queue: queue, Fetcher: &rss.Fetcher{}}
	runner := &jobs.Runner{
		Queue:        queue,
		Handlers:     worker.Handlers(),
		PollInterval: 20 * time.Millisecond,
		Concurrency:  2,
	}
	go runner.Run(runCtx)

	var mu sync.Mutex
	var reasons []string
	outboxWorker := &outbox.Worker{
		Pool:         pool,
		PollInterval: 20 * time.Millisecond,
		Handlers: map[string]outbox.Handler{
			catalog.EventContentChanged: func(ctx context.Context, e outbox.Event) error {
				var p catalog.ContentChangedPayload
				if err := json.Unmarshal(e.Payload, &p); err != nil {
					return err
				}
				mu.Lock()
				reasons = append(reasons, p.Reason)
				mu.Unlock()
				return nil
			},
		},
	}
	go outboxWorker.Run(runCtx)

	if _, err := queue.Enqueue(ctx, catalog.JobFeedRefresh,
		catalog.FeedRefreshPayload{PodcastID: podcastID}); err != nil {
		t.Fatalf("enqueue refresh: %v", err)
	}

	// The transcript lands parsed in Postgres.
	var content, format, language string
	waitFor(t, runCtx, "transcript to be ingested", func() bool {
		err := pool.QueryRow(ctx, `
			SELECT t.content, t.format, t.language FROM transcripts t
			JOIN episodes e ON e.id = t.episode_id
			WHERE e.rss_guid = 'tr-ep-1'`).Scan(&content, &format, &language)
		return err == nil
	})
	if format != "srt" || language != "en" {
		t.Errorf("format=%q language=%q", format, language)
	}
	if content != "Hello from the transcript pipeline. Searchable speech." {
		t.Errorf("content = %q", content)
	}

	// The episode without a transcript ingested fine alongside it.
	if n := countEpisodes(t, pool, podcastID); n != 2 {
		t.Errorf("episode count = %d, want 2", n)
	}

	// The outbox drains: feed events for both episodes plus one transcript
	// event, all consumed.
	waitFor(t, runCtx, "outbox to drain", func() bool {
		mu.Lock()
		defer mu.Unlock()
		feedEvents, transcriptEvents := 0, 0
		for _, r := range reasons {
			switch r {
			case "feed":
				feedEvents++
			case "transcript":
				transcriptEvents++
			}
		}
		return feedEvents == 2 && transcriptEvents == 1
	})
	var unprocessed int
	if err := pool.QueryRow(ctx,
		"SELECT count(*) FROM outbox WHERE processed_at IS NULL").Scan(&unprocessed); err != nil {
		t.Fatalf("count outbox: %v", err)
	}
	if unprocessed != 0 {
		t.Fatalf("%d outbox events left unprocessed", unprocessed)
	}
}

// A transcript job for an unreachable file dead-letters without touching the
// episode, which stays searchable through its metadata.
func TestMissingTranscriptDoesNotBlockEpisode(t *testing.T) {
	pool := pgtest.Pool(t, "catalog_transcripts")
	ctx := context.Background()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/feed.xml" {
			fmt.Fprintf(w, feedWithTranscriptTemplate, "http://"+r.Host+"/missing")
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	podcastID := insertPodcast(t, pool, server.URL+"/feed.xml")
	_, queue, runCtx := startWorker(t, pool)

	if _, err := queue.Enqueue(ctx, catalog.JobFeedRefresh,
		catalog.FeedRefreshPayload{PodcastID: podcastID}); err != nil {
		t.Fatalf("enqueue refresh: %v", err)
	}

	waitFor(t, runCtx, "transcript job to dead-letter", func() bool {
		var status string
		err := pool.QueryRow(ctx, `
			SELECT status FROM jobs WHERE job_type = $1`,
			catalog.JobTranscriptFetch).Scan(&status)
		return err == nil && status == "dead"
	})

	// Both episodes ingested; no transcript rows exist.
	if n := countEpisodes(t, pool, podcastID); n != 2 {
		t.Errorf("episode count = %d, want 2", n)
	}
	var transcripts int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM transcripts").Scan(&transcripts); err != nil {
		t.Fatalf("count transcripts: %v", err)
	}
	if transcripts != 0 {
		t.Errorf("transcript rows = %d, want 0", transcripts)
	}
}
