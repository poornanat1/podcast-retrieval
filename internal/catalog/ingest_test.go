package catalog_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"podfind/internal/catalog"
	"podfind/internal/pgtest"
	"podfind/internal/rss"
)

func loadFeed(t *testing.T) *rss.Feed {
	t.Helper()
	data, err := os.ReadFile("../rss/testdata/feed.xml")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	feed, err := rss.Parse(data)
	if err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	return feed
}

func insertPodcast(t *testing.T, pool *pgxpool.Pool, feedURL string) int64 {
	t.Helper()
	var id int64
	if err := pool.QueryRow(context.Background(),
		"INSERT INTO podcasts (feed_url) VALUES ($1) RETURNING id", feedURL,
	).Scan(&id); err != nil {
		t.Fatalf("insert podcast: %v", err)
	}
	return id
}

func countEpisodes(t *testing.T, pool *pgxpool.Pool, podcastID int64) int {
	t.Helper()
	var n int
	if err := pool.QueryRow(context.Background(),
		"SELECT count(*) FROM episodes WHERE podcast_id = $1", podcastID,
	).Scan(&n); err != nil {
		t.Fatalf("count episodes: %v", err)
	}
	return n
}

func TestReingestingFeedCreatesNoDuplicates(t *testing.T) {
	pool := pgtest.Pool(t, "catalog")
	ctx := context.Background()
	feed := loadFeed(t)
	id := insertPodcast(t, pool, "https://example.com/feed.xml")

	first, _, err := catalog.IngestFeed(ctx, pool, id, feed)
	if err != nil {
		t.Fatalf("first ingest: %v", err)
	}
	if first.NewEpisodes != 3 || first.UpdatedEpisodes != 0 {
		t.Fatalf("first ingest stats = %+v", first)
	}

	second, _, err := catalog.IngestFeed(ctx, pool, id, feed)
	if err != nil {
		t.Fatalf("second ingest: %v", err)
	}
	if second.NewEpisodes != 0 || second.UpdatedEpisodes != 0 || second.UnchangedEpisodes != 3 {
		t.Fatalf("second ingest stats = %+v", second)
	}
	if n := countEpisodes(t, pool, id); n != 3 {
		t.Fatalf("episode count = %d after double ingest, want 3", n)
	}
}

func TestChangedContentIsDetectedAndUpdated(t *testing.T) {
	pool := pgtest.Pool(t, "catalog")
	ctx := context.Background()
	feed := loadFeed(t)
	id := insertPodcast(t, pool, "https://example.com/feed.xml")

	if _, _, err := catalog.IngestFeed(ctx, pool, id, feed); err != nil {
		t.Fatalf("first ingest: %v", err)
	}

	feed.Items[0].Description = "Two-tower models beyond the paper — now with corrections."
	stats, _, err := catalog.IngestFeed(ctx, pool, id, feed)
	if err != nil {
		t.Fatalf("second ingest: %v", err)
	}
	if stats.UpdatedEpisodes != 1 || stats.UnchangedEpisodes != 2 || stats.NewEpisodes != 0 {
		t.Fatalf("stats = %+v, want exactly one update", stats)
	}

	var desc string
	if err := pool.QueryRow(ctx,
		"SELECT description FROM episodes WHERE podcast_id = $1 AND rss_guid = 'ep-3-guid'", id,
	).Scan(&desc); err != nil {
		t.Fatalf("read episode: %v", err)
	}
	if desc != feed.Items[0].Description {
		t.Fatalf("description not updated: %q", desc)
	}
}

func TestDedupFallsBackToEnclosureAndHash(t *testing.T) {
	pool := pgtest.Pool(t, "catalog")
	ctx := context.Background()
	id := insertPodcast(t, pool, "https://example.com/feed.xml")

	dur := 100
	when := time.Date(2026, 8, 1, 0, 0, 0, 0, time.UTC)
	base := rss.Item{
		Title:           "No GUID episode",
		Description:     "d",
		EnclosureURL:    "https://cdn.example.com/no-guid.mp3",
		DurationSeconds: &dur,
		PublishedAt:     &when,
	}

	// Same enclosure, no GUID: second ingest is a duplicate.
	feed := &rss.Feed{Language: "en", Items: []rss.Item{base}}
	if _, _, err := catalog.IngestFeed(ctx, pool, id, feed); err != nil {
		t.Fatalf("ingest: %v", err)
	}
	stats, _, err := catalog.IngestFeed(ctx, pool, id, feed)
	if err != nil {
		t.Fatalf("re-ingest: %v", err)
	}
	if stats.UnchangedEpisodes != 1 || stats.NewEpisodes != 0 {
		t.Fatalf("enclosure dedup stats = %+v", stats)
	}

	// A GUID appearing later for a known enclosure adopts the row instead
	// of colliding.
	withGUID := base
	withGUID.GUID = "late-guid"
	stats, _, err = catalog.IngestFeed(ctx, pool, id, &rss.Feed{Language: "en", Items: []rss.Item{withGUID}})
	if err != nil {
		t.Fatalf("guid adoption ingest: %v", err)
	}
	if stats.NewEpisodes != 0 {
		t.Fatalf("guid adoption created a duplicate: %+v", stats)
	}
	if n := countEpisodes(t, pool, id); n != 1 {
		t.Fatalf("episode count = %d, want 1", n)
	}

	// Neither GUID nor enclosure: content hash is the identity.
	bare := rss.Item{Title: "Hash-only", Description: "only text"}
	hashFeed := &rss.Feed{Language: "en", Items: []rss.Item{bare}}
	if _, _, err := catalog.IngestFeed(ctx, pool, id, hashFeed); err != nil {
		t.Fatalf("hash ingest: %v", err)
	}
	stats, _, err = catalog.IngestFeed(ctx, pool, id, hashFeed)
	if err != nil {
		t.Fatalf("hash re-ingest: %v", err)
	}
	if stats.UnchangedEpisodes != 1 || stats.NewEpisodes != 0 {
		t.Fatalf("hash dedup stats = %+v", stats)
	}
}

// The GUID→enclosure fallback must survive inside the ingestion
// transaction: a failed statement aborts a Postgres transaction unless it
// ran in a savepoint.
func TestGUIDAdoptionInsideTransaction(t *testing.T) {
	pool := pgtest.Pool(t, "catalog")
	ctx := context.Background()
	id := insertPodcast(t, pool, "https://example.com/feed.xml")

	when := time.Date(2026, 8, 1, 0, 0, 0, 0, time.UTC)
	base := rss.Item{
		Title:        "Adopted episode",
		Description:  "d",
		EnclosureURL: "https://cdn.example.com/adopted.mp3",
		PublishedAt:  &when,
	}
	if _, _, err := catalog.IngestFeed(ctx, pool, id, &rss.Feed{Language: "en", Items: []rss.Item{base}}); err != nil {
		t.Fatalf("initial ingest: %v", err)
	}

	withGUID := base
	withGUID.GUID = "late-guid"
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatalf("begin: %v", err)
	}
	defer tx.Rollback(ctx)
	stats, _, err := catalog.IngestFeed(ctx, tx, id, &rss.Feed{Language: "en", Items: []rss.Item{withGUID}})
	if err != nil {
		t.Fatalf("ingest inside transaction: %v", err)
	}
	// The transaction is still usable after the fallback.
	var n int
	if err := tx.QueryRow(ctx, "SELECT count(*) FROM episodes WHERE podcast_id = $1", id).Scan(&n); err != nil {
		t.Fatalf("transaction unusable after fallback: %v", err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatalf("commit: %v", err)
	}
	if stats.NewEpisodes != 0 || n != 1 {
		t.Fatalf("stats=%+v count=%d, want adoption without duplicate", stats, n)
	}
}

func TestCanonicalFeedURL(t *testing.T) {
	cases := map[string]string{
		"HTTPS://Example.COM:443/Feed.xml":  "https://example.com/Feed.xml",
		"http://example.com:80/feed":        "http://example.com/feed",
		"https://example.com/feed#fragment": "https://example.com/feed",
		" https://example.com/feed ":        "https://example.com/feed",
	}
	for in, want := range cases {
		got, err := catalog.CanonicalFeedURL(in)
		if err != nil {
			t.Errorf("CanonicalFeedURL(%q): %v", in, err)
			continue
		}
		if got != want {
			t.Errorf("CanonicalFeedURL(%q) = %q, want %q", in, got, want)
		}
	}
	if _, err := catalog.CanonicalFeedURL("ftp://example.com/feed"); err == nil {
		t.Error("ftp scheme accepted")
	}
}

func TestNextFetchInterval(t *testing.T) {
	now := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	hoursAgo := func(h int) *time.Time {
		t := now.Add(-time.Duration(h) * time.Hour)
		return &t
	}
	cases := []struct {
		newest *time.Time
		want   time.Duration
	}{
		{hoursAgo(2), time.Hour},
		{hoursAgo(48), 3 * time.Hour},
		{hoursAgo(5 * 24), 6 * time.Hour},
		{hoursAgo(20 * 24), 12 * time.Hour},
		{hoursAgo(90 * 24), catalog.MaxFetchInterval},
		{nil, 6 * time.Hour},
	}
	for _, c := range cases {
		if got := catalog.NextFetchInterval(c.newest, now); got != c.want {
			t.Errorf("NextFetchInterval(%v) = %v, want %v", c.newest, got, c.want)
		}
	}
}
