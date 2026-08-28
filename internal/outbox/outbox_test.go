package outbox_test

import (
	"context"
	"errors"
	"testing"

	"podfind/internal/outbox"
	"podfind/internal/pgtest"
)

func TestStateAndEventCommitAtomically(t *testing.T) {
	pool := pgtest.Pool(t, "outbox")
	ctx := context.Background()

	// Committed transaction: podcast row and its event both persist.
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatalf("begin: %v", err)
	}
	var podcastID int64
	if err := tx.QueryRow(ctx, `
		INSERT INTO podcasts (feed_url, title) VALUES ($1, $2) RETURNING id`,
		"https://example.com/feed.xml", "Example Show",
	).Scan(&podcastID); err != nil {
		t.Fatalf("insert podcast: %v", err)
	}
	if err := outbox.Append(ctx, tx, "podcast.created", 1, map[string]int64{"podcast_id": podcastID}); err != nil {
		t.Fatalf("append event: %v", err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatalf("commit: %v", err)
	}

	// Rolled-back transaction: neither the row nor the event survive.
	tx2, err := pool.Begin(ctx)
	if err != nil {
		t.Fatalf("begin: %v", err)
	}
	if _, err := tx2.Exec(ctx, `
		INSERT INTO podcasts (feed_url, title) VALUES ($1, $2)`,
		"https://example.com/other.xml", "Ghost Show"); err != nil {
		t.Fatalf("insert podcast: %v", err)
	}
	if err := outbox.Append(ctx, tx2, "podcast.created", 1, map[string]string{"feed": "other"}); err != nil {
		t.Fatalf("append event: %v", err)
	}
	if err := tx2.Rollback(ctx); err != nil {
		t.Fatalf("rollback: %v", err)
	}

	var podcasts, events int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM podcasts").Scan(&podcasts); err != nil {
		t.Fatalf("count podcasts: %v", err)
	}
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM outbox").Scan(&events); err != nil {
		t.Fatalf("count events: %v", err)
	}
	if podcasts != 1 || events != 1 {
		t.Fatalf("got %d podcasts and %d events, want 1 and 1", podcasts, events)
	}
}

func TestDrainLifecycle(t *testing.T) {
	pool := pgtest.Pool(t, "outbox")
	ctx := context.Background()

	for i := 0; i < 3; i++ {
		if err := outbox.Append(ctx, pool, "episode.content_changed", 1, map[string]int{"n": i}); err != nil {
			t.Fatalf("append: %v", err)
		}
	}

	events, err := outbox.Unprocessed(ctx, pool, 10)
	if err != nil {
		t.Fatalf("unprocessed: %v", err)
	}
	if len(events) != 3 {
		t.Fatalf("got %d events, want 3", len(events))
	}
	for i := 1; i < len(events); i++ {
		if events[i].ID <= events[i-1].ID {
			t.Fatal("events not in insertion order")
		}
	}

	// A transient failure keeps the event pending with the error recorded.
	if err := outbox.MarkFailed(ctx, pool, events[0].ID, errors.New("downstream timeout")); err != nil {
		t.Fatalf("mark failed: %v", err)
	}
	remaining, err := outbox.Unprocessed(ctx, pool, 10)
	if err != nil {
		t.Fatalf("unprocessed after failure: %v", err)
	}
	if len(remaining) != 3 {
		t.Fatalf("got %d events after transient failure, want 3", len(remaining))
	}
	if remaining[0].Attempts != 1 {
		t.Fatalf("attempts = %d, want 1", remaining[0].Attempts)
	}

	for _, e := range remaining {
		if err := outbox.MarkProcessed(ctx, pool, e.ID); err != nil {
			t.Fatalf("mark processed: %v", err)
		}
	}
	drained, err := outbox.Unprocessed(ctx, pool, 10)
	if err != nil {
		t.Fatalf("unprocessed after drain: %v", err)
	}
	if len(drained) != 0 {
		t.Fatalf("got %d events after drain, want 0", len(drained))
	}
}
