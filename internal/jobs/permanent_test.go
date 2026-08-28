package jobs_test

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"podfind/internal/jobs"
)

func TestPermanentFailureDeadLettersImmediately(t *testing.T) {
	q, pool := newQueue(t)
	ctx := context.Background()

	id, err := q.Enqueue(ctx, "test.permanent", nil) // default 5 attempts
	if err != nil {
		t.Fatalf("enqueue: %v", err)
	}

	job, err := q.Claim(ctx, []string{"test.permanent"}, time.Minute)
	if err != nil || job == nil {
		t.Fatalf("claim: job=%v err=%v", job, err)
	}

	// Wrapping deeper in an error chain must still be detected.
	wrapped := fmt.Errorf("refresh feed: %w", jobs.Permanent(errors.New("feed is not valid XML")))
	if err := q.Fail(ctx, job, wrapped); err != nil {
		t.Fatalf("fail: %v", err)
	}

	var status, lastError string
	var attempts int
	if err := pool.QueryRow(ctx,
		"SELECT status, last_error, attempts FROM jobs WHERE id = $1", id,
	).Scan(&status, &lastError, &attempts); err != nil {
		t.Fatalf("read job: %v", err)
	}
	if status != "dead" {
		t.Fatalf("status = %q after permanent failure on attempt %d, want dead", status, attempts)
	}
	if lastError == "" {
		t.Fatal("last_error empty")
	}
}
