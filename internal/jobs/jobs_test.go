package jobs_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"podfind/internal/jobs"
	"podfind/internal/pgtest"
)

func newQueue(t *testing.T, opts ...jobs.Option) (*jobs.Queue, *pgxpool.Pool) {
	t.Helper()
	pool := pgtest.Pool(t, "jobs")
	return jobs.New(pool, opts...), pool
}

func TestConcurrentClaimsNeverDuplicate(t *testing.T) {
	q, _ := newQueue(t)
	ctx := context.Background()

	const total = 50
	for i := 0; i < total; i++ {
		if _, err := q.Enqueue(ctx, "test.claim", map[string]int{"n": i}); err != nil {
			t.Fatalf("enqueue: %v", err)
		}
	}

	var mu sync.Mutex
	claimCounts := make(map[int64]int)
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				job, err := q.Claim(ctx, []string{"test.claim"}, time.Minute)
				if err != nil {
					t.Errorf("claim: %v", err)
					return
				}
				if job == nil {
					return
				}
				mu.Lock()
				claimCounts[job.ID]++
				mu.Unlock()
				if err := q.Complete(ctx, job.ID); err != nil {
					t.Errorf("complete: %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()

	if len(claimCounts) != total {
		t.Fatalf("claimed %d distinct jobs, want %d", len(claimCounts), total)
	}
	for id, n := range claimCounts {
		if n != 1 {
			t.Fatalf("job %d claimed %d times, want exactly once", id, n)
		}
	}
}

func TestExpiredLeaseIsReclaimed(t *testing.T) {
	q, _ := newQueue(t)
	ctx := context.Background()

	id, err := q.Enqueue(ctx, "test.lease", nil)
	if err != nil {
		t.Fatalf("enqueue: %v", err)
	}

	// Simulate a worker that claims the job and crashes without finishing.
	first, err := q.Claim(ctx, []string{"test.lease"}, 100*time.Millisecond)
	if err != nil || first == nil {
		t.Fatalf("first claim: job=%v err=%v", first, err)
	}
	if first.ID != id || first.Attempts != 1 {
		t.Fatalf("first claim: got id=%d attempts=%d", first.ID, first.Attempts)
	}

	// Nothing is claimable until the lease expires and is reaped.
	if job, _ := q.Claim(ctx, []string{"test.lease"}, time.Minute); job != nil {
		t.Fatalf("claimed job %d while lease held", job.ID)
	}

	deadline := time.Now().Add(10 * time.Second)
	reaped := 0
	for reaped == 0 {
		if time.Now().After(deadline) {
			t.Fatal("lease never expired")
		}
		time.Sleep(50 * time.Millisecond)
		if reaped, err = q.ReapExpiredLeases(ctx); err != nil {
			t.Fatalf("reap: %v", err)
		}
	}

	second, err := q.Claim(ctx, []string{"test.lease"}, time.Minute)
	if err != nil || second == nil {
		t.Fatalf("reclaim: job=%v err=%v", second, err)
	}
	if second.ID != id || second.Attempts != 2 {
		t.Fatalf("reclaim: got id=%d attempts=%d, want id=%d attempts=2", second.ID, second.Attempts, id)
	}
}

func TestFailingJobDeadLettersWithErrorRetained(t *testing.T) {
	q, pool := newQueue(t, jobs.WithBackoff(0, 0)) // retry immediately
	ctx := context.Background()

	id, err := q.Enqueue(ctx, "test.dead", nil, jobs.WithMaxAttempts(2))
	if err != nil {
		t.Fatalf("enqueue: %v", err)
	}

	for attempt := 1; attempt <= 2; attempt++ {
		job, err := q.Claim(ctx, []string{"test.dead"}, time.Minute)
		if err != nil || job == nil {
			t.Fatalf("claim attempt %d: job=%v err=%v", attempt, job, err)
		}
		if job.Attempts != attempt {
			t.Fatalf("attempt %d: got attempts=%d", attempt, job.Attempts)
		}
		if err := q.Fail(ctx, job, errors.New("boom")); err != nil {
			t.Fatalf("fail attempt %d: %v", attempt, err)
		}
	}

	var status, lastError string
	if err := pool.QueryRow(ctx,
		"SELECT status, last_error FROM jobs WHERE id = $1", id,
	).Scan(&status, &lastError); err != nil {
		t.Fatalf("read job: %v", err)
	}
	if status != "dead" {
		t.Fatalf("status = %q, want dead", status)
	}
	if lastError != "boom" {
		t.Fatalf("last_error = %q, want boom", lastError)
	}

	// Dead jobs are never claimable again.
	if job, _ := q.Claim(ctx, []string{"test.dead"}, time.Minute); job != nil {
		t.Fatalf("claimed dead job %d", job.ID)
	}
}

func TestIdempotencyKeyDeduplicates(t *testing.T) {
	q, pool := newQueue(t)
	ctx := context.Background()

	first, err := q.Enqueue(ctx, "test.idem", nil, jobs.WithIdempotencyKey("feed:42"))
	if err != nil {
		t.Fatalf("first enqueue: %v", err)
	}
	if first == 0 {
		t.Fatal("first enqueue returned no id")
	}

	second, err := q.Enqueue(ctx, "test.idem", nil, jobs.WithIdempotencyKey("feed:42"))
	if err != nil {
		t.Fatalf("second enqueue: %v", err)
	}
	if second != 0 {
		t.Fatalf("duplicate enqueue created job %d", second)
	}

	var n int
	if err := pool.QueryRow(ctx,
		"SELECT count(*) FROM jobs WHERE job_type = 'test.idem'").Scan(&n); err != nil {
		t.Fatalf("count: %v", err)
	}
	if n != 1 {
		t.Fatalf("found %d jobs, want 1", n)
	}
}

func TestRunnerProcessesJobs(t *testing.T) {
	q, pool := newQueue(t)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	const total = 3
	for i := 0; i < total; i++ {
		if _, err := q.Enqueue(ctx, "test.runner", map[string]int{"n": i}); err != nil {
			t.Fatalf("enqueue: %v", err)
		}
	}

	var mu sync.Mutex
	processed := 0
	runner := &jobs.Runner{
		Queue:        q,
		PollInterval: 20 * time.Millisecond,
		Handlers: map[string]jobs.Handler{
			"test.runner": func(ctx context.Context, job *jobs.Job) error {
				mu.Lock()
				processed++
				mu.Unlock()
				return nil
			},
		},
	}
	go runner.Run(ctx)

	deadline := time.Now().Add(10 * time.Second)
	for {
		var completed int
		if err := pool.QueryRow(ctx,
			"SELECT count(*) FROM jobs WHERE job_type = 'test.runner' AND status = 'completed'",
		).Scan(&completed); err != nil {
			t.Fatalf("count completed: %v", err)
		}
		if completed == total {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("only %d/%d jobs completed", completed, total)
		}
		time.Sleep(50 * time.Millisecond)
	}

	mu.Lock()
	defer mu.Unlock()
	if processed != total {
		t.Fatalf("handler ran %d times, want %d", processed, total)
	}
}
