// Package jobs implements the PostgreSQL-backed job queue. Replicas claim
// jobs with FOR UPDATE SKIP LOCKED; claims carry a lease so work from a
// crashed worker is reclaimed, retries back off exponentially, and jobs that
// exhaust their attempts are parked in dead-letter state for inspection.
package jobs

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Job is a claimed unit of work.
type Job struct {
	ID          int64
	Type        string
	Payload     json.RawMessage
	Attempts    int
	MaxAttempts int
}

// Querier is satisfied by both *pgxpool.Pool and pgx.Tx, so enqueues can
// join a caller's transaction (e.g. alongside an outbox event).
type Querier interface {
	Exec(ctx context.Context, sql string, args ...any) (pgconn.CommandTag, error)
	QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
}

// Queue provides enqueue, claim, and completion operations over one pool.
type Queue struct {
	pool        *pgxpool.Pool
	backoffBase time.Duration
	backoffCap  time.Duration
}

// Option configures a Queue.
type Option func(*Queue)

// WithBackoff overrides the retry backoff schedule. A zero base retries
// failed jobs immediately (useful in tests).
func WithBackoff(base, cap time.Duration) Option {
	return func(q *Queue) {
		q.backoffBase = base
		q.backoffCap = cap
	}
}

// New returns a Queue with a 30s..1h exponential backoff by default.
func New(pool *pgxpool.Pool, opts ...Option) *Queue {
	q := &Queue{pool: pool, backoffBase: 30 * time.Second, backoffCap: time.Hour}
	for _, opt := range opts {
		opt(q)
	}
	return q
}

// EnqueueOption configures a single enqueue.
type EnqueueOption func(*enqueueParams)

type enqueueParams struct {
	idempotencyKey *string
	runAt          *time.Time
	maxAttempts    int
}

// WithIdempotencyKey makes re-enqueues of the same (type, key) a no-op.
func WithIdempotencyKey(key string) EnqueueOption {
	return func(p *enqueueParams) { p.idempotencyKey = &key }
}

// WithRunAt delays the job until the given time.
func WithRunAt(at time.Time) EnqueueOption {
	return func(p *enqueueParams) { p.runAt = &at }
}

// WithMaxAttempts overrides the default of 5 attempts.
func WithMaxAttempts(n int) EnqueueOption {
	return func(p *enqueueParams) { p.maxAttempts = n }
}

// Enqueue inserts a job and returns its ID. If an idempotency key is given
// and a job with that (type, key) already exists, it returns (0, nil).
func (q *Queue) Enqueue(ctx context.Context, jobType string, payload any, opts ...EnqueueOption) (int64, error) {
	return q.EnqueueIn(ctx, q.pool, jobType, payload, opts...)
}

// EnqueueIn is Enqueue against an arbitrary Querier, typically a transaction.
func (q *Queue) EnqueueIn(ctx context.Context, db Querier, jobType string, payload any, opts ...EnqueueOption) (int64, error) {
	p := enqueueParams{maxAttempts: 5}
	for _, opt := range opts {
		opt(&p)
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return 0, fmt.Errorf("marshal payload: %w", err)
	}

	var id int64
	err = db.QueryRow(ctx, `
		INSERT INTO jobs (job_type, payload, idempotency_key, run_at, max_attempts)
		VALUES ($1, $2, $3, COALESCE($4, now()), $5)
		ON CONFLICT (job_type, idempotency_key) WHERE idempotency_key IS NOT NULL
		DO NOTHING
		RETURNING id`,
		jobType, body, p.idempotencyKey, p.runAt, p.maxAttempts,
	).Scan(&id)
	if errors.Is(err, pgx.ErrNoRows) {
		return 0, nil // duplicate idempotency key: already enqueued
	}
	if err != nil {
		return 0, fmt.Errorf("enqueue %s: %w", jobType, err)
	}
	return id, nil
}

// Claim atomically claims the oldest due pending job of one of the given
// types, holding it under a lease. It returns (nil, nil) when no job is due.
func (q *Queue) Claim(ctx context.Context, types []string, lease time.Duration) (*Job, error) {
	job := &Job{}
	err := q.pool.QueryRow(ctx, `
		WITH candidate AS (
			SELECT id FROM jobs
			WHERE job_type = ANY($1) AND status = 'pending' AND run_at <= now()
			ORDER BY run_at
			LIMIT 1
			FOR UPDATE SKIP LOCKED
		)
		UPDATE jobs j
		SET status = 'running',
		    attempts = j.attempts + 1,
		    lease_expires_at = now() + $2,
		    updated_at = now()
		FROM candidate
		WHERE j.id = candidate.id
		RETURNING j.id, j.job_type, j.payload, j.attempts, j.max_attempts`,
		types, lease,
	).Scan(&job.ID, &job.Type, &job.Payload, &job.Attempts, &job.MaxAttempts)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("claim: %w", err)
	}
	return job, nil
}

// RenewLease extends the lease on a running job.
func (q *Queue) RenewLease(ctx context.Context, jobID int64, lease time.Duration) error {
	tag, err := q.pool.Exec(ctx, `
		UPDATE jobs SET lease_expires_at = now() + $2, updated_at = now()
		WHERE id = $1 AND status = 'running'`,
		jobID, lease)
	if err != nil {
		return fmt.Errorf("renew lease: %w", err)
	}
	if tag.RowsAffected() == 0 {
		return fmt.Errorf("renew lease: job %d is not running", jobID)
	}
	return nil
}

// Complete marks a running job as successfully finished.
func (q *Queue) Complete(ctx context.Context, jobID int64) error {
	_, err := q.pool.Exec(ctx, `
		UPDATE jobs
		SET status = 'completed', lease_expires_at = NULL,
		    completed_at = now(), updated_at = now()
		WHERE id = $1 AND status = 'running'`,
		jobID)
	if err != nil {
		return fmt.Errorf("complete job %d: %w", jobID, err)
	}
	return nil
}

// PermanentError marks a failure that retrying cannot fix (e.g. a malformed
// feed); Fail dead-letters the job immediately instead of backing off.
type PermanentError struct{ Err error }

func (e *PermanentError) Error() string { return e.Err.Error() }
func (e *PermanentError) Unwrap() error { return e.Err }

// Permanent wraps err so Fail dead-letters the job without further retries.
func Permanent(err error) error {
	if err == nil {
		return nil
	}
	return &PermanentError{Err: err}
}

// Fail records a failed attempt. The job is retried with exponential backoff
// until its attempts are exhausted — or immediately dead-lettered on a
// PermanentError — with the final error retained.
func (q *Queue) Fail(ctx context.Context, job *Job, jobErr error) error {
	msg := ""
	if jobErr != nil {
		msg = jobErr.Error()
	}

	var perm *PermanentError
	if job.Attempts >= job.MaxAttempts || errors.As(jobErr, &perm) {
		_, err := q.pool.Exec(ctx, `
			UPDATE jobs
			SET status = 'dead', lease_expires_at = NULL,
			    last_error = $2, updated_at = now()
			WHERE id = $1`,
			job.ID, msg)
		if err != nil {
			return fmt.Errorf("dead-letter job %d: %w", job.ID, err)
		}
		return nil
	}

	_, err := q.pool.Exec(ctx, `
		UPDATE jobs
		SET status = 'pending', lease_expires_at = NULL,
		    run_at = now() + $2, last_error = $3, updated_at = now()
		WHERE id = $1`,
		job.ID, q.backoff(job.Attempts), msg)
	if err != nil {
		return fmt.Errorf("retry job %d: %w", job.ID, err)
	}
	return nil
}

// ReapExpiredLeases requeues running jobs whose lease has expired, or
// dead-letters them when their attempts are exhausted. It returns how many
// jobs it transitioned.
func (q *Queue) ReapExpiredLeases(ctx context.Context) (int, error) {
	tag, err := q.pool.Exec(ctx, `
		UPDATE jobs
		SET status = CASE WHEN attempts >= max_attempts THEN 'dead' ELSE 'pending' END,
		    last_error = CASE WHEN attempts >= max_attempts
		                      THEN 'lease expired after final attempt'
		                      ELSE last_error END,
		    lease_expires_at = NULL,
		    updated_at = now()
		WHERE status = 'running' AND lease_expires_at < now()`)
	if err != nil {
		return 0, fmt.Errorf("reap expired leases: %w", err)
	}
	return int(tag.RowsAffected()), nil
}

// backoff computes the delay before retry attempt n+1 (n = attempts so far).
func (q *Queue) backoff(attempts int) time.Duration {
	if q.backoffBase <= 0 {
		return 0
	}
	d := q.backoffBase << (attempts - 1)
	if d > q.backoffCap || d <= 0 {
		return q.backoffCap
	}
	return d
}
