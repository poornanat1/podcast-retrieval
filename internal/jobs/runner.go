package jobs

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"
)

// Handler processes one claimed job. A non-nil error counts as a failed
// attempt; the job retries with backoff until dead-lettered.
type Handler func(ctx context.Context, job *Job) error

// Runner polls the queue and dispatches claimed jobs to registered handlers,
// renewing the lease in the background while a handler runs.
type Runner struct {
	Queue        *Queue
	Handlers     map[string]Handler
	PollInterval time.Duration // default 1s
	Lease        time.Duration // default 1m
	Concurrency  int           // parallel workers, default 1
	Log          *slog.Logger  // default slog.Default()
}

// Run processes jobs until ctx is canceled, using Concurrency parallel
// workers over the shared queue.
func (r *Runner) Run(ctx context.Context) error {
	n := r.Concurrency
	if n <= 0 {
		n = 1
	}
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r.runWorker(ctx)
		}()
	}
	wg.Wait()
	return ctx.Err()
}

func (r *Runner) runWorker(ctx context.Context) error {
	poll := r.PollInterval
	if poll <= 0 {
		poll = time.Second
	}
	lease := r.Lease
	if lease <= 0 {
		lease = time.Minute
	}
	logger := r.Log
	if logger == nil {
		logger = slog.Default()
	}

	types := make([]string, 0, len(r.Handlers))
	for t := range r.Handlers {
		types = append(types, t)
	}

	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		if n, err := r.Queue.ReapExpiredLeases(ctx); err != nil {
			logger.Error("reap expired leases", "error", err)
		} else if n > 0 {
			logger.Info("reaped expired leases", "count", n)
		}

		job, err := r.Queue.Claim(ctx, types, lease)
		if err != nil {
			logger.Error("claim job", "error", err)
		}
		if job == nil {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(poll):
			}
			continue
		}

		r.process(ctx, logger, job, lease)
	}
}

func (r *Runner) process(ctx context.Context, logger *slog.Logger, job *Job, lease time.Duration) {
	// Renew the lease at half-life while the handler runs.
	renewCtx, stopRenewal := context.WithCancel(ctx)
	defer stopRenewal()
	go func() {
		ticker := time.NewTicker(lease / 2)
		defer ticker.Stop()
		for {
			select {
			case <-renewCtx.Done():
				return
			case <-ticker.C:
				if err := r.Queue.RenewLease(renewCtx, job.ID, lease); err != nil {
					logger.Error("renew lease", "job", job.ID, "error", err)
					return
				}
			}
		}
	}()

	err := func() (err error) {
		defer func() {
			if p := recover(); p != nil {
				err = fmt.Errorf("panic: %v", p)
			}
		}()
		return r.Handlers[job.Type](ctx, job)
	}()
	stopRenewal()

	if err != nil {
		logger.Warn("job failed", "job", job.ID, "type", job.Type,
			"attempt", job.Attempts, "max_attempts", job.MaxAttempts, "error", err)
		if failErr := r.Queue.Fail(ctx, job, err); failErr != nil {
			logger.Error("record job failure", "job", job.ID, "error", failErr)
		}
		return
	}
	if err := r.Queue.Complete(ctx, job.ID); err != nil {
		logger.Error("complete job", "job", job.ID, "error", err)
	}
}
