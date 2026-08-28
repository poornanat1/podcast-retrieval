package outbox

import (
	"context"
	"log/slog"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// Handler processes one outbox event. A non-nil error leaves the event
// unprocessed to be retried on the next drain.
type Handler func(ctx context.Context, event Event) error

// Worker drains the outbox in insertion order, dispatching events by type.
// Processing is at-least-once: handlers must be idempotent.
type Worker struct {
	Pool         *pgxpool.Pool
	Handlers     map[string]Handler
	PollInterval time.Duration // default 1s
	BatchSize    int           // default 50
	MaxAttempts  int           // per-event cap before it is skipped, default 10
	Log          *slog.Logger  // default slog.Default()
}

// Run drains events until ctx is canceled.
func (w *Worker) Run(ctx context.Context) error {
	poll := w.PollInterval
	if poll <= 0 {
		poll = time.Second
	}
	batch := w.BatchSize
	if batch <= 0 {
		batch = 50
	}
	maxAttempts := w.MaxAttempts
	if maxAttempts <= 0 {
		maxAttempts = 10
	}
	logger := w.Log
	if logger == nil {
		logger = slog.Default()
	}

	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		events, err := Unprocessed(ctx, w.Pool, batch)
		if err != nil {
			logger.Error("list outbox events", "error", err)
		}
		if len(events) == 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(poll):
			}
			continue
		}

		for _, e := range events {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			handler, ok := w.Handlers[e.Type]
			switch {
			case !ok:
				// Unknown types are consumed so they cannot block the
				// ordered queue; the row and its payload remain auditable.
				logger.Warn("no handler for outbox event", "event", e.ID, "type", e.Type)
				err = MarkProcessed(ctx, w.Pool, e.ID)
			case e.Attempts >= maxAttempts:
				logger.Error("outbox event exhausted retries; skipping",
					"event", e.ID, "type", e.Type, "attempts", e.Attempts)
				err = MarkProcessed(ctx, w.Pool, e.ID)
			default:
				if handleErr := handler(ctx, e); handleErr != nil {
					logger.Warn("outbox event failed", "event", e.ID, "type", e.Type,
						"attempt", e.Attempts+1, "error", handleErr)
					err = MarkFailed(ctx, w.Pool, e.ID, handleErr)
				} else {
					err = MarkProcessed(ctx, w.Pool, e.ID)
				}
			}
			if err != nil {
				logger.Error("update outbox event", "event", e.ID, "error", err)
			}
		}
	}
}
