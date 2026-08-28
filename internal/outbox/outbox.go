// Package outbox implements the transactional-outbox pattern: application
// state changes and the versioned domain events describing them commit in
// one transaction, and workers drain the events asynchronously.
package outbox

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// Querier is satisfied by *pgxpool.Pool and pgx.Tx. Append must be called
// with the same transaction that mutates application state.
type Querier interface {
	Exec(ctx context.Context, sql string, args ...any) (pgconn.CommandTag, error)
	Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
	QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
}

// Event is one domain event awaiting or past processing.
type Event struct {
	ID        int64
	Type      string
	Version   int
	Payload   json.RawMessage
	Attempts  int
	CreatedAt time.Time
}

// Append inserts a versioned event. Call it inside the transaction that
// performs the corresponding state change so both commit or neither does.
func Append(ctx context.Context, db Querier, eventType string, version int, payload any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal %s payload: %w", eventType, err)
	}
	if _, err := db.Exec(ctx, `
		INSERT INTO outbox (event_type, event_version, payload)
		VALUES ($1, $2, $3)`,
		eventType, version, body); err != nil {
		return fmt.Errorf("append %s: %w", eventType, err)
	}
	return nil
}

// Unprocessed returns up to limit unprocessed events in insertion order.
func Unprocessed(ctx context.Context, db Querier, limit int) ([]Event, error) {
	rows, err := db.Query(ctx, `
		SELECT id, event_type, event_version, payload, attempts, created_at
		FROM outbox
		WHERE processed_at IS NULL
		ORDER BY id
		LIMIT $1`, limit)
	if err != nil {
		return nil, fmt.Errorf("list unprocessed: %w", err)
	}
	defer rows.Close()

	var events []Event
	for rows.Next() {
		var e Event
		if err := rows.Scan(&e.ID, &e.Type, &e.Version, &e.Payload, &e.Attempts, &e.CreatedAt); err != nil {
			return nil, fmt.Errorf("scan event: %w", err)
		}
		events = append(events, e)
	}
	return events, rows.Err()
}

// MarkProcessed records successful processing of an event.
func MarkProcessed(ctx context.Context, db Querier, id int64) error {
	if _, err := db.Exec(ctx,
		"UPDATE outbox SET processed_at = now() WHERE id = $1", id); err != nil {
		return fmt.Errorf("mark processed %d: %w", id, err)
	}
	return nil
}

// MarkFailed records a transient processing failure; the event stays
// unprocessed and will be retried by the next drain.
func MarkFailed(ctx context.Context, db Querier, id int64, procErr error) error {
	msg := ""
	if procErr != nil {
		msg = procErr.Error()
	}
	if _, err := db.Exec(ctx, `
		UPDATE outbox SET attempts = attempts + 1, last_error = $2
		WHERE id = $1`, id, msg); err != nil {
		return fmt.Errorf("mark failed %d: %w", id, err)
	}
	return nil
}
