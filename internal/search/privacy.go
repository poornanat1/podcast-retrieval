package search

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"
)

// Data controls. A session is the unit of ownership: the API has no
// accounts, and the web app keeps its session id in the browser. Export
// returns everything stored for a session; Delete removes it, retained
// voice recordings included.

// SessionData is everything stored for one session.
type SessionData struct {
	SessionID  string            `json:"session_id"`
	ExportedAt time.Time         `json:"exported_at"`
	Searches   []StoredSearch    `json:"searches"`
	Events     []json.RawMessage `json:"events"`
	Recordings []RecordingInfo   `json:"recordings"`
}

// RecordingInfo describes a retained voice recording.
type RecordingInfo struct {
	QueryID      string    `json:"query_id"`
	ObjectKey    string    `json:"object_key"`
	Bytes        int       `json:"bytes"`
	AudioSeconds float64   `json:"audio_seconds"`
	ConsentedAt  time.Time `json:"consented_at"`
}

// Deleted counts what DeleteSession removed.
type Deleted struct {
	SessionID  string `json:"session_id"`
	Searches   int64  `json:"searches"`
	Events     int64  `json:"events"`
	Recordings int64  `json:"recordings"`
}

// ExportSession returns the session's searches (with transcripts), its
// interaction events, and the recordings it asked to keep.
func (s *Service) ExportSession(ctx context.Context, sessionID string) (SessionData, error) {
	out := SessionData{SessionID: sessionID, ExportedAt: time.Now().UTC(),
		Searches: []StoredSearch{}, Events: []json.RawMessage{}, Recordings: []RecordingInfo{}}
	if err := checkSession(sessionID); err != nil {
		return out, err
	}
	rows, err := s.Pool.Query(ctx, `SELECT query_id FROM searches WHERE session_id = $1 ORDER BY created_at`, sessionID)
	if err != nil {
		return out, err
	}
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			rows.Close()
			return out, err
		}
		ids = append(ids, id)
	}
	rows.Close()
	for _, id := range ids {
		stored, err := s.GetSearch(ctx, id)
		if err != nil {
			return out, err
		}
		out.Searches = append(out.Searches, stored)
	}

	events, err := s.Pool.Query(ctx, `
		SELECT row_to_json(e) FROM interaction_events e WHERE session_id = $1 ORDER BY occurred_at, id`, sessionID)
	if err != nil {
		return out, err
	}
	defer events.Close()
	for events.Next() {
		var raw []byte
		if err := events.Scan(&raw); err != nil {
			return out, err
		}
		out.Events = append(out.Events, json.RawMessage(raw))
	}
	if err := events.Err(); err != nil {
		return out, err
	}

	recs, err := s.Pool.Query(ctx, `
		SELECT query_id, object_key, bytes, audio_seconds, consented_at FROM voice_recordings
		WHERE session_id = $1 ORDER BY created_at`, sessionID)
	if err != nil {
		return out, err
	}
	defer recs.Close()
	for recs.Next() {
		var r RecordingInfo
		if err := recs.Scan(&r.QueryID, &r.ObjectKey, &r.Bytes, &r.AudioSeconds, &r.ConsentedAt); err != nil {
			return out, err
		}
		out.Recordings = append(out.Recordings, r)
	}
	return out, recs.Err()
}

// DeleteSession removes the session's searches, events and retained
// recordings. Objects are deleted from the store first, so a failure
// there leaves the rows (and a retry) rather than an orphaned recording.
func (s *Service) DeleteSession(ctx context.Context, sessionID string) (Deleted, error) {
	out := Deleted{SessionID: sessionID}
	if err := checkSession(sessionID); err != nil {
		return out, err
	}
	rows, err := s.Pool.Query(ctx, `SELECT object_key FROM voice_recordings WHERE session_id = $1`, sessionID)
	if err != nil {
		return out, err
	}
	var keys []string
	for rows.Next() {
		var key string
		if err := rows.Scan(&key); err != nil {
			rows.Close()
			return out, err
		}
		keys = append(keys, key)
	}
	rows.Close()
	for _, key := range keys {
		if s.Recordings == nil {
			return out, fmt.Errorf("recording %s cannot be deleted: storage is not configured", key)
		}
		if err := s.Recordings.Delete(ctx, key); err != nil {
			return out, fmt.Errorf("delete recording: %w", err)
		}
	}

	tx, err := s.Pool.Begin(ctx)
	if err != nil {
		return out, err
	}
	defer tx.Rollback(ctx) //nolint:errcheck
	tag, err := tx.Exec(ctx, `DELETE FROM voice_recordings WHERE session_id = $1`, sessionID)
	if err != nil {
		return out, err
	}
	out.Recordings = tag.RowsAffected()
	tag, err = tx.Exec(ctx, `DELETE FROM interaction_events WHERE session_id = $1`, sessionID)
	if err != nil {
		return out, err
	}
	out.Events = tag.RowsAffected()
	tag, err = tx.Exec(ctx, `DELETE FROM searches WHERE session_id = $1`, sessionID)
	if err != nil {
		return out, err
	}
	out.Searches = tag.RowsAffected()
	if err := tx.Commit(ctx); err != nil {
		return out, err
	}
	slog.Info("session data deleted", "session_id", sessionID, "searches", out.Searches,
		"events", out.Events, "recordings", out.Recordings)
	return out, nil
}
