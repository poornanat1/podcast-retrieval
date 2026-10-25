package search

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// ImpressionItem is one displayed result.
type ImpressionItem struct {
	EpisodeID     int64 `json:"episode_id"`
	DisplayedRank int   `json:"displayed_rank"`
}

// Impressions records results a client displayed for a stored search.
type Impressions struct {
	SessionID  string           `json:"session_id"`
	QueryID    string           `json:"query_id"`
	Items      []ImpressionItem `json:"items"`
	ClientTime *time.Time       `json:"client_time,omitempty"`
}

// Playback records a playback event.
type Playback struct {
	SessionID       string     `json:"session_id"`
	QueryID         string     `json:"query_id,omitempty"`
	EpisodeID       int64      `json:"episode_id"`
	PositionSeconds float64    `json:"position_seconds"`
	DurationSeconds float64    `json:"duration_seconds,omitempty"`
	ClientTime      *time.Time `json:"client_time,omitempty"`
}

// PlaybackTypes maps URL actions to event types.
var PlaybackTypes = map[string]string{
	"start": "playback_start", "progress": "playback_progress",
	"skip": "playback_skip", "complete": "playback_complete",
}

type attribution struct {
	versions Versions
	results  map[int64]storedResult
}

func (s *Service) attribution(ctx context.Context, queryID string) (attribution, error) {
	stored, err := s.GetSearch(ctx, queryID)
	if err != nil {
		return attribution{}, err
	}
	a := attribution{versions: stored.Versions, results: map[int64]storedResult{}}
	for _, r := range stored.Results {
		a.results[r.EpisodeID] = r
	}
	return a, nil
}

func checkSession(id string) error {
	if id == "" || len(id) > 128 {
		return fmt.Errorf("%w: session_id is required (at most 128 characters)", ErrBadRequest)
	}
	return nil
}

// RecordImpressions stores one impression event per displayed item, with
// rank, score, candidate sources, and versions taken from the stored search
// rather than from the client. Every item must be a result of that search.
func (s *Service) RecordImpressions(ctx context.Context, in Impressions) (int, error) {
	if err := checkSession(in.SessionID); err != nil {
		return 0, err
	}
	if len(in.Items) == 0 || len(in.Items) > maxLimit {
		return 0, fmt.Errorf("%w: 1 to %d items required", ErrBadRequest, maxLimit)
	}
	a, err := s.attribution(ctx, in.QueryID)
	if errors.Is(err, ErrNotFound) {
		return 0, fmt.Errorf("%w: unknown query_id %q", ErrBadRequest, in.QueryID)
	}
	if err != nil {
		return 0, err
	}
	for _, item := range in.Items {
		if _, ok := a.results[item.EpisodeID]; !ok {
			return 0, fmt.Errorf("%w: episode %d was not a result of %s", ErrBadRequest, item.EpisodeID, in.QueryID)
		}
		if item.DisplayedRank < 1 {
			return 0, fmt.Errorf("%w: displayed_rank must be at least 1", ErrBadRequest)
		}
	}
	tx, err := s.Pool.Begin(ctx)
	if err != nil {
		return 0, err
	}
	defer tx.Rollback(ctx) //nolint:errcheck
	now := time.Now().UTC()
	for _, item := range in.Items {
		r := a.results[item.EpisodeID]
		if _, err := tx.Exec(ctx, `
			INSERT INTO interaction_events (event_type, occurred_at, session_id, query_id, episode_id,
			    displayed_rank, candidate_sources, score, feature_version, model_version, index_version, payload)
			VALUES ('impression', $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)`,
			now, in.SessionID, in.QueryID, item.EpisodeID, item.DisplayedRank, r.Sources, r.Score,
			a.versions.FeatureVersion, a.versions.BundleID, a.versions.IndexVersion,
			jsonText(map[string]any{"served_rank": r.Rank, "ordering": a.versions.Ordering,
				"client_time": in.ClientTime})); err != nil {
			return 0, fmt.Errorf("impression: %w", err)
		}
	}
	return len(in.Items), tx.Commit(ctx)
}

// RecordPlayback stores a playback event. With a query_id, the episode must
// be a result of that search and the event carries its attribution;
// without one (playback started outside search) it is recorded
// unattributed.
func (s *Service) RecordPlayback(ctx context.Context, action string, in Playback) error {
	eventType, ok := PlaybackTypes[action]
	if !ok {
		return fmt.Errorf("%w: unknown playback action %q", ErrBadRequest, action)
	}
	if err := checkSession(in.SessionID); err != nil {
		return err
	}
	if in.EpisodeID <= 0 || in.PositionSeconds < 0 {
		return fmt.Errorf("%w: episode_id and a non-negative position_seconds are required", ErrBadRequest)
	}
	var (
		rank           *int
		score          *float64
		sources        = []string{}
		versions       Versions
		attributedFrom string
	)
	if in.QueryID != "" {
		a, err := s.attribution(ctx, in.QueryID)
		if errors.Is(err, ErrNotFound) {
			return fmt.Errorf("%w: unknown query_id %q", ErrBadRequest, in.QueryID)
		}
		if err != nil {
			return err
		}
		r, ok := a.results[in.EpisodeID]
		if !ok {
			return fmt.Errorf("%w: episode %d was not a result of %s", ErrBadRequest, in.EpisodeID, in.QueryID)
		}
		rank, score, sources, versions, attributedFrom = &r.Rank, &r.Score, r.Sources, a.versions, in.QueryID
	}
	tag, err := s.Pool.Exec(ctx, `
		INSERT INTO interaction_events (event_type, occurred_at, session_id, query_id, episode_id,
		    displayed_rank, candidate_sources, score, feature_version, model_version, index_version, payload)
		SELECT $1, $2, $3, $4, e.id, $5, $6, $7, $8, $9, $10, $11 FROM episodes e WHERE e.id = $12`,
		eventType, time.Now().UTC(), in.SessionID, attributedFrom, rank, sources, score,
		versions.FeatureVersion, versions.BundleID, versions.IndexVersion,
		jsonText(map[string]any{"position_seconds": in.PositionSeconds, "duration_seconds": in.DurationSeconds,
			"client_time": in.ClientTime}), in.EpisodeID)
	if err != nil {
		return fmt.Errorf("playback: %w", err)
	}
	if tag.RowsAffected() == 0 {
		return fmt.Errorf("%w: unknown episode %d", ErrBadRequest, in.EpisodeID)
	}
	return nil
}
