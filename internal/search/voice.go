package search

import (
	"context"
	"fmt"
	"log/slog"
	"time"
)

// Input modes.
const (
	InputText        = "text"         // typed
	InputVoice       = "voice"        // spoken; the transcript was searched as returned
	InputVoiceEdited = "voice_edited" // spoken, then the transcript was edited before searching
)

// Input records how a query was entered. Clients may send mode text or
// voice_edited (with the voice search the transcript came from); mode
// voice, with its transcript, is set by SearchVoice only.
type Input struct {
	Mode          string      `json:"mode"`
	ParentQueryID string      `json:"parent_query_id,omitempty"`
	Transcript    *Transcript `json:"transcript,omitempty"`
}

// VoiceRequest is a spoken search: a WAV recording plus the fields of a
// text search. Retain asks to keep the recording (consent); without it the
// audio is dropped as soon as the transcript is back.
type VoiceRequest struct {
	Audio     []byte
	Language  string // ISO 639-1 hint for the recognizer, or empty to detect
	Filters   Filters
	SessionID string
	Limit     int
	Retain    bool

	// Debug-only, as on Request.
	Debug             bool
	ReferenceTime     *time.Time
	SkipUnderstanding bool
}

// VoiceResponse is a served voice search: the text search's response plus
// the transcript it ran and whether the recording was kept.
type VoiceResponse struct {
	Response
	Transcript Transcript `json:"transcript"`
	Retained   bool       `json:"retained"`
	Retention  string     `json:"retention,omitempty"` // why a requested retention did not happen
}

// Recordings stores retained voice recordings (object storage).
type Recordings interface {
	Put(ctx context.Context, key string, data []byte, contentType string) error
	Delete(ctx context.Context, key string) error
}

// RecordingKey is the object key of a retained recording.
func RecordingKey(queryID string) string { return "voice/" + queryID + ".wav" }

// SearchVoice transcribes the recording through the speech server, runs
// the transcript as a text search (query understanding included), stores
// the transcript with the search, and keeps the recording only when asked
// to and a store is configured.
func (s *Service) SearchVoice(ctx context.Context, req VoiceRequest) (VoiceResponse, error) {
	start := time.Now()
	if len(req.Audio) > MaxVoiceBytes {
		return VoiceResponse{}, fmt.Errorf("%w: recording larger than %d bytes", ErrBadRequest, MaxVoiceBytes)
	}
	info, err := ParseWAV(req.Audio)
	if err != nil {
		return VoiceResponse{}, err
	}
	if err := checkSession(req.SessionID); err != nil {
		return VoiceResponse{}, err
	}
	transcript, err := s.Speech.Transcribe(ctx, req.Audio, req.Language)
	if err != nil {
		return VoiceResponse{}, err
	}
	transcribeMS := msSince(start)
	if transcript.Text == "" {
		return VoiceResponse{}, fmt.Errorf("%w: no speech recognized", ErrBadRequest)
	}
	resp, err := s.Search(ctx, Request{
		Query: transcript.Text, Filters: req.Filters, SessionID: req.SessionID, Limit: req.Limit,
		Debug: req.Debug, ReferenceTime: req.ReferenceTime, SkipUnderstanding: req.SkipUnderstanding,
		Input:   &Input{Mode: InputVoice, Transcript: &transcript},
		started: start, timings: map[string]float64{"transcribe_ms": transcribeMS},
	})
	if err != nil {
		return VoiceResponse{}, err
	}
	out := VoiceResponse{Response: resp, Transcript: transcript}
	if req.Retain {
		out.Retained, out.Retention = s.retain(ctx, req, resp.QueryID, info)
	}
	return out, nil
}

func (s *Service) retain(ctx context.Context, req VoiceRequest, queryID string, info WAVInfo) (bool, string) {
	if s.Recordings == nil {
		return false, "recording storage is not configured"
	}
	key := RecordingKey(queryID)
	if err := s.Recordings.Put(ctx, key, req.Audio, "audio/wav"); err != nil {
		slog.Error("retain recording", "query_id", queryID, "error", err)
		return false, "storing the recording failed"
	}
	if _, err := s.Pool.Exec(ctx, `
		INSERT INTO voice_recordings (query_id, session_id, object_key, content_type, bytes, audio_seconds, consented_at)
		VALUES ($1, $2, $3, 'audio/wav', $4, $5, $6)`,
		queryID, req.SessionID, key, len(req.Audio), info.Seconds, time.Now().UTC()); err != nil {
		slog.Error("record retention", "query_id", queryID, "error", err)
		_ = s.Recordings.Delete(ctx, key)
		return false, "recording the retention failed"
	}
	return true, ""
}

// checkInput validates a client-supplied Input on a text search.
func (s *Service) checkInput(ctx context.Context, in *Input, sessionID string) error {
	switch {
	case in == nil, in.Mode == "" || in.Mode == InputText:
		if in != nil && (in.ParentQueryID != "" || in.Transcript != nil) {
			return fmt.Errorf("%w: input.parent_query_id and input.transcript need mode voice_edited", ErrBadRequest)
		}
		return nil
	case in.Mode == InputVoiceEdited:
		if in.Transcript != nil {
			return fmt.Errorf("%w: input.transcript is set by the server", ErrBadRequest)
		}
		parent, err := s.GetSearch(ctx, in.ParentQueryID)
		if err != nil || parent.SessionID != sessionID || parent.Input.Mode != InputVoice {
			return fmt.Errorf("%w: input.parent_query_id must be a voice search of this session", ErrBadRequest)
		}
		return nil
	default:
		return fmt.Errorf("%w: input.mode must be text or voice_edited", ErrBadRequest)
	}
}
