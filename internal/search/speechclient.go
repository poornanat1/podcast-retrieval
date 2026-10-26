package search

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

// Transcript is what the speech server returned for a recording. It is
// stored with the search so a spoken query can be traced to the recognizer
// and its confidence.
type Transcript struct {
	Text         string   `json:"text"`
	Language     string   `json:"language"`
	Confidence   *float64 `json:"confidence"`
	Backend      string   `json:"backend"`
	AudioSeconds float64  `json:"audio_seconds"`
	ElapsedMS    float64  `json:"elapsed_ms"`
}

// ErrUnavailable marks a failed dependency (the speech server).
var ErrUnavailable = errors.New("unavailable")

// SpeechClient calls the speech server.
type SpeechClient struct {
	BaseURL string
	HTTP    *http.Client
}

// NewSpeechClient returns a client with a bounded per-request timeout.
func NewSpeechClient(baseURL string, timeout time.Duration) *SpeechClient {
	return &SpeechClient{BaseURL: baseURL, HTTP: &http.Client{Timeout: timeout}}
}

// Transcribe posts a WAV recording and returns its transcript. language is
// an optional ISO 639-1 hint; empty means detect.
func (c *SpeechClient) Transcribe(ctx context.Context, wav []byte, language string) (Transcript, error) {
	var t Transcript
	if c == nil {
		return t, fmt.Errorf("%w: no speech server configured", ErrUnavailable)
	}
	endpoint := c.BaseURL + "/v1/transcribe"
	if language != "" {
		endpoint += "?language=" + url.QueryEscape(language)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(wav))
	if err != nil {
		return t, err
	}
	req.Header.Set("Content-Type", "audio/wav")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return t, fmt.Errorf("%w: speech server: %v", ErrUnavailable, err)
	}
	defer resp.Body.Close()
	msg, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	switch {
	case resp.StatusCode == http.StatusUnprocessableEntity:
		return t, fmt.Errorf("%w: %s", ErrBadRequest, detail(msg))
	case resp.StatusCode != http.StatusOK:
		return t, fmt.Errorf("%w: speech server: %s: %s", ErrUnavailable, resp.Status, detail(msg))
	}
	if err := json.Unmarshal(msg, &t); err != nil {
		return t, fmt.Errorf("%w: speech server: %v", ErrUnavailable, err)
	}
	return t, nil
}

func detail(msg []byte) string {
	var body struct {
		Detail string `json:"detail"`
	}
	if json.Unmarshal(msg, &body) == nil && body.Detail != "" {
		return body.Detail
	}
	return string(msg)
}
