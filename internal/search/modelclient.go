package search

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// ModelInfo is the served bundle's manifest, as GET /v1/model returns it.
type ModelInfo struct {
	BundleID       string            `json:"bundle_id"`
	FeatureVersion string            `json:"feature_version"`
	Features       []string          `json:"features"`
	CandidateDepth int               `json:"candidate_depth"`
	IndexModels    map[string]string `json:"index_models"`
}

// IndexVersion identifies the vector indexes a search read, for attribution.
func (m ModelInfo) IndexVersion() string {
	return fmt.Sprintf("two_tower=%s;e5=%s;transcript=%s",
		m.IndexModels["two_tower"], m.IndexModels["e5"], m.IndexModels["transcript"])
}

// ModelQuery is the query as the model server and the shared feature code
// see it.
type ModelQuery struct {
	QueryText          string  `json:"query_text"`
	Language           *string `json:"language"`
	MaxDurationSeconds *int    `json:"max_duration_seconds"`
	PublishedAfter     *string `json:"published_after"`
	NoExplicit         bool    `json:"no_explicit"`
}

func modelQuery(u Understood) ModelQuery {
	q := ModelQuery{QueryText: u.Text, NoExplicit: u.Filters.NoExplicit}
	if u.Filters.Language != "" {
		q.Language = &u.Filters.Language
	}
	if u.Filters.MaxDurationSeconds > 0 {
		q.MaxDurationSeconds = &u.Filters.MaxDurationSeconds
	}
	if u.Filters.PublishedAfter != "" {
		q.PublishedAfter = &u.Filters.PublishedAfter
	}
	return q
}

// Vectors are the query embeddings the vector legs search with.
type Vectors struct {
	BundleID  string    `json:"bundle_id"`
	TwoTower  []float32 `json:"two_tower"`
	E5        []float32 `json:"e5"`
	ElapsedMS float64   `json:"elapsed_ms"`
}

// Rerank is the model server's scoring of one candidate set.
type Rerank struct {
	BundleID       string      `json:"bundle_id"`
	FeatureVersion string      `json:"feature_version"`
	Scores         []float64   `json:"scores"`
	Features       [][]float64 `json:"features"`
	ElapsedMS      float64     `json:"elapsed_ms"`
}

// ModelClient calls the model server.
type ModelClient struct {
	BaseURL string
	HTTP    *http.Client
}

// NewModelClient returns a client with a bounded per-request timeout.
func NewModelClient(baseURL string, timeout time.Duration) *ModelClient {
	return &ModelClient{BaseURL: baseURL, HTTP: &http.Client{Timeout: timeout}}
}

func (c *ModelClient) do(ctx context.Context, method, path string, body, out any) error {
	var reader io.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			return err
		}
		reader = bytes.NewReader(raw)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.BaseURL+path, reader)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return fmt.Errorf("model server %s: %w", path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return fmt.Errorf("model server %s: %s: %s", path, resp.Status, msg)
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// Model fetches the served bundle's manifest.
func (c *ModelClient) Model(ctx context.Context) (ModelInfo, error) {
	var m ModelInfo
	err := c.do(ctx, http.MethodGet, "/v1/model", nil, &m)
	return m, err
}

// Encode returns the query's two-tower and E5 vectors.
func (c *ModelClient) Encode(ctx context.Context, q ModelQuery) (Vectors, error) {
	var v Vectors
	err := c.do(ctx, http.MethodPost, "/v1/encode", map[string]any{"query": q}, &v)
	return v, err
}

// Rerank scores the candidates in order. referenceTime is honored only by
// callers that need reproducible freshness features (parity checks); nil
// means "now" on the server.
func (c *ModelClient) Rerank(ctx context.Context, q ModelQuery, rows []Evidence, referenceTime *time.Time, withFeatures bool) (Rerank, error) {
	var r Rerank
	body := map[string]any{"query": q, "candidates": rows, "return_features": withFeatures}
	if referenceTime != nil {
		body["reference_time"] = referenceTime.UTC().Format(time.RFC3339Nano)
	}
	err := c.do(ctx, http.MethodPost, "/v1/rerank", body, &r)
	return r, err
}
