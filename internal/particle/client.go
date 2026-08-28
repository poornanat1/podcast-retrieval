// Package particle is a client for the Particle data platform
// (https://docs.particle.pro), used to discover podcasts and their RSS feed
// URLs. Only discovery metadata is consumed; episode content always comes
// from the publisher's RSS feed.
package particle

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"

	"podfind/internal/discovery"
)

// DefaultBaseURL is the production API root.
const DefaultBaseURL = "https://api.particle.pro"

// maxPageSize is the API's documented per-page ceiling.
const maxPageSize = 100

// Client calls the Particle REST API with an API key and client-side rate
// limiting.
type Client struct {
	APIKey      string
	BaseURL     string        // default DefaultBaseURL
	HTTP        *http.Client  // default 30s timeout
	UserAgent   string        // default podfind identifier
	MinInterval time.Duration // min gap between requests, default 250ms

	mu   sync.Mutex
	last time.Time
}

// Source implements discovery.Provider.
func (c *Client) Source() string { return "particle" }

// podcastJSON mirrors the API's Podcast object; url is the RSS feed URL.
type podcastJSON struct {
	ID          string  `json:"id"`
	Title       string  `json:"title"`
	URL         string  `json:"url"`
	Description string  `json:"description"`
	Language    string  `json:"language"`
	ImageURL    string  `json:"image_url"`
	Popularity  float64 `json:"popularity"`
	Publisher   struct {
		Name string `json:"name"`
	} `json:"publisher"`
	Topics []struct {
		Name string `json:"name"`
	} `json:"topics"`
}

type listResponse struct {
	Data    []podcastJSON `json:"data"`
	Cursor  string        `json:"cursor"`
	HasMore bool          `json:"has_more"`
}

// Search implements discovery.Provider via GET /v1/podcasts/search.
func (c *Client) Search(ctx context.Context, query string, max int) ([]discovery.Podcast, error) {
	return c.listPages(ctx, "/v1/podcasts/search", url.Values{"q": {query}}, max)
}

// Trending implements discovery.Provider via GET /v1/podcasts ordered by
// global popularity.
func (c *Client) Trending(ctx context.Context, max int, language string) ([]discovery.Podcast, error) {
	v := url.Values{"sort": {"popularity"}}
	if language != "" {
		v.Set("language", language)
	}
	return c.listPages(ctx, "/v1/podcasts", v, max)
}

// listPages follows the cursor until max podcasts are collected or the
// listing is exhausted.
func (c *Client) listPages(ctx context.Context, path string, query url.Values, max int) ([]discovery.Podcast, error) {
	if max <= 0 {
		max = 25
	}
	var out []discovery.Podcast
	cursor := ""
	for len(out) < max {
		pageSize := max - len(out)
		if pageSize > maxPageSize {
			pageSize = maxPageSize
		}
		q := url.Values{}
		for k, vs := range query {
			q[k] = vs
		}
		q.Set("limit", strconv.Itoa(pageSize))
		if cursor != "" {
			q.Set("cursor", cursor)
		}
		page, next, hasMore, err := c.list(ctx, path, q)
		if err != nil {
			return out, err
		}
		out = append(out, page...)
		if !hasMore || next == "" || len(page) == 0 {
			break
		}
		cursor = next
	}
	if len(out) > max {
		out = out[:max]
	}
	return out, nil
}

func (c *Client) list(ctx context.Context, path string, query url.Values) ([]discovery.Podcast, string, bool, error) {
	if c.APIKey == "" {
		return nil, "", false, fmt.Errorf("particle: API key not configured")
	}
	c.throttle()

	base := c.BaseURL
	if base == "" {
		base = DefaultBaseURL
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+path+"?"+query.Encode(), nil)
	if err != nil {
		return nil, "", false, fmt.Errorf("particle: build request: %w", err)
	}
	req.Header.Set("X-API-Key", c.APIKey)
	ua := c.UserAgent
	if ua == "" {
		ua = "PodFind/0.1"
	}
	req.Header.Set("User-Agent", ua)

	client := c.HTTP
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, "", false, fmt.Errorf("particle: %s: %w", path, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 10<<20))
	if err != nil {
		return nil, "", false, fmt.Errorf("particle: read %s: %w", path, err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, "", false, fmt.Errorf("particle: %s: HTTP %d: %s", path, resp.StatusCode, truncate(body, 200))
	}

	var parsed listResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, "", false, fmt.Errorf("particle: decode %s: %w", path, err)
	}

	out := make([]discovery.Podcast, 0, len(parsed.Data))
	for _, p := range parsed.Data {
		if p.URL == "" {
			continue // no RSS feed: nothing for the ingestion pipeline
		}
		pod := discovery.Podcast{
			ID:          p.ID,
			Title:       p.Title,
			FeedURL:     p.URL,
			Description: p.Description,
			Publisher:   p.Publisher.Name,
			ArtworkURL:  p.ImageURL,
			Language:    p.Language,
		}
		for _, t := range p.Topics {
			if t.Name != "" {
				pod.Categories = append(pod.Categories, t.Name)
			}
		}
		out = append(out, pod)
	}
	return out, parsed.Cursor, parsed.HasMore, nil
}

func truncate(b []byte, n int) string {
	if len(b) > n {
		return string(b[:n]) + "…"
	}
	return string(b)
}

// throttle enforces the client-side minimum interval between requests.
func (c *Client) throttle() {
	interval := c.MinInterval
	if interval <= 0 {
		interval = 250 * time.Millisecond
	}
	c.mu.Lock()
	wait := interval - time.Since(c.last)
	if wait > 0 {
		time.Sleep(wait)
	}
	c.last = time.Now()
	c.mu.Unlock()
}
