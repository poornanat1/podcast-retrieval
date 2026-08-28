// Package rss fetches and parses public podcast RSS feeds. Only metadata is
// processed — audio enclosures are recorded as URLs, never downloaded.
package rss

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

// maxFeedBytes bounds how much of a feed we read; feeds beyond this are
// almost certainly malformed or abusive.
const maxFeedBytes = 20 << 20 // 20 MiB

// DefaultUserAgent identifies the crawler politely, per provider terms.
const DefaultUserAgent = "PodFindBot/0.1 (+https://github.com/poornanat1/podcast-retrieval)"

// StatusError reports a non-success HTTP response.
type StatusError struct {
	URL  string
	Code int
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("fetch %s: HTTP %d", e.URL, e.Code)
}

// Permanent reports whether retrying is pointless: client errors other than
// rate limiting mean the feed is gone or was never valid.
func (e *StatusError) Permanent() bool {
	return e.Code >= 400 && e.Code < 500 && e.Code != http.StatusTooManyRequests
}

// IsPermanent reports whether err represents an unretryable fetch failure.
func IsPermanent(err error) bool {
	var se *StatusError
	return errors.As(err, &se) && se.Permanent()
}

// Result is the outcome of one conditional fetch.
type Result struct {
	NotModified  bool
	Body         []byte
	ETag         string
	LastModified string
}

// Fetcher retrieves feeds with conditional requests so unchanged feeds cost
// a 304 instead of a full transfer.
type Fetcher struct {
	Client    *http.Client // default: 30s timeout
	UserAgent string       // default: DefaultUserAgent
}

// Fetch requests url, sending If-None-Match / If-Modified-Since when prior
// validators are known.
func (f *Fetcher) Fetch(ctx context.Context, url, etag, lastModified string) (*Result, error) {
	client := f.Client
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	ua := f.UserAgent
	if ua == "" {
		ua = DefaultUserAgent
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("build request for %s: %w", url, err)
	}
	req.Header.Set("User-Agent", ua)
	req.Header.Set("Accept", "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.5")
	if etag != "" {
		req.Header.Set("If-None-Match", etag)
	}
	if lastModified != "" {
		req.Header.Set("If-Modified-Since", lastModified)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetch %s: %w", url, err)
	}
	defer resp.Body.Close()

	switch {
	case resp.StatusCode == http.StatusNotModified:
		return &Result{NotModified: true, ETag: etag, LastModified: lastModified}, nil
	case resp.StatusCode != http.StatusOK:
		return nil, &StatusError{URL: url, Code: resp.StatusCode}
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxFeedBytes))
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", url, err)
	}
	return &Result{
		Body:         body,
		ETag:         resp.Header.Get("ETag"),
		LastModified: resp.Header.Get("Last-Modified"),
	}, nil
}
