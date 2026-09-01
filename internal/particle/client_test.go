package particle

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

const listBody = `{
	"data": [
		{
			"id": "1a2b3c",
			"title": "Practical AI",
			"url": "https://changelog.com/practicalai/feed",
			"description": "Making AI practical.",
			"language": "en",
			"image_url": "https://example.com/art.jpg",
			"popularity": 0.97,
			"publisher": {"id": "p1", "name": "Changelog Media", "slug": "changelog"},
			"topics": [{"name": "Technology"}, {"name": "Machine Learning"}]
		},
		{
			"id": "no-feed",
			"title": "Video-only show",
			"url": "",
			"language": "en"
		}
	],
	"cursor": "", "has_more": false
}`

func TestSearchAuthenticatesAndMaps(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/podcasts/search" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if q := r.URL.Query().Get("q"); q != "machine learning" {
			t.Errorf("q = %q", q)
		}
		if key := r.Header.Get("X-API-Key"); key != "test-key" {
			t.Errorf("X-API-Key = %q", key)
		}
		if r.Header.Get("User-Agent") == "" {
			t.Error("no user agent")
		}
		w.Write([]byte(listBody))
	}))
	defer server.Close()

	c := &Client{APIKey: "test-key", BaseURL: server.URL, MinInterval: time.Millisecond}
	pods, err := c.Search(context.Background(), "machine learning", 10)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	// The feed-less entry is dropped: nothing to ingest without RSS.
	if len(pods) != 1 {
		t.Fatalf("got %d podcasts, want 1", len(pods))
	}
	p := pods[0]
	if p.ID != "1a2b3c" || p.FeedURL != "https://changelog.com/practicalai/feed" {
		t.Errorf("podcast = %+v", p)
	}
	if p.Publisher != "Changelog Media" || p.Language != "en" || p.Popularity != 0.97 {
		t.Errorf("podcast = %+v", p)
	}
	if len(p.Categories) != 2 || p.Categories[0] != "Technology" {
		t.Errorf("categories = %v", p.Categories)
	}
}

func TestTrendingSortsByPopularity(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/podcasts" {
			t.Errorf("path = %q", r.URL.Path)
		}
		q := r.URL.Query()
		if q.Get("sort") != "popularity" {
			t.Errorf("sort = %q", q.Get("sort"))
		}
		if q.Get("language") != "en" {
			t.Errorf("language = %q", q.Get("language"))
		}
		if q.Get("limit") != "25" {
			t.Errorf("limit = %q", q.Get("limit"))
		}
		w.Write([]byte(listBody))
	}))
	defer server.Close()

	c := &Client{APIKey: "test-key", BaseURL: server.URL, MinInterval: time.Millisecond}
	pods, err := c.Trending(context.Background(), 25, "en")
	if err != nil {
		t.Fatalf("trending: %v", err)
	}
	if len(pods) != 1 {
		t.Fatalf("got %d podcasts, want 1", len(pods))
	}
}

func TestTrendingFollowsCursorPagination(t *testing.T) {
	pages := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		pages++
		q := r.URL.Query()
		switch q.Get("cursor") {
		case "":
			if q.Get("limit") != "100" {
				t.Errorf("first page limit = %q", q.Get("limit"))
			}
			w.Write([]byte(`{"data":[{"id":"a","title":"A","url":"https://a.example/feed"}],"cursor":"next-1","has_more":true}`))
		case "next-1":
			w.Write([]byte(`{"data":[{"id":"b","title":"B","url":"https://b.example/feed"}],"cursor":"","has_more":false}`))
		default:
			t.Errorf("unexpected cursor %q", q.Get("cursor"))
		}
	}))
	defer server.Close()

	c := &Client{APIKey: "k", BaseURL: server.URL, MinInterval: time.Millisecond}
	pods, err := c.Trending(context.Background(), 150, "")
	if err != nil {
		t.Fatalf("trending: %v", err)
	}
	if pages != 2 || len(pods) != 2 {
		t.Fatalf("pages=%d podcasts=%d, want 2 and 2", pages, len(pods))
	}
	if pods[0].ID != "a" || pods[1].ID != "b" {
		t.Fatalf("order wrong: %+v", pods)
	}
}

func TestHTTPErrorSurfacesBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"error": {"message": "invalid api key"}}`))
	}))
	defer server.Close()

	c := &Client{APIKey: "bad", BaseURL: server.URL, MinInterval: time.Millisecond}
	if _, err := c.Search(context.Background(), "x", 1); err == nil {
		t.Fatal("HTTP 401 returned no error")
	}
}

func TestMissingKeyFailsFast(t *testing.T) {
	c := &Client{}
	if _, err := c.Trending(context.Background(), 5, ""); err == nil {
		t.Fatal("missing API key returned no error")
	}
}
