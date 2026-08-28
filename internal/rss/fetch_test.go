package rss

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestFetchConditionalRequests(t *testing.T) {
	const body = "<rss><channel><title>x</title></channel></rss>"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if ua := r.Header.Get("User-Agent"); ua != DefaultUserAgent {
			t.Errorf("user agent = %q", ua)
		}
		if r.Header.Get("If-None-Match") == `"v1"` {
			w.WriteHeader(http.StatusNotModified)
			return
		}
		w.Header().Set("ETag", `"v1"`)
		w.Header().Set("Last-Modified", "Mon, 17 Aug 2026 09:30:00 GMT")
		w.Write([]byte(body))
	}))
	defer server.Close()

	f := &Fetcher{}
	ctx := context.Background()

	first, err := f.Fetch(ctx, server.URL, "", "")
	if err != nil {
		t.Fatalf("first fetch: %v", err)
	}
	if first.NotModified || string(first.Body) != body {
		t.Fatalf("first fetch: notModified=%v body=%q", first.NotModified, first.Body)
	}
	if first.ETag != `"v1"` || first.LastModified == "" {
		t.Fatalf("validators not captured: etag=%q lastModified=%q", first.ETag, first.LastModified)
	}

	second, err := f.Fetch(ctx, server.URL, first.ETag, first.LastModified)
	if err != nil {
		t.Fatalf("second fetch: %v", err)
	}
	if !second.NotModified {
		t.Fatal("second fetch was not a 304")
	}
}

func TestFetchStatusClassification(t *testing.T) {
	cases := map[int]bool{ // status → permanent?
		http.StatusNotFound:            true,
		http.StatusGone:                true,
		http.StatusForbidden:           true,
		http.StatusTooManyRequests:     false,
		http.StatusInternalServerError: false,
		http.StatusBadGateway:          false,
	}
	for status, wantPermanent := range cases {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(status)
		}))
		f := &Fetcher{}
		_, err := f.Fetch(context.Background(), server.URL, "", "")
		server.Close()
		if err == nil {
			t.Errorf("status %d: no error", status)
			continue
		}
		if IsPermanent(err) != wantPermanent {
			t.Errorf("status %d: IsPermanent = %v, want %v", status, IsPermanent(err), wantPermanent)
		}
	}
}
