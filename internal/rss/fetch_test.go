package rss

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
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

	f := &Fetcher{AllowPrivate: true}
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
		f := &Fetcher{AllowPrivate: true}
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

func TestFetchRefusesUnsafeURLs(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("<rss/>"))
	}))
	defer server.Close()
	redirecting := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, server.URL, http.StatusFound)
	}))
	defer redirecting.Close()

	f := &Fetcher{} // production defaults
	for _, u := range []string{
		server.URL,                   // loopback
		"http://10.0.0.1/feed.xml",   // private
		"http://169.254.169.254/",    // link-local (cloud metadata)
		"http://localhost:8080/feed", // loopback by name
		"ftp://example.com/feed.xml", // not http(s)
		"file:///etc/passwd",
	} {
		_, err := f.Fetch(context.Background(), u, "", "")
		if err == nil {
			t.Errorf("%s: fetched", u)
			continue
		}
		if !IsPermanent(err) {
			t.Errorf("%s: not permanent: %v", u, err)
		}
	}

	// A test-only client that can reach loopback still refuses to follow a
	// redirect to a private address.
	g := &Fetcher{Client: NewSafeClient(5 * time.Second), AllowPrivate: true}
	g.Client.Transport = http.DefaultTransport // reach the loopback redirector
	if _, err := g.Fetch(context.Background(), redirecting.URL, "", ""); err == nil {
		t.Error("redirect to a loopback address was followed")
	}
}

func TestPublicIP(t *testing.T) {
	cases := map[string]bool{
		"8.8.8.8": true, "93.184.216.34": true, "2606:4700::1111": true,
		"127.0.0.1": false, "10.1.2.3": false, "172.16.5.5": false, "192.168.1.1": false,
		"169.254.169.254": false, "100.64.0.1": false, "0.0.0.0": false, "224.0.0.1": false,
		"255.255.255.255": false, "::1": false, "fe80::1": false, "fc00::1": false, "2001:db8::1": false,
	}
	for ip, want := range cases {
		if got := publicIP(net.ParseIP(ip)); got != want {
			t.Errorf("publicIP(%s) = %v, want %v", ip, got, want)
		}
	}
}
