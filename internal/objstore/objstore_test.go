package objstore

import "testing"

func TestEndpointURL(t *testing.T) {
	cases := []struct {
		raw       string
		ssl       bool
		url       string
		pathStyle bool
	}{
		{"localhost:9000", false, "http://localhost:9000", true},
		{"rustfs:9000", true, "https://rustfs:9000", true},
		{"https://t3.storageapi.dev", false, "https://t3.storageapi.dev", false},
		{"https://acct.r2.cloudflarestorage.com", true, "https://acct.r2.cloudflarestorage.com", false},
	}
	for _, c := range cases {
		url, pathStyle := EndpointURL(c.raw, c.ssl)
		if url != c.url || pathStyle != c.pathStyle {
			t.Errorf("EndpointURL(%q, %v) = %q, %v; want %q, %v", c.raw, c.ssl, url, pathStyle, c.url, c.pathStyle)
		}
	}
}

func TestConfigFromEnv(t *testing.T) {
	t.Setenv("OBJECT_STORE_ENDPOINT", "")
	if _, ok, err := ConfigFromEnv(); ok || err != nil {
		t.Fatalf("unset endpoint: ok=%v err=%v", ok, err)
	}
	t.Setenv("OBJECT_STORE_ENDPOINT", "localhost:9000")
	if _, _, err := ConfigFromEnv(); err == nil {
		t.Fatal("missing bucket accepted")
	}
	t.Setenv("OBJECT_STORE_BUCKET", "transcripts")
	t.Setenv("OBJECT_STORE_PATH_STYLE", "false")
	cfg, ok, err := ConfigFromEnv()
	if err != nil || !ok {
		t.Fatalf("ok=%v err=%v", ok, err)
	}
	if cfg.Endpoint != "http://localhost:9000" || cfg.Region != "us-east-1" || cfg.PathStyle {
		t.Errorf("config = %+v", cfg)
	}
}
