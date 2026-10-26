package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"podfind/internal/search"
	"podfind/web"
)

func TestRoutingAndRequestValidation(t *testing.T) {
	h := New(&search.Service{}, web.Static())
	cases := []struct {
		method, path, body string
		status             int
		contains           string
	}{
		{"POST", "/v1/search/text", `{"query": "x", "unknown": 1}`, http.StatusBadRequest, "unknown field"},
		{"POST", "/v1/search/text", `{not json`, http.StatusBadRequest, "error"},
		{"POST", "/v1/events/impression", `[]`, http.StatusBadRequest, "error"},
		{"POST", "/v1/search/voice", `{"query": "x"}`, http.StatusBadRequest, "error"},
		{"GET", "/v1/episodes/abc", "", http.StatusNotFound, "not found"},
		{"GET", "/v1/me/data", "", http.StatusBadRequest, "session_id"},
		{"DELETE", "/v1/me/data?session_id=", "", http.StatusBadRequest, "session_id"},
		{"DELETE", "/v1/search/text", "", http.StatusMethodNotAllowed, ""},
		{"GET", "/", "", http.StatusOK, "<title>PodFind</title>"},
		{"GET", "/app.js", "", http.StatusOK, "/v1/events/impression"},
	}
	for _, c := range cases {
		req := httptest.NewRequest(c.method, c.path, strings.NewReader(c.body))
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != c.status || !strings.Contains(rec.Body.String(), c.contains) {
			t.Errorf("%s %s: status %d body %q", c.method, c.path, rec.Code, rec.Body.String())
		}
	}
}
