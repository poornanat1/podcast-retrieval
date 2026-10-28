package metrics

import (
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"podfind/internal/search"
)

func TestObserveSearchAndVoice(t *testing.T) {
	conf := -1.7
	resp := &search.VoiceResponse{
		Response: search.Response{
			Input:     search.Input{Mode: search.InputVoice},
			Versions:  search.Versions{Ordering: "ranker", BundleID: "b1"},
			TimingsMS: map[string]float64{"retrieve": 300, "rerank": 12},
			Legs: map[string]*search.LegResult{
				"lexical": {Status: "timeout"}, "e5": {Status: "ok", Count: 200}, "two_tower": nil},
			Results: []search.Result{{Score: 3.5, Sources: []string{"e5", "two_tower"}}, {Score: 1, Sources: []string{"e5"}}},
		},
		Transcript: search.Transcript{Confidence: &conf, Language: "en", AudioSeconds: 4, ElapsedMS: 400},
	}
	ObserveVoice(resp)
	ObserveSearch(&search.Response{Versions: search.Versions{Ordering: "rrf", BundleID: "b1"}})

	if got := testutil.ToFloat64(Searches.WithLabelValues("voice", "ranker", "b1")); got != 1 {
		t.Errorf("voice searches = %v", got)
	}
	if got := testutil.ToFloat64(Searches.WithLabelValues("text", "rrf", "b1")); got != 1 {
		t.Errorf("text searches = %v", got)
	}
	if got := testutil.ToFloat64(ZeroResults.WithLabelValues("text")); got != 1 {
		t.Errorf("zero results = %v", got)
	}
	if got := testutil.ToFloat64(SearchLegs.WithLabelValues("lexical", "timeout")); got != 1 {
		t.Errorf("lexical timeouts = %v", got)
	}
	if got := testutil.ToFloat64(ResultSources.WithLabelValues("e5")); got != 2 {
		t.Errorf("e5 sources = %v", got)
	}
	if got := testutil.ToFloat64(VoiceLowConfidence); got != 1 {
		t.Errorf("low confidence = %v", got)
	}
	if got := testutil.ToFloat64(VoiceLanguage.WithLabelValues("en")); got != 1 {
		t.Errorf("language = %v", got)
	}
	if n := testutil.CollectAndCount(SearchStage); n == 0 {
		t.Error("no stage observations")
	}
	if err := testutil.CollectAndCompare(VoiceRetained, strings.NewReader(`
# HELP podfind_voice_retained_total Voice searches by whether the recording was kept.
# TYPE podfind_voice_retained_total counter
podfind_voice_retained_total{retained="false"} 1
`)); err != nil {
		t.Error(err)
	}
}
