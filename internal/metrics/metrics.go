// Package metrics exposes the API's Prometheus metrics: request rates and
// latency, the search pipeline's stages, legs, result counts and score
// distribution, voice transcription quality, interaction events, and the
// age of the served vector indexes. The drift check and the dashboards
// read these.
package metrics

import (
	"net/http"
	"strconv"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"podfind/internal/search"
)

// LowConfidence is the mean token log-probability below which a transcript
// counts as low confidence (Whisper: clear speech sits around -0.3).
const LowConfidence = -1.0

var (
	secondsBuckets = []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30}

	HTTPRequests = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "podfind_http_requests_total", Help: "HTTP requests by route and status."},
		[]string{"route", "method", "status"})
	HTTPDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name: "podfind_http_request_seconds", Help: "HTTP request latency by route.",
		Buckets: secondsBuckets}, []string{"route"})

	Searches = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "podfind_searches_total", Help: "Searches by input mode, ordering and bundle."},
		[]string{"input", "ordering", "bundle"})
	SearchStage = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name: "podfind_search_stage_seconds", Help: "Search pipeline stage latency.",
		Buckets: secondsBuckets}, []string{"stage"})
	SearchLegs = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "podfind_search_legs_total", Help: "Retrieval leg outcomes."},
		[]string{"leg", "status"})
	SearchLegCandidates = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name: "podfind_search_leg_candidates", Help: "Candidates returned per leg.",
		Buckets: []float64{0, 1, 5, 10, 25, 50, 100, 200, 400}}, []string{"leg"})
	SearchResults = promauto.NewHistogram(prometheus.HistogramOpts{
		Name: "podfind_search_results", Help: "Results returned per search.",
		Buckets: []float64{0, 1, 2, 5, 10, 20, 50, 100}})
	ZeroResults = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "podfind_search_zero_results_total", Help: "Searches that returned nothing."},
		[]string{"input"})
	TopScore = promauto.NewHistogram(prometheus.HistogramOpts{
		Name: "podfind_search_top_score", Help: "Ranker score of the first result.",
		Buckets: prometheus.LinearBuckets(-4, 0.5, 29)})
	ResultSources = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "podfind_search_result_sources_total", Help: "Served results by the leg that found them."},
		[]string{"source"})

	VoiceTranscribe = promauto.NewHistogram(prometheus.HistogramOpts{
		Name: "podfind_voice_transcribe_seconds", Help: "Transcription time per voice search.",
		Buckets: secondsBuckets})
	VoiceAudio = promauto.NewHistogram(prometheus.HistogramOpts{
		Name: "podfind_voice_audio_seconds", Help: "Length of voice recordings.",
		Buckets: []float64{1, 2, 3, 5, 8, 12, 20, 30}})
	VoiceConfidence = promauto.NewHistogram(prometheus.HistogramOpts{
		Name: "podfind_voice_confidence", Help: "Transcript confidence (mean token log-probability).",
		Buckets: prometheus.LinearBuckets(-3, 0.25, 13)})
	VoiceLowConfidence = promauto.NewCounter(prometheus.CounterOpts{
		Name: "podfind_voice_low_confidence_total", Help: "Transcripts below the low-confidence threshold."})
	VoiceLanguage = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "podfind_voice_language_total", Help: "Detected or requested transcript language."},
		[]string{"language"})
	VoiceRetained = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "podfind_voice_retained_total", Help: "Voice searches by whether the recording was kept."},
		[]string{"retained"})

	Events = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "podfind_events_total", Help: "Interaction events recorded."},
		[]string{"type"})

	IndexAge = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "podfind_index_age_seconds", Help: "Seconds since the newest vector of each served model."},
		[]string{"model"})
	IndexRows = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "podfind_index_rows", Help: "Vectors stored for each served model."},
		[]string{"model"})
	Dependency = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "podfind_dependency_up", Help: "1 when the dependency answered its last health check."},
		[]string{"dependency"})
	BuildInfo = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "podfind_build_info", Help: "Always 1; the labels carry the versions."},
		[]string{"version", "bundle", "feature_version"})
)

// Handler serves the metrics in the Prometheus text format.
func Handler() http.Handler { return promhttp.Handler() }

// ObserveHTTP records one served request.
func ObserveHTTP(route, method string, status int, seconds float64) {
	HTTPRequests.WithLabelValues(route, method, strconv.Itoa(status)).Inc()
	HTTPDuration.WithLabelValues(route).Observe(seconds)
}

// ObserveSearch records a served search's stages, legs and results.
func ObserveSearch(resp *search.Response) {
	input := resp.Input.Mode
	if input == "" {
		input = search.InputText
	}
	Searches.WithLabelValues(input, resp.Versions.Ordering, resp.Versions.BundleID).Inc()
	for stage, ms := range resp.TimingsMS {
		SearchStage.WithLabelValues(stage).Observe(ms / 1000)
	}
	for leg, res := range resp.Legs {
		if res == nil {
			continue
		}
		SearchLegs.WithLabelValues(leg, res.Status).Inc()
		SearchLegCandidates.WithLabelValues(leg).Observe(float64(res.Count))
	}
	SearchResults.Observe(float64(len(resp.Results)))
	if len(resp.Results) == 0 {
		ZeroResults.WithLabelValues(input).Inc()
		return
	}
	TopScore.Observe(resp.Results[0].Score)
	for _, r := range resp.Results {
		for _, s := range r.Sources {
			ResultSources.WithLabelValues(s).Inc()
		}
	}
}

// ObserveVoice records a served voice search, transcription included.
func ObserveVoice(resp *search.VoiceResponse) {
	ObserveSearch(&resp.Response)
	t := resp.Transcript
	VoiceTranscribe.Observe(t.ElapsedMS / 1000)
	VoiceAudio.Observe(t.AudioSeconds)
	if t.Confidence != nil {
		VoiceConfidence.Observe(*t.Confidence)
		if *t.Confidence < LowConfidence {
			VoiceLowConfidence.Inc()
		}
	}
	lang := t.Language
	if lang == "" {
		lang = "unknown"
	}
	VoiceLanguage.WithLabelValues(lang).Inc()
	VoiceRetained.WithLabelValues(strconv.FormatBool(resp.Retained)).Inc()
}
