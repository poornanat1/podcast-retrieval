package search

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

// fakeSpeechServer transcribes every recording to a fixed text and records
// the language hints it received.
func fakeSpeechServer(t *testing.T, text string) (*httptest.Server, *[]string) {
	t.Helper()
	var hints []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/transcribe" || r.Header.Get("Content-Type") != "audio/wav" {
			http.NotFound(w, r)
			return
		}
		hints = append(hints, r.URL.Query().Get("language"))
		if text == "" {
			http.Error(w, `{"detail": "audio is shorter than 0.1 s"}`, http.StatusUnprocessableEntity)
			return
		}
		conf := -0.2
		_ = json.NewEncoder(w).Encode(Transcript{Text: text, Language: "en", Confidence: &conf,
			Backend: "fake:test", AudioSeconds: 1.5, ElapsedMS: 3})
	}))
	t.Cleanup(srv.Close)
	return srv, &hints
}

type memoryStore struct {
	mu      sync.Mutex
	objects map[string][]byte
}

func (m *memoryStore) Put(_ context.Context, key string, data []byte, _ string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.objects == nil {
		m.objects = map[string][]byte{}
	}
	m.objects[key] = data
	return nil
}

func (m *memoryStore) Delete(_ context.Context, key string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.objects, key)
	return nil
}

func TestVoiceSearchStoresTranscriptAndRetainsOnlyWithConsent(t *testing.T) {
	svc, _ := newTestService(t, nil)
	ctx := context.Background()
	speech, hints := fakeSpeechServer(t, "Rome under 45 minutes.")
	svc.Speech = NewSpeechClient(speech.URL, 5*time.Second)
	wav := testWAV(16000, 1.5)

	resp, err := svc.SearchVoice(ctx, VoiceRequest{Audio: wav, Language: "en", SessionID: "v1", Limit: 5})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Transcript.Text != "Rome under 45 minutes." || resp.Transcript.Backend != "fake:test" {
		t.Fatalf("transcript %+v", resp.Transcript)
	}
	if resp.Query.Text != "Rome" || resp.Query.Filters.MaxDurationSeconds != 2700 {
		t.Fatalf("understood %+v", resp.Query)
	}
	if len(resp.Results) == 0 || resp.Retained || resp.Retention != "" {
		t.Fatalf("results %d retained %v %q", len(resp.Results), resp.Retained, resp.Retention)
	}
	if resp.TimingsMS["transcribe_ms"] <= 0 || resp.TimingsMS["total_ms"] < resp.TimingsMS["transcribe_ms"] {
		t.Fatalf("timings %v", resp.TimingsMS)
	}
	if (*hints)[0] != "en" {
		t.Fatalf("language hint %q", (*hints)[0])
	}
	stored, err := svc.GetSearch(ctx, resp.QueryID)
	if err != nil {
		t.Fatal(err)
	}
	if stored.Input.Mode != InputVoice || stored.Input.Transcript == nil ||
		stored.Input.Transcript.Text != "Rome under 45 minutes." || *stored.Input.Transcript.Confidence != -0.2 {
		t.Fatalf("stored input %+v", stored.Input)
	}
	var mode string
	if err := svc.Pool.QueryRow(ctx, `SELECT payload->>'input_mode' FROM interaction_events
		WHERE query_id = $1 AND event_type = 'search'`, resp.QueryID).Scan(&mode); err != nil || mode != InputVoice {
		t.Fatalf("search event input_mode %q %v", mode, err)
	}

	// Retention asked for, but no store configured: refused, search still served.
	resp, err = svc.SearchVoice(ctx, VoiceRequest{Audio: wav, SessionID: "v1", Retain: true})
	if err != nil || resp.Retained || resp.Retention == "" {
		t.Fatalf("retain without store: %v %+v", err, resp.Retention)
	}
	var kept int
	_ = svc.Pool.QueryRow(ctx, `SELECT count(*) FROM voice_recordings`).Scan(&kept)
	if kept != 0 {
		t.Fatalf("%d recordings kept without a store", kept)
	}

	// With a store, the recording is kept under the search's key, with consent time.
	store := &memoryStore{}
	svc.Recordings = store
	resp, err = svc.SearchVoice(ctx, VoiceRequest{Audio: wav, SessionID: "v1", Retain: true})
	if err != nil || !resp.Retained {
		t.Fatalf("retain with store: %v %+v", err, resp.Retention)
	}
	if got := store.objects[RecordingKey(resp.QueryID)]; len(got) != len(wav) {
		t.Fatalf("stored object %d bytes, want %d", len(got), len(wav))
	}
	var key string
	var seconds float64
	if err := svc.Pool.QueryRow(ctx, `SELECT object_key, audio_seconds FROM voice_recordings WHERE query_id = $1`,
		resp.QueryID).Scan(&key, &seconds); err != nil || key != RecordingKey(resp.QueryID) || seconds != 1.5 {
		t.Fatalf("voice_recordings row %q %v %v", key, seconds, err)
	}

	// An edited transcript links back to the voice search it came from.
	edited, err := svc.Search(ctx, Request{Query: "roman roads", SessionID: "v1",
		Input: &Input{Mode: InputVoiceEdited, ParentQueryID: resp.QueryID}})
	if err != nil {
		t.Fatal(err)
	}
	if stored, _ := svc.GetSearch(ctx, edited.QueryID); stored.Input.Mode != InputVoiceEdited ||
		stored.Input.ParentQueryID != resp.QueryID {
		t.Fatalf("edited input %+v", stored.Input)
	}
	for name, in := range map[string]*Input{
		"other session": {Mode: InputVoiceEdited, ParentQueryID: resp.QueryID},
		"text parent":   {Mode: InputVoiceEdited, ParentQueryID: edited.QueryID},
		"unknown mode":  {Mode: "telepathy"},
		"client voice":  {Mode: InputVoice, Transcript: &Transcript{Text: "x"}},
	} {
		session := "v1"
		if name == "other session" {
			session = "v2"
		}
		if _, err := svc.Search(ctx, Request{Query: "rome", SessionID: session, Input: in}); !errors.Is(err, ErrBadRequest) {
			t.Errorf("%s: err %v", name, err)
		}
	}
}

func TestVoiceSearchRejectsBadAudioAndReportsSpeechOutage(t *testing.T) {
	svc, _ := newTestService(t, nil)
	ctx := context.Background()
	silent, _ := fakeSpeechServer(t, "")
	svc.Speech = NewSpeechClient(silent.URL, 5*time.Second)

	if _, err := svc.SearchVoice(ctx, VoiceRequest{Audio: []byte("mp3?"), SessionID: "v1"}); !errors.Is(err, ErrBadRequest) {
		t.Fatalf("not wav: %v", err)
	}
	if _, err := svc.SearchVoice(ctx, VoiceRequest{Audio: testWAV(16000, 1), SessionID: ""}); !errors.Is(err, ErrBadRequest) {
		t.Fatalf("no session: %v", err)
	}
	// The speech server's 422 is the client's problem.
	if _, err := svc.SearchVoice(ctx, VoiceRequest{Audio: testWAV(16000, 1), SessionID: "v1"}); !errors.Is(err, ErrBadRequest) {
		t.Fatalf("speech 422: %v", err)
	}
	silent.Close()
	if _, err := svc.SearchVoice(ctx, VoiceRequest{Audio: testWAV(16000, 1), SessionID: "v1"}); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("speech down: %v", err)
	}
	svc.Speech = nil
	if _, err := svc.SearchVoice(ctx, VoiceRequest{Audio: testWAV(16000, 1), SessionID: "v1"}); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("no speech client: %v", err)
	}
	var n int
	_ = svc.Pool.QueryRow(ctx, `SELECT count(*) FROM searches`).Scan(&n)
	if n != 0 {
		t.Fatalf("%d searches stored for failed voice requests", n)
	}
}
