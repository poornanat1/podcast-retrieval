package search

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"podfind/internal/pgtest"
)

const (
	testE5Model = "e5-test"
	testTTModel = "two-tower:test"
	testTrModel = "e5-test#transcript"
)

func unit(i int) []float32 {
	v := make([]float32, 384)
	v[i] = 1
	return v
}

// fakeModelServer returns fixed query vectors and scores each candidate by
// its episode id, so the expected order is known.
func fakeModelServer(t *testing.T, failRerank *bool) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v1/model":
			_ = json.NewEncoder(w).Encode(ModelInfo{BundleID: "bundle-test", FeatureVersion: "1", CandidateDepth: 50,
				IndexModels: map[string]string{"e5": testE5Model, "two_tower": testTTModel, "transcript": testTrModel}})
		case "/v1/encode":
			_ = json.NewEncoder(w).Encode(Vectors{BundleID: "bundle-test", E5: unit(0), TwoTower: unit(1)})
		case "/v1/rerank":
			if failRerank != nil && *failRerank {
				http.Error(w, "boom", http.StatusInternalServerError)
				return
			}
			var req struct {
				Candidates []Evidence `json:"candidates"`
			}
			_ = json.NewDecoder(r.Body).Decode(&req)
			scores := make([]float64, len(req.Candidates))
			for i, c := range req.Candidates {
				scores[i] = float64(c.EpisodeID)
			}
			_ = json.NewEncoder(w).Encode(Rerank{BundleID: "bundle-test", FeatureVersion: "1", Scores: scores})
		default:
			http.NotFound(w, r)
		}
	}))
}

func seed(t *testing.T, pool *pgxpool.Pool) (podcast int64, episodes []int64) {
	t.Helper()
	ctx := context.Background()
	if err := pool.QueryRow(ctx, `INSERT INTO podcasts (feed_url, title, publisher, language)
		VALUES ('https://example.com/feed', 'History Hit', 'History Hit Network', 'en') RETURNING id`).Scan(&podcast); err != nil {
		t.Fatal(err)
	}
	titles := []string{"The fall of Rome", "Roman roads", "Medieval castles"}
	for i, title := range titles {
		var id int64
		if err := pool.QueryRow(ctx, `INSERT INTO episodes (podcast_id, rss_guid, enclosure_url, content_hash, title,
			description, language, duration_seconds, published_at)
			VALUES ($1, $2, $3, 'h', $4, 'history of rome', 'en', 1800, now() - interval '10 days') RETURNING id`,
			podcast, title, "https://example.com/"+title+".mp3", title).Scan(&id); err != nil {
			t.Fatal(err)
		}
		episodes = append(episodes, id)
		for model, vec := range map[string][]float32{testE5Model: unit(0), testTTModel: unit(i % 2)} {
			if _, err := pool.Exec(ctx, `INSERT INTO episode_embeddings (episode_id, model, embedding, content_hash)
				VALUES ($1, $2, $3::vector, 'h')`, id, model, vectorLiteral(vec)); err != nil {
				t.Fatal(err)
			}
		}
	}
	return podcast, episodes
}

func newTestService(t *testing.T, failRerank *bool) (*Service, []int64) {
	t.Helper()
	pool := pgtest.Pool(t, "search")
	_, episodes := seed(t, pool)
	srv := fakeModelServer(t, failRerank)
	t.Cleanup(srv.Close)
	model := NewModelClient(srv.URL, 5*time.Second)
	info, err := model.Model(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	svc := &Service{Pool: pool, Model: model, Retriever: &Retriever{
		Pool: pool, Model: model, Info: info, LexicalTimeout: 5 * time.Second, VectorTimeout: 5 * time.Second}}
	return svc, episodes
}

func TestSearchEndToEndWithAttributedEvents(t *testing.T) {
	svc, episodes := newTestService(t, nil)
	ctx := context.Background()

	resp, err := svc.Search(ctx, Request{Query: "rome under 45 minutes", SessionID: "s1", Limit: 10})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Query.Text != "rome" || resp.Query.Filters.MaxDurationSeconds != 2700 {
		t.Fatalf("understood %+v", resp.Query)
	}
	if resp.Versions.BundleID != "bundle-test" || resp.Versions.Ordering != "ranker" {
		t.Fatalf("versions %+v", resp.Versions)
	}
	// The fake ranker scores by episode id: highest id first.
	want := append([]int64(nil), episodes...)
	sort.Slice(want, func(i, j int) bool { return want[i] > want[j] })
	var got []int64
	for _, r := range resp.Results {
		got = append(got, r.EpisodeID)
		if len(r.Explanation) == 0 || len(r.Sources) == 0 || r.EnclosureURL == "" {
			t.Errorf("result missing explanation/sources/enclosure: %+v", r)
		}
	}
	if len(got) != 3 || got[0] != want[0] || got[2] != want[2] {
		t.Fatalf("order %v, want %v", got, want)
	}
	if resp.Legs["lexical"].Status != "ok" || resp.Legs["lexical"].Count != 3 { // every description mentions rome
		t.Fatalf("lexical leg %+v", resp.Legs["lexical"])
	}

	stored, err := svc.GetSearch(ctx, resp.QueryID)
	if err != nil || len(stored.Results) != 3 || stored.Versions.FeatureVersion != "1" {
		t.Fatalf("stored %+v %v", stored, err)
	}

	n, err := svc.RecordImpressions(ctx, Impressions{SessionID: "s1", QueryID: resp.QueryID,
		Items: []ImpressionItem{{EpisodeID: got[0], DisplayedRank: 1}, {EpisodeID: got[1], DisplayedRank: 2}}})
	if err != nil || n != 2 {
		t.Fatalf("impressions %d %v", n, err)
	}
	var sources []string
	var score float64
	var model, index string
	if err := svc.Pool.QueryRow(ctx, `SELECT candidate_sources, score, model_version, index_version FROM interaction_events
		WHERE event_type = 'impression' AND episode_id = $1`, got[0]).Scan(&sources, &score, &model, &index); err != nil {
		t.Fatal(err)
	}
	if len(sources) == 0 || score != float64(got[0]) || model != "bundle-test" || !strings.Contains(index, testTTModel) {
		t.Fatalf("impression attribution sources=%v score=%v model=%q index=%q", sources, score, model, index)
	}

	if err := svc.RecordPlayback(ctx, "start", Playback{SessionID: "s1", QueryID: resp.QueryID, EpisodeID: got[1]}); err != nil {
		t.Fatal(err)
	}
	if err := svc.RecordPlayback(ctx, "complete", Playback{SessionID: "s1", EpisodeID: got[2], PositionSeconds: 1800}); err != nil {
		t.Fatal(err) // unattributed playback is allowed
	}

	// Rejections: unknown search, episode not in the results, bad action.
	for name, err := range map[string]error{
		"unknown query": func() error {
			_, err := svc.RecordImpressions(ctx, Impressions{SessionID: "s1", QueryID: "q_nope",
				Items: []ImpressionItem{{EpisodeID: got[0], DisplayedRank: 1}}})
			return err
		}(),
		"foreign episode": func() error {
			_, err := svc.RecordImpressions(ctx, Impressions{SessionID: "s1", QueryID: resp.QueryID,
				Items: []ImpressionItem{{EpisodeID: 999999, DisplayedRank: 1}}})
			return err
		}(),
		"no session":  svc.RecordPlayback(ctx, "start", Playback{EpisodeID: got[0]}),
		"bad action":  svc.RecordPlayback(ctx, "rewind", Playback{SessionID: "s1", EpisodeID: got[0]}),
		"bad episode": svc.RecordPlayback(ctx, "start", Playback{SessionID: "s1", EpisodeID: 999999}),
	} {
		if !errors.Is(err, ErrBadRequest) {
			t.Errorf("%s: got %v, want ErrBadRequest", name, err)
		}
	}
	var events int
	_ = svc.Pool.QueryRow(ctx, "SELECT count(*) FROM interaction_events").Scan(&events)
	if events != 5 { // search + 2 impressions + 2 playback
		t.Fatalf("events = %d, want 5", events)
	}
}

func TestSearchFallsBackToRRFWhenRerankFails(t *testing.T) {
	fail := true
	svc, _ := newTestService(t, &fail)
	resp, err := svc.Search(context.Background(), Request{Query: "rome", SessionID: "s2"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Versions.Ordering != "rrf" || len(resp.Results) == 0 {
		t.Fatalf("ordering %q, results %d", resp.Versions.Ordering, len(resp.Results))
	}
	stored, err := svc.GetSearch(context.Background(), resp.QueryID)
	if err != nil || stored.Versions.Ordering != "rrf" {
		t.Fatalf("stored ordering %+v %v", stored.Versions, err)
	}
}

func TestSearchRejectsEmptyQuery(t *testing.T) {
	svc, _ := newTestService(t, nil)
	if _, err := svc.Search(context.Background(), Request{Query: "   "}); !errors.Is(err, ErrBadRequest) {
		t.Fatalf("got %v", err)
	}
}
