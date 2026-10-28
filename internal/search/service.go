package search

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Request is a text search.
type Request struct {
	Query     string  `json:"query"`
	Filters   Filters `json:"filters"`
	SessionID string  `json:"session_id"`
	Limit     int     `json:"limit"`

	// Debug-only (Service.Debug): a fixed freshness reference time, the
	// query used verbatim, and the full candidate set with features and
	// scores in the response. Parity checks use these.
	Debug             bool       `json:"debug,omitempty"`
	ReferenceTime     *time.Time `json:"reference_time,omitempty"`
	SkipUnderstanding bool       `json:"skip_understanding,omitempty"` // use Query and Filters verbatim

	// How the query was entered (typed by default). Clients may link an
	// edited transcript to its voice search; see Input.
	Input *Input `json:"input,omitempty"`

	// Set by SearchVoice: the clock started before transcription, and the
	// transcription time to record alongside the search's own timings.
	started time.Time
	timings map[string]float64
}

// Result is one ranked episode as served.
type Result struct {
	Rank            int        `json:"rank"`
	EpisodeID       int64      `json:"episode_id"`
	Title           string     `json:"title"`
	Show            string     `json:"show"`
	Publisher       string     `json:"publisher"`
	Language        string     `json:"language"`
	DurationSeconds *int       `json:"duration_seconds"`
	PublishedAt     *time.Time `json:"published_at"`
	Explicit        bool       `json:"explicit"`
	EnclosureURL    string     `json:"enclosure_url"`
	ArtworkURL      string     `json:"artwork_url"`
	Score           float64    `json:"score"`
	Sources         []string   `json:"sources"`
	Explanation     []string   `json:"explanation"`
}

// Versions identifies what produced a search.
type Versions struct {
	BundleID       string `json:"bundle_id"`
	FeatureVersion string `json:"feature_version"`
	IndexVersion   string `json:"index_version"`
	Ordering       string `json:"ordering"` // ranker, or rrf when reranking failed
}

// Response is a served search.
type Response struct {
	QueryID   string                `json:"query_id"`
	Query     Understood            `json:"query"`
	Input     Input                 `json:"input"`
	Results   []Result              `json:"results"`
	Versions  Versions              `json:"versions"`
	Legs      map[string]*LegResult `json:"legs"`
	TimingsMS map[string]float64    `json:"timings_ms"`
	CreatedAt time.Time             `json:"created_at"`
	Debug     *Debug                `json:"debug,omitempty"`
}

// Debug exposes the full pipeline state for parity checks.
type Debug struct {
	TwoTowerVector []float32   `json:"two_tower_vector"`
	E5Vector       []float32   `json:"e5_vector"`
	Candidates     []Candidate `json:"candidates"`
	Features       [][]float64 `json:"features"`
	Scores         []float64   `json:"scores"`
}

// Service runs searches end to end.
type Service struct {
	Pool       *pgxpool.Pool
	Retriever  *Retriever
	Model      *ModelClient
	Speech     *SpeechClient // voice search; nil makes it unavailable
	Recordings Recordings    // retained voice recordings; nil refuses retention
	Debug      bool          // allow Request.Debug and Request.ReferenceTime
	Now        func() time.Time
}

const (
	defaultLimit = 20
	maxLimit     = 100
	// Queries beyond the tokenizer budget are truncated by the encoders;
	// this only bounds request size.
	maxQueryBytes = 1000
)

func newQueryID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return "q_" + hex.EncodeToString(b)
}

// ErrBadRequest marks client errors.
var ErrBadRequest = errors.New("bad request")

// Search runs query understanding, hybrid retrieval, and reranking, stores
// the search, and returns the ranked results.
func (s *Service) Search(ctx context.Context, req Request) (Response, error) {
	start := time.Now()
	if !req.started.IsZero() {
		start = req.started
	}
	now := time.Now
	if s.Now != nil {
		now = s.Now
	}
	if req.started.IsZero() { // a client request; SearchVoice validated its own
		if err := checkSession(req.SessionID); err != nil {
			return Response{}, err
		}
		if err := s.checkInput(ctx, req.Input, req.SessionID); err != nil {
			return Response{}, err
		}
	}
	if len(req.Query) > maxQueryBytes {
		return Response{}, fmt.Errorf("%w: query longer than %d bytes", ErrBadRequest, maxQueryBytes)
	}
	limit := req.Limit
	if limit <= 0 {
		limit = defaultLimit
	}
	if limit > maxLimit {
		limit = maxLimit
	}
	debug := s.Debug && req.Debug
	refTime := (*time.Time)(nil)
	if s.Debug && req.ReferenceTime != nil {
		refTime = req.ReferenceTime
	}

	u := Understand(req.Query, req.Filters, now())
	if s.Debug && req.SkipUnderstanding {
		u = Understood{Raw: req.Query, Text: req.Query, Filters: req.Filters}
	}
	if u.Text == "" && u.Filters == (Filters{}) {
		return Response{}, fmt.Errorf("%w: empty query", ErrBadRequest)
	}
	ret, err := s.Retriever.Retrieve(ctx, u)
	if err != nil {
		return Response{}, err
	}
	input := Input{Mode: InputText}
	if req.Input != nil && req.Input.Mode != "" {
		input = *req.Input
	}
	for k, v := range req.timings {
		ret.TimingsMS[k] = v
	}
	resp := Response{
		QueryID: newQueryID(), Query: u, Input: input, Legs: ret.Legs, TimingsMS: ret.TimingsMS,
		CreatedAt: now().UTC(),
		Versions: Versions{BundleID: s.Retriever.Info.BundleID, FeatureVersion: s.Retriever.Info.FeatureVersion,
			IndexVersion: s.Retriever.Info.IndexVersion(), Ordering: "ranker"},
	}

	cands := ret.Candidates
	scores := make([]float64, len(cands))
	var features [][]float64
	if len(cands) > 0 {
		evidence := make([]Evidence, len(cands))
		for i, c := range cands {
			evidence[i] = c.Evidence
		}
		t0 := time.Now()
		rr, err := s.Model.Rerank(ctx, modelQuery(u), evidence, refTime, debug)
		resp.TimingsMS["rerank_ms"] = msSince(t0)
		if err == nil && len(rr.Scores) == len(cands) && rr.BundleID == s.Retriever.Info.BundleID {
			scores, features = rr.Scores, rr.Features
		} else {
			// Degrade to RRF ordering of the same candidates rather than fail.
			slog.Warn("rerank failed; ordering by RRF", "error", err, "bundle", rr.BundleID)
			resp.Versions.Ordering = "rrf"
			for i, c := range cands {
				scores[i] = rrf(c)
			}
		}
	}
	order := make([]int, len(cands))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		i, j := order[a], order[b]
		if scores[i] != scores[j] {
			return scores[i] > scores[j]
		}
		return cands[i].EpisodeID < cands[j].EpisodeID
	})
	if len(order) > limit {
		order = order[:limit]
	}

	ids := make([]int64, len(order))
	for k, i := range order {
		ids[k] = cands[i].EpisodeID
	}
	display, err := s.display(ctx, ids)
	if err != nil {
		return Response{}, err
	}
	for k, i := range order {
		c := cands[i]
		d := display[c.EpisodeID]
		resp.Results = append(resp.Results, Result{
			Rank: k + 1, EpisodeID: c.EpisodeID, Title: c.Title, Show: c.PodcastTitle,
			Publisher: c.Publisher, Language: c.Language, DurationSeconds: c.DurationSeconds,
			PublishedAt: c.PublishedAt, Explicit: d.explicit, EnclosureURL: d.enclosure,
			ArtworkURL: d.artwork, Score: scores[i], Sources: c.Sources, Explanation: Explain(u, c),
		})
	}
	resp.TimingsMS["total_ms"] = msSince(start)
	if debug {
		resp.Debug = &Debug{TwoTowerVector: ret.Vectors.TwoTower, E5Vector: ret.Vectors.E5,
			Candidates: cands, Features: features, Scores: scores}
	}
	if err := s.save(ctx, req, resp); err != nil {
		return Response{}, err
	}
	return resp, nil
}

// jsonText marshals a value for a JSONB parameter. Statements run in exec
// mode (see ConfigurePool), where pgx cannot infer a JSON encoding for maps
// or byte slices, so JSON goes over the wire as text.
func jsonText(v any) string {
	raw, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(raw)
}

func rrf(c Candidate) float64 {
	var s float64
	for _, r := range []*int{c.LexicalRank, c.E5Rank, c.TwoTowerRank} {
		if r != nil {
			s += 1.0 / float64(60+*r)
		}
	}
	return s
}

type displayRow struct {
	explicit  bool
	enclosure string
	artwork   string
}

func (s *Service) display(ctx context.Context, ids []int64) (map[int64]displayRow, error) {
	out := map[int64]displayRow{}
	if len(ids) == 0 {
		return out, nil
	}
	rows, err := s.Pool.Query(ctx, `
		SELECT e.id, e.explicit OR p.explicit, e.enclosure_url,
		       coalesce(nullif(e.artwork_url, ''), p.artwork_url)
		FROM episodes e JOIN podcasts p ON p.id = e.podcast_id
		WHERE e.id = ANY($1)`, ids)
	if err != nil {
		return nil, fmt.Errorf("display fields: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var id int64
		var d displayRow
		var enclosure, artwork *string
		if err := rows.Scan(&id, &d.explicit, &enclosure, &artwork); err != nil {
			return nil, err
		}
		d.enclosure, d.artwork = deref(enclosure), deref(artwork)
		out[id] = d
	}
	return out, rows.Err()
}

// storedResult is a served result as kept on the searches row: enough to
// attribute impressions and playback without trusting the client.
type storedResult struct {
	EpisodeID int64    `json:"episode_id"`
	Rank      int      `json:"rank"`
	Score     float64  `json:"score"`
	Sources   []string `json:"sources"`
}

func (s *Service) save(ctx context.Context, req Request, resp Response) error {
	stored := make([]storedResult, len(resp.Results))
	for i, r := range resp.Results {
		stored[i] = storedResult{EpisodeID: r.EpisodeID, Rank: r.Rank, Score: r.Score, Sources: r.Sources}
	}
	filters, _ := json.Marshal(resp.Query.Filters)
	legs, _ := json.Marshal(resp.Legs)
	timings, _ := json.Marshal(resp.TimingsMS)
	results, _ := json.Marshal(stored)
	input, _ := json.Marshal(resp.Input)
	tx, err := s.Pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx) //nolint:errcheck
	if _, err := tx.Exec(ctx, `
		INSERT INTO searches (query_id, session_id, raw_query, query_text, filters, bundle_id,
		                      feature_version, index_version, ordering, legs, timings_ms, results, created_at, input)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)`,
		resp.QueryID, req.SessionID, resp.Query.Raw, resp.Query.Text, string(filters), resp.Versions.BundleID,
		resp.Versions.FeatureVersion, resp.Versions.IndexVersion, resp.Versions.Ordering, string(legs),
		string(timings), string(results), resp.CreatedAt, string(input)); err != nil {
		return fmt.Errorf("save search: %w", err)
	}
	if _, err := tx.Exec(ctx, `
		INSERT INTO interaction_events (event_type, occurred_at, session_id, query_id,
		                                feature_version, model_version, index_version, payload)
		VALUES ('search', $1, $2, $3, $4, $5, $6, $7)`,
		resp.CreatedAt, req.SessionID, resp.QueryID, resp.Versions.FeatureVersion, resp.Versions.BundleID,
		resp.Versions.IndexVersion, jsonText(map[string]any{"query": resp.Query.Raw, "results": len(resp.Results),
			"ordering": resp.Versions.Ordering, "input_mode": resp.Input.Mode})); err != nil {
		return fmt.Errorf("search event: %w", err)
	}
	return tx.Commit(ctx)
}

// StoredSearch is a search as GET /v1/search/{queryId} returns it. The
// session id stays server-side: query ids are visible to anyone who sees
// a results page, and the session id is what unlocks export and deletion.
type StoredSearch struct {
	QueryID   string                `json:"query_id"`
	SessionID string                `json:"-"`
	Raw       string                `json:"raw_query"`
	Text      string                `json:"query_text"`
	Filters   Filters               `json:"filters"`
	Input     Input                 `json:"input"`
	Versions  Versions              `json:"versions"`
	Legs      map[string]*LegResult `json:"legs"`
	TimingsMS map[string]float64    `json:"timings_ms"`
	Results   []storedResult        `json:"results"`
	CreatedAt time.Time             `json:"created_at"`
}

// ErrNotFound marks a missing search or episode.
var ErrNotFound = errors.New("not found")

// GetSearch loads a stored search.
func (s *Service) GetSearch(ctx context.Context, queryID string) (StoredSearch, error) {
	var out StoredSearch
	var filters, legs, timings, results, input []byte
	err := s.Pool.QueryRow(ctx, `
		SELECT query_id, session_id, raw_query, query_text, filters, bundle_id, feature_version,
		       index_version, ordering, legs, timings_ms, results, created_at, input
		FROM searches WHERE query_id = $1`, queryID).Scan(
		&out.QueryID, &out.SessionID, &out.Raw, &out.Text, &filters, &out.Versions.BundleID,
		&out.Versions.FeatureVersion, &out.Versions.IndexVersion, &out.Versions.Ordering, &legs,
		&timings, &results, &out.CreatedAt, &input)
	if errors.Is(err, pgx.ErrNoRows) {
		return out, ErrNotFound
	}
	if err != nil {
		return out, err
	}
	_ = json.Unmarshal(filters, &out.Filters)
	_ = json.Unmarshal(legs, &out.Legs)
	_ = json.Unmarshal(timings, &out.TimingsMS)
	_ = json.Unmarshal(results, &out.Results)
	_ = json.Unmarshal(input, &out.Input)
	return out, nil
}

// Episode is GET /v1/episodes/{id}.
type Episode struct {
	EpisodeID       int64      `json:"episode_id"`
	PodcastID       int64      `json:"podcast_id"`
	Title           string     `json:"title"`
	Description     string     `json:"description"`
	Show            string     `json:"show"`
	Publisher       string     `json:"publisher"`
	Language        string     `json:"language"`
	DurationSeconds *int32     `json:"duration_seconds"`
	PublishedAt     *time.Time `json:"published_at"`
	Explicit        bool       `json:"explicit"`
	EnclosureURL    string     `json:"enclosure_url"`
	ArtworkURL      string     `json:"artwork_url"`
	LinkURL         string     `json:"link_url"`
	HasTranscript   bool       `json:"has_transcript"`
	Categories      []string   `json:"categories"`
}

// GetEpisode loads one episode with its show.
func (s *Service) GetEpisode(ctx context.Context, id int64) (Episode, error) {
	var e Episode
	var title, desc, lang, enclosure, artwork, link *string
	err := s.Pool.QueryRow(ctx, `
		SELECT e.id, e.podcast_id, e.title, e.description, p.title, p.publisher, e.language,
		       e.duration_seconds, e.published_at, e.explicit OR p.explicit, e.enclosure_url,
		       coalesce(nullif(e.artwork_url, ''), p.artwork_url), e.link_url,
		       EXISTS (SELECT 1 FROM transcripts t WHERE t.episode_id = e.id), p.categories
		FROM episodes e JOIN podcasts p ON p.id = e.podcast_id WHERE e.id = $1`, id).Scan(
		&e.EpisodeID, &e.PodcastID, &title, &desc, &e.Show, &e.Publisher, &lang, &e.DurationSeconds,
		&e.PublishedAt, &e.Explicit, &enclosure, &artwork, &link, &e.HasTranscript, &e.Categories)
	if errors.Is(err, pgx.ErrNoRows) {
		return e, ErrNotFound
	}
	e.Title, e.Description, e.Language = deref(title), deref(desc), deref(lang)
	e.EnclosureURL, e.ArtworkURL, e.LinkURL = deref(enclosure), deref(artwork), deref(link)
	return e, err
}
