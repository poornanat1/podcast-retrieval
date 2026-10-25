package search

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	mlsql "podfind/ml/sql"
)

// Shared with the Python pipeline (ml/sql/*.sql), so online candidates and
// evidence come from the same statements training and evaluation used.
var (
	lexicalSQL  = mlsql.MustNamed("lexical_search")
	vectorSQL   = mlsql.MustNamed("vector_search")
	evidenceSQL = mlsql.MustNamed("candidate_evidence")
	vectorsSQL  = mlsql.MustNamed("candidate_vectors")
)

// Legs, in the order the ranker's features name them.
var Legs = []string{"lexical", "e5", "two_tower"}

// Evidence is one candidate's row as the shared ranker features read it
// (ml.ranking.features); field names must match the Python candidates.
type Evidence struct {
	EpisodeID           int64      `json:"episode_id"`
	PodcastID           int64      `json:"podcast_id"`
	Title               string     `json:"title"`
	Description         string     `json:"description"`
	Language            string     `json:"language"`
	DurationSeconds     *int       `json:"duration_seconds"`
	PublishedAt         *time.Time `json:"published_at"`
	PodcastTitle        string     `json:"podcast_title"`
	Publisher           string     `json:"publisher"`
	Popularity          float64    `json:"popularity"`
	HasTranscript       bool       `json:"has_transcript"`
	LexEpisode          float32    `json:"lex_episode"`
	LexTranscript       float32    `json:"lex_transcript"`
	LexPodcast          float32    `json:"lex_podcast"`
	ShowEpisodes        int64      `json:"show_episodes"`
	E5Sim               float32    `json:"e5_sim"`
	TwoTowerSim         float32    `json:"two_tower_sim"`
	TranscriptSim       float32    `json:"transcript_sim"`
	HasTranscriptVector bool       `json:"has_transcript_vector"`
	LexicalRank         *int       `json:"lexical_rank"`
	E5Rank              *int       `json:"e5_rank"`
	TwoTowerRank        *int       `json:"two_tower_rank"`
}

// Candidate is a candidate's evidence plus the legs that retrieved it.
type Candidate struct {
	Evidence
	Sources []string `json:"sources"`
}

// LegResult reports one retrieval leg's outcome.
type LegResult struct {
	Status string  `json:"status"` // ok, timeout, error
	Count  int     `json:"count"`
	MS     float64 `json:"ms"`
	Error  string  `json:"error,omitempty"`
}

// Retrieval is a query's candidate set with per-leg outcomes and timings.
type Retrieval struct {
	Candidates []Candidate           `json:"-"`
	Legs       map[string]*LegResult `json:"legs"`
	Vectors    Vectors               `json:"-"`
	TimingsMS  map[string]float64    `json:"timings_ms"`
}

// Retriever runs the three legs and assembles candidate evidence.
type Retriever struct {
	Pool           *pgxpool.Pool
	Model          *ModelClient
	Info           ModelInfo
	LexicalTimeout time.Duration
	VectorTimeout  time.Duration

	mu           sync.Mutex
	showEpisodes map[int64]int64
	showsLoaded  time.Time
}

// ConfigurePool applies the shared per-connection vector search settings
// (ml/sql/session_settings.sql) to every new connection.
//
// Statements run in exec mode (an unnamed statement per query) rather than
// pgx's default of caching prepared statements. A cached statement can be
// switched to a generic plan, which cannot use the per-model partial HNSW
// indexes (their predicate names the model, a bind parameter here) or
// simplify the optional-filter predicates.
func ConfigurePool(cfg *pgxpool.Config) {
	cfg.ConnConfig.DefaultQueryExecMode = pgx.QueryExecModeExec
	prev := cfg.AfterConnect
	cfg.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		if prev != nil {
			if err := prev(ctx, conn); err != nil {
				return err
			}
		}
		for _, stmt := range mlsql.SessionStatements() {
			if _, err := conn.Exec(ctx, stmt); err != nil {
				return fmt.Errorf("session setting %q: %w", stmt, err)
			}
		}
		return nil
	}
}

func filterArgs(f Filters) pgx.NamedArgs {
	return pgx.NamedArgs{
		"lang":            f.Language,
		"max_duration":    f.MaxDurationSeconds,
		"published_after": f.PublishedAfter,
		"no_explicit":     f.NoExplicit,
	}
}

// parseLang mirrors ml.retrieval.lexical.query_params: the text-search
// configuration is the query's language, English when unspecified.
func parseLang(f Filters) string {
	if f.Language == "" {
		return "en"
	}
	return f.Language
}

func vectorLiteral(v []float32) string {
	var b strings.Builder
	b.WriteByte('[')
	for i, x := range v {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.FormatFloat(float64(x), 'g', -1, 32))
	}
	b.WriteByte(']')
	return b.String()
}

func parseVector(s string) ([]float32, error) {
	s = strings.TrimSuffix(strings.TrimPrefix(strings.TrimSpace(s), "["), "]")
	parts := strings.Split(s, ",")
	out := make([]float32, len(parts))
	for i, p := range parts {
		f, err := strconv.ParseFloat(strings.TrimSpace(p), 32)
		if err != nil {
			return nil, err
		}
		out[i] = float32(f)
	}
	return out, nil
}

func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func (r *Retriever) lexicalLeg(ctx context.Context, u Understood, depth int) ([]int64, error) {
	args := filterArgs(u.Filters)
	args["query"] = u.Text
	args["parse_lang"] = parseLang(u.Filters)
	args["top_k"] = depth
	rows, err := r.Pool.Query(ctx, lexicalSQL, args)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var ids []int64
	for rows.Next() {
		vals, err := rows.Values()
		if err != nil {
			return nil, err
		}
		ids = append(ids, vals[0].(int64))
	}
	return ids, rows.Err()
}

func (r *Retriever) vectorLeg(ctx context.Context, model string, vec []float32, f Filters, depth int) ([]int64, error) {
	args := filterArgs(f)
	args["model"] = model
	args["qvec"] = vectorLiteral(vec)
	args["top_k"] = depth
	rows, err := r.Pool.Query(ctx, vectorSQL, args)
	if err != nil {
		return nil, err
	}
	return pgx.CollectRows(rows, pgx.RowTo[int64])
}

func legOutcome(start time.Time, ids []int64, err error) *LegResult {
	res := &LegResult{Status: "ok", Count: len(ids), MS: msSince(start)}
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		res.Status = "timeout"
	case err != nil:
		res.Status, res.Error = "error", err.Error()
	}
	return res
}

func msSince(t time.Time) float64 { return float64(time.Since(t).Microseconds()) / 1000 }

// Retrieve runs the lexical leg immediately and, once the query is encoded,
// both vector legs, all concurrently; then reads every candidate's evidence.
// A lexical timeout or a failed leg degrades the candidate set rather than
// failing the search; a failed encode or evidence read fails it.
func (r *Retriever) Retrieve(ctx context.Context, u Understood) (Retrieval, error) {
	start := time.Now()
	depth := r.Info.CandidateDepth
	out := Retrieval{Legs: map[string]*LegResult{}, TimingsMS: map[string]float64{}}
	ranks := map[string][]int64{}
	var mu sync.Mutex
	var wg sync.WaitGroup
	run := func(leg string, timeout time.Duration, fn func(context.Context) ([]int64, error)) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			legCtx, cancel := context.WithTimeout(ctx, timeout)
			defer cancel()
			t0 := time.Now()
			ids, err := fn(legCtx)
			mu.Lock()
			defer mu.Unlock()
			out.Legs[leg] = legOutcome(t0, ids, err)
			if err == nil {
				ranks[leg] = ids
			}
		}()
	}

	if strings.TrimSpace(u.Text) != "" {
		run("lexical", r.LexicalTimeout, func(c context.Context) ([]int64, error) { return r.lexicalLeg(c, u, depth) })
	} else {
		out.Legs["lexical"] = &LegResult{Status: "skipped"}
	}

	t0 := time.Now()
	vecs, err := r.Model.Encode(ctx, modelQuery(u))
	out.TimingsMS["encode_ms"] = msSince(t0)
	if err != nil {
		wg.Wait()
		return out, err
	}
	out.Vectors = vecs
	run("e5", r.VectorTimeout, func(c context.Context) ([]int64, error) {
		return r.vectorLeg(c, r.Info.IndexModels["e5"], vecs.E5, u.Filters, depth)
	})
	run("two_tower", r.VectorTimeout, func(c context.Context) ([]int64, error) {
		return r.vectorLeg(c, r.Info.IndexModels["two_tower"], vecs.TwoTower, u.Filters, depth)
	})
	wg.Wait()
	out.TimingsMS["retrieve_ms"] = msSince(start)

	rankOf := map[string]map[int64]int{}
	idset := map[int64]bool{}
	for _, leg := range Legs {
		rankOf[leg] = map[int64]int{}
		for i, id := range ranks[leg] {
			rankOf[leg][id] = i + 1
			idset[id] = true
		}
	}
	ids := make([]int64, 0, len(idset))
	for id := range idset {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	if len(ids) == 0 {
		return out, nil
	}

	t1 := time.Now()
	cands, err := r.evidence(ctx, u, ids, vecs)
	out.TimingsMS["evidence_ms"] = msSince(t1)
	if err != nil {
		return out, err
	}
	for i := range cands {
		c := &cands[i]
		for _, leg := range Legs {
			if rank, ok := rankOf[leg][c.EpisodeID]; ok {
				rank := rank
				switch leg {
				case "lexical":
					c.LexicalRank = &rank
				case "e5":
					c.E5Rank = &rank
				case "two_tower":
					c.TwoTowerRank = &rank
				}
				c.Sources = append(c.Sources, leg)
			}
		}
	}
	out.Candidates = cands
	return out, nil
}

func (r *Retriever) evidence(ctx context.Context, u Understood, ids []int64, vecs Vectors) ([]Candidate, error) {
	rows, err := r.Pool.Query(ctx, evidenceSQL, pgx.NamedArgs{
		"query": u.Text, "parse_lang": parseLang(u.Filters), "ids": ids})
	if err != nil {
		return nil, fmt.Errorf("evidence: %w", err)
	}
	byID := map[int64]*Candidate{}
	for rows.Next() {
		var c Candidate
		var title, desc, lang, show, publisher *string
		var duration *int32
		if err := rows.Scan(&c.EpisodeID, &c.PodcastID, &title, &desc, &lang, &duration,
			&c.PublishedAt, &show, &publisher, &c.Popularity, &c.HasTranscript,
			&c.LexEpisode, &c.LexTranscript, &c.LexPodcast); err != nil {
			rows.Close()
			return nil, fmt.Errorf("evidence scan: %w", err)
		}
		c.Title, c.Description, c.Language = deref(title), deref(desc), deref(lang)
		c.PodcastTitle, c.Publisher = deref(show), deref(publisher)
		if duration != nil {
			d := int(*duration)
			c.DurationSeconds = &d
		}
		byID[c.EpisodeID] = &c
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("evidence: %w", err)
	}

	models := []string{r.Info.IndexModels["e5"], r.Info.IndexModels["two_tower"], r.Info.IndexModels["transcript"]}
	vrows, err := r.Pool.Query(ctx, vectorsSQL, pgx.NamedArgs{"ids": ids, "models": models})
	if err != nil {
		return nil, fmt.Errorf("candidate vectors: %w", err)
	}
	for vrows.Next() {
		var id int64
		var model, text string
		if err := vrows.Scan(&id, &model, &text); err != nil {
			vrows.Close()
			return nil, fmt.Errorf("candidate vectors scan: %w", err)
		}
		v, err := parseVector(text)
		c := byID[id]
		if err != nil || c == nil {
			continue
		}
		switch model {
		case models[0]:
			c.E5Sim = dot(v, vecs.E5)
		case models[1]:
			c.TwoTowerSim = dot(v, vecs.TwoTower)
		case models[2]:
			c.TranscriptSim = dot(v, vecs.E5)
			c.HasTranscriptVector = true
		}
	}
	vrows.Close()
	if err := vrows.Err(); err != nil {
		return nil, fmt.Errorf("candidate vectors: %w", err)
	}

	shows, err := r.showCounts(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]Candidate, 0, len(ids))
	for _, id := range ids {
		if c := byID[id]; c != nil {
			c.ShowEpisodes = shows[c.PodcastID]
			out = append(out, *c)
		}
	}
	return out, nil
}

func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

// showCounts caches episode counts per show (a ranker feature), refreshed
// every ten minutes.
func (r *Retriever) showCounts(ctx context.Context) (map[int64]int64, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.showEpisodes != nil && time.Since(r.showsLoaded) < 10*time.Minute {
		return r.showEpisodes, nil
	}
	rows, err := r.Pool.Query(ctx, "SELECT podcast_id, count(*) FROM episodes GROUP BY podcast_id")
	if err != nil {
		return nil, fmt.Errorf("show counts: %w", err)
	}
	counts := map[int64]int64{}
	for rows.Next() {
		var pid, n int64
		if err := rows.Scan(&pid, &n); err != nil {
			rows.Close()
			return nil, err
		}
		counts[pid] = n
	}
	rows.Close()
	r.showEpisodes, r.showsLoaded = counts, time.Now()
	return counts, rows.Err()
}
