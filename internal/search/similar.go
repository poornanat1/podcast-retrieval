package search

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"

	mlsql "podfind/ml/sql"
)

var similarSQL = mlsql.MustNamed("similar_episodes")

// SimilarRequest asks for episodes like one episode.
type SimilarRequest struct {
	EpisodeID int64
	Filters   Filters
	Limit     int
	Model     string // index leg: two_tower (default) or e5
}

// SimilarResult is one recommended episode.
type SimilarResult struct {
	Result
	Similarity float64 `json:"similarity"` // cosine similarity to the source episode
}

// SimilarResponse is GET /v1/episodes/{id}/similar.
type SimilarResponse struct {
	EpisodeID       int64              `json:"episode_id"`
	Model           string             `json:"model"`
	Results         []SimilarResult    `json:"results"`
	Diversification Diversification    `json:"diversification"`
	TimingsMS       map[string]float64 `json:"timings_ms"`
}

// Diversification reports what the cross-show pass did.
type Diversification struct {
	Candidates    int `json:"candidates"`      // nearest neighbours considered
	MaxPerShow    int `json:"max_per_show"`    // cap applied to every show
	SameShowShown int `json:"same_show_shown"` // results from the source's own show
	Uncapped      int `json:"uncapped"`        // results added past the cap to fill the list
	Dropped       int `json:"dropped"`         // candidates passed over for the cap
}

const (
	similarDefaultLimit = 10
	similarMaxLimit     = 50
	similarMaxPerShow   = 2
	similarOverfetch    = 20 // a serialized show can own the nearest hundred neighbours
)

// Similar returns the nearest episodes to the source in the chosen index,
// diversified across shows: at most two results per show, and episodes from
// the source's own show only after every other show in the candidate list
// has been considered. If the cap leaves the list short (a neighbourhood
// owned by one or two shows), the nearest remaining candidates fill it.
// Nearest-neighbour order is kept otherwise.
func (s *Service) Similar(ctx context.Context, req SimilarRequest) (SimilarResponse, error) {
	start := time.Now()
	limit := req.Limit
	if limit <= 0 {
		limit = similarDefaultLimit
	}
	if limit > similarMaxLimit {
		return SimilarResponse{}, fmt.Errorf("%w: limit at most %d", ErrBadRequest, similarMaxLimit)
	}
	leg := req.Model
	if leg == "" {
		leg = "two_tower"
	}
	model, ok := s.Retriever.Info.IndexModels[leg]
	if !ok || leg == "transcript" {
		return SimilarResponse{}, fmt.Errorf("%w: model must be two_tower or e5", ErrBadRequest)
	}
	if err := checkFilters(req.Filters); err != nil {
		return SimilarResponse{}, err
	}

	var srcPodcast int64
	err := s.Pool.QueryRow(ctx, "SELECT podcast_id FROM episodes WHERE id = $1", req.EpisodeID).Scan(&srcPodcast)
	if errors.Is(err, pgx.ErrNoRows) {
		return SimilarResponse{}, ErrNotFound
	}
	if err != nil {
		return SimilarResponse{}, err
	}

	var vec string
	err = s.Pool.QueryRow(ctx, "SELECT embedding::text FROM episode_embeddings WHERE episode_id = $1 AND model = $2",
		req.EpisodeID, model).Scan(&vec)
	if errors.Is(err, pgx.ErrNoRows) {
		return SimilarResponse{}, fmt.Errorf("%w: episode has no vector in this index", ErrNotFound)
	}
	if err != nil {
		return SimilarResponse{}, err
	}
	args := filterArgs(req.Filters)
	args["episode_id"] = req.EpisodeID
	args["model"] = model
	args["qvec"] = vec
	args["top_k"] = limit * similarOverfetch
	t0 := time.Now()
	rows, err := s.Pool.Query(ctx, similarSQL, args)
	if err != nil {
		return SimilarResponse{}, fmt.Errorf("similar: %w", err)
	}
	type near struct {
		id, podcast int64
		distance    float64
	}
	var cands []near
	for rows.Next() {
		var n near
		if err := rows.Scan(&n.id, &n.podcast, &n.distance); err != nil {
			rows.Close()
			return SimilarResponse{}, err
		}
		cands = append(cands, n)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return SimilarResponse{}, err
	}
	retrieveMS := msSince(t0)

	// Diversify: other shows first, each capped; then the source's own show;
	// then, only if still short, the nearest leftovers regardless of show.
	perShow := map[int64]int{}
	taken := map[int64]bool{}
	var picked []near
	div := Diversification{Candidates: len(cands), MaxPerShow: similarMaxPerShow}
	take := func(c near) {
		perShow[c.podcast]++
		taken[c.id] = true
		picked = append(picked, c)
		if c.podcast == srcPodcast {
			div.SameShowShown++
		}
	}
	for pass := 0; pass < 3 && len(picked) < limit; pass++ {
		for _, c := range cands {
			if len(picked) >= limit {
				break
			}
			if taken[c.id] {
				continue
			}
			switch pass {
			case 0:
				if c.podcast == srcPodcast || perShow[c.podcast] >= similarMaxPerShow {
					continue
				}
			case 1:
				if c.podcast != srcPodcast || perShow[c.podcast] >= similarMaxPerShow {
					continue
				}
			case 2:
				div.Uncapped++
			}
			take(c)
		}
	}
	div.Dropped = len(cands) - len(picked)

	ids := make([]int64, len(picked))
	for i, c := range picked {
		ids[i] = c.id
	}
	t1 := time.Now()
	details, err := s.episodesByID(ctx, ids)
	if err != nil {
		return SimilarResponse{}, err
	}
	out := SimilarResponse{EpisodeID: req.EpisodeID, Model: leg, Results: []SimilarResult{},
		Diversification: div, TimingsMS: map[string]float64{"retrieve_ms": retrieveMS}}
	for i, c := range picked {
		r, ok := details[c.id]
		if !ok {
			continue
		}
		r.Rank = i + 1
		r.Score = 1 - c.distance
		r.Sources = []string{leg}
		out.Results = append(out.Results, SimilarResult{Result: r, Similarity: 1 - c.distance})
	}
	out.TimingsMS["details_ms"] = msSince(t1)
	out.TimingsMS["total_ms"] = msSince(start)
	return out, nil
}

// episodesByID loads the served fields of a set of episodes.
func (s *Service) episodesByID(ctx context.Context, ids []int64) (map[int64]Result, error) {
	out := map[int64]Result{}
	if len(ids) == 0 {
		return out, nil
	}
	rows, err := s.Pool.Query(ctx, `
		SELECT e.id, e.title, p.title, p.publisher, e.language, e.duration_seconds, e.published_at,
		       e.explicit OR p.explicit, coalesce(e.enclosure_url, ''),
		       coalesce(nullif(e.artwork_url, ''), p.artwork_url)
		FROM episodes e JOIN podcasts p ON p.id = e.podcast_id WHERE e.id = ANY($1)`, ids)
	if err != nil {
		return nil, fmt.Errorf("episodes: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var r Result
		var duration *int32
		var lang, title, enclosure, artwork *string
		if err := rows.Scan(&r.EpisodeID, &title, &r.Show, &r.Publisher, &lang, &duration, &r.PublishedAt,
			&r.Explicit, &enclosure, &artwork); err != nil {
			return nil, err
		}
		r.Title, r.Language, r.EnclosureURL, r.ArtworkURL = deref(title), deref(lang), deref(enclosure), deref(artwork)
		if duration != nil {
			d := int(*duration)
			r.DurationSeconds = &d
		}
		r.Explanation = []string{}
		out[r.EpisodeID] = r
	}
	return out, rows.Err()
}

// checkFilters validates caller-supplied filters as the query parser would
// have produced them.
func checkFilters(f Filters) error {
	if len(f.Language) > 8 {
		return fmt.Errorf("%w: language must be a short code", ErrBadRequest)
	}
	if f.MaxDurationSeconds < 0 {
		return fmt.Errorf("%w: max_duration_seconds must not be negative", ErrBadRequest)
	}
	if f.PublishedAfter != "" {
		if _, err := time.Parse("2006-01-02", f.PublishedAfter); err != nil {
			return fmt.Errorf("%w: published_after must be YYYY-MM-DD", ErrBadRequest)
		}
	}
	return nil
}
