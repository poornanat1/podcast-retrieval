package search

import (
	"context"
	"strings"
)

// Suggestion is a show or publisher name completing what the user typed.
type Suggestion struct {
	Text string `json:"text"`
	Kind string `json:"kind"` // show or publisher
}

const maxSuggestions = 8

// Suggest completes a prefix (at least two characters) against show and
// publisher names, most popular first. Matches at the start of the name or
// of any word in it.
func (s *Service) Suggest(ctx context.Context, q string) ([]Suggestion, error) {
	q = strings.TrimSpace(q)
	if len(q) < 2 || len(q) > 80 {
		return []Suggestion{}, nil
	}
	pattern := escapeLike(q)
	rows, err := s.Pool.Query(ctx, `
		SELECT title, publisher
		FROM podcasts
		WHERE title ILIKE $1 || '%' OR title ILIKE '% ' || $1 || '%'
		   OR publisher ILIKE $1 || '%' OR publisher ILIKE '% ' || $1 || '%'
		ORDER BY popularity DESC, title
		LIMIT 40`, pattern)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Suggestion{}
	seen := map[string]bool{}
	add := func(text, kind string) {
		key := strings.ToLower(text)
		if text == "" || seen[key] || len(out) >= maxSuggestions {
			return
		}
		seen[key] = true
		out = append(out, Suggestion{Text: text, Kind: kind})
	}
	lower := strings.ToLower(q)
	for rows.Next() {
		var title, publisher string
		if err := rows.Scan(&title, &publisher); err != nil {
			return nil, err
		}
		if matchesPrefix(strings.ToLower(title), lower) {
			add(title, "show")
		}
		if matchesPrefix(strings.ToLower(publisher), lower) {
			add(publisher, "publisher")
		}
	}
	return out, rows.Err()
}

func matchesPrefix(name, prefix string) bool {
	return strings.HasPrefix(name, prefix) || strings.Contains(name, " "+prefix)
}

func escapeLike(s string) string {
	r := strings.NewReplacer(`\`, `\\`, `%`, `\%`, `_`, `\_`)
	return r.Replace(s)
}
