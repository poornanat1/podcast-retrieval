package search

import (
	"fmt"
	"strings"
)

// Normalize matches ml.relevance.queries.normalize: lower-case, every run of
// characters outside [0-9a-z] becomes one space, trimmed. The ranker's
// match features are computed on this form, so explanations use it too.
func Normalize(s string) string {
	var b strings.Builder
	space := false
	for _, r := range strings.ToLower(s) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			if space && b.Len() > 0 {
				b.WriteByte(' ')
			}
			b.WriteRune(r)
			space = false
		} else {
			space = true
		}
	}
	return b.String()
}

// Explain lists why a result was returned, using only fields that matched
// the query or the constraints it satisfied. It never describes episode
// content beyond those fields.
func Explain(query Understood, c Candidate) []string {
	var out []string
	q := Normalize(query.Text)
	show := Normalize(c.PodcastTitle)
	publisher := Normalize(c.Publisher)
	title := Normalize(c.Title)

	switch {
	case q != "" && show != "" && q == show:
		out = append(out, fmt.Sprintf("Show name matches “%s”", c.PodcastTitle))
	case len(show) >= 3 && containsTokens(q, show):
		out = append(out, fmt.Sprintf("Show “%s” is named in your search", c.PodcastTitle))
	case len(q) >= 3 && containsTokens(show, q):
		out = append(out, fmt.Sprintf("Show name contains “%s”", query.Text))
	}
	if q != "" && publisher != "" && (q == publisher || (len(publisher) >= 3 && containsTokens(q, publisher))) {
		out = append(out, fmt.Sprintf("Published by %s", c.Publisher))
	}
	if words := matchedWords(q, title); len(words) > 0 {
		out = append(out, "Episode title mentions "+quoteList(words))
	}
	if words := matchedWords(q, Normalize(c.Description)); len(words) > 0 && len(out) == 0 {
		out = append(out, "Description mentions "+quoteList(words))
	}
	if c.LexTranscript > 0 {
		out = append(out, "Transcript mentions your search terms")
	}

	f := query.Filters
	if f.Language != "" {
		out = append(out, "In "+languageNames[f.Language])
	}
	if f.MaxDurationSeconds > 0 && c.DurationSeconds != nil {
		out = append(out, fmt.Sprintf("%d min (under %d)", (*c.DurationSeconds+59)/60, f.MaxDurationSeconds/60))
	}
	if f.PublishedAfter != "" && c.PublishedAt != nil {
		out = append(out, "Published "+c.PublishedAt.UTC().Format("Jan 2, 2006"))
	}
	if f.NoExplicit {
		out = append(out, "Not marked explicit")
	}
	if len(out) == 0 {
		out = append(out, "Found by "+sourceNames(c.Sources)+" (no exact word match)")
	}
	return out
}

// containsTokens reports whether needle occurs in haystack on whole-token
// boundaries (both normalized).
func containsTokens(haystack, needle string) bool {
	if needle == "" {
		return false
	}
	return strings.Contains(" "+haystack+" ", " "+needle+" ")
}

var stopwords = map[string]bool{
	"a": true, "an": true, "the": true, "of": true, "and": true, "or": true, "in": true,
	"on": true, "for": true, "to": true, "with": true, "about": true, "is": true, "how": true,
	"what": true, "de": true, "la": true, "el": true, "en": true, "y": true, "le": true, "der": true,
	"die": true, "das": true, "und": true, "do": true, "da": true, "e": true, "o": true,
}

func matchedWords(query, text string) []string {
	have := map[string]bool{}
	for _, w := range strings.Fields(text) {
		have[w] = true
	}
	var out []string
	seen := map[string]bool{}
	for _, w := range strings.Fields(query) {
		if stopwords[w] || seen[w] || len(w) < 2 || !have[w] {
			continue
		}
		seen[w] = true
		out = append(out, w)
	}
	return out
}

func quoteList(words []string) string {
	q := make([]string, len(words))
	for i, w := range words {
		q[i] = "“" + w + "”"
	}
	if len(q) > 4 {
		q = append(q[:4], "…")
	}
	return strings.Join(q, ", ")
}

func sourceNames(sources []string) string {
	names := map[string]string{"lexical": "keyword search", "e5": "semantic search", "two_tower": "semantic search"}
	seen := map[string]bool{}
	var out []string
	for _, s := range sources {
		if n := names[s]; n != "" && !seen[n] {
			seen[n] = true
			out = append(out, n)
		}
	}
	if len(out) == 0 {
		return "search"
	}
	return strings.Join(out, " and ")
}
