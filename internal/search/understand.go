// Package search is the online search path: query understanding, hybrid
// retrieval over three legs, learned reranking through the model server,
// explanations, and attribution for interaction events.
package search

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// Filters are hard constraints applied inside every retrieval leg. Zero
// values mean "no constraint". PublishedAfter is an ISO date (YYYY-MM-DD).
type Filters struct {
	Language           string `json:"language,omitempty"`
	MaxDurationSeconds int    `json:"max_duration_seconds,omitempty"`
	PublishedAfter     string `json:"published_after,omitempty"`
	NoExplicit         bool   `json:"no_explicit,omitempty"`
}

// Understood is a raw query split into search text and filters.
type Understood struct {
	Raw     string   `json:"raw"`
	Text    string   `json:"text"`
	Filters Filters  `json:"filters"`
	Parsed  []string `json:"parsed,omitempty"` // human-readable list of what was recognized
}

var languages = map[string]string{
	"english": "en", "spanish": "es", "espanol": "es", "español": "es",
	"french": "fr", "francais": "fr", "français": "fr",
	"german": "de", "deutsch": "de",
	"portuguese": "pt", "portugues": "pt", "português": "pt",
}

var languageNames = map[string]string{
	"en": "English", "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese",
}

const langAlt = `english|spanish|espa[nñ]ol|french|fran[cç]ais|german|deutsch|portugu[eê]se?|portugu[eê]s`

var (
	reLanguage = []*regexp.Regexp{
		regexp.MustCompile(`(?i)\b(?:in|en)\s+(` + langAlt + `)\b`),
		regexp.MustCompile(`(?i)\b(` + langAlt + `)(?:[\s-]+language)?(?:\s+(?:episodes?|podcasts?|shows?))\b`),
		regexp.MustCompile(`(?i)\b(` + langAlt + `)[\s-]+language\b`),
	}
	reDurationMinutes = regexp.MustCompile(`(?i)\b(?:under|less than|shorter than|below|at most|max(?:imum)?|up to)\s+(\d{1,3})\s*(?:minutes?|mins?|m)\b`)
	reDurationHours   = regexp.MustCompile(`(?i)\b(?:under|less than|shorter than|below|at most|up to)\s+(an|one|\d)\s+hours?\b`)
	reThisYear        = regexp.MustCompile(`(?i)\b(?:published\s+)?this\s+year\b`)
	reThisMonth       = regexp.MustCompile(`(?i)\b(?:published\s+)?this\s+month\b`)
	reRelative        = regexp.MustCompile(`(?i)\b(?:published\s+)?(?:(?:in|from|over|during)\s+the\s+)?(?:past|last)\s+(week|month|year)\b`)
	reSinceYear       = regexp.MustCompile(`(?i)\b(?:published\s+)?(?:since|after|from)\s+((?:19|20)\d{2})\b`)
	reNoExplicit      = regexp.MustCompile(`(?i)\b(?:no|not|non)[\s-]+explicit(?:\s+(?:content|language|episodes?))?\b|\bfamily[\s-]+friendly\b|\bkid[s]?[\s-]+friendly\b|\bclean\s+(?:(?:episodes?|content|version)\s+)?only\b|\bclean\s+(?:episodes?|content|version)\b`)
	reLeadIn          = regexp.MustCompile(`(?i)^(?:(?:episodes?|podcasts?|shows?)\s+(?:about|on|for)\s+|about\s+)`)
	reDangling        = regexp.MustCompile(`(?i)(?:^|[\s,;]+)(?:published|and|with|that are|which are)(?:[\s,;]+|$)`)
	reSpaces          = regexp.MustCompile(`[\s,;]+`)
)

// Understand extracts structured filters from free text and returns the
// remaining search text. Filters given explicitly (from UI controls) take
// precedence over ones parsed from the text. Patterns are deliberately
// conservative: a phrase is only treated as a filter when it cannot be a
// topic ("today" alone is not a date filter, "clean" alone is not a
// content filter).
func Understand(raw string, explicit Filters, now time.Time) Understood {
	u := Understood{Raw: raw, Filters: explicit}
	text := " " + raw + " "
	remove := func(re *regexp.Regexp) []string {
		m := re.FindStringSubmatch(text)
		if m == nil {
			return nil
		}
		text = strings.Replace(text, m[0], " ", 1)
		return m
	}

	for _, re := range reLanguage {
		if m := remove(re); m != nil {
			code := languages[normalizeLanguageWord(m[1])]
			if code != "" && u.Filters.Language == "" {
				u.Filters.Language = code
				u.Parsed = append(u.Parsed, "language: "+languageNames[code])
			}
			break
		}
	}
	if m := remove(reDurationMinutes); m != nil {
		if n, err := strconv.Atoi(m[1]); err == nil && n > 0 && u.Filters.MaxDurationSeconds == 0 {
			u.Filters.MaxDurationSeconds = n * 60
			u.Parsed = append(u.Parsed, fmt.Sprintf("under %d minutes", n))
		}
	} else if m := remove(reDurationHours); m != nil {
		n := 1
		if v, err := strconv.Atoi(m[1]); err == nil {
			n = v
		}
		if u.Filters.MaxDurationSeconds == 0 {
			u.Filters.MaxDurationSeconds = n * 3600
			u.Parsed = append(u.Parsed, fmt.Sprintf("under %d hour(s)", n))
		}
	}
	setDate := func(t time.Time, label string) {
		if u.Filters.PublishedAfter == "" {
			u.Filters.PublishedAfter = t.Format("2006-01-02")
			u.Parsed = append(u.Parsed, label)
		}
	}
	switch {
	case remove(reThisYear) != nil:
		setDate(time.Date(now.Year(), 1, 1, 0, 0, 0, 0, time.UTC), "published this year")
	case remove(reThisMonth) != nil:
		setDate(time.Date(now.Year(), now.Month(), 1, 0, 0, 0, 0, time.UTC), "published this month")
	default:
		if m := remove(reRelative); m != nil {
			days := map[string]int{"week": 7, "month": 30, "year": 365}[strings.ToLower(m[1])]
			setDate(now.UTC().AddDate(0, 0, -days).Truncate(24*time.Hour), "published in the past "+strings.ToLower(m[1]))
		} else if m := remove(reSinceYear); m != nil {
			y, _ := strconv.Atoi(m[1])
			setDate(time.Date(y, 1, 1, 0, 0, 0, 0, time.UTC), "published since "+m[1])
		}
	}
	if remove(reNoExplicit) != nil && !u.Filters.NoExplicit {
		u.Filters.NoExplicit = true
		u.Parsed = append(u.Parsed, "no explicit content")
	}

	text = reSpaces.ReplaceAllString(text, " ")
	for {
		cleaned := strings.TrimSpace(reDangling.ReplaceAllString(" "+text+" ", " "))
		cleaned = reLeadIn.ReplaceAllString(cleaned, "")
		if cleaned == strings.TrimSpace(text) {
			break
		}
		text = cleaned
	}
	u.Text = strings.Trim(strings.TrimSpace(text), " ,;.-")
	if u.Text == "" && len(u.Parsed) == 0 {
		u.Text = strings.TrimSpace(raw)
	}
	return u
}

func normalizeLanguageWord(w string) string {
	w = strings.ToLower(w)
	switch {
	case strings.HasPrefix(w, "portugu"):
		return "portuguese"
	case strings.HasPrefix(w, "espa"):
		return "spanish"
	case strings.HasPrefix(w, "fran"):
		return "french"
	}
	return w
}
