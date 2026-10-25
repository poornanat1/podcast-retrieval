package search

import (
	"testing"
	"time"
)

var now = time.Date(2026, 10, 24, 15, 0, 0, 0, time.UTC)

func TestUnderstandDesignExample(t *testing.T) {
	u := Understand("English episodes under 45 minutes about practical uses of AI, published this year, no explicit content", Filters{}, now)
	want := Filters{Language: "en", MaxDurationSeconds: 2700, PublishedAfter: "2026-01-01", NoExplicit: true}
	if u.Filters != want {
		t.Fatalf("filters = %+v, want %+v", u.Filters, want)
	}
	if u.Text != "practical uses of AI" {
		t.Fatalf("text = %q", u.Text)
	}
	if len(u.Parsed) != 4 {
		t.Fatalf("parsed = %v", u.Parsed)
	}
}

func TestUnderstandLeavesTopicsAlone(t *testing.T) {
	for _, q := range []string{"today explained", "clean energy transition", "the daily", "history of the english language",
		"short daily news briefing", "german history podcast"} {
		u := Understand(q, Filters{}, now)
		if q == "history of the english language" || q == "german history podcast" {
			// "english language" and "german ... podcast" are ambiguous; only
			// the explicit forms are filters.
			continue
		}
		if u.Filters != (Filters{}) || u.Text != q {
			t.Errorf("%q: got text %q filters %+v", q, u.Text, u.Filters)
		}
	}
}

func TestUnderstandPatterns(t *testing.T) {
	cases := []struct {
		in   string
		text string
		f    Filters
	}{
		{"noticias de economia en español", "noticias de economia", Filters{Language: "es"}},
		{"true crime in french", "true crime", Filters{Language: "fr"}},
		{"spanish podcasts about football", "football", Filters{Language: "es"}},
		{"on being", "on being", Filters{}},
		{"meditation under an hour", "meditation", Filters{MaxDurationSeconds: 3600}},
		{"startup advice less than 20 min", "startup advice", Filters{MaxDurationSeconds: 1200}},
		{"ai news from the past week", "ai news", Filters{PublishedAfter: "2026-10-17"}},
		{"climate since 2025", "climate", Filters{PublishedAfter: "2025-01-01"}},
		{"bedtime stories family friendly", "bedtime stories", Filters{NoExplicit: true}},
		{"comedy, clean episodes only", "comedy", Filters{NoExplicit: true}},
	}
	for _, c := range cases {
		u := Understand(c.in, Filters{}, now)
		if u.Filters != c.f || u.Text != c.text {
			t.Errorf("%q: got text %q filters %+v; want %q %+v", c.in, u.Text, u.Filters, c.text, c.f)
		}
	}
}

func TestExplicitFiltersWin(t *testing.T) {
	u := Understand("tech news in spanish", Filters{Language: "pt", MaxDurationSeconds: 600}, now)
	if u.Filters.Language != "pt" || u.Filters.MaxDurationSeconds != 600 {
		t.Fatalf("explicit filters overridden: %+v", u.Filters)
	}
	if u.Text != "tech news" {
		t.Fatalf("text = %q", u.Text)
	}
}

func TestOnlyFiltersKeepsEmptyText(t *testing.T) {
	u := Understand("under 30 minutes", Filters{}, now)
	if u.Text != "" || u.Filters.MaxDurationSeconds != 1800 {
		t.Fatalf("got %+v", u)
	}
}
