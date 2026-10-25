package search

import (
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestNormalizeMatchesPython(t *testing.T) {
	// Same rule as ml.relevance.queries.normalize.
	cases := map[string]string{
		"Today, Explained":         "today explained",
		"  The  Rest Is History! ": "the rest is history",
		"Noticias de Economía":     "noticias de econom a",
		"AI/ML 101":                "ai ml 101",
		"":                         "",
	}
	for in, want := range cases {
		if got := Normalize(in); got != want {
			t.Errorf("Normalize(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestExplainUsesOnlyMatchedFields(t *testing.T) {
	d := 1500
	pub := time.Date(2026, 3, 2, 0, 0, 0, 0, time.UTC)
	c := Candidate{Evidence: Evidence{
		Title: "How People Actually Use AI Agents", PodcastTitle: "The AI Daily Brief", Publisher: "Nathaniel Whittemore",
		Description: "agents everywhere", DurationSeconds: &d, PublishedAt: &pub,
	}, Sources: []string{"e5", "two_tower"}}
	u := Understood{Text: "ai agents", Filters: Filters{MaxDurationSeconds: 2700, PublishedAfter: "2026-01-01", NoExplicit: true}}
	got := Explain(u, c)
	want := []string{"Episode title mentions “ai”, “agents”", "25 min (under 45)", "Published Mar 2, 2026", "Not marked explicit"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q\nwant %q", got, want)
	}
}

func TestExplainShowAndFallback(t *testing.T) {
	c := Candidate{Evidence: Evidence{Title: "Episode 12", PodcastTitle: "Today, Explained", Publisher: "Vox"}}
	if got := Explain(Understood{Text: "today explained"}, c); got[0] != "Show name matches “Today, Explained”" {
		t.Fatalf("got %q", got)
	}
	c = Candidate{Evidence: Evidence{Title: "Episode 12", PodcastTitle: "Hidden Brain"}, Sources: []string{"two_tower"}}
	got := Explain(Understood{Text: "why we procrastinate"}, c)
	if len(got) != 1 || !strings.Contains(got[0], "semantic search") {
		t.Fatalf("got %q", got)
	}
	c.LexTranscript = 0.2
	if got := Explain(Understood{Text: "why we procrastinate"}, c); got[0] != "Transcript mentions your search terms" {
		t.Fatalf("got %q", got)
	}
}
