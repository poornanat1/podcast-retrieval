package rss

import (
	"os"
	"testing"
	"time"
)

func loadFixture(t *testing.T) *Feed {
	t.Helper()
	data, err := os.ReadFile("testdata/feed.xml")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	feed, err := Parse(data)
	if err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	return feed
}

func TestParseChannel(t *testing.T) {
	feed := loadFixture(t)

	if feed.Title != "Practical AI Weekly" {
		t.Errorf("title = %q", feed.Title)
	}
	if feed.Publisher != "Example Media" {
		t.Errorf("publisher = %q", feed.Publisher)
	}
	if feed.Language != "en-us" {
		t.Errorf("language = %q", feed.Language)
	}
	if feed.ArtworkURL != "https://example.com/art.jpg" {
		t.Errorf("artwork = %q", feed.ArtworkURL)
	}
	if feed.Explicit {
		t.Error("channel marked explicit")
	}
	want := []string{"Technology", "Machine Learning", "Science"}
	if len(feed.Categories) != len(want) {
		t.Fatalf("categories = %v, want %v", feed.Categories, want)
	}
	for i, c := range want {
		if feed.Categories[i] != c {
			t.Errorf("categories[%d] = %q, want %q", i, feed.Categories[i], c)
		}
	}
}

func TestParseItems(t *testing.T) {
	feed := loadFixture(t)
	if len(feed.Items) != 3 {
		t.Fatalf("got %d items, want 3", len(feed.Items))
	}

	ep3 := feed.Items[0]
	if ep3.GUID != "ep-3-guid" {
		t.Errorf("guid = %q", ep3.GUID)
	}
	if ep3.EnclosureURL != "https://cdn.example.com/ep3.mp3" {
		t.Errorf("enclosure = %q", ep3.EnclosureURL)
	}
	if ep3.DurationSeconds == nil || *ep3.DurationSeconds != 3753 {
		t.Errorf("duration = %v, want 3753 (1:02:33)", ep3.DurationSeconds)
	}
	wantTime := time.Date(2026, 8, 17, 9, 30, 0, 0, time.UTC)
	if ep3.PublishedAt == nil || !ep3.PublishedAt.Equal(wantTime) {
		t.Errorf("published = %v, want %v", ep3.PublishedAt, wantTime)
	}
	if ep3.Explicit {
		t.Error("ep3 marked explicit")
	}

	if !feed.Items[1].Explicit {
		t.Error("ep2 not marked explicit")
	}
	if d := feed.Items[1].DurationSeconds; d == nil || *d != 2710 {
		t.Errorf("ep2 duration = %v, want 2710 (45:10)", d)
	}
	if d := feed.Items[2].DurationSeconds; d == nil || *d != 2100 {
		t.Errorf("ep1 duration = %v, want 2100 (plain seconds)", d)
	}
	if feed.Items[2].GUID != "" {
		t.Errorf("ep1 guid = %q, want empty", feed.Items[2].GUID)
	}
}

func TestParseRejectsGarbage(t *testing.T) {
	if _, err := Parse([]byte("this is not xml at all <<<")); err == nil {
		t.Fatal("garbage input parsed without error")
	}
	if _, err := Parse([]byte("<html><body>404</body></html>")); err == nil {
		t.Fatal("non-RSS XML parsed without error")
	}
}

func TestParseDurationFormats(t *testing.T) {
	cases := map[string]struct {
		want int
		ok   bool
	}{
		"3600":    {3600, true},
		"45:10":   {2710, true},
		"1:02:33": {3753, true},
		"":        {0, false},
		"abc":     {0, false},
		"1:2:3:4": {0, false},
		"-5":      {0, false},
	}
	for in, c := range cases {
		got, ok := parseDuration(in)
		if ok != c.ok || (ok && got != c.want) {
			t.Errorf("parseDuration(%q) = (%d, %v), want (%d, %v)", in, got, ok, c.want, c.ok)
		}
	}
}
