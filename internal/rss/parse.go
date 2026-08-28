package rss

import (
	"encoding/xml"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Feed is a normalized podcast feed.
type Feed struct {
	Title       string
	Description string
	Publisher   string
	Link        string
	ArtworkURL  string
	Language    string
	Categories  []string
	Explicit    bool
	Items       []Item
}

// Item is a normalized episode entry.
type Item struct {
	Title           string
	Description     string
	GUID            string
	EnclosureURL    string
	Link            string
	ArtworkURL      string
	DurationSeconds *int
	PublishedAt     *time.Time
	Explicit        bool
	Transcripts     []TranscriptRef
}

// TranscriptRef is a publisher-provided transcript link from the podcast
// namespace (<podcast:transcript>).
type TranscriptRef struct {
	URL      string
	MimeType string
	Language string
}

const itunesNS = "http://www.itunes.com/dtds/podcast-1.0.dtd"

type xmlRSS struct {
	Channel xmlChannel `xml:"channel"`
}

type xmlCategory struct {
	Text     string        `xml:"text,attr"`
	Value    string        `xml:",chardata"`
	Children []xmlCategory `xml:"http://www.itunes.com/dtds/podcast-1.0.dtd category"`
}

type xmlImage struct {
	URL  string `xml:"url"`
	Href string `xml:"href,attr"`
}

type xmlChannel struct {
	Title            string        `xml:"title"`
	Description      string        `xml:"description"`
	Language         string        `xml:"language"`
	Link             string        `xml:"link"`
	Author           string        `xml:"http://www.itunes.com/dtds/podcast-1.0.dtd author"`
	Explicit         string        `xml:"http://www.itunes.com/dtds/podcast-1.0.dtd explicit"`
	ItunesImage      xmlImage      `xml:"http://www.itunes.com/dtds/podcast-1.0.dtd image"`
	Image            xmlImage      `xml:"image"`
	ItunesCategories []xmlCategory `xml:"http://www.itunes.com/dtds/podcast-1.0.dtd category"`
	PlainCategories  []string      `xml:"category"`
	Items            []xmlItem     `xml:"item"`
}

type xmlEnclosure struct {
	URL string `xml:"url,attr"`
}

type xmlTranscript struct {
	URL      string `xml:"url,attr"`
	Type     string `xml:"type,attr"`
	Language string `xml:"language,attr"`
}

type xmlItem struct {
	Title       string       `xml:"title"`
	Description string       `xml:"description"`
	GUID        string       `xml:"guid"`
	Link        string       `xml:"link"`
	PubDate     string       `xml:"pubDate"`
	Enclosure   xmlEnclosure `xml:"enclosure"`
	Duration    string       `xml:"http://www.itunes.com/dtds/podcast-1.0.dtd duration"`
	Explicit    string       `xml:"http://www.itunes.com/dtds/podcast-1.0.dtd explicit"`
	ItunesImage xmlImage     `xml:"http://www.itunes.com/dtds/podcast-1.0.dtd image"`
	// The podcast namespace appears in the wild under two URIs.
	Transcripts    []xmlTranscript `xml:"https://podcastindex.org/namespace/1.0 transcript"`
	TranscriptsAlt []xmlTranscript `xml:"https://github.com/Podcastindex-org/podcast-namespace/blob/main/docs/1.0.md transcript"`
}

// Parse decodes an RSS 2.0 podcast feed into its normalized form.
func Parse(data []byte) (*Feed, error) {
	var doc xmlRSS
	if err := xml.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("parse feed: %w", err)
	}
	ch := doc.Channel
	if ch.Title == "" && len(ch.Items) == 0 {
		return nil, fmt.Errorf("parse feed: no channel content")
	}

	feed := &Feed{
		Title:       strings.TrimSpace(ch.Title),
		Description: strings.TrimSpace(ch.Description),
		Publisher:   strings.TrimSpace(ch.Author),
		Link:        strings.TrimSpace(ch.Link),
		Language:    normalizeLanguage(ch.Language),
		Explicit:    parseExplicit(ch.Explicit),
		Categories:  collectCategories(ch),
	}
	if feed.ArtworkURL = strings.TrimSpace(ch.ItunesImage.Href); feed.ArtworkURL == "" {
		feed.ArtworkURL = strings.TrimSpace(ch.Image.URL)
	}

	for _, it := range ch.Items {
		item := Item{
			Title:        strings.TrimSpace(it.Title),
			Description:  strings.TrimSpace(it.Description),
			GUID:         strings.TrimSpace(it.GUID),
			EnclosureURL: strings.TrimSpace(it.Enclosure.URL),
			Link:         strings.TrimSpace(it.Link),
			ArtworkURL:   strings.TrimSpace(it.ItunesImage.Href),
			Explicit:     parseExplicit(it.Explicit),
		}
		if d, ok := parseDuration(it.Duration); ok {
			item.DurationSeconds = &d
		}
		if t, ok := parseDate(it.PubDate); ok {
			item.PublishedAt = &t
		}
		for _, tr := range append(it.Transcripts, it.TranscriptsAlt...) {
			if u := strings.TrimSpace(tr.URL); u != "" {
				item.Transcripts = append(item.Transcripts, TranscriptRef{
					URL:      u,
					MimeType: strings.TrimSpace(tr.Type),
					Language: normalizeLanguage(tr.Language),
				})
			}
		}
		feed.Items = append(feed.Items, item)
	}
	return feed, nil
}

func collectCategories(ch xmlChannel) []string {
	seen := map[string]bool{}
	var out []string
	add := func(s string) {
		s = strings.TrimSpace(s)
		if s != "" && !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	var walk func(cats []xmlCategory)
	walk = func(cats []xmlCategory) {
		for _, c := range cats {
			if c.Text != "" {
				add(c.Text)
			} else {
				add(c.Value)
			}
			walk(c.Children)
		}
	}
	walk(ch.ItunesCategories)
	for _, c := range ch.PlainCategories {
		add(c)
	}
	return out
}

func normalizeLanguage(lang string) string {
	return strings.ToLower(strings.TrimSpace(lang))
}

func parseExplicit(s string) bool {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "yes", "true", "explicit":
		return true
	}
	return false
}

// parseDuration accepts plain seconds ("3600"), MM:SS, or HH:MM:SS.
func parseDuration(s string) (int, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, false
	}
	parts := strings.Split(s, ":")
	if len(parts) == 1 {
		n, err := strconv.Atoi(parts[0])
		if err != nil || n < 0 {
			return 0, false
		}
		return n, true
	}
	if len(parts) > 3 {
		return 0, false
	}
	total := 0
	for _, p := range parts {
		n, err := strconv.Atoi(strings.TrimSpace(p))
		if err != nil || n < 0 {
			return 0, false
		}
		total = total*60 + n
	}
	return total, true
}

var dateLayouts = []string{
	time.RFC1123Z,
	time.RFC1123,
	time.RFC822Z,
	time.RFC822,
	"Mon, 2 Jan 2006 15:04:05 -0700",
	"Mon, 2 Jan 2006 15:04:05 MST",
	"2006-01-02T15:04:05Z07:00",
	"2006-01-02",
}

func parseDate(s string) (time.Time, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Time{}, false
	}
	for _, layout := range dateLayouts {
		if t, err := time.Parse(layout, s); err == nil {
			return t.UTC(), true
		}
	}
	return time.Time{}, false
}
