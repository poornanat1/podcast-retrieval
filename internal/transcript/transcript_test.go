package transcript

import (
	"strings"
	"testing"
)

func TestDetectFormat(t *testing.T) {
	cases := []struct{ mime, url, want string }{
		{"text/plain", "", FormatText},
		{"text/plain; charset=utf-8", "", FormatText},
		{"application/srt", "", FormatSRT},
		{"application/x-subrip", "", FormatSRT},
		{"text/vtt", "", FormatVTT},
		{"application/json", "", FormatJSON},
		{"", "https://cdn.example.com/ep1.srt", FormatSRT},
		{"", "https://cdn.example.com/ep1.vtt?sig=abc", FormatVTT},
		{"", "https://cdn.example.com/ep1.TXT", FormatText},
		{"application/pdf", "https://cdn.example.com/ep1.pdf", ""},
		{"", "", ""},
	}
	for _, c := range cases {
		if got := DetectFormat(c.mime, c.url); got != c.want {
			t.Errorf("DetectFormat(%q, %q) = %q, want %q", c.mime, c.url, got, c.want)
		}
	}
}

func TestParseSRT(t *testing.T) {
	srt := "1\r\n00:00:01,000 --> 00:00:04,000\r\nWelcome to the show.\r\n\r\n2\r\n00:00:04,500 --> 00:00:08,000\r\nToday we talk about\r\nretrieval systems.\r\n"
	got, err := Parse(FormatSRT, []byte(srt))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	want := "Welcome to the show. Today we talk about retrieval systems."
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestParseVTT(t *testing.T) {
	vtt := `WEBVTT

NOTE
This is a comment block
that spans lines.

00:01.000 --> 00:04.000 align:start
<v Host>Welcome to the show.</v>

cue-2
00:04.500 --> 00:08.000
Today we talk about <b>retrieval</b>.
`
	got, err := Parse(FormatVTT, []byte(vtt))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	want := "Welcome to the show. Today we talk about retrieval."
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestParseJSON(t *testing.T) {
	body := `{"version":"1.0.0","segments":[
		{"speaker":"Host","startTime":0,"endTime":4,"body":"Welcome to the show."},
		{"speaker":"Guest","startTime":4,"endTime":8,"body":"Thanks for having me."}
	]}`
	got, err := Parse(FormatJSON, []byte(body))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	want := "Welcome to the show. Thanks for having me."
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestParsePlainTextCollapsesWhitespace(t *testing.T) {
	got, err := Parse(FormatText, []byte("  Welcome\n\nto   the\tshow.  "))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got != "Welcome to the show." {
		t.Fatalf("got %q", got)
	}
}

func TestParseRejectsEmptyAndUnknown(t *testing.T) {
	if _, err := Parse(FormatText, []byte("   \n ")); err == nil {
		t.Error("empty transcript accepted")
	}
	if _, err := Parse("doc", []byte("x")); err == nil {
		t.Error("unknown format accepted")
	}
	if _, err := Parse(FormatJSON, []byte("not json")); err == nil {
		t.Error("bad json accepted")
	}
	// An SRT that is nothing but numbers and timestamps has no text.
	if _, err := Parse(FormatSRT, []byte("1\n00:00:01,000 --> 00:00:02,000\n")); err == nil {
		t.Error("textless SRT accepted")
	}
	if !strings.Contains(mustErr(t), "no text") {
		t.Error("unexpected error text")
	}
}

func mustErr(t *testing.T) string {
	t.Helper()
	_, err := Parse(FormatSRT, []byte("1\n00:00:01,000 --> 00:00:02,000\n"))
	if err == nil {
		return ""
	}
	return err.Error()
}
