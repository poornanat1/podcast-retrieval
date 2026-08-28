// Package transcript normalizes publisher-provided transcripts (plain text,
// SRT, VTT, and podcast-namespace JSON) into searchable plain text.
package transcript

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// Supported normalized formats.
const (
	FormatText = "text"
	FormatSRT  = "srt"
	FormatVTT  = "vtt"
	FormatJSON = "json"
)

// DetectFormat maps a transcript's MIME type (falling back to its URL
// extension) to a supported format, or "" when unsupported.
func DetectFormat(mimeType, url string) string {
	mt := strings.ToLower(strings.TrimSpace(mimeType))
	if i := strings.Index(mt, ";"); i >= 0 {
		mt = strings.TrimSpace(mt[:i])
	}
	switch mt {
	case "text/plain":
		return FormatText
	case "application/srt", "application/x-subrip", "text/srt":
		return FormatSRT
	case "text/vtt":
		return FormatVTT
	case "application/json":
		return FormatJSON
	}
	lower := strings.ToLower(url)
	if i := strings.IndexAny(lower, "?#"); i >= 0 {
		lower = lower[:i]
	}
	switch {
	case strings.HasSuffix(lower, ".txt"):
		return FormatText
	case strings.HasSuffix(lower, ".srt"):
		return FormatSRT
	case strings.HasSuffix(lower, ".vtt"):
		return FormatVTT
	case strings.HasSuffix(lower, ".json"):
		return FormatJSON
	}
	return ""
}

// Parse converts transcript data of the given format into normalized plain
// text. It errors on undecodable input or when no text survives.
func Parse(format string, data []byte) (string, error) {
	var text string
	var err error
	switch format {
	case FormatText:
		text = string(data)
	case FormatSRT:
		text = parseCues(string(data), false)
	case FormatVTT:
		text = parseCues(string(data), true)
	case FormatJSON:
		text, err = parseJSON(data)
	default:
		return "", fmt.Errorf("unsupported transcript format %q", format)
	}
	if err != nil {
		return "", err
	}
	text = collapseWhitespace(text)
	if text == "" {
		return "", fmt.Errorf("transcript contains no text")
	}
	return text, nil
}

var (
	timestampRe = regexp.MustCompile(`\d{1,2}:\d{2}(:\d{2})?[.,]\d{3}`)
	tagRe       = regexp.MustCompile(`<[^>]*>`)
	spaceRe     = regexp.MustCompile(`\s+`)
)

// parseCues extracts spoken text from SRT and VTT cue blocks. Cue text is
// whatever follows a timestamp line within its block; indices and cue
// identifiers precede the timestamp and are dropped, as are (for VTT) the
// header, NOTE/STYLE/REGION blocks, and inline tags.
func parseCues(s string, vtt bool) string {
	var out []string
	skipBlock := false
	inCueText := false
	s = strings.TrimPrefix(s, "\uFEFF") // strip UTF-8 BOM
	for i, line := range strings.Split(strings.ReplaceAll(s, "\r\n", "\n"), "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			skipBlock = false
			inCueText = false
			continue
		}
		if skipBlock {
			continue
		}
		if vtt {
			if i == 0 && strings.HasPrefix(trimmed, "WEBVTT") {
				continue
			}
			if strings.HasPrefix(trimmed, "NOTE") || trimmed == "STYLE" || trimmed == "REGION" {
				skipBlock = true
				continue
			}
		}
		if timestampRe.MatchString(trimmed) && strings.Contains(trimmed, "-->") {
			inCueText = true
			continue
		}
		if !inCueText {
			continue // SRT index or VTT cue identifier
		}
		if vtt {
			trimmed = strings.TrimSpace(tagRe.ReplaceAllString(trimmed, ""))
			if trimmed == "" {
				continue
			}
		}
		out = append(out, trimmed)
	}
	return strings.Join(out, " ")
}

// jsonTranscript is the podcast-namespace JSON transcript shape.
type jsonTranscript struct {
	Segments []struct {
		Body string `json:"body"`
	} `json:"segments"`
}

func parseJSON(data []byte) (string, error) {
	var t jsonTranscript
	if err := json.Unmarshal(data, &t); err != nil {
		return "", fmt.Errorf("decode json transcript: %w", err)
	}
	parts := make([]string, 0, len(t.Segments))
	for _, seg := range t.Segments {
		if s := strings.TrimSpace(seg.Body); s != "" {
			parts = append(parts, s)
		}
	}
	return strings.Join(parts, " "), nil
}

func collapseWhitespace(s string) string {
	return strings.TrimSpace(spaceRe.ReplaceAllString(s, " "))
}
