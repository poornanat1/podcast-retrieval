// Package sql embeds the SQL statements shared with the Python pipeline.
//
// Each statement lives once in ml/sql/<name>.sql with psycopg-style
// %(name)s parameters. Named returns it rewritten for pgx named arguments
// (@name, with %% unescaped to %), so the Go API runs exactly the text the
// Python training and evaluation code runs.
package sql

import (
	"embed"
	"fmt"
	"regexp"
	"strings"
)

//go:embed *.sql
var files embed.FS

var param = regexp.MustCompile(`%\(([a-z_][a-z0-9_]*)\)s`)

// Named returns the statement in <name>.sql in pgx named-argument form.
func Named(name string) (string, error) {
	raw, err := files.ReadFile(name + ".sql")
	if err != nil {
		return "", fmt.Errorf("shared sql %q: %w", name, err)
	}
	return Rewrite(string(raw)), nil
}

// MustNamed is Named for package-level statements; it panics on a missing file.
func MustNamed(name string) string {
	q, err := Named(name)
	if err != nil {
		panic(err)
	}
	return q
}

// Rewrite converts %(name)s placeholders to @name and %% to %.
func Rewrite(q string) string {
	q = param.ReplaceAllString(q, "@$1")
	return strings.ReplaceAll(q, "%%", "%")
}

// SessionStatements returns the per-connection settings in
// session_settings.sql, one statement per non-comment line.
func SessionStatements() []string {
	raw, err := files.ReadFile("session_settings.sql")
	if err != nil {
		panic(err)
	}
	var out []string
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "--") {
			continue
		}
		out = append(out, strings.TrimSuffix(line, ";"))
	}
	return out
}
