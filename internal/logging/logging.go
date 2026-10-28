// Package logging configures the process-wide slog logger: text for a
// terminal, one JSON object per line when PODFIND_LOG_FORMAT=json or when
// running on Railway (RAILWAY_ENVIRONMENT_NAME is set), whose log search
// indexes JSON fields.
package logging

import (
	"log/slog"
	"os"
)

// Setup installs the default logger and returns it.
func Setup() *slog.Logger {
	var handler slog.Handler
	format := os.Getenv("PODFIND_LOG_FORMAT")
	if format == "" && os.Getenv("RAILWAY_ENVIRONMENT_NAME") != "" {
		format = "json"
	}
	if format == "json" {
		handler = slog.NewJSONHandler(os.Stderr, nil)
	} else {
		handler = slog.NewTextHandler(os.Stderr, nil)
	}
	logger := slog.New(handler)
	slog.SetDefault(logger)
	return logger
}
