// Package version exposes the build version stamped into every binary.
package version

// Version is overridden at build time via -ldflags "-X podfind/internal/version.Version=...".
var Version = "0.0.0-dev"

// String returns the human-readable version of this build.
func String() string {
	return Version
}
