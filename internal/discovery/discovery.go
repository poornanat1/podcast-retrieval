// Package discovery defines the provider-agnostic contract for finding
// podcasts and their canonical RSS feed URLs. Providers surface feeds; RSS
// remains the authoritative source for episode metadata.
package discovery

import "context"

// Podcast is one feed surfaced by a provider.
type Podcast struct {
	ID          string // provider-native identifier
	Title       string
	FeedURL     string
	Description string
	Publisher   string
	ArtworkURL  string
	Language    string
	Categories  []string
	Popularity  float64 // global popularity percentile in [0, 1]; 0 if unknown
}

// Provider finds podcasts by query or popularity.
type Provider interface {
	// Source names the provider, recorded on discovered podcasts.
	Source() string
	// Search finds podcasts matching a free-text query.
	Search(ctx context.Context, query string, max int) ([]Podcast, error)
	// Trending lists currently popular podcasts, optionally by language.
	Trending(ctx context.Context, max int, language string) ([]Podcast, error)
}
