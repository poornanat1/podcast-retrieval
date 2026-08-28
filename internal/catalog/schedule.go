package catalog

import "time"

// Polling bounds: a feed is never polled more often than every 15 minutes
// nor less often than daily.
const (
	MinFetchInterval = 15 * time.Minute
	MaxFetchInterval = 24 * time.Hour
)

// NextFetchInterval derives a refresh interval from the feed's observed
// update frequency, approximated by the age of its newest episode: feeds
// that published recently are polled often, dormant feeds daily.
func NextFetchInterval(newestPublished *time.Time, now time.Time) time.Duration {
	if newestPublished == nil {
		return 6 * time.Hour
	}
	age := now.Sub(*newestPublished)
	var interval time.Duration
	switch {
	case age < 24*time.Hour:
		interval = time.Hour
	case age < 72*time.Hour:
		interval = 3 * time.Hour
	case age < 7*24*time.Hour:
		interval = 6 * time.Hour
	case age < 30*24*time.Hour:
		interval = 12 * time.Hour
	default:
		interval = MaxFetchInterval
	}
	if interval < MinFetchInterval {
		interval = MinFetchInterval
	}
	if interval > MaxFetchInterval {
		interval = MaxFetchInterval
	}
	return interval
}
