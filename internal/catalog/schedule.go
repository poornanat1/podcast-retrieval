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

// TransientBackoff is how far a feed's next poll moves out after its n-th
// consecutive failed fetch attempt: the minimum interval, doubling each
// time, capped at the maximum.
func TransientBackoff(attempts int) time.Duration {
	if attempts < 1 {
		attempts = 1
	}
	d := MinFetchInterval
	for i := 1; i < attempts && d < MaxFetchInterval; i++ {
		d *= 2
	}
	if d > MaxFetchInterval {
		d = MaxFetchInterval
	}
	return d
}
