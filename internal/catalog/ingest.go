// Package catalog normalizes discovered podcasts and ingests their RSS
// feeds: deduplicated upserts, searchable-content change detection, and the
// polling schedule derived from each feed's observed update frequency.
package catalog

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"

	"podfind/internal/rss"
)

// Querier is satisfied by *pgxpool.Pool and pgx.Tx.
type Querier interface {
	Exec(ctx context.Context, sql string, args ...any) (pgconn.CommandTag, error)
	QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
}

// Stats summarizes one feed ingestion.
type Stats struct {
	NewEpisodes       int
	UpdatedEpisodes   int
	UnchangedEpisodes int
	SkippedItems      int
}

// EpisodeChange records an episode whose searchable content was inserted or
// updated during ingestion, with the feed item it came from.
type EpisodeChange struct {
	EpisodeID int64
	Outcome   string // "inserted" or "updated"
	Item      rss.Item
}

// CanonicalFeedURL normalizes a feed URL for feed-level deduplication:
// lowercased scheme and host, default ports and fragments stripped.
func CanonicalFeedURL(raw string) (string, error) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return "", fmt.Errorf("parse feed url: %w", err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return "", fmt.Errorf("unsupported feed url scheme %q", u.Scheme)
	}
	u.Scheme = strings.ToLower(u.Scheme)
	u.Host = strings.ToLower(u.Host)
	if (u.Scheme == "http" && strings.HasSuffix(u.Host, ":80")) ||
		(u.Scheme == "https" && strings.HasSuffix(u.Host, ":443")) {
		u.Host = u.Host[:strings.LastIndex(u.Host, ":")]
	}
	u.Fragment = ""
	return u.String(), nil
}

// contentHash fingerprints an episode's searchable fields; a changed hash
// means embeddings and search text must be regenerated.
func contentHash(item rss.Item) string {
	var duration, published string
	if item.DurationSeconds != nil {
		duration = strconv.Itoa(*item.DurationSeconds)
	}
	if item.PublishedAt != nil {
		published = item.PublishedAt.UTC().Format(time.RFC3339)
	}
	h := sha256.New()
	for _, field := range []string{
		item.Title, item.Description, duration, published,
		strconv.FormatBool(item.Explicit), item.EnclosureURL,
	} {
		h.Write([]byte(field))
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

// UpsertPodcastMetadata writes the channel-level fields from a fetched feed.
func UpsertPodcastMetadata(ctx context.Context, db Querier, podcastID int64, feed *rss.Feed) error {
	if feed.Categories == nil {
		feed.Categories = []string{}
	}
	_, err := db.Exec(ctx, `
		UPDATE podcasts
		SET title = $2, description = $3, publisher = $4, link_url = $5,
		    artwork_url = $6, language = $7, categories = $8, explicit = $9,
		    updated_at = now()
		WHERE id = $1`,
		podcastID, feed.Title, feed.Description, feed.Publisher, feed.Link,
		feed.ArtworkURL, feed.Language, feed.Categories, feed.Explicit)
	if err != nil {
		return fmt.Errorf("update podcast %d: %w", podcastID, err)
	}
	return nil
}

// IngestFeed upserts every episode of a parsed feed. Items are deduplicated
// in order of trust — RSS GUID, then enclosure URL, then content hash — and
// episodes whose searchable content is unchanged are left untouched. It
// returns the episodes that were inserted or updated.
func IngestFeed(ctx context.Context, db Querier, podcastID int64, feed *rss.Feed) (Stats, []EpisodeChange, error) {
	var stats Stats
	var changes []EpisodeChange
	seenGUIDs := map[string]bool{}
	seenEnclosures := map[string]bool{}

	for _, item := range feed.Items {
		// Within-feed duplicates (some feeds repeat items).
		if item.GUID != "" && seenGUIDs[item.GUID] {
			stats.SkippedItems++
			continue
		}
		if item.GUID == "" && item.EnclosureURL != "" && seenEnclosures[item.EnclosureURL] {
			stats.SkippedItems++
			continue
		}
		if item.GUID != "" {
			seenGUIDs[item.GUID] = true
		}
		if item.EnclosureURL != "" {
			seenEnclosures[item.EnclosureURL] = true
		}
		if item.Title == "" && item.Description == "" && item.EnclosureURL == "" {
			stats.SkippedItems++
			continue
		}

		// Feeds rarely set a per-item language; inherit the channel's.
		language := feed.Language
		hash := contentHash(item)

		var id int64
		var outcome string
		var err error
		switch {
		case item.GUID != "":
			id, outcome, err = upsertByGUID(ctx, db, podcastID, item, language, hash)
			// A changed GUID on a known enclosure collides with the
			// enclosure key; trust the enclosure identity instead.
			if isUniqueViolation(err, "episodes_enclosure_key") && item.EnclosureURL != "" {
				id, outcome, err = upsertByEnclosure(ctx, db, podcastID, item, language, hash)
			}
		case item.EnclosureURL != "":
			id, outcome, err = upsertByEnclosure(ctx, db, podcastID, item, language, hash)
		default:
			id, outcome, err = upsertByHash(ctx, db, podcastID, item, language, hash)
		}
		if err != nil {
			return stats, changes, fmt.Errorf("ingest %q: %w", item.Title, err)
		}
		switch outcome {
		case "inserted":
			stats.NewEpisodes++
		case "updated":
			stats.UpdatedEpisodes++
		default:
			stats.UnchangedEpisodes++
		}
		if outcome == "inserted" || outcome == "updated" {
			changes = append(changes, EpisodeChange{EpisodeID: id, Outcome: outcome, Item: item})
		}
	}
	return stats, changes, nil
}

func isUniqueViolation(err error, constraint string) bool {
	var pgErr *pgconn.PgError
	return errors.As(err, &pgErr) && pgErr.Code == "23505" && pgErr.ConstraintName == constraint
}

const episodeInsertColumns = `
	INSERT INTO episodes (podcast_id, rss_guid, enclosure_url, content_hash,
	                      title, description, link_url, artwork_url, language,
	                      duration_seconds, published_at, explicit)
	VALUES ($1, NULLIF($2, ''), NULLIF($3, ''), $4, $5, $6, $7, $8, $9, $10, $11, $12)`

const episodeUpdateSet = `
	DO UPDATE SET
		rss_guid = EXCLUDED.rss_guid,
		enclosure_url = EXCLUDED.enclosure_url,
		content_hash = EXCLUDED.content_hash,
		title = EXCLUDED.title,
		description = EXCLUDED.description,
		link_url = EXCLUDED.link_url,
		artwork_url = EXCLUDED.artwork_url,
		language = EXCLUDED.language,
		duration_seconds = EXCLUDED.duration_seconds,
		published_at = EXCLUDED.published_at,
		explicit = EXCLUDED.explicit,
		updated_at = now()
	WHERE episodes.content_hash IS DISTINCT FROM EXCLUDED.content_hash
	RETURNING episodes.id, (xmax = 0) AS inserted`

func scanOutcome(row pgx.Row) (int64, string, error) {
	var id int64
	var inserted bool
	err := row.Scan(&id, &inserted)
	if errors.Is(err, pgx.ErrNoRows) {
		return 0, "unchanged", nil // conflict hit, content hash identical
	}
	if err != nil {
		return 0, "", err
	}
	if inserted {
		return id, "inserted", nil
	}
	return id, "updated", nil
}

func upsertByGUID(ctx context.Context, db Querier, podcastID int64, item rss.Item, language, hash string) (int64, string, error) {
	row := db.QueryRow(ctx, episodeInsertColumns+`
		ON CONFLICT (podcast_id, rss_guid) WHERE rss_guid IS NOT NULL `+episodeUpdateSet,
		podcastID, item.GUID, item.EnclosureURL, hash, item.Title, item.Description,
		item.Link, item.ArtworkURL, language, item.DurationSeconds, item.PublishedAt, item.Explicit)
	return scanOutcome(row)
}

func upsertByEnclosure(ctx context.Context, db Querier, podcastID int64, item rss.Item, language, hash string) (int64, string, error) {
	row := db.QueryRow(ctx, episodeInsertColumns+`
		ON CONFLICT (podcast_id, enclosure_url) WHERE enclosure_url IS NOT NULL `+episodeUpdateSet,
		podcastID, item.GUID, item.EnclosureURL, hash, item.Title, item.Description,
		item.Link, item.ArtworkURL, language, item.DurationSeconds, item.PublishedAt, item.Explicit)
	return scanOutcome(row)
}

// upsertByHash handles items with neither GUID nor enclosure: the content
// hash is the only identity, so identical content is a duplicate and
// changed content is indistinguishable from a new episode.
func upsertByHash(ctx context.Context, db Querier, podcastID int64, item rss.Item, language, hash string) (int64, string, error) {
	var exists bool
	if err := db.QueryRow(ctx, `
		SELECT EXISTS (SELECT 1 FROM episodes WHERE podcast_id = $1 AND content_hash = $2)`,
		podcastID, hash).Scan(&exists); err != nil {
		return 0, "", err
	}
	if exists {
		return 0, "unchanged", nil
	}
	var id int64
	if err := db.QueryRow(ctx, episodeInsertColumns+` RETURNING id`,
		podcastID, item.GUID, item.EnclosureURL, hash, item.Title, item.Description,
		item.Link, item.ArtworkURL, language, item.DurationSeconds, item.PublishedAt, item.Explicit,
	).Scan(&id); err != nil {
		return 0, "", err
	}
	return id, "inserted", nil
}
