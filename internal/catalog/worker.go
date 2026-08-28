package catalog

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"podfind/internal/discovery"
	"podfind/internal/jobs"
	"podfind/internal/objstore"
	"podfind/internal/outbox"
	"podfind/internal/rss"
	"podfind/internal/transcript"
)

// Job types owned by the catalog worker.
const (
	JobFeedRefresh     = "feed.refresh"
	JobDiscover        = "podcast.discover"
	JobSchedule        = "catalog.schedule"
	JobTranscriptFetch = "transcript.fetch"
	JobPodcastDelete   = "podcast.delete"
)

// EventContentChanged is the versioned outbox event emitted whenever an
// episode's searchable content is created or changes (feed text or a newly
// arrived transcript); downstream consumers regenerate search artifacts.
const EventContentChanged = "episode.content_changed"

// scheduleEvery is how often the scheduler job scans for due feeds.
const scheduleEvery = time.Minute

// Worker owns podcast discovery and feed ingestion.
type Worker struct {
	Pool      *pgxpool.Pool
	Queue     *jobs.Queue
	Fetcher   *rss.Fetcher
	Discovery discovery.Provider // nil when no provider is configured
	Objects   *objstore.Store    // nil disables raw transcript retention
	Log       *slog.Logger
}

// Handlers returns the job handlers to register with a jobs.Runner.
func (w *Worker) Handlers() map[string]jobs.Handler {
	return map[string]jobs.Handler{
		JobFeedRefresh:     w.handleFeedRefresh,
		JobDiscover:        w.handleDiscover,
		JobSchedule:        w.handleSchedule,
		JobTranscriptFetch: w.handleTranscriptFetch,
		JobPodcastDelete:   w.handlePodcastDelete,
	}
}

func (w *Worker) logger() *slog.Logger {
	if w.Log != nil {
		return w.Log
	}
	return slog.Default()
}

// FeedRefreshPayload identifies the podcast whose feed to refresh.
type FeedRefreshPayload struct {
	PodcastID int64 `json:"podcast_id"`
}

// DiscoverPayload describes one discovery request. Recurring discoveries
// re-enqueue themselves a day ahead after completing.
type DiscoverPayload struct {
	Query     string `json:"query,omitempty"`
	Trending  bool   `json:"trending,omitempty"`
	Max       int    `json:"max,omitempty"`
	Language  string `json:"language,omitempty"`
	Recurring bool   `json:"recurring,omitempty"`
}

func (w *Worker) handleFeedRefresh(ctx context.Context, job *jobs.Job) error {
	var p FeedRefreshPayload
	if err := json.Unmarshal(job.Payload, &p); err != nil {
		return jobs.Permanent(fmt.Errorf("decode payload: %w", err))
	}

	var feedURL, etag, lastModified string
	err := w.Pool.QueryRow(ctx,
		"SELECT feed_url, etag, last_modified FROM podcasts WHERE id = $1", p.PodcastID,
	).Scan(&feedURL, &etag, &lastModified)
	if errors.Is(err, pgx.ErrNoRows) {
		return jobs.Permanent(fmt.Errorf("podcast %d does not exist", p.PodcastID))
	}
	if err != nil {
		return fmt.Errorf("load podcast %d: %w", p.PodcastID, err)
	}

	res, err := w.Fetcher.Fetch(ctx, feedURL, etag, lastModified)
	if err != nil {
		if rss.IsPermanent(err) {
			return w.parkFeed(ctx, p.PodcastID, err)
		}
		return err
	}

	if res.NotModified {
		return w.recordFetch(ctx, p.PodcastID, etag, lastModified)
	}

	feed, err := rss.Parse(res.Body)
	if err != nil {
		// A feed that does not parse will not parse on retry either.
		return w.parkFeed(ctx, p.PodcastID, err)
	}

	tx, err := w.Pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin: %w", err)
	}
	defer tx.Rollback(ctx)

	if err := UpsertPodcastMetadata(ctx, tx, p.PodcastID, feed); err != nil {
		return err
	}
	stats, changes, err := IngestFeed(ctx, tx, p.PodcastID, feed)
	if err != nil {
		return err
	}
	// Content-change events and transcript jobs commit atomically with the
	// episodes they describe.
	for _, ch := range changes {
		if err := outbox.Append(ctx, tx, EventContentChanged, 1, ContentChangedPayload{
			EpisodeID: ch.EpisodeID, PodcastID: p.PodcastID, Reason: "feed",
		}); err != nil {
			return err
		}
		if err := w.enqueueTranscripts(ctx, tx, ch, feed.Language); err != nil {
			return err
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit: %w", err)
	}

	if err := w.recordFetch(ctx, p.PodcastID, res.ETag, res.LastModified); err != nil {
		return err
	}
	w.logger().Info("feed refreshed", "podcast", p.PodcastID,
		"new", stats.NewEpisodes, "updated", stats.UpdatedEpisodes,
		"unchanged", stats.UnchangedEpisodes, "skipped", stats.SkippedItems)
	return nil
}

// parkFeed pushes a permanently failing feed's next poll to the maximum
// interval before dead-lettering the job, so the scheduler does not
// re-enqueue a fresh doomed refresh every cycle.
func (w *Worker) parkFeed(ctx context.Context, podcastID int64, cause error) error {
	if _, err := w.Pool.Exec(ctx, `
		UPDATE podcasts SET next_fetch_at = now() + $2, updated_at = now()
		WHERE id = $1`,
		podcastID, MaxFetchInterval); err != nil {
		return fmt.Errorf("park feed %d: %w", podcastID, err)
	}
	return jobs.Permanent(cause)
}

// recordFetch stores conditional-request validators and schedules the next
// poll from the feed's newest episode age.
func (w *Worker) recordFetch(ctx context.Context, podcastID int64, etag, lastModified string) error {
	var newest *time.Time
	if err := w.Pool.QueryRow(ctx,
		"SELECT max(published_at) FROM episodes WHERE podcast_id = $1", podcastID,
	).Scan(&newest); err != nil {
		return fmt.Errorf("newest episode for %d: %w", podcastID, err)
	}
	interval := NextFetchInterval(newest, time.Now())
	if _, err := w.Pool.Exec(ctx, `
		UPDATE podcasts
		SET etag = $2, last_modified = $3, last_fetched_at = now(),
		    next_fetch_at = now() + $4, updated_at = now()
		WHERE id = $1`,
		podcastID, etag, lastModified, interval); err != nil {
		return fmt.Errorf("record fetch for %d: %w", podcastID, err)
	}
	return nil
}

func (w *Worker) handleDiscover(ctx context.Context, job *jobs.Job) error {
	if w.Discovery == nil {
		return jobs.Permanent(errors.New("no discovery provider configured"))
	}
	var p DiscoverPayload
	if err := json.Unmarshal(job.Payload, &p); err != nil {
		return jobs.Permanent(fmt.Errorf("decode payload: %w", err))
	}
	if p.Max <= 0 {
		p.Max = 25
	}

	var found []discovery.Podcast
	var err error
	if p.Trending {
		found, err = w.Discovery.Trending(ctx, p.Max, p.Language)
	} else {
		found, err = w.Discovery.Search(ctx, p.Query, p.Max)
	}
	if err != nil {
		return err
	}

	added := 0
	for _, pod := range found {
		id, err := w.registerPodcast(ctx, pod)
		if err != nil {
			w.logger().Warn("skip discovered podcast", "title", pod.Title, "error", err)
			continue
		}
		if id != 0 {
			added++
		}
	}
	w.logger().Info("discovery finished", "found", len(found), "added", added)

	if p.Recurring {
		return w.EnqueueDailyDiscovery(ctx, time.Now().Add(24*time.Hour), p.Max, p.Language)
	}
	return nil
}

// EnqueueDailyDiscovery schedules the recurring trending-discovery job. The
// day-bucketed idempotency key collapses duplicate enqueues, so calling this
// on every worker startup both seeds the chain and heals it after a
// dead-lettered run.
func (w *Worker) EnqueueDailyDiscovery(ctx context.Context, at time.Time, max int, language string) error {
	key := "podcast.discover:daily:" + strconv.FormatInt(at.Unix()/86400, 10)
	_, err := w.Queue.Enqueue(ctx, JobDiscover, DiscoverPayload{
		Trending: true, Max: max, Language: language, Recurring: true,
	}, jobs.WithIdempotencyKey(key), jobs.WithRunAt(at))
	if err != nil {
		return fmt.Errorf("enqueue daily discovery: %w", err)
	}
	return nil
}

// registerPodcast inserts a discovered feed (deduplicated by canonical feed
// URL) and enqueues its first refresh. It returns 0 when already known.
func (w *Worker) registerPodcast(ctx context.Context, pod discovery.Podcast) (int64, error) {
	canonical, err := CanonicalFeedURL(pod.FeedURL)
	if err != nil {
		return 0, err
	}
	if pod.Categories == nil {
		pod.Categories = []string{}
	}

	var id int64
	err = w.Pool.QueryRow(ctx, `
		INSERT INTO podcasts (feed_url, discovery_source, discovery_id, title,
		                      description, publisher, artwork_url, language, categories)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
		ON CONFLICT (feed_url) DO NOTHING
		RETURNING id`,
		canonical, w.Discovery.Source(), pod.ID, pod.Title, pod.Description,
		pod.Publisher, pod.ArtworkURL, normalizeLanguage(pod.Language), pod.Categories,
	).Scan(&id)
	if errors.Is(err, pgx.ErrNoRows) {
		return 0, nil // feed already known
	}
	if err != nil {
		return 0, err
	}

	_, err = w.Queue.Enqueue(ctx, JobFeedRefresh, FeedRefreshPayload{PodcastID: id},
		jobs.WithIdempotencyKey(fmt.Sprintf("feed.refresh:%d:initial", id)))
	if err != nil {
		return 0, err
	}
	return id, nil
}

func normalizeLanguage(lang string) string {
	lang = strings.ToLower(strings.TrimSpace(lang))
	if len(lang) > 16 {
		lang = lang[:16]
	}
	return lang
}

// ContentChangedPayload is the body of EventContentChanged events.
type ContentChangedPayload struct {
	EpisodeID int64  `json:"episode_id"`
	PodcastID int64  `json:"podcast_id"`
	Reason    string `json:"reason"` // "feed" or "transcript"
}

// TranscriptFetchPayload identifies one publisher transcript to ingest.
type TranscriptFetchPayload struct {
	EpisodeID int64  `json:"episode_id"`
	URL       string `json:"url"`
	MimeType  string `json:"mime_type,omitempty"`
	Language  string `json:"language,omitempty"`
}

// PodcastDeletePayload identifies a podcast to remove (feed removal or
// correction requests); episodes and transcripts cascade.
type PodcastDeletePayload struct {
	PodcastID int64 `json:"podcast_id"`
}

// enqueueTranscripts creates one fetch job per supported transcript link on
// a changed episode, inside the ingestion transaction.
func (w *Worker) enqueueTranscripts(ctx context.Context, tx jobs.Querier, ch EpisodeChange, feedLanguage string) error {
	for _, tr := range ch.Item.Transcripts {
		if transcript.DetectFormat(tr.MimeType, tr.URL) == "" {
			continue
		}
		language := tr.Language
		if language == "" {
			language = feedLanguage
		}
		sum := sha256.Sum256([]byte(tr.URL))
		key := fmt.Sprintf("transcript.fetch:%d:%s", ch.EpisodeID, hex.EncodeToString(sum[:8]))
		if _, err := w.Queue.EnqueueIn(ctx, tx, JobTranscriptFetch, TranscriptFetchPayload{
			EpisodeID: ch.EpisodeID, URL: tr.URL, MimeType: tr.MimeType, Language: language,
		}, jobs.WithIdempotencyKey(key)); err != nil {
			return fmt.Errorf("enqueue transcript for episode %d: %w", ch.EpisodeID, err)
		}
	}
	return nil
}

// handleTranscriptFetch downloads, parses, and stores one publisher
// transcript, retaining the raw file in object storage when configured.
func (w *Worker) handleTranscriptFetch(ctx context.Context, job *jobs.Job) error {
	var p TranscriptFetchPayload
	if err := json.Unmarshal(job.Payload, &p); err != nil {
		return jobs.Permanent(fmt.Errorf("decode payload: %w", err))
	}
	format := transcript.DetectFormat(p.MimeType, p.URL)
	if format == "" {
		return jobs.Permanent(fmt.Errorf("unsupported transcript type %q for %s", p.MimeType, p.URL))
	}

	res, err := w.Fetcher.Fetch(ctx, p.URL, "", "")
	if err != nil {
		if rss.IsPermanent(err) {
			return jobs.Permanent(err)
		}
		return err
	}
	text, err := transcript.Parse(format, res.Body)
	if err != nil {
		return jobs.Permanent(err)
	}

	objectKey := ""
	if w.Objects != nil {
		objectKey = fmt.Sprintf("episodes/%d/transcript.%s", p.EpisodeID, format)
		contentType := p.MimeType
		if contentType == "" {
			contentType = "application/octet-stream"
		}
		// Object-store failures are transient: retry the whole job.
		if err := w.Objects.Put(ctx, objectKey, res.Body, contentType); err != nil {
			return err
		}
	}

	tx, err := w.Pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin: %w", err)
	}
	defer tx.Rollback(ctx)

	var podcastID int64
	err = tx.QueryRow(ctx,
		"SELECT podcast_id FROM episodes WHERE id = $1", p.EpisodeID).Scan(&podcastID)
	if errors.Is(err, pgx.ErrNoRows) {
		return jobs.Permanent(fmt.Errorf("episode %d does not exist", p.EpisodeID))
	}
	if err != nil {
		return fmt.Errorf("load episode %d: %w", p.EpisodeID, err)
	}

	if _, err := tx.Exec(ctx, `
		INSERT INTO transcripts (episode_id, source_url, format, content, object_key, language)
		VALUES ($1, $2, $3, $4, $5, $6)
		ON CONFLICT (episode_id) DO UPDATE SET
			source_url = EXCLUDED.source_url,
			format = EXCLUDED.format,
			content = EXCLUDED.content,
			object_key = EXCLUDED.object_key,
			language = EXCLUDED.language,
			updated_at = now()`,
		p.EpisodeID, p.URL, format, text, objectKey, normalizeLanguage(p.Language)); err != nil {
		return fmt.Errorf("store transcript for episode %d: %w", p.EpisodeID, err)
	}
	if err := outbox.Append(ctx, tx, EventContentChanged, 1, ContentChangedPayload{
		EpisodeID: p.EpisodeID, PodcastID: podcastID, Reason: "transcript",
	}); err != nil {
		return err
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit: %w", err)
	}
	w.logger().Info("transcript ingested", "episode", p.EpisodeID, "format", format, "chars", len(text))
	return nil
}

// handlePodcastDelete honors feed removal and correction requests: the
// podcast row is deleted and episodes, transcripts, and pending work cascade
// away. Raw transcript objects are retained only until the next storage
// sweep (object-store GC arrives with the dataset pipeline).
func (w *Worker) handlePodcastDelete(ctx context.Context, job *jobs.Job) error {
	var p PodcastDeletePayload
	if err := json.Unmarshal(job.Payload, &p); err != nil {
		return jobs.Permanent(fmt.Errorf("decode payload: %w", err))
	}
	tag, err := w.Pool.Exec(ctx, "DELETE FROM podcasts WHERE id = $1", p.PodcastID)
	if err != nil {
		return fmt.Errorf("delete podcast %d: %w", p.PodcastID, err)
	}
	w.logger().Info("podcast deleted", "podcast", p.PodcastID, "existed", tag.RowsAffected() > 0)
	return nil
}

// handleSchedule enqueues refreshes for every due feed, then re-enqueues
// itself, making the poll loop a job like any other work.
func (w *Worker) handleSchedule(ctx context.Context, job *jobs.Job) error {
	rows, err := w.Pool.Query(ctx, `
		SELECT id FROM podcasts
		WHERE next_fetch_at IS NULL OR next_fetch_at <= now()
		ORDER BY next_fetch_at NULLS FIRST
		LIMIT 500`)
	if err != nil {
		return fmt.Errorf("list due podcasts: %w", err)
	}
	var due []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			rows.Close()
			return fmt.Errorf("scan due podcast: %w", err)
		}
		due = append(due, id)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return err
	}

	now := time.Now()
	for _, id := range due {
		// Bucketed idempotency key: one refresh per feed per poll window.
		key := fmt.Sprintf("feed.refresh:%d:%d", id, now.Unix()/int64(MinFetchInterval.Seconds()))
		if _, err := w.Queue.Enqueue(ctx, JobFeedRefresh, FeedRefreshPayload{PodcastID: id},
			jobs.WithIdempotencyKey(key)); err != nil {
			return fmt.Errorf("enqueue refresh for %d: %w", id, err)
		}
	}
	if len(due) > 0 {
		w.logger().Info("scheduled feed refreshes", "count", len(due))
	}

	return w.EnqueueSchedulerTick(ctx, now.Add(scheduleEvery))
}

// EnqueueSeed enqueues the standard catalog-seeding discovery set: global
// and per-language trending for breadth plus topical searches for long-tail
// coverage. Idempotency keys make re-running the seed a no-op.
func (w *Worker) EnqueueSeed(ctx context.Context) (int, error) {
	type seed struct {
		key     string
		payload DiscoverPayload
	}
	seeds := []seed{
		{"seed:trending:global", DiscoverPayload{Trending: true, Max: 500}},
	}
	for _, lang := range []string{"es", "fr", "de", "pt"} {
		seeds = append(seeds, seed{
			"seed:trending:" + lang,
			DiscoverPayload{Trending: true, Max: 100, Language: lang},
		})
	}
	for _, topic := range []string{
		"history", "science", "true crime", "technology", "business",
		"health", "comedy", "politics", "sports", "education",
	} {
		seeds = append(seeds, seed{
			"seed:search:" + strings.ReplaceAll(topic, " ", "-"),
			DiscoverPayload{Query: topic, Max: 50},
		})
	}

	enqueued := 0
	for _, s := range seeds {
		id, err := w.Queue.Enqueue(ctx, JobDiscover, s.payload, jobs.WithIdempotencyKey(s.key))
		if err != nil {
			return enqueued, fmt.Errorf("enqueue %s: %w", s.key, err)
		}
		if id != 0 {
			enqueued++
		}
	}
	return enqueued, nil
}

// EnqueueSchedulerTick schedules the next scheduler run; the bucketed key
// makes concurrent enqueues (startup plus self-renewal) collapse into one.
func (w *Worker) EnqueueSchedulerTick(ctx context.Context, at time.Time) error {
	key := "catalog.schedule:" + strconv.FormatInt(at.Unix()/int64(scheduleEvery.Seconds()), 10)
	_, err := w.Queue.Enqueue(ctx, JobSchedule, struct{}{},
		jobs.WithIdempotencyKey(key), jobs.WithRunAt(at))
	if err != nil {
		return fmt.Errorf("enqueue scheduler tick: %w", err)
	}
	return nil
}
