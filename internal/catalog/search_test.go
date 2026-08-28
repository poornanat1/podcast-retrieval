package catalog_test

import (
	"context"
	"testing"
	"time"

	"podfind/internal/pgtest"
)

// The lexical search query mirrored in data/samples/search.sql: candidates
// from a UNION of independently indexed matches, filters as hard predicates.
const searchSQL = `
	WITH q AS (
		SELECT websearch_to_tsquery(podfind_ts_config($1), $2) AS tsq
	),
	cand AS (
		SELECT e.id FROM episodes e, q WHERE e.search_tsv @@ q.tsq
		UNION
		SELECT t.episode_id FROM transcripts t, q WHERE t.search_tsv @@ q.tsq
		UNION
		SELECT e.id FROM episodes e, q
		WHERE e.podcast_id IN (SELECT p.id FROM podcasts p WHERE p.search_tsv @@ q.tsq)
	)
	SELECT e.id
	FROM episodes e
	JOIN cand ON cand.id = e.id
	JOIN podcasts p ON p.id = e.podcast_id
	LEFT JOIN transcripts t ON t.episode_id = e.id
	CROSS JOIN q
	WHERE ($1 = '' OR e.language LIKE $1 || '%')
	  AND ($3 = 0 OR e.duration_seconds <= $3)
	  AND (NOT $4 OR NOT e.explicit)
	ORDER BY coalesce(ts_rank(e.search_tsv, q.tsq), 0)
	       + coalesce(ts_rank(t.search_tsv, q.tsq), 0)
	       + 0.5 * coalesce(ts_rank(p.search_tsv, q.tsq), 0) DESC
	LIMIT 20`

func TestFullTextSearchCoversMetadataAndTranscripts(t *testing.T) {
	pool := pgtest.Pool(t, "catalog_search")
	ctx := context.Background()

	var podcastID int64
	if err := pool.QueryRow(ctx, `
		INSERT INTO podcasts (feed_url, title, publisher, language, categories)
		VALUES ('https://example.com/f', 'Practical AI Weekly', 'Example Media', 'en',
		        ARRAY['Technology','Machine Learning'])
		RETURNING id`).Scan(&podcastID); err != nil {
		t.Fatalf("insert podcast: %v", err)
	}

	when := time.Date(2026, 8, 17, 0, 0, 0, 0, time.UTC)
	insertEpisode := func(guid, title, desc string, duration int, explicit bool) int64 {
		t.Helper()
		var id int64
		if err := pool.QueryRow(ctx, `
			INSERT INTO episodes (podcast_id, rss_guid, title, description, language,
			                      duration_seconds, published_at, explicit)
			VALUES ($1, $2, $3, $4, 'en', $5, $6, $7) RETURNING id`,
			podcastID, guid, title, desc, duration, when, explicit).Scan(&id); err != nil {
			t.Fatalf("insert episode: %v", err)
		}
		return id
	}

	withTranscript := insertEpisode("s1", "Weekly news roundup", "Short takes.", 1800, false)
	noTranscript := insertEpisode("s2", "Vector retrieval deep dive", "Embeddings in production.", 3600, false)
	explicitEp := insertEpisode("s3", "Vector databases, uncensored", "Strong opinions.", 2400, true)

	if _, err := pool.Exec(ctx, `
		INSERT INTO transcripts (episode_id, format, content, language)
		VALUES ($1, 'text', 'Today we discuss quantum computing breakthroughs at length.', 'en')`,
		withTranscript); err != nil {
		t.Fatalf("insert transcript: %v", err)
	}

	run := func(query string, maxDuration int, noExplicit bool) map[int64]bool {
		t.Helper()
		rows, err := pool.Query(ctx, searchSQL, "en", query, maxDuration, noExplicit)
		if err != nil {
			t.Fatalf("search %q: %v", query, err)
		}
		defer rows.Close()
		got := map[int64]bool{}
		for rows.Next() {
			var id int64
			if err := rows.Scan(&id); err != nil {
				t.Fatalf("scan: %v", err)
			}
			got[id] = true
		}
		return got
	}

	// An episode with no transcript row is found through its own metadata.
	if got := run("vector retrieval", 0, false); !got[noTranscript] {
		t.Errorf("metadata-only episode not found: %v", got)
	}
	// Transcript text alone can surface an episode whose title never
	// mentions the topic.
	if got := run("quantum computing", 0, false); !got[withTranscript] {
		t.Errorf("transcript match not found: %v", got)
	}
	// Structured filters are hard constraints.
	if got := run("vector", 2500, false); got[noTranscript] {
		t.Errorf("duration filter leaked: %v", got)
	}
	if got := run("vector", 0, true); got[explicitEp] {
		t.Errorf("explicit filter leaked: %v", got)
	}
	if got := run("vector", 0, false); !got[explicitEp] || !got[noTranscript] {
		t.Errorf("unfiltered vector search incomplete: %v", got)
	}
	// Podcast metadata alone surfaces its episodes: none of these episode
	// titles or transcripts mention the show name.
	if got := run("practical ai weekly", 0, false); !got[withTranscript] || !got[noTranscript] || !got[explicitEp] {
		t.Errorf("podcast-metadata search incomplete: %v", got)
	}
}
