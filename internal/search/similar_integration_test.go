package search

import (
	"context"
	"errors"
	"testing"
)

func TestSimilarDiversifiesAcrossShows(t *testing.T) {
	svc, episodes := newTestService(t, nil)
	ctx := context.Background()
	// A second show whose episode shares the first episode's two-tower vector.
	var other, otherEp int64
	if err := svc.Pool.QueryRow(ctx, `INSERT INTO podcasts (feed_url, title, publisher, language)
		VALUES ('https://example.com/other', 'Ancient Worlds', 'Other', 'en') RETURNING id`).Scan(&other); err != nil {
		t.Fatal(err)
	}
	if err := svc.Pool.QueryRow(ctx, `INSERT INTO episodes (podcast_id, rss_guid, enclosure_url, content_hash, title,
		description, language, duration_seconds, published_at)
		VALUES ($1, 'g', 'https://example.com/o.mp3', 'h', 'Rome again', 'rome', 'en', 600, now()) RETURNING id`, other).Scan(&otherEp); err != nil {
		t.Fatal(err)
	}
	if _, err := svc.Pool.Exec(ctx, `INSERT INTO episode_embeddings (episode_id, model, embedding, content_hash)
		VALUES ($1, $2, $3::vector, 'h')`, otherEp, testTTModel, vectorLiteral(unit(0))); err != nil {
		t.Fatal(err)
	}

	resp, err := svc.Similar(ctx, SimilarRequest{EpisodeID: episodes[0], Limit: 3})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Model != "two_tower" || len(resp.Results) != 3 {
		t.Fatalf("model %s, %d results", resp.Model, len(resp.Results))
	}
	// The other show's identical vector comes first (other shows before the
	// source's own), then the source show's episodes nearest first.
	if resp.Results[0].EpisodeID != otherEp || resp.Results[0].Similarity < 0.999 {
		t.Errorf("first = %+v", resp.Results[0])
	}
	if resp.Results[1].EpisodeID != episodes[2] || resp.Results[2].EpisodeID != episodes[1] {
		t.Errorf("order = %d, %d", resp.Results[1].EpisodeID, resp.Results[2].EpisodeID)
	}
	if resp.Diversification.SameShowShown != 2 || resp.Diversification.MaxPerShow != 2 {
		t.Errorf("diversification = %+v", resp.Diversification)
	}
	for _, r := range resp.Results {
		if r.EpisodeID == episodes[0] {
			t.Error("source episode returned")
		}
		if r.Title == "" || r.Show == "" {
			t.Errorf("missing details: %+v", r)
		}
	}

	// Filters apply; the source's own show is never excluded by them.
	short, err := svc.Similar(ctx, SimilarRequest{EpisodeID: episodes[0], Limit: 5, Filters: Filters{MaxDurationSeconds: 900}})
	if err != nil {
		t.Fatal(err)
	}
	if len(short.Results) != 1 || short.Results[0].EpisodeID != otherEp {
		t.Errorf("filtered = %+v", short.Results)
	}
	// The pretrained leg and errors.
	if e5, err := svc.Similar(ctx, SimilarRequest{EpisodeID: episodes[0], Model: "e5"}); err != nil || e5.Model != "e5" || len(e5.Results) != 2 {
		t.Errorf("e5: %v %+v", err, e5)
	}
	if _, err := svc.Similar(ctx, SimilarRequest{EpisodeID: 999999}); !errors.Is(err, ErrNotFound) {
		t.Errorf("missing episode: %v", err)
	}
	if _, err := svc.Similar(ctx, SimilarRequest{EpisodeID: episodes[0], Model: "transcript"}); !errors.Is(err, ErrBadRequest) {
		t.Errorf("bad model: %v", err)
	}
	if _, err := svc.Similar(ctx, SimilarRequest{EpisodeID: episodes[0], Filters: Filters{PublishedAfter: "yesterday"}}); !errors.Is(err, ErrBadRequest) {
		t.Errorf("bad date: %v", err)
	}
}

func TestSuggestCompletesShowAndPublisherNames(t *testing.T) {
	svc, _ := newTestService(t, nil)
	ctx := context.Background()
	got, err := svc.Suggest(ctx, "hist")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Text != "History Hit" || got[0].Kind != "show" || got[1].Text != "History Hit Network" {
		t.Errorf("suggest = %+v", got)
	}
	if got, _ := svc.Suggest(ctx, "h"); len(got) != 0 {
		t.Errorf("one character suggested %+v", got)
	}
	if got, _ := svc.Suggest(ctx, "istory"); len(got) != 0 {
		t.Errorf("mid-word matched %+v", got)
	}
}
