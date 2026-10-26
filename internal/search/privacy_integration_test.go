package search

import (
	"context"
	"testing"
	"time"
)

func TestExportAndDeleteSessionData(t *testing.T) {
	svc, episodes := newTestService(t, nil)
	ctx := context.Background()
	speech, _ := fakeSpeechServer(t, "rome")
	svc.Speech = NewSpeechClient(speech.URL, 5*time.Second)
	store := &memoryStore{}
	svc.Recordings = store

	typed, err := svc.Search(ctx, Request{Query: "rome", SessionID: "mine", Limit: 3})
	if err != nil {
		t.Fatal(err)
	}
	spoken, err := svc.SearchVoice(ctx, VoiceRequest{Audio: testWAV(16000, 1), SessionID: "mine", Retain: true})
	if err != nil || !spoken.Retained {
		t.Fatalf("voice: %v %+v", err, spoken.Retention)
	}
	if _, err := svc.RecordImpressions(ctx, Impressions{SessionID: "mine", QueryID: typed.QueryID,
		Items: []ImpressionItem{{EpisodeID: typed.Results[0].EpisodeID, DisplayedRank: 1}}}); err != nil {
		t.Fatal(err)
	}
	if _, err := svc.Search(ctx, Request{Query: "castles", SessionID: "theirs"}); err != nil {
		t.Fatal(err)
	}

	data, err := svc.ExportSession(ctx, "mine")
	if err != nil {
		t.Fatal(err)
	}
	if len(data.Searches) != 2 || data.Searches[1].Input.Transcript == nil || data.Searches[1].Input.Transcript.Text != "rome" {
		t.Fatalf("searches %+v", data.Searches)
	}
	if len(data.Events) != 3 { // two search events and one impression
		t.Fatalf("%d events", len(data.Events))
	}
	if len(data.Recordings) != 1 || data.Recordings[0].QueryID != spoken.QueryID {
		t.Fatalf("recordings %+v", data.Recordings)
	}

	deleted, err := svc.DeleteSession(ctx, "mine")
	if err != nil {
		t.Fatal(err)
	}
	if deleted.Searches != 2 || deleted.Events != 3 || deleted.Recordings != 1 {
		t.Fatalf("deleted %+v", deleted)
	}
	if _, ok := store.objects[RecordingKey(spoken.QueryID)]; ok {
		t.Fatal("recording object still stored")
	}
	if _, err := svc.GetSearch(ctx, typed.QueryID); err != ErrNotFound {
		t.Fatalf("search still stored: %v", err)
	}
	// The other session's data is untouched, and a second delete is empty.
	if theirs, _ := svc.ExportSession(ctx, "theirs"); len(theirs.Searches) != 1 {
		t.Fatalf("other session: %+v", theirs)
	}
	if again, _ := svc.DeleteSession(ctx, "mine"); again.Searches != 0 || again.Events != 0 {
		t.Fatalf("second delete %+v", again)
	}
	if empty, _ := svc.ExportSession(ctx, "mine"); len(empty.Searches) != 0 || empty.Events == nil {
		t.Fatalf("export after delete %+v", empty)
	}
	_ = episodes
}
