// Command loadtest drives the search API with concurrent clients and
// reports latency percentiles, throughput, availability, and interaction
// event persistence against the SLOs (docs/16-monitoring.md).
//
//	go run ./cmd/loadtest -api https://podfind.example -concurrency 8 -duration 2m \
//	    -queries data/relevance/queries.jsonl -out experiments/load/run.json
//
// Each client loops over the queries: a text search, then (unless -events
// is false) an impression event for the results and a playback start for
// the first one, as the web app sends them. With -database-url the stored
// event count is compared with the accepted count, which is the event
// persistence SLO; without it, acceptance (HTTP 202) stands in.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

type sample struct {
	kind   string
	status int
	ms     float64
}

type result struct {
	Started       time.Time          `json:"started"`
	Duration      string             `json:"duration"`
	Concurrency   int                `json:"concurrency"`
	API           string             `json:"api"`
	Searches      int                `json:"searches"`
	SearchesPerS  float64            `json:"searches_per_second"`
	SearchStatus  map[string]int     `json:"search_status"`
	Availability  float64            `json:"availability"`
	SearchMS      map[string]float64 `json:"search_ms"`
	EventsSent    int64              `json:"events_sent"`
	EventsOK      int64              `json:"events_accepted"`
	EventsStored  int64              `json:"events_stored"`
	EventStoreSLO float64            `json:"event_persistence"`
	EventMS       map[string]float64 `json:"event_ms"`
	Errors        []string           `json:"errors,omitempty"`
}

func main() {
	api := flag.String("api", "http://localhost:8080", "API base URL")
	queriesPath := flag.String("queries", "data/relevance/queries.jsonl", "JSONL with query_text per line")
	concurrency := flag.Int("concurrency", 8, "concurrent clients")
	duration := flag.Duration("duration", time.Minute, "how long to run")
	limit := flag.Int("limit", 20, "results per search")
	events := flag.Bool("events", true, "send impression and playback events after each search")
	session := flag.String("session", "load-"+time.Now().UTC().Format("20060102-150405"), "session id prefix")
	dbURL := flag.String("database-url", os.Getenv("DATABASE_URL"), "count stored events (optional)")
	out := flag.String("out", "", "write the JSON report here")
	flag.Parse()

	queries, err := loadQueries(*queriesPath)
	if err != nil {
		log.Fatalf("queries: %v", err)
	}
	client := &http.Client{Timeout: 60 * time.Second}
	ctx, cancel := context.WithTimeout(context.Background(), *duration)
	defer cancel()

	var (
		mu       sync.Mutex
		samples  []sample
		errs     []string
		sent, ok atomic.Int64
		next     atomic.Int64
	)
	record := func(s sample) { mu.Lock(); samples = append(samples, s); mu.Unlock() }
	fail := func(msg string) {
		mu.Lock()
		if len(errs) < 20 {
			errs = append(errs, msg)
		}
		mu.Unlock()
	}

	started := time.Now()
	var wg sync.WaitGroup
	for c := 0; c < *concurrency; c++ {
		wg.Add(1)
		go func(c int) {
			defer wg.Done()
			sid := fmt.Sprintf("%s-%d", *session, c)
			for ctx.Err() == nil {
				q := queries[int(next.Add(1)-1)%len(queries)]
				body, _ := json.Marshal(map[string]any{"query": q, "session_id": sid, "limit": *limit})
				status, resp, ms, err := post(ctx, client, *api+"/v1/search/text", body)
				if err != nil {
					if ctx.Err() != nil {
						return
					}
					record(sample{"search", 0, ms})
					fail("search: " + err.Error())
					continue
				}
				record(sample{"search", status, ms})
				if status != 200 || !*events {
					if status != 200 {
						fail(fmt.Sprintf("search %d: %s", status, truncate(resp)))
					}
					continue
				}
				var sr struct {
					QueryID string `json:"query_id"`
					Results []struct {
						EpisodeID int64 `json:"episode_id"`
						Rank      int   `json:"rank"`
					} `json:"results"`
				}
				if err := json.Unmarshal(resp, &sr); err != nil || len(sr.Results) == 0 {
					continue
				}
				items := make([]map[string]any, len(sr.Results))
				for i, r := range sr.Results {
					items[i] = map[string]any{"episode_id": r.EpisodeID, "displayed_rank": r.Rank}
				}
				imp, _ := json.Marshal(map[string]any{"session_id": sid, "query_id": sr.QueryID, "items": items})
				sent.Add(int64(len(items)))
				status, resp, ms, err = post(ctx, client, *api+"/v1/events/impression", imp)
				record(sample{"event", status, ms})
				if err == nil && status == 202 {
					var acc struct {
						Recorded int64 `json:"recorded"`
					}
					_ = json.Unmarshal(resp, &acc)
					ok.Add(acc.Recorded)
				} else if ctx.Err() == nil {
					fail(fmt.Sprintf("impression %d: %v %s", status, err, truncate(resp)))
				}
				play, _ := json.Marshal(map[string]any{"session_id": sid, "query_id": sr.QueryID,
					"episode_id": sr.Results[0].EpisodeID, "position_seconds": 0})
				sent.Add(1)
				status, resp, ms, err = post(ctx, client, *api+"/v1/events/playback/start", play)
				record(sample{"event", status, ms})
				if err == nil && status == 202 {
					ok.Add(1)
				} else if ctx.Err() == nil {
					fail(fmt.Sprintf("playback %d: %v %s", status, err, truncate(resp)))
				}
			}
		}(c)
	}
	wg.Wait()
	elapsed := time.Since(started)

	r := result{Started: started.UTC(), Duration: elapsed.Round(time.Millisecond).String(),
		Concurrency: *concurrency, API: *api, SearchStatus: map[string]int{}, Errors: errs,
		EventsSent: sent.Load(), EventsOK: ok.Load()}
	var searchMS, eventMS []float64
	for _, s := range samples {
		switch s.kind {
		case "search":
			r.Searches++
			r.SearchStatus[fmt.Sprint(s.status)]++
			searchMS = append(searchMS, s.ms)
		case "event":
			eventMS = append(eventMS, s.ms)
		}
	}
	r.SearchesPerS = float64(r.Searches) / elapsed.Seconds()
	r.Availability = float64(r.SearchStatus["200"]) / float64(max(r.Searches, 1))
	r.SearchMS = percentiles(searchMS)
	r.EventMS = percentiles(eventMS)
	r.EventsStored = -1
	if *dbURL != "" {
		if n, err := storedEvents(context.Background(), *dbURL, *session); err != nil {
			log.Printf("stored events: %v", err)
		} else {
			r.EventsStored = n
		}
	}
	if r.EventsOK > 0 {
		if r.EventsStored >= 0 {
			r.EventStoreSLO = float64(r.EventsStored) / float64(r.EventsOK)
		} else {
			r.EventStoreSLO = float64(r.EventsOK) / float64(max(r.EventsSent, 1))
		}
	}

	fmt.Printf("%d searches in %s at concurrency %d: %.1f/s, availability %.3f%%\n",
		r.Searches, r.Duration, r.Concurrency, r.SearchesPerS, 100*r.Availability)
	fmt.Printf("search ms   p50 %.0f  p95 %.0f  p99 %.0f  max %.0f\n",
		r.SearchMS["p50"], r.SearchMS["p95"], r.SearchMS["p99"], r.SearchMS["max"])
	if len(eventMS) > 0 {
		fmt.Printf("event ms    p50 %.0f  p95 %.0f  p99 %.0f\n", r.EventMS["p50"], r.EventMS["p95"], r.EventMS["p99"])
		fmt.Printf("events      sent %d  accepted %d  stored %d  persistence %.4f%%\n",
			r.EventsSent, r.EventsOK, r.EventsStored, 100*r.EventStoreSLO)
	}
	for k, v := range r.SearchStatus {
		if k != "200" {
			fmt.Printf("status %s: %d\n", k, v)
		}
	}
	for _, e := range errs {
		fmt.Println("  ", e)
	}
	if *out != "" {
		data, _ := json.MarshalIndent(r, "", "  ")
		if err := os.WriteFile(*out, data, 0o644); err != nil {
			log.Fatalf("write %s: %v", *out, err)
		}
	}
}

func post(ctx context.Context, client *http.Client, url string, body []byte) (int, []byte, float64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return 0, nil, 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	start := time.Now()
	resp, err := client.Do(req)
	ms := float64(time.Since(start).Microseconds()) / 1000
	if err != nil {
		return 0, nil, ms, err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	return resp.StatusCode, data, ms, nil
}

func loadQueries(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var row struct {
			Text string `json:"query_text"`
		}
		if err := json.Unmarshal(sc.Bytes(), &row); err == nil && strings.TrimSpace(row.Text) != "" {
			out = append(out, row.Text)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no query_text rows in %s", path)
	}
	return out, sc.Err()
}

func percentiles(v []float64) map[string]float64 {
	if len(v) == 0 {
		return map[string]float64{}
	}
	sort.Float64s(v)
	at := func(q float64) float64 { return v[min(len(v)-1, int(float64(len(v))*q))] }
	return map[string]float64{"p50": at(0.5), "p95": at(0.95), "p99": at(0.99), "max": v[len(v)-1],
		"n": float64(len(v))}
}

func storedEvents(ctx context.Context, url, prefix string) (int64, error) {
	pool, err := pgxpool.New(ctx, url)
	if err != nil {
		return 0, err
	}
	defer pool.Close()
	var n int64
	err = pool.QueryRow(ctx, `SELECT count(*) FROM interaction_events
		WHERE session_id LIKE $1 AND event_type IN ('impression', 'playback_start')`, prefix+"-%").Scan(&n)
	return n, err
}

func truncate(b []byte) string {
	s := strings.TrimSpace(string(b))
	if len(s) > 120 {
		return s[:120] + "…"
	}
	return s
}
