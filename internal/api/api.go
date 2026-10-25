// Package api is the public HTTP surface of PodFind: search, episodes,
// interaction events, health, and the embedded web app.
package api

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"log/slog"
	"net/http"
	"strconv"
	"time"

	"podfind/internal/search"
)

const maxBody = 64 << 10

// Handler routes the API.
type Handler struct {
	Search *search.Service
	Web    fs.FS // static web app, served at /
	mux    *http.ServeMux
}

// New builds the router.
func New(svc *search.Service, web fs.FS) *Handler {
	h := &Handler{Search: svc, Web: web, mux: http.NewServeMux()}
	h.mux.HandleFunc("POST /v1/search/text", h.searchText)
	h.mux.HandleFunc("GET /v1/search/{queryId}", h.getSearch)
	h.mux.HandleFunc("GET /v1/episodes/{episodeId}", h.getEpisode)
	h.mux.HandleFunc("POST /v1/events/impression", h.impression)
	h.mux.HandleFunc("POST /v1/events/playback/{action}", h.playback)
	h.mux.HandleFunc("GET /v1/model", h.model)
	h.mux.HandleFunc("GET /healthz", h.health)
	if web != nil {
		h.mux.Handle("GET /", http.FileServerFS(web))
	}
	return h
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
	h.mux.ServeHTTP(rec, r)
	if r.URL.Path != "/healthz" {
		slog.Info("http", "method", r.Method, "path", r.URL.Path, "status", rec.status,
			"ms", float64(time.Since(start).Microseconds())/1000)
	}
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, search.ErrBadRequest):
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
	case errors.Is(err, search.ErrNotFound):
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
	case errors.Is(err, context.DeadlineExceeded):
		writeJSON(w, http.StatusGatewayTimeout, map[string]string{"error": "timed out"})
	default:
		slog.Error("request failed", "error", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "internal error"})
	}
}

func decode(w http.ResponseWriter, r *http.Request, v any) error {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBody))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return errors.Join(search.ErrBadRequest, err)
	}
	return nil
}

func (h *Handler) searchText(w http.ResponseWriter, r *http.Request) {
	var req search.Request
	if err := decode(w, r, &req); err != nil {
		writeError(w, err)
		return
	}
	resp, err := h.Search.Search(r.Context(), req)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (h *Handler) getSearch(w http.ResponseWriter, r *http.Request) {
	s, err := h.Search.GetSearch(r.Context(), r.PathValue("queryId"))
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, s)
}

func (h *Handler) getEpisode(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(r.PathValue("episodeId"), 10, 64)
	if err != nil {
		writeError(w, search.ErrNotFound)
		return
	}
	e, err := h.Search.GetEpisode(r.Context(), id)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, e)
}

func (h *Handler) impression(w http.ResponseWriter, r *http.Request) {
	var in search.Impressions
	if err := decode(w, r, &in); err != nil {
		writeError(w, err)
		return
	}
	n, err := h.Search.RecordImpressions(r.Context(), in)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]int{"recorded": n})
}

func (h *Handler) playback(w http.ResponseWriter, r *http.Request) {
	var in search.Playback
	if err := decode(w, r, &in); err != nil {
		writeError(w, err)
		return
	}
	if err := h.Search.RecordPlayback(r.Context(), r.PathValue("action"), in); err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]int{"recorded": 1})
}

func (h *Handler) model(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, h.Search.Retriever.Info)
}

func (h *Handler) health(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	status := map[string]string{"database": "ok", "model_server": "ok",
		"bundle_id": h.Search.Retriever.Info.BundleID}
	code := http.StatusOK
	if err := h.Search.Pool.Ping(ctx); err != nil {
		status["database"], code = err.Error(), http.StatusServiceUnavailable
	}
	if m, err := h.Search.Model.Model(ctx); err != nil {
		status["model_server"], code = err.Error(), http.StatusServiceUnavailable
	} else if m.BundleID != h.Search.Retriever.Info.BundleID {
		status["model_server"], code = "serving bundle "+m.BundleID+", expected "+
			h.Search.Retriever.Info.BundleID, http.StatusServiceUnavailable
	}
	writeJSON(w, code, status)
}
