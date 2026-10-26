// Package api is the public HTTP surface of PodFind: search, episodes,
// interaction events, health, and the embedded web app.
package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
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
	h.mux.HandleFunc("POST /v1/search/voice", h.searchVoice)
	h.mux.HandleFunc("GET /v1/search/{queryId}", h.getSearch)
	h.mux.HandleFunc("GET /v1/episodes/{episodeId}", h.getEpisode)
	h.mux.HandleFunc("POST /v1/events/impression", h.impression)
	h.mux.HandleFunc("POST /v1/events/playback/{action}", h.playback)
	h.mux.HandleFunc("GET /v1/me/data", h.exportData)
	h.mux.HandleFunc("DELETE /v1/me/data", h.deleteData)
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
	case errors.Is(err, search.ErrUnavailable):
		slog.Error("dependency unavailable", "error", err)
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
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

// searchVoice takes a multipart form: the WAV recording as "audio", and
// the text search's fields as form values (session_id, limit, language,
// filters as JSON, retain as "true"). The recording lives only in this
// request unless retain is set and a store is configured.
func (h *Handler) searchVoice(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, search.MaxVoiceBytes+maxBody)
	if err := r.ParseMultipartForm(search.MaxVoiceBytes); err != nil {
		writeError(w, errors.Join(search.ErrBadRequest, err))
		return
	}
	defer r.MultipartForm.RemoveAll() //nolint:errcheck
	file, _, err := r.FormFile("audio")
	if err != nil {
		writeError(w, fmt.Errorf("%w: a WAV file is required as \"audio\"", search.ErrBadRequest))
		return
	}
	defer file.Close()
	audio, err := io.ReadAll(io.LimitReader(file, search.MaxVoiceBytes+1))
	if err != nil {
		writeError(w, errors.Join(search.ErrBadRequest, err))
		return
	}
	req := search.VoiceRequest{Audio: audio, Language: r.FormValue("language"),
		SessionID: r.FormValue("session_id"), Retain: r.FormValue("retain") == "true",
		Debug: r.FormValue("debug") == "true", SkipUnderstanding: r.FormValue("skip_understanding") == "true"}
	if v := r.FormValue("filters"); v != "" {
		dec := json.NewDecoder(strings.NewReader(v))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&req.Filters); err != nil {
			writeError(w, fmt.Errorf("%w: filters: %v", search.ErrBadRequest, err))
			return
		}
	}
	if v := r.FormValue("limit"); v != "" {
		if req.Limit, err = strconv.Atoi(v); err != nil {
			writeError(w, fmt.Errorf("%w: limit must be an integer", search.ErrBadRequest))
			return
		}
	}
	if v := r.FormValue("reference_time"); v != "" {
		t, err := time.Parse(time.RFC3339Nano, v)
		if err != nil {
			writeError(w, fmt.Errorf("%w: reference_time must be RFC 3339", search.ErrBadRequest))
			return
		}
		req.ReferenceTime = &t
	}
	resp, err := h.Search.SearchVoice(r.Context(), req)
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

// Data controls, keyed by session id (the API has no accounts).
func (h *Handler) exportData(w http.ResponseWriter, r *http.Request) {
	data, err := h.Search.ExportSession(r.Context(), r.URL.Query().Get("session_id"))
	if err != nil {
		writeError(w, err)
		return
	}
	w.Header().Set("Content-Disposition", "attachment; filename=\"podfind-"+data.SessionID+".json\"")
	writeJSON(w, http.StatusOK, data)
}

func (h *Handler) deleteData(w http.ResponseWriter, r *http.Request) {
	deleted, err := h.Search.DeleteSession(r.Context(), r.URL.Query().Get("session_id"))
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, deleted)
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
