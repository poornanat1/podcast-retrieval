"""PodFind speech server: transcription for voice search.

    uv run python -m ml.speech.server [--backend whisper:<model>] [--device cpu] [--port 8082]

Separate from the model server on purpose: that one runs ONNX only and
must stay small, while the recognizer is a PyTorch model. The Go API posts
the recording here and discards it once the transcript is back; this
server never writes audio to disk.

Endpoints:

- ``GET  /healthz``: liveness and the backend name.
- ``GET  /v1/stt``: the backend and limits.
- ``POST /v1/transcribe``: body is a WAV file (``audio/wav``); optional
  ``?language=xx`` hint. Returns the transcript, the language, a
  confidence, the backend, the audio length and the time taken.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from ml.speech import transcribe as stt
from ml.speech.audio import decode, to_mono_16k

MAX_BYTES = 8 << 20

TRANSCRIBE_SECONDS = Histogram("podfind_speech_transcribe_seconds", "Transcription time.",
                               buckets=(.1, .25, .5, 1, 2, 4, 8, 16, 30))
AUDIO_SECONDS = Histogram("podfind_speech_audio_seconds", "Length of audio transcribed.",
                          buckets=(1, 2, 3, 5, 8, 12, 20, 30, 60))
CONFIDENCE = Histogram("podfind_speech_confidence", "Mean token log-probability.",
                       buckets=[round(-3 + 0.25 * i, 2) for i in range(13)])
LANGUAGE = Counter("podfind_speech_language_total", "Transcript language.", ["language"])
ERRORS = Counter("podfind_speech_errors_total", "Rejected or failed requests.", ["kind"])


def create_app(backend: stt.Transcriber) -> FastAPI:
    app = FastAPI(title="podfind-speech-server", version=backend.name)

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "backend": backend.name}

    @app.get("/v1/stt")
    def info() -> dict:
        return {"backend": backend.name, "max_seconds": stt.MAX_SECONDS, "max_bytes": MAX_BYTES,
                "formats": ["audio/wav"]}

    @app.post("/v1/transcribe")
    async def transcribe_endpoint(request: Request, language: str | None = None) -> dict:
        start = time.perf_counter()
        data = await request.body()
        if len(data) > MAX_BYTES:
            ERRORS.labels("too_large").inc()
            raise HTTPException(status_code=413, detail=f"audio larger than {MAX_BYTES} bytes")
        try:
            samples = to_mono_16k(decode(data))
            result = backend.transcribe(samples, language or None)
        except ValueError as err:
            ERRORS.labels("bad_audio").inc()
            raise HTTPException(status_code=422, detail=str(err)) from err
        TRANSCRIBE_SECONDS.observe(time.perf_counter() - start)
        AUDIO_SECONDS.observe(len(samples) / 16_000)
        if result.confidence is not None:
            CONFIDENCE.observe(result.confidence)
        LANGUAGE.labels(result.language or "unknown").inc()
        return {"text": result.text, "language": result.language,
                "confidence": result.confidence, "backend": result.backend,
                "audio_seconds": round(len(samples) / 16_000, 3),
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 3)}

    return app


def main(argv: list[str] | None = None) -> int:
    from ml.serving.listen import serve

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default=os.environ.get("PODFIND_STT", stt.DEFAULT_BACKEND))
    parser.add_argument("--device", default=os.environ.get("PODFIND_STT_DEVICE") or None,
                        help="cpu, mps or cuda for local backends (default: GPU if present)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8082")))
    args = parser.parse_args(argv)
    backend = stt.load(args.backend, args.device)
    print(f"speech server: {backend.name} on {getattr(backend, 'device', 'remote')}", flush=True)
    serve(create_app(backend), args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
