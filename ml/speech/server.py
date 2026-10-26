"""PodFind speech server: transcription for voice search.

    uv run python -m ml.speech.server [--backend whisper:openai/whisper-small] [--port 8082]

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

from fastapi import FastAPI, HTTPException, Request

from ml.speech import transcribe as stt
from ml.speech.audio import decode, to_mono_16k

MAX_BYTES = 8 << 20


def create_app(backend: stt.Transcriber) -> FastAPI:
    app = FastAPI(title="podfind-speech-server", version=backend.name)

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
            raise HTTPException(status_code=413, detail=f"audio larger than {MAX_BYTES} bytes")
        try:
            samples = to_mono_16k(decode(data))
            result = backend.transcribe(samples, language or None)
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        return {"text": result.text, "language": result.language,
                "confidence": result.confidence, "backend": result.backend,
                "audio_seconds": round(len(samples) / 16_000, 3),
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 3)}

    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default=os.environ.get("PODFIND_STT", stt.DEFAULT_BACKEND))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8082")))
    args = parser.parse_args(argv)
    backend = stt.load(args.backend)
    print(f"speech server: {backend.name}", flush=True)
    uvicorn.run(create_app(backend), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
