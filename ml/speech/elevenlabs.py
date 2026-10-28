"""ElevenLabs text-to-speech for the speech test set.

A hosted alternative to Chatterbox: real-accent voices from the voice
library, one multilingual model (Eleven v4), no hallucination on short
names. It costs money (characters on a paid plan), sends the query texts
to a third party, and is reproducible only as "same voice, model and seed",
since the hosted model can change; the manifest records all three.

Plain HTTP through urllib, so the main environment gains no dependency.
``ELEVENLABS_API_KEY`` must be set. Output is requested as WAV at the
test set's rate and decoded with ``ml.speech.audio``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from ml.speech import audio

API = "https://api.elevenlabs.io/v1"
DEFAULT_MODEL = "eleven_v4"


class ElevenLabs:
    def __init__(self, api_key: str | None = None, model_id: str = DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")
        self.model_id = model_id

    def _request(self, method: str, path: str, body: dict | None = None,
                 query: dict | None = None, retries: int = 4) -> bytes:
        url = f"{API}{path}"
        if query:
            url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data, method=method, headers={
            "xi-api-key": self.api_key, "content-type": "application/json",
            "accept": "application/json, audio/wav, audio/*"})
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return resp.read()
            except urllib.error.HTTPError as err:
                detail = err.read()[:500].decode(errors="replace")
                if err.code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"ElevenLabs {method} {path}: {err.code} {detail}") from err
        raise AssertionError("unreachable")

    def _json(self, method: str, path: str, **kwargs) -> dict:
        return json.loads(self._request(method, path, **kwargs) or b"{}")

    def subscription(self) -> dict:
        return self._json("GET", "/user/subscription")

    def voices(self) -> list[dict]:
        """The account's own voices (premade, cloned, and library voices
        added to it)."""
        return self._json("GET", "/voices").get("voices", [])

    def library(self, language: str | None = None, accent: str | None = None,
                gender: str | None = None, use_cases: str | None = None,
                page_size: int = 30) -> list[dict]:
        """Search the shared voice library."""
        return self._json("GET", "/shared-voices", query={
            "language": language, "accent": accent, "gender": gender,
            "use_cases": use_cases, "page_size": page_size}).get("voices", [])

    def add_from_library(self, public_owner_id: str, voice_id: str, name: str) -> str:
        """Add a library voice to the account (required before it can be
        used for speech). Returns the voice id to synthesize with."""
        out = self._json("POST", f"/voices/add/{public_owner_id}/{voice_id}",
                         body={"new_name": name})
        return out.get("voice_id", voice_id)

    def synthesize(self, text: str, voice_id: str, rate: int = 24_000, seed: int | None = None,
                   language: str | None = None, settings: dict | None = None) -> audio.Clip:
        body = {"text": text, "model_id": self.model_id}
        if seed is not None:
            body["seed"] = seed
        if language:
            body["language_code"] = language
        if settings:
            body["voice_settings"] = settings
        wav = self._request("POST", f"/text-to-speech/{voice_id}", body,
                            query={"output_format": f"wav_{rate}"})
        return audio.decode(wav)


def main(argv: list[str] | None = None) -> int:
    """List candidate library voices for a locale, or add one to the account.

        uv run python -m ml.speech.elevenlabs --language en --accent indian
        uv run python -m ml.speech.elevenlabs --add <owner_id> <voice_id> --name en_in
        uv run python -m ml.speech.elevenlabs --voices
    """
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--language")
    parser.add_argument("--accent")
    parser.add_argument("--gender")
    parser.add_argument("--add", nargs=2, metavar=("PUBLIC_OWNER_ID", "VOICE_ID"))
    parser.add_argument("--name", default="")
    parser.add_argument("--voices", action="store_true", help="the account's own voices")
    args = parser.parse_args(argv)
    client = ElevenLabs()
    if args.add:
        print(client.add_from_library(args.add[0], args.add[1], args.name or args.add[1]))
        return 0
    rows = client.voices() if args.voices else client.library(
        language=args.language, accent=args.accent, gender=args.gender)
    for v in rows:
        labels = v.get("labels", {})
        print(v.get("voice_id"), v.get("public_owner_id", ""), repr(v.get("name")),
              v.get("accent") or labels.get("accent", ""),
              v.get("language") or labels.get("language", ""), v.get("locale", ""),
              v.get("gender") or labels.get("gender", ""),
              "free" if v.get("free_users_allowed") else "")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
