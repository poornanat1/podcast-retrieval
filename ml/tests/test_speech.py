import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml.speech import audio
from ml.speech.server import create_app
from ml.speech.transcribe import Transcript, check_length


def tone(rate: int, seconds: float = 0.5, hz: float = 440.0) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_wav_round_trip_and_resampling() -> None:
    clip = audio.decode(audio.encode(tone(24_000), 24_000))
    assert clip.rate == 24_000 and clip.channels == 1
    assert clip.seconds == pytest.approx(0.5)
    mono = audio.to_mono_16k(clip)
    assert mono.dtype == np.float32 and len(mono) == 8_000
    assert np.abs(mono).max() == pytest.approx(0.5, abs=0.02)


def test_decode_float_and_stereo_wav() -> None:
    left, right = tone(16_000), np.zeros(8_000, dtype=np.float32)
    interleaved = np.stack([left, right], axis=1).astype("<f4").tobytes()
    header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(interleaved), b"WAVE", b"fmt ",
                         16, audio.IEEE_FLOAT, 2, 16_000, 16_000 * 8, 8, 32, b"data",
                         len(interleaved))
    clip = audio.decode(header + interleaved)
    assert clip.channels == 2
    mono = audio.to_mono_16k(clip)
    assert np.allclose(mono, left / 2, atol=1e-6)


def test_decode_rejects_non_wav() -> None:
    with pytest.raises(ValueError):
        audio.decode(b"ID3 not a wav")
    with pytest.raises(ValueError):
        audio.decode(b"RIFF\x00\x00\x00\x00WAVE")


def test_length_limits() -> None:
    with pytest.raises(ValueError):
        check_length(np.zeros(16_000 * 61, dtype=np.float32))
    with pytest.raises(ValueError):
        check_length(np.zeros(100, dtype=np.float32))
    check_length(np.zeros(16_000, dtype=np.float32))


class FakeBackend:
    name = "fake:test"

    def __init__(self):
        self.calls = []

    def transcribe(self, samples, language=None):
        check_length(samples)
        self.calls.append((len(samples), language))
        return Transcript(text="history hit", language=language or "en", confidence=-0.1,
                          backend=self.name)


def test_speech_server_transcribes_wav_and_rejects_bad_input() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(backend))
    body = audio.encode(tone(44_100, 1.0), 44_100)
    r = client.post("/v1/transcribe?language=en", content=body,
                    headers={"content-type": "audio/wav"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["text"] == "history hit" and out["language"] == "en"
    assert out["backend"] == "fake:test" and out["audio_seconds"] == pytest.approx(1.0)
    assert backend.calls == [(16_000, "en")]

    r = client.post("/v1/transcribe", content=b"not audio")
    assert r.status_code == 422
    r = client.post("/v1/transcribe", content=audio.encode(np.zeros(10, np.float32), 16_000))
    assert r.status_code == 422
    assert client.get("/v1/stt").json()["backend"] == "fake:test"
