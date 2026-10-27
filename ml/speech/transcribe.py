"""Speech-to-text backends.

``load(name)`` returns a ``Transcriber`` for a backend name:

- ``whisper:<model>``: Whisper through ``transformers``, locally
  (default ``whisper:openai/whisper-small``; multilingual, runs on CPU or
  Apple GPU). Nothing leaves the machine.
- ``openai:<model>``: OpenAI's hosted transcription (``OPENAI_API_KEY``).
  The recording is sent to a third party; only for deployments that accept
  that.

Both return a ``Transcript``: the text, the language (given or detected),
a confidence (mean token log-probability for Whisper, absent for hosted
models), and the backend name, which the search stores so a transcript can
be traced to the recognizer that produced it.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ml.speech.audio import TARGET_RATE, encode

MAX_SECONDS = 60.0  # one Whisper window is 30 s; longer input is rejected, not chunked
DEFAULT_BACKEND = "whisper:openai/whisper-small"


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None
    confidence: float | None
    backend: str


class Transcriber(Protocol):
    name: str

    def transcribe(self, samples: np.ndarray, language: str | None = None) -> Transcript:
        """``samples``: 16 kHz mono float32. ``language``: ISO 639-1 hint,
        or None to detect."""


def check_length(samples: np.ndarray) -> None:
    seconds = len(samples) / TARGET_RATE
    if seconds > MAX_SECONDS:
        raise ValueError(f"audio is {seconds:.1f} s; at most {MAX_SECONDS:.0f} s is accepted")
    if seconds < 0.1:
        raise ValueError("audio is shorter than 0.1 s")


class WhisperTranscriber:
    def __init__(self, model: str = "openai/whisper-small", device: str | None = None):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.name = f"whisper:{model}"
        self.processor = WhisperProcessor.from_pretrained(model)
        self.model = WhisperForConditionalGeneration.from_pretrained(model)
        self.model.eval()
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.torch = torch

    def transcribe(self, samples: np.ndarray, language: str | None = None) -> Transcript:
        check_length(samples)
        torch = self.torch
        feats = self.processor(samples, sampling_rate=TARGET_RATE, return_tensors="pt")
        features = feats.input_features.to(self.device)
        tokenizer = self.processor.tokenizer
        with torch.inference_mode():
            if not language:
                detected = self.model.detect_language(features).tolist()
                token = tokenizer.convert_ids_to_tokens(detected)[0]
                language = token[2:-2]
            generated = self.model.generate(features, task="transcribe", language=language,
                                            num_beams=1, max_new_tokens=128)[0]
            # Confidence: mean log-probability of the generated tokens under
            # the model, from one teacher-forced decoder pass over the same
            # prompt (start, language, task, no-timestamps) generate used.
            prompt = [self.model.generation_config.decoder_start_token_id] + [
                tok for _, tok in tokenizer.get_decoder_prompt_ids(language=language,
                                                                    task="transcribe")]
            decoder_ids = torch.cat([torch.tensor(prompt, device=generated.device), generated])
            logits = self.model(input_features=features,
                                decoder_input_ids=decoder_ids[None]).logits[0]
        logprobs = torch.log_softmax(logits[len(prompt) - 1:-1].float(), dim=-1)
        picked = logprobs.gather(1, generated[:, None].to(logprobs.device))[:, 0]
        special = set(tokenizer.all_special_ids)
        kept = [lp for tok, lp in zip(generated.tolist(), picked.tolist(), strict=True)
                if tok not in special]
        confidence = float(np.mean(kept)) if kept else None
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return Transcript(text=text, language=language, confidence=confidence, backend=self.name)


class OpenAITranscriber:
    def __init__(self, model: str = "gpt-4o-mini-transcribe"):
        from openai import OpenAI

        self.name = f"openai:{model}"
        self.model = model
        self.client = OpenAI()

    def transcribe(self, samples: np.ndarray, language: str | None = None) -> Transcript:
        check_length(samples)
        audio = io.BytesIO(encode(samples, TARGET_RATE))
        audio.name = "query.wav"
        kwargs = {"model": self.model, "file": audio}
        if language:
            kwargs["language"] = language
        result = self.client.audio.transcriptions.create(**kwargs)
        return Transcript(text=result.text.strip(), language=language, confidence=None,
                          backend=self.name)


def load(name: str = DEFAULT_BACKEND, device: str | None = None) -> Transcriber:
    """``device`` applies to local backends: cpu, mps or cuda; None picks
    the Apple GPU when present, else the CPU."""
    kind, _, model = name.partition(":")
    if kind == "whisper":
        return WhisperTranscriber(model or "openai/whisper-small", device)
    if kind == "openai":
        return OpenAITranscriber(model or "gpt-4o-mini-transcribe")
    raise ValueError(f"unknown speech backend {name!r}; use whisper:<model> or openai:<model>")
