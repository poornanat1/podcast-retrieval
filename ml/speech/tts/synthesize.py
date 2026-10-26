"""Synthesize the speech test set's clips with Chatterbox.

    uv run --project ml/speech/tts python ml/speech/tts/synthesize.py --jobs jobs.jsonl

Each job line names the clip's text, the reference recording whose voice
and accent to clone, the language, the output path, and Chatterbox's
``exaggeration`` and ``cfg_weight``. English jobs use the English model;
other languages use the multilingual one. Existing outputs are skipped, so
an interrupted run resumes. Every clip is seeded from its id, so the set
is reproducible on the same hardware and library versions. Output is
16-bit mono WAV at the model's 24 kHz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def seed_for(clip_id: str) -> int:
    return int.from_bytes(hashlib.sha256(clip_id.encode()).digest()[:4], "little")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    import torch
    import torchaudio

    device = args.device or ("mps" if torch.backends.mps.is_available() else
                             "cuda" if torch.cuda.is_available() else "cpu")
    jobs = [json.loads(line) for line in Path(args.jobs).read_text().splitlines() if line.strip()]
    pending = [j for j in jobs if not Path(j["path"]).exists()]
    print(f"{len(jobs)} jobs, {len(pending)} to synthesize on {device}", flush=True)
    models: dict[str, object] = {}

    def model_for(language: str):
        key = "en" if language == "en" else "mtl"
        if key not in models:
            if key == "en":
                from chatterbox.tts import ChatterboxTTS
                models[key] = ChatterboxTTS.from_pretrained(device=device)
            else:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                models[key] = ChatterboxMultilingualTTS.from_pretrained(device=device)
        return models[key]

    started = time.time()
    for n, job in enumerate(pending, 1):
        model = model_for(job["language"])
        torch.manual_seed(seed_for(job["clip_id"]))
        kwargs = {"audio_prompt_path": job["reference"], "exaggeration": job["exaggeration"],
                  "cfg_weight": job["cfg_weight"]}
        if job["language"] != "en":
            kwargs["language_id"] = job["language"]
        wav = model.generate(job["text"], **kwargs)
        out = Path(job["path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(out), wav.cpu(), model.sr, encoding="PCM_S", bits_per_sample=16)
        if n % 10 == 0 or n == len(pending):
            rate = (time.time() - started) / n
            print(f"{n}/{len(pending)} clips, {rate:.1f} s each, "
                  f"{rate * (len(pending) - n) / 60:.0f} min left", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
