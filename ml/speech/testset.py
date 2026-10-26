"""Build the speech test set: spoken versions of the relevance queries.

    uv run python -m ml.speech.testset --config experiments/speech/testset-v1.json

No human recordings exist for these queries, so the set is synthetic and
says so. Chatterbox (a neural text-to-speech model, in its own environment
under ``ml/speech/tts``) speaks each query while cloning the voice of a
short reference recording; the references are macOS system voices in six
English locales and four other languages, so "accent" here means the
accent of a synthetic voice as cloned by another model. Noise,
microphones, rooms and speaking rates are then applied to the clean clips
by ``ml.speech.degrade``, so each variant differs from its clean clip in
exactly one way.

Tiers (counts for the v1 config):

- **base**: every relevance query, terse (the query text as written), in
  the base voice: 180 clips.
- **accent**: a stratified subset of English queries in five other
  English voices: 300 clips.
- **conversational**: a subset of exploratory and filtered queries phrased
  as a spoken request ("could you find me some episodes about …"), which
  query understanding must strip: 40 clips.
- **condition**: the subset's base clips under each degradation: 540.
- **speed**: the subset's base clips at 0.8×, 1.3× and 1.6× speaking rate: 180.

Everything is reproducible from the config's seed. The manifest records,
per clip, the query, what was spoken, the voice, condition and speed, the
query's filters, and its named entities (``ml.speech.entities``). Audio is
not tracked in git; the manifest and config are.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ml.evaluation.run import load_queries
from ml.speech import audio, degrade
from ml.speech.entities import annotate, catalog_names

CONVERSATIONAL = (
    "could you find me some podcast episodes about {}",
    "I'm looking for a podcast on {}",
    "find episodes about {} please",
    "are there any podcasts that talk about {}",
    "I want to listen to something about {}",
    "play me an episode about {}",
)

FILTER_KEYS = ("language", "max_duration_seconds", "published_after", "no_explicit")


def say_reference(path: Path, voice: str, text: str, rate: int) -> None:
    subprocess.run(["say", "-v", voice, "-o", str(path), f"--data-format=LEI16@{rate}", text],
                   check=True)


def run_synthesizer(jobs_file: Path) -> None:
    project = Path(__file__).parent / "tts"
    subprocess.run(["uv", "run", "--project", str(project), "python",
                    str(project / "synthesize.py"), "--jobs", str(jobs_file)], check=True)


def stratified_subset(queries: list[dict], per_type: dict[str, int], rng: np.random.Generator,
                      text_language: dict[str, str] | None = None) -> list[dict]:
    """A seeded sample per query type, English texts only."""
    chosen = []
    for query_type, n in per_type.items():
        pool = [q for q in queries if q["query_type"] == query_type
                and query_language(q, text_language or {}) == "en"]
        pool.sort(key=lambda q: q["query_id"])
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        chosen.extend(pool[i] for i in sorted(idx))
    return chosen


def stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def query_language(q: dict, text_language: dict[str, str]) -> str:
    """The language the query text is written in. A language filter does
    not decide it ("short daily spanish lesson" is English with filter
    es), so the config names the language of each non-English text."""
    return text_language.get(q["query_id"], "en")


def build(config: dict, synthesize: bool = True,
          say: Callable[[Path, str, str, int], None] = say_reference,
          synthesizer: Callable[[Path], None] = run_synthesizer,
          names: set[str] | None = None) -> list[dict]:
    rng = np.random.default_rng(config["seed"])
    out = Path(config["out_dir"])
    rate = config["sample_rate"]
    (out / "references").mkdir(parents=True, exist_ok=True)
    (out / "clips").mkdir(exist_ok=True)
    queries = load_queries(Path(config["relevance_dir"]) / "queries.jsonl")
    if names is None:
        import psycopg

        with psycopg.connect(os.environ.get(
                "DATABASE_URL", "postgres://podfind:podfind@localhost:5432/podfind")) as conn:
            names = catalog_names(conn)
    entities = annotate(queries, names)
    text_language = config.get("text_language", {})

    voices = config["voices"]
    for key, voice in voices.items():
        ref = out / "references" / f"{key}.wav"
        if not ref.exists():
            say(ref, voice["say"], config["reference_text"][voice["language"]], rate)
    by_language = {v["language"]: k for k, v in voices.items() if k not in config["accents"]}
    by_language["en"] = config["base_voice"]

    subset = stratified_subset(queries, config["subset_per_type"], rng, text_language)
    conversational = stratified_subset(queries, config["conversational_per_type"], rng,
                                       text_language)
    synth = config["synthesis"]

    def clip(q: dict, tier: str, phrasing: str, text: str, voice: str) -> dict:
        clip_id = f"{q['query_id']}-{phrasing[0]}-{voice}"
        return {"clip_id": clip_id, "path": f"clips/{clip_id}.wav", "tier": tier,
                "query_id": q["query_id"], "query_type": q["query_type"],
                "language": query_language(q, text_language), "phrasing": phrasing,
                "text": text,
                "terse_text": q["query_text"], "voice": voice,
                "locale": voices[voice]["locale"], "condition": "clean", "speed": 1.0,
                "entities": entities[q["query_id"]],
                "filters": {k: q[k] for k in FILTER_KEYS if q.get(k)}}

    rows = [clip(q, "base", "terse", q["query_text"],
                 by_language[query_language(q, text_language)]) for q in queries]
    rows += [clip(q, "accent", "terse", q["query_text"], accent)
             for q in subset for accent in config["accents"]]
    for q in conversational:
        template = CONVERSATIONAL[int(rng.integers(len(CONVERSATIONAL)))]
        rows.append(clip(q, "conversational", "conversational", template.format(q["query_text"]),
                         by_language[query_language(q, text_language)]))

    jobs = [{"clip_id": r["clip_id"], "text": r["text"], "language": r["language"],
             "reference": str(out / "references" / f"{r['voice']}.wav"),
             "path": str(out / r["path"]), **synth} for r in rows]
    jobs_file = out / "jobs.jsonl"
    jobs_file.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    if synthesize:
        synthesizer(jobs_file)
    missing = [j["clip_id"] for j in jobs if not (out / f"clips/{j['clip_id']}.wav").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} clips were not synthesized, e.g. {missing[:3]}")

    # Derived variants of the subset's base clips.
    base = {r["query_id"]: r for r in rows if r["tier"] == "base"}
    loaded = {qid: audio.to_mono_16k(audio.read(out / r["path"])) for qid, r in base.items()}
    (out / "clips" / "derived").mkdir(exist_ok=True)
    derived = []
    for q in subset:
        src = base[q["query_id"]]
        samples = loaded[q["query_id"]]
        others = [loaded[o["query_id"]] for o in subset if o["query_id"] != q["query_id"]]
        variants = [(c, c, 1.0) for c in config["conditions"]]
        variants += [(f"speed_{name}", "clean", factor)
                     for name, factor in config["speeds"].items()]
        for suffix, condition, factor in variants:
            clip_id = f"{src['clip_id']}-{suffix}"
            path = f"clips/derived/{clip_id}.wav"
            if not (out / path).exists():
                local = np.random.default_rng([config["seed"], stable_hash(clip_id)])
                out_samples = degrade.apply(condition, samples, audio.TARGET_RATE, local, others) \
                    if factor == 1.0 else degrade.speed(samples, factor, audio.TARGET_RATE)
                audio.write(out / path, out_samples, audio.TARGET_RATE)
            derived.append({**src, "clip_id": clip_id, "path": path,
                            "tier": "speed" if factor != 1.0 else "condition",
                            "condition": condition, "speed": factor})
    rows += derived

    for r in rows:
        r["seconds"] = round(audio.read(out / r["path"]).seconds, 3)
    manifest = out / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="experiments/speech/testset-v1.json")
    parser.add_argument("--no-synthesize", action="store_true",
                        help="use clips already synthesized; fail on any missing one")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    rows = build(config, synthesize=not args.no_synthesize)
    tiers = {}
    for r in rows:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    print(f"{len(rows)} clips: {tiers}; manifest in {config['out_dir']}/manifest.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
