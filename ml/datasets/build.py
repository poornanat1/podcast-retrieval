"""Deterministic dataset builder.

(config, code revision, raw-data snapshot) → identical output, always:
generation is pure given the snapshot, sampling uses per-example seeded RNGs,
and the dataset version embeds a content hash, so rebuilding the same inputs
reproduces the same version and byte-for-byte identical rows.

    uv run python -m ml.datasets.build \
        --config experiments/datasets/search-bootstrap-v1.json \
        --snapshot data/snapshots --out data/datasets [--publish]

A config lists one or more generators; each contributes clearly-labeled
examples (synthetic queries, weak topic-similarity positives, ...) and the
manifest reports the resulting real/human/weak/synthetic composition.

Output: ``<out>/<name>-<version>/`` with ``train.parquet``, ``validation.parquet``,
``test.parquet``, and ``manifest.json``. Splits are time-based on
``event_time`` so later interactions never leak into earlier predictions.
``--publish`` uploads the dataset to the object store as an immutable version.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import random
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.datasets import snapshot as snapshot_mod
from ml.datasets.contracts import COLUMNS, LabelSource
from ml.datasets.validate import ValidationConfig, validate_examples


@dataclasses.dataclass(frozen=True)
class GeneratorSpec:
    """One example generator's slice of a dataset."""

    type: str
    max_examples: int = 50_000
    label_strength: float = 0.3
    language_prefix: str = ""
    max_per_category: int = 500  # topic_similarity only
    # llm_query only: raw inputs relative to the snapshot directory and the
    # negative mix. Each example gets ``hard_negatives`` mined candidates
    # (seeded sample from the top ``hard_pool`` per query), then
    # ``popular_negatives`` drawn with probability proportional to the
    # show's episode count, then random catalog episodes up to
    # ``negatives_per_example``.
    queries_file: str = ""
    negatives_file: str = ""
    hard_negatives: int = 0
    hard_pool: int = 20
    popular_negatives: int = 0
    # Keep only mined candidates surfaced by one of these systems (empty: all).
    # The false-negative audit found vector-mined candidates relevant ~22% of
    # the time and lexical-mined ones ~3%.
    hard_sources: tuple[str, ...] = ()
    # Example ids are ``<id_prefix>:<episode id>``; distinct prefixes let two
    # query files share a dataset.
    id_prefix: str = "llm"


@dataclasses.dataclass(frozen=True)
class BuildConfig:
    name: str
    base_version: str
    seed: int
    validation_start: str  # ISO timestamp: rows at/after go to validation
    test_start: str  # ISO timestamp: rows at/after go to test
    generators: tuple[GeneratorSpec, ...] = ()
    negatives_per_example: int = 4
    validation: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> BuildConfig:
        raw = json.loads(Path(path).read_text())
        raw["generators"] = tuple(
            GeneratorSpec(**{**spec, "hard_sources": tuple(spec.get("hard_sources", ()))})
            for spec in raw.get("generators", [])
        )
        config = cls(**raw)
        if not config.generators:
            raise ValueError("config needs at least one generator")
        if not pd.Timestamp(config.validation_start) < pd.Timestamp(config.test_start):
            raise ValueError("validation_start must precede test_start")
        return config

    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(dataclasses.asdict(self), sort_keys=True).encode()
        ).hexdigest()


_WORD_RE = re.compile(r"[^0-9a-z]+")


def normalize_query(title: str) -> str:
    """Deterministic query text derived from an episode title."""
    return _WORD_RE.sub(" ", title.lower()).strip()


def _stable_order_key(seed: int, kind: str, value: object) -> str:
    return hashlib.sha256(f"{seed}:{kind}:{value}".encode()).hexdigest()


def _sample_negatives(
    seed: int,
    key: object,
    universe: list[int],
    k: int,
    excluded: frozenset[int] = frozenset(),
) -> list[int]:
    """Per-example seeded sampling: stable regardless of how many other
    examples exist or in what order they are generated."""
    rng = random.Random(f"{seed}:neg:{key}")
    chosen: list[int] = []
    seen: set[int] = set()
    while len(chosen) < k and len(seen) < len(universe):
        candidate = universe[rng.randrange(len(universe))]
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate not in excluded:
            chosen.append(candidate)
    return chosen


def _example_row(
    example_id: str,
    event_time: object,
    query: str,
    positive: int,
    negatives: list[int],
    source: LabelSource,
    strength: float,
) -> dict:
    return {
        "example_id": example_id,
        "event_time": event_time,
        "query_text": query,
        "query_audio_variant": "",
        "user_history": [],
        "positive_episode_id": positive,
        "negative_episode_ids": negatives,
        "label_source": source.value,
        "label_strength": strength,
        "dataset_version": "0",  # stamped after hashing
    }


def generate_synthetic_query(
    config: BuildConfig, spec: GeneratorSpec, frames: dict[str, pd.DataFrame]
) -> list[dict]:
    """Bootstrap search-retrieval examples: the normalized episode title is
    the query and the episode is its positive. Clearly labeled synthetic."""
    episodes = frames["episodes"]
    eligible = episodes[
        episodes["published_at"].notna() & (episodes["title"].str.strip() != "")
    ]
    if spec.language_prefix:
        eligible = eligible[
            eligible["language"].str.startswith(spec.language_prefix, na=False)
        ]

    universe = sorted(int(i) for i in episodes["id"])
    picked = set(
        sorted(
            (int(i) for i in eligible["id"]),
            key=lambda i: _stable_order_key(config.seed, "sample", i),
        )[: spec.max_examples]
    )

    rows = []
    for row in eligible.itertuples():
        episode_id = int(row.id)
        if episode_id not in picked:
            continue
        query = normalize_query(row.title)
        if len(query) < 3:
            continue
        rows.append(_example_row(
            f"syn:{episode_id}", row.published_at, query, episode_id,
            _sample_negatives(config.seed, episode_id, universe,
                              config.negatives_per_example,
                              excluded=frozenset({episode_id})),
            LabelSource.SYNTHETIC_QUERY, spec.label_strength,
        ))
    return rows


def generate_topic_similarity(
    config: BuildConfig, spec: GeneratorSpec, frames: dict[str, pd.DataFrame]
) -> list[dict]:
    """Weak positives from category membership: the category name is the
    query; episodes of podcasts in that category weakly satisfy it.
    Negatives are drawn from outside the category, making them meaningful
    rather than purely random."""
    episodes, podcasts = frames["episodes"], frames["podcasts"]

    category_podcasts: dict[str, set[int]] = {}
    for row in podcasts.itertuples():
        # Parquet deserializes text[] as a numpy array; missing as None/NaN.
        raw_categories = row.categories
        if raw_categories is None or isinstance(raw_categories, float):
            continue
        for raw in list(raw_categories):
            name = normalize_query(str(raw))
            if len(name) >= 3:
                category_podcasts.setdefault(name, set()).add(int(row.id))

    eligible = episodes[episodes["published_at"].notna()]
    if spec.language_prefix:
        eligible = eligible[
            eligible["language"].str.startswith(spec.language_prefix, na=False)
        ]
    by_podcast: dict[int, list] = {}
    for row in eligible.itertuples():
        by_podcast.setdefault(int(row.podcast_id), []).append(row)
    universe = sorted(int(i) for i in episodes["id"])
    episode_podcast = dict(
        zip((int(i) for i in episodes["id"]),
            (int(p) for p in episodes["podcast_id"]), strict=True)
    )

    rows = []
    for category in sorted(category_podcasts):
        podcast_ids = category_podcasts[category]
        members = [r for pid in sorted(podcast_ids) for r in by_podcast.get(pid, [])]
        if len(members) < 10:
            continue  # too small to be a meaningful topic
        members = sorted(
            members,
            key=lambda r: _stable_order_key(config.seed, f"topic:{category}", int(r.id)),
        )[: spec.max_per_category]
        slug = category.replace(" ", "-")
        in_category = frozenset(
            i for i in universe if episode_podcast[i] in podcast_ids
        )
        for member in members:
            episode_id = int(member.id)
            negatives = _sample_negatives(
                config.seed, f"{category}:{episode_id}", universe,
                config.negatives_per_example, excluded=in_category,
            )
            rows.append(_example_row(
                f"topic:{slug}:{episode_id}", member.published_at, category,
                episode_id, negatives,
                LabelSource.TOPIC_SIMILARITY, spec.label_strength,
            ))

    rows.sort(key=lambda r: _stable_order_key(config.seed, "topic-cap", r["example_id"]))
    return rows[: spec.max_examples]


def _sample_popular(
    seed: int, key: object, universe: list[int], cum_weights: list[float], k: int,
    excluded: frozenset[int],
) -> list[int]:
    rng = random.Random(f"{seed}:pop:{key}")
    chosen: list[int] = []
    for _ in range(20 * max(k, 1)):
        if len(chosen) >= k:
            break
        candidate = rng.choices(universe, cum_weights=cum_weights, k=1)[0]
        if candidate not in excluded and candidate not in chosen:
            chosen.append(candidate)
    return chosen


def generate_llm_query(
    config: BuildConfig, spec: GeneratorSpec, frames: dict[str, pd.DataFrame]
) -> list[dict]:
    """LLM-written natural-language queries (``ml.datasets.llm_queries``)
    paired with the episode they were written for, with a negative mix of
    mined hard negatives, popularity-weighted negatives, and random ones.
    A query row may list ``exclude_podcasts`` (shows the query names); no
    episode of those shows becomes a negative. Clearly labeled synthetic."""
    episodes = frames["episodes"]
    queries = frames["llm_queries"]
    mined = frames.get("mined_negatives", {})
    by_id = episodes.set_index("id")
    rows_by_id = by_id[["podcast_id", "published_at", "language"]].to_dict("index")

    universe = sorted(int(i) for i in episodes["id"])
    counts = episodes["podcast_id"].value_counts()
    weights = [float(counts[rows_by_id[i]["podcast_id"]]) for i in universe]
    cum_weights = list(itertools.accumulate(weights))

    candidates = []
    for q in queries.itertuples(index=False):
        episode_id = int(q.episode_id)
        meta = rows_by_id.get(episode_id)
        if meta is None or pd.isna(meta["published_at"]):
            continue
        if spec.language_prefix and not str(meta["language"] or "").startswith(
            spec.language_prefix
        ):
            continue
        if len(str(q.query_text).strip()) < 3:
            continue
        shows = getattr(q, "exclude_podcasts", None)
        shows = frozenset(int(i) for i in shows) if isinstance(shows, list | np.ndarray) else (
            frozenset())
        candidates.append((episode_id, str(q.query_text).strip(), meta, shows))
    picked = set(
        sorted((c[0] for c in candidates),
               key=lambda i: _stable_order_key(config.seed, "llm-sample", i))[: spec.max_examples]
    )

    rows = []
    def allowed(candidate: int, shows: frozenset[int]) -> bool:
        return not shows or rows_by_id.get(candidate, {}).get("podcast_id") not in shows

    def from_sources(entry: dict) -> bool:
        return not spec.hard_sources or any(
            src["system"] in spec.hard_sources for src in entry.get("sources", []))

    for episode_id, query, meta, shows in candidates:
        if episode_id not in picked:
            continue
        excluded = frozenset({episode_id})
        negatives: list[int] = []
        hard = [int(h["episode_id"]) for h in mined.get(episode_id, []) if from_sources(h)]
        hard = [h for h in hard[: spec.hard_pool] if h != episode_id and allowed(h, shows)]
        if spec.hard_negatives and hard:
            rng = random.Random(f"{config.seed}:hard:{episode_id}")
            negatives.extend(rng.sample(hard, min(spec.hard_negatives, len(hard))))
        if spec.popular_negatives:
            negatives.extend(c for c in _sample_popular(
                config.seed, episode_id, universe, cum_weights, spec.popular_negatives,
                excluded | frozenset(negatives),
            ) if allowed(c, shows))
        remaining = config.negatives_per_example - len(negatives)
        if remaining > 0:
            # Oversample when shows are excluded so filtering never leaves
            # the example short (a show is a tiny share of the catalog).
            extra = 4 if shows else 0
            negatives.extend([c for c in _sample_negatives(
                config.seed, episode_id, universe, remaining + extra,
                excluded=excluded | frozenset(negatives),
            ) if allowed(c, shows)][:remaining])
        rows.append(_example_row(
            f"{spec.id_prefix}:{episode_id}", meta["published_at"], query, episode_id,
            negatives[: config.negatives_per_example], LabelSource.SYNTHETIC_QUERY,
            spec.label_strength,
        ))
    return rows


GENERATORS = {
    "synthetic_query": generate_synthetic_query,
    "topic_similarity": generate_topic_similarity,
    "llm_query": generate_llm_query,
}


def content_hash(df: pd.DataFrame) -> str:
    """Canonical hash of the rows, independent of file format and of the
    version stamp itself."""
    rows = df.drop(columns=["dataset_version"]).sort_values("example_id")
    lines = []
    for record in rows.to_dict("records"):
        record["event_time"] = pd.Timestamp(record["event_time"]).isoformat()
        record["user_history"] = [int(x) for x in record["user_history"]]
        record["negative_episode_ids"] = [int(x) for x in record["negative_episode_ids"]]
        lines.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def split_by_time(df: pd.DataFrame, config: BuildConfig) -> dict[str, pd.DataFrame]:
    validation_start = pd.Timestamp(config.validation_start)
    test_start = pd.Timestamp(config.test_start)
    t = df["event_time"]
    return {
        "train": df[t < validation_start],
        "validation": df[(t >= validation_start) & (t < test_start)],
        "test": df[t >= test_start],
    }


def code_revision() -> dict[str, Any]:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return {"git_commit": rev, "git_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": "unknown", "git_dirty": True}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dataset(
    config: BuildConfig, snapshot_dir: Path, out_root: Path
) -> tuple[Path, dict[str, Any]]:
    """Build, validate, split, and write one dataset. Returns the output
    directory and its manifest."""
    snapshot_manifest = json.loads((snapshot_dir / "manifest.json").read_text())
    frames = {
        "episodes": pd.read_parquet(snapshot_dir / "episodes.parquet"),
        "podcasts": pd.read_parquet(snapshot_dir / "podcasts.parquet"),
    }

    inputs: dict[str, str] = {}
    rows: list[dict] = []
    for spec in config.generators:
        generator = GENERATORS.get(spec.type)
        if generator is None:
            raise ValueError(f"unknown generator {spec.type!r}")
        frames.pop("llm_queries", None)
        frames.pop("mined_negatives", None)
        if spec.queries_file:
            path = snapshot_dir / spec.queries_file
            frames["llm_queries"] = pd.read_json(path, lines=True)
            inputs[spec.queries_file] = _file_sha256(path)
        if spec.negatives_file:
            path = snapshot_dir / spec.negatives_file
            frames["mined_negatives"] = {
                int(rec["episode_id"]): rec["hard"]
                for rec in (json.loads(line) for line in path.read_text().splitlines() if line)
            }
            inputs[spec.negatives_file] = _file_sha256(path)
        rows.extend(generator(config, spec, frames))

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["positive_episode_id"] = df["positive_episode_id"].astype(np.int64)
    df = df.sort_values("example_id", ignore_index=True)

    digest = content_hash(df)
    version = f"{config.base_version}-{digest[:12]}"
    df["dataset_version"] = version

    result = validate_examples(df, ValidationConfig(**config.validation))
    if not result.ok:
        raise SystemExit(f"dataset failed validation:\n{result.summary()}")

    splits = split_by_time(df, config)
    out_dir = out_root / f"{config.name}-{version}"
    out_dir.mkdir(parents=True, exist_ok=True)

    split_meta = {}
    for name, part in splits.items():
        path = out_dir / f"{name}.parquet"
        part.to_parquet(path, index=False)
        split_meta[name] = {
            "rows": len(part),
            "event_time_min": str(part["event_time"].min()) if len(part) else "",
            "event_time_max": str(part["event_time"].max()) if len(part) else "",
            "sha256": _file_sha256(path),
        }

    manifest = {
        "name": config.name,
        "dataset_version": version,
        "content_hash": digest,
        "created_at": datetime.now(UTC).isoformat(),
        **code_revision(),
        "config": dataclasses.asdict(config),
        "config_sha256": config.sha256(),
        "snapshot_id": snapshot_manifest["snapshot_id"],
        "inputs": inputs,
        "splits": split_meta,
        "label_class_shares": result.report.class_shares if result.report else {},
        "label_source_counts": result.report.source_counts if result.report else {},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_dir, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--snapshot", default="data/snapshots")
    parser.add_argument("--out", default="data/datasets")
    parser.add_argument("--publish", action="store_true",
                        help="upload the dataset to the object store")
    args = parser.parse_args(argv)

    config = BuildConfig.load(args.config)
    snapshot_dir = snapshot_mod.resolve(args.snapshot)
    out_dir, manifest = build_dataset(config, snapshot_dir, Path(args.out))
    print(json.dumps({k: manifest[k] for k in
                      ("name", "dataset_version", "snapshot_id", "splits",
                       "label_class_shares")}, indent=2))
    print(f"dataset written to {out_dir}")

    if args.publish:
        from ml.datasets.publish import publish_dataset

        location = publish_dataset(out_dir, manifest)
        print(f"published to {location}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
