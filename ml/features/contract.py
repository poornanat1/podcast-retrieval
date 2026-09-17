"""Feature contract shared by two-tower training and serving.

The contract pins everything that turns a raw query or catalog row into
model inputs: the text model and its role prefixes, token budgets, the
language and category vocabularies, and the bucket edges for duration and
publication age. It is serialized next to every model artifact, and the
featurizer is built *from* it, so the training and serving paths cannot
drift apart without changing the contract's fingerprint.

Index conventions (stable across versions of the same contract):

- language / duration / age id ``0`` means unknown or missing;
- category features are a multi-hot vector over ``categories`` (unknown
  categories are dropped rather than mapped to a bucket);
- intent features are ``0/1`` floats in the order of ``intent_features``.
"""

from __future__ import annotations

import bisect
import dataclasses
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from ml.embeddings.text import MODEL_ID

FILENAME = "feature_contract.json"

_LANGUAGE_ALIASES = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "portuguese": "pt",
    "italian": "it",
    "dutch": "nl",
}
_WS_RE = re.compile(r"\s+")


def normalize_language(value: object) -> str:
    """Primary subtag of a BCP-47-ish feed language ("en-US" -> "en")."""
    if not isinstance(value, str):
        return ""
    primary = value.strip().lower().split("-")[0].split("_")[0]
    return _LANGUAGE_ALIASES.get(primary, primary)


def normalize_category(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return _WS_RE.sub(" ", value.strip().lower())


@dataclasses.dataclass(frozen=True)
class FeatureContract:
    """Everything the featurizer needs; see the module docstring."""

    languages: tuple[str, ...]
    categories: tuple[str, ...]
    version: str = "1"
    text_model_id: str = MODEL_ID
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    max_query_tokens: int = 64
    max_metadata_tokens: int = 256
    max_transcript_tokens: int = 256
    max_description_chars: int = 1000
    max_transcript_chars: int = 2000
    # Upper edges in minutes: bucket i covers (edge[i-1], edge[i]]; the last
    # bucket is open-ended. Bucket 0 is reserved for unknown durations.
    duration_bucket_minutes: tuple[float, ...] = (5, 15, 30, 45, 60, 90, 120)
    # Upper edges in days since publication, relative to a caller-supplied
    # reference time (example event time in training, request time online).
    age_bucket_days: tuple[float, ...] = (7, 30, 90, 365, 3 * 365)
    intent_features: tuple[str, ...] = (
        "has_language", "has_max_duration", "has_published_after", "no_explicit",
    )

    def __post_init__(self) -> None:
        for name in ("languages", "categories"):
            values = getattr(self, name)
            if len(set(values)) != len(values) or any(not v for v in values):
                raise ValueError(f"{name} must be unique and non-empty strings")
        for name in ("duration_bucket_minutes", "age_bucket_days"):
            edges = getattr(self, name)
            if list(edges) != sorted(edges) or len(set(edges)) != len(edges):
                raise ValueError(f"{name} edges must be strictly increasing")

    # -- vocab lookups ------------------------------------------------------

    @property
    def num_languages(self) -> int:
        return len(self.languages) + 1  # + unknown

    @property
    def num_categories(self) -> int:
        return len(self.categories)

    @property
    def num_duration_buckets(self) -> int:
        return len(self.duration_bucket_minutes) + 2  # unknown + open-ended

    @property
    def num_age_buckets(self) -> int:
        return len(self.age_bucket_days) + 2

    def language_id(self, value: object) -> int:
        code = normalize_language(value)
        try:
            return self.languages.index(code) + 1
        except ValueError:
            return 0

    def category_ids(self, values: Iterable[object] | None) -> list[int]:
        if values is None or isinstance(values, float | str):
            return []
        index = {c: i for i, c in enumerate(self.categories)}
        ids = {index[c] for c in (normalize_category(v) for v in values) if c in index}
        return sorted(ids)

    def duration_bucket(self, seconds: object) -> int:
        minutes = _as_float(seconds)
        if minutes is None or minutes < 0:
            return 0
        return bisect.bisect_left(self.duration_bucket_minutes, minutes / 60.0) + 1

    def age_bucket(self, age_days: object) -> int:
        days = _as_float(age_days)
        if days is None:
            return 0
        return bisect.bisect_left(self.age_bucket_days, max(days, 0.0)) + 1

    # -- construction and serialization ------------------------------------

    @classmethod
    def from_catalog(
        cls,
        episode_languages: Iterable[object],
        podcast_categories: Iterable[Iterable[object] | None],
        min_language_count: int = 100,
        min_category_count: int = 3,
        **overrides: object,
    ) -> FeatureContract:
        """Build vocabularies from a catalog snapshot. Ordered by frequency
        then name so the same catalog always yields the same contract."""
        lang_counts = Counter(
            code for code in map(normalize_language, episode_languages) if code
        )
        languages = tuple(
            code for code, n in sorted(lang_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if n >= min_language_count
        )
        cat_counts: Counter[str] = Counter()
        for cats in podcast_categories:
            if cats is None or isinstance(cats, float | str):
                continue
            cat_counts.update({normalize_category(c) for c in cats} - {""})
        categories = tuple(
            name for name, n in sorted(cat_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if n >= min_category_count
        )
        return cls(languages=languages, categories=categories, **overrides)  # type: ignore[arg-type]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> FeatureContract:
        raw = dict(raw)
        for name in ("languages", "categories", "duration_bucket_minutes",
                     "age_bucket_days", "intent_features"):
            if name in raw:
                raw[name] = tuple(raw[name])
        return cls(**raw)

    def save(self, directory: str | Path) -> Path:
        path = Path(directory) / FILENAME
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> FeatureContract:
        return cls.from_dict(json.loads((Path(directory) / FILENAME).read_text()))

    def fingerprint(self) -> str:
        """Content hash; any change to the contract changes it."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def _as_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if number != number else number  # NaN check
