"""Turn query dicts and catalog rows into two-tower input tensors.

One ``Featurizer`` serves training, batch embedding, and online inference;
it is constructed from a ``FeatureContract`` plus the matching tokenizer
and nothing else, so every path computes features the same way.

Query rows use the relevance-set / API shape: ``query_text``, optional
``language``, ``max_duration_seconds``, ``published_after``,
``no_explicit``. Episode rows use catalog column names: ``podcast_title``,
``title``, ``description``, ``publisher``, ``categories``, ``language``,
``duration_seconds``, ``published_at``, optional ``transcript_excerpt`` and
``podcast_id``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import torch

from ml.features.contract import FeatureContract
from ml.features.text import build_metadata_text, build_query_text, build_transcript_text

TOKENIZER_DIR = "tokenizer"


def _to_datetime(value: object) -> datetime | None:
    if value is None or isinstance(value, float | bool) or value != value:  # NaN / NaT
        return None
    if isinstance(value, datetime):
        dt = value
    elif hasattr(value, "to_pydatetime"):  # pandas.Timestamp
        dt = value.to_pydatetime()
    elif isinstance(value, str) and value:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def age_days(published_at: object, reference_time: object) -> float | None:
    published, reference = _to_datetime(published_at), _to_datetime(reference_time)
    if published is None or reference is None:
        return None
    return (reference - published).total_seconds() / 86400.0


class Featurizer:
    def __init__(self, contract: FeatureContract, tokenizer) -> None:
        self.contract = contract
        self.tokenizer = tokenizer

    @classmethod
    def load(cls, directory: str | Path) -> Featurizer:
        """Rebuild from a saved model artifact (contract + tokenizer)."""
        from transformers import AutoTokenizer

        directory = Path(directory)
        return cls(
            FeatureContract.load(directory),
            AutoTokenizer.from_pretrained(directory / TOKENIZER_DIR),
        )

    @classmethod
    def from_contract(cls, contract: FeatureContract) -> Featurizer:
        """Tokenizer fetched from the contract's text model."""
        from transformers import AutoTokenizer

        return cls(contract, AutoTokenizer.from_pretrained(contract.text_model_id))

    def save_tokenizer(self, directory: str | Path) -> None:
        self.tokenizer.save_pretrained(Path(directory) / TOKENIZER_DIR)

    # -- tokenization --------------------------------------------------------

    def _tokenize(self, texts: list[str], max_tokens: int) -> dict[str, torch.Tensor]:
        # Empty strings still get [CLS]/[SEP]; callers mask them out.
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_tokens,
            return_tensors="pt",
        )
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    # -- queries ---------------------------------------------------------------

    def query_batch(self, queries: Sequence[dict]) -> dict[str, torch.Tensor]:
        contract = self.contract
        texts = [build_query_text(contract, q.get("query_text", "")) for q in queries]
        tokens = self._tokenize(texts, contract.max_query_tokens)
        intents = [
            [
                float(bool(q.get("language"))),
                float(bool(q.get("max_duration_seconds"))),
                float(bool(q.get("published_after"))),
                float(bool(q.get("no_explicit"))),
            ]
            for q in queries
        ]
        return {
            **tokens,
            "language_id": torch.tensor(
                [contract.language_id(q.get("language")) for q in queries], dtype=torch.long
            ),
            "intent": torch.tensor(intents, dtype=torch.float32).reshape(
                len(queries), len(contract.intent_features)
            ),
        }

    # -- episodes --------------------------------------------------------------

    def episode_batch(
        self, episodes: Sequence[dict], reference_time: object | Sequence[object] = None
    ) -> dict[str, torch.Tensor]:
        """``reference_time`` (one value or one per episode) anchors the
        publication-age feature; ``None`` means "now"."""
        contract = self.contract
        n = len(episodes)
        if reference_time is None:
            references: list[object] = [datetime.now(UTC)] * n
        elif isinstance(reference_time, str | datetime) or hasattr(reference_time, "to_pydatetime"):
            references = [reference_time] * n
        else:
            references = list(reference_time)
            if len(references) != n:
                raise ValueError("reference_time must be scalar or one per episode")

        meta_texts, transcript_texts = [], []
        has_transcript, language_ids, durations, ages, podcast_ids = [], [], [], [], []
        categories = torch.zeros(n, contract.num_categories, dtype=torch.float32)
        for i, (row, reference) in enumerate(zip(episodes, references, strict=True)):
            meta_texts.append(build_metadata_text(
                contract, row.get("podcast_title"), row.get("title"), row.get("description"),
                row.get("publisher", ""), row.get("categories"),
            ))
            transcript = build_transcript_text(contract, row.get("transcript_excerpt"))
            transcript_texts.append(transcript)
            has_transcript.append(1.0 if transcript else 0.0)
            language_ids.append(contract.language_id(row.get("language")))
            durations.append(contract.duration_bucket(row.get("duration_seconds")))
            ages.append(contract.age_bucket(age_days(row.get("published_at"), reference)))
            for c in contract.category_ids(row.get("categories")):
                categories[i, c] = 1.0
            podcast_ids.append(int(row.get("podcast_id") or 0))

        meta = self._tokenize(meta_texts, contract.max_metadata_tokens)
        transcript = self._tokenize(transcript_texts, contract.max_transcript_tokens)
        return {
            "meta_input_ids": meta["input_ids"],
            "meta_attention_mask": meta["attention_mask"],
            "transcript_input_ids": transcript["input_ids"],
            "transcript_attention_mask": transcript["attention_mask"],
            "has_transcript": torch.tensor(has_transcript, dtype=torch.float32),
            "language_id": torch.tensor(language_ids, dtype=torch.long),
            "categories": categories,
            "duration_bucket": torch.tensor(durations, dtype=torch.long),
            "age_bucket": torch.tensor(ages, dtype=torch.long),
            "podcast_id": torch.tensor(podcast_ids, dtype=torch.long),
        }


def to_device(
    batch: dict[str, torch.Tensor], device: torch.device | str
) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}
