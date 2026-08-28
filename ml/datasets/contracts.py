"""Typed contract for normalized training examples.

Every dataset (search retrieval, personalized retrieval, similar episodes)
serializes to this one row shape. The contract is the single source of truth
for field names, types, and invariants; the human-readable copy lives in
``data/contracts/training-example.md`` and must be kept in sync.
"""

from __future__ import annotations

import enum

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series


class LabelSource(enum.StrEnum):
    """Where a (query, positive) pair's label came from."""

    SEARCH_PLAY = "search_play"  # search followed by playback
    HIGH_COMPLETION = "high_completion"  # high completion rate
    LIKE = "like"
    SAVE = "save"
    IMPRESSION_NO_PLAY = "impression_no_play"  # weak negative signal
    EARLY_SKIP = "early_skip"  # negative signal
    TOPIC_SIMILARITY = "topic_similarity"  # weak positive
    RANDOM_NEGATIVE = "random_negative"  # easy negative from catalog
    HUMAN_JUDGMENT = "human_judgment"  # human-reviewed relevance set
    SYNTHETIC_QUERY = "synthetic_query"  # generated query for an episode


class LabelClass(enum.StrEnum):
    """Reporting buckets: datasets must break labels down by these."""

    REAL = "real"
    HUMAN = "human"
    WEAK = "weak"
    SYNTHETIC = "synthetic"


LABEL_CLASS: dict[LabelSource, LabelClass] = {
    LabelSource.SEARCH_PLAY: LabelClass.REAL,
    LabelSource.HIGH_COMPLETION: LabelClass.REAL,
    LabelSource.LIKE: LabelClass.REAL,
    LabelSource.SAVE: LabelClass.REAL,
    LabelSource.IMPRESSION_NO_PLAY: LabelClass.REAL,
    LabelSource.EARLY_SKIP: LabelClass.REAL,
    LabelSource.HUMAN_JUDGMENT: LabelClass.HUMAN,
    LabelSource.TOPIC_SIMILARITY: LabelClass.WEAK,
    LabelSource.RANDOM_NEGATIVE: LabelClass.WEAK,
    LabelSource.SYNTHETIC_QUERY: LabelClass.SYNTHETIC,
}

# Version tags are semver-ish: "1.0.0" or "2026-08-26.1".
DATASET_VERSION_PATTERN = r"^[0-9][0-9A-Za-z.\-]*$"


def _episode_id_list(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(x, (int, np.integer)) and not isinstance(x, bool) and x > 0 for x in value
    )


class TrainingExampleModel(pa.DataFrameModel):
    """Schema for one normalized training example row."""

    example_id: Series[str] = pa.Field(unique=True, str_length={"min_value": 1})
    event_time: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"unit": "ns", "tz": "UTC"}
    )
    # Empty for personalized-retrieval examples that have no typed query.
    query_text: Series[str] = pa.Field(nullable=True)
    # Identifier of the speech recording variant this example was built from;
    # empty for text-only examples.
    query_audio_variant: Series[str] = pa.Field(nullable=True)
    user_history: Series[object] = pa.Field()
    positive_episode_id: Series[np.int64] = pa.Field(gt=0)
    negative_episode_ids: Series[object] = pa.Field()
    label_source: Series[str] = pa.Field(isin=[s.value for s in LabelSource])
    label_strength: Series[float] = pa.Field(ge=0.0, le=1.0)
    dataset_version: Series[str] = pa.Field(str_matches=DATASET_VERSION_PATTERN)

    class Config:
        strict = True
        coerce = True

    @pa.check("user_history", name="history_is_episode_id_list", element_wise=True)
    def _history_ids(cls, value: object) -> bool:
        return _episode_id_list(value)

    @pa.check("negative_episode_ids", name="negatives_are_episode_id_list", element_wise=True)
    def _negative_ids(cls, value: object) -> bool:
        return _episode_id_list(value)

    @pa.dataframe_check(name="positive_not_in_negatives")
    def _positive_not_negative(cls, df: pd.DataFrame) -> Series[bool]:
        pairs = zip(df["positive_episode_id"], df["negative_episode_ids"], strict=True)
        return pd.Series([p not in set(negs) for p, negs in pairs], index=df.index)

    @pa.dataframe_check(name="positive_not_in_history")
    def _positive_not_in_history(cls, df: pd.DataFrame) -> Series[bool]:
        # The positive is the prediction target; having it in the history
        # would leak the answer into the input.
        pairs = zip(df["positive_episode_id"], df["user_history"], strict=True)
        return pd.Series([p not in set(h) for p, h in pairs], index=df.index)

    @pa.dataframe_check(name="event_time_not_in_future")
    def _event_time_bounded(cls, df: pd.DataFrame) -> Series[bool]:
        cutoff = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
        return df["event_time"] <= cutoff


#: Column order for serialized datasets.
COLUMNS: list[str] = list(TrainingExampleModel.to_schema().columns)
