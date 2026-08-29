"""Typed contracts for relevance queries and graded judgments."""

from __future__ import annotations

import enum

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series


class QueryType(enum.StrEnum):
    NAVIGATIONAL = "navigational"  # find a specific show
    ENTITY = "entity"  # host, guest, or publisher name
    EXPLORATORY = "exploratory"  # topical, open-ended
    FILTERED = "filtered"  # exploratory plus structured constraints


#: Graded relevance scale (standard 4-point).
GRADES = {
    0: "irrelevant",
    1: "marginal: topic adjacent, would not satisfy the query",
    2: "relevant: satisfies the query",
    3: "highly relevant: near-perfect answer for the query",
}


class QueryModel(pa.DataFrameModel):
    query_id: Series[str] = pa.Field(unique=True, str_matches=r"^q[0-9a-f]{8}$")
    query_text: Series[str] = pa.Field(str_length={"min_value": 3})
    query_type: Series[str] = pa.Field(isin=[t.value for t in QueryType])
    language: Series[str] = pa.Field(nullable=True)
    max_duration_seconds: Series[pd.Int64Dtype] = pa.Field(nullable=True, coerce=True)
    published_after: Series[str] = pa.Field(nullable=True)
    no_explicit: Series[bool] = pa.Field(coerce=True)
    source: Series[str] = pa.Field(isin=["generated", "curated"])

    class Config:
        strict = True


class JudgmentModel(pa.DataFrameModel):
    query_id: Series[str] = pa.Field(str_matches=r"^q[0-9a-f]{8}$")
    episode_id: Series[int] = pa.Field(gt=0, coerce=True)
    grade: Series[int] = pa.Field(ge=0, le=3, coerce=True)
    judge: Series[str] = pa.Field(str_length={"min_value": 1})
    judged_at: Series[str] = pa.Field(str_length={"min_value": 4})
    notes: Series[str] = pa.Field(nullable=True)

    class Config:
        strict = True

    @pa.dataframe_check(name="one_judgment_per_pair_and_judge")
    def _unique_pairs(cls, df: pd.DataFrame) -> bool:
        # A human may re-grade a pair an LLM judged (human wins at export),
        # but the same judge grading the same pair twice is an error.
        return not df.duplicated(subset=["query_id", "episode_id", "judge"]).any()


def is_human_judge(judge: pd.Series) -> pd.Series:
    """Machine judges are namespaced ``llm:<model>``; everything else is a
    person."""
    return ~judge.str.startswith("llm:")
