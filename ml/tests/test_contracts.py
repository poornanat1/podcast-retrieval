"""Contract tests: the schema accepts well-formed examples and rejects each
invariant violation individually."""

import pandas as pd
import pandera.pandas as pa
import pytest

from ml.datasets.contracts import COLUMNS, LabelSource, TrainingExampleModel


def make_examples(n: int = 4) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "example_id": f"ex-{i}",
                "event_time": pd.Timestamp("2026-08-01T12:00:00Z") + pd.Timedelta(hours=i),
                "query_text": f"practical ai episode {i}",
                "query_audio_variant": "",
                "user_history": [100 + i, 200 + i],
                "positive_episode_id": 1000 + i,
                "negative_episode_ids": [2000 + i, 3000 + i],
                "label_source": LabelSource.SEARCH_PLAY.value,
                "label_strength": 0.8,
                "dataset_version": "2026-08-26.1",
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def test_valid_examples_pass() -> None:
    validated = TrainingExampleModel.validate(make_examples(), lazy=True)
    assert len(validated) == 4


def expect_failure(df: pd.DataFrame, check_fragment: str) -> None:
    with pytest.raises(pa.errors.SchemaErrors) as excinfo:
        TrainingExampleModel.validate(df, lazy=True)
    checks = " ".join(str(c) for c in excinfo.value.failure_cases["check"])
    assert check_fragment in checks, f"{check_fragment!r} not in failures: {checks}"


def test_duplicate_example_id_rejected() -> None:
    df = make_examples()
    df.loc[1, "example_id"] = df.loc[0, "example_id"]
    expect_failure(df, "field_uniqueness")


def test_positive_in_negatives_rejected() -> None:
    df = make_examples()
    df.at[2, "negative_episode_ids"] = [int(df.at[2, "positive_episode_id"])]
    expect_failure(df, "positive_not_in_negatives")


def test_positive_in_history_rejected() -> None:
    df = make_examples()
    df.at[1, "user_history"] = [int(df.at[1, "positive_episode_id"])]
    expect_failure(df, "positive_not_in_history")


def test_unknown_label_source_rejected() -> None:
    df = make_examples()
    df.loc[0, "label_source"] = "vibes"
    expect_failure(df, "isin")


def test_label_strength_out_of_range_rejected() -> None:
    df = make_examples()
    df.loc[0, "label_strength"] = 1.5
    expect_failure(df, "less_than_or_equal_to")


def test_future_event_time_rejected() -> None:
    df = make_examples()
    df.loc[0, "event_time"] = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=30)
    expect_failure(df, "event_time_not_in_future")


def test_malformed_history_rejected() -> None:
    df = make_examples()
    df.at[0, "user_history"] = ["not-an-id"]
    expect_failure(df, "history_is_episode_id_list")


def test_extra_column_rejected() -> None:
    df = make_examples()
    df["surprise"] = 1
    with pytest.raises(pa.errors.SchemaErrors):
        TrainingExampleModel.validate(df, lazy=True)
