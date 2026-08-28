"""Validation-stage tests: distribution reporting and threshold enforcement."""

from ml.datasets.contracts import LabelSource
from ml.datasets.validate import (
    ValidationConfig,
    load_examples,
    validate_examples,
)
from ml.tests.test_contracts import make_examples


def relaxed(**overrides) -> ValidationConfig:
    base = {"min_examples": 1, "min_distinct_positives": 1}
    base.update(overrides)
    return ValidationConfig(**base)


def test_valid_dataset_passes_and_reports() -> None:
    df = make_examples(6)
    df.loc[0, "label_source"] = LabelSource.HUMAN_JUDGMENT.value
    df.loc[1, "label_source"] = LabelSource.TOPIC_SIMILARITY.value
    df.loc[2, "label_source"] = LabelSource.SYNTHETIC_QUERY.value

    result = validate_examples(df, relaxed())
    assert result.ok, result.summary()
    report = result.report
    assert report is not None
    assert report.examples == 6
    # real: 3 search_play, human: 1, weak: 1, synthetic: 1
    assert report.class_shares == {
        "real": 0.5,
        "human": round(1 / 6, 4),
        "weak": round(1 / 6, 4),
        "synthetic": round(1 / 6, 4),
    }
    assert report.source_counts[LabelSource.SEARCH_PLAY.value] == 3
    assert report.negatives_min == 2 and report.negatives_max == 2


def test_schema_failure_skips_distribution_stage() -> None:
    df = make_examples()
    df.loc[0, "label_strength"] = 7.0
    result = validate_examples(df, relaxed())
    assert not result.ok
    assert result.schema_errors and result.report is None


def test_weak_share_threshold_enforced() -> None:
    df = make_examples(4)
    df.loc[:1, "label_source"] = LabelSource.TOPIC_SIMILARITY.value
    result = validate_examples(df, relaxed(max_weak_share=0.25))
    assert not result.ok
    assert any("weak share" in f for f in result.distribution_failures)


def test_min_examples_threshold_enforced() -> None:
    result = validate_examples(make_examples(3), ValidationConfig(min_examples=100))
    assert not result.ok
    assert any("min_examples" in f for f in result.distribution_failures)


def test_min_negatives_threshold_enforced() -> None:
    df = make_examples(3)
    df.at[0, "negative_episode_ids"] = []
    result = validate_examples(df, relaxed(min_negatives_per_example=1))
    assert not result.ok
    assert any("negatives per example" in f for f in result.distribution_failures)


def test_load_examples_round_trips_jsonl(tmp_path) -> None:
    df = make_examples(3)
    path = tmp_path / "examples.jsonl"
    out = df.copy()
    out["event_time"] = out["event_time"].map(lambda t: t.isoformat())
    out.to_json(path, orient="records", lines=True)

    loaded = load_examples(path)
    result = validate_examples(loaded, relaxed())
    assert result.ok, result.summary()


def test_load_examples_round_trips_parquet(tmp_path) -> None:
    df = make_examples(3)
    path = tmp_path / "examples.parquet"
    df.to_parquet(path)

    loaded = load_examples(path)
    # Parquet deserializes the id lists as numpy arrays; normalize to lists
    # the way the dataset builder will.
    for column in ("user_history", "negative_episode_ids"):
        loaded[column] = loaded[column].map(lambda v: [int(x) for x in v])
    result = validate_examples(loaded, relaxed())
    assert result.ok, result.summary()
