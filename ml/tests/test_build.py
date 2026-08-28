"""Builder tests: determinism, time-based splits, and contract compliance."""

import json

import pandas as pd
import pytest

from ml.datasets.build import BuildConfig, build_dataset, normalize_query
from ml.datasets.validate import ValidationConfig, load_examples, validate_examples


def write_snapshot(tmp_path, n_episodes: int = 60):
    root = tmp_path / "snap"
    root.mkdir()
    episodes = pd.DataFrame(
        {
            "id": range(1, n_episodes + 1),
            "podcast_id": [1 + i % 5 for i in range(n_episodes)],
            "title": [f"Episode {i}: topic number {i}" for i in range(1, n_episodes + 1)],
            "description": ["d"] * n_episodes,
            "language": ["en"] * (n_episodes - 10) + ["fr"] * 10,
            "duration_seconds": [1800] * n_episodes,
            "published_at": pd.date_range(
                "2026-01-01", periods=n_episodes, freq="2D", tz="UTC"
            ),
            "explicit": [False] * n_episodes,
            "content_hash": ["h"] * n_episodes,
            "has_transcript": [False] * n_episodes,
        }
    )
    episodes.to_parquet(root / "episodes.parquet", index=False)
    (root / "manifest.json").write_text(
        json.dumps({"snapshot_id": "snaptest12345", "rows": {"episodes": n_episodes}})
    )
    return root


def make_config(**overrides) -> BuildConfig:
    base = dict(
        name="test-dataset",
        base_version="1.0.0",
        generator="synthetic_query",
        seed=42,
        validation_start="2026-03-01T00:00:00+00:00",
        test_start="2026-04-01T00:00:00+00:00",
        language_prefix="en",
        max_examples=1000,
        negatives_per_example=3,
        validation={"min_examples": 1, "min_distinct_positives": 1},
    )
    base.update(overrides)
    return BuildConfig(**base)


def test_identical_inputs_produce_identical_versions(tmp_path) -> None:
    snap = write_snapshot(tmp_path)
    _, first = build_dataset(make_config(), snap, tmp_path / "out1")
    _, second = build_dataset(make_config(), snap, tmp_path / "out2")

    assert first["dataset_version"] == second["dataset_version"]
    assert first["content_hash"] == second["content_hash"]
    for split in ("train", "validation", "test"):
        assert first["splits"][split]["sha256"] == second["splits"][split]["sha256"]


def test_seed_change_changes_version(tmp_path) -> None:
    snap = write_snapshot(tmp_path)
    _, first = build_dataset(make_config(), snap, tmp_path / "out1")
    _, second = build_dataset(make_config(seed=43), snap, tmp_path / "out2")
    assert first["content_hash"] != second["content_hash"]
    assert first["dataset_version"] != second["dataset_version"]


def test_time_splits_have_no_leakage(tmp_path) -> None:
    snap = write_snapshot(tmp_path)
    out, manifest = build_dataset(make_config(), snap, tmp_path / "out")

    train = pd.read_parquet(out / "train.parquet")
    validation = pd.read_parquet(out / "validation.parquet")
    test = pd.read_parquet(out / "test.parquet")

    assert len(train) and len(validation) and len(test)
    assert train["event_time"].max() < pd.Timestamp("2026-03-01T00:00:00Z")
    assert validation["event_time"].min() >= pd.Timestamp("2026-03-01T00:00:00Z")
    assert validation["event_time"].max() < pd.Timestamp("2026-04-01T00:00:00Z")
    assert test["event_time"].min() >= pd.Timestamp("2026-04-01T00:00:00Z")
    total = manifest["splits"]
    assert total["train"]["rows"] + total["validation"]["rows"] + total["test"]["rows"] == len(
        train
    ) + len(validation) + len(test)


def test_output_conforms_to_contract(tmp_path) -> None:
    snap = write_snapshot(tmp_path)
    out, manifest = build_dataset(make_config(), snap, tmp_path / "out")

    for split in ("train", "validation", "test"):
        df = load_examples(out / f"{split}.parquet")
        for column in ("user_history", "negative_episode_ids"):
            df[column] = df[column].map(lambda v: [int(x) for x in v])
        result = validate_examples(
            df, ValidationConfig(min_examples=1, min_distinct_positives=1)
        )
        assert result.ok, f"{split}: {result.summary()}"
        assert (df["dataset_version"] == manifest["dataset_version"]).all()


def test_language_filter_and_negatives(tmp_path) -> None:
    snap = write_snapshot(tmp_path)
    out, _ = build_dataset(make_config(), snap, tmp_path / "out")
    frames = [pd.read_parquet(out / f"{s}.parquet") for s in ("train", "validation", "test")]
    df = pd.concat(frames, ignore_index=True)

    # French episodes (ids 51..60) are excluded as positives.
    assert (df["positive_episode_id"] <= 50).all()
    for row in df.itertuples():
        negatives = [int(x) for x in row.negative_episode_ids]
        assert len(negatives) == 3
        assert row.positive_episode_id not in negatives
        assert len(set(negatives)) == 3


def test_invalid_split_order_rejected(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    bad = {
        "name": "x",
        "base_version": "1.0.0",
        "generator": "synthetic_query",
        "seed": 1,
        "validation_start": "2026-05-01T00:00:00+00:00",
        "test_start": "2026-04-01T00:00:00+00:00",
    }
    config_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="validation_start"):
        BuildConfig.load(config_path)


def test_normalize_query() -> None:
    assert normalize_query("Ep. 12 — AI & You!") == "ep 12 ai you"
    assert normalize_query("   ") == ""
