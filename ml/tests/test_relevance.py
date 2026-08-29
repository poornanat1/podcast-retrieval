"""Relevance-set tests: query generation determinism, judgment contract,
and qrels export."""

import json

import pandas as pd
import pandera.pandas as pa
import pytest

from ml.relevance.contracts import JudgmentModel
from ml.relevance.queries import build_queries, query_id, write_jsonl
from ml.tests.test_build import write_snapshot


def snapshot_with_podcasts(tmp_path):
    root = write_snapshot(tmp_path)
    podcasts = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "title": ["The Daily Brief", "Hard History", "Tech Things",
                      "Mystery Hour", "Le Journal"],
            "publisher": ["Example Media", "Jane Smith", "John Doe",
                          "audiochuck", "Radio France"],
            "language": ["en", "en", "en", "en", "fr"],
            "categories": [["News"], ["History"], ["Technology"],
                           ["True Crime"], ["News"]],
            "explicit": [False] * 5,
        }
    )
    podcasts.to_parquet(root / "podcasts.parquet", index=False)
    return root


def test_query_build_is_deterministic(tmp_path) -> None:
    snap = snapshot_with_podcasts(tmp_path)
    first = build_queries(snap)
    second = build_queries(snap)
    pd.testing.assert_frame_equal(first, second)

    out = tmp_path / "queries.jsonl"
    write_jsonl(first, out)
    lines = out.read_text().splitlines()
    assert len(lines) == len(first)
    assert all(json.loads(line)["query_id"].startswith("q") for line in lines)


def test_query_set_composition(tmp_path) -> None:
    snap = snapshot_with_podcasts(tmp_path)
    df = build_queries(snap)
    by_type = df["query_type"].value_counts()

    # All four English shows are navigational targets; the French one is not.
    assert by_type["navigational"] == 4
    assert "le journal" not in set(df["query_text"])
    # Person-like publishers become entity queries; org-like ones do not.
    entities = set(df[df["query_type"] == "entity"]["query_text"])
    assert entities == {"jane smith", "john doe"}
    # Curated sets are present in full.
    assert by_type["exploratory"] >= 80
    assert by_type["filtered"] >= 30
    # Filtered queries carry their structured constraints.
    filtered = df[df["query_type"] == "filtered"]
    assert filtered["max_duration_seconds"].notna().any()
    assert (filtered[filtered["language"] == "fr"]["query_text"].str.len() > 0).any()
    assert df["query_id"].is_unique


def test_query_ids_are_stable() -> None:
    assert query_id("history of the roman empire", "exploratory") == query_id(
        "history of the roman empire", "exploratory"
    )
    assert query_id("a", "exploratory") != query_id("a", "navigational")


def make_judgments(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [f"q{i:08x}" for i in range(n)],
            "episode_id": [100 + i for i in range(n)],
            "grade": [i % 4 for i in range(n)],
            "judge": ["tester"] * n,
            "judged_at": ["2026-08-29T00:00:00+00:00"] * n,
            "notes": [""] * n,
        }
    )


def test_judgment_contract_accepts_valid() -> None:
    validated = JudgmentModel.validate(make_judgments())
    assert len(validated) == 3


def test_judgment_contract_rejects_duplicates_and_bad_grades() -> None:
    dup = make_judgments()
    dup.loc[1, ["query_id", "episode_id"]] = dup.loc[0, ["query_id", "episode_id"]]
    with pytest.raises(pa.errors.SchemaError):
        JudgmentModel.validate(dup)

    bad = make_judgments()
    bad.loc[0, "grade"] = 5
    with pytest.raises(pa.errors.SchemaError):
        JudgmentModel.validate(bad)


def test_export_produces_qrels(tmp_path) -> None:
    from ml.relevance import export

    snap = snapshot_with_podcasts(tmp_path)
    queries = build_queries(snap)
    queries_path = tmp_path / "queries.jsonl"
    write_jsonl(queries, queries_path)

    judgments = make_judgments(2)
    judgments["query_id"] = list(queries["query_id"][:2])
    judgments["grade"] = [3, 0]
    judgments_path = tmp_path / "judgments.jsonl"
    judgments.to_json(judgments_path, orient="records", lines=True)

    out = tmp_path / "qrels.txt"
    code = export.main([
        "--queries", str(queries_path),
        "--judgments", str(judgments_path),
        "--out", str(out),
    ])
    assert code == 0
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    qid, zero, episode, grade = lines[0].split()
    assert zero == "0" and grade in {"0", "3"} and int(episode) > 0

    # Judgments referencing unknown queries are refused.
    judgments.loc[0, "query_id"] = "qdeadbeef"
    judgments.to_json(judgments_path, orient="records", lines=True)
    with pytest.raises(SystemExit):
        export.main([
            "--queries", str(queries_path),
            "--judgments", str(judgments_path),
            "--out", str(out),
        ])
