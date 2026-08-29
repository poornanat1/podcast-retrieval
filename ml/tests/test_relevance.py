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


def test_llm_judge_appends_and_resumes(tmp_path) -> None:
    from ml.relevance.llm_judge import EpisodeGrade, judge

    tasks = [
        {"query_id": "q00000001", "query_text": "a", "query_type": "exploratory",
         "episode_id": 10, "podcast_title": "P", "episode_title": "E10",
         "description": "", "rank": 1},
        {"query_id": "q00000001", "query_text": "a", "query_type": "exploratory",
         "episode_id": 11, "podcast_title": "P", "episode_title": "E11",
         "description": "", "rank": 2},
        {"query_id": "q00000002", "query_text": "b", "query_type": "navigational",
         "episode_id": 20, "podcast_title": "P2", "episode_title": "E20",
         "description": "", "rank": 1},
    ]
    calls: list[list[int]] = []

    def fake_grader(batch):
        calls.append([t["episode_id"] for t in batch])
        return [EpisodeGrade(episode_id=t["episode_id"], grade=2, rationale="ok")
                for t in batch] + [EpisodeGrade(episode_id=999, grade=3, rationale="hallucinated")]

    path = tmp_path / "judgments.jsonl"
    stats = judge(tasks, path, fake_grader, "llm:test-model")
    assert stats == {"queries": 2, "judged": 3, "missing": 0, "failed_queries": 0}

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert {r["episode_id"] for r in rows} == {10, 11, 20}  # 999 dropped
    assert all(r["judge"] == "llm:test-model" for r in rows)

    # Second run: everything already judged, no grader calls.
    calls.clear()
    stats = judge(tasks, path, fake_grader, "llm:test-model")
    assert stats["queries"] == 0 and calls == []

    # A failing query is recorded but does not abort the run.
    more = [dict(tasks[0], query_id="q00000003", episode_id=30)]

    def exploding_grader(batch):
        raise RuntimeError("api down")

    stats = judge(more, path, exploding_grader, "llm:test-model")
    assert stats["failed_queries"] == 1 and stats["judged"] == 0


def test_export_prefers_human_over_llm(tmp_path) -> None:
    from ml.relevance import export

    snap = snapshot_with_podcasts(tmp_path)
    queries = build_queries(snap)
    queries_path = tmp_path / "queries.jsonl"
    write_jsonl(queries, queries_path)
    qid = queries["query_id"].iloc[0]

    rows = [
        {"query_id": qid, "episode_id": 100, "grade": 3, "judge": "llm:test-model",
         "judged_at": "2026-08-29T00:00:00+00:00", "notes": ""},
        {"query_id": qid, "episode_id": 101, "grade": 2, "judge": "llm:test-model",
         "judged_at": "2026-08-29T00:00:00+00:00", "notes": ""},
        # Human disagrees with the LLM on episode 100.
        {"query_id": qid, "episode_id": 100, "grade": 0, "judge": "poorna",
         "judged_at": "2026-08-29T01:00:00+00:00", "notes": "off topic"},
    ]
    judgments_path = tmp_path / "judgments.jsonl"
    judgments_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    out = tmp_path / "qrels.txt"
    code = export.main([
        "--queries", str(queries_path),
        "--judgments", str(judgments_path),
        "--out", str(out),
    ])
    assert code == 0
    qrels = dict()
    for line in out.read_text().splitlines():
        q, _, episode, grade = line.split()
        qrels[(q, int(episode))] = int(grade)
    assert qrels[(qid, 100)] == 0  # human override wins
    assert qrels[(qid, 101)] == 2
    assert len(qrels) == 2


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
