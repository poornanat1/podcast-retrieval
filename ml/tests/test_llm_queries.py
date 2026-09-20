"""LLM query generation: seeded sampling and resumable, validated output."""

import json

from ml.datasets.llm_queries import EpisodeQuery, generate, sample_episodes
from ml.tests.test_build import write_snapshot


def test_sample_is_seeded_and_filters_thin_descriptions(tmp_path) -> None:
    root = write_snapshot(tmp_path, n_episodes=40)
    # The fixture's descriptions are one character: nothing qualifies.
    assert len(sample_episodes(root, 10, seed=1)) == 0
    import pandas as pd

    episodes = pd.read_parquet(root / "episodes.parquet")
    episodes["description"] = "a long enough description " * 5
    episodes.to_parquet(root / "episodes.parquet", index=False)
    first = sample_episodes(root, 10, seed=1)
    again = sample_episodes(root, 10, seed=1)
    assert list(first["id"]) == list(again["id"]) and len(first) == 10
    assert "podcast_title" in first.columns
    assert list(sample_episodes(root, 10, seed=2)["id"]) != list(first["id"])


def test_generate_resumes_and_drops_bad_ids(tmp_path) -> None:
    rows = [{"id": i, "language": "en", "podcast_title": "Show", "title": f"Ep {i}",
             "description": "d" * 100} for i in range(1, 6)]
    out = tmp_path / "queries.jsonl"
    calls = []

    def fake(batch):
        calls.append([r["id"] for r in batch])
        return [EpisodeQuery(episode_id=r["id"], query=f"how to {r['id']}") for r in batch] + [
            EpisodeQuery(episode_id=999, query="hallucinated"),
            EpisodeQuery(episode_id=batch[0]["id"], query="duplicate"),
        ]

    stats = generate(rows[:3], out, fake, "test-model", batch_size=2)
    assert stats == {"episodes": 3, "already_done": 0, "written": 3, "failed_batches": 0}
    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert [w["episode_id"] for w in written] == [1, 2, 3]
    assert written[0]["query_text"] == "how to 1" and written[0]["model"] == "test-model"

    stats = generate(rows, out, fake, "test-model", batch_size=2)  # resumes past 1-3
    assert stats["already_done"] == 3 and stats["written"] == 2
    assert calls[-1] == [4, 5]
