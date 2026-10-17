import pandas as pd

from ml.datasets.name_queries import build


def _catalog():
    podcasts = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "title": ["The Daily", "Rest Is History", "Rest Is Football", "Tiny"],
        "publisher": ["The New York Times", "Goalhanger", "Goalhanger", "Ab"],
    })
    episodes = pd.DataFrame({
        "id": range(1, 41),
        "podcast_id": [1 + i % 4 for i in range(40)],
        "published_at": pd.date_range("2026-01-01", periods=40, tz="UTC"),
    })
    return episodes, podcasts


def test_held_out_names_drop_the_whole_show() -> None:
    episodes, podcasts = _catalog()
    rows = build(episodes, podcasts, held_out={"the new york times"})
    texts = {r["query_text"] for r in rows}
    assert "the daily" not in texts and "the new york times" not in texts
    assert {"rest is history", "rest is football", "goalhanger", "tiny"} <= texts


def test_publisher_queries_exclude_every_show_of_the_publisher() -> None:
    episodes, podcasts = _catalog()
    rows = build(episodes, podcasts, held_out=set(), per_show=3, per_publisher=2)
    goal = [r for r in rows if r["query_text"] == "goalhanger"]
    assert len(goal) == 4 and all(r["exclude_podcasts"] == [2, 3] for r in goal)
    assert not any(r["query_text"] == "ab" for r in rows)  # too short
    shows = [r for r in rows if r["query_kind"] == "show"]
    assert all(r["exclude_podcasts"] == [episodes.set_index("id").loc[r["episode_id"],
               "podcast_id"]] for r in shows)
    ids = [r["episode_id"] for r in rows]
    assert len(ids) == len(set(ids))  # every episode used at most once
    assert build(episodes, podcasts, held_out=set(), per_show=3, per_publisher=2) == rows
