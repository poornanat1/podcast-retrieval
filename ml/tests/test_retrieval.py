"""Retrieval-system tests that need no database."""

import pytest

from ml.retrieval import build_system
from ml.retrieval.popularity import match_categories, tokens


def test_match_categories_requires_all_category_words() -> None:
    categories = ["True Crime", "Crime", "History", "Science", "Society & Culture"]
    assert match_categories("clean true crime for road trips", categories) == [
        "True Crime", "Crime",
    ]
    assert match_categories("history of the roman empire", categories) == ["History"]
    assert match_categories("practical uses of artificial intelligence", categories) == []
    # Partial category matches do not count.
    assert match_categories("society today", categories) == []
    assert tokens("Society & Culture!") == ["society", "culture"]


def test_unknown_system_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown retrieval system"):
        build_system("does-not-exist", conn=None)


def test_rrf_merge_math_and_determinism() -> None:
    from ml.retrieval.hybrid import rrf_merge

    # Item 3 appears at rank 1 in both lists and must win; 1 and 2 swap
    # ranks across lists and tie on score, breaking toward the smaller id.
    merged = rrf_merge([[3, 1, 2], [3, 2, 1]], constant=60)
    assert merged[0] == 3
    assert merged[1:] == [1, 2]

    # Weights shift the balance: heavily weighting the second list puts its
    # top item first.
    merged = rrf_merge([[1, 2], [2, 1]], weights=[1.0, 3.0], constant=60)
    assert merged[0] == 2

    # An item present in only one list scores lower than one in both, even
    # at a worse single rank.
    merged = rrf_merge([[9, 5], [5]], constant=60)
    assert merged[0] == 5


def test_e5_text_preparation() -> None:
    from ml.embeddings.text import build_passage, build_query

    passage = build_passage("The Daily", "A Big Story", "What happened today.")
    assert passage == "passage: The Daily. A Big Story. What happened today."
    assert build_passage("", "", "") == ""
    # Long descriptions are bounded.
    long_passage = build_passage("P", "E", "x" * 5000)
    assert len(long_passage) < 1100
    assert build_query("  practical ai  ") == "query: practical ai"


def test_registry_passes_system_options() -> None:
    # Options reach the constructor: a learned system without its artifact
    # fails at artifact loading, and unknown options are rejected.
    with pytest.raises(FileNotFoundError):
        build_system("two-tower", conn=None, options={"artifact": "/nonexistent/artifact"})
    with pytest.raises(TypeError):
        build_system("lexical-fts", conn=None, options={"bogus": 1})


def test_two_tower_index_naming() -> None:
    from ml.embeddings.embed import index_name
    from ml.embeddings.two_tower_index import model_id_for_artifact, rows_to_episodes
    from ml.retrieval.two_tower import TwoTowerSearch

    assert TwoTowerSearch.name == "two-tower"  # registry key; instances add the artifact

    model_id = model_id_for_artifact("data/models/two-tower-bootstrap-1.1.0-abc-0123beef")
    assert model_id == "two-tower:two-tower-bootstrap-1.1.0-abc-0123beef"
    assert index_name(model_id) == (
        "episode_embeddings_hnsw_two_tower_two_tower_bootstrap_1_1_0_abc_0123beef"
    )
    episodes = rows_to_episodes([
        (1, 7, "Ep", "desc", "en", 60, None, "h", "Show", "Pub", ["News"], ""),
    ])
    assert episodes[0]["podcast_title"] == "Show" and episodes[0]["categories"] == ["News"]


def test_shared_sql_comments_hold_no_placeholders() -> None:
    """psycopg parses %(name)s placeholders even inside SQL comments, so a
    comment that shows the syntax breaks the statement."""
    from pathlib import Path

    for path in Path("ml/sql").glob("*.sql"):
        for line in path.read_text().splitlines():
            if line.lstrip().startswith("--"):
                assert "%" not in line, f"{path.name}: {line}"


def test_shared_sql_placeholders_match_python_params() -> None:
    import re

    from ml.retrieval.lexical import SEARCH_SQL as lexical_sql
    from ml.retrieval.lexical import query_params
    from ml.retrieval.vector import SEARCH_SQL as vector_sql
    from ml.retrieval.vector import filter_params

    def names(sql):
        return set(re.findall(r"%\((\w+)\)s", sql))

    query = {"query_text": "x", "language": "es", "max_duration_seconds": 600}
    assert names(lexical_sql) == set(query_params(query, 5))
    assert names(vector_sql) == set(filter_params(query)) | {"model", "qvec", "top_k"}
