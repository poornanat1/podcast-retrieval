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
