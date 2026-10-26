import pytest

from ml.speech.wer import (
    character_error_rate,
    edit_ops,
    entity_error_rate,
    entity_matches,
    normalize,
    word_errors,
)


def test_normalization_ignores_case_punctuation_and_accents() -> None:
    assert normalize("The Glenn Beck Program.") == ["the", "glenn", "beck", "program"]
    assert normalize("Año nuevo — ¿podcasts?") == ["ano", "nuevo", "podcasts"]
    assert normalize("I'm looking") == ["i", "m", "looking"]


def test_edit_ops_counts_each_kind() -> None:
    assert edit_ops(list("abc"), list("abc")) == (0, 0, 0)
    assert edit_ops(["a", "b", "c"], ["a", "x", "c"]) == (1, 0, 0)
    assert edit_ops(["a", "b", "c"], ["a", "c"]) == (0, 1, 0)
    assert edit_ops(["a", "b", "c"], ["a", "b", "y", "c"]) == (0, 0, 1)
    assert edit_ops([], ["a"]) == (0, 0, 1)
    assert edit_ops(["a"], []) == (0, 1, 0)


def test_word_error_rate() -> None:
    w = word_errors("the glenn beck program", "The Glenn Back Program.")
    assert (w.words, w.substitutions, w.deletions, w.insertions) == (4, 1, 0, 0)
    assert w.wer == pytest.approx(0.25)
    assert word_errors("joe rogan", "joe rogan").wer == 0.0
    assert word_errors("joe rogan", "").wer == 1.0
    assert word_errors("mountaineering disasters on everest",
                       "mountaineering disasters on the everest").wer == pytest.approx(0.25)


def test_entity_matching_reports_near_misses() -> None:
    m = entity_matches(["glenn beck"], "The Glenn Beck Program.")
    assert m[0].found and m[0].cer == 0.0
    m = entity_matches(["glenn beck"], "the glenn back program")
    assert not m[0].found and m[0].closest == "glenn back" and 0 < m[0].cer < 0.2
    m = entity_matches(["dejan kovacevic"], "the one about football")
    assert not m[0].found and m[0].cer > 0.5
    assert entity_matches(["x"], "")[0].found is False
    assert entity_error_rate([]) is None
    assert entity_error_rate(entity_matches(["joe rogan", "sky sports"], "joe rogan on sky")) == 0.5
    assert character_error_rate("beck", "back") == pytest.approx(0.25)
