import json

import numpy as np
import pytest

from ml.speech import audio, degrade
from ml.speech.entities import annotate, entity_span, normalize
from ml.speech.testset import build, stratified_subset

RATE = 16_000


def speech_like(seconds: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(int(RATE * seconds)) / RATE
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    tone = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((180, 360, 720, 1440, 2880), 1))
    return (0.3 * envelope * tone + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def band_energy(samples: np.ndarray, lo: float, hi: float) -> float:
    spectrum = np.abs(np.fft.rfft(samples)) ** 2
    freqs = np.fft.rfftfreq(len(samples), 1 / RATE)
    return float(spectrum[(freqs >= lo) & (freqs < hi)].sum())


def test_noise_hits_requested_snr() -> None:
    clean = speech_like()
    for snr in (20.0, 5.0):
        noisy = degrade.white_noise(clean, snr, np.random.default_rng(1))
        measured = 20 * np.log10(degrade.rms(clean) / degrade.rms(noisy - clean))
        assert measured == pytest.approx(snr, abs=0.5)
        assert len(noisy) == len(clean) and noisy.dtype == np.float32


def test_telephone_removes_wideband_energy() -> None:
    clean = speech_like()
    narrow = degrade.telephone(clean, RATE)
    assert len(narrow) == len(clean)
    assert band_energy(narrow, 4000, 8000) < 0.01 * band_energy(clean, 4000, 8000)
    assert band_energy(narrow, 300, 3400) > 0.5 * band_energy(clean, 300, 3400)


def test_speed_changes_length_not_level() -> None:
    clean = speech_like(2.0)
    fast = degrade.speed(clean, 1.3, RATE)
    slow = degrade.speed(clean, 0.8, RATE)
    assert len(fast) == pytest.approx(len(clean) / 1.3, rel=0.02)
    assert len(slow) == pytest.approx(len(clean) / 0.8, rel=0.02)
    assert degrade.rms(fast) == pytest.approx(degrade.rms(clean), rel=0.05)


def test_every_condition_applies_and_is_deterministic() -> None:
    clean = speech_like()
    others = [speech_like(0.7), speech_like(1.3)]
    for condition in degrade.CONDITIONS:
        a = degrade.apply(condition, clean, RATE, np.random.default_rng(7), others)
        b = degrade.apply(condition, clean, RATE, np.random.default_rng(7), others)
        assert np.array_equal(a, b), condition
        assert len(a) == len(clean) and np.abs(a).max() <= 1.0, condition
    with pytest.raises(ValueError):
        degrade.apply("vinyl", clean, RATE, np.random.default_rng(0))


def test_entity_span_prefers_longest_catalog_match_or_whole_query() -> None:
    names = {" the joe rogan experience ", " today explained ", " sports talk ", " the economist ",
             " dk pittsburgh sports daily podcasts on steelers penguins pirates "}
    assert entity_span("the joe rogan experience", names) == "the joe rogan experience"
    assert entity_span("joe rogan", names) == "joe rogan"
    assert entity_span("Today, Explained!", names) == "today explained"
    assert entity_span("sports 180", names) == "sports 180"  # one word matched: whole query
    assert entity_span("economist podcasts", names) == "economist podcasts"
    assert entity_span("vince ferrara", names) == "vince ferrara"  # nothing matched
    assert normalize("Fr. Mike Schmitz — Año") == "fr mike schmitz ano"
    out = annotate([{"query_id": "a", "query_type": "entity", "query_text": "the economist"},
                    {"query_id": "b", "query_type": "exploratory", "query_text": "the economist"}],
                   names)
    assert out == {"a": ["the economist"], "b": []}


def _queries():
    rows = []
    for i in range(12):
        rows.append({"query_id": f"q{i}", "query_text": f"topic {i}",
                     "query_type": ["navigational", "entity", "exploratory", "filtered"][i % 4],
                     "language": "es" if i == 11 else None, "max_duration_seconds": None,
                     "published_after": "2026-01-01" if i % 4 == 3 else None, "no_explicit": False})
    return rows


def test_stratified_subset_is_seeded_and_english() -> None:
    spanish = {"q11": "es", "q7": "es"}
    a = stratified_subset(_queries(), {"exploratory": 2, "filtered": 1}, np.random.default_rng(3),
                          spanish)
    b = stratified_subset(_queries(), {"exploratory": 2, "filtered": 1}, np.random.default_rng(3),
                          spanish)
    assert [q["query_id"] for q in a] == [q["query_id"] for q in b] and len(a) == 3
    assert not {q["query_id"] for q in a} & spanish.keys()


def test_build_manifest_with_fake_synthesizer(tmp_path) -> None:
    relevance = tmp_path / "relevance"
    relevance.mkdir()
    (relevance / "queries.jsonl").write_text(
        "".join(json.dumps(q) + "\n" for q in _queries()))
    config = {
        "seed": 1, "sample_rate": 24_000, "relevance_dir": str(relevance),
        "out_dir": str(tmp_path / "set"),
        "voices": {"en_us": {"say": "A", "locale": "en_US", "language": "en"},
                   "en_gb": {"say": "B", "locale": "en_GB", "language": "en"},
                   "es": {"say": "C", "locale": "es_MX", "language": "es"}},
        "reference_text": {"en": "hello", "es": "hola"}, "base_voice": "en_us",
        "text_language": {"q11": "es"},
        "accents": ["en_gb"], "synthesis": {"exaggeration": 0.5, "cfg_weight": 0.5},
        "subset_per_type": {"exploratory": 2, "entity": 1},
        "conversational_per_type": {"filtered": 1},
        "conditions": ["noise_10db", "babble_10db", "telephone"], "speeds": {"fast": 1.3},
    }
    said = []

    def say(path, voice, text, rate):
        said.append((voice, text))
        audio.write(path, speech_like(0.5), RATE)

    def synthesizer(jobs_file):
        for line in jobs_file.read_text().splitlines():
            job = json.loads(line)
            assert job["reference"].endswith(f"{job['clip_id'].split('-')[-1]}.wav")
            audio.write(job["path"], np.resize(speech_like(1.0), 24_000), 24_000)  # 1.0 s

    rows = build(config, say=say, synthesizer=synthesizer, names={" topic 1 "})
    assert sorted(said) == [("A", "hello"), ("B", "hello"), ("C", "hola")]
    tiers = {}
    for r in rows:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    assert tiers == {"base": 12, "accent": 3, "conversational": 1, "condition": 9, "speed": 3}
    spanish = next(r for r in rows if r["query_id"] == "q11")
    assert spanish["voice"] == "es" and spanish["language"] == "es"
    conv = next(r for r in rows if r["tier"] == "conversational")
    assert conv["terse_text"] in conv["text"] and conv["text"] != conv["terse_text"]
    assert conv["filters"] == {"published_after": "2026-01-01"}
    entity = next(r for r in rows if r["query_id"] == "q1" and r["tier"] == "base")
    assert entity["entities"] == ["topic 1"]
    fast = next(r for r in rows if r["tier"] == "speed")
    assert fast["speed"] == 1.3 and fast["condition"] == "clean"
    assert fast["seconds"] == pytest.approx(1.0 / 1.3, rel=0.05)
    manifest = (tmp_path / "set" / "manifest.jsonl").read_text().splitlines()
    assert len(manifest) == len(rows) and all((tmp_path / "set" / r["path"]).exists() for r in rows)
    # Rebuilding reuses every file and gives the same manifest.
    again = build(config, say=say, synthesizer=synthesizer, names={" topic 1 "})
    assert again == rows
