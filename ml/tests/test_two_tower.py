"""Feature contract, featurizer, and two-tower model tests.

Everything runs offline: a tiny randomly-initialized BERT stands in for the
pretrained text model and a hand-written vocabulary for its tokenizer.
"""

import json
from datetime import UTC, datetime

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from ml.features.contract import FeatureContract, normalize_category, normalize_language
from ml.features.featurize import Featurizer, age_days
from ml.features.text import build_metadata_text
from ml.models.two_tower import TextEncoder, TwoTowerConfig, TwoTowerModel

VOCAB = (
    "[PAD] [UNK] [CLS] [SEP] [MASK] passage query : , . the a of and about show episode "
    "news science comedy history space rockets launch cooking bread transcript hello world "
    "publisher categories acme"
).split()


@pytest.fixture(scope="module")
def contract() -> FeatureContract:
    return FeatureContract(languages=("en", "es"), categories=("news", "science", "comedy"))


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    from transformers import BertTokenizerFast

    vocab_file = tmp_path_factory.mktemp("tok") / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB) + "\n")
    return BertTokenizerFast(vocab_file=str(vocab_file), model_max_length=128)


@pytest.fixture(scope="module")
def featurizer(contract, tokenizer) -> Featurizer:
    return Featurizer(contract, tokenizer)


def tiny_backbone():
    from transformers import BertConfig, BertModel

    return BertModel(BertConfig(
        vocab_size=len(VOCAB), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=128, pad_token_id=0,
    ))


def make_model(contract, **overrides) -> TwoTowerModel:
    torch.manual_seed(0)
    config = TwoTowerConfig(**{"embedding_dim": 32, "hidden_dim": 16, "dropout": 0.0, **overrides})
    return TwoTowerModel(config, contract, tiny_backbone()).eval()


EPISODES = [
    {
        "podcast_id": 7, "podcast_title": "Space Show", "title": "Rockets launch",
        "description": "About rockets.", "publisher": "Acme", "categories": ["Science", "Space"],
        "language": "en-US", "duration_seconds": 1800,
        "published_at": datetime(2026, 1, 1, tzinfo=UTC),
        "transcript_excerpt": "hello world transcript about rockets",
    },
    {
        "podcast_id": 9, "podcast_title": "Cooking", "title": "Bread",
        "description": "", "publisher": "", "categories": None,
        "language": "xx", "duration_seconds": None, "published_at": None,
    },
]
QUERIES = [
    {"query_text": "rockets", "language": "en", "max_duration_seconds": 2700,
     "published_after": "2026-01-01", "no_explicit": True},
    {"query_text": "bread"},
]


# -- contract ------------------------------------------------------------------


def test_normalizers() -> None:
    assert normalize_language("en-US") == "en"
    assert normalize_language("English") == "en"
    assert normalize_language(None) == ""
    assert normalize_category("  True   Crime ") == "true crime"


def test_contract_from_catalog_is_deterministic_and_thresholded() -> None:
    episodes = pd.Series(["en"] * 5 + ["es-ES"] * 3 + ["fr"] * 1 + [None])
    podcasts = pd.Series([["News", "Science"], ["news"], ["Comedy"], None, float("nan")])
    contract = FeatureContract.from_catalog(
        episodes, podcasts, min_language_count=2, min_category_count=2
    )
    assert contract.languages == ("en", "es")  # "fr" below threshold
    assert contract.categories == ("news",)
    again = FeatureContract.from_catalog(
        episodes, podcasts, min_language_count=2, min_category_count=2
    )
    assert again == contract and again.fingerprint() == contract.fingerprint()


def test_contract_roundtrip_and_fingerprint(contract, tmp_path) -> None:
    contract.save(tmp_path)
    loaded = FeatureContract.load(tmp_path)
    assert loaded == contract
    assert loaded.fingerprint() == contract.fingerprint()
    changed = FeatureContract.from_dict({**contract.to_dict(), "max_query_tokens": 32})
    assert changed.fingerprint() != contract.fingerprint()
    with pytest.raises(ValueError, match="strictly increasing"):
        FeatureContract(languages=("en",), categories=(), age_bucket_days=(30, 7))
    with pytest.raises(ValueError, match="unique"):
        FeatureContract(languages=("en", "en"), categories=())


def test_contract_lookups(contract) -> None:
    assert contract.language_id("en-GB") == 1
    assert contract.language_id("es") == 2
    assert contract.language_id("fr") == 0
    assert contract.language_id(None) == 0
    assert contract.category_ids(["News", " science ", "unknown"]) == [0, 1]
    assert contract.category_ids(None) == [] and contract.category_ids(float("nan")) == []

    assert contract.duration_bucket(None) == 0
    assert contract.duration_bucket(float("nan")) == 0
    assert contract.duration_bucket(0) == 1
    assert contract.duration_bucket(5 * 60) == 1  # upper edge inclusive
    assert contract.duration_bucket(5 * 60 + 1) == 2
    assert contract.duration_bucket(10 * 3600) == contract.num_duration_buckets - 1

    assert contract.age_bucket(None) == 0
    assert contract.age_bucket(-3) == 1  # future-dated counts as brand new
    assert contract.age_bucket(7) == 1
    assert contract.age_bucket(8) == 2
    assert contract.age_bucket(10_000) == contract.num_age_buckets - 1


# -- text and featurizer --------------------------------------------------------


def test_metadata_text_orders_short_fields_before_description(contract) -> None:
    text = build_metadata_text(
        contract, "Space Show", "Rockets", "x" * 5000, "Acme", ["Science", ""]
    )
    assert text.startswith("passage: Space Show. Rockets. Categories: Science. Publisher: Acme. x")
    assert len(text) < contract.max_description_chars + 100
    assert build_metadata_text(contract, "", "", "") == ""


def test_age_days() -> None:
    assert age_days(None, datetime(2026, 1, 8, tzinfo=UTC)) is None
    assert age_days("2026-01-01T00:00:00Z", datetime(2026, 1, 8, tzinfo=UTC)) == 7.0
    assert age_days(pd.Timestamp("2026-01-01", tz="UTC"), "2026-01-01T12:00:00+00:00") == 0.5
    assert age_days(pd.NaT, datetime.now(UTC)) is None


def test_query_batch(featurizer, contract) -> None:
    batch = featurizer.query_batch(QUERIES)
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["language_id"].tolist() == [1, 0]
    assert batch["intent"].shape == (2, len(contract.intent_features))
    assert batch["intent"].tolist() == [[1, 1, 1, 1], [0, 0, 0, 0]]


def test_episode_batch(featurizer, contract) -> None:
    reference = datetime(2026, 1, 31, tzinfo=UTC)  # 30 days after episode 0
    batch = featurizer.episode_batch(EPISODES, reference_time=reference)
    assert batch["meta_input_ids"].shape[0] == 2
    assert batch["has_transcript"].tolist() == [1.0, 0.0]
    assert batch["language_id"].tolist() == [1, 0]
    assert batch["categories"].tolist() == [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    assert batch["duration_bucket"].tolist() == [3, 0]  # 30 min -> (15, 30]
    assert batch["age_bucket"].tolist() == [2, 0]  # 30 days -> (7, 30]
    assert batch["podcast_id"].tolist() == [7, 9]

    per_episode = featurizer.episode_batch(EPISODES, reference_time=[reference, reference])
    assert torch.equal(per_episode["age_bucket"], batch["age_bucket"])
    now_batch = featurizer.episode_batch(EPISODES)  # reference = now
    assert now_batch["age_bucket"].tolist()[1] == 0
    with pytest.raises(ValueError, match="one per episode"):
        featurizer.episode_batch(EPISODES, reference_time=[reference])


# -- model ---------------------------------------------------------------------


def test_towers_emit_unit_vectors(featurizer, contract) -> None:
    model = make_model(contract)
    with torch.no_grad():
        q = model.encode_query(featurizer.query_batch(QUERIES))
        e = model.encode_episode(featurizer.episode_batch(EPISODES))
    assert q.shape == (2, 32) and e.shape == (2, 32)
    assert torch.allclose(q.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(e.norm(dim=-1), torch.ones(2), atol=1e-5)
    sim = model.similarity(q, e)
    assert sim.shape == (2, 2)
    assert torch.allclose(model.logits(q, e), sim / model.config.temperature)
    assert torch.allclose(model.score(q, e), sim.diagonal())
    assert torch.allclose(model(featurizer.query_batch(QUERIES),
                                featurizer.episode_batch(EPISODES)), model.logits(q, e))


def test_untrained_model_equals_pretrained_text_embedding(featurizer, contract) -> None:
    """Zero-initialized heads: before training, both towers are exactly the
    mean-pooled, normalized text encoder — the pretrained-embedding baseline."""
    model = make_model(contract)
    encoder = TextEncoder(model.query_tower.encoder.backbone)
    qb, eb = featurizer.query_batch(QUERIES), featurizer.episode_batch(EPISODES)
    with torch.no_grad():
        expected_q = F.normalize(encoder(qb["input_ids"], qb["attention_mask"]), dim=-1)
        expected_e = F.normalize(
            encoder(eb["meta_input_ids"], eb["meta_attention_mask"]), dim=-1
        )
        assert torch.allclose(model.encode_query(qb), expected_q, atol=1e-6)
        assert torch.allclose(model.encode_episode(eb), expected_e, atol=1e-6)


def _randomize_heads(model: TwoTowerModel) -> None:
    torch.manual_seed(1)
    for tower in (model.query_tower, model.episode_tower):
        for p in tower.head.parameters():
            p.data.normal_(0, 0.1)


def test_transcript_only_influences_episodes_that_have_one(featurizer, contract) -> None:
    model = make_model(contract)
    _randomize_heads(model)
    batch = featurizer.episode_batch(EPISODES)
    altered = dict(batch)
    altered["transcript_input_ids"] = torch.roll(batch["transcript_input_ids"], 1, dims=1)
    with torch.no_grad():
        before, after = model.encode_episode(batch), model.encode_episode(altered)
    assert not torch.allclose(before[0], after[0])  # has a transcript
    assert torch.allclose(before[1], after[1])  # no transcript: masked out

    # Structured features matter once the head is non-zero.
    altered = dict(batch)
    altered["duration_bucket"] = torch.tensor([5, 5])
    with torch.no_grad():
        assert not torch.allclose(model.encode_episode(altered)[1], before[1])


def test_ablation_configs(featurizer, contract) -> None:
    batch = featurizer.episode_batch(EPISODES)
    variants = {
        "metadata-only": {"use_transcript": False, "use_structured": False},
        "id-only": {"episode_text": False, "use_transcript": False, "use_structured": False,
                    "podcast_id_buckets": 1000},
        "structured-only": {"episode_text": False, "use_transcript": False},
        "separate-encoders": {"share_text_encoder": False},
        "projected": {"embedding_dim": 16},
    }
    for name, overrides in variants.items():
        model = make_model(contract, **overrides)
        with torch.no_grad():
            vectors = model.encode_episode(batch)
        assert torch.isfinite(vectors).all(), name
        assert torch.allclose(vectors.norm(dim=-1), torch.ones(2), atol=1e-5), name

    shared, separate = make_model(contract), make_model(contract, share_text_encoder=False)
    assert separate.parameter_counts()["total"] > shared.parameter_counts()["total"]

    with pytest.raises(ValueError, match="requires episode_text"):
        TwoTowerConfig(episode_text=False, use_transcript=True)
    with pytest.raises(ValueError, match="at least one input"):
        TwoTowerConfig(episode_text=False, use_transcript=False, use_structured=False)
    with pytest.raises(ValueError, match="temperature"):
        TwoTowerConfig(temperature=0)


def test_save_and_load_roundtrip(featurizer, contract, tmp_path) -> None:
    model = make_model(contract, share_text_encoder=False)
    _randomize_heads(model)
    qb, eb = featurizer.query_batch(QUERIES), featurizer.episode_batch(EPISODES)
    with torch.no_grad():
        q, e = model.encode_query(qb), model.encode_episode(eb)

    out = model.save(tmp_path / "artifact", featurizer=featurizer,
                     metadata={"dataset_version": "1.1.0-test", "intended_use": "tests"})
    artifact = json.loads((out / "artifact.json").read_text())
    assert artifact["architecture"] == "two-tower"
    assert artifact["embedding_dim"] == 32
    assert artifact["feature_contract_sha256"] == contract.fingerprint()
    assert artifact["dataset_version"] == "1.1.0-test"
    assert artifact["parameters"]["total"] == model.parameter_counts()["total"]
    assert "git_commit" in artifact

    loaded = TwoTowerModel.load(out).eval()
    assert loaded.config == model.config and loaded.contract == contract
    reloaded_featurizer = Featurizer.load(out)
    qb2 = reloaded_featurizer.query_batch(QUERIES)
    eb2 = reloaded_featurizer.episode_batch(EPISODES)
    assert torch.equal(qb2["input_ids"], qb["input_ids"])
    with torch.no_grad():
        assert torch.allclose(loaded.encode_query(qb2), q, atol=1e-6)
        assert torch.allclose(loaded.encode_episode(eb2), e, atol=1e-6)
