"""Training tests: candidate masking, the loss, batching, and an offline
end-to-end smoke run that writes a model artifact."""

import json

import pandas as pd
import pytest
import torch

from ml.features import FeatureContract, Featurizer
from ml.models.two_tower import TwoTowerConfig, TwoTowerModel
from ml.tests.helpers import tiny_backbone, tiny_tokenizer
from ml.tests.test_build import write_snapshot
from ml.training.data import ExampleBatcher, load_episode_rows
from ml.training.loss import candidate_mask, contrastive_softmax_loss
from ml.training.train import TrainConfig, build_optimizer, freeze_lower_encoder, train


def test_candidate_mask_blocks_duplicate_and_sibling_positives() -> None:
    # Three queries; q0 and q1 share query text ("history"); q2's positive
    # (episode 30) also appears as q0's random negative (candidate 3).
    positive_index = torch.tensor([0, 1, 2])
    positive_ids = torch.tensor([10, 20, 30])
    candidate_ids = torch.tensor([10, 20, 30, 30, 99])
    query_keys = torch.tensor([0, 0, 1])
    candidate_keys = torch.tensor([0, 0, 1, -1, -1])
    mask = candidate_mask(positive_index, positive_ids, candidate_ids, query_keys, candidate_keys)
    assert mask.tolist() == [
        [True, False, True, True, True],  # q0: q1's positive is a sibling -> masked
        [False, True, True, True, True],  # q1: likewise for q0's positive
        [True, True, True, False, True],  # q2: candidate 3 *is* its positive -> masked
    ]
    assert mask.diagonal().all()  # every query keeps its own positive


def test_contrastive_loss_and_diagnostics() -> None:
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 1.0, 3.0]])
    positive_index = torch.tensor([0, 1])
    loss, diag = contrastive_softmax_loss(logits, positive_index)
    expected = torch.nn.functional.cross_entropy(logits, positive_index)
    assert torch.isclose(loss, expected)
    assert diag["accuracy"] == 0.5 and diag["mrr"] == pytest.approx(0.75)
    assert diag["candidates"] == 3.0

    # Masking the distractor makes the second query's positive win.
    mask = torch.tensor([[True, True, True], [True, True, False]])
    _, diag = contrastive_softmax_loss(logits, positive_index, mask)
    assert diag["accuracy"] == 1.0

    # Weights: a zero-weight example contributes nothing.
    weighted, _ = contrastive_softmax_loss(
        logits, positive_index, weights=torch.tensor([1.0, 0.0])
    )
    per_query = torch.nn.functional.cross_entropy(logits, positive_index, reduction="none")
    assert torch.isclose(weighted, per_query[0])


@pytest.fixture
def snapshot(tmp_path):
    return write_snapshot(tmp_path, n_episodes=30)


@pytest.fixture
def featurizer(tmp_path):
    contract = FeatureContract(languages=("en", "fr"), categories=("history", "technology"),
                               max_metadata_tokens=32, max_query_tokens=16)
    return Featurizer(contract, tiny_tokenizer(tmp_path))


def make_examples(ids: list[int], negatives: dict[int, list[int]], query=None) -> pd.DataFrame:
    return pd.DataFrame({
        "example_id": [f"syn:{i}" for i in ids],
        "event_time": pd.to_datetime(["2026-02-01T00:00:00+00:00"] * len(ids), utc=True),
        "query_text": [query or f"topic number {i}" for i in ids],
        "query_audio_variant": [""] * len(ids),
        "user_history": [[] for _ in ids],
        "positive_episode_id": ids,
        "negative_episode_ids": [negatives.get(i, []) for i in ids],
        "label_source": ["synthetic_query"] * len(ids),
        "label_strength": [0.3] * len(ids),
        "dataset_version": ["1.0.0-test"] * len(ids),
    })


def test_batcher_collates_positives_then_negatives(snapshot, featurizer) -> None:
    examples = make_examples([1, 2, 3], {1: [4, 5], 2: [6, 999], 3: [7, 8]})
    rows = load_episode_rows(snapshot, {1, 2, 3, 4, 5, 6, 7, 8})
    assert rows[1]["podcast_title"] == "Show 1" and rows[1]["categories"] == ["History"]

    batcher = ExampleBatcher(examples, rows, featurizer, negatives_per_example=2)
    batches = list(batcher.batches(batch_size=3, shuffle=False))
    assert len(batches) == 1
    batch = batches[0]
    assert batch.positive_index.tolist() == [0, 1, 2]
    assert batch.positive_ids.tolist() == [1, 2, 3]
    # Episode 999 is not in the snapshot and is dropped.
    assert batch.candidate_ids.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert batch.candidate_keys.tolist() == [0, 1, 2, -1, -1, -1, -1, -1]
    assert batch.episodes["meta_input_ids"].shape[0] == 8
    assert batch.query["input_ids"].shape[0] == 3
    assert batch.weights.tolist() == pytest.approx([0.3, 0.3, 0.3])

    # Identical query text shares a key so sibling positives get masked.
    shared = ExampleBatcher(make_examples([1, 2], {}, query="history"), rows, featurizer, 2)
    assert next(shared.batches(2, shuffle=False)).query_keys.tolist() == [0, 0]

    with pytest.raises(ValueError, match="missing from the snapshot"):
        ExampleBatcher(make_examples([1, 2, 31], {}), rows, featurizer, 2)


def test_batcher_augmentation_is_seeded(snapshot, featurizer) -> None:
    examples = make_examples([1, 2, 3, 4, 5, 6], {})
    rows = load_episode_rows(snapshot, {1, 2, 3, 4, 5, 6})
    plain = ExampleBatcher(examples, rows, featurizer, 0)
    plain_batch = next(plain.batches(6, shuffle=False))
    # Fixture episodes were published 21-31 days before the event time.
    assert set(plain_batch.episodes["age_bucket"].tolist()) <= {2, 3}
    assert plain_batch.query["language_id"].tolist() == [0] * 6

    jittered = ExampleBatcher(examples, rows, featurizer, 0, age_jitter_days=3 * 365,
                              query_language_prob=1.0, seed=7)
    batch = next(jittered.batches(6, shuffle=False))
    buckets = batch.episodes["age_bucket"].tolist()
    assert all(1 <= b <= featurizer.contract.num_age_buckets - 1 for b in buckets)
    assert max(buckets) > 3, "jitter should reach older age buckets"
    assert len(set(buckets)) > 1, "jitter should spread examples across age buckets"
    assert batch.query["language_id"].tolist() == [1] * 6  # snapshot rows are "en"
    again = next(ExampleBatcher(examples, rows, featurizer, 0, age_jitter_days=3 * 365,
                                query_language_prob=1.0, seed=7).batches(6, shuffle=False))
    assert again.episodes["age_bucket"].tolist() == buckets  # deterministic per example


def test_optimizer_separates_encoder_and_head_learning_rates(featurizer) -> None:
    model = TwoTowerModel(TwoTowerConfig(embedding_dim=32, hidden_dim=16), featurizer.contract,
                          tiny_backbone())
    config = TrainConfig(name="t", dataset_dir="", intended_use="", encoder_lr=1e-5, head_lr=1e-3)
    optimizer = build_optimizer(model, config)
    encoder_group, head_group = optimizer.param_groups
    assert encoder_group["lr"] == 1e-5 and head_group["lr"] == 1e-3
    n_encoder = sum(p.numel() for p in encoder_group["params"])
    assert n_encoder == sum(p.numel() for p in model.query_tower.encoder.parameters())
    total = sum(p.numel() for p in model.parameters())
    assert n_encoder + sum(p.numel() for p in head_group["params"]) == total

    # Freezing the lower encoder (the tiny backbone has one layer, so
    # keeping one trainable layer freezes only the embeddings) drops those
    # parameters from the optimizer.
    assert freeze_lower_encoder(model, trainable_layers=1) == 0
    backbone = model.query_tower.encoder.backbone
    assert not any(p.requires_grad for p in backbone.embeddings.parameters())
    optimizer = build_optimizer(model, config)
    n_encoder_trainable = sum(p.numel() for p in optimizer.param_groups[0]["params"])
    n_embeddings = sum(p.numel() for p in backbone.embeddings.parameters())
    assert n_encoder_trainable == n_encoder - n_embeddings


def test_train_smoke_writes_artifact_with_metadata(tmp_path, snapshot, featurizer) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    train_ids = list(range(1, 13))
    negatives = {i: [i + 12, i + 13] for i in train_ids}
    make_examples(train_ids, negatives).to_parquet(dataset_dir / "train.parquet", index=False)
    make_examples([26, 27, 28, 29], {26: [1], 27: [2], 28: [3], 29: [4]}).to_parquet(
        dataset_dir / "validation.parquet", index=False
    )
    (dataset_dir / "manifest.json").write_text(json.dumps({
        "name": "test-dataset", "dataset_version": "1.0.0-test",
        "content_hash": "abc", "snapshot_id": "snaptest12345",
    }))

    config = TrainConfig(
        name="smoke", dataset_dir=str(dataset_dir), snapshot=str(snapshot),
        output_dir=str(tmp_path / "models"), intended_use="unit test only",
        batch_size=4, negatives_per_example=2, max_steps=3, log_every=1,
        eval_every=2, eval_batches=1, warmup_steps=1, seed=1,
    )
    torch.manual_seed(0)
    model = TwoTowerModel(TwoTowerConfig(embedding_dim=32, hidden_dim=16, dropout=0.0),
                          featurizer.contract, tiny_backbone())
    before = {k: v.clone() for k, v in model.state_dict().items()}

    out_dir, metadata = train(config, model=model, featurizer=featurizer,
                              use_mlflow=False, device="cpu")

    assert metadata["steps"] == 3
    assert metadata["final"] is True and metadata["interrupted"] is False
    assert metadata["dataset_version"] == "1.0.0-test"
    for key in ("train_loss", "train_accuracy", "validation_loss", "validation_mrr"):
        assert key in metadata["metrics"] and metadata["metrics"][key] == metadata["metrics"][key]
    changed = any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert changed, "training did not update any weights"

    artifact = json.loads((out_dir / "artifact.json").read_text())
    for key in ("architecture", "embedding_dim", "feature_contract_sha256", "git_commit",
                "dataset_version", "dataset_content_hash", "snapshot_id", "metrics",
                "intended_use", "training_config", "training_config_sha256"):
        assert key in artifact, key
    assert artifact["intended_use"] == "unit test only"
    assert artifact["feature_contract_sha256"] == featurizer.contract.fingerprint()

    loaded = TwoTowerModel.load(out_dir)
    reloaded = Featurizer.load(out_dir)
    # Weights match the trained model, not the eval-step checkpoint that was
    # overwritten by the final save.
    batch = reloaded.query_batch([{"query_text": "topic number 3"}])
    with torch.no_grad():
        assert torch.allclose(loaded.eval().encode_query(batch),
                              model.eval().encode_query(batch), atol=1e-6)

    # Resume: rewind the saved state to step 2 (as if the run had died after
    # its step-2 checkpoint) and continue with the same config.
    state_path = out_dir / "training_state.pt"
    saved = torch.load(state_path, weights_only=True)
    assert saved["step"] == 3
    torch.save({**saved, "step": 2}, state_path)
    _, resumed = train(config, model=model, featurizer=featurizer,
                       use_mlflow=False, device="cpu", resume=True)
    assert resumed["steps"] == 3 and resumed["resumed_from"] == 2
    # A checkpoint written under a different config is refused.
    torch.save({**saved, "step": 2, "training_config_sha256": "other"}, state_path)
    with pytest.raises(SystemExit, match="different config"):
        train(config, model=model, featurizer=featurizer, use_mlflow=False, device="cpu",
              resume=True)

    # A dataset built from a different snapshot is refused.
    (dataset_dir / "manifest.json").write_text(json.dumps({
        "name": "test-dataset", "dataset_version": "1.0.0-test",
        "content_hash": "abc", "snapshot_id": "other",
    }))
    with pytest.raises(SystemExit, match="built from snapshot"):
        train(config, model=model, featurizer=featurizer, use_mlflow=False, device="cpu")


def test_sync_run_copies_params_metric_history_and_artifacts(tmp_path) -> None:
    import mlflow

    from ml.training.sync_run import sync_run

    source = f"sqlite:///{tmp_path / 'src.db'}"
    target = f"sqlite:///{tmp_path / 'dst.db'}"
    mlflow.set_tracking_uri(source)
    mlflow.create_experiment("two-tower-training", artifact_location=str(tmp_path / "src-art"))
    mlflow.set_experiment("two-tower-training")
    with mlflow.start_run(run_name="offline") as run:
        mlflow.log_param("batch_size", 8)
        for step in range(3):
            mlflow.log_metric("train_loss", 1.0 / (step + 1), step=step)
        mlflow.log_dict({"steps": 3}, "artifact.json")
    mlflow.tracking.MlflowClient(tracking_uri=target).create_experiment(
        "two-tower-training", artifact_location=str(tmp_path / "dst-art")
    )
    copied_id = sync_run(run.info.run_id, source, target)

    client = mlflow.tracking.MlflowClient(tracking_uri=target)
    copied = client.get_run(copied_id)
    assert copied.info.run_name == "offline" and copied.info.status == "FINISHED"
    assert copied.data.params == {"batch_size": "8"}
    assert copied.data.tags["source_run_id"] == run.info.run_id
    history = client.get_metric_history(copied_id, "train_loss")
    assert [(m.step, m.value) for m in history] == [(0, 1.0), (1, 0.5), (2, 1 / 3)]
    assert [a.path for a in client.list_artifacts(copied_id)] == ["artifact.json"]


def test_pool_validation_distillation_and_accumulation(tmp_path, snapshot, featurizer) -> None:
    from ml.evaluation.pool_rerank import evaluate_pool, load_pool

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    train_ids = list(range(1, 13))
    make_examples(train_ids, {i: [i + 12] for i in train_ids}).to_parquet(
        dataset_dir / "train.parquet", index=False
    )
    make_examples([26, 27], {26: [1], 27: [2]}).to_parquet(
        dataset_dir / "validation.parquet", index=False
    )
    (dataset_dir / "manifest.json").write_text(json.dumps({
        "name": "test-dataset", "dataset_version": "1.0.0-test",
        "content_hash": "abc", "snapshot_id": "snaptest12345",
    }))
    # A judged pool over fixture episodes: two queries, graded candidates.
    pool_dir = tmp_path / "relevance"
    pool_dir.mkdir()
    pool_queries = [
        {"query_id": "q1", "query_text": "topic number 1", "query_type": "entity"},
        {"query_id": "q2", "query_text": "topic number 5", "query_type": "entity"},
    ]
    (pool_dir / "queries.jsonl").write_text("".join(json.dumps(q) + "\n" for q in pool_queries))
    (pool_dir / "qrels.txt").write_text("q1 0 1 3\nq1 0 2 0\nq1 0 3 1\nq2 0 5 2\nq2 0 6 0\n")
    rows = load_episode_rows(snapshot, {1, 2, 3, 5, 6})
    pd.DataFrame(list(rows.values())).to_parquet(pool_dir / "judged-episodes.parquet", index=False)
    queries, qrels, judged = load_pool(pool_dir)
    assert len(judged) == 5 and set(qrels) == {"q1", "q2"}

    config = TrainConfig(
        name="pool", dataset_dir=str(dataset_dir), snapshot=str(snapshot),
        output_dir=str(tmp_path / "models"), intended_use="unit test only",
        batch_size=2, negatives_per_example=1, max_steps=2, accumulate_steps=2,
        log_every=1, eval_every=1, eval_batches=1, warmup_steps=1, seed=3,
        distill_weight=0.5, pool_validation_dir=str(pool_dir),
    )
    torch.manual_seed(0)
    model = TwoTowerModel(TwoTowerConfig(embedding_dim=32, hidden_dim=16, dropout=0.0),
                          featurizer.contract, tiny_backbone())
    teacher = TwoTowerModel(TwoTowerConfig(embedding_dim=32, hidden_dim=16, dropout=0.0),
                            featurizer.contract, tiny_backbone())
    out_dir, metadata = train(config, model=model, featurizer=featurizer, use_mlflow=False,
                              device="cpu", teacher=teacher)
    assert metadata["steps"] == 2  # 4 micro-batches of 2 rows, accumulated in pairs
    assert metadata["selected_checkpoint"] == "best"
    assert metadata["best_step"] in (1, 2)
    assert 0.0 <= metadata["metrics"]["pool_ndcg_at_10"] <= 1.0
    assert metadata["metrics"]["pool_queries"] == 2
    assert metadata["best_pool_ndcg_at_10"] == metadata["metrics"]["best_pool_ndcg_at_10"]
    # The saved weights are the best checkpoint's: re-scoring them reproduces its pool score.
    saved = TwoTowerModel.load(out_dir).eval()
    rescored = evaluate_pool(saved, Featurizer.load(out_dir), queries, qrels, judged)
    assert rescored["pool_ndcg_at_10"] == metadata["best_pool_ndcg_at_10"]
