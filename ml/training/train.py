"""Train the two-tower model from a checked-in experiment config.

    uv run python -m ml.training.train --config experiments/training/two-tower-v1.json

Reads a versioned dataset and the snapshot it was built from, builds the
feature contract from that snapshot, fine-tunes the pretrained text model
with the contrastive softmax loss (in-batch + random catalog negatives),
and writes a self-contained model artifact whose ``artifact.json`` records
the architecture, feature contract, embedding dimension, dataset version,
code revision, metrics, and intended use. Every run logs to MLflow unless
``--no-mlflow`` is given.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

from ml.datasets import snapshot as snapshot_mod
from ml.datasets.build import code_revision
from ml.datasets.validate import load_examples
from ml.evaluation.run import DEFAULT_MLFLOW_URI
from ml.features import FeatureContract, Featurizer
from ml.models.two_tower import TwoTowerConfig, TwoTowerModel
from ml.training.data import ExampleBatcher, load_episode_rows
from ml.training.loss import candidate_mask, contrastive_softmax_loss


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: str
    dataset_dir: str
    intended_use: str
    snapshot: str = "data/snapshots"
    output_dir: str = "data/models"
    seed: int = 0
    model: dict[str, Any] = dataclasses.field(default_factory=dict)  # TwoTowerConfig
    contract: dict[str, Any] = dataclasses.field(default_factory=dict)  # from_catalog kwargs
    batch_size: int = 64
    negatives_per_example: int = 4
    epochs: int = 1
    max_steps: int = 0  # 0 = full epochs
    encoder_lr: float = 2e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    # Recompute encoder activations in the backward pass: a ~2-3x slower
    # encoder for a fraction of the memory, which is what lets a batch of
    # queries plus hundreds of 256-token candidates train on a laptop GPU.
    gradient_checkpointing: bool = True
    weight_by_label_strength: bool = False
    log_every: int = 10
    eval_every: int = 200
    eval_batches: int = 20
    mlflow_experiment: str = "two-tower-training"

    @classmethod
    def load(cls, path: str | Path, **overrides: Any) -> TrainConfig:
        raw = json.loads(Path(path).read_text())
        raw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**raw)

    def sha256(self) -> str:
        canonical = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_optimizer(model: TwoTowerModel, config: TrainConfig) -> torch.optim.Optimizer:
    """Small learning rate for the pretrained encoder(s), larger for the
    freshly initialized heads and embeddings."""
    encoder_params = {
        id(p): p for tower in (model.query_tower, model.episode_tower)
        if hasattr(tower, "encoder") for p in tower.encoder.parameters()
    }
    head_params = [p for p in model.parameters() if id(p) not in encoder_params]
    return torch.optim.AdamW(
        [
            {"params": list(encoder_params.values()), "lr": config.encoder_lr},
            {"params": head_params, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )


def enable_gradient_checkpointing(model: TwoTowerModel) -> None:
    seen: set[int] = set()
    for tower in (model.query_tower, model.episode_tower):
        backbone = getattr(getattr(tower, "encoder", None), "backbone", None)
        if backbone is None or id(backbone) in seen:
            continue
        seen.add(id(backbone))
        if getattr(backbone, "supports_gradient_checkpointing", False):
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )


def lr_lambda(total_steps: int, warmup_steps: int):
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        remaining = max(total_steps - warmup_steps, 1)
        return max(0.0, (total_steps - step) / remaining)
    return schedule


def step_loss(model: TwoTowerModel, batch, weight_by_strength: bool):
    query_vectors = model.encode_query(batch.query)
    episode_vectors = model.encode_episode(batch.episodes)
    logits = model.logits(query_vectors, episode_vectors)
    mask = candidate_mask(
        batch.positive_index, batch.positive_ids, batch.candidate_ids,
        batch.query_keys, batch.candidate_keys,
    )
    weights = batch.weights if weight_by_strength else None
    return contrastive_softmax_loss(logits, batch.positive_index, mask, weights)


@torch.no_grad()
def evaluate_split(
    model: TwoTowerModel, batcher: ExampleBatcher, config: TrainConfig, device: str
) -> dict[str, float]:
    """Loss and candidate-set ranking diagnostics on a fixed set of batches."""
    model.eval()
    totals: dict[str, float] = {"loss": 0.0, "accuracy": 0.0, "mrr": 0.0}
    n = 0
    for batch in batcher.batches(config.batch_size, shuffle=False):
        if n >= config.eval_batches:
            break
        loss, diag = step_loss(model, batch.to(device), config.weight_by_label_strength)
        totals["loss"] += loss.item()
        totals["accuracy"] += diag["accuracy"]
        totals["mrr"] += diag["mrr"]
        n += 1
    model.train()
    if n == 0:
        return {}
    return {f"validation_{k}": round(v / n, 4) for k, v in totals.items()}


def build_contract(snapshot_dir: Path, overrides: dict[str, Any]) -> FeatureContract:
    episodes = pd.read_parquet(snapshot_dir / "episodes.parquet", columns=["language"])
    podcasts = pd.read_parquet(snapshot_dir / "podcasts.parquet", columns=["categories"])
    return FeatureContract.from_catalog(episodes["language"], podcasts["categories"], **overrides)


def train(
    config: TrainConfig,
    model: TwoTowerModel | None = None,
    featurizer: Featurizer | None = None,
    use_mlflow: bool = True,
    device: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run one training job; returns the artifact directory and its metadata.
    ``model``/``featurizer`` are injectable for tests; by default both come
    from the contract's pretrained text model."""
    torch.manual_seed(config.seed)
    device = device or pick_device()
    started = time.time()

    dataset_dir = Path(config.dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    snapshot_dir = snapshot_mod.resolve(config.snapshot)
    snapshot_manifest = json.loads((snapshot_dir / "manifest.json").read_text())
    if snapshot_manifest["snapshot_id"] != manifest["snapshot_id"]:
        raise SystemExit(
            f"dataset was built from snapshot {manifest['snapshot_id']} but "
            f"{snapshot_dir} is {snapshot_manifest['snapshot_id']}"
        )

    train_df = load_examples(dataset_dir / "train.parquet")
    validation_df = load_examples(dataset_dir / "validation.parquet")
    if config.max_steps:
        # Smoke runs: only load the rows the steps can touch.
        train_df = train_df.iloc[: config.max_steps * config.batch_size]
    referenced: set[int] = set()
    for df in (train_df, validation_df):
        referenced.update(int(i) for i in df["positive_episode_id"])
        for negs in df["negative_episode_ids"]:
            referenced.update(int(n) for n in list(negs)[: config.negatives_per_example])
    episode_rows = load_episode_rows(snapshot_dir, referenced)

    if featurizer is None:
        featurizer = Featurizer.from_contract(build_contract(snapshot_dir, config.contract))
    contract = featurizer.contract
    if model is None:
        model = TwoTowerModel.from_pretrained_text(contract, TwoTowerConfig(**config.model))
    model = model.to(device)
    model.train()
    if config.gradient_checkpointing:
        enable_gradient_checkpointing(model)

    train_batcher = ExampleBatcher(train_df, episode_rows, featurizer, config.negatives_per_example)
    val_batcher = ExampleBatcher(
        validation_df, episode_rows, featurizer, config.negatives_per_example
    )
    steps_per_epoch = math.ceil(len(train_batcher) / config.batch_size)
    total_steps = config.max_steps or steps_per_epoch * config.epochs

    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(total_steps, min(config.warmup_steps, total_steps // 10))
    )

    mlflow = None
    if use_mlflow:
        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI))
        mlflow.set_experiment(config.mlflow_experiment)
        mlflow.start_run(run_name=config.name)
        mlflow.log_params({
            "dataset_version": manifest["dataset_version"],
            "snapshot_id": manifest["snapshot_id"],
            "config_sha256": config.sha256()[:12],
            "feature_contract_sha256": contract.fingerprint()[:12],
            "device": device,
            "total_steps": total_steps,
            "train_rows": len(train_batcher),
            **{f"train_{k}": v for k, v in dataclasses.asdict(config).items()
               if not isinstance(v, dict)},
            **{f"model_{k}": v for k, v in dataclasses.asdict(model.config).items()},
            **code_revision(),
        })

    def log_metrics(values: dict[str, float], step: int) -> None:
        print(json.dumps({"step": step, **values}), flush=True)
        if mlflow is not None:
            mlflow.log_metrics(values, step=step)

    step = 0
    last_train: dict[str, float] = {}
    validation: dict[str, float] = {}
    running: dict[str, float] = {"loss": 0.0, "accuracy": 0.0, "mrr": 0.0, "n": 0.0}
    done = False
    window_started = time.time()
    for epoch in range(config.epochs):
        for batch in train_batcher.batches(config.batch_size, shuffle=True,
                                           seed=config.seed + epoch):
            loss, diag = step_loss(model, batch.to(device), config.weight_by_label_strength)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            step += 1

            running["loss"] += loss.item()
            running["accuracy"] += diag["accuracy"]
            running["mrr"] += diag["mrr"]
            running["n"] += 1
            if step % config.log_every == 0 or step == total_steps:
                n = running["n"]
                last_train = {
                    "train_loss": round(running["loss"] / n, 4),
                    "train_accuracy": round(running["accuracy"] / n, 4),
                    "train_mrr": round(running["mrr"] / n, 4),
                    "encoder_lr": scheduler.get_last_lr()[0],
                    "seconds_per_step": round((time.time() - window_started) / n, 2),
                    "epoch": epoch,
                }
                log_metrics(last_train, step)
                running = {"loss": 0.0, "accuracy": 0.0, "mrr": 0.0, "n": 0.0}
                window_started = time.time()
            if config.eval_every and step % config.eval_every == 0 and step != total_steps:
                validation = evaluate_split(model, val_batcher, config, device)
                log_metrics(validation, step)
            if step >= total_steps:
                done = True
                break
        if done:
            break

    validation = evaluate_split(model, val_batcher, config, device)
    log_metrics(validation, step)
    seconds = round(time.time() - started, 1)
    metrics = {**last_train, **validation}

    run_id = f"{config.name}-{manifest['dataset_version']}-{config.sha256()[:8]}"
    metadata = {
        "name": config.name,
        "intended_use": config.intended_use,
        "dataset_version": manifest["dataset_version"],
        "dataset_content_hash": manifest["content_hash"],
        "dataset_name": manifest["name"],
        "snapshot_id": manifest["snapshot_id"],
        "training_config": dataclasses.asdict(config),
        "training_config_sha256": config.sha256(),
        "train_rows": len(train_batcher),
        "validation_rows": len(val_batcher),
        "steps": step,
        "epochs_completed": step / steps_per_epoch,
        "device": device,
        "seconds": seconds,
        "metrics": metrics,
        "finished_at": datetime.now(UTC).isoformat(),
    }
    if mlflow is not None:
        metadata["mlflow_run_id"] = mlflow.active_run().info.run_id
    out_dir = model.cpu().save(Path(config.output_dir) / run_id, featurizer, metadata)
    if mlflow is not None:
        mlflow.log_dict(json.loads((out_dir / "artifact.json").read_text()), "artifact.json")
        mlflow.log_param("artifact_dir", str(out_dir))
        mlflow.end_run()
    return out_dir, metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="override for smoke runs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    config = TrainConfig.load(
        args.config, max_steps=args.max_steps, batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    out_dir, metadata = train(config, use_mlflow=not args.no_mlflow, device=args.device)
    print(json.dumps({k: metadata[k] for k in
                      ("dataset_version", "steps", "seconds", "metrics")}, indent=2))
    print(f"artifact written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
