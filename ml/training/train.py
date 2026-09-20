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
import signal
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
    # Training-time feature augmentation (see ExampleBatcher): jitter the
    # publication-age reference time so every age bucket is trained, and
    # attach the positive's language to the query with this probability.
    age_jitter_days: float = 0.0
    query_language_prob: float = 0.0
    epochs: int = 1
    max_steps: int = 0  # 0 = full epochs
    encoder_lr: float = 2e-5
    head_lr: float = 1e-3
    # Optimizer steps happen every ``accumulate_steps`` micro-batches.
    accumulate_steps: int = 1
    # Weight of a distillation term keeping both towers' vectors close to
    # the frozen pretrained text model's (1 - cosine, query and episode
    # sides). 0 disables it.
    distill_weight: float = 0.0
    # Directory with queries.jsonl, qrels.txt, and judged-episodes.parquet:
    # every validation pass re-ranks each query's judged pool (see
    # ml.evaluation.pool_rerank) and the best checkpoint by pool NDCG@10 is
    # the one saved as final. Empty disables it.
    pool_validation_dir: str = ""
    # Fine-tune only the top N encoder layers (0 = all). Freezing the token
    # embeddings and lower layers removes their activations and optimizer
    # state from memory and keeps the pretrained lexical representation.
    trainable_encoder_layers: int = 0
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
    # Save the artifact (plus optimizer and scheduler state) at every
    # validation pass and on interrupt, so a killed run still yields its
    # latest model and can be resumed with ``--resume``.
    checkpoint: bool = True
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


def _backbones(model: TwoTowerModel) -> list[nn.Module]:
    seen: dict[int, nn.Module] = {}
    for tower in (model.query_tower, model.episode_tower):
        backbone = getattr(getattr(tower, "encoder", None), "backbone", None)
        if backbone is not None:
            seen.setdefault(id(backbone), backbone)
    return list(seen.values())


def freeze_lower_encoder(model: TwoTowerModel, trainable_layers: int) -> int:
    """Freeze token embeddings and all but the top ``trainable_layers``
    transformer layers of each backbone; returns the number frozen."""
    frozen = 0
    if trainable_layers <= 0:
        return frozen
    for backbone in _backbones(model):
        layers = backbone.encoder.layer
        keep = set(range(max(len(layers) - trainable_layers, 0), len(layers)))
        for p in backbone.embeddings.parameters():
            p.requires_grad_(False)
        for i, layer in enumerate(layers):
            if i not in keep:
                frozen += 1
                for p in layer.parameters():
                    p.requires_grad_(False)
    return frozen


def build_optimizer(model: TwoTowerModel, config: TrainConfig) -> torch.optim.Optimizer:
    """Small learning rate for the pretrained encoder(s), larger for the
    freshly initialized heads and embeddings. Frozen parameters are left out."""
    encoder_params = {
        id(p): p for backbone in _backbones(model) for p in backbone.parameters()
    }
    trainable = [p for p in model.parameters() if p.requires_grad]
    encoder = [p for p in trainable if id(p) in encoder_params]
    heads = [p for p in trainable if id(p) not in encoder_params]
    return torch.optim.AdamW(
        [{"params": encoder, "lr": config.encoder_lr}, {"params": heads, "lr": config.head_lr}],
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


def step_loss(
    model: TwoTowerModel, batch, weight_by_strength: bool,
    teacher: TwoTowerModel | None = None, distill_weight: float = 0.0,
):
    query_vectors = model.encode_query(batch.query)
    episode_vectors = model.encode_episode(batch.episodes)
    logits = model.logits(query_vectors, episode_vectors)
    mask = candidate_mask(
        batch.positive_index, batch.positive_ids, batch.candidate_ids,
        batch.query_keys, batch.candidate_keys,
    )
    weights = batch.weights if weight_by_strength else None
    loss, diag = contrastive_softmax_loss(logits, batch.positive_index, mask, weights)
    if teacher is not None and distill_weight > 0:
        with torch.no_grad():
            teacher_q = teacher.encode_query(batch.query)
            teacher_e = teacher.encode_episode(batch.episodes)
        distill = (1 - (query_vectors * teacher_q).sum(-1)).mean() + (
            1 - (episode_vectors * teacher_e).sum(-1)
        ).mean()
        diag["contrastive_loss"] = loss.item()
        diag["distill_loss"] = distill.item()
        loss = loss + distill_weight * distill
    return loss, diag


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


TRAINING_STATE_FILENAME = "training_state.pt"


def train(
    config: TrainConfig,
    model: TwoTowerModel | None = None,
    featurizer: Featurizer | None = None,
    use_mlflow: bool = True,
    device: str | None = None,
    resume: bool = False,
    teacher: TwoTowerModel | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run one training job; returns the artifact directory and its metadata.
    ``model``/``featurizer`` are injectable for tests; by default both come
    from the contract's pretrained text model. ``resume`` continues from the
    artifact directory's last checkpoint (same config, same data order)."""
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
    cap = config.max_steps * config.batch_size * max(config.accumulate_steps, 1)
    if config.max_steps and cap < len(train_df):
        # Capped runs train on a seeded random subset of the split (a prefix
        # would be all of one generator's examples); the rest is never loaded.
        train_df = train_df.sample(n=cap, random_state=config.seed).sort_values("example_id")
    referenced: set[int] = set()
    for df in (train_df, validation_df):
        referenced.update(int(i) for i in df["positive_episode_id"])
        for negs in df["negative_episode_ids"]:
            referenced.update(int(n) for n in list(negs)[: config.negatives_per_example])
    episode_rows = load_episode_rows(snapshot_dir, referenced)

    run_id = f"{config.name}-{manifest['dataset_version']}-{config.sha256()[:8]}"
    out_dir = Path(config.output_dir) / run_id
    resume_state = None
    if resume and (out_dir / TRAINING_STATE_FILENAME).exists():
        resume_state = torch.load(out_dir / TRAINING_STATE_FILENAME, map_location="cpu",
                                  weights_only=True)
        if resume_state["training_config_sha256"] != config.sha256():
            raise SystemExit(f"checkpoint in {out_dir} was trained with a different config")
        if featurizer is None:
            featurizer = Featurizer.load(out_dir)
        if model is None:
            model = TwoTowerModel.load(out_dir)
        else:
            model.load_state_dict(torch.load(out_dir / "weights.pt", map_location="cpu",
                                             weights_only=True))
        print(json.dumps({"resumed_from": resume_state["step"], "artifact": str(out_dir)}),
              flush=True)

    if featurizer is None:
        featurizer = Featurizer.from_contract(build_contract(snapshot_dir, config.contract))
    contract = featurizer.contract
    if model is None:
        model = TwoTowerModel.from_pretrained_text(contract, TwoTowerConfig(**config.model))
    model = model.to(device)
    model.train()
    frozen_layers = freeze_lower_encoder(model, config.trainable_encoder_layers)
    if config.gradient_checkpointing:
        enable_gradient_checkpointing(model)

    if config.distill_weight > 0 and teacher is None:
        # Zero-initialized heads make an untrained model exactly the
        # pretrained text model, so this is the E5 baseline as a teacher.
        teacher = TwoTowerModel.from_pretrained_text(contract, model.config)
    if teacher is not None:
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    pool = None
    if config.pool_validation_dir:
        from ml.evaluation.pool_rerank import evaluate_pool, load_pool

        pool_queries, pool_qrels, pool_rows = load_pool(config.pool_validation_dir)
        pool = (pool_queries, pool_qrels, pool_rows)

    augmentation = {
        "age_jitter_days": config.age_jitter_days,
        "query_language_prob": config.query_language_prob,
        "seed": config.seed,
    }
    train_batcher = ExampleBatcher(
        train_df, episode_rows, featurizer, config.negatives_per_example, **augmentation
    )
    val_batcher = ExampleBatcher(
        validation_df, episode_rows, featurizer, config.negatives_per_example, **augmentation
    )
    accumulate = max(config.accumulate_steps, 1)
    batches_per_epoch = math.ceil(len(train_batcher) / config.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / accumulate)
    total_steps = config.max_steps or steps_per_epoch * config.epochs

    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(total_steps, min(config.warmup_steps, total_steps // 10))
    )
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])

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
            "frozen_encoder_layers": frozen_layers,
            **{f"parameters_{k}": v for k, v in model.parameter_counts().items()},
            **{f"train_{k}": v for k, v in dataclasses.asdict(config).items()
               if not isinstance(v, dict)},
            **{f"model_{k}": v for k, v in dataclasses.asdict(model.config).items()},
            **code_revision(),
        })

    def log_metrics(values: dict[str, float], step: int) -> None:
        print(json.dumps({"step": step, **values}), flush=True)
        if mlflow is not None:
            # Asynchronous so a slow tracking server never blocks the loop.
            mlflow.log_metrics(values, step=step, synchronous=False)

    state = {"step": 0, "epoch": 0, "train": {}, "validation": {}, "interrupted": False,
             "best_step": 0, "best_pool_ndcg_at_10": -1.0}
    best_weights: dict[str, torch.Tensor] | None = None
    if resume_state is not None:
        state["step"] = int(resume_state["step"])
        state["epoch"] = int(resume_state["epoch"])
        state["resumed_from"] = state["step"]

    def save_artifact(final: bool) -> dict[str, Any]:
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
            "steps": state["step"],
            "total_steps": total_steps,
            "epochs_completed": round(state["step"] / steps_per_epoch, 4),
            "final": final,
            "interrupted": state["interrupted"],
            "resumed_from": state.get("resumed_from"),
            "best_step": state["best_step"] if pool is not None else None,
            "best_pool_ndcg_at_10": state["best_pool_ndcg_at_10"] if pool is not None else None,
            "selected_checkpoint": ("best" if final and pool is not None else "last"),
            "device": device,
            "seconds": round(time.time() - started, 1),
            "metrics": {**state["train"], **state["validation"]},
            "finished_at": datetime.now(UTC).isoformat(),
        }
        if mlflow is not None:
            metadata["mlflow_run_id"] = mlflow.active_run().info.run_id
        model.save(out_dir, featurizer, metadata)
        torch.save(
            {
                "step": state["step"],
                "epoch": state["epoch"],
                "training_config_sha256": config.sha256(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            out_dir / TRAINING_STATE_FILENAME,
        )
        print(json.dumps({"step": state["step"], "checkpoint": str(out_dir), "final": final}),
              flush=True)
        return metadata

    def validate() -> None:
        nonlocal best_weights
        state["validation"] = evaluate_split(model, val_batcher, config, device)
        if pool is not None:
            state["validation"].update(evaluate_pool(model, featurizer, *pool, device=device))
            score = state["validation"].get("pool_ndcg_at_10", -1.0)
            if score > state["best_pool_ndcg_at_10"]:
                state["best_pool_ndcg_at_10"] = score
                state["best_step"] = state["step"]
                best_weights = {k: v.detach().to("cpu", copy=True)
                                for k, v in model.state_dict().items()}
            state["validation"]["best_pool_ndcg_at_10"] = state["best_pool_ndcg_at_10"]
        log_metrics(state["validation"], state["step"])
        if device == "mps":
            torch.mps.empty_cache()

    running: dict[str, float] = {"loss": 0.0, "accuracy": 0.0, "mrr": 0.0, "n": 0.0}
    window_started = time.time()
    try:
        for epoch in range(state["epoch"], config.epochs):
            state["epoch"] = epoch
            skip = max(state["step"] - epoch * steps_per_epoch, 0)
            micro = 0
            optimizer.zero_grad(set_to_none=True)
            for batch in train_batcher.batches(config.batch_size, shuffle=True,
                                               seed=config.seed + epoch, skip=skip * accumulate):
                loss, diag = step_loss(model, batch.to(device), config.weight_by_label_strength,
                                       teacher, config.distill_weight)
                (loss / accumulate).backward()
                micro += 1
                if micro % accumulate:
                    continue
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step = state["step"] = state["step"] + 1
                if device == "mps" and step % 25 == 0:
                    torch.mps.empty_cache()  # return cached blocks to the OS

                running["loss"] += loss.item()
                running["accuracy"] += diag["accuracy"]
                running["mrr"] += diag["mrr"]
                running["n"] += 1
                if step % config.log_every == 0 or step == total_steps:
                    n = running["n"]
                    state["train"] = {
                        "train_loss": round(running["loss"] / n, 4),
                        "train_accuracy": round(running["accuracy"] / n, 4),
                        "train_mrr": round(running["mrr"] / n, 4),
                        "encoder_lr": scheduler.get_last_lr()[0],
                        "seconds_per_step": round((time.time() - window_started) / n, 2),
                        "epoch": epoch,
                    }
                    log_metrics(state["train"], step)
                    running = {"loss": 0.0, "accuracy": 0.0, "mrr": 0.0, "n": 0.0}
                    window_started = time.time()
                if config.eval_every and step % config.eval_every == 0 and step != total_steps:
                    validate()
                    if config.checkpoint:
                        save_artifact(final=False)
                if step >= total_steps:
                    break
            if state["step"] >= total_steps:
                break
    except KeyboardInterrupt:
        state["interrupted"] = True
        print(json.dumps({"step": state["step"], "interrupted": True}), flush=True)

    validate()
    if pool is not None and best_weights is not None and state["best_step"] != state["step"]:
        model.load_state_dict(best_weights)
        print(json.dumps({"restored_best_step": state["best_step"],
                          "best_pool_ndcg_at_10": state["best_pool_ndcg_at_10"]}), flush=True)
    metadata = save_artifact(final=True)
    if mlflow is not None:
        mlflow.flush_async_logging()
        mlflow.log_dict(json.loads((out_dir / "artifact.json").read_text()), "artifact.json")
        mlflow.log_param("artifact_dir", str(out_dir))
        mlflow.end_run(status="FAILED" if state["interrupted"] else "FINISHED")
    return out_dir, metadata


def _raise_interrupt(signum, frame) -> None:  # noqa: ARG001
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGTERM, _raise_interrupt)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="override for smoke runs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the artifact's last checkpoint if one exists")
    args = parser.parse_args(argv)

    config = TrainConfig.load(
        args.config, max_steps=args.max_steps, batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    out_dir, metadata = train(config, use_mlflow=not args.no_mlflow, device=args.device,
                              resume=args.resume)
    print(json.dumps({k: metadata[k] for k in
                      ("dataset_version", "steps", "seconds", "metrics")}, indent=2))
    print(f"artifact written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
