"""Two-tower retrieval model.

Both towers map into one normalized space and are scored by dot product::

    score(query, episode) = dot(normalize(q(query)), normalize(e(episode)))

**Query tower** — a pretrained text encoder over the (STT or typed) query
text, plus a residual head fed with the query's context: language and the
intent flags derived from its structured filters.

**Episode tower** — the same text encoder (shared by default) over the
flattened metadata passage (titles, categories, publisher, bounded
description), an optional second pass over a bounded transcript excerpt,
learned embeddings for language, categories, duration bucket, and
publication-age bucket, and an optional hashed podcast-id embedding for
collaborative signal once interaction data exists. All of it goes through
a residual head added to the metadata text vector.

Both heads' final layers are zero-initialized, so an untrained model is
*exactly* the pretrained-embedding baseline: fine-tuning can only move
away from it where the data says so, and ablations (transcript, structured
features, ids, text) are configuration switches, not code forks.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ml.features.contract import FeatureContract

CONFIG_FILENAME = "model_config.json"
ARTIFACT_FILENAME = "artifact.json"
WEIGHTS_FILENAME = "weights.pt"
BACKBONE_DIR = "backbone"
EPISODE_BACKBONE_DIR = "episode_backbone"


@dataclasses.dataclass(frozen=True)
class TwoTowerConfig:
    """Architecture knobs. Feature semantics live in ``FeatureContract``."""

    embedding_dim: int = 384
    hidden_dim: int = 512
    dropout: float = 0.1
    # Softmax temperature applied to cosine similarities (logits = sim / T).
    temperature: float = 0.05
    share_text_encoder: bool = True
    # Query-side context: language embedding + intent flags.
    query_context: bool = True
    # Episode-side inputs; each is an ablation switch.
    episode_text: bool = True
    use_transcript: bool = True
    use_structured: bool = True
    podcast_id_buckets: int = 0  # 0 disables the learned podcast id
    language_dim: int = 16
    category_dim: int = 32
    duration_dim: int = 8
    age_dim: int = 8
    podcast_id_dim: int = 32

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.embedding_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("embedding_dim and hidden_dim must be positive")
        if not self.episode_text and self.use_transcript:
            raise ValueError("use_transcript requires episode_text")
        if not (self.episode_text or self.use_structured or self.podcast_id_buckets):
            raise ValueError("the episode tower needs at least one input")

    @classmethod
    def from_dict(cls, raw: dict) -> TwoTowerConfig:
        return cls(**raw)


class TextEncoder(nn.Module):
    """Masked mean pooling over a Hugging Face encoder — the pooling the E5
    family was trained with."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.dim: int = int(backbone.config.hidden_size)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def _head(in_dim: int, hidden_dim: int, out_dim: int, dropout: float, zero_init: bool) -> nn.Module:
    final = nn.Linear(hidden_dim, out_dim)
    if zero_init:
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), final)


def _projection(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)


class QueryTower(nn.Module):
    def __init__(self, config: TwoTowerConfig, contract: FeatureContract, encoder: TextEncoder):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.proj = _projection(encoder.dim, config.embedding_dim)
        head_in = encoder.dim
        if config.query_context:
            self.language = nn.Embedding(contract.num_languages, config.language_dim)
            head_in += config.language_dim + len(contract.intent_features)
        self.head = _head(head_in, config.hidden_dim, config.embedding_dim, config.dropout,
                          zero_init=True)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        text = self.encoder(batch["input_ids"], batch["attention_mask"])
        parts = [text]
        if self.config.query_context:
            parts += [self.language(batch["language_id"]), batch["intent"]]
        out = self.proj(text) + self.head(torch.cat(parts, dim=-1))
        return F.normalize(out, dim=-1)


class EpisodeTower(nn.Module):
    def __init__(self, config: TwoTowerConfig, contract: FeatureContract, encoder: TextEncoder):
        super().__init__()
        self.config = config
        head_in = 0
        if config.episode_text:
            self.encoder = encoder
            self.proj = _projection(encoder.dim, config.embedding_dim)
            head_in += encoder.dim
        if config.use_transcript:
            head_in += encoder.dim + 1  # + has_transcript flag
        if config.use_structured:
            self.language = nn.Embedding(contract.num_languages, config.language_dim)
            self.duration = nn.Embedding(contract.num_duration_buckets, config.duration_dim)
            self.age = nn.Embedding(contract.num_age_buckets, config.age_dim)
            head_in += config.language_dim + config.duration_dim + config.age_dim
            self.num_categories = contract.num_categories
            if self.num_categories:
                # Mean of category embeddings, expressed as a linear map over
                # the normalized multi-hot vector.
                self.category = nn.Linear(self.num_categories, config.category_dim, bias=False)
                head_in += config.category_dim
        if config.podcast_id_buckets:
            self.podcast = nn.Embedding(config.podcast_id_buckets, config.podcast_id_dim)
            head_in += config.podcast_id_dim
        # Without a text vector to add to, a zero head would emit the zero
        # vector, so id-only / structured-only towers start from random.
        self.head = _head(head_in, config.hidden_dim, config.embedding_dim, config.dropout,
                          zero_init=config.episode_text)

    def _transcript(self, batch: dict[str, Tensor], size: int) -> Tensor:
        """Encode only episodes that have a transcript; zeros elsewhere."""
        present = batch["has_transcript"] > 0
        out = torch.zeros(size, self.encoder.dim, device=present.device,
                          dtype=next(self.encoder.parameters()).dtype)
        if bool(present.any()):
            idx = present.nonzero(as_tuple=True)[0]
            out = out.index_copy(0, idx, self.encoder(
                batch["transcript_input_ids"][idx], batch["transcript_attention_mask"][idx]
            ))
        return out

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        config = self.config
        size = batch["language_id"].shape[0]
        parts: list[Tensor] = []
        base: Tensor | None = None
        if config.episode_text:
            meta = self.encoder(batch["meta_input_ids"], batch["meta_attention_mask"])
            base = self.proj(meta)
            parts.append(meta)
        if config.use_transcript:
            parts += [self._transcript(batch, size), batch["has_transcript"].unsqueeze(-1)]
        if config.use_structured:
            parts += [
                self.language(batch["language_id"]),
                self.duration(batch["duration_bucket"]),
                self.age(batch["age_bucket"]),
            ]
            if self.num_categories:
                cats = batch["categories"]
                parts.append(self.category(cats / cats.sum(dim=-1, keepdim=True).clamp(min=1.0)))
        if config.podcast_id_buckets:
            buckets = torch.remainder(batch["podcast_id"], config.podcast_id_buckets)
            parts.append(self.podcast(buckets))
        out = self.head(torch.cat(parts, dim=-1))
        if base is not None:
            out = out + base
        return F.normalize(out, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        config: TwoTowerConfig,
        contract: FeatureContract,
        text_backbone: nn.Module,
        episode_backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.contract = contract
        query_encoder = TextEncoder(text_backbone)
        if config.share_text_encoder:
            if episode_backbone is not None:
                raise ValueError("episode_backbone is only used when share_text_encoder=False")
            episode_encoder = query_encoder
        else:
            episode_encoder = TextEncoder(episode_backbone or copy.deepcopy(text_backbone))
        self.query_tower = QueryTower(config, contract, query_encoder)
        self.episode_tower = EpisodeTower(config, contract, episode_encoder)

    @classmethod
    def from_pretrained_text(
        cls, contract: FeatureContract, config: TwoTowerConfig | None = None
    ) -> TwoTowerModel:
        """Start from the contract's pretrained text model (hub or cache)."""
        from transformers import AutoModel

        config = config or TwoTowerConfig()
        backbone = AutoModel.from_pretrained(contract.text_model_id)
        return cls(config, contract, backbone)

    # -- inference ---------------------------------------------------------

    def encode_query(self, batch: dict[str, Tensor]) -> Tensor:
        return self.query_tower(batch)

    def encode_episode(self, batch: dict[str, Tensor]) -> Tensor:
        return self.episode_tower(batch)

    @staticmethod
    def score(query_vectors: Tensor, episode_vectors: Tensor) -> Tensor:
        """Row-wise dot product of paired, already-normalized vectors."""
        return (query_vectors * episode_vectors).sum(dim=-1)

    @staticmethod
    def similarity(query_vectors: Tensor, episode_vectors: Tensor) -> Tensor:
        """All-pairs cosine similarity, ``[n_queries, n_episodes]``."""
        return query_vectors @ episode_vectors.transpose(0, 1)

    def logits(self, query_vectors: Tensor, episode_vectors: Tensor) -> Tensor:
        return self.similarity(query_vectors, episode_vectors) / self.config.temperature

    def forward(self, query_batch: dict[str, Tensor], episode_batch: dict[str, Tensor]) -> Tensor:
        return self.logits(self.encode_query(query_batch), self.encode_episode(episode_batch))

    # -- artifacts ---------------------------------------------------------

    def parameter_counts(self) -> dict[str, int]:
        params = list(self.parameters())  # shared modules are counted once
        return {
            "total": sum(p.numel() for p in params),
            "trainable": sum(p.numel() for p in params if p.requires_grad),
        }

    def describe(self) -> dict:
        """Architecture summary recorded in every artifact."""
        encoder = self.query_tower.encoder.backbone.config
        return {
            "architecture": "two-tower",
            "embedding_dim": self.config.embedding_dim,
            "text_model_id": self.contract.text_model_id,
            "text_encoder": {
                "model_type": encoder.model_type,
                "hidden_size": int(encoder.hidden_size),
                "layers": int(encoder.num_hidden_layers),
                "pooling": "mean",
                "shared": self.config.share_text_encoder,
            },
            "model_config": dataclasses.asdict(self.config),
            "feature_contract_version": self.contract.version,
            "feature_contract_sha256": self.contract.fingerprint(),
            "parameters": self.parameter_counts(),
        }

    def save(self, directory: str | Path, featurizer=None, metadata: dict | None = None) -> Path:
        """Write a self-contained artifact: weights, architecture config,
        feature contract, tokenizer (when a featurizer is given), and an
        ``artifact.json`` merging ``describe()`` with caller metadata
        (dataset version, metrics, intended use, ...)."""
        from ml.datasets.build import code_revision

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / CONFIG_FILENAME).write_text(
            json.dumps(dataclasses.asdict(self.config), indent=2, sort_keys=True) + "\n"
        )
        self.contract.save(directory)
        self.query_tower.encoder.backbone.config.save_pretrained(directory / BACKBONE_DIR)
        if not self.config.share_text_encoder:
            self.episode_tower.encoder.backbone.config.save_pretrained(
                directory / EPISODE_BACKBONE_DIR
            )
        torch.save(self.state_dict(), directory / WEIGHTS_FILENAME)
        if featurizer is not None:
            featurizer.save_tokenizer(directory)
        artifact = {
            **self.describe(),
            "created_at": datetime.now(UTC).isoformat(),
            **code_revision(),
            **(metadata or {}),
        }
        (directory / ARTIFACT_FILENAME).write_text(json.dumps(artifact, indent=2) + "\n")
        return directory

    @classmethod
    def load(cls, directory: str | Path, map_location: str | torch.device = "cpu") -> TwoTowerModel:
        from transformers import AutoConfig, AutoModel

        directory = Path(directory)
        config = TwoTowerConfig.from_dict(json.loads((directory / CONFIG_FILENAME).read_text()))
        contract = FeatureContract.load(directory)
        backbone = AutoModel.from_config(AutoConfig.from_pretrained(directory / BACKBONE_DIR))
        episode_backbone = None
        if not config.share_text_encoder:
            episode_backbone = AutoModel.from_config(
                AutoConfig.from_pretrained(directory / EPISODE_BACKBONE_DIR)
            )
        model = cls(config, contract, backbone, episode_backbone)
        state = torch.load(directory / WEIGHTS_FILENAME, map_location=map_location,
                           weights_only=True)
        model.load_state_dict(state, strict=True)
        return model.to(map_location)
