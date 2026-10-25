"""Query-side model inputs without PyTorch, for the model server.

``TwoTowerQueryInputs`` reproduces ``ml.features.Featurizer.query_batch``
(token ids, attention mask, language id, intent flags) from the feature
contract and the artifact's ``tokenizer.json``, using the ``tokenizers``
library directly and NumPy arrays. ``E5QueryInputs`` does the same for the
pretrained E5 query encoder (``query: <text>``). The parity checks compare
both against the training-side code on golden queries, token for token.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from ml.embeddings.text import build_query
from ml.features.contract import FeatureContract
from ml.features.text import build_query_text

E5_MAX_TOKENS = 512


PAD_TOKENS = ("<pad>", "[PAD]")  # XLM-R / sentencepiece, BERT wordpiece


def _tokenizer(path: Path, max_length: int, pad_to: int | None) -> Tokenizer:
    tok = Tokenizer.from_file(str(path))
    tok.enable_truncation(max_length=max_length)
    if pad_to is None:
        tok.no_padding()
        return tok
    pad = next((t for t in PAD_TOKENS if tok.token_to_id(t) is not None), None)
    if pad is None:
        raise ValueError(f"{path}: no padding token among {PAD_TOKENS}")
    tok.enable_padding(length=pad_to, pad_id=tok.token_to_id(pad), pad_token=pad)
    return tok


class TwoTowerQueryInputs:
    """Inputs for the two-tower query tower, padded to the contract's
    ``max_query_tokens`` exactly as training pads them."""

    def __init__(self, contract: FeatureContract, tokenizer_json: Path):
        self.contract = contract
        self.tokenizer = _tokenizer(tokenizer_json, contract.max_query_tokens,
                                    contract.max_query_tokens)

    def __call__(self, queries: list[dict]) -> dict[str, np.ndarray]:
        contract = self.contract
        encoded = self.tokenizer.encode_batch(
            [build_query_text(contract, q.get("query_text", "")) for q in queries])
        return {
            "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
            "language_id": np.array([contract.language_id(q.get("language")) for q in queries],
                                    dtype=np.int64),
            "intent": np.array([[float(bool(q.get("language"))),
                                 float(bool(q.get("max_duration_seconds"))),
                                 float(bool(q.get("published_after"))),
                                 float(bool(q.get("no_explicit")))] for q in queries],
                               dtype=np.float32).reshape(len(queries),
                                                         len(contract.intent_features)),
        }


class E5QueryInputs:
    """Inputs for the pretrained E5 query encoder: one query at a time, no
    padding, truncated at the encoder's 512 tokens (as sentence-transformers
    does)."""

    def __init__(self, tokenizer_json: Path):
        self.tokenizer = _tokenizer(tokenizer_json, E5_MAX_TOKENS, None)

    def __call__(self, query_text: str) -> dict[str, np.ndarray]:
        e = self.tokenizer.encode(build_query(query_text))
        return {"input_ids": np.array([e.ids], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask], dtype=np.int64)}
