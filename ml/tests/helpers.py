"""Offline stand-ins for the pretrained text model used across model tests."""

from pathlib import Path

VOCAB = (
    "[PAD] [UNK] [CLS] [SEP] [MASK] passage query : , . the a of and about show episode "
    "news science comedy history space rockets launch cooking bread transcript hello world "
    "publisher categories acme topic number"
).split()


def tiny_tokenizer(directory: Path):
    from transformers import BertTokenizerFast

    vocab_file = Path(directory) / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB) + "\n")
    return BertTokenizerFast(vocab_file=str(vocab_file), model_max_length=128)


def tiny_backbone():
    from transformers import BertConfig, BertModel

    return BertModel(BertConfig(
        vocab_size=len(VOCAB), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=128, pad_token_id=0,
    ))
