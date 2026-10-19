"""Default vocab-population recipe used by Qwen3/DeepSeek adapters.

BPE-style GGUF tokenizer: ``tokenizer.ggml.model = "gpt2"`` plus the
HF token list and ``SpecialVocab(load_merges=True)`` for merges + chat
template.  Lifted out of ``gguf_writer.py`` so per-arch adapters can
override vocab population when their tokenizer is not GPT-2 BPE.
"""
from __future__ import annotations


def populate_gpt2_bpe_vocab(writer, model_dir, vocab_size: int | None = None) -> None:
    from gguf import SpecialVocab
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    vocab = tok.get_vocab()
    added = tok.get_added_vocab()
    n_tok = max(vocab.values()) + 1
    target = max(n_tok, int(vocab_size or n_tok))
    inv = [""] * target
    for tok_str, idx in vocab.items():
        inv[idx] = tok_str
    GGUF_TOKEN_TYPE_NORMAL = 1
    GGUF_TOKEN_TYPE_CONTROL = 3
    GGUF_TOKEN_TYPE_UNUSED = 5
    token_types = [GGUF_TOKEN_TYPE_NORMAL] * target
    for tok_str, idx in added.items():
        token_types[idx] = GGUF_TOKEN_TYPE_CONTROL
    for i in range(n_tok, target):
        inv[i] = f"[PAD{i}]"
        token_types[i] = GGUF_TOKEN_TYPE_UNUSED
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list([s.encode("utf-8") for s in inv])
    writer.add_token_types(token_types)
    sv = SpecialVocab(model_dir, load_merges=True)
    sv.add_to_gguf(writer)
