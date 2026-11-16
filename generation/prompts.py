"""Prompt construction.

One template for every strategy: the *only* difference between strategies is whether the
context block is present.  Keeping the template fixed is what makes the routing comparison
meaningful (a retrieval-induced prompt shift cannot be blamed on a different instruction).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

PASSAGE_TEMPLATE = "[{i}] {title}\n{text}"


def build_context(passages: Sequence[Dict[str, Any]], max_context_tokens: int, tokenizer=None) -> Dict[str, Any]:
    """Pack passages into a context block, capped at `max_context_tokens` tokens.

    Whole passages are dropped first; only the final passage is truncated.
    """
    kept: List[Dict[str, Any]] = []
    used = 0
    blocks: List[str] = []
    for i, p in enumerate(passages):
        block = PASSAGE_TEMPLATE.format(i=i + 1, title=p["title"], text=p["text"])
        n_tok = len(tokenizer.encode(block, add_special_tokens=False)) if tokenizer is not None else max(1, len(block) // 4)
        if used + n_tok > max_context_tokens:
            remaining = max_context_tokens - used
            if remaining > 32:  # truncate the last passage on a word boundary
                words = p["text"].split()
                keep_words = max(8, int(remaining * 0.75 / 1.3))
                block = PASSAGE_TEMPLATE.format(i=i + 1, title=p["title"], text=" ".join(words[:keep_words]))
                kept.append(p)
                blocks.append(block)
            break
        kept.append(p)
        blocks.append(block)
        used += n_tok
    return {"context": "\n\n".join(blocks), "passages": kept, "n_passages": len(kept), "approx_ctx_tokens": used}


def build_user_message(question: str, context: Optional[str]) -> str:
    """The user turn.  Identical for every strategy except for the context block."""
    if context:
        return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    return f"Question: {question}\nAnswer:"


def build_chat_prompt(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
