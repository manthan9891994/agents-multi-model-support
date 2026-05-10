"""Pluggable tokenizers for accurate cost estimation.

Backends (in priority order):
  - tiktoken (OpenAI / GPT-* / generic English)
  - anthropic SDK (Claude)
  - HuggingFace tokenizers (anything else if installed)
  - wordcount (always-available fallback)

Use:
    from classifier.infra.tokenizers import count_tokens
    n = count_tokens("hello world", model="gpt-4o-mini")
"""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Custom tokenizer registry: {model_or_provider: callable(text) -> int}
_CUSTOM_TOKENIZERS: dict[str, Callable[[str], int]] = {}

# Cached tiktoken encoders keyed by model name. None means "no encoder for this model".
_TIKTOKEN_CACHE: dict[str, Callable[[str], int] | None] = {}


def register_tokenizer(name: str, fn: Callable[[str], int]) -> None:
    """Register a tokenizer for a model or model-prefix.

    `name` matches by:
      - exact match against full model name, OR
      - prefix match (so "gpt-" catches all GPT models).
    """
    _CUSTOM_TOKENIZERS[name] = fn


def _wordcount(text: str) -> int:
    return max(1, len(text.split()))


def _tiktoken(model: str) -> Callable[[str], int] | None:
    if model in _TIKTOKEN_CACHE:
        return _TIKTOKEN_CACHE[model]
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        fn = lambda text: len(enc.encode(text))  # noqa: E731
    except ImportError:
        fn = None
    _TIKTOKEN_CACHE[model] = fn
    return fn


def _anthropic_tokenizer() -> Callable[[str], int] | None:
    try:
        from anthropic import Anthropic  # noqa: F401

        # Claude SDK exposes count_tokens via client; expensive — fall back to char/3.5
        return lambda text: max(1, len(text) // 4)
    except ImportError:
        return None


def count_tokens(text: str, model: str = "") -> int:
    """Best-effort token count for `text` under `model`'s tokenizer.

    Priority:
      1. Custom registered tokenizer matching model exactly
      2. Custom registered tokenizer matching prefix
      3. tiktoken (if installed) for OpenAI-family models
      4. anthropic char/4 heuristic for Claude
      5. word-count fallback
    """
    if not text:
        return 0
    # Exact match
    if model in _CUSTOM_TOKENIZERS:
        return _CUSTOM_TOKENIZERS[model](text)
    # Prefix match
    for prefix, fn in _CUSTOM_TOKENIZERS.items():
        if model.startswith(prefix):
            return fn(text)

    # OpenAI / GPT family — use tiktoken
    if model.startswith(("gpt-", "o1-", "o3-")):
        tk = _tiktoken(model)
        if tk:
            return tk(text)

    # Anthropic / Claude
    if model.startswith("claude-"):
        ant = _anthropic_tokenizer()
        if ant:
            return ant(text)

    # Generic fallback
    return _wordcount(text)
