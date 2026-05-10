"""DSPy integration — dynamic model routing for DSPy programs.

Install with:
    pip install dspy

DSPy uses LiteLLM-style provider-qualified model names. Two patterns:

1. **`get_lm(task)`** — returns a `dspy.LM` configured for the routed model.

    import dspy
    from classifier.integrations.dspy import get_lm

    dspy.configure(lm=get_lm("Generate code to compute Fibonacci"))

2. **`route(task)`** — returns a context manager that swaps `dspy.settings.lm`
   for a single block of execution.

    with route("Translate to French"):
        out = my_module(text="Hello")
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# DMR provider -> LiteLLM provider prefix (DSPy uses LiteLLM under the hood)
_PROVIDER_PREFIX = {
    "google":    "gemini",
    "anthropic": "anthropic",
    "openai":    "openai",
    "groq":      "groq",
    "mistral":   "mistral",
    "cohere":    "cohere",
    "bedrock":   "bedrock",
}


def _qualify(model_name: str, provider: str) -> str:
    if "/" in model_name:
        return model_name
    prefix = _PROVIDER_PREFIX.get(provider, provider)
    return f"{prefix}/{model_name}"


def _route_one(task: str, provider: str | None, fallback_model: str | None) -> str:
    from classifier import classify_task
    from classifier.core.exceptions import ClassificationError
    from classifier.infra.config import settings

    resolved = provider or settings.default_provider
    try:
        decision = classify_task(task, provider=resolved)
        model = decision.model_name
        logger.info(
            "DSPy: routed [%s | %s/%s] -> %s",
            decision.tier.value.upper(),
            decision.task_type.value, decision.complexity.value, model,
        )
    except ClassificationError:
        if not fallback_model:
            raise
        model = fallback_model
    return _qualify(model, resolved)


def get_lm(
    task: str,
    *,
    provider: str | None = None,
    fallback_model: str | None = None,
    **lm_kwargs: Any,
) -> Any:
    """Return a `dspy.LM` configured for the routed model."""
    try:
        import dspy
    except ImportError as exc:
        raise ImportError("dspy is not installed. Install: pip install dspy") from exc

    qualified = _route_one(task, provider, fallback_model)
    return dspy.LM(qualified, **lm_kwargs)


@contextmanager
def route(
    task: str,
    *,
    provider: str | None = None,
    fallback_model: str | None = None,
    **lm_kwargs: Any,
):
    """Context manager that temporarily swaps `dspy.settings.lm` to the routed model.

    Usage:
        with route("Hard task"):
            output = my_dspy_module(input="...")
    """
    import dspy
    new_lm  = get_lm(task, provider=provider, fallback_model=fallback_model, **lm_kwargs)
    prev_lm = dspy.settings.lm
    dspy.configure(lm=new_lm)
    try:
        yield new_lm
    finally:
        dspy.configure(lm=prev_lm)


def get_model_string(
    task: str,
    *,
    provider: str | None = None,
    fallback_model: str | None = None,
) -> str:
    """Just the qualified `provider/model` string (when you need raw access)."""
    return _route_one(task, provider, fallback_model)
