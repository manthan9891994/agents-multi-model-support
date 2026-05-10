"""Haystack integration — dynamic model routing for Haystack pipelines.

Install with:
    pip install haystack-ai
    pip install google-ai-haystack          # for Google Gemini
    pip install anthropic-haystack          # for Anthropic
    # OpenAI generator ships in haystack-ai itself

One pattern:

**`get_generator(task)`** — returns the right Haystack generator instance.

    from haystack import Pipeline
    from classifier.integrations.haystack import get_generator

    gen = get_generator("Summarise this contract")
    pipe = Pipeline()
    pipe.add_component("llm", gen)
    result = pipe.run({"llm": {"prompt": "Summarise this contract"}})
"""
from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Provider -> (package, class name) for the appropriate Haystack generator
_PROVIDER_MAP = {
    "google":    ("haystack_integrations.components.generators.google_genai", "GoogleGenAIGenerator"),
    "anthropic": ("haystack_integrations.components.generators.anthropic",    "AnthropicGenerator"),
    "openai":    ("haystack.components.generators",                            "OpenAIGenerator"),
}


def _build_generator(model_name: str, provider: str, **kwargs: Any) -> Any:
    if provider not in _PROVIDER_MAP:
        raise ValueError(
            f"Provider '{provider}' not supported by Haystack integration. "
            f"Choose from: {sorted(_PROVIDER_MAP)}"
        )
    pkg, cls_name = _PROVIDER_MAP[provider]
    try:
        mod = importlib.import_module(pkg)
        cls = getattr(mod, cls_name)
    except ImportError as exc:
        # Map back to the user-facing extras hint
        hint = {
            "google":    "google-ai-haystack",
            "anthropic": "anthropic-haystack",
            "openai":    "haystack-ai",
        }.get(provider, pkg.replace("_", "-"))
        raise ImportError(
            f"Haystack provider package '{pkg}' is not installed. "
            f"Install with: pip install {hint}"
        ) from exc

    return cls(model=model_name, **kwargs)


def get_generator(
    task: str,
    *,
    provider: str | None = None,
    fallback_model: str | None = None,
    **generator_kwargs: Any,
) -> Any:
    """Classify a task and return the Haystack generator pinned to the routed model."""
    from classifier import classify_task
    from classifier.core.exceptions import ClassificationError
    from classifier.infra.config import settings

    resolved_provider = provider or settings.default_provider
    try:
        decision = classify_task(task, provider=resolved_provider)
        model_name = decision.model_name
        logger.info(
            "Haystack: routed [%s | %s/%s] -> %s",
            decision.tier.value.upper(),
            decision.task_type.value, decision.complexity.value, model_name,
        )
    except ClassificationError:
        if not fallback_model:
            raise
        model_name = fallback_model

    return _build_generator(model_name, resolved_provider, **generator_kwargs)
