"""HuggingFace smolagents integration — dynamic model routing for smolagents Agents.

Install with:
    pip install smolagents

smolagents typically uses LiteLLM-style provider-qualified model IDs.

Two patterns:

1. **`get_model(task)`** — returns a smolagents `LiteLLMModel` configured for the routed model.

    from smolagents import CodeAgent
    from classifier.integrations.smolagents import get_model

    agent = CodeAgent(tools=[...], model=get_model("Solve this puzzle"))

2. **`DynamicModel(provider=...)`** — wrapper that classifies each call's prompt
   and dispatches to the routed underlying model.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# DMR provider -> LiteLLM provider prefix
_PROVIDER_PREFIX = {
    "google":    "gemini",
    "anthropic": "anthropic",
    "openai":    "openai",
    "groq":      "groq",
    "mistral":   "mistral",
    "cohere":    "cohere",
}


def _qualify(model_name: str, provider: str) -> str:
    if "/" in model_name:
        return model_name
    prefix = _PROVIDER_PREFIX.get(provider, provider)
    return f"{prefix}/{model_name}"


def _route_one(task: str, provider: Optional[str], fallback_model: Optional[str]) -> str:
    from classifier import classify_task
    from classifier.core.exceptions import ClassificationError
    from classifier.infra.config import settings

    resolved = provider or settings.default_provider
    try:
        decision = classify_task(task, provider=resolved)
        model = decision.model_name
        logger.info(
            "smolagents: routed [%s | %s/%s] -> %s",
            decision.tier.value.upper(),
            decision.task_type.value, decision.complexity.value, model,
        )
    except ClassificationError:
        if not fallback_model:
            raise
        model = fallback_model
    return _qualify(model, resolved)


def get_model(
    task: str,
    *,
    provider: Optional[str] = None,
    fallback_model: Optional[str] = None,
    **model_kwargs: Any,
) -> Any:
    """Classify a task and return a smolagents `LiteLLMModel` pinned to the routed model."""
    try:
        from smolagents import LiteLLMModel
    except ImportError as exc:
        raise ImportError(
            "smolagents is not installed. Install: pip install smolagents"
        ) from exc

    qualified = _route_one(task, provider, fallback_model)
    return LiteLLMModel(model_id=qualified, **model_kwargs)


class DynamicModel:
    """smolagents-compatible model that classifies each call's prompt and dispatches.

    Usage:
        from smolagents import CodeAgent
        from classifier.integrations.smolagents import DynamicModel

        agent = CodeAgent(tools=[...], model=DynamicModel(provider="google"))
    """

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        fallback_model: Optional[str] = None,
        **model_kwargs: Any,
    ) -> None:
        self._provider = provider
        self._fallback_model = fallback_model
        self._model_kwargs = model_kwargs

    @staticmethod
    def _as_text(messages_or_str: Any) -> str:
        if isinstance(messages_or_str, str):
            return messages_or_str
        if isinstance(messages_or_str, list):
            for m in reversed(messages_or_str):
                if isinstance(m, dict):
                    content = m.get("content", "")
                    if isinstance(content, list):
                        # smolagents content blocks
                        for block in reversed(content):
                            if isinstance(block, dict) and block.get("type") == "text":
                                return block.get("text", "")
                    if content:
                        return str(content)
                if hasattr(m, "content"):
                    return str(m.content)
        return str(messages_or_str)

    def _build(self, prompt: str) -> Any:
        return get_model(
            prompt,
            provider=self._provider,
            fallback_model=self._fallback_model,
            **self._model_kwargs,
        )

    def __call__(self, messages, **kwargs):
        prompt = self._as_text(messages)
        return self._build(prompt)(messages, **kwargs)

    def generate(self, messages, **kwargs):
        prompt = self._as_text(messages)
        return self._build(prompt).generate(messages, **kwargs)
