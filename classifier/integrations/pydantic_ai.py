"""Pydantic AI integration — dynamic model routing for Pydantic AI Agents.

Install with:
    pip install pydantic-ai

Pydantic AI takes models as strings of the form `"provider:model-name"`.
Two patterns:

1. **`get_model_string(task)`** — returns a provider-qualified model string.

    from pydantic_ai import Agent
    from classifier.integrations.pydantic_ai import get_model_string

    model = get_model_string("Summarise this contract", provider="google")
    agent = Agent(model=model)

2. **`get_agent(task, **agent_kwargs)`** — returns a constructed Agent.

    agent = get_agent("Summarise this contract", system_prompt="...")
    result = agent.run_sync("Summarise...")
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# DMR provider name -> Pydantic AI provider prefix
_PROVIDER_PREFIX = {
    "google": "google-gla",
    "anthropic": "anthropic",
    "openai": "openai",
    "groq": "groq",
    "mistral": "mistral",
    "cohere": "cohere",
}


def _qualify(model_name: str, provider: str) -> str:
    """Return Pydantic-AI-shaped 'provider:model' string."""
    prefix = _PROVIDER_PREFIX.get(provider, provider)
    if ":" in model_name:  # already qualified
        return model_name
    return f"{prefix}:{model_name}"


def get_model_string(
    task: str,
    *,
    provider: str | None = None,
    fallback_model: str | None = None,
) -> str:
    """Classify a task and return the Pydantic-AI provider-qualified model string."""
    from classifier import classify_task
    from classifier.core.exceptions import ClassificationError
    from classifier.infra.config import settings

    resolved_provider = provider or settings.default_provider
    try:
        decision = classify_task(task, provider=resolved_provider)
        model_name = decision.model_name
        logger.info(
            "PydanticAI: routed [%s | %s/%s] -> %s",
            decision.tier.value.upper(),
            decision.task_type.value,
            decision.complexity.value,
            model_name,
        )
    except ClassificationError:
        if not fallback_model:
            raise
        model_name = fallback_model

    return _qualify(model_name, resolved_provider)


def get_agent(
    task: str,
    *,
    provider: str | None = None,
    fallback_model: str | None = None,
    **agent_kwargs: Any,
) -> Any:
    """Classify a task and return a Pydantic-AI Agent pinned to the routed model."""
    try:
        from pydantic_ai import Agent
    except ImportError as exc:
        raise ImportError("pydantic-ai is not installed. Install: pip install pydantic-ai") from exc

    model_string = get_model_string(task, provider=provider, fallback_model=fallback_model)
    return Agent(model_string, **agent_kwargs)
