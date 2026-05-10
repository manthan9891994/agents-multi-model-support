"""AutoGen / OpenAI Agents SDK integration — dynamic model routing.

Supports two popular agent frameworks:

**AutoGen (microsoft/autogen)**
    pip install pyautogen

    from classifier.integrations.autogen import get_autogen_llm_config
    config = get_autogen_llm_config("Analyse quarterly revenue trends")
    agent = AssistantAgent("analyst", llm_config=config)

**OpenAI Agents SDK (openai/openai-agents-python)**
    pip install openai-agents

    from classifier.integrations.autogen import get_openai_agent_model
    model = get_openai_agent_model("Write a compliance report")
    agent = Agent(name="Writer", model=model)

Both return the model string / config dict that the respective framework needs —
no wrapping of internal classes so you retain full framework behaviour.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Maps provider → API-key env var name (used in AutoGen llm_config)
_PROVIDER_API_KEY_ENV = {
    "google":    "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
}

# Maps provider → AutoGen api_type string
_AUTOGEN_API_TYPE = {
    "google":    "google",
    "anthropic": "anthropic",
    "openai":    "openai",
}


def _classify(task: str, provider: str) -> tuple[str, Any]:
    """Return (model_name, decision). Shared by all helpers."""
    from classifier import classify_task
    decision = classify_task(task, provider=provider)
    logger.info(
        "AutoGen: routed [%s | %s/%s | conf=%.2f] → %s",
        decision.tier.value.upper(),
        decision.task_type.value, decision.complexity.value,
        decision.confidence, decision.model_name,
    )
    return decision.model_name, decision


# ── AutoGen ──────────────────────────────────────────────────────────────────

def get_autogen_llm_config(
    task: str,
    *,
    provider: str | None = None,
    fallback_model: str | None = None,
    extra_config: dict | None = None,
) -> dict:
    """Return an AutoGen-compatible `llm_config` dict for the routed model.

    Args:
        task:           Task description to classify.
        provider:       "google" | "anthropic" | "openai". Default: DEFAULT_PROVIDER.
        fallback_model: Used only on classification failure.
        extra_config:   Additional keys merged into llm_config (e.g. {"temperature": 0}).

    Returns:
        Dict suitable for AssistantAgent(llm_config=...).

    Example:
        from autogen import AssistantAgent
        from classifier.integrations.autogen import get_autogen_llm_config

        config = get_autogen_llm_config("Summarise this earnings call transcript")
        agent = AssistantAgent("summariser", llm_config=config)
    """
    import os

    from classifier.core.exceptions import ClassificationError
    from classifier.infra.config import settings

    resolved = provider or settings.default_provider

    try:
        model_name, decision = _classify(task, resolved)
    except ClassificationError as exc:
        if fallback_model:
            logger.warning("AutoGen: classification failed (%s) — fallback to %s", exc, fallback_model)
            model_name = fallback_model
        else:
            raise

    api_key_env = _PROVIDER_API_KEY_ENV.get(resolved, "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env, "")

    config_list = [{
        "model":    model_name,
        "api_key":  api_key,
        "api_type": _AUTOGEN_API_TYPE.get(resolved, resolved),
    }]
    llm_config: dict = {"config_list": config_list}
    if extra_config:
        llm_config.update(extra_config)
    return llm_config


# ── OpenAI Agents SDK ─────────────────────────────────────────────────────────

def get_openai_agent_model(
    task: str,
    *,
    provider: str | None = None,
    fallback_model: str | None = None,
) -> str:
    """Return a model name string for the OpenAI Agents SDK.

    Args:
        task:           Task description to classify.
        provider:       "google" | "anthropic" | "openai". Default: DEFAULT_PROVIDER.
        fallback_model: Used only on classification failure.

    Returns:
        Model name string (e.g. "gpt-4o", "claude-opus-4-7").

    Example:
        from agents import Agent
        from classifier.integrations.autogen import get_openai_agent_model

        model = get_openai_agent_model("Implement a rate limiter in Python")
        agent = Agent(name="Coder", model=model, instructions="You are a senior engineer.")
    """
    from classifier.core.exceptions import ClassificationError
    from classifier.infra.config import settings

    resolved = provider or settings.default_provider

    try:
        model_name, _ = _classify(task, resolved)
    except ClassificationError as exc:
        if fallback_model:
            logger.warning("OpenAI Agents: classification failed (%s) — fallback to %s", exc, fallback_model)
            model_name = fallback_model
        else:
            raise

    return model_name


# ── DynamicAutoGenAgent helper ────────────────────────────────────────────────

class DynamicModelRouter:
    """Utility class that re-classifies each new task and returns updated configs.

    Useful when you're reusing one agent definition but the task changes between
    runs — call `.llm_config(task)` to get a fresh config each time.

    Example:
        router = DynamicModelRouter(provider="openai")

        for task in tasks:
            agent = AssistantAgent("worker", llm_config=router.llm_config(task))
            agent.initiate_chat(...)
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        fallback_model: str | None = None,
    ) -> None:
        self._provider = provider
        self._fallback_model = fallback_model

    def llm_config(self, task: str, **extra_config) -> dict:
        """AutoGen llm_config for this task."""
        return get_autogen_llm_config(
            task,
            provider=self._provider,
            fallback_model=self._fallback_model,
            extra_config=extra_config or None,
        )

    def model(self, task: str) -> str:
        """Model name string for OpenAI Agents SDK."""
        return get_openai_agent_model(
            task,
            provider=self._provider,
            fallback_model=self._fallback_model,
        )
