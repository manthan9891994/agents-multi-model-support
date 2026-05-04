"""CrewAI integration — pick the right LLM for each Task before the Agent runs.

Install with:
    pip install 'dynamic-model-router[crewai]'   # or pip install crewai

CrewAI agents accept an `llm` parameter. This module gives you two ways to wire
in dynamic routing:

1. **`pick_llm_for_task(task_description)`** — returns the LLM instance the
   router would choose. Use it when constructing Agents one task at a time.

2. **`DynamicLLM(provider=...)`** — a thin LangChain-compatible wrapper that
   classifies each call's prompt and dispatches to the right underlying model.
   Drop it in as `Agent(llm=DynamicLLM())`.

Both share the same routing logic — pick whichever fits your code shape.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def pick_llm_for_task(
    task_description: str,
    *,
    provider: Optional[str] = None,
    fallback_model: Optional[str] = None,
):
    """Classify a task and return a CrewAI-compatible LLM instance pre-configured
    with the right model.

    Args:
        task_description: The task / goal text from your CrewAI Task.
        provider:         Override the default provider ("google" | "anthropic" | "openai").
        fallback_model:   Used only when classification raises (extreme failure mode).

    Returns:
        A `crewai.LLM` instance pinned to the router-selected model.

    Raises:
        ImportError if `crewai` is not installed.
    """
    try:
        from crewai import LLM
    except ImportError as exc:
        raise ImportError(
            "CrewAI is not installed. Install with: pip install crewai"
        ) from exc

    from classifier import classify_task
    from classifier.core.exceptions import ClassificationError
    from classifier.infra.config import settings

    resolved_provider = provider or settings.default_provider

    try:
        decision = classify_task(task_description, provider=resolved_provider)
        model_name = decision.model_name
        logger.info(
            "CrewAI: routed [%s | %s/%s | conf=%.2f] → %s",
            decision.tier.value.upper(),
            decision.task_type.value, decision.complexity.value,
            decision.confidence, model_name,
        )
    except ClassificationError as exc:
        if fallback_model:
            logger.warning("CrewAI: classification failed (%s) — using fallback %s",
                           exc, fallback_model)
            model_name = fallback_model
        else:
            raise

    # CrewAI's LLM class accepts a model string and routes via litellm under the hood
    return LLM(model=_qualify_model(model_name, resolved_provider))


def _qualify_model(model_name: str, provider: str) -> str:
    """CrewAI / litellm uses provider-prefixed model names like 'gemini/gemini-2.5-flash'."""
    prefix_map = {
        "google":    "gemini",
        "anthropic": "anthropic",
        "openai":    "openai",
    }
    prefix = prefix_map.get(provider, provider)
    if model_name.startswith(f"{prefix}/"):
        return model_name
    return f"{prefix}/{model_name}"


class DynamicLLM:
    """Drop-in LLM wrapper for CrewAI that classifies each call's prompt
    and dispatches to the router-selected underlying model.

    Use when you want the same Agent to handle different complexity tasks:

        from classifier.integrations.crewai import DynamicLLM
        from crewai import Agent

        agent = Agent(role="Researcher", goal="...", llm=DynamicLLM())
    """

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        fallback_model: str = "gemini/gemini-2.5-flash",
        report_outcomes: bool = True,
    ):
        self._provider = provider
        self._fallback_model = fallback_model
        self._report_outcomes = report_outcomes
        self._cache: dict[str, Any] = {}

    def _classify_and_build(self, task_text: str):
        """Returns (llm, decision) so we can report outcome after the call."""
        from classifier import classify_task
        from classifier.core.exceptions import ClassificationError
        from classifier.infra.config import settings
        try:
            from crewai import LLM
        except ImportError as exc:
            raise ImportError("CrewAI is not installed. Install with: pip install crewai") from exc

        provider = self._provider or settings.default_provider
        try:
            decision = classify_task(task_text, provider=provider)
            model = decision.model_name
        except ClassificationError:
            if not self._fallback_model:
                raise
            decision = None
            model = self._fallback_model
        return LLM(model=_qualify_model(model, provider)), decision

    def call(self, messages, *args, **kwargs):
        """Entry point CrewAI uses. Inspects messages, classifies, dispatches."""
        import time
        task_text = self._extract_task_text(messages)
        llm, decision = self._classify_and_build(task_text)
        t0 = time.perf_counter()
        try:
            response = llm.call(messages, *args, **kwargs)
            if self._report_outcomes and decision is not None:
                from classifier import log_outcome, OutcomeRecord
                # CrewAI's LLM.call typically returns a string; we don't have token counts.
                # Approximate with tokenizer for analytics.
                from classifier.infra.tokenizers import count_tokens
                tokens_in  = count_tokens(task_text, model=decision.model_name)
                tokens_out = count_tokens(str(response), model=decision.model_name)
                log_outcome(OutcomeRecord(
                    decision_id=decision.decision_id,
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    wall_ms=(time.perf_counter() - t0) * 1000,
                    success=True,
                ))
            return response
        except Exception as exc:
            if self._report_outcomes and decision is not None:
                from classifier import log_outcome, OutcomeRecord
                log_outcome(OutcomeRecord(
                    decision_id=decision.decision_id,
                    wall_ms=(time.perf_counter() - t0) * 1000,
                    success=False, error_message=str(exc),
                ))
            raise

    @staticmethod
    def _extract_task_text(messages) -> str:
        """Get the most recent user message from a chat-format message list."""
        if isinstance(messages, str):
            return messages
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return str(msg.get("content", ""))
            # Fallback: stringify the last message
            if messages:
                return str(messages[-1])
        return str(messages)
