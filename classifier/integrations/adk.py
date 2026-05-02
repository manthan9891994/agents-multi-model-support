"""Google ADK integration — drop the router into any LlmAgent via callback.

Install with:
    pip install 'dynamic-model-router[adk]'

Usage:
    from google.adk.agents import LlmAgent
    from classifier.integrations.adk import dynamic_model_selector

    agent = LlmAgent(
        name="MyAgent",
        model="gemini-2.5-flash",   # placeholder — replaced per-request
        before_model_callback=dynamic_model_selector,
    )

The callback inspects each `LlmRequest`, extracts the user's task + context
signals (call number, errors, multimodal data, tool count), classifies via
the router, and overwrites `llm_request.model` with the selected model.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_call_counter: dict[str, int] = {}
_ERROR_SIGNALS = {"error", "exception", "traceback", "failed", "failure", "timeout", "refused"}


def _extract_context_signals(llm_request, agent_name: str):
    """Inspect an ADK LlmRequest and produce ContextSignals for the router."""
    from classifier.core.types import ContextSignals

    _call_counter[agent_name] = _call_counter.get(agent_name, 0) + 1
    call_number = _call_counter[agent_name]

    total_chars   = 0
    last_role     = "user"
    last_non_user = ""
    has_multimodal = False

    for content in llm_request.contents:
        last_role = content.role or "user"
        for part in (content.parts or []):
            text = getattr(part, "text", "") or ""
            total_chars += len(text)
            if content.role in ("tool", "model"):
                last_non_user = text
            if (
                getattr(part, "inline_data", None) is not None
                or getattr(part, "file_data", None) is not None
            ):
                has_multimodal = True

    has_error = False
    if last_non_user:
        lower = last_non_user[-2000:].lower()
        has_error = any(sig in lower for sig in _ERROR_SIGNALS)

    available_tools = len(getattr(llm_request, "tools", None) or [])

    return ContextSignals(
        total_context_tokens=total_chars // 4,
        call_number=call_number,
        has_error=has_error,
        last_role=last_role,
        has_multimodal=has_multimodal,
        available_tools=available_tools,
    )


def dynamic_model_selector(callback_context, llm_request):
    """ADK `before_model_callback` — fires before every LLM API call.

    Mutates `llm_request.model` to the router-selected model. Returns None
    so ADK proceeds with the (now-modified) request.
    """
    from classifier import classify_task
    from classifier.core.exceptions import ClassificationError
    from classifier.infra.config import settings

    # Extract the user's task from the latest user message
    task = ""
    for content in reversed(llm_request.contents):
        if content.role == "user" and content.parts:
            task = content.parts[0].text or ""
            break

    if not task:
        logger.warning("dynamic_model_selector: no user message found — keeping default model.")
        return None

    agent_name = getattr(callback_context, "agent_name", "Agent")
    ctx_signals = _extract_context_signals(llm_request, agent_name=agent_name)

    try:
        decision = classify_task(
            task,
            provider=settings.default_provider,
            context_signals=ctx_signals,
        )
    except ClassificationError as exc:
        logger.error("dynamic_model_selector: classification failed (%s) — keeping default model.", exc)
        return None

    original = llm_request.model
    llm_request.model = decision.model_name

    logger.info(
        "Model selected | %s => %s [%s | %s | %s | call=%d | ctx_tokens=%d%s%s]",
        original, decision.model_name,
        decision.tier.value.upper(),
        decision.task_type.value, decision.complexity.value,
        ctx_signals.call_number, ctx_signals.total_context_tokens,
        " | PII" if decision.compliance_flag else "",
        f" | tools={ctx_signals.available_tools}" if ctx_signals.available_tools else "",
    )
    return None


# Back-compat alias for the underscore-prefixed name used in earlier examples
_dynamic_model_selector = dynamic_model_selector
