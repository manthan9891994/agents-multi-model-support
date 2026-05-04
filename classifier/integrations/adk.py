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

# Per-callback decision tracking. Pairs before_model_callback (routing) with
# after_model_callback (outcome reporting). Key strategy:
#   1. callback_context.invocation_id   (preferred — stable across the request)
#   2. callback_context.state["_dmr_decision"]  (preferred — survives GC)
#   3. id(callback_context)             (fallback — bounded by an LRU)
# Bounded so unmatched decisions (after_model_callback never fires) don't leak.
import collections as _collections
_PENDING_MAX = 1024
_pending_decisions: "_collections.OrderedDict[str, dict]" = _collections.OrderedDict()
import threading as _threading
_pending_lock = _threading.Lock()


def _pending_key(callback_context) -> str:
    """Pick the most stable key available on a callback_context."""
    # Prefer stable ADK-provided IDs over object identity
    inv_id = getattr(callback_context, "invocation_id", None)
    if inv_id:
        return f"inv:{inv_id}"
    # callback_context.state is a dict in current ADK — store the key there too
    return f"obj:{id(callback_context)}"


def _store_pending(callback_context, payload: dict) -> None:
    key = _pending_key(callback_context)
    # Also stash on state so the after-callback can find it even if id() recycles.
    state = getattr(callback_context, "state", None)
    if isinstance(state, dict):
        state["_dmr_decision_key"] = key
    with _pending_lock:
        _pending_decisions[key] = payload
        _pending_decisions.move_to_end(key)
        # Bound the dict — drop oldest unmatched entries
        while len(_pending_decisions) > _PENDING_MAX:
            _pending_decisions.popitem(last=False)


def _pop_pending(callback_context) -> dict | None:
    state = getattr(callback_context, "state", None)
    state_key = state.get("_dmr_decision_key") if isinstance(state, dict) else None
    keys = [state_key, _pending_key(callback_context)]
    with _pending_lock:
        for k in keys:
            if k and k in _pending_decisions:
                return _pending_decisions.pop(k)
    return None


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

    # Stash the decision so the paired after_model_callback can report the outcome.
    # Bounded LRU + stable key via invocation_id / callback_context.state to
    # prevent leaks if after_model_callback never fires (agent crash).
    import time
    _store_pending(callback_context, {
        "decision_id": decision.decision_id,
        "model":       decision.model_name,
        "task":        task,
        "t0":          time.perf_counter(),
    })

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


def report_model_outcome(callback_context, llm_response):
    """ADK `after_model_callback` — fires after every LLM API call.

    Pair with `dynamic_model_selector` to feed continual-learning telemetry:

        agent = LlmAgent(
            name="MyAgent",
            model="gemini-2.5-flash",
            before_model_callback=dynamic_model_selector,
            after_model_callback=report_model_outcome,
        )
    """
    import time
    pending = _pop_pending(callback_context)
    if pending is None:
        return None

    from classifier import log_outcome, OutcomeRecord
    from classifier.infra.tokenizers import count_tokens

    wall_ms = (time.perf_counter() - pending["t0"]) * 1000

    # Try to extract usage / response text from various ADK response shapes
    tokens_in  = count_tokens(pending["task"], model=pending["model"])
    tokens_out = 0
    success    = True
    error      = None
    try:
        usage = getattr(llm_response, "usage_metadata", None) or {}
        if hasattr(usage, "get"):
            tokens_in  = int(usage.get("prompt_token_count", tokens_in))
            tokens_out = int(usage.get("candidates_token_count", 0))
        else:
            tokens_in  = int(getattr(usage, "prompt_token_count", tokens_in))
            tokens_out = int(getattr(usage, "candidates_token_count", 0))
    except Exception:
        # Fall through with the heuristic count
        for content in (getattr(llm_response, "content", None) or []):
            for part in getattr(content, "parts", []) or []:
                if getattr(part, "text", None):
                    tokens_out += count_tokens(part.text, model=pending["model"])

    log_outcome(OutcomeRecord(
        decision_id=pending["decision_id"],
        tokens_in=tokens_in, tokens_out=tokens_out,
        wall_ms=wall_ms, success=success, error_message=error,
    ))
    return None


# Back-compat alias for the underscore-prefixed name used in earlier examples
_dynamic_model_selector = dynamic_model_selector
