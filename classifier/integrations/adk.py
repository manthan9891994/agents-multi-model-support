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

import collections as _collections
import logging
import threading as _threading

logger = logging.getLogger(__name__)

_call_counter: dict[str, int] = {}
_ERROR_SIGNALS = {"error", "exception", "traceback", "failed", "failure", "timeout", "refused"}

# Per-callback decision tracking. Pairs before_model_callback (routing) with
# after_model_callback (outcome reporting). Key strategy:
#   1. callback_context.invocation_id   (preferred — stable across the request)
#   2. callback_context.state["_dmr_decision"]  (preferred — survives GC)
#   3. id(callback_context)             (fallback — bounded by an LRU)
# Bounded so unmatched decisions (after_model_callback never fires) don't leak.
_PENDING_MAX = 1024
_pending_decisions: _collections.OrderedDict[str, dict] = _collections.OrderedDict()

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

    total_chars = 0
    last_role = "user"
    last_non_user = ""
    has_multimodal = False

    for content in llm_request.contents:
        last_role = content.role or "user"
        for part in content.parts or []:
            text = getattr(part, "text", "") or ""
            total_chars += len(text)
            if content.role in ("tool", "model"):
                last_non_user = text
            if getattr(part, "inline_data", None) is not None or getattr(part, "file_data", None) is not None:
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


def _build_agent_ctx(callback_context, llm_request):
    """Translate an ADK LlmRequest into the framework-neutral AgentCallContext.
    Counts tool-output (function_response) tokens, not just text, and uses the
    invocation/session id as the routing scope key."""
    from classifier.integrations._agentic import AgentCallContext

    task = ""
    history: list[dict] = []
    total_chars = 0
    last_role = "user"
    had_error = False
    for content in llm_request.contents or []:
        role = getattr(content, "role", None) or "user"
        last_role = role
        first_text = ""
        for part in getattr(content, "parts", None) or []:
            t = getattr(part, "text", "") or ""
            if t:
                total_chars += len(t)
                first_text = first_text or t
            fr = getattr(part, "function_response", None)
            if fr is not None:
                s = str(getattr(fr, "response", "") or "")
                total_chars += len(s)
                last_role = "tool"
                if any(k in s.lower() for k in _ERROR_SIGNALS):
                    had_error = True
            fc = getattr(part, "function_call", None)
            if fc is not None:
                total_chars += len(str(getattr(fc, "args", "") or ""))
        history.append({"role": role, "text": first_text})

    for content in reversed(llm_request.contents or []):
        if getattr(content, "role", None) == "user":
            for part in getattr(content, "parts", None) or []:
                if getattr(part, "text", None):
                    task = part.text
                    break
            if task:
                break

    scope_key = str(
        getattr(callback_context, "invocation_id", "")
        or getattr(getattr(callback_context, "session", None), "id", "")
        or ""
    )
    return AgentCallContext(
        task=task or None,
        scope_key=scope_key,
        configured_model=getattr(llm_request, "model", "") or "",
        history=history,
        tool_count=len(getattr(llm_request, "tools", None) or []),
        context_tokens=total_chars // 4,
        last_role=last_role,
        had_error=had_error,
    )


def dynamic_model_selector(callback_context, llm_request):
    """ADK `before_model_callback` — fires before every LLM API call.

    Thin translator: builds an AgentCallContext and delegates the decision to the
    framework-neutral core (real-question recovery, ceiling, capability gate,
    stickiness, effort). Mutates `llm_request.model`; returns None so ADK proceeds.
    """
    import time

    from classifier.integrations._agentic import route_agent_call

    try:
        ctx = _build_agent_ctx(callback_context, llm_request)
    except Exception as exc:
        logger.warning("dynamic_model_selector: could not read request (%s) — keeping default", exc)
        return None

    if not (ctx.task or ctx.history):
        logger.warning("dynamic_model_selector: no user message found — keeping default model.")
        return None

    try:
        decision = route_agent_call(ctx)
    except Exception as exc:
        logger.error("dynamic_model_selector: routing failed (%s) — keeping default model.", exc)
        return None

    original = llm_request.model
    llm_request.model = decision.model_name
    # Apply effort (thinking budget) best-effort — provider-specific; ignored if unsupported.
    if decision.effort and decision.effort != "none":
        llm_request.dmr_effort = decision.effort

    _store_pending(
        callback_context,
        {
            "decision_id": decision.decision_id,
            "model": decision.model_name,
            "task": ctx.task or "",
            "t0": time.perf_counter(),
        },
    )

    logger.info(
        "Model selected | %s => %s [%s | role=%s | effort=%s%s]",
        original,
        decision.model_name,
        decision.tier.value.upper(),
        decision.call_role,
        decision.effort,
        " | sticky" if decision.sticky else "",
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

    from classifier import OutcomeRecord, log_outcome
    from classifier.infra.tokenizers import count_tokens

    wall_ms = (time.perf_counter() - pending["t0"]) * 1000

    # Try to extract usage / response text from various ADK response shapes
    tokens_in = count_tokens(pending["task"], model=pending["model"])
    tokens_out = 0
    success = True
    error = None
    try:
        usage = getattr(llm_response, "usage_metadata", None) or {}
        if hasattr(usage, "get"):
            tokens_in = int(usage.get("prompt_token_count", tokens_in))
            tokens_out = int(usage.get("candidates_token_count", 0))
        else:
            tokens_in = int(getattr(usage, "prompt_token_count", tokens_in))
            tokens_out = int(getattr(usage, "candidates_token_count", 0))
    except Exception:
        # Fall through with the heuristic count
        for content in getattr(llm_response, "content", None) or []:
            for part in getattr(content, "parts", []) or []:
                if getattr(part, "text", None):
                    tokens_out += count_tokens(part.text, model=pending["model"])

    log_outcome(
        OutcomeRecord(
            decision_id=pending["decision_id"],
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            wall_ms=wall_ms,
            success=success,
            error_message=error,
        )
    )

    # Feed the agentic escalation loop (no-op unless escalate_on_failure is on).
    try:
        from classifier.integrations._agentic import report_agent_outcome

        text = ""
        resp_content = getattr(llm_response, "content", None)
        if resp_content is not None:
            for part in getattr(resp_content, "parts", []) or []:
                if getattr(part, "text", None):
                    text += part.text
        sk = str(getattr(callback_context, "invocation_id", "") or "")
        if sk:
            report_agent_outcome(sk, text)
    except Exception:
        pass
    return None


# Back-compat alias for the underscore-prefixed name used in earlier examples
_dynamic_model_selector = dynamic_model_selector
