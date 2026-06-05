"""Multi-agent ADK routing with dynamic-model-router — the CORRECT pattern.

Drop-in `before_model_callback` / `after_model_callback` for a multi-agent Google
ADK app (an orchestrator that transfers to specialist sub-agents). It fixes the
two things a naive per-call router gets wrong in multi-agent ADK:

1. REAL-QUESTION RECOVERY.
   When the orchestrator transfers to a sub-agent, ADK hands that sub-agent a
   wrapper message like ``"For context:"`` — NOT the user's real question. If you
   route on "the last user message" (what `dynamic_model_selector` does), every
   specialist call looks like trivial chatter and gets the cheapest model, which
   quietly destroys answer quality. Here we cache the originating question per
   ``invocation_id`` (shared by all calls in one turn) and route the whole turn
   on it.

2. CONFIGURED MODEL = CEILING.
   Each ``LlmAgent`` has a configured ``model=...``. Treat that as the HIGHEST
   tier available to that agent: route dynamically *below* it, but never *above*
   it. A flash-configured orchestrator should never be silently upgraded to pro.

Usage — wire into every agent (keep each agent's ``model=`` as its ceiling):

    from examples.adk_multiagent_routing import before_model_callback, after_model_callback
    from google.adk.agents import LlmAgent

    specialist = LlmAgent(
        name="clinical_intelligence",
        model="gemini-2.5-pro",                       # ceiling: may use pro / flash / flash-lite
        before_model_callback=before_model_callback,
        after_model_callback=after_model_callback,
        ...,
    )

This module imports lazily and has NO hard dependency on google-adk — the
callbacks are duck-typed against ADK's callback_context / llm_request objects.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

_CONTEXT_PREFIX = "for context"     # ADK's agent-to-agent transfer wrapper
_HEAVY_CTX_TOKENS = 1500           # accumulated context that marks a synthesis call

# Per-turn state shared across the whole agent loop (orchestrator + specialists
# share one invocation_id). Bounded so long-running processes don't leak.
_inv_state: dict[str, dict] = {}
_pending: dict[str, dict] = {}


def _tier_order():
    from classifier.core.types import ModelTier
    return [ModelTier.LOW, ModelTier.MEDIUM, ModelTier.HIGH]


def _ceiling_tier(model: str, provider: str):
    """The tier of an agent's CONFIGURED model = its highest available tier.
    Looks the model up in the active registry; unknown models default to HIGH."""
    from classifier.core.registry import MODEL_REGISTRY
    from classifier.core.types import ModelTier

    for tier, name in (MODEL_REGISTRY.get(provider) or {}).items():
        if name == model:
            return tier
    return ModelTier.HIGH


def _last_user_text(llm_request) -> str:
    for content in reversed(llm_request.contents):
        if content.role == "user" and content.parts:
            for part in content.parts:
                if getattr(part, "text", None):
                    return part.text
    return ""


def _resolve_real_task(llm_request, invocation_id: str) -> str:
    """The turn's originating question — never the ``"For context:"`` wrapper.

    The orchestrator's first call carries the real question; we cache it per
    invocation_id and reuse it for every downstream specialist call so the whole
    loop routes on the actual intent."""
    if len(_inv_state) > 2000:
        _inv_state.clear()
    st = _inv_state.setdefault(invocation_id, {"root_query": "", "call_number": 0})
    msg = _last_user_text(llm_request).strip()
    if msg and not msg.lower().startswith(_CONTEXT_PREFIX):
        if not st["root_query"]:
            st["root_query"] = msg
        return msg
    return st["root_query"] or "conversation"


def _context_signals(llm_request, invocation_id: str):
    from classifier.core.types import ContextSignals

    st = _inv_state[invocation_id]
    st["call_number"] += 1
    total_chars = sum(
        len(getattr(p, "text", "") or "")
        for c in llm_request.contents for p in (c.parts or [])
    )
    last = llm_request.contents[-1] if llm_request.contents else None
    last_role = "user"
    if last is not None:
        if any(getattr(p, "function_response", None) for p in (last.parts or [])):
            last_role = "tool"
        elif (last.role or "user") == "model":
            last_role = "model"
    return ContextSignals(
        total_context_tokens=total_chars // 4,
        call_number=st["call_number"],
        has_error=False,
        last_role=last_role,
        has_multimodal=False,
        available_tools=len(getattr(llm_request, "tools", None) or []),
    )


def before_model_callback(callback_context, llm_request):
    """Route this call on the turn's REAL question, capped at the agent's
    configured model. Mutates ``llm_request.model``; returns None so ADK proceeds."""
    from classifier import classify_task
    from classifier.core.registry import MODEL_REGISTRY
    from classifier.infra.config import settings

    provider = settings.default_provider
    invocation_id = getattr(callback_context, "invocation_id", "default")
    configured = llm_request.model                       # the agent's ceiling
    ceiling = _ceiling_tier(configured, provider)

    real_task = _resolve_real_task(llm_request, invocation_id)
    ctx = _context_signals(llm_request, invocation_id)

    try:
        decision = classify_task(real_task[:500], provider=provider, context_signals=ctx)
    except Exception as exc:
        logger.warning("DMR: classify failed (%s) — keeping %s", exc, configured)
        return None

    order = _tier_order()
    # CEILING: never exceed the configured model.
    final_tier = decision.tier if order.index(decision.tier) <= order.index(ceiling) else ceiling
    selected = (MODEL_REGISTRY.get(provider) or {}).get(final_tier, configured)
    llm_request.model = selected

    _pending[invocation_id] = {"decision_id": decision.decision_id, "t0": time.perf_counter()}
    logger.info(
        "DMR | %s -> %s [%s<=%s] task=%r",
        configured, selected, final_tier.value, ceiling.value, real_task[:60],
    )
    return None


def after_model_callback(callback_context, llm_response):
    """Report the outcome to DMR's continual-learning loop (optional but useful)."""
    from classifier import OutcomeRecord, log_outcome

    invocation_id = getattr(callback_context, "invocation_id", "default")
    p = _pending.pop(invocation_id, None)
    if not p:
        return None
    usage = getattr(llm_response, "usage_metadata", None)
    tin = (getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
    tout = (getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
    try:
        log_outcome(OutcomeRecord(
            decision_id=p["decision_id"],
            tokens_in=tin, tokens_out=tout,
            wall_ms=(time.perf_counter() - p["t0"]) * 1000,
            success=True,
        ))
    except Exception:
        pass
    return None
