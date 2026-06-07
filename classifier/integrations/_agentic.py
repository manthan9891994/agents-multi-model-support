"""Framework-neutral agentic routing core + universal API.

ALL agentic intelligence lives here — real-question recovery, scope stickiness
(don't thrash models mid-turn), configured-model ceiling, capability gating
(via the pipeline), and escalate-on-failure. Per-framework adapters are thin
translators that build an `AgentCallContext` and call `route_agent_call`; anything
else (LangGraph nodes, AutoGen, Strands, bespoke loops) uses the universal
`route_scope` / `route` / `report` API.

    from classifier import route_scope, route, report
    with route_scope(scope_key=thread_id, ceiling="gpt-4o"):
        model = route("summarize these labs", role="synthesis")
        ... call your provider with `model` ...
        report(response_text)            # enables escalate-on-failure
"""

from __future__ import annotations

import contextvars
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass

from classifier.core.types import _TIER_ORDER, ModelTier

_CONTEXT_PREFIX = "for context"

# Per-scope sticky decision cache + per-scope escalation flag (process-local).
_scope_decisions: dict[str, tuple] = {}  # scope_key -> (decision, expires_at)
_scope_escalated: set[str] = set()


@dataclass
class AgentCallContext:
    """Framework-neutral description of one model call inside an agent."""

    task: str | None
    scope_key: str
    configured_model: str
    call_role: str | None = None
    history: list | None = None
    tool_count: int = 0
    context_tokens: int = 0
    last_role: str = "user"
    had_error: bool = False


def _recover_task(task: str | None, history: list | None) -> str:
    """The originating question — never the framework's 'For context:' wrapper."""
    if task and not task.strip().lower().startswith(_CONTEXT_PREFIX):
        return task
    for item in history or []:
        text = item.get("text", "") if isinstance(item, dict) else str(item)
        if (item.get("role") if isinstance(item, dict) else "user") == "user":
            if text and not text.strip().lower().startswith(_CONTEXT_PREFIX):
                return text
    return task or "conversation"


def _infer_role(ctx: AgentCallContext) -> str:
    """Best-effort call-role when the adapter didn't supply one."""
    if ctx.call_role:
        return ctx.call_role
    if ctx.tool_count and ctx.tool_count > 0:
        # mid-loop with tools available: a tool-driving step unless context is huge
        return "tool_call"
    return "synthesis"


def _cap_to_ceiling(decision, ceiling: ModelTier):
    """Lower a decision's tier/model to the configured ceiling (never above it)."""
    from classifier.core.registry import MODEL_REGISTRY

    if _TIER_ORDER.index(decision.tier) <= _TIER_ORDER.index(ceiling):
        return decision
    decision.tier = ceiling
    model = (MODEL_REGISTRY.get(decision.provider) or {}).get(ceiling)
    if model:
        decision.model_name = model
    decision.reasoning += f" [capped at configured {ceiling.value}]"
    return decision


def route_agent_call(ctx: AgentCallContext):
    """Route one agent call: recover the real task, route on it, cap at the
    configured model, honor scope stickiness + escalation. Returns a
    ClassificationDecision (with .model_name to use)."""
    from classifier import classify
    from classifier.core.registry import tier_of_model
    from classifier.core.types import ContextSignals
    from classifier.infra.config import settings

    scope = settings.dmr_routing_scope
    ceiling = tier_of_model(ctx.configured_model) if ctx.configured_model else ModelTier.HIGH

    # Escalation wins: a prior failure in this scope forces the ceiling model.
    if ctx.scope_key and ctx.scope_key in _scope_escalated:
        from classifier.core.registry import MODEL_REGISTRY
        from classifier.core.types import ClassificationDecision

        model = (MODEL_REGISTRY.get(settings.default_provider) or {}).get(ceiling, ctx.configured_model)
        return ClassificationDecision(
            model_name=model,
            tier=ceiling,
            task_type=_dummy_tt(),
            complexity=_dummy_cx(),
            reasoning="escalated to ceiling after a failure in this scope",
            confidence=1.0,
            provider=settings.default_provider,
            layer_used="escalated",
            call_role=_infer_role(ctx),
            sticky=False,
        )

    # Scope stickiness: reuse a fresh decision for this scope (preserves prompt cache).
    if scope != "call" and ctx.scope_key:
        hit = _scope_decisions.get(ctx.scope_key)
        if hit and hit[1] > time.time():
            d = hit[0]
            d.sticky = True
            return d

    task = _recover_task(ctx.task, ctx.history)
    cs = ContextSignals(
        total_context_tokens=ctx.context_tokens,
        call_number=1,
        has_error=ctx.had_error,
        last_role=ctx.last_role,
        available_tools=ctx.tool_count,
        scope_key=ctx.scope_key,
        call_role=_infer_role(ctx),
    )
    decision = classify(task[:500], context_signals=cs)
    decision = _cap_to_ceiling(decision, ceiling)

    if scope != "call" and ctx.scope_key:
        _scope_decisions[ctx.scope_key] = (decision, time.time() + settings.dmr_scope_decision_ttl_s)
    return decision


def report_agent_outcome(scope_key: str, response_text: str | None, usage=None) -> None:
    """Feed an outcome back: flag the scope for escalation if the response failed."""
    from classifier.infra.config import settings

    if settings.dmr_escalate_on_failure and scope_key:
        from classifier.quality.failure_detect import looks_like_failure

        failed, _reason = looks_like_failure(response_text)
        if failed:
            _scope_escalated.add(scope_key)
            _scope_decisions.pop(scope_key, None)  # drop sticky cheap decision


def reset_scope(scope_key: str) -> None:
    """Forget a scope's sticky decision + escalation state (call when a turn ends)."""
    _scope_decisions.pop(scope_key, None)
    _scope_escalated.discard(scope_key)


def _dummy_tt():
    from classifier.core.types import TaskType

    return TaskType.REASONING


def _dummy_cx():
    from classifier.core.types import TaskComplexity

    return TaskComplexity.STANDARD


# ── Universal API (works for any framework / bespoke loop; async-safe) ─────────
_current = contextvars.ContextVar("dmr_scope", default=None)


@contextmanager
def route_scope(scope_key: str | None = None, ceiling: str = ""):
    """Set the ambient routing scope for `route()`/`report()` inside the block."""
    token = _current.set((scope_key or uuid.uuid4().hex[:12], ceiling))
    try:
        yield
    finally:
        _current.reset(token)


def route(task: str, role: str = "synthesis", **kw) -> str:
    """Return the model name to use for this call, honoring the ambient scope."""
    sk, ceiling = _current.get() or (uuid.uuid4().hex[:12], "")
    ctx = AgentCallContext(task=task, scope_key=sk, configured_model=ceiling, call_role=role, **kw)
    return route_agent_call(ctx).model_name


def report(response_text: str | None) -> None:
    """Report the ambient scope's outcome (enables escalate-on-failure)."""
    cur = _current.get()
    if cur:
        report_agent_outcome(cur[0], response_text)
