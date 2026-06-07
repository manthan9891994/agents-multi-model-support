"""Capability gate — the agentic-cliff fix.

Cheap models fail at *driving tools* (not at writing prose). So a call whose role
needs reliable tool-calling must never be routed to a model that can't do it,
regardless of how cheap it is. This raises the chosen tier to the cheapest tier
whose provider model meets the role's minimum tool-calling reliability.
"""

from __future__ import annotations

from classifier.core.registry import MODEL_REGISTRY, get_tool_calling
from classifier.core.types import _TIER_ORDER, ModelTier

# Minimum tool-calling reliability required per call role.
ROLE_MIN_TOOL_CALLING: dict[str, str] = {
    "tool_call": "reliable",
    "orchestration": "reliable",
    "synthesis": "basic",
    "conversational": "none",
}
_RANK = {"none": 0, "basic": 1, "reliable": 2}


def enforce_capability(provider: str, tier: ModelTier, call_role: str | None) -> ModelTier:
    """Return `tier` raised to the cheapest tier whose model satisfies the role's
    minimum tool-calling reliability. No-op when the role needs nothing, the
    provider is unknown, or nothing higher qualifies (never crashes/​downgrades)."""
    need = _RANK.get(ROLE_MIN_TOOL_CALLING.get(call_role or "synthesis", "none"), 0)
    if need == 0:
        return tier
    order = _TIER_ORDER
    if tier not in order:
        return tier
    reg = MODEL_REGISTRY.get(provider) or {}
    for i in range(order.index(tier), len(order)):
        model = reg.get(order[i])
        if model and _RANK.get(get_tool_calling(model), 2) >= need:
            return order[i]
    return tier  # nothing higher qualifies — keep current rather than fail
