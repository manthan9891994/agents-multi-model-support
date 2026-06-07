"""Unit tests for the agentic cost levers (plan_docs/09): capability gate,
failure detector, context reduction, cache-aware cost, posture dial, and the
framework-neutral universal API (route_scope / route / report)."""

import pytest

from classifier.core.types import ModelTier


@pytest.fixture(autouse=True)
def restore_agentic_settings():
    from classifier.infra.config import settings

    fields = [
        "dmr_savings_level",
        "dmr_cache_aware",
        "dmr_context_reduction",
        "dmr_effort_routing",
        "dmr_model_routing",
        "dmr_escalate_on_failure",
        "dmr_routing_scope",
    ]
    saved = {f: getattr(settings, f) for f in fields}
    yield
    for f, v in saved.items():
        setattr(settings, f, v)


# ── Capability gate (T4) ───────────────────────────────────────────────────────


def test_capability_gate_blocks_basic_tool_calling_for_tool_role():
    from classifier.routing.capability import enforce_capability

    # google LOW = flash-lite (tool_calling=basic); a tool_call must be raised.
    assert enforce_capability("google", ModelTier.LOW, "tool_call") == ModelTier.MEDIUM
    # synthesis only needs 'basic' → LOW is fine.
    assert enforce_capability("google", ModelTier.LOW, "synthesis") == ModelTier.LOW
    # conversational needs nothing.
    assert enforce_capability("google", ModelTier.LOW, "conversational") == ModelTier.LOW


# ── Failure detector (T8) ──────────────────────────────────────────────────────


def test_failure_detector():
    from classifier.quality.failure_detect import looks_like_failure

    assert looks_like_failure("I cannot access the patient data.")[0] is True
    assert looks_like_failure("")[0] is True
    assert looks_like_failure("ok")[0] is True  # too short
    assert looks_like_failure("x" * 200)[0] is False


# ── Context reduction (T7) ─────────────────────────────────────────────────────


def test_context_prune_keeps_system_question_and_last_tools():
    from classifier.context.reduce import prune_context

    contents = [
        {"role": "system", "text": "you are a clinical agent"},
        {"role": "user", "text": "complete assessment for patient 12345"},
        {"role": "user", "text": "For context: ..."},
        {"role": "tool", "text": "OLD labs " + "x" * 5000},
        {"role": "tool", "text": "mid result"},
        {"role": "tool", "text": "recent result"},
    ]
    out = prune_context(contents, keep_last_tool_results=2, max_tool_chars=100)
    roles = [c["role"] for c in out]
    assert "system" in roles and roles.count("tool") == 2  # only last 2 tools
    assert all(not c["text"].lower().startswith("for context") for c in out)
    assert all(len(c["text"]) <= 120 for c in out if c["role"] == "tool")  # truncated


# ── Cache-aware cost (T6) ──────────────────────────────────────────────────────


def test_cache_aware_cost_and_switch_penalty():
    from classifier.infra.cost_tracker import estimate_cost, switch_penalty

    full = estimate_cost("gemini-2.5-pro", 10000, 500, cached_fraction=0.0)
    cached = estimate_cost("gemini-2.5-pro", 10000, 500, cached_fraction=0.9)
    assert cached < full  # caching reduces input cost
    # switching away from a cacheable model costs a re-warm penalty
    assert switch_penalty("gemini-2.5-pro", "gemini-2.5-flash", 8000) > 0
    assert switch_penalty("gemini-2.5-pro", "gemini-2.5-pro", 8000) == 0  # no switch


# ── Universal API (T9): stickiness + escalate-on-failure ───────────────────────


def test_universal_api_sticky_then_escalate():
    from classifier.infra.config import settings
    from classifier.integrations._agentic import report, reset_scope, route, route_scope

    settings.dmr_routing_scope = "turn"
    settings.dmr_escalate_on_failure = True
    reset_scope("ut-1")
    with route_scope("ut-1", ceiling="gemini-2.5-pro"):
        m1 = route("hello there friend", role="synthesis")
        m2 = route("another light step", role="synthesis")
        assert m1 == m2  # sticky within the turn (preserves prompt cache)
        report("I cannot access the data, please provide more information")  # failure
        m3 = route("now the hard synthesis step", role="synthesis")
        assert m3 == "gemini-2.5-pro"  # escalated to ceiling
    reset_scope("ut-1")


# ── Posture dial (T10) ─────────────────────────────────────────────────────────


def test_posture_dial_presets_levers():
    from classifier.infra.config import settings
    from classifier.routing.posture import apply_posture

    apply_posture(3)
    assert settings.dmr_cache_aware is True
    assert settings.dmr_effort_routing is True
    assert settings.dmr_context_reduction == "prune"
    assert settings.dmr_model_routing == "dispatch_downgrade"
    assert settings.dmr_escalate_on_failure is True
