"""Conformance tests for the agent adapters (thin translators over the neutral
core). Verifies the four guarantees with mocked framework objects — no google-adk
needed: real-question recovery, configured-model ceiling, capability gate, and
scope stickiness."""

import pytest

# ── Minimal ADK-shaped mocks ───────────────────────────────────────────────────


class _Part:
    def __init__(self, text=None, function_response=None, function_call=None):
        self.text = text
        self.function_response = function_response
        self.function_call = function_call


class _Content:
    def __init__(self, role, parts):
        self.role = role
        self.parts = parts


class _Req:
    def __init__(self, model, contents, tools=None):
        self.model = model
        self.contents = contents
        self.tools = tools or []


class _Ctx:
    def __init__(self, invocation_id):
        self.invocation_id = invocation_id
        self.agent_name = "agent"
        self.state = {}


_DEEP = "Provide a complete clinical assessment for patient 12345 with labs, eGFR, interactions, eligibility"


@pytest.fixture(autouse=True)
def restore_scope_setting():
    from classifier.infra.config import settings
    from classifier.integrations import _agentic

    saved = settings.dmr_routing_scope
    _agentic._scope_decisions.clear()
    _agentic._scope_escalated.clear()
    yield
    settings.dmr_routing_scope = saved
    _agentic._scope_decisions.clear()
    _agentic._scope_escalated.clear()


def test_recovery_specialist_does_not_route_on_for_context():
    """A specialist call whose last user msg is 'For context:' must route on the
    recovered real task — NOT on the wrapper (which would collapse to flash-lite)."""
    from classifier.integrations.adk import dynamic_model_selector

    req = _Req(
        "gemini-2.5-pro",
        [
            _Content("user", [_Part(text=_DEEP)]),
            _Content("model", [_Part(text="transferring to specialist")]),
            _Content("user", [_Part(text="For context:")]),
        ],
    )
    dynamic_model_selector(_Ctx("turn-A"), req)
    # Recovery proven: routed on the real (non-trivial) task, not the "For context:"
    # wrapper that would classify as conversation -> flash-lite.
    assert req.model != "gemini-2.5-flash-lite"


def test_ceiling_caps_at_configured_model():
    """A flash-configured orchestrator must never be upgraded to pro."""
    from classifier.integrations.adk import dynamic_model_selector

    req = _Req("gemini-2.5-flash", [_Content("user", [_Part(text=_DEEP)])])
    dynamic_model_selector(_Ctx("turn-B"), req)
    assert req.model in ("gemini-2.5-flash", "gemini-2.5-flash-lite")  # capped at flash ceiling


def test_capability_gate_never_flash_lite_for_tool_driving():
    """A tool-driving call (tools present) must not land on flash-lite (basic)."""
    from classifier.integrations.adk import dynamic_model_selector

    req = _Req("gemini-2.5-pro", [_Content("user", [_Part(text="hi there")])], tools=[1, 2, 3])
    dynamic_model_selector(_Ctx("turn-C"), req)
    assert req.model != "gemini-2.5-flash-lite"


def test_stickiness_same_model_within_turn():
    """With turn scope, all calls in one invocation reuse the same model."""
    from classifier.infra.config import settings
    from classifier.integrations.adk import dynamic_model_selector

    settings.dmr_routing_scope = "turn"
    req1 = _Req("gemini-2.5-pro", [_Content("user", [_Part(text=_DEEP)])])
    req2 = _Req("gemini-2.5-pro", [_Content("user", [_Part(text="For context:")])])
    dynamic_model_selector(_Ctx("turn-D"), req1)
    dynamic_model_selector(_Ctx("turn-D"), req2)
    assert req1.model == req2.model  # sticky within the turn
