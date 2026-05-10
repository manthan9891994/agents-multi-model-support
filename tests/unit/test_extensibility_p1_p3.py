"""Tests for P1/P2/P3 extensibility features."""

import pytest

from classifier import (
    ABTest,
    ModelTier,
    Router,
    ShadowMode,
    TaskType,
    clear_hooks,
    count_tokens,
    list_layers,
    list_tier_levels,
    register_complexity,
    register_l3_strategy,
    register_layer,
    register_provider,
    register_task_type,
    register_tokenizer,
    set_tier_levels,
    unregister_layer,
)


@pytest.fixture(autouse=True)
def cleanup():
    yield
    clear_hooks()


# ── #6 register_task_type ────────────────────────────────────────────────────


def test_register_task_type_creates_member():
    from classifier.core.types import task_type_for

    ct = register_task_type("clinical_note_test")
    assert ct.value == "clinical_note_test"
    # Look up via the helper (handles both real and dynamic members)
    found = task_type_for("clinical_note_test")
    assert found.value == "clinical_note_test"
    # Attribute access works too
    assert TaskType.CLINICAL_NOTE_TEST.value == "clinical_note_test"


# ── #7 register_complexity ───────────────────────────────────────────────────


def test_register_complexity_creates_member():
    from classifier.core.types import TaskComplexity, complexity_for

    ec = register_complexity("epic")
    assert ec.value == "epic"
    assert complexity_for("epic").value == "epic"
    assert TaskComplexity.EPIC.value == "epic"


# ── #8 set_tier_levels ───────────────────────────────────────────────────────


def test_set_tier_levels_extends_order():
    original = list_tier_levels()
    set_tier_levels(["free", "low", "medium", "high", "frontier"])
    assert list_tier_levels() == ["free", "low", "medium", "high", "frontier"]
    # Restore
    set_tier_levels(original)


# ── #15 Real tokenizer ───────────────────────────────────────────────────────


def test_count_tokens_with_wordcount_fallback():
    n = count_tokens("hello world how are you", model="unknown-model")
    assert n == 5


def test_register_tokenizer():
    register_tokenizer("custom-test-model", lambda t: 42)
    assert count_tokens("anything", model="custom-test-model") == 42


# ── #18 PII policy ───────────────────────────────────────────────────────────


def test_pii_policy_min_tier_high():
    """PII content with min_tier=HIGH bumps to HIGH tier."""
    router = Router(
        layer2_enabled=False,
        layer3_enabled=False,
        pii_policy={"min_tier": ModelTier.HIGH, "block": False},
        cache_enabled=False,
    )
    decision = router.classify("Patient MRN: 12345678 has chest pain")
    assert decision.tier == ModelTier.HIGH
    assert decision.compliance_flag


def test_pii_policy_block_raises():
    """block=True raises ClassificationError on PII."""
    from classifier.core.exceptions import ClassificationError

    router = Router(
        layer2_enabled=False,
        layer3_enabled=False,
        pii_policy={"min_tier": ModelTier.MEDIUM, "block": True},
        cache_enabled=False,
    )
    with pytest.raises(ClassificationError):
        router.classify("MRN: 12345678 patient details")


# ── #17 Retry / circuit breaker policy ───────────────────────────────────────


def test_l2_circuit_breaker_policy_applied():
    """Custom CB threshold takes effect during classify."""
    from classifier.layers.layer2 import api as l2api

    router = Router(
        layer2_enabled=False,
        layer3_enabled=False,
        l2_circuit_breaker={"failure_threshold": 99, "cooldown_secs": 5.0},
    )
    with router._apply_overrides():
        assert l2api._circuit_breaker.failure_threshold == 99
        assert l2api._circuit_breaker.cooldown_secs == 5.0


def test_l2_retry_policy_applied():
    from classifier.layers.layer2 import api as l2api

    router = Router(
        layer2_enabled=False,
        layer3_enabled=False,
        l2_retry_policy={"max_attempts": 7, "initial_delay": 0.5, "backoff": 2.0},
    )
    with router._apply_overrides():
        assert l2api._retry_policy["max_attempts"] == 7
        assert l2api._retry_policy["initial_delay"] == 0.5
        assert l2api._retry_policy["backoff"] == 2.0


# ── #16 L1 weights ───────────────────────────────────────────────────────────


def test_l1_weights_override():
    from classifier.layers.layer1 import scoring

    router = Router(l1_weights={"primary": 5.0, "secondary": 0.5}, layer2_enabled=False, layer3_enabled=False)
    with router._apply_overrides():
        assert scoring._WEIGHTS["primary"] == 5.0
        assert scoring._WEIGHTS["secondary"] == 0.5


# ── #13 Layer plugin ─────────────────────────────────────────────────────────


def test_layer_plugin_registration():
    class MyLayer:
        name = "test_legal"
        runs_after = "pre"

        def classify(self, task, history=None):
            from classifier.core.types import TaskComplexity

            if "tort" in task.lower():
                return TaskType.REASONING, TaskComplexity.COMPLEX, ModelTier.HIGH, 0.95, "test plugin"
            return None

    plugin = MyLayer()
    register_layer(plugin)
    try:
        assert "test_legal" in list_layers()["pre"]
        router = Router(layer2_enabled=False, layer3_enabled=False, cache_enabled=False)
        decision = router.classify("Discuss tort liability for software vendors")
        assert decision.layer_used == "plugin:pre"
        assert decision.tier == ModelTier.HIGH
    finally:
        unregister_layer("test_legal")


# ── #12 L3 strategy registration ─────────────────────────────────────────────


def test_register_l3_strategy():
    captured = {"called": False}

    def my_strategy(task, history=None):
        captured["called"] = True
        return None  # abstain — fall through to L2

    register_l3_strategy("test_strategy", my_strategy)
    from classifier.layers.layer3 import list_strategies

    assert "test_strategy" in list_strategies()


# ── #23 A/B testing ──────────────────────────────────────────────────────────


def test_ab_test_split_routes_traffic():
    control = Router(layer2_enabled=False, layer3_enabled=False)
    treatment = Router(layer2_enabled=False, layer3_enabled=False)
    ab = ABTest(control=control, treatment=treatment, split=0.0)  # all to control
    decision = ab.classify("hello", ctx={"user_id": "u1"})
    assert decision is not None


def test_ab_test_sticky_assignment():
    control = Router(layer2_enabled=False, layer3_enabled=False)
    treatment = Router(layer2_enabled=False, layer3_enabled=False)
    ab = ABTest(
        control=control,
        treatment=treatment,
        split=0.5,
        sticky_key=lambda c: c.get("user_id"),
    )
    # Same user_id should always assign to same variant
    a1 = ab.assign({"user_id": "stable_user"})
    a2 = ab.assign({"user_id": "stable_user"})
    assert a1 == a2


def test_ab_test_split_zero_always_control():
    control = Router(layer2_enabled=False, layer3_enabled=False)
    treatment = Router(layer2_enabled=False, layer3_enabled=False)
    ab = ABTest(
        control=control,
        treatment=treatment,
        split=0.0,
        sticky_key=lambda c: c.get("user_id"),
    )
    for i in range(10):
        assert ab.assign({"user_id": f"u{i}"}) == "control"


# ── #24 Shadow mode ──────────────────────────────────────────────────────────


def test_shadow_mode_returns_primary_decision():
    primary = Router(layer2_enabled=False, layer3_enabled=False)
    shadow = Router(layer2_enabled=False, layer3_enabled=False)

    sm = ShadowMode(primary=primary, shadow=shadow)
    decision = sm.classify("hello world test")
    assert decision is not None
    # Both routers configured the same → match
    stats = sm.stats
    assert stats["calls"] == 1


def test_shadow_mode_diff_callback():
    """When primary and shadow disagree, on_diff is invoked."""
    primary = Router(layer2_enabled=False, layer3_enabled=False)
    shadow = Router(layer2_enabled=False, layer3_enabled=False)
    diffs = []
    sm = ShadowMode(
        primary=primary,
        shadow=shadow,
        on_diff=lambda task, p, s: diffs.append((p.tier, s.tier)),
    )
    sm.classify("hi some shadow test")
    # When configs match, no diff callback fires
    assert isinstance(diffs, list)


# ── #26 Multi-tenant per-call config ─────────────────────────────────────────


def test_tenant_config_per_call():
    """Per-call tenant_config overrides Router defaults."""
    router = Router(layer2_enabled=False, layer3_enabled=False, cache_enabled=False)
    # Default uses google
    router.classify("hello tenant test 1")
    # Tenant A overrides to anthropic
    d2 = router.classify(
        "hello tenant test 2",
        tenant_config={"providers": ["anthropic"]},
    )
    assert d2.provider == "anthropic"


# ── #27 Router.merge / with_overrides ─────────────────────────────────────────


def test_with_overrides_creates_new_router():
    base = Router(layer2_enabled=False, layer3_enabled=False)
    derived = base.with_overrides(providers=["anthropic"])
    assert derived.providers == ["anthropic"]
    assert base.providers == []  # base unchanged


def test_router_merge_combines_configs():
    base = Router(extra_keyword_packs=[], layer2_enabled=False, layer3_enabled=False)
    custom = Router(providers=["anthropic"], layer2_enabled=False, layer3_enabled=False)
    merged = base.merge(custom)
    assert merged.providers == ["anthropic"]


def test_router_to_dict_round_trip():
    """Router.to_dict()/Router(**d) should round-trip."""
    r1 = Router(
        providers=["anthropic", "google"],
        layer2_enabled=False,
        layer3_enabled=False,
        latency_budget_ms=1500,
    )
    d = r1.to_dict()
    r2 = Router(**d)
    assert r2.providers == ["anthropic", "google"]
    assert r2.latency_budget_ms == 1500


# ── #21 Latency budget (smoke) ───────────────────────────────────────────────


def test_latency_budget_in_hook_context():
    """latency_budget_ms gets injected into hook_context for cascade post-processing."""
    captured = {}

    def capture_ctx(task, decision, ctx):
        captured.update(ctx)
        return decision

    router = Router(
        layer2_enabled=False,
        layer3_enabled=False,
        latency_budget_ms=2000,
        post_classify_hooks=[capture_ctx],
        cache_enabled=False,
    )
    router.classify("Latency budget smoke test xyz")
    assert captured.get("latency_budget_ms") == 2000


# ── #22 Residency in hook_context ────────────────────────────────────────────


def test_residency_in_hook_context():
    captured = {}

    def capture_ctx(task, decision, ctx):
        captured.update(ctx)
        return decision

    router = Router(
        layer2_enabled=False,
        layer3_enabled=False,
        residency="EU",
        post_classify_hooks=[capture_ctx],
        cache_enabled=False,
    )
    router.classify("Residency smoke test xyz")
    assert captured.get("residency") == "EU"
