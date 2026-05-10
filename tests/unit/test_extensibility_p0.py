"""Tests for P0 extensibility (custom provider, cost table, L2 provider, L3 embed,
custom router, hooks)."""
from unittest.mock import MagicMock, patch

import pytest

from classifier import (
    ModelTier,
    Router,
    capabilities_for,
    clear_hooks,
    get_model_cost,
    list_models,
    list_providers,
    register_hook,
    register_model_cost,
    register_provider,
)


@pytest.fixture(autouse=True)
def cleanup_hooks():
    yield
    clear_hooks()


# ── #5 Custom provider registration ─────────────────────────────────────────

def test_register_provider_adds_to_registry():
    register_provider("groq_test", {
        ModelTier.LOW:    "llama-3.3-8b",
        ModelTier.MEDIUM: "llama-3.3-70b",
        ModelTier.HIGH:   "llama-3.3-405b",
    })
    assert "groq_test" in list_providers()


def test_register_provider_with_capabilities():
    register_provider("mistral_test", {
        ModelTier.LOW: "mistral-small",
        ModelTier.HIGH: "mistral-large",
    }, capabilities={
        "mistral-large": {"context_window": 128_000, "supports_function_calling": True},
    })
    caps = capabilities_for("mistral-large")
    assert caps["context_window"] == 128_000


def test_router_uses_custom_provider():
    register_provider("cohere_test", {
        ModelTier.LOW: "command-r", ModelTier.MEDIUM: "command-r-plus", ModelTier.HIGH: "command-r-plus",
    })
    router = Router(layer2_enabled=False, layer3_enabled=False)
    decision = router.classify("Hello", provider="cohere_test")
    assert decision.provider == "cohere_test"
    assert decision.model_name in ("command-r", "command-r-plus")


# ── #1 Pluggable cost table ─────────────────────────────────────────────────

def test_register_model_cost_overrides_existing():
    register_model_cost("gemini-2.5-flash", input_per_1m=0.99, output_per_1m=2.99)
    assert get_model_cost("gemini-2.5-flash") == {"input": 0.99, "output": 2.99}
    # Reset for other tests
    register_model_cost("gemini-2.5-flash", input_per_1m=0.25, output_per_1m=0.75)


def test_router_model_costs_constructor_arg():
    Router(model_costs={
        "my-custom-model": {"input": 1.0, "output": 3.0},
    })
    assert get_model_cost("my-custom-model") == {"input": 1.0, "output": 3.0}


# ── #2 Configurable L2 provider ─────────────────────────────────────────────

def test_layer2_provider_dispatch():
    """When layer2_provider='openai', L2 calls go through openai caller."""
    from classifier.layers.layer2 import api as l2api
    from classifier.layers.layer2 import providers as l2_providers

    fake_caller = MagicMock(return_value=MagicMock(text='{"task_type":"reasoning","complexity":"simple","confidence":0.9,"reason":"x"}'))
    l2_providers.register_l2_provider("test_provider", fake_caller)

    router = Router(layer2_provider="test_provider", layer2_model="x-1", layer2_enabled=True)
    with router._apply_overrides():
        l2api._shared_client = None
        l2api._circuit_breaker._failures = 0
        l2api._call_api("hello world")
        assert fake_caller.called


# ── #4 Configurable L3 embedding model ──────────────────────────────────────

def test_set_embedding_model_updates_global():
    from classifier.ml.embeddings import current_embedding_model, set_embedding_model
    original = current_embedding_model()
    set_embedding_model("BAAI/bge-large-en-v1.5")
    assert current_embedding_model() == "BAAI/bge-large-en-v1.5"
    set_embedding_model(original)   # restore


def test_router_layer3_embedding_model_constructor():
    from classifier.ml.embeddings import current_embedding_model, set_embedding_model
    original = current_embedding_model()
    Router(layer3_embedding_model="sentence-transformers/all-mpnet-base-v2")
    assert current_embedding_model() == "sentence-transformers/all-mpnet-base-v2"
    set_embedding_model(original)


# ── #14 Custom router function ──────────────────────────────────────────────

def test_custom_classifier_overrides_cascade():
    from classifier.core.types import ClassificationDecision, TaskComplexity, TaskType

    def force_high(task: str, ctx: dict):
        return ClassificationDecision(
            model_name="claude-opus-4-7",
            tier=ModelTier.HIGH,
            task_type=TaskType.REASONING,
            complexity=TaskComplexity.COMPLEX,
            reasoning="custom",
            confidence=1.0,
            provider="anthropic",
            layer_used="custom",
        )

    router = Router(custom_classifier=force_high, layer2_enabled=False, layer3_enabled=False)
    decision = router.classify("anything")
    assert decision.tier == ModelTier.HIGH
    assert decision.layer_used == "custom"


def test_custom_classifier_returns_none_falls_through():
    """If custom_classifier returns None, fall back to cascade."""
    def maybe_skip(task: str, ctx: dict):
        return None   # always fall through

    router = Router(custom_classifier=maybe_skip, layer2_enabled=False, layer3_enabled=False)
    decision = router.classify("Hello")
    assert decision.layer_used == "layer1"


# ── #9 Middleware / hooks ───────────────────────────────────────────────────

def test_pre_classify_hook_modifies_task():
    captured = []
    def upper_hook(task: str, ctx: dict) -> str:
        captured.append(task)
        return task.upper()

    router = Router(pre_classify_hooks=[upper_hook], layer2_enabled=False, layer3_enabled=False)
    router.classify("hello")
    assert captured == ["hello"]


def test_post_classify_hook_modifies_decision():
    from classifier.core.types import TaskType
    def force_reasoning(task, decision, ctx):
        decision.task_type = TaskType.REASONING
        return decision

    router = Router(post_classify_hooks=[force_reasoning],
                    layer2_enabled=False, layer3_enabled=False, cache_enabled=False)
    decision = router.classify("Some unique post-hook task xyz123")
    assert decision.task_type == TaskType.REASONING


def test_hook_context_passed_through():
    captured_ctx = {}
    def capture_hook(task, decision, ctx):
        captured_ctx.update(ctx)
        return decision

    router = Router(post_classify_hooks=[capture_hook], layer2_enabled=False, layer3_enabled=False, cache_enabled=False)
    router.classify("Hi some unique tenant context xyz123", hook_context={"tenant_id": "acme", "user_id": "u123"})
    assert captured_ctx["tenant_id"] == "acme"
    assert captured_ctx["user_id"] == "u123"


def test_hook_exception_doesnt_break_classify():
    def broken_hook(task, decision, ctx):
        raise ValueError("oops")

    router = Router(post_classify_hooks=[broken_hook], layer2_enabled=False, layer3_enabled=False)
    decision = router.classify("Hi")   # should not raise
    assert decision is not None


def test_pre_hook_can_block_with_exception():
    def reject_pii(task: str, ctx: dict) -> str:
        if "[REDACTED]" in task:
            raise ValueError("blocked")
        return task

    router = Router(pre_classify_hooks=[reject_pii], layer2_enabled=False, layer3_enabled=False)
    with pytest.raises(ValueError, match="blocked"):
        router.classify("[REDACTED] task")


def test_router_hooks_unregistered_after_call():
    from classifier.hooks import hook_manager

    def my_hook(task, decision, ctx):
        return decision

    router = Router(post_classify_hooks=[my_hook], layer2_enabled=False, layer3_enabled=False)
    router.classify("hi")
    # After call, the hook should NOT remain registered globally
    assert my_hook not in hook_manager.post_classify
