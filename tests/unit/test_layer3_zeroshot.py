"""Unit tests for Layer 3 Stage 1 — zero-shot classifier.

All tests mock the transformers pipeline. transformers does NOT need to be installed
to run this test file; the lazy-loaded pipeline is replaced by a MagicMock.
"""
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def reset_zeroshot_singletons():
    """Reset module-level pipeline cache between tests."""
    import classifier.layers.layer3.zeroshot as zs
    zs._pipeline = None
    zs._load_failed = False
    yield
    zs._pipeline = None
    zs._load_failed = False


def _make_mock_pipe(tt_label: str, tt_score: float, cx_label: str, cx_score: float):
    """Build a mock pipeline that returns the given top labels for tt then cx calls."""
    mock_pipe = MagicMock()
    mock_pipe.side_effect = [
        # First call → task_type result (top label first, descending scores)
        {
            "labels": [tt_label, "filler"],
            "scores": [tt_score, 1.0 - tt_score],
            "sequence": "task",
        },
        # Second call → complexity result
        {
            "labels": [cx_label, "filler"],
            "scores": [cx_score, 1.0 - cx_score],
            "sequence": "task",
        },
    ]
    return mock_pipe


# ── Happy path ────────────────────────────────────────────────────────────────

def test_confident_zeroshot_returns_decision():
    from classifier.core.types import TaskType, TaskComplexity, ModelTier
    from classifier.layers.layer3 import zeroshot as zs

    tt_label = next(k for k, v in zs._TASK_TYPE_HYPOTHESES.items() if v == TaskType.CODE_CREATION)
    cx_label = next(k for k, v in zs._COMPLEXITY_HYPOTHESES.items() if v == TaskComplexity.STANDARD)

    zs._pipeline = _make_mock_pipe(tt_label, 0.95, cx_label, 0.90)

    result = zs.classify_layer3_zeroshot("Implement a REST API endpoint")
    assert result is not None

    task_type, complexity, tier, conf, reasoning = result
    assert task_type  == TaskType.CODE_CREATION
    assert complexity == TaskComplexity.STANDARD
    assert tier       == ModelTier.MEDIUM
    # Geometric mean of 0.95 and 0.90 ≈ 0.924
    assert 0.92 < conf < 0.93
    assert reasoning.startswith("layer3 | zeroshot | code_creation/standard")


def test_reasoning_string_includes_both_probs():
    from classifier.core.types import TaskType, TaskComplexity
    from classifier.layers.layer3 import zeroshot as zs

    tt_label = next(k for k, v in zs._TASK_TYPE_HYPOTHESES.items() if v == TaskType.REASONING)
    cx_label = next(k for k, v in zs._COMPLEXITY_HYPOTHESES.items() if v == TaskComplexity.SIMPLE)
    zs._pipeline = _make_mock_pipe(tt_label, 0.92, cx_label, 0.88)

    _, _, _, _, reasoning = zs.classify_layer3_zeroshot("Compare X vs Y briefly")
    assert "tt=0.92" in reasoning
    assert "cx=0.88" in reasoning
    assert "conf=" in reasoning


# ── Abstain logic ─────────────────────────────────────────────────────────────

def test_low_confidence_abstains():
    """Geometric mean below threshold → returns None → cascade falls to L2."""
    from classifier.core.types import TaskType, TaskComplexity
    from classifier.layers.layer3 import zeroshot as zs

    tt_label = next(k for k, v in zs._TASK_TYPE_HYPOTHESES.items() if v == TaskType.CODE_CREATION)
    cx_label = next(k for k, v in zs._COMPLEXITY_HYPOTHESES.items() if v == TaskComplexity.SIMPLE)
    # 0.70 * 0.65 = 0.455 → sqrt ≈ 0.675, below 0.85 threshold
    zs._pipeline = _make_mock_pipe(tt_label, 0.70, cx_label, 0.65)

    result = zs.classify_layer3_zeroshot("ambiguous task")
    assert result is None


def test_asymmetric_confidence_abstains():
    """One head sure, other guessing → geometric mean penalises → abstain."""
    from classifier.core.types import TaskType, TaskComplexity
    from classifier.layers.layer3 import zeroshot as zs

    tt_label = next(k for k, v in zs._TASK_TYPE_HYPOTHESES.items() if v == TaskType.REASONING)
    cx_label = next(k for k, v in zs._COMPLEXITY_HYPOTHESES.items() if v == TaskComplexity.STANDARD)
    # 0.99 * 0.55 = 0.5445 → sqrt ≈ 0.738, below 0.85 threshold
    zs._pipeline = _make_mock_pipe(tt_label, 0.99, cx_label, 0.55)

    result = zs.classify_layer3_zeroshot("a task")
    assert result is None


# ── Failure modes (return None gracefully) ────────────────────────────────────

def test_transformers_not_installed_returns_none():
    """When transformers package missing, _load_pipeline returns None → classify returns None."""
    from classifier.layers.layer3 import zeroshot as zs

    with patch.object(zs, "_load_pipeline", return_value=None):
        result = zs.classify_layer3_zeroshot("any task")
    assert result is None


def test_pipeline_exception_returns_none():
    """If the pipeline call raises, we return None and cascade falls to L2."""
    from classifier.layers.layer3 import zeroshot as zs

    failing_pipe = MagicMock(side_effect=RuntimeError("CUDA OOM"))
    zs._pipeline = failing_pipe

    result = zs.classify_layer3_zeroshot("any task")
    assert result is None


def test_load_failed_short_circuits():
    """After a failed load, subsequent calls don't retry — they return None immediately."""
    from classifier.layers.layer3 import zeroshot as zs

    zs._load_failed = True
    assert zs._load_pipeline() is None


# ── History-aware truncation ──────────────────────────────────────────────────

def test_history_is_prepended_to_input():
    """Last history turn should be visible to the NLI model for continuation cases."""
    from classifier.core.types import TaskType, TaskComplexity
    from classifier.layers.layer3 import zeroshot as zs

    tt_label = next(k for k, v in zs._TASK_TYPE_HYPOTHESES.items() if v == TaskType.CODE_CREATION)
    cx_label = next(k for k, v in zs._COMPLEXITY_HYPOTHESES.items() if v == TaskComplexity.SIMPLE)
    zs._pipeline = _make_mock_pipe(tt_label, 0.95, cx_label, 0.92)

    zs.classify_layer3_zeroshot("now make it faster", history=["implement binary search"])

    # First call (tt) should have received the combined input
    called_with = zs._pipeline.call_args_list[0]
    sent_text = called_with.args[0]
    assert "implement binary search" in sent_text
    assert "now make it faster" in sent_text


# ── Strategy router ───────────────────────────────────────────────────────────

def test_strategy_router_dispatches_zeroshot():
    """settings.layer3_strategy='zeroshot' → calls zeroshot implementation."""
    from classifier.layers.layer3 import classify_layer3, zeroshot as zs
    from classifier.core.types import TaskType, TaskComplexity

    tt_label = next(k for k, v in zs._TASK_TYPE_HYPOTHESES.items() if v == TaskType.MATH)
    cx_label = next(k for k, v in zs._COMPLEXITY_HYPOTHESES.items() if v == TaskComplexity.SIMPLE)
    zs._pipeline = _make_mock_pipe(tt_label, 0.93, cx_label, 0.91)

    result = classify_layer3("solve x + 2 = 5")
    assert result is not None
    assert result[0] == TaskType.MATH


def test_strategy_router_skips_unimplemented_head():
    """'head' strategy not yet implemented → returns None gracefully."""
    from classifier.layers.layer3 import classify_layer3
    from classifier.infra.config import settings

    original = settings.layer3_strategy
    try:
        settings.layer3_strategy = "head"
        assert classify_layer3("any task") is None
    finally:
        settings.layer3_strategy = original


def test_strategy_router_skips_unimplemented_distilbert():
    from classifier.layers.layer3 import classify_layer3
    from classifier.infra.config import settings

    original = settings.layer3_strategy
    try:
        settings.layer3_strategy = "distilbert"
        assert classify_layer3("any task") is None
    finally:
        settings.layer3_strategy = original


def test_strategy_router_unknown_returns_none():
    from classifier.layers.layer3 import classify_layer3
    from classifier.infra.config import settings

    original = settings.layer3_strategy
    try:
        settings.layer3_strategy = "future_v4_strategy"
        assert classify_layer3("any task") is None
    finally:
        settings.layer3_strategy = original


# ── Cascade integration: L3 confident → skips L2 ──────────────────────────────

def test_cascade_l3_confident_skips_l2():
    """When L3 returns a confident result, L2 should not fire even if L1 was low-conf."""
    from classifier.core.types import TaskType, TaskComplexity
    from classifier.infra.config import settings
    from classifier.layers.layer3 import zeroshot as zs
    from classifier.infra.cache import cache

    tt_label = next(k for k, v in zs._TASK_TYPE_HYPOTHESES.items() if v == TaskType.REASONING)
    cx_label = next(k for k, v in zs._COMPLEXITY_HYPOTHESES.items() if v == TaskComplexity.COMPLEX)
    zs._pipeline = _make_mock_pipe(tt_label, 0.95, cx_label, 0.90)

    cache.clear()  # ensure fresh classification

    original_l3 = settings.layer3_enabled
    original_l2 = settings.layer2_enabled
    try:
        settings.layer3_enabled = True
        settings.layer2_enabled = True

        # Use a task with no L1 keyword matches → L1 confidence ~0.3 (DOC_CREATION fallback)
        # This guarantees L3 trigger condition `confidence < 0.75` is met.
        ambiguous_task = "xyzzy frobnicate quuxify the wibble"

        with patch("classifier.layers.layer2.api.genai.Client") as mock_l2_client:
            from classifier import classify_task
            decision = classify_task(ambiguous_task, provider="google")

        assert decision.layer_used == "layer3", (
            f"expected layer3 but got {decision.layer_used} "
            f"(conf={decision.confidence}, reasoning={decision.reasoning})"
        )
        mock_l2_client.assert_not_called()
    finally:
        settings.layer3_enabled = original_l3
        settings.layer2_enabled = original_l2
        cache.clear()


def test_cascade_l3_abstains_falls_through_to_l1():
    """When L3 abstains and L2 is disabled, the L1 result is preserved."""
    from classifier.infra.config import settings
    from classifier.layers.layer3 import zeroshot as zs
    from classifier.core.types import TaskType, TaskComplexity

    tt_label = next(k for k, v in zs._TASK_TYPE_HYPOTHESES.items() if v == TaskType.CODE_CREATION)
    cx_label = next(k for k, v in zs._COMPLEXITY_HYPOTHESES.items() if v == TaskComplexity.SIMPLE)
    # Low confidence → L3 abstains
    zs._pipeline = _make_mock_pipe(tt_label, 0.60, cx_label, 0.60)

    original_l3 = settings.layer3_enabled
    original_l2 = settings.layer2_enabled
    try:
        settings.layer3_enabled = True
        settings.layer2_enabled = False  # force L1 fallback after L3 abstain
        from classifier import classify_task
        decision = classify_task("Write a README", provider="google")
        # Layer 1 handled it (L3 abstained, L2 disabled)
        assert decision.layer_used == "layer1"
    finally:
        settings.layer3_enabled = original_l3
        settings.layer2_enabled = original_l2
