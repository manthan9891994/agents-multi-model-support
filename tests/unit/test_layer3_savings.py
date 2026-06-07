"""Unit tests for the Layer 3 quality<->savings dial (L3_DMR_SAVINGS_LEVEL).

The dial biases L3's chosen tier toward cheaper models without retraining:
0 = quality (L3's natural tier), each step = one tier cheaper, clamped at LOW.
"""

import pytest

from classifier.core.types import ModelTier, TaskComplexity, TaskType
from classifier.infra.config import settings
from classifier.layers.layer3 import (
    _apply_savings_level,
    classify_layer3,
    register_strategy,
)


@pytest.fixture(autouse=True)
def restore_settings():
    """Save/restore the L3 settings this dial touches."""
    saved = (settings.layer3_strategy, settings.l3_dmr_savings_level)
    yield
    settings.layer3_strategy, settings.l3_dmr_savings_level = saved


# ── Pure helper ───────────────────────────────────────────────────────────────


def test_apply_savings_level_shifts_and_clamps():
    assert _apply_savings_level(ModelTier.HIGH, 0) is ModelTier.HIGH  # no-op
    assert _apply_savings_level(ModelTier.HIGH, 1) is ModelTier.MEDIUM
    assert _apply_savings_level(ModelTier.HIGH, 2) is ModelTier.LOW
    assert _apply_savings_level(ModelTier.HIGH, 5) is ModelTier.LOW  # clamp
    assert _apply_savings_level(ModelTier.MEDIUM, 1) is ModelTier.LOW
    assert _apply_savings_level(ModelTier.LOW, 3) is ModelTier.LOW  # already lowest
    assert _apply_savings_level(ModelTier.HIGH, -1) is ModelTier.HIGH  # negative = no-op


# ── Dispatcher applies the dial (deterministic via a custom strategy) ──────────


def _fake_high_strategy(task, history=None):
    return TaskType.REASONING, TaskComplexity.COMPLEX, ModelTier.HIGH, 0.99, "fake"


def test_classify_layer3_default_level_is_noop():
    register_strategy("fake_high", _fake_high_strategy)
    settings.layer3_strategy = "fake_high"
    settings.l3_dmr_savings_level = 0
    result = classify_layer3("anything")
    assert result is not None
    assert result[2] is ModelTier.HIGH
    assert "savings_level" not in result[4]  # no annotation when unchanged


@pytest.mark.parametrize(
    "level,expected",
    [(1, ModelTier.MEDIUM), (2, ModelTier.LOW), (9, ModelTier.LOW)],
)
def test_classify_layer3_applies_savings(level, expected):
    register_strategy("fake_high", _fake_high_strategy)
    settings.layer3_strategy = "fake_high"
    settings.l3_dmr_savings_level = level
    result = classify_layer3("anything")
    assert result[2] is expected
    assert f"savings_level={level}" in result[4]  # reasoning annotated for transparency


# ── Router param sets + restores the dial via _apply_overrides ────────────────


def test_router_savings_level_override_and_restore():
    from classifier import Router

    before = settings.l3_dmr_savings_level
    r = Router(layer3_savings_level=2)
    with r._apply_overrides():
        assert settings.l3_dmr_savings_level == 2
    assert settings.l3_dmr_savings_level == before  # restored after the context
