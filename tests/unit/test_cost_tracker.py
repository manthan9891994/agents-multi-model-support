"""Unit tests for classifier/infra/cost_tracker.py."""
from classifier.infra.cost_tracker import CostTracker


def test_initial_state():
    t = CostTracker(monthly_budget_usd=100.0)
    assert t.total_cost == 0.0
    assert t.budget_remaining == 100.0
    assert t.budget_utilization == 0.0
    assert not t.should_downgrade()
    assert not t.is_exhausted()


def test_record_adds_cost():
    t = CostTracker(monthly_budget_usd=100.0)
    cost = t.record("gemini-2.5-pro", input_tokens=1_000_000, output_tokens=0)
    assert cost > 0
    assert t.total_cost == pytest.approx(cost)


def test_should_downgrade_at_80pct():
    t = CostTracker(monthly_budget_usd=1.0)
    # 330,000 tokens × $2.50/1M = $0.825 = 82.5% of $1 budget → triggers > 80%
    t.record("gemini-2.5-pro", input_tokens=330_000, output_tokens=0)
    assert t.should_downgrade()
    assert not t.is_exhausted()


def test_is_exhausted_at_100pct():
    t = CostTracker(monthly_budget_usd=1.0)
    t.record("gemini-2.5-pro", input_tokens=400_000, output_tokens=0)
    assert t.is_exhausted()


def test_unknown_model_uses_default_rate():
    t = CostTracker(monthly_budget_usd=100.0)
    cost = t.record("unknown-model-xyz", input_tokens=1_000_000, output_tokens=0)
    assert cost == 0.25  # default rate


def test_summary_structure():
    t = CostTracker(monthly_budget_usd=100.0)
    t.record("gpt-4o-mini", input_tokens=100, output_tokens=50)
    s = t.summary
    assert "total_cost_usd" in s
    assert "budget_usd" in s
    assert "by_model" in s
    assert "gpt-4o-mini" in s["by_model"]


import pytest
