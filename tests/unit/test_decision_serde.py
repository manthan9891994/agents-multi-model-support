"""Round-trip tests for ClassificationDecision serde."""

import json

from classifier import classify_task
from classifier.core.types import (
    ClassificationDecision,
    ModelTier,
    TaskComplexity,
    TaskType,
)


def test_to_dict_and_back():
    d = ClassificationDecision(
        model_name="gemini-2.5-flash",
        tier=ModelTier.MEDIUM,
        task_type=TaskType.REASONING,
        complexity=TaskComplexity.STANDARD,
        reasoning="test",
        confidence=0.85,
        provider="google",
        layer_used="layer1",
        latency_ms=12.3,
        compliance_flag=False,
        disagreement=False,
    )
    serialized = d.to_dict()
    assert serialized["tier"] == "medium"
    assert serialized["task_type"] == "reasoning"

    d2 = ClassificationDecision.from_dict(serialized)
    assert d2.model_name == d.model_name
    assert d2.tier == d.tier
    assert d2.task_type == d.task_type
    assert d2.complexity == d.complexity


def test_to_json_and_back():
    d = ClassificationDecision(
        model_name="claude-opus-4-7",
        tier=ModelTier.HIGH,
        task_type=TaskType.CODE_CREATION,
        complexity=TaskComplexity.COMPLEX,
        reasoning="r",
        confidence=0.9,
        provider="anthropic",
    )
    raw = d.to_json()
    assert isinstance(raw, str)
    parsed = json.loads(raw)
    assert parsed["provider"] == "anthropic"

    d2 = ClassificationDecision.from_json(raw)
    assert d2.tier == ModelTier.HIGH
    assert d2.complexity == TaskComplexity.COMPLEX


def test_real_decision_roundtrip():
    d = classify_task("Write a Python function", provider="google")
    raw = d.to_json()
    d2 = ClassificationDecision.from_json(raw)
    assert d.tier == d2.tier
    assert d.model_name == d2.model_name
    assert d.task_type == d2.task_type


def test_from_dict_with_defaults():
    """Missing optional fields should fall back to defaults."""
    d = ClassificationDecision.from_dict(
        {
            "model_name": "x",
            "tier": "low",
            "task_type": "conversation",
            "complexity": "simple",
            "reasoning": "",
            "confidence": 0.5,
            "provider": "google",
        }
    )
    assert d.layer_used == "layer1"
    assert d.compliance_flag is False
