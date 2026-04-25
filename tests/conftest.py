"""Shared pytest fixtures for the classifier test suite."""
import pytest
from classifier.core.types import TaskType, TaskComplexity, ModelTier, ClassificationDecision


@pytest.fixture
def simple_doc_decision():
    return ClassificationDecision(
        model_name="gemini-2.5-flash-lite",
        tier=ModelTier.LOW,
        task_type=TaskType.DOC_CREATION,
        complexity=TaskComplexity.SIMPLE,
        reasoning="test fixture",
        confidence=0.9,
        provider="google",
        layer_used="layer1",
        latency_ms=0.5,
    )


@pytest.fixture
def complex_code_decision():
    return ClassificationDecision(
        model_name="gemini-2.5-pro",
        tier=ModelTier.HIGH,
        task_type=TaskType.CODE_CREATION,
        complexity=TaskComplexity.COMPLEX,
        reasoning="test fixture",
        confidence=0.85,
        provider="google",
        layer_used="layer1",
        latency_ms=0.8,
    )
