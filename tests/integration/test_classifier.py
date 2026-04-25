"""Integration tests for classifier.classify_task() — tests full Layer 1 pipeline."""
import pytest
from classifier import classify_task
from classifier.core.types import ModelTier, TaskType, TaskComplexity
from classifier.core.exceptions import ClassificationError, UnsupportedProviderError


# ── Tier routing ───────────────────────────────────────────────────────────────

def test_simple_doc_routes_to_low():
    d = classify_task("Write a README for this project")
    assert d.tier == ModelTier.LOW
    assert d.task_type == TaskType.DOC_CREATION
    assert d.complexity == TaskComplexity.SIMPLE


def test_complex_thinking_routes_to_high():
    d = classify_task("Design a distributed cache with LRU eviction and thread safety")
    assert d.tier == ModelTier.HIGH


def test_reasoning_routes_to_medium():
    d = classify_task("Compare Python vs JavaScript for building REST APIs")
    assert d.tier == ModelTier.MEDIUM
    assert d.task_type == TaskType.REASONING


def test_simple_code_routes_to_low():
    d = classify_task("Write a function to check if a string is a palindrome")
    assert d.tier == ModelTier.LOW
    assert d.task_type == TaskType.CODE_CREATION


def test_standard_code_routes_to_medium():
    d = classify_task("Implement a REST API endpoint with input validation")
    assert d.tier == ModelTier.MEDIUM


def test_research_routes_to_high():
    d = classify_task("Comprehensive research on AI adoption across 10 industries with market data")
    assert d.tier == ModelTier.HIGH
    assert d.complexity in (TaskComplexity.COMPLEX, TaskComplexity.RESEARCH)


# ── Provider model names ───────────────────────────────────────────────────────

def test_google_model_name():
    d = classify_task("Write a README", provider="google")
    assert "gemini" in d.model_name
    assert d.provider == "google"


def test_openai_model_name():
    d = classify_task("Write a README", provider="openai")
    assert "gpt" in d.model_name
    assert d.provider == "openai"


def test_anthropic_model_name():
    d = classify_task("Write a README", provider="anthropic")
    assert "claude" in d.model_name
    assert d.provider == "anthropic"


# ── Error handling ─────────────────────────────────────────────────────────────

def test_empty_task_raises():
    with pytest.raises(ClassificationError):
        classify_task("")


def test_whitespace_only_task_raises():
    with pytest.raises(ClassificationError):
        classify_task("   ")


def test_invalid_provider_raises():
    with pytest.raises(UnsupportedProviderError):
        classify_task("Write a README", provider="cohere")


# ── Decision fields ────────────────────────────────────────────────────────────

def test_decision_has_all_fields():
    d = classify_task("Write a README")
    assert d.model_name
    assert d.tier in ModelTier
    assert d.task_type in TaskType
    assert d.complexity in TaskComplexity
    assert 0.0 <= d.confidence <= 1.0
    assert d.reasoning
    assert d.provider
    assert d.layer_used
    assert d.latency_ms >= 0.0
