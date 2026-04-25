"""Unit tests for classifier/infra/cache.py."""
import time
from classifier.infra.cache import ClassificationCache
from classifier.core.types import TaskType, TaskComplexity, ModelTier, ClassificationDecision


def _decision(model="gemini-2.5-flash-lite"):
    return ClassificationDecision(
        model_name=model,
        tier=ModelTier.LOW,
        task_type=TaskType.DOC_CREATION,
        complexity=TaskComplexity.SIMPLE,
        reasoning="test",
        confidence=0.9,
        provider="google",
    )


def test_miss_on_empty_cache():
    c = ClassificationCache()
    assert c.get("hello", "google") is None


def test_set_then_get():
    c = ClassificationCache()
    d = _decision()
    c.set("hello world", "google", d)
    result = c.get("hello world", "google")
    assert result is not None
    assert result.model_name == d.model_name


def test_normalizes_whitespace():
    c = ClassificationCache()
    d = _decision()
    c.set("hello   world", "google", d)
    assert c.get("hello world", "google") is not None


def test_provider_isolation():
    c = ClassificationCache()
    d = _decision()
    c.set("hello", "google", d)
    assert c.get("hello", "openai") is None


def test_ttl_expiry():
    c = ClassificationCache(ttl_seconds=1)
    d = _decision()
    c.set("hello", "google", d)
    time.sleep(1.1)
    assert c.get("hello", "google") is None


def test_evicts_oldest_on_full():
    c = ClassificationCache(max_size=2)
    c.set("task1", "google", _decision())
    c.set("task2", "google", _decision())
    c.set("task3", "google", _decision())
    assert c.size == 2


def test_clear_resets_stats():
    c = ClassificationCache()
    c.set("hello", "google", _decision())
    c.get("hello", "google")
    c.clear()
    assert c.size == 0
    assert c.hit_rate == 0.0


def test_hit_rate_calculation():
    c = ClassificationCache()
    c.set("hello", "google", _decision())
    c.get("hello", "google")   # hit
    c.get("missing", "google") # miss
    assert c.hit_rate == 0.5
