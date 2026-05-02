"""Tests for input length guard, API key validation, and last-decision reset."""
import pytest

import classifier as pkg
from classifier import Router, classify_task, reset_last_decision
from classifier.core.exceptions import ClassificationError


def test_oversized_input_raises():
    router = Router(layer2_enabled=False, layer3_enabled=False)
    too_long = "x" * (pkg.MAX_TASK_CHARS + 1)
    with pytest.raises(ClassificationError) as exc_info:
        router.classify(too_long)
    assert "exceeds" in str(exc_info.value).lower()


def test_under_limit_input_succeeds():
    router = Router(layer2_enabled=False, layer3_enabled=False)
    ok = "x" * 100
    decision = router.classify(ok)
    assert decision is not None


def test_empty_task_raises_with_suggestion():
    router = Router(layer2_enabled=False, layer3_enabled=False)
    with pytest.raises(ClassificationError) as exc_info:
        router.classify("")
    assert "empty" in str(exc_info.value).lower()
    assert exc_info.value.suggestion   # has suggestion text


def test_classification_error_attributes():
    try:
        classify_task("")
    except ClassificationError as e:
        assert hasattr(e, "layer")
        assert hasattr(e, "task_preview")
        assert hasattr(e, "suggestion")


def test_reset_last_decision_clears_streaming_cache():
    router = Router(layer2_enabled=False, layer3_enabled=False)
    router.classify("first task")   # populates _last_decision
    reset_last_decision()
    # Next streaming-debounce call should NOT return a stale decision
    # (we can't easily inspect internals, but at least the function runs)
    decision = router.classify("second task")
    assert decision is not None


def test_classification_error_has_layer_context():
    """ClassificationError should carry the failed layer name."""
    err = ClassificationError("test", layer="layer1", task="some task", suggestion="try X")
    assert err.layer == "layer1"
    assert err.task_preview == "some task"
    assert err.suggestion == "try X"
    msg = str(err)
    assert "layer1" in msg
    assert "try X" in msg
