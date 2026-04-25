"""Unit tests for classifier/layers/layer2.py — mocks all external calls."""
import json
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    import classifier.layers.layer2 as l2
    l2._rate_limiter = None
    yield
    l2._rate_limiter = None


def _mock_response(data: dict) -> MagicMock:
    m = MagicMock()
    m.text = json.dumps(data)
    return m


def _valid_payload(**overrides):
    return {"task_type": "code_creation", "complexity": "standard",
            "confidence": 0.88, "reason": "implement endpoint with validation", **overrides}


# ── Happy path ─────────────────────────────────────────────────────────────────

def test_valid_response_returns_tuple():
    from classifier.core.types import TaskType, TaskComplexity, ModelTier
    with patch("classifier.layers.layer2.genai.Client") as mock_cls:
        mock_cls.return_value.models.generate_content.return_value = _mock_response(_valid_payload())
        from classifier.layers.layer2 import classify_layer2
        result = classify_layer2("Implement a REST API endpoint with input validation")

    assert result is not None
    task_type, complexity, tier, conf, reasoning = result
    assert task_type  == TaskType.CODE_CREATION
    assert complexity == TaskComplexity.STANDARD
    assert tier       == ModelTier.MEDIUM
    assert conf       == pytest.approx(0.88)
    assert reasoning.startswith("layer2 |")
    assert "conf=0.88" in reasoning


def test_reasoning_string_format():
    with patch("classifier.layers.layer2.genai.Client") as mock_cls:
        mock_cls.return_value.models.generate_content.return_value = _mock_response(
            _valid_payload(task_type="reasoning", complexity="simple", reason="simple comparison task")
        )
        from classifier.layers.layer2 import classify_layer2
        _, _, _, conf, reasoning = classify_layer2("Compare X and Y briefly")

    assert reasoning.startswith("layer2 |")
    assert "simple comparison task" in reasoning
    assert "conf=" in reasoning


# ── Validation failures → None ─────────────────────────────────────────────────

def test_invalid_task_type_returns_none():
    with patch("classifier.layers.layer2.genai.Client") as mock_cls:
        mock_cls.return_value.models.generate_content.return_value = _mock_response(
            _valid_payload(task_type="flying_unicorn")
        )
        from classifier.layers.layer2 import classify_layer2
        assert classify_layer2("some task") is None


def test_invalid_complexity_returns_none():
    with patch("classifier.layers.layer2.genai.Client") as mock_cls:
        mock_cls.return_value.models.generate_content.return_value = _mock_response(
            _valid_payload(complexity="extreme")
        )
        from classifier.layers.layer2 import classify_layer2
        assert classify_layer2("some task") is None


# ── Error conditions → None ────────────────────────────────────────────────────

def test_api_exception_returns_none():
    with patch("classifier.layers.layer2.genai.Client") as mock_cls:
        mock_cls.return_value.models.generate_content.side_effect = Exception("API down")
        from classifier.layers.layer2 import classify_layer2
        assert classify_layer2("some task") is None


def test_timeout_returns_none():
    from concurrent.futures import TimeoutError as FuturesTimeout
    with patch("classifier.layers.layer2._executor") as mock_exec:
        mock_future = MagicMock()
        mock_future.result.side_effect = FuturesTimeout()
        mock_exec.submit.return_value = mock_future
        from classifier.layers.layer2 import classify_layer2
        assert classify_layer2("some task") is None


def test_malformed_json_returns_none():
    with patch("classifier.layers.layer2.genai.Client") as mock_cls:
        m = MagicMock(); m.text = "not-json-at-all"
        mock_cls.return_value.models.generate_content.return_value = m
        from classifier.layers.layer2 import classify_layer2
        assert classify_layer2("some task") is None


# ── Rate limiter ───────────────────────────────────────────────────────────────

def test_rate_limit_blocks():
    from classifier.layers.layer2 import _RateLimiter
    limiter = _RateLimiter(max_rpm=3)
    assert limiter.allow() is True
    assert limiter.allow() is True
    assert limiter.allow() is True
    assert limiter.allow() is False  # 4th call within 60s → blocked


def test_rate_limiter_resets_after_window():
    import time
    from classifier.layers.layer2 import _RateLimiter
    limiter = _RateLimiter(max_rpm=2)
    assert limiter.allow() is True
    assert limiter.allow() is True
    assert limiter.allow() is False

    # Manually expire the window by backdating the timestamps
    with limiter._lock:
        limiter._calls = [t - 61 for t in limiter._calls]

    assert limiter.allow() is True  # window reset


def test_rate_limit_returns_none_in_classify():
    with patch("classifier.layers.layer2._get_rate_limiter") as mock_rl:
        mock_rl.return_value.allow.return_value = False
        from classifier.layers.layer2 import classify_layer2
        assert classify_layer2("some task") is None


# ── All task types and complexities accepted ───────────────────────────────────

@pytest.mark.parametrize("task_type", [
    "reasoning", "thinking", "analyzing", "code_creation", "doc_creation",
    "translation", "math", "conversation", "multimodal",
])
def test_all_task_types_accepted(task_type):
    with patch("classifier.layers.layer2.genai.Client") as mock_cls:
        mock_cls.return_value.models.generate_content.return_value = _mock_response(
            _valid_payload(task_type=task_type, complexity="simple")
        )
        from classifier.layers.layer2 import classify_layer2
        result = classify_layer2("some task")
    assert result is not None
    assert result[0].value == task_type


@pytest.mark.parametrize("complexity", ["simple", "standard", "complex", "research"])
def test_all_complexities_accepted(complexity):
    with patch("classifier.layers.layer2.genai.Client") as mock_cls:
        mock_cls.return_value.models.generate_content.return_value = _mock_response(
            _valid_payload(complexity=complexity, task_type="reasoning")
        )
        from classifier.layers.layer2 import classify_layer2
        result = classify_layer2("some task")
    assert result is not None
    assert result[1].value == complexity
