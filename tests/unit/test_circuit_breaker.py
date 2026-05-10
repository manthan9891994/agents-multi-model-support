"""Tests for the L2 circuit breaker and Retry-After parsing."""
import time

from classifier.layers.layer2.api import _CircuitBreaker, _retry_after_seconds


def test_circuit_starts_closed():
    cb = _CircuitBreaker(failure_threshold=3, cooldown_secs=0.5)
    assert not cb.is_open()


def test_circuit_opens_after_threshold_failures():
    cb = _CircuitBreaker(failure_threshold=3, cooldown_secs=10.0)
    for _ in range(3):
        cb.record_failure()
    assert cb.is_open()


def test_circuit_does_not_open_on_intermittent_failures():
    cb = _CircuitBreaker(failure_threshold=3, cooldown_secs=10.0)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()   # resets counter
    cb.record_failure()
    assert not cb.is_open()


def test_circuit_half_opens_after_cooldown():
    cb = _CircuitBreaker(failure_threshold=2, cooldown_secs=0.05)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open()
    time.sleep(0.06)
    # After cooldown, allow one trial through (half-open)
    assert not cb.is_open()


def test_retry_after_parses_int():
    class FakeExc:
        retry_after = 5
    assert _retry_after_seconds(FakeExc()) == 5.0


def test_retry_after_parses_string():
    class FakeExc:
        retry_after = "12"
    assert _retry_after_seconds(FakeExc()) == 12.0


def test_retry_after_parses_dict_headers():
    class FakeExc:
        headers = {"Retry-After": "7"}
    assert _retry_after_seconds(FakeExc()) == 7.0


def test_retry_after_returns_none_when_absent():
    class FakeExc:
        pass
    assert _retry_after_seconds(FakeExc()) is None
