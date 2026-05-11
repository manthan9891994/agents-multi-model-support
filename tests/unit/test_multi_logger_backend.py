"""Tests for MultiLoggerBackend fan-out."""

import threading
import time

import pytest

from classifier.logger_backends import MultiLoggerBackend, NullLoggerBackend, StdoutLoggerBackend


class _CollectingBackend:
    def __init__(self):
        self.logged = []

    def log(self, entry: dict) -> None:
        self.logged.append(entry)


class _ReadableBackend(_CollectingBackend):
    def read(self, *, since=None, until=None, decision_ids=None):
        return [{"decision_id": "r1"}]


class _BrokenBackend:
    def log(self, entry: dict) -> None:
        raise RuntimeError("intentional failure")


def test_fanout_reaches_all_backends():
    a, b = _CollectingBackend(), _CollectingBackend()
    multi = MultiLoggerBackend([a, b])
    multi.log({"decision_id": "x1", "tier": "low"})
    assert len(a.logged) == 1
    assert len(b.logged) == 1
    assert a.logged[0]["decision_id"] == "x1"


def test_broken_backend_does_not_block_others():
    good = _CollectingBackend()
    multi = MultiLoggerBackend([_BrokenBackend(), good])
    multi.log({"decision_id": "y1"})  # must not raise
    assert len(good.logged) == 1


def test_broken_backend_error_not_raised():
    multi = MultiLoggerBackend([_BrokenBackend()])
    multi.log({"decision_id": "z1"})  # no exception


def test_read_delegates_to_first_readable_backend():
    multi = MultiLoggerBackend([NullLoggerBackend(), _ReadableBackend()])
    rows = multi.read(since=None, until=None)
    assert rows == [{"decision_id": "r1"}]


def test_read_returns_empty_when_no_readable_backend():
    multi = MultiLoggerBackend([NullLoggerBackend()])
    assert multi.read(since=None, until=None) == []


def test_async_write_returns_immediately():
    barrier = threading.Event()
    completed = []

    class _SlowBackend:
        def log(self, entry):
            barrier.wait(timeout=3)
            completed.append(entry)

    multi = MultiLoggerBackend([_SlowBackend()], async_write=True)
    t0 = time.monotonic()
    multi.log({"decision_id": "async1"})
    elapsed = time.monotonic() - t0
    assert elapsed < 0.1, f"async_write should return immediately, took {elapsed:.3f}s"
    barrier.set()
    time.sleep(0.1)
    assert len(completed) == 1


def test_multiple_entries_fanned_out():
    a = _CollectingBackend()
    multi = MultiLoggerBackend([a])
    for i in range(10):
        multi.log({"decision_id": f"d{i:02d}"})
    assert len(a.logged) == 10


def test_null_backend_compatible():
    multi = MultiLoggerBackend([NullLoggerBackend()])
    multi.log({"decision_id": "null1"})  # no error
