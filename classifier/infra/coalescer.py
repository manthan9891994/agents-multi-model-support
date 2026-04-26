"""Single-flight request coalescer — prevents cache stampede when many concurrent
requests miss the cache for the same task simultaneously."""
import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class SingleFlight:
    """Ensures only one computation runs per key; other callers wait and share the result."""

    def __init__(self):
        self._inflight: dict[str, threading.Event] = {}
        self._results:  dict[str, Any] = {}
        self._lock = threading.Lock()

    def do(self, key: str, fn: Callable[[], T]) -> T:
        with self._lock:
            if key in self._inflight:
                event = self._inflight[key]
                leader = False
            else:
                event = threading.Event()
                self._inflight[key] = event
                leader = True

        if leader:
            try:
                result = fn()
                with self._lock:
                    self._results[key] = result
                return result
            finally:
                event.set()
                with self._lock:
                    self._inflight.pop(key, None)
        else:
            event.wait(timeout=10)
            with self._lock:
                result = self._results.pop(key, None)
            # If leader failed (result=None), compute independently
            return result if result is not None else fn()


single_flight = SingleFlight()
