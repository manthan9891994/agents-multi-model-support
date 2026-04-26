"""Sliding-window p95 latency tracker for adaptive routing.

When a (provider, tier) combination exceeds the SLO p95 threshold, is_degraded()
returns True and classify_task() demotes one tier automatically.
"""
import threading
from collections import defaultdict, deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from classifier.core.types import ModelTier


class TierHealthTracker:
    def __init__(self, slo_ms: float = 8000.0, window: int = 50):
        self._slo_ms = slo_ms
        self._window = window
        self._latencies: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=window))
        self._lock = threading.Lock()

    def record(self, provider: str, tier: "ModelTier", latency_ms: float) -> None:
        with self._lock:
            self._latencies[(provider, tier)].append(latency_ms)

    def is_degraded(self, provider: str, tier: "ModelTier") -> bool:
        with self._lock:
            samples = list(self._latencies[(provider, tier)])
        if len(samples) < 10:
            return False
        p95 = sorted(samples)[int(len(samples) * 0.95)]
        return p95 > self._slo_ms

    def p95(self, provider: str, tier: "ModelTier") -> float | None:
        with self._lock:
            samples = list(self._latencies[(provider, tier)])
        if len(samples) < 2:
            return None
        return sorted(samples)[int(len(samples) * 0.95)]

    def stats(self) -> dict:
        with self._lock:
            result = {}
            for (provider, tier), dq in self._latencies.items():
                samples = list(dq)
                if not samples:
                    continue
                key = f"{provider}/{tier.value}"
                result[key] = {
                    "count":      len(samples),
                    "p50":        round(sorted(samples)[len(samples) // 2], 1),
                    "p95":        round(sorted(samples)[int(len(samples) * 0.95)], 1),
                    "degraded":   self.is_degraded(provider, tier),
                }
        return result


health_tracker = TierHealthTracker()
