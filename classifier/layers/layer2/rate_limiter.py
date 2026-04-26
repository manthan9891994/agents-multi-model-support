import threading
import time
from collections import deque

from classifier.infra.config import settings


class _RateLimiter:
    def __init__(self, max_rpm: int):
        self._lock  = threading.Lock()
        self._calls: deque[float] = deque()
        self._max   = max_rpm

    def allow(self) -> bool:
        now    = time.time()
        cutoff = now - 60
        with self._lock:
            while self._calls and self._calls[0] < cutoff:
                self._calls.popleft()
            if len(self._calls) >= self._max:
                return False
            self._calls.append(now)
            return True


_rate_limiter: _RateLimiter | None = None


def _get_rate_limiter() -> _RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = _RateLimiter(settings.layer2_max_rpm)
    return _rate_limiter
