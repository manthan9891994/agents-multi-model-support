"""LRU classification cache — avoids re-classifying identical or near-identical tasks."""
import hashlib
import logging
import threading
import time
from collections import OrderedDict
from typing import Optional

from classifier.core.types import ClassificationDecision

logger = logging.getLogger(__name__)


class ClassificationCache:
    def __init__(self, max_size: int = 10_000, ttl_seconds: int = 3600):
        self._cache: OrderedDict[str, tuple[ClassificationDecision, float]] = OrderedDict()
        self._max_size  = max_size
        self._ttl       = ttl_seconds
        self._lock      = threading.Lock()
        self._hits      = 0
        self._misses    = 0

    def _key(self, task: str, provider: str) -> str:
        normalized = " ".join(task.lower().split())
        return hashlib.sha256(f"{provider}:{normalized}".encode()).hexdigest()

    def get(self, task: str, provider: str) -> Optional[ClassificationDecision]:
        key = self._key(task, provider)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            decision, ts = entry
            if time.time() - ts > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            logger.debug("Cache hit: %s...", task[:40])
            return decision

    def set(self, task: str, provider: str, decision: ClassificationDecision) -> None:
        key = self._key(task, provider)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (decision, time.time())
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def stats(self) -> dict:
        return {
            "size":     self.size,
            "hits":     self._hits,
            "misses":   self._misses,
            "hit_rate": round(self.hit_rate, 3),
        }


cache = ClassificationCache()
