"""Pluggable cache backends.

Default is `InMemoryCacheBackend` (current behaviour). For multi-instance
deployments use Redis or implement your own:

    class CacheBackend(Protocol):
        def get(self, key: str) -> Any | None: ...
        def set(self, key: str, value: Any, ttl: int) -> None: ...
        def clear(self) -> None: ...

Wire to a Router:
    from classifier.cache_backends import RedisCacheBackend
    router = Router(cache_backend=RedisCacheBackend(host="localhost"))
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class CacheBackend(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl: int) -> None: ...
    def clear(self) -> None: ...


class InMemoryCacheBackend:
    """LRU + TTL in-process cache. Default backend."""

    def __init__(self, max_size: int = 10_000):
        self._store: dict[str, tuple[float, Any]] = {}
        self._max_size = max_size
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at < time.time():
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl: int) -> None:
        with self._lock:
            if len(self._store) >= self._max_size:
                # Drop oldest
                oldest = min(self._store.items(), key=lambda kv: kv[1][0])[0]
                self._store.pop(oldest, None)
            self._store[key] = (time.time() + ttl, value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class RedisCacheBackend:
    """Redis-backed cache. Requires `pip install redis`.

    Stores ClassificationDecision objects as JSON; deserializes on get().
    """

    def __init__(self, *, host: str = "localhost", port: int = 6379,
                 db: int = 0, prefix: str = "dmr:cache:", **redis_kwargs):
        try:
            import redis
        except ImportError as exc:
            raise ImportError(
                "RedisCacheBackend requires the redis package. "
                "Install with: pip install 'dynamic-model-router[redis]'"
            ) from exc
        self._client = redis.Redis(host=host, port=port, db=db, **redis_kwargs)
        self._prefix = prefix

    def get(self, key: str) -> Any | None:
        raw = self._client.get(self._prefix + key)
        if raw is None:
            return None
        try:
            from classifier.core.types import ClassificationDecision
            return ClassificationDecision.from_json(raw.decode("utf-8"))
        except Exception as exc:
            logger.warning("RedisCacheBackend: deserialize failed: %s", exc)
            return None

    def set(self, key: str, value: Any, ttl: int) -> None:
        try:
            payload = value.to_json() if hasattr(value, "to_json") else str(value)
            self._client.setex(self._prefix + key, ttl, payload)
        except Exception as exc:
            logger.warning("RedisCacheBackend: set failed: %s", exc)

    def clear(self) -> None:
        for k in self._client.scan_iter(match=self._prefix + "*"):
            self._client.delete(k)


class FileCacheBackend:
    """Single-machine persistence between process restarts. JSONL append-only."""

    def __init__(self, path: str = ".dmr_cache.jsonl"):
        from pathlib import Path
        self._path = Path(path)
        self._lock = threading.RLock()
        self._mem = InMemoryCacheBackend()
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            import json
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._mem.set(rec["key"], rec["value"], rec.get("ttl", 3600))
        except Exception as exc:
            logger.warning("FileCacheBackend: load failed: %s", exc)

    def get(self, key: str) -> Any | None:
        return self._mem.get(key)

    def set(self, key: str, value: Any, ttl: int) -> None:
        import json
        self._mem.set(key, value, ttl)
        with self._lock:
            try:
                payload = value.to_dict() if hasattr(value, "to_dict") else value
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "value": payload, "ttl": ttl}) + "\n")
            except Exception as exc:
                logger.warning("FileCacheBackend: write failed: %s", exc)

    def clear(self) -> None:
        self._mem.clear()
        if self._path.exists():
            self._path.unlink()


class NullCacheBackend:
    """No-op cache (always miss). Useful for testing and bypass."""
    def get(self, key: str) -> Any | None: return None
    def set(self, key: str, value: Any, ttl: int) -> None: pass
    def clear(self) -> None: pass
