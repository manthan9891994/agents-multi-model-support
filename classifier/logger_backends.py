"""Pluggable decision logger backends.

Default writes JSONL to a local file. For production, swap in:
    - StdoutLoggerBackend   (write JSON lines to stdout — for K8s collectors)
    - WebhookLoggerBackend  (POST each decision to an HTTP endpoint)
    - KafkaLoggerBackend    (publish to a Kafka topic)
    - S3LoggerBackend       (batched writes to S3)
    - MultiLoggerBackend    (fan-out to N backends simultaneously)
    - NullLoggerBackend     (no-op for tests)

Wire to a Router:
    from classifier.logger_backends import KafkaLoggerBackend, MultiLoggerBackend
    backend = MultiLoggerBackend([
        KafkaLoggerBackend(brokers=["k1:9092"], topic="dmr-decisions"),
        StdoutLoggerBackend(),
    ])
    router = Router(decision_logger=backend, outcome_logger=backend)

See examples/custom_backends/ for BigQuery, DynamoDB, PostgreSQL, GCS examples.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class LoggerBackend(Protocol):
    def log(self, entry: dict) -> None: ...


class JSONLLoggerBackend:
    """Default backend — append-only JSONL file. Safe across threads."""

    def __init__(self, path: str = "routing_decisions.jsonl"):
        import threading
        from pathlib import Path

        self._path = Path(path)
        self._lock = threading.Lock()

    def log(self, entry: dict) -> None:
        try:
            with self._lock:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
        except OSError as exc:
            logger.warning("JSONLLoggerBackend: write failed: %s", exc)


class StdoutLoggerBackend:
    """Print JSON lines to stdout — for Kubernetes log-collector pipelines."""

    def log(self, entry: dict) -> None:
        sys.stdout.write(json.dumps(entry) + "\n")
        sys.stdout.flush()


class NullLoggerBackend:
    """No-op backend. Useful for tests."""

    def log(self, entry: dict) -> None:
        pass


class WebhookLoggerBackend:
    """POST each decision as JSON to a webhook URL.

    Best effort, fire-and-forget. Failures logged at WARNING but never propagate.
    """

    def __init__(self, url: str, *, timeout: float = 2.0, headers: dict | None = None):
        self._url = url
        self._timeout = timeout
        self._headers = headers or {"Content-Type": "application/json"}

    def log(self, entry: dict) -> None:
        try:
            import urllib.request

            req = urllib.request.Request(
                self._url,
                data=json.dumps(entry).encode("utf-8"),
                headers=self._headers,
                method="POST",
            )
            urllib.request.urlopen(req, timeout=self._timeout).read()
        except Exception as exc:
            logger.warning("WebhookLoggerBackend: post failed: %s", exc)


class KafkaLoggerBackend:
    """Publish each decision to a Kafka topic.

    Requires `confluent-kafka`. Install: `pip install 'dynamic-model-router[kafka]'`.
    """

    def __init__(self, *, brokers: list[str], topic: str, **producer_kwargs):
        try:
            from confluent_kafka import Producer
        except ImportError as exc:
            raise ImportError(
                "KafkaLoggerBackend requires confluent-kafka. "
                "Install: pip install 'dynamic-model-router[kafka]'"
            ) from exc
        self._producer = Producer(
            {
                "bootstrap.servers": ",".join(brokers),
                **producer_kwargs,
            }
        )
        self._topic = topic

    def log(self, entry: dict) -> None:
        try:
            self._producer.produce(self._topic, value=json.dumps(entry).encode("utf-8"))
            self._producer.poll(0)
        except Exception as exc:
            logger.warning("KafkaLoggerBackend: produce failed: %s", exc)


class MultiLoggerBackend:
    """Fan-out to N backends simultaneously.

    Individual backend failures are caught and logged at WARNING — they never
    block other backends or propagate to the caller.

    Usage:
        from classifier.logger_backends import MultiLoggerBackend, StdoutLoggerBackend, S3LoggerBackend
        backend = MultiLoggerBackend([
            StdoutLoggerBackend(),
            S3LoggerBackend(bucket="my-dmr-logs"),
        ])
        router = Router(decision_logger=backend, outcome_logger=backend)

    async_write=True: spawns a daemon thread per backend per call (fire-and-forget).
    Not durable on abrupt process exit — use sync mode (default) for write guarantees.
    """

    def __init__(self, backends: list, *, async_write: bool = False):
        self._backends = list(backends)
        self._async = async_write

    def log(self, entry: dict) -> None:
        if self._async:
            import threading

            for b in self._backends:
                threading.Thread(target=self._safe_log, args=(b, entry), daemon=True).start()
        else:
            for b in self._backends:
                self._safe_log(b, entry)

    def _safe_log(self, backend, entry: dict) -> None:
        try:
            backend.log(entry)
        except Exception as exc:
            logger.warning("MultiLoggerBackend: %s.log() failed: %s", type(backend).__name__, exc)

    def read(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        decision_ids: set | None = None,
    ) -> list[dict]:
        """Delegate read to the first backend that implements it."""
        for b in self._backends:
            read_fn = getattr(b, "read", None)
            if callable(read_fn):
                try:
                    result = read_fn(since=since, until=until, decision_ids=decision_ids)
                    if result is not None:
                        return list(result)
                except Exception as exc:
                    logger.warning("MultiLoggerBackend: %s.read() failed: %s", type(b).__name__, exc)
        return []


class S3LoggerBackend:
    """Buffered writes to S3 — flushes every N decisions or T seconds."""

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "dmr/",
        flush_interval: int = 60,
        flush_size: int = 100,
        **boto3_kwargs,
    ):
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "S3LoggerBackend requires boto3. Install: pip install 'dynamic-model-router[s3]'"
            ) from exc
        import threading
        import time

        self._client = boto3.client("s3", **boto3_kwargs)
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/"
        self._buf: list[dict] = []
        self._lock = threading.Lock()
        self._flush_size = flush_size
        self._flush_interval = flush_interval
        self._last_flush = time.time()

    def log(self, entry: dict) -> None:
        import time

        with self._lock:
            self._buf.append(entry)
            now = time.time()
            if len(self._buf) >= self._flush_size or now - self._last_flush > self._flush_interval:
                self._flush_locked()
                self._last_flush = now

    def _flush_locked(self) -> None:
        if not self._buf:
            return
        import time
        from datetime import datetime, timezone

        body = "\n".join(json.dumps(e) for e in self._buf).encode("utf-8")
        key = f"{self._prefix}{datetime.now(timezone.utc).strftime('%Y/%m/%d/%H%M%S')}-{int(time.time() * 1000)}.jsonl"
        try:
            self._client.put_object(Bucket=self._bucket, Key=key, Body=body)
            self._buf.clear()
        except Exception as exc:
            logger.warning("S3LoggerBackend: put failed: %s", exc)
