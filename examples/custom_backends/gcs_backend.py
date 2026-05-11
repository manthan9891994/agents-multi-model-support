"""GCS logger backend — batched JSONL uploads to Google Cloud Storage.

Buffers events in memory and flushes to GCS when either:
  - flush_size events accumulated, OR
  - flush_interval seconds have elapsed since last flush

Each flush writes one JSONL object at:
    gs://<bucket>/<prefix>YYYYMMDD/HHmmss-<epoch_ms>.jsonl

Install: pip install google-cloud-storage

Usage:
    from examples.custom_backends.gcs_backend import GCSBackend
    from classifier import Router

    backend = GCSBackend(bucket="my-dmr-logs", prefix="decisions/")
    router = Router(decision_logger=backend, outcome_logger=backend)

    # Call backend.flush() explicitly on shutdown to drain the buffer.
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class GCSBackend:
    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "dmr/",
        flush_interval: int = 60,
        flush_size: int = 100,
    ):
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise ImportError(
                "google-cloud-storage is required: pip install google-cloud-storage"
            ) from exc
        client = storage.Client()
        self._bucket = client.bucket(bucket)
        self._prefix = prefix.rstrip("/") + "/"
        self._buf: list[dict] = []
        self._lock = threading.Lock()
        self._flush_size = flush_size
        self._flush_interval = flush_interval
        self._last_flush = time.monotonic()

    def log(self, entry: dict) -> None:
        with self._lock:
            self._buf.append(entry)
            elapsed = time.monotonic() - self._last_flush
            if len(self._buf) >= self._flush_size or elapsed > self._flush_interval:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buf:
            return
        ts = datetime.now(timezone.utc)
        key = (
            f"{self._prefix}"
            f"{ts.strftime('%Y%m%d/%H%M%S')}-"
            f"{int(time.monotonic() * 1000)}.jsonl"
        )
        body = "\n".join(json.dumps(e) for e in self._buf).encode("utf-8")
        try:
            self._bucket.blob(key).upload_from_string(body, content_type="application/jsonl")
            self._buf.clear()
            self._last_flush = time.monotonic()
            logger.debug("GCSBackend: flushed %d events to gs://%s/%s", len(self._buf), self._bucket.name, key)
        except Exception as exc:
            logger.warning("GCSBackend: upload failed: %s", exc)
