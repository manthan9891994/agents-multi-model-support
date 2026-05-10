"""Logs every classification decision as a JSONL entry for analysis and retraining."""

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from classifier.core.types import ClassificationDecision

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_LOG_FILE = Path(__file__).parent.parent.parent / "routing_decisions.jsonl"
_TEST_LOG = Path(__file__).parent.parent.parent / "routing_decisions.test.jsonl"

# PII patterns — spans matched here are replaced with [REDACTED] before logging
_REDACT_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"),  # credit card
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),  # email
    re.compile(r"\b\+?1?[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}\b"),  # phone
    re.compile(r"\b(sk-|pk_|AIza|ghp_|xox[baprs]-)[A-Za-z0-9_-]{16,}"),  # API key
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),  # JWT
    re.compile(r"\bMRN[\s:]*\d{4,}\b", re.IGNORECASE),  # MRN
    re.compile(r"\bDOB[\s:]*\d{4}-\d{2}-\d{2}\b", re.IGNORECASE),  # DOB
]


def _redact_pii(text: str) -> str:
    for pat in _REDACT_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def _is_test_mode() -> bool:
    return os.environ.get("CLASSIFIER_TEST_MODE", "").lower() in ("1", "true", "yes")


_backend = None  # set by Router(decision_logger=...) — None falls back to default JSONL


def read_decisions(
    *,
    since: str | None = None,
    until: str | None = None,
) -> list[dict]:
    """Read decision rows — from the configured backend if it implements `.read()`,
    else from the local JSONL fallback.

    Args:
        since: ISO 8601 — only return decisions at-or-after this time.
        until: ISO 8601 — only return decisions strictly before this time.

    Used by the AutoLabeler to feed weak supervision over real production traffic.
    """
    if _backend is not None:
        backend_read = getattr(_backend, "read", None)
        if callable(backend_read):
            try:
                rows = backend_read(since=since, until=until)
                if rows is not None:
                    return list(rows)
            except Exception as exc:
                logger.warning(
                    "decision_logger backend.read() failed (%s) — falling back to local JSONL",
                    exc,
                )

    log_file = _TEST_LOG if _is_test_mode() else _LOG_FILE
    if not log_file.exists():
        return []

    out: list[dict] = []
    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since and rec.get("timestamp", "") < since:
                continue
            if until and rec.get("timestamp", "") >= until:
                continue
            out.append(rec)
    return out


def log_decision(
    task: str,
    decision: "ClassificationDecision",
    layer_used: str,
    latency_ms: float,
) -> None:
    safe_preview = _redact_pii(task[:200])
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision_id": getattr(decision, "decision_id", ""),  # join key for outcome log
        "task_preview": safe_preview,
        "task_length": len(task or ""),
        "layer": layer_used,
        "model": decision.model_name,
        "tier": decision.tier.value,
        "task_type": decision.task_type.value,
        "complexity": decision.complexity.value,
        "confidence": round(decision.confidence, 4),
        "latency_ms": round(latency_ms, 2),
        "provider": decision.provider,
        "compliance_flag": decision.compliance_flag,
        "disagreement": decision.disagreement,
        "exploration": getattr(decision, "exploration", False),
        "cached": getattr(decision, "cached", False),
        "cached_from": getattr(decision, "cached_from", ""),
    }

    # Pluggable backend takes precedence
    if _backend is not None:
        try:
            _backend.log(entry)
            return
        except Exception as exc:
            logger.warning("decision_logger backend failed, falling back to file: %s", exc)

    # Default: append-only JSONL file
    log_file = _TEST_LOG if _is_test_mode() else _LOG_FILE
    try:
        with _lock:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
    except OSError as exc:
        logger.warning("Failed to write decision log: %s", exc)
