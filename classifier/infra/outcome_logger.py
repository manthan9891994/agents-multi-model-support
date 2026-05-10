"""Outcome logger — append-only log of LLM-call outcomes joined by `decision_id`.

The decision log (`decision_logger.py`) records *what we picked*. This module
records *what happened*: tokens, wall time, success, user feedback, retries,
escalations.

Outcomes feed the auto-labeler (`ml/auto_labeler.py`) and drift detector
(`infra/drift_detector.py`). The two streams join on `decision_id`.

Public API:

    from classifier.infra.outcome_logger import log_outcome, OutcomeRecord

    log_outcome(OutcomeRecord(
        decision_id="...",
        tokens_in=142, tokens_out=38,
        wall_ms=412.3,
        success=True,
    ))

Or use the `Router.report_outcome(...)` shortcut that wraps this.

Pluggable backend: identical mechanism to `decision_logger._backend` —
set with `Router(outcome_logger=KafkaLoggerBackend(...))`.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_lock     = threading.Lock()
_LOG_FILE = Path(__file__).parent.parent.parent / "routing_outcomes.jsonl"
_TEST_LOG = Path(__file__).parent.parent.parent / "routing_outcomes.test.jsonl"

# Pluggable backend (e.g. KafkaLoggerBackend). Set by Router(outcome_logger=...).
# When non-None, outcomes go through it instead of the local JSONL fallback.
_backend = None


def _is_test_mode() -> bool:
    return os.environ.get("CLASSIFIER_TEST_MODE", "").lower() in ("1", "true", "yes")


@dataclass
class OutcomeRecord:
    """What happened after a routing decision was made.

    Joins to ClassificationDecision via `decision_id`. Fields beyond the
    required tokens/wall_ms are optional signals — the auto-labeler uses
    whichever ones are populated.

    `tokens_estimated`: True when token counts came from a heuristic
    tokenizer (e.g. word count) rather than the provider's `usage_metadata`.
    The auto-labeler can downweight or skip estimated rows.
    """
    decision_id:    str
    tokens_in:      int                  = 0
    tokens_out:     int                  = 0
    tokens_estimated: bool               = False
    wall_ms:        float                = 0.0
    success:        bool                 = True
    cost_usd:       float | None         = None
    user_retried:   bool                 = False
    user_escalated_model: str | None     = None
    user_feedback:  str | None           = None       # "up" | "down" | None
    edit_distance:  int | None           = None       # if user edited the response
    error_message:  str | None           = None
    timestamp:      str                  = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def _redact_outcome(entry: dict) -> dict:
    """Redact PII from string fields before writing.

    Same patterns as `decision_logger._redact_pii` (SSN, credit card, email,
    phone, API keys, JWT, MRN, DOB). Applied to free-form string fields:
    `error_message`, `user_escalated_model`. Numeric/bool/timestamp fields
    are passed through unchanged.
    """
    from classifier.infra.decision_logger import _redact_pii
    redacted = dict(entry)
    for key in ("error_message", "user_escalated_model"):
        val = redacted.get(key)
        if isinstance(val, str) and val:
            redacted[key] = _redact_pii(val)
    return redacted


def log_outcome(rec: OutcomeRecord) -> None:
    """Append an outcome to the configured backend (or the JSONL fallback).

    String fields are PII-redacted via `_redact_outcome` before writing.
    """
    entry = _redact_outcome(asdict(rec))

    if _backend is not None:
        try:
            _backend.log(entry)
            return
        except Exception as exc:
            logger.warning("outcome_logger backend failed, falling back to file: %s", exc)

    log_file = _TEST_LOG if _is_test_mode() else _LOG_FILE
    try:
        with _lock:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
    except OSError as exc:
        logger.warning("Failed to write outcome log: %s", exc)


def read_outcomes(
    *,
    since: str | None = None,
    until: str | None = None,
    decision_ids: set[str] | None = None,
) -> list[dict]:
    """Read outcomes — from the configured backend if it implements `.read()`,
    else from the local JSONL fallback.

    Args:
        since: ISO 8601 — only return outcomes at-or-after this time.
        until: ISO 8601 — only return outcomes strictly before this time.
        decision_ids: if set, only return outcomes whose decision_id is in here.

    Backend protocol:
        Any backend with a `.read(*, since, until, decision_ids) -> list[dict]`
        method is consulted first. If absent, the local JSONL is read instead.
        Backends explicitly opting out of read should set `read = None`.
    """
    # Try the configured backend first (if it supports read)
    if _backend is not None:
        backend_read = getattr(_backend, "read", None)
        if callable(backend_read):
            try:
                rows = backend_read(since=since, until=until, decision_ids=decision_ids)
                if rows is not None:
                    return list(rows)
            except Exception as exc:
                logger.warning(
                    "outcome_logger backend.read() failed (%s) — falling back to local JSONL",
                    exc,
                )
        elif backend_read is None:
            # Backend explicitly opted out (read=None). Fall through to local JSONL.
            pass

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
            if decision_ids and rec.get("decision_id") not in decision_ids:
                continue
            out.append(rec)
    return out


def prune_old_outcomes(*, days: int = 90) -> int:
    """Delete outcome rows older than `days`. Returns count of rows pruned.

    Operates on the local JSONL fallback only — for cloud-backed pipelines
    (Kafka / S3 with object lock), enforce retention at the infra layer.

    Wire this to a cron / scheduler for ongoing housekeeping:

        # crontab
        0 4 * * * dmr stats prune --days 90
    """
    from datetime import datetime, timedelta, timezone
    log_file = _TEST_LOG if _is_test_mode() else _LOG_FILE
    if not log_file.exists():
        return 0

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    pruned = 0
    kept_lines: list[str] = []
    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                kept_lines.append(line)   # keep malformed lines for debugging
                continue
            ts = rec.get("timestamp", "")
            if ts and ts < cutoff:
                pruned += 1
            else:
                kept_lines.append(line)

    with _lock:
        with log_file.open("w", encoding="utf-8") as f:
            for line in kept_lines:
                f.write(line + "\n")
    return pruned


def join_decisions_outcomes(
    decisions: list[dict],
    outcomes:  list[dict],
) -> list[dict]:
    """Inner-join decisions ⨝ outcomes on `decision_id`. Decisions without
    a matching outcome are dropped — the caller never reported what happened
    so we have no signal to label from.
    """
    by_id = {o["decision_id"]: o for o in outcomes if o.get("decision_id")}
    joined: list[dict] = []
    for d in decisions:
        did = d.get("decision_id")
        if not did or did not in by_id:
            continue
        joined.append({"decision": d, "outcome": by_id[did]})
    return joined
