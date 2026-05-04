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
from typing import Optional

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
    """
    decision_id:    str
    tokens_in:      int                  = 0
    tokens_out:     int                  = 0
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


def log_outcome(rec: OutcomeRecord) -> None:
    """Append an outcome to the configured backend (or the JSONL fallback)."""
    entry = asdict(rec)

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
    """Read outcomes from the JSONL fallback log. Returns a list of dicts.

    Args:
        since: ISO 8601 — only return outcomes at-or-after this time.
        until: ISO 8601 — only return outcomes strictly before this time.
        decision_ids: if set, only return outcomes whose decision_id is in here.
    """
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
