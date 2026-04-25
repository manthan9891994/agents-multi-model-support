"""Logs every classification decision as a JSONL entry for analysis and retraining."""
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from classifier.core.types import ClassificationDecision

logger = logging.getLogger(__name__)

_lock    = threading.Lock()
_LOG_FILE = Path("routing_decisions.jsonl")


def log_decision(
    task: str,
    decision: "ClassificationDecision",
    layer_used: str,
    latency_ms: float,
) -> None:
    entry = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "task_preview": task[:200],
        "layer":        layer_used,
        "model":        decision.model_name,
        "tier":         decision.tier.value,
        "task_type":    decision.task_type.value,
        "complexity":   decision.complexity.value,
        "confidence":   round(decision.confidence, 4),
        "latency_ms":   round(latency_ms, 2),
        "provider":     decision.provider,
    }
    try:
        with _lock:
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
    except OSError as exc:
        logger.warning("Failed to write decision log: %s", exc)
