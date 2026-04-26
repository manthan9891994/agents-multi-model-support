"""Record classification feedback for retraining and accuracy tracking."""
import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_REFERENCE_FILE = Path(__file__).parent.parent / "data" / "reference_tasks.jsonl"


def record_feedback(
    task: str,
    expected_type: str,
    expected_complexity: str,
    original_type: str = "",
    original_complexity: str = "",
    user_id: str = "",
    tier_too_low: bool = False,
    tier_too_high: bool = False,
) -> None:
    """Append a feedback entry to reference_tasks.jsonl for Layer 3 training data.

    Pass user_id + tier_too_low/tier_too_high to update per-user tier bias.
    """
    entry = {
        "task":                 task[:500],
        "expected_type":        expected_type,
        "expected_complexity":  expected_complexity,
        "original_type":        original_type,
        "original_complexity":  original_complexity,
        "source":               "feedback",
    }
    try:
        with _lock:
            with open(_REFERENCE_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        logger.info("Feedback recorded for task: %s...", task[:40])
    except OSError as exc:
        logger.warning("Failed to write feedback: %s", exc)

    if user_id and (tier_too_low or tier_too_high):
        try:
            from classifier.infra.personalization import update_user_bias
            update_user_bias(user_id, tier_too_low=tier_too_low, tier_too_high=tier_too_high)
        except Exception as e:
            logger.debug("Personalization update skipped: %s", e)
