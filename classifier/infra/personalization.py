"""Per-user tier bias storage. Learns from explicit feedback to personalize routing.

Bias range: [-0.5, +0.5]
  > +0.3  → bump tier up one level (user prefers more powerful models)
  < -0.3  → drop tier down one level (user prefers cheaper/faster models)

Biases decay with a 30-day half-life so old preferences don't lock users in.
"""
import json
import math
import threading
import time
from pathlib import Path

_DATA_FILE  = Path(__file__).parent.parent / "data" / "user_biases.json"
_lock       = threading.Lock()
_DECAY_DAYS = 30
_MAX_BIAS   = 0.5
_NUDGE      = 0.1

_biases: dict[str, dict] = {}
_loaded = False


def _ensure_loaded() -> None:
    global _biases, _loaded
    if _loaded:
        return
    try:
        if _DATA_FILE.exists():
            with open(_DATA_FILE, encoding="utf-8") as f:
                _biases = json.load(f)
    except Exception:
        _biases = {}
    _loaded = True


def _save() -> None:
    try:
        _DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(_biases, f, indent=2)
    except Exception:
        pass


def get_user_bias(user_id: str) -> float:
    """Return decayed tier bias in [-0.5, 0.5]. Positive = prefer higher tier."""
    if not user_id:
        return 0.0
    with _lock:
        _ensure_loaded()
        entry = _biases.get(user_id)
    if not entry:
        return 0.0
    age_days = (time.time() - entry.get("updated", time.time())) / 86400
    decay = math.exp(-age_days * math.log(2) / _DECAY_DAYS)
    return entry.get("bias", 0.0) * decay


def update_user_bias(
    user_id: str,
    tier_too_low: bool = False,
    tier_too_high: bool = False,
) -> None:
    """Nudge a user's tier bias based on explicit feedback."""
    if not user_id or not (tier_too_low or tier_too_high):
        return
    delta = _NUDGE if tier_too_low else -_NUDGE
    with _lock:
        _ensure_loaded()
        entry = _biases.setdefault(user_id, {"bias": 0.0, "updated": time.time()})
        entry["bias"]    = max(-_MAX_BIAS, min(_MAX_BIAS, entry["bias"] + delta))
        entry["updated"] = time.time()
        _save()
