"""Detect a failed/degraded model response so the router can escalate.

The dominant failure mode on agentic workloads is a cheap model *giving up*:
"I cannot access the data", "need more information", empty/too-short, or an
explicit tool error. `looks_like_failure` is a fast, dependency-free heuristic
used by `escalate_on_failure` (see integrations/_agentic.py).
"""

from __future__ import annotations

_FAIL_PHRASES = (
    "i cannot access",
    "i don't have access",
    "i do not have access",
    "unable to access",
    "unable to retrieve",
    "cannot retrieve",
    "need more information",
    "need additional information",
    "cannot fulfill",
    "can't fulfill",
    "unable to fulfill",
    "please provide",
    "i'm unable to",
    "i am unable to",
    "no data available",
    "error:",
    "traceback (most recent call last)",
)


def looks_like_failure(text: str | None, *, min_chars: int = 40) -> tuple[bool, str]:
    """Return (is_failure, reason). Heuristic: empty, too short, or a refusal phrase."""
    if not text or not text.strip():
        return True, "empty"
    t = text.strip()
    if len(t) < min_chars:
        return True, "too_short"
    low = t.lower()
    for p in _FAIL_PHRASES:
        if p in low:
            return True, f"refusal:{p}"
    return False, ""
